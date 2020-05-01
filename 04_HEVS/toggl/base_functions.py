#!/usr/bin/env python3
###############################################################################
#          __ _____            __
#   /\  /\/__\\_   \   /\   /\/ _\
#  / /_/ /_\   / /\/___\ \ / /\ \
# / __  //__/\/ /_|_____\ V / _\ \
# \/ /_/\__/\____/       \_/  \__/
# (c) - zas
###############################################################################
# toggl - Custom functions
# **Copyright (C) 2019 HES-SO Valais-Wallis - All Rights Reserved**
###############################################################################

###############################################################################
# Import sub-modules
# python
import sys
import os
import os.path
# import enum
import datetime
from dateutil.parser import parse
import time
import pytz
import json
import re  # noqa
import math

# watermark
import watermark

# wavedrom
import nbwavedrom

# project base config
from base_config import *

# import hevslib
from hevslib.general import *
from hevslib.pandas import *
from hevslib.plotly import *
from hevslib.time import *
from hevslib.toggl import *
from hevslib.md import *

###############################################################################
# Custom Time Functions
#
def fillCleanTimeentries(df, projects, user_clients):
  # Add project name and client name to dataframe
  df['project'] = df.pid.map(projects.set_index('id')['name'].to_dict())
  projects['client'] = projects.cid.map(user_clients.set_index('id')['name'].to_dict())
  df['client'] = df.pid.map(projects.set_index('id')['client'].to_dict())
  # drop unused columns
  drop_columns = ['at', 'billable', 'duronly', 'guid', 'id', 'pid', 'uid', 'wid']
  df = df.drop(drop_columns, axis=1)
  # rearrange columns
  df = df[['client', 'project', 'description', 'start', 'stop', 'duration']]
  # calculate timedelta
  df['duration'] = pd.to_timedelta(df['duration'], unit='sec')
  # transform to datetime
  df['start'] = pd.to_datetime(df['start'])
  df['stop'] = pd.to_datetime(df['stop'])
  return df


def getJoinDate(user):
  join_date = parse(user['data']['created_at'])
  join_date = join_date.replace(tzinfo=None)
  return join_date


# Function that turns datetimes back to strings since that's what the API likes
def dateOnly(datetimeVal):
  datePart = datetimeVal.strftime("%Y-%m-%d")
  return datePart


# Extract Timelogs Between Two Dates and Export to a CSV
def togglTimelogExtractor(toggl, toggl_time_entries_url, input_date1, input_date2, save_csv=False, outputdir='output'):
  date1 = parse(input_date1).isoformat() + '+00:00'
  date2 = parse(input_date2).isoformat() + '+00:00'
  param = {
    'start_date': date1,
    'end_date': date2,
  }
  try:
    temp_log =  pd.DataFrame.from_dict(toggl.request(toggl_time_entries_url, parameters=param))
    if save_csv:
      if not os.path.exists(outputdir + '/detailed/'):
        os.makedirs(outputdir + '/detailed/')
      temp_log.to_csv(outputdir + '/detailed/toggl-time-entries-' + input_date1 + '.csv')
  except:
    # try again if there is an issue the first time
    temp_log =  pd.DataFrame.from_dict(toggl.request(toggl_time_entries_url, parameters=param))
    if save_csv:
      if not os.path.exists(outputdir + '/daily-detailed/'):
        os.makedirs(outputdir + '/detailed/')
      temp_log.to_csv(outputdir + '/detailed/toggl-time-entries-' + input_date1 + '.csv')
  return temp_log


###############################################################################
# Custom Visualization Functions
#
def workingHoursPieChart(df, outputDir, graph_subdir, startDate, endDate=None, timeFrame="Week", plotlySettings=None):
  title = "Work by Clients"
  labels = df['client']
  values = df['duration']
  trace1 = go.Pie(title=title,
                  labels=labels,
                  values=values,
                  hoverinfo='label+percent+value',
                  domain=dict(x=[0,0.5]))
    
  title = "Work by Projects"
  labels = df['project']
  values = df['duration']
  trace2 = go.Pie(title=title,
                  labels=labels,
                  values=values,
                  hoverinfo='label+percent+value',
                  domain=dict(x=[0.5,1.0]))
    
  if timeFrame == "Week":
    title = "{} {} {} Work partition Pie".format(timeFrame, endDate.strftime('%Y-%m-%d'), getWeekNbr(startDate))
  else:
    title = "{} {} Work partition Pie".format(timeFrame, startDate.strftime('%Y-%m'))
  layout = go.Layout(title=title,
                     #annotations=[ann1,ann2],
                     # Hide legend if you want
                     #showlegend=False
                     )
    
  data = [trace1, trace2]
  # Create fig with data and layout
  fig = go.Figure(data=data,layout=layout)
  
  graphFilename = (title + ext_file).replace(" ", "_")
  plotFigure(outputDir + graph_subdir + graphFilename, fig, plotlySettings[1], plotlySettings[2], plotlySettings[3], plotlySettings[4])
  
  # return md for report
  return mdImage("." + os.sep + graph_subdir + graphFilename, title)


def workingHoursBarChart(clientDfs, outputDir, graph_subdir, uniqueEntries, startDate, endDate=None, timeFrame="Week", plotlySettings=None):
  if timeFrame == "Week":
    title = "Week {} {} Work Partition Bar".format(endDate.strftime('%Y-%m-%d'), getWeekNbr(startDate))
  else:
    title = "{} {} Work Partition Bar".format(timeFrame, startDate.strftime('%Y-%m'))
  clientDfs[0]['duration'] = clientDfs[0]['duration'].apply(lambda x: pd.to_timedelta(x, unit='h'))
  traces = []
  for i in range(len(clientDfs)):
    labels = clientDfs[i]['project']
    values = clientDfs[i]['duration']
    traces.append(go.Bar(name=uniqueEntries[i],
                         x=labels,
                         y=values
                         #hoverinfo='label+percent+value'
                         #domain=dict(x=[0,0.5])
                  ))
    
  layout = go.Layout(title=title,
                       #annotations=[ann1,ann2],
                       # Hide legend if you want
                       #showlegend=False
                       )
    
  # Create fig with data and layout
  fig = go.Figure(data=traces, layout=layout)
  
  graphFilename = (title + ext_file).replace(" ", "_")
  plotFigure(outputDir + graph_subdir + graphFilename, fig, plotlySettings[1], plotlySettings[2], plotlySettings[3], plotlySettings[4])
    
  # return md for report
  return mdImage("." + os.sep + graph_subdir + graphFilename, title)


###############################################################################
# Report functions
#
def pandasGroupbyMd(pd_report):
    string  = "| Project | Time |\n"
    string += "| ------- | ---- |\n"
    for line in pd_report.splitlines():
        if (not(re.search(r"project", line)) and not(re.search(r"Name:", line))):
            string += "| "
            string += re.sub('\s{2,}', " | ", line).strip()
            string += " |\n"
    return string

def parameterUniqueValues(TimeEntiresDf, reportFilePath,  verbose=2):
  if verbose >= 2:
    dfSize = TimeEntiresDf.shape
    paramList = list(TimeEntiresDf)
    printFile("| {:>11} | {:13} | {:10} | ".format("Parameter", "Unique Values", "Total Rows"), reportFilePath, False, True)
    printFile("|{}|{}|{}|".format(13*"-", 15*"-", 12*"-"), reportFilePath, False, True)
    for param in paramList:
      uniqueEntries = TimeEntiresDf[param].unique()
      uniqueEntryOccurence = uniqueEntries.shape[0]
      printFile("| {:>11} | {:13} | {:10} | ".format(param, uniqueEntryOccurence, dfSize[0]), reportFilePath, False, True)
    printFile("", reportFilePath, False, False)

        
# Split Dataframe by Clients
def getUniqueClientDfs(df):
  uniqueEntries = df['client'].unique()
  clientDfs = []
  for uniqueEntry in uniqueEntries:
    clientDfs.append(df[df['client'] == uniqueEntry])
  return clientDfs, uniqueEntries

  
# Monthly Report
def monthlyReport(df, addMonth, outputDir, md_subdir, graph_subdir, plotlySettings):
  report = ""
  # Filter by month
  (monthlyTimeEntiresDf, monthlystartdate, monthlyenddate) = filterByMonth(df, 'start', addMonth)
  print("{} -> {}".format(monthlystartdate,monthlyenddate))
  # Define output file
  monthlyreportFile = "Month_{}_Report".format(monthlystartdate.strftime('%Y-%m'))
  monthlyreportFilePath = outputDir + md_subdir + monthlyreportFile + ".md"
  report += mdH1("{}".format(monthlyreportFile.replace("_", " "))) + mdLinesep()
   
  # Parameter Unique Values
  parameterUniqueValues(monthlyTimeEntiresDf, monthlyreportFilePath, verbose)
  # Summary of Working Hours
  clientDfs, uniqueEntries = getUniqueClientDfs(monthlyTimeEntiresDf)
  report += mdH2("Monthly Working Hours per Clients and Project") + mdLinesep()
  report += "Timeframe: {} => {}".format(monthlystartdate.strftime('%d-%m-%Y'), monthlyenddate.strftime('%d-%m-%Y')) + mdLinesep()
  for i in range(len(clientDfs)):
    report += mdH3("Client {}".format(uniqueEntries[i])) + mdLinesep()
    clientDfs[i]['duration_hours'] = (clientDfs[i]['duration'].dt.total_seconds() / 3600)
    report += pandasGroupbyMd(str(clientDfs[i].groupby('project')['duration_hours'].sum())) + mdLinesep()
  # Create Visualizations
  # Working Hours Pie Chart
  report += workingHoursPieChart(monthlyTimeEntiresDf, outputDir + md_subdir, graph_subdir, monthlystartdate, timeFrame="Month", plotlySettings=plotlySettings)
  report += mdLinesep() + mdLinesep()
  # Working Hours Bar Chart
  report += workingHoursBarChart(clientDfs, outputDir + md_subdir, graph_subdir, uniqueEntries, monthlystartdate, timeFrame="Month", plotlySettings=plotlySettings)
  report += mdLinesep() + mdLinesep()
  printFile(report, monthlyreportFilePath, True, True, False)
  return monthlyreportFilePath


# Weekly Report
def weeklyReport(df, addWeek, outputDir, md_subdir, graph_subdir, plotlySettings):
  report = ""
  # Filter by week
  (weeklyTimeEntiresDf, weeklystartdate, weeklyenddate) = filterByWeek(df, 'start', addWeek, getTodayDate())
  if verbose >= 2:
    weeklyTimeEntiresDf

  # Define output file
  weeklyreportFile = "Week_{}_{}_Report".format(weeklyenddate.strftime('%Y-%m-%d'), getWeekNbr(weeklystartdate))
  weeklyreportFilePath = outputDir + md_subdir + weeklyreportFile + ".md"
  report += mdH1("{}".format(weeklyreportFile.replace("_", " "))) + mdLinesep()
  
  # Parameter Unique Values
  parameterUniqueValues(weeklyTimeEntiresDf, weeklyreportFilePath, verbose)
  # Summary of Working Hours
  clientDfs, uniqueEntries = getUniqueClientDfs(weeklyTimeEntiresDf)

  report += mdH2("Weekly Working Hours per Clients and Project") + mdLinesep()
  report += "Week Number: {}".format(getWeekNbr(weeklystartdate)) + mdLinesep()
  report += "Timeframe: {} => {}".format(weeklystartdate.strftime('%d-%m-%Y'), weeklyenddate.strftime('%d-%m-%Y')) + mdLinesep()
  for i in range(len(clientDfs)):
    report += mdH3("Client {}".format(uniqueEntries[i])) + mdLinesep()
    clientDfs[i]['duration_hours'] = (clientDfs[i]['duration'].dt.total_seconds() / 3600)
    report += pandasGroupbyMd(str(clientDfs[i].groupby('project')['duration_hours'].sum())) + mdLinesep()
    
  # Create Visualizations
  # Working Hours Pie Chart
  report += workingHoursPieChart(weeklyTimeEntiresDf, outputDir + md_subdir, graph_subdir, weeklystartdate, weeklyenddate, timeFrame="Week", plotlySettings=plotlySettings)
  report += mdLinesep() + mdLinesep()
  # Working Hours Bar Chart
  report += workingHoursBarChart(clientDfs, outputDir + md_subdir, graph_subdir, uniqueEntries, weeklystartdate, weeklyenddate, timeFrame="Week", plotlySettings=plotlySettings)
  report += mdLinesep() + mdLinesep()
  printFile(report, weeklyreportFilePath, True, True, False)
    
  return weeklyreportFilePath


def generateAllReports(srcdir, outputDir):
  dirlist = os.listdir(srcdir)
  previous_dir = os.getcwd()
  os.chdir(srcdir)
  for file in dirlist:
    if re.search(".md", file):
      cmd = "markdown-pdf {}".format(file)
      print(cmd)
      if os.system(cmd) == 0:
        print("  * PDF Report {} generated".format(file.replace(".md", ".pdf")))
      else:
        print("  * PDF Report {} failed!!!!".format(file.replace(".md", ".pdf")))
  os.chdir(previous_dir)
  moveFiles(os.path.abspath(srcdir), os.path.abspath(outputDir), ".pdf")

    
def generateReport(src_file_path, outputDir):
  previous_dir = os.getcwd()
  os.chdir(os.path.dirname(src_file_path))
  if re.search(".md", src_file_path):
    cmd = "markdown-pdf {}".format(os.path.basename(src_file_path))
    print(cmd)
    if os.system(cmd) == 0:
      print("  * PDF Report {} generated".format(os.path.basename(src_file_path).replace(".md", ".pdf")))
    else:
      print("  * PDF Report {} failed!!!!".format(os.path.basename(src_file_path).replace(".md", ".pdf")))
  os.chdir(previous_dir)
  moveFiles(os.path.abspath(os.path.dirname(src_file_path)), os.path.abspath(outputDir), ".pdf")