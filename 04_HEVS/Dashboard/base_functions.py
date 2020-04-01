#!/usr/bin/env python3
###############################################################################
#          __ _____            __
#   /\  /\/__\\_   \   /\   /\/ _\
#  / /_/ /_\   / /\/___\ \ / /\ \
# / __  //__/\/ /_|_____\ V / _\ \
# \/ /_/\__/\____/       \_/  \__/
# (c) - zas
###############################################################################
# Dashboard - Custom functions
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

###############################################################################
# Custom Dashboard Plot figures
###############################################################################
def project_plot_combined(df, df_col, projectList, projectListColumns, projectConf, outputGraphDir, ext_file):
  fig = go.Figure()
  fig = make_subplots(specs=[[{"secondary_y": True}]])
  title = datetime.date.today().strftime("%Y.%m.%d") + " " + projectConf[0] + " " + \
          filterRows(projectList, [projectListColumns["project_number"], [projectConf[0]]])[
            projectListColumns["acronym"]].iloc[0] + " - Project Overview"
  title_long = datetime.date.today().strftime("%d.%m.%Y") + " - " + projectConf[0] + " - " + \
               filterRows(projectList, [projectListColumns["project_number"], [projectConf[0]]])[
                 projectListColumns["title_humanreadable"]].iloc[0] + " - Project Overview"
  # Add member data
  for member in df[df_col["collaborator"]].unique():
    labels = df.loc[df[df_col["collaborator"]] == member][df_col["date"]]
    values = df.loc[df[df_col["collaborator"]] == member][df_col["hours"]]
    fig.add_trace(go.Bar(x=labels, y=values, name=member, texttemplate="%{label}", textposition="inside"),
                  secondary_y=False)

  # Add global data
  labels = df[df_col["date"]]
  values = df[df_col['total_budget']]
  fig.add_trace(go.Scatter(x=labels, y=values, name=df_col['total_budget']), secondary_y=True)
  values = df[df_col['monthy_budget']]
  fig.add_trace(go.Scatter(x=labels, y=values, name=df_col['monthy_budget']), secondary_y=True)
  values = df[df_col['remaining_budget']]
  fig.add_trace(go.Scatter(x=labels, y=values, name=df_col['remaining_budget']), secondary_y=True)

  # Update Graph
  fig.update_layout(barmode='stack',
                    xaxis={'categoryorder': 'total descending', 'type': 'date'},
                    title={'text': title_long, 'x': 0.5, 'y': 0.9},
                    xaxis_title=df_col["date"],
                    yaxis_title=df_col["hours"],
                    )
  fig.update_yaxes(title_text=df_col["hours"], secondary_y=False)
  fig.update_yaxes(title_text='CHF', secondary_y=True)
  graphFilename = (title + ext_file).replace(" ", "_")
  plot_figure(outputGraphDir + graphFilename, fig)

  
def project_plot_budget(df, df_col, projectList, projectListColumns, projectConf, outputGraphDir, ext_file):
  fig = go.Figure()
  title = projectConf[0] + " " + filterRows(projectList, [projectListColumns["project_number"], [projectConf[0]]])[
            projectListColumns["acronym"]].iloc[0] + " Project Budget"
  title_long = datetime.date.today().strftime("%d.%m.%Y") + " - " + projectConf[0] + " - " + \
               filterRows(projectList, [projectListColumns["project_number"], [projectConf[0]]])[
                 projectListColumns["title_humanreadable"]].iloc[0] + " - Project Budget"

  labels = df[df_col["date"]]
  values = df[df_col['total_budget']]
  fig.add_trace(go.Scatter(x=labels, y=values, name=df_col['total_budget']))
  values = df[df_col['monthy_budget']]
  fig.add_trace(go.Scatter(x=labels, y=values, name=df_col['monthy_budget']))
  values = df[df_col['remaining_budget']]
  fig.add_trace(go.Scatter(x=labels, y=values, name=df_col['remaining_budget']))

  # Update Graph
  fig.update_layout(barmode='stack',
                    xaxis={'categoryorder': 'total descending', 'type': 'date'},
                    title={'text': title_long, 'x': 0.5, 'y': 0.9},
                    xaxis_title=df_col["date"],
                    yaxis_title=df_col["hours"],
                    )
  fig.update_yaxes(title_text='CHF')
  graphFilename = (title + ext_file).replace(" ", "_")
  plot_figure(outputGraphDir + graphFilename, fig)

  
def project_plot_hours(df, df_col, projectList, projectListColumns, projectConf, outputGraphDir, ext_file):
  fig = go.Figure()
  title = datetime.date.today().strftime("%Y.%m.%d") + " " + projectConf[0] + " " + \
          filterRows(projectList, [projectListColumns["project_number"], [projectConf[0]]])[
            projectListColumns["acronym"]].iloc[0] + " - Project Hours"
  title_long = datetime.date.today().strftime("%d.%m.%Y") + " - " + projectConf[0] + " - " + \
               filterRows(projectList, [projectListColumns["project_number"], [projectConf[0]]])[
                 projectListColumns["title_humanreadable"]].iloc[0] + " - Project Hours"
  # Add member data
  for member in df[df_col["collaborator"]].unique():
    labels = df.loc[df[df_col["collaborator"]] == member][df_col["date"]]
    values = df.loc[df[df_col["collaborator"]] == member][df_col["hours"]]
    fig.add_trace(go.Bar(x=labels, y=values, name=member, texttemplate="%{label}", textposition="inside"))

  # Update Graph
  fig.update_layout(barmode='stack',
                    xaxis={'categoryorder': 'total descending', 'type': 'date'},
                    title={'text': title_long, 'x': 0.5, 'y': 0.9},
                    xaxis_title=df_col["date"],
                    yaxis_title=df_col["hours"],
                    )
  fig.update_yaxes(title_text=df_col["hours"])
  graphFilename = (title + ext_file).replace(" ", "_")
  plot_figure(outputGraphDir + graphFilename, fig)

  
  
  # Add to report
  #print_file("![{}]({})".format(title, "." + os.sep + graph_subdir + graphFilename), reportFilePath, True, False)
  #print_file("", reportFilePath, True, False)


###############################################################################
# Reports
###############################################################################
def projectReport(df, addMonth):
  # Filter by month
  (monthlyTimeEntiresDf, monthlystartdate, monthlyenddate) = filterByMonth(df, addMonth)
  # Define output file
  monthlyreportFile = "Month_{}_Report".format(monthlystartdate.strftime('%Y-%m'))
  monthlyreportFilePath = graphFilename = outputDir + md_subdir + monthlyreportFile + ".md"
  print_file("# {}".format(monthlyreportFile.replace("_", " ")), monthlyreportFilePath, True, False, False)
  print_file("", monthlyreportFilePath, True, False)

  # Parameter Unique Values
  parameterUniqueValues(monthlyTimeEntiresDf, monthlyreportFilePath, verbose)
  # Summary of Working Hours
  clientDfs, uniqueEntries = getUniqueClientDfs(monthlyTimeEntiresDf)

  print_file("## Monthly Working Hours per Clients and Project", monthlyreportFilePath, True, True)
  print_file("Timeframe: {} => {}".format(monthlystartdate.strftime('%d-%m-%Y'), monthlyenddate.strftime('%d-%m-%Y')),
             monthlyreportFilePath, False, True)
  for i in range(len(clientDfs)):
    print_file("### Client {}".format(uniqueEntries[i]), monthlyreportFilePath, True, True)
    clientDfs[i]['duration_hours'] = (clientDfs[i]['duration'].dt.total_seconds() / 3600)
    print_file(panda_groupby_md(str(clientDfs[i].groupby('project')['duration_hours'].sum())), monthlyreportFilePath,
               True, True)
    print_file("", monthlyreportFilePath, True, True)

  # Create Visualizations
  # Working Hours Pie Chart
  workingHoursPieChart(monthlyTimeEntiresDf, monthlyreportFilePath, monthlystartdate, timeFrame="Month")
  # Working Hours Bar Chart
  workingHoursBarChart(clientDfs, monthlyreportFilePath, uniqueEntries, monthlystartdate, timeFrame="Month")

  return monthlyreportFilePath
