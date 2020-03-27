# python
import sys
import os
import enum
import datetime
from dateutil.parser import parse
import time
import pytz
import json
import re

# iPython
import IPython
from IPython.display import display
from IPython.display import Image

# pandas
import pandas as pd

# numpy
import numpy as np

# plotly
import plotly as ply
import plotly.graph_objs as go
import plotly.figure_factory as ff
ply.offline.init_notebook_mode(connected=True)
import plotly.io as pio

#--------------------------------------------------------------
# My Functions
#--------------------------------------------------------------

#--------------------------------------------------------------
# General Functions
#--------------------------------------------------------------
def createDir(directory):
    if (os.path.exists(directory)) is False:
        os.makedirs(directory)
    if (os.path.isdir(directory)) is False:
        raise NotADirectoryError("{} is not a directory".format(directory))

def print_file(text="", file=None, fileoutput=True, consoleoutput=True, append=True):
    if fileoutput:
        if append:
            with open(file, "a+") as file:
                file.write(text+"\n")#os.linesep)
        else:
            with open(file, "w+") as file:
                file.write(text+"\n")#os.linesep)
    if consoleoutput:
        print(text)

#--------------------------------------------------------------
# Time Functions
#--------------------------------------------------------------
def fill_clean_timeentries(df, projects, user_clients):
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

def get_join_date(user):
    join_date = parse(user['data']['created_at'])
    join_date = join_date.replace(tzinfo=None)
    return join_date

def get_today_date():
    today = datetime.datetime.now()
    return today

def get_last_dom(any_day):
    next_month = any_day.replace(day=28) + datetime.timedelta(days=4)
    return next_month - datetime.timedelta(days=next_month.day)

def get_first_dom(any_day):
    any_day = any_day.replace(day=1)
    return any_day

def get_month(addMonth = 0):
    if addMonth > 0:
        date = get_today_date() + datetime.timedelta(addMonth*365/12)
    elif addMonth < 0:
        date = get_today_date() - datetime.timedelta(-1*addMonth*365/12)
    else:
        date = get_today_date()
    startdate = get_first_dom(date)
    enddate = get_last_dom(date)
    return startdate, enddate

def get_week(date, addWeek = 0):
    if addWeek > 0:
        startdate = date - datetime.timedelta(days=date.weekday()) + datetime.timedelta(days=addWeek*7)
    elif addWeek < 0:
        startdate = date - datetime.timedelta(days=date.weekday()) - datetime.timedelta(days=-1*addWeek*7)
    else:
        startdate = date - datetime.timedelta(days=date.weekday())
    enddate = startdate + datetime.timedelta(days=6)
    return startdate, enddate

def get_weekNbr(date):
    return date.strftime("%V")

# Function that turns datetimes back to strings since that's what the API likes
def date_only(datetimeVal):
    datePart = datetimeVal.strftime("%Y-%m-%d")
    return datePart

def filterByMonth(df, addMonth = 0):
    (startdate, enddate) = get_month(addMonth)
    filteredTimeEntriesDf = df[(df['start'] > startdate.strftime('%Y-%m-%d')) & (df['start'] < enddate.strftime('%Y-%m-%d'))]
    return filteredTimeEntriesDf, startdate, enddate
    #timeEntriesDf[(timeEntriesDf['start'] > '2019-03-23 07:30:00') & (timeEntriesDf['start'] < '2019-04-23 09:00:00')]
    #timeEntriesDf[(timeEntriesDf['start'] > '2019-03-23') & (timeEntriesDf['start'] < '2019-04-23')]

def filterByWeek(df, date, addWeek = 0):
    (startdate, enddate) = get_week(get_today_date(), addWeek)
    filteredTimeEntriesDf = df[(df['start'] > startdate.strftime('%Y-%m-%d')) & (df['start'] < enddate.strftime('%Y-%m-%d'))]
    return filteredTimeEntriesDf, startdate, enddate

# Extract Timelogs Between Two Dates and Export to a CSV
def toggl_timelog_extractor(toggl, toggl_time_entries_url, input_date1, input_date2, save_csv=False, outputdir='output'):
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

#--------------------------------------------------------------
# Pandas Functions
#--------------------------------------------------------------
# Beautify Panda Groupby report
def panda_groupby_md(pd_report):
    string  = "| Project | Time |\n"
    string += "| ------- | ---- |\n"
    for line in pd_report.splitlines():
        if (not(re.search(r"project", line)) and not(re.search(r"Name:", line))):
            string += "| "
            string += re.sub('\s{2,}', " | ", line).strip()
            string += " |\n"
    return string

def saveDf(df, outputDir):
    file_name = "{}{}-zas-timerecords.csv".format(outputDir, get_today_date().strftime('%Y-%m-%d'))
    print("  * DataFrame save to {}".format(file_name))
    df.to_csv(file_name, encoding='utf-8')

