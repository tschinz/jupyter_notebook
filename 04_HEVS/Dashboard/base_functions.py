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
from hevslib.md import *


###############################################################################
# Custom Dashboard Plot figures
###############################################################################
def projectPlotCombined(df, df_col, projectDf, projectCol, projectConf, outputGraphDir, plotlySettings):
  fig = go.Figure()
  fig = make_subplots(specs=[[{"secondary_y": True}]])
  title = projectConf[0] + " " + filterRows(projectDf, [projectCol["project_number"], [projectConf[0]]])[
            projectCol["acronym"]].iloc[0] + " Project Overview"
  title_graph = "Project Overview"
  # Add member data
  for member in df[df_col["collaborator"]].unique():
    labels = df.loc[df[df_col["collaborator"]] == member][df_col["date"]]
    values = df.loc[df[df_col["collaborator"]] == member][df_col["hours"]]
    fig.add_trace(go.Bar(x=labels, y=values, name=member), secondary_y=False)#, texttemplate="%{label}", textposition="outside"),
                  

  # Add global data
  labels = df[df_col["date"]]
  values = df[df_col['total_budget']]
  fig.add_trace(go.Scatter(x=labels, y=values, name=df_col['total_budget']), secondary_y=True)
  values = df[df_col['monthly_budget']]
  fig.add_trace(go.Scatter(x=labels, y=values, name=df_col['monthly_budget']), secondary_y=True)
  values = df[df_col['remaining_budget']]
  fig.add_trace(go.Scatter(x=labels, y=values, name=df_col['remaining_budget']), secondary_y=True)

  # Update Graph
  fig.update_layout(barmode='stack',
                    xaxis={'categoryorder': 'total descending', 'type': 'date'},
                    title={'text': title_graph, 'x': 0.5, 'y': 0.9},
                    xaxis_title=df_col["date"],
                    yaxis_title=df_col["hours"],
                    )
  fig.update_yaxes(title_text=df_col["hours"], secondary_y=False)
  fig.update_yaxes(title_text='CHF', secondary_y=True)
  graphFilename = (title + plotlySettings[0]).replace(" ", "_")
  plot_figure(outputGraphDir + graphFilename, fig, plotlySettings[1], plotlySettings[2], plotlySettings[3], plotlySettings[4])
  
  return outputGraphDir + graphFilename
  
def projectBarBudget(df, df_col, projectDf, projectCol, projectConf, outputGraphDir, plotlySettings):
  fig = go.Figure()
  title = projectConf[0] + " " + filterRows(projectDf, [projectCol["project_number"], [projectConf[0]]])[
            projectCol["acronym"]].iloc[0] + " Project Budget"
  title_graph = "Project Budget"

  labels = df[df_col["date"]]
  values = df[df_col['total_budget']]
  fig.add_trace(go.Scatter(x=labels, y=values, name=df_col['total_budget']))
  values = df[df_col['monthly_budget']]
  fig.add_trace(go.Scatter(x=labels, y=values, name=df_col['monthly_budget']))
  values = df[df_col['remaining_budget']]
  fig.add_trace(go.Scatter(x=labels, y=values, name=df_col['remaining_budget']))

  # Update Graph
  fig.update_layout(barmode='stack',
                    xaxis={'categoryorder': 'total descending', 'type': 'date'},
                    title={'text': title_graph, 'x': 0.5, 'y': 0.9},
                    xaxis_title=df_col["date"],
                    yaxis_title=df_col["hours"],
                    )
  fig.update_yaxes(title_text='CHF')
  graphFilename = (title + plotlySettings[0]).replace(" ", "_")
  plot_figure(outputGraphDir + graphFilename, fig, plotlySettings[1], plotlySettings[2], plotlySettings[3], plotlySettings[4])
  return outputGraphDir + graphFilename
  
def projectLinesHours(df, df_col, projectDf, projectCol, projectConf, outputGraphDir, ext_file):
  fig = go.Figure()
  title = projectConf[0] + " " + filterRows(projectDf, [projectCol["project_number"], [projectConf[0]]])[
            projectCol["acronym"]].iloc[0] + " Project Hours"
  title_graph = "Project Hours"
  # Add member data
  for member in df[df_col["collaborator"]].unique():
    labels = df.loc[df[df_col["collaborator"]] == member][df_col["date"]]
    values = df.loc[df[df_col["collaborator"]] == member][df_col["hours"]]
    fig.add_trace(go.Bar(x=labels, y=values, name=member))#, texttemplate="%{label}", textposition="outside"))

  # Update Graph
  fig.update_layout(barmode='stack',
                    xaxis={'categoryorder': 'total descending', 'type': 'date'},
                    title={'text': title_graph, 'x': 0.5, 'y': 0.9},
                    xaxis_title=df_col["date"],
                    yaxis_title=df_col["hours"],
                    )
  fig.update_yaxes(title_text=df_col["hours"])
  graphFilename = (title + plotlySettings[0]).replace(" ", "_")
  plot_figure(outputGraphDir + graphFilename, fig, plotlySettings[1], plotlySettings[2], plotlySettings[3], plotlySettings[4])
  return outputGraphDir + graphFilename

def projectPieCollaborators(df1, df2, df_col, projectDf, projectCol, projectConf, outputGraphDir, plotlySettings):
  title = "Budget by Collaborator"
  labels = df2[df_col['collaborator']]
  values = df2[df_col['amount']]
  trace1 = go.Pie(title=title,
                  labels=labels,
                  values=values,
                  
                  textinfo='label+percent+value',
                  hoverinfo='label+percent+value',
                  domain=dict(x=[0.0,0.5]))
    
  title = "Hours by Collaborator"
  labels = df2[df_col['collaborator']]
  values = df2[df_col['hours']]
  trace2 = go.Pie(title=title,
                  labels=labels,
                  values=values,
                  textinfo='label+percent+value',
                  hoverinfo='label+percent+value',
                  domain=dict(x=[0.5,1.0]))
  
  title = "Split by Collaborator"  
  layout = go.Layout(title={'text': title, 'x': 0.5, 'y': 0.9},
                     #annotations=[ann1,ann2],
                     # Hide legend if you want
                     #showlegend=False
                     )
  data = [trace1, trace2]
  # Create fig with data and layout
  fig = go.Figure(data=data,layout=layout)
    
  graphFilename = (title + plotlySettings[0]).replace(" ", "_")
  plot_figure(outputGraphDir + graphFilename, fig, plotlySettings[1], plotlySettings[2], plotlySettings[3], plotlySettings[4])
  return outputGraphDir + graphFilename


def projectPieBudgetActivity(df1, df3, df_col, projectDf, projectCol, projectConf, outputGraphDir, plotlySettings):
  title = "Budget"
  labels = ['HEI', 'Material', 'Extern', 'Remaining']
  budget_total = filterRows(projectDf, [projectCol["project_number"], [projectConf[0]]])[projectCol["budget_total"]].iloc[0]
  budget_material = filterRows(projectDf, [projectCol["project_number"], [projectConf[0]]])[projectCol["budget_material"]].iloc[0]
  budget_extern = filterRows(projectDf, [projectCol["project_number"], [projectConf[0]]])[projectCol["budget_extern"]].iloc[0]
  budget_remaining = df3[df_col['remaining_budget']].iloc[-1]
  budget_hei = budget_total - budget_material- budget_extern - budget_remaining
  budget = [budget_hei, budget_material, budget_extern, budget_remaining]
  values = budget
  trace1 = go.Pie(title=title,
                  labels=labels,
                  values=values,
                  textinfo='label+percent+value',
                  hoverinfo='label+percent+value',
                  domain=dict(x=[0.0,0.4]))
  
  title = "Activity by Collaborator"
  uniqueEntries = df1[df_col['activity']].unique()
  uniqueEntryOccurence = uniqueEntries.shape[0]
  nbrOfOccurences = []
  for uniqueEntry in np.sort(uniqueEntries):
    nbrOfOccurences.append(df1.loc[df1[df_col['activity']] == uniqueEntry].shape[0])
  labels = np.sort(uniqueEntries)
  values = nbrOfOccurences
  trace2 = go.Pie(title=title,
                  labels=labels,
                  values=values,
                  textinfo='label+percent',
                  hoverinfo='label+percent+value',
                  domain=dict(x=[0.6,1.0]))
  
  title = projectConf[0] + " - " + filterRows(projectDf, [projectCol["project_number"], [projectConf[0]]])[projectCol["acronym"]].iloc[0] + " Project Statistics"
  layout = go.Layout(title={'text': title, 'x': 0.5, 'y': 0.9},
                     #annotations=[ann1,ann2],
                     # Hide legend if you want
                     showlegend=False
                     )
  data = [trace1, trace2]
  # Create fig with data and layout
  fig = go.Figure(data=data,layout=layout)
    
  graphFilename = (title + plotlySettings[0]).replace(" ", "_")
  plot_figure(outputGraphDir + graphFilename, fig, plotlySettings[1], plotlySettings[2], plotlySettings[3], plotlySettings[4])
  return outputGraphDir + graphFilename

def projectPieCollaborators(df2, df_col, projectDf, projectCol, projectConf, outputGraphDir, plotlySettings):
  title = "Budget by Collaborator"
  labels = df2[df_col['collaborator']]
  values = df2[df_col['amount']]
  trace1 = go.Pie(title=title,
                  labels=labels,
                  values=values,
                  textinfo='label+percent+value',
                  hoverinfo='label+percent+value',
                  domain=dict(x=[0,0.4]))
    
  title = "Hours by Collaborator"
  labels = df2[df_col['collaborator']]
  values = df2[df_col['hours']]
  trace2 = go.Pie(title=title,
                  labels=labels,
                  values=values,
                  textinfo='label+percent+value',
                  hoverinfo='label+percent+value',
                  domain=dict(x=[0.6,1.0]))
  title = projectConf[0] + " - " + filterRows(projectDf, [projectCol["project_number"], [projectConf[0]]])[projectCol["acronym"]].iloc[0] + " Split by Collaborator"  
  layout = go.Layout(title={'text': title, 'x': 0.5, 'y': 0.9},
                     #annotations=[ann1,ann2],
                     # Hide legend if you want
                     showlegend=False
                     )
  data = [trace1, trace2]
  # Create fig with data and layout
  fig = go.Figure(data=data,layout=layout)
    
  graphFilename = (title + plotlySettings[0]).replace(" ", "_")
  plot_figure(outputGraphDir + graphFilename, fig, plotlySettings[1], plotlySettings[2], plotlySettings[3], plotlySettings[4])
  return outputGraphDir + graphFilename


###############################################################################
# Reports
###############################################################################
def projectReport(df1, df2, df3, ash_col, projectDf, projectCol, projectConf, outputMdDir, outputPdfDir, plotlySettings, verbose):
  # create graph subdir
  img_subdir = "img" + os.sep
  outputImgDir = outputMdDir + img_subdir
  createDir(os.path.realpath(outputImgDir))
  
  mdContent = ""
  # Title
  mdContent += mdH1("Project Report - {} - {}".format(projectConf[0], filterRows(projectDf, [projectCol["project_number"], [projectConf[0]]])[projectCol["title_humanreadable"]].iloc[0]))
  # Basic Stats
  basicStats = []
  basicStats.append("Report Data exported: {}".format(filterRows(projectDf, [projectCol["project_number"], [projectConf[0]]])[projectCol["updated"]].iloc[0]))
  basicStats.append("Project begin: {}".format(filterRows(projectDf, [projectCol["project_number"], [projectConf[0]]])[projectCol["date_begin"]].iloc[0]))#.dt.strftime('%d.%m.%Y')))
  basicStats.append("Project end: {}".format(filterRows(projectDf, [projectCol["project_number"], [projectConf[0]]])[projectCol["date_end"]].iloc[0]))#.strftime('%d.%m.%Y')))
  basicStats.append("HEI Project Budget: {:d} CHF".format(int(df3[ash_col['total_budget']].iloc[0])))
  basicStats.append("HEI Budget Remaining: {:d} CHF".format(int(df3[ash_col['remaining_budget']].iloc[-1])))
  basicStats.append("HEI Montlhy Budget: {:d} CHF".format(int(df3[ash_col['monthly_budget']].iloc[0])))
  basicStats.append("HEI Month Remaining: ~{:.2f} Months".format(df3[ash_col['remaining_budget']].iloc[0]/df3[ash_col['monthly_budget']].iloc[0]))
  mdContent += mdList(basicStats) + mdlinesep()
  
  # Pie charts
  
  imagePath = projectPieBudgetActivity(df1, df3, ash_col, projectDf, projectCol, projectConf, outputImgDir, plotlySettings)
  mdContent += mdImage("." + os.sep + img_subdir + os.path.basename(imagePath), "Project Pie Budget") + mdlinesep()
  imagePath = projectPieCollaborators(df2, ash_col, projectDf, projectCol, projectConf, outputImgDir, plotlySettings)
  mdContent += mdImage("." + os.sep + img_subdir + os.path.basename(imagePath), "Project Pie Collaborators") + mdlinesep()
    
  # Bar charts
  imagePath = projectPlotCombined(df3, ash_col, projectDf, projectCol, projectConf, outputImgDir, plotlySettings)
  mdContent += mdImage("." + os.sep + img_subdir + os.path.basename(imagePath), "Project Plot Combined") + mdlinesep()
  imagePath = projectBarBudget(df3, ash_col, projectDf, projectCol, projectConf, outputImgDir, plotlySettings)
  mdContent += mdImage("." + os.sep + img_subdir + os.path.basename(imagePath), "Project Plot Budget") + mdlinesep()
  imagePath = projectLinesHours(df3, ash_col, projectDf, projectCol, projectConf, outputImgDir, plotlySettings)  
  mdContent += mdImage("." + os.sep + img_subdir + os.path.basename(imagePath), "Project Plot Hours") + mdlinesep()

  mdContent += mdlinesep() + mdlinesep() + mdItalics("Report automatically generated - (c) zas")
  
  filePath = (outputMdDir + projectConf[0] + "_" + filterRows(projectDf, [projectCol["project_number"], [projectConf[0]]])[projectCol["title_humanreadable"]].iloc[0]).replace(" ", "_")
  mdFilePath = (filePath + ".md")
  
  print_file(mdContent, mdFilePath, fileoutput=True, consoleoutput=False,  append=False)
  generate_report(mdFilePath, outputPdfDir, verbose)

  if verbose >=1:
     print("Report created: " + filePath + ".pdf")