#!/usr/bin/env python3
###############################################################################
#          __ _____            __
#   /\  /\/__\\_   \   /\   /\/ _\
#  / /_/ /_\   / /\/___\ \ / /\ \
# / __  //__/\/ /_|_____\ V / _\ \
# \/ /_/\__/\____/       \_/  \__/
# (c) - zas
###############################################################################
# Dashboard - base config
# **Copyright (C) 2019 HES-SO Valais-Wallis - All Rights Reserved**
###############################################################################
import enum
import pandas as pd

from hevslib.plotly import *

###############################################################################
# Constants
#
# 0 = no output
# 1 = normal output
# 2 = verbose output
verbose = 2



# Data Storage constants
data_inputDir = "in"
data_outputDir = "out"

# Project Definition
projectListFile = "projectList.xls"
projectCol = {"project_number": "project_number",
              "acronym": "acronym",
              "title_humanreadable": "title_humanreadable",
              "budget_total": "budget_total",
              "budget_material": "budget_material",
              "date_begin": "date_begin",
              "date_end": "date_end",
              "updated": "updated",
              }

# activity constans
activityData = []
activityCol = ["german", "french"]
activityData = [["Allgemeine administrative Arbeiten", "Tâches administratives générales"],
                ["Durchführung von aF&E Projekten oder -Arbeiten", "Réalisation de projects ou d'activités Ra&D"],
                ["Durchführung von Mandaten", "Réalisation de mandats"],
                ["Projektakquisition", "Acquisition de projects"],
                ["Unterricht (Anstellungsplan)", "Cours dispensé (plan d'engagement)"],
                ["Unterrichtsspezifische Aufgaben (1.4)", "Tâche spécifiques d'enseignement"],
                ["Weiterbildung", "Formation"],
                ]
activityDf = pd.DataFrame(activityData, columns=activityCol)

# Input ash file config
ashSheetname = "Sheet1"

ashCols = {'german': {"date": "Datum",
                      "collaborator": "Mitarbeiter",
                      "activity": "Tätigkeit",
                      "rate": "Zugeteilter Satz",
                      "hours": "Anz. Stunden",
                      "amount": "Betrag",
                      "description": "Beschrieb",
                     },
          'french': {"date": "Date",
                     "collaborator": "Collaborateur",
                     "activity": "Activité",
                     "rate": "Taux affecté",
                     "hours": "Nb heures",
                     "amount": "Montant",
                     "description": "Description"
                    }
          }


###############################################################################
# Graph output Options
#
notebookGraphicInteraction = GraphInteractionOption('static')
notebookGraphicOutputs = GraphOutputOption('both')

ext_file = ".svg"

staticImageSize = {'width': 1280, 'height': 720, 'scale': 1}

GraphAutoOpenHTML = False  # Auto open external HTML files [True/False]
