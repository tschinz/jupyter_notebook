#!/usr/bin/env python3
# Dashboard Config

###############################################################################
# Confidentiality
# **All information in this document is strictly confidiental**
# **Copyright (C) 2020 HES-SO Valais-Wallis - All Rights Reserved**

import enum
import pandas as pd

###############################################################################
# Constants
#
# Month - Day
month2year = 1/12
year2month = 12
day2month = 0.03285421
month2day = 1/day2month

# Data Storage constants
data_inputDir = "in"
data_outputDir = "out"

# Project Definition
projectListFile = "projectList.xls"
projectListColumns = ["project_number", "acronym", "title_humanreadable", "budget_total", "budget_material", "date_begin", "date_end"]

# activity constans
activityData = []
activityDfColumns = ["german", "french"]
activityData = [["Allgemeine administrative Arbeiten", "Tâches administratives générales"],
                ["Durchführung von aF&E Projekten oder -Arbeiten", "Réalisation de projects ou d'activités Ra&D"],
                ["Durchführung von Mandaten", "Réalisation de mandats"],
                ["Projektakquisition", "Acquisition de projects"],
                ["Unterricht (Anstellungsplan)", "Cours dispensé (plan d'engagement)"],
                ["Unterrichtsspezifische Aufgaben (1.4)", "Tâche spécifiques d'enseignement"],
                ["Weiterbildung", "Formation"],
               ]
activityDf = pd.DataFrame(activityData, columns=activityDfColumns)

# Input ash file config
ash_sheetname = "Sheet1"

ash_columns = { 'german' : ["Datum",
                  "Mitarbeiter",
                  "Tätigkeit",
                  "Zugeteilter Satz",
                  "Anz. Stunden",
                  "Betrag",
                  "Beschrieb",                  
                 ],
               'french' : ["Date",
                  "Collaborateur",
                  "Activité",
                  "Taux affecté",
                  "Nb heures",
                  "Montant",
                  "Description"
                  ]
              }

###############################################################################
# Graph output Options
#
class GraphOutputOption(enum.Enum):
    none = 'none'                # Do not generate any plots
    inline = 'inline'            # Generate inline plots only
    htmlFile = 'extFile'         # Generate plots in external file (html|png|...)
    both = 'both'                # Generate all plots inline and external


class GraphInteractionOption(enum.Enum):
    static = 'static'            # Generate static inline plots (as images)
    interactive = 'interactive'  # Generate interactive inline plots


notebookGraphicInteraction = GraphInteractionOption('static')
notebookGraphicOutputs = GraphOutputOption('both')

ext_file = ".svg"

staticImageSize = {'width':1280, 'height':720, 'scale':1}

GraphAutoOpenHTML = False        # Auto open external HTML files [True/False]
