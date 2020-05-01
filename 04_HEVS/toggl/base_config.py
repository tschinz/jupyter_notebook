#!/usr/bin/env python3
###############################################################################
#          __ _____            __
#   /\  /\/__\\_   \   /\   /\/ _\
#  / /_/ /_\   / /\/___\ \ / /\ \
# / __  //__/\/ /_|_____\ V / _\ \
# \/ /_/\__/\____/       \_/  \__/
# (c) - zas
###############################################################################
# toggl - base config
# **Copyright (C) 2019 HES-SO Valais-Wallis - All Rights Reserved**
###############################################################################

# hevslib
import sys 
sys.path.append("hevslib")
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

###############################################################################
# Graph output Options
#
#notebookGraphicInteraction = GraphInteractionOption('interactive')
notebookGraphicInteraction = GraphInteractionOption('static')
notebookGraphicOutputs = GraphOutputOption('both')

ext_file = ".svg"
#ext_file = ".png"

staticImageSize = {'width': 1000, 'height': 400, 'scale': 1}

GraphAutoOpenHTML = False  # Auto open external HTML files [True/False]

plotlySettings = [ext_file, staticImageSize, notebookGraphicOutputs, notebookGraphicInteraction, GraphAutoOpenHTML]
