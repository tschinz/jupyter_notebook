#!/usr/bin/env python3
# Digital Twin- Base Config

###############################################################################
# Confidentiality
# **All information in this document is strictly confidiental**
# **Copyright (C) 2019 HES-SO Valais-Wallis - All Rights Reserved**

import enum

###############################################################################
# Constants
#
# 01 Data Preparation
opencv_inputDir = "in/"
opencv_outputDir = "out/"

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


notebookGraphingInteraction = GraphInteractionOption('static')
notebookGraphingOutputs = GraphOutputOption('inline')

ext_file = ".html"

staticImageSize = {'width': 1000, 'height': 500, 'scale': 1}

GraphAutoOpenHTML = False        # Auto open external HTML files [True/False]
