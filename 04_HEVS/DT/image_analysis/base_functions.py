#!/usr/bin/env python3
# Data Science - Base functions

###############################################################################
# Confidentiality
# **All information in this document is strictly confidiental**
# **Copyright (C) 2019 HES-SO Valais-Wallis - All Rights Reserved**

###############################################################################
# Import sub-modules
# Import required sub-modules

# python
# import sys
import os
import os.path
# import enum
import datetime
import re  # noqa
import math

import string
import random

# iPython
from IPython.display import display
from IPython.display import HTML

# color
from sty import fg, Style, RgbFg

# pandas
import pandas as pd

# numpy
import numpy as np

# matplotlib and seaborn
import matplotlib as mpl
import matplotlib.pyplot as plt  # noqa
import seaborn as sns  # noqa

# plotly
import plotly as ply # noqa
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from plotly.offline import plot  # noqa
from plotly.subplots import make_subplots  # noqa
import plotly.express as px  # noqa

# watermark
import watermark

# wavedrom
import nbwavedrom

# Plotly options
# ply.offline.init_notebook_mode(connected=True)

# Pandas output options
pd.set_option('display.max_rows', 200)
pd.set_option('display.max_columns', 150)
pd.set_option('display.width', 500)
pd.set_option('display.max_info_columns', 150)

# Add custom colors for print_color function:
fg.orange = Style(RgbFg(255, 150, 50))
fg.purple = Style(RgbFg(150, 50, 255))

###############################################################################
# Common functions
###############################################################################
def createScoreGraph(df, thickness, time_column, ruleset, target, target_quality_order, verbose=False):
    """ Create graph showing percentage of excellent/normal plate

    Args:
        df: The dataframe containing the data to display
        time_column: The column to use for x in the scatter plot
        ml_rule: List of rules that will be used to filter plates from df
        rule_type: List of the type of the rules (equal to, less than, etc..)
        target: The target that will be used to differentiate excellent/normal plates
        target_quality_order: Dictionary that indicate if a plate is excellent when the
                              target is below (0) or above (1) the median
    Returns:
        None
    Raises:
        None
    """
    df_raw = df.copy()
    ml_rule, rule_type = formatRuleListForFilter(ruleset, verbose=False)

    # filter for thickness
    print("Filter dataframe for thickness {}".format(thickness))
    df_thickness = filterRows(df_raw, ["EPAISSEUR_FINALE", [thickness]], type="eq", verbose=False)

    # Clean column of df that are affected by ruleset
    df_clean = df_thickness.copy()
    feature_list = [item[0] for item in ml_rule]
    if len(feature_list) > 0:
        df_clean = removeNaN(df_clean, feature_list, False)

    # Calculate score
    print("Calculate rule score: ", ml_rule, rule_type)
    # score, nbSample, nbGoodPlates, nbBadPlates, filteredDf, ruleWeightDict = bf.calculateRuleScore(df_clean, ruleset, target, target_quality_order, verbose)
    score_dict = calculateRuleScore(df_clean, ruleset, target, target_quality_order, verbose)
    score = score_dict["score"]
    scoreUnique = score_dict["scoreUnique"]
    scoreExp = score_dict["scoreExp"]
    nbSample = score_dict["nbSample"]
    nbGoodPlates = score_dict["nbGoodPlates"]
    nbBadPlates = score_dict["nbBadPlates"]
    filteredDf = score_dict["filteredDf"]

    # Calculate median of target
    # targetMedian = np.median(df_clean[target])
    median_list = df_clean[target + "_MEDIAN"].unique()
    if len(median_list) > 1:
        bf.print_color("Warning: {} median column contains multiple different values".format(target), fg.orange)
    targetMedian = median_list[0]
    medianLine_x = [min(pd.to_datetime(df_raw[time_column], format='%Y%m%d')), max(pd.to_datetime(df_raw[time_column], format='%Y%m%d'))]
    medianLine_y = [targetMedian, targetMedian]

    # Get/Calculate the threshold for unique plates
    unique_list = df_clean[target + "_UNIQUE_THRESHOLD"].unique()
    if len(unique_list) > 1:
        bf.print_color("Warning: {}_UNIQUE_THRESHOLD column contains multiple different values".format(target), fg.orange)
    targetUnique = unique_list[0]
    uniqueLine_x = [min(pd.to_datetime(df_raw[time_column], format='%Y%m%d')), max(pd.to_datetime(df_raw[time_column], format='%Y%m%d'))]
    uniqueLine_y = [targetUnique, targetUnique]

    # Add traces to plot
    traceUsable = go.Scattergl(x=pd.to_datetime(df_clean[time_column], format='%Y%m%d'),
                               y=df_clean[target],
                               name='Usables plates ({})'.format(len(df_clean)),
                               mode='markers',
#                                hovertext=df_clean["hover_text"],
                               hoverinfo="text",
                               )

    if filteredDf is not None:
        traceFlt = go.Scattergl(x=pd.to_datetime(filteredDf[time_column], format='%Y%m%d'),
                                y=filteredDf[target],
                                name='Plates following ML rule ({})'.format(len(filteredDf)),
                                mode='markers',
#                                 hovertext=filteredDf["hover_text"],
                                hoverinfo="text",
                                )

    traceMedian = go.Scattergl(x=medianLine_x,
                               y=medianLine_y,
                               name='Target median',
                               mode='lines',
                               )

    traceUniqueThreshold = go.Scattergl(x=uniqueLine_x,
                                        y=uniqueLine_y,
                                        name='Target unique',
                                        mode='lines',
                                        )

    intermediateTrace = []
    for index_rule, rule in enumerate(ml_rule):
        subRuleDf = filterRows(df_clean, rule, rule_type[index_rule], verbose=False)
        traceTmp = go.Scattergl(x=pd.to_datetime(subRuleDf[time_column], format='%Y%m%d'),
                                y=subRuleDf[target],
                                name='{} {} {} ({})'.format(rule[0], rule_type[index_rule], rule[1], len(subRuleDf)),
                                mode='markers',
                                visible="legendonly"
                                )
        intermediateTrace.append(traceTmp)

    # Create title in html (add cariage return if ruleset too long)
    title = ""
    for i in range(len(ml_rule)):
        if i in [2, 4, 6]:
            title += "<br>"
        title += "{} {} {}, ".format(ml_rule[i][0], rule_type[i], ml_rule[i][1])
    title = title[:-2]
    title += "<br>Score Exponential: {}".format(scoreExp)

    # Pie chart, where the slices will be ordered and plotted counter-clockwise:
    labels = ['Nb bad plates', 'Nb good plates']
    colors = ['red', 'green']
    pie_parts = [nbBadPlates, nbGoodPlates]
    piePlot = go.Pie(labels=labels, values=pie_parts)

    # Second pie plot with unique score
    pie_parts = [(1 - scoreUnique) * nbSample, scoreUnique * nbSample]
    uniquePiePlot = go.Pie(labels=labels, values=pie_parts)

    # Create plotly graph
    fig = make_subplots(rows=2, cols=2, column_widths=[0.75, 0.25], specs=[[{"rowspan": 2, "type": "xy"}, {"type": "domain"}], [None, {"type": "domain"}]],
                        subplot_titles=(title, "score median: {}".format(score), "score unique: {}".format(scoreUnique)))
    fig.add_trace(traceUsable, row=1, col=1)
    if filteredDf is not None:
        fig.add_trace(traceFlt, row=1, col=1)
    fig.add_trace(traceMedian, row=1, col=1)
    fig.add_trace(traceUniqueThreshold, row=1, col=1)
    for trace in intermediateTrace:
        fig.add_trace(trace, row=1, col=1)
    fig.add_trace(piePlot, row=1, col=2)
    fig.add_trace(uniquePiePlot, row=2, col=2)
    fig.update_traces(col=2,
                      # hoverinfo='label+percent', textinfo='value', textfont_size=20,
                      marker=dict(colors=colors))

    fig.update_xaxes(title_text=time_column)
    fig.update_yaxes(title_text=target)

    fig_features = go.Figure()
    if filteredDf is not None:
        cols = ["TEMPS_TREMPE_TRAC", "TR_RATIOS_AA_DERNIERE_PASSE",
                "TR_RATIOS_A_DERNIERE_PASSE", "TR_RATIOS_DERNIERE_PASSE",
                "TR_T_C_DERNIERE_PASSE", "TR_VITESSE_DERNIERE_PASSE",
                "TR_LNZ_DERNIERE_PASSE", "TR_COEF_VARIATION_FORCE",
                "COMPO_V", "COMPO_H", "RASPTAB_TIL_VAL",
                "RASPTAB_TIL_VAL_TARGET_SUP"]
        if thickness == 15.0 or thickness == 20.0:
            cols = ["NUMERO_POSITION_BARRE", "TEMPS_TREMPE_TRAC",
                    "TR_RATIOS_AA_DERNIERE_PASSE", "TR_RATIOS_A_DERNIERE_PASSE",
                    "TR_RATIOS_DERNIERE_PASSE",
                    "TR_LNZ_DERNIERE_PASSE", "HOMOG_TPS_MAINTIEN_HALTEN1_TCC_MAXI",
                    "COMPO_NI", "REVENU_M1_T_C_TCC_Mini",
                    "TR_VITESSE_DERNIERE_PASSE", "TR_T_C_DERNIERE_PASSE",
                    "TEMPS_TREMPE_REVENU",
                    "COMPO_V", "CU_TO_MG",
                    "RASPTAB_TIL_VAL",
                    "RASPTAB_TIL_VAL_TARGET_SUP"]

        fig_features = px.parallel_coordinates(filteredDf[cols], color="RASPTAB_TIL_VAL_TARGET_SUP",
                                               color_continuous_scale=px.colors.diverging.Tealrose, color_continuous_midpoint=0.5)

    return fig



# def createScoreGraph(df, time_column, rules_list, target, target_quality_order, verbose=False):
#     """ Create graph showing percentage of excellent/normal plate

#     Args:
#         df: The dataframe containing the data to display
#         time_column: The column to use for x in the scatter plot
#         rules_list: List of rules that will be used to filter plates from df
#         target: The target that will be used to differentiate excellent/normal plates
#         target_quality_order: Dictionary that indicate if a plate is excellent when the
#                               target is below (0) or above (1) the median
#     Returns:
#         fig: the plotly figure ready to be exported
#     Raises:
#         None
#     """
#     # Format rules to be used in filter function
#     ml_rule, rule_type = formatRuleListForFilter(rules_list, False)

#     # Clean column of df that are affected by ruleset
#     df_clean = df.copy()
#     feature_list = [item[0] for item in ml_rule]
#     if len(feature_list) > 0:
#         df_clean = removeNaN(df_clean, feature_list, False)

#     # Calculate score
#     # score, nbSample, nbGoodPlates, nbBadPlates, filteredDf, ruleWeightDict = calculateRuleScore(df_clean, rules_list, target, target_quality_order, verbose)
#     score_dict = calculateRuleScore(df_clean, rules_list, target, target_quality_order, verbose)
#     score = score_dict["score"]
#     nbGoodPlates = score_dict["nbGoodPlates"]
#     nbBadPlates = score_dict["nbBadPlates"]
#     filteredDf = score_dict["filteredDf"]

#     # Calculate median of target
#     # targetMedian = np.median(df_clean[target])
#     median_list = df_clean[target + "_MEDIAN"].unique()
#     if len(median_list) > 1:
#         print_color("Warning: {} median column contains multiple different values".format(target), fg.orange)
#     targetMedian = median_list[0]
#     medianLine_x = [min(pd.to_datetime(df_clean[time_column], format='%Y%m%d')), max(pd.to_datetime(df_clean[time_column], format='%Y%m%d'))]
#     medianLine_y = [targetMedian, targetMedian]

#     # Get/Calculate threshold for unique plates
#     unique_list = df_clean[target + "_UNIQUE_THRESHOLD"].unique()
#     if len(unique_list) > 1:
#         print_color("Warning: {}_UNIQUE_THRESHOLD column contains multiple different values".format(target), fg.orange)
#     targetUnique = unique_list[0]
#     uniqueLine_x = [min(pd.to_datetime(df_clean[time_column], format='%Y%m%d')), max(pd.to_datetime(df_clean[time_column], format='%Y%m%d'))]
#     uniqueLine_y = [targetUnique, targetUnique]

#     # Add traces to plot
#     traceAll = go.Scatter(x=pd.to_datetime(df_clean[time_column], format='%Y%m%d'),
#                           y=df_clean[target],
#                           name='Usable plates',
#                           mode='markers',
#                           )

#     if filteredDf is not None:
#         traceFlt = go.Scatter(x=pd.to_datetime(filteredDf[time_column], format='%Y%m%d'),
#                               y=filteredDf[target],
#                               name='Plates following ML rule',
#                               mode='markers',
#                               )

#     traceMedian = go.Scatter(x=medianLine_x,
#                              y=medianLine_y,
#                              name='Target median',
#                              mode='lines',
#                              )

#     traceUniqueThreshold = go.Scatter(x=uniqueLine_x,
#                                       y=uniqueLine_y,
#                                       name='Target unique',
#                                       mode='lines',
#                                       )

#     intermediateTrace = []
#     for index_rule, rule in enumerate(ml_rule):
#         subRuleDf = filterRows(df_clean, rule, rule_type[index_rule], verbose=False)
#         traceTmp = go.Scatter(x=pd.to_datetime(subRuleDf[time_column], format='%Y%m%d'),
#                               y=subRuleDf[target],
#                               name='{} {} {}'.format(rule[0], rule_type[index_rule], rule[1]),
#                               mode='markers',
#                               visible="legendonly"
#                               )
#         intermediateTrace.append(traceTmp)

#     # Create title in html (add cariage return if ruleset too long)
#     title = ""
#     for i in range(len(ml_rule)):
#         if i in [2, 4, 6]:
#             title += "<br>"
#         title += "{} {} {}, ".format(ml_rule[i][0], rule_type[i], ml_rule[i][1])
#     title = title[:-2]
#     # print(title)

#     # scatterPlot.set_title("{} {} {}".format(ml_rule[0], rule_type, ml_rule[1]))
#     # scatterPlot.set_title(title)

#     # legend = scatterPlot.legend(loc='upper right', shadow=True, fontsize='x-large')
#     # Put a nicer background color on the legend.
#     # legend.get_frame().set_facecolor('C0')

#     # Pie chart, where the slices will be ordered and plotted counter-clockwise:
#     labels = ['Nb bad plates', 'Nb good plates']
#     colors = ['red', 'green']
#     pie_parts = [nbBadPlates, nbGoodPlates]
#     # explode = (0, 0.1, 0, 0)  # only "explode" the 2nd slice (i.e. 'Hogs')
#     # piePlot.pie(pie_parts, labels=labels, autopct='%1.1f%%',
#     #             shadow=True, startangle=90, colors=colors)
#     piePlot = go.Pie(labels=labels, values=pie_parts)

#     # Create plotly graph
#     fig = make_subplots(rows=1, cols=2, column_widths=[0.75, 0.25], specs=[[{"type": "xy"}, {"type": "domain"}]],
#                         subplot_titles=(title, "score: {}".format(score)))
#     fig.add_trace(traceAll, row=1, col=1)
#     if filteredDf is not None:
#         fig.add_trace(traceFlt, row=1, col=1)
#     fig.add_trace(traceMedian, row=1, col=1)
#     fig.add_trace(traceUniqueThreshold, row=1, col=1)
#     for trace in intermediateTrace:
#         fig.add_trace(trace, row=1, col=1)
#     fig.add_trace(piePlot, row=1, col=2)
#     fig.update_traces(col=2,
#                       # hoverinfo='label+percent', textinfo='value', textfont_size=20,
#                       marker=dict(colors=colors))

#     fig.update_xaxes(title_text=time_column)
#     fig.update_yaxes(title_text=target)
#     # fig.update_layout(col=1,
#     #                  xaxis=go.layout.XAxis(title=go.layout.xaxis.Title(text="{} {} {}".format(ml_rule[0], rule_type, ml_rule[1]))),
#     #                  yaxis=go.layout.XAxis(title=go.layout.xaxis.Title(text=target))
#     #                  )

#     # piePlot.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
#     # piePlot.set_title("Score: {}".format(score))
#     # f.show()
#     # fig.show()

#     return fig


###############################################################################
# General
#
def createDir(directory):
    """Checks if directory exists, and creates it if not

    Args:
        dir: string location of directory
    Returns:
        None
    Raises:
        NotADirectoryError
    """
    if (os.path.exists(directory)) is False:
        os.makedirs(directory)
    if (os.path.isdir(directory)) is False:
        raise NotADirectoryError("{} is not a directory".format(directory))


def print_color(s, fg_color):
    """Function to print text in specific color

    Args:
        s: text to print
        fg_color: foreground color from "Sty" package
    Returns:
        None
    Raises:
        None
    """
    text = fg_color + s + fg.rs
    print(text)


def print_color_html(s, color='black'):
    """Function to display html text in specific color

    Args:
        s: text to print
        color: color of the text
    Returns:
        None
    Raises:
        None
    """
    s = s.replace(" ", "&nbsp;")
    s = s.replace("\t", 4 * "&nbsp;")
    display(HTML("<text style=color:{};font-size:95%;font-family:consolas>".format(color) + s + "</text>"))


def colorGradient(c1, c2, mix):
    """Return the gradient of color between two colors

    Args:
        c1: rgb color of max value (1)
        c2: rgb color of min value (0)
        mix: value of the gradient we want (between 0 and 1)
    Returns:
        RGB of gradient corresponding to mix
    Raises:
        None
    """
    c1 = np.array(mpl.colors.to_rgb(c1))
    cmid = np.array(mpl.colors.to_rgb('white'))
    c2 = np.array(mpl.colors.to_rgb(c2))

    if mix <= 0.5:
        mix *= 2
        return mpl.colors.to_hex((1 - mix) * c1 + mix * cmid)
    else:
        mix = 2 * (mix - 0.5)
        return mpl.colors.to_hex((1 - mix) * cmid + mix * c2)


###############################################################################
# Calculation Functions
#
def calcPSI(tit, til, coeff, alloy_type):
    r"""Calculate Rapstab PSI Value depending on alloy type
       Factor depends on Alloy
       * ``7214`` => $FACTOR = 1.2$
       * ``7213`` or ``7215`` =>  $FACTOR = 2.0$
       :math:`PSI = FACTOR*\frac{\sqrt{RASPTAB\_TIT\_VAL^2 + RASPTAB\_TIL\_VAL^2}}{\sqrt{2*RASPTAB\_VAL\_MAX\_TIL\_TIT^2}}`

    Args:
        df: pandas dataframe with zero values
        columns: list of df columns to search for zeros
        verbose: bool give some informational output
    Returns:
        float PSI value
    Raises:
        None
    """
    factor = {7213: 2.0, 7214: 1.2, 7215: 2.0}
    return factor[alloy_type] * (np.sqrt((np.power(tit, 2) + np.power(til, 2))) / np.sqrt(2 * np.power(coeff, 2)))


def calcDfPSI(df, verbose=True):
    """Calculate Rapstab PSI Value for full pandas dataframe.
       The dataframe needs the following columns
       * RASPTAB_TIT_VAL
       * RASPTAB_TIL_VAL
       * RASPTAB_VAL_MAX_TIL_TIT
       * ALLIAGE_INTERNE

    Args:
        df: pandas dataframe with zero values
        verbose: bool give some informational output
    Returns:
        pandas dataframe column for RASPTAB_PSI
    Raises:
        None
    """
    if verbose:
        print("  * Add RAPSTAB_PSI_2")
        print("=> 1 column added!!!")
    return df.apply(lambda row: calcPSI(row['RASPTAB_TIT_VAL'], row['RASPTAB_TIL_VAL'], row['RASPTAB_VAL_MAX_TIL_TIT'], row['ALLIAGE_INTERNE']), axis=1)


def calcPSI2(tit, til, coeff):
    r"""Calculate Rapstab PSI Value depending without factor
       :math:`PSI = \frac{\sqrt{RASPTAB\_TIT\_VAL^2 + RASPTAB\_TIL\_VAL^2}}{\sqrt{2*RASPTAB\_VAL\_MAX\_TIL\_TIT^2}}`

    Args:
        df: pandas dataframe with zero values
        columns: list of df columns to search for zeros
        verbose: bool give some informational output
    Returns:
        float PSI value
    Raises:
        None
    """
    return (np.sqrt((np.power(tit, 2) + np.power(til, 2))) / np.sqrt(2 * np.power(coeff, 2)))


def calcDfPSI2(df, verbose=True):
    """Calculate Rapstab PSI Value without alloy factorfor full pandas dataframe.
       The dataframe needs the following columns
       * RASPTAB_TIT_VAL
       * RASPTAB_TIL_VAL
       * RASPTAB_VAL_MAX_TIL_TIT

    Args:
        df: pandas dataframe with zero values
        verbose: bool give some informational output
    Returns:
        pandas dataframe column for RASPTAB_PSI
    Raises:
        None
    """
    if verbose:
        print("  * Add RAPSTAB_PSI_3")
        print("=> 1 column added!!!")
    return df.apply(lambda row: calcPSI2(row['RASPTAB_TIT_VAL'], row['RASPTAB_TIL_VAL'], row['RASPTAB_VAL_MAX_TIL_TIT']), axis=1)


def calcTmax(d_start, d_cut, thickness, length=0.38, E=71.7, verbose=False):
    r"""Calculate Rapstab internal force. Value is thickness independent
       :math:`T_{max}=\frac{4Et\delta}{l^2}`
       * :math:`T_{max}\,[GPa]` = Stability Pression
       * :math:`t\,[m]` = Thickness after machining (:math:`\frac{1}{2}$ or :math:`\frac{1}{4}` nominal thickness measured)
       * :math:`\delta\,[m]` = :math:`Distance_{\frac{1}{2}}-Distance_{start}` or :math:`Distance_{\frac{1}{4}}-Distance_{start}`
       * :math:`l\,[m]` = Length between contact points :math:`(\approx 38cm)`

    Args:
        d_start: deformation at start
        d_cut: defiormation after cut
        thickness: thickness after machining
        length: length between measurement contact rest points
        E: Module de young for alloy 7xxx
        verbose: bool give some informational output
    Returns:
        float Tmax value
    Raises:
        None
    """
    deformation = d_cut - d_start
    tmax = (4 * E * thickness * deformation) / length**2
    if verbose:
        if tmax < 0:
            print("tmax = {}".format(tmax))
            print("  * d_cut = {}".format(d_cut))
            print("  * d_start = {}".format(d_start))
            print("    * deformation = {}".format(deformation))
            print("  * E = {}".format(E))
            print("  * t = {}".format(thickness))
            print("  * l = {}".format(length))
    return tmax


def calcDfTmax(df, verbose=True):
    """Calculate Rapstab PSI Value for full pandas dataframe.
       The dataframe needs the following columns
       * EPAISSEUR_FINALE
       * TIL_INITIALE
       * TIL_DEMI_EP
       * TIL_QUARTER_EP
       * TIT_INITIALE
       * TIT_DEMI_EP
       * TIT_QUARTER_EP

    Args:
        df: pandas dataframe with zero values
        verbose: bool give some informational output
    Returns:
        pandas dataframe column for
        * RAPSTAB_TmaxL_1/2
        * RAPSTAB_TmaxL_1/4
        * RAPSTAB_TmaxT_1/2
        * RAPSTAB_TmaxT_1/4
    Raises:
        None
    """
    measurement_distance = 0.38
    module_young = 71.7
    rapstab_tmaxl_half = df.apply(lambda row: calcTmax(row['RASPTAB_TIL_INITIALE'], row['RASPTAB_TIL_DEMI_EP'], row['EPAISSEUR_FINALE'] / 2, measurement_distance, module_young), axis=1)
    rapstab_tmaxl_quarter = df.apply(lambda row: calcTmax(row['RASPTAB_TIL_INITIALE'], row['RASPTAB_TIL_QUART_EP'], row['EPAISSEUR_FINALE'] * 3 / 4, measurement_distance, module_young), axis=1)
    rapstab_tmaxt_half = df.apply(lambda row: calcTmax(row['RASPTAB_TIT_INITIALE'], row['RASPTAB_TIT_DEMI_EP'], row['EPAISSEUR_FINALE'] / 2, measurement_distance, module_young), axis=1)
    rapstab_tmaxt_quarter = df.apply(lambda row: calcTmax(row['RASPTAB_TIT_INITIALE'], row['RASPTAB_TIT_QUART_EP'], row['EPAISSEUR_FINALE'] * 3 / 4, measurement_distance, module_young), axis=1)
    if verbose:
        print("  * Add RASPTAB_TMAXL_HALF")
        print("  * Add RASPTAB_TMAXL_QUARTER")
        print("  * Add RASPTAB_TMAXT_HALF")
        print("  * Add RASPTAB_TMAXT_QUARTER")
        print("=> 4 column added!!!")
    return (rapstab_tmaxl_half, rapstab_tmaxl_quarter, rapstab_tmaxt_half, rapstab_tmaxt_quarter)


###############################################################################
# Pandas dataframe
#
# Fill stuff
def fillZeroToNaN(df, columns, verbose=True):
    """Fill all ``0`` values with NaN for given columns

    Args:
        df: pandas dataframe with zero values
        columns: list of df columns to search for zeros
        verbose: bool give some informational output
    Returns:
        Pandas dataframe with zeros filled by NaN for selected columns
    Raises:
        None
    """
    df_tmp = df.copy()
    if verbose:
        print("Zeros filled with NaN for columns:")
    for column in columns:
        if column in df_tmp:
            df_tmp.loc[df_tmp[column] == 0, column] = np.nan
            if verbose:
                print("  * {}".format(column))
    return df_tmp


def fillNaNToZero(df, columns, verbose=True):
    """Fill all NaN values with ``0`` for given columns

    Args:
        df: pandas dataframe with NaN values
        columns: list of df columns to search for NaN
        verbose: bool give some informational output
    Returns:
        Pandas dataframe with NaN filled by Zeros for selected columns
    Raises:
        None
    """
    df_tmp = df.copy()
    if verbose:
        print("NaN filled with Zeros for columns:")
    for column in columns:
        if column in df_tmp:
            df_tmp[columns] = df_tmp[columns].fillna(0)
            if verbose:
                print("  * {}".format(column))
    return df_tmp


def fillNegTime(df, column, verbose=True):
    """Fill all negative time values with zero time for given columns

    Args:
        df: pandas dataframe with negative time values
        columns: list of df columns to search for negative times
        verbose: bool give some informational output
    Returns:
        Pandas dataframe with negative times filled by zeros for selected columns
    Raises:
        None
    """
    df_tmp = df.copy()
    if column in df_tmp:
        df_tmp[column][df_tmp[column] < pd.Timedelta(0)] = pd.Timedelta(0)
        # mask = df_tmp[column] < pd.Timedelta(0)
        # df_tmp[column] = df_tmp[column].mask(mask)
    else:
        print("Column {} not found".format(column))
    return df_tmp


# Remove Stuff
def removeNaN(df, column=None, verbose=True):
    """Removes all NaN values from a pandas dataframe

    Args:
        df: pandas dataframe with NaN values
        column: list of column in which the NaN will be removed
                if none: remove NaN in all columns
        verbose: give some informational output
    Returns:
        Pandas bool dataframe without any NaN values
    Raises:
        None
    """
    # df_na = df[df.isna().any(axis=1)]
    # display(df_na)
    if column is None:
        df_tmp = df.dropna()
    else:
        df_tmp = df.dropna(subset=column)
    len_with_na = len(df)
    len_without_na = len(df_tmp)
    if verbose:
        print("Remove NA-Values from Table")
        if len_with_na == len_without_na:
            print("No duplicates found")
        else:
            print("  * {}/{} NaN Values found".format(len_with_na - len_without_na, len_with_na))
            print("  * {} Rows without NaN Values available".format(len_without_na))
    return df_tmp


def filterRows(df, filter, type="eq", verbose=False):
    """Filter dataframe by filter criteria (keep values defined in filter criteria)

    Args:
        df: pandas dataframe
        filter: list ["<column>",[<filtervalue_1>, <filtervalue_2>]]
        type: string ("eq"|"neq"|"lt"|"lte"|"gt"|"gte")
        verbose: bool give some informational output
    Returns:
        Dataframe panda table selected data
    Raises:
        None
    """
    df_tmp = df.copy()
    if filter[0] in df_tmp:
        if type == "eq":
            df_tmp = df_tmp[df_tmp[filter[0]].isin(filter[1])]
        elif type == "neq":
            df_tmp = df_tmp[~df_tmp[filter[0]].isin(filter[1])]
        elif type == "lt":
            df_tmp = df_tmp[df_tmp[filter[0]] < filter[1]]
        elif type == "lte":
            df_tmp = df_tmp[df_tmp[filter[0]] <= filter[1]]
        elif type == "gt":
            df_tmp = df_tmp[df_tmp[filter[0]] > filter[1]]
        elif type == "gte":
            df_tmp = df_tmp[df_tmp[filter[0]] >= filter[1]]
        else:
            print_color(" Unknown filter type \"{}\", use \"eq\" \"neq\" \"lte\" \"lt\" \"gte\" \"gt\" ".format(type), fg.red)

        if verbose:
            print("Filter Data {}, {} {}".format(filter[0], type, filter[1]))
            print("  * {}/{} Rows removed".format(df.shape[0] - df_tmp.shape[0], df.shape[0]))
            print("  * {} Rows available".format(df_tmp.shape[0]))
    else:
        print("Filter Data {}, {} {}".format(filter[0], type, filter[1]))
        print_color("  Missing Datafield, nothing done", fg.orange)

    return df_tmp


def filterEbauche_old(df, verbose=True):
    """Keeps all data TR_NOMBRE_EBAUCHE == 1 and removes if
        * TR_NOMBRE_EBAUCHE == 2 & NUMERO_TOLE_MERE_MAX == 4 & NUMERO_TOLE_MERE = (1|2)
        * TR_NOMBRE_EBAUCHE == 2 & NUMERO_TOLE_MERE_MAX == (5|6) & NUMERO_TOLE_MERE = (1|2|3)
        * TR_NOMBRE_EBAUCHE == 2 & NUMERO_TOLE_MERE_MAX == (7|8) & NUMERO_TOLE_MERE = (1|2|3|4)
        Needs columns: TR_NOMBRE_EBAUCHE, NUMERO_TOLE_MERE, NUMERO_TOLE_MERE_MAX
    Args:
        df: pandas dataframe
        verbose: bool give some informational output
    Returns:
        Dataframe panda table with removes columns
    Raises:
        None
    """
    if 'TR_NOMBRE_EBAUCHE' in df.columns and \
       'NUMERO_TOLE_MERE_MAX' in df.columns and \
       'NUMERO_TOLE_MERE' in df.columns:

        df_tmp = df.copy()
        indexNames = df_tmp[(df_tmp['TR_NOMBRE_EBAUCHE'] == 2)
                            & (df_tmp['NUMERO_TOLE_MERE_MAX'] == 4)
                            & ((df_tmp['NUMERO_TOLE_MERE'] >= 1) & (df_tmp['NUMERO_TOLE_MERE'] <= 2))
                            ].index
        display(indexNames)
        df_tmp.drop(indexNames, inplace=True)

        indexNames = df_tmp[(df_tmp['TR_NOMBRE_EBAUCHE'] == 2)
                            & ((df_tmp['NUMERO_TOLE_MERE_MAX'] >= 5) & (df_tmp['NUMERO_TOLE_MERE_MAX'] <= 6))
                            & ((df_tmp['NUMERO_TOLE_MERE'] >= 1) & (df_tmp['NUMERO_TOLE_MERE'] <= 3))
                            ].index
        display(indexNames)
        df_tmp.drop(indexNames, inplace=True)

        indexNames = df_tmp[(df_tmp['TR_NOMBRE_EBAUCHE'] == 2)
                            & ((df_tmp['NUMERO_TOLE_MERE_MAX'] >= 7) & (df_tmp['NUMERO_TOLE_MERE_MAX'] <= 8))
                            & ((df_tmp['NUMERO_TOLE_MERE'] >= 1) & (df_tmp['NUMERO_TOLE_MERE'] <= 4))
                            ].index
        display(indexNames)
        df_tmp.drop(indexNames, inplace=True)

        if verbose:
            print("Filter Ebauche Issoire Data")
            print("  * {}/{} Rows removed".format(df.shape[0] - df_tmp.shape[0], df.shape[0]))
            print("  * {} Rows available".format(df_tmp.shape[0]))
        return df_tmp
    else:
        if verbose:
            print("Filter Ebauche Issoire Data")
            print_color("  Missing Datafield, nothing done", fg.orange)
        return df


def filterEbauche(df, verbose=True):
    """Keeps all data TR_NOMBRE_EBAUCHE == 1 and removes if
        * TR_NOMBRE_EBAUCHE == 2 & NUMERO_TOLE_MERE <= ceil(NUMERO_TOLE_MERE_MAX /2)
        We want to keep only the superior half of the total Plates
        (only plates from the second ebauche)
        Needs columns: TR_NOMBRE_EBAUCHE, NUMERO_TOLE_MERE, NUMERO_TOLE_MERE_MAX
    Args:
        df: pandas dataframe
        verbose: bool give some informational output
    Returns:
        Dataframe panda table with removes columns
    Raises:
        None
    """
    if 'TR_NOMBRE_EBAUCHE' in df.columns and \
       'NUMERO_TOLE_MERE_MAX' in df.columns and \
       'NUMERO_TOLE_MERE' in df.columns:

        df_tmp = df.copy()
        for tole_mere_max in range(1, 9):
            indexNames = df_tmp[((df_tmp['TR_NOMBRE_EBAUCHE'] == 2)
                                & (df_tmp['NUMERO_TOLE_MERE_MAX'] == tole_mere_max)
                                & (df_tmp['NUMERO_TOLE_MERE'] <= math.ceil(tole_mere_max / 2)))
                                ].index
            # display(indexNames)
            df_tmp.drop(indexNames, inplace=True)

        if verbose:
            print("Filter Ebauche Issoire Data")
            print("  * {}/{} Rows removed".format(df.shape[0] - df_tmp.shape[0], df.shape[0]))
            print("  * {} Rows available".format(df_tmp.shape[0]))
        return df_tmp
    else:
        if verbose:
            print("Filter Ebauche Issoire Data")
            print_color("  Missing Datafield, nothing done", fg.orange)
        return df


def keepColumns(df, columns_keep, text=None, verbose=True):
    """Only keeps all specified columns

    Args:
        df: pandas dataframe
        columns: list of columns to keep
        text: string to print
        verbose: bool give some informational output
    Returns:
        Dataframe panda table selected columns
    Raises:
        None
    """
    if verbose:
        print("Keep only selected columns")
        if not(text is None):
            print(text)
    df_tmp = df.copy()
    dropped = 0
    for column in df.columns:
        if not(column in columns_keep):
            df_tmp.drop(column, inplace=True, axis=1)
            dropped += 1
            if verbose:
                print("  * Drop {}".format(column))
    if verbose:
        print("=> {} columns dropped!!!".format(dropped))
    return df_tmp


def removeColumns(df, columns, text=None, verbose=True):
    """Removes all specified columns

    Args:
        df: pandas dataframe
        columns: list of columns to remove
        text: string to print
        verbose: bool give some informational output
    Returns:
        Dataframe panda table selected columns removed
    Raises:
        None
    """
    if verbose:
        print("Remove selected columns")
        if not(text is None):
            print(text)
    df_tmp = df.copy()
    for column in columns:
        if column in df_tmp:
            df_tmp.drop(column, inplace=True, axis=1)
            if verbose:
                print("  * Drop {}".format(column))
        else:
            if verbose:
                print_color("  * Not exists {}".format(column), fg.orange)
    if verbose:
        print("=> {} columns dropped!!!".format(len(columns)))
    return df_tmp


def removeFiniteColumns(df, verbose=True):
    """Removes all columns with only one value

    Args:
        df: pandas dataframe
        verbose: bool give some informational output
    Returns:
        Dataframe panda table without these columns
    Raises:
        None
    """
    if verbose:
        print("Delete finite columns")
    dropped_col = 0
    df_tmp = df.copy()
    for column in df_tmp.columns:
        if len(df_tmp[column].unique()) == 1:
            dropped_col += 1
            if verbose:
                print("  * Drop {} only value '{}' present".format(column, df_tmp[column].iloc[0]))
            df_tmp.drop(column, inplace=True, axis=1)
    if verbose:
        print("=> {} columns dropped!!!".format(dropped_col))
    return df_tmp


def removeColumnsLessThan(df, minRowNbr=1000, verbose=True):
    """Removes all Columns with less than x non NaN Values

    Args:
        df: pandas dataframe
        verbose: bool give some informational output
    Returns:
        Dataframe panda table without these columns
    Raises:
        None
    """
    if verbose:
        print("Delete columns with less than {} values".format(minRowNbr))
    dropped_col = 0
    df_tmp = df.copy()
    for column in df_tmp.columns:
        nbrOfValues = df_tmp.shape[0] - pd.isnull(df_tmp[column]).sum()
        if nbrOfValues < minRowNbr:
            dropped_col += 1
            if verbose:
                print("  * Drop {} only \"{}\" values present".format(column, nbrOfValues))
            df_tmp.drop(column, inplace=True, axis=1)
    if verbose:
        print("=> {} columns dropped!!!".format(dropped_col))
    return df_tmp


def removeDuplicates(df, verbose=True):
    """Removes all duplicated from a table

    Args:
        df: pandas dataframe
        verbose: bool give some informational output
    Returns:
        Dataframe panda table without duplicated
    Raises:
        None
    """
    if not(df.duplicated().empty):
        if (len(df[df.duplicated()]) > 0):
            len_with_duplicates = len(df)
            df = df.drop_duplicates()
            len_without_duplicates = len(df)
            if verbose:
                print("Duplicates found and removed from the Table")
                print("  * {}/{} Elements found".format(len_with_duplicates - len_without_duplicates, len_with_duplicates))
                print("  * {} Unique Elements available ".format(len_without_duplicates))
        else:
            if verbose:
                print("No Duplicates found")
    else:
        if verbose:
            print("No Duplicates found")
    return df


# def filterTwoDf(df_master, df_slave, verbose=True):
#     """Filter df_slave leaving only rows with same index as df_master

#     Args:
#         df_master: pandas dataframe used as master
#         df_master: pandas dataframe users as slave
#         verbose: bool give some informational output
#     Returns:
#         Dataframe panda table with same index rows as df_master
#     Raises:
#         None
#     """
#     len_before = target_df.shape[0]
#     df_tmp = target_df.iloc[dataset.index]
#     len_after = target_df.shape[0]
#     if verbose:
#         print("{}".format(len_after - len_before))
#     return df_tmp


def cleanDf(df, verbose=True):
    """Cleans pandas dataframe
         * Removes Duplicates
         * Removes Finite Columns
         * Removes NaN

    Args:
        df: pandas dataframe
        verbose: bool give some informational output
    Returns:
        Dataframe panda table cleaned
    Raises:
        None
    """
    if verbose:
        print("Clean DataFrame")
        print("  * remove duplicates")
        print("  * remove finite columns")
        print("  * remove NaN")

    df = removeDuplicates(df, verbose)
    df = removeFiniteColumns(df, verbose)
    df = removeNaN(df, None, verbose)
    return df


def fix_types(df, dtypes, verbose=True):
    """Changes types of columns, df columns and dtypes list need to match

    Args:
        df: pandas input table with set of given columns
        dtypes: list of types for table columns
        verbose: bool give some informational output
    Returns:
        Pandas dataframe with changed columns types
    Raises:
        None
    """
    df_tmp = df.copy()
    if verbose:
        print("Change types of table columns")
    for i, column in enumerate(df):
        df_tmp[column] = df_tmp[column].astype(dtypes[i])
        if verbose:
            print("  * {} - {}".format(column, dtypes[i]))
    return df_tmp


# Test Stuff
def testNull(df, columns, verbose=1):
    """Check if Null (NaN) values in pandas dataframe exist

    Args:
        df: pandas dataframe
        columns: list of colums to search for
        verbose: bool give some informational output
    Returns:
        Bool
    Raises:
        None
    """
    exist = False
    if len(np.where(pd.isnull(df))[0]) != 0:
        if verbose:
            print_color(' WARNING!!: Dataframe has null (NaN or empty) cells', fg.orange)
        for column in columns:
            if column in df:
                listofNullCells = np.where(pd.isnull(df[column]))
                lenNullCells = len(listofNullCells[0])
                if lenNullCells != 0:
                    if verbose >= 1:
                        print("  * Column '{}' has {}/{} null cells".format(column, lenNullCells, df.shape[0]))
                    exist = True
                else:
                    if verbose >= 2:
                        print("* Column {}: No Null Cells found!".format(column))
    else:
        if verbose:
            print("All good: no empty cells in Dataframe")
    return exist


def testNaT(df, columns, verbose=1):
    """Check if not a time (NaT) values in pandas dataframe exist

    Args:
        df: pandas dataframe
        columns: list of colums to search for
        verbose: bool give some informational output (1|2)
    Returns:
        Bool
    Raises:
        None
    """
    exist = False
    for column in columns:
        if column in df:
            if pd.api.types.is_timedelta64_dtype(df[column]) or pd.api.types.is_datetime64_ns_dtype(df[column]):
                listofNaTCells = df.loc[np.isnat(df[column])]
                lenNaTCells = len(listofNaTCells)
                if lenNaTCells != 0:
                    if verbose >= 1:
                        print("  * Column '{}' has {}/{} NaT cells".format(column, lenNaTCells, df.shape[0]))
                    exist = True
                else:
                    if verbose >= 2:
                        print("* Column {}: No NaT Cells found!".format(column))
    return exist


def testNegTime(df, columns, verbose=1):
    """Check if negative time values in pandas dataframe exist

    Args:
        df: pandas dataframe
        columns: list of colums to search for
        verbose: int give some informational output (1|2|3)
    Returns:
        Bool
    Raises:
        None
    """
    exist = False
    for column in columns:
        if column in df:
            if pd.api.types.is_timedelta64_dtype(df[column]) or pd.api.types.is_datetime64_ns_dtype(df[column]):
                listofNegTimeCells = df[df[column] < pd.Timedelta(0)]
                lenNegTimeCells = len(listofNegTimeCells)
                if lenNegTimeCells != 0:
                    if verbose >= 1:
                        print("* Column '{}' has {}/{} Negative Time".format(column, lenNegTimeCells, df[column].count()))
                    if verbose >= 3:
                        print(listofNegTimeCells[column])
                    exist = True
                else:
                    if verbose >= 2:
                        print("* Column '{}': No Negative Time!".format(column))
    return exist


# Display stuff
def countRowWithNaN(df, verbose=False):
    """Counts row with at least 1 NaN value

    Args:
        df: pandas dataframe to analyse
        verbose: bool give some informational output
    Returns:
        int number of rows with a NaN Value
    Raises:
        None
    """
    x = df.isnull().sum(axis=1)
    rows_with_NaN = x[x > 0].count()
    if verbose:
        print("Table contains {} rows with NaN Values".format(rows_with_NaN))
    return rows_with_NaN


def countNaN(df, verbose=False):
    """Count all NaN cells in a dataframe

    Args:
        df: pandas dataframe to analyse
        verbose: bool give some informational output
    Returns:
        int number of rows with a NaN Value
    Raises:
        None
    """
    nbrNaN = np.count_nonzero(df.isnull().values)
    if verbose:
        print("Table contains {} NaN cells".format(nbrNaN))
    return nbrNaN


def df_info(df_name, df_description, df, verbose=True):
    """Display info about one dataframe

    Args:
        df_name: string shortname of the dataframe
        df_description: string description of the dataframe
        df: pandas dataframe
        verbose: bool give some informational output
    Returns:
        None
    Raises:
        None
    """
    data = [df_name,
            df_description,
            df.shape[1],
            df.shape[0],
            df.shape[0] - countRowWithNaN(df),
            countRowWithNaN(df),
            np.count_nonzero(df.isnull().values)
            ]
    if verbose:
        print("| {:10} | {:30} | {:8} | {:7} | {:15} | {:11} | {:15} |".format(data[0], data[1], data[2], data[3], data[4], data[5], data[6]))
    return data


def dfs_info(dfs_name, dfs_description, dfs, verbose=False):
    """Display info about multiple dataframes

    Args:
        dfs_name: list of strings with shortname of the dataframe
        dfs_description: list of strings with description of the dataframe
        dfs: list of pandas dataframe
        verbose: bool give some informational output
    Returns:
        None
    Raises:
        None
    """
    columns = ['Name',
               'Description',
               'Features',
               'Entries',
               'Entries_Non_NaN',
               'Entries_NaN',
               'Total_count_NaN',
               ]
    column_types = ['category',
                    'category',
                    'int64',
                    'int64',
                    'int64',
                    'int64',
                    'int64',
                    ]
    dfs_informations = pd.DataFrame(columns=columns)
    if verbose:
        print("| {:10} | {}                    | {} | {} | {} | {} | {} |".format(columns[0], columns[1], columns[2], columns[3], columns[4], columns[5], columns[6]))
        print("|------------|--------------------------------|----------|---------|-----------------|-------------|-----------------|")
    for i in range(len(dfs_name)):
        dfs_informations.loc[i] = df_info(dfs_name[i], dfs_description[i], dfs[i], verbose)
    # Fix datatypes
    dfs_informations = fix_types(dfs_informations, column_types, verbose=False)
    return dfs_informations


def df_info_append(df_information, df_name, df_description, df, verbose=False):
    """Append dataframe informations to pandas information table

    Args:
        df_informations: existing information table
        df_name: string shortname of the dataframe
        df_description: string description of the dataframe
        dfs: pandas dataframe
        verbose: bool give some informational output
    Returns:
        None
    Raises:
        None
    """
    row = pd.DataFrame([df_info(df_name, df_description, df, verbose)], columns=df_information.columns)
    print(row)
    df_information.append(row)
    return df_information


def displayNegTimes(df, column_t1, column_t2, column_deltatime):
    """Display negative times values of given columns

    Args:
        df: pandas dataframe
        columns_t1: datetime first time to display
        columns_t2: datetime second time to display
        column_deltatime: deltatime to search for negative values
    Returns:
        dataframe of only selected columns and negative times
    Raises:
        None
    """
    columns = [column_t1, column_t2, column_deltatime]
    return df.loc[df[column_deltatime] < pd.Timedelta(0)][columns]


def displaySummary(df, columns=None):
    """Display a summary of the dataframe

    Args:
        df: pandas dataframe
        columns: list of colums to display
    Returns:
        None
    Raises:
        None
    """
    if not columns:
        columns = df.columns

    print("| {:>50} | {:15} | {:10} | {:10} | {:18} | {:18} | {:18} | ".format("Parameter", "Type", "Total Rows", "Unique Val", "Min", "Max", "Mean"))
    print("|{}|{}|{}|{}|{}|{}|{}|".format(52 * "-", 17 * "-", 12 * "-", 12 * "-", 20 * "-", 20 * "-", 20 * "-"))
    for column in columns:
        if column in df:
            columnType = str(df[column].dtypes)
            totalRow = df.shape[0]
            minV, maxV, mean = "-", "-", "-"
            if columnType in ["int64", "Int64", "float64", "timedelta64[ns]"]:
                minV = np.min(df[column])
                maxV = np.max(df[column])
                mean = np.mean(df[column])
            nbUniqueVal = df[column].unique().shape[0]
            # nbUniqueVal = 0
            if columnType == "timedelta64[ns]":
                print("| {:>50} | {:15} | {:10} | {:10} | {:18} | {:18} | {:18} | ".format(column, columnType, totalRow, nbUniqueVal, str(minV)[:18], str(maxV)[:18], str(mean)[:18]))
            elif columnType in ["int64", "Int64"]:
                print("| {:>50} | {:15} | {:10} | {:10} | {:18} | {:18} | {:18.2f} | ".format(column, columnType, totalRow, nbUniqueVal, minV, maxV, mean))
            elif columnType == "float64":
                print("| {:>50} | {:15} | {:10} | {:10} | {:18.2f} | {:18.2f} | {:18.2f} | ".format(column, columnType, totalRow, nbUniqueVal, minV, maxV, mean))
            else:
                print("| {:>50} | {:15} | {:10} | {:10} | {:>18} | {:>18} | {:>18} | ".format(column, columnType, totalRow, nbUniqueVal, minV, maxV, mean))

    print()


def displayDiff(df, index1, index2, columns=None):
    """Display difference between two elements of the dataframe

    Args:
        df: pandas dataframe
        index1: index of the first item we want to compare
        index2: index of the second item we want to compare
        columns: list of colums to display
    Returns:
        None
    Raises:
        None
    """
    if not columns:
        columns = df.columns

    print("| {:>50} | {:18} | {:18} | ".format("Parameter", index1, index2,))
    print("|{}|{}|{}|".format(52 * "-", 20 * "-", 20 * "-"))
    for column in columns:
        if column in df:

            columnType = str(df[column].dtypes)
            val1 = df[df.index == index1][column].values[0]
            val2 = df[df.index == index2][column].values[0]

            if val1 == val2:
                fg.color = fg.green
            else:
                fg.color = fg.red

            if columnType in ["int64", "Int64, timedelta64[ns]"]:
                str1 = "{:18}".format(val1)
                str2 = "{:18}".format(val2)
            elif columnType == "float64":
                str1 = "{:18.2f}".format(val1)
                str2 = "{:18.2f}".format(val2)
            else:
                str1 = "{:>18}".format(val1)
                str2 = "{:>18}".format(val2)
            print("| {:>50} | {}{}{} | {}{}{} | ".format(column, fg.color, str1, fg.rs, fg.color, str2, fg.rs))

    print()


def listUniqueValues(df, columns):
    """Display unique values of given columns

    Args:
        df: pandas dataframe
        columns: list of colums to display
    Returns:
        None
    Raises:
        None
    """
    print("| {:>50} | {:13} | {:10} | ".format("Parameter", "Unique Values", "Total Rows"))
    print("|{}|{}|{}|".format(52 * "-", 15 * "-", 12 * "-"))
    for column in columns:
        uniqueEntries = df[column].unique()
        uniqueEntryOccurence = uniqueEntries.shape[0]
        print("| {:>50} | {:13} | {:10} | ".format(column, uniqueEntryOccurence, df.shape[0]))


def displayEntryOccurences(df, columns=None, showOccurencesWhen=None):
    """Display occurences of selected columns

    Args:
        df: pandas dataframe
        columns: list of colums to display, None for all columns of df
        showOccurencesWhen: int filer to selected on which number to display, None for remove filtering
    Returns:
        None
    Raises:
        None
    """
    if not(showOccurencesWhen is None):
        print("Display Unique values if they have less than {} unique values!!".format(showOccurencesWhen))
    if not columns:
        columns = df.columns
    for column in columns:
        if column in df:
            if not(showOccurencesWhen is None):
                if df[column].unique().shape[0] < showOccurencesWhen:
                    uniqueEntries = df[column].unique()
                    print("| {:27} |".format(column))
                    print("|          Value | Occurences |")
                    print("|----------------|------------|")
                    if uniqueEntries.dtype == np.float64 or uniqueEntries.dtype in ["int64", "Int64"]:
                        for uniqueEntry in np.sort(uniqueEntries):
                            nbrOfOccurences = df.loc[df[column] == uniqueEntry].shape[0]
                            print("| {:>14} | {:10} |".format(uniqueEntry, nbrOfOccurences))
                    else:
                        for uniqueEntry in uniqueEntries:
                            nbrOfOccurences = df.loc[df[column] == uniqueEntry].shape[0]
                            print("| {:>14} | {:10} |".format(uniqueEntry, nbrOfOccurences))
                    print()
            else:
                uniqueEntries = df[column].unique()
                print("| {:27} |".format(column))
                print("|          Value | Occurences |")
                print("|----------------|------------|")
                if uniqueEntries.dtype == np.float64 or uniqueEntries.dtype in ["int64", "Int64"]:
                    for uniqueEntry in np.sort(uniqueEntries):
                        nbrOfOccurences = df.loc[df[column] == uniqueEntry].shape[0]
                        print("| {:>14} | {:10} |".format(uniqueEntry, nbrOfOccurences))
                else:
                    for uniqueEntry in uniqueEntries:
                        nbrOfOccurences = df.loc[df[column] == uniqueEntry].shape[0]
                        print("| {:>14} | {:10} |".format(uniqueEntry, nbrOfOccurences))
                print()


# Action on dataframe
def absoluteNegValues(df, columns):
    """Absolute value for specified columns

    Args:
        df: pandas dataframe
        columns: list of column to process
    Returns:
        df: with absolute values
    Raises:
        None
    """
    df_tmp = df.copy()
    for target in columns:
        if target in df_tmp:
            df_tmp[target] = np.abs(df_tmp[target])
    return df_tmp


def addTargetColumnBasedOnMedian(df, columns=None, verbose=True):
    """Add new column for targets where 1 indicates
       the value is superior than the median and 0 if
       it's inferior
    Args:
        df: pandas dataframe
        columns: list of existing colums that we want to process
    Returns:
        df with new target columns
    Raises:
        None
    """
    if not columns:
        print_color(" Error: you must indicate at least one target column", fg.red)
    else:
        # List thickness
        thicknessList = df['EPAISSEUR_FINALE'].unique()
        # For each thickness
        for thickness in thicknessList:
            if verbose:
                print(" * Thickness of {}".format(thickness))
            for target in columns:
                if target in df:
                    if verbose:
                        print("  - process target column {}".format(target))
                    # print(df[df['EPAISSEUR_FINALE'] == thickness][['EPAISSEUR_FINALE', target]])
                    # Calculate target median of samples with this specific thickness
                    median = np.median(df.loc[df['EPAISSEUR_FINALE'] == thickness, target])
                    # Calculate threshold for unique plates
                    uniqueThreshold = np.percentile(df.loc[df['EPAISSEUR_FINALE'] == thickness, target], 10)
                    # Add median column and threshold for unique plates
                    df.loc[df['EPAISSEUR_FINALE'] == thickness, '{}_MEDIAN'.format(target)] = median
                    df.loc[df['EPAISSEUR_FINALE'] == thickness, '{}_UNIQUE_THRESHOLD'.format(target)] = uniqueThreshold
                    # Add column with 1 if above or equal median, else 0
                    df.loc[df['EPAISSEUR_FINALE'] == thickness, '{}_TARGET_SUP'.format(target)] = np.where(df.loc[df['EPAISSEUR_FINALE'] == thickness, target] >= median, 1, 0)
                    df.loc[df['EPAISSEUR_FINALE'] == thickness, '{}_TARGET_SUP_UNIQUE'.format(target)] = np.where(df.loc[df['EPAISSEUR_FINALE'] == thickness, target] >= uniqueThreshold, 1, 0)
                    # print(df[df['EPAISSEUR_FINALE'] == thickness][['EPAISSEUR_FINALE', target, '{}_TARGET_SUP'.format(target)]])
                else:
                    print_color(" Warning: Column {} not found in dataframe".format(target), fg.orange)

            # displaySummary(df[df['EPAISSEUR_FINALE'] == thickness][["RASPTAB_TIL_VAL_MEDIAN", "RASPTAB_TIL_VAL_TARGET_SUP"]])
            # displayEntryOccurences(df[df['EPAISSEUR_FINALE'] == thickness][["RASPTAB_TIT_VAL", "RASPTAB_TIT_VAL_TARGET_SUP", "RASPTAB_TIL_VAL", "RASPTAB_TIL_VAL_TARGET_SUP"]])

    return df


def convertTimeDurationToSec(df, columns, verbose=True):
    """Convert time duration to sec in dataframe
    Args:
        df: pandas dataframe
        columns: list of existing columns that we want to process
    Returns:
        df with new converted columns
    Raises:
        None
    """
    df_tmp = df.copy()
    if verbose:
        print("Convert time duration to second")
    for column in columns:
        if column in df_tmp:
            if verbose:
                print("  * Convert column {}".format(column))
            df_tmp[column] = np.floor(df_tmp[column].dt.total_seconds()).astype('Int64')
        else:
            print_color("Warning: column {} not found in dataframe".format(column), fg.orange)
    return df_tmp


def convertSecToTimeDuration(df, columns, verbose=False):
    """Convert second to time duration in dataframe
    Args:
        df: pandas dataframe
        columns: list of existing columns that we want to process
    Returns:
        df with new converted columns
    Raises:
        None
    """
    df_tmp = df.copy()
    if verbose:
        print("Convert time duration to second")
    for column in columns:
        if column in df_tmp:
            if verbose:
                print("  * Convert column {}".format(column))
            df_tmp[column] = df_tmp.apply(lambda x: datetime.timedelta(seconds=x[column]), 1)
        else:
            print_color("Warning: column {} not found in dataframe".format(column), fg.orange)
    return df_tmp


def generate_hover_text(df, thickness):
    """ Generate a html text for each ruleset of the dataframe to be displayed on hover mode of a plotly graph
    Args:
        df: pandas dataframe
        thickness: thickness we want to use for the score / nb samples
    Returns:
        hover_text: pandas serie with the html text
    Raises:
        None
    """
    def get_html_text(a, b, c, d, e):
        text = "<b>{}</b><br>Nb_samples:{}<br>Score:{}<br>".format(a, b, c)

        # Add cariage return if ruleset too long
        for i, rule in enumerate(d):
            # if i in [3, 6, 9, 12]:
            #    text += "<br>"
            text += "<br>"
            text += "{} ({} / {}), ".format(rule, e[rule][0], e[rule][1])
        text = text[:-2]
        return text

    # Generate hover text for graph
    hover_text = df.apply(lambda x: get_html_text(x["rule_index"], x["score_nbSample_{}".format(thickness)], x["score_{}".format(thickness)], x["rules_list"], x["rules_weight_{}".format(thickness)]), axis=1)
    return hover_text


def reorder_columns(df, columns):
    """ Reorder columns of a dataframe
    Args:
        df: pandas dataframe
        columns: list of dataframe columns in the order we want them to appear in the df
    Returns:
        df_ordered: pandas dataframe with ordered columns
    Raises:
        None
    """
    # Check if all columns are present in the dataframe
    valid_columns = []
    for item in columns:
        if item in df:
            valid_columns.append(item)
        else:
            print_color("Warning: column {} not in df".format(item), fg.orange)
    # Reorder existing columns
    df_ordered = df[valid_columns].copy()
    return df_ordered


# Function about twins plates
def areTwins(item1, item2, columnsToCompare=None, verbose=False):
    """ Compare to series object and return 1 if they are twins, otherwise 0
    Args:
        item1: pandas series to compare
        item2: pandas series to compare
        columnsToCompare: columns to use for the comparison
    Returns:
        areTwins: boolean telling if items are twins
    Raises:
        None
    """
    def typesafe_isnan(obj):
        return isinstance(obj, float) and obj != obj

    # Use all columns if they are not specified
    if columnsToCompare is None:
        columnsToCompare = item1.keys().values
    # Compare each column of the two series
    for column in columnsToCompare:
        if column in item1 and column in item2:
            if (not typesafe_isnan(item1[column])) and (not typesafe_isnan(item2[column])) and (item1[column] != item2[column]):
                if verbose:
                    print(" Column {} not similar {} / {}".format(column, item1[column], item2[column]))
                return 0
        elif verbose:
            print_color(' Warning: column {} not found'.format(column), fg.orange)
    return 1


def findTwins(df, columns=None, verbose=False):
    """ Find twins in a dataframe
    Args:
        df: pandas dataframe
        columns: list of dataframe columns to use to find twins
    Returns:
        df_temp: pandas dataframe with new column containing the list of twins
        twin_list_flat: list of the index of all the twins in the dataframe
    Raises:
        None
    """
    df_temp = df.copy()
    twin_list = []
    # For each plate, find twins
    for index, row in df_temp.iterrows():
        # Add column to indicate twins
        df_temp["twin"] = df_temp.apply(lambda x: areTwins(x, row, columns, verbose), axis=1)
        # Filter to have only twins
        df_twins = df_temp[(df_temp["twin"] == 1)]
        df_twins = df_twins.loc[df_twins.index != index]

        # Store twins list
        twin_list.append(df_twins.index.values.tolist())

        # Delete twin column, we don't need it anymore
        df_temp = df_temp.drop(["twin"], axis=1)

    df_temp["twins_list"] = twin_list

    # Flatten the twin list
    twin_list_flat = [twin for elem in twin_list for twin in elem]
    twin_list_flat = list(set(twin_list_flat))
    return df_temp, twin_list_flat


def plotTwins(df, thickness, targetMedian, graphName="Test.html"):
    """ Create graph to plot twins from a dataframe
    Args:
        df: pandas dataframe, must contains "twins_list" column
        thickness: thickness of the plates
        targetMedian: value of the target median to be plotted on the graph
        graphName: name of the plot
    Returns:
        None
    Raises:
        None
    """
    diff_list = []
    x, y, text, color = [], [], [], []
    alreadyPlotted = []
    # Run through plates
    for index, row in df.iterrows():
        # Skip it if it was already plotted
        if str(index) in alreadyPlotted:
            continue

        # Add plates to list for plotting
        x.append("gr {}".format(index))
        ytmp = [row["RASPTAB_TIL_VAL"]]
        text.append(index)
        # Keep trace of plates that are already plotted
        alreadyPlotted.append(str(index))

        # Run through twins of the current plates
        for twin_index in row["twins_list"]:
            # Add plates to list for plotting
            x.append("gr {}".format(index))
            ytmp += [df[df.index == twin_index]["RASPTAB_TIL_VAL"].values[0]]
            text.append(twin_index)
            # Keep trace of plates that are already plotted
            alreadyPlotted.append(str(twin_index))

        # Choose color
        if len(ytmp) == 1:
            color += ["blue" for i in range(len(ytmp))]
        elif max(ytmp) > targetMedian and min(ytmp) <= targetMedian:
            color += ["red" for i in range(len(ytmp))]
        else:
            color += ["green" for i in range(len(ytmp))]

        y += ytmp

        # Calculate max target delta between twins
        if len(ytmp) > 1:
            delta = max(ytmp) - min(ytmp)
            diff_list.append(delta)

    # Count twins
    nbUnique = color.count("blue")
    nbGood = color.count("green")
    nbBad = color.count("red")

    # Print a describe of the delta target between twins
    twin_df = pd.DataFrame(columns=["RASPTAB_TIL_VAL_DIFF"], data=diff_list)
    print(twin_df.describe())

    # Add traces to plot
    traceAll = go.Scatter(x=x, y=y, name='All', mode='markers', hovertext=text, hoverinfo="text", marker_color=color)
    traceMedian = go.Scatter(x=[x[0], x[-1]], y=[targetMedian, targetMedian], name='Target median', mode='lines', marker_color="magenta")

    # Create graph
    data = [traceAll, traceMedian]
    layout = go.Layout(title="Twins Plates Thickness {} (Nb Unique: {}, nb Good: {}, nb Bad: {})".format(thickness, nbUnique, nbGood, nbBad))
    fig = go.Figure(data=data, layout=layout)
    fig.update_yaxes(title_text="RASPTAB_TIL_VAL")
    plot(fig, filename=graphName, auto_open=False)


###############################################################################
# Scikit learn functions
# Encoders
def categoryToInt(df, verbose=True):
    """Encodes columns of type category to integers

    Args:
        df: pandas dataframe
        verbose: bool give some informational output
    Returns:
        df: dataset encoded
        labelencoders: label encoder function objects
    Raises:
        None
    """
    if verbose:
        print("Convert category to integer")
    labelEncoders = dict()
    for column in df.columns:
        if df[column].dtype.name == 'category':
            le = LabelEncoder()
            le.fit(df[column])
            df[column] = le.transform(df[column])
            labelEncoders[column] = le
            if verbose:
                print("  * Encode category: {}".format(column))
    return df, labelEncoders


def encodePdOneHot(df, columns, verbose=True):
    """Encodes columns to one hot pandas style

    Args:
        df: pandas dataframe
        columns: list of colums to encode
        verbose: bool give some informational output
    Returns:
        df: dataset encoded
    Raises:
        None
    """
    if verbose:
        print("Convert columns to One-Hot pandas style")

    columnToEncode = []
    for column in columns:
        if column in df:
            columnToEncode.append(column)
            if verbose:
                print("  * Encode column: {}".format(column))
        else:
            if verbose:
                print_color("   * Warning: column {} not in dataframe".format(column), fg.orange)

    df = pd.get_dummies(df, columns=columnToEncode)
    return df


def encodeOneHot(df, columns, verbose=True):
    """Encodes columns to one hot sklearn style

    Args:
        df: pandas dataframe
        columns: list of colums to encode
        verbose: bool give some informational output
    Returns:
        df: dataset encoded
        oneHotEncoders: one hot encoder function objects
    Raises:
        None
    """
    df_tmp = df.copy()
    oneHotEncoders = dict()
    for column in columns:
        if verbose:
            print("encoding column:", column)
        if column in df_tmp:
            ohe = OneHotEncoder(sparse=True, handle_unknown="ignore")
            ohe.fit(df_tmp[column])
            df_tmp[column] = ohe.transform(df_tmp[column])
            oneHotEncoders[column] = ohe
    return df_tmp, oneHotEncoders


# Scaler
def scaleStandard(df, verbose=True):
    """Scales Values with the standard method. Uses all columns with int64 and float64 type
       $$\frac{x_i-mean(x)}{stdev(x)}$$

    Args:
        df: pandas dataframe
        verbose: bool give some informational output
    Returns:
        df: dataset encoded
    Raises:
        None
    """
    df_num = df.select_dtypes(include=['int64', 'Int64', 'float64']).astype('float64')
    columns = df_num.columns
    index = df_num.index

    scaler = StandardScaler()

    scaled_df = scaler.fit_transform(df_num)
    scaled_df = pd.DataFrame(scaled_df, index=index, columns=columns)

    df = df.drop(columns, 1)
    df = df.join(scaled_df)

    if verbose:
        print("Standard scaling of all int64 and float64 columns")
        for col in columns:
            print("  * Scale column: {}".format(col))

    return df


def scaleMinMax(df, center_zero=False, verbose=True):
    """Scales Values with the Min Max method. Uses all columns with int64 and float64 type
       $$\frac{x_i-min(x)}{max(x)-min(x)}$$

    Args:
        df: pandas dataframe
        center_zero: center around zero
        verbose: bool give some informational output
    Returns:
        df: dataset encoded
    Raises:
        None
    """
    df_num = df.select_dtypes(include=['int64', 'Int64', 'float64']).astype('float64')
    columns = df_num.columns
    index = df_num.index

    scaler = MinMaxScaler()

    if center_zero:
        scaled_df = 2 * scaler.fit_transform(df_num) - 1
    else:
        scaled_df = scaler.fit_transform(df_num)
    scaled_df = pd.DataFrame(scaled_df, index=index, columns=columns)

    df = df.drop(columns, 1)
    df = df.join(scaled_df)

    if verbose:
        print("Standard scaling of all int64 and float64 columns: {}".format(columns))
    return df


# Splitting functions
def train_test_split_target(df, target, by=None, testSize=0.2):
    """Splits dataset into train and test set by a feature

    Args:
        df: pandas dataframe ml set
        target: pandas dataframe target feature set
        verbose: bool give some informational output
    Returns:
        df: dataset encoded
    Raises:
        None
    """
    if by is None:
        return train_test_split(df.values, target, test_size=testSize)

    choices = df[by].drop_duplicates()
    choices = random.choices(choices, k=int(testSize * len(choices)))
    choices = df[df[by].isin(choices)]

    xTrain = df.drop(choices.index)
    xTest = choice

    yTrain = target.drop(choices.index)
    yTest = target.loc[choices.index]

    xTrain = xTrain.drop(by, axis=1)
    xTest = xTest.drop(by, axis=1)

    if verbose:
        print("Split dataset into test and train set")
        print("  * Split by  = {}".format(by))
        print("  * Testset {}% = {}x{}".format(test_size * 100, shape[0], shape[1]))
        print("  * Trainset {}%= {}x{}".format(100 - (test_size * 100), shape[0], shape[1]))

    return xTrain, xTest, yTrain, yTest


# DecisionTrees
def getTreeInfo(tree_index=0, tree=None, df=None):
    """Get informations about a sklearn decision tree
    Args:
        tree_index: optional index if we have multiple trees
        tree: sklearn tree model
        df: dataframe that was used to train the tree, it's needed
            to get the features name
    Returns:
        df: dataset containing the informations of the tree
    Raises:
        None
    """
    # Read info about the tree
    n_nodes = tree.tree_.node_count
    children_left = tree.tree_.children_left
    children_right = tree.tree_.children_right
    feature = tree.tree_.feature
    threshold = tree.tree_.threshold
    impurity = tree.tree_.impurity
    value = tree.tree_.value

    # Get features name
    feature_name = df.columns

    # Create dataframe to store info of tree
    tree_df = pd.DataFrame(columns=["tree_index", "node_index", "node_depth", "is_leaf", "rules_list", "impurity", "value", "class", "node_children", "node_parent"])

    # run through nodes
    for node_index in range(n_nodes):
        rule_list = []
        parent_node = -1
        # Check if it's a leaf
        if (children_left[node_index] != children_right[node_index]):
            is_leaf = False
        else:
            is_leaf = True

        # get all the rules that point to this node
        parent_node_id = -1
        current_node_id = node_index
        if node_index > 0:  # First node is the root: doesn't have any rule
            # Go back until we reach the tree root
            while(parent_node_id != 0):
                # find parent node and get its rule
                if current_node_id in children_left:
                    parent_node_id = children_left.tolist().index(current_node_id)
                    rule = "{} {} {}".format(feature_name[feature[parent_node_id]], "<=", threshold[parent_node_id])
                else:
                    parent_node_id = children_right.tolist().index(current_node_id)
                    rule = "{} {} {}".format(feature_name[feature[parent_node_id]], ">", threshold[parent_node_id])

                # rule_list.append(rule)
                rule_list.insert(0, rule)
                current_node_id = parent_node_id
                if parent_node == -1:
                    parent_node = parent_node_id

        # Add in dataframe
        index = len(tree_df)
        tree_df.loc[index] = ""
        tree_df.iloc[index]["tree_index"] = tree_index
        tree_df.iloc[index]["node_index"] = node_index
        tree_df.iloc[index]["node_depth"] = len(rule_list)
        tree_df.iloc[index]["is_leaf"] = is_leaf
        tree_df.iloc[index]["rules_list"] = rule_list
        tree_df.iloc[index]["impurity"] = impurity[node_index]
        tree_df.iloc[index]["value"] = list(value[node_index][0])
        tree_df.iloc[index]["class"] = np.argmax(value[node_index][0])
        tree_df.iloc[index]["node_children"] = [children_left[node_index], children_right[node_index]]
        tree_df.iloc[index]["node_parent"] = parent_node
    return tree_df


def getForestInfo(forest, df):
    """Get informations about a sklearn forest classifier
    Args:
        forest: sklearn forest classifier model
        df: dataframe that was used to train the model, it's needed
            to get the features name
    Returns:
        df: dataset containing the informations of the forest
    Raises:
        None
    """
    for index_tree, tree in enumerate(forest.estimators_):
        # Get info about each tree
        tree_df = getTreeInfo(index_tree, tree, df)
        # Add tree df to forest df
        if index_tree == 0:
            forest_df = tree_df.copy()
        else:
            forest_df = forest_df.append(tree_df, ignore_index=True)
    return forest_df


def keepOnlyBiggestLeaf(forest_df):
    problematic_tree_list = []
    for tree_id in range(max(forest_df['tree_index']) + 1):
        # print(tree_id)
        # Get tree
        tree_df = forest_df[forest_df['tree_index'] == tree_id]

        # Recursive function to run through tree and store the branch that lead to the biggest leaf
        def storeBiggestLeaf(tree_df, node_index, tree_df2):
            # Get current node
            node = tree_df[tree_df["node_index"] == node_index].iloc[0]
            # print_color("{}".format(node_index), fg.green)
            # print_color("{}".format(node), fg.cyan)
            # store it
            tree_df2 = tree_df2.append(node, ignore_index=True)
            # print(tree_df2)

            if node["is_leaf"]:
                return tree_df2

            # Find next node
            children_node_left = tree_df[tree_df["node_index"] == node["node_children"][0]].iloc[0]
            children_node_right = tree_df[tree_df["node_index"] == node["node_children"][1]].iloc[0]
            # print(children_node_left)
            # print(children_node_right)

            # Check if children have the correct classification
            class_left = children_node_left["class"]
            class_right = children_node_right["class"]
            if class_left < class_right:
                biggest_node_index = node["node_children"][0]
            elif class_right < class_left:
                biggest_node_index = node["node_children"][1]
            else:
                # If both classification are correct choose the child with the less impurity
                impurity_left = children_node_left["impurity"]
                impurity_right = children_node_right["impurity"]

                if impurity_left < impurity_right:
                    biggest_node_index = node["node_children"][0]
                else:
                    biggest_node_index = node["node_children"][1]

            tree_df2 = storeBiggestLeaf(tree_df, biggest_node_index, tree_df2)
            return tree_df2

        tree_df2 = pd.DataFrame(columns=tree_df.columns)
        tree_df2 = storeBiggestLeaf(tree_df, 0, tree_df2)
        # print(tree_df2)
        # print_color("{}".format(tree_df2.iloc[-1]["node_index"]), fg.magenta)

        # Check if this is really the biggest leaf
        tree_leaf_df = tree_df[(tree_df["is_leaf"]) & (tree_df["class"] == 0)]
        # print_color("{}".format(tree_leaf_df), fg.green)
        argmax_of_val = np.argmax([item[0] for item in tree_leaf_df["value"]])
        biggest_leaf_index = tree_leaf_df["node_index"].tolist()[argmax_of_val]
        # tree_leaf_df.iloc[np.argmax([item[0] for item in tree_leaf_df["value"]])]["node_index"]
        # print_color("{}".format(biggest_leaf_index), fg.red)

        def getPathOfLeaf(tree_df, leaf_index):
            path = [leaf_index]
            parent_index = tree_df[tree_df["node_index"] == leaf_index].iloc[0]["node_parent"]
            while parent_index != -1:
                path.insert(0, parent_index)
                parent_index = tree_df[tree_df["node_index"] == parent_index].iloc[0]["node_parent"]
            return path

        # get all the parents nodes of the biggest leaf
        leaf_path = getPathOfLeaf(tree_df, biggest_leaf_index)
        # print(leaf_path)
        tree_df3 = tree_df[tree_df["node_index"].isin(leaf_path)]
        # print(tree_df3)

        if tree_df3.iloc[-1]["value"][0] < tree_df3.iloc[-1]["value"][0]:
            problematic_tree_list.append(tree_id)
            print_color("Warning: leaf has more bad than good plates in tree {}".format(tree_id), fg.orange)
            print(tree_df3)

        # If two methods don't return same results, store the index of the tree to plot it later
        if tree_df2.iloc[-1]["node_index"] != biggest_leaf_index:
            problematic_tree_list.append(tree_id)
            print_color("Warning: mismatch in biggest leafs for tree {}".format(tree_id), fg.orange)

        # Merge trees to recreate forest
        if tree_id == 0:
            forest_df2 = tree_df3.copy()
        else:
            forest_df2 = forest_df2.append(tree_df3, ignore_index=True)

    return forest_df2, problematic_tree_list


def convertBackOneHotRules(forest_df, label_encoders):
    """Convert back one hot rules to category from a forest dataframe
       to be used directly on original data
    Args:
        forest_df: dataframe containing the informations of the forest
                   (returned by getForestInfo function)
        label_encoders: the one-hot encoder
                        (returned by categoryToInt function)
    Returns:
        df: forest dataframe with converted rules
    Raises:
        None
    """
    df = forest_df.copy()

    # Run through each ruleset
    for index, row in df.iterrows():
        rules_list = []
        # Run thtough each rule
        for rule in row["rules_list"]:
            # print(rule)
            # Split rule to get feature, sign and threshold value
            feature, sign, val = rule.split(" ")
            # Check if feature is one-hot formated
            f1 = feature.split("_")[-1]
            f2 = feature[:-(len(f1) + 1)]
            if f2 in label_encoders.keys():
                # This is an one-hot: Convert back to category
                f3 = label_encoders[f2].inverse_transform([int(f1)])[0]
                # Convert to int in case of number
                # print(feature, type(f3), f3)
                if type(f3) == np.float64:
                    f3 = int(f3)
                    # print(feature, type(f3), f3)
                # if f3.replace('.', '', 1).isdigit():
                #     f3 = int(float(f3))

                # Verify that we only have threshold val at 0.5
                if val != str(0.5):
                    print_color("ERROR decoding rule {} {} {}: Invalid value".format(feature, sign, val), fg.red)

                # Convert sign, we should only have two cases:
                #    Feature_N <= 0.5: means Feature must not be equal to N
                #    Feature_N > 0.5: means Feature must be equal to N
                if sign == "<=":
                    rules_list.append("{} != {}".format(f2, f3))
                elif sign == ">":
                    rules_list.append("{} = {}".format(f2, f3))
                else:
                    print_color("ERROR decoding rule {} {} {}: Sign not valid".format(feature, sign, val), fg.red)

            else:
                # This is not an one-hot: Keep the same rule
                rules_list.append(rule)

        # Replace rules in dataframe
        row["rules_list"] = rules_list

    return df


def formatRuleListForFilter(rule_list, verbose=False):
    """ Format a list of rules given by a forest/tree to be
        compatible with the filterRows function
    Args:
        rule_list: List of rule to format
        verbose: bool give some informational output
    Returns:
        ml_rule: list of rule formated like: [[feature1, threshold1], ..., [featureN, thresholdN]]
        rule_type: list of operation (lte | gt | eq | neq)
    Raises:
        None
    """
    if verbose:
        print("Format rule for filerRows function, nb rules: {}".format(len(rule_list)))
    ml_rule = []
    rule_type = []
    # Run through rule in the list
    for rule in rule_list:
        if verbose:
            print(" *", rule)

        # Split rule string
        feature, sign, val = rule.split(" ")

        # Convert type of val
        try:
            ml_val = int(val)
        except ValueError:
            try:
                ml_val = float(val)
            except ValueError:
                ml_val = val

        # Convert sign for filter function
        if sign == "<=":
            rule_type.append("lte")
        elif sign == "<":
            rule_type.append("lt")
        elif sign == ">=":
            rule_type.append("gte")
        elif sign == ">":
            rule_type.append("gt")
        elif sign == "=":
            rule_type.append("eq")
            ml_val = [ml_val]
        elif sign == "!=":
            rule_type.append("neq")
            ml_val = [ml_val]
        else:
            print_color("ERROR: sign '{}' not handle by formatRuleListForFilter()".format(sign), fg.red)

        ml_rule.append([feature, ml_val])

    if verbose:
        print(" ->", ml_rule, rule_type)

    return ml_rule, rule_type


def calculate_exponential_linear_score(filteredDf, target):
    """ Calculate exponential and linear score for a given set of plates

    Args:
        df: The dataframe containing the plates filtered by the rule
        target: The target that will be used to differentiate excellent/normal plates
    Returns:
        The exponential score (looks like y = -x^3)
        The linear score (looks like y = -x)
    Raises:
        None
    """
    # Get threshold
    # threshold_list = filteredDf[target + "_MEDIAN"].unique()
    threshold_list = filteredDf[target + "_UNIQUE_THRESHOLD"].unique()
    if len(threshold_list) > 1:
        print_color("Warning: {} threshold column contains multiple different values".format(target), fg.orange)
    threshold = threshold_list[0]
    # Calculate score using y = -x^3 with x = (target - threshold) / (threshold/10). The min x should be equal to -10
    x_list = (filteredDf[target] - threshold) / (threshold / 10.0)
    score_exp_list = -x_list**3
    score_exp = np.sum(score_exp_list)

    # Calculate score using y = -x
    score_lin = np.sum(-x_list)

    return score_exp, score_lin


# def calculateRuleScore(df, ml_rule, rule_type, target, target_quality_order, verbose=False):
def calculateRuleScore(df, rules_list, target, target_quality_order, verbose=False):
    """ Calculate score for a given set of rules and the target

    Args:
        df: The dataframe containing the data (must contains only plates filterable by rules_list)
        rules_list: List of rules that will be used to filter plates from df
        target: The target that will be used to differentiate excellent/normal plates
        target_quality_order: Dictionary that indicate if a plate is excellent when the
                              target is below (0) or above (1) the median
    Returns:
        A dictionary with the following item:
         - score: The score of the ml_rule
         - nbSample: Number of samples that follow the ml_rule
         - nbGoodPlates: Number of good plates that follow the ml_rule
         - nbBadPlates: Number of bad plates that follow the ml_rule
         - filteredDf: Dataframe containing the samples that follow the ml_rule
         - ruleWeightDict: Dictionary containing the number of sample filtered by each subrules
    Raises:
        None
    """
    # Format rules to be used in filter function
    ml_rule, rule_type = formatRuleListForFilter(rules_list, False)

    # Filter df to have only point following the rule
    filteredDf = df.copy()
    ruleWeightDict = {}
    for index_rule, rule in enumerate(ml_rule):
        nb_sample_before_filter = len(filteredDf)
        filteredDf = filterRows(filteredDf, rule, rule_type[index_rule], verbose=False)
        # Keep track of how many sample are kept by each subrules
        ruleWeightDict[rules_list[index_rule]] = [len(filteredDf), len(filteredDf) - nb_sample_before_filter]

    # Create default output dict
    result_dict = {
        "score": 0,
        "scoreUnique": 0,
        "scoreExp": 0,
        "scoreLin": 0,
        "nbSample": 0,
        "nbGoodPlates": 0,
        "nbBadPlates": 0,
        "filteredDf": None,
        "ruleWeightDict": ruleWeightDict
    }

    # Get nb plates that are filtered by the rules set
    nbSample = len(filteredDf)
    if verbose:
        print("Plates filtered with ml_rule: {}/{}".format(nbSample, len(df)))
    if nbSample == 0:
        if verbose:
            print_color("Warning: 0 valid plates after filtering for rule {} {}".format(ml_rule, rule_type), fg.orange)
        return result_dict

    # Calculate median of target
    # targetMedian = np.median(df[target])

    # Filter dataframe to get only plates below the target median
    below_target_rule = [target + "_TARGET_SUP", 0.5]
    belowTargetDf = filterRows(filteredDf, below_target_rule, type="lte", verbose=False)
    if verbose:
        print("Selected plates below median: {}/{}".format(len(belowTargetDf), len(filteredDf)))

    # Filter dataframe to get only plates below the target Unique Threshold
    below_target_rule = [target + "_TARGET_SUP_UNIQUE", 0.5]
    belowUThresholdTargetDf = filterRows(filteredDf, below_target_rule, type="lte", verbose=False)
    if verbose:
        print("Selected plates below Unique Threshold: {}/{}".format(len(belowUThresholdTargetDf), len(filteredDf)))

    # Calculate score using median (nbExcellentPlates / TotalPlates)
    nbPlatesBelowTarget = len(belowTargetDf)
    nbPlatesAboveTarget = nbSample - nbPlatesBelowTarget
    if target_quality_order == 1:
        score = nbPlatesAboveTarget / nbSample
        nbGoodPlates = nbPlatesAboveTarget
        nbBadPlates = nbPlatesBelowTarget
    else:
        score = nbPlatesBelowTarget / nbSample
        nbGoodPlates = nbPlatesBelowTarget
        nbBadPlates = nbPlatesAboveTarget

    # Calculate score using Unique Threshold (nbExcellentPlates / TotalPlates)
    nbPlatesBelowTarget = len(belowUThresholdTargetDf)
    nbPlatesAboveTarget = nbSample - nbPlatesBelowTarget
    if target_quality_order == 1:
        scoreUnique = nbPlatesAboveTarget / nbSample
    else:
        scoreUnique = nbPlatesBelowTarget / nbSample

    # Calculate exponential score using y = -(x^3)
    scoreExp, scoreLin = calculate_exponential_linear_score(filteredDf, target)

    # Store result in output dictionary
    result_dict["score"] = score
    result_dict["scoreUnique"] = scoreUnique
    result_dict["scoreExp"] = scoreExp
    result_dict["scoreLin"] = scoreLin
    result_dict["nbSample"] = nbSample
    result_dict["nbGoodPlates"] = nbGoodPlates
    result_dict["nbBadPlates"] = nbBadPlates
    result_dict["filteredDf"] = filteredDf
    return result_dict


def calculateRandomForestScore(forest_df, df, target, target_quality_order, normalize=False, verbose=False):
    """ Calculate the score of each ruleset of a forest for a given target

    Args:
        forest_df: dataframe of a sklearn forest (given by getForestInfo())
        df: dataframe containing the data to use
        target: column of df to use as the target
        target_quality_order: Dictionary that indicate if a plate is excellent when the
                              target is below (0) or above (1) the median
    Returns:
        score_df: dataframe containing the score for each ruleset
    Raises:
        None
    """
    if verbose:
        print("Spliting dataframe for each thickness...")
    thickness_list = [15.0, 20.0, 25.0, 30.0, 35.0, 40.0]
    df_list = []
    # Create new dataframe to store the score
    score_columns = ["rules_list", "occurence", "origin", "similar_rules", "score_total"]
    for thickness in thickness_list:
        score_columns.append("totalSample_{}".format(thickness))
        score_columns.append("usableSample_{}".format(thickness))
        score_columns.append("goodPlates_{}".format(thickness))
        score_columns.append("badPlates_{}".format(thickness))
        score_columns.append("score_{}".format(thickness))
        score_columns.append("score_unique_{}".format(thickness))
        score_columns.append("score_expo_{}".format(thickness))
        score_columns.append("score_lin_{}".format(thickness))
        score_columns.append("score_nbSample_{}".format(thickness))
        score_columns.append("rules_weight_{}".format(thickness))

        # Create list of dataframe grouped by thickness
        df_by_thickness = filterRows(df, ["EPAISSEUR_FINALE", [thickness]], type="eq", verbose=False)
        df_list.append(df_by_thickness)
        if verbose:
            print(" * {} samples for thickness {}".format(len(df_by_thickness), thickness))

    score_df = pd.DataFrame(columns=score_columns, dtype='int64')

    # Calculating score for each rule
    if verbose:
        print("Calculating score for each ruleset...")
    for index, row in forest_df.iterrows():
        if verbose:
            print(" * Ruleset {}/{}".format(index, len(forest_df)))
        # Doesn't process root of trees
        if "node_index" in row and row["node_index"] == 0:
            continue

        # Add new entry in score_df
        scoreIndex = len(score_df)
        score_df.loc[scoreIndex] = ""
        score_df.iloc[scoreIndex]["rules_list"] = row["rules_list"]
        score_df.iloc[scoreIndex]["occurence"] = row["occurence"]
        score_df.iloc[scoreIndex]["origin"] = row["origin"]
        score_df.iloc[scoreIndex]["similar_rules"] = row["similar_rules"]

        # Format rules to be used in filter function
        # ml_rule, rule_type = formatRuleListForFilter(row["rules_list"], False)
        # Get feature list
        feature_list = [item.split(" ")[0] for item in row["rules_list"]]

        # for thickness in np.sort(df['EPAISSEUR_FINALE'].unique()):
        score_total = 0
        for index_thickness, thickness in enumerate(thickness_list):
            # Get only plates with corresponding thickness
            df_tmp = df_list[index_thickness]
            nb_plates_for_this_thickness = len(df_tmp)

            # Clean columns of df that are affected by the ruleset
            # feature_list = [item[0] for item in ml_rule]
            if len(feature_list) > 0:
                df_tmp = removeNaN(df_tmp, feature_list, False)
            usableSamples = len(df_tmp)

            # Calculate score for this rules set
            # score, score_max, goodPlates, badPlates, filteredDf, ruleWeightDict = calculateRuleScore(df_tmp, row["rules_list"], target, target_quality_order, verbose)
            score_dict = calculateRuleScore(df_tmp, row["rules_list"], target, target_quality_order, verbose)
            score = score_dict["score"]
            scoreUnique = score_dict["scoreUnique"]
            scoreExp = score_dict["scoreExp"]
            scoreLin = score_dict["scoreLin"]
            nbSample = score_dict["nbSample"]
            goodPlates = score_dict["nbGoodPlates"]
            badPlates = score_dict["nbBadPlates"]
            ruleWeightDict = score_dict["ruleWeightDict"]

            score_df.iloc[scoreIndex]["totalSample_{}".format(thickness)] = nb_plates_for_this_thickness
            score_df.iloc[scoreIndex]["usableSample_{}".format(thickness)] = usableSamples
            score_df.iloc[scoreIndex]["goodPlates_{}".format(thickness)] = goodPlates
            score_df.iloc[scoreIndex]["badPlates_{}".format(thickness)] = badPlates
            score_df.iloc[scoreIndex]["score_{}".format(thickness)] = score
            score_df.iloc[scoreIndex]["score_unique_{}".format(thickness)] = scoreUnique
            score_df.iloc[scoreIndex]["score_expo_{}".format(thickness)] = scoreExp
            score_df.iloc[scoreIndex]["score_lin_{}".format(thickness)] = scoreLin
            score_df.iloc[scoreIndex]["score_nbSample_{}".format(thickness)] = nbSample
            score_df.iloc[scoreIndex]["rules_weight_{}".format(thickness)] = ruleWeightDict

            # Calculate total score for this rules set
            score_total += score
        score_df.iloc[scoreIndex]["score_total"] = score_total

    # Normalize score for each thickness
    if normalize:
        if verbose:
            print("Normalize score for each thickness...")
        for index_thickness, thickness in enumerate(thickness_list):
            max_score = max(np.abs(score_df["score_{}".format(thickness)]))
            score_df["score_{}".format(thickness)] = score_df["score_{}".format(thickness)] / max_score * 100
        # Recalculate score total
        columns_score = ["score_{}".format(thickness) for thickness in thickness_list]
        score_df["score_total"] = score_df.loc[:, columns_score].sum(axis=1)
    return score_df


# Family of rules
def get_family(rule):
    """ Return the family of a rule (feature + sign)

    Args:
        rule: the rule (feature + sign + threshold)
    Returns:
        family: the family (feature + sign)
    Raises:
        None
    """
    feature, sign, threshold = rule.split(" ")
    family = "{} {}".format(feature, sign)
    return family


def get_threshold(rule):
    """ Return the threshold of a rule

    Args:
        rule: the rule
    Returns:
        threshold: the threshold
    Raises:
        None
    """
    feature, sign, threshold = rule.split(" ")
    return float(threshold)


def get_reverse_family(family):
    """ Return the inverse of a family (feature + inverted sign)

    Args:
        family: the family
    Returns:
        inverse_family: the inverse of the family
    Raises:
        None
    """
    feature, sign = family.split(" ")
    if sign == "<=":
        sign = ">"
    elif sign == ">":
        sign = "<="
    elif sign == "=":
        sign = "!="
    elif sign == "!=":
        sign = "="
    else:
        print_color("Error: Sign {} not handle by get_reverse_family()".format(sign), fg.red)
    inverse_family = "{} {}".format(feature, sign)
    return inverse_family


def get_sub_family(ruleset_list, family_list):
    """ Return the list of sub family that are present in ruleset_list without the one in family_list

    Args:
        ruleset_list: list of ruleset from where we want to export the family
        family_list: list of family to exclude from the returned list
    Returns:
        sub_family_list: The sub_family list
    Raises:
        None
    """
    # Get all sub rules of a each ruleset that are not present in family_list
    sub_family_list = [get_family(rule) for ruleset in ruleset_list for rule in ruleset if get_family(rule) not in family_list]
    # List uniques item
    sub_family_list = list(set(sub_family_list))

    return sub_family_list


def replace_sign_to_text(family):
    """ Convert the sign of a family to the corresponding text
        used for html files when the family is in the name of the file

    Args:
        family: The family
    Returns:
        formated_family: The family as a string with the sign converted
    Raises:
        None
    """
    return family.replace(" <=", "_lte").replace(" >", "_gt").replace(" !=", "_neq").replace(" =", "_eq")


def check_if_is_family(ruleset, family):
    """ Check if a ruleset is part of a family

    Args:
        ruleset: The set of rules to check
        family: The family, must be a list
    Returns:
        part_of_family: 1 if ruleset is part of the family, 0 otherwise
    Raises:
        None
    """
    # Convert ruleset to string
    # ruleset_str = ', '.join(ruleset)
    # Transform ruleset to family list (remove threshold)
    ruleset_fam = [get_family(item) for item in ruleset]
    # Check if every elements of the family is present in the ruleset
    for item in family:
        if item not in ruleset_fam:
            return 0
    return 1


def add_family_count(df):
    """ Add a new column in the dataframe containing the number of rules
        of the same family

    Args:
        df: The dataframe
    Returns:
        df: The dataframe with the new column "family_count"
    Raises:
        None
    """
    df_tmp = df.copy()

    # Trial to improve speed..
    # family_list = df["rules_list"].apply(lambda x: [get_family(rule) for rule in x])
    # def bla(family, df):
    #    family_count = df["rules_list"].apply(lambda x:)
    # family_count_list = family_list.apply(lambda x: bla(x ,df))

    # Count same family rules
    family_count_list = []
    for ruleset in df["rules_list"]:
        # Get family from ruleset
        family = [get_family(rule) for rule in ruleset]
        # Extract rows that contains this family
        df_tmp["in_family"] = df_tmp['rules_list'].apply(lambda x: check_if_is_family(x, family))
        df_tmp2 = df_tmp[df_tmp["in_family"] == 1]
        # Sum occurence
        family_count_list.append(sum(df_tmp2["occurence"]))

    # Add the new column in the dataset
    df["family_count"] = family_count_list
    return df


def extract_family_root(df):
    """ Extract and list all root families from a dataframe
        A family with only 1 condition is considered root

    Args:
        df: The dataframe, must contains a column "rules_list"
    Returns:
        family_root: list of root families
    Raises:
        None
    """
    # Get ruleset with only 1 sub-rule, extract the feature + sign and put them in a set to have only 1 occurence of each
    rules_list = df["rules_list"].tolist()
    family_root = sorted(list(set([get_family(item[0]) for item in rules_list if len(item) == 1])))
    return family_root


def get_family_sample_count(df, thickness, family):
    """ Extract the number of sample filtered by the family from a dataframe

    Args:
        df: The score dataframe, must contains rules_weight_{} column
        thickness: the thickness to use
        family: the family from which we want the sample count
    Returns:
        sample_count: the number of sample filtered by the family
    Raises:
        None
    """
    sample_count = 0
    for index, row in df.iterrows():
        rules_weight = row["rules_weight_{}".format(thickness)]
        for key in rules_weight.keys():
            if get_family(key) == family:
                sample_count += rules_weight[key][0]
    return sample_count


def get_family_importance(df, thickness, family_list, threshold_sample_min, threshold_inf, threshold_sup, verbose=False):
    """ Extract the number of significant rules for each family in family_list and return
        the result as a sorted list of dictionary

    Args:
        df: The score dataframe, must contains score for each thickness
        thickness: the thickness to get the score from
        family_list: the family to use to filter rules
        threshold_sample_min: threshold to filers valid rules
        threshold_inf: dict of score threshold to filters significant bad rules
                       must contains 1 entry for each thickness
        threshold_sup: dict of score threshold to filters significant good rules
                       must contains 1 entry for each thickness
    Returns:
        family_importance: list of dict containing for each family:
                           family: The current family (Feature + sign),
                           reverse_family: The reverse of the current family (Feature + inverted sign),
                           nb_significant_rules: The number of rules of this family in the "significant" area
                           display_df: Dataframe containing all the rules of the current family
                           good_df: Dataframe containing only the good rules of the current family in the significant area
                           bad_df: Dataframe containing only the bad rules of the current family in the significant area

    Raises:
        None
    """
    family_importance = []
    for rule_family in family_list:
        if verbose:
            print_color(rule_family, fg.cyan)
            # print(rule_family)
        score_df_d1 = df.copy()

        # Get reverse rule
        reverse_rule_family = get_reverse_family(rule_family)

        # Create new column that indicate if it's part of the family
        score_df_d1['in_family'] = score_df_d1['rules_list'].apply(lambda x: check_if_is_family(x, [rule_family]))
        score_df_d1['in_reverse_family'] = score_df_d1['rules_list'].apply(lambda x: check_if_is_family(x, [reverse_rule_family]))

        # Get only entries that are part of the family
        score_df_family = score_df_d1[(score_df_d1["in_family"] == 1) & (score_df_d1["score_{}".format(thickness)] > 0.5)]
        score_df_reverse_family = score_df_d1[(score_df_d1["in_reverse_family"] == 1) & (score_df_d1["score_{}".format(thickness)] <= 0.5)]
        if verbose:
            print(" Nb good rules using {}: {}/{}".format(rule_family, len(score_df_family), len(score_df_d1)))
            print(" Nb bad rules using {}: {}/{}".format(reverse_rule_family, len(score_df_reverse_family), len(score_df_d1)))

        # Merge two dataframes to have a global df for display
        score_df_display = score_df_family.append(score_df_reverse_family, ignore_index=True)

        # Count how many good rules are significant (drop rules that are below threshold)
        score_df_family_significant = score_df_family.drop(score_df_family[score_df_family["score_{}".format(thickness)] < threshold_sup[thickness]].index)
        score_df_family_significant = score_df_family_significant.drop(score_df_family_significant[score_df_family_significant["score_nbSample_{}".format(thickness)] < threshold_sample_min].index)
        # Count how many bad rules are significant (drop rules that are above threshold)
        score_df_reverse_family_significant = score_df_reverse_family.drop(score_df_reverse_family[score_df_reverse_family["score_{}".format(thickness)] > threshold_inf[thickness]].index)
        score_df_reverse_family_significant = score_df_reverse_family_significant.drop(score_df_reverse_family_significant[score_df_reverse_family_significant["score_nbSample_{}".format(thickness)] < threshold_sample_min].index)

        if verbose:
            print(" Nb significant good rules: {}".format(len(score_df_family_significant)))
            print(" Nb significant bad rules: {}".format(len(score_df_reverse_family_significant)))

        # Count total of significant rules
        # nb_significant_rules = len(score_df_reverse_family_significant) + len(score_df_family_significant)
        nb_good_rules = len(score_df_family_significant)
        nb_bad_rules = len(score_df_reverse_family_significant)

        # Count the number of samples filtered by the rule
        sample_count = get_family_sample_count(score_df_family_significant, thickness, rule_family)

        # Store in a list the family only if there is some rules of this family above the threshold
        if nb_good_rules > 0:
            family_json = {"family": rule_family, "reverse_family": reverse_rule_family,
                           "nb_good_rules": nb_good_rules, "nb_bad_rules": nb_bad_rules,
                           "display_df": score_df_display["rule_index"].tolist(), "good_df": score_df_family_significant[["rules_list"]], "bad_df": score_df_reverse_family_significant[["rules_list"]],
                           "sample_count": sample_count}
            family_importance.append(family_json)

    # Sort family by number of good rules
    family_importance.sort(key=lambda tup: tup["nb_good_rules"], reverse=True)

    return family_importance


def extract_threshold_list(df, family):
    """ Extract and list all thresholds of a family from a dataframe

    Args:
        df: The dataframe, must contains a column "rules_list"
        family: the family to filter threshold
    Returns:
        threshold_list: list of thresholds values
    Raises:
        None
    """
    # Get ruleset_list
    ruleset_list = df["rules_list"].tolist()
    # Get only rules of the family and extract threshold
    threshold_list = [get_threshold(rule) for ruleset in ruleset_list for rule in ruleset if family in rule]
    return threshold_list


def merge_rules_order(list_of_ruleset):
    """ Merge a list of unordered but similar rules to keep only the one 
        that appear the most
    Args:
        list_of_ruleset: List of similar ruleset in format: [[A < 5, B > 3], [B > 3, A < 5], [B > 3, A < 5], ...]
    Returns:
        merged_order_list: The ruleset with the most present order ex: [B > 3, A < 5]
    Raises:
        None
    """
    order_dict = {}
    # Run through rules_list
    for rules_list in list_of_ruleset:
        # print("-", rules_list)
        for index, rule in enumerate(rules_list):
            # Init dict entry if not already there
            if rule not in order_dict:
                order_dict[rule] = 0
            # Increment importance score for rule
            order_dict[rule] += 1 / (index + 1)

    # Order from higher to lower rule importance
    merged_order_list = [k for k, v in sorted(order_dict.items(), key=lambda item: item[1], reverse=True)]

    return merged_order_list


def group_similar_rules(df):
    """ Count number of occurence and similar rules inside a dataframe

    Args:
        df: The dataframe, must contains columns "rules_list" and "forest_id"
    Returns:
        df: The dataframe with columns "rules_list", "occurence", "origin", "similar_rules"
    Raises:
        None
    """
    df_tmp = df.copy()

    # Create new column with rules_list in alphabetical order
    df_tmp['rules_list_sorted'] = df_tmp['rules_list'].apply(lambda x: sorted(x))

    # Convert ruleset to string to regroup them
    df_tmp['rules_list_sorted'] = df_tmp['rules_list_sorted'].apply(lambda x: ', '.join(x))

    # Create list of origin
    df1 = df_tmp.groupby('rules_list_sorted')['forest_id'].apply(lambda x: x.unique().tolist())
    # print(df1)
    df2 = df_tmp.groupby('rules_list_sorted').size().reset_index(name='occurence')
    # print(df2)
    # Keep the more relevant order of rules
    df3 = df_tmp.groupby('rules_list_sorted')['rules_list'].apply(lambda x: merge_rules_order(x.tolist()))

    # Merge temp datasets
    df_result = pd.merge(df1, df2, on='rules_list_sorted')
    df_result.rename(columns={"forest_id": "origin"}, inplace=True)
    df_result = pd.merge(df_result, df3, on="rules_list_sorted")

    # Convert back ruleset to list
    df_result['rules_list_sorted'] = df_result['rules_list_sorted'].apply(lambda x: x.split(", "))
    # print(df_result)

    # Generate rule_uid_similar to find similar rules
    def generate_rule_uid_similar(ruleset):
        # This function run through the ruleset and keep:
        # a. only the feature and sign of each rule if the sign is "<" ">", "<=", ">="
        # b. everything if the sign is "==" or "!="
        return [item if item.split(" ")[1] in ["=", "!="] else item[:-len(item.split(" ")[2]) - 1] for item in ruleset]
    df_result['rule_uid_similar'] = df_result['rules_list_sorted'].apply(lambda x: generate_rule_uid_similar(x))
    # Convert to string
    df_result['rule_uid_similar'] = df_result['rule_uid_similar'].apply(lambda x: ', '.join(x))

    # Sum all rules that are similar
    df4 = df_result.groupby('rule_uid_similar')['occurence'].sum().reset_index(name='similar_rules')
    # print(df4)

    # Merge into dataframe
    df_result = pd.merge(df_result, df4, on='rule_uid_similar')
    # print(df_result)

    return df_result[["rules_list", "occurence", "origin", "similar_rules"]]


# Storage class for dataframe
class DataframeList:
    def __init__(self, filename=None):
        """Create a new dataframeList

        Args:
            filename: filepath/name of an existing .h5 file if we want to load
        Returns:
            The DataframeList
        Raises:
            None
        """
        self.datasets_count = 0
        self.datasets_name = []
        self.datasets_description = []
        self.datasets = []
        self.default_name = "split"

        # Load from an existing h5 file
        if filename:
            with pd.HDFStore(filename) as hdf:
                # Read Datasets_information
                df_info = hdf.select("Datasets_Information")
                self.datasets_name = df_info["Name"].values.tolist()
                self.datasets_description = df_info["Description"].values.tolist()
                # Read each dataframe listed in Datasets_Information
                for df_name in self.datasets_name:
                    df = hdf.select(df_name)
                    self.datasets.append(df)
                    self.datasets_count += 1

    def addDf(self, name=None, description=None, df=None):
        """Add a new dataframe to the dataframeList

        Args:
            name: name of the DataFrame
            description: description of the DataFrame
            df: the DataFrame itself
        Returns:
            None
        Raises:
            None
        """
        if df is None:
            print_color(" Error: you must pass a dataframe", fg.red)
        else:
            if name is None:
                name = "{}_{}".format(self.default_name, self.datasets_count)
            self.datasets_name.append(name)
            self.datasets_description.append(description)
            self.datasets.append(df)
            self.datasets_count += 1

    def getInfo(self, verbose=False):
        """Get/Display info about this dataframeList

        Args:
            None
        Returns:
            dfs_informations: Contains info about the dataframeList
        Raises:
            None
        """
        columns = ['Name',
                   'Description',
                   'Features',
                   'Entries',
                   'Entries_Non_NaN',
                   'Entries_NaN',
                   'Total_count_NaN',
                   ]
        column_types = ['category',
                        'category',
                        'int64',
                        'int64',
                        'int64',
                        'int64',
                        'int64',
                        ]
        dfs_informations = pd.DataFrame(columns=columns)
        if verbose:
            print("| {:10} | {}                    | {} | {} | {} | {} | {} |".format(columns[0], columns[1], columns[2], columns[3], columns[4], columns[5], columns[6]))
            print("|------------|--------------------------------|----------|---------|-----------------|-------------|-----------------|")
        for i in range(self.datasets_count):
            dfs_informations.loc[i] = df_info(self.datasets_name[i], self.datasets_description[i], self.datasets[i], verbose)
        # Fix datatypes
        dfs_informations = fix_types(dfs_informations, column_types, verbose=False)
        return dfs_informations

    def getDfFromName(self, name):
        """Get a specific dataframe given its name

        Args:
            name: name of the dataframe
        Returns:
            the dataframe
        Raises:
            None
        """
        if name in self.datasets_name:
            return self.datasets[self.datasets_name.index(name)]
        else:
            print_color(" Error: {} not found in DataframeList".format(name), fg.red)

    def exportToH5(self, filename, verbose=True):
        """Export to a .h5 file

        Args:
            filename: filepath and name of where we want to export
        Returns:
            None
        Raises:
            None
        """
        if os.path.exists(filename):
            os.remove(filename)
        with pd.HDFStore(filename, mode='w') as hdf:
            datasets_info = self.getInfo()
            hdf.put("Datasets_Information", datasets_info, format='table', data_columns=True)
            # hdf.put("Target_Dataset", target_df, format='table', data_columns=True)
            if verbose:
                print("Dataset stored in HDF\n{}".format(filename))
                print("  * [{:03}] - {}".format(1, "Datasets_Information"))
                # print("  * [{:03}] - {}".format(target_loc, target_name))
            for i in range(len(self.datasets_name)):
                hdf.put(self.datasets_name[i], self.datasets[i], format='table', data_columns=True)
                if verbose:
                    print("  * [{:03}] - {} - {}".format(i + 2, self.datasets_name[i], datasets_info['Description'][i]))
            print("Done")
