import csv
import getopt
import json
import os
import statistics
import sys
import numpy as np
import pandas as pd

"""
* This file makes the following assumptions:
    * numpy and pandas libraries are required
    * All CSV files in the dataset directory will be parsed, so the directory should only contain CSV files from the same runs
        * It also does not filter by model! So different models' CSV log files are expected to be split up into different directories
            * This lack of filtering by model is important because multiple models in the same directory will cause inaccurate reporting
    * Other file formats are not supported, only .csv files
    * Experimental group is expected to exist; otherwise, this script will not work
    * The "config" column in the output CSV will have the name of the config file (referred to as a prefix)
    * The following metrics are reported (see the headers in append_to_csv for the output file format):
        * The following metrics are aggregated first by taking the average across all timesteps per run,
        * weighted by population at each timestep, then taking the mean and std dev across runs:
            * Life expectancy
            * TTL
            * Wealth
        * The following interaction metrics are aggregated by taking their rates across all timesteps per run,
        * then taking the mean and std dev of those rates across runs:
            * Lending interaction totals and rates (mean and std dev across runs' total rates), same-group and different-group
            * Reproduction interaction totals and rates (mean and std dev across runs' total rates), same-group and different-group
            * Trade interaction totals and rates (mean and std dev across runs' total rates), same-group and different-group
        * Other metrics reported:
            * Population at final timestep (mean and std dev across runs) by group and in total
            * Extinction of control group, experimental group, and both (total count and percentage)

"""


def get_terminal_width(default=80):
    try:
        return os.get_terminal_size().columns
    except OSError:
        return default

def parse_options():
    command_line_args = sys.argv[1:]
    short_options = "c:p:o:h"
    long_options = (
        "conf=",
        "path=",
        "output=",
        "help",
    )
    options = {
        "config": None,
        "path": None,
        "output": None,
    }

    try:
        args, _ = getopt.getopt(command_line_args, short_options, long_options)
    except getopt.GetoptError as err:
        print(f"Error: {str(err)}")
        print_help()

    for curr_arg, curr_val in args:
        if curr_arg in ("-c", "--conf"):
            options["config"] = curr_val
            if curr_val == "":
                print("No config file provided.")
                print_help()
        elif curr_arg in ("-p", "--path"):
            options["path"] = curr_val
            if curr_val == "":
                print("No dataset path provided.")
                print_help()
        elif curr_arg in ("-o", "--output"):
            options["output"] = curr_val
            if curr_val == "":
                print("No output CSV path provided.")
                print_help()
        elif curr_arg in ("-h", "--help"):
            print_help()

    missing_required = False
    if options["path"] is None:
        print("Dataset path required.")
        missing_required = True
    if options["config"] is None:
        print("Configuration file path required.")
        missing_required = True
    if missing_required:
        print_help()

    return options


def print_help():
    print(
        "Usage:\n"
        "\tpython report_life_expectancy.py --path /path/to/data --conf /path/to/config [options]\n\n"
        "Options:\n"
        "\t-c,--conf\tUse the specified path to configurable settings file.\n"
        "\t-p,--path\tUse the specified path to find dataset CSV files.\n"
        "\t-o,--output\tCSV output path. A new file is created if it does not exist.\n"
        "\t-h,--help\tDisplay this message."
    )
    sys.exit(0)


# Build columns functions have a fallback for if experimental group is not specified, but this script is really only intended to be used with an experimental group

def build_population_columns(experimental_group):
    if experimental_group is None:
        control_population_column = None
        experimental_population_column = "population"
    else:
        control_population_column = "controlPopulation"
        experimental_population_column = experimental_group + "Population"
    return control_population_column, experimental_population_column


def build_life_expectancy_columns(experimental_group):
    if experimental_group is None:
        control_life_expectancy_column = None
        experimental_life_expectancy_column = "meanAgeAtDeath"
    else:
        control_life_expectancy_column = "controlMeanAgeAtDeath"
        experimental_life_expectancy_column = experimental_group + "MeanAgeAtDeath"
    return control_life_expectancy_column, experimental_life_expectancy_column

def build_TTL_columns(experimental_group):
    if experimental_group is None:
        control_TTL_column = None
        experimental_TTL_column = "agentMeanTimeToLive"
    else:
        control_TTL_column = "controlAgentMeanTimeToLive"
        experimental_TTL_column = experimental_group + "AgentMeanTimeToLive"
    return control_TTL_column, experimental_TTL_column

def build_wealth_columns(experimental_group):
    if experimental_group is None:
        control_wealth_column = None
        experimental_wealth_column = "meanWealth"
    else:
        control_wealth_column = "controlMeanWealth"
        experimental_wealth_column = experimental_group + "MeanWealth"
    return control_wealth_column, experimental_wealth_column

def build_interaction_columns(experimental_group):
    if experimental_group is None:
        print("No experimental group specified, skipping interaction columns.")
        return {}, {}, {}
    else:
        lending_columns = {}
        lending_columns["control_to_control_lending_column"] = "lendingControlGroupToControlGroup"
        lending_columns["control_to_experimental_lending_column"] = "lendingControlGroupToExperimentalGroup"
        lending_columns["experimental_to_control_lending_column"] = "lendingExperimentalGroupToControlGroup"
        lending_columns["experimental_to_experimental_lending_column"] = "lendingExperimentalGroupToExperimentalGroup"

        reproduction_columns = {}
        reproduction_columns["control_to_control_reproduction_column"] = "reproductionControlGroupToControlGroup"
        reproduction_columns["control_to_experimental_reproduction_column"] = "reproductionControlGroupToExperimentalGroup"
        reproduction_columns["experimental_to_control_reproduction_column"] = "reproductionExperimentalGroupToControlGroup"
        reproduction_columns["experimental_to_experimental_reproduction_column"] = "reproductionExperimentalGroupToExperimentalGroup"

        trade_columns = {}
        trade_columns["control_to_control_trade_column"] = "tradeControlGroupToControlGroup"
        trade_columns["control_to_experimental_trade_column"] = "tradeControlGroupToExperimentalGroup"
        trade_columns["experimental_to_control_trade_column"] = "tradeExperimentalGroupToControlGroup"
        trade_columns["experimental_to_experimental_trade_column"] = "tradeExperimentalGroupToExperimentalGroup"

    return lending_columns, reproduction_columns, trade_columns


def print_progress(filename, files_parsed, total_files, file_length, decimals=2):
    if total_files == 0:
        return
    bar_length = get_terminal_width() // 2
    progress = round(((files_parsed / total_files) * 100), decimals)
    filled_length = (bar_length * files_parsed) // total_files
    bar = "█" * filled_length + "-" * (bar_length - filled_length)
    print_string = f"\rParsing {filename:>{file_length}}: |{bar}| {files_parsed} / {total_files} ({progress}%)"
    if files_parsed == total_files:
        print(f"\r{' ' * get_terminal_width()}", end="\r")
    else:
        print(f"\r{print_string}", end="\r")


def get_single_run_stats(
    df,
    control_population_column,
    experimental_population_column,
    control_life_expectancy_column,
    experimental_life_expectancy_column,
    control_TTL_column,
    experimental_TTL_column,
    control_wealth_column,
    experimental_wealth_column,
    lending_columns,
    reproduction_columns,
    trade_columns
):  
    if any(column not in df.columns for column in [
        control_population_column,
        experimental_population_column,
    ] if column is not None):
        raise ValueError("Population columns are missing from the DataFrame.")
    
    if all(column in df.columns for column in [
        control_life_expectancy_column,
        experimental_life_expectancy_column,
    ] if column is not None):
        try:
            control_life_expectancy_weighted_avg = np.average(df[control_life_expectancy_column], weights=df[control_population_column])
            experimental_life_expectancy_weighted_avg = np.average(df[experimental_life_expectancy_column], weights=df[experimental_population_column])
            total_life_expectancy_weighted_avg = np.average(df['meanAgeAtDeath'], weights=df['population'])
        except ZeroDivisionError:
            print("Warning: Population weights sum to zero for this file, cannot compute weighted average. Setting to 0.")
            control_life_expectancy_weighted_avg = 0
            experimental_life_expectancy_weighted_avg = 0
            total_life_expectancy_weighted_avg = 0
    else:
        print("One or more required columns are missing for life expectancy calculations from the DataFrame.")
        control_life_expectancy_weighted_avg = 0
        experimental_life_expectancy_weighted_avg = 0
        total_life_expectancy_weighted_avg = 0

    if all(column in df.columns for column in [
        control_TTL_column,
        experimental_TTL_column,
    ] if column is not None):
        try:
            control_TTL_weighted_avg = np.average(df[control_TTL_column], weights=df[control_population_column])
            experimental_TTL_weighted_avg = np.average(df[experimental_TTL_column], weights=df[experimental_population_column])
            total_TTL_weighted_avg = np.average(df['agentMeanTimeToLive'], weights=df['population'])
        except ZeroDivisionError:
            print("Warning: Population weights sum to zero for this file, cannot compute weighted average. Setting to 0.")
            control_TTL_weighted_avg = 0
            experimental_TTL_weighted_avg = 0
            total_TTL_weighted_avg = 0
    else:
        print("One or more required columns are missing for TTL calculations from the DataFrame.")
        control_TTL_weighted_avg = 0
        experimental_TTL_weighted_avg = 0
        total_TTL_weighted_avg = 0

    if all(column in df.columns for column in [
        control_wealth_column,
        experimental_wealth_column,
    ] if column is not None):
        try:
            control_wealth_weighted_avg = np.average(df[control_wealth_column], weights=df[control_population_column])
            experimental_wealth_weighted_avg = np.average(df[experimental_wealth_column], weights=df[experimental_population_column])
            total_wealth_weighted_avg = np.average(df['meanWealth'], weights=df['population'])
        except ZeroDivisionError:
            print("Warning: Population weights sum to zero for this file, cannot compute weighted average. Setting to 0.")
            control_wealth_weighted_avg = 0
            experimental_wealth_weighted_avg = 0
            total_wealth_weighted_avg = 0
    else:
        print("One or more required columns are missing for wealth calculations from the DataFrame.")
        control_wealth_weighted_avg = 0
        experimental_wealth_weighted_avg = 0
        total_wealth_weighted_avg = 0
    
    if all(column in df.columns for column in [
        lending_columns["control_to_control_lending_column"],
        lending_columns["control_to_experimental_lending_column"],
        lending_columns["experimental_to_control_lending_column"],
        lending_columns["experimental_to_experimental_lending_column"],
    ]):
        same_group_lending = (df[lending_columns["control_to_control_lending_column"]] + df[lending_columns["experimental_to_experimental_lending_column"]]).sum()
        different_group_lending = (df[lending_columns["control_to_experimental_lending_column"]] + df[lending_columns["experimental_to_control_lending_column"]]).sum()
        total_lending = same_group_lending + different_group_lending
        same_group_lending_rate = same_group_lending / total_lending if total_lending > 0 else 0
        different_group_lending_rate = different_group_lending / total_lending if total_lending > 0 else 0
    else:
        print("One or more required columns are missing for lending interactions from the DataFrame.")
        same_group_lending_rate = 0
        different_group_lending_rate = 0

    if all(column in df.columns for column in [
        reproduction_columns["control_to_control_reproduction_column"],
        reproduction_columns["control_to_experimental_reproduction_column"],
        reproduction_columns["experimental_to_control_reproduction_column"],
        reproduction_columns["experimental_to_experimental_reproduction_column"],
    ]):
        same_group_reproduction = (df[reproduction_columns["control_to_control_reproduction_column"]] + df[reproduction_columns["experimental_to_experimental_reproduction_column"]]).sum()
        different_group_reproduction = (df[reproduction_columns["control_to_experimental_reproduction_column"]] + df[reproduction_columns["experimental_to_control_reproduction_column"]]).sum()
        total_reproduction = same_group_reproduction + different_group_reproduction
        same_group_reproduction_rate = same_group_reproduction / total_reproduction if total_reproduction > 0 else 0
        different_group_reproduction_rate = different_group_reproduction / total_reproduction if total_reproduction > 0 else 0
    else:
        print("One or more required columns are missing for reproduction interactions from the DataFrame.")
        same_group_reproduction_rate = 0
        different_group_reproduction_rate = 0

    if all(column in df.columns for column in [
        trade_columns["control_to_control_trade_column"],
        trade_columns["control_to_experimental_trade_column"],
        trade_columns["experimental_to_control_trade_column"],
        trade_columns["experimental_to_experimental_trade_column"],
    ]):
        same_group_trade = (df[trade_columns["control_to_control_trade_column"]] + df[trade_columns["experimental_to_experimental_trade_column"]]).sum()
        different_group_trade = (df[trade_columns["control_to_experimental_trade_column"]] + df[trade_columns["experimental_to_control_trade_column"]]).sum()
        total_trade = same_group_trade + different_group_trade
        same_group_trade_rate = same_group_trade / total_trade if total_trade > 0 else 0
        different_group_trade_rate = different_group_trade / total_trade if total_trade > 0 else 0
    else:
        print("One or more required columns are missing for trade interactions from the DataFrame.")
        same_group_trade_rate = 0
        different_group_trade_rate = 0

    control_extinct = True if control_population_column is not None and df[control_population_column].iloc[-1] == 0 else False
    experimental_extinct = True if experimental_population_column is not None and df[experimental_population_column].iloc[-1] == 0 else False
    both_extinct = control_extinct and experimental_extinct

    control_population_at_final_timestep = df[control_population_column].iloc[-1] if control_population_column is not None else None
    experimental_population_at_final_timestep = df[experimental_population_column].iloc[-1] if experimental_population_column is not None else None
    total_population_at_final_timestep = df['population'].iloc[-1]

    return {"life_expectancy": (control_life_expectancy_weighted_avg, experimental_life_expectancy_weighted_avg, total_life_expectancy_weighted_avg),
            "TTL": (control_TTL_weighted_avg, experimental_TTL_weighted_avg, total_TTL_weighted_avg),
            "wealth": (control_wealth_weighted_avg, experimental_wealth_weighted_avg, total_wealth_weighted_avg),
            "lending_rates": (same_group_lending_rate, different_group_lending_rate),
            "reproduction_rates": (same_group_reproduction_rate, different_group_reproduction_rate),
            "trade_rates": (same_group_trade_rate, different_group_trade_rate),
            "extinctions": (control_extinct, experimental_extinct, both_extinct),
            "population_at_final_timestep": (control_population_at_final_timestep, experimental_population_at_final_timestep, total_population_at_final_timestep)
            }

def scan_dataset(path, experimental_group=None):
    encoded_dir = os.fsencode(path)
    # FIXME: This scans through ANYTHING in the directory that ends with .csv,
    # so if there are other csv files in the directory that don't match the expected format, it might silently fail and report inaccurate metrics
    files = [
        f
        for f in os.listdir(encoded_dir)
        if os.fsdecode(f).endswith(".csv")
    ]
    print(f"Parsing dataset at {path}: {len(files)} files found")
    print(f"Total files to parse: {len(files)}")

    if len(files) == 0:
        print("No CSV files found in dataset path.")
        return (None, None, None)

    file_count = 1
    print_file_length = len(max(files, key=len))

    control_population_column, experimental_population_column = build_population_columns(experimental_group)
    control_life_expectancy_column, experimental_life_expectancy_column = build_life_expectancy_columns(experimental_group)
    control_TTL_column, experimental_TTL_column = build_TTL_columns(experimental_group)
    control_wealth_column, experimental_wealth_column = build_wealth_columns(experimental_group)
    lending_columns, reproduction_columns, trade_columns = build_interaction_columns(experimental_group)

    control_stats = {"life_expectancy": [], "TTL": [], "wealth": [], "population_at_final_timestep": []}
    experimental_stats = {"life_expectancy": [], "TTL": [], "wealth": [], "population_at_final_timestep": []}
    total_stats = {"life_expectancy": [], "TTL": [], "wealth": [], "population_at_final_timestep": []}
    lending_rates = {"same_group": [], "different_group": []}
    reproduction_rates = {"same_group": [], "different_group": []}
    trade_rates = {"same_group": [], "different_group": []}
    all_runs, control_extinct_runs, experimental_extinct_runs, both_extinct_runs = 0, 0, 0, 0
    for file in files:
        filename = os.fsdecode(file)
        file_path = os.path.join(path, filename)

        with open(file_path) as log_file:
            try:
                df = pd.read_csv(log_file)
            except pd.errors.EmptyDataError:
                print(f"Warning: {filename} is empty.")
                continue

        print_progress(filename, file_count, len(files), print_file_length)
        file_count += 1

        single_run_stats = get_single_run_stats(
            df,
            control_population_column,
            experimental_population_column,
            control_life_expectancy_column,
            experimental_life_expectancy_column,
            control_TTL_column,
            experimental_TTL_column,
            control_wealth_column,
            experimental_wealth_column,
            lending_columns,
            reproduction_columns,
            trade_columns
        )
        control_stats["life_expectancy"].append(single_run_stats["life_expectancy"][0])
        experimental_stats["life_expectancy"].append(single_run_stats["life_expectancy"][1])
        total_stats["life_expectancy"].append(single_run_stats["life_expectancy"][2])

        control_stats["TTL"].append(single_run_stats["TTL"][0])
        experimental_stats["TTL"].append(single_run_stats["TTL"][1])
        total_stats["TTL"].append(single_run_stats["TTL"][2])
        
        control_stats["wealth"].append(single_run_stats["wealth"][0])
        experimental_stats["wealth"].append(single_run_stats["wealth"][1])
        total_stats["wealth"].append(single_run_stats["wealth"][2])

        control_stats["population_at_final_timestep"].append(single_run_stats["population_at_final_timestep"][0])
        experimental_stats["population_at_final_timestep"].append(single_run_stats["population_at_final_timestep"][1])
        total_stats["population_at_final_timestep"].append(single_run_stats["population_at_final_timestep"][2])

        lending_rates["same_group"].append(single_run_stats["lending_rates"][0])
        lending_rates["different_group"].append(single_run_stats["lending_rates"][1])

        reproduction_rates["same_group"].append(single_run_stats["reproduction_rates"][0])
        reproduction_rates["different_group"].append(single_run_stats["reproduction_rates"][1])

        trade_rates["same_group"].append(single_run_stats["trade_rates"][0])
        trade_rates["different_group"].append(single_run_stats["trade_rates"][1])

        all_runs += 1
        if single_run_stats["extinctions"][0]:
            control_extinct_runs += 1
        if single_run_stats["extinctions"][1]:
            experimental_extinct_runs += 1
        if single_run_stats["extinctions"][2]:
            both_extinct_runs += 1

    controlStandardDeviations = {"life_expectancy": None, "TTL": None, "wealth": None, "population_at_final_timestep": None}
    experimentalStandardDeviations = {"life_expectancy": None, "TTL": None, "wealth": None, "population_at_final_timestep": None}
    totalStandardDeviations = {"life_expectancy": None, "TTL": None, "wealth": None, "population_at_final_timestep": None}
    controlFinalMeans = {"life_expectancy": None, "TTL": None, "wealth": None, "population_at_final_timestep": None}
    experimentalFinalMeans = {"life_expectancy": None, "TTL": None, "wealth": None, "population_at_final_timestep": None}
    totalFinalMeans = {"life_expectancy": None, "TTL": None, "wealth": None, "population_at_final_timestep": None}
    interactionRateMeans = {"lending_rates": None, "reproduction_rates": None, "trade_rates": None}
    interactionRateStandardDeviations = {"lending_rate_std_devs": (None, None), "reproduction_rate_std_devs": (None, None), "trade_rate_std_devs": (None, None)}

    if len(control_stats["life_expectancy"]) < 2 or len(experimental_stats["life_expectancy"]) < 2 or len(total_stats["life_expectancy"]) < 2:
        print("Warning: Not enough valid files to compute standard deviation.")
        controlStandardDeviations["life_expectancy"] = None
        experimentalStandardDeviations["life_expectancy"] = None
        totalStandardDeviations["life_expectancy"] = None
    else:
        controlStandardDeviations["life_expectancy"] = statistics.stdev(control_stats["life_expectancy"])
        experimentalStandardDeviations["life_expectancy"] = statistics.stdev(experimental_stats["life_expectancy"])
        totalStandardDeviations["life_expectancy"] = statistics.stdev(total_stats["life_expectancy"])

    if len(control_stats["TTL"]) < 2 or len(experimental_stats["TTL"]) < 2 or len(total_stats["TTL"]) < 2:
        print("Warning: Not enough valid files to compute standard deviation.")
        controlStandardDeviations["TTL"] = None
        experimentalStandardDeviations["TTL"] = None
        totalStandardDeviations["TTL"] = None
    else:
        controlStandardDeviations["TTL"] = statistics.stdev(control_stats["TTL"])
        experimentalStandardDeviations["TTL"] = statistics.stdev(experimental_stats["TTL"])
        totalStandardDeviations["TTL"] = statistics.stdev(total_stats["TTL"])

    if len(control_stats["wealth"]) < 2 or len(experimental_stats["wealth"]) < 2 or len(total_stats["wealth"]) < 2:
        print("Warning: Not enough valid files to compute standard deviation.")
        controlStandardDeviations["wealth"] = None
        experimentalStandardDeviations["wealth"] = None
        totalStandardDeviations["wealth"] = None
    else:
        controlStandardDeviations["wealth"] = statistics.stdev(control_stats["wealth"])
        experimentalStandardDeviations["wealth"] = statistics.stdev(experimental_stats["wealth"])
        totalStandardDeviations["wealth"] = statistics.stdev(total_stats["wealth"])

    if len(control_stats["population_at_final_timestep"]) < 2 or len(experimental_stats["population_at_final_timestep"]) < 2 or len(total_stats["population_at_final_timestep"]) < 2:
        print("Warning: Not enough valid files to compute standard deviation.")
        controlStandardDeviations["population_at_final_timestep"] = None
        experimentalStandardDeviations["population_at_final_timestep"] = None
        totalStandardDeviations["population_at_final_timestep"] = None
    else:
        controlStandardDeviations["population_at_final_timestep"] = statistics.stdev(control_stats["population_at_final_timestep"])
        experimentalStandardDeviations["population_at_final_timestep"] = statistics.stdev(experimental_stats["population_at_final_timestep"])
        totalStandardDeviations["population_at_final_timestep"] = statistics.stdev(total_stats["population_at_final_timestep"])

    interactionRateStandardDeviations["lending_rate_std_devs"] = (statistics.stdev(lending_rates["same_group"]), statistics.stdev(lending_rates["different_group"])) if len(lending_rates["same_group"]) >= 2 and len(lending_rates["different_group"]) >= 2 else (None, None)
    interactionRateStandardDeviations["reproduction_rate_std_devs"] = (statistics.stdev(reproduction_rates["same_group"]), statistics.stdev(reproduction_rates["different_group"])) if len(reproduction_rates["same_group"]) >= 2 and len(reproduction_rates["different_group"]) >= 2 else (None, None)
    interactionRateStandardDeviations["trade_rate_std_devs"] = (statistics.stdev(trade_rates["same_group"]), statistics.stdev(trade_rates["different_group"])) if len(trade_rates["same_group"]) >= 2 and len(trade_rates["different_group"]) >= 2 else (None, None)

    controlFinalMeans["life_expectancy"] = statistics.mean(control_stats["life_expectancy"]) if len(control_stats["life_expectancy"]) > 0 else None
    experimentalFinalMeans["life_expectancy"] = statistics.mean(experimental_stats["life_expectancy"]) if len(experimental_stats["life_expectancy"]) > 0 else None
    totalFinalMeans["life_expectancy"] = statistics.mean(total_stats["life_expectancy"]) if len(total_stats["life_expectancy"]) > 0 else None

    controlFinalMeans["TTL"] = statistics.mean(control_stats["TTL"]) if len(control_stats["TTL"]) > 0 else None
    experimentalFinalMeans["TTL"] = statistics.mean(experimental_stats["TTL"]) if len(experimental_stats["TTL"]) > 0 else None
    totalFinalMeans["TTL"] = statistics.mean(total_stats["TTL"]) if len(total_stats["TTL"]) > 0 else None

    controlFinalMeans["wealth"] = statistics.mean(control_stats["wealth"]) if len(control_stats["wealth"]) > 0 else None
    experimentalFinalMeans["wealth"] = statistics.mean(experimental_stats["wealth"]) if len(experimental_stats["wealth"]) > 0 else None
    totalFinalMeans["wealth"] = statistics.mean(total_stats["wealth"]) if len(total_stats["wealth"]) > 0 else None

    controlFinalMeans["population_at_final_timestep"] = statistics.mean(control_stats["population_at_final_timestep"]) if len(control_stats["population_at_final_timestep"]) > 0 else None
    experimentalFinalMeans["population_at_final_timestep"] = statistics.mean(experimental_stats["population_at_final_timestep"]) if len(experimental_stats["population_at_final_timestep"]) > 0 else None
    totalFinalMeans["population_at_final_timestep"] = statistics.mean(total_stats["population_at_final_timestep"]) if len(total_stats["population_at_final_timestep"]) > 0 else None

    interactionRateMeans["lending_rates"] = (statistics.mean(lending_rates["same_group"]), statistics.mean(lending_rates["different_group"])) if len(lending_rates["same_group"]) > 0 and len(lending_rates["different_group"]) > 0 else (None, None)
    interactionRateMeans["reproduction_rates"] = (statistics.mean(reproduction_rates["same_group"]), statistics.mean(reproduction_rates["different_group"])) if len(reproduction_rates["same_group"]) > 0 and len(reproduction_rates["different_group"]) > 0 else (None, None)
    interactionRateMeans["trade_rates"] = (statistics.mean(trade_rates["same_group"]), statistics.mean(trade_rates["different_group"])) if len(trade_rates["same_group"]) > 0 and len(trade_rates["different_group"]) > 0 else (None, None)

    stdDevs = {
        "control": controlStandardDeviations,
        "experimental": experimentalStandardDeviations,
        "total": totalStandardDeviations,
        "interaction_rates": interactionRateStandardDeviations,
    }
    means = {
        "control": controlFinalMeans,
        "experimental": experimentalFinalMeans,
        "total": totalFinalMeans,
        "interaction_rates": interactionRateMeans
    }
    extinctions = {
        "control": control_extinct_runs,
        "experimental": experimental_extinct_runs,
        "both": both_extinct_runs,
        "all_runs": all_runs,
    }
    return (stdDevs, means, extinctions)


def append_to_csv(
    csv_path,
    stdDevs,
    means,
    extinctions,
    config_prefix,
):
    headers = [
        "configuration",
        "total_runs",
        "control_extinctions",
        "control_extinction_rate",
        "experimental_extinctions",
        "experimental_extinction_rate",
        "both_extinctions",
        "both_extinction_rate",
        "control_population_at_final_timestep_mean",
        "control_population_at_final_timestep_stdDev",
        "experimental_population_at_final_timestep_mean",
        "experimental_population_at_final_timestep_stdDev",
        "total_population_at_final_timestep_mean",
        "total_population_at_final_timestep_stdDev",
        "control_life_expectancy_mean",
        "control_life_expectancy_stdDev",
        "experimental_life_expectancy_mean",
        "experimental_life_expectancy_stdDev",
        "total_life_expectancy_mean",
        "total_life_expectancy_stdDev",
        "control_TTL_mean",
        "control_TTL_stdDev",
        "experimental_TTL_mean",
        "experimental_TTL_stdDev",
        "total_TTL_mean",
        "total_TTL_stdDev",
        "control_wealth_mean",
        "control_wealth_stdDev",
        "experimental_wealth_mean",
        "experimental_wealth_stdDev",
        "total_wealth_mean",
        "total_wealth_stdDev",
        "same_group_lending_rate_mean",
        "different_group_lending_rate_mean",
        "same_group_lending_rate_stdDev",
        "different_group_lending_rate_stdDev",
        "same_group_reproduction_rate_mean",
        "different_group_reproduction_rate_mean",
        "same_group_reproduction_rate_stdDev",
        "different_group_reproduction_rate_stdDev",
        "same_group_trade_rate_mean",
        "different_group_trade_rate_mean",
        "same_group_trade_rate_stdDev",
        "different_group_trade_rate_stdDev",
    ]

    file_exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as output_file:
        writer = csv.writer(output_file)
        if not file_exists:
            writer.writerow(headers)

        if means is None or stdDevs is None:
            print("No valid data found to compute metrics. Writing Nones to CSV.")
            row = [
                config_prefix, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None
            ]
        else:
            row = [
                config_prefix,
                extinctions['all_runs'],
                extinctions['control'] if extinctions['control'] is not None else 0,
                f"{(extinctions['control'] / extinctions['all_runs'] * 100):.2f}" if extinctions['all_runs'] > 0 and extinctions['control'] is not None else "0.00",
                extinctions['experimental'] if extinctions['experimental'] is not None else 0,
                f"{(extinctions['experimental'] / extinctions['all_runs'] * 100):.2f}" if extinctions['all_runs'] > 0 and extinctions['experimental'] is not None else "0.00",
                extinctions['both'] if extinctions['both'] is not None else 0,
                f"{(extinctions['both'] / extinctions['all_runs'] * 100):.2f}" if extinctions['all_runs'] > 0 and extinctions['both'] is not None else "0.00",
                f"{means['control']['population_at_final_timestep']:.4f}" if means['control']['population_at_final_timestep'] is not None else 0,
                f"{stdDevs['control']['population_at_final_timestep']:.4f}" if stdDevs['control']['population_at_final_timestep'] is not None else 0,
                f"{means['experimental']['population_at_final_timestep']:.4f}" if means['experimental']['population_at_final_timestep'] is not None else 0,
                f"{stdDevs['experimental']['population_at_final_timestep']:.4f}" if stdDevs['experimental']['population_at_final_timestep'] is not None else 0,
                f"{means['total']['population_at_final_timestep']:.4f}" if means['total']['population_at_final_timestep'] is not None else 0,
                f"{stdDevs['total']['population_at_final_timestep']:.4f}" if stdDevs['total']['population_at_final_timestep'] is not None else 0,
                f"{means['control']['life_expectancy']:.4f}" if means['control']['life_expectancy'] is not None else "N/A",
                f"{stdDevs['control']['life_expectancy']:.4f}" if stdDevs['control']['life_expectancy'] is not None else "N/A",
                f"{means['experimental']['life_expectancy']:.4f}" if means['experimental']['life_expectancy'] is not None else "N/A",
                f"{stdDevs['experimental']['life_expectancy']:.4f}" if stdDevs['experimental']['life_expectancy'] is not None else "N/A",
                f"{means['total']['life_expectancy']:.4f}" if means['total']['life_expectancy'] is not None else "N/A",
                f"{stdDevs['total']['life_expectancy']:.4f}" if stdDevs['total']['life_expectancy'] is not None else "N/A",
                f"{means['control']['TTL']:.4f}" if means['control']['TTL'] is not None else "N/A",
                f"{stdDevs['control']['TTL']:.4f}" if stdDevs['control']['TTL'] is not None else "N/A",
                f"{means['experimental']['TTL']:.4f}" if means['experimental']['TTL'] is not None else "N/A",
                f"{stdDevs['experimental']['TTL']:.4f}" if stdDevs['experimental']['TTL'] is not None else "N/A",
                f"{means['total']['TTL']:.4f}" if means['total']['TTL'] is not None else "N/A",
                f"{stdDevs['total']['TTL']:.4f}" if stdDevs['total']['TTL'] is not None else "N/A",
                f"{means['control']['wealth']:.4f}" if means['control']['wealth'] is not None else "N/A",
                f"{stdDevs['control']['wealth']:.4f}" if stdDevs['control']['wealth'] is not None else "N/A",
                f"{means['experimental']['wealth']:.4f}" if means['experimental']['wealth'] is not None else "N/A",
                f"{stdDevs['experimental']['wealth']:.4f}" if stdDevs['experimental']['wealth'] is not None else "N/A",
                f"{means['total']['wealth']:.4f}" if means['total']['wealth'] is not None else "N/A",
                f"{stdDevs['total']['wealth']:.4f}" if stdDevs['total']['wealth'] is not None else "N/A",
                f"{(means['interaction_rates']['lending_rates'][0] * 100):.4f}" if means['interaction_rates']['lending_rates'][0] is not None else "N/A",
                f"{(means['interaction_rates']['lending_rates'][1] * 100):.4f}" if means['interaction_rates']['lending_rates'][1] is not None else "N/A",
                f"{(stdDevs['interaction_rates']['lending_rate_std_devs'][0] * 100):.4f}" if stdDevs['interaction_rates']['lending_rate_std_devs'][0] is not None else "N/A",
                f"{(stdDevs['interaction_rates']['lending_rate_std_devs'][1] * 100):.4f}" if stdDevs['interaction_rates']['lending_rate_std_devs'][1] is not None else "N/A",
                f"{(means['interaction_rates']['reproduction_rates'][0] * 100):.4f}" if means['interaction_rates']['reproduction_rates'][0] is not None else "N/A",
                f"{(means['interaction_rates']['reproduction_rates'][1] * 100):.4f}" if means['interaction_rates']['reproduction_rates'][1] is not None else "N/A",
                f"{(stdDevs['interaction_rates']['reproduction_rate_std_devs'][0] * 100):.4f}" if stdDevs['interaction_rates']['reproduction_rate_std_devs'][0] is not None else "N/A",
                f"{(stdDevs['interaction_rates']['reproduction_rate_std_devs'][1] * 100):.4f}" if stdDevs['interaction_rates']['reproduction_rate_std_devs'][1] is not None else "N/A",
                f"{(means['interaction_rates']['trade_rates'][0] * 100):.4f}" if means['interaction_rates']['trade_rates'][0] is not None else "N/A",
                f"{(means['interaction_rates']['trade_rates'][1] * 100):.4f}" if means['interaction_rates']['trade_rates'][1] is not None else "N/A",
                f"{(stdDevs['interaction_rates']['trade_rate_std_devs'][0] * 100):.4f}" if stdDevs['interaction_rates']['trade_rate_std_devs'][0] is not None else "N/A",
                f"{(stdDevs['interaction_rates']['trade_rate_std_devs'][1] * 100):.4f}" if stdDevs['interaction_rates']['trade_rate_std_devs'][1] is not None else "N/A",
            ]
        writer.writerow(row)


def main():
    options = parse_options()
    path = options["path"]
    config_path = options["config"]
    output_csv = options["output"]

    if output_csv is None:
        print("Output CSV path required.")
        print_help()

    if os.path.exists(output_csv) and os.path.isdir(output_csv):
        print(f"Output target {output_csv} must be a file, not a directory.")
        print_help()

    if not os.path.exists(path):
        print(f"Path {path} not recognized.")
        print_help()

    if not os.path.exists(config_path):
        print(f"Config file {config_path} not recognized.")
        print_help()
    
    with open(config_path) as config_file:
        config = json.loads(config_file.read())

    experimental_group = config.get("sugarscapeOptions", {}).get("experimentalGroup")

    config_prefix = os.path.splitext(os.path.basename(config_path))[0]
    stdDevs, means, extinctions = scan_dataset(path, experimental_group)
    if means is None:
        print("No valid data found to compute metrics.")
        exit(1)

    append_to_csv(
        output_csv,
        stdDevs,
        means,
        extinctions,
        config_prefix,
    )
    print(f"Appended 1 row to {output_csv}")


if __name__ == "__main__":
    main()
