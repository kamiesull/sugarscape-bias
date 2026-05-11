import getopt
import json
import os
import re
import statistics
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot
import matplotlib.ticker

# TODO: can choose to keep np/pd usage, but consider doing everything with built-in python

# TODO: Integrate with plot.py

def get_terminal_width(default=80):
    try:
        return os.get_terminal_size().columns
    except OSError:
        return default

def parse_options():
    command_line_args = sys.argv[1:]
    short_options = "c:p:l:h"
    long_options = (
        "conf=",
        "path=",
        "column=",
        "help",
    )
    options = {
        "config": None,
        "path": None,
        "column": None,
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
        elif curr_arg in ("-l", "--column"):
            options["column"] = curr_val
            if curr_val == "":
                print("No column specified.")
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
        "\tpython plot_by_config_generalized.py --path /path/to/data --conf /path/to/config [options]\n\n"
        "Options:\n"
        "\t-c,--conf\tUse the specified path to configurable settings file.\n"
        "\t-p,--path\tUse the specified path to find dataset CSV files.\n"
        "\t-l,--column\tSpecify the column to analyze.\n"
        "\t-h,--help\tDisplay this message."
    )
    sys.exit(0)


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

def build_population_columns(experimental_group):
    if experimental_group is None:
        return None, None
    else:
        control_population_column = "controlPopulation"
        experimental_population_column = experimental_group + "Population"
    return control_population_column, experimental_population_column

def get_single_run_stats(
    df,
    experimental_group_name
):  
    col_stats = {}
        
    # If the column is in this list, we will do a weighted average by population (specific to control/experimental if doing these columns).
    # Otherwise, all timesteps are weighted the same regardless of population size
    weighted_average_columns = ["meanAgeAtDeath", "agentMeanTimeToLive", "meanWealth",
                                "controlMeanAgeAtDeath", "controlAgentMeanTimeToLive", "controlMeanWealth"]
    if experimental_group_name is not None:
        weighted_average_columns.extend([experimental_group_name + "MeanAgeAtDeath", experimental_group_name + "AgentMeanTimeToLive", experimental_group_name + "MeanWealth"])

    control_population_column, experimental_population_column = build_population_columns(experimental_group_name)

    for column in df:
        col_stats[column] = {}
        if column in weighted_average_columns:
            # Weight the average by population, according to group
            if experimental_group_name and "control" in column.lower() and column != control_population_column:
                weighted_avg = np.average(df[column], weights=df[control_population_column]) if control_population_column is not None else None
            elif experimental_group_name and experimental_group_name in column.lower() and column != experimental_population_column:
                weighted_avg = np.average(df[column], weights=df[experimental_population_column]) if experimental_population_column is not None else None
            else:
                weighted_avg = np.average(df[column], weights=df['population']) if 'population' in df.columns else None
        else:
            weighted_avg = np.average(df[column])
        col_stats[column]["average_across_timesteps"] = weighted_avg

        # final_timestep_value = df[column].iloc[-1] if len(df[column]) > 0 else None
        # col_stats[column]["final_timestep"] = final_timestep_value
    
    return col_stats

def scan_dataset(models, path, experimental_group=None):
    encoded_dir = os.fsencode(path)
    files = [
        f
        for f in os.listdir(encoded_dir)
        if os.fsdecode(f).endswith(".csv")
    ]
    print(f"Parsing dataset at {path}: {len(files)} files found")
    print(f"Total files to parse: {len(files)}")

    if len(files) == 0:
        print("No CSV files found in dataset path.")
        return None

    file_count = 1
    print_file_length = len(max(files, key=len))

    stats = {}
    for file in files:
        filename = os.fsdecode(file)
        file_path = os.path.join(path, filename)

        # Assumes a config prefix exists at start of filename, partitioned by _ from modelSeed.json|.csv
        # This means that the configPrefix cannot contain any underscores
        if "_" in filename:
            configPrefix, _, rest_of_filename = filename.partition("_")
        else:
            configPrefix = "no-config-prefix"
            rest_of_filename = filename
        fileDecisionModel = re.compile(r"^([A-z]*)(\d*)\.(json|csv)")
        fileSearch = re.search(fileDecisionModel, rest_of_filename)
        if fileSearch == None:
            continue
        model = fileSearch.group(1)
        if model not in models:
            continue

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
            experimental_group
        )

        stats.setdefault(model, {})
        stats[model].setdefault(configPrefix, {})
        for column, col_stats in single_run_stats.items():
            stats[model][configPrefix].setdefault(
                column,
                {"averaged_across_timesteps": [],
                #  "final_timestep_values_per_run": []
                }
            )
            val = col_stats["average_across_timesteps"]
            if val is not None and not pd.isna(val):
                stats[model][configPrefix][column]["averaged_across_timesteps"].append(val)
            # stats[model][configPrefix][column]["final_timestep_values_per_run"].append(col_stats["final_timestep"])

    aggregates_by_model = {}
    for model, configs in stats.items():
        aggregates_by_model[model] = {
            "aggregates_averaged_across_timesteps_per_config": {},
            "standardDeviations_averaged_across_timesteps_per_config": {},
            # "aggregates_final_timestep_values_per_config": {},
            # "standardDeviations_final_timestep_values_per_config": {}
        }
        for config, cols in configs.items():
            for column, colStats in cols.items():
                aggregates_by_model[model]["aggregates_averaged_across_timesteps_per_config"].setdefault(column, {})
                aggregates_by_model[model]["standardDeviations_averaged_across_timesteps_per_config"].setdefault(column, {})
                # aggregates_by_model[model]["aggregates_final_timestep_values_per_config"].setdefault(column, {})
                # aggregates_by_model[model]["standardDeviations_final_timestep_values_per_config"].setdefault(column, {})

                aggregates_by_model[model]["aggregates_averaged_across_timesteps_per_config"][column][config] = (
                    np.average(colStats["averaged_across_timesteps"]) if colStats["averaged_across_timesteps"] else None
                )
                aggregates_by_model[model]["standardDeviations_averaged_across_timesteps_per_config"][column][config] = (
                    statistics.stdev(colStats["averaged_across_timesteps"]) if len(colStats["averaged_across_timesteps"]) > 1 else 0
                )
                # aggregates_by_model[model]["aggregates_final_timestep_values_per_config"][column][config] = (
                #     np.average(colStats["final_timestep_values_per_run"]) if colStats["final_timestep_values_per_run"] else None
                # )
                # aggregates_by_model[model]["standardDeviations_final_timestep_values_per_config"][column][config] = (
                #     statistics.stdev(colStats["final_timestep_values_per_run"]) if len(colStats["final_timestep_values_per_run"]) > 1 else 0
                # )
    # Final structure: aggregates_by_model[model][aggregate_type][column][config] = value
    return aggregates_by_model

def generateSimpleBarChart(aggregates_by_model, configsInOrder, outfile, column, label, positioning, percentage=False, experimentalGroup=None, plotGroups=False):
    matplotlib.pyplot.rcParams["font.family"] = "serif"
    matplotlib.pyplot.rcParams["font.size"] = 18
    x = list(range(len(configsInOrder)))
    modelStrings = {"asimov": "Asimov's Robot", "bentham": "Utilitarian", "egoist": "Egoist", "altruist": "Altruist", "none": "Raw Sugarscape", "rawSugarscape": "Raw Sugarscape",
                    "temperance": "Simple Temperance", "temperancePECS": "Complex Temperance", "multiple": "Multiple", "unknown": "Unknown"}
    modelColors = {"asimov": "blue", "bentham": "magenta", "egoist": "cyan", "altruist": "gold", "none": "black", "rawSugarscape": "black ", "temperance": "blue", "temperancePECS": "purple", "multiple": "red", "unknown": "green"}
    groupColors = {"experimental": "magenta", "control": "blue"}

    # Calculate bar width and positions based on number of bars being plotted to ensure bars fit within axis bounds
    group_width = 0.8
    number_of_bars = 2 if experimentalGroup != None and plotGroups == True else len(aggregates_by_model)
    bar_width = group_width / max(1, number_of_bars)
    start_offset = -group_width / 2 + bar_width / 2
    
    # If plotting separate groups, split up models into different charts to avoid overcrowding
    if experimentalGroup != None and plotGroups == True:
        for model in aggregates_by_model:
            figure, axes = matplotlib.pyplot.subplots()
            axes.set(xlabel = "Configuration", ylabel = label, xlim = [-0.5, len(configsInOrder) + 0.5])
            axes.set_xticks(x)
            axes.set_xticklabels(configsInOrder, rotation=45, ha="right")
            modelString = model
            if '_' in model:
                modelString = "multiple"
            elif model not in modelStrings:
                modelString = "unknown"
            controlGroupColumn = "control" + column[0].upper() + column[1:]
            controlGroupLabel = f"Control {modelStrings[modelString]}"
            experimentalGroupColumn = experimentalGroup + column[0].upper() + column[1:]
            experimentalGroupLabel = experimentalGroup[0].upper() + experimentalGroup[1:] + f" {modelStrings[modelString]}"
            # Prevent key error if all seeds went extinct for model
            if controlGroupColumn in aggregates_by_model[model]["aggregates_averaged_across_timesteps_per_config"]:
                bar_positions_control = [curr_x + start_offset for curr_x in x]
                y = [aggregates_by_model[model]["aggregates_averaged_across_timesteps_per_config"][controlGroupColumn][config] for config in configsInOrder]
                y_err = [aggregates_by_model[model]["standardDeviations_averaged_across_timesteps_per_config"][controlGroupColumn][config] for config in configsInOrder]
                axes.bar(bar_positions_control, y, color=groupColors["control"], label=controlGroupLabel, width=bar_width)
                axes.errorbar(bar_positions_control, y, yerr=y_err, fmt="none", ecolor="black", capsize=10, elinewidth=2)
            if experimentalGroupColumn in aggregates_by_model[model]["aggregates_averaged_across_timesteps_per_config"]:
                bar_positions_experimental = [curr_x + start_offset + bar_width for curr_x in x]
                y = [aggregates_by_model[model]["aggregates_averaged_across_timesteps_per_config"][experimentalGroupColumn][config] for config in configsInOrder]
                y_err = [aggregates_by_model[model]["standardDeviations_averaged_across_timesteps_per_config"][experimentalGroupColumn][config] for config in configsInOrder]
                axes.bar(bar_positions_experimental, y, color=groupColors["experimental"], label=experimentalGroupLabel, width=bar_width)
                axes.errorbar(bar_positions_experimental, y, yerr=y_err, fmt="none", ecolor="black", capsize=10, elinewidth=2)
            axes.legend(loc=positioning, labelspacing=0.1, frameon=True, fontsize=16, facecolor='white', framealpha=1.0)
            if percentage == True:
                axes.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter())
            modelOutfile = outfile.replace(".pdf", f"_{model}_model.pdf")
            figure.savefig(modelOutfile, format="pdf", bbox_inches="tight")
            matplotlib.pyplot.close(figure)
    else:
        figure, axes = matplotlib.pyplot.subplots()
        axes.set(xlabel = "Configuration", ylabel = label, xlim = [-0.5, len(configsInOrder) + 0.5])
        axes.set_xticks(x)
        axes.set_xticklabels(configsInOrder, rotation=45, ha="right")
        for index, model in enumerate(aggregates_by_model):
            modelString = model
            if '_' in model:
                modelString = "multiple"
            elif model not in modelStrings:
                modelString = "unknown"
            # Prevent key error if all seeds went extinct for model
            if column in aggregates_by_model[model]["aggregates_averaged_across_timesteps_per_config"]:
                bar_positions = [curr_x + start_offset + index * bar_width for curr_x in x]
                y = [aggregates_by_model[model]["aggregates_averaged_across_timesteps_per_config"][column][config] for config in configsInOrder]
                y_err = [aggregates_by_model[model]["standardDeviations_averaged_across_timesteps_per_config"][column][config] for config in configsInOrder]
                axes.bar(bar_positions, y, color=modelColors[modelString], label=modelStrings[modelString], width=bar_width)
                axes.errorbar(bar_positions, y, yerr=y_err, fmt="none", ecolor="black", capsize=10, elinewidth=2)
        axes.legend(loc=positioning, labelspacing=0.1, frameon=False, fontsize=16, facecolor='white', framealpha=1.0)
        if percentage == True:
            axes.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter())
        figure.savefig(outfile, format="pdf", bbox_inches="tight")
        matplotlib.pyplot.close(figure)

def main():
    options = parse_options()
    path = options["path"]
    config_path = options["config"]
    column = options["column"]

    if not os.path.exists(path):
        print(f"Path {path} not recognized.")
        print_help()

    if not os.path.exists(config_path):
        print(f"Config file {config_path} not recognized.")
        print_help()
    
    with open(config_path) as config_file:
        config = json.loads(config_file.read())

    experimental_group = config.get("sugarscapeOptions", {}).get("experimentalGroup")
    
    # Expects a list in the config file with the config prefixes in order of how they will be plotted
    # These MUST MATCH EXACTLY the config prefixes of the data files, which are assumed to be the part of the filename before the first underscore
    # Any configs that don't match exactly will cause an error
    configs_in_order = config.get("dataCollectionOptions", {}).get("plotConfigsInOrder", [])

    models = config.get("dataCollectionOptions", {}).get("decisionModels", [])
    models = [model for sublist in models for model in sublist] # flatten list

    aggregates_by_model = scan_dataset(models, path, experimental_group)
    if aggregates_by_model is None:
        print("No valid data found to compute metrics.")
        exit(1)

    generateSimpleBarChart(aggregates_by_model, configs_in_order, "test_config_plotting.pdf", column, column, "best", percentage=False, experimentalGroup=experimental_group, plotGroups=True)

if __name__ == "__main__":
    main()
