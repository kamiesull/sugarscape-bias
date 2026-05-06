import csv
import getopt
import os
import sys

import matplotlib.pyplot
import matplotlib.ticker


VALID_INTERACTION_COLUMNS = [
    ("same_group_lending_rate_mean", "same_group_lending_rate_stdDev", "Lending", "magenta"),
    ("same_group_reproduction_rate_mean", "same_group_reproduction_rate_stdDev", "Reproduction", "gold"),
    ("same_group_trade_rate_mean", "same_group_trade_rate_stdDev", "Trade", "cyan"),
]


def parse_options():
    command_line_args = sys.argv[1:]
    short_options = "p:o:x:y:i:h"
    long_options = ("path=", "output=", "xlabel=", "ylabel=", "interactions=", "help")
    options = {"path": None, "output": None, "xlabel": None, "ylabel": None, "interactions": None}

    try:
        args, _ = getopt.getopt(command_line_args, short_options, long_options)
    except getopt.GetoptError as err:
        print(err)
        print_help()

    for curr_arg, curr_val in args:
        if curr_arg in ("-p", "--path"):
            options["path"] = curr_val
            if curr_val == "":
                print("No CSV path provided.")
                print_help()
        elif curr_arg in ("-o", "--output"):
            options["output"] = curr_val
            if curr_val == "":
                print("No output path provided.")
                print_help()
        elif curr_arg in ("-x", "--xlabel"):
            options["xlabel"] = curr_val
            if curr_val == "":
                print("No x-axis label provided.")
                print_help()
        elif curr_arg in ("-y", "--ylabel"):
            options["ylabel"] = curr_val
            if curr_val == "":
                print("No y-axis label provided.")
                print_help()
        elif curr_arg in ("-i", "--interactions"):
            options["interactions"] = [val.strip().lower() for val in curr_val.split(",")]
            if "" in options["interactions"]:
                print("Invalid interaction type provided.")
                print_help()
        elif curr_arg in ("-h", "--help"):
            print_help()

    if options["path"] is None:
        print("CSV path required.")
        print_help()

    return options


def print_help():
    print(
        "Usage:\n\tpython plot-interaction-proportions-bar-chart.py --path /path/to/interactions.csv [--output /path/to/output.pdf]\n\n"
        "Options:\n"
        "\t-p,--path\tUse the specified CSV file produced by print-reporting.py's interactions output.\n"
        "\t-o,--output\tOptional path for the generated plot. Defaults to a PDF next to the input CSV.\n"
        "\t-x,--xlabel\tOptional custom x-axis label. Defaults to the CSV's first column name.\n"
        "\t-y,--ylabel\tOptional custom y-axis label. Defaults to Same-Group Interactions.\n"
        "\t-i,--interactions\tOptional interaction types to plot: lending, reproduction, or trade. Defaults to all.\n"
        "\t-h,--help\tDisplay this message."
    )
    sys.exit(0)


def parse_percentage(value):
    if value in [None, "", "N/A", "None"]:
        return None
    if isinstance(value, str):
        value = value.strip().rstrip("%")
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def load_interaction_csv(csv_path):
    with open(csv_path, newline="") as input_file:
        reader = csv.DictReader(input_file)
        if reader.fieldnames is None or len(reader.fieldnames) == 0:
            raise ValueError(f"CSV file {csv_path} does not contain headers.")

        configuration_column = reader.fieldnames[0]
        configurations = []
        columns = VALID_INTERACTION_COLUMNS
        series = {label: [] for _, _, label, _ in columns}

        for row in reader:
            configuration = row.get(configuration_column, "")
            if "disabled" in configuration.lower():
                continue
            configurations.append(configuration)

            for column_name, std_dev_column_name, label, _ in columns:
                series[label].append((parse_percentage(row.get(column_name)), parse_percentage(row.get(std_dev_column_name))))

    return configuration_column, configurations, series


def default_output_path(input_path):
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    return os.path.join(os.path.dirname(input_path), f"{base_name}_interaction_proportions.pdf")


def generate_bar_chart(csv_path, output_path=None, xlabel=None, ylabel=None, interaction_type=None):
    matplotlib.pyplot.rcParams["font.family"] = "serif"
    matplotlib.pyplot.rcParams["font.size"] = 18

    configuration_column, configurations, series = load_interaction_csv(csv_path)
    if len(configurations) == 0:
        raise ValueError(f"CSV file {csv_path} does not contain any data rows.")

    if output_path is None:
        output_path = default_output_path(csv_path)

    # Filter columns by interaction type if specified
    columns = VALID_INTERACTION_COLUMNS
    if interaction_type is not None:
        columns = [col for col in VALID_INTERACTION_COLUMNS if col[2].lower() in interaction_type]
        if len(columns) == 0:
            raise ValueError(f"Interaction types '{interaction_type}' not found. Must be one of: lending, reproduction, trade")

    figure, axes = matplotlib.pyplot.subplots(figsize=(max(10, len(configurations) * 0.75), 7))
    x = [i for i in range(len(configurations))]
    x_label = configuration_column.title() if xlabel is None else xlabel
    # Set default ylabel: if single interaction type, use "Same-Group {Type}", otherwise "Same-Group Interactions"
    if ylabel is None:
        if len(columns) == 1:
            y_label = f"Same-Group {columns[0][2]}"
        else:
            y_label = "Same-Group Interactions"
    else:
        y_label = ylabel

    axes.set(
        xlabel=x_label,
        ylabel=y_label,
        ylim=[40, 100],
        xticks=x,
    )
    axes.set_xticklabels(configurations, rotation=45, ha="right")
    axes.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(5))
    axes.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=100))

    group_width = 0.8
    bar_width = group_width / max(1, len(columns))
    start_offset = -group_width / 2 + bar_width / 2

    for index, (column_name, std_dev_column_name, label, color) in enumerate(columns):
        y = series[label]
        if any(value is not None for value in y):
            bar_positions = [curr_x + start_offset + index * bar_width for curr_x in x]
            axes.bar(bar_positions, [val[0] for val in y], yerr=[val[1] for val in y], width=bar_width, color=color, label=label)
            axes.errorbar(bar_positions, [val[0] for val in y], yerr=[val[1] for val in y], fmt="none", ecolor="black", capsize=10, elinewidth=2)

    # Ensure the outer bars and their caps fit within the axis bounds.
    x_padding = max(0.5, group_width / 2 + 0.1)
    axes.set_xlim(-x_padding, max(x_padding, len(configurations) - 1 + x_padding))

    # Only show legend when multiple interaction types are plotted
    if len(columns) > 1:
        axes.legend(loc="lower right", labelspacing=0.1, frameon=True, fontsize=16, facecolor='white', framealpha=1.0)

    figure.tight_layout()
    figure.savefig(output_path, format="pdf", bbox_inches="tight")
    matplotlib.pyplot.close(figure)


def main():
    options = parse_options()
    csv_path = options["path"]
    output_path = options["output"]
    xlabel = options["xlabel"]
    ylabel = options["ylabel"]
    interaction_types = options["interactions"]

    if not os.path.exists(csv_path):
        print(f"CSV path {csv_path} not recognized.")
        print_help()

    if not csv_path.lower().endswith(".csv"):
        print(f"CSV path {csv_path} must point to a .csv file.")
        print_help()

    if interaction_types is not None:
        valid_types = ["lending", "reproduction", "trade"]
        if not all(interaction in valid_types for interaction in interaction_types):
            print(f"Invalid interaction types '{interaction_types}'. Must be one of: {', '.join(valid_types)}")
            print_help()

    print(f"Generating interaction proportion plot from {csv_path}")
    generate_bar_chart(csv_path, output_path, xlabel, ylabel, interaction_types)
    print(f"Saved plot to {output_path if output_path is not None else default_output_path(csv_path)}")


if __name__ == "__main__":
    main()