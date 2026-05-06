import csv
import getopt
import os
import sys

import matplotlib.pyplot
import matplotlib.ticker


LIFE_EXPECTANCY_COLUMNS = [
    ("experimental_life_expectancy_mean", "experimental_life_expectancy_stdDev", "experimental", "magenta"),
    ("control_life_expectancy_mean", "control_life_expectancy_stdDev", "control", "cyan"),
]
TOTAL_LIFE_EXPECTANCY_COLUMN = ("total_life_expectancy_mean", "total_life_expectancy_stdDev", "total", "cyan")

def parse_options():
    command_line_args = sys.argv[1:]
    short_options = "p:o:x:y:c:e:th"
    long_options = (
        "path=",
        "output=",
        "xlabel=",
        "ylabel=",
        "control-group=",
        "experimental-group=",
        "total-only",
        "help",
    )
    options = {
        "path": None,
        "output": None,
        "xlabel": None,
        "ylabel": None,
        "controlGroup": "Control",
        "experimentalGroup": "Experimental",
        "totalOnly": False,
    }

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
        elif curr_arg in ("-c", "--control-group"):
            options["controlGroup"] = curr_val
            if curr_val == "":
                print("No control group name provided.")
                print_help()
        elif curr_arg in ("-e", "--experimental-group"):
            options["experimentalGroup"] = curr_val
            if curr_val == "":
                print("No experimental group name provided.")
                print_help()
        elif curr_arg in ("-t", "--total-only"):
            options["totalOnly"] = True
        elif curr_arg in ("-h", "--help"):
            print_help()

    if options["path"] is None:
        print("CSV path required.")
        print_help()

    return options


def print_help():
    print(
        "Usage:\n\tpython plot-life-expectancy-bar-chart.py --path /path/to/life-expectancy.csv [--output /path/to/output.pdf]\n\n"
        "Options:\n"
        "\t-p,--path\tUse the specified CSV file produced by print-reporting.py's life-expectancy output.\n"
        "\t-o,--output\tOptional path for the generated plot. Defaults to a PDF next to the input CSV.\n"
        "\t-x,--xlabel\tOptional custom x-axis label. Defaults to the CSV's first column name.\n"
        "\t-y,--ylabel\tOptional custom y-axis label. Defaults to Mean Life Expectancy.\n"
        "\t-c,--control-group\tOptional custom legend label for the control group. Defaults to Control.\n"
        "\t-e,--experimental-group\tOptional custom legend label for the experimental group. Defaults to Experimental.\n"
        "\t-t,--total-only\tPlot only total_life_expectancy_averaged as a single line.\n"
        "\t-h,--help\tDisplay this message."
    )
    sys.exit(0)


def parse_numeric(value):
    if value in [None, "", "N/A", "None"]:
        return None
    if isinstance(value, str):
        value = value.strip().rstrip("%")
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def load_life_expectancy_csv(csv_path, total_only=False):
    with open(csv_path, newline="") as input_file:
        reader = csv.DictReader(input_file)
        if reader.fieldnames is None or len(reader.fieldnames) == 0:
            raise ValueError(f"CSV file {csv_path} does not contain headers.")

        configuration_column = reader.fieldnames[0]
        configurations = []
        columns = [TOTAL_LIFE_EXPECTANCY_COLUMN] if total_only is True else LIFE_EXPECTANCY_COLUMNS
        series = {label: [] for _, _, label, _ in columns}

        for row in reader:
            configuration = row.get(configuration_column, "")
            if "disabled" in configuration.lower():
                continue

            configurations.append(configuration)

            for column_name, std_dev_column_name, label, _ in columns:
                series[label].append((parse_numeric(row.get(column_name)), parse_numeric(row.get(std_dev_column_name))))

    return configuration_column, configurations, series


def default_output_path(input_path):
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    return os.path.join(os.path.dirname(input_path), f"{base_name}_averaged_life_expectancy.pdf")


def generate_bar_chart(
    csv_path,
    output_path=None,
    xlabel=None,
    ylabel=None,
    control_group_name="Control",
    experimental_group_name="Experimental",
    total_only=False,
):
    matplotlib.pyplot.rcParams["font.family"] = "serif"
    matplotlib.pyplot.rcParams["font.size"] = 18

    configuration_column, configurations, series = load_life_expectancy_csv(csv_path, total_only)
    if len(configurations) == 0:
        raise ValueError(f"CSV file {csv_path} does not contain any data rows.")

    if output_path is None:
        output_path = default_output_path(csv_path)

    figure, axes = matplotlib.pyplot.subplots(figsize=(max(10, len(configurations) * 0.75), 7))
    x = [i for i in range(len(configurations))]
    x_label = configuration_column.title() if xlabel is None else xlabel
    y_label = "Mean Life Expectancy" if ylabel is None else ylabel
    columns = [TOTAL_LIFE_EXPECTANCY_COLUMN] if total_only is True else LIFE_EXPECTANCY_COLUMNS
    legend_labels = {
        "experimental": experimental_group_name,
        "control": control_group_name,
        "total": "Total",
    }

    means = []
    stdDevs = []
    for _, _, label, _ in columns:
        means.extend([value[0] for value in series[label] if value[0] is not None])
        stdDevs.extend([value[1] for value in series[label] if value[1] is not None])
    if len(means) == 0:
        raise ValueError(f"CSV file {csv_path} does not contain any averaged life expectancy values.")

    y_min = min(means) - max(stdDevs) if len(stdDevs) > 0 else min(means)
    y_max = max(means) + max(stdDevs) if len(stdDevs) > 0 else max(means)
    y_padding = max(1, max(stdDevs) * 0.1)
    y_lower = max(0, y_min - y_padding)
    y_upper = y_max + y_padding

    axes.set(
        xlabel=x_label,
        ylabel=y_label,
        ylim=[y_lower, y_upper],
        xticks=x,
    )
    axes.set_xticklabels(configurations, rotation=45, ha="right")
    axes.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=False, nbins=10))

    group_width = 0.8
    bar_width = group_width / max(1, len(columns))
    start_offset = -group_width / 2 + bar_width / 2

    for index, (column_name, std_dev_column_name, label, color) in enumerate(columns):
        y = series[label]
        if any(value is not None for value in y):
            bar_positions = [curr_x + start_offset + index * bar_width for curr_x in x]
            axes.bar(bar_positions, [val[0] for val in y], yerr=[val[1] for val in y], width=bar_width, color=color, label=legend_labels[label])
            axes.errorbar(bar_positions, [val[0] for val in y], yerr=[val[1] for val in y], fmt="none", ecolor="black", capsize=10, elinewidth=2)

    # Ensure the outer bars and their caps fit within the axis bounds.
    x_padding = max(0.5, group_width / 2 + 0.1)
    axes.set_xlim(-x_padding, max(x_padding, len(configurations) - 1 + x_padding))

    if not total_only:
        axes.legend(loc="best", labelspacing=0.1, frameon=True, fontsize=16, facecolor='white', framealpha=1.0)

    figure.tight_layout()
    figure.savefig(output_path, format="pdf", bbox_inches="tight")
    matplotlib.pyplot.close(figure)


def main():
    options = parse_options()
    csv_path = options["path"]
    output_path = options["output"]
    xlabel = options["xlabel"]
    ylabel = options["ylabel"]
    control_group_name = options["controlGroup"]
    experimental_group_name = options["experimentalGroup"]
    total_only = options["totalOnly"]

    if not os.path.exists(csv_path):
        print(f"CSV path {csv_path} not recognized.")
        print_help()

    if not csv_path.lower().endswith(".csv"):
        print(f"CSV path {csv_path} must point to a .csv file.")
        print_help()

    print(f"Generating averaged life expectancy plot from {csv_path}")
    generate_bar_chart(
        csv_path,
        output_path,
        xlabel,
        ylabel,
        control_group_name,
        experimental_group_name,
        total_only,
    )
    print(f"Saved plot to {output_path if output_path is not None else default_output_path(csv_path)}")


if __name__ == "__main__":
    main()