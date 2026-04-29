#!/usr/bin/env python3

import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt


def parse_eval_average_return(log_text: str):
    iterations = []
    eval_returns = []
    current_iteration = 0

    for line in log_text.splitlines():
        line = line.strip()

        iter_match = re.search(r"Iteration\s+(\d+)", line)
        if iter_match:
            current_iteration = int(iter_match.group(1))
            continue

        eval_match = re.search(
            r"Eval_AverageReturn\s*:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)",
            line,
        )
        if eval_match:
            iterations.append(current_iteration)
            eval_returns.append(float(eval_match.group(1)))

    return iterations, eval_returns


def parse_named_log_arg(arg: str):
    """
    Parse an argument of the form:
        label=path/to/log.txt
    If '=' is not present, use the file stem as the label.
    """
    if "=" in arg:
        label, path_str = arg.split("=", 1)
        return label, Path(path_str)
    else:
        path = Path(arg)
        return path.stem, path


def main():
    if len(sys.argv) < 3:
        print("Usage:")
        print("  python3 log_to_plot.py output.jpg label1=log1.txt label2=log2.txt ...")
        print("  or")
        print("  python3 log_to_plot.py output.jpg log1.txt log2.txt ...")
        sys.exit(1)

    output_file = Path(sys.argv[1])
    named_logs = [parse_named_log_arg(arg) for arg in sys.argv[2:]]

    plt.figure(figsize=(9, 6))
    plotted_any = False

    for label, log_file in named_logs:
        if not log_file.exists():
            print(f"Warning: file not found, skipping: {log_file}")
            continue

        log_text = log_file.read_text(encoding="utf-8", errors="ignore")
        iterations, eval_returns = parse_eval_average_return(log_text)

        if not iterations:
            print(f"Warning: no Eval_AverageReturn entries found in: {log_file}")
            continue

        plt.plot(iterations, eval_returns, marker="o", label=label)
        plotted_any = True

    if not plotted_any:
        print("Error: no valid log data found to plot.")
        sys.exit(1)

    plt.xlabel("Iteration")
    plt.ylabel("Eval Average Return")
    plt.title("CEM vs. Random Shooting for Evaluation Returns: Cheetah Env.")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_file, format="jpg", dpi=300)
    print(f"Saved plot to: {output_file}")


if __name__ == "__main__":
    main()