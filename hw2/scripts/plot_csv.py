#!/usr/bin/env python3
"""
plot_tb_csv.py

Plot one or more TensorBoard-exported CSVs with columns:
    Wall time, Step, Value

Examples:
    python plot_tb_csv.py run1.csv run2.csv run3.csv
    python plot_tb_csv.py run1.csv run2.csv --labels baseline improved
    python plot_tb_csv.py *.csv --smooth 0.9 --title "Training Curves"
"""

import argparse
from pathlib import Path
import sys

import pandas as pd
import matplotlib.pyplot as plt


def ema_smooth(values, alpha: float):
    if alpha <= 0:
        return values
    out = []
    prev = None
    for v in values:
        v = float(v)
        prev = v if prev is None else alpha * prev + (1 - alpha) * v
        out.append(prev)
    return out


def load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]

    required = {"Step", "Value"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns {sorted(missing)}; found {list(df.columns)}")

    # Numeric conversion
    df["Step"] = pd.to_numeric(df["Step"], errors="coerce")
    df["Value"] = pd.to_numeric(df["Value"], errors="coerce")
    if "Wall time" in df.columns:
        df["Wall time"] = pd.to_numeric(df["Wall time"], errors="coerce")

    df = df.dropna(subset=["Step", "Value"]).sort_values("Step")
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csvs", nargs="+", help="CSV files to plot (one or many)")
    parser.add_argument(
        "--labels",
        nargs="*",
        default=None,
        help="Optional labels (same count as CSVs). Example: --labels seed1 seed2 seed3",
    )
    parser.add_argument("--smooth", type=float, default=0.0, help="EMA alpha in [0,1), e.g. 0.9")
    parser.add_argument("--title", type=str, default="TensorBoard CSV Plot")
    parser.add_argument("--xlabel", type=str, default="Step")
    parser.add_argument("--ylabel", type=str, default="Value")
    parser.add_argument("--use-wall-time", action="store_true", help="Use 'Wall time' as x-axis if present")
    parser.add_argument("--save", type=str, default=None, help="Save figure to file instead of showing")
    args = parser.parse_args()

    if not (0.0 <= args.smooth < 1.0):
        print("Error: --smooth must be in [0,1).", file=sys.stderr)
        sys.exit(1)

    csv_paths = [Path(p).expanduser().resolve() for p in args.csvs]

    if args.labels is not None and len(args.labels) != len(csv_paths):
        print(
            f"Error: got {len(args.labels)} labels for {len(csv_paths)} CSVs.",
            file=sys.stderr,
        )
        sys.exit(1)

    plt.figure(figsize=(9, 5))

    plotted = 0
    for i, path in enumerate(csv_paths):
        if not path.exists():
            print(f"Warning: not found: {path}", file=sys.stderr)
            continue

        try:
            df = load_csv(path)
        except Exception as e:
            print(f"Warning: failed to load {path}: {e}", file=sys.stderr)
            continue

        x_col = "Wall time" if args.use_wall_time and "Wall time" in df.columns else "Step"
        x = df[x_col].to_numpy()
        y = df["Value"].to_numpy()
        y_plot = ema_smooth(y, args.smooth) if args.smooth > 0 else y

        label = args.labels[i] if args.labels is not None else path.stem
        plt.plot(x, y_plot, label=label)
        plotted += 1

    if plotted == 0:
        print("Error: no valid CSVs were plotted.", file=sys.stderr)
        sys.exit(1)

    plt.title(args.title)
    plt.xlabel("Wall time" if args.use_wall_time else args.xlabel)
    plt.ylabel(args.ylabel)
    plt.grid(True, alpha=0.3)
    if plotted > 1:
        plt.legend()
    plt.tight_layout()

    if args.save:
        out = Path(args.save).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out, dpi=150)
        print(f"Saved plot to: {out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()