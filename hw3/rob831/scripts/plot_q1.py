#!/usr/bin/env python3

import argparse
import os
import sys
from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


REQUIRED_COLUMNS = [
    "Iteration",
    "Train_EnvstepsSoFar",
    "Train_AverageReturn",
]


def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"CSV '{path}' is missing required columns: {missing}"
        )

    df = df[REQUIRED_COLUMNS].copy()
    df = df.sort_values("Train_EnvstepsSoFar").reset_index(drop=True)
    return df


def validate_timestep_alignment(dfs: List[pd.DataFrame], label: str) -> np.ndarray:
    """
    Ensure all CSVs in a category share the same timestep vector.
    Returns the common timestep array.
    """
    if not dfs:
        raise ValueError(f"No CSVs provided for {label}.")

    base_steps = dfs[0]["Train_EnvstepsSoFar"].to_numpy()

    for i, df in enumerate(dfs[1:], start=1):
        steps = df["Train_EnvstepsSoFar"].to_numpy()

        if len(steps) != len(base_steps):
            raise ValueError(
                f"{label} CSV index {i} does not have the same number of timesteps "
                f"as the first {label} CSV."
            )

        if not np.array_equal(steps, base_steps):
            raise ValueError(
                f"{label} CSV index {i} has different timestep values.\n"
                f"All CSVs within a category must have identical Train_EnvstepsSoFar."
            )

    return base_steps


def aggregate_runs(dfs: List[pd.DataFrame], label: str):
    """
    Returns:
        timesteps: shape (T,)
        mean_returns: shape (T,)
        std_returns: shape (T,)
    """
    timesteps = validate_timestep_alignment(dfs, label)

    returns_matrix = np.stack(
        [df["Train_AverageReturn"].to_numpy() for df in dfs],
        axis=0
    )  # shape: (num_runs, T)

    mean_returns = returns_matrix.mean(axis=0)
    std_returns = returns_matrix.std(axis=0, ddof=0)

    return timesteps, mean_returns, std_returns


def plot_results(
    dqn_steps,
    dqn_mean,
    dqn_std,
    ddqn_steps,
    ddqn_mean,
    ddqn_std,
    num_dqn,
    num_ddqn,
    title_env,
    errorbar_style,
    output_path=None,
    show_plot=False,
):
    plt.figure(figsize=(10.2, 6.0))

    # Mean lines
    plt.plot(dqn_steps, dqn_mean, label="DQN", linewidth=2)
    plt.plot(ddqn_steps, ddqn_mean, label="Double DQN", linewidth=2)

    # Uncertainty visualization
    if errorbar_style == "band":
        plt.fill_between(
            dqn_steps,
            dqn_mean - dqn_std,
            dqn_mean + dqn_std,
            alpha=0.2
        )
        plt.fill_between(
            ddqn_steps,
            ddqn_mean - ddqn_std,
            ddqn_mean + ddqn_std,
            alpha=0.2
        )
    elif errorbar_style == "bars":
        plt.errorbar(
            dqn_steps,
            dqn_mean,
            yerr=dqn_std,
            fmt="none",
            elinewidth=1,
            alpha=0.5,
            capsize=2
        )
        plt.errorbar(
            ddqn_steps,
            ddqn_mean,
            yerr=ddqn_std,
            fmt="none",
            elinewidth=1,
            alpha=0.5,
            capsize=2
        )
    else:
        raise ValueError(f"Unknown errorbar style: {errorbar_style}")

    total_seeds = max(num_dqn, num_ddqn)
    title = f"{title_env}: DQN vs Double DQN"
    if num_dqn == num_ddqn:
        title += f" ({num_dqn} seeds)"
    else:
        title += f" (DQN={num_dqn}, DDQN={num_ddqn})"

    plt.title(title, fontsize=16, fontweight="bold")
    plt.xlabel("Timesteps", fontsize=15)
    plt.ylabel("Average Return", fontsize=15)
    plt.legend(fontsize=13)
    plt.grid(True, alpha=0.35)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved plot to: {output_path}")

    if show_plot:
        plt.show()

    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Plot mean and std of DQN vs Double DQN CSV results."
    )
    parser.add_argument(
        "--dqn_csvs",
        type=str,
        nargs="+",
        required=True,
        help="One or more CSV paths for DQN runs."
    )
    parser.add_argument(
        "--ddqn_csvs",
        type=str,
        nargs="+",
        required=True,
        help="One or more CSV paths for Double DQN runs."
    )
    parser.add_argument(
        "--title_env",
        type=str,
        default="LunarLander-v3",
        help="Environment name to use in the plot title."
    )
    parser.add_argument(
        "--errorbar_style",
        type=str,
        choices=["band", "bars"],
        default="band",
        help="Use 'band' to match the shown figure, or 'bars' for actual error bars."
    )
    parser.add_argument(
        "--output",
        type=str,
        default="dqn_vs_ddqn_plot.png",
        help="Path to save the output figure."
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show the plot interactively."
    )

    args = parser.parse_args()

    for p in args.dqn_csvs + args.ddqn_csvs:
        if not os.path.exists(p):
            print(f"Error: file does not exist: {p}", file=sys.stderr)
            sys.exit(1)

    dqn_dfs = [load_csv(p) for p in args.dqn_csvs]
    ddqn_dfs = [load_csv(p) for p in args.ddqn_csvs]

    dqn_steps, dqn_mean, dqn_std = aggregate_runs(dqn_dfs, "DQN")
    ddqn_steps, ddqn_mean, ddqn_std = aggregate_runs(ddqn_dfs, "Double DQN")

    plot_results(
        dqn_steps=dqn_steps,
        dqn_mean=dqn_mean,
        dqn_std=dqn_std,
        ddqn_steps=ddqn_steps,
        ddqn_mean=ddqn_mean,
        ddqn_std=ddqn_std,
        num_dqn=len(dqn_dfs),
        num_ddqn=len(ddqn_dfs),
        title_env=args.title_env,
        errorbar_style=args.errorbar_style,
        output_path=args.output,
        show_plot=args.show,
    )


if __name__ == "__main__":
    main()