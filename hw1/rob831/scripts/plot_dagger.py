#!/usr/bin/env python3
"""
Plot DAgger performance:
- Mean Return vs DAgger Iteration
- Error bars = Std Return
- Horizontal dotted line = BC performance (iteration 0 mean)
- Horizontal solid line = Expert performance (iteration 0 train max)
"""

import pandas as pd
import matplotlib.pyplot as plt

def main(csv_path):
    # --------------------------------------------------
    # Load CSV
    # --------------------------------------------------
    # Use whitespace separator since your table is space/tab separated
    df = pd.read_csv(csv_path)

    # Strip any accidental whitespace in headers
    df.columns = df.columns.str.strip()

    # --------------------------------------------------
    # Extract data
    # --------------------------------------------------
    iters = df["DAgger Iterations"]
    mean_return = df["Mean Return"]
    std_return = df["Stdev Return"]

    # BC baseline = mean return at iteration 0
    bc_performance = df.loc[df["DAgger Iterations"] == 0, "Mean Return"].iloc[0]

    # Expert baseline = train max return at iteration 0
    expert_performance = df.loc[df["DAgger Iterations"] == 0, "Train Max Return"].iloc[0]

    # --------------------------------------------------
    # Plot
    # --------------------------------------------------
    plt.figure()

    # DAgger curve with error bars
    plt.errorbar(
        iters,
        mean_return,
        yerr=std_return,
        fmt="o-",
        capsize=5,
        label="DAgger Mean Return"
    )

    # BC baseline (dotted)
    plt.axhline(
        bc_performance,
        linestyle="--",
        linewidth=2,
        label="BC Performance",
        color='orange'
    )

    # Expert baseline (solid, different color automatically)
    plt.axhline(
        expert_performance,
        linestyle="--",
        linewidth=2,
        label="Expert Performance",
        color='green'

    )

    # --------------------------------------------------
    # Labels and formatting
    # --------------------------------------------------
    plt.xlabel("DAgger Iteration")
    plt.ylabel("Return")
    plt.title("DAgger Performance vs Iteration")
    plt.legend()
    plt.grid(True)
    #add minor grid
    plt.minorticks_on()
    plt.grid(which='minor', linestyle=':', linewidth='0.5', color='gray')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python plot_dagger_performance.py dagger_results.csv")
        exit(1)

    main(sys.argv[1])
