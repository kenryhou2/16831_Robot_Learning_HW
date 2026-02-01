#!/usr/bin/env python3
"""
Plot Agents per Iter vs Mean Return with Std Dev error bars.
"""

import pandas as pd
import matplotlib.pyplot as plt

def main(csv_path):
    # --------------------------------------------------
    # Load CSV
    # --------------------------------------------------
    # CSV is comma-separated for hw1 results.
    df = pd.read_csv(csv_path)

    # --------------------------------------------------
    # Extract columns
    # --------------------------------------------------
    x = df["Training Steps per Iter"]
    y = df["Mean Return"]
    yerr = df["Stdev Return"]

    # --------------------------------------------------
    # Plot
    # --------------------------------------------------
    plt.figure()
    plt.errorbar(
        x,
        y,
        yerr=yerr,
        fmt='o-',
        capsize=5
    )

    plt.xlabel("Training Steps per Iteration")
    plt.ylabel("Mean Return")
    plt.title("Mean Return vs Training Steps per Iteration (Ants-V2)")
    plt.grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python plot_agents_vs_return.py results.csv")
        sys.exit(1)

    main(sys.argv[1])
