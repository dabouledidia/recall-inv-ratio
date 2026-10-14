"""
label_precision_vs_recall.py
============================
Plots LP@100 and 1/Ratio@100 (normalised to exact search) versus synthetic recall
for the four classification datasets, under both substitution windows:
    top row    = default window    (W = 500)
    bottom row = worst-case window (W = 1000)
The dotted y=x diagonal is Recall itself. Reads semantic_stability.csv produced by
generate_synthetic_recall_data.py.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(ROOT))

DATASETS = ["mnist", "fmnist", "cifar10", "svhn"]
DATASET_LABELS = {"mnist": "MNIST", "fmnist": "Fashion-MNIST", "cifar10": "CIFAR-10", "svhn": "SVHN"}
WINDOW_ROWS = [(500, "Default window"), (1000, "Worst-case window")]

DATA_PATH = SCRIPT_DIR / "semantic_stability.csv"
OUT_DIR   = SCRIPT_DIR
RECALL_MIN = 0.4

COLOR_LP = "#8c564b"       # brown
COLOR_RATIO = "#1f77b4"    # dark blue
COLOR_DIAGONAL = "#aaaaaa" # grey

plt.rcParams.update({"font.size": 12, "figure.dpi": 150})


def make_figure(df: pd.DataFrame, datasets: list[str]) -> plt.Figure:
    nrows, ncols = len(WINDOW_ROWS), len(datasets)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.8 * ncols, 3.6 * nrows),
                             sharey=True, squeeze=False)

    pool = df[df["dataset"].isin(datasets)]
    y_min = (min(pool["lp_norm"].min(), pool["inv_ratio"].min(), pool["recall"].min()) - 0.03) * 100.0
    y_max = 104.0

    for ri, (W, row_label) in enumerate(WINDOW_ROWS):
        for ci, ds in enumerate(datasets):
            ax = axes[ri][ci]
            sub = df[(df["dataset"] == ds) & (df["window_N"] == W)].sort_values("recall")
            if sub.empty:
                ax.set_visible(False); continue
            r_vals = sub["recall"].values
            lp = sub["lp_norm"].values * 100.0
            ir = sub["inv_ratio"].values * 100.0

            ax.plot(r_vals, r_vals * 100.0, linestyle=":", color=COLOR_DIAGONAL, linewidth=1.6,
                    label="Recall (= x, reference)", zorder=1)
            ax.plot(r_vals, ir, color=COLOR_RATIO, linewidth=2.5, linestyle="--",
                    marker="s", markersize=4, markeredgewidth=0,
                    label=r"$1\ /\ \mathrm{Ratio}@100$", zorder=2)
            ax.plot(r_vals, lp, color=COLOR_LP, linewidth=2.5,
                    marker="o", markersize=4, markeredgewidth=0,
                    label=r"$\mathrm{LP@100}(r)\ /\ \mathrm{LP@100}(1.0)$", zorder=3)

            if ri == 0:
                ax.set_title(DATASET_LABELS[ds], fontsize=14, fontweight="bold", y=1.02)
            if ri == nrows - 1:
                ax.set_xlabel("Recall@100", fontsize=13)
            if ci == 0:
                ax.set_ylabel(f"{row_label}\n% of R=1.0 Performance", fontsize=12)
            ax.set_xlim(r_vals.min() - 0.025, 1.025)
            ax.set_ylim(y_min, y_max)
            ax.axhline(100.0, color="black", linewidth=0.5, alpha=0.20)
            ax.grid(True, alpha=0.25)

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3,
               bbox_to_anchor=(0.5, 1.02), frameon=True, fontsize=14)
    fig.tight_layout(rect=[0.04, 0, 1, 0.94])
    return fig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=DATASETS,
                        choices=list(DATASET_LABELS.keys()), metavar="DS")
    parser.add_argument("--recall-min", type=float, default=RECALL_MIN, dest="recall_min")
    args = parser.parse_args()

    df = pd.read_csv(DATA_PATH)
    df = df[df["recall"] >= args.recall_min - 1e-9].copy()
    available = [ds for ds in args.datasets if not df[df["dataset"] == ds].empty]
    fig = make_figure(df, available)

    for ext in ("png", "pdf"):
        path = OUT_DIR / f"fig5_lp_stability.{ext}"
        fig.savefig(path, bbox_inches="tight", pad_inches=0.15)
        print(f"  saved -> {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
