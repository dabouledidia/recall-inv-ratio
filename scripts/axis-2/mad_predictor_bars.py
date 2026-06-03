import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.ticker
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(ROOT))

DATASETS = ["mnist", "fmnist", "cifar10", "svhn"]
DATASET_LABELS = {
    "mnist":   "MNIST",
    "fmnist":  "fMNIST",
    "cifar10": "CIFAR-10",
    "svhn":
    "SVHN",
}

K_SINGLE = [1, 20, 50, 100]
K_ALL = [1, 20, 50, 100]

RECALL_LEVELS = [0.50, 0.75, 0.95, 1.00]
BAR_GROUP_NAMES = ["50% Recall", "75% Recall", "95% Recall", "Exact"]
GROUP_COLORS = ["#9B59B6", "#E67E22", "#E74C3C", "#3498DB"]

DATA_PATH = SCRIPT_DIR / "semantic_stability_multi_k.csv"
OUT_DIR = SCRIPT_DIR

plt.rcParams.update({"font.size": 11, "figure.dpi": 150})

def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df["inv_ratio"] = 1.0 / df["ratio"]
    return df

def get_values(df: pd.DataFrame, ds: str, k: int):
    sub = df[(df["dataset"] == ds) & (df["k"] == k)].sort_values("recall")
    if sub.empty:
        return [np.nan] * 4
    
    recalls = sub["recall"].values
    lps = sub["label_precision"].values
    
    lp_vals = [float(np.interp(r, recalls, lps)) for r in RECALL_LEVELS]
    return lp_vals

def draw_panel(ax, lp_vals, title, show_ylabel=False, k=None):
    n = len(BAR_GROUP_NAMES)
    x = np.arange(n)
    width = 0.6

    lp_bars = ax.bar(x, lp_vals, width, edgecolor="black", linewidth=0.8,
                     color=[c + "CC" for c in GROUP_COLORS])   # slight transparency

    ax.set_yscale("log")
    ax.set_ylim(0.1, 1.5)
    ax.set_yticks([0.1, 0.2, 0.3, 0.5, 0.7, 1.0])
    ax.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())

    ax.set_xticks(x)
    ax.set_xticklabels(BAR_GROUP_NAMES, rotation=20, ha='right', fontsize=10)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.tick_params(axis='y', labelsize=10)
    ax.grid(axis='y', alpha=0.3, which='major')
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ylabel = f"LP@{k}" if k else "LP@k"
    if show_ylabel:
        ax.set_ylabel(ylabel, fontsize=11)

    # Annotate LP values above each bar
    for bar, h in zip(lp_bars, lp_vals):
        if not np.isnan(h):
            ax.text(bar.get_x() + bar.get_width() / 2, h * 1.04,
                    f"{h:.3f}", ha='center', va='bottom', fontsize=8.5, fontweight='bold')
    return lp_bars

def make_single_k_figure(df, k, datasets):
    fig, axes = plt.subplots(1, len(datasets), figsize=(5.0 * len(datasets), 4.5), sharey=True)
    if len(datasets) == 1:
        axes = [axes]
    for i, (ax, ds) in enumerate(zip(axes, datasets)):
        lp_vals = get_values(df, ds, k)
        draw_panel(ax, lp_vals, title=DATASET_LABELS[ds], show_ylabel=(i == 0), k=k)
    fig.suptitle(f"Label Precision (LP@{k}) Stability Across Recall Levels  (k = {k})",
        fontsize=13, fontweight="bold", y=1.03
    )
    fig.tight_layout()
    return fig

def make_combined_figure(df, k_values, datasets):
    n_rows, n_cols = len(k_values), len(datasets)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(4.8 * n_cols, 4.2 * n_rows),
                             sharey=False)
    if n_rows == 1:
        axes = [axes]

    for row, k in enumerate(k_values):
        for col, ds in enumerate(datasets):
            ax = axes[row][col]
            lp_vals = get_values(df, ds, k)
            draw_panel(ax, lp_vals, title=f"{DATASET_LABELS[ds]}  (k={k})",
                       show_ylabel=(col == 0), k=k)

    fig.suptitle(
        "Label Precision (LP@k) Stability Across Recall Levels — all k values",
        fontsize=15, fontweight="bold", y=1.01
    )
    fig.tight_layout()
    return fig

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=DATASETS,
                        choices=list(DATASET_LABELS.keys()))
    args = parser.parse_args()

    df = load_data()

    # Per-k plots
    for k in K_SINGLE:
        fig = make_single_k_figure(df, k, args.datasets)
        stem = f"fig8_semantic_quality_bars_k{k}"
        for ext in ("png", "pdf"):
            path = OUT_DIR / f"{stem}.{ext}"
            fig.savefig(path, bbox_inches="tight")
            print(f"  saved -> {path}")
        plt.close(fig)

    # Combined plot
    fig = make_combined_figure(df, K_ALL, args.datasets)
    stem = "fig8_semantic_quality_bars_all"
    for ext in ("png", "pdf"):
        path = OUT_DIR / f"{stem}.{ext}"
        fig.savefig(path, bbox_inches="tight")
        print(f"  saved -> {path}")
    plt.close(fig)

main()
