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
DATASET_LABELS = {
    "mnist":   "MNIST",
    "fmnist":  "Fashion-MNIST",
    "cifar10": "CIFAR-10",
    "svhn":
    "SVHN",
}

DATA_PATH = SCRIPT_DIR / "semantic_stability.csv"
OUT_DIR   = SCRIPT_DIR
RECALL_MIN = 0.4

plt.rcParams.update({"font.size": 12, "figure.dpi": 150})

def load_data(recall_min: float = RECALL_MIN, window_N: int = 500) -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    if "inv_ratio" not in df.columns:
        df["inv_ratio"] = 1.0 / df["ratio"]
    if "window_N" in df.columns:                 # default window only
        df = df[df["window_N"] == window_N]
    return df[df["recall"] >= recall_min - 1e-9].copy()

def make_figure(df: pd.DataFrame, datasets: list[str]) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 5.5))

    mad_recall = []
    mad_ratio = []
    labels = []

    for ds in datasets:
        sub = df[df["dataset"] == ds]
        recall = sub["recall"].values
        inv_ratio = sub["inv_ratio"].values
        lp_norm = sub["lp_norm"].values

        mad_recall.append(np.mean(np.abs(lp_norm - recall)))
        mad_ratio.append(np.mean(np.abs(lp_norm - inv_ratio)))
        labels.append(DATASET_LABELS.get(ds, ds))

    x = np.arange(len(datasets))
    width = 0.35

    # Plot bars
    rects1 = ax.bar(x - width/2, mad_recall, width, label="Recall@100 (prediction error)", color="#8c564b", edgecolor="black", linewidth=0.5)
    rects2 = ax.bar(x + width/2, mad_ratio, width, label=r"$1/\mathrm{Ratio}@100$ (prediction error)", color="#1f77b4", edgecolor="black", linewidth=0.5)

    ax.set_ylabel("Mean Absolute Deviation (MAD)\nfrom True Quality", fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=14, fontweight="bold")
    ax.tick_params(axis="y", labelsize=12)

    # Increase y max a bit to leave room for text labels
    y_max = max(mad_recall) * 1.2
    ax.set_ylim(0, y_max)

    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    # Add text labels on top of bars
    for rect in rects1:
        height = rect.get_height()
        ax.annotate(f"{height:.3f}",
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 4),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=12, fontweight='bold')

    for rect in rects2:
        height = rect.get_height()
        ax.annotate(f"{height:.3f}",
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 4),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=12, fontweight='bold')

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    handles, labels_leg = ax.get_legend_handles_labels()
    fig.legend(handles, labels_leg, loc="upper center", ncol=2,
               bbox_to_anchor=(0.5, 1.02), frameon=True, framealpha=0.95, fontsize=14)
    fig.tight_layout(rect=[0.02, 0.05, 1, 0.93])
    return fig

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=DATASETS,
                        choices=list(DATASET_LABELS.keys()), metavar="DS")
    parser.add_argument("--recall-min", type=float, default=RECALL_MIN, dest="recall_min")
    args = parser.parse_args()

    df  = load_data(recall_min=args.recall_min)
    fig = make_figure(df, args.datasets)

    stem = "fig6_metric_correlation"
    for ext in ("png", "pdf"):
        path = OUT_DIR / f"{stem}.{ext}"
        fig.savefig(path, bbox_inches="tight")
        print(f"  saved -> {path}")
    plt.close(fig)

main()
