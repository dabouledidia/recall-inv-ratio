import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(ROOT))

from benchmarks import parse_hnsw, parse_rabitq, parse_symqg, parse_suco, parse_annoy

# Sorted by LID ascending
DATASETS_BY_LID = [
    "MNIST",
    "fmnist",
    "imagenet",
    "agnews",
    "simplewiki-openai",
    "gist",
]

DS_LABEL = {
    "MNIST":
    "MNIST",
    "fmnist":
    "Fashion-MNIST",
    "imagenet":
    "ImageNet",
    "agnews":
    "AGNews",
    "simplewiki-openai": "SimpleWiki",
    "gist":
    "Gist",
}
DS_LID = {
    "MNIST": 12.41, "fmnist": 13.99, "imagenet": 17.85,
    "agnews": 24.09, "simplewiki-openai": 27.45, "gist": 32.35,
}

DS_COLORS = {
    "MNIST":
    "#1B4F72",
    "fmnist":
    "#B7770D",
    "imagenet":
    "#6C3483",
    "agnews":
    "#1E8449",
    "simplewiki-openai": "#C0392B",
    "gist":
    "#555555",
}

DS_MARKERS = {
    "MNIST":
    "o",   # circle
    "fmnist":
    "s",   # square
    "imagenet":
    "D",   # diamond
    "agnews":
    "^",   # triangle up
    "simplewiki-openai": "v",   # triangle down
    "gist":
    "p",   # pentagon
}

ALGORITHMS = {
    "Annoy":
    {"file": ROOT / "data/annoy-example.txt",  "parser": parse_annoy},
    "SuCo":       {"file": ROOT / "data/suco-example.txt",
    "parser": parse_suco},
    "RaBitQ":
    {"file": ROOT / "data/rabitq-example.txt", "parser": parse_rabitq},
    "HNSW":       {"file": ROOT / "data/hnsw-example.txt",
    "parser": parse_hnsw},
    "SymphonyQG": {"file": ROOT / "data/symphonyQG-example.txt",
    "parser": parse_symqg},
}

THRESHOLD = 0.95
K_VALUES  = [1, 20, 50, 100]
OUT_DIR   = SCRIPT_DIR

def load_runs() -> dict[str, list[dict]]:
    runs = {}
    for name, cfg in ALGORITHMS.items():
        p = cfg["file"]
        if p.exists():
            runs[name] = cfg["parser"](str(p))
            print(f"  {name}: {len(runs[name])} runs")
        else:
            print(f"  [SKIP] {name}: {p} not found")
    return runs

def min_ms_at_threshold(
    algo_runs: list[dict], dataset: str, algo: str, k: int,
    threshold: float, metric: str
) -> float | None:
    best = None
    for r in algo_runs:
        if r["dataset"] != dataset or r["algo"] != algo:
            continue
        if not r.get("query_ms") or r["query_ms"] <= 0:
            continue
        for pt in r["points"]:
            if pt["k"] != k:
                continue
            score = pt["recall"] if metric == "recall" else 1.0 / pt["ratio"]
            if score >= threshold:
                if best is None or r["query_ms"] < best:
                    best = r["query_ms"]
    return best

def compute_speedups(all_runs: dict[str, list[dict]]) -> dict:
    result = {}
    for algo_name, algo_runs in all_runs.items():
        for ds in DATASETS_BY_LID:
            for k in K_VALUES:
                rc_ms = min_ms_at_threshold(
                    algo_runs, ds, algo_name, k, THRESHOLD, "recall")
                ir_ms = min_ms_at_threshold(
                    algo_runs, ds, algo_name, k, THRESHOLD, "inv_ratio")
                if rc_ms is None:
                    result[(algo_name, ds, k)] = None   # Recall unreachable
                elif ir_ms is None or ir_ms <= 0:
                    result[(algo_name, ds, k)] = 1.0
                else:
                    result[(algo_name, ds, k)] = rc_ms / ir_ms
    return result

def plot_speedup(
    all_runs: dict[str, list[dict]],
    speedups: dict,
) -> tuple:
    algo_names = list(all_runs.keys())
    n_algos = len(algo_names)

    plt.rcParams.update({
        "font.family":
        "sans-serif",
        "font.size":
        12,
        "figure.dpi":
        150,
        "axes.linewidth": 0.8,
    })

    fig, axes = plt.subplots(
        1, n_algos,
        figsize=(3.6 * n_algos, 3.5),
        sharey=True,   # same y-scale across all panels — key for comparison
    )
    if n_algos == 1:
        axes = [axes]

    for col, algo_name in enumerate(algo_names):
        ax = axes[col]

        for ds in DATASETS_BY_LID:
            color = DS_COLORS[ds]
            # Collect (k, speedup) pairs where speedup is not None
            pts = [
                (k, speedups[(algo_name, ds, k)])
                for k in K_VALUES
                if speedups.get((algo_name, ds, k)) is not None
            ]
            if not pts:
                continue

            # Plot connected segments, but only between consecutive non-None k values
            # Build segments: consecutive pairs
            ks, vals = zip(*pts)
            ax.plot(
                ks, vals,
                color=color,
                linewidth=2.0,
                marker=DS_MARKERS[ds],
                markersize=6,
                zorder=3,
                label=DS_LABEL[ds],
            )

        # Reference line at 1×
        ax.axhline(1.0, color="black", linewidth=0.8, linestyle="--",
                   alpha=0.4, zorder=2)

        ax.set_xlim(-2, 107)
        ax.set_xticks(K_VALUES)
        ax.set_xticklabels([str(k) for k in K_VALUES], fontsize=12)
        ax.set_xlabel("$k$", fontsize=13)

        ax.set_title(algo_name, fontsize=14, fontweight="bold", pad=6)

        if col == 0:
            ax.set_ylabel("QPS speedup (log scale)", fontsize=13)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, axis="y", alpha=0.18, linestyle="--", zorder=1)

    # Y-axis: log scale works well for speedups
    axes[0].set_yscale("log")
    axes[0].set_yticks([1, 2, 3, 5, 10, 15])
    axes[0].set_yticklabels(["1×", "2×", "3×", "5×", "10×", "15×"], fontsize=12)
    axes[0].set_ylim(0.7, 18)
    dataset_handles = [
        mlines.Line2D([], [], color=DS_COLORS[ds], linewidth=2.0,
                      marker=DS_MARKERS[ds], markersize=6,
                      label=f"{DS_LABEL[ds]}  (LID={DS_LID[ds]:.2f})")
        for ds in DATASETS_BY_LID
    ]

    fig.legend(
        handles=dataset_handles,
        loc="upper center",
        ncol=6,
        fontsize=13,
        framealpha=0.95,
        bbox_to_anchor=(0.5, 1.05),
        handlelength=2.0,
        columnspacing=1.0,
        handletextpad=0.5,
    )

    fig.tight_layout(pad=0.6, rect=[0, 0, 1, 0.90])

    png_path = OUT_DIR / "fig3_k_analysis.png"
    pdf_path = OUT_DIR / "fig3_k_analysis.pdf"
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(OUT_DIR / "fig_k_speedup_t095.png", dpi=150, bbox_inches="tight")
    fig.savefig(OUT_DIR / "fig_k_speedup_t095.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {png_path}")
    print(f"  Saved: {pdf_path}")
    return png_path, pdf_path

def main() -> None:
    print("Loading runs…")
    all_runs = load_runs()
    if not all_runs:
        print("ERROR: no runs loaded.")
        return

    print(f"Computing speedups at T={THRESHOLD}…")
    speedups = compute_speedups(all_runs)

    # Print summary table for verification
    print(f"\n{'Dataset':<22}", end="")
    for algo in all_runs:
        print(f"  {algo:>12}", end="")
    print()
    for ds in DATASETS_BY_LID:
        print(f"  {DS_LABEL[ds]:<20}", end="")
        for algo in all_runs:
            vals = [speedups.get((algo, ds, k)) for k in K_VALUES]
            strs = [f"{v:.2f}×" if v else "N/A" for v in vals]
            print(f"  {' '.join(strs):>12}", end="")
        print()

    print("\nPlotting…")
    plot_speedup(all_runs, speedups)

main()
