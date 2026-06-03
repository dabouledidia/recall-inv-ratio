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

# Perceptually distinct palette, works in B&W too (varies luminance)
DS_COLORS = {
    "MNIST":             "#1B4F72",   # dark steel-blue
    "fmnist":            "#B7770D",   # golden amber
    "imagenet":          "#6C3483",   # dark violet
    "agnews":            "#1E8449",   # dark green
    "simplewiki-openai": "#C0392B",   # vivid crimson
    "gist":              "#555555",   # medium charcoal
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

K_VALUES  = [1, 20, 50, 100]
OUT_DIR   = SCRIPT_DIR
MIN_SCORE = 0.10   # below this at k=100 → algorithm failed on this dataset

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

def algo_failed(algo_runs: list[dict], dataset: str, algo: str) -> bool:
    k_hi = K_VALUES[-1]
    for r in algo_runs:
        if r["dataset"] != dataset or r["algo"] != algo:
            continue
        for pt in r["points"]:
            if pt["k"] == k_hi and pt["recall"] >= MIN_SCORE:
                return False
    return True

def per_run_delta(
    algo_runs: list[dict], dataset: str, algo: str, metric: str
) -> list[list[float]]:
    k_set = set(K_VALUES)
    result = []
    for r in algo_runs:
        if r["dataset"] != dataset or r["algo"] != algo:
            continue
        score_by_k = {}
        for pt in r["points"]:
            if pt["k"] in k_set:
                s = pt["recall"] if metric == "recall" else 1.0 / pt["ratio"]
                score_by_k[pt["k"]] = s * 100.0
        if all(k in score_by_k for k in K_VALUES):
            anchor = score_by_k[K_VALUES[0]]   # score at k=1
            result.append([score_by_k[k] - anchor for k in K_VALUES])
    return result

def compute_mean_deltas(all_runs: dict[str, list[dict]]) -> dict:
    deltas = {}
    for algo_name, algo_runs in all_runs.items():
        for ds in DATASETS_BY_LID:
            failed = algo_failed(algo_runs, ds, algo_name)
            for metric in ("recall", "inv_ratio"):
                if failed:
                    deltas[(algo_name, ds, metric)] = None
                    continue
                runs_data = per_run_delta(algo_runs, ds, algo_name, metric)
                if not runs_data:
                    deltas[(algo_name, ds, metric)] = None
                    continue
                arr = np.array(runs_data)
                # shape (n_runs, n_k)
                means = np.mean(arr, axis=0)
                # shape (n_k,)
                deltas[(algo_name, ds, metric)] = list(zip(K_VALUES, means))
    return deltas

def plot_degradation(
    all_runs: dict[str, list[dict]],
    deltas: dict,
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
        "mathtext.fontset": "dejavusans",
    })

    fig, axes = plt.subplots(
        1, n_algos,
        figsize=(3.6 * n_algos, 4.9),
        sharey=True,
    )

    for col, algo_name in enumerate(algo_names):
        ax = axes[col]

        # Plot Recall (solid, dataset-specific filled marker)
        for ds in DATASETS_BY_LID:
            pts = deltas.get((algo_name, ds, "recall"))
            if pts is None:
                continue
            ks, vals = zip(*pts)
            ax.plot(
                ks, vals,
                color=DS_COLORS[ds],
                linewidth=2.0,
                linestyle="-",
                marker=DS_MARKERS[ds],
                markersize=6,
                zorder=3,
            )

        # Plot 1/Ratio (dotted, dataset-specific hollow marker)
        for ds in DATASETS_BY_LID:
            pts = deltas.get((algo_name, ds, "inv_ratio"))
            if pts is None:
                continue
            ks, vals = zip(*pts)
            ax.plot(
                ks, vals,
                color=DS_COLORS[ds],
                linewidth=2.0,
                linestyle=":",
                marker=DS_MARKERS[ds],
                markerfacecolor="white",
                markeredgecolor=DS_COLORS[ds],
                markeredgewidth=1.2,
                markersize=6,
                zorder=3,
            )

        # Zero-reference line
        ax.axhline(0, color="black", linewidth=0.9, linestyle="-", alpha=0.4, zorder=2)

        ax.set_xlim(-2, 107)
        ax.set_xticks(K_VALUES)
        ax.set_xticklabels([str(k) for k in K_VALUES], fontsize=12)
        ax.set_xlabel("$k$", fontsize=13)

        # Column titles (algorithm names)
        ax.set_title(algo_name, fontsize=14, fontweight="bold", pad=6)

        # Row labels on leftmost column only
        if col == 0:
            ax.set_ylabel(
                r"$\Delta$ Metric (% from $k$=1)",
                fontsize=13,
            )

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, axis="y", alpha=0.18, linestyle="--", zorder=1)

    # Set shared y-limits
    axes[0].set_ylim(-42, 5)
    axes[0].set_yticks([-40, -30, -20, -10, 0])
    axes[0].set_yticklabels(["-40%", "-30%", "-20%", "-10%", "0%"], fontsize=12)
    dataset_handles = []
    for ds in DATASETS_BY_LID:
        label_text = f"{DS_LABEL[ds]} (LID={DS_LID[ds]:.2f})"
        dataset_handles.append(
            mlines.Line2D([], [], color=DS_COLORS[ds], linewidth=2.2,
                          marker=DS_MARKERS[ds], markersize=6,
                          label=label_text)
        )

    metric_handles = [
        mlines.Line2D([], [], color="#666666", linewidth=2.0, linestyle="-",
                      marker="o", markerfacecolor="#666666", markersize=6,
                      label="Recall degradation"),
        mlines.Line2D([], [], color="#666666", linewidth=2.0, linestyle=":",
                      marker="o", markerfacecolor="white", markeredgecolor="#666666", markeredgewidth=1.2, markersize=6,
                      label="1/Ratio degradation"),
    ]

    fig.legend(
        handles=dataset_handles + metric_handles,
        loc="upper center",
        ncol=4,
        fontsize=16,
        framealpha=0.95,
        bbox_to_anchor=(0.5, 1.02),
        handlelength=2.5,
        columnspacing=1.6,
        handletextpad=0.6,
    )

    fig.tight_layout(pad=0.6, rect=[0, 0, 1, 0.81])

    png_path = OUT_DIR / "fig_lid_degradation.png"
    pdf_path = OUT_DIR / "fig_lid_degradation.pdf"
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
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

    print("Computing mean degradation curves (relative to k=1)…")
    deltas = compute_mean_deltas(all_runs)

    print("Plotting 1x5 merged figure…")
    plot_degradation(all_runs, deltas)

main()
