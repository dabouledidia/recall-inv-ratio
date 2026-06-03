import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(ROOT))

from benchmarks import parse_hnsw, parse_rabitq, parse_symqg
DATASETS = ["MNIST", "fmnist", "agnews", "simplewiki-openai", "gist", "imagenet"]

ALGORITHMS = {
    "HNSW":       {"color": "#8B4513", "file": ROOT / "data/hnsw-example.txt",
    "parser": "hnsw"},
    "RaBitQ":
    {"color": "#00008B", "file": ROOT / "data/rabitq-example.txt", "parser": "rabitq"},
    "SymphonyQG": {"color": "#006400", "file": ROOT / "data/symphonyQG-example.txt",
    "parser": "symqg"},
}

_PARSERS = {"hnsw": parse_hnsw, "rabitq": parse_rabitq, "symqg": parse_symqg}

THRESHOLDS = [0.90, 0.95, 0.99]
K          = 100
OUT_DIR = SCRIPT_DIR

plt.rcParams.update({"font.size": 9, "figure.dpi": 150})

def load_all_runs() -> dict[str, list[dict]]:
    all_runs: dict[str, list[dict]] = {}
    for algo_name, cfg in ALGORITHMS.items():
        path = cfg["file"]
        if not Path(path).exists():
            print(f"  [SKIP] {algo_name}: {path} not found")
            continue
        runs = _PARSERS[cfg["parser"]](str(path))
        all_runs[algo_name] = runs
        print(f"  {algo_name}: {len(runs)} runs")
    return all_runs

def min_memory_at_threshold(
    runs: list[dict], dataset: str, algo: str, k: int, threshold: float, metric: str
) -> float | None:
    best_mem = None
    for r in runs:
        if r["dataset"] != dataset or r["algo"] != algo:
            continue
        mem = r.get("search_mem")
        if mem is None or mem <= 0:
            continue
        for pt in r["points"]:
            if pt["k"] != k:
                continue
            score = pt["recall"] if metric == "recall" else 1.0 / pt["ratio"]
            if score >= threshold:
                if best_mem is None or mem < best_mem:
                    best_mem = mem
    return best_mem

def compute_table(all_runs: dict[str, list[dict]]) -> dict[tuple, float | None]:
    table = {}
    for metric in ("recall", "inv_ratio"):
        for ds in DATASETS:
            for algo_name, runs in all_runs.items():
                for T in THRESHOLDS:
                    table[(metric, ds, algo_name, T)] = min_memory_at_threshold(
                        runs, ds, algo_name, K, T, metric
                    )
    return table

def plot_bars(all_runs: dict[str, list[dict]], table: dict) -> tuple:
    algo_names = list(all_runs.keys())
    n_algos = len(algo_names)
    n_datasets = len(DATASETS)

    bar_width = 0.13
    x = np.arange(n_datasets)

    fig, axes = plt.subplots(
        1, len(THRESHOLDS),
        figsize=(15, 4.5),
        sharey=False,
    )

    for col, T in enumerate(THRESHOLDS):
        ax = axes[col]
        _na_marks = []

        for ai, algo_name in enumerate(algo_names):
            color = ALGORITHMS[algo_name]["color"]
            
            grp_offset = (ai - 1) * (bar_width * 2.2)
            rc_offset = grp_offset - bar_width / 2
            ir_offset = grp_offset + bar_width / 2

            for j, ds in enumerate(DATASETS):
                rc = table[("recall",
                ds, algo_name, T)]
                ir = table[("inv_ratio", ds, algo_name, T)]

                if rc is not None:
                    ax.bar(x[j] + rc_offset, rc, width=bar_width,
                           color=color, alpha=0.88, zorder=3)

                if ir is not None:
                    ax.bar(x[j] + ir_offset, ir, width=bar_width,
                           facecolor="white", alpha=1.0,
                           edgecolor=color, linewidth=0.8, hatch="////", zorder=3)

        ax.set_xticks(x)
        ax.set_xticklabels(
            [ds.replace("simplewiki-openai", "simplewiki") for ds in DATASETS],
            rotation=35, ha="right", fontsize=7,
        )
        ax.set_title(f"T = {T:.2f}", fontsize=9, fontweight="bold")
        if col == 0:
            ax.set_ylabel(f"Min Search Phase Memory (MB)", fontsize=8)

        ax.tick_params(labelsize=7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_yscale("log")
        ax.set_xlim(-0.6, n_datasets - 0.4)
        ax.grid(True, axis="y", alpha=0.20, linestyle="--", which="both")

    # Shared legend at bottom
    legend_handles = [
        Patch(facecolor="#666", alpha=0.88, label=f"Recall@{K} ≥ Threshold"),
        Patch(facecolor="white", alpha=1.0, edgecolor="#666", linewidth=0.8,
              hatch="////", label=f"1/Ratio@{K} ≥ Threshold"),
    ]
    for algo_name in algo_names:
        legend_handles.append(Patch(facecolor=ALGORITHMS[algo_name]["color"], alpha=0.90, label=algo_name))

    fig.legend(
        handles=legend_handles,
        loc="lower center", ncol=5, fontsize=8,
        framealpha=0.9, bbox_to_anchor=(0.5, -0.05),
    )

    fig.subplots_adjust(hspace=0.45, wspace=0.25, left=0.07, right=0.97,
                        bottom=0.15, top=0.90)

    png_path = OUT_DIR / "fig_threshold_bars_memory_1x3.png"
    pdf_path = OUT_DIR / "fig_threshold_bars_memory_1x3.pdf"
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, pdf_path

def print_report(all_runs: dict, table: dict) -> None:
    algo_names = list(all_runs.keys())
    print(f"fig_threshold_bars_memory_1x3  — ratio report  (k={K})")
    print(f"  ratio = Recall memory / 1/Ratio memory")
    print(f"  >1 means Recall requires MORE memory")
    print()
    print(f"  {'Threshold':>9s}  {'algo':12s}  {'median ratio':>14s}  unreachable datasets")
    print("  " + "-" * 65)
    for T in THRESHOLDS:
        for algo_name in algo_names:
            ratios, unreachable = [], []
            for ds in DATASETS:
                rc = table[("recall",
                ds, algo_name, T)]
                ir = table[("inv_ratio", ds, algo_name, T)]
                if rc is None:
                    unreachable.append(ds)
                elif ir is not None and ir > 0:
                    ratios.append(rc / ir)
            med = f"{np.median(ratios):.2f}x" if ratios else "N/A"
            unr = ", ".join(unreachable) if unreachable else "none"
            print(f"  T={T:.2f} {algo_name:12s}  {med:>14s}  {unr}")
            
    print("\nSearch Phase Memory (MB) required at T=0.90")
    print("────────────────────────────────────────────────────────────────────────────────────────────────")
    print("Dataset          │       HNSW        │      RaBitQ       │ SymphonyQG")
    print("                 │  Recall  1/Ratio  │  Recall  1/Ratio  │  Recall  1/Ratio")
    print("─────────────────┼───────────────────┼───────────────────┼───────────────────")
    for ds in DATASETS:
        row_str = f"{ds:16s} │"
        for algo in ["HNSW", "RaBitQ", "SymphonyQG"]:
            rc = table[("recall", ds, algo, 0.90)]
            ir = table[("inv_ratio", ds, algo, 0.90)]
            rc_s = f"{int(rc):,}" if rc is not None else "N/A"
            ir_s = f"{int(ir):,}" if ir is not None else "N/A"
            row_str += f" {rc_s:>7s}  {ir_s:>7s}  │"
        print(row_str)
    print("────────────────────────────────────────────────────────────────────────────────────────────────")
def main() -> None:
    print("Loading runs…")
    all_runs = load_all_runs()
    if not all_runs:
        print("ERROR: no runs loaded."); return

    print("Computing table…")
    table = compute_table(all_runs)

    print("Plotting…")
    png_path, pdf_path = plot_bars(all_runs, table)
    print_report(all_runs, table)
    print(f"\n  Saved: {png_path}\n  Saved: {pdf_path}")

main()
