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

from benchmarks import parse_hnsw, parse_rabitq, parse_symqg, parse_suco, parse_annoy
DATASETS = ["MNIST", "fmnist", "agnews", "simplewiki-openai", "gist", "imagenet"]

ALGORITHMS = {
    "HNSW":       {"color": "#8B4513", "file": ROOT / "data/hnsw-example.txt",
    "parser": "hnsw"},
    "RaBitQ":
    {"color": "#00008B", "file": ROOT / "data/rabitq-example.txt",   "parser": "rabitq"},
    "SymphonyQG": {"color": "#006400", "file": ROOT / "data/symphonyQG-example.txt",
    "parser": "symqg"},
    "SuCo":       {"color": "#7B2D8B", "file": ROOT / "data/suco-example.txt",
    "parser": "suco"},
    "Annoy":      {"color": "#C86400", "file": ROOT / "data/annoy-example.txt",
    "parser": "annoy"},
}

_PARSERS = {"hnsw": parse_hnsw, "rabitq": parse_rabitq, "symqg": parse_symqg,
            "suco": parse_suco, "annoy": parse_annoy}

THRESHOLDS = [0.90, 0.95, 0.99]
K_VALUES   = [20, 50, 100]
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

def min_build_at_threshold(
    runs: list[dict], dataset: str, algo: str, k: int, threshold: float, metric: str
) -> float | None:
    best_ms = None
    for r in runs:
        if r["dataset"] != dataset or r["algo"] != algo:
            continue
        if not r.get("build_ms") or r["build_ms"] <= 0:
            continue
        for pt in r["points"]:
            if pt["k"] != k:
                continue
            score = pt["recall"] if metric == "recall" else 1.0 / pt["ratio"]
            if score >= threshold:
                if best_ms is None or r["build_ms"] < best_ms:
                    best_ms = r["build_ms"]
    return best_ms

def compute_table(all_runs: dict[str, list[dict]], k: int) -> dict[tuple, float | None]:
    table = {}
    for metric in ("recall", "inv_ratio"):
        for ds in DATASETS:
            for algo_name, runs in all_runs.items():
                for T in THRESHOLDS:
                    table[(metric, ds, algo_name, T)] = min_build_at_threshold(
                        runs, ds, algo_name, k, T, metric
                    )
    return table

def plot_bars(all_runs: dict[str, list[dict]], table: dict, k: int) -> tuple:
    algo_names = list(all_runs.keys())
    n_algos = len(algo_names)
    n_thresh   = len(THRESHOLDS)
    n_datasets = len(DATASETS)

    bar_width = 0.30
    x = np.arange(n_datasets)

    fig, axes = plt.subplots(
        n_thresh, n_algos,
        figsize=(5.0 * n_algos, 3.0 * n_thresh),
        sharey="row",
    )

    for row, T in enumerate(THRESHOLDS):
        for col, algo_name in enumerate(algo_names):
            ax = axes[row][col]

            color = ALGORITHMS[algo_name]["color"]
            _na_marks = []
            for j, ds in enumerate(DATASETS):
                rc = table[("recall",
                ds, algo_name, T)]
                ir = table[("inv_ratio", ds, algo_name, T)]

                if rc is not None:
                    ax.bar(x[j] - bar_width / 2, rc, width=bar_width,
                           color=color, alpha=0.88, zorder=3)
                else:
                    _na_marks.append((x[j] - bar_width / 2, color))

                if ir is not None:
                    ax.bar(x[j] + bar_width / 2, ir, width=bar_width,
                           facecolor="white", alpha=1.0,
                           edgecolor=color, linewidth=0.8, hatch="////", zorder=3)
                else:
                    _na_marks.append((x[j] + bar_width / 2, color))

            ax._na_marks = _na_marks

            ax.set_xticks(x)
            ax.set_xticklabels(
                [ds.replace("simplewiki-openai", "simplewiki") for ds in DATASETS],
                rotation=35, ha="right", fontsize=7,
            )

            if col == 0:
                ax.set_ylabel(f"Min build time (ms)", fontsize=8)
            ax.set_title(f"{algo_name}  |  T = {T:.2f}", fontsize=9, fontweight="bold")

            ax.tick_params(labelsize=7)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.set_yscale("log")
            ax.set_xlim(-0.6, n_datasets - 0.4)
            ax.grid(True, axis="y", alpha=0.20, linestyle="--", which="both")

    # Shared legend at top
    legend_handles = [
        Patch(facecolor="#666", alpha=0.88, label=f"Recall@{k} ≥ Threshold"),
        Patch(facecolor="white", alpha=1.0, edgecolor="#666", linewidth=0.8,
              hatch="////", label=f"1/Ratio@{k} ≥ Threshold"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center", ncol=3, fontsize=7,
        framealpha=0.9, bbox_to_anchor=(0.5, 1.01),
    )

    fig.subplots_adjust(hspace=0.45, wspace=0.25, left=0.07, right=0.97,
                        bottom=0.08, top=0.92)

    png_path = OUT_DIR / f"fig2_threshold_bars_build_k{k}.png"
    pdf_path = OUT_DIR / f"fig2_threshold_bars_build_k{k}.pdf"
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    if k == 100:
        fig.savefig(OUT_DIR / "fig2_threshold_bars_build.png", dpi=150, bbox_inches="tight")
        fig.savefig(OUT_DIR / "fig2_threshold_bars_build.pdf", bbox_inches="tight")

    plt.close(fig)
    return png_path, pdf_path

def print_report(all_runs: dict, table: dict, k: int) -> None:
    algo_names = list(all_runs.keys())
    print(f"fig2_threshold_bars_build  — geo-mean ratio report  (k={k})")
    print(f"  (ratio = red/blue = Recall build / 1/Ratio build; >1 means Recall costs more)")
    print()
    print(f"  {'Threshold':>9s}  {'algo':8s}  {'geo-mean ratio (red/blue)':>24s}  unreachable")
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
            med = f"{float(np.exp(np.mean(np.log(ratios)))):.3f}x" if ratios else "N/A"
            unr = ", ".join(unreachable) if unreachable else "none"
            print(f"  T={T:.2f} {algo_name:8s}  {med:>24s}  {unr}")
def main() -> None:
    print("Loading runs…")
    all_runs = load_all_runs()
    if not all_runs:
        print("ERROR: no runs loaded."); return

    for k in K_VALUES:
        print(f"\nProcessing k={k}...")
        table = compute_table(all_runs, k)
        png_path, pdf_path = plot_bars(all_runs, table, k)
        print_report(all_runs, table, k)
        print(f"  Saved: {png_path}\n  Saved: {pdf_path}")

main()
