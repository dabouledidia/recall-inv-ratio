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
DS_LABEL  = {
    "MNIST": "MNIST", "fmnist": "Fashion-MNIST", "imagenet": "ImageNet",
    "agnews": "AGNews", "simplewiki-openai": "SimpleWiki", "gist": "Gist",
}

ALGORITHMS = {
    "Annoy":      {"color": "#C86400", "file": ROOT / "data/annoy-example.txt",
    "parser": "annoy"},
    "SuCo":       {"color": "#7B2D8B", "file": ROOT / "data/suco-example.txt",
    "parser": "suco"},
    "RaBitQ":
    {"color": "#00008B", "file": ROOT / "data/rabitq-example.txt",   "parser": "rabitq"},
    "HNSW":       {"color": "#8B4513", "file": ROOT / "data/hnsw-example.txt",
    "parser": "hnsw"},
    "SymphonyQG": {"color": "#006400", "file": ROOT / "data/symphonyQG-example.txt",
    "parser": "symqg"},
}

_PARSERS = {"hnsw": parse_hnsw, "rabitq": parse_rabitq, "symqg": parse_symqg,
            "suco": parse_suco, "annoy": parse_annoy}

THRESHOLDS = [0.90, 0.95, 0.99]
K_VALUES   = [20, 50, 100]
OUT_DIR = SCRIPT_DIR

plt.rcParams.update({"font.size": 12, "figure.dpi": 150})

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

def max_qps_at_threshold(
    runs: list[dict], dataset: str, algo: str, k: int, threshold: float, metric: str
) -> float | None:
    best_qps = None
    for r in runs:
        if r["dataset"] != dataset or r["algo"] != algo:
            continue
        if not r.get("query_ms") or r["query_ms"] <= 0:
            continue
        for pt in r["points"]:
            if pt["k"] != k:
                continue
            score = pt["recall"] if metric == "recall" else 1.0 / pt["ratio"]
            if score >= threshold:
                qps = 1000.0 / r["query_ms"]
                if best_qps is None or qps > best_qps:
                    best_qps = qps
    return best_qps

def compute_table(all_runs: dict[str, list[dict]], k: int) -> dict[tuple, float | None]:
    table = {}
    for metric in ("recall", "inv_ratio"):
        for ds in DATASETS:
            for algo_name, runs in all_runs.items():
                for T in THRESHOLDS:
                    table[(metric, ds, algo_name, T)] = max_qps_at_threshold(
                        runs, ds, algo_name, k, T, metric
                    )
    return table

def plot_bars(all_runs: dict[str, list[dict]], table: dict, k: int) -> tuple:
    algo_names  = list(all_runs.keys())
    n_algos = len(algo_names)
    n_thresh = len(THRESHOLDS)
    n_datasets  = len(DATASETS)

    bar_width = min(0.13, 0.80 / (n_algos * 2.2))
    x = np.arange(n_datasets)

    fig, axes = plt.subplots(
        1, n_thresh,
        figsize=(15, 3.6),
        sharey=False,
    )

    for col, T in enumerate(THRESHOLDS):
        ax = axes[col]
        _na_marks = []

        for j, ds in enumerate(DATASETS):
            for a_i, algo_name in enumerate(algo_names):
                color = ALGORITHMS[algo_name]["color"]
                rc = table[("recall",
                ds, algo_name, T)]
                ir = table[("inv_ratio", ds, algo_name, T)]

                grp_offset = (a_i - (n_algos - 1) / 2) * (bar_width * 2.2)
                rc_x = x[j] + grp_offset - bar_width / 2
                ir_x = x[j] + grp_offset + bar_width / 2

                if rc is not None:
                    ax.bar(rc_x, rc, width=bar_width,
                           color=color, alpha=0.90, zorder=3)
                else:
                    _na_marks.append((rc_x, color))

                if ir is not None:
                    ax.bar(ir_x, ir, width=bar_width,
                           facecolor="white", alpha=1.0,
                           edgecolor=color, linewidth=0.8, hatch="////", zorder=3)
                else:
                    _na_marks.append((ir_x, color))

        ax._na_marks = _na_marks

        ax.set_xticks(x)
        ax.set_xticklabels(
            [DS_LABEL[ds] for ds in DATASETS],
            rotation=35, ha="right", fontsize=12,
        )

        if col == 0:
            ax.set_ylabel("Max QPS (log scale)", fontsize=13)
        ax.set_title(f"T = {T:.2f}", fontsize=14, fontweight="bold", y=1.02)

        ax.tick_params(labelsize=12)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_yscale("log")
        ax.set_xlim(-0.6, n_datasets - 0.4)
        ax.grid(True, axis="y", alpha=0.20, linestyle="--", which="both")

    # Shared legend at bottom
    legend_handles = [
        Patch(facecolor="#666", alpha=0.90, label=f"Recall@{k} ≥ Threshold"),
        Patch(facecolor="white", alpha=1.0, edgecolor="#666", linewidth=0.8,
              hatch="////", label=f"1/Ratio@{k} ≥ Threshold"),
    ]
    # Add algorithm colors to legend
    for algo_name in algo_names:
        legend_handles.append(Patch(facecolor=ALGORITHMS[algo_name]["color"], alpha=0.90, label=algo_name))
        
    fig.legend(
        handles=legend_handles,
        loc="upper center", ncol=len(legend_handles), fontsize=12,
        framealpha=0.9, bbox_to_anchor=(0.5, 1.03),
    )

    fig.tight_layout(pad=0.4, h_pad=0.8, w_pad=0.8, rect=[0, 0, 1, 0.90])

    png_path = OUT_DIR / f"fig2_threshold_bars_qps_k{k}.png"
    pdf_path = OUT_DIR / f"fig2_threshold_bars_qps_k{k}.pdf"
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    if k == 100:
        fig.savefig(OUT_DIR / "fig2_threshold_bars_qps.png", dpi=150, bbox_inches="tight")
        fig.savefig(OUT_DIR / "fig2_threshold_bars_qps.pdf", bbox_inches="tight")

    plt.close(fig)
    return png_path, pdf_path

def print_report(all_runs: dict, table: dict, k: int) -> None:
    algo_names = list(all_runs.keys())
    print(f"fig2_threshold_bars_qps  — median speedup report  (k={k})")
    print()
    print(f"  {'Threshold':>9s}  {'algo':8s}  {'median speedup (blue/red)':>26s}  unreachable")
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
                elif ir is not None and rc > 0:
                    ratios.append(ir / rc)
            med = f"{np.median(ratios):.2f}x" if ratios else "N/A"
            unr = ", ".join(unreachable) if unreachable else "none"
            print(f"  T={T:.2f} {algo_name:8s}  {med:>26s}  {unr}")
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
