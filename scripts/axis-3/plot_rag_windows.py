import csv
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
OUT_DIR = SCRIPT_DIR

DATASETS = [("HotpotQA", "hotpotqa"), ("MS-MARCO", "msmarco"), ("PubMedQA", "pubmedqa"),
            ("SciFact", "scifact"), ("NFCorpus", "nfcorpus")]
SCENARIOS = [("default", "Default window"), ("worst-case", "Worst-case window")]
METRICS = ["sim", "bertscore", "grade"]
METRIC_LABELS = {"sim": "Sem. Sim.", "bertscore": "BERTScore", "grade": "Grade"}
C_RATIO = "#1f77b4"
SERIES = [("sim", "#27AE60", "D", "Semantic Similarity"),
          ("bertscore", "#9B59B6", "^", "BERTScore F1"),
          ("grade", "#8c564b", "o", "LLM Grade")]

plt.rcParams.update({"font.size": 12, "figure.dpi": 150})


def load(ds):
    path = SCRIPT_DIR / ds / "results" / "rag_window_sweep.csv"
    if not path.exists():
        return None
    data = {}
    for row in csv.DictReader(open(path)):
        d = data.setdefault(row["scenario"], {})
        r = float(row["recall"])
        cell = d.setdefault(r, {"inv": float(row["inv_ratio"])})
        cell[row["metric"]] = (float(row["value"]), float(row["value_norm"]))
    return data


def make_figure(data):
    labels = [lab for lab, ds in DATASETS if ds in data]
    keys = [ds for _, ds in DATASETS if ds in data]
    fig, axes = plt.subplots(2, len(keys), figsize=(4.4 * len(keys), 7.0),
                             sharey=True, sharex=True, squeeze=False)
    for ri, (scen, rlabel) in enumerate(SCENARIOS):
        for ci, ds in enumerate(keys):
            ax = axes[ri, ci]
            d = data[ds][scen]
            r = sorted(d)
            ax.axhline(100.0, color="#cccccc", linestyle=":", lw=1.2, zorder=1)
            ax.plot(r, [x * 100 for x in r], ":", color="#aaaaaa", lw=1.6,
                    label="Recall (= x, reference)", zorder=1)
            ax.plot(r, [d[x]["inv"] * 100 for x in r], "--s", color=C_RATIO, lw=2.0, ms=5.5,
                    label="1 / Ratio", zorder=3)
            for m, c, mk, lab in SERIES:
                ax.plot(r, [d[x][m][1] * 100 for x in r], color=c, lw=2.0, marker=mk, ms=5.0,
                        label=lab, zorder=2)
            if ri == 0:
                ax.set_title(labels[ci], fontsize=14, fontweight="bold", y=1.02)
            if ri == 1:
                ax.set_xlabel("Recall@10", fontsize=13)
            ax.set_xlim(min(r) - 0.025, 1.025)
            ax.set_ylim(38, 104)
            ax.grid(True, alpha=0.25)
        axes[ri, 0].set_ylabel(f"{rlabel}\n% of R=1.0 Performance", fontsize=13)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=5, bbox_to_anchor=(0.5, 1.02), frameon=True, fontsize=18)
    fig.tight_layout(rect=[0.02, 0, 1, 0.93])
    return fig


def fmt_range(lo, hi):
    a, b = f"{lo:.2f}", f"{hi:.2f}"
    return a if a == b else f"{a}--{b}"


def print_summary(data):
    print("\nLargest drop / rise of each metric vs its exact-search value (%), over all datasets")
    for scen, _ in SCENARIOS:
        for m in METRICS:
            devs = [(v[m][1] - 1) * 100 for ds in data for v in data[ds][scen].values()]
            print(f"  {scen:10s} {METRIC_LABELS[m]:10s} drop {-min(devs):6.2f}   rise {max(devs):6.2f}")
    print("\nLargest |deviation| of any metric from its exact-search value (%), per dataset")
    for scen, _ in SCENARIOS:
        row = "  ".join(f"{lab} {max(abs(v[m][1] - 1) * 100 for v in data[ds][scen].values() for m in METRICS):5.2f}"
                        for lab, ds in DATASETS if ds in data)
        print(f"  {scen:10s} {row}")
    print("\nTable 8 -- raw score ranges across r in [0.4, 1.0] (LLM grade divided by 10)")
    for scen, _ in SCENARIOS:
        print(f"  [{scen}]  {'Dataset':10s} {'1/Ratio':12s} {'Sem. Sim.':12s} {'BERTScore':12s} {'Grade':12s}")
        for lab, ds in DATASETS:
            if ds not in data:
                continue
            d = data[ds][scen].values()
            inv = [v["inv"] for v in d]
            cols = [fmt_range(min(inv), max(inv))]
            for m in METRICS:
                vals = [v[m][0] / (10 if m == "grade" else 1) for v in d]
                cols.append(fmt_range(min(vals), max(vals)))
            print(f"  {'':12s}{lab:10s} " + " ".join(f"{c:12s}" for c in cols))
    print("\nMAE (%) vs composite quality (mean of the three normalized metrics), r in {0.4, ..., 1.0}")
    for scen, _ in SCENARIOS:
        for lab, ds in DATASETS:
            if ds not in data:
                continue
            d = data[ds][scen]
            comp = {r: np.mean([d[r][m][1] for m in METRICS]) for r in d}
            mae_inv = np.mean([abs(d[r]["inv"] - comp[r]) for r in d]) * 100
            mae_rec = np.mean([abs(r - comp[r]) for r in d]) * 100
            print(f"  {scen:10s} {lab:10s} 1/Ratio {mae_inv:5.2f}   Recall {mae_rec:5.2f}   "
                  f"({mae_rec / mae_inv:.1f}x)   min composite {min(comp.values()):.3f}")


def main():
    data = {ds: d for _, ds in DATASETS if (d := load(ds)) is not None}
    if not data:
        print("No results found. Run rag_window_sweep.py first.")
        return
    fig = make_figure(data)
    for ext in ("pdf", "png"):
        path = OUT_DIR / f"fig_rag_windows.{ext}"
        fig.savefig(path, bbox_inches="tight")
        print(f"  saved -> {path}")
    plt.close(fig)
    print_summary(data)


if __name__ == "__main__":
    main()
