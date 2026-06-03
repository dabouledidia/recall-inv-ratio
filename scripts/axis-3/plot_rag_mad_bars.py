import csv, json
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path(__file__).parent
FINAL_DIR = BASE

RECALLS = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

DATASETS = [
    dict(label="SciFact",   results=BASE / "scifact/results",   part=1),
    dict(label="NFCorpus",  results=BASE / "nfcorpus/results",  part=1),
    dict(label="HotpotQA",  results=BASE / "hotpotqa/results",  part=2),
    dict(label="MS-MARCO",  results=BASE / "msmarco/results",   part=2),
    dict(label="PubMedQA",  results=BASE / "pubmedqa/results",  part=2),
]

COLOR_RECALL = "#8c564b"   # brown — same as Recall bar in pillar-2 fig6
COLOR_RATIO  = "#1f77b4"   # dark blue — same as 1/Ratio bar in pillar-2 fig6

def load_agg(ds: dict) -> dict[float, dict]:
    results_dir = ds["results"]
    agg: dict[float, dict] = {r: {"sim": float("nan"), "bert": float("nan"),
                                   "grade": float("nan"), "ratio": float("nan")}
                               for r in RECALLS}

    part = ds["part"]
    
    if part == 1:
        # Load from temp03_gold_agg.csv and temp03_grade_agg.csv for accurate baseline
        t03_path = results_dir / "temp03_gold_agg.csv"
        if t03_path.exists():
            with open(t03_path) as f:
                for row in csv.DictReader(f):
                    r = round(float(row["recall_at_k"]), 1)
                    if r in agg:
                        agg[r]["sim"]  = float(row["semantic_sim"])
                        agg[r]["bert"] = float(row["bertscore_f1"])

        t03g = results_dir / "temp03_grade_agg.csv"
        if t03g.exists():
            with open(t03g) as f:
                for row in csv.DictReader(f):
                    r = round(float(row["recall_at_k"]), 1)
                    if r in agg and row.get("grade_self_10"):
                        agg[r]["grade"] = float(row["grade_self_10"])
    else:
        ext_path = results_dir / "rag_results_extended.csv"
        if ext_path.exists():
            with open(ext_path) as f:
                for row in csv.DictReader(f):
                    r = round(float(row["recall_at_k"]), 1)
                    if r not in agg:
                        continue
                    if row.get("semantic_sim"):
                        agg[r]["sim"]  = float(row["semantic_sim"])
                    if row.get("bertscore_f1"):
                        agg[r]["bert"] = float(row["bertscore_f1"])
                    for key in ("grade_self_10", "llm_grade_self_10"):
                        if row.get(key):
                            agg[r]["grade"] = float(row[key])
                            break
                    if row.get("ratio"):
                        agg[r]["ratio"] = float(row["ratio"])

    # JSON fallback for ratio
    if any(np.isnan(agg[r]["ratio"]) for r in RECALLS):
        json_path = results_dir / "rag_results.json"
        if json_path.exists():
            with open(json_path) as f:
                data = json.load(f)
            data = [e for e in data if not e.get("skipped")]
            buckets: dict[float, list] = defaultdict(list)
            for entry in data:
                for sw in entry.get("sweeps", []):
                    r = round(sw.get("recall_at_k", 0), 1)
                    if r in agg and sw.get("ratio") is not None:
                        buckets[r].append(float(sw["ratio"]))
            for r, vals in buckets.items():
                if vals and np.isnan(agg[r]["ratio"]):
                    agg[r]["ratio"] = float(np.mean(vals))

    return agg

def compute_mad(agg: dict) -> tuple[float, float]:
    base_ratio = agg.get(1.0, {}).get("ratio", 1.0)
    if np.isnan(base_ratio) or base_ratio == 0:
        base_ratio = 1.0

    recall_errors, ratio_errors = [], []

    for r in RECALLS:
        v = agg[r]

        # Normalize each quality metric to R=1.0 baseline → %
        base = agg.get(1.0, {})
        quality_vals = []
        for key in ("sim", "bert"):
            val  = v.get(key, float("nan"))
            bval = base.get(key, float("nan"))
            if not (np.isnan(val) or np.isnan(bval) or bval == 0):
                quality_vals.append(val / bval * 100)
        # Grade: normalize to R=1.0 grade
        gval  = v.get("grade", float("nan"))
        gbval = base.get("grade", float("nan"))
        if not (np.isnan(gval) or np.isnan(gbval) or gbval == 0):
            quality_vals.append(gval / gbval * 100)

        if not quality_vals:
            continue

        quality_pct = float(np.mean(quality_vals))
        recall_pct  = r * 100.0

        rv = v.get("ratio", float("nan"))
        if np.isnan(rv) or rv == 0:
            continue
        inv_ratio_pct = (1.0 / rv) / (1.0 / base_ratio) * 100

        recall_errors.append(abs(recall_pct  - quality_pct))
        ratio_errors.append( abs(inv_ratio_pct - quality_pct))

    mad_recall = float(np.mean(recall_errors)) if recall_errors else float("nan")
    mad_ratio  = float(np.mean(ratio_errors))  if ratio_errors  else float("nan")
    return mad_recall, mad_ratio

def make_figure(datasets: list[dict]) -> plt.Figure:
    labels, mad_recalls, mad_ratios, parts = [], [], [], []

    for ds in datasets:
        agg = load_agg(ds)
        mr, mrat = compute_mad(agg)
        labels.append(ds["label"])
        mad_recalls.append(mr)
        mad_ratios.append(mrat)
        parts.append(ds["part"])
        print(f"  {ds['label']:12s}  MAD(Recall)={mr:.2f} pp   MAD(1/Ratio)={mrat:.2f} pp")

    x     = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5.5))

    # Solid bars for all datasets
    rects1 = ax.bar(x - width/2, [v/100 for v in mad_recalls], width,
                    label="Recall@10 (prediction error)",
                    color=COLOR_RECALL, edgecolor="black", linewidth=0.6)
    rects2 = ax.bar(x + width/2, [v/100 for v in mad_ratios], width,
                    label=r"$1/\mathrm{Ratio}@10$ (prediction error)",
                    color=COLOR_RATIO, edgecolor="black", linewidth=0.6)

    ax.set_ylabel("Mean Absolute Deviation (MAD)\nfrom True Quality", fontsize=15)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=14, fontweight="bold")
    ax.tick_params(axis="y", labelsize=12)

    y_max = max(max(v/100 for v in mad_recalls), max(v/100 for v in mad_ratios)) * 1.22
    ax.set_ylim(0, y_max)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for rect in list(rects1) + list(rects2):
        h = rect.get_height()
        ax.annotate(f"{h:.3f}",
                    xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 4), textcoords="offset points",
                    ha="center", va="bottom", fontsize=12, fontweight="bold")

    handles, labels_leg = ax.get_legend_handles_labels()
    fig.legend(handles, labels_leg, loc="upper center", ncol=2,
               bbox_to_anchor=(0.5, 1.02), frameon=True, framealpha=0.95, fontsize=16)
    fig.tight_layout(rect=[0.02, 0.05, 1, 0.93])
    return fig

print("Computing MAD values…")
fig = make_figure(DATASETS)

FINAL_DIR.mkdir(parents=True, exist_ok=True)
for ext in ("png", "pdf"):
    out = FINAL_DIR / f"fig_p3_mad.{ext}"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  Saved → {out.name}")
plt.close(fig)
print("Done.")
