import csv, json
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

BASE = Path(__file__).parent

RECALLS = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

COLOR_RATIO = "#1f77b4"
COLOR_SIM   = "#27AE60"
COLOR_BERT  = "#9B59B6"
COLOR_GRADE = "#8c564b"

# part=1: corroborating datasets with LLM-generated gold (SciFact, NFCorpus)
# part=2: primary datasets with human-written gold (HotpotQA, MS-MARCO, PubMedQA)
DATASETS = [
    dict(label="SciFact",  results=BASE / "scifact/results",  part=1),
    dict(label="NFCorpus", results=BASE / "nfcorpus/results", part=1),
    dict(label="HotpotQA", results=BASE / "hotpotqa/results", part=2),
    dict(label="MS-MARCO", results=BASE / "msmarco/results",  part=2),
    dict(label="PubMedQA", results=BASE / "pubmedqa/results", part=2),
]


def load_part1(results_dir: Path) -> dict:
    agg = {r: {"sim": float("nan"), "bert": float("nan"),
               "grade": float("nan"), "ratio": float("nan")}
           for r in RECALLS}

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

    json_path = results_dir / "rag_results.json"
    if json_path.exists():
        with open(json_path) as f:
            data = json.load(f)
        buckets: dict = defaultdict(list)
        for entry in data:
            if entry.get("skipped"):
                continue
            for sw in entry.get("sweeps", []):
                rv = sw.get("ratio")
                r  = round(sw.get("recall_at_k", 0), 1)
                if r in agg and rv is not None:
                    buckets[r].append(float(rv))
        for r, vals in buckets.items():
            if vals:
                agg[r]["ratio"] = float(np.mean(vals))

    return agg


def load_part2(results_dir: Path) -> dict:
    agg = {r: {"sim": float("nan"), "bert": float("nan"),
               "grade": float("nan"), "ratio": float("nan")}
           for r in RECALLS}

    for name in ("rag_results_extended.csv", "rag_results_v2.csv"):
        csv_path = results_dir / name
        if csv_path.exists():
            with open(csv_path) as f:
                for row in csv.DictReader(f):
                    r = round(float(row["recall_at_k"]), 1)
                    if r not in agg:
                        continue
                    for src, dst in [("semantic_sim", "sim"), ("bertscore_f1", "bert"), ("ratio", "ratio")]:
                        if row.get(src):
                            agg[r][dst] = float(row[src])
                    for gk in ("grade_self_10", "llm_grade_self_10"):
                        if row.get(gk):
                            agg[r]["grade"] = float(row[gk])
                            break
            break

    if any(np.isnan(agg[r]["ratio"]) for r in RECALLS):
        json_path = results_dir / "rag_results.json"
        if json_path.exists():
            with open(json_path) as f:
                data = json.load(f)
            data = [e for e in data if not e.get("skipped")]
            buckets: dict = defaultdict(list)
            for entry in data:
                for sw in entry.get("sweeps", []):
                    r = round(sw.get("recall_at_k", 0), 1)
                    if r in agg and sw.get("ratio") is not None:
                        buckets[r].append(float(sw["ratio"]))
            for r, vals in buckets.items():
                if vals and np.isnan(agg[r]["ratio"]):
                    agg[r]["ratio"] = float(np.mean(vals))

    return agg


def normalise(agg: dict, part: int) -> dict:
    out = {}
    base = agg.get(1.0, {})
    for r in RECALLS:
        v = agg[r]
        out[r] = {}

        if part == 1:
            # sim/bert are already normalised against the temp=0.3 r=1.0 gold (self-similarity = 1.0)
            out[r]["sim"]   = v["sim"]  * 100 if not np.isnan(v["sim"])  else float("nan")
            out[r]["bert"]  = v["bert"] * 100 if not np.isnan(v["bert"]) else float("nan")
            g = v.get("grade", float("nan"))
            out[r]["grade"] = g / 10.0 * 100 if not np.isnan(g) else float("nan")
        else:
            for key in ("sim", "bert", "grade"):
                bv = base.get(key, float("nan"))
                vv = v.get(key, float("nan"))
                out[r][key] = vv / bv * 100 if (bv and not np.isnan(bv) and not np.isnan(vv)) else float("nan")

        rv = v.get("ratio", float("nan"))
        rb = base.get("ratio", float("nan"))
        if rv and rb and not np.isnan(rv) and not np.isnan(rb):
            out[r]["ratio_inv"] = (1.0 / rv) / (1.0 / rb) * 100
        else:
            out[r]["ratio_inv"] = float("nan")

    return out


def draw_panel(ax, norm, title, show_ylabel, y_min, y_max):
    ax.axhline(100.0, color="#cccccc", linestyle=":", linewidth=1.2, zorder=1)

    def get(key):
        return [norm[r][key] for r in RECALLS]

    ax.plot(RECALLS, get("ratio_inv"), color=COLOR_RATIO, linewidth=2.0,
            marker="s", markersize=5.5, linestyle="--", label="1 / Ratio", zorder=3)
    for key, color, marker, lbl in [
        ("sim",   COLOR_SIM,   "D", "Semantic Similarity"),
        ("bert",  COLOR_BERT,  "^", "BERTScore F1"),
        ("grade", COLOR_GRADE, "o", "Grade /10 (Llama)"),
    ]:
        vals = get(key)
        if not all(np.isnan(v) for v in vals):
            ax.plot(RECALLS, vals, color=color, linewidth=1.9,
                    marker=marker, markersize=5.5, label=lbl, zorder=4)

    margin = (RECALLS[-1] - RECALLS[0]) * 0.06
    ax.set_xlim(RECALLS[0] - margin, RECALLS[-1] + margin)
    ax.set_xticks(RECALLS)
    ax.set_xticklabels([f"{r:.1f}" for r in RECALLS], fontsize=13)
    ax.set_ylim(y_min, y_max)
    ax.tick_params(axis="y", labelsize=13)
    ax.grid(True, alpha=0.25, linestyle="--")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Recall@10", fontsize=15)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if show_ylabel:
        ax.set_ylabel("% of r=1.0 Performance", fontsize=15)


def build_figure():
    data = []
    for ds in DATASETS:
        agg = load_part1(ds["results"]) if ds["part"] == 1 else load_part2(ds["results"])
        norm = normalise(agg, ds["part"])
        data.append((ds["label"], norm))

    all_vals = []
    for _, norm in data:
        for r in RECALLS:
            for k in ("ratio_inv", "sim", "bert", "grade"):
                v = norm[r].get(k, float("nan"))
                if not np.isnan(v):
                    all_vals.append(v)

    y_min = max(0, min(all_vals) - 2) if all_vals else 80
    y_max = 106

    fig, axes = plt.subplots(1, 5, figsize=(4.5 * 5, 4.4), sharey=True)

    for col, (label, norm) in enumerate(data):
        draw_panel(axes[col], norm, label, show_ylabel=(col == 0), y_min=y_min, y_max=y_max)

    handles = [
        mlines.Line2D([], [], color=COLOR_RATIO, linewidth=2.0, linestyle="--",
                      marker="s", markersize=5.5, label="1 / Ratio"),
        mlines.Line2D([], [], color=COLOR_SIM,   linewidth=1.9,
                      marker="D", markersize=5.5, label="Semantic Similarity"),
        mlines.Line2D([], [], color=COLOR_BERT,  linewidth=1.8,
                      marker="^", markersize=5.5, label="BERTScore F1"),
        mlines.Line2D([], [], color=COLOR_GRADE, linewidth=1.9,
                      marker="o", markersize=5.5, label="Grade /10 (Llama)"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=4,
               bbox_to_anchor=(0.5, 1.02), frameon=True, framealpha=0.95, fontsize=17)
    fig.tight_layout(rect=[0, 0, 1, 0.91])

    for ext in ("png", "pdf"):
        p = BASE / f"fig_p3_quality_curves.{ext}"
        fig.savefig(p, dpi=150, bbox_inches="tight")
        print(f"  Saved → {p.name}")
    plt.close(fig)


print("Loading RAG evaluation results…")
build_figure()
print("Done.")
