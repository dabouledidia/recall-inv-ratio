import json, csv, sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_FILE   = Path("results/rag_results.json")
CSV_OUT        = Path("results/rag_results_extended.csv")
RECALL_TARGETS = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
DATASET_LABEL  = "NFCorpus-BEIR"

def load_results() -> list[dict]:
    with open(RESULTS_FILE) as f:
        return json.load(f)

# ─── Semantic Similarity ──────────────────────────────────────────────────────

def compute_semantic_similarity(results: list[dict]) -> dict[int, dict[float, float]]:
    print("Computing Semantic Similarity (all-mpnet-base-v2)…")
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("  sentence-transformers not installed — skipping.")
        sim_map: dict[int, dict[float, float]] = defaultdict(dict)
        for qi in range(len(results)):
            for r in RECALL_TARGETS:
                sim_map[qi][r] = float("nan")
        return sim_map

    model = SentenceTransformer("all-mpnet-base-v2")
    cands, refs, index_map = [], [], []
    for qi, entry in enumerate(results):
        gold = entry["gold_answer"]
        for sw in entry["sweeps"]:
            r = round(sw["recall_at_k"], 1)
            if r == 1.0:
                continue
            cands.append(sw["answer"])
            refs.append(gold)
            index_map.append((qi, r))

    if not cands:
        sim_map = defaultdict(dict)
        for qi in range(len(results)):
            sim_map[qi][1.0] = 1.0
        return sim_map

    cand_embs = model.encode(cands, show_progress_bar=False, batch_size=32)
    ref_embs  = model.encode(refs,  show_progress_bar=False, batch_size=32)
    dots  = np.sum(cand_embs * ref_embs, axis=1)
    norms = np.linalg.norm(cand_embs, axis=1) * np.linalg.norm(ref_embs, axis=1)
    sims  = dots / np.maximum(norms, 1e-9)

    sim_map = defaultdict(dict)
    for idx, (qi, r) in enumerate(index_map):
        sim_map[qi][r] = float(sims[idx])
    for qi in range(len(results)):
        sim_map[qi][1.0] = 1.0
    return sim_map

# ─── BERTScore ────────────────────────────────────────────────────────────────

def compute_bertscore(results: list[dict]) -> dict[int, dict[float, float]]:
    print("Computing BERTScore (distilbert-base-uncased)…")
    try:
        from bert_score import score as bert_score_fn
    except ImportError:
        print("  bert-score not installed — skipping.")
        bs_map: dict[int, dict[float, float]] = defaultdict(dict)
        for qi in range(len(results)):
            for r in RECALL_TARGETS:
                bs_map[qi][r] = float("nan")
        return bs_map

    cands, refs, index_map = [], [], []
    for qi, entry in enumerate(results):
        gold = entry["gold_answer"]
        for sw in entry["sweeps"]:
            r = round(sw["recall_at_k"], 1)
            if r == 1.0:
                continue
            cands.append(sw["answer"])
            refs.append(gold)
            index_map.append((qi, r))

    if not cands:
        bs_map = defaultdict(dict)
        for qi in range(len(results)):
            bs_map[qi][1.0] = 1.0
        return bs_map

    _, _, F1 = bert_score_fn(
        cands, refs,
        model_type="distilbert-base-uncased",
        lang="en", verbose=False, batch_size=32,
    )
    f1_vals = F1.numpy().tolist()

    bs_map = defaultdict(dict)
    for idx, (qi, r) in enumerate(index_map):
        bs_map[qi][r] = float(f1_vals[idx])
    for qi in range(len(results)):
        bs_map[qi][1.0] = 1.0
    return bs_map

# ─── Aggregate ────────────────────────────────────────────────────────────────

def aggregate(results, sim_map, bs_map) -> dict[float, dict]:
    buckets = defaultdict(lambda: {"sims": [], "bs": [], "grades_unbiased_10": [], "grades_self_10": [], "ratios": []})
    for qi, entry in enumerate(results):
        for sw in entry["sweeps"]:
            r = round(sw["recall_at_k"], 1)
            if r == 0.0:
                continue
            b = buckets[r]
            b["sims"].append(sim_map[qi].get(r, float("nan")))
            b["bs"].append(bs_map[qi].get(r, float("nan")))
            b["grades_self_10"].append(sw.get("grade_self_10", float("nan")))
            if sw.get("ratio") is not None:
                b["ratios"].append(sw["ratio"])

    def _mean(lst):
        clean = [x for x in lst if not (isinstance(x, float) and np.isnan(x))]
        return float(np.mean(clean)) if clean else float("nan")

    return {
        r: {
            "sim":               _mean(v["sims"]),
            "bert":              _mean(v["bs"]),
            "grade_self_10":     _mean(v["grades_self_10"]),
            "ratio":             float(np.mean(v["ratios"])) if v["ratios"] else float("nan"),
            "n":                 len(v["sims"]),
        }
        for r, v in sorted(buckets.items())
    }

# ─── Save CSV ─────────────────────────────────────────────────────────────────

def save_csv(agg, path):
    fields = ["recall_at_k", "ratio", "semantic_sim", "bertscore_f1", "llm_grade_self_10", "n"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r, v in sorted(agg.items()):
            def _fmt(x):
                return round(x, 6) if not (isinstance(x, float) and np.isnan(x)) else ""
            w.writerow({
                "recall_at_k":           r,
                "ratio":                 _fmt(v["ratio"]),
                "semantic_sim":          _fmt(v["sim"]),
                "bertscore_f1":          _fmt(v["bert"]),
                "llm_grade_unbiased_10": _fmt(v["grade_unbiased_10"]),
                "llm_grade_self_10":     _fmt(v["grade_self_10"]),
                "n":                     v["n"],
            })
    print(f"CSV saved → {path}")

# ─── Plotting ─────────────────────────────────────────────────────────────────

def plot_quality_metrics(agg, save_path):
    recalls = [r for r in sorted(agg.keys()) if 0.0 < r <= 0.9]
    base_sim  = agg[1.0]["sim"]
    base_bert = agg[1.0]["bert"]

    sims  = [agg[r]["sim"]  / base_sim  * 100 if base_sim  > 0 else 0 for r in recalls]
    berts = [agg[r]["bert"] / base_bert * 100 if base_bert > 0 else 0 for r in recalls]

    has_unbiased = not np.isnan(agg[recalls[0]].get("grade_unbiased_10", float("nan")))
    has_self     = not np.isnan(agg[recalls[0]].get("grade_self_10",     float("nan")))

    # Ratio line: (1/ratio)*100 — stays near 100% showing the thin shell is geometrically tight
    ratio_vals = [agg[r].get("ratio", float("nan")) for r in recalls]
    ratio_pct  = [100.0 / rv if not np.isnan(rv) else float("nan") for rv in ratio_vals]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.axhline(100.0, color="#cccccc", linestyle=":", linewidth=1.5)

    ax.plot(recalls, ratio_pct, color="#7D3C98", linewidth=2.0, marker="s",
            markersize=7, linestyle="--", label="ANN Distance Ratio (geometric proximity)")

    ax.plot(recalls, sims,  color="#E67E22", linewidth=2.2, marker="D",
            markersize=7, label="Semantic Similarity (MPNet)")
    ax.plot(recalls, berts, color="#C0392B", linewidth=2.0, marker="^",
            markersize=7, label="BERTScore F1")

    if has_unbiased:
        base_g = agg[1.0]["grade_unbiased_10"]
        grades = [agg[r]["grade_unbiased_10"] / base_g * 100 for r in recalls]
        ax.plot(recalls, grades, color="#1ABC9C", linewidth=2.2, marker="X",
                markersize=8, label="LLM Graded Score (Unbiased — Qwen 1–10)")
    if has_self:
        base_g = agg[1.0]["grade_self_10"]
        grades = [agg[r]["grade_self_10"] / base_g * 100 for r in recalls]
        ax.plot(recalls, grades, color="#2980B9", linewidth=2.2, marker="o",
                markersize=8, label="LLM Graded Score (Self — Llama 1–10)")

    ax.set_xlabel("Recall@10 (fraction of exact gold docs retrieved)", fontsize=12)
    ax.set_ylabel("% of Exact Search Performance", fontsize=12)
    ax.set_title(
        f"{DATASET_LABEL}: Answer Quality vs. Search Recall\n"
        "Demonstrating the Thin Shell Effect",
        fontsize=11, fontweight="bold",
    )
    ax.set_xlim(0.37, 0.93)
    ax.set_ylim(50, 105)
    ax.set_xticks(recalls)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10, loc="lower right")

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Plot saved → {save_path}")

def plot_bertscore_vs_semsim(results, sim_map, bs_map, save_path):
    pts_x, pts_y = [], []
    for qi in range(len(results)):
        for r in RECALL_TARGETS:
            if r in (0.0, 1.0):
                continue
            x = sim_map[qi].get(r, float("nan"))
            y = bs_map[qi].get(r, float("nan"))
            if not np.isnan(x) and not np.isnan(y):
                pts_x.append(x)
                pts_y.append(y)

    if not pts_x:
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(pts_x, pts_y, color="#34495E", alpha=0.4, edgecolors="none")
    ax.set_xlabel("MPNet Cosine Similarity", fontsize=11)
    ax.set_ylabel("BERTScore F1", fontsize=11)
    ax.set_title(
        f"{DATASET_LABEL}: BERTScore vs Semantic Similarity\n(per query × recall level)",
        fontsize=12, fontweight="bold",
    )
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Plot saved → {save_path}")

def print_summary(agg, n_questions):
    print(f"\n  Total questions: {n_questions}")
    for rt in [0.4, 0.6, 0.8]:
        if rt in agg:
            v = agg[rt]
            g_u = f"{v['grade_unbiased_10']:.3f}" if not np.isnan(v['grade_unbiased_10']) else "N/A"
            g_s = f"{v['grade_self_10']:.3f}"     if not np.isnan(v['grade_self_10'])     else "N/A"
            print(f"  Recall={rt} →  SemanticSim={v['sim']:.3f}  BERTScore={v['bert']:.3f}"
                  f"  Grade_unbiased={g_u}  Grade_self={g_s}")

if not RESULTS_FILE.exists():
    print(f"ERROR: {RESULTS_FILE} not found. Run 3_rag_sweep.py first.")
    sys.exit(1)

results = load_results()
results = [r for r in results if not r.get("skipped")]

sim_map = compute_semantic_similarity(results)
bs_map  = compute_bertscore(results)
agg     = aggregate(results, sim_map, bs_map)

print_summary(agg, len(results))
save_csv(agg, CSV_OUT)

plot_quality_metrics(agg, Path("results/rag_quality_metrics.png"))
plot_bertscore_vs_semsim(results, sim_map, bs_map,
                         Path("results/rag_bertscore_vs_semsim.png"))
