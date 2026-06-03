import csv, json, sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches

# ── Config ─────────────────────────────────────────────────────────────────────
RESULTS_FILE = Path("results/rag_results.json")
RESULTS_DIR  = Path("results")
CSV_OUT      = RESULTS_DIR / "rag_results_extended.csv"

# No 0.0 — exclude no-context baseline from all plots and aggregation.
PLOT_RECALLS = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

# Auto-detect from directory name; falls back to the raw name.
_LABELS = {"hotpotqa": "HotpotQA", "msmarco": "MS-MARCO", "pubmedqa": "PubMedQA"}
DATASET_LABEL = _LABELS.get(Path(__file__).parent.name.lower(),
                             Path(__file__).parent.name)

# ── Load ───────────────────────────────────────────────────────────────────────
def load_results() -> list[dict]:
    if not RESULTS_FILE.exists():
        print(f"ERROR: {RESULTS_FILE} not found. Run 3_rag_sweep.py first.")
        sys.exit(1)
    with open(RESULTS_FILE) as f:
        return json.load(f)

# ── Semantic Similarity (MPNet) ────────────────────────────────────────────────
def compute_semantic_similarity(results: list[dict]) -> dict[int, dict[float, float]]:
    print("Computing Semantic Similarity (all-mpnet-base-v2)…")
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("  SKIP — pip install sentence-transformers")
        sim_map: dict[int, dict[float, float]] = defaultdict(dict)
        for qi in range(len(results)):
            for r in PLOT_RECALLS:
                sim_map[qi][r] = float("nan")
        return sim_map

    model = SentenceTransformer("all-mpnet-base-v2")

    golds: list[str]              = []
    tests: list[str]              = []
    index_map: list[tuple[int, float]] = []   # (qi, recall_at_k)

    for qi, entry in enumerate(results):
        gold = entry["gold_answer"]
        for sw in entry["sweeps"]:
            r = round(sw["recall_at_k"], 1)
            if r == 0.0:
                continue   # skip no-context baseline
            golds.append(gold)
            tests.append(sw["answer"])
            index_map.append((qi, r))

    if not golds:
        return defaultdict(dict)

    all_texts = golds + tests
    embs = model.encode(
        all_texts, batch_size=64, show_progress_bar=True,
        normalize_embeddings=True, convert_to_numpy=True,
    )
    gold_embs = embs[:len(golds)]
    test_embs = embs[len(golds):]
    sims = (gold_embs * test_embs).sum(axis=1)   # dot of unit vectors = cosine

    sim_map: dict[int, dict[float, float]] = defaultdict(dict)
    for idx, (qi, r) in enumerate(index_map):
        sim_map[qi][r] = float(sims[idx])

    print(f"  Done.  Mean sim @ R=1.0: "
          f"{np.mean([sim_map[qi].get(1.0, float('nan')) for qi in range(len(results))]):.3f}")
    return sim_map

# ── BERTScore (DistilBERT) ─────────────────────────────────────────────────────
def compute_bertscore(results: list[dict]) -> dict[int, dict[float, float]]:
    print("Computing BERTScore (distilbert-base-uncased)…")
    try:
        from bert_score import score as bert_score_fn
    except ImportError:
        print("  SKIP — pip install bert-score")
        bs_map: dict[int, dict[float, float]] = defaultdict(dict)
        for qi in range(len(results)):
            for r in PLOT_RECALLS:
                bs_map[qi][r] = float("nan")
        return bs_map

    cands: list[str]              = []
    refs:  list[str]              = []
    index_map: list[tuple[int, float]] = []

    for qi, entry in enumerate(results):
        gold = entry["gold_answer"]
        for sw in entry["sweeps"]:
            r = round(sw["recall_at_k"], 1)
            if r == 0.0:
                continue
            cands.append(sw["answer"])
            refs.append(gold)
            index_map.append((qi, r))

    if not cands:
        return defaultdict(dict)

    _, _, F1 = bert_score_fn(
        cands, refs,
        model_type="distilbert-base-uncased",
        lang="en",
        verbose=False,
        batch_size=32,
    )
    f1_vals = F1.numpy().tolist()

    bs_map: dict[int, dict[float, float]] = defaultdict(dict)
    for idx, (qi, r) in enumerate(index_map):
        bs_map[qi][r] = float(f1_vals[idx])

    print(f"  Done.  Mean BERTScore @ R=1.0: "
          f"{np.mean([bs_map[qi].get(1.0, float('nan')) for qi in range(len(results))]):.3f}")
    return bs_map

# ── Aggregate ──────────────────────────────────────────────────────────────────
def aggregate(
    results: list[dict],
    sim_map: dict[int, dict[float, float]],
    bs_map:  dict[int, dict[float, float]],
) -> dict[float, dict]:
    buckets: dict[float, dict] = defaultdict(lambda: {
        "sims": [], "bs": [],
        "grades_self_10": [], "grades_unbiased_10": [],
    })

    for qi, entry in enumerate(results):
        for sw in entry["sweeps"]:
            r = round(sw["recall_at_k"], 1)
            if r == 0.0:
                continue
            b = buckets[r]
            b["sims"].append(sim_map[qi].get(r, float("nan")))
            b["bs"].append(bs_map[qi].get(r, float("nan")))
            b["grades_self_10"].append(float(sw.get("grade_self_10", float("nan"))))

    def _mean(lst: list) -> float:
        clean = [x for x in lst if not (isinstance(x, float) and np.isnan(x))]
        return float(np.mean(clean)) if clean else float("nan")

    return {
        r: {
            "sim":               _mean(v["sims"]),
            "bert":              _mean(v["bs"]),
            "grade_self_10":     _mean(v["grades_self_10"]),
            "n":                 len(v["equivs_self"]),
        }
        for r, v in sorted(buckets.items())
    }

# ── Normalise helper ───────────────────────────────────────────────────────────
def _norm(agg: dict, key: str, recalls: list[float]) -> list[float]:
    base = agg[1.0][key]
    if not base or np.isnan(base):
        return [float("nan")] * len(recalls)
    return [agg[r][key] / base * 100 for r in recalls]

# ── CSV ────────────────────────────────────────────────────────────────────────
def save_csv(agg: dict, path: Path) -> None:
    fields = ["recall_at_k", "semantic_sim", "bertscore_f1", "grade_self_10", "n"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r, v in sorted(agg.items()):
            def _fmt(x):
                return round(x, 4) if not (isinstance(x, float) and np.isnan(x)) else ""
            w.writerow({
                "recall_at_k":       r,
                "semantic_sim":      _fmt(v["sim"]),
                "bertscore_f1":      _fmt(v["bert"]),
                "grade_self_10":     _fmt(v["grade_self_10"]),
                "n":                 v["n"],
            })
    print(f"CSV saved → {path}")

# ── Plot helpers ───────────────────────────────────────────────────────────────
def _tight_axes(ax, recalls: list[float]) -> None:
    margin = (recalls[-1] - recalls[0]) * 0.06
    ax.set_xlim(recalls[0] - margin, recalls[-1] + margin)
    ax.set_xticks(recalls)
    ax.set_xticklabels([f"{r:.1f}" for r in recalls], fontsize=9)
    ax.grid(True, alpha=0.28, linestyle="--")

# ── Plot 1: AER (Biased vs Unbiased) ──────────────────────────────────────────
def plot_semantic_metrics(agg: dict, save_path: Path) -> None:
    recalls = [r for r in PLOT_RECALLS if r in agg]

    sim_pct            = _norm(agg, "sim",               recalls)
    bert_pct           = _norm(agg, "bert",              recalls)
    grade_self_pct     = _norm(agg, "grade_self_10",     recalls)

    has_grades_self     = not all(np.isnan(v) for v in grade_self_pct)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.axhline(100.0, color="#cccccc", linestyle=":", linewidth=1.5)

    ax.plot(recalls, sim_pct,  color="#E67E22", linewidth=2.2, marker="D",
            markersize=7, label="Semantic Similarity (MPNet)")
    ax.plot(recalls, bert_pct, color="#C0392B", linewidth=2.0, marker="^",
            markersize=7, label="BERTScore F1 (DistilBERT)")

    if has_grades_self:
        ax.plot(recalls, grade_self_pct, color="#2980B9", linewidth=2.2,
                marker="o", markersize=7,
                label="LLM Grade 1-10 (Llama 3.1:8b)")

    _tight_axes(ax, recalls)
    ax.set_xlabel("Recall@10 (fraction of exact gold docs retrieved)", fontsize=12)
    ax.set_ylabel("% of Exact Search Performance (Recall=1.0)", fontsize=12)
    ax.set_title(
        f"{DATASET_LABEL}: Semantic & Graded Quality vs. Search Recall\n"
        "Normalised to Recall=1.0 Baseline  |  Human Gold Answers",
        fontsize=11, fontweight="bold",
    )
    valid    = [v for v in all_vals if not np.isnan(v)]
    ax.set_ylim(max(0, min(valid) - 8), 112)
    ax.legend(fontsize=9, loc="lower right")

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Plot saved → {save_path}")

# ── Plot 3: Per-query AER heatmap ─────────────────────────────────────────────
def print_summary(agg: dict, n_queries: int) -> None:
    sep = "─" * 72
    print(f"\n{sep}")
    print(f"  {DATASET_LABEL}  |  n={n_queries} queries")
    print(sep)
    print(f"  {'Recall':>6}  "
          f"{'Sim':>6}  {'BERT':>6}  {'G-Self':>7}  {'G-Unb':>7}  {'n':>4}")
    print(f"  {'------':>6}  "
          f"{'---':>6}  {'----':>6}  {'------':>7}  {'-----':>7}  {'-':>4}")
    for r, v in sorted(agg.items()):
        def _f(x): return f"{x:.3f}" if not np.isnan(x) else "  —  "
        print(f"  {r:>6.1f}  {_f(v['sim']):>6}  {_f(v['bert']):>6}  {_f(v['grade_self_10']):>7}  {v['n']:>4d}")
    print(sep)
    if 1.0 in agg and 0.4 in agg:
        delta_sim  = agg[1.0]["sim"]          - agg[0.4]["sim"]
        print(f"  Semantic-Sim drop  R=0.4→1.0: {delta_sim:+.3f}")
    print()

# ── Main ───────────────────────────────────────────────────────────────────────
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

results = load_results()
print(f"\n{DATASET_LABEL} — {len(results):,} queries loaded")

sim_map = compute_semantic_similarity(results)
bs_map  = compute_bertscore(results)
agg     = aggregate(results, sim_map, bs_map)

print_summary(agg, len(results))
save_csv(agg, CSV_OUT)

plot_semantic_metrics(agg,    RESULTS_DIR / "rag_semantic_metrics.png")
