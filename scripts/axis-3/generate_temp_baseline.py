import argparse, csv, json, re, time
from pathlib import Path

import numpy as np
import ollama

BASE = Path(__file__).parent

K = 10
NEAR_POOL_SIZE = 60
LLM_MODEL = "llama3.1:8b"

DATASETS = [
    dict(
        name         = "SciFact",
        results_dir  = BASE / "scifact/results",
        data_dir     = BASE / "scifact/data",
        index_file   = BASE / "scifact/index/rag_scifact_hnsw.bin",
        context_hdr  = "Context from scientific abstracts:",
        question_lbl = "Question/Claim",
    ),
    dict(
        name         = "NFCorpus",
        results_dir  = BASE / "nfcorpus/results",
        data_dir     = BASE / "nfcorpus/data",
        index_file   = BASE / "nfcorpus/index/rag_nfcorpus_hnsw.bin",
        context_hdr  = "Context from biomedical abstracts:",
        question_lbl = "Question/Claim",
    ),]

# ── LLM helpers ───────────────────────────────────────────────────────────────
def _call(prompt: str, temperature: float = 0.0, max_retries: int = 5) -> str:
    for attempt in range(max_retries):
        try:
            return ollama.generate(
                model=LLM_MODEL, prompt=prompt,
                options={"temperature": temperature},
            )["response"].strip()
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            wait = 15 * (attempt + 1)
            print(f"  [Ollama retry {attempt+1}: {e}] waiting {wait}s …", flush=True)
            time.sleep(wait)
            for _ in range(20):
                try:
                    ollama.list(); break
                except Exception:
                    time.sleep(5)
    return ""

def generate_answer(question: str, passages: list[str],
                    context_hdr: str, question_lbl: str,
                    temperature: float) -> str:
    context = "\n\n".join(passages)
    prompt = (
        f"{context_hdr}\n{context}\n\n"
        f"{question_lbl}: {question}\n\n"
        "Instructions: Answer using ONLY the information in the provided passages above. "
        "Do not use any outside knowledge or information from your training. "
        "If the provided passages do not contain enough information to answer, "
        "respond with exactly: 'Insufficient info'."
    )
    return _call(prompt, temperature=temperature)

def grade_answer(question: str, gold: str, test: str) -> int:
    if test == gold:
        return 10
    prompt = (
        f"Question: {question}\n"
        f"Gold Answer: {gold}\n"
        f"Test Answer: {test}\n\n"
        "Rate the factual consistency of the Test Answer against the Gold Answer "
        "on a scale of 1 to 10, where:\n"
        "1 = Completely contradicts or fails to mention the main facts.\n"
        "5 = Partially consistent, missing some key details but not severely hallucinating.\n"
        "10 = Perfectly consistent factual conclusion.\n"
        "Respond with ONLY the integer score (e.g., 1, 4, 7, 10)."
    )
    resp = _call(prompt, temperature=0.0)
    nums = re.findall(r'\b(?:10|[1-9])\b', resp)
    return int(nums[0]) if nums else 5

# ── Data loading ──────────────────────────────────────────────────────────────
def load_corpus(data_dir: Path) -> dict[int, dict]:
    docs: dict[int, dict] = {}
    with open(data_dir / "corpus.jsonl") as f:
        for line in f:
            docs[len(docs)] = json.loads(line)
    return docs

def load_queries(data_dir: Path) -> tuple[list[dict], np.ndarray]:
    queries = []
    with open(data_dir / "queries.jsonl") as f:
        for line in f:
            queries.append(json.loads(line))
    vecs = np.load(data_dir / "queries.npy")
    return queries, vecs

# ── Per-dataset runner ─────────────────────────────────────────────────────────
def run_dataset(ds: dict, temperature: float) -> None:
    import hnswlib
    name        = ds["name"]
    results_dir = ds["results_dir"]
    data_dir    = ds["data_dir"]
    index_file  = ds["index_file"]
    print(f"  {name}  (temperature={temperature})", flush=True)
    out_json = results_dir / "r1_temp_results.json"
    out_csv  = results_dir / "r1_temp_agg.csv"

    # Load source results (gold answers)
    with open(results_dir / "rag_results.json") as f:
        src = json.load(f)
    valid = [e for e in src if not e.get("skipped") and "gold_answer" in e]
    print(f"  {len(valid)} valid entries", flush=True)

    # Load corpus, queries, index
    docs             = load_corpus(data_dir)
    queries, vecs    = load_queries(data_dir)
    q_id_to_idx      = {q["id"]: i for i, q in enumerate(queries)}

    hnsw = hnswlib.Index(space="ip", dim=vecs.shape[1])
    hnsw.load_index(str(index_file))
    hnsw.set_ef(500)

    # Resume from checkpoint
    done: dict[str, dict] = {}
    if out_json.exists():
        with open(out_json) as f:
            for item in json.load(f):
                done[item["query_id"]] = item
        print(f"  Resuming — {len(done)} already done", flush=True)

    for i, entry in enumerate(valid):
        qid = entry["query_id"]
        if qid in done:
            continue

        q_text   = entry["query"]
        gold_ans = entry["gold_answer"]
        q_idx    = q_id_to_idx.get(qid)
        if q_idx is None:
            print(f"  WARN: {qid} not in queries.jsonl — skip", flush=True)
            continue

        q_vec = vecs[q_idx].reshape(1, -1)
        q_vec = q_vec / max(float(np.linalg.norm(q_vec)), 1e-9)

        labels, _ = hnsw.knn_query(q_vec, k=K + NEAR_POOL_SIZE)
        gold_ids  = list(labels[0])[:K]
        passages  = [docs[lid]["text"] for lid in gold_ids]

        temp_ans = generate_answer(
            q_text, passages,
            ds["context_hdr"], ds["question_lbl"],
            temperature=temperature,
        )
        grade = grade_answer(q_text, gold_ans, temp_ans)

        done[qid] = {
            "query_id":      qid,
            "query":         q_text,
            "gold_answer":   gold_ans,
            "temp_answer":   temp_ans,
            "grade_self_10": grade,
        }

        if (i + 1) % 10 == 0 or (i + 1) == len(valid):
            print(f"  [{i+1}/{len(valid)}] grade={grade}  len={len(temp_ans)}", flush=True)

        with open(out_json, "w") as f:
            json.dump(list(done.values()), f, indent=2)

    # Batch SemanticSim + BERTScore
    items = list(done.values())
    if not all("sim" in it for it in items):
        print(f"\n  Computing SemanticSim (MPNet) …", flush=True)
        from sentence_transformers import SentenceTransformer
        mpnet = SentenceTransformer("all-mpnet-base-v2")
        golds = [it["gold_answer"] for it in items]
        tests = [it["temp_answer"] for it in items]
        embs  = mpnet.encode(golds + tests, batch_size=64, normalize_embeddings=True,
                             show_progress_bar=True, convert_to_numpy=True)
        sims  = (embs[:len(golds)] * embs[len(golds):]).sum(axis=1)

        print(f"  Computing BERTScore (DistilBERT) …", flush=True)
        from bert_score import score as bert_score_fn
        _, _, F1 = bert_score_fn(tests, golds, model_type="distilbert-base-uncased",
                                  lang="en", verbose=False, batch_size=32)
        f1s = F1.numpy().tolist()

        for it, sim_val, bert_val in zip(items, sims, f1s):
            it["sim"]  = round(float(sim_val),  6)
            it["bert"] = round(float(bert_val), 6)

        with open(out_json, "w") as f:
            json.dump(items, f, indent=2)
        print(f"  JSON updated → {out_json}", flush=True)

    # Aggregate summary
    grades_ = [it["grade_self_10"] for it in items]
    sims_   = [it["sim"]           for it in items if "sim"  in it]
    berts_  = [it["bert"]          for it in items if "bert" in it]

    mean_grade = float(np.mean(grades_))
    mean_sim   = float(np.mean(sims_))  if sims_  else float("nan")
    mean_bert  = float(np.mean(berts_)) if berts_ else float("nan")

    print(f"\n  Results (n={len(items)}):  grade={mean_grade:.3f}  sim={mean_sim:.4f}  bert={mean_bert:.4f}", flush=True)

    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "mean_value", "n"])
        w.writerow(["grade_self_10", round(mean_grade, 4), len(grades_)])
        if sims_:
            w.writerow(["semantic_sim",  round(mean_sim,  6), len(sims_)])
        if berts_:
            w.writerow(["bertscore_f1",  round(mean_bert, 6), len(berts_)])
    print(f"  Aggregate CSV → {out_csv}", flush=True)

# ── Main ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--temp", type=float, default=0.3)
parser.add_argument("--dataset", choices=["scifact", "nfcorpus", ],
                    help="Run one dataset only (lowercase, underscore)")
args = parser.parse_args()

try:
    ollama.list()
except Exception:
    raise SystemExit("ERROR: Ollama is not running. Start with: ./ollama_bin/bin/ollama serve &")

for ds in DATASETS:
    key = ds["name"].lower().replace("-", "_")
    if args.dataset and args.dataset != key:
        continue
    run_dataset(ds, temperature=args.temp)

print("\nAll datasets done.", flush=True)
