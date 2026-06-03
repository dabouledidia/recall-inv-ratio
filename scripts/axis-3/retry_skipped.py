import json, sys, time, signal
from pathlib import Path

import numpy as np
import ollama
from tqdm import tqdm

BASE = Path(__file__).parent
K              = 10
RECALL_TARGETS = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
NEAR_POOL_SIZE = 60
LLM_MODEL = "llama3.1:8b"
QUERY_TIMEOUT = 7200
# 120 minutes

DATASETS = [
    {
        "label":
        "SciFact",
        "data_dir":   BASE / "scifact/data",
        "index_file": BASE / "scifact/index/rag_scifact_hnsw.bin",
        "out_file":   BASE / "scifact/results/rag_results.json",
    },
    {
        "label":
        "NFCorpus",
        "data_dir":   BASE / "nfcorpus/data",
        "index_file": BASE / "nfcorpus/index/rag_nfcorpus_hnsw.bin",
        "out_file":   BASE / "nfcorpus/results/rag_results.json",
    },
]

class QueryTimeout(BaseException):
    pass

def _timeout_handler(signum, frame):
    raise QueryTimeout()

def load_data(data_dir: Path):
    docs: dict[int, dict] = {}
    with open(data_dir / "corpus.jsonl") as f:
        for line in f:
            d = json.loads(line)
            docs[len(docs)] = d

    all_queries: list[dict] = []
    with open(data_dir / "queries.jsonl") as f:
        for line in f:
            all_queries.append(json.loads(line))

    all_vecs = np.load(data_dir / "queries.npy")

    with open(data_dir / "qrels.json") as f:
        qrels = json.load(f)
    qrel_ids = set(qrels.keys())

    grounded_queries, grounded_vecs = [], []
    for i, q in enumerate(all_queries):
        if q["id"] in qrel_ids:
            grounded_queries.append(q)
            grounded_vecs.append(all_vecs[i])

    query_vecs = np.stack(grounded_vecs)
    return docs, grounded_queries, query_vecs

def _ollama_call(prompt: str, model: str, max_retries: int = 5) -> str:
    for attempt in range(max_retries):
        try:
            return ollama.generate(
                model=model, prompt=prompt, options={"temperature": 0}
            )["response"].strip()
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            wait = 15 * (attempt + 1)
            tqdm.write(f"  [Ollama error attempt {attempt+1}: {e}] retrying in {wait}s…")
            time.sleep(wait)
            for _ in range(20):
                try:
                    ollama.list(); break
                except Exception:
                    time.sleep(5)

def get_llama_answer(question: str, passages: list[str]) -> str:
    context = "\n\n".join(passages)
    prompt = (
        f"Context from scientific abstracts:\n{context}\n\n"
        f"Question/Claim: {question}\n\n"
        "Instructions: Answer using ONLY the information in the provided abstracts above. "
        "Do not use any outside knowledge or information from your training. "
        "If the provided abstracts do not contain enough information to answer, "
        "respond with exactly: 'Insufficient info'."
    )
    return _ollama_call(prompt, model=LLM_MODEL)

def get_no_context_answer(question: str) -> str:
    prompt = (
        f"Question/Claim: {question}\n\n"
        "Instructions: Answer the question or verify the claim to the best of your knowledge. "
        "Do not refuse to answer. If you are unsure, provide your best factual guess."
    )
    return _ollama_call(prompt, model=LLM_MODEL)

def judge_equivalence_llm(q: str, gold: str, test: str, model: str) -> bool:
    if test == gold:
        return True
    prompt = (
        f"Question: {q}\n"
        f"Gold Answer (from perfect retrieval): {gold}\n"
        f"Test Answer (from approximate retrieval): {test}\n\n"
        "Does this answer match the factual conclusion of the Gold Answer? "
        "Respond with EXACTLY 'YES' or 'NO'."
    )
    return "YES" in _ollama_call(prompt, model=model).upper()

def build_retrieval_set(gold_ids, near_pool, gold_dists, near_pool_dists, recall_target):
    n_keep = round(recall_target * K)
    n_replace = K - n_keep
    ids = list(gold_ids[:n_keep]) + list(near_pool[:n_replace])
    actual_recall = n_keep / K
    ratio = float(n_keep)
    for j in range(n_keep, K):
        pool_idx = j - n_keep
        gt_d = gold_dists[j]
        ob_d = near_pool_dists[pool_idx]
        ratio += (ob_d / gt_d) ** 0.5 if gt_d > 0 else 1.0
    ratio /= K
    return ids, actual_recall, float(ratio)

def run_query(q_item, q_vec, docs_map, index) -> dict:
    q_vec = q_vec.reshape(1, -1)
    q_vec = q_vec / np.maximum(np.linalg.norm(q_vec), 1e-9)

    labels_all, dists_all = index.knn_query(q_vec, k=K + NEAR_POOL_SIZE)
    all_ids = list(labels_all[0])
    all_dists = list(dists_all[0])
    gold_ids = all_ids[:K]
    near_pool = all_ids[K:]
    gold_dists = all_dists[:K]
    near_pool_dists = all_dists[K:]

    passages_gold = [docs_map[l]["text"] for l in gold_ids]
    gold_answer   = get_llama_answer(q_item["text"], passages_gold)

    sweeps = []
    no_context_answer     = get_no_context_answer(q_item["text"])
    sweeps.append({
        "recall_target":
        0.0,
        "recall_at_k":
        0.0,
        "ratio":                   None,        "answer":
        no_context_answer,
    })

    for r_target in RECALL_TARGETS:
        ret_ids, actual_recall, ratio = build_retrieval_set(
            gold_ids, near_pool, gold_dists, near_pool_dists, r_target
        )
        passages = [docs_map[l]["text"] for l in ret_ids]

        if actual_recall == 1.0:
            test_answer = gold_answer
        else:
            test_answer = get_llama_answer(q_item["text"], passages)
        sweeps.append({
            "recall_target": r_target,
            "recall_at_k":
            actual_recall,
            "ratio":                   round(ratio, 6),            "answer":
            test_answer,
        })

    return {
        "query_id":
        q_item["id"],
        "query":
        q_item["text"],
        "gold_answer": gold_answer,
        "sweeps":
        sweeps,
    }

def process_dataset(cfg: dict) -> None:
    import hnswlib

    label = cfg["label"]
    data_dir   = cfg["data_dir"]
    index_file = cfg["index_file"]
    out_file   = cfg["out_file"]

    if not index_file.exists():
        print(f"[{label}] ERROR: index not found at {index_file} — skipping.")
        return
        print(f"[{label}] Loading data…")
    docs_map, grounded_queries, query_vecs = load_data(data_dir)

    with open(out_file) as f:
        results: list[dict] = json.load(f)

    # Remove skipped entries — we'll re-run them
    skipped_ids = {r["query_id"] for r in results if r.get("skipped")}
    results = [r for r in results if not r.get("skipped")]
    done_ids = {r["query_id"] for r in results}

    # Queries to process: skipped ones + ones never in the JSON
    to_run = [
        (gi, q) for gi, q in enumerate(grounded_queries)
        if q["id"] in skipped_ids or q["id"] not in done_ids
    ]

    print(f"[{label}] Previously skipped: {len(skipped_ids)}  |  Absent: {len([q for _,q in to_run if q['id'] not in skipped_ids])}")
    print(f"[{label}] Total to retry: {len(to_run)}")

    if not to_run:
        print(f"[{label}] Nothing to do.")
        return

    dim   = query_vecs.shape[1]
    index = hnswlib.Index(space="ip", dim=dim)
    index.load_index(str(index_file))
    index.set_ef(500)

    signal.signal(signal.SIGALRM, _timeout_handler)

    for gi, q_item in tqdm(to_run, desc=label):
        signal.alarm(QUERY_TIMEOUT)
        try:
            entry = run_query(q_item, query_vecs[gi], docs_map, index)
            signal.alarm(0)
            results.append(entry)
            tqdm.write(f"  [done] {q_item['id']} — {q_item['text'][:55]}")
        except QueryTimeout:
            signal.alarm(0)
            tqdm.write(f"  [TIMEOUT >2h] {q_item['id']} — skipping permanently")
            results.append({
                "query_id": q_item["id"],
                "query":
                q_item["text"],
                "skipped":  True,
                "sweeps":   [],
            })

        with open(out_file, "w") as f:
            json.dump(results, f, indent=2)

    done_now = sum(1 for r in results if not r.get("skipped"))
    print(f"[{label}] Done. Total={len(results)}  non-skipped={done_now}")

try:
    ollama.list()
except Exception:
    print("ERROR: Ollama not running.")
    print("Start it with:  ./ollama_bin/bin/ollama serve &")
    sys.exit(1)

    for dataset_cfg in DATASETS:
        process_dataset(dataset_cfg)

    print("\nAll datasets processed.")
