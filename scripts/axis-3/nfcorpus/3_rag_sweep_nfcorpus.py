import json, sys, time, signal
from pathlib import Path

import numpy as np
import ollama
from tqdm import tqdm

DATA_DIR = Path("data")
INDEX_FILE  = Path("index/rag_nfcorpus_hnsw.bin")
RESULTS_DIR = Path("results")
OUT_FILE = RESULTS_DIR / "rag_results.json"

K              = 10
RECALL_TARGETS = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
NEAR_POOL_SIZE = 60
LLM_MODEL = "llama3.1:8b"
JUDGE_MODEL_SELF = "llama3.1:8b"
N_QUERIES = 119
QUERY_TIMEOUT = 7200
# skip query if it exceeds this many seconds

class QueryTimeout(BaseException):
    pass

def _timeout_handler(signum, frame):
    raise QueryTimeout()

def load_data():
    print("Loading corpus and queries…")
    docs: dict[int, dict] = {}
    with open(DATA_DIR / "corpus.jsonl") as f:
        for line in f:
            d = json.loads(line)
            docs[len(docs)] = d

    # Load all queries with their original indices (needed to index query_vecs)
    all_queries: list[dict] = []
    with open(DATA_DIR / "queries.jsonl") as f:
        for line in f:
            all_queries.append(json.loads(line))

    # Filter to queries that have at least one relevant document (score >= 2)
    # These are the only queries for which the gold retrieval is meaningful.
    with open(DATA_DIR / "qrels.json") as f:
        qrels = json.load(f)
    qrel_ids = set(qrels.keys())

    all_vecs = np.load(DATA_DIR / "queries.npy")

    grounded_queries: list[dict] = []
    grounded_vecs: list[np.ndarray] = []
    for i, q in enumerate(all_queries):
        if q["id"] in qrel_ids:
            grounded_queries.append(q)
            grounded_vecs.append(all_vecs[i])

    query_vecs = np.stack(grounded_vecs)
    print(f"  {len(docs):,} corpus docs  |  "
          f"{len(grounded_queries)} grounded queries (have qrels with score≥2)")
    return docs, grounded_queries, query_vecs

def _ollama_call(prompt: str, max_retries: int = 5, model: str = None) -> str:
    if model is None: model = LLM_MODEL
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

def get_no_context_answer(question: str) -> str:
    prompt = (
        f"Question/Claim: {question}\n\n"
        "Instructions: Answer the question or verify the claim to the best of your knowledge. "
        "Do not refuse to answer. If you are unsure, provide your best factual guess."
    )
    return _ollama_call(prompt, model=LLM_MODEL)

def get_llama_answer(question: str, passages: list[str]) -> str:
    context = "\n\n".join(passages)
    prompt = (
        f"Context from biomedical abstracts:\n{context}\n\n"
        f"Question/Claim: {question}\n\n"
        "Instructions: Answer using ONLY the information in the provided abstracts above. "
        "Do not use any outside knowledge or information from your training. "
        "If the provided abstracts do not contain enough information to answer, "
        "respond with exactly: 'Insufficient info'."
    )
    return _ollama_call(prompt)

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
    # Per-position ratio matching the standard ANN benchmark formula:
    #   ratio = (1/K) * sum_j [ dist(obtained_j) / dist(groundtruth_j) ]
    # hnswlib ip-space returns 1-dot_product for L2-normalised vectors.
    # Euclidean = sqrt(2 * ip_dist), the sqrt(2) cancels in the ratio, so:
    #   ratio_j = sqrt(ip_dist_obtained_j / ip_dist_groundtruth_j)
    # Gold slots have obtained == groundtruth → ratio_j = 1.0.
    ratio = n_keep  # sum of 1.0 for each gold slot
    for j in range(n_keep, K):
        pool_idx = j - n_keep
        gt_d = gold_dists[j]
        ob_d = near_pool_dists[pool_idx]
        ratio += (ob_d / gt_d) ** 0.5 if gt_d > 0 else 1.0
    ratio /= K
    return ids, actual_recall, float(ratio)

def run_experiment():
    import hnswlib
    if not INDEX_FILE.exists():
        print(f"ERROR: {INDEX_FILE} not found. Run 2_build_index.py first.")
        sys.exit(1)

    docs_map, query_items, query_vecs = load_data()

    dim   = query_vecs.shape[1]
    index = hnswlib.Index(space="ip", dim=dim)
    index.load_index(str(INDEX_FILE))
    index.set_ef(500)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results: list[dict] = []

    if OUT_FILE.exists():
        with open(OUT_FILE) as f:
            results = json.load(f)
        print(f"Resuming from {len(results)} completed queries…")

    pbar = tqdm(total=N_QUERIES)
    pbar.update(len(results))

    signal.signal(signal.SIGALRM, _timeout_handler)

    for i in range(len(results), min(N_QUERIES, len(query_vecs))):
        q_item = query_items[i]
        signal.alarm(QUERY_TIMEOUT)
        try:
            q_vec  = query_vecs[i].reshape(1, -1)
            q_vec  = q_vec / np.maximum(np.linalg.norm(q_vec), 1e-9)

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
            # Add No-RAG baseline as recall=0.0 (no retrieval → ratio undefined)
            no_context_answer = get_no_context_answer(q_item["text"])
            no_context_equiv_self = judge_equivalence_llm(q_item["text"], gold_answer, no_context_answer, JUDGE_MODEL_SELF)
            sweeps.append({
                "recall_target": 0.0,
                "recall_at_k":   0.0,
                "ratio": None,
                "equivalent_to_gold_self": no_context_equiv_self,                "answer":
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
                    "recall_at_k": actual_recall,
                    "ratio": round(ratio, 6),
                    "equivalent_to_gold_self": llm_equiv_self,                    "answer":
                    test_answer,
                })

            results.append({
                "query_id":
                q_item["id"],
                "query":
                q_item["text"],
                "gold_answer": gold_answer,
                "sweeps":
                sweeps,
            })

            signal.alarm(0)  # cancel timer — query completed cleanly

            with open(OUT_FILE, "w") as f:
                json.dump(results, f, indent=2)

            if (i + 1) % 5 == 0:
                done = i + 1
                aer_06 = [
                    s["equivalent_to_gold_self"]
                    for r in results
                    for s in r["sweeps"]
                    if abs(s["recall_at_k"] - 0.6) < 0.01
                ]
                msg = f"  [Q{done}]"
                if aer_06:
                    msg += f"  AER@recall=0.6 so far: {sum(aer_06)/len(aer_06):.2f}"
                tqdm.write(msg)

        except QueryTimeout:
            signal.alarm(0)
            tqdm.write(f"  [Q{i+1}] TIMEOUT after {QUERY_TIMEOUT//60} min — skipping query")
            results.append({"query_id": q_item["id"], "query": q_item["text"],
                            "skipped": True, "sweeps": []})
            with open(OUT_FILE, "w") as f:
                json.dump(results, f, indent=2)

        pbar.update(1)

    with open(OUT_FILE, "w") as f:
        json.dump(results, f, indent=2)
    pbar.close()
    print(f"\nDone. Results → {OUT_FILE}")

try:
    ollama.list()
except Exception:
    print("ERROR: Ollama not running.")
    print("Start it with:  ../ollama_bin/bin/ollama serve &")
    sys.exit(1)
run_experiment()
