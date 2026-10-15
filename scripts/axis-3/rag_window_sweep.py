import argparse
import csv
import json
import os
import re
import subprocess
import threading
import time
import urllib.request
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent

K = 10
RECALLS = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
WINDOWS = {"default": 50, "worst-case": 500}
LLM_MODEL = "llama3.1:8b"
SEED = 0
CALL_DEADLINE = 240
OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
OLLAMA_BIN = os.environ.get("OLLAMA_BIN", "ollama")

DATASETS = {
    "hotpotqa": dict(ctx="Wikipedia passages",   ref_free=False, n_queries=200),
    "msmarco":  dict(ctx="web passages",         ref_free=False, n_queries=200),
    "pubmedqa": dict(ctx="biomedical abstracts", ref_free=False, n_queries=200),
    "scifact":  dict(ctx="scientific abstracts", ref_free=True,  n_queries=300),
    "nfcorpus": dict(ctx="biomedical abstracts", ref_free=True,  n_queries=None),
}


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def load_data(ds):
    base = SCRIPT_DIR / ds
    docs = {}
    with open(base / "data/corpus.jsonl") as f:
        for line in f:
            docs[len(docs)] = json.loads(line)
    allq = [json.loads(l) for l in open(base / "data/queries.jsonl")]
    vecs = np.load(base / "data/queries.npy")
    qids = set(json.load(open(base / "data/qrels.json")))
    ref_free = DATASETS[ds]["ref_free"]
    gq, gv = [], []
    for i, q in enumerate(allq):
        if q["id"] not in qids:
            continue
        if not ref_free:
            g = str(q.get("gold_human_answer", "")).strip()
            if not g or g == "No Answer Present.":
                continue
        gq.append(q)
        gv.append(vecs[i])
    V = np.stack(gv)
    V = V / np.maximum(np.linalg.norm(V, axis=1, keepdims=True), 1e-9)
    return docs, gq, V, base / "index" / f"rag_{ds}_hnsw.bin"


def build_set(gold_ids, gold_dists, pool_ids, pool_dists, r, W, rng):
    n_keep = round(r * K)
    n_rep = K - n_keep
    ids = list(gold_ids[:n_keep])
    ratio = float(n_keep)
    if n_rep > 0:
        sel = rng.choice(W - K, size=n_rep, replace=False)
        for m, j in enumerate(sel):
            ids.append(int(pool_ids[j]))
            gt = float(gold_dists[n_keep + m])
            ob = float(pool_dists[j])
            ratio += (ob / gt) ** 0.5 if gt > 0 else 1.0
    return ids, float(n_keep / K), float(ratio / K)


def restart_ollama():
    subprocess.run("pkill -x ollama", shell=True)
    time.sleep(4)
    env = dict(os.environ, OLLAMA_HOST="127.0.0.1:11434", OLLAMA_KEEP_ALIVE="-1")
    subprocess.Popen(f"{OLLAMA_BIN} serve > /dev/null 2>&1", shell=True, env=env)
    for _ in range(40):
        try:
            urllib.request.urlopen("http://127.0.0.1:11434/api/version", timeout=5)
            time.sleep(2)
            return True
        except Exception:
            time.sleep(3)
    return False


def call_llm(prompt, temperature=0.0, retries=2):
    body = json.dumps({"model": LLM_MODEL, "prompt": prompt, "stream": False,
                       "keep_alive": -1, "options": {"temperature": temperature}}).encode()
    last = None
    for _ in range(retries):
        box = {}

        def _do():
            try:
                req = urllib.request.Request(OLLAMA_URL, data=body,
                                             headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=CALL_DEADLINE) as r:
                    box["ok"] = json.loads(r.read())["response"].strip()
            except Exception as e:
                box["err"] = e

        t = threading.Thread(target=_do, daemon=True)
        t.start()
        t.join(CALL_DEADLINE)
        if t.is_alive():
            last = TimeoutError(f"hard deadline {CALL_DEADLINE}s")
            time.sleep(2)
            continue
        if "ok" in box:
            return box["ok"]
        last = box.get("err")
        time.sleep(5)
    raise last


def gen_answer(question, passages, ctx, temperature=0.0):
    prompt = (f"Context from {ctx}:\n" + "\n\n".join(passages) +
              f"\n\nQuestion: {question}\n\nInstructions: Answer using ONLY the information "
              "in the provided passages above. Do not use outside knowledge. If they do not "
              "contain enough information, respond with exactly: 'Insufficient info'.")
    return call_llm(prompt, temperature=temperature)


def grade(question, gold, test):
    if test == gold:
        return 10
    p = (f"Question: {question}\nGold Answer: {gold}\nTest Answer: {test}\n\n"
         "Rate the factual consistency of the Test Answer against the Gold Answer on a scale "
         "of 1 to 10, where:\n1 = Completely contradicts or fails to mention the main facts.\n"
         "5 = Partially consistent, missing some key details but not severely hallucinating.\n"
         "10 = Perfectly consistent factual conclusion.\nRespond with ONLY the integer score.")
    nums = re.findall(r'\b(?:10|[1-9])\b', call_llm(p))
    return int(nums[0]) if nums else 5


def phase_A(ds, nq, out):
    import hnswlib
    rng = np.random.default_rng(SEED)
    docs, gq, V, index_path = load_data(ds)
    ctx, ref_free = DATASETS[ds]["ctx"], DATASETS[ds]["ref_free"]
    nq = len(gq) if nq is None else min(nq, len(gq))
    fetch = max(WINDOWS.values())
    idx = hnswlib.Index(space="ip", dim=V.shape[1])
    idx.load_index(str(index_path))
    idx.set_ef(max(1500, fetch))
    results = json.load(open(out)) if out.exists() else []
    done = {r["qid"] for r in results}
    log(f"{ds}: {len(gq)} grounded queries; running {nq}; resuming from {len(results)}")
    for i in range(nq):
        q = gq[i]
        if q["id"] in done:
            continue
        try:
            lab, dist = idx.knn_query(V[i:i + 1], k=fetch)
            gold_ids, gold_d = lab[0][:K], dist[0][:K]
            pool_ids, pool_d = lab[0][K:], dist[0][K:]
            gold_ctx = [docs[int(x)]["text"] for x in gold_ids]
            gold_ans = gen_answer(q["text"], gold_ctx, ctx)
            gold_ref = gen_answer(q["text"], gold_ctx, ctx, temperature=0.3) if ref_free \
                else q["gold_human_answer"]
            entry = {"qid": q["id"], "question": q["text"], "gold_ref": gold_ref, "cells": []}
            g10 = grade(q["text"], gold_ref, gold_ans)
            for N in WINDOWS.values():
                entry["cells"].append(dict(window=N, recall=1.0, inv_ratio=1.0,
                                           answer=gold_ans, grade=g10))
            for N in WINDOWS.values():
                for r in RECALLS:
                    if r >= 1.0:
                        continue
                    ids, rec, ratio = build_set(gold_ids, gold_d, pool_ids, pool_d, r, N, rng)
                    ans = gen_answer(q["text"], [docs[int(x)]["text"] for x in ids], ctx)
                    entry["cells"].append(dict(window=N, recall=rec, inv_ratio=1.0 / ratio,
                                               answer=ans, grade=grade(q["text"], gold_ref, ans)))
            results.append(entry)
            json.dump(results, open(out, "w"), indent=1)
            log(f"  [{i + 1}/{nq}] {q['id']} done")
        except Exception as e:
            log(f"  [{i + 1}/{nq}] {q['id']} SKIPPED ({type(e).__name__}: {e}); restarting ollama")
            with open(out.parent / f"skipped_{ds}.txt", "a") as sf:
                sf.write(f"{q['id']}\n")
            restart_ollama()
    return results


def phase_B(out):
    from sentence_transformers import SentenceTransformer
    from bert_score import score as bert_score_fn
    results = json.load(open(out))
    pairs = [(c["answer"], r.get("gold_ref") or r.get("gold_human", ""))
             for r in results for c in r["cells"]]
    cand = [a for a, _ in pairs]
    ref = [g for _, g in pairs]
    log(f"phase B: scoring {len(pairs)} answers")
    mp = SentenceTransformer("all-mpnet-base-v2")
    ea = mp.encode(cand, batch_size=64, normalize_embeddings=True, show_progress_bar=False)
    eg = mp.encode(ref, batch_size=64, normalize_embeddings=True, show_progress_bar=False)
    sims = (ea * eg).sum(1)
    _, _, F1 = bert_score_fn(cand, ref, model_type="distilbert-base-uncased", lang="en",
                             verbose=False, batch_size=64)
    F1 = F1.numpy()
    k = 0
    for r in results:
        for c in r["cells"]:
            c["sim"] = float(sims[k])
            c["bertscore"] = float(F1[k])
            k += 1
    json.dump(results, open(out, "w"), indent=1)
    return results


def phase_C(ds, out, csv_path):
    results = json.load(open(out))
    metrics = ["grade", "sim", "bertscore"]
    rows = []
    for scenario, N in WINDOWS.items():
        agg = {}
        for r in RECALLS:
            cells = [c for res in results for c in res["cells"]
                     if c["window"] == N and abs(c["recall"] - r) < 1e-6]
            if cells:
                agg[r] = dict(inv=np.mean([c["inv_ratio"] for c in cells]),
                              **{m: np.mean([c[m] for c in cells]) for m in metrics})
        base = agg[1.0]
        for m in metrics:
            for r in sorted(agg):
                rows.append([ds, scenario, N, m, f"{r:.1f}", f"{agg[r][m]:.6f}",
                             f"{agg[r][m] / base[m]:.6f}", f"{agg[r]['inv']:.6f}"])
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "scenario", "window_N", "metric", "recall", "value",
                    "value_norm", "inv_ratio"])
        w.writerows(rows)
    log(f"{ds}: {len(results)} queries aggregated -> {csv_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(DATASETS))
    ap.add_argument("--n-queries", type=int, default=None, dest="n_queries")
    ap.add_argument("--phase", default="ABC")
    a = ap.parse_args()
    rd = SCRIPT_DIR / a.dataset / "results"
    rd.mkdir(parents=True, exist_ok=True)
    out = rd / "rag_window_sweep.json"
    csv_path = rd / "rag_window_sweep.csv"
    nq = a.n_queries if a.n_queries is not None else DATASETS[a.dataset]["n_queries"]
    if "A" in a.phase:
        phase_A(a.dataset, nq, out)
    if "B" in a.phase:
        phase_B(out)
    if "C" in a.phase:
        phase_C(a.dataset, out, csv_path)


if __name__ == "__main__":
    main()
