import json, sys
from pathlib import Path

import numpy as np

DATA_DIR = Path("data")
CORPUS_JSONL  = DATA_DIR / "corpus.jsonl"
QUERIES_JSONL = DATA_DIR / "queries.jsonl"
CORPUS_NPY = DATA_DIR / "documents.npy"
QUERIES_NPY   = DATA_DIR / "queries.npy"
QRELS_JSON = DATA_DIR / "qrels.json"

EMBED_MODEL = "all-MiniLM-L6-v2"
BATCH_SIZE  = 128

def build_data():
    try:
        from datasets import load_dataset
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("ERROR: pip install datasets sentence-transformers")
        sys.exit(1)

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading NFCorpus from HuggingFace (mteb/nfcorpus)…")
    corpus_ds  = load_dataset("mteb/nfcorpus", "corpus",  split="corpus")
    queries_ds = load_dataset("mteb/nfcorpus", "queries", split="queries")
    qrels_ds   = load_dataset("mteb/nfcorpus", "default", split="test")
    print(f"Processing {len(corpus_ds):,} documents…")
    doc_texts = []
    with open(CORPUS_JSONL, "w") as f:
        for doc in corpus_ds:
            item = {"id": doc["_id"], "title": doc.get("title", ""),
                    "text": doc["text"]}
            f.write(json.dumps(item) + "\n")
            doc_texts.append(f"{item['title']} {item['text']}".strip())
    print(f"Processing {len(queries_ds):,} queries…")
    query_texts = []
    with open(QUERIES_JSONL, "w") as f:
        for q in queries_ds:
            item = {"id": q["_id"], "text": q["text"]}
            f.write(json.dumps(item) + "\n")
            query_texts.append(q["text"])
    # NFCorpus uses multi-level relevance: 0=not relevant, 1=somewhat, 2=relevant
    # We keep score >= 2 as relevant (consistent with BEIR standard for NFCorpus)
    print("Processing qrels (score >= 2 → relevant)…")
    qrels: dict[str, list[str]] = {}
    for row in qrels_ds:
        if row["score"] >= 2:
            qid = str(row["query-id"])
            did = str(row["corpus-id"])
            qrels.setdefault(qid, []).append(did)
    with open(QRELS_JSON, "w") as f:
        json.dump(qrels, f, indent=2)
    print(f"  {len(qrels):,} queries have at least one relevant document")
    print(f"\nLoading embedding model: {EMBED_MODEL}…")
    model = SentenceTransformer(EMBED_MODEL)

    print(f"Embedding {len(doc_texts):,} documents…")
    doc_embs = model.encode(doc_texts, batch_size=BATCH_SIZE,
                            show_progress_bar=True, convert_to_numpy=True)
    np.save(CORPUS_NPY, doc_embs.astype(np.float32))
    print(f"  Saved → {CORPUS_NPY}  shape={doc_embs.shape}")

    print(f"Embedding {len(query_texts):,} queries…")
    q_embs = model.encode(query_texts, batch_size=BATCH_SIZE,
                          show_progress_bar=True, convert_to_numpy=True)
    np.save(QUERIES_NPY, q_embs.astype(np.float32))
    print(f"  Saved → {QUERIES_NPY}  shape={q_embs.shape}")

    print(f"\nDone. Data in {DATA_DIR}/")

build_data()
