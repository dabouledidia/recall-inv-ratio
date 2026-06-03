import json, os, sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

DATA_DIR = Path("data")
MODELS_DIR = Path("models") # for local model caching if needed

# Dataset files
CORPUS_JSONL = DATA_DIR / "corpus.jsonl"
QUERIES_JSONL = DATA_DIR / "queries.jsonl"
CORPUS_NPY = DATA_DIR / "documents.npy"
QUERIES_NPY = DATA_DIR / "queries.npy"
QRELS_JSON = DATA_DIR / "qrels.json"

EMBED_MODEL = "all-MiniLM-L6-v2"
BATCH_SIZE = 128

def build_data():
    try:
        from datasets import load_dataset
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("ERROR: pip install datasets sentence-transformers")
        sys.exit(1)

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading SciFact dataset from HuggingFace …")
    # SciFact has 'corpus', 'queries', and 'default' (qrels) configurations
    corpus_ds = load_dataset("mteb/scifact", "corpus", split="corpus")
    queries_ds = load_dataset("mteb/scifact", "queries", split="queries")
    qrels_ds = load_dataset("mteb/scifact", "default", split="test")

    # 1. Process Corpus
    print(f"Processing {len(corpus_ds):,} documents …")
    documents = []
    doc_texts = []
    with open(CORPUS_JSONL, "w") as f:
        for doc in corpus_ds:
            item = {"id": doc["_id"], "title": doc["title"], "text": doc["text"]}
            f.write(json.dumps(item) + "\n")
            documents.append(item)
            # Combine title and text for better embedding
            doc_texts.append(f"{doc['title']} {doc['text']}")

    # 2. Process Queries
    print(f"Processing {len(queries_ds):,} queries …")
    queries = []
    query_texts = []
    with open(QUERIES_JSONL, "w") as f:
        for q in queries_ds:
            item = {"id": q["_id"], "text": q["text"]}
            f.write(json.dumps(item) + "\n")
            queries.append(item)
            query_texts.append(q["text"])

    # 3. Process Qrels
    print(f"Processing Qrels …")
    qrels = {}
    for row in qrels_ds:
        qid = str(row["query-id"])
        did = str(row["corpus-id"])
        if row["score"] > 0:
            if qid not in qrels:
                qrels[qid] = []
            qrels[qid].append(did)
    with open(QRELS_JSON, "w") as f:
        json.dump(qrels, f, indent=2)

    # 4. Generate Embeddings
    print(f"\nLoading embedding model: {EMBED_MODEL} …")
    model = SentenceTransformer(EMBED_MODEL)
    
    print(f"Embedding {len(doc_texts):,} documents (this may take 1-2 mins on CPU) …")
    doc_embs = model.encode(doc_texts, batch_size=BATCH_SIZE, show_progress_bar=True, convert_to_numpy=True)
    np.save(CORPUS_NPY, doc_embs.astype(np.float32))

    print(f"Embedding {len(query_texts):,} queries …")
    query_embs = model.encode(query_texts, batch_size=BATCH_SIZE, show_progress_bar=True, convert_to_numpy=True)
    np.save(QUERIES_NPY, query_embs.astype(np.float32))

    print(f"\nDone! Data saved to {DATA_DIR}/")

build_data()
