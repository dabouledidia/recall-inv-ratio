import json, sys, time
from pathlib import Path

import numpy as np

CORPUS_NPY = Path("data/documents.npy")
INDEX_DIR  = Path("index")
INDEX_FILE = INDEX_DIR / "rag_msmarco_hnsw.bin"
META_FILE  = INDEX_DIR / "index_meta.json"

M   = 16
EFC = 200

def build_index():
    try:
        import hnswlib
    except ImportError:
        print("ERROR: pip install hnswlib"); sys.exit(1)

    if not CORPUS_NPY.exists():
        print(f"ERROR: {CORPUS_NPY} not found. Run ingest_msmarco.py first.")
        sys.exit(1)

    vecs = np.load(CORPUS_NPY)
    n, dim = vecs.shape

    # L2-normalise, inner product = cosine similarity
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    vecs  = vecs / np.maximum(norms, 1e-9)

    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Building HNSW index  n={n:,}  dim={dim}  M={M}  efC={EFC}…")
    index = hnswlib.Index(space="ip", dim=dim)
    index.init_index(max_elements=n, ef_construction=EFC, M=M)

    t0 = time.time()
    index.add_items(vecs, np.arange(n))
    print(f"  Built in {time.time() - t0:.2f}s")

    index.save_index(str(INDEX_FILE))
    print(f"  Saved → {INDEX_FILE}")

    with open(META_FILE, "w") as f:
        json.dump({"n_vectors": n, "dim": dim, "M": M, "efC": EFC,
                   "space": "ip"}, f, indent=2)

build_index()
