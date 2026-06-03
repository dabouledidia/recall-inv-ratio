import argparse, json, sys, time
from pathlib import Path

import numpy as np

CORPUS_NPY = Path("data/documents.npy")
INDEX_DIR = Path("index")
INDEX_FILE = INDEX_DIR / "rag_scifact_hnsw.bin"
META_FILE = INDEX_DIR / "index_meta.json"

DEFAULT_M = 16
DEFAULT_EFC = 200

def build_index(M, efC, threads):
    try:
        import hnswlib
    except ImportError:
        print("ERROR: pip install hnswlib"); sys.exit(1)

    if not CORPUS_NPY.exists():
        print(f"ERROR: {CORPUS_NPY} not found. Run 1_build_corpus.py first.")
        sys.exit(1)

    vecs = np.load(CORPUS_NPY)
    n, dim = vecs.shape
    
    # L2 normalize for cosine similarity if using 'ip'
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    vecs = vecs / np.maximum(norms, 1e-9)

    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Building HNSW index (n={n:,}, dim={dim}, M={M}, efC={efC}) …")
    index = hnswlib.Index(space="ip", dim=dim)
    index.init_index(max_elements=n, ef_construction=efC, M=M)
    index.set_num_threads(threads)
    
    t0 = time.time()
    index.add_items(vecs, np.arange(n))
    print(f"  Built in {time.time()-t0:.2f}s")

    index.save_index(str(INDEX_FILE))
    
    meta = {"n_vectors": n, "dim": dim, "M": M, "efC": efC, "space": "ip"}
    with open(META_FILE, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved → {INDEX_FILE}")

ap = argparse.ArgumentParser()
ap.add_argument("--m", type=int, default=DEFAULT_M)
ap.add_argument("--efc", type=int, default=DEFAULT_EFC)
ap.add_argument("--threads", type=int, default=4)
args = ap.parse_args()
build_index(args.m, args.efc, args.threads)
