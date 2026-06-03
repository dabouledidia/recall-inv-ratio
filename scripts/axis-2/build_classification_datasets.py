import argparse
import os
import sys
import time
from pathlib import Path

import h5py
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent.parent
PILLAR2 = SCRIPT_DIR
N_QUERY    = 10_000
# test queries to keep
TOP_K      = 200
# top-K for extended neighbor cache

OUT_DIRS = {
    "cifar10": str(ROOT / "datasets/cifar10"),
    "svhn": str(ROOT / "datasets/svhn"),
}
HDF5_PATHS = {
    "cifar10": str(ROOT / "datasets/cifar10/cifar10-3072-euclidean.hdf5"),
    "svhn": str(ROOT / "datasets/svhn/svhn-3072-euclidean.hdf5"),
}

# Neighbour computation (brute-force L2 via numpy; no faiss needed)

def compute_neighbors_brute(train: np.ndarray, queries: np.ndarray,
                             top_k: int = TOP_K,
                             chunk: int = 500) -> np.ndarray:
    n_q   = len(queries)
    n_t   = len(train)
    train_sq = (train.astype(np.float64) ** 2).sum(axis=1).astype(np.float32)
    query_sq = (queries.astype(np.float64) ** 2).sum(axis=1).astype(np.float32)
    neighbors = np.zeros((n_q, top_k), dtype=np.int32)

    t0 = time.time()
    for s in range(0, n_q, chunk):
        e   = min(s + chunk, n_q)
        q   = queries[s:e].astype(np.float32)
        # (chunk, dim)
        dot = q @ train.T
        # (chunk, n_t)
        d2  = query_sq[s:e, None] + train_sq[None, :] - 2.0 * dot  # (chunk, n_t)

        # Partial sort: indices of top_k smallest per row
        idx = np.argpartition(d2, top_k, axis=1)[:, :top_k]
        for i in range(e - s):
            srt = idx[i][np.argsort(d2[i, idx[i]])]
            neighbors[s + i] = srt

        if (s // chunk) % 5 == 0:
            elapsed = time.time() - t0
            rate = (e / n_q) * 100
            eta = elapsed / max(e, 1) * (n_q - e)
            print(f"  [{e}/{n_q}]  {rate:.0f}%  elapsed={elapsed:.0f}s  ETA={eta:.0f}s")

    print(f"  Neighbour computation done in {time.time()-t0:.1f}s")
    return neighbors

# Image → flat float32 vector

def img_to_vec(img) -> np.ndarray:
    return np.array(img.convert("RGB"), dtype=np.float32).flatten()

# Dataset loaders

def load_cifar10():
    from datasets import load_dataset as hf_load
    print("  Downloading CIFAR-10 from HuggingFace …")
    ds_train = hf_load("uoft-cs/cifar10", split="train")
    ds_test  = hf_load("uoft-cs/cifar10", split="test")

    print("  Vectorising train …")
    train_X = np.array([img_to_vec(ex["img"]) for ex in ds_train], dtype=np.float32)
    train_y = np.array([ex["label"] for ex in ds_train], dtype=np.int32)
    print("  Vectorising test …")
    test_X  = np.array([img_to_vec(ex["img"]) for ex in ds_test],  dtype=np.float32)
    test_y  = np.array([ex["label"] for ex in ds_test],  dtype=np.int32)
    return train_X, train_y, test_X[:N_QUERY], test_y[:N_QUERY]

def load_svhn():
    from datasets import load_dataset as hf_load
    print("  Downloading SVHN (cropped_digits) from HuggingFace …")
    ds_train = hf_load("svhn", "cropped_digits", split="train")
    ds_test  = hf_load("svhn", "cropped_digits", split="test")

    print("  Vectorising train …")
    train_X = np.array([img_to_vec(ex["image"]) for ex in ds_train], dtype=np.float32)
    train_y = np.array([ex["label"] for ex in ds_train], dtype=np.int32)
    print("  Vectorising test …")
    test_X  = np.array([img_to_vec(ex["image"]) for ex in ds_test],  dtype=np.float32)
    test_y  = np.array([ex["label"] for ex in ds_test],  dtype=np.int32)
    return train_X, train_y, test_X[:N_QUERY], test_y[:N_QUERY]

LOADERS = {
    "cifar10": load_cifar10,
    "svhn":
    load_svhn,
}

# Build one dataset

def build_dataset(name: str) -> None:
    hdf5_path  = HDF5_PATHS[name]
    cache_path = PILLAR2 / f"extended_neighbors_{name}.npy"

    if os.path.exists(hdf5_path) and cache_path.exists():
        print(f"  {name}: HDF5 and neighbor cache already exist — skipping.")
        return

    os.makedirs(OUT_DIRS[name], exist_ok=True)
    PILLAR2.mkdir(parents=True, exist_ok=True)
    print(f"  Building: {name.upper()}")
    train_X, train_y, test_X, test_y = LOADERS[name]()
    print(f"  train={train_X.shape}  test={test_X.shape}")
    print(f"  classes: {len(np.unique(train_y))} train, {len(np.unique(test_y))} test")

    print(f"\n  Computing exact top-{TOP_K} neighbours (brute-force L2) …")
    nb200 = compute_neighbors_brute(train_X, test_X, top_k=TOP_K)

    print(f"\n  Saving HDF5 → {hdf5_path}")
    with h5py.File(hdf5_path, "w") as f:
        f.create_dataset("train",        data=train_X,
        compression="gzip")
        f.create_dataset("test",         data=test_X,
        compression="gzip")
        f.create_dataset("neighbors",
        data=nb200[:, :100], compression="gzip")
        f.create_dataset("labels_train", data=train_y)
        f.create_dataset("labels_test",  data=test_y)

    print(f"  Saving extended neighbor cache → {cache_path}")
    np.save(cache_path, nb200)

    print(f"  Done: {name.upper()}")

# Main

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+",
                    default=["cifar10", "svhn"],
                    choices=list(LOADERS.keys()))
    args = ap.parse_args()

    for name in args.datasets:
        build_dataset(name)

    print("\nAll datasets built. Run generate_synthetic_recall_data.py next.")

main()
