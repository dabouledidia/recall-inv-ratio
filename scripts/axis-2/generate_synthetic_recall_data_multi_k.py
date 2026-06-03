
import argparse
import csv
import os
import sys
import time

import h5py
import numpy as np

NEAR_K   = 200   # extended search depth; positions k…NEAR_K are the near-miss pool

RECALL_LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0]
K_VALUES_DEFAULT = [1, 10, 20, 50, 100]

DATASET_CONFIGS = {
    "mnist": {
        "hdf5":         str(ROOT / "datasets/MNIST/mnist-784-euclidean.hdf5"),
        "label_source": "hf",
        "hf_name":      "mnist",
    },
    "fmnist": {
        "hdf5":         str(ROOT / "datasets/fmnist/fashion-mnist-784-euclidean.hdf5"),
        "label_source": "hf",
        "hf_name":      "fashion_mnist",
    },    "cifar10": {
        "hdf5":         str(ROOT / "datasets/cifar10/cifar10-3072-euclidean.hdf5"),
        "label_source": "hdf5",
    },
    "svhn": {
        "hdf5":         str(ROOT / "datasets/svhn/svhn-3072-euclidean.hdf5"),
        "label_source": "hdf5",
    },}

def load_dataset(name: str):
    cfg = DATASET_CONFIGS[name]
    with h5py.File(cfg["hdf5"], "r") as f:
        train     = f["train"][:]
        queries   = f["test"][:]
        neighbors = f["neighbors"][:]
        if cfg["label_source"] == "hdf5":
            train_labels = f["labels_train"][:].astype(np.int32)
            query_labels = f["labels_test"][:].astype(np.int32)
        else:
            train_labels = query_labels = None

    if cfg["label_source"] == "hf":
        try:
            from datasets import load_dataset as hf_load
        except ImportError:
            print("ERROR: pip install datasets"); sys.exit(1)
        hf_name = cfg["hf_name"]
        print(f"  Loading {hf_name} labels from HuggingFace …")
        ds_train = hf_load(hf_name, split="train")
        ds_test  = hf_load(hf_name, split="test")
        train_labels = np.array(ds_train["label"], dtype=np.int32)
        query_labels = np.array(ds_test["label"],  dtype=np.int32)

    return train, queries, neighbors, train_labels, query_labels

def load_extended_neighbors(name: str) -> np.ndarray:
    cache = PILLAR2 / f"extended_neighbors_{name}.npy"
    if not os.path.exists(cache):
        print(f"ERROR: cache not found: {cache}")
        print("  Run pillar2_semantic_stability.py first to build the cache.")
        sys.exit(1)
    nb = np.load(cache)
    print(f"  Loaded extended neighbors from cache: {cache}  shape={nb.shape}")
    return nb

def simulate_retrieval(nb200: np.ndarray, recall: float, k: int) -> np.ndarray:
    n_exact = round(recall * k)
    n_miss  = k - n_exact
    exact_part = nb200[:, :n_exact]
    miss_part  = nb200[:, k:k + n_miss]   # near-miss pool starts just past top-k
    return np.hstack([exact_part, miss_part])

def label_precision(retrieved: np.ndarray,
                    train_labels: np.ndarray,
                    query_labels: np.ndarray) -> float:
    n = len(retrieved)
    return float(sum(
        np.mean(train_labels[retrieved[q]] == query_labels[q])
        for q in range(n)
    ) / n)

def ratio_at_k(retrieved, ground_truth, train, queries, k=100, batch=500):
    n = len(queries)
    total = 0.0
    for i in range(0, n, batch):
        q   = queries[i:i+batch]
        ret = retrieved[i:i+batch]
        gt  = ground_truth[i:i+batch, :k]
        d_ret = np.linalg.norm(train[ret] - q[:, None, :], axis=2)
        d_gt  = np.linalg.norm(train[gt]  - q[:, None, :], axis=2)
        mask  = d_gt == 0
        total += np.where(mask, 1.0, d_ret / np.where(mask, 1.0, d_gt)).sum()
    return float(total / (n * k))

def precompute_d200(train: np.ndarray, queries: np.ndarray,
                    nb200: np.ndarray, batch: int = 100) -> np.ndarray:
    n, top = len(queries), nb200.shape[1]
    d200 = np.zeros((n, top), dtype=np.float32)
    t0 = time.time()
    for i in range(0, n, batch):
        q  = queries[i:i+batch].astype(np.float32)
        nb = nb200[i:i+batch]
        diff = train[nb].astype(np.float32) - q[:, None, :]
        d200[i:i+batch] = np.linalg.norm(diff, axis=2)
    print(f"  precompute_d200 done in {time.time()-t0:.1f}s")
    return d200

def fast_ratio_from_d200(d200: np.ndarray, recall: float, k: int) -> float:
    n_exact = round(recall * k)
    n_miss  = k - n_exact
    d_ret = np.concatenate([d200[:, :n_exact], d200[:, k:k + n_miss]], axis=1)
    d_gt  = d200[:, :k]
    mask  = d_gt == 0
    return float(np.where(mask, 1.0, d_ret / np.where(mask, 1.0, d_gt)).mean())

def run_dataset(name: str, k_values: list[int]) -> list[dict]:
    print(f"  Dataset: {name.upper()}")
    train, queries, nb100, train_labels, query_labels = load_dataset(name)
    nb200 = load_extended_neighbors(name)

    print("  Precomputing distances to all 200 neighbours …")
    d200 = precompute_d200(train, queries, nb200)

    rows = []
    for k in k_values:
        print(f"\n  --- k = {k} ---")
        exact_retrieved = nb200[:, :k]
        exact_lp    = label_precision(exact_retrieved, train_labels, query_labels)
        exact_ratio = fast_ratio_from_d200(d200, 1.0, k)
        print(f"    Exact LP@{k}={exact_lp:.4f}  Ratio={exact_ratio:.6f}")

        for r in RECALL_LEVELS:
            retrieved = simulate_retrieval(nb200, r, k)
            lp_val  = label_precision(retrieved, train_labels, query_labels)
            rat_val = fast_ratio_from_d200(d200, r, k)
            rows.append(dict(
                dataset=name,
                k=k,
                recall=r,
                label_precision=lp_val,
                lp_norm=lp_val / exact_lp,
                ratio=rat_val,
                ratio_norm=rat_val / exact_ratio,
                exact_lp=exact_lp,
                exact_ratio=exact_ratio,
            ))
            print(f"    recall={r:.2f}  LP={lp_val:.4f}  lp_norm={lp_val/exact_lp:.4f}"
                  f"  ratio={rat_val:.4f}")
    return rows

def save_csv(all_rows: list[dict], path: str) -> None:
    fields = ["dataset", "k", "recall", "label_precision", "lp_norm",
              "ratio", "ratio_norm", "exact_lp", "exact_ratio"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(all_rows)
    print(f"\nCSV saved → {path}  ({len(all_rows)} rows)")

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+",
                    default=["mnist", "fmnist", "cifar10", "svhn"],
                    choices=list(DATASET_CONFIGS.keys()))
    ap.add_argument("--k-values", nargs="+", type=int,
                    default=K_VALUES_DEFAULT, dest="k_values")
    args = ap.parse_args()

    PILLAR2.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict] = []
    for name in args.datasets:
        all_rows.extend(run_dataset(name, args.k_values))

    out = PILLAR2 / "semantic_stability_multi_k.csv"
    save_csv(all_rows, out)
    print("\nDone. Re-run fig8_semantic_quality_bars.py to update plots.")

main()
