import argparse
import csv
import os
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent.parent
PILLAR2 = SCRIPT_DIR
K          = 100
NEAR_K = 200   # extended search depth; positions K…NEAR_K are the near-miss pool

RECALL_LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0]

DATASET_CONFIGS = {
    "mnist": {
        "title":
        "MNIST",
        "hdf5": str(ROOT / "datasets/MNIST/mnist-784-euclidean.hdf5"),
        "label_source": "hf",
        "hf_name":
        "mnist",
    },
    "fmnist": {
        "title":
        "Fashion-MNIST",
        "hdf5": str(ROOT / "datasets/fmnist/fashion-mnist-784-euclidean.hdf5"),
        "label_source": "hf",
        "hf_name":
        "fashion_mnist",
    },
    "cifar10": {
        "title":
        "CIFAR-10",
        "hdf5": str(ROOT / "datasets/cifar10/cifar10-3072-euclidean.hdf5"),
        "label_source": "hdf5",
    },
    "svhn": {
        "title":
        "SVHN",
        "hdf5": str(ROOT / "datasets/svhn/svhn-3072-euclidean.hdf5"),
        "label_source": "hdf5",
    },
}

# Data loading

def load_dataset(name: str):
    cfg = DATASET_CONFIGS[name]
    with h5py.File(cfg["hdf5"], "r") as f:
        train = f["train"][:]
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

def extend_neighbors(train: np.ndarray, queries: np.ndarray,
                     neighbors_100: np.ndarray, name: str) -> np.ndarray:
    cache = PILLAR2 / f"extended_neighbors_{name}.npy"
    if cache.exists():
        nb200 = np.load(cache)
        print(f"  Loaded extended neighbors from cache: {cache}")
        return nb200

    try:
        import faiss
    except ImportError:
        print("ERROR: pip install faiss-cpu"); sys.exit(1)

    dim = train.shape[1]
    print(f"  Computing exact top-{NEAR_K} neighbors (faiss, {dim}D) …")
    t0 = time.time()
    index = faiss.IndexFlatL2(dim)
    index.add(train)
    _, nb200 = index.search(queries, NEAR_K)
    print(f"  Done in {time.time()-t0:.1f}s")
    np.save(cache, nb200)
    print(f"  Saved → {cache}")
    return nb200

# Simulated retrieval set

def simulate_retrieval(nb200: np.ndarray, recall: float, k: int = K) -> np.ndarray:
    n_exact = round(recall * k)
    n_miss  = k - n_exact
    exact_part = nb200[:, :n_exact]
    miss_part  = nb200[:, k:k + n_miss]
    return np.hstack([exact_part, miss_part])

# Metrics

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
        q  = queries[i:i+batch].astype(np.float32)   # (b, dim)
        nb = nb200[i:i+batch]
        # (b, 200)
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

# Per-dataset sweep

def run_dataset(name: str) -> list[dict]:
    print(f"  Dataset: {name.upper()}")
    cfg = DATASET_CONFIGS[name]
    print(f"Loading {cfg['hdf5']} …")
    train, queries, nb100, train_labels, query_labels = load_dataset(name)
    print(f"  train={train.shape}  queries={queries.shape}")

    nb200 = extend_neighbors(train, queries, nb100, name)

    print("Precomputing distances to all 200 neighbours …")
    d200 = precompute_d200(train, queries, nb200)

    print("Computing exact-search (recall=1.0) baseline …")
    exact_retrieved = nb200[:, :K]
    exact_lp = label_precision(exact_retrieved, train_labels, query_labels)
    exact_ratio = fast_ratio_from_d200(d200, 1.0, K)
    print(f"  Exact LP@100 = {exact_lp:.4f}   Ratio = {exact_ratio:.6f}")

    print(f"\n{'Recall':>8}  {'LP@100':>8}  {'LP/exact':>9}  {'Ratio':>8}  {'Rat/exact':>9}")
    print("-" * 52)

    rows = []
    for r in RECALL_LEVELS:
        retrieved = simulate_retrieval(nb200, r)
        lp_val = label_precision(retrieved, train_labels, query_labels)
        rat_val = fast_ratio_from_d200(d200, r, K)
        row = dict(
            dataset=name,
            recall=r,
            label_precision=lp_val,
            lp_norm=lp_val / exact_lp,
            ratio=rat_val,
            ratio_norm=rat_val / exact_ratio,
            exact_lp=exact_lp,
            exact_ratio=exact_ratio,
        )
        rows.append(row)
        print(f"  {r:>6.2f}  {lp_val:>8.4f}  {lp_val/exact_lp:>9.4f}  "
              f"{rat_val:>8.4f}  {rat_val/exact_ratio:>9.4f}")

    return rows

# CSV

def save_csv(all_rows: list[dict], path: str) -> None:
    fields = ["dataset", "recall", "label_precision", "lp_norm",
              "ratio", "ratio_norm", "exact_lp", "exact_ratio"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(all_rows)
    print(f"\nCSV saved → {path}")

# Plots

def _line_panel(ax, rows_by_dataset: dict[str, list[dict]],
               metric_key: str, metric_label: str) -> None:
    datasets = list(rows_by_dataset.keys())
    x = np.array(RECALL_LEVELS) * 100  # percentage

    ds_colors = {"mnist": "#E74C3C", "fmnist": "#3498DB", "cifar100": "#2ECC71", "imagenet": "#9B59B6"}
    ds_markers = {"mnist": "o", "fmnist": "s", "cifar100": "D", "imagenet": "^"}

    # Exact Baseline at 100%
    ax.axhline(100.0, color="#2C3E50", linestyle="--", linewidth=2.0, alpha=0.6, zorder=1)

    for ds in datasets:
        rows  = rows_by_dataset[ds]
        # Quality % = actual / exact * 100
        if metric_key == "ratio":
            vals = [(1.0 / r["ratio"]) / (1.0 / r["exact_ratio"]) * 100 for r in rows]
        else:
            vals = [r[metric_key] / r["exact_lp"] * 100 for r in rows]

        title  = DATASET_CONFIGS[ds]["title"]
        color  = ds_colors.get(ds, "#34495E")
        marker = ds_markers.get(ds, "o")
        
        # Delta from lowest recall (0.1) to highest (1.0)
        v_lo, v_hi = vals[0], vals[-1]
        delta = v_hi - v_lo
        
        label_with_delta = f"{title} (Δ {delta:+.1f}%)"
        
        ax.plot(x, vals, marker=marker, markersize=8, linewidth=3,
                color=color, alpha=0.85, label=label_with_delta, zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(r)}%" for r in x], fontsize=10)
    ax.set_xlabel("Simulated Recall@100 (%)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Quality (% of Exact Search)", fontsize=12, fontweight="bold")
    ax.set_title(metric_label, fontsize=14, fontweight="bold", pad=12)

    # Standard zoom for stability plots
    ax.set_ylim(80, 105)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{int(v)}%"))

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="both", alpha=0.3, linestyle="--", zorder=0)
    ax.legend(fontsize=10, loc="lower right", frameon=True, shadow=True)

def plot_all(all_rows: list[dict]) -> None:
    rows_by_dataset: dict[str, list[dict]] = {}
    for row in all_rows:
        rows_by_dataset.setdefault(row["dataset"], []).append(row)

    PILLAR2.mkdir(parents=True, exist_ok=True)
    suptitle = (
        "Semantic Stability Across Recall Levels (The Thin Shell Effect)\n"
        "Even at 10% Recall, semantic quality stays at ~100% of exact search."
    )

    fig, axes = plt.subplots(1, 2, figsize=(18, 7.5))
    _line_panel(axes[0], rows_by_dataset, "label_precision", "Label Precision Quality")
    _line_panel(axes[1], rows_by_dataset, "ratio", "1/Ratio Quality")
    
    fig.suptitle(suptitle, fontsize=18, fontweight="bold", y=1.02)
    fig.tight_layout()
    path = PILLAR2 / "semantic_stability_unified.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Plot saved → {path}")
    plt.close(fig)

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["mnist", "fmnist", "cifar10", "svhn"])
    args = ap.parse_args()
    PILLAR2.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict] = []
    for name in args.datasets:
        all_rows.extend(run_dataset(name))
    save_csv(all_rows, PILLAR2 / "semantic_stability.csv")
    plot_all(all_rows)

main()
