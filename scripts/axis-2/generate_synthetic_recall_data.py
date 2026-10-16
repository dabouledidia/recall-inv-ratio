import argparse
import csv
import sys
import time
from pathlib import Path

import h5py
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent.parent
PILLAR2 = SCRIPT_DIR

K        = 100
WINDOWS  = [500, 1000]
TOPN     = max(WINDOWS) + 1
SEED     = 0
RECALL_LEVELS = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

DATASET_CONFIGS = {
    "mnist":   {"title": "MNIST",         "hdf5": str(ROOT / "datasets/MNIST/mnist-784-euclidean.hdf5"),          "label_source": "hf",   "hf_name": "mnist"},
    "fmnist":  {"title": "Fashion-MNIST", "hdf5": str(ROOT / "datasets/fmnist/fashion-mnist-784-euclidean.hdf5"),  "label_source": "hf",   "hf_name": "fashion_mnist"},
    "cifar10": {"title": "CIFAR-10",      "hdf5": str(ROOT / "datasets/cifar10/cifar10-3072-euclidean.hdf5"),      "label_source": "hdf5"},
    "svhn":    {"title": "SVHN",          "hdf5": str(ROOT / "datasets/svhn/svhn-3072-euclidean.hdf5"),            "label_source": "hdf5"},
}


def log(msg): print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_dataset(name):
    cfg = DATASET_CONFIGS[name]
    with h5py.File(cfg["hdf5"], "r") as f:
        train   = f["train"][:].astype(np.float32)
        queries = f["test"][:].astype(np.float32)
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
        log(f"  loading {cfg['hf_name']} labels from HuggingFace …")
        train_labels = np.array(hf_load(cfg["hf_name"], split="train")["label"], dtype=np.int32)
        query_labels = np.array(hf_load(cfg["hf_name"], split="test")["label"],  dtype=np.int32)
    assert len(train_labels) == len(train) and len(query_labels) == len(queries), \
        f"{name}: label/vector count mismatch"
    return train, queries, train_labels, query_labels


def exact_topN(train, queries, N, batch=400):
    tn2 = (train * train).sum(1)
    idx_out  = np.empty((len(queries), N), np.int64)
    dist_out = np.empty((len(queries), N), np.float32)
    for s in range(0, len(queries), batch):
        qb = queries[s:s+batch]
        d2 = tn2[None, :] - 2.0 * (qb @ train.T) + (qb * qb).sum(1)[:, None]
        np.maximum(d2, 0, out=d2)
        part = np.argpartition(d2, N-1, axis=1)[:, :N]
        rows = np.arange(len(qb))[:, None]
        order = np.argsort(d2[rows, part], axis=1)
        idx_out[s:s+len(qb)]  = part[rows, order]
        dist_out[s:s+len(qb)] = np.sqrt(d2[rows, part][rows, order])
    return idx_out, dist_out


def sweep(idx, dist, tr_lab, te_lab, W, rng):
    nq = idx.shape[0]
    exact_lp = (tr_lab[idx[:, :K]] == te_lab[:, None]).mean(1)
    rows = []
    for r in RECALL_LEVELS:
        n_gold = int(round(r * K)); n_fill = K - n_gold
        if n_fill == 0:
            lp = exact_lp; inv = np.ones(nq)
        else:
            gold = idx[:, :n_gold]
            ranks = rng.integers(K, W, size=(nq, n_fill))
            fill  = np.take_along_axis(idx, ranks, axis=1)
            ret   = np.concatenate([gold, fill], axis=1)
            lp = (tr_lab[ret] == te_lab[:, None]).mean(1)
            dfill = np.take_along_axis(dist, ranks, axis=1)
            dgt   = dist[:, n_gold:K]
            rj    = np.where(dgt > 0, dfill / dgt, 1.0)
            inv   = 1.0 / ((n_gold + rj.sum(1)) / K)
        inv_mean = float(inv.mean())
        rows.append(dict(window_N=W, recall=r,
                         label_precision=float(lp.mean()),
                         lp_norm=float(lp.mean() / exact_lp.mean()),
                         inv_ratio=inv_mean,
                         ratio=(1.0 / inv_mean if inv_mean else 1.0),
                         exact_lp=float(exact_lp.mean())))
    return rows


def run_dataset(name):
    log(f"Dataset: {name.upper()}")
    train, queries, tr_lab, te_lab = load_dataset(name)
    log(f"  train={train.shape}  queries={queries.shape}")
    log(f"  computing exact top-{TOPN} neighbours …")
    idx, dist = exact_topN(train, queries, TOPN)
    out = []
    for W in WINDOWS:
        rng = np.random.default_rng(SEED)
        for row in sweep(idx, dist, tr_lab, te_lab, W, rng):
            out.append(dict(dataset=name, **row))
        r04 = next(x for x in out if x["window_N"] == W and abs(x["recall"] - 0.4) < 1e-9)
        log(f"  W={W}: at r=0.4  LP_norm={r04['lp_norm']:.3f}  1/Ratio={r04['inv_ratio']:.3f}")
    return out


def save_csv(all_rows, path):
    fields = ["dataset", "window_N", "recall", "label_precision", "lp_norm", "inv_ratio", "ratio", "exact_lp"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(all_rows)
    log(f"CSV saved → {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["mnist", "fmnist", "cifar10", "svhn"])
    args = ap.parse_args()
    all_rows = []
    for name in args.datasets:
        all_rows.extend(run_dataset(name))
    save_csv(all_rows, PILLAR2 / "semantic_stability.csv")



main()
