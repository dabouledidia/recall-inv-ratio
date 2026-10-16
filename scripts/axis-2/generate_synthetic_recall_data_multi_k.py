import argparse
import csv
import time
from pathlib import Path

import h5py
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent.parent
PILLAR2 = SCRIPT_DIR

K_VALUES_DEFAULT = [1, 20, 50, 100]
SCENARIOS = {"default": 5, "worst-case": 10}
RECALL_LEVELS = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
SEED = 0

DATASET_CONFIGS = {
    "mnist":   {"hdf5": str(ROOT / "datasets/MNIST/mnist-784-euclidean.hdf5"),           "label_source": "hf",   "hf_name": "mnist"},
    "fmnist":  {"hdf5": str(ROOT / "datasets/fmnist/fashion-mnist-784-euclidean.hdf5"),   "label_source": "hf",   "hf_name": "fashion_mnist"},
    "cifar10": {"hdf5": str(ROOT / "datasets/cifar10/cifar10-3072-euclidean.hdf5"),       "label_source": "hdf5"},
    "svhn":    {"hdf5": str(ROOT / "datasets/svhn/svhn-3072-euclidean.hdf5"),             "label_source": "hdf5"},
}


def log(msg): print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_dataset(name):
    cfg = DATASET_CONFIGS[name]
    with h5py.File(cfg["hdf5"], "r") as f:
        train   = f["train"][:].astype(np.float32)
        queries = f["test"][:].astype(np.float32)
        if cfg["label_source"] == "hdf5":
            tr_lab = f["labels_train"][:].astype(np.int32)
            te_lab = f["labels_test"][:].astype(np.int32)
        else:
            tr_lab = te_lab = None
    if cfg["label_source"] == "hf":
        from datasets import load_dataset as hf_load
        log(f"  loading {cfg['hf_name']} labels from HuggingFace …")
        tr_lab = np.array(hf_load(cfg["hf_name"], split="train")["label"], dtype=np.int32)
        te_lab = np.array(hf_load(cfg["hf_name"], split="test")["label"],  dtype=np.int32)
    return train, queries, tr_lab, te_lab


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


def sweep_k(idx, dist, tr_lab, te_lab, k, W, rng):
    nq = idx.shape[0]
    exact_lp = (tr_lab[idx[:, :k]] == te_lab[:, None]).mean(1)
    rows = []
    for r in RECALL_LEVELS:
        n_gold = int(round(r * k)); n_fill = k - n_gold
        if n_fill == 0:
            lp = exact_lp; inv = np.ones(nq)
        else:
            gold = idx[:, :n_gold]
            ranks = rng.integers(k, W, size=(nq, n_fill))
            fill  = np.take_along_axis(idx, ranks, axis=1)
            ret   = np.concatenate([gold, fill], axis=1)
            lp = (tr_lab[ret] == te_lab[:, None]).mean(1)
            dfill = np.take_along_axis(dist, ranks, axis=1)
            dgt   = dist[:, n_gold:k]
            rj    = np.where(dgt > 0, dfill / dgt, 1.0)
            inv   = 1.0 / ((n_gold + rj.sum(1)) / k)
        inv_mean = float(inv.mean())
        rows.append(dict(k=k, recall=r,
                         label_precision=float(lp.mean()),
                         lp_norm=float(lp.mean() / exact_lp.mean()),
                         inv_ratio=inv_mean,
                         ratio=(1.0 / inv_mean if inv_mean else 1.0),
                         exact_lp=float(exact_lp.mean())))
    return rows


def run_dataset(name, k_values):
    log(f"Dataset: {name.upper()}")
    train, queries, tr_lab, te_lab = load_dataset(name)
    N = max(k_values) * max(SCENARIOS.values()) + 1
    log(f"  computing exact top-{N} neighbours …")
    idx, dist = exact_topN(train, queries, N)
    out = []
    for scenario, mult in SCENARIOS.items():
        rng = np.random.default_rng(SEED)
        for k in k_values:
            for row in sweep_k(idx, dist, tr_lab, te_lab, k, k * mult, rng):
                out.append(dict(dataset=name, scenario=scenario, window_N=k * mult, **row))
        log(f"  {scenario} (W = {mult}k)")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=list(DATASET_CONFIGS), choices=list(DATASET_CONFIGS))
    ap.add_argument("--k-values", nargs="+", type=int, default=K_VALUES_DEFAULT, dest="k_values")
    args = ap.parse_args()
    all_rows = []
    for name in args.datasets:
        all_rows.extend(run_dataset(name, args.k_values))
    out = PILLAR2 / "semantic_stability_multi_k.csv"
    fields = ["dataset", "scenario", "window_N", "k", "recall", "label_precision", "lp_norm", "inv_ratio", "ratio", "exact_lp"]
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(all_rows)
    log(f"CSV saved → {out}  ({len(all_rows)} rows)")



main()
