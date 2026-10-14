import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_PATH = SCRIPT_DIR / "semantic_stability_multi_k.csv"

DATASETS = ["mnist", "fmnist", "cifar10", "svhn"]
DATASET_LABELS = {"mnist": "MNIST", "fmnist": "Fashion-MNIST", "cifar10": "CIFAR-10", "svhn": "SVHN"}
SCENARIOS = ["default", "worst-case"]
K_VALUES = [1, 20, 50, 100]
RECALL_MIN = 0.4


def mae(df, ds, scenario, k, predictor):
    sub = df[(df.dataset == ds) & (df.scenario == scenario) & (df.k == k) & (df.recall >= RECALL_MIN - 1e-9)]
    if sub.empty:
        return None
    target = sub["lp_norm"].values
    pred = sub["recall"].values if predictor == "recall" else sub["inv_ratio"].values
    return float(np.mean(np.abs(pred - target)) * 100.0)


def main():
    df = pd.read_csv(DATA_PATH)
    if "scenario" not in df.columns:
        print("ERROR: regenerate semantic_stability_multi_k.csv with generate_synthetic_recall_data_multi_k.py")
        sys.exit(1)

    cw = 8
    print("Table 6  —  MAE of Recall and 1/Ratio from normalized Label Precision (%)")
    print()
    hdr = f"{'Dataset':<16}{'Predictor':<10}"
    for scen in SCENARIOS:
        for k in K_VALUES:
            hdr += f"{scen[:4]}/k{k}".rjust(cw)
    print(hdr)
    print("-" * len(hdr))
    for ds in DATASETS:
        if df[df.dataset == ds].empty:
            continue
        for predictor, lab in (("recall", "Recall"), ("inv_ratio", "1/Ratio")):
            row = f"{DATASET_LABELS[ds]:<16}{lab:<10}"
            for scen in SCENARIOS:
                for k in K_VALUES:
                    v = mae(df, ds, scen, k, predictor)
                    row += (f"{v:.2f}" if v is not None else "N/A").rjust(cw)
            print(row)
        print()


if __name__ == "__main__":
    main()
