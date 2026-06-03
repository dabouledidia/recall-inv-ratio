import sys
import numpy as np
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(ROOT))

from benchmarks import parse_hnsw, parse_rabitq, parse_symqg, parse_suco, parse_annoy

DATASETS_EFF = ["MNIST", "fmnist", "agnews", "simplewiki-openai", "gist", "imagenet"]
DS_LABEL = {
    "MNIST":
    "MNIST",
    "fmnist":
    "Fashion-MNIST",
    "agnews":
    "AG News",
    "simplewiki-openai":  "SimpleWiki",
    "gist":
    "GIST",
    "imagenet":
    "ImageNet",
}
DS_LID = {
    "MNIST": 12.41, "fmnist": 13.99, "imagenet": 17.85,
    "agnews": 24.09, "simplewiki-openai": 27.45, "gist": 32.35,
}
DATASETS_BY_LID = sorted(DATASETS_EFF, key=lambda d: DS_LID[d])

K    = 100
K_LO = 1
K_HI = 100
THRESHOLDS = [0.90, 0.95, 0.99]

def load_runs() -> dict[str, list[dict]]:
    specs = {
        "HNSW":       (ROOT / "data/hnsw-example.txt",       parse_hnsw),
        "SymphonyQG": (ROOT / "data/symphonyQG-example.txt", parse_symqg),
        "SuCo":       (ROOT / "data/suco-example.txt",       parse_suco),
        "Annoy":      (ROOT / "data/annoy-example.txt",      parse_annoy),
    }
    runs = {}
    for name, (path, parser) in specs.items():
        if path.exists():
            runs[name] = parser(str(path))
            print(f"  {name}: {len(runs[name])} runs")
        else:
            print(f"  [SKIP] {name}: {path} not found")
    return runs

def min_build(runs, dataset, algo, k, threshold, metric):
    best = None
    for r in runs:
        if r["dataset"] != dataset or r["algo"] != algo:
            continue
        bms = r.get("build_ms")
        if not bms or bms <= 0:
            continue
        for pt in r["points"]:
            if pt["k"] != k:
                continue
            score = pt["recall"] if metric == "recall" else 1.0 / pt["ratio"]
            if score >= threshold:
                if best is None or bms < best:
                    best = bms
    return best

def print_table3(all_runs: dict[str, list[dict]]) -> None:
    algos = [a for a in ["HNSW", "SymphonyQG", "SuCo", "Annoy"] if a in all_runs]

    def cell(algo, ds, T):
        rc = min_build(all_runs[algo], ds, algo, K, T, "recall")
        ir = min_build(all_runs[algo], ds, algo, K, T, "inv_ratio")
        if rc is None or ir is None or ir <= 0:
            return "N/A"
        return f"{rc/ir:.2f}×"

    col_w = 8
    T_headers = [f"T={T:.2f}" for T in THRESHOLDS]

    print("Table 3  —  Min build time ratio (Recall / 1/Ratio) at k=100")
    print("N/A = Recall threshold unreachable at any configuration.")
    print("RaBitQ omitted — achieves exactly 1.00× across all datasets and thresholds.")
    print()

    # header
    hdr = f"{'Dataset':<17}"
    for algo in algos:
        hdr += f"  {algo:^{col_w*3+4}}"
    print(hdr)
    sub = f"{'':17}"
    for algo in algos:
        for th in T_headers:
            sub += f"  {th:>{col_w}}"
    print(sub)
    print("-" * (17 + len(algos) * (col_w * 3 + 6)))

    for ds in DATASETS_BY_LID:
        row = f"{DS_LABEL[ds]:<17}"
        for algo in algos:
            for T in THRESHOLDS:
                v = cell(algo, ds, T)
                row += f"  {v:>{col_w}}"
        print(row)
    print()

MIN_SCORE = 0.10  # if the best recall@k_hi across all runs < this, the algo has failed on the dataset

def algo_failed(runs, dataset, algo, k_hi) -> bool:
    for r in runs:
        if r["dataset"] != dataset or r["algo"] != algo:
            continue
        for pt in r["points"]:
            if pt["k"] == k_hi and pt["recall"] >= MIN_SCORE:
                return False
    return True

def degradation_per_run(runs, dataset, algo, k_lo, k_hi, metric):
    deltas = []
    for r in runs:
        if r["dataset"] != dataset or r["algo"] != algo:
            continue
        score_lo = score_hi = None
        for pt in r["points"]:
            s = pt["recall"] if metric == "recall" else 1.0 / pt["ratio"]
            if pt["k"] == k_lo:
                score_lo = s
            if pt["k"] == k_hi:
                score_hi = s
        if score_lo is not None and score_hi is not None:
            deltas.append(score_lo - score_hi)
    return deltas

def print_table6(all_runs: dict[str, list[dict]]) -> None:
    algos_rabitq = {}
    rabitq_path = ROOT / "data/rabitq-recall-ratio.txt"
    if rabitq_path.exists():
        algos_rabitq["RaBitQ"] = parse_rabitq(str(rabitq_path))

    all_runs_t6 = {**algos_rabitq, **all_runs}
    algos = [a for a in ["HNSW", "RaBitQ", "SymphonyQG", "SuCo", "Annoy"] if a in all_runs_t6]

    def cell(algo, ds):
        runs = all_runs_t6[algo]
        if algo_failed(runs, ds, algo, K_HI):
            return None, None
        rec_d = degradation_per_run(runs, ds, algo, K_LO, K_HI, "recall")
        rat_d = degradation_per_run(runs, ds, algo, K_LO, K_HI, "inv_ratio")
        if not rec_d or not rat_d:
            return None, None
        return np.median(rec_d) * 100, np.median(rat_d) * 100

    def fmt(v):
        if v is None:
            return "N/A"
        return f"{'+' if v >= 0 else ''}{v:.2f}"

    print("Table 6  —  Median degradation of Recall and 1/Ratio (k = 1 → 100)")
    print("Positive values = metric worsened (percentage points). Datasets sorted by LID.")
    print()

    col_w = 7
    hdr = f"{'Dataset':<22}"
    for algo in algos:
        hdr += f"  {algo:^{col_w*2+3}}"
    print(hdr)
    sub = f"{'':22}"
    for algo in algos:
        sub += f"  {'ΔRec':>{col_w}}  {'ΔRat':>{col_w}}"
    print(sub)
    print("-" * (22 + len(algos) * (col_w * 2 + 6)))

    for ds in DATASETS_BY_LID:
        row = f"{DS_LABEL[ds]} (LID={DS_LID[ds]:.2f})"
        row = f"{row:<22}"
        for algo in algos:
            rec, rat = cell(algo, ds)
            row += f"  {fmt(rec):>{col_w}}  {fmt(rat):>{col_w}}"
        print(row)

    print()
    print("N/A reasons:")
    print("  SymphonyQG SimpleWiki — quantization failure at 3072-d (~0.4% recall).")
    print("  Annoy ImageNet — algorithm fails completely (max recall ~3.5%); values would be noise.")
    print()

def main() -> None:
    print("Loading runs…")
    all_runs = load_runs()
    print()

    print_table3(all_runs)
    print_table6(all_runs)

main()
