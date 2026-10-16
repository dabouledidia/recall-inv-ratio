import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(ROOT))

from benchmarks import parse_hnsw, parse_rabitq, parse_symqg, parse_suco, parse_annoy

DATASETS = ["MNIST", "fmnist", "imagenet", "agnews", "simplewiki-openai", "gist"]
DS_LABEL = {
    "MNIST": "MNIST",
    "fmnist": "Fashion-MNIST",
    "imagenet": "ImageNet",
    "agnews": "AGNews",
    "simplewiki-openai": "SimpleWiki",
    "gist": "Gist",
}

ALGORITHMS = {
    "HNSW":       (ROOT / "data/hnsw-example.txt",       parse_hnsw),
    "SymphonyQG": (ROOT / "data/symphonyQG-example.txt", parse_symqg),
    "SuCo":       (ROOT / "data/suco-example.txt",       parse_suco),
    "RaBitQ":     (ROOT / "data/rabitq-example.txt",     parse_rabitq),
    "Annoy":      (ROOT / "data/annoy-example.txt",      parse_annoy),
}
TABLE_ALGOS = ["HNSW", "SymphonyQG", "SuCo"]
CHECK_ALGOS = ["RaBitQ", "Annoy"]

THRESHOLDS = [0.90, 0.95, 0.99]
K = 100


def load_all_runs() -> dict[str, list[dict]]:
    all_runs: dict[str, list[dict]] = {}
    for algo_name, (path, parser) in ALGORITHMS.items():
        if not path.exists():
            print(f"  [SKIP] {algo_name}: {path} not found")
            continue
        all_runs[algo_name] = parser(str(path))
        print(f"  {algo_name}: {len(all_runs[algo_name])} runs")
    return all_runs


def graph_key(r: dict) -> tuple:
    a = r["algo"]
    if a in ("HNSW", "SymphonyQG"):
        return (r.get("M"), r.get("efC"))
    if a == "SuCo":
        return (r.get("sub"),)
    if a == "RaBitQ":
        return (r.get("clusters"), r.get("train_config"))
    if a == "Annoy":
        return (r.get("num_trees"),)
    return (r.get("label"),)


def size_key(r: dict) -> tuple:
    if r["algo"] in ("HNSW", "SymphonyQG"):
        return (r.get("M"),)
    return graph_key(r)


def score_at_k(r: dict, metric: str) -> float | None:
    for p in r["points"]:
        if p["k"] == K:
            return p["recall"] if metric == "recall" else 1.0 / p["ratio"]
    return None


def select_config(runs: list[dict], dataset: str, metric: str, T: float) -> dict | None:
    rows = [r for r in runs if r["dataset"] == dataset and r.get("build_ms")
            and r.get("query_ms") and r["query_ms"] > 0 and score_at_k(r, "recall") is not None]
    builds: dict[tuple, list[float]] = {}
    for r in rows:
        builds.setdefault(graph_key(r), []).append(r["build_ms"])
    cost = {g: float(np.mean(v)) for g, v in builds.items()}
    cand = [r for r in rows if score_at_k(r, metric) >= T]
    if not cand:
        return None
    best = min(cost[graph_key(r)] for r in cand)
    tied = [r for r in cand if cost[graph_key(r)] == best]
    return max(tied, key=lambda r: 1000.0 / r["query_ms"])


def index_size(runs: list[dict], sel: dict) -> int | None:
    for r in runs:
        if r["dataset"] == sel["dataset"] and size_key(r) == size_key(sel) and r.get("index_bytes"):
            return r["index_bytes"]
    return None


def footprint_ratio(runs: list[dict], dataset: str, T: float):
    rec = select_config(runs, dataset, "recall", T)
    inv = select_config(runs, dataset, "inv_ratio", T)
    if rec is None or inv is None:
        return "N/A"
    if size_key(rec) == size_key(inv):
        return 1.0
    a, b = index_size(runs, rec), index_size(runs, inv)
    if a is None or b is None:
        return "no size"
    return a / b


def fmt(v) -> str:
    return f"{v:.2f}x" if isinstance(v, float) else v


def print_table5(all_runs: dict[str, list[dict]]) -> None:
    algos = [a for a in TABLE_ALGOS if a in all_runs]
    print(f"\nTable 5  --  Index footprint ratio at k={K}")
    print("  size of the index selected for Recall@100 >= T / size of the one selected for")
    print("  1/Ratio@100 >= T; index size = 'Index Size (structure)', i.e. without the base vectors;")
    print("  selection = cheapest-to-build index reaching T (as in Table 4).")
    print("  N/A = a target is unreachable; 'no size' = the two targets select differently sized")
    print("  indexes but a log lacks the 'Index Size (structure)' line.\n")
    cw = 9
    hdr = f"{'Dataset':15s}" + "".join(f"{a:^{cw * 3}s}" for a in algos)
    sub = f"{'':15s}" + "".join(f"{f'T={T:.2f}':>{cw}s}" for _ in algos for T in THRESHOLDS)
    print(hdr)
    print(sub)
    print("-" * len(sub))
    for ds in DATASETS:
        row = f"{DS_LABEL[ds]:15s}"
        for a in algos:
            for T in THRESHOLDS:
                row += f"{fmt(footprint_ratio(all_runs[a], ds, T)):>{cw}s}"
        print(row)
    print()
    for a in CHECK_ALGOS:
        if a not in all_runs:
            continue
        cells = [footprint_ratio(all_runs[a], ds, T) for ds in DATASETS for T in THRESHOLDS]
        reach = [c for c in cells if c != "N/A"]
        same = sum(1 for c in reach if c == 1.0)
        print(f"  {a}: same index under both targets in {same} of {len(reach)} reachable cases "
              f"(omitted from Table 5 when all are the same)")


def main() -> None:
    print("Loading runs...")
    all_runs = load_all_runs()
    if not all_runs:
        print("ERROR: no runs loaded.")
        return
    print_table5(all_runs)


main()
