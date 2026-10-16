import os
import re
import sys
import json
import argparse
from collections import Counter
from typing import Optional


_KV_RE = re.compile(
    r"When k = (\d+), \(recall, ratio\) = \(([\d.]+), ([\d.]+)\)"
)

_QT_RE = re.compile(
    r"average query time.*?is[:\s]*([\d.]+)\s*ms",
    re.IGNORECASE,
)

_IT_RE = re.compile(
    r"The indexing time is.*?:\s*([0-9eE.+\-]+)\s*ms",
    re.IGNORECASE,
)

_BUILD_MEM_RE = re.compile(r"Peak Construction Memory \(RSS\):\s*([\d.]+)\s*MB", re.IGNORECASE)
_SEARCH_MEM_RE = re.compile(r"Search Phase Memory \(RSS\):\s*([\d.]+)\s*MB", re.IGNORECASE)
_INDEX_SIZE_RE = re.compile(r"Index Size \(structure\):\s*(\d+)", re.IGNORECASE)


def parse_hnsw(path: str) -> list[dict]:
    runs: list[dict] = []
    cur: Optional[dict] = None

    hdr = re.compile(
        r"Dataset=([\w-]+)\s*\|\s*M=(\d+)\s*\|\s*efC=(\d+)\s*\|\s*efS=(\d+)"
    )
    _DC_RE = re.compile(r"Average Distance Computations per query:\s*([\d.]+)", re.IGNORECASE)
    _DC_EXACT_RE = re.compile(r"Exact Distance Computations per query:\s*([\d.]+)", re.IGNORECASE)

    for line in open(path, encoding="utf-8", errors="replace"):
        m = hdr.search(line)
        if m:
            cur = dict(
                algo="HNSW",
                dataset=m.group(1),
                M=int(m.group(2)),
                efC=int(m.group(3)),
                efS=int(m.group(4)),
                label=f"M={m.group(2)} efS={m.group(4)}",
                points=[],
                query_ms=None,
                build_ms=None,
                dist_comps=None,
                build_mem=None,
                search_mem=None,
                index_bytes=None,
            )
            runs.append(cur)
            continue

        m = _KV_RE.search(line)
        if m and cur:
            cur["points"].append(
                dict(k=int(m.group(1)), recall=float(m.group(2)), ratio=float(m.group(3)))
            )
            continue

        m = _QT_RE.search(line)
        if m and cur and cur["query_ms"] is None:
            cur["query_ms"] = float(m.group(1))
            continue

        m = _IT_RE.search(line)
        if m and cur and cur["build_ms"] is None:
            cur["build_ms"] = float(m.group(1))
            continue

        m = _DC_EXACT_RE.search(line)
        if m and cur:
            cur["dist_comps"] = float(m.group(1))
            continue

        m = _DC_RE.search(line)
        if m and cur and cur["dist_comps"] is None:
            cur["dist_comps"] = float(m.group(1))

        m = _BUILD_MEM_RE.search(line)
        if m and cur and cur["build_mem"] is None:
            cur["build_mem"] = float(m.group(1))

        m = _SEARCH_MEM_RE.search(line)
        if m and cur and cur["search_mem"] is None:
            cur["search_mem"] = float(m.group(1))

        m = _INDEX_SIZE_RE.search(line)
        if m and cur and cur["index_bytes"] is None:
            cur["index_bytes"] = int(m.group(1))

    return [r for r in runs if r["points"]]

def parse_suco(path: str) -> list[dict]:
    runs: list[dict] = []
    cur: Optional[dict] = None

    hdr   = re.compile(
        r"CONFIG: DS=([\w-]+)\s*\|\s*SUB=([\w]+)\s*\|\s*CR=([\d.]+)\s*\|\s*CAND=([\d.]+)"
    )
    DC_RE = re.compile(r"Average Distance Computations per query:\s*([\d.]+)", re.IGNORECASE)

    for line in open(path, encoding="utf-8", errors="replace"):
        m = hdr.search(line)
        if m:
            cur = dict(
                algo="SuCo",
                dataset=m.group(1),
                sub=m.group(2),
                cr=float(m.group(3)),
                cand=float(m.group(4)),
                label=f"sub={m.group(2)} cand={m.group(4)}",
                points=[],
                query_ms=None,
                build_ms=None,
                dist_comps=None,
                build_mem=None,
                search_mem=None,
                index_bytes=None,
            )
            runs.append(cur)
            continue

        m = _KV_RE.search(line)
        if m and cur:
            cur["points"].append(
                dict(k=int(m.group(1)), recall=float(m.group(2)), ratio=float(m.group(3)))
            )
            continue

        m = _QT_RE.search(line)
        if m and cur and cur["query_ms"] is None:
            cur["query_ms"] = float(m.group(1))
            continue

        m = _IT_RE.search(line)
        if m and cur and cur["build_ms"] is None:
            cur["build_ms"] = float(m.group(1))
            continue

        m = DC_RE.search(line)
        if m and cur and cur["dist_comps"] is None:
            cur["dist_comps"] = float(m.group(1))
            continue

        m = _BUILD_MEM_RE.search(line)
        if m and cur and cur["build_mem"] is None:
            cur["build_mem"] = float(m.group(1))
            continue

        m = _SEARCH_MEM_RE.search(line)
        if m and cur and cur["search_mem"] is None:
            cur["search_mem"] = float(m.group(1))

        m = _INDEX_SIZE_RE.search(line)
        if m and cur and cur["index_bytes"] is None:
            cur["index_bytes"] = int(m.group(1))

    return [r for r in runs if r["points"]]

def parse_symqg(path: str) -> list[dict]:
    runs: list[dict] = []
    cur = None

    hdr    = re.compile(r"Dataset=([\w-]+)\s*\|\s*M=(\d+)\s*\|\s*efC=(\d+)\s*\|\s*efS=(\d+)")
    _DC_RE = re.compile(r"Average Distance Computations per query:\s*([\d.]+)", re.IGNORECASE)
    _DC_EXACT_RE = re.compile(r"Exact Distance Computations per query:\s*([\d.]+)", re.IGNORECASE)

    for line in open(path, encoding="utf-8", errors="replace"):
        m = hdr.search(line)
        if m:
            cur = dict(
                algo="SymphonyQG",
                dataset=m.group(1),
                M=int(m.group(2)),
                efC=int(m.group(3)),
                efS=int(m.group(4)),
                label=f"M={m.group(2)} efS={m.group(4)}",
                points=[],
                query_ms=None,
                build_ms=None,
                dist_comps=None,
                build_mem=None,
                search_mem=None,
                index_bytes=None,
            )
            runs.append(cur)
            continue

        m = _KV_RE.search(line)
        if m and cur:
            cur["points"].append(
                dict(k=int(m.group(1)), recall=float(m.group(2)), ratio=float(m.group(3)))
            )
            continue

        m = _QT_RE.search(line)
        if m and cur and cur["query_ms"] is None:
            cur["query_ms"] = float(m.group(1))
            continue

        m = _IT_RE.search(line)
        if m and cur and cur["build_ms"] is None:
            cur["build_ms"] = float(m.group(1))
            continue

        m = _DC_EXACT_RE.search(line)
        if m and cur:
            cur["dist_comps"] = float(m.group(1))
            continue

        m = _DC_RE.search(line)
        if m and cur and cur["dist_comps"] is None:
            cur["dist_comps"] = float(m.group(1))

        m = _BUILD_MEM_RE.search(line)
        if m and cur and cur["build_mem"] is None:
            cur["build_mem"] = float(m.group(1))

        m = _SEARCH_MEM_RE.search(line)
        if m and cur and cur["search_mem"] is None:
            cur["search_mem"] = float(m.group(1))

        m = _INDEX_SIZE_RE.search(line)
        if m and cur and cur["index_bytes"] is None:
            cur["index_bytes"] = int(m.group(1))

    return [r for r in runs if r["points"]]

def parse_rabitq(path: str) -> list[dict]:
    import csv

    _K_COLS = [
        (1,   "Recall@1",   "Ratio@1"),
        (10,  "Recall@10",  "Ratio@10"),
        (20,  "Recall@20",  "Ratio@20"),
        (50,  "Recall@50",  "Ratio@50"),
        (70,  "Recall@70",  "Ratio@70"),
        (100, "Recall@100", "Ratio@100"),
    ]

    runs: list[dict] = []
    with open(path, encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            dataset    = row["Dataset"].strip()
            clusters   = int(row["Clusters"])
            train_cfg  = row["TrainConfig"].strip()
            nprobe     = int(row["NProbe"])
            query_ms   = float(row["SearchTime(ms)"])
            build_ms   = float(row["TotalConstructionTime(ms)"])
            build_mem  = float(row["ConstructionMemory(MB)"]) if row.get("ConstructionMemory(MB)") else None
            search_mem = float(row["SearchMemory(MB)"]) if row.get("SearchMemory(MB)") else None

            points = [
                dict(k=k, recall=float(row[rc_col]) / 100.0, ratio=float(row[rt_col]))
                for k, rc_col, rt_col in _K_COLS
                if rc_col in row and rt_col in row
            ]

            if not points:
                continue

            dc_raw = row.get("DistanceComps", "").strip()
            dist_comps = float(dc_raw) if dc_raw else None

            runs.append(dict(
                algo="RaBitQ",
                dataset=dataset,
                clusters=clusters,
                train_config=train_cfg,
                nprobe=nprobe,
                label=f"C={clusters} TC={train_cfg} np={nprobe}",
                points=points,
                query_ms=query_ms,
                build_ms=build_ms,
                dist_comps=dist_comps,
                build_mem=build_mem,
                search_mem=search_mem,
            ))
    return runs

def parse_annoy(path: str) -> list[dict]:
    runs: list[dict] = []
    cur_dataset: str | None = None
    cur_num_trees: int | None = None
    cur_build_ms: float | None = None
    cur_index_bytes: int | None = None
    cur_block_runs: list[dict] = []
    cur_run: dict | None = None

    DS_RE    = re.compile(r"^Dataset:\s*([\w-]+)")
    TREES_RE = re.compile(r"Parameters:\s*NUM_TREES\s*=\s*(\d+)")
    SK_RE    = re.compile(r"Evaluating SEARCH_K\s*=\s*(\d+)")
    DC_RE    = re.compile(r"Average Distance Computations per query:\s*([\d.]+)", re.IGNORECASE)
    MEM_RE   = re.compile(r"Peak Memory:\s*([\d.]+)\s*MB", re.IGNORECASE)

    for line in open(path, encoding="utf-8", errors="replace"):
        m = DS_RE.match(line)
        if m:
            cur_dataset    = m.group(1)
            cur_num_trees  = None
            cur_build_ms   = None
            cur_index_bytes = None
            cur_block_runs = []
            cur_run        = None
            continue

        m = TREES_RE.search(line)
        if m:
            cur_num_trees  = int(m.group(1))
            cur_build_ms   = None
            cur_index_bytes = None
            cur_block_runs = []
            cur_run        = None
            continue

        m = _IT_RE.search(line)
        if m and cur_build_ms is None and cur_run is None:
            cur_build_ms = float(m.group(1))
            continue

        m = SK_RE.search(line)
        if m and cur_dataset and cur_num_trees is not None:
            sk = int(m.group(1))
            cur_run = dict(
                algo="Annoy",
                dataset=cur_dataset,
                num_trees=cur_num_trees,
                search_k=sk,
                label=f"nt={cur_num_trees} sk={sk}",
                points=[],
                query_ms=None,
                build_ms=cur_build_ms,
                dist_comps=None,
                search_mem=None,
                index_bytes=cur_index_bytes,
            )
            runs.append(cur_run)
            cur_block_runs.append(cur_run)
            continue

        m = DC_RE.search(line)
        if m and cur_run and cur_run["dist_comps"] is None:
            cur_run["dist_comps"] = float(m.group(1))
            continue

        m = _QT_RE.search(line)
        if m and cur_run and cur_run["query_ms"] is None:
            cur_run["query_ms"] = float(m.group(1))
            continue

        m = _KV_RE.search(line)
        if m and cur_run:
            cur_run["points"].append(
                dict(k=int(m.group(1)), recall=float(m.group(2)), ratio=float(m.group(3)))
            )
            continue

        m = _INDEX_SIZE_RE.search(line)
        if m and cur_dataset and cur_num_trees is not None:
            cur_index_bytes = int(m.group(1))
            for r in cur_block_runs:
                r["index_bytes"] = cur_index_bytes
            continue

        m = MEM_RE.search(line)
        if m:
            mem = float(m.group(1))
            for r in cur_block_runs:
                if r["search_mem"] is None:
                    r["search_mem"] = mem

    return [r for r in runs if r["points"]]

def load_runs(
    hnsw: Optional[str] = None,
    suco: Optional[str] = None,
    rabitq: Optional[str] = None,
    symqg: Optional[str] = None,
    annoy: Optional[str] = None,
    verbose: bool = True,
) -> list[dict]:
    all_runs: list[dict] = []

    if hnsw:
        parsed = parse_hnsw(hnsw)
        if verbose:
            print(f"[HNSW]  {len(parsed):>5} runs  ← {hnsw}")
        all_runs += parsed

    if suco:
        parsed = parse_suco(suco)
        if verbose:
            print(f"[SuCo]  {len(parsed):>5} runs  ← {suco}")
        all_runs += parsed



    if rabitq:
        parsed = parse_rabitq(rabitq)
        if verbose:
            print(f"[RaBitQ] {len(parsed):>5} runs  ← {rabitq}")
        all_runs += parsed

    if symqg:
        parsed = parse_symqg(symqg)
        if verbose:
            print(f"[SymphonyQG] {len(parsed):>5} runs  ← {symqg}")
        all_runs += parsed

    if annoy:
        parsed = parse_annoy(annoy)
        if verbose:
            print(f"[Annoy]  {len(parsed):>5} runs  ← {annoy}")
        all_runs += parsed

    if verbose and all_runs:
        datasets = sorted(set(r["dataset"] for r in all_runs))
        algos    = sorted(set(r["algo"]    for r in all_runs))
        print(f"[TOTAL] {len(all_runs)} runs | "
              f"{len(datasets)} datasets: {', '.join(datasets)} | "
              f"algos: {', '.join(algos)}")

    return all_runs



def _print_breakdown(runs: list[dict]) -> None:
    algos    = sorted(set(r["algo"]    for r in runs))
    datasets = sorted(set(r["dataset"] for r in runs))

    col_w = max(len(d) for d in datasets) + 2
    header = f"{'dataset':<{col_w}}" + "".join(f"{a:>8}" for a in algos) + "  total"
    print("\n" + header)
    print("-" * len(header))

    counts: dict[tuple, int] = Counter((r["algo"], r["dataset"]) for r in runs)
    for ds in datasets:
        row = f"{ds:<{col_w}}"
        total = 0
        for algo in algos:
            n = counts.get((algo, ds), 0)
            total += n
            row += f"{n:>8}"
        row += f"  {total}"
        print(row)

    print("-" * len(header))
    totals = f"{'TOTAL':<{col_w}}"
    gt = 0
    for algo in algos:
        n = sum(counts.get((algo, ds), 0) for ds in datasets)
        gt += n
        totals += f"{n:>8}"
    print(totals + f"  {gt}")

    missing_qms = sum(1 for r in runs if r["query_ms"] is None)
    if missing_qms:
        print(f"\nWARNING: {missing_qms} runs have no query_ms")
    else:
        print("\nAll runs have query_ms. ✓")

def _print_sample(runs: list[dict], algo: str, dataset: str, n: int = 3) -> None:
    subset = [r for r in runs if r["algo"] == algo and r["dataset"] == dataset][:n]
    if not subset:
        print(f"  No runs found for {algo} / {dataset}")
        return
    print(f"\nSample runs — {algo} / {dataset}:")
    for r in subset:
        print(f"  label     : {r['label']}")
        print(f"  query_ms  : {r['query_ms']}")

        for key in ("M", "efC", "efS", "sub", "cr", "cand"):
            if key in r:
                print(f"  {key:<10}: {r[key]}")
        print(f"  points    : {r['points']}")
        print()



class _Tee:
    def __init__(self, *streams):
        self._streams = streams

    def write(self, data: str) -> int:
        for s in self._streams:
            s.write(data)
        return len(data)

    def flush(self) -> None:
        for s in self._streams:
            s.flush()


    def __getattr__(self, name: str):
        return getattr(self._streams[0], name)

def setup_log(path: str) -> None:
    import atexit
    import sys

    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    log_file = open(path, "w", encoding="utf-8")
    sys.stdout = _Tee(sys.__stdout__, log_file)

    def _cleanup():
        sys.stdout = sys.__stdout__
        log_file.close()

    atexit.register(_cleanup)
    print(f"Terminal output will also be saved → {path}\n")

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Parse HNSW / SuCo benchmark logs and report run counts.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--hnsw",    metavar="FILE", help="Path to hnsw-example.txt")
    ap.add_argument("--suco",    metavar="FILE", help="Path to suco-example.txt")

    ap.add_argument("--dataset", metavar="DS",   help="Filter to one dataset")
    ap.add_argument("--algo",    metavar="ALGO",
                    choices=["HNSW", "SuCo"],
                    help="Filter to one algorithm")
    ap.add_argument("--sample",  action="store_true",
                    help="Print 3 sample runs per algo/dataset combination")
    ap.add_argument("--json",    metavar="FILE",
                    help="Dump all parsed runs to a JSON file")
    ap.add_argument("--log",   metavar="FILE", default="pillar-1/log_benchmarks.txt",
                    help="Save terminal output to FILE (default: pillar-1/log_benchmarks.txt)")
    ap.add_argument("--no-log", action="store_true", help="Disable terminal logging to file")
    args = ap.parse_args()

    if not args.no_log:
        setup_log(args.log)

    if not any([args.hnsw, args.suco]):
        ap.error("Provide at least one of --hnsw, --suco")

    runs = load_runs(args.hnsw, args.suco)
    if not runs:
        print("No runs parsed. Check file paths.")
        sys.exit(1)

    if args.dataset:
        runs = [r for r in runs if r["dataset"] == args.dataset]
        print(f"[FILTER] dataset={args.dataset} → {len(runs)} runs")
    if args.algo:
        runs = [r for r in runs if r["algo"] == args.algo]
        print(f"[FILTER] algo={args.algo} → {len(runs)} runs")
    if not runs:
        print("No runs after filtering.")
        sys.exit(1)

    _print_breakdown(runs)

    if args.sample:
        algos    = sorted(set(r["algo"]    for r in runs))
        datasets = sorted(set(r["dataset"] for r in runs))
        for algo in algos:
            for ds in datasets:
                _print_sample(runs, algo, ds, n=1)

    if args.json:
        with open(args.json, "w") as f:
            json.dump(runs, f, indent=2)
        print(f"\nDumped {len(runs)} runs → {args.json}")



main()
