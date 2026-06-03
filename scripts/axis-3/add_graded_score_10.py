import json
import re
from pathlib import Path
from tqdm import tqdm
import ollama

BASE = Path(__file__).parent
LLM_MODEL = "llama3.1:8b"

def get_grade_10(q: str, gold: str, test: str) -> int:
    if test == gold:
        return 10
    prompt = (
        f"Question: {q}\n"
        f"Gold Answer: {gold}\n"
        f"Test Answer: {test}\n\n"
        "Rate the factual consistency of the Test Answer against the Gold Answer on a scale of 1 to 10, where:\n"
        "1 = Completely contradicts or fails to mention the main facts.\n"
        "5 = Partially consistent, missing some key details but not severely hallucinating.\n"
        "10 = Perfectly consistent factual conclusion.\n"
        "Respond with ONLY the integer score (e.g., 1, 4, 7, 10)."
    )
    for attempt in range(3):
        try:
            resp = ollama.generate(model=LLM_MODEL, prompt=prompt, options={"temperature": 0})["response"].strip()
            nums = re.findall(r'\b(?:10|[1-9])\b', resp)
            if nums:
                return int(nums[0])
            return 5
        except Exception:
            pass
    return 5

dirs = [
    BASE / "scifact",
    BASE / "nfcorpus",
    BASE / "hotpotqa",
    BASE / "msmarco",
    BASE / "pubmedqa",
]

for d in dirs:
    results_path = d / "results" / "rag_results.json"
    if not results_path.exists():
        print(f"Skipping {d}, no results file.")
        continue
    
    print(f"Processing {d}...")
    with open(results_path) as f:
        results = json.load(f)
        
    changed = False
    for entry in tqdm(results):
        if entry.get("skipped"):
            continue
        q = entry["query"]
        gold = entry["gold_answer"]
        for sw in entry["sweeps"]:
            if "grade_self_10" not in sw:
                sw["grade_self_10"] = get_grade_10(q, gold, sw["answer"])
                changed = True
                
    if changed:
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Updated {results_path}")
    else:
        print("No updates needed.")
