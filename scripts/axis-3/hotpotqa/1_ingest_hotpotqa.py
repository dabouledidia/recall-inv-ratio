import hashlib, json, sys
from pathlib import Path

import numpy as np
DATA_DIR = Path(__file__).parent / "data"
CORPUS_JSONL  = DATA_DIR / "corpus.jsonl"
QUERIES_JSONL = DATA_DIR / "queries.jsonl"
CORPUS_NPY = DATA_DIR / "documents.npy"
QUERIES_NPY   = DATA_DIR / "queries.npy"
QRELS_JSON = DATA_DIR / "qrels.json"

EMBED_MODEL = "all-MiniLM-L6-v2"
BATCH_SIZE  = 128

# Use the validation split (7,405 questions) - held-out, no label leakage.
HF_DATASET  = "hotpot_qa"
HF_CONFIG   = "distractor"
HF_SPLIT = "validation"

def title_to_doc_id(title: str) -> str:
    return "hpqa_" + hashlib.md5(title.encode()).hexdigest()[:16]

def build_data() -> None:
    try:
        from datasets import load_dataset
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("ERROR: pip install datasets sentence-transformers")
        sys.exit(1)

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading HotpotQA ({HF_CONFIG}, {HF_SPLIT}) from HuggingFace…")
    ds = load_dataset(HF_DATASET, HF_CONFIG, split=HF_SPLIT)
    print(f"  {len(ds):,} questions loaded")

    print("Building corpus (deduplicating by Wikipedia title)…")
    docs: dict[str, dict] = {}
    # title: {id, title, text}
    for item in ds:
        titles = item["context"]["title"]
        sentences = item["context"]["sentences"]
        for title, sents in zip(titles, sentences):
            if title not in docs:
                docs[title] = {
                    "id":
                    title_to_doc_id(title),
                    "title": title,
                    "text":  " ".join(sents),
                }

    doc_list   = list(docs.values())
    title_to_id = {d["title"]: d["id"] for d in doc_list}
    print(f"  {len(doc_list):,} unique Wikipedia paragraphs")

    doc_texts = [f"{d['title']} {d['text']}".strip() for d in doc_list]

    with open(CORPUS_JSONL, "w") as f:
        for d in doc_list:
            f.write(json.dumps(d) + "\n")
    print(f"  Saved → {CORPUS_JSONL}")
    print("Building queries and qrels…")
    queries: list[dict] = []
    query_texts: list[str] = []
    qrels: dict[str, list[str]] = {}
    skipped_no_supporting = 0

    for item in ds:
        qid = str(item["id"])
        answer = item["answer"].strip()

        # Supporting fact titles, doc IDs
        sup_titles = set(item["supporting_facts"]["title"])
        sup_doc_ids = [
            title_to_id[t] for t in sup_titles if t in title_to_id
        ]
        if not sup_doc_ids:
            # All supporting titles were absent from context - skip.
            skipped_no_supporting += 1
            continue

        queries.append({
            "id":
            qid,
            "text":
            item["question"],
            "gold_human_answer": answer,
        })
        query_texts.append(item["question"])
        qrels[qid] = sup_doc_ids

    if skipped_no_supporting:
        print(f"  Skipped {skipped_no_supporting} queries with unresolvable "
              f"supporting facts")

    print(f"  {len(queries):,} queries with valid qrels")

    with open(QUERIES_JSONL, "w") as f:
        for q in queries:
            f.write(json.dumps(q) + "\n")
    print(f"  Saved → {QUERIES_JSONL}")

    with open(QRELS_JSON, "w") as f:
        json.dump(qrels, f, indent=2)
    print(f"  Saved → {QRELS_JSON}")
    print(f"\nLoading embedding model: {EMBED_MODEL}…")
    model = SentenceTransformer(EMBED_MODEL)

    print(f"Embedding {len(doc_texts):,} corpus documents…")
    doc_embs = model.encode(
        doc_texts, batch_size=BATCH_SIZE,
        show_progress_bar=True, convert_to_numpy=True,
    )
    np.save(CORPUS_NPY, doc_embs.astype(np.float32))
    print(f"  Saved → {CORPUS_NPY}  shape={doc_embs.shape}")

    print(f"Embedding {len(query_texts):,} queries…")
    q_embs = model.encode(
        query_texts, batch_size=BATCH_SIZE,
        show_progress_bar=True, convert_to_numpy=True,
    )
    np.save(QUERIES_NPY, q_embs.astype(np.float32))
    print(f"  Saved → {QUERIES_NPY}  shape={q_embs.shape}")

    print(f"\nDone.  All artefacts written to {DATA_DIR}/")
    print(f"  Corpus  : {len(doc_list):,} docs")
    print(f"  Queries : {len(queries):,}  (with gold_human_answer)")
    print(f"  Qrels   : {len(qrels):,} query → supporting-doc mappings")

build_data()
