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

HF_DATASET  = "ms_marco"
HF_CONFIG   = "v1.1"
HF_SPLIT = "validation"

NO_ANSWER   = "No Answer Present."

def passage_to_doc_id(text: str) -> str:
    return "msm_" + hashlib.md5(text.encode()).hexdigest()[:16]

def extract_gold_answer(answers: list[str]) -> str | None:
    valid = [a.strip() for a in answers if a.strip() and a.strip() != NO_ANSWER]
    return " ".join(valid) if valid else None

def build_data() -> None:
    try:
        from datasets import load_dataset
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("ERROR: pip install datasets sentence-transformers")
        sys.exit(1)

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading MS-MARCO ({HF_CONFIG}, {HF_SPLIT}) from HuggingFace…")
    ds = load_dataset(HF_DATASET, HF_CONFIG, split=HF_SPLIT)
    print(f"  {len(ds):,} examples loaded")
    # We need two passes: first collect all passages so the corpus is complete
    # before writing, then write queries/qrels referencing those IDs.
    print("Scanning passages and filtering queries…")

    passage_registry: dict[str, str] = {}
    # text: doc_id   (dedup store)

    # Per-query data accumulated for the valid queries only.
    valid_queries: list[dict] = []
    valid_qrels:   dict[str, list[str]] = {}

    skipped_no_answer   = 0
    skipped_no_selected = 0

    for item in ds:
        qid = str(item["query_id"])
        texts  = item["passages"]["passage_text"]
        flags  = item["passages"]["is_selected"]
        answers = item["answers"]

        gold = extract_gold_answer(answers)
        if gold is None:
            skipped_no_answer += 1
            continue

        # Collect all selected passage IDs for this query.
        selected_ids: list[str] = []
        for text, flag in zip(texts, flags):
            doc_id = passage_to_doc_id(text)
            if doc_id not in passage_registry:
                passage_registry[doc_id] = text
            if flag == 1:
                selected_ids.append(doc_id)

        if not selected_ids:
            skipped_no_selected += 1
            continue

        valid_queries.append({
            "id":
            qid,
            "text":
            item["query"],
            "gold_human_answer": gold,
        })
        valid_qrels[qid] = selected_ids

    print(f"  Skipped {skipped_no_answer} queries with no human answer")
    print(f"  Skipped {skipped_no_selected} queries with no selected passage")
    print(f"  {len(valid_queries):,} valid queries kept")
    print(f"  {len(passage_registry):,} unique passages in corpus")
    print("Writing corpus.jsonl…")
    doc_list  = [{"id": did, "title": "", "text": text}
                 for did, text in passage_registry.items()]
    doc_texts = [d["text"] for d in doc_list]

    with open(CORPUS_JSONL, "w") as f:
        for d in doc_list:
            f.write(json.dumps(d) + "\n")
    print(f"  Saved → {CORPUS_JSONL}")
    query_texts = [q["text"] for q in valid_queries]
    with open(QUERIES_JSONL, "w") as f:
        for q in valid_queries:
            f.write(json.dumps(q) + "\n")
    print(f"  Saved → {QUERIES_JSONL}")
    with open(QRELS_JSON, "w") as f:
        json.dump(valid_qrels, f, indent=2)
    print(f"  Saved → {QRELS_JSON}")
    print(f"\nLoading embedding model: {EMBED_MODEL}…")
    model = SentenceTransformer(EMBED_MODEL)

    print(f"Embedding {len(doc_texts):,} corpus passages…")
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
    print(f"  Corpus  : {len(doc_list):,} passages")
    print(f"  Queries : {len(valid_queries):,}  (with gold_human_answer)")
    print(f"  Qrels   : {len(valid_qrels):,} query → selected-passage mappings")

build_data()
