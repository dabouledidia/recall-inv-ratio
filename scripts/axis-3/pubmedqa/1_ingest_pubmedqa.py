import json, sys
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

HF_DATASET  = "pubmed_qa"
HF_CONFIG   = "pqa_labeled"
HF_SPLIT    = "train"
# pqa_labeled only has a 'train' split (1,000 items)

def build_data() -> None:
    try:
        from datasets import load_dataset
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("ERROR: pip install datasets sentence-transformers")
        sys.exit(1)

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading PubMedQA ({HF_CONFIG}, {HF_SPLIT}) from HuggingFace…")
    ds = load_dataset(HF_DATASET, HF_CONFIG, split=HF_SPLIT)
    print(f"  {len(ds):,} labelled examples loaded")

    # Accumulated output lists (built in a single pass).
    doc_list:     list[dict] = []
    doc_texts:    list[str] = []
    queries:      list[dict] = []
    query_texts:  list[str] = []
    qrels: dict[str, list[str]]  = {}
    skipped = 0

    print("Processing questions and context paragraphs…")
    for item in ds:
        pubid = str(item["pubid"])
        question = item["question"].strip()
        long_answer = item["long_answer"].strip()
        contexts = item["context"]["contexts"]   # list[str]

        if not long_answer:
            skipped += 1
            continue

        # Each context paragraph → one corpus document.
        ctx_doc_ids: list[str] = []
        for i, para in enumerate(contexts):
            doc_id = f"{pubid}_ctx_{i}"
            doc = {
                "id":
                doc_id,
                "title": "",
                # PubMedQA paragraphs carry no individual title
                "text":  para.strip(),
            }
            doc_list.append(doc)
            doc_texts.append(para.strip())
            ctx_doc_ids.append(doc_id)

        queries.append({
            "id":
            pubid,
            "text":
            question,
            "gold_human_answer": long_answer,
        })
        query_texts.append(question)
        qrels[pubid] = ctx_doc_ids

    if skipped:
        print(f"  Skipped {skipped} questions with empty long_answer")

    print(f"  {len(queries):,} questions with valid gold answers")
    print(f"  {len(doc_list):,} context paragraphs in corpus "
          f"(avg {len(doc_list)/max(len(queries),1):.1f} per question)")
    with open(CORPUS_JSONL, "w") as f:
        for d in doc_list:
            f.write(json.dumps(d) + "\n")
    print(f"  Saved → {CORPUS_JSONL}")
    with open(QUERIES_JSONL, "w") as f:
        for q in queries:
            f.write(json.dumps(q) + "\n")
    print(f"  Saved → {QUERIES_JSONL}")
    with open(QRELS_JSON, "w") as f:
        json.dump(qrels, f, indent=2)
    print(f"  Saved → {QRELS_JSON}")
    print(f"\nLoading embedding model: {EMBED_MODEL}…")
    model = SentenceTransformer(EMBED_MODEL)

    print(f"Embedding {len(doc_texts):,} corpus paragraphs…")
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
    print(f"  Corpus  : {len(doc_list):,} paragraphs")
    print(f"  Queries : {len(queries):,}  (with gold_human_answer)")
    print(f"  Qrels   : {len(qrels):,} query → context-paragraph mappings")

build_data()
