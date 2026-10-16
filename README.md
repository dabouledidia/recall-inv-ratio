# ANN Search: Recall What Matters

---

The paper is structured around three experimental axes:

| Axis | Question |
|------|----------|
| [Cost-Accuracy](#axis-1-cost-accuracy) | How much overhead does optimising for Recall impose vs 1/Ratio? |
| [Classification](#axis-2-classification) | Does recall loss translate to label precision loss? |
| [RAG](#axis-3-rag) | Does recall loss hurt LLM answer quality? |

---

## Repository Structure

```
.
├── benchmarks.py               # Parser for all Axis 1 raw data files
├── data/
│   ├── hnsw-example.txt        # Example: HNSW benchmark runs
│   └── ...                     # One file per algorithm
├── datasets/                   # Place downloaded datasets here
├── scripts/
│   ├── axis-1/                 # Cost-accuracy scripts and outputs
│   ├── axis-2/                 # Classification scripts and outputs
│   └── axis-3/                 # RAG scripts, pipeline, and outputs
```

---

**Hardware.** All experiments ran on an 11th Gen Intel Core i7-11700K (3.60 GHz), 32 GB RAM, with AVX-512 support, Ubuntu 22.04 LTS, gcc 11.4.

---

## Axis 1: Cost-Accuracy

### What it tests

Five ANN algorithms are swept across their full hyperparameter grids on six datasets. For every configuration we record QPS, index build time, distance computations per query, and memory footprint.

### Algorithms

Each algorithm is compiled from its official repository following the project's recommended build configuration. Annoy, SuCo, HNSW and SymphonyQG use AVX-512; RaBitQ uses AVX2.

Follow each algorithm's own build instructions. The requirements below cover the **analysis and plotting scripts only**.

```bash
pip install numpy pandas      # parsing benchmark log files, data manipulation
pip install matplotlib        # figure generation
```

| Algorithm | Repository |
|-----------|------------|
| **HNSW** | [facebookresearch/faiss](https://github.com/facebookresearch/faiss) |
| **RaBitQ** | [gaoj0017/RaBitQ](https://github.com/gaoj0017/RaBitQ) |
| **SymphonyQG** | [gouyt13/SymphonyQG](https://github.com/gouyt13/SymphonyQG) |
| **SuCo** | [WeiJiuQi/SuCo](https://github.com/WeiJiuQi/SuCo) |
| **Annoy** | [spotify/annoy](https://github.com/spotify/annoy) |

### Datasets

All datasets for the Cost-Accuracy sweeps are available for download [here](https://drive.google.com/drive/u/0/folders/1f76UCrU52N2wToGMFg9ir1MY8ZocrN34).

| Dataset | N | Q | D |
|---------|---|---|---|
| GIST | 1,000,000 | 1,000 | 960 |
| AGNews | 769,382 | 1,000 | 1,024 |
| SimpleWiki | 260,372 | 1,000 | 3,072 |
| ImageNet-640 | 1,281,167 | 1,000 | 640 |
| MNIST | 69,000 | 10,000 | 784 |
| Fashion-MNIST | 60,000 | 10,000 | 784 |

Place dataset files under `datasets/<name>/` following the paths expected by `benchmarks.py`.

### Generating Benchmark Data

If you wish to re-run the benchmark sweeps from scratch to generate the log files in `data/`:
1. Build and run each algorithm (Faiss HNSW, RaBitQ, SymphonyQG, SuCo, Annoy) over the full hyperparameter grid sweep on the datasets.
2. Ensure the stdout formatting matches the template files provided in `data/`. For example, search recall output must print lines matching:
   `When k = [K], (recall, ratio) = ([RECALL], [RATIO])`
   For the index footprint (Table 5), each build must also print its index size without the base vectors:
   `Index Size (structure): [BYTES]`
3. Save the redirected outputs into the `data/` directory:
   * **HNSW**: `data/hnsw-example.txt`
   * **RaBitQ**: `data/rabitq-example.txt`
   * **SymphonyQG**: `data/symphonyQG-example.txt`
   * **SuCo**: `data/suco-example.txt`
   * **Annoy**: `data/annoy-example.txt`

### Plotting and Analysis

To analyze the parsed data and generate the figures (PDF and PNG formats) in the `scripts/axis-1/` directory:

```bash
cd scripts/axis-1
python qps_threshold_bars.py
python distance_computations.py
python build_time_bars.py
python speedup_vs_k.py
python recall_ratio_vs_k.py
python memory_footprint.py    # prints Table 5 (index footprint ratios)
python comparison_table.py    # prints per-dataset build-time ratios and the k=1 to k=100 change of Recall and 1/Ratio
```

---

## Axis 2: Classification

*(Note: Standard classification benchmarks are programmatically downloaded and vectorised by the setup scripts.)*

### What it tests

For each of four image datasets, the top-100 nearest neighbours of every test query are synthetically degraded from perfect recall (r=1.0) down to r=0.4. For a target recall r, the closest round(100·r) exact neighbours are kept and every remaining slot is filled with a neighbour drawn uniformly at random from the exact ranks [100, W): W = 500 is the default window and W = 1000 the worst-case window. Label Precision@100 (LP@100) and 1/Ratio@100 are measured at each recall level and compared against the Recall diagonal. The multi-k sweep (k ∈ {1, 20, 50, 100}) scales the window with k: W = 5k (default) and W = 10k (worst-case).

### Prerequisites

```bash
pip install numpy pandas      # data manipulation, reading/writing CSV and HDF5 files
pip install matplotlib        # figure generation
pip install datasets h5py     # loading HuggingFace datasets and HDF5 files
```

### Dataset Preparation

For datasets with no pre-computed Euclidean HDF5 benchmarks (CIFAR-10, SVHN), they are generated locally by running:
```bash
cd scripts/axis-2
python build_classification_datasets.py
```

For MNIST and Fashion-MNIST, the HDF5 benchmark files from Axis 1 are reused, and their corresponding labels are programmatically downloaded from Hugging Face (`mnist` and `fashion_mnist` via the `datasets` library) at runtime. We have verified that the image vectors in the HDF5 benchmark files match the Hugging Face dataset byte-for-byte, ensuring perfect alignment of the downloaded labels.

### Running the Sweeps

Once the four HDF5 datasets are in place under `datasets/`, generate the synthetic degradation results:
```bash
cd scripts/axis-2
python generate_synthetic_recall_data.py
python generate_synthetic_recall_data_multi_k.py
```
This produces `semantic_stability.csv` (k = 100, both windows) and `semantic_stability_multi_k.csv` (k ∈ {1, 20, 50, 100}, both windows) containing the Label Precision and 1/Ratio values for simulated recall levels $r \in [0.4, 1.0]$. The committed CSVs are small MNIST-only examples of the output format.

### Plotting and Analysis

Run the plotting scripts to produce the figures (PDF and PNG formats) in the `scripts/axis-2/` directory:
```bash
python label_precision_vs_recall.py
python metric_correlation.py
python mad_predictor_bars.py
python mae_vs_k_table.py      # prints Table 6 (MAE of Recall and 1/Ratio per k and window)
```

---

## Axis 3: RAG

*(Note: Standard QA datasets are programmatically downloaded and embedded by the setup scripts.)*

### What it tests

For each of five QA datasets, an HNSW index is built over document embeddings (`all-MiniLM-L6-v2`, 384-dim, cosine). For each query, the top-10 retrieved documents are synthetically degraded from r=1.0 down to r=0.4: the closest round(10·r) documents are kept and every remaining slot is filled with a document drawn uniformly at random from exact ranks 11–50 (default window) or 11–500 (worst-case window). An LLM (Llama 3.1:8b via Ollama) generates an answer from the mixed context. Three metrics judge quality: Semantic Similarity (`all-mpnet-base-v2`), BERTScore F1 (`distilbert-base-uncased`), and LLM Grade (1–10). All metrics are normalised to the exact-search (r=1.0) baseline.

The three primary datasets (HotpotQA, MS-MARCO, PubMedQA) evaluate against **human-written** gold answers, eliminating any LLM self-evaluation bias. The two corroborating datasets (SciFact, NFCorpus) are BEIR benchmarks where the gold answer is an LLM-generated response at r=1.0 with temperature=0.3.

### Prerequisites

```bash
pip install numpy pandas           # data manipulation and result aggregation
pip install matplotlib             # figure generation
pip install sentence-transformers  # document/query embedding (all-MiniLM-L6-v2, all-mpnet-base-v2)
pip install hnswlib                # building the retrieval index
pip install bert-score             # BERTScore F1 answer quality evaluation
pip install datasets transformers  # loading HuggingFace QA datasets
```

**LLM Configuration:**
1. Install Ollama:
   ```bash
   curl -fsSL https://ollama.com/install.sh | sh
   ```
2. Pull the required model:
   ```bash
   ollama pull llama3.1:8b
   ```
3. Ensure the service is running locally on port 11434:
   ```bash
   ollama serve &
   ```

### Running the Pipeline

The RAG evaluation is executed end-to-end for each dataset. Replace `<dataset>` with `hotpotqa`, `msmarco`, `pubmedqa`, `scifact`, or `nfcorpus` in the commands below.

1. **Ingest the Dataset**: Downloads raw corpora and queries from HuggingFace, embeds them via `all-MiniLM-L6-v2`, and dumps files:
   ```bash
   cd scripts/axis-3/<dataset>
   python 1_ingest_<dataset>.py          # hotpotqa, msmarco, pubmedqa
   python 1_build_corpus_<dataset>.py    # scifact, nfcorpus
   ```
2. **Build Retrieval Index**: Builds the local HNSW database (M=16, efConstruction=200):
   ```bash
   python 2_build_index_<dataset>.py
   ```
3. **Run the RAG Sweep**: For every recall level and both windows, generates an answer from the degraded context, grades it with the LLM (1–10), scores Semantic Similarity and BERTScore F1 against the reference, and aggregates the results. For SciFact and NFCorpus the reference (temperature-0.3 answer from the exact top-10) is generated in the same run:
   ```bash
   cd scripts/axis-3
   python rag_window_sweep.py --dataset <dataset>
   ```
   Results are written to `<dataset>/results/rag_window_sweep.json` (per-query answers and scores) and `rag_window_sweep.csv` (per window and recall level). By default the first 200 queries are used for HotpotQA, MS-MARCO and PubMedQA, all 300 SciFact claims, and all NFCorpus queries with a fully relevant document; `--n-queries` overrides this. A query whose LLM call exceeds 240 s is skipped (listed in `results/skipped_<dataset>.txt`) and Ollama is restarted (set `OLLAMA_BIN` if `ollama` is not on your PATH). The three phases can also be run separately with `--phase A` (generation and grading), `B` (scoring) and `C` (aggregation).

### Plotting and Analysis

Plot the normalized quality metrics and 1/Ratio@10 against recall for both windows (`fig_rag_windows.pdf`/`.png`), and print the raw score ranges (Table 8), the largest deviation of each metric from its exact-search value, and the MAE of Recall and 1/Ratio against the composite quality (the average of the three normalized metrics):
```bash
cd scripts/axis-3
python plot_rag_windows.py
```
The committed `msmarco/results/rag_window_sweep.csv` is an example of the output format.

---

## Reference

References to cite:
```
@misc{dimitropoulos2026annsearchrecallmatters,
      title={ANN Search: Recall What Matters}, 
      author={Dimitris Dimitropoulos and Nikos Mamoulis},
      year={2026},
      eprint={2606.04522},
      archivePrefix={arXiv},
      primaryClass={cs.IR},
      url={https://arxiv.org/abs/2606.04522}, 
}
```
