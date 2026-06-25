# GraphRAG for Scientific Literature

## Project Goal

This project builds a **GraphRAG** pipeline over PubMed scientific abstracts. The end-to-end vision:

- **PubMed GraphRAG** — ingest a reproducible subset of biomedical literature
- **Semantic chunking** — split abstracts into meaning-aware units for retrieval
- **Neo4j knowledge graph** — store entities, relations, and document structure
- **Graph-enhanced retrieval** — combine vector search with graph traversal
- **LLM generation** — answer questions grounded in retrieved evidence
- **Evaluation** — measure retrieval quality and answer faithfulness
- **Demo application** — interactive query interface over the graph

> **Project state:** All phases (1–7) are complete and deployed. The current focus is maintenance, evaluation, and incremental UX improvements.
>
> **Phase 1** — data loading, chunking, embedding, visualization.  
> **Phase 2** — entity extraction and Neo4j-importable graph export.  
> **Phase 3** — graph-enhanced retrieval using offline artifacts.  
> **Phase 4** — LLM generation and evaluation over PubMedQA.  
> **Phase 5** — multiple embedding indexes / chunking strategies and query-driven routing.  
> **Phase 6** — HNSW approximate-nearest-neighbor vector search.  
> **Phase 7** — streaming sources/citations with event-sequence proof.

## Environment

**OS:** Windows 11 / WSL2 Ubuntu (deployed to Streamlit Community Cloud)

**Project location:** `/mnt/d/pubmed-graphrag`

**Repository:** https://github.com/Slayer025/pubmed-graphrag-v2

**Streamlit Cloud URL:** https://pubmed-graphrag-kamfpkughsfmstpcrv8r23.streamlit.app/

**Virtual environments:**
- Linux/WSL: `/mnt/d/pubmed-graphrag/.venv` (Python 3.14)
- Windows: `/mnt/d/pubmed-graphrag/.venv_win` (Python 3.10)

**Python version (Streamlit Cloud):** 3.11.9 (pinned in `runtime.txt`)

Activate the environment before running commands:

```bash
# Windows PowerShell
.venv_win\Scripts\activate

# WSL/Linux
source .venv/bin/activate
```

**HF cache:** `HF_HOME=/tmp/hf_cache` (set automatically by `configure_environment()`)

## Dataset

**Dataset source:** HuggingFace `scientific_papers/pubmed` (`armanc/scientific_papers`)

**Sampling strategy:** Stream train split; collect first **5000** valid abstracts only

**Storage format:** gzip-compressed JSONL

**Input file:** `data/pubmed_5000.jsonl.gz` (~2.1 MB)

Each record:

```json
{"article_id": "0", "abstract": "..."}
```

## Current Architecture

### `src/load_data.py`

| | |
|---|---|
| **Purpose** | Stream PubMed abstracts from HuggingFace, sample 5000, save compressed subset |
| **Input** | HuggingFace `scientific_papers/pubmed` (streaming) |
| **Output** | `data/pubmed_5000.jsonl.gz` |

### `src/chunker.py`

| | |
|---|---|
| **Purpose** | Three chunking strategies over abstract text |
| **Input** | Abstract records (`article_id`, `abstract`) |
| **Output** | Chunk records (`article_id`, `chunk_id`, `text`, `strategy`) |

Strategies: `fixed` (100-token windows), `sentence` (~100-token sentence groups), `semantic` (embedding clustering).

### `src/create_chunks.py`

| | |
|---|---|
| **Purpose** | Orchestrate chunking and persist three strategy datasets |
| **Input** | `data/pubmed_5000.jsonl.gz` |
| **Output** | `data/chunks/chunks_{fixed,sentence,semantic}.jsonl.gz` |

Estimates output size before writing; aborts if estimate exceeds 1 GB.

### `src/embeddings.py`

| | |
|---|---|
| **Purpose** | Batch-encode semantic chunks with sentence-transformers |
| **Input** | `data/chunks/chunks_semantic.jsonl.gz` |
| **Output** | `data/embeddings/semantic_embeddings.npy` (L2-normalized float32) |

Model: `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions).

### `src/visualize_chunks.py`

| | |
|---|---|
| **Purpose** | 2-D projection of semantic embeddings for quality inspection |
| **Input** | Semantic chunks + `semantic_embeddings.npy` |
| **Output** | `outputs/semantic_clusters.png` |

Uses UMAP when available; falls back to t-SNE. Subsamples to 10 000 points for memory efficiency.

### `src/graph_schema.py`

| | |
|---|---|
| **Purpose** | Define Neo4j node/relationship schema and generate import Cypher |
| **Input** | None (schema constants) |
| **Output** | `GraphSchema` object; `data/graph/schema.cypher` |

Schema: `:Article {article_id, abstract}`, `:Chunk {chunk_id, article_id, text, strategy, embedding}`, `:Entity {entity_id, name, label}`; relationships `:HAS_CHUNK` (Article → Chunk) and `:MENTIONS` (Chunk → Entity).

### `src/entity_extraction.py`

| | |
|---|---|
| **Purpose** | Extract named entities and noun phrases from semantic chunks |
| **Input** | `data/chunks/chunks_semantic.jsonl.gz` |
| **Output** | `data/graph/entities.jsonl.gz` |

Uses spaCy `en_core_web_sm` (~13 MB); falls back to a lightweight regex extractor if spaCy is unavailable.

### `src/graph_loader.py`

| | |
|---|---|
| **Purpose** | Build Article / Chunk / Entity nodes and HAS_CHUNK / MENTIONS edges as CSVs |
| **Input** | Semantic chunks, embeddings, source abstracts, and extracted entity mentions |
| **Output** | `data/graph/*.csv` and `data/graph/schema.cypher` |

Estimates output size before writing; aborts if estimate exceeds 1 GB.

### `src/create_graph.py`

| | |
|---|---|
| **Purpose** | Orchestrate Phase 2: entity extraction + graph CSV export |
| **Input** | Phase 1 artifacts |
| **Output** | All files under `data/graph/` |

### `src/llm_client.py`

| | |
|---|---|
| **Purpose** | Real LLM generation backends that implement the `LLMClient` protocol |
| **Input** | Prompt string; environment variables (`OPENAI_API_KEY`, `LLM_MODEL`, `OLLAMA_URL`) |
| **Output** | Generated answer text |

Supports `OpenAIClient` (OpenAI-compatible chat completions), `OllamaClient` (local Ollama `/api/generate`), and re-exports `MockLLMClient` for testing.

### `src/evaluation.py`

| | |
|---|---|
| **Purpose** | Retrieval evaluation metrics and vector-only vs GraphRAG comparison |
| **Input** | PubMedQA questions with matched article IDs; `Retriever` instance |
| **Output** | `RetrievalMetrics`, per-question results, `outputs/retrieval_results.csv`, `outputs/retrieval_summary.json` |

Metrics: Recall@5, Recall@10, MRR. Compares vector-only baseline (`expand_depth=0, alpha=1.0`) against default GraphRAG (`expand_depth=2, alpha=0.8`).

### `src/generation_eval.py`

| | |
|---|---|
| **Purpose** | Generation quality metrics |
| **Input** | Generated answers and PubMedQA reference long answers |
| **Output** | ROUGE-L and BERTScore (precision/recall/F1) per question |

### `src/eval_dataset.py`

| | |
|---|---|
| **Purpose** | Build the Phase 4 evaluation dataset from PubMedQA |
| **Input** | PubMedQA `pqa_labeled` split (via HuggingFace or raw JSON fallback); our 5,000 abstract subset |
| **Output** | `data/evaluation/pubmedqa_filtered.jsonl.gz` |

### `scripts/run_evaluation.py`

| | |
|---|---|
| **Purpose** | End-to-end Phase 4 evaluation runner |
| **Input** | Evaluation questions and pre-computed query embeddings |
| **Output** | `outputs/retrieval_results.csv`, `outputs/retrieval_summary.json`, `outputs/generation_results.csv`, `outputs/evaluation_summary.json` |

CLI: `--max-questions`, `--retrieval-only`, `--llm-client {mock,openai,ollama}`, `--use-llm`.

### `notebooks/phase4_evaluation.ipynb`

| | |
|---|---|
| **Purpose** | Interactive summary of dataset statistics, retrieval comparison, and generation metrics |

### `src/storage.py`

Shared disk-budget utilities: gzip I/O, HF cache configuration, cleanup helpers, size estimation and 1 GB warnings (disk02 policy).

### `src/config.py`

| | |
|---|---|
| **Purpose** | Central configuration: Neo4j (optional), embedding model, artifact paths, retrieval hyperparameters |
| **Input** | Environment variables and defaults |
| **Output** | `AppConfig` dataclass |

### `src/retriever.py`

| | |
|---|---|
| **Purpose** | Graph-enhanced retrieval: query embedding, vector search, graph expansion, deduplication, re-ranking |
| **Input** | Query string or vector; Phase 1/2 artifacts |
| **Output** | `list[RetrievalResult]` with `chunk_id`, `article_id`, `text`, `vector_score`, `graph_score`, `combined_score` |

### `src/rag_pipeline.py`

| | |
|---|---|
| **Purpose** | End-to-end RAG interface: `retrieve()` + `generate()` |
| **Input** | Query string |
| **Output** | `RAGResponse` with query, ranked context, and answer |

Notes:
- `generate()` is currently a mock/placeholder with an `LLMClient` protocol for future OpenAI/Ollama integration.
- Phase 3 does **not** require a Neo4j instance; graph expansion is performed offline from `data/graph/*.csv`.

### `scripts/retrieval_debug.py`

| | |
|---|---|
| **Purpose** | Manual retrieval test and diagnostic tool |
| **Input** | Query string, or `--query-chunk-id`, or `--query-vector-file` |
| **Output** | Ranked retrieval results and optional mock generation |

## Data Artifacts

| File | Size | Description |
|------|------|-------------|
| `data/pubmed_5000.jsonl.gz` | 2.1 MB | 5000 sampled abstracts |
| `data/chunks/chunks_fixed.jsonl.gz` | 2.2 MB | 15 582 fixed-token chunks |
| `data/chunks/chunks_sentence.jsonl.gz` | 2.2 MB | 17 449 sentence chunks |
| `data/chunks/chunks_semantic.jsonl.gz` | 2.2 MB | 15 556 semantic chunks |
| `data/embeddings/semantic_embeddings.npy` | 23 MB | Shape `(15556, 384)` |
| `outputs/semantic_clusters.png` | 712 KB | 2-D embedding visualization |
| `data/graph/entities.jsonl.gz` | 407 KB | Entity mentions per semantic chunk |
| `data/graph/articles.csv` | 5.9 MB | 5 000 Article nodes |
| `data/graph/chunks.csv` | 60.5 MB | 15 556 Chunk nodes with embeddings |
| `data/graph/entities.csv` | 7.5 MB | 137 219 deduplicated Entity nodes |
| `data/graph/has_chunk.csv` | 373 KB | 15 556 Article → Chunk edges |
| `data/graph/mentions.csv` | 10.6 MB | 258 464 Chunk → Entity edges |
| `data/graph/schema.cypher` | 1.8 KB | Neo4j constraints, indexes, and `LOAD CSV` script |

## Phase Progress

### Phase 0 — Project setup

**Status:** Complete

- Repository structure (`data/`, `outputs/`, `src/`, `notebooks/`)
- Storage-efficient loading policy (streaming, gzip, disk02 rules)
- Environment on D: drive with external HF/pip caches

### Phase 1 — Chunking, embeddings, visualization

**Status:** Complete

- [x] Load 5000 PubMed abstracts (`load_data.py`)
- [x] Implement three chunking strategies (`chunker.py`)
- [x] Save chunk datasets (`create_chunks.py`)
- [x] Encode semantic chunks (`embeddings.py`)
- [x] Visualize embedding clusters (`visualize_chunks.py`)

### Phase 2 — Neo4j graph construction

**Status:** Complete

- [x] Define graph schema (`graph_schema.py`)
- [x] Extract entities from semantic chunks (`entity_extraction.py`)
- [x] Build Neo4j-importable node/edge CSVs (`graph_loader.py`)
- [x] Orchestrate Phase 2 (`create_graph.py`)

### Phase 3 — Retrieval

**Status:** Complete

- [x] Central configuration (`src/config.py`)
- [x] Graph-enhanced retriever (`src/retriever.py`)
- [x] RAG pipeline scaffold with mock generation (`src/rag_pipeline.py`)
- [x] Manual retrieval debug tool (`scripts/retrieval_debug.py`)
- [x] Verified artifact alignment (embeddings, chunks, CSV)
- [x] Vector search over `data/embeddings/semantic_embeddings.npy`
- [x] Offline graph expansion via `MENTIONS` and `HAS_CHUNK`
- [x] Re-ranking with configurable `alpha` weighted sum
- [x] Manual retrieval test executed successfully

### Phase 4 — LLM generation & evaluation

**Status:** Complete

- [x] PubMedQA evaluation dataset pipeline (`src/eval_dataset.py`)
- [x] Filter PubMedQA questions to our 5,000 abstract subset (`data/evaluation/pubmedqa_filtered.jsonl.gz`)
- [x] Text-similarity matching between PubMedQA contexts and our abstracts
- [x] Retrieval metrics: Recall@5, Recall@10, MRR (`src/evaluation.py`)
- [x] Vector-only vs GraphRAG retrieval comparison (`outputs/retrieval_comparison.csv`)
- [x] Real LLM clients: OpenAI-compatible and Ollama (`src/llm_client.py`)
- [x] Generation metrics: ROUGE-L and BERTScore (`src/generation_eval.py`)
- [x] End-to-end Phase 4 runner (`scripts/run_evaluation.py`)
- [x] Evaluation notebook (`notebooks/phase4_evaluation.ipynb`)

### Phase 5 — Multiple Embedding Indexes

**Status:** Complete

- [x] Offline builders for `fixed`, `sentence`, and `semantic` indexes (`scripts/build_indexes.py`, `scripts/build_hnsw_indexes.py`)
- [x] `MultiIndexVectorStore` and `SwitchableVectorStore` adapters
- [x] Query-driven index routing in `src/domain/services/strategy_router.py`
- [x] Streamlit controls for multi-index routing

### Phase 6 — HNSW Search

**Status:** Complete

- [x] hnswlib-based approximate nearest neighbor indexes built offline
- [x] Runtime HNSW/NumPy switching via `SwitchableVectorStore`
- [x] Exact cosine re-scoring to preserve recall
- [x] GitHub Release `v2.1-hnsw` with all 16 artifacts

### Phase 7 — Streaming Sources / Citations

**Status:** Complete

- [x] `StreamEvent` domain model with timestamps
- [x] `RetrieveAndGenerateStreamUseCase` (generator-based)
- [x] LLM `stream_answer()` for Mock, OpenAI, and Ollama clients
- [x] Streamlit streaming UI with live sources, graph evidence, and answer tokens
- [x] "🕒 Event Sequence" proof table showing sources arrive before the answer finishes

### Demo application

**Status:** Complete

- [x] LLM-driven query decomposition (`src/query_decomposer.py`)
- [x] Graph-signal re-ranking (`src/graph_reranker.py`)
- [x] Optional Neo4j GDS PageRank integration
- [x] Streamlit demo (`scripts/demo.py`)

**New components:**

| File | Purpose |
|---|---|
| `src/query_decomposer.py` | Decompose complex questions into LLM-generated sub-questions; falls back to the original query on failure. |
| `src/graph_reranker.py` | Re-rank `RetrievalResult` lists using graph connectivity signals (shared entities, connected chunks, inverse entity degree). Optional Neo4j GDS PageRank when a graph is loaded. |
| `src/application/use_cases/retrieve_and_generate_stream.py` | Generator-based streaming retrieve-and-generate use case. |
| `src/domain/entities/stream_events.py` | Streaming event dataclasses (RetrievalStarted, ChunksFound, GraphEvidenceFound, TextChunkEvent, StreamComplete). |
| `scripts/demo.py` | Lightweight Streamlit UI for interactive retrieval and answer generation. |

## Demo

Start the Streamlit demo after installing dependencies:

```bash
cd /mnt/d/pubmed-graphrag
# Windows
.venv_win\Scripts\activate
streamlit run scripts/demo.py

# WSL/Linux
source .venv/bin/activate
streamlit run scripts/demo.py
```

Configure secrets via `.streamlit/secrets.toml` or environment variables:

```toml
[llm]
openai = "sk-..."

[embedding]
provider = "huggingface_api"   # or "local" / "remote_http"
hf_api_token = "hf_..."
model = "sentence-transformers/all-MiniLM-L6-v2"

[app]
artifact_base_url = "https://github.com/<owner>/<repo>/releases/download/v2.1-hnsw"
```

The demo runs locally in a browser and supports:

* Mock, OpenAI, and Ollama LLM clients
* Dense, hybrid (BM25 + RRF), HNSW, multi-index, and routed retrieval
* Metadata-aware entity-label boosting
* Adjustable retrieval parameters
* Optional LLM-driven query decomposition
* Optional graph-signal re-ranking
* Optional Neo4j GDS PageRank (requires a running Neo4j graph named `pubmed-graph`)
* Streaming mode with sources/citations and event-sequence proof

## Commands

Run the full Phase 1 pipeline from the project root:

```bash
cd /mnt/d/pubmed-graphrag
source .venv/bin/activate
export HF_HOME=/mnt/d/hf_cache_backup
export PIP_CACHE_DIR=/mnt/d/pip_cache_backup

# Step 1: Load abstracts (already done if pubmed_5000.jsonl.gz exists)
python -m src.load_data

# Step 2: Generate chunk datasets
python -m src.create_chunks

# Step 3: Encode semantic chunks
python -m src.embeddings

# Step 4: Visualize embeddings
python -m src.visualize_chunks

# Step 5: Build Neo4j-importable graph (Phase 2)
python -m src.create_graph
```

## Phase 3 — Graph-enhanced retrieval

Retrieval works entirely from existing repository artifacts. No running Neo4j instance is required.

### Retrieval architecture

```
Query
  │
  ▼
┌─────────────────┐
│ Query embedding │  sentence-transformers/all-MiniLM-L6-v2
│ (384-d, L2      │
│  normalized)    │
└────────┬────────┘
         ▼
┌─────────────────┐
│ Vector search   │  cosine similarity via dot product over
│ (top_k)         │  data/embeddings/semantic_embeddings.npy
└────────┬────────┘
         ▼
┌─────────────────┐
│ Graph expansion │  depth ≤ 2 using data/graph/mentions.csv
│ (bounded BFS)   │  and data/graph/has_chunk.csv
└────────┬────────┘
         ▼
┌─────────────────┐
│ Deduplication   │  by chunk_id (keep best depth/score)
└────────┬────────┘
         ▼
┌─────────────────┐
│ Re-ranking      │  combined = α·vector_score + (1‑α)·graph_score
│                 │  default α = 0.8
└────────┬────────┘
         ▼
   list[RetrievalResult]
```

### Graph expansion rules

| Depth | Path | Source label | `graph_score` |
|---|---|---|---|
| 0 | Vector-retrieved chunk | `vector` | 1.0 |
| 1 | Same article → chunk | `same_article` | 0.5 |
| 1 | Shared entity → chunk | `shared_entity` | 0.5 |
| 2 | Entity → chunk → entity → chunk | `shared_entity` | 0.25 |

Expansion is bounded by:

- `expand_depth` (default 2)
- `max_entity_degree` (default 500) — skip entities shared by too many chunks
- `max_expansion_per_entity` (default 100) — deterministic neighbor cap per entity
- `max_expanded_nodes` (default 2 000) — hard BFS node cap

### Configuration

All retrieval parameters live in `src/config.py` (`RetrievalConfig`):

| Parameter | Default | Description |
|---|---|---|
| `top_k` | 10 | Vector search candidates |
| `expand_depth` | 2 | Maximum graph traversal depth |
| `max_entity_degree` | 500 | Degree filter for shared entities |
| `max_expansion_per_entity` | 100 | Max neighbors per entity |
| `max_expanded_nodes` | 2 000 | Total expansion budget |
| `alpha` | 0.8 | Weight for vector score in combined ranking |
| `depth_scores` | (1.0, 0.5, 0.25) | Graph score by depth |
| `max_results` | 20 | Final ranked result cap |

Neo4j settings are optional (`Neo4jConfig`, default `enabled=False`). They are reserved for future database-backed phases.

### Example commands

Activate the environment first:

```bash
cd /mnt/d/pubmed-graphrag
source .venv/bin/activate
export HF_HOME=/mnt/d/hf_cache_backup
export PIP_CACHE_DIR=/mnt/d/pip_cache_backup
```

Run a string query (embeds the query, then retrieves):

```bash
python scripts/retrieval_debug.py "risk factors for type 2 diabetes" --top-k 5 --max-results 10
```

Run with mock LLM generation:

```bash
python scripts/retrieval_debug.py "risk factors for type 2 diabetes" --top-k 5 --max-results 10 --generate
```

Use a chunk embedding as the query vector (bypasses model loading, useful for cold-start testing):

```bash
python scripts/retrieval_debug.py --query-chunk-id 0_semantic_0000 --top-k 5 --max-results 10
```

Use a pre-computed query vector from a `.npy` file:

```bash
python scripts/retrieval_debug.py --query-vector-file query.npy --top-k 5 --max-results 10
```

Adjust graph expansion and ranking:

```bash
python scripts/retrieval_debug.py "diabetes prevention" \
  --top-k 10 \
  --max-entity-degree 200 \
  --alpha 0.7 \
  --max-results 15
```

Run the Phase 4 evaluation (retrieval only, fast smoke test):

```bash
python scripts/run_evaluation.py --max-questions 10 --retrieval-only
```

Run the full Phase 4 evaluation with mock generation:

```bash
python scripts/run_evaluation.py --max-questions 50 --llm-client mock
```

Run with a real OpenAI-compatible model:

```bash
export OPENAI_API_KEY=sk-...
export LLM_MODEL=gpt-3.5-turbo
python scripts/run_evaluation.py --max-questions 50 --llm-client openai
```

Run with a local Ollama model:

```bash
export LLM_MODEL=llama3
export OLLAMA_URL=http://localhost:11434
python scripts/run_evaluation.py --max-questions 50 --llm-client ollama
```

Build or refresh the evaluation dataset:

```bash
python -m src.eval_dataset
```

Pre-compute query embeddings (slow once, reused by the runner):

```bash
python -c "
from src.evaluation import load_questions, precompute_query_embeddings
q = load_questions('data/evaluation/pubmedqa_filtered.jsonl.gz')
precompute_query_embeddings(q)
"
```

## Phase 2 import (after Neo4j is running)

Copy `data/graph/*.csv` and `data/graph/schema.cypher` to your Neo4j import directory, then run the Cypher script in Neo4j Browser or `cypher-shell`:

```cypher
:source schema.cypher
```

No Neo4j server is required to generate the CSV files — `create_graph.py` produces import-ready artifacts offline.

. **Mock LLM "Insufficient evidence" with hybrid mode** — FIXED. Threshold was lowered from 0.55 to 0.05 so normal retrieval scores produce extractive answers. Real LLM still recommended for generative answers.

2. **HuggingFace Inference endpoint deprecation** — FIXED. Updated `DEFAULT_HF_API_URL` to `https://router.huggingface.co/hf-inference/models` and appended `/pipeline/feature-extraction` to the request path.

3. **Query classifier is conservative** — 30/40 evaluation queries classified as "general". Architecture proven; needs LLM-based classifier for better accuracy.

4. **Metadata boost metrics unchanged** — Evaluation queries don't contain entity-label keywords (gene, drug, disease). Architecture proven; needs targeted query set to show effect.

5. **Secret leakage** — FIXED. Added `scrub_secrets()` utility, hid `HF_API_TOKEN` from `EmbeddingConfig` repr, and scrubbed tokens from logs, errors, and UI output.

## Limitations

- **No Neo4j vector index is used.** Embeddings are loaded from `data/embeddings/*.npy` and searched with NumPy / HNSW. Neo4j integration remains optional for future phases.
- **Cold-start latency.** The first string query incurs the cost of importing PyTorch/sentence-transformers and loading the model from disk. This can take 10–30 seconds on a warm Streamlit Cloud container or 2–4 minutes locally. Subsequent queries in the same process are fast (~0.5–1 s retrieval time).
- **Graph expansion is bounded but can still add thousands of candidates.** Use `max_entity_degree` and `max_expanded_nodes` to tune recall/latency trade-offs.
- **No SEMANTIC_SIMILAR edges were added.** Expansion uses only `HAS_CHUNK` and `MENTIONS` relationships from Phase 2.
- **Query decomposition requires a working LLM.** The deterministic fallback returns the original query if the LLM fails.
- **Graph reranker PageRank requires Neo4j GDS.** Without it, the reranker falls back to offline connectivity heuristics.
- **HNSW requires `hnswlib`, which has no Windows wheel.** Local Windows development falls back to NumPy search; Linux deployments (Streamlit Cloud) use HNSW when `use_hnsw=True`.

## Next Steps

1. **Faithfulness evaluation** — add answer-level metrics (consistency with retrieved context)
2. **LLM-based query classifier** — replace keyword classifier with a small LLM or fine-tuned model
3. **Citation extraction** — link answer sentences to the chunks they summarize
4. **Production LLM backend** — move from mock to OpenAI / Ollama for generative answers
5. **Neo4j-backed retrieval mode** — optional live graph traversal (currently offline CSV only)

## Constraints

- Do not download the full PubMed dataset (~880 MB zip / ~2.5 GB Arrow)
- Prefer HuggingFace **streaming** for remote data
- Prefer **gzip-compressed** storage (`.jsonl.gz`, `.npy`)
- **Estimate disk usage** before large writes; **abort if > 1 GB**
- Keep modules separate with type hints and logging
- HuggingFace cache pinned via `HF_HOME`

## Handoff Notes

**Current state:** Phases 0–7 are complete and deployed. The project has a 5000-abstract working set, three chunk strategies persisted as gzip JSONL, L2-normalized semantic embeddings, a cluster visualization, a Neo4j-importable graph export (137k entities, 258k mentions) generated offline, an offline graph-enhanced retriever, a full Phase 4 evaluation harness over PubMedQA, a lightweight Streamlit demo, HNSW approximate-nearest-neighbor search, multi-index routing, metadata-aware boosting, and streaming sources/citations with an event-sequence proof.

**Completed source files:**

- `src/load_data.py` — streaming data loader
- `src/chunker.py` — fixed / sentence / semantic chunking
- `src/create_chunks.py` — chunk dataset builder
- `src/embeddings.py` — semantic embedding encoder
- `src/visualize_chunks.py` — 2-D embedding plot
- `src/storage.py` — disk-budget and I/O helpers
- `src/graph_schema.py` — Neo4j graph schema and Cypher generator
- `src/entity_extraction.py` — spaCy NER + noun-phrase entity extractor
- `src/graph_loader.py` — Neo4j CSV node/relationship exporter
- `src/create_graph.py` — Phase 2 orchestrator
- `src/config.py` — central configuration
- `src/retriever.py` — graph-enhanced retriever
- `src/rag_pipeline.py` — RAG pipeline with optional decomposition and graph re-ranking
- `scripts/retrieval_debug.py` — manual retrieval test
- `src/eval_dataset.py` — PubMedQA evaluation dataset builder
- `src/evaluation.py` — retrieval metrics and comparison
- `src/llm_client.py` — OpenAI / Ollama / mock LLM clients
- `src/generation_eval.py` — ROUGE-L and BERTScore evaluation
- `scripts/run_evaluation.py` — end-to-end Phase 4 runner
- `notebooks/phase4_evaluation.ipynb` — evaluation summary notebook
- `src/query_decomposer.py` — LLM-driven query decomposition
- `src/graph_reranker.py` — graph-signal re-ranking
- `scripts/demo.py` — Streamlit demo application
- `tests/test_query_decomposer.py` — decomposition unit tests
- `tests/test_graph_reranker.py` — reranker unit tests

**Generated artifacts:** See [Data Artifacts](#data-artifacts) above, plus:

- `data/evaluation/pubmedqa_filtered.jsonl.gz` — filtered PubMedQA questions
- `data/evaluation/query_embeddings.npy` — cached query embeddings
- `data/evaluation/query_index.json` — query-to-row mapping
- `outputs/retrieval_results.csv` — per-question retrieval results
- `outputs/retrieval_summary.json` — aggregated retrieval metrics
- `outputs/generation_results.csv` — per-question generation metrics
- `outputs/evaluation_summary.json` — combined retrieval + generation summary

**Next implementation target:** All requested phases are complete. Future work could include:
1. A richer graph visualization in the demo (e.g., `pyvis` subgraph rendering).
2. Faithfulness evaluation for generated answers.
3. Neo4j-backed retrieval mode (currently optional and not required).

**Do not yet:** redesign the retrieval architecture, add SEMANTIC_SIMILAR edges, or migrate entity extraction to SciSpaCy unless explicitly requested.
