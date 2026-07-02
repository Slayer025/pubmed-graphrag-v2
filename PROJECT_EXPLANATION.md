# PubMed GraphRAG — Full Project Explanation

> **Internal reference document — not committed to GitHub.**
> This guide explains what the project does, how every feature works, and what we implemented across all phases.

---

## 1. What is this project?

**PubMed GraphRAG** is a retrieval-augmented generation (RAG) system built over a 5,000-abstract sample of PubMed scientific literature. It combines:

- **Dense vector search** (semantic similarity)
- **Sparse keyword search** (BM25)
- **Graph traversal** (article/entity relationships)
- **Metadata-aware reranking**
- **Multiple embedding indexes** (semantic, fixed-window, sentence-level)
- **Approximate-nearest-neighbor (HNSW) indexing**
- **Streaming sources/citations** in the UI

The final app is deployed as a **Streamlit Community Cloud** application.

---

## 2. High-level data flow

```
User question
    │
    ▼
[Query embedding] ── local sentence-transformers OR HuggingFace Inference API
    │
    ▼
[Vector search] ── NumPy exact OR HNSW approximate
    │
    ▼
[Optional] BM25 keyword search + RRF fusion
    │
    ▼
[Graph expansion] ── same-article + shared-entity BFS
    │
    ▼
[Optional] Metadata-aware boost / graph reranker
    │
    ▼
[LLM generation] ── mock extractive OR OpenAI / Ollama streaming
    │
    ▼
Answer + ranked sources + graph evidence + event-sequence proof
```

---

## 3. Project structure

| Folder | Purpose |
|---|---|
| `src/bootstrap/` | **Dependency Injection container.** Builds the entire object graph (embeddings, vector store, graph repo, use cases, LLM). |
| `src/domain/` | Pure domain logic: entities (`Chunk`, `RetrievalResult`, `StreamEvent`), value objects (`Query`, `RetrievalHyperparameters`), and services (`RRF`, `metadata boost`, `graph traversal`, `query classifier`, `strategy router`). |
| `src/application/` | Use cases and ports: `RetrieveDocumentsUseCase`, `VectorSearchUseCase`, `GenerateAnswerUseCase`, `RetrieveAndGenerateStreamUseCase`, DTOs like `SearchConfig`. |
| `src/infrastructure/` | Concrete adapters: embedding clients, vector stores, graph repositories, retrievers, storage, secret scrubbing. |
| `src/interfaces/streamlit/` | Streamlit UI (`demo.py`). |
| `tests/` | Unit tests — 115 passing. |
| `evaluation/` | 40 frozen evaluation queries, `run_eval.py`, result JSONLs. |
| `scripts/` | `demo.py` (Streamlit launcher), `build_indexes.py`, `build_hnsw_indexes.py`. |

### Clean Architecture rule
- Domain does **not** import infrastructure or frameworks.
- Application defines **ports** (Protocols).
- Infrastructure implements those ports.
- Only `src/bootstrap/__init__.py` wires concrete implementations together.

---

## 4. Phase-by-phase breakdown

### Phase 1 — Hybrid Retrieval (Dense + Sparse + RRF)

**What problem does it solve?**
Dense embeddings miss exact keyword matches. BM25 finds exact keywords but misses semantic meaning. Combining both improves recall.

**How it works:**
1. `VectorSearchUseCase` gets top-k chunks by cosine similarity.
2. `BM25Retriever` (`src/infrastructure/retrievers/bm25_retriever.py`) tokenizes chunks with regex and gets top-k by keyword relevance.
3. `RRFFusionService` (`src/domain/services/rrf_fusion_service.py`) merges both ranked lists:
   - For each list, every rank contributes `1 / (k + rank)` to the chunk's fused score.
   - Default `k=20`.
4. Top fused chunks feed into graph expansion.

**Files:** `bm25_retriever.py`, `rrf_fusion_service.py`, `retrieve_documents.py`

**UI toggle:** `Enable Hybrid Retrieval`

---

### Phase 2 — Remote Embedding Service

**What problem does it solve?**
Decouples embedding generation from the app so we can use a remote API instead of loading sentence-transformers locally.

**How it works:**
`RemoteEmbeddingClient` supports three providers:
- `local` — loads `sentence-transformers/all-MiniLM-L6-v2` (384-dim).
- `huggingface_api` — calls HuggingFace Inference API.
- `remote_http` — calls any custom HTTP endpoint.

If the remote call fails, it **gracefully falls back** to the local model.

**Endpoint fix we applied:**
HuggingFace moved from `api-inference.huggingface.co` to `router.huggingface.co/hf-inference/models/{model}/pipeline/feature-extraction`.

**Secrets:** `HF_API_TOKEN` is read from `st.secrets` first, then env vars. It is scrubbed from all logs and UI output.

**Files:** `remote_embedding_client.py`, `config.py`

**UI panel:** `System Status → Embedding provider diagnostics`

---

### Phase 3 — Query Understanding Layer

**What problem does it solve?**
Different question types benefit from different retrieval strategies (e.g., relationship questions need deeper graph expansion).

**How it works:**
1. `query_classifier.py` classifies the question into:
   - `definition`, `entity_specific`, `relationship`, `mechanism`, `comparison`, `general`
2. `strategy_router.py` picks a strategy:
   - `expand_depth` (how far to traverse the graph)
   - `use_hybrid` (whether to use BM25)
   - `rrf_k`
   - `index_name` (multi-index routing, Phase 5)
3. `RetrieveDocumentsUseCase._apply_strategy()` rebuilds a routed `SearchConfig` and logs the decision.

**Files:** `query_classifier.py`, `strategy_router.py`, `retrieve_documents.py`

**UI toggle:** `Enable Query Understanding & Routing`

**UI display:** `🧠 Query Understanding` expander shows query type, matched keywords, detected entities, selected strategy, selected index, and reason.

---

### Phase 4 — Metadata-Aware Retrieval

**What problem does it solve?**
Semantic similarity alone can miss domain-specific relevance. If a query mentions "diabetes" and a chunk has entities labeled `Disease`, boosting that chunk can improve ranking.

**How it works:**
1. `MetadataBoostService` reads entities attached to each chunk from the graph repository.
2. `metadata_boost_service.py` checks whether query tokens match entity labels.
3. Matching chunks have their `combined_score` multiplied by `metadata_boost_factor` (default 1.1).
4. Results are re-sorted.

**Files:** `metadata_boost_service.py`, `metadata_boost.py`

**UI toggle:** `Enable Metadata-Aware Boosting`

---

### Phase 5 — Multiple Embedding Indexes

**What problem does it solve?**
Different chunking strategies capture different kinds of evidence:
- `semantic` — meaning-aware clusters (default, good for general questions)
- `fixed` — fixed 500-char windows (good for factoid spans)
- `sentence` — sentence-level chunks (good for relationship / mechanism questions)

**How it works:**
1. `scripts/build_indexes.py` creates three chunk+embedding datasets offline.
2. `SwitchableVectorStore` holds all three NumPy stores and can switch by `index_name` at query time.
3. `strategy_router.py` maps query types to preferred indexes:
   - `relationship` / `mechanism` → `sentence`
   - everything else → `semantic`
4. Manual override is available in the UI.

**Files:** `build_indexes.py`, `numpy_vector_store.py`, `switchable_vector_store.py`, `strategy_router.py`

**UI toggles:** `Enable Multi-Index Routing`, `Manual index override`

---

### Phase 6 — HNSW Search

**What problem does it solve?**
Exact NumPy search over ~15k × 384 embeddings is fine for this scale, but HNSW provides sub-linear approximate search and lower latency.

**How it works:**
1. `scripts/build_hnsw_indexes.py` builds `hnswlib` indexes offline for each index:
   - `M=16`, `ef_construction=200`, `ef_search=100`
2. `HnswVectorStore` loads the `.bin` index.
3. At query time, HNSW returns candidate IDs; exact cosine similarity is recomputed using the original `.npy` embeddings to preserve recall.
4. `SwitchableVectorStore` holds both NumPy and HNSW stores and selects based on `use_hnsw`.

**Note:** `hnswlib` has no Windows wheel, so local Windows dev falls back to NumPy. Streamlit Cloud Linux installs `hnswlib` normally.

**Files:** `build_hnsw_indexes.py`, `hnsw_vector_store.py`, `switchable_vector_store.py`

**UI toggle:** `⚡ Enable HNSW Search`

---

### Phase 7 — Streaming Sources / Citations

**What problem does it solve?**
In a RAG UI, users should see **which sources grounded the answer** before the LLM finishes generating text. This proves the answer is retrieval-based, not hallucinated.

**How it works:**
1. Domain events (`src/domain/entities/stream_events.py`):
   - `RetrievalStarted`
   - `ChunksFound`
   - `GraphEvidenceFound`
   - `TextChunkEvent`
   - `StreamComplete`
   Each event has a `timestamp`.
2. `RetrieveAndGenerateStreamUseCase` is a Python generator that:
   - Yields `RetrievalStarted`
   - Runs vector/hybrid search + graph expansion + metadata boost
   - Yields `ChunksFound`
   - Yields `GraphEvidenceFound`
   - Calls `llm_client.stream_answer(prompt)` and yields `TextChunkEvent` for each token
   - Yields `StreamComplete`
3. Streamlit UI iterates the generator and renders each event as it arrives:
   - Sources appear first.
   - Graph evidence appears second.
   - Answer tokens stream in real time.
   - `🕒 Event Sequence` table proves timing.

**Files:** `stream_events.py`, `retrieve_and_generate_stream.py`, `llm_client.py`, `demo.py`

**UI toggle:** `🌊 Enable Streaming Mode`

---

## 5. Retrieval pipeline internals

### RetrieveDocumentsUseCase

```python
apply_strategy(query, config)
    → vector_search.execute(query, config, index_name=..., use_hnsw=...)
    → [if hybrid] BM25 + RRF
    → graph_expand.execute(seed_chunks, config)
    → rerank.execute(search_results, expanded, config)
    → [if metadata boost] metadata_boost_service.apply_boost(...)
    → return top results
```

### Streaming variant

`RetrieveAndGenerateStreamUseCase` does the same thing but yields events instead of returning a list. It uses `GenerateAnswerUseCase._build_prompt()` to format context with chunk IDs and scores, then streams the LLM answer.

---

## 6. The LLM clients

| Client | Mode | Behavior |
|---|---|---|
| `MockLLMClient` | `mock` | Extractive QA: picks the best sentence from top chunks and returns labeled bullets + source IDs. Good for demos without API keys. |
| `OpenAIClient` | `openai` | Chat completions (`gpt-3.5-turbo` or `LLM_MODEL`). Falls back to mock on failure. Now uses streaming. |
| `OllamaClient` | `ollama` | Calls local Ollama `/api/generate`. Uses streaming. |

All three implement:
- `complete(prompt) -> str`
- `stream_answer(prompt) -> Iterator[str]`

---

## 7. Bootstrap / Deployment

`src/bootstrap/bootstrap_artifacts.py` downloads 16 artifacts from a GitHub Release at runtime:
- 3 chunk JSONLs
- 3 embedding `.npy` files
- 3 graph CSVs
- 3 HNSW `.bin` files
- 3 HNSW chunk-id JSONs
- 1 `manifest.json`

This makes the Streamlit Cloud deployment self-contained: the repo only holds code; data is fetched on first run.

---

## 8. Configuration

`src/config.py` defines:
- `EmbeddingConfig` — provider, model, token, service URL, timeout
- `RetrievalConfig` — all retrieval hyperparameters
- `Neo4jConfig` — optional future DB
- `AppConfig` — top-level container

`SearchConfig` in `src/application/dto/search_config.py` is the request-scoped DTO passed into use cases.

---

## 9. Testing

- 115 unit tests across the project.
- Key test files:
  - `test_bm25_retriever.py`
  - `test_rrf_fusion_service.py`
  - `test_query_classifier.py`
  - `test_strategy_router.py`
  - `test_metadata_boost_service.py`
  - `test_hnsw_vector_store.py`
  - `test_remote_embedding_client.py`
  - `test_retrieve_and_generate_stream.py`

---

## 10. Known issues we already fixed

1. **Mock LLM said "Insufficient evidence" too often**
   - Fixed by lowering the top-chunk score threshold from 0.55 to 0.05.

2. **HuggingFace endpoint was unreachable**
   - Fixed by migrating to `router.huggingface.co/hf-inference/models/{model}/pipeline/feature-extraction`.

3. **HF_API_TOKEN leaked into UI/logs**
   - Fixed via `scrub_secrets()` and hiding the token from `EmbeddingConfig` repr.

4. **Streamlit CORS/Xsrf warning**
   - Fixed by removing conflicting `enableCORS = false`.

5. **`TRANSFORMERS_CACHE` deprecation warning**
   - Fixed by removing the deprecated env var.

6. **HNSW fallback printed scary traceback on Windows**
   - Fixed by lowering log level to INFO.

7. **Missing `__init__.py` in several packages**
   - Fixed by adding them.

---

## 11. How to run locally

```bash
# Windows PowerShell
.venv_win\Scripts\activate
pytest tests/ -q
streamlit run scripts/demo.py

# WSL/Linux
source .venv/bin/activate
pytest tests/ -q
streamlit run scripts/demo.py
```

---

## 12. How to configure secrets

Create `.streamlit/secrets.toml` (do **not** commit it):

```toml
[llm]
openai = "sk-..."

[embedding]
provider = "huggingface_api"
hf_api_token = "hf_..."
model = "sentence-transformers/all-MiniLM-L6-v2"

[app]
artifact_base_url = "https://github.com/Slayer025/pubmed-graphrag-v2/releases/download/v2.1-hnsw"
```

---

## 13. What each UI control does

| Control | Effect |
|---|---|
| `⚡ Enable HNSW Search` | Uses pre-built HNSW index instead of exact NumPy search. |
| `Enable Hybrid Retrieval` | Adds BM25 keyword search and fuses with dense via RRF. |
| `Enable Metadata-Aware Boosting` | Boosts chunks whose entity labels match query keywords. |
| `Enable Query Understanding & Routing` | Classifies the query and picks a retrieval strategy + index. |
| `Enable Multi-Index Routing` | Allows the router to choose between semantic/fixed/sentence indexes. |
| `🌊 Enable Streaming Mode` | Streams sources, graph evidence, and answer tokens progressively. |
| `Enable query decomposition` | Breaks complex questions into sub-questions (disabled in streaming). |
| `Enable graph re-ranking` | Post-processes results using graph connectivity signals (disabled in streaming). |

---

## 14. What "Event Sequence" proves

The `🕒 Event Sequence` table shows:
- `RetrievalStarted` at T+0.00s
- `ChunksFound` at T+0.XXs (sources exist before answer)
- `GraphEvidenceFound` at T+0.YYs (graph evidence before answer)
- Many `TextChunkEvent` rows from T+0.ZZs onward
- `StreamComplete` at the end

This proves the system retrieves and displays grounding evidence **before** the LLM finishes generating the answer.
