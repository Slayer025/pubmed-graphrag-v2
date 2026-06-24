# PubMed GraphRAG Improvement Plan

## 1. Objective

The current PubMed GraphRAG project already supports semantic chunking, embeddings, graph-enhanced retrieval, evaluation, LLM generation, and a Streamlit demo. The next goal is to improve retrieval quality beyond the current dense/vector-based approach and make the improvements visible through logs, comparisons, and Streamlit UI evidence.

The focus of this improvement phase is:

1. Hybrid Retrieval (Dense + Sparse + RRF)
2. Remote Embedding Service (Decoupled API)
3. Query Understanding Layer (Intent Routing)
4. Streamlit Cloud compatible deployment

The final application must continue to run on Streamlit Community Cloud using the existing Streamlit entrypoint, repository artifacts, and lightweight runtime configuration. The ultimate goal is to transition the project from a "proof-of-concept demo" to an **evaluated, production-grade retrieval system**.

---

## 2. Current System Summary

The current retrieval flow is:

```text
User Question
    ↓
Query Embedding (Local sentence-transformers)
    ↓
Vector Search over semantic_embeddings.npy
    ↓
Offline Graph Expansion using graph CSV artifacts
    ↓
Re-ranking & Deduplication
    ↓
Retrieved Chunks
    ↓
LLM / Mock Answer
```

This works, but it has specific limitations:

* **Exact Match Failure:** Dense retrieval struggles with exact biomedical terms (drug names, gene symbols like *BRCA1*, abbreviations, MeSH codes) because it relies on semantic proximity rather than exact lexical overlap.
* **Static Retrieval:** All queries are treated with the exact same retrieval behavior, regardless of whether the user is asking for a simple definition or a complex multi-entity relationship.
* **Tight Coupling:** Embedding generation is tightly connected to the application flow, making it difficult to scale or swap models without restarting the app.
* **Lack of Observability:** The Streamlit UI does not clearly prove *why* one retrieval strategy is better than another; it just shows the final answer.

---

## 3. Target Improved Architecture

The new retrieval flow introduces a routing layer and a dual-retrieval mechanism:

```text
User Question
    ↓
[1] Query Understanding Layer (Classifier)
    ↓
[2] Strategy Router (Selects: dense_only, hybrid_rrf, graph_expand, etc.)
    ↓
[3] Dual Retrieval Engine
    ├─→ Dense Retrieval (Vector Search)
    └─→ Sparse Retrieval (BM25 Keyword Search)
    ↓
[4] Reciprocal Rank Fusion (RRF)
    ↓
[5] Optional Graph Expansion / Re-ranking
    ↓
[6] Evidence Chunks + Retrieval Logs
    ↓
[7] Answer Generation
    ↓
[8] Streamlit Debug View (Proves the improvement)
```

The goal is not only to generate answers, but to **prove that better context is being retrieved** through transparent logging and UI evidence.

---

## 4. Phase 1: Hybrid Retrieval

### Goal

Add sparse keyword retrieval alongside existing dense retrieval, then combine both result sets using Reciprocal Rank Fusion (RRF). Dense retrieval is useful for semantic similarity; sparse retrieval is useful for exact biomedical terms.

### Planned Changes

Add a BM25-based sparse retriever that uses the existing semantic chunks file.

**New or updated files:**

```text
src/infrastructure/retrievers/bm25_retriever.py
src/domain/services/rrf_fusion_service.py
src/application/use_cases/retrieve_documents.py
requirements.txt (add rank-bm25)
tests/test_bm25_retriever.py
tests/test_rrf_fusion_service.py
```

### Implementation Details

**The BM25 Retriever:**

* Load chunk text from `data/chunks/chunks_semantic.jsonl.gz` (or the cached artifact).
* **Streamlit Constraint:** The BM25 index must be initialized using `@st.cache_resource` to prevent reloading the massive index on every UI interaction.
* Tokenize chunk text and user query using a simple whitespace/lowercase tokenizer.
* Return top-k chunks with BM25 scores.

**The RRF Fusion Service:**

* Accept dense results and sparse results.
* Apply the RRF formula: `rrf_score = Σ 1 / (k + rank)` (Standard constant: `k = 60`).
* Rank by reciprocal rank instead of raw score (this avoids the problem of comparing cosine similarity scores with BM25 scores).
* Merge duplicate chunks and return a final ranked list.

### Expected Result

Hybrid retrieval should improve retrieval for biomedical entity-heavy questions.

**Example comparison:**

```text
Query: "Does metformin reduce cardiovascular risk in type 2 diabetes?"

Dense-only:
1. General diabetes management article
2. Obesity treatment article
3. Insulin resistance article

Hybrid (Dense + BM25 + RRF):
1. Exact metformin-related cardiovascular outcomes article (Boosted by BM25 matching "metformin")
2. Insulin resistance article
3. Diabetes treatment article
```

---

## 5. Phase 2: Remote Embedding Service

### Goal

Decouple embedding generation from the retrieval layer. The app should no longer depend only on directly loading the heavy `sentence-transformers` model inside the main Streamlit flow.

### Planned Changes

**New or updated files:**

```text
src/infrastructure/embeddings/remote_embedding_client.py
src/config.py
.streamlit/secrets.toml.example
requirements.txt (add httpx)
```

### Provider Modes

The embedding client will support three modes via configuration:

1. `local`: Uses the existing `sentence-transformers` model (fallback/development).
2. `remote_http`: Calls a custom FastAPI embedding service.
3. `huggingface_api`: Calls the HuggingFace Inference API directly.

### Streamlit-Compatible Rule

**Crucial:** The deployed Streamlit app runs in a single container. It cannot assume a secondary FastAPI server is running locally.

* For Streamlit Cloud, `huggingface_api` is the easiest "remote" proof.
* If using `remote_http`, the service must be deployed externally (e.g., Render, Fly.io).

### Configuration

The app should read embedding configuration from Streamlit secrets:

```toml
# .streamlit/secrets.toml.example
EMBEDDING_PROVIDER = "huggingface_api" # Options: local, remote_http, huggingface_api
EMBEDDING_SERVICE_URL = "https://your-custom-api.com/embed"
HF_API_TOKEN = "hf_xxxxxxxxxxxx"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
```

**Error Handling:** If the remote API fails or times out, the client must gracefully fallback to `local` mode (if the model is cached) or return a clear UI error, **never crashing the Streamlit app**.

### Expected Result

The UI and logs should show:

```text
Embedding provider: huggingface_api
Embedding model: all-MiniLM-L6-v2
Embedding latency: 120 ms
Retrieval latency: 90 ms
```

---

## 6. Phase 3: Query Understanding Layer

### Goal

Add a lightweight classifier before retrieval so different query types can use different retrieval strategies.

### Query Types & Strategy Mapping

The first version will use a simple, rule-based keyword/regex classifier.

| Query Type | Trigger Keywords | Selected Strategy |
| :--- | :--- | :--- |
| `definition` | "what is", "define" | `dense_only` |
| `entity_specific` | "gene", "mutation", "BRCA" | `hybrid_rrf` |
| `relationship` | "associated with", "linked to" | `hybrid_rrf` + `graph_expansion` |
| `mechanism` | "pathway", "how does" | `dense_only` + `graph_expansion` |
| `comparison` | "compare", "versus", "vs" | `hybrid_rrf` |
| `general` | (default) | `hybrid_rrf` |

### Planned Changes

**New or updated files:**

```text
src/domain/services/query_classifier.py
src/domain/services/strategy_router.py
src/interfaces/streamlit/demo.py (UI updates)
```

### Streamlit UI Changes

The Streamlit app should display the routing logic transparently. Use `st.expander("🧠 Query Understanding & Routing")` to show:

* Detected query type
* Selected retrieval strategy
* Reason for selection (e.g., "Matched keyword: 'associated with'")

### Expected Result

For a query like: *"Is EGFR associated with lung cancer treatment response?"*
The app should show:

```text
Query type: relationship
Selected strategy: hybrid_rrf + graph_expansion
Reason: Matched phrase "associated with" + detected entity "EGFR"
```

---

## 7. Evaluation Plan

The project must prove improvements using before/after retrieval metrics.

### Evaluation Files

```text
evaluation/queries.jsonl         # 30-50 frozen test questions from PubMedQA
evaluation/run_eval.py           # CLI script to run the evaluation
evaluation/results_dense_only.jsonl
evaluation/results_hybrid.jsonl
evaluation/results_routed.jsonl
outputs/retrieval_improvement_summary.json
```

### Metrics

Required metrics to calculate and log:

* **Recall@5** and **Recall@10**: Did the correct `pubmed_id` appear in the top chunks?
* **MRR@10**: Mean Reciprocal Rank (was the correct result ranked early?)
* **Average & p95 Latency**: Speed impact of the new layers.

### Comparison Table

The final report in `outputs/retrieval_improvement_summary.json` should generate a table like:

```text
Mode                  Recall@5   Recall@10   MRR@10   Avg Latency
Dense only            0.52       0.70        0.31     180 ms
Hybrid RRF            0.68       0.82        0.46     260 ms
Query routed hybrid   0.72       0.85        0.50     285 ms
```

---

## 8. Streamlit Cloud Deployment Plan

The app will be deployed through Streamlit Community Cloud.

### Entry Point & Dependencies

* **Entry Point:** `scripts/demo.py`
* **Dependencies:** Add `rank-bm25` and `httpx` to `requirements.txt`. Avoid heavy new dependencies.

### Secrets Configuration

Do not commit real secrets. Configure these in the Streamlit Cloud dashboard:

```text
OPENAI_API_KEY
EMBEDDING_PROVIDER
HF_API_TOKEN
```

*(Note: Ollama is treated as optional/ignored on Cloud since it requires a local GPU server).*

---

## 9. UI Evidence Requirements

The deployed app must make improvements visible without needing terminal access.

Add a dedicated **"🔍 Retrieval Debug & Evidence"** tab or sidebar expander in `src/interfaces/streamlit/demo.py` showing:

1. **Query Classification:** `st.metric` or `st.info` showing the detected intent.
2. **Retrieval Breakdown:** Use `st.tabs` to show:
   * *Tab 1:* Top 3 Dense Results
   * *Tab 2:* Top 3 Sparse (BM25) Results
   * *Tab 3:* Final Fused (RRF) Results
3. **Latency Breakdown:** A simple bar chart or metrics row showing Embedding time vs. Retrieval time vs. LLM time.

This makes the demo highly persuasive for technical reviews.

---

## 10. Final Deliverables

After implementation, the repository should contain:

```text
src/infrastructure/retrievers/bm25_retriever.py
src/domain/services/rrf_fusion_service.py
src/infrastructure/embeddings/remote_embedding_client.py
src/domain/services/query_classifier.py
src/domain/services/strategy_router.py
evaluation/results_dense_only.jsonl
evaluation/results_hybrid.jsonl
evaluation/results_routed.jsonl
outputs/retrieval_improvement_summary.json
```

---

## 11. Implementation Order

Recommended order for AI coding assistants:

1. Add BM25 retriever (with `@st.cache_resource`).
2. Add RRF fusion service.
3. Compare dense-only vs hybrid (run eval).
4. Add embedding client abstraction (Local/Remote/HF).
5. Add query classifier and strategy router.
6. Update Streamlit UI with the Debug/Evidence panel.
7. Run full evaluation suite.
8. Deploy to Streamlit Cloud and verify secrets.

---

## 12. Success Criteria

The improvement phase is successful when:

* [ ] The current Streamlit app still works and deploys without crashing.
* [ ] Dense-only retrieval remains available as a baseline.
* [ ] Hybrid retrieval successfully fuses dense + sparse results using RRF.
* [ ] Query classification and routing are visibly proven in the Streamlit UI.
* [ ] Remote embedding mode can be configured via secrets and works on Cloud.
* [ ] Evaluation files definitively show before/after retrieval metric improvements.
* [ ] The app respects Streamlit's single-container constraint (no local FastAPI assumptions).
