# PubMed GraphRAG — Claude Context

**Last updated:** 2026-06-25
**Current state:** Phases 1–7 complete and deployed.
**Most recent commit:** `chore: fix warnings and package structure from project health check`

This file gives an AI assistant everything needed to resume work on the PubMed GraphRAG project without re-reading the entire repository.

---

## Deployment
- URL: https://pubmed-graphrag-kamfpkughsfmstpcrv8r23.streamlit.app/
- Repository: https://github.com/Slayer025/pubmed-graphrag-v2
- Release (artifacts): https://github.com/Slayer025/pubmed-graphrag-v2/releases/tag/v2.1-hnsw
- `ARTIFACT_BASE_URL`: `https://github.com/Slayer025/pubmed-graphrag-v2/releases/download/v2.1-hnsw`

### Required Streamlit secrets (`.streamlit/secrets.toml`)
```toml
[llm]
# openai = "sk-..."           # optional, for OpenAI generation

[embedding]
provider = "huggingface_api"  # or "local" / "remote_http"
hf_api_token = "hf_..."        # required when provider = "huggingface_api"
model = "sentence-transformers/all-MiniLM-L6-v2"

[app]
artifact_base_url = "https://github.com/Slayer025/pubmed-graphrag-v2/releases/download/v2.1-hnsw"
```
Environment variables are read as a fallback (`OPENAI_API_KEY`, `HF_API_TOKEN`, `EMBEDDING_PROVIDER`, etc.).

---

## Project Goal

Transition a PubMed semantic-search demo into an evaluated, production-grade retrieval system. The app must remain deployable on Streamlit Community Cloud and run in a single container with ephemeral storage and CPU-only inference.

---

## Repository Layout

```text
pubmed-graphrag/
├── src/
│   ├── bootstrap/                # DI container + artifact bootstrap
│   ├── domain/                   # Pure domain logic + entities
│   ├── application/              # Use cases, ports, DTOs
│   ├── infrastructure/           # Adapters (embeddings, retrievers, storage, vector stores, utils)
│   └── interfaces/               # Streamlit demo
├── tests/                        # Unit tests (115 passing)
├── evaluation/                   # 40 frozen queries + run_eval.py + result JSONLs
├── outputs/                      # retrieval_improvement_summary.json
├── docs/
│   └── metadata_inventory.md
├── scripts/
│   ├── demo.py                   # Streamlit entry point
│   ├── build_indexes.py          # Multi-index offline builder
│   └── build_hnsw_indexes.py     # HNSW offline builder
├── .streamlit/
│   ├── config.toml
│   └── secrets.toml.example
├── requirements.txt
├── runtime.txt                   # Python 3.11.9
├── README.md
├── CLAUDE_CONTEXT.md             # This file
└── .gitignore
```

---

## Architecture & Constraints

### Clean Architecture
- Domain logic lives in `src/domain/` (no infrastructure imports).
- Application layer lives in `src/application/` (depends only on domain + ports).
- Infrastructure lives in `src/infrastructure/` (implements ports).
- `src/bootstrap/__init__.py` is the **sole DI container**.

### Streamlit Cloud Rules
- Single container; no local FastAPI server.
- Ephemeral storage; artifacts bootstrapped from GitHub Release at runtime.
- CPU-only; use small models (`sentence-transformers/all-MiniLM-L6-v2`).
- No heavy NLP libraries (no `nltk`, `spacy`).

### Backwards Compatibility
- All new features are **disabled by default**.
- Opt-in via config flags.
- Existing tests must continue to pass.

---

## Implemented Phases

### ✅ Phase 1 — Hybrid Retrieval (Dense + Sparse + RRF)
- `src/infrastructure/retrievers/bm25_retriever.py` + `src/domain/services/rrf_fusion_service.py`
- `src/application/use_cases/retrieve_documents.py` fuses dense + sparse when `use_hybrid=True`.
- UI: "Enable Hybrid Retrieval" checkbox.
- Proof: `evaluation/results_dense_only.jsonl`, `evaluation/results_hybrid_k*.jsonl`.

### ✅ Phase 2 — Remote Embedding Service
- `src/infrastructure/embeddings/remote_embedding_client.py` supports `local`, `huggingface_api`, `remote_http`.
- Falls back to local model on remote failure.
- UI System Status panel shows provider, latency, fallback reason.
- Latest endpoint: `https://router.huggingface.co/hf-inference/models/{model}/pipeline/feature-extraction`.

### ✅ Phase 3 — Query Understanding Layer
- `src/domain/services/query_classifier.py` + `src/domain/services/strategy_router.py`
- `src/application/use_cases/retrieve_documents.py` wires classifier/router.
- UI: "Enable Query Understanding & Routing" + "🧠 Query Understanding" expander.

### ✅ Phase 4 — Metadata-Aware Retrieval
- `src/domain/services/metadata_boost_service.py` + `src/application/use_cases/metadata_boost.py`
- Boosts `combined_score` when query keywords match chunk entity labels.
- UI: "Enable Metadata-Aware Boosting".

### ✅ Phase 5 — Multiple Embedding Indexes
- `scripts/build_indexes.py` builds `semantic`, `fixed`, `sentence` indexes.
- `SwitchableVectorStore` loads all three; routing chooses per query.
- UI: "Enable Multi-Index Routing" + manual index override.

### ✅ Phase 6 — HNSW Search
- `scripts/build_hnsw_indexes.py` builds hnswlib `.bin` indexes.
- Runtime HNSW/NumPy switching via `SwitchableVectorStore`.
- UI: "⚡ Enable HNSW Search" + backend caption.

### ✅ Phase 7 — Streaming Sources/Citations
- `src/domain/entities/stream_events.py` — events with timestamps.
- `src/application/use_cases/retrieve_and_generate_stream.py` — generator-based pipeline.
- LLM clients implement `stream_answer()`.
- UI: "🌊 Enable Streaming Mode" + live sources/graph evidence/answer tokens + "🕒 Event Sequence" proof table.

---

## Key Config Flags

| Flag | Default | Meaning |
|------|---------|---------|
| `use_hybrid` | `False` | Dense + BM25 + RRF |
| `rrf_k` | `20` | RRF damping constant |
| `enable_query_routing` | `False` | Classifier + strategy router |
| `enable_metadata_boost` | `False` | Entity-label boosting |
| `metadata_boost_factor` | `1.1` | Score multiplier when labels match |
| `default_index` | `semantic` | Default vector index name |
| `enable_multi_index` | `False` | Multi-index / routing |
| `index_name` | `None` | Manual index override |
| `use_hnsw` | `False` | Use pre-built HNSW indexes |

---

## Known Issues Already Fixed

1. **Mock LLM "Insufficient evidence"** — Fixed by lowering `_MOCK_MIN_TOP_CHUNK_SCORE` from `0.55` to `0.05`.
2. **HuggingFace Inference endpoint deprecation** — Fixed by migrating to `router.huggingface.co`.
3. **HF_API_TOKEN leakage in logs/UI** — Fixed via `scrub_secrets()` and `repr=False` on `EmbeddingConfig.api_token`.
4. **Streamlit CORS/Xsrf warning** — Fixed by removing conflicting `enableCORS = false`.
5. **TRANSFORMERS_CACHE deprecation warning** — Fixed by removing the deprecated env var.
6. **Noisy HNSW fallback traceback on Windows** — Fixed by lowering log level to INFO.
7. **Missing `__init__.py` files** — Fixed in `src/application/dto` and several `src/infrastructure/*` packages.

---

## How to Resume Work

```bash
# Windows (primary dev environment in this repo)
.venv_win\Scripts\activate
pytest tests/ -q
streamlit run scripts/demo.py

# Evaluation examples
python evaluation/run_eval.py              # dense-only baseline
python evaluation/run_eval.py --hybrid       # hybrid retrieval
python evaluation/run_eval.py --routed       # query routing
python evaluation/run_eval.py --metadata-boost
python evaluation/run_eval.py --multi-index --hybrid
python evaluation/run_eval.py --hnsw --hybrid
```

---

## Notes for Future AI Assistants

- The DI container is `src/bootstrap/__init__.py`. Do not instantiate infrastructure elsewhere.
- The `pure_build_guard` blocks filesystem writes during pipeline construction.
- Entity IDs in the graph are formatted as `label:name`.
- Evaluation metrics are low because the 40-query set is a random sample. Prefer reproducible before/after comparisons.
- `scripts/build_indexes.py` and `scripts/build_hnsw_indexes.py` are strictly offline; never run them inside the Streamlit runtime.
- HNSW is Linux-only in this project (`hnswlib` has no Windows wheel). Windows dev falls back to NumPy search.
