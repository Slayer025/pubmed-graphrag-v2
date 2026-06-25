"""Streamlit interface for the PubMed GraphRAG pipeline."""

from __future__ import annotations

import csv
import io
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

# Repo root (…/pubmed-graphrag), not src/.
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.bootstrap.environment import configure_environment

configure_environment()

CACHE_DIR = os.environ.get("ARTIFACT_CACHE_DIR", "").strip() or "/tmp/pubmed-graphrag"
HF_HOME = os.environ.get("HF_HOME", "/tmp/hf_cache")

from src.bootstrap.bootstrap_artifacts import bootstrap_artifacts, is_bootstrap_complete, mark_streamlit_runtime

if not is_bootstrap_complete():
    try:
        bootstrap_artifacts(CACHE_DIR)
    except RuntimeError as exc:
        print(f"ARTIFACT BOOTSTRAP FAILED: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

try:
    import streamlit as st
except ImportError as exc:
    print(
        "Streamlit is not installed. Install it with: pip install streamlit\n"
        f"Original error: {exc}",
        file=sys.stderr,
    )
    raise SystemExit(1)

mark_streamlit_runtime()

from src.application.dto.rerank_config import RerankConfig
from src.application.dto.search_config import SearchConfig
from src.application.use_cases.generate_answer import GenerateAnswerUseCase
from src.application.use_cases.retrieve_and_generate_stream import (
    RetrieveAndGenerateStreamUseCase,
)
from src.application.use_cases.retrieve_documents import RetrieveDocumentsUseCase
from src.bootstrap import build_pipeline, default_search_config
from src.domain.entities.stream_events import (
    ChunksFound,
    GraphEvidenceFound,
    RetrievalStarted,
    StreamComplete,
    TextChunkEvent,
)
from src.bootstrap.bootstrap_artifacts import get_preloaded_artifacts
from src.config import AppConfig
from src.infrastructure.embeddings.remote_embedding_client import create_embedding_client
from src.domain.entities.retrieval_result import RetrievalResult
from src.domain.value_objects.query import Query
from src.graph_reranker import GraphReranker
from src.llm_client import (
    LLM_MODE_MOCK,
    LLM_MODE_OPENAI,
    UNABLE_TO_GENERATE_ANSWER,
    create_llm_client_with_mode,
    log_llm_startup_diagnostics,
    safe_llm_complete,
)
from src.query_decomposer import DecomposerConfig, QueryDecomposer
from src.rag_pipeline import RAGPipeline

logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


@st.cache_resource(show_spinner=False)
def get_pipeline(hf_home: str) -> RAGPipeline:
    """Bootstrap the heavy retrieval stack once per session (pure, no IO)."""
    print("PIPELINE BUILD START (PURE)", flush=True)
    logger.info("PIPELINE BUILD START (PURE)")
    pipeline = build_pipeline(hf_home=hf_home, artifacts=get_preloaded_artifacts())
    print("PIPELINE BUILD END (PURE)", flush=True)
    logger.info("PIPELINE BUILD END (PURE)")
    print("PIPELINE INIT CALLED", flush=True)
    logger.info("PIPELINE INIT CALLED")
    return pipeline


def _build_search_config(base: SearchConfig, overrides: dict[str, Any]) -> SearchConfig:
    """Build a request-scoped ``SearchConfig`` from UI overrides."""
    use_hnsw = overrides.get("use_hnsw", base.use_hnsw)
    logger.info("BUILD CONFIG: use_hnsw = %s", use_hnsw)
    return SearchConfig(
        top_k=overrides.get("top_k", base.top_k),
        expand_depth=overrides.get("expand_depth", base.expand_depth),
        max_entity_degree=overrides.get("max_entity_degree", base.max_entity_degree),
        max_expansion_per_entity=overrides.get(
            "max_expansion_per_entity", base.max_expansion_per_entity
        ),
        max_expanded_nodes=overrides.get("max_expanded_nodes", base.max_expanded_nodes),
        alpha=overrides.get("alpha", base.alpha),
        depth_scores=base.depth_scores,
        max_results=overrides.get("max_results", base.max_results),
        use_hnsw=use_hnsw,
        use_hybrid=overrides.get("use_hybrid", base.use_hybrid),
        rrf_k=overrides.get("rrf_k", base.rrf_k),
        enable_query_routing=overrides.get("enable_query_routing", base.enable_query_routing),
        enable_metadata_boost=overrides.get("enable_metadata_boost", base.enable_metadata_boost),
        metadata_boost_factor=overrides.get(
            "metadata_boost_factor", base.metadata_boost_factor
        ),
        default_index=overrides.get("default_index", base.default_index),
        enable_multi_index=overrides.get("enable_multi_index", base.enable_multi_index),
        index_name=overrides.get("index_name", base.index_name),
    )


def _maybe_rerank(
    graph_repository: Any,
    query: str,
    results: list[RetrievalResult],
    *,
    enabled: bool,
    beta: float,
) -> list[RetrievalResult]:
    if not enabled:
        return results
    reranker = GraphReranker(
        index=graph_repository,
        config=RerankConfig(enabled=True, beta=beta),
    )
    return reranker.rerank(query, results)


def _retrieve_results(
    retrieve_documents: RetrieveDocumentsUseCase,
    graph_repository: Any,
    query: str,
    search_config: SearchConfig,
    *,
    llm_client_type: str,
    use_reranker: bool,
    reranker_beta: float,
    use_decomposer: bool,
) -> tuple[list[str], list[RetrievalResult], dict, dict]:
    classification: dict = {}
    strategy: dict = {}

    def _unpack_results(
        raw: list[RetrievalResult] | tuple[list[RetrievalResult], dict, dict],
    ) -> tuple[list[RetrievalResult], dict, dict]:
        if isinstance(raw, tuple):
            return raw[0], raw[1], raw[2]
        return raw, {}, {}

    if use_decomposer:
        llm = create_llm_client_with_mode(llm_client_type).client
        decomposer = QueryDecomposer(llm=llm, config=DecomposerConfig(enabled=True))
        sub_queries = decomposer.decompose(query)
        if len(sub_queries) <= 1:
            results, classification, strategy = _unpack_results(
                retrieve_documents.execute(Query(query), search_config)
            )
            results = _maybe_rerank(
                graph_repository,
                query,
                results,
                enabled=use_reranker,
                beta=reranker_beta,
            )
            return sub_queries, results, classification, strategy

        best_by_chunk: dict[str, RetrievalResult] = {}
        for sub_query in sub_queries:
            sub_results, sub_classification, sub_strategy = _unpack_results(
                retrieve_documents.execute(Query(sub_query), search_config)
            )
            sub_results = _maybe_rerank(
                graph_repository,
                sub_query,
                sub_results,
                enabled=use_reranker,
                beta=reranker_beta,
            )
            for result in sub_results:
                existing = best_by_chunk.get(result.chunk_id)
                if existing is None or result.combined_score > existing.combined_score:
                    best_by_chunk[result.chunk_id] = result
            classification = sub_classification
            strategy = sub_strategy

        merged = sorted(best_by_chunk.values(), key=lambda r: r.combined_score, reverse=True)
        return sub_queries, merged[: search_config.max_results], classification, strategy

    results, classification, strategy = _unpack_results(
        retrieve_documents.execute(Query(query), search_config)
    )
    results = _maybe_rerank(
        graph_repository,
        query,
        results,
        enabled=use_reranker,
        beta=reranker_beta,
    )
    return [query], results, classification, strategy


def _results_to_csv(results: list[RetrievalResult]) -> str:
    output = io.StringIO()
    writer = csv.DictWriter(
        output,
        fieldnames=[
            "rank",
            "chunk_id",
            "article_id",
            "source",
            "depth",
            "vector_score",
            "graph_score",
            "combined_score",
            "text",
        ],
    )
    writer.writeheader()
    for rank, result in enumerate(results, start=1):
        writer.writerow(
            {
                "rank": rank,
                "chunk_id": result.chunk_id,
                "article_id": result.article_id,
                "source": result.source,
                "depth": result.depth,
                "vector_score": f"{result.vector_score:.4f}",
                "graph_score": f"{result.graph_score:.4f}",
                "combined_score": f"{result.combined_score:.4f}",
                "text": result.text,
            }
        )
    return output.getvalue()


def _render_result_card(rank: int, result: RetrievalResult) -> None:
    with st.expander(
        f"#{rank} {result.chunk_id} | {result.source} | score={result.combined_score:.4f}"
    ):
        st.markdown(
            f"""
            **Article:** `{result.article_id}`  
            **Source:** `{result.source}`  
            **Depth:** `{result.depth}`  
            **Vector score:** `{result.vector_score:.4f}`  
            **Graph score:** `{result.graph_score:.4f}`  
            **Combined score:** `{result.combined_score:.4f}`
            """
        )
        st.markdown(f"> {result.text}")


def _openai_api_key_available() -> bool:
    return bool((os.environ.get("OPENAI_API_KEY") or "").strip())


def _active_llm_mode(selection: Any) -> str:
    """Return the effective LLM mode from an instantiated client selection."""
    return selection.mode


@st.cache_data(show_spinner=False)
def _probe_embedding_client(hf_home: str) -> dict[str, Any]:
    """Probe the configured embedding provider and return runtime diagnostics."""
    cfg = AppConfig.default().embedding
    result = create_embedding_client(
        provider=cfg.provider,
        model_name=cfg.model_name,
        api_token=cfg.api_token,
        service_url=cfg.service_url,
        batch_size=cfg.batch_size,
        normalize=cfg.normalize,
        timeout_seconds=cfg.timeout_seconds,
        cache_folder=hf_home,
    )
    client = result.client
    t0 = time.perf_counter()
    try:
        client.embed_query("diagnostic probe")
        latency_ms = (time.perf_counter() - t0) * 1000
        error = None
    except Exception as exc:
        latency_ms = (time.perf_counter() - t0) * 1000
        error = f"{type(exc).__name__}: {exc}"

    return {
        "provider": client.provider,
        "selected_provider": result.selected_provider,
        "model_name": client.model_name,
        "fallback_reason": client.fallback_reason,
        "latency_ms": latency_ms,
        "error": error,
    }


def _render_embedding_diagnostics(hf_home: str) -> None:
    """Display embedding provider status in the sidebar."""
    st.header("System Status")
    with st.expander("Embedding provider diagnostics", expanded=True):
        try:
            diagnostics = _probe_embedding_client(hf_home)
        except Exception as exc:
            st.error(f"Could not probe embedding client: {exc}")
            return

        st.markdown(f"**Active provider:** `{diagnostics['provider']}`")
        st.markdown(f"**Requested provider:** `{diagnostics['selected_provider']}`")
        st.markdown(f"**Model:** `{diagnostics['model_name']}`")
        st.markdown(f"**Probe latency:** `{diagnostics['latency_ms']:.1f} ms`")

        if diagnostics["fallback_reason"]:
            st.warning(f"Fallback: {diagnostics['fallback_reason']}")

        if diagnostics["error"]:
            st.error(f"Probe failed: {diagnostics['error']}")
        elif diagnostics["provider"] != diagnostics["selected_provider"]:
            st.info("The active provider differs from the requested provider due to fallback.")


def _render_llm_mode_sidebar() -> tuple[str, Any]:
    """Render LLM controls and return selected type plus instantiated client."""
    llm_options = ["mock"]
    if _openai_api_key_available():
        llm_options.append("openai")
    if os.environ.get("OLLAMA_URL"):
        llm_options.append("ollama")

    if "llm_client_select" not in st.session_state:
        st.session_state.llm_client_select = "mock"
    if st.session_state.llm_client_select == "openai" and not _openai_api_key_available():
        st.session_state.llm_client_select = "mock"
    if st.session_state.llm_client_select not in llm_options:
        st.session_state.llm_client_select = "mock"

    llm_client_type = st.selectbox(
        "LLM client",
        options=llm_options,
        index=llm_options.index(st.session_state.llm_client_select),
        help="Select the LLM used for generation. OpenAI appears only when OPENAI_API_KEY is set.",
        key="llm_client_select",
    )
    llm_selection = create_llm_client_with_mode(llm_client_type)
    effective_llm_mode = _active_llm_mode(llm_selection)
    st.caption(f"Active LLM mode: `{effective_llm_mode}`")
    if not _openai_api_key_available():
        st.caption("Add `OPENAI_API_KEY` in Streamlit secrets to enable OpenAI.")
    if (
        llm_client_type == LLM_MODE_OPENAI
        and effective_llm_mode == LLM_MODE_MOCK
        and llm_selection.fallback_reason
    ):
        st.warning("OpenAI initialization failed. Running in mock mode.")
        logger.warning("OpenAI fallback reason: %s", llm_selection.fallback_reason)
    return llm_client_type, llm_selection


def _generate_answer_safe(
    query: str,
    results: list[RetrievalResult],
    llm: Any,
) -> str:
    """Generate an answer without propagating LLM exceptions to the UI."""
    prompt = GenerateAnswerUseCase._build_prompt(query, results)
    return safe_llm_complete(llm, prompt)


def _render_query_understanding(
    classification: dict,
    strategy: dict,
) -> None:
    """Display query classification and routing decisions in the UI."""
    if not classification and not strategy:
        return

    with st.expander("🧠 Query Understanding", expanded=True):
        if classification:
            st.markdown(f"**Query type:** `{classification.get('query_type', 'general')}`")
            keywords = classification.get("matched_keywords", [])
            if keywords:
                st.markdown(f"**Matched keywords:** `{', '.join(keywords)}`")
            entities = classification.get("detected_entities", [])
            if entities:
                st.markdown(f"**Detected entities:** `{', '.join(entities)}`")
        if strategy:
            st.markdown(f"**Selected strategy:** `{strategy.get('strategy_name', 'unknown')}`")
            selected_index = strategy.get("index_name") or "semantic"
            st.markdown(f"**Selected index:** `{selected_index}`")
            st.markdown(f"**Reason:** {strategy.get('reason', '')}")


def _render_graph_evidence(graph_repository: Any, results: list[RetrievalResult]) -> None:
    st.subheader("Graph evidence")
    if not results:
        st.write("No results to visualize.")
        return

    top = results[:5]
    entity_counts: dict[str, int] = {}
    for result in top:
        for entity_id in graph_repository.get_chunk_entities(result.chunk_id):
            entity_counts[entity_id] = entity_counts.get(entity_id, 0) + 1

    if not entity_counts:
        st.write("No shared entities found for the top results.")
        return

    sorted_entities = sorted(entity_counts.items(), key=lambda kv: kv[1], reverse=True)[:10]
    st.write("Top entities mentioned by the top 5 retrieved chunks:")
    for entity_id, count in sorted_entities:
        degree = graph_repository.get_entity_degree(entity_id)
        st.write(f"- `{entity_id}` (mentions={count}, degree={degree})")


def _build_streaming_use_case(
    retrieve_documents: RetrieveDocumentsUseCase,
    llm_client: Any,
) -> RetrieveAndGenerateStreamUseCase:
    """Build the streaming use case from the synchronous retrieval stack."""
    return RetrieveAndGenerateStreamUseCase(
        vector_search=retrieve_documents.vector_search,
        llm_client=llm_client,
        chunk_repository=retrieve_documents.rerank.chunk_repository,
        graph_repository=retrieve_documents.graph_expand.graph_repository,
        sparse_retriever=retrieve_documents.sparse_retriever,
        rrf_fusion_service=retrieve_documents.rrf_fusion_service,
        query_classifier=retrieve_documents.query_classifier,
        strategy_router=retrieve_documents.strategy_router,
        metadata_boost_service=retrieve_documents.metadata_boost_service,
    )


def _render_streaming_graph_evidence(
    graph_repository: Any,
    entities: list[dict],
) -> None:
    """Render graph evidence emitted by the streaming use case."""
    st.subheader("Graph evidence")
    if not entities:
        st.write("No shared entities found for the top results.")
        return

    entity_counts: dict[str, int] = {}
    for entity in entities:
        entity_counts[entity["entity_id"]] = entity_counts.get(entity["entity_id"], 0) + 1

    sorted_entities = sorted(entity_counts.items(), key=lambda kv: kv[1], reverse=True)[:10]
    st.write("Top entities mentioned by the top retrieved chunks:")
    for entity_id, count in sorted_entities:
        degree = graph_repository.get_entity_degree(entity_id)
        st.write(f"- `{entity_id}` (mentions={count}, degree={degree})")


def main() -> int:
    st.set_page_config(page_title="PubMed GraphRAG", layout="wide")
    st.title("🧬 PubMed GraphRAG")
    st.markdown(
        "Ask a biomedical question. The demo retrieves semantic chunks from 5,000 "
        "PubMed abstracts using graph-enhanced retrieval."
    )

    if "llm_startup_logged" not in st.session_state:
        st.session_state.llm_startup_logged = False

    with st.sidebar:
        _render_embedding_diagnostics(HF_HOME)

        with st.expander("🔍 Retrieval Strategy", expanded=True):
            use_hnsw = st.checkbox(
                "⚡ Enable HNSW Search",
                value=False,
                help="Uses pre-built HNSW approximate-nearest-neighbor indexes instead of exact NumPy search.",
            )
            logger.info("UI: use_hnsw checkbox = %s", use_hnsw)
            use_hybrid = st.checkbox(
                "Enable Hybrid Retrieval",
                value=False,
                help="Combines dense vector search with BM25 keyword matching using Reciprocal Rank Fusion.",
            )
            enable_metadata_boost = st.checkbox(
                "Enable Metadata-Aware Boosting",
                value=False,
                help="Boosts chunks containing relevant biomedical entities (genes, diseases, etc.).",
            )
            enable_query_routing = st.checkbox(
                "Enable Query Understanding & Routing",
                value=False,
                help="Automatically selects retrieval strategy based on query type.",
            )
            enable_multi_index = st.checkbox(
                "Enable Multi-Index Routing",
                value=False,
                help="Routes queries across semantic, fixed, and sentence-level embedding indexes.",
            )

            manual_index = "Auto (router decides)"
            if enable_multi_index:
                manual_index = st.selectbox(
                    "Manual index override",
                    options=["Auto (router decides)", "semantic", "fixed", "sentence"],
                    index=0,
                    help="Choose a specific index for A/B testing, or leave it to the query router.",
                )

            use_streaming = st.checkbox(
                "🌊 Enable Streaming Mode",
                value=False,
                help="Streams retrieved sources and graph evidence before the answer finishes generating.",
            )

        with st.expander("⚙️ Retrieval Parameters", expanded=False):
            top_k = st.slider("top_k", 1, 50, 10)
            expand_depth = st.slider("expand_depth", 0, 3, 2)
            max_entity_degree = st.slider("max_entity_degree", 10, 2000, 500)
            alpha = st.slider("alpha (vector weight)", 0.0, 1.0, 0.8, step=0.05)
            max_results = st.slider("max_results", 1, 50, 20)

        st.header("Model")
        llm_client_type, llm_selection = _render_llm_mode_sidebar()
        if not st.session_state.llm_startup_logged:
            log_llm_startup_diagnostics(
                llm_selection.selected_mode,
                llm_selection.mode,
            )
            st.session_state.llm_startup_logged = True

        with st.expander("🔬 Advanced Options", expanded=False):
            use_decomposer = st.checkbox("Enable query decomposition", value=False)
            use_reranker = st.checkbox("Enable graph re-ranking", value=False)
            reranker_beta = st.slider(
                "reranker beta (original score weight)",
                0.0,
                1.0,
                0.7,
                step=0.05,
                disabled=not use_reranker,
            )

    index_name: str | None = None
    if enable_multi_index and manual_index != "Auto (router decides)":
        index_name = manual_index

    retrieval_overrides = {
        "top_k": top_k,
        "expand_depth": expand_depth,
        "max_entity_degree": max_entity_degree,
        "alpha": alpha,
        "max_results": max_results,
        "use_hnsw": use_hnsw,
        "use_hybrid": use_hybrid,
        "enable_query_routing": enable_query_routing,
        "enable_metadata_boost": enable_metadata_boost,
        "enable_multi_index": enable_multi_index,
        "index_name": index_name,
    }

    try:
        pipeline = get_pipeline(HF_HOME)
        base_config = default_search_config()
        search_config = _build_search_config(base_config, retrieval_overrides)
        retrieve_documents = pipeline.retrieve_documents
        graph_repository = retrieve_documents.graph_expand.graph_repository
    except Exception as exc:
        st.error(f"Failed to load pipeline: {exc}")
        return 1

    query = st.text_input(
        "Question",
        value="What are the risk factors for type 2 diabetes?",
        placeholder="Enter a biomedical question...",
    )

    col1, col2 = st.columns(2)
    retrieve_clicked = col1.button("🔍 Retrieve")
    answer_clicked = col2.button("💬 Answer")

    if use_streaming and answer_clicked:
        if use_decomposer:
            st.info("Query decomposition is disabled in streaming mode.")
        if use_reranker:
            st.info(
                "Graph re-ranking is disabled in streaming mode; the streaming use case "
                "applies its own ranking."
            )

        stream_use_case = _build_streaming_use_case(retrieve_documents, llm_selection.client)
        status = st.status("🔍 Searching...", state="running")
        events = stream_use_case.execute(Query(query), search_config)

        answer_container = st.empty()
        answer_text = ""
        answer_header_shown = False
        streamed_results: list[RetrievalResult] = []

        for event in events:
            if isinstance(event, RetrievalStarted):
                status.update(label="🔍 Searching...", state="running")
            elif isinstance(event, ChunksFound):
                streamed_results = event.chunks
                status.update(label=f"✅ Found {len(event.chunks)} chunks", state="complete")
                st.subheader(f"Retrieved context ({len(event.chunks)} chunks)")
                actual_backend = getattr(
                    retrieve_documents.vector_search.vector_store, "last_backend", None
                )
                requested_backend = "hnsw" if search_config.use_hnsw else "numpy"
                if actual_backend:
                    st.caption(f"⚡ Backend: {actual_backend} (requested: {requested_backend})")
                elif search_config.use_hnsw:
                    st.caption("⚡ HNSW requested")
                if search_config.enable_metadata_boost:
                    st.caption("🔬 Metadata boost applied")
                for rank, result in enumerate(event.chunks, start=1):
                    _render_result_card(rank, result)

                st.download_button(
                    label="Download results as CSV",
                    data=_results_to_csv(event.chunks),
                    file_name="retrieval_results.csv",
                    mime="text/csv",
                )
            elif isinstance(event, GraphEvidenceFound):
                _render_streaming_graph_evidence(graph_repository, event.entities)
            elif isinstance(event, TextChunkEvent):
                if not answer_header_shown:
                    st.subheader("Answer")
                    answer_header_shown = True
                answer_text += event.token
                answer_container.markdown(answer_text)
            elif isinstance(event, StreamComplete):
                status.update(label="✅ Answer complete", state="complete")

        return 0

    if retrieve_clicked or answer_clicked:
        with st.spinner("Retrieving..."):
            sub_queries, results, classification, strategy = _retrieve_results(
                retrieve_documents,
                graph_repository,
                query,
                search_config,
                llm_client_type=llm_client_type,
                use_reranker=use_reranker,
                reranker_beta=reranker_beta,
                use_decomposer=use_decomposer,
            )
            if use_decomposer:
                st.write(f"Sub-queries used ({len(sub_queries)}): {sub_queries}")

        _render_query_understanding(classification, strategy)

        st.subheader(f"Retrieved context ({len(results)} chunks)")
        actual_backend = getattr(
            retrieve_documents.vector_search.vector_store, "last_backend", None
        )
        requested_backend = "hnsw" if search_config.use_hnsw else "numpy"
        if actual_backend:
            st.caption(f"⚡ Backend: {actual_backend} (requested: {requested_backend})")
        elif search_config.use_hnsw:
            st.caption("⚡ HNSW requested")
        if search_config.enable_metadata_boost:
            st.caption("🔬 Metadata boost applied")
        for rank, result in enumerate(results, start=1):
            _render_result_card(rank, result)

        st.download_button(
            label="Download results as CSV",
            data=_results_to_csv(results),
            file_name="retrieval_results.csv",
            mime="text/csv",
        )

        _render_graph_evidence(graph_repository, results)

        if answer_clicked:
            with st.spinner("Generating answer..."):
                logger.info(
                    "Selected mode: %s | Effective mode: %s",
                    llm_selection.selected_mode,
                    llm_selection.mode,
                )
                answer = _generate_answer_safe(query, results, llm_selection.client)
                if answer == UNABLE_TO_GENERATE_ANSWER:
                    st.warning("Answer generation fell back to a safe default.")
            st.subheader("Answer")
            st.markdown(answer)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
