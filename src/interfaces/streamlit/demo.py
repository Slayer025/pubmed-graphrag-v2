"""Streamlit interface for the PubMed GraphRAG pipeline."""

from __future__ import annotations

import csv
import io
import logging
import os
import sys
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
from src.application.use_cases.retrieve_documents import RetrieveDocumentsUseCase
from src.bootstrap import build_pipeline, default_search_config
from src.bootstrap.bootstrap_artifacts import get_preloaded_artifacts
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
) -> tuple[list[str], list[RetrievalResult]]:
    if use_decomposer:
        llm = create_llm_client_with_mode(llm_client_type).client
        decomposer = QueryDecomposer(llm=llm, config=DecomposerConfig(enabled=True))
        sub_queries = decomposer.decompose(query)
        if len(sub_queries) <= 1:
            results = retrieve_documents.execute(Query(query), search_config)
            results = _maybe_rerank(
                graph_repository,
                query,
                results,
                enabled=use_reranker,
                beta=reranker_beta,
            )
            return sub_queries, results

        best_by_chunk: dict[str, RetrievalResult] = {}
        for sub_query in sub_queries:
            sub_results = retrieve_documents.execute(Query(sub_query), search_config)
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

        merged = sorted(best_by_chunk.values(), key=lambda r: r.combined_score, reverse=True)
        return sub_queries, merged[: search_config.max_results]

    results = retrieve_documents.execute(Query(query), search_config)
    results = _maybe_rerank(
        graph_repository,
        query,
        results,
        enabled=use_reranker,
        beta=reranker_beta,
    )
    return [query], results


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


def main() -> int:
    st.set_page_config(page_title="PubMed GraphRAG Demo", layout="wide")
    st.title("🧬 PubMed GraphRAG Demo")
    st.markdown(
        "Ask a biomedical question. The demo retrieves semantic chunks from 5,000 "
        "PubMed abstracts using graph-enhanced retrieval."
    )

    if "llm_startup_logged" not in st.session_state:
        st.session_state.llm_startup_logged = False

    with st.sidebar:
        st.header("Model")
        llm_client_type, llm_selection = _render_llm_mode_sidebar()
        if not st.session_state.llm_startup_logged:
            log_llm_startup_diagnostics(
                llm_selection.selected_mode,
                llm_selection.mode,
            )
            st.session_state.llm_startup_logged = True

        st.header("Retrieval")
        top_k = st.slider("top_k", 1, 50, 10)
        expand_depth = st.slider("expand_depth", 0, 3, 2)
        max_entity_degree = st.slider("max_entity_degree", 10, 2000, 500)
        alpha = st.slider("alpha (vector weight)", 0.0, 1.0, 0.8, step=0.05)
        max_results = st.slider("max_results", 1, 50, 20)
        use_hybrid = st.checkbox(
            "Enable Hybrid Retrieval (Dense + BM25 + RRF)",
            value=False,
            help="When enabled, dense vector search and BM25 keyword search are fused with Reciprocal Rank Fusion.",
        )

        st.header("Phase 5 options")
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

    retrieval_overrides = {
        "top_k": top_k,
        "expand_depth": expand_depth,
        "max_entity_degree": max_entity_degree,
        "alpha": alpha,
        "max_results": max_results,
        "use_hybrid": use_hybrid,
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

    if retrieve_clicked or answer_clicked:
        with st.spinner("Retrieving..."):
            sub_queries, results = _retrieve_results(
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

        st.subheader(f"Retrieved context ({len(results)} chunks)")
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
