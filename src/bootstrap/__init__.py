"""Bootstrap / Dependency Injection container.

This module owns the object graph construction. It is the ONLY place where
infrastructure adapters are instantiated and wired into application use cases.

UI layers (Streamlit, CLI, scripts) should call ``bootstrap_pipeline()`` or
``bootstrap_retriever()`` instead of importing infrastructure directly.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any

from src.application.dto.rerank_config import RerankConfig
from src.application.dto.search_config import SearchConfig
from src.application.ports import EmbeddingService, LLMClient
from src.application.use_cases.generate_answer import GenerateAnswerUseCase
from src.application.use_cases.retrieve_documents import RetrieveDocumentsUseCase
from src.config import AppConfig
from src.graph_reranker import GraphReranker
from src.domain.services.rrf_fusion_service import RRFFusionService
from src.domain.services.query_classifier import classify_query
from src.domain.services.strategy_router import route_strategy
from src.infrastructure.embeddings.remote_embedding_client import create_embedding_client
from src.infrastructure.graph.in_memory_graph_repository import InMemoryGraphRepository
from src.infrastructure.retrievers.bm25_retriever import BM25Retriever
from src.infrastructure.storage.artifact_loader import LoadedArtifacts
from src.infrastructure.storage.chunk_repository import InMemoryChunkRepository
from src.infrastructure.storage.pure_build import pure_build_guard
from src.infrastructure.vector_store.numpy_vector_store import NumpyVectorStore
from src.llm_client import create_llm_client
from src.query_decomposer import DecomposerConfig, QueryDecomposer
from src.rag_pipeline import RAGPipeline

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _load_artifacts() -> LoadedArtifacts:
    """Load Phase 1/2 artifacts exactly once per process."""
    from src.bootstrap.bootstrap_artifacts import bootstrap_artifacts, default_cache_dir, get_preloaded_artifacts

    logger.info("Loading artifacts...")
    bootstrap_artifacts(default_cache_dir())
    return get_preloaded_artifacts()


def _build_embedding_service(config: AppConfig | None = None) -> EmbeddingService:
    """Build the embedding service adapter based on AppConfig."""
    cfg = config or AppConfig.default()
    result = create_embedding_client(
        provider=cfg.embedding.provider,
        model_name=cfg.embedding.model_name,
        api_token=cfg.embedding.api_token,
        service_url=cfg.embedding.service_url,
        batch_size=cfg.embedding.batch_size,
        normalize=cfg.embedding.normalize,
        timeout_seconds=cfg.embedding.timeout_seconds,
    )
    if result.fallback_reason:
        logger.warning("Embedding client fallback: %s", result.fallback_reason)
    return result.client


class _QueryClassifierPort:
    """Lightweight adapter exposing the pure domain classifier as a port."""

    def classify_query(self, question: str) -> dict:
        return classify_query(question)


class _StrategyRouterPort:
    """Lightweight adapter exposing the pure domain router as a port."""

    def route_strategy(self, classification: dict) -> dict:
        return route_strategy(classification)


def _build_sparse_retriever(chunks: list[dict[str, Any]]) -> BM25Retriever:
    """Build the BM25 sparse retriever directly from chunk records."""
    return BM25Retriever(chunks)


def _build_retrieve_documents(config: AppConfig | None = None) -> RetrieveDocumentsUseCase:
    """Build the main retrieval use case with cached artifacts and model."""
    cfg = config or AppConfig.default()
    artifacts = _load_artifacts()

    embedding_service = _build_embedding_service(cfg)
    vector_store = NumpyVectorStore(artifacts.chunks, artifacts.embeddings)
    graph_repository = InMemoryGraphRepository(
        artifacts.mentions,
        artifacts.has_chunk,
        artifacts.chunks,
    )
    chunk_repository = InMemoryChunkRepository(artifacts.chunks)
    sparse_retriever = _build_sparse_retriever(artifacts.chunks)

    return RetrieveDocumentsUseCase(
        embedding_service=embedding_service,
        vector_store=vector_store,
        graph_repository=graph_repository,
        chunk_repository=chunk_repository,
        sparse_retriever=sparse_retriever,
        rrf_fusion_service=RRFFusionService(),
        query_classifier=_QueryClassifierPort(),
        strategy_router=_StrategyRouterPort(),
    )


def _search_config_from_app(config: AppConfig | None = None) -> SearchConfig:
    """Convert ``AppConfig.retrieval`` into application-layer ``SearchConfig``."""
    cfg = config or AppConfig.default()
    return SearchConfig.from_retrieval_config(cfg.retrieval)


def build_pipeline(
    *,
    hf_home: str,
    artifacts: LoadedArtifacts,
) -> RAGPipeline:
    """Build the retrieval stack from preloaded in-memory artifacts (pure: no IO)."""
    from src.bootstrap.bootstrap_artifacts import require_bootstrap_success

    require_bootstrap_success()
    with pure_build_guard():
        app_config = AppConfig.default()
        embedding_service = create_embedding_client(
            provider=app_config.embedding.provider,
            model_name=app_config.embedding.model_name,
            api_token=app_config.embedding.api_token,
            service_url=app_config.embedding.service_url,
            batch_size=app_config.embedding.batch_size,
            normalize=app_config.embedding.normalize,
            timeout_seconds=app_config.embedding.timeout_seconds,
            cache_folder=hf_home,
        ).client
        graph_repository = InMemoryGraphRepository(
            artifacts.mentions,
            artifacts.has_chunk,
            artifacts.chunks,
        )
        chunk_repository = InMemoryChunkRepository(artifacts.chunks)
        sparse_retriever = _build_sparse_retriever(artifacts.chunks)
        retrieve_documents = RetrieveDocumentsUseCase(
            embedding_service=embedding_service,
            vector_store=NumpyVectorStore(artifacts.chunks, artifacts.embeddings),
            graph_repository=graph_repository,
            chunk_repository=chunk_repository,
            sparse_retriever=sparse_retriever,
            rrf_fusion_service=RRFFusionService(),
            query_classifier=_QueryClassifierPort(),
            strategy_router=_StrategyRouterPort(),
        )
        return RAGPipeline(
            retrieve_documents=retrieve_documents,
            generate_answer=None,
            llm=None,
            decomposer=None,
            reranker=None,
        )


def bootstrap_retriever(config: AppConfig | None = None) -> "Retriever":
    """Build the backward-compatible retriever facade.

    This is deprecated; prefer ``bootstrap_pipeline`` for new code.
    """
    from src.retriever import Retriever

    cfg = config or AppConfig.default()
    artifacts = _load_artifacts()

    graph_repository = InMemoryGraphRepository(
        artifacts.mentions,
        artifacts.has_chunk,
        artifacts.chunks,
    )
    chunk_repository = InMemoryChunkRepository(artifacts.chunks)

    class _Index:
        def __init__(self, chunks: list[dict[str, Any]], embeddings: Any) -> None:
            self.chunks = chunks
            self.embeddings = embeddings
            self.chunk_by_id = chunk_repository.get_chunks({str(c["chunk_id"]) for c in chunks})
            self.row_by_chunk_id = {
                str(chunk["chunk_id"]): row for row, chunk in enumerate(chunks)
            }
            self.article_chunks = graph_repository.article_chunks
            self.entity_chunks = graph_repository.entity_chunks
            self.chunk_entities = graph_repository.chunk_entities
            self.entity_degrees = graph_repository.entity_degrees

    index = _Index(artifacts.chunks, artifacts.embeddings)
    return Retriever(index, cfg)


def bootstrap_pipeline(
    config: AppConfig | None = None,
    llm: LLMClient | None = None,
    *,
    llm_client_type: str | None = None,
    use_decomposer: bool = False,
    use_reranker: bool = False,
    reranker_beta: float = 0.7,
) -> RAGPipeline:
    """Build the main RAG orchestrator.

    This is the preferred entry point for UI and script layers. If ``llm`` is
    not provided but ``llm_client_type`` is given, the LLM client is created by
    the bootstrap container.
    """
    if llm is None and llm_client_type:
        llm = create_llm_client(llm_client_type)

    retrieve_documents = _build_retrieve_documents(config)
    generate_answer = GenerateAnswerUseCase(llm=llm) if llm else None
    decomposer = _build_decomposer(llm, use_decomposer) if llm else None
    reranker = _build_reranker(retrieve_documents, use_reranker, reranker_beta)
    return RAGPipeline(
        retrieve_documents=retrieve_documents,
        generate_answer=generate_answer,
        llm=llm,
        decomposer=decomposer,
        reranker=reranker,
    )


def _build_decomposer(
    llm: LLMClient,
    enabled: bool = False,
) -> QueryDecomposer | None:
    """Build a query decomposer if requested."""
    if not enabled:
        return None
    return QueryDecomposer(llm=llm, config=DecomposerConfig(enabled=True))


def _build_reranker(
    retrieve_documents: RetrieveDocumentsUseCase,
    enabled: bool = False,
    beta: float = 0.7,
) -> GraphReranker | None:
    """Build a graph reranker using the pipeline's graph repository."""
    if not enabled:
        return None
    graph_repository = retrieve_documents.graph_expand.graph_repository
    return GraphReranker(index=graph_repository, config=RerankConfig(enabled=True, beta=beta))


def default_search_config(config: AppConfig | None = None) -> SearchConfig:
    """Return the default ``SearchConfig`` for the given ``AppConfig``."""
    return _search_config_from_app(config)
