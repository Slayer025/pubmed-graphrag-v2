"""Bootstrap / Dependency Injection container.

This module owns the object graph construction. It is the ONLY place where
infrastructure adapters are instantiated and wired into application use cases.

UI layers (Streamlit, CLI, scripts) should call ``bootstrap_pipeline()`` or
``bootstrap_retriever()`` instead of importing infrastructure directly.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from src.application.dto.rerank_config import RerankConfig
from src.application.dto.search_config import SearchConfig
from src.application.ports import EmbeddingService, LLMClient, VectorStore
from src.application.use_cases.generate_answer import GenerateAnswerUseCase
from src.application.use_cases.metadata_boost import MetadataBoostService
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
from src.infrastructure.vector_store.hnsw_vector_store import HnswVectorStore
from src.infrastructure.vector_store.multi_index_vector_store import MultiIndexVectorStore
from src.infrastructure.vector_store.numpy_vector_store import NumpyVectorStore
from src.infrastructure.vector_store.switchable_vector_store import SwitchableVectorStore
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


def _load_single_index(
    chunks_path: str,
    embeddings_path: str,
    embedding_dim: int,
) -> NumpyVectorStore:
    """Load one chunk+embedding index from disk into a NumpyVectorStore."""
    from src.embeddings import normalize_embeddings
    from src.storage import iter_jsonl_gz

    chunks = list(iter_jsonl_gz(Path(chunks_path)))
    embeddings = np.load(embeddings_path)

    if embeddings.shape[0] != len(chunks):
        raise ValueError(
            f"Embedding rows ({embeddings.shape[0]}) do not match chunk count ({len(chunks)}) "
            f"for {chunks_path}"
        )
    if embeddings.shape[1] != embedding_dim:
        raise ValueError(
            f"Embedding dimension ({embeddings.shape[1]}) does not match config ({embedding_dim}) "
            f"for {embeddings_path}"
        )

    return NumpyVectorStore(chunks, normalize_embeddings(embeddings))


def _load_hnsw_index(
    index_name: str,
    base_path: Path,
    ef_search: int = 100,
) -> HnswVectorStore:
    """Load a single HNSW index plus its sidecar data.

    Args:
        index_name: Identifier used for the file prefix (semantic, fixed, sentence).
        base_path: Directory containing the HNSW ``.bin`` and ``.json`` sidecars.
        ef_search: Query-time HNSW accuracy parameter.

    Returns:
        An ``HnswVectorStore`` backed by the persisted index.

    Raises:
        FileNotFoundError: If the HNSW ``.bin`` file is missing.
    """
    hnsw_dir = base_path / "data/hnsw"
    embeddings_dir = base_path / "data/embeddings"
    index_file = hnsw_dir / f"{index_name}_index.bin"
    chunk_ids_file = hnsw_dir / f"{index_name}_chunk_ids.json"
    embeddings_file = embeddings_dir / f"{index_name}_embeddings.npy"
    return HnswVectorStore(
        str(index_file),
        str(chunk_ids_file),
        str(embeddings_file),
        ef_search=ef_search,
    )


def _build_named_vector_stores(
    index_name: str,
    chunks_path: Path,
    embeddings_path: Path,
    hnsw_base_path: Path,
    ef_search: int,
    embedding_dim: int,
) -> tuple[NumpyVectorStore, HnswVectorStore | None]:
    """Build both NumPy and (optionally) HNSW stores for a single index.

    Returns:
        A tuple of ``(numpy_store, hnsw_store_or_none)``. The HNSW store is
        ``None`` when the index file is missing or hnswlib cannot load it.
    """
    numpy_store = _load_single_index(
        str(chunks_path),
        str(embeddings_path),
        embedding_dim,
    )

    hnsw_index_path = hnsw_base_path / "data/hnsw" / f"{index_name}_index.bin"
    hnsw_store: HnswVectorStore | None = None
    if hnsw_index_path.exists():
        try:
            hnsw_store = _load_hnsw_index(index_name, hnsw_base_path, ef_search=ef_search)
            logger.info("Loaded %s index with HNSW", index_name)
        except Exception as exc:
            logger.warning(
                "Failed to load HNSW index for %s; using NumPy only: %s",
                index_name,
                exc,
            )
    else:
        logger.info("No HNSW index found for %s; using NumPy only", index_name)

    return numpy_store, hnsw_store


def _build_vector_store(
    config: AppConfig,
    artifacts: LoadedArtifacts,
) -> VectorStore:
    """Build the vector store from available chunk+embedding indexes.

    The semantic index is always loaded. Additional indexes (fixed, sentence)
    are loaded opportunistically when their files exist in the artifact cache.
    For every index, both a ``NumpyVectorStore`` and an optional
    ``HnswVectorStore`` are built and wrapped in a ``SwitchableVectorStore``.
    The active backend is selected at query time via ``use_hnsw`` in
    ``SearchConfig``, so the Streamlit UI can toggle HNSW without restarting.
    Missing HNSW files are ignored and the store falls back to NumPy.
    """
    cfg = config or AppConfig.default()

    # Additional indexes are expected in the artifact cache directory, which is
    # where bootstrap_artifacts() materializes them (locally or from a release).
    from src.bootstrap.bootstrap_artifacts import default_cache_dir

    cache_root = Path(default_cache_dir()).resolve()
    hnsw_base_path = Path("data").resolve()
    ef_search = 100  # keep in sync with scripts/build_hnsw_indexes.py defaults

    numpy_stores: dict[str, VectorStore] = {}
    hnsw_stores: dict[str, VectorStore] = {}

    index_definitions = [
        (
            "semantic",
            cache_root / "data/chunks/chunks_semantic.jsonl.gz",
            cache_root / "data/embeddings/semantic_embeddings.npy",
        ),
        (
            "fixed",
            cache_root / "data/chunks/chunks_fixed.jsonl.gz",
            cache_root / "data/embeddings/fixed_embeddings.npy",
        ),
        (
            "sentence",
            cache_root / "data/chunks/chunks_sentence.jsonl.gz",
            cache_root / "data/embeddings/sentence_embeddings.npy",
        ),
    ]

    for name, chunks_path, embeddings_path in index_definitions:
        if not chunks_path.exists() or not embeddings_path.exists():
            logger.info(
                "Skipping %s index: missing %s or %s", name, chunks_path, embeddings_path
            )
            continue
        try:
            numpy_store, hnsw_store = _build_named_vector_stores(
                name,
                chunks_path,
                embeddings_path,
                hnsw_base_path,
                ef_search,
                cfg.embedding.embedding_dim,
            )
            numpy_stores[name] = numpy_store
            if hnsw_store is not None:
                hnsw_stores[name] = hnsw_store
            logger.info("Loaded %s index", name)
        except Exception as exc:
            logger.warning("Failed to load %s index: %s", name, exc)

    default_index = cfg.retrieval.default_index
    if default_index not in numpy_stores:
        logger.warning(
            "Default index '%s' not available; falling back to 'semantic'",
            default_index,
        )
        default_index = "semantic"

    if not numpy_stores:
        raise RuntimeError("No vector indexes could be loaded")

    switchable = SwitchableVectorStore(
        hnsw_stores,
        numpy_stores,
        default_index=default_index,
    )

    loaded_hnsw = sorted(hnsw_stores)
    logger.info(
        "Vector store: switchable with indexes=%s, hnsw=%s, default=%s",
        sorted(numpy_stores),
        loaded_hnsw,
        default_index,
    )
    return switchable


class _QueryClassifierPort:
    """Lightweight adapter exposing the pure domain classifier as a port."""

    def classify_query(self, question: str) -> dict:
        return classify_query(question)


class _StrategyRouterPort:
    """Lightweight adapter exposing the pure domain router as a port."""

    def route_strategy(
        self,
        classification: dict,
        *,
        enable_multi_index: bool = False,
    ) -> dict:
        return route_strategy(classification, enable_multi_index=enable_multi_index)


def _build_sparse_retriever(chunks: list[dict[str, Any]]) -> BM25Retriever:
    """Build the BM25 sparse retriever directly from chunk records."""
    return BM25Retriever(chunks)


def _build_retrieve_documents(config: AppConfig | None = None) -> RetrieveDocumentsUseCase:
    """Build the main retrieval use case with cached artifacts and model."""
    cfg = config or AppConfig.default()
    artifacts = _load_artifacts()

    embedding_service = _build_embedding_service(cfg)
    vector_store = _build_vector_store(cfg, artifacts)
    graph_repository = InMemoryGraphRepository(
        artifacts.mentions,
        artifacts.has_chunk,
        artifacts.chunks,
        artifacts.entities,
    )
    chunk_repository = InMemoryChunkRepository(artifacts.chunks)
    sparse_retriever = _build_sparse_retriever(artifacts.chunks)
    metadata_boost_service = MetadataBoostService(graph_repository)

    return RetrieveDocumentsUseCase(
        embedding_service=embedding_service,
        vector_store=vector_store,
        graph_repository=graph_repository,
        chunk_repository=chunk_repository,
        sparse_retriever=sparse_retriever,
        rrf_fusion_service=RRFFusionService(),
        query_classifier=_QueryClassifierPort(),
        strategy_router=_StrategyRouterPort(),
        metadata_boost_service=metadata_boost_service,
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
            artifacts.entities,
        )
        chunk_repository = InMemoryChunkRepository(artifacts.chunks)
        sparse_retriever = _build_sparse_retriever(artifacts.chunks)
        retrieve_documents = RetrieveDocumentsUseCase(
            embedding_service=embedding_service,
            vector_store=_build_vector_store(app_config, artifacts),
            graph_repository=graph_repository,
            chunk_repository=chunk_repository,
            sparse_retriever=sparse_retriever,
            rrf_fusion_service=RRFFusionService(),
            query_classifier=_QueryClassifierPort(),
            strategy_router=_StrategyRouterPort(),
            metadata_boost_service=MetadataBoostService(graph_repository),
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
        artifacts.entities,
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
