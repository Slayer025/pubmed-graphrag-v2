"""Retrieve documents use case."""

from __future__ import annotations

import logging
from typing import Any, Protocol

import numpy as np

from src.application.dto.search_config import SearchConfig
from src.application.ports import (
    ChunkRepository,
    EmbeddingService,
    GraphRepository,
    SparseRetriever,
    VectorStore,
)
from src.application.use_cases.graph_expand import GraphExpandUseCase
from src.application.use_cases.metadata_boost import MetadataBoostService
from src.application.use_cases.rerank import RerankUseCase
from src.application.use_cases.vector_search import VectorSearchUseCase
from src.domain.entities.retrieval_result import RetrievalResult
from src.domain.services.rrf_fusion_service import RRFFusionService
from src.domain.value_objects.query import Query

logger = logging.getLogger(__name__)


class QueryClassifier(Protocol):
    """Port for query classification."""

    def classify_query(self, question: str) -> dict:
        """Return classification dict for the question."""
        ...


class StrategyRouter(Protocol):
    """Port for strategy routing."""

    def route_strategy(
        self,
        classification: dict,
        *,
        enable_multi_index: bool = False,
    ) -> dict:
        """Return strategy dict for the classification."""
        ...


class RetrieveDocumentsUseCase:
    """End-to-end retrieval: vector search (+ optional BM25) + graph expand + rerank."""

    def __init__(
        self,
        embedding_service: EmbeddingService,
        vector_store: VectorStore,
        graph_repository: GraphRepository,
        chunk_repository: ChunkRepository,
        sparse_retriever: SparseRetriever | None = None,
        rrf_fusion_service: RRFFusionService | None = None,
        query_classifier: QueryClassifier | None = None,
        strategy_router: StrategyRouter | None = None,
        metadata_boost_service: MetadataBoostService | None = None,
    ) -> None:
        self.vector_search = VectorSearchUseCase(embedding_service, vector_store)
        self.graph_expand = GraphExpandUseCase(graph_repository)
        self.rerank = RerankUseCase(chunk_repository)
        self.sparse_retriever = sparse_retriever
        self.rrf_fusion_service = rrf_fusion_service or RRFFusionService()
        self.query_classifier = query_classifier
        self.strategy_router = strategy_router
        self.metadata_boost_service = metadata_boost_service

    def _apply_metadata_boost(
        self,
        results: list[RetrievalResult],
        query: Query,
        config: SearchConfig,
    ) -> list[RetrievalResult]:
        """Apply optional metadata-aware boosting and re-sort by combined_score."""
        if not config.enable_metadata_boost:
            return results
        if self.metadata_boost_service is None:
            logger.warning("Metadata boost enabled but no MetadataBoostService provided")
            return results

        boosted = self.metadata_boost_service.apply_boost(
            results,
            query.text,
            config.metadata_boost_factor,
        )
        logger.info(
            "METADATA BOOST APPLIED: factor=%.2f, top_chunk=%s, top_score=%.4f",
            config.metadata_boost_factor,
            boosted[0].chunk_id if boosted else "none",
            boosted[0].combined_score if boosted else 0.0,
        )
        return sorted(boosted, key=lambda r: r.combined_score, reverse=True)

    def _apply_strategy(
        self,
        query: Query,
        config: SearchConfig,
    ) -> tuple[SearchConfig, dict, dict, str | None]:
        """Return a possibly modified config plus classification, strategy, and index metadata."""
        if not config.enable_query_routing:
            return config, {}, {}, None

        classification = {}
        strategy = {}
        if self.query_classifier is not None:
            classification = self.query_classifier.classify_query(query.text)
        if self.strategy_router is not None:
            strategy = self.strategy_router.route_strategy(
                classification,
                enable_multi_index=config.enable_multi_index,
            )

        if not strategy:
            return config, classification, strategy, None

        index_name = (
            strategy.get("index_name")
            if config.enable_query_routing and config.enable_multi_index
            else None
        )

        logger.info(
            "QUERY ROUTING: type=%s strategy=%s index=%s reason=%s",
            classification.get("query_type", "general"),
            strategy.get("strategy_name", "unknown"),
            index_name or "default",
            strategy.get("reason", ""),
        )

        routed = SearchConfig(
            top_k=config.top_k,
            expand_depth=int(strategy.get("expand_depth", config.expand_depth)),
            max_entity_degree=config.max_entity_degree,
            max_expansion_per_entity=config.max_expansion_per_entity,
            max_expanded_nodes=config.max_expanded_nodes,
            alpha=config.alpha,
            depth_scores=config.depth_scores,
            use_hybrid=bool(strategy.get("use_hybrid", config.use_hybrid)),
            rrf_k=int(strategy.get("rrf_k", config.rrf_k)),
            max_results=config.max_results,
            enable_query_routing=config.enable_query_routing,
            enable_metadata_boost=config.enable_metadata_boost,
            metadata_boost_factor=config.metadata_boost_factor,
            default_index=config.default_index,
            enable_multi_index=config.enable_multi_index,
            index_name=index_name or config.default_index,
        )
        return routed, classification, strategy, index_name

    def execute(
        self,
        query: Query,
        config: SearchConfig,
    ) -> list[RetrievalResult] | tuple[list[RetrievalResult], dict, dict]:
        """Retrieve and rank context chunks for a query.

        When ``enable_query_routing`` is enabled, returns a tuple of
        (results, classification, strategy). When disabled, returns only the
        list of results to preserve backwards compatibility.
        """
        routed_config, classification, strategy, index_name = self._apply_strategy(query, config)

        logger.info("INDEX ROUTING: index=%s", index_name or "default")

        vector_results = self.vector_search.execute(
            query,
            routed_config,
            index_name=index_name,
            use_hnsw=routed_config.use_hnsw,
        )

        if not routed_config.use_hybrid or self.sparse_retriever is None:
            logger.info("RETRIEVAL: mode=dense_only")
            seed_ids = {chunk_id for chunk_id, _ in vector_results}
            expanded = self.graph_expand.execute(seed_ids, routed_config)
            results = self.rerank.execute(vector_results, expanded, routed_config)
        else:
            logger.info("RETRIEVAL: mode=hybrid")
            sparse_results = self.sparse_retriever.search(query.text, routed_config.top_k)
            fused = self.rrf_fusion_service.fuse(
                [
                    {"chunk_id": chunk_id, "score": score}
                    for chunk_id, score in vector_results
                ],
                [
                    {"chunk_id": chunk_id, "score": score}
                    for chunk_id, score in sparse_results
                ],
                k=routed_config.rrf_k,
            )
            fused_results = [
                (result.chunk_id, result.rrf_score) for result in fused[: routed_config.top_k]
            ]

            seed_ids = {chunk_id for chunk_id, _ in fused_results}
            expanded = self.graph_expand.execute(seed_ids, routed_config)
            results = self.rerank.execute(fused_results, expanded, routed_config)

        results = self._apply_metadata_boost(results, query, config)

        if config.enable_query_routing:
            return results, classification, strategy
        return results

    def retrieve_by_vector(
        self,
        query_vector: Any,
        config: SearchConfig,
        *,
        index_name: str | None = None,
    ) -> list[RetrievalResult]:
        """Retrieve by a pre-computed query vector.

        Vector-based retrieval skips query classification because there is no
        query text, so this method always returns a plain list of results for
        backwards compatibility. An optional ``index_name`` selects the vector
        index to search; when omitted the configured default index is used.
        """
        if isinstance(query_vector, np.ndarray):
            query_vector = query_vector.tolist()
        vector_results = self.vector_search.search_by_vector(
            query_vector,
            config,
            index_name=index_name,
            use_hnsw=config.use_hnsw,
        )
        seed_ids = {chunk_id for chunk_id, _ in vector_results}
        expanded = self.graph_expand.execute(seed_ids, config)
        return self.rerank.execute(vector_results, expanded, config)
