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

    def route_strategy(self, classification: dict) -> dict:
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
    ) -> None:
        self.vector_search = VectorSearchUseCase(embedding_service, vector_store)
        self.graph_expand = GraphExpandUseCase(graph_repository)
        self.rerank = RerankUseCase(chunk_repository)
        self.sparse_retriever = sparse_retriever
        self.rrf_fusion_service = rrf_fusion_service or RRFFusionService()
        self.query_classifier = query_classifier
        self.strategy_router = strategy_router

    def _apply_strategy(
        self,
        query: Query,
        config: SearchConfig,
    ) -> tuple[SearchConfig, dict, dict]:
        """Return a possibly modified config plus classification and strategy metadata."""
        if not config.enable_query_routing:
            return config, {}, {}

        classification = {}
        strategy = {}
        if self.query_classifier is not None:
            classification = self.query_classifier.classify_query(query.text)
        if self.strategy_router is not None:
            strategy = self.strategy_router.route_strategy(classification)

        if not strategy:
            return config, classification, strategy

        logger.info(
            "QUERY ROUTING: type=%s strategy=%s reason=%s",
            classification.get("query_type", "general"),
            strategy.get("strategy_name", "unknown"),
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
        )
        return routed, classification, strategy

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
        routed_config, classification, strategy = self._apply_strategy(query, config)

        vector_results = self.vector_search.execute(query, routed_config)

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

        if config.enable_query_routing:
            return results, classification, strategy
        return results

    def retrieve_by_vector(
        self,
        query_vector: Any,
        config: SearchConfig,
    ) -> list[RetrievalResult]:
        """Retrieve by a pre-computed query vector.

        Vector-based retrieval skips query classification because there is no
        query text, so this method always returns a plain list of results for
        backwards compatibility.
        """
        if isinstance(query_vector, np.ndarray):
            query_vector = query_vector.tolist()
        vector_results = self.vector_search.search_by_vector(query_vector, config)
        seed_ids = {chunk_id for chunk_id, _ in vector_results}
        expanded = self.graph_expand.execute(seed_ids, config)
        return self.rerank.execute(vector_results, expanded, config)
