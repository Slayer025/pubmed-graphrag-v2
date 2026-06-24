"""Retrieve documents use case."""

from __future__ import annotations

import logging
from typing import Any

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
    ) -> None:
        self.vector_search = VectorSearchUseCase(embedding_service, vector_store)
        self.graph_expand = GraphExpandUseCase(graph_repository)
        self.rerank = RerankUseCase(chunk_repository)
        self.sparse_retriever = sparse_retriever
        self.rrf_fusion_service = rrf_fusion_service or RRFFusionService()

    def execute(self, query: Query, config: SearchConfig) -> list[RetrievalResult]:
        """Retrieve and rank context chunks for a query."""
        vector_results = self.vector_search.execute(query, config)

        if not config.use_hybrid or self.sparse_retriever is None:
            logger.info("RETRIEVAL: mode=dense_only")
            seed_ids = {chunk_id for chunk_id, _ in vector_results}
            expanded = self.graph_expand.execute(seed_ids, config)
            return self.rerank.execute(vector_results, expanded, config)

        logger.info("RETRIEVAL: mode=hybrid")
        sparse_results = self.sparse_retriever.search(query.text, config.top_k)
        fused = self.rrf_fusion_service.fuse(
            [
                {"chunk_id": chunk_id, "score": score}
                for chunk_id, score in vector_results
            ],
            [
                {"chunk_id": chunk_id, "score": score}
                for chunk_id, score in sparse_results
            ],
            k=config.rrf_k,
        )
        fused_results = [
            (result.chunk_id, result.rrf_score) for result in fused[: config.top_k]
        ]

        seed_ids = {chunk_id for chunk_id, _ in fused_results}
        expanded = self.graph_expand.execute(seed_ids, config)
        return self.rerank.execute(fused_results, expanded, config)

    def retrieve_by_vector(
        self,
        query_vector: Any,
        config: SearchConfig,
    ) -> list[RetrievalResult]:
        """Retrieve by a pre-computed query vector."""
        if isinstance(query_vector, np.ndarray):
            query_vector = query_vector.tolist()
        vector_results = self.vector_search.search_by_vector(query_vector, config)
        seed_ids = {chunk_id for chunk_id, _ in vector_results}
        expanded = self.graph_expand.execute(seed_ids, config)
        return self.rerank.execute(vector_results, expanded, config)
