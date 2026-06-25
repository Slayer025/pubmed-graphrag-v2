"""Vector search use case."""

from __future__ import annotations

from typing import Any

import numpy as np

from src.application.dto.search_config import SearchConfig
from src.application.ports import EmbeddingService, VectorStore
from src.domain.value_objects.query import Query


class VectorSearchUseCase:
    """Find top-k chunks by vector similarity."""

    def __init__(self, embedding_service: EmbeddingService, vector_store: VectorStore) -> None:
        self.embedding_service = embedding_service
        self.vector_store = vector_store

    def execute(
        self,
        query: Query,
        config: SearchConfig,
        *,
        index_name: str | None = None,
    ) -> list[tuple[str, float]]:
        """Return (chunk_id, vector_score) pairs for the query."""
        query_vector = self.embedding_service.embed_query(query.text)
        return self.search_by_vector(query_vector, config, index_name=index_name)

    def search_by_vector(
        self,
        query_vector: Any,
        config: SearchConfig,
        *,
        index_name: str | None = None,
    ) -> list[tuple[str, float]]:
        """Return (chunk_id, vector_score) pairs for a pre-computed vector."""
        if isinstance(query_vector, np.ndarray):
            query_vector = query_vector.tolist()
        return self.vector_store.search(query_vector, config.top_k, index_name=index_name)
