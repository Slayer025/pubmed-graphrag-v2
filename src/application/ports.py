"""Application-layer ports (interfaces) for infrastructure adapters."""

from __future__ import annotations

from typing import Any, Protocol

from src.domain.entities.retrieval_result import RetrievalResult


class EmbeddingService(Protocol):
    """Port for embedding text strings."""

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one embedding vector per input text."""
        ...

    def embed_query(self, query: str) -> list[float]:
        """Embed a single query string."""
        ...


class VectorStore(Protocol):
    """Port for vector similarity search."""

    def search(
        self,
        query_vector: list[float],
        top_k: int,
        *,
        index_name: str | None = None,
    ) -> list[tuple[str, float]]:
        """Return top-k (chunk_id, similarity_score) pairs.

        ``index_name`` is optional and may be ignored by single-index stores.
        """
        ...


class GraphRepository(Protocol):
    """Port for graph adjacency lookups."""

    def get_chunk_article(self, chunk_id: str) -> str:
        ...

    def get_article_chunks(self, article_id: str) -> set[str]:
        ...

    def get_chunk_entities(self, chunk_id: str) -> set[str]:
        ...

    def get_entity_chunks(self, entity_id: str) -> set[str]:
        ...

    def get_entity_degree(self, entity_id: str) -> int:
        ...


class ChunkRepository(Protocol):
    """Port for chunk metadata lookup."""

    def get_chunk(self, chunk_id: str) -> dict[str, Any] | None:
        ...

    def get_chunks(self, chunk_ids: set[str]) -> dict[str, dict[str, Any]]:
        ...


class SparseRetriever(Protocol):
    """Port for keyword/sparse retrieval (e.g., BM25)."""

    def search(self, query: str, top_k: int) -> list[tuple[str, float]]:
        """Return top-k (chunk_id, score) pairs for the query."""
        ...


class LLMClient(Protocol):
    """Port for text-generation backends."""

    def complete(self, prompt: str, **kwargs: Any) -> str:
        """Return a text completion for the given prompt."""
        ...


class Decomposer(Protocol):
    """Port for query decomposition strategies."""

    def decompose(self, query: str) -> list[str]:
        """Split a query into sub-queries."""
        ...


class GraphReranker(Protocol):
    """Port for graph-signal reranking of retrieval results."""

    def rerank(self, query: str, results: list[RetrievalResult]) -> list[RetrievalResult]:
        """Re-rank results using graph-derived signals."""
        ...
