"""Switchable vector store adapter.

Holds both HNSW and NumPy-backed ``VectorStore`` instances for each index and
selects the active backend at query time via ``use_hnsw``.  This lets the
Streamlit UI toggle HNSW without restarting the app.
"""

from __future__ import annotations

import logging
from typing import Any

from src.application.ports import VectorStore

logger = logging.getLogger(__name__)


class SwitchableVectorStore:
    """Registry of named vector stores with runtime HNSW/NumPy switching."""

    def __init__(
        self,
        hnsw_stores: dict[str, VectorStore],
        numpy_stores: dict[str, VectorStore],
        *,
        default_index: str = "semantic",
    ) -> None:
        """Build a switchable store.

        Args:
            hnsw_stores: Mapping from index name to an ``HnswVectorStore`` (or any
                ``VectorStore``). May be empty if HNSW indexes are unavailable.
            numpy_stores: Mapping from index name to a ``NumpyVectorStore``.
                Must contain at least one index.
            default_index: Name used when ``search`` is called without an explicit
                ``index_name``.
        """
        if not numpy_stores:
            raise ValueError("At least one NumPy-backed index must be provided")

        self.hnsw_stores: dict[str, VectorStore] = dict(hnsw_stores)
        self.numpy_stores: dict[str, VectorStore] = dict(numpy_stores)
        self.default_index = default_index
        self.last_backend: str | None = None
        self.last_index: str | None = None

        if self.default_index not in self.numpy_stores:
            available = ", ".join(sorted(self.numpy_stores))
            raise ValueError(
                f"Default index '{self.default_index}' not available. "
                f"Available indexes: {available}"
            )

    def search(
        self,
        query_vector: list[float],
        top_k: int,
        *,
        index_name: str | None = None,
        use_hnsw: bool = False,
    ) -> list[tuple[str, float]]:
        """Return top-k results using the requested backend.

        Args:
            query_vector: Dense query embedding.
            top_k: Number of results to return.
            index_name: Optional index to query. Falls back to the default.
            use_hnsw: If ``True`` and an HNSW store exists for the selected index,
                use HNSW. Otherwise fall back to NumPy.

        Returns:
            Top-k ``(chunk_id, similarity_score)`` pairs.
        """
        name = index_name or self.default_index
        if name not in self.numpy_stores:
            available = ", ".join(sorted(self.numpy_stores))
            raise ValueError(
                f"Unknown index '{name}'. Available indexes: {available}"
            )

        self.last_index = name
        if use_hnsw and name in self.hnsw_stores:
            self.last_backend = "hnsw"
            logger.info("VECTOR STORE: using hnsw for index=%s", name)
            return self.hnsw_stores[name].search(query_vector, top_k)

        self.last_backend = "numpy"
        logger.info("VECTOR STORE: using numpy for index=%s", name)
        return self.numpy_stores[name].search(query_vector, top_k)
