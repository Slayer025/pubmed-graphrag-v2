"""Multi-index vector store adapter.

Holds a registry of named vector indexes and routes searches to the
requested index.  With only one index loaded it behaves identically to a
single NumpyVectorStore, preserving today's backwards-compatible behavior.
"""

from __future__ import annotations

from typing import Any

from src.application.ports import VectorStore


class MultiIndexVectorStore(VectorStore):
    """Registry of named :class:`VectorStore` implementations.

    This adapter itself satisfies the :class:`VectorStore` port: a plain
    ``search(query_vector, k)`` call routes to the configured default
    index, so callers that do not know about multiple indexes keep working.
    Callers that *do* know about multiple indexes can pass
    ``index_name="..."`` to select a specific index.
    """

    def __init__(
        self,
        indexes: dict[str, VectorStore],
        *,
        default_index: str | None = None,
    ) -> None:
        """Build a multi-index store.

        Args:
            indexes: Mapping from index name to a ``VectorStore`` instance.
            default_index: Name used when ``search`` is called without an
                explicit index.  Defaults to the first key in ``indexes``.

        Raises:
            ValueError: If ``indexes`` is empty or ``default_index`` is not
                present in the registry.
        """
        if not indexes:
            raise ValueError("At least one index must be provided to MultiIndexVectorStore")

        self.indexes: dict[str, VectorStore] = dict(indexes)
        self.default_index = default_index or next(iter(self.indexes))

        if self.default_index not in self.indexes:
            available = ", ".join(sorted(self.indexes))
            raise ValueError(
                f"Default index '{self.default_index}' not in registry. "
                f"Available indexes: {available}"
            )

    def search(
        self,
        query_vector: list[float],
        top_k: int,
        *,
        index_name: str | None = None,
    ) -> list[tuple[str, float]]:
        """Return top-k results from the requested index.

        Args:
            query_vector: Dense query embedding.
            top_k: Number of results to return.
            index_name: Optional index to query.  Falls back to the default.

        Returns:
            Top-k ``(chunk_id, similarity_score)`` pairs.
        """
        name = index_name or self.default_index
        if name not in self.indexes:
            available = ", ".join(sorted(self.indexes))
            raise ValueError(
                f"Unknown index '{name}'. Available indexes: {available}"
            )
        return self.indexes[name].search(query_vector, top_k)
