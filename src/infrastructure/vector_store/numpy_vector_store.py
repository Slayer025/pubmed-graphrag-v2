"""In-memory numpy vector store adapter."""

from __future__ import annotations

from typing import Any

import numpy as np


class NumpyVectorStore:
    """Vector store backed by a numpy embedding matrix."""

    def __init__(self, chunks: list[dict[str, Any]], embeddings: np.ndarray) -> None:
        self.chunks = chunks
        self.embeddings = embeddings

    def search(
        self,
        query_vector: list[float],
        top_k: int,
        *,
        index_name: str | None = None,
        use_hnsw: bool = False,
    ) -> list[tuple[str, float]]:
        """Return top-k (chunk_id, cosine_similarity) pairs by dot product.

        Assumes embeddings and query vector are L2-normalized. The optional
        ``index_name`` and ``use_hnsw`` are accepted for API compatibility with
        multi-index and switchable stores but are ignored by this single-index
        implementation.
        """
        del index_name, use_hnsw
        vector = np.asarray(query_vector, dtype=np.float32)
        scores = self.embeddings @ vector
        top_indices = np.argpartition(scores, -top_k)[-top_k:]
        top_indices = top_indices[np.argsort(-scores[top_indices])]

        results: list[tuple[str, float]] = []
        for idx in top_indices:
            chunk_id = str(self.chunks[int(idx)]["chunk_id"])
            results.append((chunk_id, float(scores[idx])))
        return results
