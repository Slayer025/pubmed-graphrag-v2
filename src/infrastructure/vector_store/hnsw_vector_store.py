"""HNSW-backed vector store adapter.

Loads a pre-built hnswlib index, its chunk-id sidecar, and the original
embedding matrix.  Search uses HNSW to retrieve approximate neighbors, then
re-scores those candidates with exact cosine similarity against the loaded
embeddings so downstream consumers receive deterministic, comparable scores.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class HnswVectorStore:
    """Vector store backed by an hnswlib approximate-nearest-neighbor index."""

    def __init__(
        self,
        index_path: str,
        chunk_ids_path: str,
        embeddings_path: str,
        ef_search: int = 100,
    ) -> None:
        """Load a persisted HNSW index and its sidecars.

        Args:
            index_path: Path to the persisted hnswlib ``.bin`` file.
            chunk_ids_path: Path to the JSON sidecar containing chunk IDs in
                the same order as the original embedding rows (which become the
                internal hnswlib IDs).
            embeddings_path: Path to the original ``.npy`` embedding matrix.
                Kept in memory so search results can be re-scored with exact
                cosine similarity instead of HNSW's approximate distances.
            ef_search: Query-time accuracy parameter passed to hnswlib's
                ``set_ef``. Higher values improve recall at the cost of speed.

        Raises:
            FileNotFoundError: If ``index_path`` does not exist.
        """
        import hnswlib

        index_file = Path(index_path)
        if not index_file.exists():
            raise FileNotFoundError(
                f"HNSW index file not found: {index_path}. "
                "Run 'python scripts/build_hnsw_indexes.py' to build it."
            )

        chunk_ids_file = Path(chunk_ids_path)
        with chunk_ids_file.open("r", encoding="utf-8") as handle:
            self._chunk_ids: list[str] = [str(cid) for cid in json.load(handle)]

        embeddings = np.load(embeddings_path)
        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype(np.float32)
        self._embeddings: np.ndarray = embeddings

        num_elements = len(self._chunk_ids)
        if self._embeddings.shape[0] != num_elements:
            raise ValueError(
                f"Embedding rows ({self._embeddings.shape[0]}) do not match "
                f"chunk-id count ({num_elements})."
            )

        dim = self._embeddings.shape[1]
        self._index: Any = hnswlib.Index(space="cosine", dim=dim)
        self._index.load_index(
            str(index_file),
            max_elements=max(num_elements * 2, 1024),
        )
        self._index.set_ef(ef_search)
        self._ef_search = ef_search

        logger.info(
            "Loaded HNSW index from %s (%d elements, dim=%d, ef_search=%d)",
            index_file,
            num_elements,
            dim,
            ef_search,
        )

    def search(
        self,
        query_vector: list[float],
        top_k: int,
        *,
        index_name: str | None = None,
        use_hnsw: bool = False,
    ) -> list[tuple[str, float]]:
        """Return top-k (chunk_id, cosine_similarity) pairs.

        ``index_name`` is accepted for API compatibility with multi-index stores
        but is ignored by this single-index implementation. ``use_hnsw`` is
        accepted for API compatibility with switchable stores but has no effect
        because this store is already HNSW-backed.
        """
        query = np.asarray(query_vector, dtype=np.float32)
        query_norm = query / (np.linalg.norm(query) + 1e-10)

        # Ask HNSW for a slightly larger candidate pool so exact re-ranking has
        # a better chance of recovering the true top-k.
        candidate_k = min(
            len(self._chunk_ids),
            max(top_k, self._ef_search, top_k * 2),
        )

        labels, _ = self._index.knn_query(query.reshape(1, -1), k=candidate_k)
        candidate_ids = labels[0].astype(int)

        candidate_embeddings = self._embeddings[candidate_ids]
        scores = candidate_embeddings @ query_norm

        effective_k = min(top_k, len(scores))
        top_indices = np.argpartition(scores, -effective_k)[-effective_k:]
        top_indices = top_indices[np.argsort(-scores[top_indices])]

        results: list[tuple[str, float]] = []
        for idx in top_indices:
            internal_id = int(candidate_ids[idx])
            chunk_id = self._chunk_ids[internal_id]
            results.append((chunk_id, float(scores[idx])))
        return results
