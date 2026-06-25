"""Tests for the HNSW vector store adapter.

These tests do not require the real ``hnswlib`` extension; a minimal fake
module is injected into ``sys.modules`` before importing the implementation.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
import pytest


class _FakeHnswIndex:
    """Minimal stand-in for hnswlib.Index used by HnswVectorStore."""

    def __init__(self, space: str, dim: int) -> None:
        self.space = space
        self.dim = dim
        self._ef: int | None = None
        self._loaded_path: str | None = None
        self._max_elements: int | None = None
        self._data: np.ndarray | None = None

    def init_index(self, *, max_elements: int, ef_construction: int, M: int) -> None:
        self._max_elements = max_elements

    def set_ef(self, ef: int) -> None:
        self._ef = ef

    def load_index(self, path: str, max_elements: int = 0) -> None:
        self._loaded_path = path
        if max_elements:
            self._max_elements = max_elements

    def add_items(self, data: np.ndarray, ids: np.ndarray | None = None) -> None:
        self._data = data

    def knn_query(
        self,
        query: np.ndarray,
        k: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return the k nearest neighbors.

        When populated with data via ``add_items``, this fake performs exact
        dot-product search.  Otherwise it simply returns the first ``k``
        internal IDs so tests can still exercise the calling code path.
        """
        query_vec = query[0]
        if self._data is None:
            labels = np.arange(min(k, 1000), dtype=np.int32)
            distances = np.zeros_like(labels, dtype=np.float32)
            return labels.reshape(1, -1), distances.reshape(1, -1)

        scores = self._data @ query_vec
        k = min(k, len(scores))
        top_indices = np.argpartition(scores, -k)[-k:]
        top_indices = top_indices[np.argsort(-scores[top_indices])]
        distances = 1.0 - scores[top_indices]
        return top_indices.reshape(1, -1), distances.reshape(1, -1)


class _FakeHnswlib(types.ModuleType):
    Index = _FakeHnswIndex


# Inject the fake hnswlib module before importing the store.
if "hnswlib" not in sys.modules:
    sys.modules["hnswlib"] = _FakeHnswlib("hnswlib")

from src.infrastructure.vector_store.hnsw_vector_store import HnswVectorStore


_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent


def _build_artifacts(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Create a small synthetic index, chunk-id sidecar, and embeddings file."""
    index_path = tmp_path / "test_index.bin"
    index_path.write_bytes(b"fake-hnsw-index")

    chunk_ids = ["chunk-0", "chunk-1", "chunk-2", "chunk-3", "chunk-4"]
    chunk_ids_path = tmp_path / "chunk_ids.json"
    chunk_ids_path.write_text(json.dumps(chunk_ids), encoding="utf-8")

    rng = np.random.default_rng(0)
    embeddings = rng.random((len(chunk_ids), 8)).astype(np.float32)
    # L2-normalize so dot product == cosine similarity.
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings_path = tmp_path / "embeddings.npy"
    np.save(embeddings_path, embeddings)

    return index_path, chunk_ids_path, embeddings_path


def _make_store(
    tmp_path: Path,
    ef_search: int = 100,
) -> tuple[HnswVectorStore, np.ndarray, list[str]]:
    index_path, chunk_ids_path, embeddings_path = _build_artifacts(tmp_path)
    embeddings = np.load(embeddings_path)
    chunk_ids = json.loads(chunk_ids_path.read_text(encoding="utf-8"))
    store = HnswVectorStore(
        str(index_path),
        str(chunk_ids_path),
        str(embeddings_path),
        ef_search=ef_search,
    )
    # Populate the fake index so its knn_query performs exact dot-product search.
    store._index._data = embeddings
    return store, embeddings, chunk_ids


def test_loads_chunk_ids_and_embeddings(tmp_path: Path) -> None:
    """The store exposes the sidecar data internally after construction."""
    store, embeddings, chunk_ids = _make_store(tmp_path)
    assert np.array_equal(store._embeddings, embeddings)
    assert store._chunk_ids == chunk_ids


def test_search_returns_top_k_with_exact_scores(tmp_path: Path) -> None:
    """Search uses HNSW candidates and re-ranks by exact cosine similarity."""
    store, embeddings, chunk_ids = _make_store(tmp_path)

    # L2-normalized query vector.
    query = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    query /= np.linalg.norm(query)

    results = store.search(query.tolist(), top_k=3)

    assert len(results) == 3
    chunk_ids_out, scores = zip(*results)

    # Scores are monotonically decreasing and within cosine range.
    assert list(scores) == sorted(scores, reverse=True)
    assert all(0.0 <= s <= 1.0 + 1e-6 for s in scores)

    # Verify scores match exact cosine similarity computed directly.
    exact_scores = embeddings @ query
    expected_top_indices = np.argsort(-exact_scores)[:3]
    expected = [(chunk_ids[i], float(exact_scores[i])) for i in expected_top_indices]
    assert results == expected


def test_search_ignores_index_name_for_compatibility(tmp_path: Path) -> None:
    """The optional index_name argument is accepted but ignored."""
    store, _, _ = _make_store(tmp_path)
    query = np.ones(8, dtype=np.float32)
    query /= np.linalg.norm(query)

    results_default = store.search(query.tolist(), top_k=2)
    results_named = store.search(query.tolist(), top_k=2, index_name="semantic")
    assert results_default == results_named


def test_ef_search_is_set_on_index(tmp_path: Path) -> None:
    """The ef_search value is forwarded to the underlying hnswlib index."""
    store, _, _ = _make_store(tmp_path, ef_search=50)
    assert store._ef_search == 50
    assert store._index._ef == 50


def test_missing_index_file_raises_clear_error(tmp_path: Path) -> None:
    """A non-existent index file produces a descriptive FileNotFoundError."""
    missing_index = tmp_path / "missing.bin"
    chunk_ids_path = tmp_path / "chunk_ids.json"
    embeddings_path = tmp_path / "embeddings.npy"
    chunk_ids_path.write_text(json.dumps(["c1"]), encoding="utf-8")
    np.save(embeddings_path, np.ones((1, 4), dtype=np.float32))

    with pytest.raises(FileNotFoundError, match="HNSW index file not found"):
        HnswVectorStore(
            str(missing_index),
            str(chunk_ids_path),
            str(embeddings_path),
        )


def test_mismatched_chunk_count_raises(tmp_path: Path) -> None:
    """A mismatch between embeddings and chunk IDs is rejected early."""
    index_path = tmp_path / "test_index.bin"
    index_path.write_bytes(b"fake-hnsw-index")
    chunk_ids_path = tmp_path / "chunk_ids.json"
    chunk_ids_path.write_text(json.dumps(["chunk-0", "chunk-1"]), encoding="utf-8")
    embeddings_path = tmp_path / "embeddings.npy"
    np.save(embeddings_path, np.ones((3, 4), dtype=np.float32))

    with pytest.raises(ValueError, match="Embedding rows .* do not match chunk-id count"):
        HnswVectorStore(
            str(index_path),
            str(chunk_ids_path),
            str(embeddings_path),
        )


def test_search_top_k_larger_than_collection(tmp_path: Path) -> None:
    """Asking for more results than available items returns all items."""
    store, embeddings, chunk_ids = _make_store(tmp_path)
    query = np.ones(8, dtype=np.float32)
    query /= np.linalg.norm(query)

    results = store.search(query.tolist(), top_k=100)
    assert len(results) == len(chunk_ids)
    assert [r[0] for r in results] == [
        chunk_ids[i] for i in np.argsort(-(embeddings @ query))
    ]
