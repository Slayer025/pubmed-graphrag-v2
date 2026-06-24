"""Lightweight BM25 sparse retriever backed by in-memory chunks.

This adapter uses only the ``rank_bm25`` package and a pure-Python regex
tokenizer.  It intentionally avoids heavy NLP dependencies so it remains safe
to initialize inside a Streamlit Community Cloud container while still capturing
hyphenated biomedical terms like ``BRCA-1`` or ``non-small-cell``.
"""

from __future__ import annotations

import re
from typing import Any

from rank_bm25 import BM25Okapi


# Lightweight biomedical-aware tokenizer: keeps words, digits, and hyphenated
# tokens (e.g., "BRCA-1", "IL-6", "T-cell", "non-small-cell").
_TOKEN_RE = re.compile(r"\b[\w-]+\b")


def _tokenize(text: str) -> list[str]:
    """Return lowercase tokens from ``text`` using a regex tokenizer."""
    return [token for token in _TOKEN_RE.findall(text.lower())]


class BM25Retriever:
    """BM25 sparse retriever initialized from a list of chunk records."""

    def __init__(
        self,
        chunks: list[dict[str, Any]],
        *,
        text_field: str = "text",
        chunk_id_field: str = "chunk_id",
    ) -> None:
        """Build a BM25 index from the provided chunks.

        Args:
            chunks: Chunk records (e.g., from ``InMemoryChunkRepository``). Each
                record must contain the configured ``text_field`` and
                ``chunk_id_field``.
            text_field: Key holding the chunk text to index.
            chunk_id_field: Key holding the chunk identifier.
        """
        self._text_field = text_field
        self._chunk_id_field = chunk_id_field

        self._chunks: list[dict[str, Any]] = []
        self._chunk_ids: list[str] = []
        corpus: list[list[str]] = []

        for chunk in chunks:
            chunk_id = str(chunk.get(chunk_id_field, ""))
            text = str(chunk.get(text_field, ""))
            if not chunk_id or not text:
                continue
            self._chunks.append(chunk)
            self._chunk_ids.append(chunk_id)
            corpus.append(_tokenize(text))

        if not corpus:
            # rank_bm25 requires a non-empty corpus; store an empty index safely.
            self._bm25: BM25Okapi | None = None
            return

        self._bm25 = BM25Okapi(corpus)

    def search(self, query: str, top_k: int) -> list[tuple[str, float]]:
        """Return the top-k matching (chunk_id, bm25_score) pairs.

        Args:
            query: Raw query text.
            top_k: Maximum number of results to return. Values <= 0 return an
                empty list.

        Returns:
            Ranked list of ``(chunk_id, score)`` tuples, highest score first.
        """
        if top_k <= 0 or not self._bm25 or not query:
            return []

        query_tokens = _tokenize(query)
        if not query_tokens:
            return []

        scores = self._bm25.get_scores(query_tokens)
        # argsort descending
        ranked_indices = sorted(
            range(len(scores)),
            key=lambda idx: scores[idx],
            reverse=True,
        )[:top_k]

        return [
            (self._chunk_ids[idx], float(scores[idx]))
            for idx in ranked_indices
        ]
