"""Deduplicate and cap retrieval results before LLM context assembly."""

from __future__ import annotations

import logging

from src.domain.entities.retrieval_result import RetrievalResult

logger = logging.getLogger(__name__)

MAX_UNIQUE_CONTEXT_CHUNKS = 10


def deduplicate_retrieval_results(
    results: list[RetrievalResult],
    *,
    max_chunks: int = MAX_UNIQUE_CONTEXT_CHUNKS,
) -> list[RetrievalResult]:
    """Keep one result per chunk_id (highest combined_score), then rank and cap."""
    if not results:
        return []

    before = len(results)
    best_by_chunk: dict[str, RetrievalResult] = {}
    for result in results:
        existing = best_by_chunk.get(result.chunk_id)
        if existing is None or result.combined_score > existing.combined_score:
            best_by_chunk[result.chunk_id] = result

    ranked = sorted(
        best_by_chunk.values(),
        key=lambda r: (-r.combined_score, r.chunk_id),
    )
    capped = ranked[:max_chunks]
    after = len(capped)

    if before != after:
        logger.info("DEDUP: reduced %d chunks to %d unique chunks", before, after)

    return capped
