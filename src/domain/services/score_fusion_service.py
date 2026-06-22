"""Pure-domain score fusion service.

Combines vector and graph expansion scores into a ranked list of
``RetrievalResult`` objects.  No IO, no frameworks.
"""

from __future__ import annotations

from src.domain.entities.chunk import Chunk
from src.domain.entities.retrieval_result import RetrievalResult
from src.domain.services.retrieval_dedup_service import (
    MAX_UNIQUE_CONTEXT_CHUNKS,
    deduplicate_retrieval_results,
)
from src.domain.value_objects.retrieval_hyperparameters import RetrievalHyperparameters


class ScoreFusionService:
    """Fuse vector scores with graph expansion scores."""

    def fuse(
        self,
        vector_results: list[tuple[str, float]],
        expanded: dict[str, tuple[int, float, str]],
        chunks: dict[str, Chunk],
        hyperparameters: RetrievalHyperparameters,
    ) -> list[RetrievalResult]:
        """Combine vector and graph scores into a ranked result list."""
        alpha = hyperparameters.alpha
        max_results = hyperparameters.max_results

        # Build O(1) lookup for vector scores.
        vector_scores = {chunk_id: score for chunk_id, score in vector_results}

        candidates: dict[str, RetrievalResult] = {}

        # Seed vector results with depth 0, graph_score 1.0.
        for chunk_id, vector_score in vector_results:
            chunk = chunks.get(chunk_id)
            if chunk is None:
                continue
            combined = alpha * vector_score + (1 - alpha) * 1.0
            candidates[chunk_id] = RetrievalResult(
                chunk_id=chunk_id,
                article_id=chunk.article_id,
                text=chunk.text,
                vector_score=vector_score,
                graph_score=1.0,
                combined_score=combined,
                depth=0,
                source="vector",
            )

        # Merge graph-expanded results.
        for chunk_id, (depth, graph_score, source) in expanded.items():
            chunk = chunks.get(chunk_id)
            if chunk is None:
                continue

            vector_score = vector_scores.get(chunk_id, 0.0)
            combined = alpha * vector_score + (1 - alpha) * graph_score

            if chunk_id in candidates:
                existing = candidates[chunk_id]
                if combined > existing.combined_score:
                    candidates[chunk_id] = RetrievalResult(
                        chunk_id=chunk_id,
                        article_id=chunk.article_id,
                        text=chunk.text,
                        vector_score=vector_score or existing.vector_score,
                        graph_score=graph_score,
                        combined_score=combined,
                        depth=min(depth, existing.depth),
                        source=existing.source if existing.source == "vector" else source,
                    )
            else:
                candidates[chunk_id] = RetrievalResult(
                    chunk_id=chunk_id,
                    article_id=chunk.article_id,
                    text=chunk.text,
                    vector_score=vector_score,
                    graph_score=graph_score,
                    combined_score=combined,
                    depth=depth,
                    source=source,
                )

        ranked = sorted(
            candidates.values(),
            key=lambda r: (-r.combined_score, r.chunk_id),
        )
        return deduplicate_retrieval_results(
            ranked,
            max_chunks=min(max_results, MAX_UNIQUE_CONTEXT_CHUNKS),
        )
