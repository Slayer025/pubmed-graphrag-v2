"""Pure-domain Reciprocal Rank Fusion (RRF) service.

RRF merges ranked lists from different retrieval strategies without requiring
score normalization.  For each result list, every rank contributes
``1 / (k + rank)`` to the fused score of the corresponding chunk.  The constant
``k`` dampens the impact of low ranks; the literature default is ``60``.

This module contains no infrastructure or framework imports.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


DEFAULT_RRF_K = 60


@dataclass(frozen=True)
class RRFResult:
    """One fused result produced by the RRF service."""

    chunk_id: str
    rrf_score: float
    metadata: dict[str, Any]


class RRFFusionService:
    """Fuse multiple ranked result lists using Reciprocal Rank Fusion."""

    def fuse(
        self,
        *result_lists: list[dict[str, Any]],
        k: int = DEFAULT_RRF_K,
    ) -> list[RRFResult]:
        """Merge ranked lists and score by reciprocal rank.

        Args:
            result_lists: One or more ranked lists. Each inner list is assumed
                to be ordered from best (rank 1) to worst. Each item must be a
                mapping containing at least ``chunk_id`` and ``score`` keys.
                The ``score`` value is preserved in ``metadata`` but is not used
                for ranking.
            k: RRF damping constant. Defaults to 60.

        Returns:
            List of ``RRFResult`` objects sorted by descending ``rrf_score``.
        """
        scores: dict[str, float] = {}
        metadata: dict[str, dict[str, Any]] = {}

        for ranked_list in result_lists:
            for rank, item in enumerate(ranked_list, start=1):
                chunk_id = str(item.get("chunk_id", ""))
                if not chunk_id:
                    continue
                scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
                if chunk_id not in metadata:
                    metadata[chunk_id] = dict(item)

        fused = [
            RRFResult(
                chunk_id=chunk_id,
                rrf_score=score,
                metadata=metadata[chunk_id],
            )
            for chunk_id, score in scores.items()
        ]
        return sorted(fused, key=lambda r: (-r.rrf_score, r.chunk_id))
