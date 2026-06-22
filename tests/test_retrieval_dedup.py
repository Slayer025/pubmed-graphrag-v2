"""Tests for retrieval result deduplication."""

from __future__ import annotations

from src.domain.entities.retrieval_result import RetrievalResult
from src.domain.services.retrieval_dedup_service import deduplicate_retrieval_results


def _result(chunk_id: str, score: float) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk_id,
        article_id="a1",
        text=f"text for {chunk_id}",
        vector_score=score,
        graph_score=score,
        combined_score=score,
        depth=0,
        source="vector",
    )


def test_deduplicate_keeps_highest_score_per_chunk() -> None:
    results = [
        _result("c1", 0.5),
        _result("c1", 0.9),
        _result("c2", 0.7),
    ]

    deduped = deduplicate_retrieval_results(results, max_chunks=10)

    assert [r.chunk_id for r in deduped] == ["c1", "c2"]
    assert deduped[0].combined_score == 0.9


def test_deduplicate_stable_sort_by_chunk_id_tiebreaker() -> None:
    results = [_result("b", 0.8), _result("a", 0.8), _result("c", 0.5)]

    deduped = deduplicate_retrieval_results(results, max_chunks=10)

    assert [r.chunk_id for r in deduped] == ["a", "b", "c"]


def test_deduplicate_caps_at_max_chunks() -> None:
    results = [_result(f"c{i}", float(i) / 10) for i in range(12)]

    deduped = deduplicate_retrieval_results(results, max_chunks=10)

    assert len(deduped) == 10
    assert deduped[0].chunk_id == "c11"
