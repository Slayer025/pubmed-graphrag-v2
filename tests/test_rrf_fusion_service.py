"""Unit tests for the pure-domain RRF fusion service."""

from __future__ import annotations

import pytest

from src.domain.services.rrf_fusion_service import DEFAULT_RRF_K, RRFFusionService


def _dense() -> list[dict]:
    return [
        {"chunk_id": "c1", "score": 0.9},
        {"chunk_id": "c2", "score": 0.8},
        {"chunk_id": "c3", "score": 0.7},
    ]


def _sparse() -> list[dict]:
    return [
        {"chunk_id": "c3", "score": 12.0},
        {"chunk_id": "c4", "score": 10.0},
        {"chunk_id": "c1", "score": 8.0},
    ]


def test_core_rrf_math_with_default_k() -> None:
    service = RRFFusionService()
    # Single list: rank 1 and rank 2 with k=60.
    results = service.fuse(
        [
            {"chunk_id": "a", "score": 1.0},
            {"chunk_id": "b", "score": 0.5},
        ]
    )

    assert len(results) == 2
    assert results[0].chunk_id == "a"
    assert round(results[0].rrf_score, 6) == round(1.0 / (DEFAULT_RRF_K + 1), 6)
    assert results[1].chunk_id == "b"
    assert round(results[1].rrf_score, 6) == round(1.0 / (DEFAULT_RRF_K + 2), 6)


def test_core_rrf_math_with_custom_k() -> None:
    service = RRFFusionService()
    results = service.fuse(
        [{"chunk_id": "a", "score": 1.0}],
        k=10,
    )

    assert len(results) == 1
    assert round(results[0].rrf_score, 6) == round(1.0 / (10 + 1), 6)


def test_duplicate_chunk_scores_are_summed() -> None:
    service = RRFFusionService()
    results = service.fuse(_dense(), _sparse())

    by_id = {r.chunk_id: r for r in results}
    # c1: rank 1 in dense, rank 3 in sparse.
    expected_c1 = 1.0 / (DEFAULT_RRF_K + 1) + 1.0 / (DEFAULT_RRF_K + 3)
    assert by_id["c1"].rrf_score == pytest.approx(expected_c1)
    # c3: rank 3 in dense, rank 1 in sparse.
    expected_c3 = 1.0 / (DEFAULT_RRF_K + 3) + 1.0 / (DEFAULT_RRF_K + 1)
    assert by_id["c3"].rrf_score == pytest.approx(expected_c3)


def test_duplicate_chunks_top_two_positions() -> None:
    service = RRFFusionService()
    results = service.fuse(_dense(), _sparse())

    # c1 and c3 are the only chunks present in both lists, so they tie for the
    # top score.  Both must occupy positions 0 and 1 after tie-breaking by ID.
    top_ids = {results[0].chunk_id, results[1].chunk_id}
    assert top_ids == {"c1", "c3"}
    assert results[0].rrf_score == pytest.approx(results[1].rrf_score)


def test_both_lists_empty_returns_empty_list() -> None:
    service = RRFFusionService()
    assert service.fuse([]) == []
    assert service.fuse([], []) == []


def test_one_empty_list_uses_other_list_ranks() -> None:
    service = RRFFusionService()
    results = service.fuse(_dense(), [])

    assert len(results) == 3
    expected_scores = [1.0 / (DEFAULT_RRF_K + 1), 1.0 / (DEFAULT_RRF_K + 2), 1.0 / (DEFAULT_RRF_K + 3)]
    for result, expected in zip(results, expected_scores):
        assert result.rrf_score == pytest.approx(expected)


def test_results_sorted_descending_by_rrf_score() -> None:
    service = RRFFusionService()
    results = service.fuse(_dense(), _sparse())

    scores = [r.rrf_score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_metadata_is_preserved_from_first_seen_chunk() -> None:
    service = RRFFusionService()
    dense = [
        {"chunk_id": "c1", "score": 0.9, "source": "dense"},
    ]
    sparse = [
        {"chunk_id": "c1", "score": 8.0, "source": "sparse"},
    ]
    results = service.fuse(dense, sparse)

    assert len(results) == 1
    assert results[0].metadata["source"] == "dense"
    assert results[0].metadata["score"] == 0.9


def test_item_missing_chunk_id_is_skipped() -> None:
    service = RRFFusionService()
    results = service.fuse(
        [
            {"chunk_id": "c1", "score": 0.9},
            {"score": 0.8},  # missing chunk_id
        ]
    )

    assert len(results) == 1
    assert results[0].chunk_id == "c1"


def test_three_lists_are_fused() -> None:
    service = RRFFusionService()
    list_a = [{"chunk_id": "x", "score": 1.0}]
    list_b = [{"chunk_id": "x", "score": 1.0}]
    list_c = [{"chunk_id": "x", "score": 1.0}]

    results = service.fuse(list_a, list_b, list_c)

    assert len(results) == 1
    assert results[0].rrf_score == pytest.approx(3.0 / (DEFAULT_RRF_K + 1))
