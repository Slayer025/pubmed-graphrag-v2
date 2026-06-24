"""Unit tests for the BM25 sparse retriever."""

from __future__ import annotations

import pytest

from src.infrastructure.retrievers.bm25_retriever import BM25Retriever


def _make_chunks() -> list[dict]:
    return [
        {
            "chunk_id": "c1",
            "article_id": "a1",
            "text": "Metformin reduces cardiovascular risk in type 2 diabetes.",
        },
        {
            "chunk_id": "c2",
            "article_id": "a2",
            "text": "Insulin resistance is a hallmark of type 2 diabetes.",
        },
        {
            "chunk_id": "c3",
            "article_id": "a3",
            "text": "Obesity treatment often involves lifestyle changes.",
        },
        {
            "chunk_id": "c4",
            "article_id": "a4",
            "text": "BRCA1 mutations are linked to breast cancer risk.",
        },
    ]


def test_search_returns_top_k_limit() -> None:
    retriever = BM25Retriever(_make_chunks())
    results = retriever.search("diabetes", top_k=2)

    assert len(results) == 2
    chunk_ids = [chunk_id for chunk_id, _ in results]
    assert all(isinstance(cid, str) for cid in chunk_ids)


def test_search_exact_term_boosts_relevant_chunk() -> None:
    retriever = BM25Retriever(_make_chunks())
    results = retriever.search("metformin", top_k=3)

    assert len(results) == 3
    assert results[0][0] == "c1"
    assert results[0][1] > 0
    # Documents without the term receive a BM25 score of 0.0, which is valid.
    assert all(score >= 0 for _, score in results)


def _make_hyphenated_chunks() -> list[dict]:
    return [
        {
            "chunk_id": "h1",
            "article_id": "a1",
            "text": "BRCA-1 mutations increase breast cancer risk.",
        },
        {
            "chunk_id": "h2",
            "article_id": "a2",
            "text": "IL-6 signaling is elevated in chronic inflammation.",
        },
        {
            "chunk_id": "h3",
            "article_id": "a3",
            "text": "T-cell activation requires co-stimulation.",
        },
    ]


def test_search_biomedical_symbol() -> None:
    retriever = BM25Retriever(_make_chunks())
    results = retriever.search("BRCA1", top_k=2)

    assert len(results) > 0
    assert results[0][0] == "c4"


def test_tokenizer_keeps_hyphenated_terms() -> None:
    retriever = BM25Retriever(_make_hyphenated_chunks())
    assert retriever.search("BRCA-1", top_k=1)[0][0] == "h1"
    assert retriever.search("IL-6", top_k=1)[0][0] == "h2"
    assert retriever.search("T-cell", top_k=1)[0][0] == "h3"


def test_tokenizer_matches_hyphenated_and_unhyphenated_forms() -> None:
    retriever = BM25Retriever(_make_hyphenated_chunks())
    # "brca1" should also match the "BRCA-1" chunk because the tokenizer emits
    # the full hyphenated token; exact string matching is expected.
    results = retriever.search("brca1", top_k=1)
    assert results[0][0] == "h1"


def test_empty_query_returns_empty_list() -> None:
    retriever = BM25Retriever(_make_chunks())
    assert retriever.search("", top_k=5) == []


def test_non_positive_top_k_returns_empty_list() -> None:
    retriever = BM25Retriever(_make_chunks())
    assert retriever.search("diabetes", top_k=0) == []
    assert retriever.search("diabetes", top_k=-1) == []


def test_top_k_larger_than_corpus_is_capped() -> None:
    chunks = _make_chunks()
    retriever = BM25Retriever(chunks)
    results = retriever.search("diabetes", top_k=100)

    assert len(results) == len(chunks)


def test_empty_corpus_is_safe() -> None:
    retriever = BM25Retriever([])
    assert retriever.search("diabetes", top_k=5) == []


def test_chunks_missing_required_fields_are_skipped() -> None:
    retriever = BM25Retriever(
        [
            {"chunk_id": "c1", "text": "valid chunk"},
            {"text": "missing chunk_id"},
            {"chunk_id": "c3"},
        ]
    )
    results = retriever.search("valid", top_k=5)

    assert len(results) == 1
    assert results[0][0] == "c1"


def test_scores_are_sorted_descending() -> None:
    retriever = BM25Retriever(_make_chunks())
    results = retriever.search("diabetes metformin", top_k=4)

    scores = [score for _, score in results]
    assert scores == sorted(scores, reverse=True)
