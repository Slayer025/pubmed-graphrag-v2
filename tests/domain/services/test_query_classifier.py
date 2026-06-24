"""Tests for the lightweight biomedical query classifier."""

from __future__ import annotations

import pytest

from src.domain.services.query_classifier import classify_query


def test_definition_query() -> None:
    result = classify_query("What is the definition of type 2 diabetes?")
    assert result["query_type"] == "definition"
    assert "what is" in result["matched_keywords"]
    assert result["detected_entities"] == []


def test_entity_specific_query_by_keywords() -> None:
    result = classify_query("Which gene mutations are linked to breast cancer?")
    assert result["query_type"] == "entity_specific"
    assert "gene" in result["matched_keywords"]


def test_entity_specific_query_by_token() -> None:
    result = classify_query("Explain the role of BRCA1 and TP53 in cancer.")
    assert result["query_type"] == "entity_specific"
    assert "BRCA1" in result["detected_entities"]
    assert "TP53" in result["detected_entities"]


def test_relationship_query() -> None:
    result = classify_query("Is obesity associated with heart disease?")
    assert result["query_type"] == "relationship"
    assert "associated with" in result["matched_keywords"]


def test_mechanism_query() -> None:
    result = classify_query("How does insulin regulate glucose uptake?")
    assert result["query_type"] == "mechanism"
    assert "how does" in result["matched_keywords"]


def test_comparison_query() -> None:
    result = classify_query("Compare chemotherapy versus radiotherapy.")
    assert result["query_type"] == "comparison"
    assert "compare" in result["matched_keywords"]


def test_vs_keyword_matches() -> None:
    result = classify_query("Aspirin vs ibuprofen for inflammation")
    assert result["query_type"] == "comparison"
    assert " vs " in result["matched_keywords"]


def test_general_fallback() -> None:
    result = classify_query("Recent advances in cancer treatment")
    assert result["query_type"] == "general"
    assert result["matched_keywords"] == []


def test_multiple_keywords_pick_most_specific() -> None:
    # "what is" is definition, "associated with" is relationship.
    # Definition is earlier in the pattern list, so it wins.
    result = classify_query("What is the relationship between sleep and memory?")
    assert result["query_type"] == "definition"
    assert "what is" in result["matched_keywords"]


def test_entity_detection_il6() -> None:
    result = classify_query("What is the function of IL-6?")
    assert "IL-6" in result["detected_entities"]


def test_entity_detection_egfr() -> None:
    result = classify_query("EGFR mutations in lung cancer")
    assert "EGFR" in result["detected_entities"]


def test_case_insensitivity() -> None:
    result = classify_query("WHAT IS ALZHEIMER'S DISEASE?")
    assert result["query_type"] == "definition"


def test_empty_query() -> None:
    result = classify_query("")
    assert result["query_type"] == "general"
    assert result["matched_keywords"] == []
    assert result["detected_entities"] == []


def test_whitespace_only_query() -> None:
    result = classify_query("   ")
    assert result["query_type"] == "general"


def test_non_string_input() -> None:
    result = classify_query(12345)
    assert result["query_type"] == "general"
