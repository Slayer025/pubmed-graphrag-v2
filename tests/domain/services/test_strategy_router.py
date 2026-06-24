"""Tests for the strategy router."""

from __future__ import annotations

import pytest

from src.domain.services.strategy_router import route_strategy


_REQUIRED_FIELDS = {
    "strategy_name",
    "use_hybrid",
    "use_graph_expansion",
    "rrf_k",
    "expand_depth",
    "reason",
}


def _classify(query_type: str, keywords: list[str] | None = None) -> dict:
    return {
        "query_type": query_type,
        "matched_keywords": keywords or [],
        "detected_entities": [],
    }


def test_definition_strategy() -> None:
    result = route_strategy(_classify("definition", ["what is"]))
    assert result["strategy_name"] == "dense_only"
    assert result["use_hybrid"] is False
    assert result["use_graph_expansion"] is False
    assert result["expand_depth"] == 0
    assert result["rrf_k"] == 20
    assert "definition" in result["reason"]


def test_entity_specific_strategy() -> None:
    result = route_strategy(_classify("entity_specific", ["gene"]))
    assert result["strategy_name"] == "hybrid_rrf"
    assert result["use_hybrid"] is True
    assert result["use_graph_expansion"] is False
    assert result["expand_depth"] == 0
    assert result["rrf_k"] == 20


def test_relationship_strategy() -> None:
    result = route_strategy(_classify("relationship", ["associated with"]))
    assert result["strategy_name"] == "hybrid_rrf_graph_expand"
    assert result["use_hybrid"] is True
    assert result["use_graph_expansion"] is True
    assert result["expand_depth"] == 2
    assert result["rrf_k"] == 20
    assert "associated with" in result["reason"]


def test_mechanism_strategy() -> None:
    result = route_strategy(_classify("mechanism", ["how does"]))
    assert result["strategy_name"] == "dense_graph_expand"
    assert result["use_hybrid"] is False
    assert result["use_graph_expansion"] is True
    assert result["expand_depth"] == 2
    assert result["rrf_k"] == 20


def test_comparison_strategy() -> None:
    result = route_strategy(_classify("comparison", ["compare"]))
    assert result["strategy_name"] == "hybrid_rrf"
    assert result["use_hybrid"] is True
    assert result["use_graph_expansion"] is False


def test_general_strategy() -> None:
    result = route_strategy(_classify("general", []))
    assert result["strategy_name"] == "hybrid_rrf"
    assert result["use_hybrid"] is True
    assert result["use_graph_expansion"] is False


def test_all_required_fields_present() -> None:
    result = route_strategy(_classify("relationship", ["linked to"]))
    assert set(result.keys()) >= _REQUIRED_FIELDS


def test_invalid_query_type_defaults_to_general() -> None:
    result = route_strategy(_classify("unknown_type", ["foo"]))
    assert result["strategy_name"] == "hybrid_rrf"
    assert result["use_hybrid"] is True
    assert "general" in result["reason"]


def test_empty_classification_defaults_to_general() -> None:
    result = route_strategy({})
    assert result["strategy_name"] == "hybrid_rrf"
    assert "query_type" not in result  # strategy does not echo input
    assert "general" in result["reason"]


def test_non_dict_input_defaults_to_general() -> None:
    result = route_strategy(None)
    assert result["strategy_name"] == "hybrid_rrf"


def test_reason_uses_first_matched_keyword() -> None:
    result = route_strategy(_classify("relationship", ["associated with", "linked to"]))
    assert "associated with" in result["reason"]


def test_reason_without_keywords() -> None:
    result = route_strategy(_classify("mechanism", []))
    assert result["reason"] == "Query type 'mechanism' detected"
