"""Strategy router: maps query classification to retrieval strategy.

This is pure domain logic. It takes the output of ``query_classifier`` and
returns a retrieval strategy dict without side effects.
"""

from __future__ import annotations

DEFAULT_RRF_K = 20

_STRATEGIES: dict[str, dict[str, object]] = {
    "definition": {
        "strategy_name": "dense_only",
        "use_hybrid": False,
        "use_graph_expansion": False,
        "expand_depth": 0,
    },
    "entity_specific": {
        "strategy_name": "hybrid_rrf",
        "use_hybrid": True,
        "use_graph_expansion": False,
        "expand_depth": 0,
    },
    "relationship": {
        "strategy_name": "hybrid_rrf_graph_expand",
        "use_hybrid": True,
        "use_graph_expansion": True,
        "expand_depth": 2,
    },
    "mechanism": {
        "strategy_name": "dense_graph_expand",
        "use_hybrid": False,
        "use_graph_expansion": True,
        "expand_depth": 2,
    },
    "comparison": {
        "strategy_name": "hybrid_rrf",
        "use_hybrid": True,
        "use_graph_expansion": False,
        "expand_depth": 0,
    },
    "general": {
        "strategy_name": "hybrid_rrf",
        "use_hybrid": True,
        "use_graph_expansion": False,
        "expand_depth": 0,
    },
}


def route_strategy(classification: dict) -> dict[str, object]:
    """Map a query classification to a retrieval strategy.

    Args:
        classification: Output of ``query_classifier`` containing at least
            ``query_type``, ``matched_keywords``, and ``detected_entities``.

    Returns:
        A dict describing the chosen strategy, including the tuned ``rrf_k``.
    """
    if not isinstance(classification, dict):
        classification = {}

    query_type = classification.get("query_type", "general")
    if query_type not in _STRATEGIES:
        query_type = "general"

    strategy = dict(_STRATEGIES[query_type])
    strategy["rrf_k"] = DEFAULT_RRF_K

    matched_keywords = classification.get("matched_keywords", [])
    first_keyword = matched_keywords[0] if matched_keywords else ""
    if first_keyword:
        reason = (
            f"Query type '{query_type}' detected with keyword '{first_keyword}'"
        )
    else:
        reason = f"Query type '{query_type}' detected"

    strategy["reason"] = reason
    return strategy
