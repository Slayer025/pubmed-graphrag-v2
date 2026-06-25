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
        "index_name": "semantic",
    },
    "entity_specific": {
        "strategy_name": "hybrid_rrf",
        "use_hybrid": True,
        "use_graph_expansion": False,
        "expand_depth": 0,
        "index_name": "semantic",
    },
    "relationship": {
        "strategy_name": "hybrid_rrf_graph_expand",
        "use_hybrid": True,
        "use_graph_expansion": True,
        "expand_depth": 2,
        "index_name": "sentence",
    },
    "mechanism": {
        "strategy_name": "dense_graph_expand",
        "use_hybrid": False,
        "use_graph_expansion": True,
        "expand_depth": 2,
        "index_name": "sentence",
    },
    "comparison": {
        "strategy_name": "hybrid_rrf",
        "use_hybrid": True,
        "use_graph_expansion": False,
        "expand_depth": 0,
        "index_name": "semantic",
    },
    "general": {
        "strategy_name": "hybrid_rrf",
        "use_hybrid": True,
        "use_graph_expansion": False,
        "expand_depth": 0,
        "index_name": "semantic",
    },
}


def route_strategy(
    classification: dict,
    *,
    enable_multi_index: bool = False,
) -> dict[str, object]:
    """Map a query classification to a retrieval strategy.

    Args:
        classification: Output of ``query_classifier`` containing at least
            ``query_type``, ``matched_keywords``, and ``detected_entities``.
        enable_multi_index: If ``True``, the returned strategy includes an
            ``index_name`` chosen for the query type. If ``False``, the
            ``index_name`` key is omitted so the caller uses its default index.

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

    if not enable_multi_index:
        strategy.pop("index_name", None)

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
