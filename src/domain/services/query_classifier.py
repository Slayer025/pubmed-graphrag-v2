"""Lightweight biomedical query classifier.

This is pure domain logic: it classifies a natural-language question into a
query intent based on simple keyword/regex matching. It intentionally avoids
LLM calls and heavy NLP dependencies so it is safe to run on every request.
"""

from __future__ import annotations

import re

# Map query types to the keyword triggers that indicate them.  Order matters:
# earlier, more specific types are preferred when multiple patterns match.
# entity_specific is placed before relationship because terms like "gene"
# and "mutation" are more specific than generic relationship phrases.
_QUERY_TYPE_PATTERNS: list[tuple[str, list[str]]] = [
    ("definition", ["what is", "define", "definition of"]),
    ("comparison", ["compare", "versus", " vs ", "difference between"]),
    ("mechanism", ["how does", "mechanism", "pathway", "why does"]),
    ("entity_specific", ["mutation", "gene", "protein"]),
    ("relationship", ["associated with", "linked to", "correlated with", "relationship between"]),
]

# Match biomedical entity tokens: either uppercase letters with digits (BRCA1,
# TP53, IL-6) or short all-caps gene/protein symbols (EGFR, HER2).
_ENTITY_PATTERN = re.compile(r"\b[A-Z]{2,}[A-Z0-9-]*\d*\b")


def classify_query(question: str) -> dict[str, list[str]]:
    """Classify a biomedical question by intent and detect entity mentions.

    Args:
        question: Raw user query string.

    Returns:
        A dict with keys ``query_type``, ``matched_keywords``, and
        ``detected_entities``.
    """
    if not isinstance(question, str):
        question = str(question)

    text = question.lower().strip()
    if not text:
        return {
            "query_type": "general",
            "matched_keywords": [],
            "detected_entities": [],
        }

    matched_keywords: list[str] = []
    query_type = "general"

    for candidate_type, keywords in _QUERY_TYPE_PATTERNS:
        hits = [kw for kw in keywords if kw.lower() in text]
        if hits:
            query_type = candidate_type
            matched_keywords = hits
            break

    detected_entities = sorted(set(_ENTITY_PATTERN.findall(question)))

    # Entity-specific queries are also triggered when the text explicitly talks
    # about genes/proteins/mutations, even if no entity token was found.
    if query_type == "general" and detected_entities:
        query_type = "entity_specific"

    return {
        "query_type": query_type,
        "matched_keywords": matched_keywords,
        "detected_entities": detected_entities,
    }
