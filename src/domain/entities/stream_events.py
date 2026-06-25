"""Lightweight streaming events for the PubMed GraphRAG pipeline.

These dataclasses carry information from the Application/Domain layers to the
UI without importing any presentation framework. Every event records its
creation time so the presentation layer can prove that sources and graph
evidence arrive before the streaming answer finishes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from time import time
from typing import Iterator

from src.domain.entities.retrieval_result import RetrievalResult


@dataclass(frozen=True)
class RetrievalStarted:
    """Emitted when streaming retrieval begins for a query."""

    query: str
    timestamp: float = field(default_factory=time)


@dataclass(frozen=True)
class ChunksFound:
    """Emitted when the retriever has found a set of chunks."""

    chunks: list[RetrievalResult]
    timestamp: float = field(default_factory=time)


@dataclass(frozen=True)
class GraphEvidenceFound:
    """Emitted when graph-derived evidence has been collected."""

    entities: list[dict]
    timestamp: float = field(default_factory=time)


@dataclass(frozen=True)
class TextChunkEvent:
    """Emitted for each token/word produced by a streaming LLM."""

    token: str
    timestamp: float = field(default_factory=time)


@dataclass(frozen=True)
class StreamComplete:
    """Emitted when the streaming pipeline has finished."""

    timestamp: float = field(default_factory=time)


StreamEvent = RetrievalStarted | ChunksFound | GraphEvidenceFound | TextChunkEvent | StreamComplete


def is_stream_event(obj: object) -> bool:
    """Return True when ``obj`` is one of the known streaming event types."""
    return isinstance(
        obj,
        (RetrievalStarted, ChunksFound, GraphEvidenceFound, TextChunkEvent, StreamComplete),
    )


__all__ = [
    "RetrievalStarted",
    "ChunksFound",
    "GraphEvidenceFound",
    "TextChunkEvent",
    "StreamComplete",
    "StreamEvent",
    "is_stream_event",
]
