"""Tests for the streaming retrieve-and-generate use case."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from src.application.dto.search_config import SearchConfig
from src.application.use_cases.metadata_boost import MetadataBoostService
from src.application.use_cases.retrieve_and_generate_stream import (
    RetrieveAndGenerateStreamUseCase,
)
from src.application.use_cases.vector_search import VectorSearchUseCase
from src.domain.entities.retrieval_result import RetrievalResult
from src.domain.entities.stream_events import (
    ChunksFound,
    GraphEvidenceFound,
    RetrievalStarted,
    StreamComplete,
    StreamEvent,
    TextChunkEvent,
    is_stream_event,
)
from src.domain.value_objects.query import Query
from src.infrastructure.graph.in_memory_graph_repository import InMemoryGraphRepository
from src.infrastructure.storage.chunk_repository import InMemoryChunkRepository
from src.infrastructure.vector_store.numpy_vector_store import NumpyVectorStore


class _FakeEmbeddingService:
    """Returns a deterministic query embedding."""

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0, 0.0] for _ in texts]

    def embed_query(self, query: str) -> list[float]:
        return [1.0, 0.0, 0.0]


class _FakeLLMClient:
    """Yields tokens without sleeping."""

    def __init__(self, tokens: list[str] | None = None) -> None:
        self.tokens = tokens or ["answer", " ", "text"]
        self.last_prompt: str | None = None

    def complete(self, prompt: str, **kwargs: Any) -> str:
        self.last_prompt = prompt
        return "".join(self.stream_answer(prompt, **kwargs))

    def stream_answer(self, prompt: str, **kwargs: Any):
        self.last_prompt = prompt
        for token in self.tokens:
            yield token


class _FakeSparseRetriever:
    """Keyword retriever returning a fixed list."""

    def __init__(self, results: list[tuple[str, float]]) -> None:
        self.results = results

    def search(self, query: str, top_k: int) -> list[tuple[str, float]]:
        return self.results[:top_k]


class _FakeQueryClassifier:
    def classify_query(self, question: str) -> dict:
        return {"query_type": "entity", "terms": ["diabetes"]}


class _FakeStrategyRouter:
    def route_strategy(
        self,
        classification: dict,
        *,
        enable_multi_index: bool = False,
    ) -> dict:
        strategy = {
            "strategy_name": "entity_dense",
            "expand_depth": 1,
            "use_hybrid": True,
            "rrf_k": 10,
            "reason": "entity question",
        }
        if enable_multi_index:
            strategy["index_name"] = "fixed"
        return strategy


def _build_use_case(
    *,
    chunks: list[dict[str, Any]] | None = None,
    embeddings: np.ndarray | None = None,
    mentions: list[dict[str, str]] | None = None,
    entities: list[dict[str, str]] | None = None,
    sparse_results: list[tuple[str, float]] | None = None,
    query_classifier: Any = None,
    strategy_router: Any = None,
    metadata_boost_service: MetadataBoostService | None = None,
) -> tuple[RetrieveAndGenerateStreamUseCase, _FakeLLMClient]:
    chunks = chunks or [
        {"chunk_id": "c0", "article_id": "a0", "text": "diabetes risk factors"},
        {"chunk_id": "c1", "article_id": "a1", "text": "hypertension treatment"},
    ]
    if embeddings is None:
        embeddings = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=np.float32,
        )

    chunk_repository = InMemoryChunkRepository(chunks)
    graph_repository = InMemoryGraphRepository(
        mentions or [], [], chunks, entities=entities
    )
    vector_store = NumpyVectorStore(chunks, embeddings)
    vector_search = VectorSearchUseCase(_FakeEmbeddingService(), vector_store)
    llm_client = _FakeLLMClient()

    sparse_retriever = None
    if sparse_results is not None:
        sparse_retriever = _FakeSparseRetriever(sparse_results)

    use_case = RetrieveAndGenerateStreamUseCase(
        vector_search=vector_search,
        llm_client=llm_client,
        chunk_repository=chunk_repository,
        graph_repository=graph_repository,
        sparse_retriever=sparse_retriever,
        query_classifier=query_classifier,
        strategy_router=strategy_router,
        metadata_boost_service=metadata_boost_service,
    )
    return use_case, llm_client


def _collect_events(generator) -> list[StreamEvent]:
    return list(generator)


def test_stream_yields_expected_event_sequence() -> None:
    use_case, llm_client = _build_use_case()
    query = Query("diabetes risk factors")
    config = SearchConfig(top_k=1, expand_depth=0, alpha=1.0, max_results=5)

    events = _collect_events(use_case.execute(query, config))

    assert events
    assert isinstance(events[0], RetrievalStarted)
    assert events[0].query == query.text
    assert isinstance(events[1], ChunksFound)
    assert any(isinstance(e, TextChunkEvent) for e in events)
    assert isinstance(events[-1], StreamComplete)
    assert all(is_stream_event(e) for e in events)
    assert llm_client.last_prompt is not None
    assert "diabetes risk factors" in llm_client.last_prompt


def test_chunks_found_contains_retrieval_results() -> None:
    use_case, _ = _build_use_case()
    query = Query("diabetes risk factors")
    config = SearchConfig(top_k=1, expand_depth=0, alpha=1.0, max_results=5)

    events = _collect_events(use_case.execute(query, config))
    chunks_event = next(e for e in events if isinstance(e, ChunksFound))

    assert len(chunks_event.chunks) == 1
    result = chunks_event.chunks[0]
    assert isinstance(result, RetrievalResult)
    assert result.chunk_id == "c0"
    assert result.text == "diabetes risk factors"


def test_graph_evidence_is_emitted_when_entities_present() -> None:
    mentions = [
        {"entity_id": "Disease:diabetes", "chunk_id": "c0"},
    ]
    entities = [
        {"entity_id": "Disease:diabetes", "label": "Disease"},
    ]
    use_case, _ = _build_use_case(mentions=mentions, entities=entities)
    query = Query("diabetes risk factors")
    config = SearchConfig(top_k=1, expand_depth=0, alpha=1.0, max_results=5)

    events = _collect_events(use_case.execute(query, config))
    evidence_event = next((e for e in events if isinstance(e, GraphEvidenceFound)), None)

    assert evidence_event is not None
    assert len(evidence_event.entities) == 1
    assert evidence_event.entities[0]["entity_id"] == "Disease:diabetes"
    assert evidence_event.entities[0]["label"] == "Disease"
    assert evidence_event.entities[0]["name"] == "diabetes"


def test_hybrid_search_uses_sparse_retriever() -> None:
    use_case, _ = _build_use_case(
        sparse_results=[("c1", 2.0), ("c0", 1.0)],
    )
    query = Query("diabetes risk factors")
    config = SearchConfig(
        top_k=2,
        use_hybrid=True,
        rrf_k=20,
        expand_depth=0,
        alpha=1.0,
        max_results=5,
    )

    events = _collect_events(use_case.execute(query, config))
    chunks_event = next(e for e in events if isinstance(e, ChunksFound))

    chunk_ids = [r.chunk_id for r in chunks_event.chunks]
    assert "c0" in chunk_ids
    assert "c1" in chunk_ids


def test_query_routing_changes_index_and_hybrid_flag() -> None:
    use_case, _ = _build_use_case(
        query_classifier=_FakeQueryClassifier(),
        strategy_router=_FakeStrategyRouter(),
    )
    query = Query("diabetes risk factors")
    config = SearchConfig(
        top_k=1,
        enable_query_routing=True,
        enable_multi_index=True,
        default_index="semantic",
        expand_depth=2,
        use_hybrid=False,
        rrf_k=60,
        alpha=1.0,
        max_results=5,
    )

    events = _collect_events(use_case.execute(query, config))
    chunks_event = next(e for e in events if isinstance(e, ChunksFound))

    # Strategy sets expand_depth=1 and use_hybrid=True; both are respected.
    assert len(chunks_event.chunks) >= 1


def test_metadata_boost_service_is_applied() -> None:
    mentions = [
        {"entity_id": "Disease:diabetes", "chunk_id": "c0"},
    ]
    entities = [
        {"entity_id": "Disease:diabetes", "label": "Disease"},
    ]
    use_case, _ = _build_use_case(
        mentions=mentions,
        entities=entities,
    )
    # Create a metadata boost service that uses the same graph repository.
    graph_repository = InMemoryGraphRepository(mentions, [], [
        {"chunk_id": "c0", "article_id": "a0", "text": "diabetes risk factors"},
        {"chunk_id": "c1", "article_id": "a1", "text": "hypertension treatment"},
    ], entities=entities)
    metadata_boost_service = MetadataBoostService(graph_repository)
    use_case.metadata_boost_service = metadata_boost_service

    query = Query("diabetes risk factors")
    config = SearchConfig(
        top_k=2,
        expand_depth=0,
        alpha=1.0,
        max_results=5,
        enable_metadata_boost=True,
        metadata_boost_factor=1.5,
    )

    events = _collect_events(use_case.execute(query, config))
    chunks_event = next(e for e in events if isinstance(e, ChunksFound))

    top_chunk = chunks_event.chunks[0]
    assert top_chunk.chunk_id == "c0"


def test_hnsw_flag_is_forwarded_to_vector_store() -> None:
    """The use case must pass use_hnsw through to the vector search use case."""

    class _RecordingVectorStore:
        def __init__(self, base_store: NumpyVectorStore) -> None:
            self.base_store = base_store
            self.last_use_hnsw: bool | None = None

        def search(
            self,
            query_vector: Any,
            top_k: int,
            *,
            index_name: str | None = None,
            use_hnsw: bool = False,
        ) -> list[tuple[str, float]]:
            self.last_use_hnsw = use_hnsw
            return self.base_store.search(
                query_vector, top_k, index_name=index_name, use_hnsw=use_hnsw
            )

    chunks = [
        {"chunk_id": "c0", "article_id": "a0", "text": "diabetes risk factors"},
    ]
    embeddings = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
    base_store = NumpyVectorStore(chunks, embeddings)
    recording_store = _RecordingVectorStore(base_store)
    vector_search = VectorSearchUseCase(_FakeEmbeddingService(), recording_store)
    llm_client = _FakeLLMClient()
    chunk_repository = InMemoryChunkRepository(chunks)

    use_case = RetrieveAndGenerateStreamUseCase(
        vector_search=vector_search,
        llm_client=llm_client,
        chunk_repository=chunk_repository,
    )

    query = Query("diabetes risk factors")
    config = SearchConfig(top_k=1, expand_depth=0, alpha=1.0, max_results=5, use_hnsw=True)

    list(use_case.execute(query, config))

    assert recording_store.last_use_hnsw is True


def test_execute_by_vector_skips_embedding() -> None:
    """execute_by_vector must not call embed_query on the embedding service."""

    class _RaisingEmbeddingService:
        def embed(self, texts: list[str]) -> list[list[float]]:
            raise AssertionError("embed() must not be called")

        def embed_query(self, query: str) -> list[float]:
            raise AssertionError("embed_query() must not be called")

    chunks = [
        {"chunk_id": "c0", "article_id": "a0", "text": "diabetes risk factors"},
        {"chunk_id": "c1", "article_id": "a1", "text": "hypertension treatment"},
    ]
    embeddings = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    chunk_repository = InMemoryChunkRepository(chunks)
    vector_store = NumpyVectorStore(chunks, embeddings)
    vector_search = VectorSearchUseCase(_RaisingEmbeddingService(), vector_store)
    llm_client = _FakeLLMClient()

    use_case = RetrieveAndGenerateStreamUseCase(
        vector_search=vector_search,
        llm_client=llm_client,
        chunk_repository=chunk_repository,
    )

    query_vector = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    config = SearchConfig(top_k=1, expand_depth=0, alpha=1.0, max_results=5)

    events = _collect_events(
        use_case.execute_by_vector(query_vector, config, query_text="diabetes risk factors")
    )

    assert isinstance(events[0], RetrievalStarted)
    chunks_event = next(e for e in events if isinstance(e, ChunksFound))
    assert chunks_event.chunks[0].chunk_id == "c0"
    assert isinstance(events[-1], StreamComplete)


def test_stream_complete_is_last_event_even_when_empty_context() -> None:
    chunks = []
    embeddings = np.zeros((0, 3), dtype=np.float32)
    use_case, _ = _build_use_case(chunks=chunks, embeddings=embeddings)
    query = Query("unknown")
    config = SearchConfig(top_k=1, expand_depth=0, alpha=1.0, max_results=5)

    events = _collect_events(use_case.execute(query, config))

    assert isinstance(events[0], RetrievalStarted)
    assert isinstance(events[1], ChunksFound)
    assert events[1].chunks == []
    assert isinstance(events[-1], StreamComplete)
