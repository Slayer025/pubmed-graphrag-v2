"""Streaming retrieve-and-generate use case.

This use case orchestrates vector search, optional hybrid/BM25 fusion, graph
expansion, metadata boosting, and streaming LLM answer generation.  It yields
application-neutral ``StreamEvent`` objects so the presentation layer can
consume them without any framework-specific imports.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from typing import Any, Protocol

from src.application.dto.search_config import SearchConfig
from src.application.ports import (
    ChunkRepository,
    GraphRepository,
    LLMClient,
    SparseRetriever,
)
from src.application.use_cases.generate_answer import GenerateAnswerUseCase
from src.application.use_cases.graph_expand import GraphExpandUseCase
from src.application.use_cases.metadata_boost import MetadataBoostService
from src.application.use_cases.rerank import RerankUseCase
from src.application.use_cases.vector_search import VectorSearchUseCase
from src.domain.entities.retrieval_result import RetrievalResult
from src.domain.entities.stream_events import (
    ChunksFound,
    GraphEvidenceFound,
    RetrievalStarted,
    StreamComplete,
    StreamEvent,
    TextChunkEvent,
)
from src.domain.services.rrf_fusion_service import RRFFusionService
from src.domain.value_objects.query import Query

logger = logging.getLogger(__name__)


class QueryClassifier(Protocol):
    """Port for query classification."""

    def classify_query(self, question: str) -> dict:
        """Return classification dict for the question."""
        ...


class StrategyRouter(Protocol):
    """Port for strategy routing."""

    def route_strategy(
        self,
        classification: dict,
        *,
        enable_multi_index: bool = False,
    ) -> dict:
        """Return strategy dict for the classification."""
        ...


class RetrieveAndGenerateStreamUseCase:
    """Stream retrieval events and generated answer tokens for a query.

    The generator yields:

    1. ``RetrievalStarted`` when the query is accepted.
    2. ``ChunksFound`` after vector/hybrid search + optional graph expansion.
    3. ``GraphEvidenceFound`` when a graph repository can supply entity evidence.
    4. ``TextChunkEvent`` for each token/chunk produced by ``llm_client.stream_answer``.
    5. ``StreamComplete`` when the pipeline finishes.
    """

    def __init__(
        self,
        vector_search: VectorSearchUseCase,
        llm_client: LLMClient,
        *,
        chunk_repository: ChunkRepository | None = None,
        graph_repository: GraphRepository | None = None,
        sparse_retriever: SparseRetriever | None = None,
        rrf_fusion_service: RRFFusionService | None = None,
        query_classifier: QueryClassifier | None = None,
        strategy_router: StrategyRouter | None = None,
        metadata_boost_service: MetadataBoostService | None = None,
    ) -> None:
        self.vector_search = vector_search
        self.llm_client = llm_client
        self.chunk_repository = chunk_repository
        self.graph_repository = graph_repository
        self.sparse_retriever = sparse_retriever
        self.rrf_fusion_service = rrf_fusion_service or RRFFusionService()
        self.query_classifier = query_classifier
        self.strategy_router = strategy_router
        self.metadata_boost_service = metadata_boost_service
        self._graph_expand: GraphExpandUseCase | None = None
        if graph_repository is not None:
            self._graph_expand = GraphExpandUseCase(graph_repository)
        self._rerank_use_case: RerankUseCase | None = None
        if chunk_repository is not None:
            self._rerank_use_case = RerankUseCase(chunk_repository)

    def _apply_strategy(
        self,
        query: Query,
        config: SearchConfig,
    ) -> tuple[SearchConfig, dict, dict, str | None]:
        """Return a possibly modified config plus classification, strategy, and index metadata."""
        if not config.enable_query_routing:
            return config, {}, {}, None

        classification: dict = {}
        strategy: dict = {}
        if self.query_classifier is not None:
            classification = self.query_classifier.classify_query(query.text)
        if self.strategy_router is not None:
            strategy = self.strategy_router.route_strategy(
                classification,
                enable_multi_index=config.enable_multi_index,
            )

        if not strategy:
            return config, classification, strategy, None

        index_name = (
            strategy.get("index_name")
            if config.enable_query_routing and config.enable_multi_index
            else None
        )

        logger.info(
            "QUERY ROUTING: type=%s strategy=%s index=%s reason=%s",
            classification.get("query_type", "general"),
            strategy.get("strategy_name", "unknown"),
            index_name or "default",
            strategy.get("reason", ""),
        )

        routed = SearchConfig(
            top_k=config.top_k,
            expand_depth=int(strategy.get("expand_depth", config.expand_depth)),
            max_entity_degree=config.max_entity_degree,
            max_expansion_per_entity=config.max_expansion_per_entity,
            max_expanded_nodes=config.max_expanded_nodes,
            alpha=config.alpha,
            depth_scores=config.depth_scores,
            use_hybrid=bool(strategy.get("use_hybrid", config.use_hybrid)),
            rrf_k=int(strategy.get("rrf_k", config.rrf_k)),
            max_results=config.max_results,
            enable_query_routing=config.enable_query_routing,
            enable_metadata_boost=config.enable_metadata_boost,
            metadata_boost_factor=config.metadata_boost_factor,
            default_index=config.default_index,
            enable_multi_index=config.enable_multi_index,
            index_name=index_name or config.default_index,
            use_hnsw=bool(strategy.get("use_hnsw", config.use_hnsw)),
        )
        return routed, classification, strategy, index_name

    def _apply_metadata_boost(
        self,
        results: list[RetrievalResult],
        query: Query,
        config: SearchConfig,
    ) -> list[RetrievalResult]:
        """Apply optional metadata-aware boosting and re-sort by combined_score."""
        if not config.enable_metadata_boost:
            return results
        if self.metadata_boost_service is None:
            logger.warning("Metadata boost enabled but no MetadataBoostService provided")
            return results
        if self.graph_repository is None:
            logger.warning("Metadata boost requires a GraphRepository")
            return results

        boosted = self.metadata_boost_service.apply_boost(
            results,
            query.text,
            config.metadata_boost_factor,
        )
        logger.info(
            "METADATA BOOST APPLIED: factor=%.2f, top_chunk=%s, top_score=%.4f",
            config.metadata_boost_factor,
            boosted[0].chunk_id if boosted else "none",
            boosted[0].combined_score if boosted else 0.0,
        )
        return sorted(boosted, key=lambda r: r.combined_score, reverse=True)

    def _collect_graph_evidence(
        self,
        results: list[RetrievalResult],
    ) -> list[dict]:
        """Collect unique entity evidence from the retrieved chunks."""
        if self.graph_repository is None or not results:
            return []

        seen: set[str] = set()
        entities: list[dict] = []
        for result in results:
            for entity_id in self.graph_repository.get_chunk_entities(result.chunk_id):
                if entity_id in seen:
                    continue
                seen.add(entity_id)
                label: str | None = None
                name: str | None = None
                if ":" in entity_id:
                    label, name = entity_id.split(":", 1)
                entities.append(
                    {
                        "entity_id": entity_id,
                        "label": label,
                        "name": name,
                        "article_id": result.article_id,
                        "chunk_id": result.chunk_id,
                    }
                )
        return entities

    def _rerank(
        self,
        search_results: list[tuple[str, float]],
        expanded: dict[str, tuple[int, float, str]],
        config: SearchConfig,
    ) -> list[RetrievalResult]:
        """Convert search + graph scores into ranked ``RetrievalResult`` objects."""
        if self._rerank_use_case is not None:
            return self._rerank_use_case.execute(search_results, expanded, config)

        # Fallback when no chunk repository is supplied: build minimal results.
        return [
            RetrievalResult(
                chunk_id=chunk_id,
                article_id="",
                text="",
                vector_score=score,
                graph_score=expanded.get(chunk_id, (0, 0.0, ""))[1],
                combined_score=score,
                depth=expanded.get(chunk_id, (0, 0, ""))[0],
                source=expanded.get(chunk_id, (0, 0.0, "vector"))[2] or "vector",
            )
            for chunk_id, score in search_results
        ]

    def execute(
        self,
        query: Query,
        config: SearchConfig,
    ) -> Iterator[StreamEvent]:
        """Yield streaming events for retrieval and answer generation.

        When ``enable_query_routing`` is enabled, the routed configuration is
        used for retrieval.  All other ``SearchConfig`` flags (hybrid, multi-index,
        HNSW, metadata boost) are respected.
        """
        yield RetrievalStarted(query=query.text)

        routed_config, _classification, _strategy, index_name = self._apply_strategy(
            query, config
        )

        logger.info("INDEX ROUTING: index=%s", index_name or "default")

        vector_results = self.vector_search.execute(
            query,
            routed_config,
            index_name=index_name,
            use_hnsw=routed_config.use_hnsw,
        )

        if not routed_config.use_hybrid or self.sparse_retriever is None:
            logger.info("RETRIEVAL: mode=dense_only")
            seed_ids = {chunk_id for chunk_id, _ in vector_results}
            search_results = vector_results
        else:
            logger.info("RETRIEVAL: mode=hybrid")
            sparse_results = self.sparse_retriever.search(query.text, routed_config.top_k)
            fused = self.rrf_fusion_service.fuse(
                [
                    {"chunk_id": chunk_id, "score": score}
                    for chunk_id, score in vector_results
                ],
                [
                    {"chunk_id": chunk_id, "score": score}
                    for chunk_id, score in sparse_results
                ],
                k=routed_config.rrf_k,
            )
            search_results = [
                (result.chunk_id, result.rrf_score) for result in fused[: routed_config.top_k]
            ]
            seed_ids = {chunk_id for chunk_id, _ in search_results}

        expanded: dict[str, tuple[int, float, str]] = {}
        if self._graph_expand is not None:
            expanded = self._graph_expand.execute(seed_ids, routed_config)

        results = self._rerank(search_results, expanded, routed_config)
        results = self._apply_metadata_boost(results, query, config)

        top_results = results[: config.max_results]
        yield ChunksFound(chunks=top_results)

        entities = self._collect_graph_evidence(top_results)
        if entities:
            yield GraphEvidenceFound(entities=entities)

        prompt = GenerateAnswerUseCase._build_prompt(query.text, top_results)
        logger.info("STREAMING ANSWER for query: %s", query.text)
        for token in self.llm_client.stream_answer(prompt):
            yield TextChunkEvent(token=token)

        yield StreamComplete()

    def execute_by_vector(
        self,
        query_vector: Any,
        config: SearchConfig,
        *,
        query_text: str = "",
        index_name: str | None = None,
    ) -> Iterator[StreamEvent]:
        """Stream retrieval and generation starting from a pre-computed vector.

        This mirrors ``RetrieveDocumentsUseCase.retrieve_by_vector`` and is useful
        when the caller has already embedded the query.  It skips query
        classification because there is no query text.
        """
        yield RetrievalStarted(query=query_text)

        vector_results = self.vector_search.search_by_vector(
            query_vector,
            config,
            index_name=index_name,
            use_hnsw=config.use_hnsw,
        )

        seed_ids = {chunk_id for chunk_id, _ in vector_results}
        expanded: dict[str, tuple[int, float, str]] = {}
        if self._graph_expand is not None:
            expanded = self._graph_expand.execute(seed_ids, config)

        results = self._rerank(vector_results, expanded, config)
        results = self._apply_metadata_boost(results, Query(query_text), config)

        top_results = results[: config.max_results]
        yield ChunksFound(chunks=top_results)

        entities = self._collect_graph_evidence(top_results)
        if entities:
            yield GraphEvidenceFound(entities=entities)

        prompt = GenerateAnswerUseCase._build_prompt(query_text, top_results)
        for token in self.llm_client.stream_answer(prompt):
            yield TextChunkEvent(token=token)

        yield StreamComplete()
