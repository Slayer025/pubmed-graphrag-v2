"""Thin RAG orchestrator for the PubMed GraphRAG pipeline.

This module is a pure facade over application-layer use cases and ports.
It contains no concrete implementation imports.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from src.application.dto.search_config import SearchConfig
from src.application.ports import Decomposer, GraphReranker, LLMClient
from src.application.use_cases.generate_answer import GenerateAnswerUseCase
from src.application.use_cases.retrieve_documents import RetrieveDocumentsUseCase
from src.domain.entities.retrieval_result import RetrievalResult
from src.domain.services.retrieval_dedup_service import (
    MAX_UNIQUE_CONTEXT_CHUNKS,
    deduplicate_retrieval_results,
)
from src.domain.value_objects.query import Query

if TYPE_CHECKING:
    from src.config import RetrievalConfig

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RAGResponse:
    """Output of a single RAG query."""

    query: str
    context: list[RetrievalResult]
    answer: str


class RAGPipeline:
    """End-to-end RAG orchestrator."""

    def __init__(
        self,
        retrieve_documents: RetrieveDocumentsUseCase,
        generate_answer: GenerateAnswerUseCase | None = None,
        llm: LLMClient | None = None,
        decomposer: Decomposer | None = None,
        reranker: GraphReranker | None = None,
    ) -> None:
        """Initialize the pipeline with injected application-layer dependencies."""
        if retrieve_documents is None:
            raise ValueError("retrieve_documents is required.")
        self.retrieve_documents = retrieve_documents
        self.generate_answer = generate_answer
        self.llm = llm
        self.decomposer = decomposer
        self.reranker = reranker

    @staticmethod
    def _to_search_config(config: SearchConfig | RetrievalConfig) -> SearchConfig:
        """Convert a config object into the application-layer ``SearchConfig``."""
        if isinstance(config, SearchConfig):
            return config
        return SearchConfig.from_retrieval_config(config)

    def retrieve(
        self,
        query: str,
        config: SearchConfig | RetrievalConfig,
    ) -> list[RetrievalResult]:
        """Return ranked context chunks for the query."""
        search_config = self._to_search_config(config)
        results = self.retrieve_documents.execute(Query(query), search_config)
        return self._finalize_results(query, results, search_config.max_results)

    def retrieve_reranked(
        self,
        query: str,
        config: SearchConfig | RetrievalConfig,
    ) -> list[RetrievalResult]:
        """Retrieve and optionally apply graph re-ranking."""
        return self.retrieve(query, config)

    def retrieve_by_vector(
        self,
        query_vector: Any,
        config: SearchConfig | RetrievalConfig,
        *,
        query_text: str = "",
    ) -> list[RetrievalResult]:
        """Retrieve by a pre-computed query vector."""
        search_config = self._to_search_config(config)
        results = self.retrieve_documents.retrieve_by_vector(query_vector, search_config)
        return self._finalize_results(query_text, results, search_config.max_results)

    def retrieve_decomposed(
        self,
        query: str,
        config: SearchConfig | RetrievalConfig,
        *,
        apply_reranker: bool = True,
    ) -> tuple[list[str], list[RetrievalResult]]:
        """Retrieve for the original query and any decomposed sub-queries."""
        search_config = self._to_search_config(config)
        if self.decomposer is None:
            return [query], self.retrieve(query, search_config)

        sub_queries = self.decomposer.decompose(query)
        if len(sub_queries) <= 1:
            return sub_queries, self.retrieve(query, search_config)

        logger.info("Retrieving for %d sub-queries.", len(sub_queries))
        best_by_chunk: dict[str, RetrievalResult] = {}

        for sub_query in sub_queries:
            sub_results = self.retrieve_documents.execute(Query(sub_query), search_config)
            if apply_reranker and self.reranker is not None:
                sub_results = self.reranker.rerank(sub_query, sub_results)
            for result in sub_results:
                existing = best_by_chunk.get(result.chunk_id)
                if existing is None or result.combined_score > existing.combined_score:
                    best_by_chunk[result.chunk_id] = result

        merged = deduplicate_retrieval_results(
            list(best_by_chunk.values()),
            max_chunks=min(search_config.max_results, MAX_UNIQUE_CONTEXT_CHUNKS),
        )
        return sub_queries, merged

    def _finalize_results(
        self,
        query: str,
        results: list[RetrievalResult],
        max_results: int,
    ) -> list[RetrievalResult]:
        """Apply optional reranking, then deduplicate and cap context."""
        if self.reranker is not None:
            results = self.reranker.rerank(query, results)
        return deduplicate_retrieval_results(
            results,
            max_chunks=min(max_results, MAX_UNIQUE_CONTEXT_CHUNKS),
        )

    def generate(
        self,
        query: str,
        config: SearchConfig | RetrievalConfig | None = None,
        context: list[RetrievalResult] | None = None,
    ) -> RAGResponse:
        """Retrieve (if needed) and generate an answer."""
        if context is None:
            if config is None:
                raise ValueError("config is required when context is not provided")
            context = self.retrieve_reranked(query, config)

        if self.generate_answer is not None:
            answer = self.generate_answer.execute(Query(query), context)
        elif self.llm is not None:
            answer = self.llm.complete(self._build_prompt(query, context))
        else:
            raise ValueError(
                "Cannot generate an answer: provide generate_answer or llm at construction."
            )

        logger.info("Generated answer length: %d chars", len(answer))
        return RAGResponse(query=query, context=context, answer=answer)

    def generate_decomposed(
        self,
        query: str,
        config: SearchConfig | RetrievalConfig,
    ) -> RAGResponse:
        """Decompose the query, retrieve per sub-query, and generate an answer."""
        search_config = self._to_search_config(config)
        sub_queries, context = self.retrieve_decomposed(query, search_config)
        logger.info(
            "Generating answer for query using %d sub-question(s).",
            len(sub_queries),
        )
        return self.generate(query, context=context)

    def run(
        self,
        query: str,
        config: SearchConfig | RetrievalConfig,
    ) -> RAGResponse:
        """Convenience alias for ``generate``."""
        return self.generate(query, config)

    @staticmethod
    def _build_prompt(query: str, context: list[RetrievalResult]) -> str:
        """Build a grounded QA prompt from retrieved chunks."""
        prompt_parts = [
            "You are a biomedical research assistant. Answer the question using only the context below.\n",
            "Context:\n",
        ]
        for rank, result in enumerate(context, start=1):
            prompt_parts.append(
                f"[{rank}] chunk_id={result.chunk_id} article_id={result.article_id} "
                f"combined_score={result.combined_score:.4f}\n{result.text}\n"
            )
        prompt_parts.append(f"\nQuestion: {query}\n\nAnswer:")
        return "\n".join(prompt_parts)
