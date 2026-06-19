"""RAG pipeline combining graph-enhanced retrieval with optional generation.

Phase 3 implements retrieval only. ``generate()`` is a placeholder interface
with a clear contract for future OpenAI/Ollama integration.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Protocol

from src.config import AppConfig
from src.retriever import RetrievalResult, Retriever, create_retriever

logger = logging.getLogger(__name__)


class LLMClient(Protocol):
    """Protocol for future LLM generation backends (OpenAI, Ollama, etc.)."""

    def complete(self, prompt: str, **kwargs: Any) -> str:
        """Return a text completion for the given prompt."""
        ...


class MockLLMClient:
    """Placeholder LLM that echoes the prompt context.

    Useful for testing the RAG pipeline without external API keys.
    """

    def __init__(self, max_chars: int = 500) -> None:
        self.max_chars = max_chars

    def complete(self, prompt: str, **kwargs: Any) -> str:
        return (
            "[MOCK LLM] I would answer based on the retrieved context.\n\n"
            f"Prompt preview:\n{prompt[:self.max_chars]}..."
        )


@dataclass(frozen=True)
class RAGResponse:
    """Output of a single RAG query."""

    query: str
    context: list[RetrievalResult]
    answer: str


class RAGPipeline:
    """End-to-end RAG interface: retrieve, then generate."""

    def __init__(
        self,
        retriever: Retriever,
        llm: LLMClient | None = None,
    ) -> None:
        self.retriever = retriever
        self.llm = llm or MockLLMClient()

    def retrieve(self, query: str) -> list[RetrievalResult]:
        """Return ranked context chunks for the query."""
        return self.retriever.retrieve(query)

    def _build_prompt(self, query: str, context: list[RetrievalResult]) -> str:
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

    def generate(self, query: str, context: list[RetrievalResult] | None = None) -> RAGResponse:
        """Retrieve (if needed) and generate an answer.

        Args:
            query: User question.
            context: Optional pre-retrieved context. If None, retrieve is called.

        Returns:
            A ``RAGResponse`` containing the query, context, and generated answer.
        """
        if context is None:
            context = self.retrieve(query)

        prompt = self._build_prompt(query, context)
        logger.info("Generating answer for query: %s", query)
        answer = self.llm.complete(prompt)
        logger.info("Generated answer length: %d chars", len(answer))
        return RAGResponse(query=query, context=context, answer=answer)

    def run(self, query: str) -> RAGResponse:
        """Convenience alias for ``generate``."""
        return self.generate(query)


def create_rag_pipeline(config: AppConfig | None = None, llm: LLMClient | None = None) -> RAGPipeline:
    """Factory helper for building a fully configured RAG pipeline."""
    retriever = create_retriever(config)
    return RAGPipeline(retriever=retriever, llm=llm)
