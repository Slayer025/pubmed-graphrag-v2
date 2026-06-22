"""LLM client implementations for the PubMed GraphRAG pipeline.

This module provides concrete ``LLMClient`` implementations that conform to the
protocol defined in ``src.rag_pipeline`` without modifying it:

* ``OpenAIClient`` — OpenAI-compatible chat completions API.
* ``OllamaClient`` — Local Ollama ``/api/generate`` endpoint.
* ``MockLLMClient`` — kept here as a re-export for convenience.

Configuration is read exclusively from environment variables; no secrets are
hard-coded.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from typing import Any

from src.application.ports import LLMClient

logger = logging.getLogger(__name__)

LLM_MODE_MOCK = "mock"
LLM_MODE_OPENAI = "openai"
LLM_MODE_OLLAMA = "ollama"
LLM_MODE_DISABLED_OPENAI_MISSING_KEY = "disabled_openai_missing_key"

__all__ = [
    "LLMClient",
    "LLMClientResult",
    "LLM_MODE_DISABLED_OPENAI_MISSING_KEY",
    "LLM_MODE_MOCK",
    "LLM_MODE_OPENAI",
    "LLM_MODE_OLLAMA",
    "MockLLMClient",
    "OpenAIClient",
    "OllamaClient",
    "create_llm_client",
    "create_llm_client_with_mode",
    "resolve_effective_llm_mode",
]


@dataclass(frozen=True)
class LLMClientResult:
    """LLM client plus explicit runtime mode for UI and logging."""

    client: LLMClient
    mode: str


_MOCK_KEYWORDS = frozenset(
    {"risk", "factor", "factors", "associated", "includes", "linked", "related"}
)
_MOCK_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "how",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "that",
        "the",
        "to",
        "was",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
        "with",
    }
)
_CHUNK_HEADER_RE = re.compile(
    r"\[(\d+)\] chunk_id=([^\s]+) article_id=([^\s]+) combined_score=([\d.]+)\n",
    re.MULTILINE,
)
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def _question_terms(question: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z]{3,}", question.lower())
        if token not in _MOCK_STOPWORDS
    }


def _split_sentences(text: str) -> list[str]:
    sentences = [segment.strip() for segment in _SENTENCE_SPLIT_RE.split(text) if segment.strip()]
    if sentences:
        return sentences
    return [text.strip()] if text.strip() else []


def _parse_answer_prompt(prompt: str) -> tuple[str, list[tuple[str, float, str]]] | None:
    """Parse ``GenerateAnswerUseCase`` prompts into question + ranked chunks."""
    if "Context:\n" not in prompt or "\nQuestion:" not in prompt:
        return None

    question_match = re.search(r"\nQuestion:\s*(.+?)\s*\n\nAnswer:\s*$", prompt, re.DOTALL)
    if question_match is None:
        return None

    context_section = prompt.split("Context:\n", 1)[1].split("\nQuestion:", 1)[0]
    chunks: list[tuple[str, float, str]] = []
    matches = list(_CHUNK_HEADER_RE.finditer(context_section))
    if not matches:
        return None

    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(context_section)
        text = context_section[start:end].strip()
        if text:
            chunks.append((match.group(2), float(match.group(4)), text))

    if not chunks:
        return None
    return question_match.group(1).strip(), chunks


def _score_sentence(sentence: str, question_words: set[str]) -> int:
    lower = sentence.lower()
    score = sum(1 for keyword in _MOCK_KEYWORDS if keyword in lower)
    sentence_words = set(re.findall(r"[a-z]{3,}", lower))
    score += len(sentence_words & question_words)
    return score


def _build_extractive_answer(question: str, chunks: list[tuple[str, float, str]]) -> str:
    """Build a grounded extractive answer from top-scoring retrieved chunks."""
    question_words = _question_terms(question)
    ranked_chunks = sorted(chunks, key=lambda item: item[1], reverse=True)[:5]

    selected_sentences: list[tuple[int, str, str]] = []
    for chunk_id, chunk_score, text in ranked_chunks:
        for sentence in _split_sentences(text):
            normalized = " ".join(sentence.split())
            if len(normalized) < 40:
                continue
            lower = normalized.lower()
            has_keyword = any(keyword in lower for keyword in _MOCK_KEYWORDS)
            sentence_words = set(re.findall(r"[a-z]{3,}", lower))
            has_overlap = bool(sentence_words & question_words)
            if not has_keyword and not has_overlap:
                continue
            sentence_score = _score_sentence(normalized, question_words)
            selected_sentences.append((sentence_score + int(chunk_score * 10), normalized, chunk_id))

    selected_sentences.sort(key=lambda item: item[0], reverse=True)

    bullets: list[str] = []
    source_ids: list[str] = []
    seen: set[str] = set()
    for _, sentence, chunk_id in selected_sentences:
        key = sentence.lower()
        if key in seen:
            continue
        seen.add(key)
        bullets.append(sentence)
        if chunk_id not in source_ids:
            source_ids.append(chunk_id)
        if len(bullets) >= 6:
            break

    if len(bullets) < 3:
        for chunk_id, _, text in ranked_chunks:
            for sentence in _split_sentences(text):
                normalized = " ".join(sentence.split())
                if len(normalized) < 20:
                    continue
                key = normalized.lower()
                if key in seen:
                    continue
                seen.add(key)
                bullets.append(normalized)
                if chunk_id not in source_ids:
                    source_ids.append(chunk_id)
                if len(bullets) >= 3:
                    break
            if len(bullets) >= 3:
                break

    if not bullets:
        return (
            "Answer:\n"
            "- No extractive summary could be built from the retrieved context.\n\n"
            "Sources:\n"
            + "\n".join(f"- {chunk_id}" for chunk_id, _, _ in ranked_chunks[:3])
        )

    answer_lines = "\n".join(f"- {bullet}" for bullet in bullets[:6])
    source_lines = "\n".join(f"- {chunk_id}" for chunk_id in source_ids[:5])
    return f"Answer:\n{answer_lines}\n\nSources:\n{source_lines}"


class MockLLMClient:
    """Offline extractive QA mock that summarizes retrieved context only."""

    def __init__(self, max_chars: int = 500) -> None:
        self.max_chars = max_chars

    def complete(self, prompt: str, **kwargs: Any) -> str:
        del kwargs
        parsed = _parse_answer_prompt(prompt)
        if parsed is not None:
            question, chunks = parsed
            return _build_extractive_answer(question, chunks)

        if "Decompose the following question" in prompt:
            question_match = re.search(r"Question:\s*(.+?)\s*\n\nOutput:\s*$", prompt, re.DOTALL)
            if question_match is not None:
                return f'["{question_match.group(1).strip()}"]'

        return (
            "[MOCK LLM] Provide retrieved context chunks to generate an extractive answer.\n\n"
            f"Prompt preview:\n{prompt[: self.max_chars]}..."
        )


class OpenAIClient:
    """OpenAI-compatible chat completion client.

    Reads ``OPENAI_API_KEY`` (required) and ``LLM_MODEL`` (optional, defaults to
    ``gpt-3.5-turbo``). ``OPENAI_BASE_URL`` can be set to target proxies or other
    OpenAI-compatible services.
    """

    DEFAULT_MODEL = "gpt-3.5-turbo"
    DEFAULT_BASE_URL = "https://api.openai.com/v1"
    DEFAULT_MAX_TOKENS = 512
    DEFAULT_TEMPERATURE = 0.3

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
    ) -> None:
        resolved_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not resolved_key:
            raise RuntimeError(
                "OpenAIClient requires OPENAI_API_KEY environment variable or api_key argument."
            )
        self.api_key = resolved_key
        self.model = model or os.environ.get("LLM_MODEL") or self.DEFAULT_MODEL
        self.base_url = (base_url or os.environ.get("OPENAI_BASE_URL") or self.DEFAULT_BASE_URL).rstrip("/")
        self.max_tokens = max_tokens
        self.temperature = temperature

        # Optional import — only needed when this client is instantiated.
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "OpenAI client requested but 'openai' package is not installed. "
                "Install it with: pip install openai"
            ) from exc
        self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def complete(self, prompt: str, **kwargs: Any) -> str:
        """Request a chat completion from the configured endpoint."""
        logger.info("Calling OpenAI-compatible model %s", self.model)
        messages = [{"role": "user", "content": prompt}]
        response = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
            temperature=kwargs.get("temperature", self.temperature),
        )
        content = response.choices[0].message.content or ""
        logger.info("OpenAI response received (%d chars)", len(content))
        return content.strip()


class OllamaClient:
    """Local Ollama ``/api/generate`` client.

    Reads ``OLLAMA_URL`` (optional, defaults to ``http://localhost:11434``) and
    ``LLM_MODEL`` (required). Uses plain ``requests`` so no extra heavy
    dependencies are required.
    """

    DEFAULT_URL = "http://localhost:11434"
    DEFAULT_OPTIONS: dict[str, Any] = {"temperature": 0.3, "num_predict": 512}

    def __init__(
        self,
        url: str | None = None,
        model: str | None = None,
        options: dict[str, Any] | None = None,
    ) -> None:
        self.url = (url or os.environ.get("OLLAMA_URL") or self.DEFAULT_URL).rstrip("/")
        self.model = model or os.environ.get("LLM_MODEL")
        if not self.model:
            raise RuntimeError(
                "OllamaClient requires LLM_MODEL environment variable or model argument."
            )
        self.options = options or self.DEFAULT_OPTIONS
        self._session = self._create_session()

    @staticmethod
    def _create_session():
        import requests

        return requests.Session()

    def complete(self, prompt: str, **kwargs: Any) -> str:
        """Generate text using the Ollama ``/api/generate`` endpoint."""
        logger.info("Calling Ollama model %s at %s", self.model, self.url)
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": kwargs.get("options", self.options),
        }
        response = self._session.post(
            f"{self.url}/api/generate",
            json=payload,
            timeout=kwargs.get("timeout", 120),
        )
        response.raise_for_status()
        data = response.json()
        content = data.get("response", "")
        logger.info("Ollama response received (%d chars)", len(content))
        return content.strip()


def _resolve_openai_api_key(api_key: str | None = None) -> str | None:
    resolved = (api_key or os.environ.get("OPENAI_API_KEY") or "").strip()
    return resolved or None


def resolve_effective_llm_mode(
    client_type: str,
    *,
    api_key: str | None = None,
) -> str:
    """Return the explicit runtime LLM mode for a requested client type."""
    normalized = client_type.lower().strip()
    if normalized == "openai":
        if not _resolve_openai_api_key(api_key):
            return LLM_MODE_DISABLED_OPENAI_MISSING_KEY
        return LLM_MODE_OPENAI
    if normalized == "ollama":
        return LLM_MODE_OLLAMA
    if normalized == "mock":
        return LLM_MODE_MOCK
    return LLM_MODE_MOCK


def create_llm_client_with_mode(
    client_type: str = "mock",
    *,
    api_key: str | None = None,
    model: str | None = None,
    base_url: str | None = None,
    ollama_url: str | None = None,
) -> LLMClientResult:
    """Factory returning both client and explicit ``effective_llm_mode``."""
    normalized = client_type.lower().strip()
    mode = resolve_effective_llm_mode(normalized, api_key=api_key)
    logger.info("LLM MODE: %s", mode)

    if mode == LLM_MODE_DISABLED_OPENAI_MISSING_KEY:
        logger.warning(
            "OpenAI selected but API key missing in Streamlit secrets. "
            "Running with mock client (mode=disabled_openai_missing_key)."
        )
        return LLMClientResult(client=MockLLMClient(), mode=mode)

    try:
        if mode == LLM_MODE_OPENAI:
            resolved_key = _resolve_openai_api_key(api_key)
            assert resolved_key is not None
            return LLMClientResult(
                client=OpenAIClient(api_key=resolved_key, model=model, base_url=base_url),
                mode=LLM_MODE_OPENAI,
            )
        if mode == LLM_MODE_OLLAMA:
            return LLMClientResult(
                client=OllamaClient(url=ollama_url, model=model),
                mode=LLM_MODE_OLLAMA,
            )
        return LLMClientResult(client=MockLLMClient(), mode=LLM_MODE_MOCK)
    except Exception as exc:
        logger.warning(
            "Failed to create LLM client %r (%s), falling back to mock",
            normalized,
            exc,
        )
        logger.info("LLM MODE: %s", LLM_MODE_MOCK)
        return LLMClientResult(client=MockLLMClient(), mode=LLM_MODE_MOCK)


def create_llm_client(
    client_type: str = "mock",
    *,
    api_key: str | None = None,
    model: str | None = None,
    base_url: str | None = None,
    ollama_url: str | None = None,
) -> LLMClient:
    """Factory for selecting an LLM client by name.

    Returns only the client. Use ``create_llm_client_with_mode`` when the
    explicit runtime mode is required (for example in Streamlit UI).
    """
    return create_llm_client_with_mode(
        client_type,
        api_key=api_key,
        model=model,
        base_url=base_url,
        ollama_url=ollama_url,
    ).client


def main() -> int:
    """Quick smoke test for LLM client selection."""
    import argparse

    parser = argparse.ArgumentParser(description="Smoke-test an LLM client.")
    parser.add_argument(
        "--client",
        choices=["mock", "openai", "ollama"],
        default="mock",
        help="LLM client type",
    )
    parser.add_argument("--prompt", default="What is PubMedQA?", help="Prompt to send")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    client = create_llm_client_with_mode(args.client).client
    answer = client.complete(args.prompt)
    print("\nAnswer:\n", answer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
