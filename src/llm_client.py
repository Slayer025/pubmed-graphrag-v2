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
import time
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
    "ANSWER_EVIDENCE_SUBTITLE",
    "UNABLE_TO_GENERATE_ANSWER",
    "safe_llm_complete",
    "create_llm_client",
    "create_llm_client_with_mode",
    "get_openai_package_version",
    "is_openai_package_installed",
    "log_llm_startup_diagnostics",
    "log_openai_package_version",
    "resolve_effective_llm_mode",
]


@dataclass(frozen=True)
class LLMClientResult:
    """LLM client plus explicit runtime mode for UI and logging."""

    client: LLMClient
    mode: str
    selected_mode: str
    fallback_reason: str | None = None


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
_MOCK_TOP_K_CHUNKS = 3
_MOCK_MIN_TOP_CHUNK_SCORE = 0.05
_MOCK_BULLET_LABELS = ("Risk factor", "Association", "Evidence")
ANSWER_EVIDENCE_SUBTITLE = "Answer generated from retrieved PubMed evidence."
UNABLE_TO_GENERATE_ANSWER = "Unable to generate answer from retrieved context."
_INSUFFICIENT_EVIDENCE = "Insufficient evidence in retrieved context."
_LEGACY_MODE_LABEL = "MODE: RETRIEVAL-ONLY (NO LLM REASONING)"
_NOISE_SECTION_WORDS = frozenset(
    {
        "abstract",
        "aim",
        "background",
        "conclusion",
        "design",
        "introduction",
        "methods",
        "objective",
        "purpose",
        "results",
        "setting",
    }
)
_SECTION_PREFIX_RE = re.compile(
    r"^(background|objective|methods|results|conclusion|introduction|abstract|purpose|aim|design|setting)\s*[:.\-]\s*",
    re.IGNORECASE,
)


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


def _lexical_overlap(sentence: str, question_words: set[str]) -> int:
    sentence_words = set(re.findall(r"[a-z]{3,}", sentence.lower()))
    return len(sentence_words & question_words)


def _sentence_score(sentence: str, question: str, question_words: set[str]) -> int:
    """Score a sentence by query overlap plus lightweight biomedical cues."""
    overlap = _lexical_overlap(sentence, question_words)
    if overlap == 0:
        return 0

    score = overlap * 10
    if re.search(r"\d", question) and re.search(r"\d", sentence):
        score += 1
    return score


def _best_sentence_for_chunk(
    text: str,
    question: str,
    question_words: set[str],
) -> str | None:
    """Return the single best overlapping sentence for a chunk, or None."""
    sentences = [" ".join(sentence.split()) for sentence in _split_sentences(text) if sentence.strip()]
    if not sentences or not question_words:
        return None

    best_sentence: str | None = None
    best_score = 0
    best_index = 0
    for index, sentence in enumerate(sentences):
        if _is_noisy_sentence(sentence):
            continue
        score = _sentence_score(sentence, question, question_words)
        if score == 0:
            continue
        if score > best_score or (score == best_score and index < best_index):
            best_score = score
            best_index = index
            best_sentence = sentence
    if best_sentence is not None:
        return _clean_extracted_sentence(best_sentence)
    return None


def _select_top_chunks(chunks: list[tuple[str, float, str]]) -> list[tuple[str, float, str]]:
    """Select top 3 chunks by retrieval ``combined_score`` (deterministic)."""
    return sorted(chunks, key=lambda item: (-item[1], item[0]))[:_MOCK_TOP_K_CHUNKS]


def _normalize_text(text: str) -> str:
    return " ".join(text.split())


def _clean_extracted_sentence(sentence: str) -> str:
    cleaned = _normalize_text(sentence)
    cleaned = _SECTION_PREFIX_RE.sub("", cleaned).strip()
    return cleaned


def _is_noisy_sentence(sentence: str) -> bool:
    cleaned = _clean_extracted_sentence(sentence)
    if not cleaned:
        return True
    first_token = cleaned.lower().split(None, 1)[0].rstrip(":")
    return first_token in _NOISE_SECTION_WORDS


def _sanitize_answer_text(answer: str) -> str:
    """Remove legacy debug labels from user-visible answer text."""
    if _LEGACY_MODE_LABEL in answer:
        answer = answer.replace(_LEGACY_MODE_LABEL, ANSWER_EVIDENCE_SUBTITLE)
    return answer.strip()


def _format_mock_answer(labeled_bullets: list[tuple[str, str]], source_ids: list[str]) -> str:
    lines = [ANSWER_EVIDENCE_SUBTITLE, "", "Answer:"]
    for label, text in labeled_bullets:
        lines.append(f"- {label}: {text}")
    lines.append("")
    lines.append("Sources:")
    for chunk_id in source_ids:
        lines.append(f"- {chunk_id}")
    return "\n".join(lines)


def _insufficient_evidence_answer() -> str:
    return f"{ANSWER_EVIDENCE_SUBTITLE}\n\nAnswer:\n- {_INSUFFICIENT_EVIDENCE}\n\nSources:"


def _build_extractive_answer(question: str, chunks: list[tuple[str, float, str]]) -> str:
    """Build a strict retrieval-only extractive answer from top-3 chunks."""
    ranked_chunks = _select_top_chunks(chunks)
    if not ranked_chunks or ranked_chunks[0][1] < _MOCK_MIN_TOP_CHUNK_SCORE:
        return _insufficient_evidence_answer()

    question_words = _question_terms(question)
    labeled_bullets: list[tuple[str, str]] = []
    source_ids: list[str] = []
    for chunk_id, _, text in ranked_chunks:
        sentence = _best_sentence_for_chunk(text, question, question_words)
        if sentence is None:
            continue
        label = _MOCK_BULLET_LABELS[len(labeled_bullets)]
        labeled_bullets.append((label, sentence))
        source_ids.append(chunk_id)
        if len(labeled_bullets) >= len(_MOCK_BULLET_LABELS):
            break

    deduped_bullets: list[tuple[str, str]] = []
    deduped_sources: list[str] = []
    seen: set[str] = set()
    for (label, bullet), chunk_id in zip(labeled_bullets, source_ids, strict=True):
        key = _normalize_text(bullet).lower()
        if key in seen:
            continue
        seen.add(key)
        deduped_bullets.append((label, bullet))
        deduped_sources.append(chunk_id)

    if not deduped_bullets:
        return _insufficient_evidence_answer()

    return _format_mock_answer(deduped_bullets, deduped_sources)


class MockLLMClient:
    """Retrieval-only extractive QA mock (no generative reasoning)."""

    def __init__(self, max_chars: int = 500) -> None:
        self.max_chars = max_chars

    def complete(self, prompt: str, **kwargs: Any) -> str:
        del kwargs
        return _sanitize_answer_text("".join(self.stream_answer(prompt)))

    def stream_answer(self, prompt: str, **kwargs: Any):
        """Yield the mock answer word-by-word to simulate streaming."""
        del kwargs
        parsed = _parse_answer_prompt(prompt)
        if parsed is not None:
            question, chunks = parsed
            answer = _build_extractive_answer(question, chunks)
        elif "Decompose the following question" in prompt:
            question_match = re.search(r"Question:\s*(.+?)\s*\n\nOutput:\s*$", prompt, re.DOTALL)
            answer = (
                f'["{question_match.group(1).strip()}"]'
                if question_match is not None
                else '["decompose me"]'
            )
        else:
            answer = (
                "[MOCK LLM] Provide retrieved context chunks to generate an extractive answer.\n\n"
                f"Prompt preview:\n{prompt[: self.max_chars]}..."
            )

        answer = _sanitize_answer_text(answer)
        words = answer.split(" ")
        for index, word in enumerate(words):
            token = word if index == len(words) - 1 else word + " "
            yield token
            time.sleep(0.05)


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
        """Request a chat completion; fall back to mock retrieval QA on any failure."""
        return "".join(self.stream_answer(prompt, **kwargs))

    def stream_answer(self, prompt: str, **kwargs: Any):
        """Stream a chat completion; fall back to mock retrieval QA on any failure."""
        logger.info("Calling OpenAI-compatible model %s", self.model)
        try:
            messages = [{"role": "user", "content": prompt}]
            response = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=kwargs.get("max_tokens", self.max_tokens),
                temperature=kwargs.get("temperature", self.temperature),
                stream=True,
            )
            for chunk in response:
                content = chunk.choices[0].delta.content or ""
                if content:
                    yield content
        except Exception as exc:
            logger.warning(
                "OpenAI API call failed (%s: %s); falling back to mock retrieval QA",
                type(exc).__name__,
                exc,
            )
            yield from MockLLMClient().stream_answer(prompt, **kwargs)


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
        return "".join(self.stream_answer(prompt, **kwargs))

    def stream_answer(self, prompt: str, **kwargs: Any):
        """Stream text using the Ollama ``/api/generate`` endpoint."""
        logger.info("Calling Ollama model %s at %s", self.model, self.url)
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": True,
            "options": kwargs.get("options", self.options),
        }
        response = self._session.post(
            f"{self.url}/api/generate",
            json=payload,
            stream=True,
            timeout=kwargs.get("timeout", 120),
        )
        response.raise_for_status()
        for line in response.iter_lines():
            if not line:
                continue
            import json

            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            content = data.get("response", "")
            if content:
                yield content
        logger.info("Ollama streaming response complete")


def _resolve_openai_api_key(api_key: str | None = None) -> str | None:
    resolved = (api_key or os.environ.get("OPENAI_API_KEY") or "").strip()
    return resolved or None


def is_openai_package_installed() -> bool:
    try:
        import openai  # noqa: F401
    except ImportError:
        return False
    return True


def get_openai_package_version() -> str | None:
    try:
        import openai
    except ImportError:
        return None
    return getattr(openai, "__version__", None)


def log_openai_package_version() -> None:
    if is_openai_package_installed():
        logger.info("OpenAI package version: %s", get_openai_package_version())
    else:
        logger.info("OpenAI package not installed")


def log_llm_startup_diagnostics(selected_mode: str, effective_mode: str) -> None:
    """Emit startup diagnostics for LLM configuration."""
    logger.info("OPENAI_API_KEY present: %s", bool(_resolve_openai_api_key()))
    logger.info("OpenAI package installed: %s", is_openai_package_installed())
    log_openai_package_version()
    logger.info("Selected mode: %s", selected_mode)
    logger.info("Effective mode: %s", effective_mode)


def _normalize_selected_mode(client_type: str) -> str:
    normalized = client_type.lower().strip()
    if normalized in {LLM_MODE_MOCK, LLM_MODE_OPENAI, LLM_MODE_OLLAMA}:
        return normalized
    return LLM_MODE_MOCK


def _mock_fallback(
    selected_mode: str,
    reason: str,
) -> LLMClientResult:
    logger.warning("%s Falling back to mock mode.", reason)
    return LLMClientResult(
        client=MockLLMClient(),
        mode=LLM_MODE_MOCK,
        selected_mode=selected_mode,
        fallback_reason=reason,
    )


def resolve_effective_llm_mode(
    client_type: str,
    *,
    api_key: str | None = None,
) -> str:
    """Return the effective runtime LLM mode for a requested client type."""
    return create_llm_client_with_mode(client_type, api_key=api_key).mode


def create_llm_client_with_mode(
    client_type: str = "mock",
    *,
    api_key: str | None = None,
    model: str | None = None,
    base_url: str | None = None,
    ollama_url: str | None = None,
) -> LLMClientResult:
    """Factory returning both client and explicit effective runtime mode."""
    selected_mode = _normalize_selected_mode(client_type)

    if selected_mode == LLM_MODE_OPENAI:
        if not _resolve_openai_api_key(api_key):
            reason = "OPENAI_API_KEY is not set."
            result = _mock_fallback(selected_mode, reason)
            logger.info("Effective mode: %s", result.mode)
            return result
        if not is_openai_package_installed():
            reason = "OpenAI package is not installed."
            result = _mock_fallback(selected_mode, reason)
            logger.info("Effective mode: %s", result.mode)
            return result

    try:
        if selected_mode == LLM_MODE_OPENAI:
            resolved_key = _resolve_openai_api_key(api_key)
            assert resolved_key is not None
            client = OpenAIClient(api_key=resolved_key, model=model, base_url=base_url)
            result = LLMClientResult(
                client=client,
                mode=LLM_MODE_OPENAI,
                selected_mode=selected_mode,
            )
            logger.info("Effective mode: %s", result.mode)
            return result
        if selected_mode == LLM_MODE_OLLAMA:
            result = LLMClientResult(
                client=OllamaClient(url=ollama_url, model=model),
                mode=LLM_MODE_OLLAMA,
                selected_mode=selected_mode,
            )
            logger.info("Effective mode: %s", result.mode)
            return result
        result = LLMClientResult(
            client=MockLLMClient(),
            mode=LLM_MODE_MOCK,
            selected_mode=selected_mode,
        )
        logger.info("Effective mode: %s", result.mode)
        return result
    except Exception as exc:
        reason = f"OpenAI initialization failed: {exc}"
        logger.exception(reason)
        result = _mock_fallback(selected_mode, reason)
        logger.info("Effective mode: %s", result.mode)
        return result


def safe_llm_complete(llm: LLMClient, prompt: str, **kwargs: Any) -> str:
    """Complete a prompt without raising; always return user-visible text."""
    from src.infrastructure.utils.secrets import scrub_secrets

    try:
        return _sanitize_answer_text(llm.complete(prompt, **kwargs))
    except Exception as exc:
        logger.warning(
            "LLM complete() failed (%s: %s); attempting mock retrieval fallback",
            type(exc).__name__,
            scrub_secrets(str(exc)),
        )
        try:
            return MockLLMClient().complete(prompt, **kwargs)
        except Exception as mock_exc:
            logger.exception("Mock retrieval fallback failed: %s", scrub_secrets(str(mock_exc)))
            return UNABLE_TO_GENERATE_ANSWER


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
