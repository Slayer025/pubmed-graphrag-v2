"""Embedding client supporting local, remote HTTP, and HuggingFace API modes.

The client conforms to the ``EmbeddingService`` application port.  It is
intended for deployment environments such as Streamlit Community Cloud where a
co-located FastAPI server is not available; the "remote" proof is therefore
provided by an external API (HuggingFace Inference API or an externally hosted
service).

Configuration is read from environment variables / Streamlit secrets:

* ``EMBEDDING_PROVIDER`` - ``local`` (default), ``remote_http``, or ``huggingface_api``
* ``EMBEDDING_SERVICE_URL`` - URL for ``remote_http`` mode
* ``HF_API_TOKEN`` - HuggingFace API token for ``huggingface_api`` mode
* ``EMBEDDING_MODEL`` - model identifier, defaults to ``all-MiniLM-L6-v2``

If a remote call fails, the client gracefully falls back to the local
sentence-transformers model (if it can be loaded) so the Streamlit app never
crashes.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any

import httpx
import numpy as np

from src.embeddings import create_embedding_model, embed_texts, normalize_embeddings
from src.infrastructure.utils.secrets import scrub_secrets

logger = logging.getLogger(__name__)

DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_TIMEOUT_SECONDS = 30
DEFAULT_HF_API_URL = "https://router.huggingface.co/hf-inference/models"
DEFAULT_REMOTE_RETRIES = 3
DEFAULT_REMOTE_BACKOFF_SECONDS = 1.0


class EmbeddingClientResult:
    """Result returned by the embedding factory: client plus runtime metadata."""

    def __init__(
        self,
        client: "RemoteEmbeddingClient",
        provider: str,
        selected_provider: str,
        fallback_reason: str | None = None,
    ) -> None:
        self.client = client
        self.provider = provider
        self.selected_provider = selected_provider
        self.fallback_reason = fallback_reason


class RemoteEmbeddingClient:
    """Embedding client with local fallback and external remote modes."""

    def __init__(
        self,
        provider: str = "local",
        model_name: str | None = None,
        *,
        api_token: str | None = None,
        service_url: str | None = None,
        batch_size: int = 64,
        normalize: bool = True,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        cache_folder: str | None = None,
    ) -> None:
        """Initialize the embedding client.

        Args:
            provider: ``local``, ``remote_http``, or ``huggingface_api``.
            model_name: HuggingFace model identifier. Falls back to the configured
                default when omitted.
            api_token: API token for remote providers.
            service_url: Custom HTTP endpoint for ``remote_http`` mode.
            batch_size: Batch size used by the local model.
            normalize: Whether to L2-normalize local embeddings.
            timeout_seconds: HTTP timeout for remote calls.
            cache_folder: HuggingFace cache directory for the local fallback model.
        """
        self._provider = provider.lower().strip()
        self._model_name = model_name or DEFAULT_MODEL_NAME
        self._api_token = api_token
        self._service_url = service_url
        self._batch_size = batch_size
        self._normalize = normalize
        self._timeout_seconds = timeout_seconds
        self._cache_folder = cache_folder

        self._local_model: Any | None = None
        self._effective_provider = self._provider
        self._fallback_reason: str | None = None

        # Validate remote config eagerly. Missing credentials do not trigger an
        # immediate local model load so that construction stays lightweight and
        # safe inside pure-build guards.
        if self._provider == "huggingface_api" and not self._api_token:
            self._record_fallback("HF_API_TOKEN is not set; falling back to local model.")
            return

        if self._provider == "remote_http" and not self._service_url:
            self._record_fallback(
                "EMBEDDING_SERVICE_URL is not set; falling back to local model."
            )
            return

        # Local mode is lazy: do not load the sentence-transformers model here
        # so that this client is safe to instantiate inside pure-build guards.

    @property
    def provider(self) -> str:
        return self._effective_provider

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def fallback_reason(self) -> str | None:
        return self._fallback_reason

    def _load_local_model(self) -> None:
        """Load the sentence-transformers model into memory (cached)."""
        if self._local_model is None:
            logger.info("Loading local embedding model: %s", self._model_name)
            self._local_model = create_embedding_model(
                self._model_name, cache_folder=self._cache_folder
            )

    def _record_fallback(self, reason: str) -> None:
        """Switch to local mode and record why, without loading the model."""
        safe_reason = scrub_secrets(reason)
        logger.warning("%s", safe_reason)
        self._effective_provider = "local"
        self._fallback_reason = safe_reason

    def _fallback(self, reason: str) -> None:
        """Record a fallback reason, switch to local mode, and load the model."""
        self._record_fallback(reason)
        try:
            self._load_local_model()
        except Exception as exc:
            logger.error("Local fallback model failed to load: %s", scrub_secrets(str(exc)))
            self._local_model = None

    def _local_embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings using the local sentence-transformers model."""
        self._load_local_model()
        if self._local_model is None:
            raise RuntimeError("No local embedding model available for fallback.")
        vectors = embed_texts(texts, self._local_model, batch_size=self._batch_size)
        if self._normalize:
            vectors = normalize_embeddings(vectors)
        return vectors.tolist()

    @staticmethod
    def _with_retry(
        operation: Any,
        *,
        retries: int = DEFAULT_REMOTE_RETRIES,
        backoff_seconds: float = DEFAULT_REMOTE_BACKOFF_SECONDS,
    ) -> Any:
        """Run ``operation`` with simple exponential-backoff retry.

        Only transient network errors (connection/DNS/timeout) are retried;
        HTTP 4xx/5xx and malformed payloads raise immediately.
        """
        last_exception: Exception | None = None
        for attempt in range(retries):
            try:
                return operation()
            except httpx.NetworkError as exc:
                last_exception = exc
                logger.warning(
                    "Remote embedding network error on attempt %d/%d: %s",
                    attempt + 1,
                    retries,
                    scrub_secrets(str(exc)),
                )
                if attempt < retries - 1:
                    time.sleep(backoff_seconds * (2 ** attempt))
            except httpx.TimeoutException as exc:
                last_exception = exc
                logger.warning(
                    "Remote embedding timeout on attempt %d/%d: %s",
                    attempt + 1,
                    retries,
                    scrub_secrets(str(exc)),
                )
                if attempt < retries - 1:
                    time.sleep(backoff_seconds * (2 ** attempt))
        assert last_exception is not None
        raise last_exception

    def _huggingface_embed(self, texts: list[str]) -> list[list[float]]:
        """Call the HuggingFace Inference API for embeddings."""
        url = f"{DEFAULT_HF_API_URL}/{self._model_name}/pipeline/feature-extraction"
        headers: dict[str, str] = {}
        if self._api_token:
            headers["Authorization"] = f"Bearer {self._api_token}"

        logger.info("Calling HuggingFace Inference API: %s", url)

        def _call() -> list[list[float]]:
            with httpx.Client(timeout=self._timeout_seconds) as client:
                response = client.post(
                    url,
                    headers=headers,
                    json={"inputs": texts},
                )
                response.raise_for_status()
                payload = response.json()

            if isinstance(payload, list) and len(payload) == len(texts):
                return [np.asarray(vec, dtype=np.float32).tolist() for vec in payload]
            raise RuntimeError(f"Unexpected HuggingFace API response shape: {type(payload)}")

        return self._with_retry(_call)

    def _remote_http_embed(self, texts: list[str]) -> list[list[float]]:
        """Call a custom HTTP embedding service."""
        assert self._service_url is not None
        logger.info("Calling remote HTTP embedding service: %s", self._service_url)

        def _call() -> list[list[float]]:
            with httpx.Client(timeout=self._timeout_seconds) as client:
                response = client.post(
                    self._service_url,
                    json={"texts": texts, "model": self._model_name},
                )
                response.raise_for_status()
                payload = response.json()

            embeddings = payload.get("embeddings") if isinstance(payload, dict) else payload
            if isinstance(embeddings, list) and len(embeddings) == len(texts):
                return [np.asarray(vec, dtype=np.float32).tolist() for vec in embeddings]
            raise RuntimeError(f"Unexpected remote HTTP response shape: {type(payload)}")

        return self._with_retry(_call)

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one embedding vector per input text."""
        if not texts:
            return []

        if self._effective_provider == "local":
            return self._local_embed(texts)

        t0 = time.perf_counter()
        try:
            if self._effective_provider == "huggingface_api":
                vectors = self._huggingface_embed(texts)
            elif self._effective_provider == "remote_http":
                vectors = self._remote_http_embed(texts)
            else:
                raise ValueError(f"Unknown embedding provider: {self._effective_provider}")
            logger.info(
                "Embedding provider=%s model=%s latency=%.2f ms texts=%d",
                self._effective_provider,
                self._model_name,
                (time.perf_counter() - t0) * 1000,
                len(texts),
            )
            return vectors
        except Exception as exc:
            if self._effective_provider != "local":
                self._fallback(
                    f"Remote embedding failed ({type(exc).__name__}: {exc}); "
                    "falling back to local model."
                )
                return self._local_embed(texts)
            logger.error("Embedding failed: %s", scrub_secrets(str(exc)))
            raise

    def embed_query(self, query: str) -> list[float]:
        """Embed a single query string."""
        vectors = self.embed([query])
        return vectors[0]


def create_embedding_client(
    provider: str | None = None,
    *,
    model_name: str | None = None,
    api_token: str | None = None,
    service_url: str | None = None,
    batch_size: int = 64,
    normalize: bool = True,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    cache_folder: str | None = None,
) -> EmbeddingClientResult:
    """Factory for creating an embedding client with explicit mode reporting."""
    selected_provider = (provider or os.environ.get("EMBEDDING_PROVIDER", "local")).lower().strip()
    resolved_model = model_name or os.environ.get("EMBEDDING_MODEL") or DEFAULT_MODEL_NAME
    resolved_token = api_token or os.environ.get("HF_API_TOKEN")
    resolved_url = service_url or os.environ.get("EMBEDDING_SERVICE_URL")

    client = RemoteEmbeddingClient(
        provider=selected_provider,
        model_name=resolved_model,
        api_token=resolved_token,
        service_url=resolved_url,
        batch_size=batch_size,
        normalize=normalize,
        timeout_seconds=timeout_seconds,
        cache_folder=cache_folder,
    )

    return EmbeddingClientResult(
        client=client,
        provider=client.provider,
        selected_provider=selected_provider,
        fallback_reason=client.fallback_reason,
    )
