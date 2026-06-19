"""Embedding generation for PubMed text chunks."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import numpy as np

from src.storage import (
    WARN_THRESHOLD_BYTES,
    configure_hf_home,
    estimate_model_download,
    format_bytes,
    iter_jsonl_gz,
    log_disk_estimate,
)

if False:  # noqa: SIM108
    from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_EMBEDDING_DIM = 384
DEFAULT_BATCH_SIZE = 64
DEFAULT_INPUT_PATH = Path("data/chunks/chunks_semantic.jsonl.gz")
DEFAULT_OUTPUT_PATH = Path("data/embeddings/semantic_embeddings.npy")
BYTES_PER_FLOAT32 = 4


def _configure_logging() -> None:
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )


def create_embedding_model(model_name: str = DEFAULT_MODEL_NAME) -> Any:
    """Initialize an embedding model.

    Args:
        model_name: Identifier or path of the embedding model to load.

    Returns:
        A loaded SentenceTransformer model instance.
    """
    from sentence_transformers import SentenceTransformer

    configure_hf_home()
    log_disk_estimate(estimate_model_download(model_name))
    logger.info("Loading embedding model %s (local_files_only=True)", model_name)
    t0 = time.perf_counter()
    model = SentenceTransformer(model_name, local_files_only=True)
    logger.info(
        "Embedding model loaded in %.2f seconds (device=%s)",
        time.perf_counter() - t0,
        getattr(model, "device", "unknown"),
    )
    return model


def load_semantic_chunks(input_path: Path | str = DEFAULT_INPUT_PATH) -> list[dict[str, Any]]:
    """Load semantic chunk records from gzip JSONL."""
    path = Path(input_path)
    logger.info("Loading semantic chunks from %s", path)
    chunks = list(iter_jsonl_gz(path))
    logger.info("Loaded %d semantic chunks", len(chunks))
    return chunks  # type: ignore[return-value]


def extract_chunk_texts(chunks: list[dict[str, Any]]) -> list[str]:
    """Return chunk text fields in file order."""
    return [str(chunk["text"]) for chunk in chunks]


def estimate_embeddings_bytes(
    num_chunks: int,
    embedding_dim: int = DEFAULT_EMBEDDING_DIM,
) -> int:
    """Estimate on-disk size for a float32 embedding matrix."""
    return num_chunks * embedding_dim * BYTES_PER_FLOAT32


def log_embeddings_estimate(num_chunks: int, output_path: Path = DEFAULT_OUTPUT_PATH) -> int:
    """Log estimated embedding output size and return byte estimate."""
    estimated_bytes = estimate_embeddings_bytes(num_chunks)
    logger.info(
        "Embedding output estimate: %d chunks x %d dims -> ~%s at %s",
        num_chunks,
        DEFAULT_EMBEDDING_DIM,
        format_bytes(estimated_bytes),
        output_path,
    )
    return estimated_bytes


def ensure_within_size_limit(estimated_bytes: int, limit_bytes: int = WARN_THRESHOLD_BYTES) -> None:
    """Abort before saving when estimated embedding output exceeds the disk budget."""
    if estimated_bytes > limit_bytes:
        raise RuntimeError(
            f"Estimated embedding output {format_bytes(estimated_bytes)} exceeds "
            f"limit of {format_bytes(limit_bytes)}; aborting."
        )
    logger.info(
        "Estimated embedding output %s is within limit %s",
        format_bytes(estimated_bytes),
        format_bytes(limit_bytes),
    )


def embed_texts(
    texts: list[str],
    model: Any,
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> np.ndarray:
    """Generate embeddings for a list of text strings.

    Args:
        texts: Input strings to embed.
        model: Loaded embedding model.
        batch_size: Encoding batch size.

    Returns:
        A 2-D array of shape ``(len(texts), embedding_dim)``.
    """
    if not texts:
        return np.empty((0, DEFAULT_EMBEDDING_DIM), dtype=np.float32)

    logger.info("Encoding %d texts in batches of %d", len(texts), batch_size)
    vectors = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        show_progress_bar=True,
    )
    return np.asarray(vectors, dtype=np.float32)


def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    """L2-normalize embedding rows for cosine similarity via dot product."""
    if embeddings.size == 0:
        return embeddings.astype(np.float32, copy=False)

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.clip(norms, a_min=1e-12, a_max=None)
    normalized = embeddings / norms
    logger.info("Normalized embeddings to unit length (shape=%s)", normalized.shape)
    return normalized.astype(np.float32, copy=False)


def save_embeddings(embeddings: np.ndarray, output_path: Path | str = DEFAULT_OUTPUT_PATH) -> Path:
    """Persist embedding matrix to disk."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, embeddings)
    logger.info("Saved embeddings shape=%s to %s (%s)", embeddings.shape, path, format_bytes(path.stat().st_size))
    return path


def load_embeddings(input_path: Path | str = DEFAULT_OUTPUT_PATH) -> np.ndarray:
    """Load a saved embedding matrix."""
    path = Path(input_path)
    embeddings = np.load(path)
    logger.info("Loaded embeddings shape=%s from %s", embeddings.shape, path)
    return embeddings


def embed_chunks(
    chunks: list[dict[str, Any]],
    model: Any,
    text_field: str = "text",
) -> list[dict[str, Any]]:
    """Attach embedding vectors to chunk records.

    Args:
        chunks: Chunk records produced by the chunking step.
        model: Loaded embedding model.
        text_field: Key used to read text from each chunk record.

    Returns:
        Chunk records augmented with an ``embedding`` field.
    """
    texts = [str(chunk[text_field]) for chunk in chunks]
    vectors = embed_texts(texts, model)
    enriched: list[dict[str, Any]] = []
    for chunk, vector in zip(chunks, vectors, strict=True):
        record = dict(chunk)
        record["embedding"] = vector.tolist()
        enriched.append(record)
    return enriched


def create_semantic_embeddings(
    input_path: Path | str = DEFAULT_INPUT_PATH,
    output_path: Path | str = DEFAULT_OUTPUT_PATH,
    model_name: str = DEFAULT_MODEL_NAME,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> np.ndarray:
    """Load semantic chunks, encode, normalize, and save embeddings.

    Args:
        input_path: Semantic chunk gzip JSONL from Phase 1 chunking.
        output_path: Destination ``.npy`` file.
        model_name: Sentence-transformers model identifier.
        batch_size: Batch size for encoding.

    Returns:
        Normalized embedding matrix of shape ``(n_chunks, embedding_dim)``.
    """
    _configure_logging()

    input_path = Path(input_path)
    output_path = Path(output_path)

    if not input_path.is_file():
        raise FileNotFoundError(f"Semantic chunk file not found: {input_path}")

    chunks = load_semantic_chunks(input_path)
    estimated_bytes = log_embeddings_estimate(len(chunks), output_path=output_path)
    ensure_within_size_limit(estimated_bytes)

    model = create_embedding_model(model_name)
    texts = extract_chunk_texts(chunks)
    embeddings = embed_texts(texts, model, batch_size=batch_size)
    normalized = normalize_embeddings(embeddings)
    save_embeddings(normalized, output_path)
    return normalized


if __name__ == "__main__":
    create_semantic_embeddings()
