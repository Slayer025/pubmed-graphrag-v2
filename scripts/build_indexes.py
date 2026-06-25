#!/usr/bin/env python3
"""Build alternate chunking indexes and their embedding matrices.

This script is strictly offline. It reads the raw PubMed abstracts, chunks them
using two lightweight strategies (fixed-size windows and regex sentence
splitting), embeds each chunk with ``all-MiniLM-L6-v2``, and writes the
resulting chunk JSONL and ``.npy`` embedding matrices.

No heavy NLP libraries (nltk/spacy) are used; splitting is regex-only.
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("HF_HOME", str(Path(tempfile.gettempdir()) / "hf_cache"))

from src.embeddings import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_MODEL_NAME,
    create_embedding_model,
    embed_texts,
    normalize_embeddings,
    save_embeddings,
)
from src.storage import iter_jsonl_gz

logger = logging.getLogger(__name__)

RAW_ARTICLES_PATH = Path("data/pubmed_5000.jsonl.gz")
CHUNKS_DIR = Path("data/chunks")
EMBEDDINGS_DIR = Path("data/embeddings")

FIXED_WINDOW_SIZE = 500
FIXED_OVERLAP = 50
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.?!])\s+(?=[A-Z0-9])")


def _configure_logging() -> None:
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )


def load_articles(path: Path) -> list[dict[str, Any]]:
    """Load raw PubMed articles from a gzip JSONL file."""
    articles: list[dict[str, Any]] = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                articles.append(json.loads(line))
    logger.info("Loaded %d raw articles from %s", len(articles), path)
    return articles


def _chunk_id(article_id: int | str, strategy: str, index: int) -> str:
    return f"{article_id}_{strategy}_{index:04d}"


def _move_to_word_boundary(text: str, position: int, direction: int = -1) -> int:
    """Shift ``position`` to the nearest word boundary inside ``text``."""
    if position <= 0 or position >= len(text):
        return position
    if direction < 0:
        while position > 0 and text[position] != " ":
            position -= 1
    else:
        while position < len(text) and text[position] != " ":
            position += 1
    return position


def chunk_fixed(
    article_id: int | str,
    text: str,
    *,
    window_size: int = FIXED_WINDOW_SIZE,
    overlap: int = FIXED_OVERLAP,
) -> list[dict[str, Any]]:
    """Create fixed-size character-window chunks with overlap."""
    chunks: list[dict[str, Any]] = []
    if not text:
        return chunks

    start = 0
    index = 0
    while start < len(text):
        end = min(start + window_size, len(text))
        if end < len(text):
            end = _move_to_word_boundary(text, end, direction=-1)
        if end <= start:
            end = min(start + window_size, len(text))

        chunk_text = text[start:end].strip()
        if chunk_text:
            chunks.append(
                {
                    "article_id": article_id,
                    "chunk_id": _chunk_id(article_id, "fixed", index),
                    "text": chunk_text,
                    "strategy": "fixed",
                }
            )
            index += 1

        next_start = end - overlap
        if next_start <= start:
            next_start = end
        next_start = _move_to_word_boundary(text, next_start, direction=1)
        if next_start >= len(text) or next_start <= start:
            break
        start = next_start

    return chunks


def chunk_sentence(
    article_id: int | str,
    text: str,
) -> list[dict[str, Any]]:
    """Create sentence-level chunks using regex-only splitting."""
    chunks: list[dict[str, Any]] = []
    if not text:
        return chunks

    sentences = [s.strip() for s in SENTENCE_SPLIT_RE.split(text) if s.strip()]
    for index, sentence in enumerate(sentences):
        chunks.append(
            {
                "article_id": article_id,
                "chunk_id": _chunk_id(article_id, "sentence", index),
                "text": sentence,
                "strategy": "sentence",
            }
        )
    return chunks


def save_chunks(chunks: list[dict[str, Any]], output_path: Path) -> None:
    """Persist chunk records to a gzip JSONL file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(output_path, "wt", encoding="utf-8") as handle:
        for chunk in chunks:
            handle.write(json.dumps(chunk, ensure_ascii=False) + "\n")
    logger.info("Saved %d chunks to %s", len(chunks), output_path)


def build_index(
    articles: list[dict[str, Any]],
    strategy: str,
    model: Any,
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> tuple[list[dict[str, Any]], "np.ndarray"]:
    """Chunk articles and embed the resulting texts."""
    import numpy as np

    chunker = chunk_fixed if strategy == "fixed" else chunk_sentence
    chunks: list[dict[str, Any]] = []
    for article in articles:
        chunks.extend(chunker(article["article_id"], article.get("abstract", "")))

    if not chunks:
        return chunks, np.empty((0, DEFAULT_EMBEDDING_DIM), dtype=np.float32)

    texts = [chunk["text"] for chunk in chunks]
    embeddings = embed_texts(texts, model, batch_size=batch_size)
    normalized = normalize_embeddings(embeddings)
    return chunks, normalized


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build fixed and sentence chunk indexes.")
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL_NAME,
        help="Sentence-transformers model identifier (default: all-MiniLM-L6-v2).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Embedding batch size.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing chunk/embedding files.",
    )
    parser.add_argument(
        "--strategies",
        nargs="+",
        choices=["fixed", "sentence"],
        default=["fixed", "sentence"],
        help="Which alternate indexes to build (default: both).",
    )
    return parser.parse_args()


def main() -> int:
    _configure_logging()
    args = _parse_args()

    if not RAW_ARTICLES_PATH.exists():
        logger.error("Raw articles not found: %s", RAW_ARTICLES_PATH)
        return 1

    CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)

    articles = load_articles(RAW_ARTICLES_PATH)
    if not articles:
        logger.error("No articles loaded; aborting.")
        return 1

    logger.info("Loading embedding model %s...", args.model)
    model = create_embedding_model(args.model)

    strategy_files = {
        "fixed": (CHUNKS_DIR / "chunks_fixed.jsonl.gz", EMBEDDINGS_DIR / "fixed_embeddings.npy"),
        "sentence": (
            CHUNKS_DIR / "chunks_sentence.jsonl.gz",
            EMBEDDINGS_DIR / "sentence_embeddings.npy",
        ),
    }

    for strategy in args.strategies:
        chunks_path, embeddings_path = strategy_files[strategy]
        if not args.force and chunks_path.exists() and embeddings_path.exists():
            logger.info("Skipping %s index: %s and %s already exist", strategy, chunks_path, embeddings_path)
            continue

        logger.info("Building %s index...", strategy)
        chunks, embeddings = build_index(articles, strategy, model, batch_size=args.batch_size)
        save_chunks(chunks, chunks_path)
        save_embeddings(embeddings, embeddings_path)
        logger.info("%s index complete: %d chunks, shape=%s", strategy, len(chunks), embeddings.shape)

    logger.info("Index build finished.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
