#!/usr/bin/env python3
"""Build HNSW (Hierarchical Navigable Small World) indexes for all chunking strategies.

This is an offline, pre-deployment step. It loads the existing numpy embedding
matrices and chunk metadata, constructs an hnswlib index per strategy, and writes
``.bin`` index files plus ``.json`` chunk-id sidecars to ``data/hnsw/``.

Usage:
    python scripts/build_hnsw_indexes.py

Parameters (tunable balance of speed vs recall):
    - M = 16
    - ef_construction = 200
    - ef_search = 100
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.storage import iter_jsonl_gz

logger = logging.getLogger(__name__)

DATA_DIR = Path("data")
EMBEDDINGS_DIR = DATA_DIR / "embeddings"
CHUNKS_DIR = DATA_DIR / "chunks"
HNSW_DIR = DATA_DIR / "hnsw"

# Tunable HNSW parameters.
DEFAULT_M = 16
DEFAULT_EF_CONSTRUCTION = 200
DEFAULT_EF_SEARCH = 100

INDEXES: list[tuple[str, str, str]] = [
    ("semantic", "semantic_embeddings.npy", "chunks_semantic.jsonl.gz"),
    ("fixed", "fixed_embeddings.npy", "chunks_fixed.jsonl.gz"),
    ("sentence", "sentence_embeddings.npy", "chunks_sentence.jsonl.gz"),
]


def _configure_logging() -> None:
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )


def load_embeddings(path: Path) -> np.ndarray:
    """Load a numpy embedding matrix and ensure it is float32."""
    embeddings = np.load(path)
    if embeddings.dtype != np.float32:
        embeddings = embeddings.astype(np.float32)
    return embeddings


def load_chunk_ids(path: Path) -> list[str]:
    """Return chunk_id values in file order."""
    return [str(record["chunk_id"]) for record in iter_jsonl_gz(path)]


def build_hnsw_index(
    embeddings: np.ndarray,
    *,
    m: int = DEFAULT_M,
    ef_construction: int = DEFAULT_EF_CONSTRUCTION,
    ef_search: int = DEFAULT_EF_SEARCH,
) -> Any:
    """Build and return an hnswlib index for the given normalized embeddings."""
    import hnswlib

    num_elements, dim = embeddings.shape
    index = hnswlib.Index(space="cosine", dim=dim)
    # max_elements must be at least the current count; leave headroom for future growth.
    max_elements = max(num_elements * 2, 1024)
    index.init_index(
        max_elements=max_elements,
        ef_construction=ef_construction,
        M=m,
    )
    index.set_ef(ef_search)
    index.add_items(embeddings, ids=np.arange(num_elements, dtype=np.int32))
    return index


def save_index(
    index: Any,
    chunk_ids: list[str],
    output_dir: Path,
    name: str,
) -> tuple[Path, Path]:
    """Persist an HNSW index binary and its chunk-id sidecar."""
    output_dir.mkdir(parents=True, exist_ok=True)
    index_path = output_dir / f"{name}_index.bin"
    sidecar_path = output_dir / f"{name}_chunk_ids.json"

    index.save_index(str(index_path))
    with open(sidecar_path, "w", encoding="utf-8") as handle:
        json.dump(chunk_ids, handle, ensure_ascii=False)

    return index_path, sidecar_path


def build_index_for_strategy(
    name: str,
    embeddings_file: str,
    chunks_file: str,
    output_dir: Path,
    *,
    m: int,
    ef_construction: int,
    ef_search: int,
) -> dict[str, Any]:
    """Load data, build HNSW index, save binary + sidecar, return metadata."""
    embeddings_path = EMBEDDINGS_DIR / embeddings_file
    chunks_path = CHUNKS_DIR / chunks_file

    if not embeddings_path.exists():
        raise FileNotFoundError(f"Embeddings missing: {embeddings_path}")
    if not chunks_path.exists():
        raise FileNotFoundError(f"Chunks missing: {chunks_path}")

    logger.info("Building %s HNSW index from %s", name, embeddings_path)
    embeddings = load_embeddings(embeddings_path)
    chunk_ids = load_chunk_ids(chunks_path)

    if len(chunk_ids) != embeddings.shape[0]:
        raise ValueError(
            f"Chunk count ({len(chunk_ids)}) does not match embedding rows ({embeddings.shape[0]}) for {name}"
        )

    index = build_hnsw_index(
        embeddings,
        m=m,
        ef_construction=ef_construction,
        ef_search=ef_search,
    )
    index_path, sidecar_path = save_index(index, chunk_ids, output_dir, name)

    return {
        "name": name,
        "embeddings_shape": list(embeddings.shape),
        "index_path": str(index_path),
        "chunk_ids_path": str(sidecar_path),
        "num_elements": len(chunk_ids),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build HNSW indexes for PubMed GraphRAG.")
    parser.add_argument(
        "--m",
        type=int,
        default=DEFAULT_M,
        help="HNSW M parameter (default: 16).",
    )
    parser.add_argument(
        "--ef-construction",
        type=int,
        default=DEFAULT_EF_CONSTRUCTION,
        help="HNSW ef_construction parameter (default: 200).",
    )
    parser.add_argument(
        "--ef-search",
        type=int,
        default=DEFAULT_EF_SEARCH,
        help="HNSW ef_search parameter (default: 100).",
    )
    parser.add_argument(
        "--strategies",
        nargs="+",
        choices=["semantic", "fixed", "sentence"],
        default=["semantic", "fixed", "sentence"],
        help="Which indexes to build (default: all).",
    )
    return parser.parse_args()


def main() -> int:
    _configure_logging()
    args = _parse_args()

    HNSW_DIR.mkdir(parents=True, exist_ok=True)

    strategy_lookup = {name: (emb, chk) for name, emb, chk in INDEXES}
    summaries: list[dict[str, Any]] = []

    for name in args.strategies:
        embeddings_file, chunks_file = strategy_lookup[name]
        summary = build_index_for_strategy(
            name,
            embeddings_file,
            chunks_file,
            HNSW_DIR,
            m=args.m,
            ef_construction=args.ef_construction,
            ef_search=args.ef_search,
        )
        summaries.append(summary)
        logger.info(
            "%s index saved: %s (%d elements)",
            summary["name"],
            summary["index_path"],
            summary["num_elements"],
        )

    manifest_path = HNSW_DIR / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "m": args.m,
                "ef_construction": args.ef_construction,
                "ef_search": args.ef_search,
                "indexes": summaries,
            },
            handle,
            indent=2,
            ensure_ascii=False,
        )
    logger.info("HNSW manifest saved to %s", manifest_path)
    logger.info("HNSW index build complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
