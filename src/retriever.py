"""Graph-enhanced retriever for the PubMed GraphRAG pipeline.

Retrieval is implemented offline using existing Phase 1/2 artifacts:

* ``data/embeddings/semantic_embeddings.npy`` — L2-normalized chunk embeddings
* ``data/chunks/chunks_semantic.jsonl.gz`` — chunk records with ``chunk_id``,
  ``article_id``, and ``text``
* ``data/graph/mentions.csv`` — Chunk → Entity edges
* ``data/graph/has_chunk.csv`` — Article → Chunk edges
* ``data/graph/entities.csv`` — entity metadata (for diagnostics/filters)

The pipeline performs query embedding, vector search, graph expansion,
deduplication, and re-ranking. No live Neo4j instance is required.
"""

from __future__ import annotations

import csv
import gzip
import json
import logging
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.config import AppConfig, RetrievalConfig
from src.embeddings import normalize_embeddings
from src.storage import iter_jsonl_gz

logger = logging.getLogger(__name__)

# TYPE_CHECKING-only import to avoid loading torch/sentence-transformers at import time.
if False:  # noqa: SIM108
    from sentence_transformers import SentenceTransformer


@dataclass(frozen=True)
class RetrievalResult:
    """One ranked context chunk returned by the retriever."""

    chunk_id: str
    article_id: str
    text: str
    vector_score: float
    graph_score: float
    combined_score: float
    depth: int
    source: str  # "vector", "same_article", or "shared_entity"


class ArtifactIndex:
    """Lightweight in-memory indexes built from repository artifacts."""

    def __init__(
        self,
        chunks: list[dict[str, Any]],
        embeddings: np.ndarray,
        mentions: list[dict[str, str]],
        has_chunk: list[dict[str, str]],
    ) -> None:
        self.chunks = chunks
        self.embeddings = embeddings

        # chunk index by row order (matches embedding matrix)
        self.chunk_by_id: dict[str, dict[str, Any]] = {}
        self.row_by_chunk_id: dict[str, int] = {}
        for row, chunk in enumerate(chunks):
            chunk_id = str(chunk["chunk_id"])
            self.chunk_by_id[chunk_id] = chunk
            self.row_by_chunk_id[chunk_id] = row

        # Article -> chunks
        self.article_chunks: dict[str, set[str]] = {}
        for rel in has_chunk:
            article_id = str(rel["article_id"])
            chunk_id = str(rel["chunk_id"])
            self.article_chunks.setdefault(article_id, set()).add(chunk_id)

        # Entity -> chunks
        self.entity_chunks: dict[str, set[str]] = {}
        for rel in mentions:
            entity_id = str(rel["entity_id"])
            chunk_id = str(rel["chunk_id"])
            self.entity_chunks.setdefault(entity_id, set()).add(chunk_id)

        # Chunk -> entities
        self.chunk_entities: dict[str, set[str]] = {}
        for rel in mentions:
            entity_id = str(rel["entity_id"])
            chunk_id = str(rel["chunk_id"])
            self.chunk_entities.setdefault(chunk_id, set()).add(entity_id)

        self.entity_degrees: dict[str, int] = {
            entity_id: len(chunks) for entity_id, chunks in self.entity_chunks.items()
        }

    @classmethod
    def load(cls, config: AppConfig) -> "ArtifactIndex":
        """Load and validate all required Phase 1/2 artifacts."""
        artifact = config.artifact

        logger.info("Loading chunks from %s", artifact.chunks_path)
        chunks = list(iter_jsonl_gz(artifact.chunks_path))

        logger.info("Loading embeddings from %s", artifact.embeddings_path)
        t0 = time.perf_counter()
        embeddings = np.load(artifact.embeddings_path)
        logger.info(
            "Loaded embeddings %s in %.4f seconds",
            embeddings.shape,
            time.perf_counter() - t0,
        )

        if embeddings.shape[0] != len(chunks):
            raise ValueError(
                f"Embedding rows ({embeddings.shape[0]}) do not match chunk count ({len(chunks)})."
            )

        expected_dim = config.embedding.embedding_dim
        if embeddings.shape[1] != expected_dim:
            raise ValueError(
                f"Embedding dimension ({embeddings.shape[1]}) does not match config ({expected_dim})."
            )

        # Ensure unit-length embeddings for cosine via dot product.
        t0 = time.perf_counter()
        embeddings = normalize_embeddings(embeddings)
        logger.info("Normalized embeddings in %.4f seconds", time.perf_counter() - t0)

        logger.info("Loading mentions from %s", artifact.mentions_path)
        mentions = _load_csv(artifact.mentions_path, ["chunk_id", "entity_id"])

        logger.info("Loading has_chunk from %s", artifact.has_chunk_path)
        has_chunk = _load_csv(artifact.has_chunk_path, ["article_id", "chunk_id"])

        logger.info("Loading entities from %s", artifact.entities_path)
        # Entities are loaded primarily for diagnostics; the index can function
        # without them if the file is missing, but fail fast if expected.
        _load_csv(artifact.entities_path, ["entity_id", "name", "label"])
        logger.info("Entities loaded; starting validation")

        # Validate that every mention references a known chunk.
        logger.info("Building chunk id set")
        chunk_id_set = cls._chunk_id_set(chunks)
        logger.info("Chunk id set built")
        unknown_chunks = {
            rel["chunk_id"] for rel in mentions if rel["chunk_id"] not in chunk_id_set
        }
        if unknown_chunks:
            sample = sorted(unknown_chunks)[:5]
            raise ValueError(f"mentions.csv references unknown chunk_ids: {sample}")

        logger.info("Building ArtifactIndex instance")
        index = cls(chunks, embeddings, mentions, has_chunk)
        logger.info(
            "ArtifactIndex ready: %d chunks, embeddings %s, %d mentions, %d has_chunk",
            len(chunks),
            embeddings.shape,
            len(mentions),
            len(has_chunk),
        )
        return index

    @staticmethod
    def _chunk_id_set(chunks: list[dict[str, Any]]) -> set[str]:
        return {str(chunk["chunk_id"]) for chunk in chunks}


def _load_csv(path: Path, expected_columns: list[str]) -> list[dict[str, str]]:
    """Load a CSV file and validate its header."""
    with open(path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV {path} has no header")
        missing = set(expected_columns) - set(reader.fieldnames)
        if missing:
            raise ValueError(f"CSV {path} missing columns: {missing}")
        rows = [dict(row) for row in reader]
    logger.debug("Loaded %d rows from %s", len(rows), path)
    return rows


class Retriever:
    """Graph-enhanced vector retriever backed by repository artifacts."""

    def __init__(self, index: ArtifactIndex, config: AppConfig) -> None:
        self.index = index
        self.config = config
        self.retrieval = config.retrieval
        self._model: Any | None = None

    def _get_model(self) -> Any:
        """Lazily load the sentence-transformers model (requires torch)."""
        if self._model is None:
            from src.embeddings import create_embedding_model

            self._model = create_embedding_model(self.config.embedding.model_name)
        return self._model

    def embed_query(self, query: str) -> np.ndarray:
        """Embed and normalize a query string."""
        model = self._get_model()
        logger.info("Embedding query (len=%d)", len(query))
        t0 = time.perf_counter()
        vector = model.encode(
            [query],
            batch_size=1,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        logger.info("Query embedded in %.2f seconds", time.perf_counter() - t0)
        vector = np.asarray(vector, dtype=np.float32)
        if self.config.embedding.normalize:
            vector = normalize_embeddings(vector)
        return vector[0]

    def vector_search(self, query_vector: np.ndarray, top_k: int | None = None) -> list[tuple[str, float]]:
        """Return top-k (chunk_id, cosine_similarity) by dot product."""
        if top_k is None:
            top_k = self.retrieval.top_k

        logger.info("Starting vector search (top_k=%d)", top_k)
        t0 = time.perf_counter()
        # Embeddings are L2-normalized, so cosine similarity == dot product.
        scores = self.index.embeddings @ query_vector
        # Sort descending.
        top_indices = np.argpartition(scores, -top_k)[-top_k:]
        top_indices = top_indices[np.argsort(-scores[top_indices])]

        results = []
        for idx in top_indices:
            chunk_id = str(self.index.chunks[int(idx)]["chunk_id"])
            results.append((chunk_id, float(scores[idx])))

        logger.info(
            "Vector search returned %d chunks in %.4f seconds",
            len(results),
            time.perf_counter() - t0,
        )
        return results

    def graph_expand(
        self,
        seed_chunk_ids: set[str],
    ) -> dict[str, tuple[int, float, str]]:
        """Expand from seed chunks via same-article and shared-entity edges.

        Bounded BFS with degree filtering and a hard cap on total expanded
        nodes to keep retrieval latency predictable.

        Returns a mapping chunk_id -> (depth, graph_score, source).
        """
        cfg = self.retrieval
        depth_scores = cfg.depth_scores
        max_depth = min(cfg.expand_depth, len(depth_scores) - 1)

        logger.info(
            "Starting graph expansion (seeds=%d, max_depth=%d, max_entity_degree=%d, max_nodes=%d)",
            len(seed_chunk_ids),
            max_depth,
            cfg.max_entity_degree,
            cfg.max_expanded_nodes,
        )
        t0 = time.perf_counter()

        expanded: dict[str, tuple[int, float, str]] = {}
        frontier: deque[tuple[str, int, str]] = deque(
            (chunk_id, 0, "vector") for chunk_id in seed_chunk_ids
        )
        # A chunk may be reached at multiple depths; only enqueue each
        # (chunk_id, depth) pair once.
        enqueued: set[tuple[str, int]] = set()

        for chunk_id in seed_chunk_ids:
            enqueued.add((chunk_id, 0))

        while frontier and len(expanded) < cfg.max_expanded_nodes:
            chunk_id, depth, source = frontier.popleft()

            graph_score = depth_scores[min(depth, len(depth_scores) - 1)]
            if chunk_id not in expanded or graph_score > expanded[chunk_id][1]:
                expanded[chunk_id] = (depth, graph_score, source)

            if depth >= max_depth:
                continue

            next_depth = depth + 1

            # 1) Same-article expansion: Article -> Chunk
            chunk = self.index.chunk_by_id.get(chunk_id)
            article_id = str(chunk["article_id"]) if chunk else ""
            if article_id:
                for related_chunk_id in self.index.article_chunks.get(article_id, set()):
                    key = (related_chunk_id, next_depth)
                    if key not in enqueued:
                        enqueued.add(key)
                        frontier.append((related_chunk_id, next_depth, "same_article"))

            # 2) Shared-entity expansion: Chunk -> Entity -> Chunk
            for entity_id in self.index.chunk_entities.get(chunk_id, set()):
                degree = self.index.entity_degrees.get(entity_id, 0)
                if degree > cfg.max_entity_degree:
                    continue

                related = self.index.entity_chunks.get(entity_id, set())
                if len(related) > cfg.max_expansion_per_entity:
                    related = sorted(related)[: cfg.max_expansion_per_entity]

                for related_chunk_id in related:
                    key = (related_chunk_id, next_depth)
                    if key not in enqueued:
                        enqueued.add(key)
                        frontier.append((related_chunk_id, next_depth, "shared_entity"))

        logger.info(
            "Graph expansion produced %d unique chunks from %d seeds in %.4f seconds",
            len(expanded),
            len(seed_chunk_ids),
            time.perf_counter() - t0,
        )
        return expanded

    def rerank(
        self,
        vector_results: list[tuple[str, float]],
        expanded: dict[str, tuple[int, float, str]],
    ) -> list[RetrievalResult]:
        """Combine vector and graph scores into a ranked result list."""
        alpha = self.retrieval.alpha

        logger.info(
            "Starting reranking (vector_results=%d, expanded=%d, alpha=%.2f)",
            len(vector_results),
            len(expanded),
            alpha,
        )
        t0 = time.perf_counter()

        # Seed vector results with depth 0, graph_score 1.0.
        candidates: dict[str, RetrievalResult] = {}
        for chunk_id, vector_score in vector_results:
            chunk = self.index.chunk_by_id.get(chunk_id)
            if chunk is None:
                continue
            article_id = str(chunk.get("article_id", ""))
            text = str(chunk.get("text", ""))
            combined = alpha * vector_score + (1 - alpha) * 1.0
            candidates[chunk_id] = RetrievalResult(
                chunk_id=chunk_id,
                article_id=article_id,
                text=text,
                vector_score=vector_score,
                graph_score=1.0,
                combined_score=combined,
                depth=0,
                source="vector",
            )

        # Merge graph-expanded results.
        for chunk_id, (depth, graph_score, source) in expanded.items():
            chunk = self.index.chunk_by_id.get(chunk_id)
            if chunk is None:
                continue

            vector_score = 0.0
            for seed_chunk_id, seed_score in vector_results:
                if seed_chunk_id == chunk_id:
                    vector_score = seed_score
                    break

            combined = alpha * vector_score + (1 - alpha) * graph_score

            if chunk_id in candidates:
                existing = candidates[chunk_id]
                # Keep the better combined score; prefer vector source.
                if combined > existing.combined_score:
                    candidates[chunk_id] = RetrievalResult(
                        chunk_id=chunk_id,
                        article_id=str(chunk.get("article_id", "")),
                        text=str(chunk.get("text", "")),
                        vector_score=vector_score or existing.vector_score,
                        graph_score=graph_score,
                        combined_score=combined,
                        depth=min(depth, existing.depth),
                        source=existing.source if existing.source == "vector" else source,
                    )
            else:
                candidates[chunk_id] = RetrievalResult(
                    chunk_id=chunk_id,
                    article_id=str(chunk.get("article_id", "")),
                    text=str(chunk.get("text", "")),
                    vector_score=vector_score,
                    graph_score=graph_score,
                    combined_score=combined,
                    depth=depth,
                    source=source,
                )

        ranked = sorted(candidates.values(), key=lambda r: r.combined_score, reverse=True)
        logger.info(
            "Reranking produced %d results in %.4f seconds",
            len(ranked),
            time.perf_counter() - t0,
        )
        return ranked[: self.retrieval.max_results]

    def retrieve_by_vector(
        self,
        query_vector: np.ndarray,
        *,
        query_text: str = "",
    ) -> list[RetrievalResult]:
        """Run graph-enhanced retrieval given an already-computed query vector."""
        logger.info(
            "Retrieving by vector (query=%s, shape=%s)",
            query_text[:60] if query_text else "",
            query_vector.shape,
        )
        t_total = time.perf_counter()

        vector_results = self.vector_search(query_vector)
        seed_ids = {chunk_id for chunk_id, _ in vector_results}

        expanded = self.graph_expand(seed_ids)
        deduped_count = len(expanded) - len(seed_ids.intersection(expanded))
        logger.info("Graph expansion added %d new chunks after deduplication", deduped_count)

        ranked = self.rerank(vector_results, expanded)
        logger.info(
            "Retrieval complete: %d results in %.4f seconds",
            len(ranked),
            time.perf_counter() - t_total,
        )
        return ranked

    def retrieve(self, query: str) -> list[RetrievalResult]:
        """Run the full retrieval pipeline for a query string."""
        logger.info("Retrieving for query: %s", query)

        query_vector = self.embed_query(query)
        logger.info("Query embedding shape: %s", query_vector.shape)

        return self.retrieve_by_vector(query_vector, query_text=query)


def create_retriever(config: AppConfig | None = None) -> Retriever:
    """Factory helper: load the artifact index and build a retriever."""
    if config is None:
        config = AppConfig.default()
    index = ArtifactIndex.load(config)
    return Retriever(index, config)
