#!/usr/bin/env python3
"""Manual debug script for the Phase 3 graph-enhanced retriever.

Usage:
    # Embed a query string (requires the sentence-transformers model)
    python scripts/retrieval_debug.py "risk factors for type 2 diabetes"

    # Use a pre-computed chunk embedding as the query vector
    # (useful when the embedding model cannot be loaded in this environment)
    python scripts/retrieval_debug.py --query-chunk-id 0_semantic_0000
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

from src.config import AppConfig
from src.retriever import ArtifactIndex, Retriever, RetrievalResult
from src.rag_pipeline import create_rag_pipeline


def _configure_logging(level: int = logging.INFO) -> None:
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=level,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            stream=sys.stderr,
        )


def _build_config(args: argparse.Namespace) -> AppConfig:
    """Build an AppConfig from CLI overrides."""
    base_config = AppConfig.default()
    return AppConfig(
        neo4j=base_config.neo4j,
        embedding=base_config.embedding,
        artifact=base_config.artifact,
        retrieval=base_config.retrieval.__class__(
            top_k=args.top_k,
            expand_depth=base_config.retrieval.expand_depth,
            max_entity_degree=args.max_entity_degree,
            max_expansion_per_entity=base_config.retrieval.max_expansion_per_entity,
            alpha=args.alpha,
            depth_scores=base_config.retrieval.depth_scores,
            max_results=args.max_results,
        ),
    )


def _print_results(query_text: str, results: list[RetrievalResult]) -> None:
    print(f"\nQuery: {query_text}")
    print("=" * 80)
    print(f"\nRanked context ({len(results)} results):\n")
    for rank, result in enumerate(results, start=1):
        print(
            f"{rank}. chunk_id={result.chunk_id}  "
            f"article_id={result.article_id}  "
            f"depth={result.depth}  source={result.source}\n"
            f"   vector_score={result.vector_score:.4f}  "
            f"graph_score={result.graph_score:.4f}  "
            f"combined_score={result.combined_score:.4f}\n"
            f"   text: {result.text[:280]}...\n"
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a manual retrieval test against the PubMed GraphRAG index.",
    )
    parser.add_argument(
        "query",
        nargs="?",
        default="",
        help="Query string (used when --query-chunk-id is not given)",
    )
    parser.add_argument(
        "--query-chunk-id",
        default="",
        help="Use the embedding of this chunk as the query vector",
    )
    parser.add_argument(
        "--query-vector-file",
        type=Path,
        default=None,
        help="Load query vector from a .npy file (shape (384,))",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of vector search results",
    )
    parser.add_argument(
        "--max-results",
        type=int,
        default=20,
        help="Maximum ranked results to return",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.8,
        help="Weight for vector score in combined ranking",
    )
    parser.add_argument(
        "--max-entity-degree",
        type=int,
        default=500,
        help="Skip entities with degree above this threshold",
    )
    parser.add_argument(
        "--generate",
        action="store_true",
        help="Also run the mock LLM generation step",
    )
    args = parser.parse_args()

    if not args.query and not args.query_chunk_id and args.query_vector_file is None:
        parser.error("Provide a query string, --query-chunk-id, or --query-vector-file")

    _configure_logging()
    config = _build_config(args)

    # When bypassing the LLM generation interface, build the retriever directly
    # so we can inject a pre-computed query vector.
    if args.query_chunk_id or args.query_vector_file is not None:
        index = ArtifactIndex.load(config)
        retriever = Retriever(index, config)

        if args.query_chunk_id:
            row = index.row_by_chunk_id.get(args.query_chunk_id)
            if row is None:
                print(f"Unknown chunk_id: {args.query_chunk_id}", file=sys.stderr)
                return 1
            query_vector = index.embeddings[row]
            query_text = index.chunk_by_id[args.query_chunk_id]["text"]
        else:
            query_vector = np.load(args.query_vector_file)
            if query_vector.shape != (config.embedding.embedding_dim,):
                print(
                    f"Query vector shape {query_vector.shape} != ({config.embedding.embedding_dim},)",
                    file=sys.stderr,
                )
                return 1
            query_text = args.query

        results = retriever.retrieve_by_vector(query_vector, query_text=query_text)
        _print_results(query_text, results)

        if args.generate:
            from src.rag_pipeline import RAGPipeline, MockLLMClient

            pipeline = RAGPipeline(retriever=retriever, llm=MockLLMClient())
            print("\n" + "=" * 80)
            print("Mock generation:\n")
            response = pipeline.generate(query_text, context=results)
            print(response.answer)

        return 0

    # Standard path: embed the query string and retrieve.
    pipeline = create_rag_pipeline(config)
    results = pipeline.retrieve(args.query)
    _print_results(args.query, results)

    if args.generate:
        print("\n" + "=" * 80)
        print("Mock generation:\n")
        response = pipeline.generate(args.query, context=results)
        print(response.answer)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
