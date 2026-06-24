"""Configuration for the PubMed GraphRAG retrieval pipeline.

Phase 3 uses offline artifacts only; Neo4j settings are kept optional for
future phases.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from src.application.dto.rerank_config import RerankConfig


@dataclass(frozen=True)
class Neo4jConfig:
    """Optional Neo4j connection parameters for future database-backed phases."""

    uri: str = "bolt://localhost:7687"
    user: str = "neo4j"
    password: str = "password"
    database: str = "neo4j"
    enabled: bool = False

    @classmethod
    def from_env(cls) -> "Neo4jConfig":
        return cls(
            uri=os.environ.get("NEO4J_URI", "bolt://localhost:7687"),
            user=os.environ.get("NEO4J_USER", "neo4j"),
            password=os.environ.get("NEO4J_PASSWORD", "password"),
            database=os.environ.get("NEO4J_DATABASE", "neo4j"),
            enabled=os.environ.get("NEO4J_ENABLED", "false").lower() in {"1", "true", "yes"},
        )


@dataclass(frozen=True)
class EmbeddingConfig:
    """Embedding model and artifact settings."""

    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dim: int = 384
    batch_size: int = 64
    normalize: bool = True


@dataclass(frozen=True)
class ArtifactConfig:
    """Paths to Phase 1/2 artifacts used by the retrieval pipeline."""

    chunks_path: Path = Path("data/chunks/chunks_semantic.jsonl.gz")
    embeddings_path: Path = Path("data/embeddings/semantic_embeddings.npy")
    mentions_path: Path = Path("data/graph/mentions.csv")
    has_chunk_path: Path = Path("data/graph/has_chunk.csv")
    entities_path: Path = Path("data/graph/entities.csv")


@dataclass(frozen=True)
class RetrievalConfig:
    """Retrieval hyperparameters."""

    # Vector search
    top_k: int = 10

    # Graph expansion
    expand_depth: int = 2
    max_entity_degree: int = 500
    max_expansion_per_entity: int = 100
    max_expanded_nodes: int = 2_000

    # Re-ranking: combined_score = alpha * vector_score + (1 - alpha) * graph_score
    alpha: float = 0.8

    # Graph score by traversal depth
    depth_scores: tuple[float, float, float] = (1.0, 0.5, 0.25)

    # Hybrid retrieval settings
    use_hybrid: bool = False
    rrf_k: int = 20

    # Final result cap
    max_results: int = 20


from src.application.dto.rerank_config import RerankConfig  # noqa: F401  # compatibility re-export


@dataclass(frozen=True)
class DecomposerConfig:
    """Optional query decomposition configuration."""

    enabled: bool = False
    max_sub_queries: int = 4


@dataclass(frozen=True)
class AppConfig:
    """Top-level application configuration."""

    neo4j: Neo4jConfig
    embedding: EmbeddingConfig
    artifact: ArtifactConfig
    retrieval: RetrievalConfig
    rerank: RerankConfig = RerankConfig()
    decomposer: DecomposerConfig = DecomposerConfig()

    @classmethod
    def default(cls) -> "AppConfig":
        return cls(
            neo4j=Neo4jConfig.from_env(),
            embedding=EmbeddingConfig(),
            artifact=ArtifactConfig(),
            retrieval=RetrievalConfig(),
            rerank=RerankConfig(),
            decomposer=DecomposerConfig(),
        )
