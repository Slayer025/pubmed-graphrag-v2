"""Configuration for the PubMed GraphRAG retrieval pipeline.

Phase 3 uses offline artifacts only; Neo4j settings are kept optional for
future phases.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
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
    """Embedding model, provider, and artifact settings."""

    provider: str = "local"
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dim: int = 384
    batch_size: int = 64
    normalize: bool = True
    api_token: str | None = field(default=None, repr=False)
    service_url: str | None = None
    timeout_seconds: float = 30.0

    @classmethod
    def from_env(cls) -> "EmbeddingConfig":
        """Build embedding config from environment / Streamlit secrets.

        When running on Streamlit Cloud, ``st.secrets`` is the preferred source
        for sensitive values.  Environment variables are used as a fallback.
        """
        defaults = cls()
        try:
            import streamlit as st  # type: ignore

            secrets = st.secrets if hasattr(st, "secrets") else {}
        except Exception:
            secrets = {}

        def _get(name: str, env_name: str) -> str | None:
            """Read from Streamlit secrets first, then environment."""
            try:
                value = secrets.get(name)
            except Exception:
                value = None
            if value:
                return str(value).strip() or None
            value = os.environ.get(env_name)
            if value:
                return value.strip() or None
            return None

        provider = _get("EMBEDDING_PROVIDER", "EMBEDDING_PROVIDER") or defaults.provider
        normalize_raw = _get("EMBEDDING_NORMALIZE", "EMBEDDING_NORMALIZE")
        normalize = (
            normalize_raw.lower() in {"1", "true", "yes"}
            if normalize_raw
            else defaults.normalize
        )

        return cls(
            provider=provider,
            model_name=_get("EMBEDDING_MODEL", "EMBEDDING_MODEL") or defaults.model_name,
            embedding_dim=int(_get("EMBEDDING_DIM", "EMBEDDING_DIM") or defaults.embedding_dim),
            batch_size=int(
                _get("EMBEDDING_BATCH_SIZE", "EMBEDDING_BATCH_SIZE") or defaults.batch_size
            ),
            normalize=normalize,
            api_token=_get("HF_API_TOKEN", "HF_API_TOKEN"),
            service_url=_get("EMBEDDING_SERVICE_URL", "EMBEDDING_SERVICE_URL"),
            timeout_seconds=float(
                _get("EMBEDDING_TIMEOUT_SECONDS", "EMBEDDING_TIMEOUT_SECONDS")
                or defaults.timeout_seconds
            ),
        )


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

    # Phase 3: query understanding routing
    enable_query_routing: bool = False

    # Phase 4: metadata-aware boosting
    enable_metadata_boost: bool = False
    metadata_boost_factor: float = 1.1

    # Phase 5: multiple embedding indexes / chunking strategies
    default_index: str = "semantic"
    enable_multi_index: bool = False

    # Phase 6: HNSW approximate-nearest-neighbor search
    use_hnsw: bool = False

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
            embedding=EmbeddingConfig.from_env(),
            artifact=ArtifactConfig(),
            retrieval=RetrievalConfig(),
            rerank=RerankConfig(),
            decomposer=DecomposerConfig(),
        )
