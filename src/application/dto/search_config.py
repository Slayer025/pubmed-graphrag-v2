"""SearchConfig DTO and mapping helpers."""

from __future__ import annotations

from dataclasses import dataclass

from src.domain.value_objects.retrieval_hyperparameters import RetrievalHyperparameters


@dataclass(frozen=True)
class SearchConfig:
    """Request-scoped retrieval hyperparameters.

    This DTO mirrors ``src.config.RetrievalConfig`` but lives in the
    application layer so it can be passed into use cases without violating
    Clean Architecture dependency rules.
    """

    top_k: int = 10
    expand_depth: int = 2
    max_entity_degree: int = 500
    max_expansion_per_entity: int = 100
    max_expanded_nodes: int = 2_000
    alpha: float = 0.8
    depth_scores: tuple[float, float, float] = (1.0, 0.5, 0.25)
    use_hybrid: bool = False
    rrf_k: int = 20
    max_results: int = 20
    enable_query_routing: bool = False
    enable_metadata_boost: bool = False
    metadata_boost_factor: float = 1.1

    # Phase 5: multiple embedding indexes
    default_index: str = "semantic"
    enable_multi_index: bool = False
    index_name: str | None = None

    # Phase 6: HNSW approximate-nearest-neighbor search
    use_hnsw: bool = False

    @classmethod
    def from_retrieval_config(cls, config) -> "SearchConfig":
        """Build a ``SearchConfig`` from ``src.config.RetrievalConfig``."""
        default_index = getattr(config, "default_index", "semantic")
        return cls(
            top_k=config.top_k,
            expand_depth=config.expand_depth,
            max_entity_degree=config.max_entity_degree,
            max_expansion_per_entity=config.max_expansion_per_entity,
            max_expanded_nodes=config.max_expanded_nodes,
            alpha=config.alpha,
            depth_scores=config.depth_scores,
            max_results=config.max_results,
            use_hybrid=getattr(config, "use_hybrid", False),
            rrf_k=getattr(config, "rrf_k", 60),
            enable_query_routing=getattr(config, "enable_query_routing", False),
            enable_metadata_boost=getattr(config, "enable_metadata_boost", False),
            metadata_boost_factor=getattr(config, "metadata_boost_factor", 1.1),
            default_index=default_index,
            enable_multi_index=getattr(config, "enable_multi_index", False),
            index_name=default_index,
            use_hnsw=getattr(config, "use_hnsw", False),
        )

    def to_hyperparameters(self) -> RetrievalHyperparameters:
        """Map this application DTO to a domain value object."""
        return RetrievalHyperparameters(
            expand_depth=self.expand_depth,
            max_entity_degree=self.max_entity_degree,
            max_expansion_per_entity=self.max_expansion_per_entity,
            max_expanded_nodes=self.max_expanded_nodes,
            depth_scores=self.depth_scores,
            alpha=self.alpha,
            max_results=self.max_results,
        )
