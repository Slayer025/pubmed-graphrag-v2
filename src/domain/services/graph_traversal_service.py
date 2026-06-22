"""Pure-domain graph traversal service.

This module contains only graph-expansion logic. It has no knowledge of files,
numpy, embedding models, or web frameworks.
"""

from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING, Protocol

from src.domain.value_objects.retrieval_hyperparameters import RetrievalHyperparameters

if TYPE_CHECKING:
    from collections.abc import Set as AbstractSet


class GraphRepository(Protocol):
    """Port for graph adjacency lookups."""

    def get_chunk_article(self, chunk_id: str) -> str:
        ...

    def get_article_chunks(self, article_id: str) -> set[str]:
        ...

    def get_chunk_entities(self, chunk_id: str) -> set[str]:
        ...

    def get_entity_chunks(self, entity_id: str) -> set[str]:
        ...

    def get_entity_degree(self, entity_id: str) -> int:
        ...


class GraphTraversalService:
    """Bounded BFS expansion over article and entity edges."""

    def expand(
        self,
        seed_chunk_ids: AbstractSet[str],
        graph: GraphRepository,
        hyperparameters: RetrievalHyperparameters,
    ) -> dict[str, tuple[int, float, str]]:
        """Expand from seed chunks via same-article and shared-entity edges.

        Returns a mapping chunk_id -> (depth, graph_score, source).
        """
        depth_scores = hyperparameters.depth_scores
        max_depth = min(hyperparameters.expand_depth, len(depth_scores) - 1)

        expanded: dict[str, tuple[int, float, str]] = {}
        visited: set[str] = set(seed_chunk_ids)
        frontier: deque[tuple[str, int, str]] = deque(
            (chunk_id, 0, "vector") for chunk_id in seed_chunk_ids
        )

        while frontier and len(expanded) < hyperparameters.max_expanded_nodes:
            chunk_id, depth, source = frontier.popleft()

            graph_score = depth_scores[min(depth, len(depth_scores) - 1)]
            if chunk_id not in expanded or graph_score > expanded[chunk_id][1]:
                expanded[chunk_id] = (depth, graph_score, source)

            if depth >= max_depth:
                continue

            next_depth = depth + 1

            # Same-article expansion.
            article_id = graph.get_chunk_article(chunk_id)
            if article_id:
                for related_chunk_id in graph.get_article_chunks(article_id):
                    if related_chunk_id in visited:
                        continue
                    visited.add(related_chunk_id)
                    frontier.append((related_chunk_id, next_depth, "same_article"))

            # Shared-entity expansion.
            for entity_id in graph.get_chunk_entities(chunk_id):
                degree = graph.get_entity_degree(entity_id)
                if degree > hyperparameters.max_entity_degree:
                    continue

                related = graph.get_entity_chunks(entity_id)
                if len(related) > hyperparameters.max_expansion_per_entity:
                    related = sorted(related)[: hyperparameters.max_expansion_per_entity]

                for related_chunk_id in related:
                    if related_chunk_id in visited:
                        continue
                    visited.add(related_chunk_id)
                    frontier.append((related_chunk_id, next_depth, "shared_entity"))

        return expanded
