"""ConceptNet adapter for KG retrieval server.

Wraps the existing ConceptNet NetworkX graph with the 4 query methods
required by the KG server interface:
  - get_tail_relations(entity) -> list of relations from entity
  - get_head_relations(entity) -> list of relations to entity
  - get_tail_entities(entity, relation) -> list of tail entities
  - get_head_entities(entity, relation) -> list of head entities

Reuses load_triples() and build_graph() from src.kg.conceptnet_extractor.

Supports pickle caching: on first load, saves the graph to a .pkl file
next to the assertions file. Subsequent loads use the cache (<5s vs ~80s).
"""

from __future__ import annotations

import logging
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Protocol

import networkx as nx

from src.kg.conceptnet_extractor import build_graph, load_triples

logger = logging.getLogger(__name__)


class KGAdapter(Protocol):
    """Protocol for KG adapters — all adapters must implement these methods."""

    def get_tail_relations(self, entity: str) -> list[str]: ...
    def get_head_relations(self, entity: str) -> list[str]: ...
    def get_tail_entities(self, entity: str, relation: str) -> list[str]: ...
    def get_head_entities(self, entity: str, relation: str) -> list[str]: ...
    def has_entity(self, entity: str) -> bool: ...
    def get_all_entities(self) -> list[str]: ...


class ConceptNetAdapter:
    """Wraps ConceptNet MultiDiGraph for KG server queries."""

    def __init__(self, graph: nx.MultiDiGraph) -> None:
        self._graph = graph

        # Pre-compute lookup indices for fast querying
        self._outgoing: dict[str, dict[str, list[str]]] = defaultdict(
            lambda: defaultdict(list)
        )
        self._incoming: dict[str, dict[str, list[str]]] = defaultdict(
            lambda: defaultdict(list)
        )

        for u, v, data in graph.edges(data=True):
            rel = data["relation"]
            self._outgoing[u][rel].append(v)
            self._incoming[v][rel].append(u)

        logger.info(
            "ConceptNetAdapter initialized: %d nodes, %d edges",
            graph.number_of_nodes(),
            graph.number_of_edges(),
        )

    @classmethod
    def from_assertions(
        cls,
        assertions_path: Path,
        min_weight: float = 2.0,
        max_triples: int | None = None,
    ) -> ConceptNetAdapter:
        """Load ConceptNet from assertions file and build adapter.

        Uses pickle cache when available. Cache file is stored next to the
        assertions file as ``conceptnet_graph_w{min_weight}.pkl``.
        """
        cache_path = assertions_path.parent / f"conceptnet_graph_w{min_weight}.pkl"

        if cache_path.exists():
            logger.info("Loading cached graph from %s ...", cache_path)
            with open(cache_path, "rb") as f:
                graph = pickle.load(f)
            logger.info(
                "Loaded cached graph: %d nodes, %d edges",
                graph.number_of_nodes(),
                graph.number_of_edges(),
            )
            return cls(graph)

        logger.info("No cache found at %s, loading from CSV ...", cache_path)
        triples = load_triples(assertions_path, min_weight, max_triples)
        graph = build_graph(triples)

        # Save cache for next time
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_path, "wb") as f:
                pickle.dump(graph, f, protocol=pickle.HIGHEST_PROTOCOL)
            logger.info("Saved graph cache to %s", cache_path)
        except OSError as e:
            logger.warning("Failed to save graph cache: %s", e)

        return cls(graph)

    def get_tail_relations(self, entity: str) -> list[str]:
        """Get all relation types going out from entity."""
        entity = entity.lower().replace("_", " ")
        if entity in self._outgoing:
            return sorted(set(self._outgoing[entity].keys()))
        return []

    def get_head_relations(self, entity: str) -> list[str]:
        """Get all relation types coming into entity."""
        entity = entity.lower().replace("_", " ")
        if entity in self._incoming:
            return sorted(set(self._incoming[entity].keys()))
        return []

    def get_tail_entities(self, entity: str, relation: str) -> list[str]:
        """Get entities reachable from entity via relation."""
        entity = entity.lower().replace("_", " ")
        if entity in self._outgoing and relation in self._outgoing[entity]:
            return sorted(set(self._outgoing[entity][relation]))
        return []

    def get_head_entities(self, entity: str, relation: str) -> list[str]:
        """Get entities that connect to entity via relation."""
        entity = entity.lower().replace("_", " ")
        if entity in self._incoming and relation in self._incoming[entity]:
            return sorted(set(self._incoming[entity][relation]))
        return []

    def has_entity(self, entity: str) -> bool:
        """Check if entity exists in the graph."""
        entity = entity.lower().replace("_", " ")
        return entity in self._outgoing or entity in self._incoming

    def get_all_entities(self) -> list[str]:
        """Get all entity names in the graph."""
        return list(self._graph.nodes())

    @property
    def num_nodes(self) -> int:
        return self._graph.number_of_nodes()

    @property
    def num_edges(self) -> int:
        return self._graph.number_of_edges()
