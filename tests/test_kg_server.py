"""Integration tests for KG server.

Tests the ConceptNet and Freebase adapters (unit tests, no server needed)
and the FastAPI server endpoints (requires mocking or live server).

Usage:
    # Unit tests (no server needed):
    pytest tests/test_kg_server.py -v -k "not live"

    # Full integration tests (requires KG server running):
    KG_SERVER_URL=http://localhost:8001 pytest tests/test_kg_server.py -v
"""

import os
from unittest.mock import MagicMock, patch

import pytest

from src_verl.kg_server.conceptnet_adapter import ConceptNetAdapter
from src_verl.kg_server.shortest_paths import bfs_distances


# =========================================================================
# ConceptNet adapter tests (mock graph)
# =========================================================================

class TestConceptNetAdapter:
    @pytest.fixture
    def adapter(self) -> ConceptNetAdapter:
        """Create adapter with a small mock graph."""
        import networkx as nx

        graph = nx.MultiDiGraph()
        graph.add_edge("dog", "animal", relation="IsA", weight=3.0)
        graph.add_edge("cat", "animal", relation="IsA", weight=3.0)
        graph.add_edge("animal", "alive", relation="HasProperty", weight=2.5)
        graph.add_edge("dog", "loyal", relation="HasProperty", weight=2.0)
        graph.add_edge("dog", "park", relation="AtLocation", weight=2.0)

        return ConceptNetAdapter(graph)

    def test_get_tail_relations(self, adapter: ConceptNetAdapter) -> None:
        rels = adapter.get_tail_relations("dog")
        assert "IsA" in rels
        assert "HasProperty" in rels
        assert "AtLocation" in rels

    def test_get_tail_relations_missing(self, adapter: ConceptNetAdapter) -> None:
        rels = adapter.get_tail_relations("nonexistent_entity")
        assert rels == []

    def test_get_head_relations(self, adapter: ConceptNetAdapter) -> None:
        rels = adapter.get_head_relations("animal")
        assert "IsA" in rels

    def test_get_tail_entities(self, adapter: ConceptNetAdapter) -> None:
        entities = adapter.get_tail_entities("dog", "IsA")
        assert "animal" in entities

    def test_get_tail_entities_no_match(self, adapter: ConceptNetAdapter) -> None:
        entities = adapter.get_tail_entities("dog", "MadeOf")
        assert entities == []

    def test_get_head_entities(self, adapter: ConceptNetAdapter) -> None:
        entities = adapter.get_head_entities("animal", "IsA")
        assert "dog" in entities
        assert "cat" in entities

    def test_has_entity(self, adapter: ConceptNetAdapter) -> None:
        assert adapter.has_entity("dog")
        assert adapter.has_entity("animal")
        assert not adapter.has_entity("spaceship")

    def test_case_insensitive(self, adapter: ConceptNetAdapter) -> None:
        """Entities should match case-insensitively."""
        rels = adapter.get_tail_relations("Dog")
        assert "IsA" in rels

    def test_num_nodes(self, adapter: ConceptNetAdapter) -> None:
        assert adapter.num_nodes == 6  # dog, cat, animal, alive, loyal, park

    def test_num_edges(self, adapter: ConceptNetAdapter) -> None:
        assert adapter.num_edges == 5


# =========================================================================
# BFS shortest paths tests
# =========================================================================

class TestBFSDistances:
    def test_simple_graph(self) -> None:
        adj = {
            "a": ["b", "c"],
            "b": ["a", "d"],
            "c": ["a"],
            "d": ["b"],
        }
        distances = bfs_distances(adj, "a", max_depth=5)
        assert distances["a"] == 0
        assert distances["b"] == 1
        assert distances["c"] == 1
        assert distances["d"] == 2

    def test_max_depth_limit(self) -> None:
        adj = {
            "a": ["b"],
            "b": ["a", "c"],
            "c": ["b", "d"],
            "d": ["c"],
        }
        distances = bfs_distances(adj, "a", max_depth=2)
        assert "a" in distances
        assert "b" in distances
        assert "c" in distances
        assert "d" not in distances  # beyond max_depth

    def test_disconnected(self) -> None:
        adj = {"a": ["b"], "b": ["a"], "c": ["d"], "d": ["c"]}
        distances = bfs_distances(adj, "a", max_depth=5)
        assert "c" not in distances
        assert "d" not in distances


# =========================================================================
# FastAPI server tests (mock adapter)
# =========================================================================

class TestServerEndpoints:
    @pytest.fixture
    def client(self) -> "TestClient":
        """Create test client with mock adapter."""
        from fastapi.testclient import TestClient

        from src_verl.kg_server import server as server_module

        # Create mock adapter
        mock_adapter = MagicMock()
        mock_adapter.get_tail_relations.return_value = ["IsA", "HasProperty"]
        mock_adapter.get_tail_entities.return_value = ["animal"]
        mock_adapter.get_head_entities.return_value = ["dog", "cat"]
        mock_adapter.get_head_relations.return_value = ["IsA"]
        mock_adapter.num_nodes = 100

        server_module._adapter = mock_adapter
        return TestClient(server_module.app)

    def test_health(self, client: "TestClient") -> None:
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"

    def test_retrieve_tail_relations(self, client: "TestClient") -> None:
        resp = client.post("/retrieve", json={
            "action": "get_tail_relations",
            "entity": "dog",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["found"] is True
        assert "IsA" in data["results"]

    def test_retrieve_tail_entities(self, client: "TestClient") -> None:
        resp = client.post("/retrieve", json={
            "action": "get_tail_entities",
            "entity": "dog",
            "relation": "IsA",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "animal" in data["results"]

    def test_retrieve_unknown_action(self, client: "TestClient") -> None:
        resp = client.post("/retrieve", json={
            "action": "unknown_action",
            "entity": "dog",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["results"] == []

    def test_batch_retrieve(self, client: "TestClient") -> None:
        resp = client.post("/batch_retrieve", json={
            "queries": [
                {"action": "get_tail_relations", "entity": "dog"},
                {"action": "get_tail_entities", "entity": "dog", "relation": "IsA"},
            ]
        })
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["responses"]) == 2


# =========================================================================
# Live server tests (skip if no server)
# =========================================================================

@pytest.mark.skipif(
    "KG_SERVER_URL" not in os.environ,
    reason="KG_SERVER_URL not set — skip live server tests",
)
class TestLiveServer:
    @pytest.fixture
    def server_url(self) -> str:
        return os.environ["KG_SERVER_URL"]

    def test_live_health(self, server_url: str) -> None:
        import requests

        resp = requests.get(f"{server_url}/health", timeout=5)
        assert resp.status_code == 200

    def test_live_retrieve(self, server_url: str) -> None:
        import requests

        resp = requests.post(
            f"{server_url}/retrieve",
            json={"action": "get_tail_relations", "entity": "dog"},
            timeout=5,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data["results"], list)
