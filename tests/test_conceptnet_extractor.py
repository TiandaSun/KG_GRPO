"""Unit tests for ConceptNet path extraction (Stage 1)."""

from __future__ import annotations

import gzip
import json
import random
import tempfile
from collections import Counter
from pathlib import Path

import pytest

from src.kg.conceptnet_extractor import (
    ExtractionConfig,
    KGPath,
    Triple,
    _clean_entity,
    _relation_diversity_ok,
    build_graph,
    extract_1hop_paths,
    extract_multihop_paths,
    extract_paths,
    load_triples,
    parse_conceptnet_line,
    save_paths,
)


# --- Fixtures ---

def _make_assertion_line(
    subj: str = "/c/en/dog",
    rel: str = "/r/IsA",
    obj: str = "/c/en/animal",
    weight: float = 3.0,
) -> str:
    """Create a ConceptNet-format assertion line."""
    uri = f"/a/[{rel}/,{subj}/,{obj}/]"
    metadata = json.dumps({"weight": weight})
    return f"{uri}\t{rel}\t{subj}\t{obj}\t{metadata}"


SAMPLE_LINES = [
    _make_assertion_line("/c/en/dog", "/r/IsA", "/c/en/animal", 3.0),
    _make_assertion_line("/c/en/animal", "/r/HasProperty", "/c/en/alive", 2.5),
    _make_assertion_line("/c/en/cat", "/r/IsA", "/c/en/animal", 4.0),
    _make_assertion_line("/c/en/car", "/r/UsedFor", "/c/en/transportation", 2.0),
    _make_assertion_line("/c/en/bird", "/r/CapableOf", "/c/en/fly", 2.1),
    _make_assertion_line("/c/en/fire", "/r/Causes", "/c/en/burn", 3.5),
    _make_assertion_line("/c/en/tree", "/r/HasA", "/c/en/leaf", 2.8),
    _make_assertion_line("/c/en/wheel", "/r/PartOf", "/c/en/car", 3.0),
    _make_assertion_line("/c/en/book", "/r/AtLocation", "/c/en/library", 2.2),
    _make_assertion_line("/c/en/dog", "/r/HasProperty", "/c/en/loyal", 2.3),
    _make_assertion_line("/c/en/alive", "/r/HasProperty", "/c/en/breathing", 2.0),
    _make_assertion_line("/c/en/fly", "/r/HasPrerequisite", "/c/en/wing", 2.5),
    _make_assertion_line("/c/en/transportation", "/r/UsedFor", "/c/en/travel", 2.0),
    _make_assertion_line("/c/en/burn", "/r/Causes", "/c/en/pain", 2.2),
    _make_assertion_line("/c/en/leaf", "/r/HasProperty", "/c/en/green", 2.4),
]


@pytest.fixture
def sample_assertions_file(tmp_path: Path) -> Path:
    """Create a temporary ConceptNet assertions file."""
    fpath = tmp_path / "assertions.csv"
    fpath.write_text("\n".join(SAMPLE_LINES) + "\n", encoding="utf-8")
    return fpath


@pytest.fixture
def sample_assertions_gz(tmp_path: Path) -> Path:
    """Create a gzipped temporary ConceptNet assertions file."""
    fpath = tmp_path / "assertions.csv.gz"
    with gzip.open(fpath, "wt", encoding="utf-8") as f:
        f.write("\n".join(SAMPLE_LINES) + "\n")
    return fpath


@pytest.fixture
def sample_triples() -> list[Triple]:
    """Pre-parsed triples matching SAMPLE_LINES."""
    return [
        Triple("dog", "IsA", "animal", 3.0),
        Triple("animal", "HasProperty", "alive", 2.5),
        Triple("cat", "IsA", "animal", 4.0),
        Triple("car", "UsedFor", "transportation", 2.0),
        Triple("bird", "CapableOf", "fly", 2.1),
        Triple("fire", "Causes", "burn", 3.5),
        Triple("tree", "HasA", "leaf", 2.8),
        Triple("wheel", "PartOf", "car", 3.0),
        Triple("book", "AtLocation", "library", 2.2),
        Triple("dog", "HasProperty", "loyal", 2.3),
        Triple("alive", "HasProperty", "breathing", 2.0),
        Triple("fly", "HasPrerequisite", "wing", 2.5),
        Triple("transportation", "UsedFor", "travel", 2.0),
        Triple("burn", "Causes", "pain", 2.2),
        Triple("leaf", "HasProperty", "green", 2.4),
    ]


# --- Test parse_conceptnet_line ---

class TestParseConceptNetLine:
    def test_valid_english_line(self) -> None:
        line = _make_assertion_line("/c/en/dog", "/r/IsA", "/c/en/animal", 3.0)
        triple = parse_conceptnet_line(line)
        assert triple is not None
        assert triple.subject == "dog"
        assert triple.relation == "IsA"
        assert triple.obj == "animal"
        assert triple.weight == 3.0

    def test_non_english_filtered(self) -> None:
        line = _make_assertion_line("/c/fr/chien", "/r/IsA", "/c/en/animal", 3.0)
        assert parse_conceptnet_line(line) is None

    def test_excluded_relation_filtered(self) -> None:
        line = _make_assertion_line("/c/en/dog", "/r/SimilarTo", "/c/en/wolf", 5.0)
        assert parse_conceptnet_line(line) is None

        line = _make_assertion_line("/c/en/dog", "/r/RelatedTo", "/c/en/pet", 5.0)
        assert parse_conceptnet_line(line) is None

    def test_self_loop_filtered(self) -> None:
        line = _make_assertion_line("/c/en/dog", "/r/IsA", "/c/en/dog", 3.0)
        assert parse_conceptnet_line(line) is None

    def test_short_entity_filtered(self) -> None:
        line = _make_assertion_line("/c/en/a", "/r/IsA", "/c/en/animal", 3.0)
        assert parse_conceptnet_line(line) is None

    def test_underscore_cleaned(self) -> None:
        line = _make_assertion_line("/c/en/hot_dog", "/r/IsA", "/c/en/food", 3.0)
        triple = parse_conceptnet_line(line)
        assert triple is not None
        assert triple.subject == "hot dog"

    def test_malformed_line(self) -> None:
        assert parse_conceptnet_line("incomplete\tline") is None

    def test_bad_json_metadata(self) -> None:
        line = "/a/x\t/r/IsA\t/c/en/dog\t/c/en/animal\t{bad json"
        assert parse_conceptnet_line(line) is None


# --- Test _clean_entity ---

class TestCleanEntity:
    def test_basic(self) -> None:
        assert _clean_entity("/c/en/dog") == "dog"

    def test_with_pos_tag(self) -> None:
        assert _clean_entity("/c/en/dog/n") == "dog"

    def test_with_wordnet_suffix(self) -> None:
        assert _clean_entity("/c/en/dog/n/wn/animal") == "dog"

    def test_underscore(self) -> None:
        assert _clean_entity("/c/en/hot_dog") == "hot dog"


# --- Test load_triples ---

class TestLoadTriples:
    def test_loads_plain_csv(self, sample_assertions_file: Path) -> None:
        triples = load_triples(sample_assertions_file, min_weight=2.0)
        assert len(triples) == 15

    def test_loads_gzipped_csv(self, sample_assertions_gz: Path) -> None:
        triples = load_triples(sample_assertions_gz, min_weight=2.0)
        assert len(triples) == 15

    def test_weight_filter(self, sample_assertions_file: Path) -> None:
        triples = load_triples(sample_assertions_file, min_weight=3.0)
        assert all(t.weight >= 3.0 for t in triples)
        assert len(triples) < 15

    def test_max_triples_limit(self, sample_assertions_file: Path) -> None:
        triples = load_triples(sample_assertions_file, min_weight=2.0, max_triples=5)
        assert len(triples) == 5


# --- Test build_graph ---

class TestBuildGraph:
    def test_basic_graph(self, sample_triples: list[Triple]) -> None:
        graph = build_graph(sample_triples)
        assert graph.number_of_nodes() > 0
        assert graph.number_of_edges() == len(sample_triples)

    def test_edge_attributes(self, sample_triples: list[Triple]) -> None:
        graph = build_graph(sample_triples)
        # dog -> animal edge should exist with IsA relation
        edges = list(graph.edges("dog", data=True))
        relations = [d["relation"] for _, _, d in edges]
        assert "IsA" in relations


# --- Test relation diversity ---

class TestRelationDiversity:
    def test_allows_when_below_threshold(self) -> None:
        counter = Counter({"IsA": 5, "HasA": 5})
        assert _relation_diversity_ok("IsA", counter, 50, 0.20)

    def test_blocks_when_above_threshold(self) -> None:
        counter = Counter({"IsA": 10})
        assert not _relation_diversity_ok("IsA", counter, 50, 0.20)

    def test_allows_when_few_paths(self) -> None:
        counter = Counter({"IsA": 5})
        # With < 10 total paths, diversity is not enforced
        assert _relation_diversity_ok("IsA", counter, 5, 0.20)


# --- Test path extraction ---

class TestExtract1HopPaths:
    def test_extracts_paths(self, sample_triples: list[Triple]) -> None:
        graph = build_graph(sample_triples)
        rng = random.Random(42)
        counter: Counter[str] = Counter()
        seen: set[frozenset[str]] = set()

        paths = extract_1hop_paths(graph, rng, 5, counter, 0.5, seen)
        assert len(paths) > 0
        assert all(p.hops == 1 for p in paths)
        assert all(len(p.path) == 1 for p in paths)

    def test_no_duplicate_entity_sets(self, sample_triples: list[Triple]) -> None:
        graph = build_graph(sample_triples)
        rng = random.Random(42)
        counter: Counter[str] = Counter()
        seen: set[frozenset[str]] = set()

        paths = extract_1hop_paths(graph, rng, 20, counter, 0.5, seen)
        entity_sets = [p.entity_set_key for p in paths]
        assert len(entity_sets) == len(set(entity_sets))


class TestExtractMultihopPaths:
    def test_extracts_2hop_paths(self, sample_triples: list[Triple]) -> None:
        graph = build_graph(sample_triples)
        rng = random.Random(42)
        counter: Counter[str] = Counter()
        seen: set[frozenset[str]] = set()

        paths = extract_multihop_paths(
            graph, rng, 3, num_hops=2,
            relation_counter=counter, max_relation_share=0.5,
            seen_entity_sets=seen,
        )
        assert len(paths) > 0
        assert all(p.hops == 2 for p in paths)
        assert all(len(p.path) == 2 for p in paths)

    def test_path_entities_are_connected(self, sample_triples: list[Triple]) -> None:
        graph = build_graph(sample_triples)
        rng = random.Random(42)
        counter: Counter[str] = Counter()
        seen: set[frozenset[str]] = set()

        paths = extract_multihop_paths(
            graph, rng, 5, num_hops=2,
            relation_counter=counter, max_relation_share=0.5,
            seen_entity_sets=seen,
        )
        for path in paths:
            # Each hop's object should be the next hop's subject
            for i in range(len(path.path) - 1):
                assert path.path[i][2] == path.path[i + 1][0]

    def test_no_cycles_in_paths(self, sample_triples: list[Triple]) -> None:
        graph = build_graph(sample_triples)
        rng = random.Random(42)
        counter: Counter[str] = Counter()
        seen: set[frozenset[str]] = set()

        paths = extract_multihop_paths(
            graph, rng, 5, num_hops=2,
            relation_counter=counter, max_relation_share=0.5,
            seen_entity_sets=seen,
        )
        for path in paths:
            # All entities should be unique (no revisits)
            assert len(path.entities) == len(set(path.entities))


# --- Test KGPath ---

class TestKGPath:
    def test_to_dict(self) -> None:
        path = KGPath(
            path=[["dog", "IsA", "animal"], ["animal", "HasProperty", "alive"]],
            hops=2,
            relations=["IsA", "HasProperty"],
            entities=["dog", "animal", "alive"],
        )
        d = path.to_dict()
        assert d["hops"] == 2
        assert len(d["path"]) == 2
        assert d["relations"] == ["IsA", "HasProperty"]
        assert d["entities"] == ["dog", "animal", "alive"]

    def test_entity_set_key(self) -> None:
        path = KGPath(
            path=[["dog", "IsA", "animal"]],
            hops=1,
            relations=["IsA"],
            entities=["dog", "animal"],
        )
        assert path.entity_set_key == frozenset(["dog", "animal"])


# --- Test save_paths ---

class TestSavePaths:
    def test_save_and_load(self, tmp_path: Path) -> None:
        paths = [
            KGPath(
                path=[["dog", "IsA", "animal"]],
                hops=1,
                relations=["IsA"],
                entities=["dog", "animal"],
            ),
            KGPath(
                path=[["dog", "IsA", "animal"], ["animal", "HasProperty", "alive"]],
                hops=2,
                relations=["IsA", "HasProperty"],
                entities=["dog", "animal", "alive"],
            ),
        ]
        output = tmp_path / "test_paths.jsonl"
        save_paths(paths, output)

        assert output.exists()
        lines = output.read_text().strip().split("\n")
        assert len(lines) == 2

        loaded = [json.loads(line) for line in lines]
        assert loaded[0]["hops"] == 1
        assert loaded[1]["hops"] == 2


# --- Test full extraction pipeline ---

class TestExtractPaths:
    def test_end_to_end(self, sample_assertions_file: Path, tmp_path: Path) -> None:
        config = ExtractionConfig(
            assertions_path=sample_assertions_file,
            output_path=tmp_path / "output.jsonl",
            min_weight=2.0,
            num_paths=10,
            max_relation_share=0.5,  # Relaxed for small test set
            seed=42,
        )
        paths = extract_paths(config)
        assert len(paths) > 0

        # Check all paths have valid structure
        for path in paths:
            assert path.hops >= 1
            assert len(path.path) == path.hops
            assert len(path.relations) == path.hops
            assert len(path.entities) == path.hops + 1

    def test_reproducibility(self, sample_assertions_file: Path, tmp_path: Path) -> None:
        config = ExtractionConfig(
            assertions_path=sample_assertions_file,
            output_path=tmp_path / "output.jsonl",
            min_weight=2.0,
            num_paths=10,
            max_relation_share=0.5,
            seed=42,
        )
        paths1 = extract_paths(config)
        paths2 = extract_paths(config)

        assert len(paths1) == len(paths2)
        for p1, p2 in zip(paths1, paths2):
            assert p1.to_dict() == p2.to_dict()
