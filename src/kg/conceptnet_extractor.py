"""Stage 1: ConceptNet path extraction for KG-Align-RL.

Downloads ConceptNet assertions, filters to high-quality English triples,
builds a graph, and extracts stratified multi-hop paths with relation-type
diversity constraints.

Usage:
    python src/kg/conceptnet_extractor.py \
        --assertions_path data/raw/conceptnet-assertions-5.7.0.csv.gz \
        --output_path data/processed/conceptnet_paths.jsonl \
        --num_paths 8000 \
        --seed 42
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import networkx as nx

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ConceptNet download URL
CONCEPTNET_URL = (
    "https://s3.amazonaws.com/conceptnet/downloads/2019/edges/"
    "conceptnet-assertions-5.7.0.csv.gz"
)

# Relations to exclude (too vague to form meaningful reasoning paths)
EXCLUDED_RELATIONS = frozenset({"SimilarTo", "RelatedTo", "Synonym", "EtymologicallyRelatedTo"})

# Major relation types to ensure diversity across
MAJOR_RELATIONS = frozenset({
    "IsA", "HasProperty", "UsedFor", "CapableOf", "Causes",
    "HasA", "PartOf", "AtLocation", "HasPrerequisite", "MotivatedByGoal",
    "HasSubevent", "HasFirstSubevent", "HasLastSubevent", "DefinedAs",
    "ReceivesAction", "CreatedBy", "MadeOf", "CausesDesire",
})


@dataclass
class ExtractionConfig:
    """Configuration for ConceptNet path extraction."""

    assertions_path: Path
    output_path: Path
    min_weight: float = 2.0
    num_paths: int = 8000
    max_relation_share: float = 0.20
    hop_distribution: dict[int, float] = field(
        default_factory=lambda: {1: 0.40, 2: 0.35, 3: 0.25}
    )
    seed: int = 42
    max_triples_to_load: int | None = None  # None = load all


@dataclass
class Triple:
    """A single ConceptNet triple."""

    subject: str
    relation: str
    obj: str
    weight: float


@dataclass
class KGPath:
    """A multi-hop path through the knowledge graph."""

    path: list[list[str]]  # [[subj, rel, obj], ...]
    hops: int
    relations: list[str]
    entities: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "hops": self.hops,
            "relations": self.relations,
            "entities": self.entities,
        }

    @property
    def entity_set_key(self) -> frozenset[str]:
        """Key for deduplication by entity set."""
        return frozenset(self.entities)


def download_conceptnet(output_path: Path) -> Path:
    """Download ConceptNet assertions if not already present."""
    import urllib.request

    if output_path.exists():
        logger.info("ConceptNet assertions already exist at %s", output_path)
        return output_path

    output_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading ConceptNet assertions from %s ...", CONCEPTNET_URL)
    urllib.request.urlretrieve(CONCEPTNET_URL, output_path)
    logger.info("Download complete: %s", output_path)
    return output_path


def parse_conceptnet_line(line: str) -> Triple | None:
    """Parse a single line from ConceptNet assertions CSV.

    Format: URI<tab>Relation<tab>Subject<tab>Object<tab>JSON_metadata
    Returns None if the line doesn't pass basic validity checks.
    """
    parts = line.strip().split("\t")
    if len(parts) < 5:
        return None

    relation_uri = parts[1]
    subject_uri = parts[2]
    object_uri = parts[3]
    metadata_json = parts[4]

    # English-only filter
    if not subject_uri.startswith("/c/en/") or not object_uri.startswith("/c/en/"):
        return None

    # Extract clean relation name (e.g. /r/IsA -> IsA)
    relation = relation_uri.split("/")[-1]

    # Exclude vague relations
    if relation in EXCLUDED_RELATIONS:
        return None

    # Extract clean entity names (e.g. /c/en/dog/n -> dog)
    subject = _clean_entity(subject_uri)
    obj = _clean_entity(object_uri)

    # Skip self-loops
    if subject == obj:
        return None

    # Skip very short or very long entity names
    if len(subject) < 2 or len(obj) < 2 or len(subject) > 50 or len(obj) > 50:
        return None

    # Parse weight from metadata
    try:
        metadata = json.loads(metadata_json)
        weight = float(metadata.get("weight", 0.0))
    except (json.JSONDecodeError, ValueError):
        return None

    return Triple(subject=subject, relation=relation, obj=obj, weight=weight)


def _clean_entity(uri: str) -> str:
    """Extract clean entity name from ConceptNet URI.

    /c/en/dog/n/wn/animal -> dog
    /c/en/hot_dog -> hot dog
    """
    parts = uri.split("/")
    # /c/en/entity_name[/pos[/...]]
    if len(parts) >= 4:
        entity = parts[3]
    else:
        entity = parts[-1]
    return entity.replace("_", " ")


def load_triples(
    assertions_path: Path,
    min_weight: float,
    max_triples: int | None = None,
) -> list[Triple]:
    """Load and filter ConceptNet triples from assertions file.

    Filters:
    - English entities only
    - Weight >= min_weight
    - Excludes vague relations (SimilarTo, RelatedTo, etc.)
    - Excludes self-loops and very short/long entity names
    """
    triples: list[Triple] = []
    total_lines = 0
    skipped = 0

    open_fn = gzip.open if str(assertions_path).endswith(".gz") else open

    logger.info("Loading triples from %s (min_weight=%.1f) ...", assertions_path, min_weight)

    with open_fn(assertions_path, "rt", encoding="utf-8") as f:
        for line in f:
            total_lines += 1

            if total_lines % 1_000_000 == 0:
                logger.info(
                    "  Processed %dM lines, kept %d triples so far",
                    total_lines // 1_000_000,
                    len(triples),
                )

            triple = parse_conceptnet_line(line)
            if triple is None:
                skipped += 1
                continue

            if triple.weight < min_weight:
                skipped += 1
                continue

            triples.append(triple)

            if max_triples is not None and len(triples) >= max_triples:
                logger.info("Reached max_triples limit (%d), stopping.", max_triples)
                break

    logger.info(
        "Loaded %d triples from %d lines (skipped %d)",
        len(triples), total_lines, skipped,
    )

    # Log relation distribution
    relation_counts = Counter(t.relation for t in triples)
    logger.info("Relation distribution (top 15):")
    for rel, count in relation_counts.most_common(15):
        logger.info("  %s: %d (%.1f%%)", rel, count, 100 * count / len(triples))

    return triples


def build_graph(triples: list[Triple]) -> nx.MultiDiGraph:
    """Build a directed graph from filtered triples.

    Edge attributes include the relation type and weight.
    Multi-edges between the same pair are stored as separate edges
    using a MultiDiGraph converted to DiGraph with composite keys.
    """
    graph = nx.MultiDiGraph()

    for triple in triples:
        graph.add_edge(
            triple.subject,
            triple.obj,
            relation=triple.relation,
            weight=triple.weight,
        )

    logger.info(
        "Built graph: %d nodes, %d edges",
        graph.number_of_nodes(),
        graph.number_of_edges(),
    )

    return graph


def extract_1hop_paths(
    graph: nx.MultiDiGraph,
    rng: random.Random,
    target_count: int,
    relation_counter: Counter[str],
    max_relation_share: float,
    seen_entity_sets: set[frozenset[str]],
) -> list[KGPath]:
    """Extract 1-hop paths (single edges) with diversity constraints."""
    edges = list(graph.edges(data=True))
    rng.shuffle(edges)

    paths: list[KGPath] = []
    total_paths_so_far = sum(relation_counter.values())

    for subj, obj, data in edges:
        if len(paths) >= target_count:
            break

        relation = data["relation"]

        # Diversity check: no relation > max_relation_share
        if not _relation_diversity_ok(
            relation, relation_counter, total_paths_so_far + len(paths), max_relation_share
        ):
            continue

        entity_set = frozenset([subj, obj])
        if entity_set in seen_entity_sets:
            continue

        path = KGPath(
            path=[[subj, relation, obj]],
            hops=1,
            relations=[relation],
            entities=[subj, obj],
        )

        seen_entity_sets.add(entity_set)
        relation_counter[relation] += 1
        paths.append(path)

    logger.info("Extracted %d 1-hop paths (target: %d)", len(paths), target_count)
    return paths


def extract_multihop_paths(
    graph: nx.MultiDiGraph,
    rng: random.Random,
    target_count: int,
    num_hops: int,
    relation_counter: Counter[str],
    max_relation_share: float,
    seen_entity_sets: set[frozenset[str]],
    max_attempts_multiplier: int = 50,
) -> list[KGPath]:
    """Extract multi-hop paths via random walks with diversity constraints.

    For each path, starts at a random node and performs a random walk
    of the specified length, checking diversity and dedup constraints.
    """
    nodes = list(graph.nodes())
    if not nodes:
        return []

    paths: list[KGPath] = []
    attempts = 0
    max_attempts = target_count * max_attempts_multiplier
    # Track total relation occurrences (not path count) for diversity check
    # _relation_diversity_ok compares per-relation count against total, so both
    # must be in the same unit (relation occurrences)
    total_relations_so_far = sum(relation_counter.values())

    # Pre-compute adjacency for faster random walks
    adj: dict[str, list[tuple[str, dict[str, Any]]]] = defaultdict(list)
    for u, v, data in graph.edges(data=True):
        adj[u].append((v, data))

    while len(paths) < target_count and attempts < max_attempts:
        attempts += 1

        # Random starting node
        start = rng.choice(nodes)
        current = start
        triples: list[list[str]] = []
        visited: set[str] = {current}
        entities: list[str] = [current]
        relations: list[str] = []
        valid = True

        for _ in range(num_hops):
            neighbors = adj.get(current, [])
            if not neighbors:
                valid = False
                break

            # Filter to unvisited neighbors
            unvisited = [(v, d) for v, d in neighbors if v not in visited]
            if not unvisited:
                valid = False
                break

            next_node, edge_data = rng.choice(unvisited)
            relation = edge_data["relation"]

            # Diversity check for this relation
            # Use total relation occurrences (not path count) for consistent units
            current_total_rels = total_relations_so_far + sum(
                len(p.relations) for p in paths
            )
            if not _relation_diversity_ok(
                relation, relation_counter,
                current_total_rels, max_relation_share,
            ):
                # Try other neighbors with different relations
                alternatives = [
                    (v, d) for v, d in unvisited
                    if d["relation"] != relation
                    and _relation_diversity_ok(
                        d["relation"], relation_counter,
                        current_total_rels, max_relation_share,
                    )
                ]
                if alternatives:
                    next_node, edge_data = rng.choice(alternatives)
                    relation = edge_data["relation"]
                else:
                    valid = False
                    break

            triples.append([current, relation, next_node])
            relations.append(relation)
            visited.add(next_node)
            entities.append(next_node)
            current = next_node

        if not valid or len(triples) != num_hops:
            continue

        entity_set = frozenset(entities)
        if entity_set in seen_entity_sets:
            continue

        path = KGPath(
            path=triples,
            hops=num_hops,
            relations=relations,
            entities=entities,
        )

        seen_entity_sets.add(entity_set)
        for rel in relations:
            relation_counter[rel] += 1
        paths.append(path)

    logger.info(
        "Extracted %d %d-hop paths (target: %d, attempts: %d)",
        len(paths), num_hops, target_count, attempts,
    )
    return paths


def _relation_diversity_ok(
    relation: str,
    relation_counter: Counter[str],
    total_paths: int,
    max_share: float,
) -> bool:
    """Check if adding this relation would violate the diversity constraint."""
    if total_paths < 10:
        # Too few paths to enforce diversity
        return True
    current_count = relation_counter.get(relation, 0)
    projected_share = (current_count + 1) / (total_paths + 1)
    return projected_share <= max_share


def extract_paths(config: ExtractionConfig) -> list[KGPath]:
    """Main extraction pipeline: load, filter, build graph, extract paths."""
    rng = random.Random(config.seed)

    # Load and filter triples
    triples = load_triples(
        config.assertions_path,
        min_weight=config.min_weight,
        max_triples=config.max_triples_to_load,
    )

    if not triples:
        logger.error("No triples loaded. Check assertions file path and filters.")
        return []

    # Build graph
    graph = build_graph(triples)

    # Calculate hop targets
    hop_targets: dict[int, int] = {}
    for hops, share in config.hop_distribution.items():
        hop_targets[hops] = int(config.num_paths * share)

    logger.info("Hop targets: %s", hop_targets)

    # Shared state for diversity and dedup
    relation_counter: Counter[str] = Counter()
    seen_entity_sets: set[frozenset[str]] = set()

    all_paths: list[KGPath] = []

    # Extract 1-hop paths first (simplest, most abundant)
    if 1 in hop_targets:
        paths_1hop = extract_1hop_paths(
            graph, rng, hop_targets[1],
            relation_counter, config.max_relation_share, seen_entity_sets,
        )
        all_paths.extend(paths_1hop)

    # Extract 2-hop paths
    if 2 in hop_targets:
        paths_2hop = extract_multihop_paths(
            graph, rng, hop_targets[2], num_hops=2,
            relation_counter=relation_counter,
            max_relation_share=config.max_relation_share,
            seen_entity_sets=seen_entity_sets,
        )
        all_paths.extend(paths_2hop)

    # Extract 3-hop paths
    if 3 in hop_targets:
        paths_3hop = extract_multihop_paths(
            graph, rng, hop_targets[3], num_hops=3,
            relation_counter=relation_counter,
            max_relation_share=config.max_relation_share,
            seen_entity_sets=seen_entity_sets,
        )
        all_paths.extend(paths_3hop)

    # Log final statistics
    _log_extraction_stats(all_paths, relation_counter)

    return all_paths


def _log_extraction_stats(
    paths: list[KGPath],
    relation_counter: Counter[str],
) -> None:
    """Log summary statistics of extracted paths."""
    total = len(paths)
    if total == 0:
        logger.warning("No paths extracted!")
        return

    hop_counts = Counter(p.hops for p in paths)
    logger.info("=== Extraction Summary ===")
    logger.info("Total paths: %d", total)
    for hops in sorted(hop_counts):
        count = hop_counts[hops]
        logger.info("  %d-hop: %d (%.1f%%)", hops, count, 100 * count / total)

    logger.info("Relation distribution (top 15):")
    total_relations = sum(relation_counter.values())
    for rel, count in relation_counter.most_common(15):
        logger.info("  %s: %d (%.1f%%)", rel, count, 100 * count / total_relations)

    # Check diversity constraint
    if not relation_counter or total_relations == 0:
        logger.info("No relations found (empty extraction)")
        return
    max_share = max(count / total_relations for count in relation_counter.values())
    logger.info("Max relation share: %.1f%%", 100 * max_share)


def save_paths(paths: list[KGPath], output_path: Path) -> None:
    """Save extracted paths to JSONL file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for path in paths:
            f.write(json.dumps(path.to_dict(), ensure_ascii=False) + "\n")

    logger.info("Saved %d paths to %s", len(paths), output_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract multi-hop paths from ConceptNet for KG-Align-RL."
    )
    parser.add_argument(
        "--assertions_path",
        type=Path,
        default=Path("data/raw/conceptnet-assertions-5.7.0.csv.gz"),
        help="Path to ConceptNet assertions CSV (plain or gzipped).",
    )
    parser.add_argument(
        "--output_path",
        type=Path,
        default=Path("data/processed/conceptnet_paths.jsonl"),
        help="Output JSONL path for extracted KG paths.",
    )
    parser.add_argument(
        "--num_paths",
        type=int,
        default=8000,
        help="Target number of unique paths to extract.",
    )
    parser.add_argument(
        "--min_weight",
        type=float,
        default=2.0,
        help="Minimum ConceptNet edge weight to include.",
    )
    parser.add_argument(
        "--max_relation_share",
        type=float,
        default=0.20,
        help="Maximum fraction of paths that can use any single relation type.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download ConceptNet assertions if not present.",
    )
    parser.add_argument(
        "--max_triples",
        type=int,
        default=None,
        help="Maximum number of triples to load (for testing). None = load all.",
    )
    args = parser.parse_args()

    config = ExtractionConfig(
        assertions_path=args.assertions_path,
        output_path=args.output_path,
        min_weight=args.min_weight,
        num_paths=args.num_paths,
        max_relation_share=args.max_relation_share,
        seed=args.seed,
        max_triples_to_load=args.max_triples,
    )

    logger.info("ExtractionConfig: %s", config)

    # Optionally download ConceptNet
    if args.download:
        download_conceptnet(config.assertions_path)

    if not config.assertions_path.exists():
        logger.error(
            "Assertions file not found at %s. "
            "Use --download to fetch it, or provide the correct path.",
            config.assertions_path,
        )
        return

    # Run extraction
    paths = extract_paths(config)

    if not paths:
        logger.error("No paths extracted. Exiting.")
        return

    # Save results
    save_paths(paths, config.output_path)


if __name__ == "__main__":
    main()
