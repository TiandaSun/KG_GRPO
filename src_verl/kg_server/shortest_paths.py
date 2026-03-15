"""BFS shortest path precomputation for R_progress reward.

Computes shortest distances from all answer entities in the training set
to every reachable entity. This enables the R_progress reward component
to measure whether each agent step moves closer to the answer.

Usage:
    python src_verl/kg_server/shortest_paths.py \
        --kg conceptnet \
        --assertions_path data/raw/conceptnet-assertions-5.7.0.csv.gz \
        --train_file data/processed/conceptnet_qa_train.jsonl \
        --output_path data/processed/conceptnet_distances.pkl
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
from collections import deque
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def bfs_distances(
    graph_adj: dict[str, list[str]],
    source: str,
    max_depth: int = 5,
) -> dict[str, int]:
    """BFS from source, returning {entity: distance} for all reachable nodes.

    Args:
        graph_adj: Adjacency list (undirected — both directions).
        source: Starting entity.
        max_depth: Maximum BFS depth (limits computation).

    Returns:
        Dict mapping reachable entities to their shortest distance from source.
    """
    distances: dict[str, int] = {source: 0}
    queue: deque[tuple[str, int]] = deque([(source, 0)])

    while queue:
        node, depth = queue.popleft()
        if depth >= max_depth:
            continue
        for neighbor in graph_adj.get(node, []):
            if neighbor not in distances:
                distances[neighbor] = depth + 1
                queue.append((neighbor, depth + 1))

    return distances


def build_undirected_adj(
    adapter: Any,
) -> dict[str, list[str]]:
    """Build undirected adjacency list from a KG adapter's graph.

    Works with both ConceptNetAdapter (has _graph) and FreebaseAdapter (has _outgoing/_incoming).
    """
    adj: dict[str, list[str]] = {}

    if hasattr(adapter, "_graph"):
        # ConceptNet: use NetworkX graph
        for u, v in adapter._graph.edges():
            adj.setdefault(u, []).append(v)
            adj.setdefault(v, []).append(u)
    elif hasattr(adapter, "_outgoing"):
        # Freebase: use outgoing/incoming dicts
        for entity_id, rels in adapter._outgoing.items():
            entity_name = adapter._entity_name(entity_id)
            for rel_id, tails in rels.items():
                for tail_id in tails:
                    tail_name = adapter._entity_name(tail_id)
                    adj.setdefault(entity_name, []).append(tail_name)
                    adj.setdefault(tail_name, []).append(entity_name)
    else:
        logger.warning("Unknown adapter type, cannot build adjacency list")

    logger.info("Built undirected adjacency: %d nodes", len(adj))
    return adj


def precompute_distances(
    adapter: Any,
    answer_entities: set[str],
    max_depth: int = 5,
) -> dict[str, dict[str, int]]:
    """Precompute BFS distances from each answer entity.

    Args:
        adapter: KG adapter (ConceptNet or Freebase).
        answer_entities: Set of answer entity strings.
        max_depth: Max BFS depth.

    Returns:
        {answer_entity: {reachable_entity: distance}}
    """
    adj = build_undirected_adj(adapter)

    all_distances: dict[str, dict[str, int]] = {}
    found = 0
    skipped = 0

    for i, entity in enumerate(answer_entities):
        entity_lower = entity.lower().replace("_", " ")
        if entity_lower in adj:
            all_distances[entity_lower] = bfs_distances(adj, entity_lower, max_depth)
            found += 1
        else:
            skipped += 1

        if (i + 1) % 100 == 0:
            logger.info(
                "BFS progress: %d/%d (found: %d, skipped: %d)",
                i + 1, len(answer_entities), found, skipped,
            )

    logger.info(
        "Precomputed distances for %d answer entities (%d not found in graph)",
        found, skipped,
    )
    return all_distances


def extract_answer_entities(train_file: Path) -> set[str]:
    """Extract unique answer entities from training data."""
    entities: set[str] = set()
    with open(train_file, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line.strip())
            gold = record.get("gold_answer_short", "")
            if gold:
                entities.add(gold.strip())
            # Also add last entity from KG path
            kg_path = record.get("kg_path", [])
            if kg_path:
                last_triple = kg_path[-1]
                if len(last_triple) >= 3:
                    entities.add(last_triple[2].strip())

    logger.info("Extracted %d unique answer entities from %s", len(entities), train_file)
    return entities


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Precompute BFS shortest distances from answer entities."
    )
    parser.add_argument(
        "--kg",
        type=str,
        choices=["conceptnet", "freebase"],
        required=True,
        help="Which KG to use.",
    )
    parser.add_argument(
        "--assertions_path",
        type=Path,
        default=Path("data/raw/conceptnet-assertions-5.7.0.csv.gz"),
        help="Path to ConceptNet assertions (for conceptnet KG).",
    )
    parser.add_argument(
        "--freebase_dir",
        type=Path,
        default=Path("data_kg"),
        help="Path to KG-R1 Freebase data directory.",
    )
    parser.add_argument(
        "--train_file",
        type=Path,
        required=True,
        help="Training data JSONL to extract answer entities from.",
    )
    parser.add_argument(
        "--output_path",
        type=Path,
        required=True,
        help="Output path for pickled distance dict.",
    )
    parser.add_argument(
        "--max_depth",
        type=int,
        default=5,
        help="Maximum BFS depth.",
    )
    parser.add_argument(
        "--min_weight",
        type=float,
        default=2.0,
        help="Min weight for ConceptNet triples.",
    )
    args = parser.parse_args()

    # Load adapter
    if args.kg == "conceptnet":
        from src_verl.kg_server.conceptnet_adapter import ConceptNetAdapter
        adapter = ConceptNetAdapter.from_assertions(
            args.assertions_path, min_weight=args.min_weight
        )
    else:
        from src_verl.kg_server.freebase_adapter import FreebaseAdapter
        adapter = FreebaseAdapter.from_data_dir(args.freebase_dir)

    # Extract answer entities
    answer_entities = extract_answer_entities(args.train_file)

    # Precompute distances
    distances = precompute_distances(adapter, answer_entities, args.max_depth)

    # Save
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_path, "wb") as f:
        pickle.dump(distances, f)
    logger.info("Saved distances to %s", args.output_path)


if __name__ == "__main__":
    main()
