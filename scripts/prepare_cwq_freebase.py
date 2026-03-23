"""Download and preprocess CWQ/Freebase data from RoG HuggingFace datasets.

This script:
1. Downloads RoG-CWQ and RoG-WebQSP from HuggingFace
2. Builds a global Freebase subgraph (entities.txt, relations.txt, triples.txt)
3. Converts QA data to verl parquet format
4. Extracts hop count estimates from subgraph structure
5. Optionally downloads original CWQ for SPARQL-based hop counts

Usage (login node OK — downloads + text processing only):
    python scripts/prepare_cwq_freebase.py --output_dir data/freebase

Output structure:
    data/freebase/
        kg/                     # Global KG for server
            entities.txt
            relations.txt
            triples.txt
        verl_cwq/               # verl parquet for GRPO training
            train.parquet
            val.parquet
            test.parquet
        verl_webqsp/            # verl parquet for cross-dataset eval
            train.parquet
            test.parquet
        stats.json              # Dataset statistics
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

import datasets

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a knowledge graph reasoning agent. You can query a knowledge graph to answer questions.

Available tools:
- get_tail_relations(entity): Get all relation types going out from an entity
- get_head_relations(entity): Get all relation types coming into an entity
- get_tail_entities(entity, relation): Get entities reachable from entity via relation
- get_head_entities(entity, relation): Get entities that connect to entity via relation

To use a tool, write: <search>tool_name(arguments)</search>
To give your final answer, write: <answer>your answer</answer>

Think step-by-step about what to query. Use <think>...</think> for reasoning."""


def download_rog_dataset(name: str, cache_dir: str | None = None) -> datasets.DatasetDict:
    """Download a RoG dataset from HuggingFace."""
    logger.info("Downloading %s from HuggingFace...", name)
    ds = datasets.load_dataset(name, cache_dir=cache_dir)
    logger.info("Downloaded %s: %s", name, {k: len(v) for k, v in ds.items()})
    return ds


def build_global_kg(
    cwq_ds: datasets.DatasetDict,
    webqsp_ds: datasets.DatasetDict,
    output_dir: Path,
) -> dict[str, Any]:
    """Build global Freebase subgraph from per-question graphs.

    Unions all triples from all questions across both datasets.
    Writes entities.txt, relations.txt, triples.txt for the KG server.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    entities: set[str] = set()
    relations: set[str] = set()
    triples: set[tuple[str, str, str]] = set()

    for ds_name, ds in [("cwq", cwq_ds), ("webqsp", webqsp_ds)]:
        for split_name, split_ds in ds.items():
            logger.info("Processing %s/%s (%d samples)...", ds_name, split_name, len(split_ds))
            for row in split_ds:
                graph = row.get("graph", [])
                for triple in graph:
                    if len(triple) >= 3:
                        h, r, t = str(triple[0]), str(triple[1]), str(triple[2])
                        entities.add(h)
                        entities.add(t)
                        relations.add(r)
                        triples.add((h, r, t))

    # Sort for deterministic output
    entity_list = sorted(entities)
    relation_list = sorted(relations)

    # Write entities.txt (name-based, no separate IDs — entity name IS the ID)
    with open(output_dir / "entities.txt", "w", encoding="utf-8") as f:
        for e in entity_list:
            f.write(f"{e}\t{e}\n")

    # Write relations.txt
    with open(output_dir / "relations.txt", "w", encoding="utf-8") as f:
        for r in relation_list:
            f.write(f"{r}\t{r}\n")

    # Write triples.txt
    with open(output_dir / "triples.txt", "w", encoding="utf-8") as f:
        for h, r, t in sorted(triples):
            f.write(f"{h}\t{r}\t{t}\n")

    stats = {
        "num_entities": len(entity_list),
        "num_relations": len(relation_list),
        "num_triples": len(triples),
    }
    logger.info("Global KG: %d entities, %d relations, %d triples", *stats.values())
    return stats


def estimate_hop_count(graph: list[list[str]], q_entities: list[str], a_entities: list[str]) -> int:
    """Estimate hop count via BFS from question entities to answer entities on the subgraph.

    Returns the shortest path length, or 0 if no path found.
    """
    if not q_entities or not a_entities:
        return 0

    # Build adjacency from subgraph
    adj: dict[str, set[str]] = defaultdict(set)
    for triple in graph:
        if len(triple) >= 3:
            adj[triple[0]].add(triple[2])
            adj[triple[2]].add(triple[0])

    q_set = set(str(e) for e in q_entities)
    a_set = set(str(e) for e in a_entities)

    # BFS from question entities
    visited: set[str] = set()
    frontier = list(q_set)
    visited.update(q_set)
    depth = 0

    while frontier and depth < 10:
        if any(e in a_set for e in frontier):
            return max(depth, 1)
        next_frontier: list[str] = []
        for node in frontier:
            for neighbor in adj.get(node, set()):
                if neighbor not in visited:
                    visited.add(neighbor)
                    next_frontier.append(neighbor)
        frontier = next_frontier
        depth += 1

    return 0  # No path found


def rog_to_verl_rows(
    split_ds: datasets.Dataset,
    data_source: str = "cwq",
) -> list[dict[str, Any]]:
    """Convert RoG dataset split to verl parquet rows."""
    rows: list[dict[str, Any]] = []

    for row in split_ds:
        question = row.get("question", "")
        answers = row.get("answer", [])
        q_entities = row.get("q_entity", [])
        a_entities = row.get("a_entity", [])
        graph = row.get("graph", [])
        sample_id = row.get("id", "")

        if not question or not answers:
            continue

        # Primary answer (short entity — good for EM)
        gold_answer = answers[0] if answers else ""

        # Estimate hops
        hops = estimate_hop_count(graph, q_entities, a_entities)

        # Extract gold KG path (triples connecting q_entity to a_entity)
        # This is approximate — the full subgraph contains many irrelevant triples
        # For r_on_path reward, we'd need the actual shortest path
        kg_path = []
        if q_entities and a_entities:
            # Find triples that mention q or a entities as a rough path
            q_set = set(str(e).lower() for e in q_entities)
            a_set = set(str(e).lower() for e in a_entities)
            for triple in graph:
                if len(triple) >= 3:
                    h_lower = str(triple[0]).lower()
                    t_lower = str(triple[2]).lower()
                    if (h_lower in q_set or t_lower in q_set or
                            h_lower in a_set or t_lower in a_set):
                        kg_path.append([str(triple[0]), str(triple[1]), str(triple[2])])

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]

        extra = {
            "gold_answer_short": gold_answer,
            "all_answers": answers,
            "hops": hops,
            "kg_path": kg_path,
            "query_entities": q_entities,
            "answer_entities": a_entities,
            "sample_id": sample_id,
            "subgraph_size": len(graph),
            "tools_kwargs": {
                "kg_query": {
                    "create_kwargs": {
                        "gold_answer": gold_answer,
                        "kg_path": kg_path,
                    }
                }
            },
        }

        reward_model = {
            "style": "rule",
            "ground_truth": gold_answer,
        }

        rows.append({
            "data_source": data_source,
            "prompt": messages,
            "extra_info": extra,
            "reward_model": reward_model,
            "agent_name": "tool_agent",
        })

    return rows


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    parser = argparse.ArgumentParser(description="Prepare CWQ/Freebase data for verl GRPO training.")
    parser.add_argument("--output_dir", type=Path, default=Path("data/freebase"))
    parser.add_argument("--hf_cache", type=str, default=None,
                        help="HuggingFace cache dir (default: HF_HOME env var)")
    args = parser.parse_args()

    # Download datasets
    cwq_ds = download_rog_dataset("rmanluo/RoG-cwq", cache_dir=args.hf_cache)
    webqsp_ds = download_rog_dataset("rmanluo/RoG-webqsp", cache_dir=args.hf_cache)

    # Build global KG (skip if already exists)
    kg_dir = args.output_dir / "kg"
    if (kg_dir / "triples.txt").exists():
        import subprocess
        n_ent = int(subprocess.check_output(["wc", "-l", str(kg_dir / "entities.txt")]).split()[0])
        n_rel = int(subprocess.check_output(["wc", "-l", str(kg_dir / "relations.txt")]).split()[0])
        n_tri = int(subprocess.check_output(["wc", "-l", str(kg_dir / "triples.txt")]).split()[0])
        kg_stats = {"num_entities": n_ent, "num_relations": n_rel, "num_triples": n_tri}
        logger.info("KG already exists, skipping build: %s", kg_stats)
    else:
        kg_stats = build_global_kg(cwq_ds, webqsp_ds, kg_dir)

    # Convert CWQ to verl parquet
    cwq_output = args.output_dir / "verl_cwq"
    cwq_output.mkdir(parents=True, exist_ok=True)

    # RoG-CWQ has train/validation/test splits
    split_mapping = {"train": "train", "validation": "val", "test": "test"}
    cwq_stats: dict[str, int] = {}

    for rog_split, our_split in split_mapping.items():
        if rog_split not in cwq_ds:
            logger.warning("CWQ split %s not found, skipping", rog_split)
            continue
        rows = rog_to_verl_rows(cwq_ds[rog_split], data_source="cwq")
        ds = datasets.Dataset.from_list(rows)
        out_path = cwq_output / f"{our_split}.parquet"
        ds.to_parquet(str(out_path))
        cwq_stats[our_split] = len(rows)
        logger.info("CWQ %s: %d rows -> %s", our_split, len(rows), out_path)

        # Log hop distribution
        hops = [r["extra_info"]["hops"] for r in rows]
        hop_dist = defaultdict(int)
        for h in hops:
            hop_dist[h] += 1
        logger.info("  Hop distribution: %s", dict(sorted(hop_dist.items())))

    # Convert WebQSP to verl parquet
    webqsp_output = args.output_dir / "verl_webqsp"
    webqsp_output.mkdir(parents=True, exist_ok=True)

    webqsp_stats: dict[str, int] = {}
    for rog_split, our_split in split_mapping.items():
        if rog_split not in webqsp_ds:
            continue
        rows = rog_to_verl_rows(webqsp_ds[rog_split], data_source="webqsp")
        ds = datasets.Dataset.from_list(rows)
        out_path = webqsp_output / f"{our_split}.parquet"
        ds.to_parquet(str(out_path))
        webqsp_stats[our_split] = len(rows)
        logger.info("WebQSP %s: %d rows -> %s", our_split, len(rows), out_path)

    # Save stats
    all_stats = {
        "kg": kg_stats,
        "cwq": cwq_stats,
        "webqsp": webqsp_stats,
    }
    with open(args.output_dir / "stats.json", "w") as f:
        json.dump(all_stats, f, indent=2)

    logger.info("=== Done ===")
    logger.info("KG: %s", kg_stats)
    logger.info("CWQ: %s", cwq_stats)
    logger.info("WebQSP: %s", webqsp_stats)
    logger.info("Output: %s", args.output_dir)


if __name__ == "__main__":
    main()
