"""Generate gold multi-turn trajectories from ConceptNet QA pairs.

For each QA pair with a known KG path, constructs the oracle agent
trajectory: think → query KG → observe → think → query → ... → answer.

Queries the live KG server for realistic observations so the SFT
training data matches what the model will see during GRPO rollouts.

Usage:
    # Start KG server first:
    python src_verl/kg_server/server.py --kg conceptnet --port 8001

    # Then generate trajectories:
    python src_verl/data/generate_sft_trajectories.py \
        --input_file data/processed/conceptnet_qa_train.jsonl \
        --output_file data/processed/sft_trajectories.jsonl \
        --kg_server_url http://localhost:8001
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import requests

from src_verl.data.prepare_conceptnet import SYSTEM_PROMPT

logger = logging.getLogger(__name__)


def query_kg_server(
    kg_server_url: str,
    action: str,
    entity: str,
    relation: str | None = None,
) -> list[str]:
    """Query the KG server for a single action."""
    payload = {
        "action": action,
        "entity": entity,
    }
    if relation:
        payload["relation"] = relation

    try:
        resp = requests.post(f"{kg_server_url}/retrieve", json=payload, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        return data.get("results", [])
    except (requests.RequestException, json.JSONDecodeError) as e:
        logger.warning("KG server query failed: %s", e)
        return []


def generate_trajectory(
    question: str,
    kg_path: list[list[str]],
    gold_answer: str,
    kg_server_url: str,
) -> list[dict[str, str]]:
    """Generate a gold multi-turn trajectory for a single QA pair.

    Strategy:
    1. For each hop in the KG path:
       a. Think about what to query next
       b. Query for relations from current entity
       c. Observe relations
       d. Think about which relation to follow
       e. Query for entities via the correct relation
       f. Observe entities
    2. Final: think and give answer

    Returns list of messages in chat format.
    """
    trajectory: list[dict[str, str]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]

    for hop_idx, triple in enumerate(kg_path):
        if len(triple) < 3:
            continue

        head, relation, tail = triple[0], triple[1], triple[2]

        # Step 1: Think + query relations from head entity
        think_text = _generate_think_relations(hop_idx, head, kg_path)
        trajectory.append({
            "role": "assistant",
            "content": f"<think>{think_text}</think>\n<search>get_tail_relations({head})</search>",
        })

        # Get real observation from KG server
        relations = query_kg_server(kg_server_url, "get_tail_relations", head)
        if not relations:
            relations = [relation]  # Fallback to gold relation
        trajectory.append({
            "role": "tool",
            "content": f"<information>{relations}</information>",
        })

        # Step 2: Think + query entities via the correct relation
        think_text = _generate_think_entities(hop_idx, head, relation, relations)
        trajectory.append({
            "role": "assistant",
            "content": f"<think>{think_text}</think>\n<search>get_tail_entities({head}, {relation})</search>",
        })

        # Get real observation from KG server
        entities = query_kg_server(kg_server_url, "get_tail_entities", head, relation)
        if not entities:
            entities = [tail]  # Fallback to gold tail
        trajectory.append({
            "role": "tool",
            "content": f"<information>{entities}</information>",
        })

    # Final step: think and answer
    think_text = _generate_think_answer(kg_path, gold_answer)
    trajectory.append({
        "role": "assistant",
        "content": f"<think>{think_text}</think>\n<answer>{gold_answer}</answer>",
    })

    return trajectory


def _generate_think_relations(
    hop_idx: int,
    entity: str,
    kg_path: list[list[str]],
) -> str:
    """Generate thinking text for a relation query step."""
    if hop_idx == 0:
        return (
            f"I need to answer this question. Let me start by exploring "
            f"what relations are available for '{entity}' in the knowledge graph."
        )
    else:
        prev_tail = kg_path[hop_idx - 1][2] if len(kg_path[hop_idx - 1]) >= 3 else "the previous entity"
        return (
            f"From the previous step, I found '{prev_tail}'. "
            f"Now let me explore what relations '{entity}' has to continue the reasoning chain."
        )


def _generate_think_entities(
    hop_idx: int,
    entity: str,
    relation: str,
    available_relations: list[str],
) -> str:
    """Generate thinking text for an entity query step."""
    other_rels = [r for r in available_relations if r != relation][:3]
    if other_rels:
        return (
            f"I can see several relations from '{entity}': {', '.join(available_relations[:5])}. "
            f"The '{relation}' relation seems most relevant to the question. "
            f"Let me follow this relation to find connected entities."
        )
    return (
        f"The '{relation}' relation from '{entity}' is relevant to the question. "
        f"Let me find what entities are connected via this relation."
    )


def _generate_think_answer(
    kg_path: list[list[str]],
    gold_answer: str,
) -> str:
    """Generate thinking text for the final answer step."""
    path_desc_parts: list[str] = []
    for triple in kg_path:
        if len(triple) >= 3:
            path_desc_parts.append(f"{triple[0]} --{triple[1]}--> {triple[2]}")
    path_desc = " → ".join(path_desc_parts) if path_desc_parts else "the knowledge graph"

    return (
        f"Following the knowledge graph path: {path_desc}. "
        f"Based on this reasoning chain, the answer to the question is '{gold_answer}'."
    )


def generate_all_trajectories(
    records: list[dict[str, Any]],
    kg_server_url: str,
    use_server: bool = True,
) -> list[dict[str, Any]]:
    """Generate trajectories for all QA records."""
    trajectories: list[dict[str, Any]] = []
    skipped = 0

    for i, record in enumerate(records):
        question = record["question"]
        kg_path = record.get("kg_path", [])
        gold_answer = record.get("gold_answer_short", "")

        if not kg_path or not gold_answer:
            skipped += 1
            continue

        if use_server:
            traj = generate_trajectory(question, kg_path, gold_answer, kg_server_url)
        else:
            # Offline mode: use synthetic observations from KG path
            traj = _generate_offline_trajectory(question, kg_path, gold_answer)

        trajectories.append({
            "trajectory": traj,
            "question": question,
            "gold_answer_short": gold_answer,
            "kg_path": kg_path,
            "hops": record.get("hops", len(kg_path)),
            "num_turns": len([m for m in traj if m["role"] == "assistant"]),
        })

        if (i + 1) % 100 == 0:
            logger.info("Generated %d/%d trajectories (skipped %d)", len(trajectories), i + 1, skipped)

    logger.info(
        "Generated %d trajectories total (skipped %d records without path/answer)",
        len(trajectories), skipped,
    )
    return trajectories


def _generate_offline_trajectory(
    question: str,
    kg_path: list[list[str]],
    gold_answer: str,
) -> list[dict[str, str]]:
    """Generate trajectory without KG server (synthetic observations)."""
    trajectory: list[dict[str, str]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]

    for hop_idx, triple in enumerate(kg_path):
        if len(triple) < 3:
            continue

        head, relation, tail = triple[0], triple[1], triple[2]

        # Query relations
        think_text = _generate_think_relations(hop_idx, head, kg_path)
        trajectory.append({
            "role": "assistant",
            "content": f"<think>{think_text}</think>\n<search>get_tail_relations({head})</search>",
        })
        trajectory.append({
            "role": "tool",
            "content": f"<information>['{relation}']</information>",
        })

        # Query entities
        think_text = _generate_think_entities(hop_idx, head, relation, [relation])
        trajectory.append({
            "role": "assistant",
            "content": f"<think>{think_text}</think>\n<search>get_tail_entities({head}, {relation})</search>",
        })
        trajectory.append({
            "role": "tool",
            "content": f"<information>['{tail}']</information>",
        })

    # Final answer
    think_text = _generate_think_answer(kg_path, gold_answer)
    trajectory.append({
        "role": "assistant",
        "content": f"<think>{think_text}</think>\n<answer>{gold_answer}</answer>",
    })

    return trajectory


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Generate gold multi-turn trajectories for SFT warmup."
    )
    parser.add_argument(
        "--input_file",
        type=Path,
        default=Path("data/processed/conceptnet_qa_train.jsonl"),
        help="Input JSONL with QA pairs + KG paths.",
    )
    parser.add_argument(
        "--output_file",
        type=Path,
        default=Path("data/processed/sft_trajectories.jsonl"),
        help="Output JSONL with multi-turn trajectories.",
    )
    parser.add_argument(
        "--kg_server_url",
        type=str,
        default="http://localhost:8001",
        help="URL of the running KG server.",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Generate without KG server (synthetic observations from gold path).",
    )
    parser.add_argument(
        "--max_records",
        type=int,
        default=None,
        help="Maximum records to process.",
    )
    args = parser.parse_args()

    # Load records
    records: list[dict[str, Any]] = []
    with open(args.input_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if record.get("is_negative", False):
                continue
            records.append(record)
            if args.max_records and len(records) >= args.max_records:
                break
    logger.info("Loaded %d records from %s", len(records), args.input_file)

    # Check KG server health (unless offline)
    use_server = not args.offline
    if use_server:
        try:
            resp = requests.get(f"{args.kg_server_url}/health", timeout=5)
            resp.raise_for_status()
            health = resp.json()
            logger.info("KG server healthy: %s (%d entities)", health["kg_type"], health["num_entities"])
        except requests.RequestException:
            logger.warning("KG server not reachable at %s, falling back to offline mode", args.kg_server_url)
            use_server = False

    # Generate trajectories
    trajectories = generate_all_trajectories(records, args.kg_server_url, use_server)

    # Save
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_file, "w", encoding="utf-8") as f:
        for traj in trajectories:
            f.write(json.dumps(traj, ensure_ascii=False) + "\n")
    logger.info("Saved %d trajectories to %s", len(trajectories), args.output_file)

    # Stats
    if trajectories:
        turn_counts = [t["num_turns"] for t in trajectories]
        hop_counts = [t["hops"] for t in trajectories]
        logger.info("Turn stats: min=%d, max=%d, mean=%.1f", min(turn_counts), max(turn_counts), sum(turn_counts) / len(turn_counts))
        logger.info("Hop distribution: %s", {h: hop_counts.count(h) for h in sorted(set(hop_counts))})
    else:
        logger.warning("No trajectories generated")


if __name__ == "__main__":
    main()
