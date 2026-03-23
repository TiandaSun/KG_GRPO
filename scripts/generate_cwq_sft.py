"""Generate SFT trajectories from CWQ/Freebase data.

Constructs gold multi-turn agent trajectories from KG paths in the CWQ data.
Each trajectory shows the model how to use <search> tools to answer a question.

Output: JSONL file where each line is a conversation with tool calls.

Usage:
    python scripts/generate_cwq_sft.py \
        --input data/freebase/verl_cwq/train.parquet \
        --output data/freebase/sft_trajectories.jsonl \
        --max_samples 5000
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _to_python(obj: any) -> any:
    """Recursively convert numpy types to native Python types."""
    if isinstance(obj, np.ndarray):
        return [_to_python(x) for x in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.str_,)):
        return str(obj)
    return obj

SYSTEM_PROMPT = """You are a knowledge graph reasoning agent. You can query a knowledge graph to answer questions.

Available tools:
- get_tail_relations(entity): Get all relation types going out from an entity
- get_head_relations(entity): Get all relation types coming into an entity
- get_tail_entities(entity, relation): Get entities reachable from entity via relation
- get_head_entities(entity, relation): Get entities that connect to entity via relation

To use a tool, write: <search>tool_name(arguments)</search>
To give your final answer, write: <answer>your answer</answer>

Think step-by-step about what to query. Use <think>...</think> for reasoning."""


def build_trajectory(
    question: str,
    gold_answer: str,
    kg_path: list,
    query_entities: list,
) -> list[dict[str, str]] | None:
    """Build a gold multi-turn trajectory from a KG path.

    Returns a list of chat messages (system, user, assistant turns).
    Returns None if the data is insufficient to build a trajectory.
    """
    if not kg_path or not gold_answer:
        return None

    # Filter valid triples and cap at 5 to keep trajectories manageable
    # (kg_path from RoG contains the full per-question subgraph, not just shortest path)
    valid_triples = []
    for triple in kg_path:
        if hasattr(triple, '__len__') and len(triple) >= 3:
            valid_triples.append((str(triple[0]), str(triple[1]), str(triple[2])))
        if len(valid_triples) >= 5:
            break

    if not valid_triples:
        return None

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]

    # Determine starting entity
    start_entity = str(query_entities[0]) if query_entities else valid_triples[0][0]

    # Build multi-turn trajectory following the KG path
    prev_entity = start_entity

    for i, (head, relation, tail) in enumerate(valid_triples):
        # Step 1: Think about what to query
        if i == 0:
            think = f"I need to find information about {start_entity}. Let me start by checking its relations in the knowledge graph."
        else:
            think = f"Now I know about {prev_entity}. Let me follow the relation '{relation}' to continue reasoning."

        # Step 2: Decide which tool to call
        # If we know the relation, use get_tail_entities; otherwise get_tail_relations first
        if i == 0 and len(valid_triples) > 1:
            # First step: explore relations
            assistant_msg = (
                f"<think>{think}</think>\n"
                f"<search>get_tail_relations({head})</search>"
            )
            messages.append({"role": "assistant", "content": assistant_msg})

            # Simulate tool response with relations
            relations_list = [t[1] for t in valid_triples if t[0].lower() == head.lower()]
            if not relations_list:
                relations_list = [relation]
            tool_response = json.dumps(relations_list[:10])
            messages.append({"role": "user", "content": f"<tool_response>{tool_response}</tool_response>"})

            # Now query for entities along the relation
            think2 = f"I can see the relation '{relation}'. Let me follow it to find connected entities."
            assistant_msg2 = (
                f"<think>{think2}</think>\n"
                f"<search>get_tail_entities({head}, {relation})</search>"
            )
            messages.append({"role": "assistant", "content": assistant_msg2})

            # Simulate tool response
            tool_response2 = json.dumps([tail])
            messages.append({"role": "user", "content": f"<tool_response>{tool_response2}</tool_response>"})

        else:
            # Subsequent steps: directly query entities
            assistant_msg = (
                f"<think>{think}</think>\n"
                f"<search>get_tail_entities({head}, {relation})</search>"
            )
            messages.append({"role": "assistant", "content": assistant_msg})

            # Simulate tool response
            tool_response = json.dumps([tail])
            messages.append({"role": "user", "content": f"<tool_response>{tool_response}</tool_response>"})

        prev_entity = tail

    # Final answer
    conclusion = f"Based on my knowledge graph exploration, I found that the answer is {gold_answer}."
    final_msg = (
        f"<think>{conclusion}</think>\n"
        f"<answer>{gold_answer}</answer>"
    )
    messages.append({"role": "assistant", "content": final_msg})

    return messages


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    parser = argparse.ArgumentParser(description="Generate CWQ SFT trajectories.")
    parser.add_argument("--input", type=Path, default=Path("data/freebase/verl_cwq/train.parquet"))
    parser.add_argument("--output", type=Path, default=Path("data/freebase/sft_trajectories.jsonl"))
    parser.add_argument("--max_samples", type=int, default=5000,
                        help="Max trajectories to generate (use samples with best KG paths)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    logger.info("Loading data from %s", args.input)
    df = pd.read_parquet(args.input)
    logger.info("Loaded %d samples", len(df))

    # Generate trajectories
    trajectories = []
    skipped = 0

    for _, row in df.iterrows():
        extra = row["extra_info"]
        kg_path = extra.get("kg_path", [])
        gold_answer = extra.get("gold_answer_short", "")
        query_entities = extra.get("query_entities", [])
        question = row["prompt"][1]["content"]  # user message

        # Deep-convert numpy arrays to Python lists (parquet stores nested ndarrays)
        kg_path = _to_python(kg_path)
        query_entities = _to_python(query_entities)

        trajectory = build_trajectory(question, gold_answer, kg_path, query_entities)
        if trajectory is None:
            skipped += 1
            continue

        trajectories.append({"trajectory": trajectory})

    logger.info("Generated %d trajectories, skipped %d (no KG path)", len(trajectories), skipped)

    # Sample if we have more than max_samples
    if len(trajectories) > args.max_samples:
        random.shuffle(trajectories)
        trajectories = trajectories[:args.max_samples]
        logger.info("Sampled %d trajectories", len(trajectories))

    # Write output
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for traj in trajectories:
            f.write(json.dumps(traj, ensure_ascii=False) + "\n")

    logger.info("Wrote %d trajectories to %s", len(trajectories), args.output)

    # Stats
    num_turns = [len(t["trajectory"]) for t in trajectories]
    logger.info("Turns per trajectory: min=%d, max=%d, mean=%.1f",
                min(num_turns), max(num_turns), sum(num_turns) / len(num_turns))


if __name__ == "__main__":
    main()
