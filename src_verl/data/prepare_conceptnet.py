"""Convert ConceptNet QA JSONL to verl parquet format.

Reads existing data/processed/conceptnet_qa_train.jsonl and converts
to the parquet schema expected by verl's data pipeline.

verl parquet columns (verl 0.7.0):
  - data_source: str (e.g. "conceptnet")
  - prompt: list[dict] (chat messages as native list, NOT JSON string)
  - extra_info: dict with nested tools_kwargs, interaction_kwargs, index
  - agent_name: str (e.g. "tool_agent")

Usage:
    python src_verl/data/prepare_conceptnet.py \
        --input_file data/processed/conceptnet_qa_train.jsonl \
        --output_dir data/processed/verl_conceptnet \
        --split train
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import datasets

logger = logging.getLogger(__name__)

# System prompt for the KG agent
SYSTEM_PROMPT = """You are a knowledge graph reasoning agent. You can query a knowledge graph to answer questions.

Available tools:
- get_tail_relations(entity): Get all relation types going out from an entity
- get_head_relations(entity): Get all relation types coming into an entity
- get_tail_entities(entity, relation): Get entities reachable from entity via relation
- get_head_entities(entity, relation): Get entities that connect to entity via relation

To use a tool, write: <search>tool_name(arguments)</search>
To give your final answer, write: <answer>your answer</answer>

Think step-by-step about what to query. Use <think>...</think> for reasoning."""


def load_qa_records(input_file: Path, max_records: int | None = None) -> list[dict[str, Any]]:
    """Load QA records from JSONL."""
    records: list[dict[str, Any]] = []
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if record.get("is_negative", False):
                continue
            records.append(record)
            if max_records and len(records) >= max_records:
                break
    logger.info("Loaded %d records from %s", len(records), input_file)
    return records


def records_to_verl_parquet(
    records: list[dict[str, Any]],
    data_source: str = "conceptnet",
) -> list[dict[str, Any]]:
    """Convert QA records to verl dict format.

    verl 0.7.0 expects (via HF datasets):
    - data_source: str
    - prompt: list[dict] (chat messages as native list)
    - extra_info: dict with tools_kwargs nested inside
    - agent_name: str
    """
    rows: list[dict[str, Any]] = []

    for idx, record in enumerate(records):
        question = record["question"]
        gold_answer = record.get("gold_answer_short", "")
        kg_path = record.get("kg_path", [])
        hops = record.get("hops", 1)

        # Format as chat messages (native list, not JSON string)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]

        # Extract query entities (first entity in path)
        query_entities = []
        if kg_path:
            query_entities.append(kg_path[0][0])

        # verl reads tools_kwargs from extra_info (rl_dataset.py line 349)
        extra = {
            "index": idx,
            "kg_path": kg_path,
            "gold_answer_short": gold_answer,
            "hops": hops,
            "query_entities": query_entities,
            "tools_kwargs": {
                "kg_query": {
                    "create_kwargs": {
                        "gold_answer": gold_answer,
                        "kg_path": kg_path,
                    }
                }
            },
        }

        # verl reward manager reads ground_truth from reward_model column
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

    parser = argparse.ArgumentParser(
        description="Convert ConceptNet QA JSONL to verl parquet."
    )
    parser.add_argument(
        "--input_file",
        type=Path,
        default=Path("data/processed/conceptnet_qa_train.jsonl"),
        help="Input JSONL file.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("data/processed/verl_conceptnet"),
        help="Output directory for parquet files.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "val", "test"],
        help="Data split name.",
    )
    parser.add_argument(
        "--max_records",
        type=int,
        default=None,
        help="Maximum records to include.",
    )
    args = parser.parse_args()

    records = load_qa_records(args.input_file, args.max_records)
    if not records:
        logger.error("No records loaded from %s", args.input_file)
        return

    rows = records_to_verl_parquet(records)

    # Use HF datasets to write parquet — preserves native list/dict types
    ds = datasets.Dataset.from_list(rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / f"{args.split}.parquet"
    ds.to_parquet(str(output_path))
    logger.info("Wrote %d records to %s", len(records), output_path)
    logger.info("Columns: %s", ds.column_names)


if __name__ == "__main__":
    main()
