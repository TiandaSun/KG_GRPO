"""Convert KG-R1 Freebase data (WebQSP/CWQ) to verl parquet format.

Reads KG-R1's pre-processed QA data and converts to verl's expected schema.

KG-R1 data layout:
  data_kg/
    WebQSP/
      train.json or train.jsonl
    CWQ/
      train.json or train.jsonl

Usage:
    python src_verl/data/prepare_freebase.py \
        --input_dir data_kg/WebQSP \
        --output_dir data/processed/verl_freebase \
        --split train
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import datasets

from src_verl.data.prepare_conceptnet import SYSTEM_PROMPT

logger = logging.getLogger(__name__)


def load_kgr1_data(input_dir: Path, split: str = "train") -> list[dict[str, Any]]:
    """Load KG-R1 format QA data.

    Tries multiple file patterns that KG-R1 might use.
    """
    records: list[dict[str, Any]] = []

    # Try different file patterns
    candidates = [
        input_dir / f"{split}.json",
        input_dir / f"{split}.jsonl",
        input_dir / f"{split}_simple.json",
    ]

    data_file = None
    for candidate in candidates:
        if candidate.exists():
            data_file = candidate
            break

    if data_file is None:
        logger.warning("No data file found in %s for split=%s", input_dir, split)
        return records

    logger.info("Loading KG-R1 data from %s", data_file)

    if data_file.suffix == ".json":
        with open(data_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                records = data
            elif isinstance(data, dict):
                records = list(data.values())
    else:
        # JSONL
        with open(data_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))

    logger.info("Loaded %d records from %s", len(records), data_file)
    return records


def kgr1_to_verl_rows(
    records: list[dict[str, Any]],
    data_source: str = "freebase",
) -> list[dict[str, Any]]:
    """Convert KG-R1 records to verl dict format (matching ConceptNet schema).

    Uses the same native list/dict schema as prepare_conceptnet.py so verl
    can load both datasets interchangeably.
    """
    rows: list[dict[str, Any]] = []

    for record in records:
        # KG-R1 uses various field names
        question = (
            record.get("question", "")
            or record.get("ProcessedQuestion", "")
            or record.get("RawQuestion", "")
        )
        if not question:
            continue

        # Answer extraction (KG-R1 format varies)
        gold_answer = ""
        if "answer" in record:
            ans = record["answer"]
            if isinstance(ans, list):
                gold_answer = ans[0] if ans else ""
            else:
                gold_answer = str(ans)
        elif "Parses" in record:
            # WebQSP format
            parses = record["Parses"]
            if parses and isinstance(parses, list):
                answers = parses[0].get("Answers", [])
                if answers:
                    gold_answer = answers[0].get("EntityName", "") or answers[0].get("AnswerArgument", "")

        # Extract topic entity
        query_entities = []
        if "topic_entity" in record:
            query_entities = [record["topic_entity"]]
        elif "Parses" in record and record["Parses"]:
            te = record["Parses"][0].get("TopicEntityName", "")
            if te:
                query_entities = [te]

        # KG path (if available in KG-R1 format)
        kg_path = record.get("kg_path", record.get("path", []))

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]

        extra = {
            "kg_path": kg_path,
            "gold_answer_short": gold_answer,
            "hops": len(kg_path) if kg_path else 0,
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
        description="Convert KG-R1 Freebase data to verl parquet."
    )
    parser.add_argument(
        "--input_dir",
        type=Path,
        required=True,
        help="Input directory (e.g., data_kg/WebQSP).",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("data/processed/verl_freebase"),
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
        "--data_source",
        type=str,
        default="freebase",
        help="Data source identifier.",
    )
    args = parser.parse_args()

    records = load_kgr1_data(args.input_dir, args.split)
    if not records:
        logger.error("No records loaded")
        return

    rows = kgr1_to_verl_rows(records, args.data_source)
    if not rows:
        logger.error("No valid rows produced")
        return

    # Use HF datasets to write parquet — preserves native list/dict types
    # (same approach as prepare_conceptnet.py)
    ds = datasets.Dataset.from_list(rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / f"{args.split}.parquet"
    ds.to_parquet(str(output_path))
    logger.info("Wrote %d records to %s", len(rows), output_path)
    logger.info("Columns: %s", ds.column_names)


if __name__ == "__main__":
    main()
