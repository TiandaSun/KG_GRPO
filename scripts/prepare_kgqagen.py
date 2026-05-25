"""Convert KGQAGen-10k from HuggingFace dataset to verl parquet format.

Matches the schema of data/freebase/verl_cwq/test.parquet so the same
eval_with_tools.py script works on both benchmarks (just point at different
parquet + KG server).

Output: data/wikidata/verl_kgqagen/{dev,test}.parquet

Usage:
    python scripts/prepare_kgqagen.py
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# System prompt for the KG agent (Wikidata variant — uses QID/PID terminology
# but the agent still queries by entity name like the Freebase prompt)
SYSTEM_PROMPT = (
    "You are a knowledge graph reasoning agent. You can query a knowledge graph to answer questions.\n\n"
    "Available tools:\n"
    "- get_tail_relations(entity): Get all relation types going out from an entity\n"
    "- get_head_relations(entity): Get all relation types coming into an entity\n"
    "- get_tail_entities(entity, relation): Get entities reachable from entity via relation\n"
    "- get_head_entities(entity, relation): Get entities that connect to entity via relation\n\n"
    "To use a tool, write: <search>tool_name(arguments)</search>\n"
    "To give your final answer, write: <answer>your answer</answer>\n\n"
    "Think step-by-step about what to query. Use <think>...</think> for reasoning."
)

QID_PATTERN = re.compile(r"\(Q(\d+)\)")
PID_PATTERN = re.compile(r"\(P(\d+)\)")


def parse_proof_string(proof_str: str) -> list[list[str]]:
    """Parse a proof string into list of [head_label, prop_label, tail_label].

    Handles both list-of-lists format and string repr.
    """
    if isinstance(proof_str, list):
        return proof_str
    if not proof_str:
        return []
    try:
        # KGQAGen stores proof as Python list literal in a string
        # Use json after replacing single quotes
        return json.loads(proof_str.replace("'", '"'))
    except Exception:
        return []


def parse_answer_field(ans) -> list[str]:
    """Parse the 'answer' field which is stored as a string repr of a list."""
    if isinstance(ans, list):
        return [str(a) for a in ans]
    if isinstance(ans, str):
        try:
            parsed = json.loads(ans.replace("'", '"'))
            if isinstance(parsed, list):
                return [str(a) for a in parsed]
            return [str(parsed)]
        except Exception:
            return [ans]
    return []


def extract_label(text: str) -> str:
    """Extract just the label part from 'Label (Q12345)' format."""
    if not isinstance(text, str):
        return str(text)
    # Strip ' (Q123)' or ' (P456)' suffix
    cleaned = re.sub(r"\s*\(Q\d+\)\s*$", "", text)
    cleaned = re.sub(r"\s*\(P\d+\)\s*$", "", cleaned)
    return cleaned.strip()


def convert_sample(sample: dict, idx: int) -> dict:
    """Convert one KGQAGen-10k sample to verl parquet row format."""
    sample_id = f"KGQAGen-{sample.get('id', idx)}"
    question = sample.get("question", "")
    answers = parse_answer_field(sample.get("answer", "[]"))
    proof = parse_proof_string(sample.get("proof", "[]"))

    # Extract kg_path: list of [head_label, prop_label, tail_label] (labels only)
    kg_path = []
    for triple in proof:
        if isinstance(triple, list) and len(triple) >= 3:
            kg_path.append([extract_label(triple[0]), extract_label(triple[1]), extract_label(triple[2])])

    # Hops = number of unique entities in proof - 1 (rough estimate)
    entities_in_proof = set()
    for triple in proof:
        if isinstance(triple, list) and len(triple) >= 3:
            entities_in_proof.add(extract_label(triple[0]))
            entities_in_proof.add(extract_label(triple[2]))
    hops = max(1, len(entities_in_proof) - 1)

    seed_qid = sample.get("seed", "")

    gold_answer = answers[0] if answers else ""

    return {
        "data_source": "kgqagen_10k",
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
        "extra_info": {
            "sample_id": sample_id,
            "all_answers": answers,
            "answer_entities": answers,
            "gold_answer_short": gold_answer,
            "hops": hops,
            "kg_path": kg_path,
            "seed_qid": seed_qid,
            "subgraph_size": len(proof),
            "tools_kwargs": {"kg_query": {"create_kwargs": {"gold_answer": gold_answer}}},
        },
        "reward_model": {
            "style": "rule",
            "ground_truth": gold_answer,
        },
        "agent_name": "tool_agent",
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    parser = argparse.ArgumentParser(description="Convert KGQAGen-10k to verl parquet")
    parser.add_argument("--output_dir", type=Path, default=Path("data/wikidata/verl_kgqagen"))
    args = parser.parse_args()

    os.environ.setdefault("HF_HOME", "/scratch/u6gg/ts1201.u6gg/hf_cache")
    from datasets import load_dataset

    logger.info("Loading KGQAGen-10k from HuggingFace")
    ds = load_dataset("lianglz/KGQAGen-10k")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for split_name in ["dev", "test"]:
        rows = []
        for i, sample in enumerate(ds[split_name]):
            row = convert_sample(dict(sample), i)
            rows.append(row)

        df = pd.DataFrame(rows)
        out_file = args.output_dir / f"{split_name}.parquet"
        df.to_parquet(out_file, index=False)
        logger.info("Saved %d %s rows to %s", len(rows), split_name, out_file)

    print("\n=== KGQAGen-10k Conversion ===")
    print(f"Output: {args.output_dir}/")
    print(f"  dev.parquet:  {len(ds['dev'])} samples")
    print(f"  test.parquet: {len(ds['test'])} samples")


if __name__ == "__main__":
    main()
