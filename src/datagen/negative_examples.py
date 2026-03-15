"""Stage 2c: Programmatic generation of negative (corrupted) QA examples.

Swaps entities in KG paths to create corrupted variants that teach the model
to distinguish valid from invalid reasoning paths. No LLM needed.

Usage:
    python src/datagen/negative_examples.py \
        --train_file data/processed/conceptnet_qa_train.jsonl \
        --paths_file data/processed/conceptnet_paths.jsonl \
        --output_file data/processed/conceptnet_qa_train_with_neg.jsonl \
        --num_negatives 500 \
        --seed 42
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class NegativeConfig:
    """Configuration for negative example generation."""

    train_file: Path
    paths_file: Path
    output_file: Path
    num_negatives: int = 500
    seed: int = 42


def load_jsonl(file_path: Path) -> list[dict[str, Any]]:
    """Load records from a JSONL file."""
    records = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def collect_entity_pool(paths_file: Path) -> list[str]:
    """Collect all unique entities from the paths file for corruption."""
    entities: set[str] = set()
    records = load_jsonl(paths_file)
    for record in records:
        for entity in record.get("entities", []):
            entities.add(entity)
    entity_list = sorted(entities)  # Sort for reproducibility
    logger.info("Collected %d unique entities for corruption pool", len(entity_list))
    return entity_list


def corrupt_path(
    kg_path: list[list[str]],
    entity_pool: list[str],
    rng: random.Random,
) -> tuple[list[list[str]], str, str]:
    """Corrupt a KG path by swapping one entity with a random replacement.

    Returns (corrupted_path, original_entity, replacement_entity).
    """
    corrupted = [list(triple) for triple in kg_path]

    # Collect all entity positions (not relations)
    # Each triple is [subject, relation, object]
    positions: list[tuple[int, int]] = []  # (triple_idx, position: 0=subj, 2=obj)
    for i, triple in enumerate(corrupted):
        positions.append((i, 0))  # subject
        positions.append((i, 2))  # object

    if not positions:
        return corrupted, "", ""
    # Pick a random position to corrupt
    triple_idx, pos_idx = rng.choice(positions)
    original_entity = corrupted[triple_idx][pos_idx]

    # Pick a replacement entity that differs from the original
    # and doesn't already appear in the path
    path_entities = {e for triple in kg_path for e in (triple[0], triple[2])}
    candidates = [e for e in entity_pool if e not in path_entities]

    if not candidates:
        # Fallback: just pick something different from original
        candidates = [e for e in entity_pool if e != original_entity]

    if not candidates:
        # Extremely unlikely but handle gracefully
        return corrupted, original_entity, original_entity

    replacement = rng.choice(candidates)
    corrupted[triple_idx][pos_idx] = replacement

    # If we corrupted a connecting entity (object of one triple = subject of next),
    # also update the corresponding position to keep the path structurally coherent
    # but factually wrong
    if pos_idx == 2 and triple_idx + 1 < len(corrupted):
        # We changed the object, which is the subject of the next triple
        if kg_path[triple_idx][2] == kg_path[triple_idx + 1][0]:
            corrupted[triple_idx + 1][0] = replacement
    elif pos_idx == 0 and triple_idx > 0:
        # We changed the subject, which is the object of the previous triple
        if kg_path[triple_idx][0] == kg_path[triple_idx - 1][2]:
            corrupted[triple_idx - 1][2] = replacement

    return corrupted, original_entity, replacement


def build_corrupted_statement(corrupted_path: list[list[str]]) -> str:
    """Build a natural language statement from a corrupted path."""
    parts = []
    for triple in corrupted_path:
        subj, rel, obj = triple
        # Convert relation to natural language
        rel_text = _relation_to_text(rel, subj, obj)
        parts.append(rel_text)
    return " ".join(parts)


def _relation_to_text(relation: str, subject: str, obj: str) -> str:
    """Convert a KG relation triple to natural language."""
    templates: dict[str, str] = {
        "IsA": f"{subject} is a type of {obj}",
        "HasProperty": f"{subject} has the property of being {obj}",
        "UsedFor": f"{subject} is used for {obj}",
        "CapableOf": f"{subject} is capable of {obj}",
        "Causes": f"{subject} causes {obj}",
        "HasA": f"{subject} has {obj}",
        "PartOf": f"{subject} is part of {obj}",
        "AtLocation": f"{subject} is found at {obj}",
        "HasPrerequisite": f"{subject} requires {obj}",
        "MotivatedByGoal": f"{subject} is motivated by {obj}",
        "HasSubevent": f"{subject} involves {obj}",
        "HasFirstSubevent": f"{subject} begins with {obj}",
        "HasLastSubevent": f"{subject} ends with {obj}",
        "DefinedAs": f"{subject} is defined as {obj}",
        "ReceivesAction": f"{subject} can be {obj}",
        "CreatedBy": f"{subject} is created by {obj}",
        "MadeOf": f"{subject} is made of {obj}",
        "CausesDesire": f"{subject} makes you want {obj}",
        "MannerOf": f"{subject} is a way to {obj}",
        "FormOf": f"{subject} is a form of {obj}",
        "HasContext": f"{subject} is used in the context of {obj}",
        "DerivedFrom": f"{subject} is derived from {obj}",
        "Antonym": f"{subject} is the opposite of {obj}",
    }
    return templates.get(relation, f"{subject} {relation} {obj}")


def build_negative_reasoning(
    corrupted_path: list[list[str]],
    original_entity: str,
    replacement_entity: str,
) -> str:
    """Build reasoning that explains why the corrupted path is wrong."""
    path_desc_parts = []
    for triple in corrupted_path:
        path_desc_parts.append(f"{triple[0]} → {triple[1]} → {triple[2]}")
    path_desc = ", ".join(path_desc_parts)

    return (
        f"Let me examine this claim by checking the reasoning path: {path_desc}. "
        f"This path contains '{replacement_entity}' where '{original_entity}' "
        f"would be expected based on common knowledge. "
        f"The relationship described does not hold because the entity "
        f"'{replacement_entity}' does not correctly fit in this knowledge path."
    )


def generate_negative_examples(config: NegativeConfig) -> list[dict[str, Any]]:
    """Generate negative QA examples by corrupting existing training paths."""
    rng = random.Random(config.seed)

    # Load training data and entity pool
    train_records = load_jsonl(config.train_file)
    entity_pool = collect_entity_pool(config.paths_file)

    if not train_records:
        logger.error("No training records found in %s", config.train_file)
        return []

    if not entity_pool:
        logger.error("No entities found in %s", config.paths_file)
        return []

    # Sample records to corrupt
    num_to_corrupt = min(config.num_negatives, len(train_records))
    source_records = rng.sample(train_records, num_to_corrupt)

    negatives: list[dict[str, Any]] = []
    for record in source_records:
        kg_path = record["kg_path"]

        corrupted_path, original_entity, replacement = corrupt_path(
            kg_path, entity_pool, rng,
        )

        if original_entity == replacement:
            continue

        statement = build_corrupted_statement(corrupted_path)
        reasoning = build_negative_reasoning(
            corrupted_path, original_entity, replacement,
        )

        negative_record = {
            "question": f"Is it true that {statement}?",
            "answer": f"<think>{reasoning}</think>\nNo, this is incorrect.",
            "kg_path": corrupted_path,
            "gold_answer_short": "No, this is incorrect.",
            "hops": record["hops"],
            "is_negative": True,
        }
        negatives.append(negative_record)

    logger.info("Generated %d negative examples", len(negatives))
    return negatives


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate negative (corrupted) QA examples from training data."
    )
    parser.add_argument(
        "--train_file",
        type=Path,
        default=Path("data/processed/conceptnet_qa_train.jsonl"),
        help="Input training JSONL file.",
    )
    parser.add_argument(
        "--paths_file",
        type=Path,
        default=Path("data/processed/conceptnet_paths.jsonl"),
        help="KG paths file (for entity pool).",
    )
    parser.add_argument(
        "--output_file",
        type=Path,
        default=Path("data/processed/conceptnet_qa_train_with_neg.jsonl"),
        help="Output training file with negatives appended.",
    )
    parser.add_argument(
        "--num_negatives",
        type=int,
        default=500,
        help="Number of negative examples to generate.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    args = parser.parse_args()

    config = NegativeConfig(
        train_file=args.train_file,
        paths_file=args.paths_file,
        output_file=args.output_file,
        num_negatives=args.num_negatives,
        seed=args.seed,
    )

    logger.info("NegativeConfig: %s", config)

    # Generate negatives
    negatives = generate_negative_examples(config)

    if not negatives:
        logger.error("No negative examples generated.")
        return

    # Load original training data and append negatives
    train_records = load_jsonl(config.train_file)
    combined = train_records + negatives

    # Shuffle the combined dataset
    rng = random.Random(config.seed + 1)  # Different seed to avoid correlation
    rng.shuffle(combined)

    # Save
    config.output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(config.output_file, "w", encoding="utf-8") as f:
        for record in combined:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    logger.info(
        "Saved %d records (%d positive + %d negative) to %s",
        len(combined), len(train_records), len(negatives), config.output_file,
    )


if __name__ == "__main__":
    main()
