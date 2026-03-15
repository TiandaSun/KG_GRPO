"""Stage 2b: Quality filtering of generated QA pairs using LLM-as-judge.

Scores each QA pair on answerability, faithfulness, and naturalness (1-5),
then filters to keep only high-quality pairs. Splits into train/val/test.

Usage:
    python src/datagen/quality_filter.py \
        --input_file data/processed/conceptnet_qa_raw.jsonl \
        --output_dir data/processed \
        --model_name Qwen/Qwen2.5-7B-Instruct \
        --min_score 4 \
        --seed 42
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class FilterConfig:
    """Configuration for quality filtering."""

    input_file: Path
    output_dir: Path
    model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    min_score: int = 4
    batch_size: int = 4
    max_new_tokens: int = 256
    target_total: int = 5000
    train_count: int = 4000
    val_count: int = 500
    test_count: int = 500
    seed: int = 42
    torch_dtype: str = "bfloat16"
    scores_cache_file: Path | None = None


def build_judge_prompt(
    question: str,
    answer: str,
    kg_path: list[list[str]],
) -> str:
    """Build the LLM-as-judge prompt for quality scoring."""
    # Format path for readability
    path_parts = []
    for triple in kg_path:
        path_parts.append(f"  {triple[0]} --{triple[1]}--> {triple[2]}")
    path_str = "\n".join(path_parts)

    return (
        f"You are evaluating the quality of a question-answer pair generated from a "
        f"knowledge graph path. Score each dimension from 1 (worst) to 5 (best).\n\n"
        f"Knowledge graph path:\n{path_str}\n\n"
        f"Question: {question}\n\n"
        f"Answer: {answer}\n\n"
        f"Score the following dimensions:\n"
        f"1. Answerability (1-5): Can the question be clearly answered using the "
        f"knowledge graph path above?\n"
        f"2. Faithfulness (1-5): Does the chain-of-thought reasoning in the answer "
        f"actually follow the knowledge graph path (mentions the correct entities "
        f"and relations in order)?\n"
        f"3. Naturalness (1-5): Does the question sound like something a human would "
        f"naturally ask? Is it grammatically correct and not awkwardly phrased?\n\n"
        f"Respond with ONLY the three scores in this exact format:\n"
        f"Answerability: <score>\n"
        f"Faithfulness: <score>\n"
        f"Naturalness: <score>"
    )


def parse_judge_scores(response: str) -> dict[str, int] | None:
    """Parse the judge response to extract scores.

    Returns dict with keys answerability, faithfulness, naturalness or None.
    """
    scores: dict[str, int] = {}

    for dimension in ["Answerability", "Faithfulness", "Naturalness"]:
        match = re.search(rf"{dimension}:\s*(\d)", response)
        if match:
            score = int(match.group(1))
            if 1 <= score <= 5:
                scores[dimension.lower()] = score
            else:
                return None
        else:
            return None

    return scores


def load_qa_records(input_file: Path) -> list[dict[str, Any]]:
    """Load raw QA records from JSONL."""
    records = []
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    logger.info("Loaded %d raw QA records from %s", len(records), input_file)
    return records


def load_model_and_tokenizer(
    model_name: str,
    torch_dtype: str,
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """Load the judge model and tokenizer."""
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    dtype = dtype_map.get(torch_dtype, torch.bfloat16)

    logger.info("Loading judge model %s with dtype %s ...", model_name, torch_dtype)
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map="auto",
    )
    model.eval()

    logger.info("Judge model loaded on device: %s", model.device)
    return model, tokenizer


def score_batch(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompts: list[str],
    max_new_tokens: int,
) -> list[str]:
    """Score a batch of QA pairs using the judge model."""
    messages_batch = [
        [{"role": "user", "content": prompt}]
        for prompt in prompts
    ]

    texts = [
        tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        for msgs in messages_batch
    ]

    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=1536,
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.1,  # Low temperature for consistent scoring
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )

    responses = []
    for i, output in enumerate(outputs):
        input_len = inputs["input_ids"][i].shape[0]
        new_tokens = output[input_len:]
        response = tokenizer.decode(new_tokens, skip_special_tokens=True)
        responses.append(response)

    return responses


def score_all_records(
    records: list[dict[str, Any]],
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    config: FilterConfig,
) -> list[dict[str, Any]]:
    """Score all QA records using the judge model."""
    scored_records: list[dict[str, Any]] = []
    failed_count = 0
    start_time = time.time()

    for batch_start in range(0, len(records), config.batch_size):
        batch = records[batch_start : batch_start + config.batch_size]

        prompts = [
            build_judge_prompt(
                record["question"],
                record["answer"],
                record["kg_path"],
            )
            for record in batch
        ]

        try:
            responses = score_batch(model, tokenizer, prompts, config.max_new_tokens)
        except Exception:
            logger.exception("Scoring failed for batch starting at %d", batch_start)
            failed_count += len(batch)
            continue

        for record, response in zip(batch, responses):
            scores = parse_judge_scores(response)
            if scores is None:
                failed_count += 1
                logger.debug("Failed to parse scores for: %s", record["question"][:60])
                continue

            record_with_scores = {**record, "scores": scores}
            scored_records.append(record_with_scores)

        total_processed = batch_start + len(batch)
        if total_processed % (config.batch_size * 50) == 0 or total_processed >= len(records):
            elapsed = time.time() - start_time
            rate = total_processed / elapsed if elapsed > 0 else 0
            logger.info(
                "Scoring progress: %d/%d (%.1f/s), scored: %d, failed: %d",
                total_processed, len(records), rate,
                len(scored_records), failed_count,
            )

    logger.info(
        "Scoring complete: %d scored, %d failed out of %d total",
        len(scored_records), failed_count, len(records),
    )
    return scored_records


def filter_and_split(
    scored_records: list[dict[str, Any]],
    config: FilterConfig,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Filter records by minimum score and split into train/val/test.

    Returns (train, val, test) lists.
    """
    rng = random.Random(config.seed)

    # Filter: all three dimensions >= min_score
    passed = [
        record for record in scored_records
        if all(
            score >= config.min_score
            for score in record["scores"].values()
        )
    ]

    logger.info(
        "Filter results: %d/%d passed (min_score=%d on all dimensions)",
        len(passed), len(scored_records), config.min_score,
    )

    # Log score distributions
    for dim in ["answerability", "faithfulness", "naturalness"]:
        scores_list = [r["scores"][dim] for r in scored_records if dim in r["scores"]]
        if scores_list:
            avg = sum(scores_list) / len(scores_list)
            logger.info("  %s: mean=%.2f", dim, avg)

    # If we have more than target, subsample; if fewer, use all
    if len(passed) > config.target_total:
        rng.shuffle(passed)
        passed = passed[: config.target_total]
        logger.info("Subsampled to %d records", len(passed))
    elif len(passed) < config.target_total:
        logger.warning(
            "Only %d records passed filtering (target: %d). "
            "Consider lowering --min_score.",
            len(passed), config.target_total,
        )

    # Shuffle before splitting
    rng.shuffle(passed)

    # Split: train / val / test
    # If we have fewer than target, split proportionally
    total = len(passed)
    if total >= config.train_count + config.val_count + config.test_count:
        train = passed[: config.train_count]
        val = passed[config.train_count : config.train_count + config.val_count]
        test = passed[config.train_count + config.val_count : config.train_count + config.val_count + config.test_count]
    else:
        # Proportional split: 80% / 10% / 10%
        val_start = int(total * 0.8)
        test_start = int(total * 0.9)
        train = passed[:val_start]
        val = passed[val_start:test_start]
        test = passed[test_start:]

    logger.info("Split: train=%d, val=%d, test=%d", len(train), len(val), len(test))
    return train, val, test


def save_split(
    records: list[dict[str, Any]],
    output_path: Path,
    remove_metadata: bool = True,
) -> None:
    """Save a split to JSONL, optionally removing intermediate metadata."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for record in records:
            out_record = dict(record)
            if remove_metadata:
                out_record.pop("scores", None)
                out_record.pop("variant", None)
            f.write(json.dumps(out_record, ensure_ascii=False) + "\n")

    logger.info("Saved %d records to %s", len(records), output_path)


def run_filter_pipeline(config: FilterConfig) -> None:
    """Main filtering pipeline."""
    # Load raw QA records
    records = load_qa_records(config.input_file)

    if not records:
        logger.error("No records to filter. Run question_generator.py first.")
        return

    # Check for cached scores
    scores_cache = config.scores_cache_file or config.input_file.with_suffix(".scored.jsonl")
    if scores_cache.exists():
        logger.info("Loading cached scores from %s", scores_cache)
        scored_records = load_qa_records(scores_cache)
    else:
        # Load model and score
        model, tokenizer = load_model_and_tokenizer(config.model_name, config.torch_dtype)
        scored_records = score_all_records(records, model, tokenizer, config)

        # Cache scores
        save_split(scored_records, scores_cache, remove_metadata=False)
        logger.info("Cached scored records to %s", scores_cache)

        # Free GPU memory
        del model
        del tokenizer
        torch.cuda.empty_cache()

    # Filter and split
    train, val, test = filter_and_split(scored_records, config)

    # Save splits
    save_split(train, config.output_dir / "conceptnet_qa_train.jsonl")
    save_split(val, config.output_dir / "conceptnet_qa_val.jsonl")
    save_split(test, config.output_dir / "conceptnet_qa_test.jsonl")

    # Also save the full scored dataset for analysis
    save_split(scored_records, config.output_dir / "conceptnet_qa_scored.jsonl", remove_metadata=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Filter generated QA pairs using LLM-as-judge scoring."
    )
    parser.add_argument(
        "--input_file",
        type=Path,
        default=Path("data/processed/conceptnet_qa_raw.jsonl"),
        help="Input JSONL file with raw QA pairs.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("data/processed"),
        help="Output directory for filtered splits.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct",
        help="HuggingFace model ID for judge scoring.",
    )
    parser.add_argument(
        "--min_score",
        type=int,
        default=4,
        choices=[1, 2, 3, 4, 5],
        help="Minimum score on all dimensions to pass filter.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Number of QA pairs to score per batch.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=256,
        help="Maximum new tokens for judge response.",
    )
    parser.add_argument(
        "--target_total",
        type=int,
        default=5000,
        help="Target number of QA pairs after filtering.",
    )
    parser.add_argument(
        "--train_count",
        type=int,
        default=4000,
        help="Number of training examples.",
    )
    parser.add_argument(
        "--val_count",
        type=int,
        default=500,
        help="Number of validation examples.",
    )
    parser.add_argument(
        "--test_count",
        type=int,
        default=500,
        help="Number of test examples.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shuffling and splitting.",
    )
    parser.add_argument(
        "--torch_dtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="Torch dtype for model loading.",
    )
    parser.add_argument(
        "--scores_cache",
        type=Path,
        default=None,
        help="Path to cache scored records (avoids re-scoring).",
    )
    args = parser.parse_args()

    config = FilterConfig(
        input_file=args.input_file,
        output_dir=args.output_dir,
        model_name=args.model_name,
        min_score=args.min_score,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        target_total=args.target_total,
        train_count=args.train_count,
        val_count=args.val_count,
        test_count=args.test_count,
        seed=args.seed,
        torch_dtype=args.torch_dtype,
        scores_cache_file=args.scores_cache,
    )

    logger.info("FilterConfig: %s", config)
    run_filter_pipeline(config)


if __name__ == "__main__":
    main()
