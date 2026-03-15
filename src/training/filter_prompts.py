"""Pre-GRPO prompt filtering (DAPO-inspired).

Generates multiple completions per prompt using the SFT model and removes
prompts where all completions receive the same reward (zero gradient signal).

Usage:
    python src/training/filter_prompts.py \
        --input_file data/processed/conceptnet_qa_train.jsonl \
        --output_file data/processed/conceptnet_qa_train_filtered.jsonl \
        --adapter_path outputs/sft-warmup \
        --num_generations 8

Expected to filter out 10-30% of prompts.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.rewards.kg_reward import compute_format_reward, compute_kg_reward

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_records(path: Path) -> list[dict[str, Any]]:
    """Load JSONL records, skipping negatives."""
    records: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if not record.get("is_negative", False):
                records.append(record)
    return records


def generate_completions(
    model: Any,
    tokenizer: Any,
    question: str,
    num_generations: int,
    max_new_tokens: int,
    temperature: float,
) -> list[str]:
    """Generate multiple completions for a single question."""
    messages = [{"role": "user", "content": question}]
    input_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=True,
        num_return_sequences=num_generations,
        pad_token_id=tokenizer.pad_token_id,
    )

    completions: list[str] = []
    input_len = inputs["input_ids"].shape[1]
    for output in outputs:
        completion = tokenizer.decode(output[input_len:], skip_special_tokens=True)
        completions.append(completion)

    return completions


def compute_reward(
    question: str,
    completion: str,
    kg_path: list[list[str]],
    gold_answer: str,
) -> float:
    """Compute combined reward for a single completion."""
    kg_r = compute_kg_reward(question, completion, kg_path, gold_answer)
    fmt_r = compute_format_reward(completion)
    return kg_r + fmt_r


def filter_prompts(
    input_file: Path,
    output_file: Path,
    model_name: str,
    adapter_path: Path | None,
    num_generations: int,
    max_new_tokens: int,
    temperature: float,
    batch_size: int,
) -> None:
    """Filter prompts by removing those with zero reward variance.

    For each prompt, generates ``num_generations`` completions, computes
    rewards, and keeps the prompt only if the rewards have non-zero
    standard deviation (i.e., the prompt provides a learning signal).
    """
    records = load_records(input_file)
    logger.info("Loaded %d records from %s", len(records), input_file)

    # Load model
    logger.info("Loading model %s", model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    if adapter_path and adapter_path.exists():
        from peft import PeftModel
        logger.info("Loading adapter from %s", adapter_path)
        model = PeftModel.from_pretrained(model, str(adapter_path))

    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    kept: list[dict[str, Any]] = []
    filtered_count = 0

    for i, record in enumerate(records):
        if (i + 1) % 50 == 0:
            logger.info(
                "Progress: %d/%d (kept=%d, filtered=%d)",
                i + 1, len(records), len(kept), filtered_count,
            )

        question = record["question"]
        kg_path = record["kg_path"]
        gold_answer = record["gold_answer_short"]

        with torch.no_grad():
            completions = generate_completions(
                model, tokenizer, question,
                num_generations, max_new_tokens, temperature,
            )

        rewards = [
            compute_reward(question, c, kg_path, gold_answer)
            for c in completions
        ]

        # Check if all rewards are the same (zero gradient signal)
        reward_set = set(round(r, 4) for r in rewards)
        if len(reward_set) <= 1:
            filtered_count += 1
            continue

        kept.append(record)

    # Write filtered dataset
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        for record in kept:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    logger.info("=== Filtering complete ===")
    logger.info("  Input: %d prompts", len(records))
    logger.info("  Kept: %d prompts", len(kept))
    logger.info("  Filtered: %d prompts (%.1f%%)", filtered_count, 100 * filtered_count / max(len(records), 1))
    logger.info("  Output: %s", output_file)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pre-GRPO prompt filtering (remove zero-gradient prompts)"
    )
    parser.add_argument(
        "--input_file",
        type=Path,
        default=Path("data/processed/conceptnet_qa_train.jsonl"),
        help="Input JSONL file with QA pairs.",
    )
    parser.add_argument(
        "--output_file",
        type=Path,
        default=Path("data/processed/conceptnet_qa_train_filtered.jsonl"),
        help="Output JSONL file with filtered QA pairs.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen2.5-1.5B-Instruct",
        help="Base model name.",
    )
    parser.add_argument(
        "--adapter_path",
        type=Path,
        default=Path("outputs/sft-warmup"),
        help="Path to SFT adapter (optional).",
    )
    parser.add_argument(
        "--num_generations",
        type=int,
        default=8,
        help="Number of completions per prompt.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=512,
        help="Max tokens per completion.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for generation (prompts processed sequentially for simplicity).",
    )
    args = parser.parse_args()

    filter_prompts(
        args.input_file,
        args.output_file,
        args.model_name,
        args.adapter_path,
        args.num_generations,
        args.max_new_tokens,
        args.temperature,
        args.batch_size,
    )


if __name__ == "__main__":
    main()
