"""Stage 2a: Question generation from KG paths using an instruction-tuned LLM.

Loads extracted KG paths and generates 2 QA variants per path in <think> CoT
format, using Qwen2.5-7B-Instruct for generation.

Usage:
    python src/datagen/question_generator.py \
        --paths_file data/processed/conceptnet_paths.jsonl \
        --output_file data/processed/conceptnet_qa_raw.jsonl \
        --model_name Qwen/Qwen2.5-7B-Instruct \
        --batch_size 4 \
        --seed 42
"""

from __future__ import annotations

import argparse
import json
import logging
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
class GenerationConfig:
    """Configuration for question generation."""

    paths_file: Path
    output_file: Path
    model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    variants_per_path: int = 2
    batch_size: int = 4
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    seed: int = 42
    save_every: int = 200
    torch_dtype: str = "bfloat16"
    resume: bool = False


def format_path_string(kg_path: list[list[str]]) -> str:
    """Format a KG path as a human-readable string.

    Example: [["dog", "IsA", "animal"], ["animal", "HasProperty", "alive"]]
    -> "[dog] --IsA--> [animal] --HasProperty--> [alive]"
    """
    if not kg_path:
        return ""

    parts = [f"[{kg_path[0][0]}]"]
    for triple in kg_path:
        parts.append(f"--{triple[1]}-->")
        parts.append(f"[{triple[2]}]")
    return " ".join(parts)


def build_generation_prompt(kg_path: list[list[str]], variant: int) -> str:
    """Build the prompt for QA generation from a KG path.

    Uses slightly different phrasing for each variant to encourage diversity.
    """
    path_str = format_path_string(kg_path)
    hops = len(kg_path)

    if variant == 0:
        return (
            f"Given this knowledge graph path:\n"
            f"{path_str}\n\n"
            f"Generate a natural question whose answer requires following this "
            f"{'single-step' if hops == 1 else 'multi-step'} reasoning path.\n"
            f"Provide the answer in the following format:\n\n"
            f"<think>\n"
            f"[Step-by-step reasoning that walks through the knowledge graph path, "
            f"explaining each hop and how it leads to the answer]\n"
            f"</think>\n"
            f"[Final answer in a short phrase]\n\n"
            f"Format your response as:\n"
            f"Question: <question>\n"
            f"Answer: <full answer with thinking and final answer>"
        )
    else:
        return (
            f"Consider this chain of knowledge:\n"
            f"{path_str}\n\n"
            f"Write a question that someone might naturally ask, where answering it "
            f"requires reasoning through {'this relationship' if hops == 1 else 'these connected relationships'}.\n"
            f"Then provide a detailed answer showing your reasoning step by step.\n\n"
            f"Use this format for the answer:\n"
            f"<think>\n"
            f"[Walk through the knowledge path step by step, explaining the "
            f"reasoning at each hop]\n"
            f"</think>\n"
            f"[A concise final answer]\n\n"
            f"Format your full response as:\n"
            f"Question: <your question>\n"
            f"Answer: <full answer with thinking tags and final answer>"
        )


def parse_qa_response(response: str) -> tuple[str, str, str] | None:
    """Parse the LLM response to extract question, full answer, and short answer.

    Returns (question, full_answer, short_answer) or None if parsing fails.
    """
    # Extract question
    question_match = re.search(r"Question:\s*(.+?)(?:\n|Answer:)", response, re.DOTALL)
    if not question_match:
        return None
    question = question_match.group(1).strip()

    # Extract full answer (everything after "Answer:")
    answer_match = re.search(r"Answer:\s*(.+)", response, re.DOTALL)
    if not answer_match:
        return None
    full_answer = answer_match.group(1).strip()

    # Extract short answer (text after </think> tag)
    if "</think>" in full_answer:
        short_answer = full_answer.split("</think>")[-1].strip()
    else:
        # No think tags — use the whole answer as short answer
        short_answer = full_answer.strip()

    # Basic validation
    if len(question) < 10 or len(full_answer) < 20:
        return None

    return question, full_answer, short_answer


def load_paths(paths_file: Path) -> list[dict[str, Any]]:
    """Load KG paths from JSONL file."""
    paths = []
    with open(paths_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                paths.append(json.loads(line))
    logger.info("Loaded %d paths from %s", len(paths), paths_file)
    return paths


def load_existing_outputs(output_file: Path) -> set[str]:
    """Load already-generated path keys for resumption.

    Returns a set of (path_str, variant) keys that have been generated.
    """
    existing: set[str] = set()
    if not output_file.exists():
        return existing

    with open(output_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                record = json.loads(line)
                # Create a key from the path and variant
                path_key = json.dumps(record.get("kg_path", []), sort_keys=True)
                variant = record.get("variant", 0)
                existing.add(f"{path_key}|{variant}")

    logger.info("Found %d existing outputs for resumption", len(existing))
    return existing


def load_model_and_tokenizer(
    model_name: str,
    torch_dtype: str,
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """Load the generation model and tokenizer."""
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    dtype = dtype_map.get(torch_dtype, torch.bfloat16)

    logger.info("Loading model %s with dtype %s ...", model_name, torch_dtype)
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map="auto",
    )
    model.eval()

    logger.info("Model loaded on device: %s", model.device)
    return model, tokenizer


def generate_batch(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompts: list[str],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> list[str]:
    """Generate responses for a batch of prompts."""
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
        max_length=1024,
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )

    # Decode only the new tokens
    responses = []
    for i, output in enumerate(outputs):
        input_len = inputs["input_ids"][i].shape[0]
        new_tokens = output[input_len:]
        response = tokenizer.decode(new_tokens, skip_special_tokens=True)
        responses.append(response)

    return responses


def generate_qa_pairs(config: GenerationConfig) -> None:
    """Main generation pipeline."""
    torch.manual_seed(config.seed)

    # Load paths
    paths = load_paths(config.paths_file)

    # Handle resumption
    existing_keys: set[str] = set()
    if config.resume:
        existing_keys = load_existing_outputs(config.output_file)

    # Build work items: (path_record, variant_index)
    work_items: list[tuple[dict[str, Any], int]] = []
    for path_record in paths:
        for variant in range(config.variants_per_path):
            path_key = json.dumps(path_record["path"], sort_keys=True)
            key = f"{path_key}|{variant}"
            if key not in existing_keys:
                work_items.append((path_record, variant))

    logger.info(
        "Work items: %d (total possible: %d, already done: %d)",
        len(work_items),
        len(paths) * config.variants_per_path,
        len(existing_keys),
    )

    if not work_items:
        logger.info("All items already generated. Nothing to do.")
        return

    # Load model
    model, tokenizer = load_model_and_tokenizer(config.model_name, config.torch_dtype)

    # Open output file in append mode
    config.output_file.parent.mkdir(parents=True, exist_ok=True)
    generated_count = 0
    failed_count = 0
    start_time = time.time()

    with open(config.output_file, "a", encoding="utf-8") as out_f:
        for batch_start in range(0, len(work_items), config.batch_size):
            batch = work_items[batch_start : batch_start + config.batch_size]

            # Build prompts
            prompts = [
                build_generation_prompt(record["path"], variant)
                for record, variant in batch
            ]

            # Generate
            try:
                responses = generate_batch(
                    model, tokenizer, prompts,
                    config.max_new_tokens, config.temperature, config.top_p,
                )
            except Exception:
                logger.exception(
                    "Generation failed for batch starting at %d", batch_start,
                )
                failed_count += len(batch)
                continue

            # Parse and save
            for (record, variant), response in zip(batch, responses):
                parsed = parse_qa_response(response)
                if parsed is None:
                    failed_count += 1
                    logger.debug(
                        "Failed to parse response for path: %s (variant %d)",
                        record["path"], variant,
                    )
                    continue

                question, full_answer, short_answer = parsed

                qa_record = {
                    "question": question,
                    "answer": full_answer,
                    "kg_path": record["path"],
                    "gold_answer_short": short_answer,
                    "hops": record["hops"],
                    "is_negative": False,
                    "variant": variant,
                    "relations": record.get("relations", []),
                }
                out_f.write(json.dumps(qa_record, ensure_ascii=False) + "\n")
                generated_count += 1

            # Periodic flush and logging
            if (batch_start // config.batch_size + 1) % config.save_every == 0:
                out_f.flush()

            total_processed = batch_start + len(batch)
            if total_processed % (config.batch_size * 50) == 0 or total_processed == len(work_items):
                elapsed = time.time() - start_time
                rate = total_processed / elapsed if elapsed > 0 else 0
                logger.info(
                    "Progress: %d/%d items (%.1f/s), generated: %d, failed: %d",
                    total_processed, len(work_items), rate,
                    generated_count, failed_count,
                )

    elapsed = time.time() - start_time
    logger.info(
        "=== Generation Complete ===\n"
        "  Total generated: %d\n"
        "  Failed to parse: %d\n"
        "  Time: %.1f seconds\n"
        "  Output: %s",
        generated_count, failed_count, elapsed, config.output_file,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate QA pairs from KG paths using an instruction-tuned LLM."
    )
    parser.add_argument(
        "--paths_file",
        type=Path,
        default=Path("data/processed/conceptnet_paths.jsonl"),
        help="Input JSONL file with extracted KG paths.",
    )
    parser.add_argument(
        "--output_file",
        type=Path,
        default=Path("data/processed/conceptnet_qa_raw.jsonl"),
        help="Output JSONL file for raw QA pairs.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct",
        help="HuggingFace model ID for generation.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Number of prompts per generation batch.",
    )
    parser.add_argument(
        "--variants_per_path",
        type=int,
        default=2,
        help="Number of QA variants to generate per KG path.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=512,
        help="Maximum new tokens per generation.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature.",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="Top-p (nucleus) sampling.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--save_every",
        type=int,
        default=200,
        help="Flush output every N batches.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing output file (skip already generated items).",
    )
    parser.add_argument(
        "--torch_dtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="Torch dtype for model loading.",
    )
    args = parser.parse_args()

    config = GenerationConfig(
        paths_file=args.paths_file,
        output_file=args.output_file,
        model_name=args.model_name,
        variants_per_path=args.variants_per_path,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        seed=args.seed,
        save_every=args.save_every,
        torch_dtype=args.torch_dtype,
        resume=args.resume,
    )

    logger.info("GenerationConfig: %s", config)
    generate_qa_pairs(config)


if __name__ == "__main__":
    main()
