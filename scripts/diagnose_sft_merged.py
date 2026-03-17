"""Diagnostic: test if the SFT-merged model generates <search> and <think> tags.

Loads the merged model with the exact same system prompt and chat template
used in GRPO rollouts, generates completions, and reports format statistics.

Usage:
    python scripts/diagnose_sft_merged.py \
        --model_path outputs/verl-sft-merged \
        --num_samples 20 \
        --temperature 0.7
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(message)s")
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


def load_test_questions(data_path: Path, num_samples: int) -> list[dict]:
    """Load a few questions from the val set."""
    records = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
            if len(records) >= num_samples:
                break
    return records


def generate_completions(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    questions: list[dict],
    temperature: float,
    max_new_tokens: int,
    num_return_sequences: int,
) -> list[dict]:
    """Generate completions for each question."""
    results = []

    for i, q in enumerate(questions):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": q["question"]},
        ]

        input_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                top_p=1.0,
                num_return_sequences=num_return_sequences,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )

        # Decode only new tokens
        input_len = inputs["input_ids"].shape[1]
        for seq_idx in range(num_return_sequences):
            generated = outputs[seq_idx][input_len:]
            text = tokenizer.decode(generated, skip_special_tokens=False)
            results.append({
                "question_idx": i,
                "question": q["question"],
                "gold_answer": q.get("gold_answer_short", ""),
                "hops": q.get("hops", -1),
                "generated": text,
            })

        logger.info("Sample %d/%d done", i + 1, len(questions))

    return results


def analyze_results(results: list[dict]) -> None:
    """Analyze format compliance of generated outputs."""
    total = len(results)
    has_search = sum(1 for r in results if "<search>" in r["generated"])
    has_think = sum(1 for r in results if "<think>" in r["generated"])
    has_answer = sum(1 for r in results if "<answer>" in r["generated"])
    has_eos = sum(1 for r in results if "<|im_end|>" in r["generated"])

    logger.info("")
    logger.info("=" * 60)
    logger.info("FORMAT ANALYSIS (%d completions)", total)
    logger.info("=" * 60)
    logger.info("  <search> tags: %d/%d (%.0f%%)", has_search, total, 100 * has_search / total)
    logger.info("  <think> tags:  %d/%d (%.0f%%)", has_think, total, 100 * has_think / total)
    logger.info("  <answer> tags: %d/%d (%.0f%%)", has_answer, total, 100 * has_answer / total)
    logger.info("  <|im_end|>:   %d/%d (%.0f%%)", has_eos, total, 100 * has_eos / total)
    logger.info("")

    # Length stats
    lengths = [len(r["generated"]) for r in results]
    logger.info("  Length: min=%d, max=%d, mean=%.0f chars",
                min(lengths), max(lengths), sum(lengths) / len(lengths))

    logger.info("")
    logger.info("=" * 60)
    logger.info("SAMPLE OUTPUTS")
    logger.info("=" * 60)
    for r in results[:10]:
        logger.info("")
        logger.info("--- Q%d [%d-hop]: %s", r["question_idx"], r["hops"], r["question"][:80])
        logger.info("--- Gold: %s", r["gold_answer"])
        # Show first 500 chars of generation
        text = r["generated"][:500]
        logger.info("--- Generated:\n%s", text)
        if len(r["generated"]) > 500:
            logger.info("... [truncated, total %d chars]", len(r["generated"]))

    # Verdict
    logger.info("")
    logger.info("=" * 60)
    if has_search == 0:
        logger.info("VERDICT: MODEL NEVER GENERATES <search> TAGS")
        logger.info("  -> Chat template mismatch or SFT format not learned")
        logger.info("  -> GRPO cannot learn tool use from this starting point")
    elif has_search < total * 0.5:
        logger.info("VERDICT: MODEL SOMETIMES GENERATES <search> TAGS (%d%%)", 100 * has_search // total)
        logger.info("  -> SFT format partially learned, may work with GRPO")
    else:
        logger.info("VERDICT: MODEL RELIABLY GENERATES <search> TAGS (%d%%)", 100 * has_search // total)
        logger.info("  -> SFT format intact, GRPO should be able to build on this")
    logger.info("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose SFT-merged model format")
    parser.add_argument("--model_path", type=str, default="outputs/verl-sft-merged")
    parser.add_argument("--data_path", type=str, default="data/processed/conceptnet_qa_val.jsonl")
    parser.add_argument("--num_samples", type=int, default=20)
    parser.add_argument("--num_return_sequences", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    args = parser.parse_args()

    logger.info("=== SFT-Merged Model Diagnostic ===")
    logger.info("Model: %s", args.model_path)
    logger.info("Temperature: %.1f", args.temperature)
    logger.info("Samples: %d x %d sequences", args.num_samples, args.num_return_sequences)
    logger.info("")

    # Also test greedy (temperature=0) for comparison
    logger.info("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="sdpa",
        trust_remote_code=False,
    )
    model.eval()
    logger.info("Model loaded on %s", model.device)

    # Print the chat template to verify it's correct
    logger.info("")
    logger.info("=== Chat Template Preview ===")
    sample_messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "What is a dog?"},
    ]
    template_output = tokenizer.apply_chat_template(
        sample_messages, tokenize=False, add_generation_prompt=True
    )
    logger.info("%s", template_output)
    logger.info("=== End Template Preview ===")
    logger.info("")

    questions = load_test_questions(Path(args.data_path), args.num_samples)
    logger.info("Loaded %d questions", len(questions))

    # Test 1: With sampling (same as GRPO)
    logger.info("")
    logger.info("=== TEST 1: Sampling (temperature=%.1f) ===", args.temperature)
    results_sampling = generate_completions(
        model, tokenizer, questions,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        num_return_sequences=args.num_return_sequences,
    )
    analyze_results(results_sampling)

    # Test 2: Greedy (temperature=0)
    logger.info("")
    logger.info("=== TEST 2: Greedy (temperature=0) ===")
    results_greedy = generate_completions(
        model, tokenizer, questions,
        temperature=0.0,
        max_new_tokens=args.max_new_tokens,
        num_return_sequences=1,
    )
    analyze_results(results_greedy)


if __name__ == "__main__":
    main()
