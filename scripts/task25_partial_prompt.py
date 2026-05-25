"""Task 25: Partial-Prompt Completion Test for contamination quantification.

Following "Reasoning or Memorization?" (arXiv:2507.10532):
- Truncate each question to first 50% of tokens
- Let model complete via greedy decoding
- Check if completion contains the gold answer

High completion rate = model has memorized the dataset.

Usage:
    python scripts/task25_partial_prompt.py \
        --models Qwen/Qwen2.5-7B-Instruct meta-llama/Llama-3.1-8B-Instruct \
        --eval_data data/freebase/verl_cwq/test.parquet \
        --output results/task25_partial_prompt.json
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import time
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)


def normalize(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"[^\w\s]", " ", text)
    return " ".join(text.split())


def contains_em(completion: str, gold: str, aliases: list[str] | None = None) -> bool:
    """Check if completion contains the gold answer (normalized)."""
    comp_norm = normalize(completion)
    if not comp_norm:
        return False
    if normalize(gold) in comp_norm:
        return True
    if aliases:
        for a in aliases:
            if normalize(str(a)) in comp_norm:
                return True
    return False


def evaluate_model(
    model_name: str,
    questions: list[dict],
    truncation_ratio: float = 0.5,
    max_new_tokens: int = 128,
    device: str = "cuda",
) -> dict:
    """Run partial-prompt completion test on a single model."""
    logger.info("Loading model: %s", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
        trust_remote_code=True,
    ).to(device).eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    n_correct = 0
    n_total = len(questions)
    t0 = time.time()

    for i, q in enumerate(questions):
        question_text = q["question"]

        # Tokenize the question
        tokens = tokenizer.encode(question_text, add_special_tokens=False)
        n_keep = max(1, int(len(tokens) * truncation_ratio))
        truncated_tokens = tokens[:n_keep]

        # Decode back to text for the partial prompt
        partial_question = tokenizer.decode(truncated_tokens, skip_special_tokens=True)

        # Generate completion (greedy)
        input_ids = tokenizer.encode(partial_question, return_tensors="pt").to(device)
        with torch.no_grad():
            output = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        completion = tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=True)

        # Check if completion contains the gold answer
        if contains_em(completion, q["gold"], q["all_answers"]):
            n_correct += 1

        if (i + 1) % 200 == 0:
            rate = n_correct / (i + 1)
            elapsed = time.time() - t0
            logger.info(
                "  %d/%d completion_rate=%.3f (%.0fs elapsed, %.1fs/sample)",
                i + 1, n_total, rate, elapsed, elapsed / (i + 1),
            )

    elapsed = time.time() - t0
    completion_rate = n_correct / n_total

    logger.info(
        "%s: completion_rate=%.4f (%d/%d) in %.0fs",
        model_name, completion_rate, n_correct, n_total, elapsed,
    )

    del model
    torch.cuda.empty_cache()

    return {
        "model": model_name,
        "completion_rate": completion_rate,
        "n_correct": n_correct,
        "n_total": n_total,
        "truncation_ratio": truncation_ratio,
        "max_new_tokens": max_new_tokens,
        "elapsed_s": elapsed,
    }


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    parser = argparse.ArgumentParser(description="Partial-prompt completion test for contamination.")
    parser.add_argument(
        "--models", nargs="+",
        default=["Qwen/Qwen2.5-7B-Instruct", "meta-llama/Llama-3.1-8B-Instruct"],
        help="Model names to test",
    )
    parser.add_argument("--eval_data", type=Path, default=Path("data/freebase/verl_cwq/test.parquet"))
    parser.add_argument("--truncation_ratio", type=float, default=0.5)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--max_samples", type=int, default=0, help="0 = all samples")
    parser.add_argument("--output", type=Path, default=Path("results/task25_partial_prompt.json"))
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    # Load eval data
    df = pd.read_parquet(args.eval_data)
    questions = []
    for i, (_, row) in enumerate(df.iterrows()):
        if args.max_samples > 0 and i >= args.max_samples:
            break
        gold = row["reward_model"]["ground_truth"]
        all_answers = row["extra_info"].get("all_answers", [gold])
        if hasattr(all_answers, "tolist"):
            all_answers = all_answers.tolist()
        questions.append({
            "question": row["prompt"][1]["content"],
            "gold": str(gold),
            "all_answers": [str(a) for a in all_answers],
            "sample_id": row["extra_info"].get("sample_id", str(i)),
        })

    logger.info("Loaded %d questions from %s", len(questions), args.eval_data)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    results = {}

    for model_name in args.models:
        logger.info("=== Testing %s ===", model_name)
        result = evaluate_model(
            model_name,
            questions,
            truncation_ratio=args.truncation_ratio,
            max_new_tokens=args.max_new_tokens,
            device=args.device,
        )
        # Use short name as key
        short_name = model_name.split("/")[-1].lower()
        results[short_name] = result

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Results saved to %s", args.output)

    # Summary
    print("\n=== Partial-Prompt Completion Test ===")
    print(f"Truncation ratio: {args.truncation_ratio}")
    print(f"Questions: {len(questions)}")
    print(f"{'Model':<30} {'Completion Rate':>16} {'Correct':>8} {'Total':>8}")
    print("-" * 70)
    for name, res in results.items():
        print(f"{name:<30} {res['completion_rate']:>16.4f} {res['n_correct']:>8} {res['n_total']:>8}")


if __name__ == "__main__":
    main()
