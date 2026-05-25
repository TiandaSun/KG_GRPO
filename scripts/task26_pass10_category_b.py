"""Task 26: Pass@10 Category B Identification.

Identify questions the SFT model genuinely cannot answer from parametric memory.
For each question, generate 10 responses (temperature=0.7, NO tools) and check EM.

Category A: pass@10 > 0 (model can answer at least once)
Category B: pass@10 = 0 (model NEVER gets it right in 10 tries)

Usage:
    python scripts/task26_pass10_category_b.py \
        --model outputs/verl-sft-cwq-7b-merged \
        --eval_data data/freebase/verl_cwq/test.parquet \
        --output results/task26_category_b.json
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)


def normalize(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"[^\w\s]", " ", text)
    return " ".join(text.split())


def exact_match(predicted: str, gold: str, aliases: list[str] | None = None) -> float:
    pred_norm = normalize(predicted)
    if not pred_norm:
        return 0.0
    if pred_norm == normalize(gold):
        return 1.0
    if aliases:
        for a in aliases:
            if pred_norm == normalize(str(a)):
                return 1.0
    return 0.0


def extract_answer(text: str) -> str:
    if "<answer>" in text and "</answer>" in text:
        return text.split("<answer>")[1].split("</answer>")[0].strip()
    if "</think>" in text:
        return text.split("</think>")[-1].strip()
    return text.strip()


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k estimator."""
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


DIRECT_ANSWER_SYSTEM_PROMPT = (
    "You are a helpful knowledge assistant. Answer the user's question directly "
    "with just the answer entity (e.g., a person's name, place, or short phrase). "
    "Do not explain your reasoning. Wrap your final answer in <answer>...</answer> tags."
)


def extract_answer_relaxed(text: str) -> str:
    """Extract answer more robustly — try <answer> tags first, then last line, then full text."""
    if "<answer>" in text and "</answer>" in text:
        return text.split("<answer>")[1].split("</answer>")[0].strip()
    # Try to find common patterns like "The answer is X" or "Answer: X"
    m = re.search(r"(?:the answer is|answer:|answer is)\s*:?\s*([^\n.]+)", text, re.IGNORECASE)
    if m:
        return m.group(1).strip().rstrip(".")
    # Fallback: last non-empty line
    lines = [line.strip() for line in text.strip().split("\n") if line.strip()]
    if lines:
        return lines[-1]
    return text.strip()


def generate_single_turn(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    question: str,
    system_prompt: str,
    temperature: float = 0.7,
    max_new_tokens: int = 128,
    device: str = "cuda",
) -> str:
    """Single-turn generation WITHOUT tools. Pure parametric memory test."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048).to(device)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=0.95,
            pad_token_id=tokenizer.pad_token_id,
        )
    response = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return response


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    parser = argparse.ArgumentParser(description="Pass@10 Category B identification.")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct",
                        help="Model for parametric memory test. Default: base Qwen (not SFT, since SFT was trained for tool use)")
    parser.add_argument("--system_prompt", type=str, default=DIRECT_ANSWER_SYSTEM_PROMPT,
                        help="System prompt. Default: direct-answer prompt (NOT the tool-use prompt)")
    parser.add_argument("--eval_data", type=Path, default=Path("data/freebase/verl_cwq/test.parquet"))
    parser.add_argument("--k_samples", type=int, default=10, help="Samples per question")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max_samples", type=int, default=0, help="0 = all samples")
    parser.add_argument("--output", type=Path, default=Path("results/task26_category_b.json"))
    parser.add_argument("--device", type=str, default="cuda")
    # For parallel splitting across jobs
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=-1, help="-1 = all")
    args = parser.parse_args()

    # Load data
    df = pd.read_parquet(args.eval_data)
    all_questions = []
    for i, (_, row) in enumerate(df.iterrows()):
        if args.max_samples > 0 and i >= args.max_samples:
            break
        gold = row["reward_model"]["ground_truth"]
        all_answers = row["extra_info"].get("all_answers", [gold])
        if hasattr(all_answers, "tolist"):
            all_answers = all_answers.tolist()
        all_questions.append({
            "question": row["prompt"][1]["content"],
            "gold": str(gold),
            "all_answers": [str(a) for a in all_answers],
            "sample_id": row["extra_info"].get("sample_id", str(i)),
        })

    # Apply index range
    end_idx = args.end_idx if args.end_idx > 0 else len(all_questions)
    questions = all_questions[args.start_idx:end_idx]
    logger.info(
        "Loaded %d questions (indices %d-%d) from %s",
        len(questions), args.start_idx, end_idx, args.eval_data,
    )

    # Load model
    logger.info("Loading model: %s", args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
        trust_remote_code=True,
    ).to(args.device).eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Use direct-answer prompt (NOT the tool-use prompt from data)
    system_prompt = args.system_prompt
    logger.info("Using system prompt: %s", system_prompt[:100])

    # Evaluate
    per_question = []
    category_a_ids = []
    category_b_ids = []
    t0 = time.time()

    for qi, q in enumerate(questions):
        correct_count = 0
        for si in range(args.k_samples):
            response = generate_single_turn(
                model, tokenizer, q["question"], system_prompt,
                temperature=args.temperature, device=args.device,
            )
            predicted = extract_answer_relaxed(response)
            # Use Contains-EM (also checks aliases): answer appears in prediction
            is_correct = exact_match(predicted, q["gold"], q["all_answers"]) > 0
            if not is_correct:
                # Fallback: does the gold answer string appear in prediction?
                pred_norm = normalize(predicted)
                if normalize(q["gold"]) and normalize(q["gold"]) in pred_norm:
                    is_correct = True
                else:
                    for a in q["all_answers"]:
                        a_norm = normalize(str(a))
                        if a_norm and a_norm in pred_norm:
                            is_correct = True
                            break
            if is_correct:
                correct_count += 1

        p10 = pass_at_k(args.k_samples, correct_count, min(10, args.k_samples))

        per_question.append({
            "sample_id": q["sample_id"],
            "question": q["question"],
            "gold": q["gold"],
            "correct_count": correct_count,
            "pass@10": p10,
            "category": "A" if correct_count > 0 else "B",
        })

        if correct_count > 0:
            category_a_ids.append(q["sample_id"])
        else:
            category_b_ids.append(q["sample_id"])

        if (qi + 1) % 100 == 0:
            elapsed = time.time() - t0
            n_a = len(category_a_ids)
            n_b = len(category_b_ids)
            logger.info(
                "  %d/%d  Cat-A=%d (%.1f%%)  Cat-B=%d (%.1f%%)  %.1fs/q  ETA=%.0fm",
                qi + 1, len(questions),
                n_a, 100 * n_a / (qi + 1),
                n_b, 100 * n_b / (qi + 1),
                elapsed / (qi + 1),
                (elapsed / (qi + 1)) * (len(questions) - qi - 1) / 60,
            )

    elapsed = time.time() - t0

    # Summary
    results = {
        "model": args.model,
        "k_samples": args.k_samples,
        "temperature": args.temperature,
        "n_total": len(questions),
        "category_a_count": len(category_a_ids),
        "category_b_count": len(category_b_ids),
        "category_a_pct": len(category_a_ids) / len(questions),
        "category_b_pct": len(category_b_ids) / len(questions),
        "category_a_ids": category_a_ids,
        "category_b_ids": category_b_ids,
        "per_question": per_question,
        "elapsed_s": elapsed,
        "start_idx": args.start_idx,
        "end_idx": end_idx,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Results saved to %s", args.output)

    print(f"\n=== Category B Identification ===")
    print(f"Model: {args.model}")
    print(f"Questions: {len(questions)}, k={args.k_samples}")
    print(f"Category A (pass@10 > 0): {len(category_a_ids)} ({100*len(category_a_ids)/len(questions):.1f}%)")
    print(f"Category B (pass@10 = 0): {len(category_b_ids)} ({100*len(category_b_ids)/len(questions):.1f}%)")
    print(f"Time: {elapsed:.0f}s ({elapsed/len(questions):.1f}s/question)")

    # Pass@k statistics
    correct_counts = [pq["correct_count"] for pq in per_question]
    for k in [1, 5, 10]:
        if k <= args.k_samples:
            scores = [pass_at_k(args.k_samples, c, k) for c in correct_counts]
            print(f"  pass@{k} (mean): {np.mean(scores):.4f}")


if __name__ == "__main__":
    main()
