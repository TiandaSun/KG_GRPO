"""Phase 7 Task 42 (v11 trimmed): Self-consistency@5 on full 3531 test set.

For each question:
  1. Generate k samples at temperature=0.7 with tools (max_turns=5)
  2. Extract the final <answer> from each sample
  3. Normalize answers and take majority vote
  4. Compare majority vote EM vs greedy EM

Also tracks:
  - Whether the majority-vote answer came from a trajectory with ≥1 tool call
  - Number of distinct answers per question (diversity)
  - CvT@5: fraction of questions where majority answer came from a tool-using
    trajectory AND was correct (proxy for "tools contributed")

v11 paper use:
  - If SC@5 lifts 39B by +2-4pp, goes in small table as "inference-time augmentation"
  - If SC@5 lifts E3 and 39B by similar amounts → same underlying distribution
  - If SC@5 doesn't help → systematic failures (not sampling noise) → diagnostic support

Usage:
  python scripts/phase7_task42_self_consistency.py \\
      --checkpoint_dir /.../grpo-cwq-7b-39b-kl-20260413 \\
      --step 400 \\
      --base_model outputs/verl-sft-cwq-7b-merged \\
      --kg_server_url http://localhost:18901 \\
      --output results/phase7/sc5_39b_step400_full.json \\
      --k 5 \\
      --max_samples 0
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
import requests
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Reuse the existing utilities from scripts/
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from task17_pass_at_k import (  # noqa: E402
    exact_match,
    extract_answer,
    generate_with_tools_sampled,
    load_checkpoint,
    normalize,
)

logger = logging.getLogger(__name__)


def majority_vote(answers: list[str]) -> tuple[str, int, int]:
    """Return (winning answer, count, distinct_count).

    Uses normalized strings for voting but returns the FIRST original answer
    that produced the winning normalized form (preserves casing).
    """
    if not answers:
        return "", 0, 0
    norm_to_orig: dict[str, str] = {}
    counts: Counter[str] = Counter()
    for a in answers:
        n = normalize(a) if a else ""
        if not n:
            continue
        counts[n] += 1
        norm_to_orig.setdefault(n, a)
    if not counts:
        return "", 0, 0
    winner_norm, winner_count = counts.most_common(1)[0]
    return norm_to_orig[winner_norm], winner_count, len(counts)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    parser = argparse.ArgumentParser(description="Phase 7 Task 42: self-consistency@k")
    parser.add_argument("--checkpoint_dir", type=Path, required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--base_model", type=str, default="outputs/verl-sft-cwq-7b-merged")
    parser.add_argument("--eval_data", type=Path, default=Path("data/freebase/verl_cwq/test.parquet"))
    parser.add_argument("--kg_server_url", type=str, default="http://localhost:18901")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max_turns", type=int, default=5)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--max_samples", type=int, default=0, help="0 = all test samples")
    parser.add_argument("--filter_ids", type=Path, default=None,
                        help="Optional JSON file with filter_ids/category_b_ids to subset")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save_every", type=int, default=50)
    args = parser.parse_args()

    # Verify KG server
    try:
        r = requests.get(f"{args.kg_server_url}/health", timeout=5)
        logger.info("KG server healthy: %s", r.status_code)
    except Exception as e:
        logger.error("KG server not reachable: %s", e)
        return

    # Load eval data
    df = pd.read_parquet(args.eval_data)
    system_prompt = df.iloc[0]["prompt"][0]["content"]

    # Optional filter
    filter_id_set = None
    if args.filter_ids:
        with open(args.filter_ids) as f:
            filter_data = json.load(f)
        filter_id_set = set(filter_data.get("filter_ids", filter_data.get("category_b_ids", [])))
        logger.info("Filtering to %d question IDs", len(filter_id_set))

    questions: list[dict] = []
    for i, (_, row) in enumerate(df.iterrows()):
        sid = str(row["extra_info"].get("sample_id", i))
        if filter_id_set is not None and sid not in filter_id_set:
            continue
        gold = row["reward_model"]["ground_truth"]
        all_answers = row["extra_info"].get("all_answers", [gold])
        if hasattr(all_answers, "tolist"):
            all_answers = all_answers.tolist()
        questions.append(
            {
                "sample_id": sid,
                "question": row["prompt"][1]["content"],
                "gold": gold,
                "all_answers": [str(a) for a in all_answers],
            }
        )
    if args.max_samples > 0:
        questions = questions[: args.max_samples]
    logger.info("Selected %d questions", len(questions))

    # Load tokenizer + checkpoint
    logger.info("Loading tokenizer from %s", args.base_model)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info("Loading checkpoint %s step %d", args.checkpoint_dir, args.step)
    model = load_checkpoint(args.checkpoint_dir, args.step, args.base_model, args.device)

    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Resume support: skip already-processed samples
    already_done: dict[str, dict] = {}
    if args.output.exists():
        try:
            prev = json.load(open(args.output))
            if isinstance(prev.get("per_sample"), list):
                for entry in prev["per_sample"]:
                    already_done[str(entry["sample_id"])] = entry
            logger.info("Resume: %d samples already processed", len(already_done))
        except Exception:
            pass

    per_sample: list[dict] = []
    sc_correct = 0
    any_correct = 0  # pass@k style — did ANY sample get it right
    greedy_equivalent = 0  # did SC agree with the most-voted answer
    em_sum = 0
    t_start = time.time()

    for qi, q in enumerate(questions):
        if q["sample_id"] in already_done:
            entry = already_done[q["sample_id"]]
            per_sample.append(entry)
            sc_correct += entry.get("sc_em", 0)
            any_correct += entry.get("any_em", 0)
            continue

        samples: list[dict] = []
        for si in range(args.k):
            full_response, ntools = generate_with_tools_sampled(
                model, tokenizer, q["question"], system_prompt,
                args.kg_server_url,
                max_turns=args.max_turns,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                device=args.device,
                use_tools=True,
            )
            predicted = extract_answer(full_response)
            em = exact_match(predicted, q["gold"], q["all_answers"])
            samples.append(
                {
                    "predicted": predicted,
                    "em": em,
                    "num_tool_calls": ntools,
                }
            )

        predictions = [s["predicted"] for s in samples]
        winner, winner_count, distinct = majority_vote(predictions)
        sc_em = exact_match(winner, q["gold"], q["all_answers"])
        any_em = max(s["em"] for s in samples)
        mean_tools = sum(s["num_tool_calls"] for s in samples) / len(samples)

        sc_correct += sc_em
        any_correct += any_em
        em_sum += sum(s["em"] for s in samples) / len(samples)  # average per-sample EM

        entry = {
            "sample_id": q["sample_id"],
            "gold": q["gold"],
            "sc_em": sc_em,
            "any_em": any_em,
            "winning_answer": winner,
            "winner_count": winner_count,
            "distinct_answers": distinct,
            "mean_tool_calls": mean_tools,
            "samples": samples,
        }
        per_sample.append(entry)

        if (qi + 1) % args.save_every == 0 or qi + 1 == len(questions):
            n = qi + 1
            sc_rate = sc_correct / n
            any_rate = any_correct / n
            elapsed = time.time() - t_start
            logger.info(
                "  %d/%d  SC@%d EM=%.3f  pass@%d(any)=%.3f  tools=%.2f  elapsed=%ds",
                n, len(questions), args.k, sc_rate, args.k, any_rate, mean_tools, int(elapsed),
            )
            # Incremental save
            summary = {
                "n": n,
                "k": args.k,
                "sc_em": sc_rate,
                "pass_at_k_any": any_rate,
                "mean_per_sample_em": em_sum / n,
                "elapsed_s": elapsed,
            }
            with open(args.output, "w") as f:
                json.dump({"summary": summary, "per_sample": per_sample}, f, indent=2)

    n = len(questions)
    summary = {
        "n": n,
        "k": args.k,
        "sc_em": sc_correct / n if n else 0.0,
        "pass_at_k_any": any_correct / n if n else 0.0,
        "mean_per_sample_em": em_sum / n if n else 0.0,
        "elapsed_s": time.time() - t_start,
        "checkpoint": str(args.checkpoint_dir),
        "step": args.step,
    }
    with open(args.output, "w") as f:
        json.dump({"summary": summary, "per_sample": per_sample}, f, indent=2)

    logger.info(
        "FINAL SC@%d EM: %.4f  pass@%d(any): %.4f  on n=%d",
        args.k, summary["sc_em"], args.k, summary["pass_at_k_any"], n,
    )


if __name__ == "__main__":
    main()
