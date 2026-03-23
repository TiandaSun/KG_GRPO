"""Offline evaluation of GRPO checkpoints on CWQ val/test.

Loads FSDP-sharded checkpoints, runs inference with KG server, computes EM/F1.
Supports evaluating multiple checkpoints in one run.

Usage:
    python scripts/eval_checkpoints.py \
        --checkpoint_dir checkpoints/kg-align-verl/grpo-cwq-7b-outcome-20260321 \
        --steps 50 250 500 750 1000 1250 \
        --eval_data data/freebase/verl_cwq/val.parquet \
        --output results/eval_outcome.json \
        --kg_server_url http://localhost:18901 \
        --max_samples 500
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)


def load_fsdp_checkpoint(
    checkpoint_dir: Path,
    step: int,
    base_model_path: str,
    device: str = "cuda",
) -> tuple[Any, Any]:
    """Load a verl FSDP checkpoint by consolidating shards.

    verl saves FSDP shards as model_world_size_N_rank_K.pt.
    We load all shards and merge state dicts.
    """
    step_dir = checkpoint_dir / f"global_step_{step}" / "actor"
    if not step_dir.exists():
        raise FileNotFoundError(f"Checkpoint not found: {step_dir}")

    # Check if huggingface format exists (verl saves model config there)
    hf_dir = step_dir / "huggingface"

    # Find all model shards
    shard_files = sorted(step_dir.glob("model_world_size_*_rank_*.pt"))
    if not shard_files:
        raise FileNotFoundError(f"No model shards in {step_dir}")

    world_size = len(shard_files)
    logger.info("Loading %d FSDP shards from step %d...", world_size, step)

    # Load all shards and collect local tensors per key
    # verl FSDP saves as DTensors with Shard(dim=0) placement
    # Each rank has 1/world_size of each parameter along dim 0
    shard_locals: dict[str, list[torch.Tensor]] = {}
    for shard_file in shard_files:
        shard = torch.load(shard_file, map_location="cpu", weights_only=False)
        for key, value in shard.items():
            # Extract local tensor from DTensor
            if hasattr(value, 'to_local'):
                local = value.to_local()
            else:
                local = value
            if key not in shard_locals:
                shard_locals[key] = []
            shard_locals[key].append(local)

    # Concatenate shards along dim 0 to reconstruct full parameters
    merged_state = {}
    for key, locals_list in shard_locals.items():
        if len(locals_list) == 1:
            merged_state[key] = locals_list[0]
        else:
            # All shards have same shape -> sharded along dim 0
            merged_state[key] = torch.cat(locals_list, dim=0)

    logger.info("Reconstructed %d parameters from %d shards", len(merged_state), world_size)

    # Load tokenizer from base model
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load base model and inject weights
    logger.info("Loading base model architecture from %s...", base_model_path)
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
    )

    # Try to load state dict (may need key remapping)
    try:
        model.load_state_dict(merged_state, strict=False)
        logger.info("Loaded checkpoint weights (strict=False)")
    except Exception as e:
        logger.warning("Could not load state dict directly: %s", e)
        # Try with prefix remapping
        remapped = {}
        for k, v in merged_state.items():
            new_key = k.replace("_fsdp_wrapped_module.", "").replace("_checkpoint_wrapped_module.", "")
            remapped[new_key] = v
        model.load_state_dict(remapped, strict=False)
        logger.info("Loaded checkpoint weights with key remapping")

    model = model.to(device).eval()
    return model, tokenizer


def generate_answer(
    model: Any,
    tokenizer: Any,
    question: str,
    system_prompt: str,
    max_new_tokens: int = 512,
    device: str = "cuda",
) -> str:
    """Generate answer for a single question (greedy, no tool calls)."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tokenizer.pad_token_id,
        )

    generated = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True)


def extract_answer(text: str) -> str:
    """Extract answer from <answer> tags."""
    if "<answer>" in text and "</answer>" in text:
        return text.split("<answer>")[1].split("</answer>")[0].strip()
    if "</think>" in text:
        return text.split("</think>")[-1].strip()
    return text.strip()


def normalize(text: str) -> str:
    """Lowercase, strip punctuation."""
    text = str(text).lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return " ".join(text.split())


def exact_match(predicted: str, gold: str, aliases: list | None = None) -> float:
    """Check EM against gold and aliases."""
    pred_norm = normalize(predicted)
    if not pred_norm:
        return 0.0
    if pred_norm == normalize(gold):
        return 1.0
    if aliases:
        for alias in aliases:
            if pred_norm == normalize(str(alias)):
                return 1.0
    return 0.0


def contains_match(full_output: str, gold: str, aliases: list | None = None) -> float:
    """Check if gold answer appears anywhere in the full output (lenient EM)."""
    output_norm = normalize(full_output)
    if not output_norm:
        return 0.0
    if normalize(gold) in output_norm:
        return 1.0
    if aliases:
        for alias in aliases:
            if normalize(str(alias)) in output_norm:
                return 1.0
    return 0.0


def token_f1(predicted: str, gold: str) -> float:
    """Token-level F1."""
    pred_tokens = normalize(predicted).split()
    gt_tokens = normalize(gold).split()
    if not gt_tokens:
        return 1.0 if not pred_tokens else 0.0
    if not pred_tokens:
        return 0.0
    pred_counts = Counter(pred_tokens)
    gt_counts = Counter(gt_tokens)
    common = sum((pred_counts & gt_counts).values())
    if common == 0:
        return 0.0
    precision = common / len(pred_tokens)
    recall = common / len(gt_tokens)
    return 2 * precision * recall / (precision + recall)


def evaluate_checkpoint(
    model: Any,
    tokenizer: Any,
    eval_df: pd.DataFrame,
    max_samples: int = 500,
) -> dict[str, float]:
    """Evaluate a model on CWQ data, return metrics."""
    system_prompt = eval_df.iloc[0]["prompt"][0]["content"]

    ems, f1s, cms = [], [], []
    samples_evaluated = 0

    for idx, row in eval_df.iterrows():
        if samples_evaluated >= max_samples:
            break

        question = row["prompt"][1]["content"]
        gold = row["reward_model"]["ground_truth"]
        extra = row["extra_info"]
        all_answers = extra.get("all_answers", [gold])
        if hasattr(all_answers, 'tolist'):
            all_answers = all_answers.tolist()

        response = generate_answer(model, tokenizer, question, system_prompt)
        predicted = extract_answer(response)

        em = exact_match(predicted, gold, all_answers)
        f1 = token_f1(predicted, gold)
        cm = contains_match(response, gold, all_answers)
        ems.append(em)
        f1s.append(f1)
        cms.append(cm)
        samples_evaluated += 1

        if samples_evaluated % 50 == 0:
            logger.info("  Evaluated %d/%d samples, EM=%.3f, F1=%.3f, Contains=%.3f",
                        samples_evaluated, min(max_samples, len(eval_df)),
                        sum(ems) / len(ems), sum(f1s) / len(f1s), sum(cms) / len(cms))

    return {
        "em": sum(ems) / len(ems) if ems else 0.0,
        "f1": sum(f1s) / len(f1s) if f1s else 0.0,
        "contains_em": sum(cms) / len(cms) if cms else 0.0,
        "n_samples": len(ems),
    }


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    parser = argparse.ArgumentParser(description="Evaluate GRPO checkpoints on CWQ.")
    parser.add_argument("--checkpoint_dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, nargs="+", default=[50, 250, 500, 750, 1000, 1250])
    parser.add_argument("--eval_data", type=Path, default=Path("data/freebase/verl_cwq/val.parquet"))
    parser.add_argument("--base_model", type=str, default="outputs/verl-sft-cwq-7b-merged")
    parser.add_argument("--output", type=Path, default=Path("results/eval.json"))
    parser.add_argument("--max_samples", type=int, default=500)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    # Load eval data
    logger.info("Loading eval data from %s", args.eval_data)
    eval_df = pd.read_parquet(args.eval_data)
    logger.info("Loaded %d eval samples (will use %d)", len(eval_df), args.max_samples)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    results = {}

    for step in args.steps:
        step_dir = args.checkpoint_dir / f"global_step_{step}"
        if not step_dir.exists():
            logger.warning("Step %d not found, skipping", step)
            continue

        logger.info("=== Evaluating step %d ===", step)
        start = time.time()

        try:
            model, tokenizer = load_fsdp_checkpoint(
                args.checkpoint_dir, step, args.base_model, args.device
            )
            metrics = evaluate_checkpoint(model, tokenizer, eval_df, args.max_samples)
            elapsed = time.time() - start

            results[str(step)] = {**metrics, "elapsed_s": elapsed}
            logger.info("Step %d: EM=%.4f, F1=%.4f (%d samples, %.0fs)",
                        step, metrics["em"], metrics["f1"], metrics["n_samples"], elapsed)

            # Free GPU memory
            del model
            torch.cuda.empty_cache()

        except Exception as e:
            logger.error("Failed to evaluate step %d: %s", step, e)
            results[str(step)] = {"error": str(e)}

    # Also eval the SFT base model (step 0)
    logger.info("=== Evaluating SFT base model (step 0) ===")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.base_model)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model, torch_dtype=torch.bfloat16, attn_implementation="eager"
        ).to(args.device).eval()
        metrics = evaluate_checkpoint(model, tokenizer, eval_df, args.max_samples)
        results["0"] = metrics
        logger.info("SFT base: EM=%.4f, F1=%.4f", metrics["em"], metrics["f1"])
        del model
        torch.cuda.empty_cache()
    except Exception as e:
        logger.error("Failed to eval SFT base: %s", e)

    # Save results
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Results saved to %s", args.output)

    # Print summary table
    print("\n=== Summary ===")
    print(f"{'Step':>6} {'EM':>8} {'ContEM':>8} {'F1':>8} {'Samples':>8}")
    print("-" * 44)
    for step in sorted(results.keys(), key=lambda x: int(x)):
        r = results[step]
        if "error" in r:
            print(f"{step:>6} {'ERROR':>8}")
        else:
            print(f"{step:>6} {r['em']:>8.4f} {r.get('contains_em',0):>8.4f} {r['f1']:>8.4f} {r['n_samples']:>8}")


if __name__ == "__main__":
    main()
