"""Pass@k analysis: distinguish capability expansion vs distribution sharpening.

Generates n samples per question (with temperature>0) for both the SFT base model
and the best GRPO checkpoint (E3@500), then computes pass@k for k=1,4,8,16,32.

If pass@1(E3) >> pass@1(SFT) but pass@32(E3) ≈ pass@32(SFT):
    → distribution sharpening (no new capability)
If pass@32(E3) >> pass@32(SFT):
    → genuine capability expansion
If pass@32(SFT) >> pass@32(E3):
    → capability narrowing (RL reduced diversity)

Reference: "Does RL Really Incentivize Reasoning Capacity in LLMs Beyond
the Base Model?" (arXiv:2504.13837, ICLR 2026)

Usage:
    # Start KG server first:
    python -m src_verl.kg_server.server --kg freebase --freebase_dir data/freebase/kg --port 18901 &

    python scripts/task17_pass_at_k.py \
        --checkpoint_dir checkpoints/kg-align-verl/grpo-cwq-7b-verifiable-20260321 \
        --step 500 \
        --n_samples 32 \
        --n_questions 200 \
        --output results/task17_pass_at_k.json
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)


def normalize(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"[^\w\s]", " ", text)
    return " ".join(text.split())


def exact_match(predicted: str, gold: str, aliases: list | None = None) -> float:
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


def call_kg_server(action: str, entity: str, relation: str | None, server_url: str) -> str:
    payload = {"action": action, "entity": entity}
    if relation:
        payload["relation"] = relation
    try:
        resp = requests.post(f"{server_url}/retrieve", json=payload, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            results = data.get("results", data.get("result", []))
            if isinstance(results, list) and len(results) > 10:
                results = results[:10]
            return json.dumps(results, ensure_ascii=False)
        return f"Error: {resp.status_code}"
    except Exception as e:
        return f"Error: {e}"


def parse_search_call(text: str) -> tuple[str, str, str | None] | None:
    match = re.search(r"<search>\s*(\w+)\(([^)]*)\)\s*</search>", text, re.DOTALL)
    if not match:
        return None
    action = match.group(1)
    args = [a.strip().strip("'\"") for a in match.group(2).split(",") if a.strip()]
    entity = args[0] if args else ""
    relation = args[1] if len(args) > 1 else None
    return action, entity, relation


def generate_with_tools_sampled(
    model: Any,
    tokenizer: Any,
    question: str,
    system_prompt: str,
    kg_server_url: str,
    max_turns: int = 5,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    device: str = "cuda",
) -> tuple[str, int]:
    """Multi-turn generation with sampling (temperature>0) for pass@k."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]
    num_tool_calls = 0
    full_response = ""

    for turn in range(max_turns):
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=4096).to(device)

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
        full_response += response

        if "<answer>" in response and "</answer>" in response:
            messages.append({"role": "assistant", "content": response})
            break

        call = parse_search_call(response)
        if call is None:
            messages.append({"role": "assistant", "content": response})
            break

        action, entity, relation = call
        num_tool_calls += 1
        kg_result = call_kg_server(action, entity, relation, kg_server_url)
        messages.append({"role": "assistant", "content": response})
        messages.append({"role": "user", "content": f"<tool_response>{kg_result}</tool_response>"})
        full_response += f"\n<tool_response>{kg_result}</tool_response>\n"

    return full_response, num_tool_calls


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k estimator. n=total samples, c=correct samples, k=k."""
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def load_checkpoint(ckpt_dir: Path, step: int, base_model: str, device: str = "cuda"):
    """Load FSDP checkpoint by concatenating shards."""
    step_dir = ckpt_dir / f"global_step_{step}" / "actor"
    shards = sorted(step_dir.glob("model_world_size_*_rank_*.pt"))
    if not shards:
        raise FileNotFoundError(f"No shards in {step_dir}")

    shard_locals: dict[str, list] = {}
    for sf in shards:
        s = torch.load(sf, map_location="cpu", weights_only=False)
        for k, v in s.items():
            local = v.to_local() if hasattr(v, "to_local") else v
            shard_locals.setdefault(k, []).append(local)

    merged = {k: torch.cat(vs, dim=0) if len(vs) > 1 else vs[0] for k, vs in shard_locals.items()}
    model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=torch.bfloat16, attn_implementation="eager")
    model.load_state_dict(merged, strict=False)
    return model.to(device).eval()


def evaluate_model_pass_at_k(
    model: Any,
    tokenizer: Any,
    questions: list[dict],
    system_prompt: str,
    kg_server_url: str,
    n_samples: int = 32,
    temperature: float = 0.7,
    device: str = "cuda",
) -> dict:
    """Generate n_samples per question and compute pass@k."""
    k_values = [1, 4, 8, 16, 32]
    k_values = [k for k in k_values if k <= n_samples]

    per_question_correct = []

    for qi, q in enumerate(questions):
        question = q["question"]
        gold = q["gold"]
        all_answers = q["all_answers"]

        correct_count = 0
        for si in range(n_samples):
            full_response, _ = generate_with_tools_sampled(
                model, tokenizer, question, system_prompt,
                kg_server_url, temperature=temperature, device=device,
            )
            predicted = extract_answer(full_response)
            if exact_match(predicted, gold, all_answers) > 0:
                correct_count += 1

        per_question_correct.append(correct_count)

        if (qi + 1) % 20 == 0:
            # Running pass@1 estimate
            running_pass1 = np.mean([pass_at_k(n_samples, c, 1) for c in per_question_correct])
            logger.info(
                "  %d/%d questions, running pass@1=%.3f",
                qi + 1, len(questions), running_pass1,
            )

    # Compute pass@k for each k
    results = {}
    for k in k_values:
        scores = [pass_at_k(n_samples, c, k) for c in per_question_correct]
        results[f"pass@{k}"] = float(np.mean(scores))

    results["per_question_correct"] = per_question_correct
    results["n_samples"] = n_samples
    results["n_questions"] = len(questions)

    return results


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    parser = argparse.ArgumentParser(description="Pass@k analysis for capability expansion.")
    parser.add_argument("--checkpoint_dir", type=Path, required=True)
    parser.add_argument("--step", type=int, default=500)
    parser.add_argument("--base_model", type=str, default="outputs/verl-sft-cwq-7b-merged")
    parser.add_argument("--eval_data", type=Path, default=Path("data/freebase/verl_cwq/val.parquet"))
    parser.add_argument("--kg_server_url", type=str, default="http://localhost:18901")
    parser.add_argument("--n_samples", type=int, default=32, help="Samples per question (n)")
    parser.add_argument("--n_questions", type=int, default=200, help="Number of questions to evaluate")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--output", type=Path, default=Path("results/task17_pass_at_k.json"))
    parser.add_argument("--device", type=str, default="cuda")
    # Optionally also evaluate E2
    parser.add_argument("--e2_checkpoint_dir", type=Path, default=None)
    parser.add_argument("--e2_step", type=int, default=1200)
    # Parallel splitting: process a subset of questions
    parser.add_argument("--start_idx", type=int, default=0, help="Start question index (for parallel splitting)")
    parser.add_argument("--end_idx", type=int, default=-1, help="End question index (-1 = all)")
    # Which models to evaluate (for splitting SFT vs E3 across jobs)
    parser.add_argument("--models", type=str, nargs="+", default=["sft", "e3", "e2"],
                        help="Which models to evaluate: sft, e3, e2")
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
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    system_prompt = df.iloc[0]["prompt"][0]["content"]

    # Prepare questions (subset based on start_idx/end_idx)
    all_questions = []
    for i, (_, row) in enumerate(df.iterrows()):
        if i >= args.n_questions:
            break
        gold = row["reward_model"]["ground_truth"]
        all_answers = row["extra_info"].get("all_answers", [gold])
        if hasattr(all_answers, "tolist"):
            all_answers = all_answers.tolist()
        all_questions.append({
            "question": row["prompt"][1]["content"],
            "gold": gold,
            "all_answers": [str(a) for a in all_answers],
        })

    end_idx = args.end_idx if args.end_idx > 0 else len(all_questions)
    questions = all_questions[args.start_idx:end_idx]
    logger.info("Evaluating questions %d-%d (%d total)", args.start_idx, end_idx, len(questions))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    all_results = {}

    # --- Evaluate selected models ---
    if "sft" in args.models:
        logger.info("=== Evaluating SFT base (n=%d, %d questions) ===", args.n_samples, len(questions))
        sft_model = AutoModelForCausalLM.from_pretrained(
            args.base_model, torch_dtype=torch.bfloat16, attn_implementation="eager"
        ).to(args.device).eval()

        t0 = time.time()
        sft_results = evaluate_model_pass_at_k(
            sft_model, tokenizer, questions, system_prompt,
            args.kg_server_url, args.n_samples, args.temperature, args.device,
        )
        sft_results["elapsed_s"] = time.time() - t0
        all_results["sft_base"] = sft_results
        logger.info("SFT base: %s (%.0fs)", {k: v for k, v in sft_results.items() if k.startswith("pass@")}, sft_results["elapsed_s"])

        del sft_model
        torch.cuda.empty_cache()

    if "e3" in args.models:
        logger.info("=== Evaluating E3 step %d (n=%d, %d questions) ===", args.step, args.n_samples, len(questions))
        e3_model = load_checkpoint(args.checkpoint_dir, args.step, args.base_model, args.device)

        t0 = time.time()
        e3_results = evaluate_model_pass_at_k(
            e3_model, tokenizer, questions, system_prompt,
            args.kg_server_url, args.n_samples, args.temperature, args.device,
        )
        e3_results["elapsed_s"] = time.time() - t0
        all_results[f"e3_step_{args.step}"] = e3_results
        logger.info("E3 step %d: %s (%.0fs)", args.step, {k: v for k, v in e3_results.items() if k.startswith("pass@")}, e3_results["elapsed_s"])

        del e3_model
        torch.cuda.empty_cache()

    if "e2" in args.models and args.e2_checkpoint_dir and args.e2_checkpoint_dir.exists():
        logger.info("=== Evaluating E2 step %d ===", args.e2_step)
        e2_model = load_checkpoint(args.e2_checkpoint_dir, args.e2_step, args.base_model, args.device)

        t0 = time.time()
        e2_results = evaluate_model_pass_at_k(
            e2_model, tokenizer, questions, system_prompt,
            args.kg_server_url, args.n_samples, args.temperature, args.device,
        )
        e2_results["elapsed_s"] = time.time() - t0
        all_results[f"e2_step_{args.e2_step}"] = e2_results
        logger.info("E2 step %d: %s (%.0fs)", args.e2_step, {k: v for k, v in e2_results.items() if k.startswith("pass@")}, e2_results["elapsed_s"])

        del e2_model
        torch.cuda.empty_cache()

    # --- Summary ---
    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info("Results saved to %s", args.output)

    print("\n=== Pass@k Summary ===")
    print(f"{'Model':<20} {'pass@1':>8} {'pass@4':>8} {'pass@8':>8} {'pass@16':>8} {'pass@32':>8}")
    print("-" * 60)
    for model_name, res in all_results.items():
        vals = " ".join(f"{res.get(f'pass@{k}', 0):>8.3f}" for k in [1, 4, 8, 16, 32])
        print(f"{model_name:<20} {vals}")

    # Interpretation
    sft_p1 = all_results["sft_base"].get("pass@1", 0)
    sft_p32 = all_results["sft_base"].get("pass@32", 0)
    e3_key = f"e3_step_{args.step}"
    e3_p1 = all_results[e3_key].get("pass@1", 0)
    e3_p32 = all_results[e3_key].get("pass@32", 0)

    print(f"\npass@1:  E3={e3_p1:.3f} vs SFT={sft_p1:.3f} (ratio={e3_p1/max(sft_p1,1e-6):.1f}x)")
    print(f"pass@32: E3={e3_p32:.3f} vs SFT={sft_p32:.3f} (ratio={e3_p32/max(sft_p32,1e-6):.1f}x)")

    if e3_p1 > sft_p1 * 1.5 and abs(e3_p32 - sft_p32) < 0.1:
        print("→ DISTRIBUTION SHARPENING: E3 pass@1 >> SFT but pass@32 ≈ SFT")
    elif e3_p32 > sft_p32 * 1.3:
        print("→ CAPABILITY EXPANSION: E3 pass@32 >> SFT pass@32")
    elif sft_p32 > e3_p32 * 1.3:
        print("→ CAPABILITY NARROWING: SFT pass@32 >> E3 pass@32")
    else:
        print("→ INCONCLUSIVE: differences within noise margin")


if __name__ == "__main__":
    main()
