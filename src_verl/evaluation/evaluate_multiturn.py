"""Multi-turn evaluation with live KG tool calls.

Evaluates models on ConceptNet QA and HotpotQA by running multi-turn
agent trajectories with the KG server. Compares base, SFT, and GRPO models.

Usage:
    # Start KG server first, then:
    python src_verl/evaluation/evaluate_multiturn.py \
        --benchmark conceptnet \
        --base_model Qwen/Qwen2.5-1.5B-Instruct \
        --sft_adapter outputs/verl-sft \
        --grpo_adapter outputs/verl-grpo/checkpoint-1000 \
        --kg_server_url http://localhost:8001
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src_verl.data.prepare_conceptnet import SYSTEM_PROMPT
from src_verl.rewards.common import (
    compute_exact_match,
    compute_token_f1,
    extract_answer,
    normalize_answer,
)
from src_verl.rewards.reward_outcome import compute_reward_outcome
from src_verl.rewards.reward_heuristic import compute_reward_heuristic
from src_verl.rewards.reward_verifiable import compute_reward_verifiable

logger = logging.getLogger(__name__)

MAX_TURNS = 10


@dataclass
class MultiTurnEvalResults:
    """Evaluation results for a single model."""

    model_name: str
    benchmark: str
    total: int = 0
    em_sum: float = 0.0
    f1_sum: float = 0.0
    reward_outcome_sum: float = 0.0
    reward_heuristic_sum: float = 0.0
    reward_verifiable_sum: float = 0.0
    avg_turns: float = 0.0
    total_turns: int = 0
    tool_call_count: int = 0
    valid_tool_calls: int = 0
    samples: list[dict[str, Any]] = field(default_factory=list)

    @property
    def em(self) -> float:
        return self.em_sum / max(self.total, 1)

    @property
    def f1(self) -> float:
        return self.f1_sum / max(self.total, 1)

    @property
    def mean_turns(self) -> float:
        return self.total_turns / max(self.total, 1)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "benchmark": self.benchmark,
            "total": self.total,
            "exact_match": round(self.em, 4),
            "token_f1": round(self.f1, 4),
            "reward_outcome": round(self.reward_outcome_sum / max(self.total, 1), 4),
            "reward_heuristic": round(self.reward_heuristic_sum / max(self.total, 1), 4),
            "reward_verifiable": round(self.reward_verifiable_sum / max(self.total, 1), 4),
            "avg_turns": round(self.mean_turns, 2),
            "tool_calls": self.tool_call_count,
            "valid_tool_calls": self.valid_tool_calls,
        }


def load_model(
    model_name: str,
    adapter_path: str | None = None,
) -> tuple[Any, Any]:
    """Load model with optional LoRA adapter or merged HF checkpoint.

    If adapter_path contains adapter_config.json, loads as LoRA adapter on top of model_name.
    If adapter_path contains config.json (full HF model), loads directly from adapter_path.
    Otherwise falls back to loading model_name only.
    """
    model_kwargs: dict[str, Any] = {
        "torch_dtype": torch.bfloat16,
        "device_map": "auto",
        "attn_implementation": "sdpa",
    }

    tokenizer_path = model_name  # default tokenizer source

    if adapter_path:
        adapter_dir = Path(adapter_path)
        if adapter_dir.exists() and (adapter_dir / "adapter_config.json").exists():
            # LoRA adapter
            model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, adapter_path)
            logger.info("Loaded LoRA adapter from %s", adapter_path)
        elif adapter_dir.exists() and (adapter_dir / "config.json").exists():
            # Full merged HF model (e.g. verl checkpoint converted via model_merger)
            model = AutoModelForCausalLM.from_pretrained(adapter_path, **model_kwargs)
            tokenizer_path = adapter_path
            logger.info("Loaded merged HF model from %s", adapter_path)
        else:
            logger.warning("No model found at %s, falling back to %s", adapter_path, model_name)
            model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def run_multiturn_inference(
    model: Any,
    tokenizer: Any,
    question: str,
    kg_server_url: str,
    max_turns: int = MAX_TURNS,
    max_new_tokens: int = 512,
) -> list[dict[str, str]]:
    """Run multi-turn agent inference with KG tool calls."""
    import requests

    messages: list[dict[str, str]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]

    for turn in range(max_turns):
        # Generate
        input_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        input_len = inputs["input_ids"].shape[1]
        response = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)

        messages.append({"role": "assistant", "content": response})

        # Check for final answer
        if "<answer>" in response:
            break

        # Check for tool call
        search_match = re.search(r"<search>(.*?)</search>", response, re.DOTALL)
        if search_match:
            action_str = search_match.group(1).strip()
            tool_result = _execute_tool_call(action_str, kg_server_url)
            messages.append({
                "role": "tool",
                "content": f"<information>{tool_result}</information>",
            })
        else:
            # No tool call and no answer — model is done
            break

    return messages


def _execute_tool_call(action_str: str, kg_server_url: str) -> str:
    """Parse and execute a tool call against the KG server."""
    import requests

    func_match = re.match(r"(\w+)\((.*)\)", action_str, re.DOTALL)
    if not func_match:
        return "Invalid tool call format."

    action_type = func_match.group(1)
    args_str = func_match.group(2).strip()
    parts = [p.strip().strip("'\"") for p in args_str.split(",")]

    payload: dict[str, Any] = {"action": action_type}
    if parts:
        payload["entity"] = parts[0]
    if len(parts) > 1:
        payload["relation"] = parts[1]

    try:
        resp = requests.post(f"{kg_server_url}/retrieve", json=payload, timeout=10)
        resp.raise_for_status()
        results = resp.json().get("results", [])
        return str(results)
    except Exception as e:
        return f"Error: {e}"


def evaluate_model(
    model: Any,
    tokenizer: Any,
    records: list[dict[str, Any]],
    model_label: str,
    benchmark: str,
    kg_server_url: str,
    num_qualitative: int = 20,
) -> MultiTurnEvalResults:
    """Evaluate a model with multi-turn KG interaction."""
    results = MultiTurnEvalResults(model_name=model_label, benchmark=benchmark)

    for i, record in enumerate(records):
        question = record["question"]
        gold = record["gold_answer_short"]
        kg_path = record.get("kg_path", [])

        # Run multi-turn inference
        trajectory = run_multiturn_inference(
            model, tokenizer, question, kg_server_url,
        )

        # Extract answer
        predicted = extract_answer(trajectory)
        em = compute_exact_match(predicted, gold) if predicted else 0.0
        f1 = compute_token_f1(predicted, gold) if predicted else 0.0

        # Compute all three reward types
        r_out = compute_reward_outcome(trajectory, gold)
        r_heur = compute_reward_heuristic(trajectory, gold, kg_path)
        r_ver = compute_reward_verifiable(trajectory, gold, kg_path)

        # Count turns and tool calls
        num_turns = sum(1 for m in trajectory if m["role"] == "assistant")
        num_tool_calls = sum(
            1 for m in trajectory
            if m["role"] == "assistant" and "<search>" in m.get("content", "")
        )

        results.total += 1
        results.em_sum += em
        results.f1_sum += f1
        results.reward_outcome_sum += r_out
        results.reward_heuristic_sum += r_heur
        results.reward_verifiable_sum += r_ver
        results.total_turns += num_turns
        results.tool_call_count += num_tool_calls

        if i < num_qualitative:
            results.samples.append({
                "question": question,
                "gold_answer": gold,
                "predicted": predicted,
                "em": em,
                "f1": round(f1, 4),
                "r_outcome": round(r_out, 4),
                "r_heuristic": round(r_heur, 4),
                "r_verifiable": round(r_ver, 4),
                "num_turns": num_turns,
                "trajectory": trajectory,
            })

        if (i + 1) % 50 == 0:
            logger.info(
                "  [%s] %d/%d — EM=%.3f, F1=%.3f, turns=%.1f",
                model_label, i + 1, len(records), results.em, results.f1, results.mean_turns,
            )

    logger.info(
        "[%s] Final — EM=%.4f, F1=%.4f, turns=%.1f (%d samples)",
        model_label, results.em, results.f1, results.mean_turns, results.total,
    )
    return results


def load_conceptnet_test(path: Path, num_samples: int | None = None) -> list[dict[str, Any]]:
    """Load ConceptNet QA test split."""
    records: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if record.get("is_negative", False):
                continue
            records.append(record)
            if num_samples and len(records) >= num_samples:
                break
    return records


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    parser = argparse.ArgumentParser(description="Multi-turn evaluation with KG tools.")
    parser.add_argument("--benchmark", type=str, default="conceptnet", choices=["conceptnet", "hotpotqa"])
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--sft_adapter", type=str, default=None)
    parser.add_argument("--grpo_adapter", type=str, default=None)
    parser.add_argument("--test_file", type=Path, default=Path("data/processed/conceptnet_qa_test.jsonl"))
    parser.add_argument("--num_samples", type=int, default=500)
    parser.add_argument("--kg_server_url", type=str, default="http://localhost:8001")
    parser.add_argument("--output_dir", type=Path, default=Path("outputs/eval_multiturn"))
    parser.add_argument("--num_qualitative", type=int, default=20)
    args = parser.parse_args()

    # Load test data
    if args.benchmark == "conceptnet":
        records = load_conceptnet_test(args.test_file, args.num_samples)
    else:
        from src.evaluation.evaluate import load_hotpotqa_bridge
        records = load_hotpotqa_bridge(args.num_samples)

    if not records:
        logger.error("No records loaded")
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Evaluate each model
    model_configs = [("base", args.base_model, None)]
    if args.sft_adapter:
        model_configs.append(("sft", args.base_model, args.sft_adapter))
    if args.grpo_adapter:
        model_configs.append(("sft+grpo", args.base_model, args.grpo_adapter))

    all_results: list[MultiTurnEvalResults] = []
    for label, model_name, adapter in model_configs:
        logger.info("=== Evaluating: %s ===", label)
        model, tokenizer = load_model(model_name, adapter)
        results = evaluate_model(
            model, tokenizer, records, label, args.benchmark,
            args.kg_server_url, args.num_qualitative,
        )
        all_results.append(results)

        # Save per-model results
        result_path = args.output_dir / f"{args.benchmark}_{label}_results.json"
        with open(result_path, "w") as f:
            json.dump({"metrics": results.to_dict(), "samples": results.samples}, f, indent=2, ensure_ascii=False)

        del model
        torch.cuda.empty_cache()

    # Save summary
    summary = {
        "benchmark": args.benchmark,
        "num_samples": len(records),
        "models": [r.to_dict() for r in all_results],
    }
    summary_path = args.output_dir / f"{args.benchmark}_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Summary saved to %s", summary_path)

    # Print comparison
    header = f"{'Model':<12} {'EM':>6} {'F1':>6} {'R_out':>6} {'R_heur':>6} {'R_ver':>6} {'Turns':>6}"
    logger.info("=== Comparison ===")
    logger.info(header)
    for r in all_results:
        logger.info(
            f"{r.model_name:<12} {r.em:>6.3f} {r.f1:>6.3f} "
            f"{r.reward_outcome_sum / max(r.total, 1):>6.3f} "
            f"{r.reward_heuristic_sum / max(r.total, 1):>6.3f} "
            f"{r.reward_verifiable_sum / max(r.total, 1):>6.3f} "
            f"{r.mean_turns:>6.1f}"
        )


if __name__ == "__main__":
    main()
