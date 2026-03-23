"""Evaluate GRPO checkpoints WITH KG tool access (multi-turn agent inference).

This is the fair evaluation for tool-using models (E3 verifiable):
the model gets to query the KG server during inference, just like during training.

Uses a simple ReAct loop: generate -> check for <search> tags -> call KG -> append response -> repeat.

Usage:
    # Start KG server first:
    python -m src_verl.kg_server.server --kg freebase --freebase_dir data/freebase/kg --port 18901 &

    # Then run eval:
    python scripts/eval_with_tools.py \
        --checkpoint_dir checkpoints/kg-align-verl/grpo-cwq-7b-verifiable-20260321 \
        --steps 500 1250 \
        --eval_data data/freebase/verl_cwq/val.parquet \
        --output results/eval_e3_with_tools.json \
        --kg_server_url http://localhost:18901 \
        --max_samples 500
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any

import requests
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd

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


def contains_match(output: str, gold: str, aliases: list | None = None) -> float:
    out_norm = normalize(output)
    if normalize(gold) in out_norm:
        return 1.0
    if aliases:
        for a in aliases:
            if normalize(str(a)) in out_norm:
                return 1.0
    return 0.0


def token_f1(predicted: str, gold: str) -> float:
    p = normalize(predicted).split()
    g = normalize(gold).split()
    if not g:
        return 1.0 if not p else 0.0
    if not p:
        return 0.0
    common = sum((Counter(p) & Counter(g)).values())
    if common == 0:
        return 0.0
    prec = common / len(p)
    rec = common / len(g)
    return 2 * prec * rec / (prec + rec)


def extract_answer(text: str) -> str:
    if "<answer>" in text and "</answer>" in text:
        return text.split("<answer>")[1].split("</answer>")[0].strip()
    if "</think>" in text:
        return text.split("</think>")[-1].strip()
    return text.strip()


def call_kg_server(action: str, entity: str, relation: str | None, server_url: str) -> str:
    """Call the KG server and return results as text."""
    payload = {"action": action, "entity": entity}
    if relation:
        payload["relation"] = relation
    try:
        resp = requests.post(f"{server_url}/retrieve", json=payload, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            results = data.get("results", data.get("result", []))
            if isinstance(results, list) and len(results) > 10:
                results = results[:10]  # Cap to avoid huge responses
            return json.dumps(results, ensure_ascii=False)
        return f"Error: {resp.status_code}"
    except Exception as e:
        return f"Error: {e}"


def parse_search_call(text: str) -> tuple[str, str, str | None] | None:
    """Parse a <search>action(args)</search> call. Returns (action, entity, relation) or None."""
    match = re.search(r"<search>\s*(\w+)\(([^)]*)\)\s*</search>", text, re.DOTALL)
    if not match:
        return None
    action = match.group(1)
    args = [a.strip().strip("'\"") for a in match.group(2).split(",") if a.strip()]
    entity = args[0] if args else ""
    relation = args[1] if len(args) > 1 else None
    return action, entity, relation


def generate_with_tools(
    model: Any,
    tokenizer: Any,
    question: str,
    system_prompt: str,
    kg_server_url: str,
    max_turns: int = 5,
    max_new_tokens: int = 512,
    device: str = "cuda",
) -> tuple[str, int]:
    """Multi-turn generation with KG tool calls.

    Returns (full_conversation_text, num_tool_calls).
    """
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
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        response = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        full_response += response

        # Check for <answer> tag — done
        if "<answer>" in response and "</answer>" in response:
            messages.append({"role": "assistant", "content": response})
            break

        # Check for <search> tag — tool call
        call = parse_search_call(response)
        if call is None:
            # No tool call and no answer — model is just reasoning, stop
            messages.append({"role": "assistant", "content": response})
            break

        action, entity, relation = call
        num_tool_calls += 1

        # Execute tool call
        kg_result = call_kg_server(action, entity, relation, kg_server_url)

        # Add to conversation
        messages.append({"role": "assistant", "content": response})
        messages.append({"role": "user", "content": f"<tool_response>{kg_result}</tool_response>"})
        full_response += f"\n<tool_response>{kg_result}</tool_response>\n"

    return full_response, num_tool_calls


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


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    parser = argparse.ArgumentParser(description="Evaluate checkpoints WITH KG tool access.")
    parser.add_argument("--checkpoint_dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, nargs="+", default=[500, 1250])
    parser.add_argument("--eval_data", type=Path, default=Path("data/freebase/verl_cwq/val.parquet"))
    parser.add_argument("--base_model", type=str, default="outputs/verl-sft-cwq-7b-merged")
    parser.add_argument("--output", type=Path, default=Path("results/eval_with_tools.json"))
    parser.add_argument("--kg_server_url", type=str, default="http://localhost:18901")
    parser.add_argument("--max_samples", type=int, default=500)
    parser.add_argument("--max_turns", type=int, default=5)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    # Verify KG server is reachable
    try:
        r = requests.get(f"{args.kg_server_url}/health", timeout=5)
        logger.info("KG server healthy: %s", r.status_code)
    except Exception as e:
        logger.error("KG server not reachable at %s: %s", args.kg_server_url, e)
        logger.error("Start the KG server first!")
        return

    df = pd.read_parquet(args.eval_data)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    system_prompt = df.iloc[0]["prompt"][0]["content"]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    results = {}

    all_steps = [0] + args.steps  # Always include SFT base

    for step in all_steps:
        logger.info("=== Evaluating step %d (WITH tools) ===", step)
        start = time.time()

        if step == 0:
            model = AutoModelForCausalLM.from_pretrained(
                args.base_model, torch_dtype=torch.bfloat16, attn_implementation="eager"
            ).to(args.device).eval()
        else:
            model = load_checkpoint(args.checkpoint_dir, step, args.base_model, args.device)

        ems, cms, f1s, tool_counts = [], [], [], []

        for i, (_, row) in enumerate(df.iterrows()):
            if i >= args.max_samples:
                break

            question = row["prompt"][1]["content"]
            gold = row["reward_model"]["ground_truth"]
            all_answers = row["extra_info"].get("all_answers", [gold])
            if hasattr(all_answers, "tolist"):
                all_answers = all_answers.tolist()

            full_response, n_tools = generate_with_tools(
                model, tokenizer, question, system_prompt,
                args.kg_server_url, args.max_turns, device=args.device,
            )

            predicted = extract_answer(full_response)
            ems.append(exact_match(predicted, gold, all_answers))
            cms.append(contains_match(full_response, gold, all_answers))
            f1s.append(token_f1(predicted, gold))
            tool_counts.append(n_tools)

            if (i + 1) % 50 == 0:
                logger.info(
                    "  %d/%d EM=%.3f ContEM=%.3f F1=%.3f ToolCalls=%.1f",
                    i + 1, args.max_samples,
                    sum(ems) / len(ems), sum(cms) / len(cms),
                    sum(f1s) / len(f1s), sum(tool_counts) / len(tool_counts),
                )

        elapsed = time.time() - start
        r = {
            "em": sum(ems) / len(ems) if ems else 0,
            "contains_em": sum(cms) / len(cms) if cms else 0,
            "f1": sum(f1s) / len(f1s) if f1s else 0,
            "avg_tool_calls": sum(tool_counts) / len(tool_counts) if tool_counts else 0,
            "n_samples": len(ems),
            "elapsed_s": elapsed,
        }
        results[str(step)] = r
        logger.info("Step %d: EM=%.4f ContEM=%.4f F1=%.4f ToolCalls=%.1f (%.0fs)",
                     step, r["em"], r["contains_em"], r["f1"], r["avg_tool_calls"], elapsed)

        del model
        torch.cuda.empty_cache()

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Results saved to %s", args.output)

    print("\n=== Summary (WITH tools) ===")
    print(f"{'Step':>6} {'EM':>8} {'ContEM':>8} {'F1':>8} {'Tools':>8}")
    print("-" * 42)
    for step in sorted(results.keys(), key=int):
        r = results[step]
        print(f"{step:>6} {r['em']:>8.4f} {r['contains_em']:>8.4f} {r['f1']:>8.4f} {r['avg_tool_calls']:>8.1f}")


if __name__ == "__main__":
    main()
