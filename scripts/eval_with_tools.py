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
    repetition_penalty: float = 1.0,
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
                repetition_penalty=repetition_penalty,
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
    parser.add_argument("--max_samples", type=int, default=500,
                        help="Max samples to evaluate. 0 = all samples in eval_data.")
    parser.add_argument("--max_turns", type=int, default=5)
    parser.add_argument("--max_new_tokens", type=int, default=512,
                        help="Max new tokens per generation turn")
    parser.add_argument("--repetition_penalty", type=float, default=1.0,
                        help="Repetition penalty for generation (1.0 = no penalty)")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save_trajectories", type=Path, default=None,
                        help="Directory to save full trajectories (one JSON per step)")
    parser.add_argument("--max_trajectory_samples", type=int, default=100,
                        help="Max trajectories to save per checkpoint")
    parser.add_argument("--filter_ids", type=Path, default=None,
                        help="JSON file with 'filter_ids' or 'category_b_ids' list to filter questions")
    parser.add_argument("--save_per_sample", type=Path, default=None,
                        help="Directory to save per-sample results JSON (for bootstrap CIs / McNemar's test)")
    parser.add_argument("--start_idx", type=int, default=0,
                        help="Start at this sample index (for splitting across jobs)")
    parser.add_argument("--end_idx", type=int, default=-1,
                        help="End at this sample index (-1 = no limit). For splitting across jobs.")
    parser.add_argument("--save_every", type=int, default=200,
                        help="Save per-sample + partial results every N samples (for resume on timeout)")
    parser.add_argument("--resume", action="store_true",
                        help="Skip sample_ids already present in the per-sample output file")
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

    # Optional: filter to specific question IDs (e.g., Category B from Task 26)
    filter_id_set = None
    if args.filter_ids:
        with open(args.filter_ids) as fid:
            filter_data = json.load(fid)
        filter_id_set = set(
            filter_data.get("filter_ids", filter_data.get("category_b_ids", []))
        )
        logger.info("Filtering to %d question IDs", len(filter_id_set))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    results = {}

    all_steps = args.steps if 0 in args.steps else args.steps  # Only include step 0 if explicitly requested

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
        trajectories: list[dict] = []
        per_sample: list[dict] = []

        # max_samples=0 means evaluate all
        max_samples = args.max_samples if args.max_samples > 0 else len(df)
        end_idx = args.end_idx if args.end_idx > 0 else len(df)

        # Determine per-sample output file path (for periodic saves + resume)
        ps_file = None
        if args.save_per_sample is not None:
            ps_dir = args.save_per_sample
            ps_dir.mkdir(parents=True, exist_ok=True)
            if args.start_idx > 0 or args.end_idx > 0:
                ps_file = ps_dir / f"step_{step}_chunk_{args.start_idx}_{end_idx}.json"
            else:
                ps_file = ps_dir / f"step_{step}_per_sample.json"

        # Resume: load already-processed records
        already_done: set[str] = set()
        if args.resume and ps_file is not None and ps_file.exists():
            try:
                with open(ps_file) as psf_in:
                    prior = json.load(psf_in)
                for rec in prior:
                    already_done.add(str(rec["sample_id"]))
                    per_sample.append(rec)
                    ems.append(float(rec.get("em", 0)))
                    cms.append(float(rec.get("contains_em", 0)))
                    f1s.append(float(rec.get("f1", 0)))
                    tool_counts.append(int(rec.get("num_tool_calls", 0)))
                logger.info("Resumed: loaded %d already-processed samples", len(already_done))
            except Exception as e:
                logger.warning("Could not resume from %s: %s", ps_file, e)

        n_evaluated = len(per_sample)
        for i, (_, row) in enumerate(df.iterrows()):
            if i < args.start_idx:
                continue
            if i >= end_idx:
                break
            if n_evaluated >= max_samples:
                break

            sample_id = row["extra_info"].get("sample_id", str(i))

            # Skip questions not in filter set (if filtering enabled)
            if filter_id_set is not None and sample_id not in filter_id_set:
                continue

            # Resume: skip already-processed samples
            if str(sample_id) in already_done:
                continue

            question = row["prompt"][1]["content"]
            gold = row["reward_model"]["ground_truth"]
            all_answers = row["extra_info"].get("all_answers", [gold])
            if hasattr(all_answers, "tolist"):
                all_answers = all_answers.tolist()
            hops = int(row["extra_info"].get("hops", 0))
            n_evaluated += 1

            full_response, n_tools = generate_with_tools(
                model, tokenizer, question, system_prompt,
                args.kg_server_url, args.max_turns,
                max_new_tokens=args.max_new_tokens,
                repetition_penalty=args.repetition_penalty,
                device=args.device,
            )

            predicted = extract_answer(full_response)
            em_score = exact_match(predicted, gold, all_answers)
            cm_score = contains_match(full_response, gold, all_answers)
            f1_score = token_f1(predicted, gold)
            ems.append(em_score)
            cms.append(cm_score)
            f1s.append(f1_score)
            tool_counts.append(n_tools)

            if args.save_per_sample is not None:
                per_sample.append({
                    "sample_id": str(sample_id),
                    "em": float(em_score),
                    "contains_em": float(cm_score),
                    "f1": float(f1_score),
                    "num_tool_calls": int(n_tools),
                    "hops": hops,
                })

            if args.save_trajectories and n_evaluated <= args.max_trajectory_samples:
                trajectories.append({
                    "sample_id": str(sample_id),
                    "question": question,
                    "gold_answer": gold,
                    "all_answers": [str(a) for a in all_answers],
                    "hops": hops,
                    "predicted": predicted,
                    "full_response": full_response,
                    "num_tool_calls": n_tools,
                    "em": em_score,
                    "f1": f1_score,
                })

            if n_evaluated % 50 == 0:
                logger.info(
                    "  %d/%d EM=%.3f ContEM=%.3f F1=%.3f ToolCalls=%.1f",
                    n_evaluated, max_samples,
                    sum(ems) / len(ems), sum(cms) / len(cms),
                    sum(f1s) / len(f1s), sum(tool_counts) / len(tool_counts),
                )

            # Periodic per-sample save (so timeouts don't waste compute)
            if ps_file is not None and n_evaluated % args.save_every == 0:
                with open(ps_file, "w") as psf_out:
                    json.dump(per_sample, psf_out, indent=2)

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

        if args.save_trajectories and trajectories:
            traj_dir = args.save_trajectories / f"step_{step}"
            traj_dir.mkdir(parents=True, exist_ok=True)
            traj_file = traj_dir / "trajectories.json"
            with open(traj_file, "w") as tf:
                json.dump(trajectories, tf, indent=2, ensure_ascii=False)
            logger.info("Saved %d trajectories to %s", len(trajectories), traj_file)

        if ps_file is not None and per_sample:
            with open(ps_file, "w") as psf:
                json.dump(per_sample, psf, indent=2)
            logger.info("Saved %d per-sample records to %s", len(per_sample), ps_file)

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
