"""Variant G Step 1a — yield pilot for self-distillation.

Runs 39B@400 greedy + tools on 2,000 randomly-sampled CWQ train questions
(seed=42) and computes the "strict yield": fraction of trajectories where
  (a) EM = 1, AND
  (b) num_tool_calls >= 1, AND
  (c) format is valid (<answer> tag present, no nested <search> in <think>).

Per v11 decision rule:
  yield >= 15%  → proceed to full 27K sweep, expect ~4K trajectories
  7.5-15%       → widen filter to EM=1 OR (F1 >= 0.8 AND tool_used)
  < 7.5%        → EEF salvage BEFORE full sweep

This is a ReST-EM style data-generation step, not a training step. 1 GPU, ~1h.

Output: data/freebase/verl_cwq/39b_pilot_yield.json
  {
    "yield_strict": 0.xx,
    "yield_widened": 0.xx,
    "n_pilot": 2000,
    "n_strict_pass": N,
    "n_widened_pass": N,
    "decision": "proceed_strict" | "proceed_widened" | "eef_salvage",
    "per_sample": [...]
  }
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import re
import sys
import time
from pathlib import Path

import pandas as pd
import requests
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Reuse the utilities from scripts/
sys.path.insert(0, str(Path(__file__).parent))
from task17_pass_at_k import (  # noqa: E402
    exact_match,
    extract_answer,
    load_checkpoint,
    normalize,
)

logger = logging.getLogger(__name__)


def _token_f1(prediction: str, ground_truth: str) -> float:
    """Token-level F1 (same normalization as eval_with_tools)."""
    pn = normalize(prediction).split()
    gn = normalize(ground_truth).split()
    if not pn or not gn:
        return 0.0
    common = set(pn) & set(gn)
    if not common:
        return 0.0
    p = len(common) / len(pn)
    r = len(common) / len(gn)
    return 2 * p * r / (p + r)


def _format_valid(full_response: str) -> bool:
    """Same binary check as _format_valid in verl_reward.py (Variant A reward).

    Returns True if:
      - No <search> nested in a <think>...</think>
      - <answer>...</answer> present and non-empty
    """
    for m in re.finditer(r"<think>(.*?)</think>", full_response, re.DOTALL):
        if "<search>" in m.group(1):
            return False
    a = re.search(r"<answer>(.*?)</answer>", full_response, re.DOTALL)
    if not a or not a.group(1).strip():
        return False
    return True


def _parse_search_call(text: str) -> tuple[str, str, str | None] | None:
    m = re.search(r"<search>\s*(\w+)\s*\(([^)]*)\)\s*</search>", text)
    if not m:
        return None
    action = m.group(1)
    args = [a.strip().strip("'\"") for a in m.group(2).split(",") if a.strip()]
    if not args:
        return None
    entity = args[0]
    relation = args[1] if len(args) > 1 else None
    return action, entity, relation


def _call_kg(action: str, entity: str, relation: str | None, url: str) -> str:
    try:
        r = requests.post(
            f"{url}/retrieve",
            json={"tool": action, "entity": entity, "relation": relation},
            timeout=30,
        )
        r.raise_for_status()
        return json.dumps(r.json().get("results", []))[:1500]
    except Exception as e:
        return f"ERROR: {e}"


def generate_greedy(
    model, tokenizer, question, system_prompt, kg_url, max_turns, max_new_tokens, device,
):
    """Greedy generation with tools — same loop as eval_with_tools.generate_with_tools."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]
    full = ""
    num_calls = 0
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
        resp = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        full += resp
        if "<answer>" in resp and "</answer>" in resp:
            messages.append({"role": "assistant", "content": resp})
            break
        call = _parse_search_call(resp)
        if call is None:
            messages.append({"role": "assistant", "content": resp})
            break
        action, entity, relation = call
        num_calls += 1
        kg_res = _call_kg(action, entity, relation, kg_url)
        messages.append({"role": "assistant", "content": resp})
        messages.append({"role": "user", "content": f"<tool_response>{kg_res}</tool_response>"})
        full += f"\n<tool_response>{kg_res}</tool_response>\n"
    return full, num_calls, messages


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    parser = argparse.ArgumentParser(description="Variant G Step 1a: yield pilot")
    parser.add_argument(
        "--checkpoint_dir", type=Path,
        default=Path("/projects/u6gg/KG_GRPO/checkpoints/kg-align-verl/grpo-cwq-7b-39b-kl-20260413"),
    )
    parser.add_argument("--step", type=int, default=400)
    parser.add_argument("--base_model", type=str, default="outputs/verl-sft-cwq-7b-merged")
    parser.add_argument("--train_parquet", type=Path, default=Path("data/freebase/verl_cwq/train.parquet"))
    parser.add_argument("--kg_server_url", type=str, default="http://localhost:18901")
    parser.add_argument("--n_pilot", type=int, default=2000,
                        help="Total pilot size after seed shuffle. Use 0 to process all rows.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_turns", type=int, default=5)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--output", type=Path, default=Path("data/freebase/verl_cwq/39b_pilot_yield.json"))
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--start_idx", type=int, default=0,
                        help="Start index within the shuffled pilot_idx (for sharding).")
    parser.add_argument("--end_idx", type=int, default=0,
                        help="End index within the shuffled pilot_idx (0 = to end).")
    parser.add_argument("--save_every", type=int, default=100,
                        help="Flush partial output every N samples.")
    parser.add_argument("--resume", action="store_true",
                        help="If --output exists, load its per_sample list and skip completed sample_ids.")
    args = parser.parse_args()

    # Verify KG server
    r = requests.get(f"{args.kg_server_url}/health", timeout=5)
    logger.info("KG server healthy: %s", r.status_code)

    # Load data + sample pilot
    df = pd.read_parquet(args.train_parquet)
    logger.info("Loaded %d train rows", len(df))
    system_prompt = df.iloc[0]["prompt"][0]["content"]

    idx = list(range(len(df)))
    random.Random(args.seed).shuffle(idx)
    n_pilot = args.n_pilot if args.n_pilot > 0 else len(idx)
    pilot_idx = idx[:n_pilot]
    end_idx = args.end_idx if args.end_idx > 0 else len(pilot_idx)
    shard_idx = pilot_idx[args.start_idx:end_idx]
    logger.info("Pilot size: %d  shard: [%d:%d] = %d rows",
                len(pilot_idx), args.start_idx, end_idx, len(shard_idx))

    # Load model
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    logger.info("Loading 39B @ step %d", args.step)
    model = load_checkpoint(args.checkpoint_dir, args.step, args.base_model, args.device)

    per_sample: list = []
    n_strict = 0
    n_widened = 0
    done_ids: set = set()

    # Resume: load prior per-sample list
    if args.resume and args.output.exists():
        try:
            prior = json.loads(args.output.read_text())
            for rec in prior.get("per_sample", []):
                per_sample.append(rec)
                done_ids.add(str(rec.get("sample_id", "")))
                n_strict += int(rec.get("strict_pass", 0))
                n_widened += int(rec.get("widened_pass", 0))
            logger.info("Resumed: loaded %d prior samples", len(done_ids))
        except Exception as e:
            logger.warning("Could not resume: %s", e)

    t_start = time.time()

    def _flush_output() -> None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        n_processed = len(per_sample)
        y_s = n_strict / n_processed if n_processed else 0.0
        y_w = n_widened / n_processed if n_processed else 0.0
        out = {
            "yield_strict": y_s,
            "yield_widened": y_w,
            "n_processed": n_processed,
            "n_pilot": n_pilot,
            "shard_start": args.start_idx,
            "shard_end": end_idx,
            "n_strict_pass": n_strict,
            "n_widened_pass": n_widened,
            "decision": (
                "proceed_strict" if y_s >= 0.15
                else ("proceed_widened" if y_s >= 0.075 else "eef_salvage")
            ),
            "elapsed_s": time.time() - t_start,
            "per_sample": per_sample,
        }
        tmp = args.output.with_suffix(args.output.suffix + ".tmp")
        tmp.write_text(json.dumps(out, indent=2))
        tmp.replace(args.output)

    for i, row_idx in enumerate(shard_idx):
        row = df.iloc[row_idx]
        question = row["prompt"][1]["content"]
        gold = row["reward_model"]["ground_truth"]
        all_answers = row["extra_info"].get("all_answers", [gold])
        if hasattr(all_answers, "tolist"):
            all_answers = all_answers.tolist()
        all_answers = [str(a) for a in all_answers]
        sample_id = str(row["extra_info"].get("sample_id", row_idx))

        if sample_id in done_ids:
            continue

        full, ntools, messages = generate_greedy(
            model, tokenizer, question, system_prompt,
            args.kg_server_url, args.max_turns, args.max_new_tokens, args.device,
        )
        predicted = extract_answer(full)
        em = exact_match(predicted, gold, all_answers)
        f1 = _token_f1(predicted, gold)
        fmt_ok = _format_valid(full)

        strict_pass = int(em >= 1 and ntools >= 1 and fmt_ok)
        widened_pass = int(
            (em >= 1 or (f1 >= 0.8 and ntools >= 1)) and fmt_ok
        )
        n_strict += strict_pass
        n_widened += widened_pass

        per_sample.append(
            {
                "sample_id": sample_id,
                "question": question,
                "gold": gold,
                "predicted": predicted,
                "em": em,
                "f1": f1,
                "num_tool_calls": ntools,
                "format_valid": fmt_ok,
                "strict_pass": strict_pass,
                "widened_pass": widened_pass,
                "full_response": full[:4000],
            }
        )

        n_processed = len(per_sample)
        if n_processed % args.save_every == 0:
            _flush_output()

        if n_processed % 100 == 0:
            logger.info(
                "  %d/%d  strict=%.1f%%  widened=%.1f%%  elapsed=%ds",
                n_processed, len(shard_idx),
                100 * n_strict / n_processed, 100 * n_widened / n_processed,
                int(time.time() - t_start),
            )

    _flush_output()

    n = len(per_sample)
    yield_strict = n_strict / n if n else 0.0
    yield_widened = n_widened / n if n else 0.0
    if yield_strict >= 0.15:
        decision = "proceed_strict"
    elif yield_strict >= 0.075:
        decision = "proceed_widened"
    else:
        decision = "eef_salvage"

    logger.info(
        "FINAL strict=%.3f widened=%.3f decision=%s",
        yield_strict, yield_widened, decision,
    )
    print(
        f"\n=== Pilot yield: strict={100*yield_strict:.1f}% "
        f"({n_strict}/{n}), widened={100*yield_widened:.1f}% ({n_widened}/{n}) ==="
    )
    print(f"=== Decision: {decision} ===")


if __name__ == "__main__":
    main()
