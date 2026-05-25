"""Evaluate a Search-R1-trained checkpoint using the Search-R1 single-tool
interface (free-form <search>query</search> -> Freebase adapter on port 19001).

This is "Arm B" in the V14-A1 Search-R1 baseline: the pure Search-R1 recipe.
For "Arm A" (our 4-tool eval), use `scripts/eval_with_tools.py` directly.

Assumes:
  - Checkpoint was trained with Search-R1's template (user prompt contains
    the `<think>...</think> ... <search>free-form query</search> ... <answer>...</answer>` instructions).
  - The Freebase adapter (scripts/phase7_ii2_searchr1_adapter.py) is running on
    --adapter_url, bridging to the KG server.

Metrics: same normalized-EM and token-F1 helpers reused from scripts/eval_with_tools.py.

Usage:
  python scripts/eval_searchr1_singletool.py \
      --checkpoint_dir /projects/u6gg/KG_GRPO/checkpoints/kg-align-verl/searchr1-cwq-7b-20260420 \
      --steps 100 200 300 400 500 \
      --eval_data data/freebase/searchr1_cwq/test.parquet \
      --adapter_url http://localhost:19001 \
      --output results/phase7/v14_a1_searchr1_step{step}_singletool_eval.json
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


def exact_match(pred: str, golds: list[str]) -> float:
    p = normalize(pred)
    if not p:
        return 0.0
    for g in golds:
        if p == normalize(g):
            return 1.0
    return 0.0


def contains_match(pred: str, golds: list[str]) -> float:
    p = normalize(pred)
    for g in golds:
        if normalize(g) in p:
            return 1.0
    return 0.0


def token_f1(pred: str, golds: list[str]) -> float:
    p = normalize(pred).split()
    if not p:
        return 0.0
    best = 0.0
    for g in golds:
        gt = normalize(g).split()
        if not gt:
            continue
        common = sum((Counter(p) & Counter(gt)).values())
        if common == 0:
            continue
        prec = common / len(p)
        rec = common / len(gt)
        f1 = 2 * prec * rec / (prec + rec)
        if f1 > best:
            best = f1
    return best


def extract_answer(text: str) -> str | None:
    # Search-R1 qa_em.py uses the LAST <answer>..</answer> occurrence,
    # but only if there are 2+ matches (their data template already contains
    # one example answer). For robustness in eval we take the last occurrence
    # unconditionally.
    matches = list(re.finditer(r"<answer>(.*?)</answer>", text, re.DOTALL))
    if not matches:
        return None
    return matches[-1].group(1).strip()


def parse_search_query(text: str) -> str | None:
    m = re.search(r"<search>(.*?)</search>", text, re.DOTALL)
    if not m:
        return None
    return m.group(1).strip()


def call_adapter(query: str, adapter_url: str, topk: int = 3) -> str:
    """Call Search-R1 adapter. Returns a formatted string of documents."""
    try:
        r = requests.post(
            f"{adapter_url}/retrieve",
            json={"queries": [query], "topk": topk, "return_scores": False},
            timeout=15,
        )
        r.raise_for_status()
        results = r.json().get("result", [[]])
        docs = results[0] if results else []
        out_lines: list[str] = []
        for i, d in enumerate(docs):
            doc = d.get("document", d)
            contents = doc.get("contents", "") if isinstance(doc, dict) else str(doc)
            out_lines.append(f"Doc {i + 1}: {contents}")
        return "\n".join(out_lines) if out_lines else "No results."
    except Exception as e:  # noqa: BLE001
        return f"Error contacting retrieval server: {e}"


def generate_single_tool(
    model: Any,
    tokenizer: Any,
    prompt_text: str,
    adapter_url: str,
    max_turns: int = 5,
    max_new_tokens: int = 512,
    topk: int = 3,
    device: str = "cuda",
) -> tuple[str, int]:
    """Search-R1-style multi-turn rollout driven by free-form <search>query</search>.

    Returns (full_generation_text, num_tool_calls).
    """
    current = prompt_text
    full = ""
    num_calls = 0
    for _ in range(max_turns):
        inputs = tokenizer(current, return_tensors="pt", truncation=True, max_length=8192).to(device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        response = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        full += response

        if "<answer>" in response and "</answer>" in response:
            break

        query = parse_search_query(response)
        if query is None:
            break

        num_calls += 1
        obs = call_adapter(query, adapter_url, topk=topk)
        appended = f"\n\n<information>{obs}</information>\n\n"
        current = current + response + appended
        full += appended

    return full, num_calls


def extract_targets(row: dict[str, Any]) -> list[str]:
    rm = row.get("reward_model") or {}
    gt = rm.get("ground_truth") if isinstance(rm, dict) else None
    if isinstance(gt, dict):
        tgts = gt.get("target", [])
        try:
            return [str(t).strip() for t in tgts if str(t).strip()]
        except TypeError:
            return [str(tgts).strip()]
    if isinstance(gt, str):
        return [gt]
    return []


def load_checkpoint(ckpt_dir: Path, step: int, base_model: str, device: str = "cuda"):
    """Load actor weights from Search-R1's checkpoint layout.

    Search-R1's verl trainer saves under {default_local_dir}/global_step_{N}/actor/.
    We reuse eval_with_tools.py's FSDP shard concatenation logic.
    """
    step_dir = ckpt_dir / f"global_step_{step}" / "actor"
    shards = sorted(step_dir.glob("model_world_size_*_rank_*.pt"))
    if not shards:
        # Fall back to HF-format saved model (some verl builds save directly)
        hf_path = step_dir if (step_dir / "config.json").exists() else None
        if hf_path is None:
            raise FileNotFoundError(f"No FSDP shards or HF model in {step_dir}")
        logger.info("Loading HF-format checkpoint from %s", hf_path)
        return AutoModelForCausalLM.from_pretrained(
            hf_path, torch_dtype=torch.bfloat16, attn_implementation="sdpa"
        ).to(device).eval()

    shard_locals: dict[str, list] = {}
    for sf in shards:
        s = torch.load(sf, map_location="cpu", weights_only=False)
        for k, v in s.items():
            local = v.to_local() if hasattr(v, "to_local") else v
            shard_locals.setdefault(k, []).append(local)
    merged = {k: torch.cat(vs, dim=0) if len(vs) > 1 else vs[0] for k, vs in shard_locals.items()}
    model = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=torch.bfloat16, attn_implementation="sdpa"
    )
    model.load_state_dict(merged, strict=False)
    return model.to(device).eval()


def eval_step(
    model: Any,
    tokenizer: Any,
    df: pd.DataFrame,
    adapter_url: str,
    max_samples: int,
    max_turns: int,
    max_new_tokens: int,
    topk: int,
    device: str,
) -> dict[str, Any]:
    rows = df.head(max_samples).to_dict(orient="records") if max_samples > 0 else df.to_dict(orient="records")
    em_sum = 0.0
    cm_sum = 0.0
    f1_sum = 0.0
    n_tools_sum = 0
    per_sample: list[dict[str, Any]] = []
    t0 = time.time()
    for i, row in enumerate(rows):
        prompt_msgs = row["prompt"]
        prompt_text = prompt_msgs[0]["content"]
        golds = extract_targets(row)
        full, n_tools = generate_single_tool(
            model, tokenizer, prompt_text, adapter_url,
            max_turns=max_turns, max_new_tokens=max_new_tokens,
            topk=topk, device=device,
        )
        pred = extract_answer(full) or ""
        em = exact_match(pred, golds)
        cm = contains_match(pred, golds)
        f1 = token_f1(pred, golds)
        em_sum += em
        cm_sum += cm
        f1_sum += f1
        n_tools_sum += n_tools
        per_sample.append({
            "index": i,
            "pred": pred,
            "gold": golds[0] if golds else "",
            "em": em,
            "contains": cm,
            "f1": f1,
            "n_tool_calls": n_tools,
        })
        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            logger.info(
                "[%d/%d] EM=%.3f CM=%.3f F1=%.3f tools/turn=%.2f elapsed=%.1fs",
                i + 1, len(rows),
                em_sum / (i + 1), cm_sum / (i + 1), f1_sum / (i + 1),
                n_tools_sum / (i + 1), elapsed,
            )
    n = max(len(rows), 1)
    return {
        "n": n,
        "em": em_sum / n,
        "contains": cm_sum / n,
        "f1": f1_sum / n,
        "avg_tool_calls": n_tools_sum / n,
        "per_sample": per_sample,
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    ap = argparse.ArgumentParser(description="Search-R1 single-tool eval (Arm B of V14-A1)")
    ap.add_argument("--checkpoint_dir", type=Path, required=True)
    ap.add_argument("--steps", type=int, nargs="+", required=True)
    ap.add_argument("--eval_data", type=Path, default=Path("data/freebase/searchr1_cwq/test.parquet"))
    ap.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--adapter_url", type=str, default="http://localhost:19001")
    ap.add_argument("--output", type=str,
                    default="results/phase7/v14_a1_searchr1_step{step}_singletool_eval.json",
                    help="Output path; '{step}' is replaced per checkpoint.")
    ap.add_argument("--max_samples", type=int, default=-1,
                    help="-1 = full test set.")
    ap.add_argument("--max_turns", type=int, default=5)
    ap.add_argument("--max_new_tokens", type=int, default=512)
    ap.add_argument("--topk", type=int, default=3)
    ap.add_argument("--device", type=str, default="cuda")
    args = ap.parse_args()

    logger.info("Loading eval data from %s", args.eval_data)
    df = pd.read_parquet(args.eval_data)
    logger.info("Loaded %d rows", len(df))
    if args.max_samples > 0:
        logger.info("Truncating to %d samples", args.max_samples)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    for step in args.steps:
        logger.info("=== Evaluating step %d ===", step)
        model = load_checkpoint(args.checkpoint_dir, step, args.base_model, device=args.device)
        metrics = eval_step(
            model=model,
            tokenizer=tokenizer,
            df=df,
            adapter_url=args.adapter_url,
            max_samples=args.max_samples if args.max_samples > 0 else len(df),
            max_turns=args.max_turns,
            max_new_tokens=args.max_new_tokens,
            topk=args.topk,
            device=args.device,
        )
        out_path = Path(args.output.format(step=step))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps({
            "step": step,
            "checkpoint_dir": str(args.checkpoint_dir),
            "eval_data": str(args.eval_data),
            "adapter_url": args.adapter_url,
            "metrics": {k: v for k, v in metrics.items() if k != "per_sample"},
            "per_sample": metrics["per_sample"],
        }, indent=2))
        logger.info(
            "Step %d: EM=%.3f CM=%.3f F1=%.3f avg_tools=%.2f -> %s",
            step, metrics["em"], metrics["contains"], metrics["f1"],
            metrics["avg_tool_calls"], out_path,
        )
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
