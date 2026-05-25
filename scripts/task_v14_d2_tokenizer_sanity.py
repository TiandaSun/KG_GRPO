"""V14-D2 tokenizer + chat-template sanity check for Qwen3-4B-Instruct-2507.

Verifies (per v13 pilot protocol) before kicking off the SFT pilot:
1. `Qwen/Qwen3-4B-Instruct-2507` never emits `<think>` tags on a 10-sample
   forward pass (non-thinking variant).
2. `tokenizer.apply_chat_template(..., enable_thinking=False)` returns usable
   ChatML with tool-role messages preserved.
3. Token-count diff vs Qwen2.5 tokenisation on the same 10 SFT rows
   (expect ~5-10% length delta from Qwen3 vocab updates).

Writes a JSON report suitable for the sanity gate, including:
- has_think_tags: bool (True if any of the 10 base-model generations contain <think>)
- qwen25_token_counts: list[int]
- qwen3_token_counts: list[int]
- length_delta_pct: mean abs percent difference
- chat_template_samples: first 500 chars of 3 rendered chat strings

Usage:
    python scripts/task_v14_d2_tokenizer_sanity.py \
        --qwen3_model Qwen/Qwen3-4B-Instruct-2507 \
        --qwen25_model Qwen/Qwen2.5-7B-Instruct \
        --sft_file data/freebase/sft_trajectories_pilot500.jsonl \
        --output results/phase7/v14_d2_tokenizer_check.json \
        --num_samples 10 \
        --generate_samples 10 \
        --do_generation
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def load_trajectories(path: Path, limit: int) -> list[list[dict[str, str]]]:
    """Load first `limit` trajectories from JSONL and return message lists."""
    records: list[list[dict[str, str]]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            traj = rec["trajectory"]
            records.append(
                [{"role": m["role"], "content": m["content"]} for m in traj]
            )
            if len(records) >= limit:
                break
    return records


def render_chat(tokenizer: Any, messages: list[dict[str, str]],
                enable_thinking: bool | None) -> str:
    """Apply chat template; tolerate tokenizers without enable_thinking kwarg."""
    kwargs: dict[str, Any] = {"tokenize": False, "add_generation_prompt": False}
    if enable_thinking is not None:
        kwargs["enable_thinking"] = enable_thinking
    try:
        return tokenizer.apply_chat_template(messages, **kwargs)
    except TypeError:
        kwargs.pop("enable_thinking", None)
        return tokenizer.apply_chat_template(messages, **kwargs)


def count_tokens(tokenizer: Any, text: str) -> int:
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])


def check_think_tags_on_base(model_name: str, num_samples: int,
                             max_new_tokens: int = 128) -> tuple[bool, list[str]]:
    """Generate `num_samples` short completions on simple prompts; flag any <think> tag.

    We use the base Qwen3-4B-Instruct-2507 (non-thinking variant). It must not
    emit <think> regardless of enable_thinking flag.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info("Loading model for <think>-tag sanity: %s", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager",
    )
    model.eval()

    prompts = [
        "What is the capital of France?",
        "Who wrote 'Pride and Prejudice'?",
        "Explain why the sky is blue in two sentences.",
        "List three primary colors.",
        "Compute 17 * 23.",
        "Name the largest planet in the solar system.",
        "Translate 'good morning' into Spanish.",
        "Give a one-sentence biography of Marie Curie.",
        "What is photosynthesis?",
        "Summarise the plot of Romeo and Juliet in two sentences.",
    ][:num_samples]

    generations: list[str] = []
    has_think = False
    for p in prompts:
        messages = [{"role": "user", "content": p}]
        text = render_chat(tokenizer, messages, enable_thinking=False)
        # Add generation prompt so we actually get a reply:
        kwargs: dict[str, Any] = {"tokenize": False, "add_generation_prompt": True}
        try:
            text = tokenizer.apply_chat_template(
                messages, enable_thinking=False, **kwargs
            )
        except TypeError:
            text = tokenizer.apply_chat_template(messages, **kwargs)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=1.0,
                top_p=1.0,
                pad_token_id=tokenizer.eos_token_id,
            )
        new_ids = out[0, inputs["input_ids"].shape[1]:]
        decoded = tokenizer.decode(new_ids, skip_special_tokens=False)
        generations.append(decoded)
        if "<think>" in decoded or "</think>" in decoded:
            has_think = True

    return has_think, generations


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    parser = argparse.ArgumentParser(description="V14-D2 tokenizer sanity check")
    parser.add_argument("--qwen3_model", type=str,
                        default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--qwen25_model", type=str,
                        default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--sft_file", type=Path,
                        default=Path("data/freebase/sft_trajectories_pilot500.jsonl"))
    parser.add_argument("--output", type=Path,
                        default=Path("results/phase7/v14_d2_tokenizer_check.json"))
    parser.add_argument("--num_samples", type=int, default=10,
                        help="Number of SFT rows for tokenisation comparison")
    parser.add_argument("--generate_samples", type=int, default=10,
                        help="Number of prompts for <think>-tag forward pass")
    parser.add_argument("--do_generation", action="store_true",
                        help="Run base-model forward pass (requires GPU).")
    parser.add_argument("--max_new_tokens", type=int, default=128)
    args = parser.parse_args()

    # Lazy import so --help works without torch.
    from transformers import AutoTokenizer

    logger.info("Loading tokenizers")
    tok_q3 = AutoTokenizer.from_pretrained(args.qwen3_model)
    tok_q25 = AutoTokenizer.from_pretrained(args.qwen25_model)

    logger.info("Loading %d trajectories from %s", args.num_samples, args.sft_file)
    trajectories = load_trajectories(args.sft_file, args.num_samples)
    if not trajectories:
        raise RuntimeError(f"No trajectories loaded from {args.sft_file}")

    # Render chat templates and count tokens for both tokenizers.
    q3_texts: list[str] = []
    q25_texts: list[str] = []
    q3_counts: list[int] = []
    q25_counts: list[int] = []
    for messages in trajectories:
        q3_text = render_chat(tok_q3, messages, enable_thinking=False)
        q25_text = render_chat(tok_q25, messages, enable_thinking=None)
        q3_texts.append(q3_text)
        q25_texts.append(q25_text)
        q3_counts.append(count_tokens(tok_q3, q3_text))
        q25_counts.append(count_tokens(tok_q25, q25_text))

    # Percent length delta per row (abs), then mean.
    deltas_pct: list[float] = []
    for q3, q25 in zip(q3_counts, q25_counts):
        if q25 == 0:
            continue
        deltas_pct.append(abs(q3 - q25) / q25 * 100.0)
    length_delta_pct = sum(deltas_pct) / len(deltas_pct) if deltas_pct else 0.0

    # Optional: base-model forward pass for <think>-tag check.
    has_think_tags = False
    sample_generations: list[str] = []
    if args.do_generation:
        has_think_tags, sample_generations = check_think_tags_on_base(
            args.qwen3_model,
            num_samples=args.generate_samples,
            max_new_tokens=args.max_new_tokens,
        )

    report: dict[str, Any] = {
        "qwen3_model": args.qwen3_model,
        "qwen25_model": args.qwen25_model,
        "num_samples": args.num_samples,
        "has_think_tags": has_think_tags,
        "qwen25_token_counts": q25_counts,
        "qwen3_token_counts": q3_counts,
        "length_delta_pct": length_delta_pct,
        "chat_template_samples": [t[:500] for t in q3_texts[:3]],
        "generation_samples_qwen3": sample_generations,
        "generation_ran": args.do_generation,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    logger.info("Report written: %s", args.output)
    logger.info("has_think_tags=%s  length_delta_pct=%.2f%%",
                has_think_tags, length_delta_pct)
    logger.info("Qwen2.5 token counts: %s", q25_counts)
    logger.info("Qwen3   token counts: %s", q3_counts)

    # Non-zero exit on hard failure (think tags emitted on non-thinking model).
    if args.do_generation and has_think_tags:
        logger.error("SANITY FAIL: Qwen3-4B-Instruct-2507 emitted <think> tags "
                     "— this contradicts the v13 non-thinking expectation.")
        raise SystemExit(2)
    if length_delta_pct > 30.0:
        logger.error("SANITY FAIL: tokenisation length delta %.2f%% > 30%%",
                     length_delta_pct)
        raise SystemExit(3)


if __name__ == "__main__":
    main()
