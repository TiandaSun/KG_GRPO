"""Variant G Step 2: filter + SFT-format conversion of 39B self-distill trajectories.

Reads the full-sweep output from `phase7_variant_g_full_sweep.py` (or the pilot),
applies the strict (or widened) filter, and converts the surviving trajectories
into the JSONL format expected by src_verl/training/sft_multiturn.py.

Output schema (matches data/freebase/sft_trajectories.jsonl):
  {
    "question": str,
    "gold_answer": str,
    "sample_id": str,
    "trajectory": [
      {"role": "system", "content": "..."},
      {"role": "user", "content": "..."},
      {"role": "assistant", "content": "<think>...</think><search>...</search>"},
      {"role": "tool", "content": "..."},
      ...
      {"role": "assistant", "content": "<think>...</think><answer>...</answer>"}
    ]
  }

For the Variant G yield pilot (only 2K samples, full_response stored directly),
we produce a simpler trajectory that wraps the full_response as a single
assistant turn — since we don't have the structured messages, we fall back to
a "single assistant turn" representation which SFTTrainer can still train on.
"""
from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)


def convert_one(rec: dict, system_prompt: str) -> dict | None:
    """Convert one pilot record to SFT trajectory format."""
    full = rec.get("full_response") or ""
    if not full:
        return None
    # Reconstruct the multi-turn trajectory by splitting on tool_response blocks
    trajectory: list[dict] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": rec["question"]},
    ]
    # Split the full response into assistant/tool alternation
    parts = re.split(r"(<tool_response>.*?</tool_response>)", full, flags=re.DOTALL)
    current_assistant_buf: list[str] = []
    for part in parts:
        if part.startswith("<tool_response>"):
            # flush pending assistant turn first
            if current_assistant_buf:
                trajectory.append(
                    {"role": "assistant", "content": "".join(current_assistant_buf).strip()}
                )
                current_assistant_buf = []
            tool_content = re.sub(r"<tool_response>|</tool_response>", "", part).strip()
            trajectory.append({"role": "tool", "content": tool_content})
        else:
            current_assistant_buf.append(part)
    if current_assistant_buf:
        tail = "".join(current_assistant_buf).strip()
        if tail:
            trajectory.append({"role": "assistant", "content": tail})

    return {
        "question": rec["question"],
        "gold_answer": rec.get("gold", ""),
        "sample_id": rec.get("sample_id", ""),
        "trajectory": trajectory,
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description="Variant G Step 2: filter + SFT conversion")
    parser.add_argument(
        "--input", type=Path,
        default=Path("data/freebase/verl_cwq/39b_pilot_yield.json"),
        help="Pilot or full-sweep JSON with per_sample records",
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("data/freebase/verl_cwq/39b_self_distill.jsonl"),
    )
    parser.add_argument(
        "--filter", type=str, choices=["strict", "widened"], default="strict",
        help="strict = EM=1 AND ntools>=1 AND format_valid; widened = also accept F1>=0.8",
    )
    parser.add_argument(
        "--train_parquet", type=Path,
        default=Path("data/freebase/verl_cwq/train.parquet"),
        help="Used only to extract the CWQ system prompt",
    )
    args = parser.parse_args()

    # Pull the system prompt
    import pandas as pd
    df = pd.read_parquet(args.train_parquet)
    system_prompt = df.iloc[0]["prompt"][0]["content"]

    data = json.load(open(args.input))
    records = data.get("per_sample") if isinstance(data, dict) else data
    logger.info("Loaded %d source records from %s", len(records), args.input)

    field = "strict_pass" if args.filter == "strict" else "widened_pass"
    kept = [r for r in records if r.get(field)]
    logger.info("After %s filter: %d kept (%.1f%%)", args.filter, len(kept), 100 * len(kept) / max(1, len(records)))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    skipped = 0
    with open(args.output, "w") as f:
        for rec in kept:
            conv = convert_one(rec, system_prompt)
            if conv is None:
                skipped += 1
                continue
            f.write(json.dumps(conv) + "\n")

    logger.info("Wrote %d trajectories (skipped %d) to %s", len(kept) - skipped, skipped, args.output)


if __name__ == "__main__":
    main()
