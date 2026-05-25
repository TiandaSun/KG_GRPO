"""Filter SFT trajectories to keep only 'correct-via-tool' examples.

Criterion: The gold answer (case-insensitive, normalized) must appear in at least
one <tool_response>...</tool_response> block in the trajectory. This filters out
'pantomime' trajectories where the model called tools but the answer was hallucinated.

Used for Task 39C2-data: SFT replay with higher-quality filtered reminders.

Usage:
    python scripts/filter_sft_trajectories.py \
        --input data/freebase/sft_trajectories.jsonl \
        --output data/freebase/sft_trajectories_correct_via_tool.jsonl
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import string
from pathlib import Path

logger = logging.getLogger(__name__)

_TOOL_RESP_RE = re.compile(r"<tool_response>(.*?)</tool_response>", re.DOTALL)
_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)


def normalize(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    return " ".join(text.split())


def answer_in_tool_responses(trajectory: list[dict[str, str]]) -> tuple[bool, str]:
    """Check if the gold answer appears in any tool response.

    Returns (is_correct_via_tool, gold_answer_normalized).
    """
    gold = ""
    tool_text_parts = []

    for msg in trajectory:
        content = msg.get("content", "")
        if msg.get("role") == "assistant":
            m = _ANSWER_RE.search(content)
            if m:
                gold = m.group(1).strip()
        # Tool responses can appear in user messages
        for tm in _TOOL_RESP_RE.finditer(content):
            tool_text_parts.append(tm.group(1))

    if not gold:
        return False, ""

    gold_norm = normalize(gold)
    if not gold_norm:
        return False, ""

    tool_text_norm = normalize(" ".join(tool_text_parts))
    # Substring match on normalized tokens (token boundaries respected via join)
    return gold_norm in tool_text_norm, gold_norm


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    parser = argparse.ArgumentParser(description="Filter SFT trajectories by correct-via-tool")
    parser.add_argument("--input", type=Path, default=Path("data/freebase/sft_trajectories.jsonl"))
    parser.add_argument("--output", type=Path,
                        default=Path("data/freebase/sft_trajectories_correct_via_tool.jsonl"))
    args = parser.parse_args()

    n_total = 0
    n_kept = 0
    n_empty_gold = 0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.input) as fin, open(args.output, "w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            n_total += 1
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            trajectory = data.get("trajectory", [])
            if not trajectory:
                continue
            ok, gold = answer_in_tool_responses(trajectory)
            if not gold:
                n_empty_gold += 1
                continue
            if ok:
                fout.write(json.dumps(data) + "\n")
                n_kept += 1

    keep_pct = 100 * n_kept / max(n_total, 1)
    logger.info("Total trajectories: %d", n_total)
    logger.info("Missing gold answer: %d", n_empty_gold)
    logger.info("Kept (correct-via-tool): %d (%.1f%%)", n_kept, keep_pct)
    logger.info("Saved to: %s", args.output)


if __name__ == "__main__":
    main()
