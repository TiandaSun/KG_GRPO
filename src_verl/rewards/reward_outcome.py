"""R_outcome: Outcome-only reward (Baseline 1).

Scores only the final answer correctness. No process supervision.
This is the simplest reward and serves as a baseline to measure
whether process rewards add value.

Score = 0.5 * EM + 0.5 * F1
Range: [0.0, 1.0]
"""

from __future__ import annotations

from typing import Any

from src_verl.rewards.common import (
    compute_exact_match,
    compute_token_f1,
    extract_answer,
)


def compute_reward_outcome(
    trajectory: list[dict[str, str]],
    gold_answer: str,
    **kwargs: Any,
) -> float:
    """Compute outcome-only reward.

    Args:
        trajectory: Full multi-turn conversation.
        gold_answer: Expected short answer string.

    Returns:
        Reward in [0.0, 1.0].
    """
    predicted = extract_answer(trajectory)
    if not predicted or not gold_answer:
        return 0.0

    em = compute_exact_match(predicted, gold_answer)
    f1 = compute_token_f1(predicted, gold_answer)

    return 0.5 * em + 0.5 * f1
