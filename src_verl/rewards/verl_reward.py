"""Custom reward function for verl GRPO training on ConceptNet KG data.

Registered via config: custom_reward_function.path / custom_reward_function.name

verl calls: compute_score(data_source, solution_str, ground_truth, extra_info=None)

IMPORTANT: verl's standard NaiveRewardManager does NOT pass tool_rewards from
ToolAgentLoop to compute_score (only the experimental NaiveRewardLoopManager does).
Therefore this reward function parses the flat solution_str directly for <search>
tags to detect tool calls, rather than relying on extra_info["tool_rewards"].

Reward components:
    r_answer   (0–1.0): Token-F1 against gold answer (extracted from <answer> tags)
    r_coverage (0–1.0): KG path entity/relation mention coverage
    r_format   (0/0.5): Proper <answer>...</answer> tags with content
    r_tool_use (0–1.5): Reward for tool calls detected in solution_str
    r_no_tool  (-1.0/0): Penalty if no tool calls detected at all
"""

from __future__ import annotations

import logging
import re
from collections import Counter

logger = logging.getLogger(__name__)


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict | None = None,
    **kwargs,
) -> dict:
    """Compute reward for KG reasoning responses.

    Returns dict with 'score' key (required by verl's NaiveRewardManager).
    """
    if extra_info is None:
        extra_info = {}

    kg_path = extra_info.get("kg_path", [])

    # --- Parse tool calls from solution_str directly ---
    # verl's NaiveRewardManager does not bridge tool_rewards from
    # ToolAgentLoop to extra_info. We detect tool use from the text.
    search_calls = re.findall(r"<search>(.*?)</search>", solution_str, re.DOTALL)
    num_tool_calls = len(search_calls)

    # Also check extra_info as fallback (in case NaiveRewardLoopManager is used)
    tool_rewards_from_loop = extra_info.get("tool_rewards", [])
    if hasattr(tool_rewards_from_loop, "tolist"):
        tool_rewards_from_loop = tool_rewards_from_loop.tolist()
    num_tool_calls_from_loop = len(tool_rewards_from_loop) if tool_rewards_from_loop else 0
    # Use whichever gives more information
    num_tool_calls = max(num_tool_calls, num_tool_calls_from_loop)

    # --- Answer reward (continuous via token F1) ---
    r_answer = _token_f1(solution_str, ground_truth)

    # --- KG path coverage reward ---
    r_coverage = 0.0
    if kg_path:
        path_tokens: set[str] = set()
        for triple in kg_path:
            for token in triple:
                path_tokens.update(token.lower().split())

        output_lower = solution_str.lower()
        output_words = set(output_lower.split())
        hits = len(path_tokens & output_words)

        if hits >= 2:
            r_coverage = min(hits / len(path_tokens), 1.0)

    # --- Format reward (uses <answer> tags for verl multi-turn) ---
    r_format = 0.0
    if "<answer>" in solution_str and "</answer>" in solution_str:
        answer_content = solution_str.split("<answer>")[1].split("</answer>")[0].strip()
        if len(answer_content) > 5:
            r_format = 0.5

    # --- Tool-use rewards (parsed from solution_str) ---
    r_tool_use = 0.0
    if num_tool_calls > 0:
        # Base bonus for using tools at all
        r_tool_use += 0.5

        # Quality bonus: reward valid tool call syntax
        valid_actions = {"get_tail_relations", "get_head_relations",
                         "get_tail_entities", "get_head_entities"}
        # Also accept aliases that the parser can resolve
        alias_actions = {"kg_query", "search", "query", "get_relations", "get_entities"}
        valid_calls = 0
        for call_str in search_calls:
            func_match = re.match(r"(\w+)\(", call_str.strip())
            if func_match and func_match.group(1) in (valid_actions | alias_actions):
                valid_calls += 1

        if valid_calls > 0:
            # Up to 0.5 for valid tool calls (diminishing returns after 3)
            r_tool_use += min(valid_calls / 3.0, 1.0) * 0.5

        # If step rewards are available from the loop, add them
        if num_tool_calls_from_loop > 0:
            r_steps_avg = sum(tool_rewards_from_loop) / num_tool_calls_from_loop
            r_tool_use += r_steps_avg * 0.5  # up to 0.5 extra from step quality

    # --- No-tool penalty ---
    # Strong penalty for outputs that don't use any tools.
    # This creates gradient pressure toward tool use in GRPO.
    r_no_tool = -1.0 if num_tool_calls == 0 else 0.0

    total = r_answer + r_coverage + r_format + r_tool_use + r_no_tool

    return {
        "score": total,
        "r_answer": r_answer,
        "r_coverage": r_coverage,
        "r_format": r_format,
        "r_tool_use": r_tool_use,
        "r_no_tool": r_no_tool,
        "num_tool_calls": float(num_tool_calls),
    }


def _token_f1(prediction: str, ground_truth: str) -> float:
    """Token-level F1 between prediction and ground truth."""
    pred_tokens = _normalize(prediction).split()
    gt_tokens = _normalize(ground_truth).split()

    if not gt_tokens:
        return 1.0 if not pred_tokens else 0.0
    if not pred_tokens:
        return 0.0

    pred_counts = Counter(pred_tokens)
    gt_counts = Counter(gt_tokens)
    common_count = sum((pred_counts & gt_counts).values())
    if common_count == 0:
        return 0.0

    precision = common_count / len(pred_tokens)
    recall = common_count / len(gt_tokens)
    return 2 * precision * recall / (precision + recall)


def _normalize(text: str) -> str:
    """Lowercase and strip punctuation, extracting answer content from tags."""
    text = text.lower()
    # Extract content BETWEEN <answer>...</answer> tags if present
    if "<answer>" in text and "</answer>" in text:
        text = text.split("<answer>")[1].split("</answer>")[0]
    elif "</think>" in text:
        # Extract text after </think> (the final answer portion)
        text = text.split("</think>")[-1]
    text = re.sub(r"[^\w\s]", " ", text)
    return " ".join(text.split())
