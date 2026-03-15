"""Shared utilities for reward functions.

Provides parsing helpers and scoring functions reused across all three
reward variants (outcome, heuristic, verifiable).
"""

from __future__ import annotations

import re
import string
from collections import Counter
from dataclasses import dataclass
from typing import Any


@dataclass
class Step:
    """A single step in a multi-turn agent trajectory."""

    turn_index: int
    role: str  # "assistant" or "tool"
    thinking: str  # content of <think>...</think> if present
    action: str  # the tool call, e.g. "get_tail_relations(dog)"
    action_type: str  # "get_tail_relations", "get_head_relations", etc.
    action_args: dict[str, str]  # parsed arguments
    observation: str  # tool response content
    raw_content: str  # full message content


def parse_steps(trajectory: list[dict[str, str]]) -> list[Step]:
    """Parse multi-turn conversation into Step objects.

    Expected trajectory format:
    [
        {"role": "user", "content": "Question..."},
        {"role": "assistant", "content": "<think>...</think>\n<search>get_tail_relations(dog)</search>"},
        {"role": "tool", "content": "<information>['IsA', 'HasProperty']</information>"},
        {"role": "assistant", "content": "<think>...</think>\n<search>get_tail_entities(dog, IsA)</search>"},
        {"role": "tool", "content": "<information>['animal', 'pet']</information>"},
        {"role": "assistant", "content": "<think>...</think>\n<answer>animal</answer>"},
    ]

    Returns list of Steps, one per assistant message that contains an action.
    """
    steps: list[Step] = []
    i = 0
    turn_idx = 0

    while i < len(trajectory):
        msg = trajectory[i]

        if msg["role"] == "assistant":
            content = msg.get("content", "")
            thinking = extract_think(content)
            action, action_type, action_args = parse_action(content)

            # Find the next tool response if any
            observation = ""
            if i + 1 < len(trajectory) and trajectory[i + 1]["role"] == "tool":
                obs_content = trajectory[i + 1].get("content", "")
                observation = extract_information(obs_content)
                i += 1  # Skip the tool message

            steps.append(
                Step(
                    turn_index=turn_idx,
                    role="assistant",
                    thinking=thinking,
                    action=action,
                    action_type=action_type,
                    action_args=action_args,
                    observation=observation,
                    raw_content=content,
                )
            )
            turn_idx += 1

        i += 1

    return steps


def extract_think(content: str) -> str:
    """Extract content between <think>...</think> tags."""
    match = re.search(r"<think>(.*?)</think>", content, re.DOTALL)
    return match.group(1).strip() if match else ""


def extract_answer(trajectory: list[dict[str, str]]) -> str:
    """Extract the final answer from <answer>...</answer> in the last assistant message."""
    for msg in reversed(trajectory):
        if msg["role"] == "assistant":
            content = msg.get("content", "")
            match = re.search(r"<answer>(.*?)</answer>", content, re.DOTALL)
            if match:
                return match.group(1).strip()
            # Fallback: text after </think> in the last message
            if "</think>" in content:
                return content.split("</think>")[-1].strip()
            return content.strip()
    return ""


def extract_information(content: str) -> str:
    """Extract content from <information>...</information> tags."""
    match = re.search(r"<information>(.*?)</information>", content, re.DOTALL)
    return match.group(1).strip() if match else content.strip()


def parse_action(content: str) -> tuple[str, str, dict[str, str]]:
    """Parse a tool call from assistant message.

    Looks for <search>action_type(arg1, arg2)</search> pattern.

    Returns:
        (full_action_string, action_type, {arg_name: arg_value})
    """
    match = re.search(r"<search>(.*?)</search>", content, re.DOTALL)
    if not match:
        # Check for <answer> tag (final step)
        answer_match = re.search(r"<answer>(.*?)</answer>", content, re.DOTALL)
        if answer_match:
            return f"answer({answer_match.group(1).strip()})", "answer", {"answer": answer_match.group(1).strip()}
        return "", "none", {}

    action_str = match.group(1).strip()

    # Parse function call: action_type(arg1, arg2)
    func_match = re.match(r"(\w+)\((.*)\)", action_str, re.DOTALL)
    if not func_match:
        return action_str, "unknown", {}

    action_type = func_match.group(1)
    args_str = func_match.group(2).strip()

    # Parse arguments
    args: dict[str, str] = {}
    if args_str:
        parts = [p.strip().strip("'\"") for p in args_str.split(",")]
        if action_type in ("get_tail_relations", "get_head_relations"):
            args["entity"] = parts[0] if parts else ""
        elif action_type in ("get_tail_entities", "get_head_entities"):
            args["entity"] = parts[0] if parts else ""
            args["relation"] = parts[1] if len(parts) > 1 else ""

    return action_str, action_type, args


def normalize_answer(text: str) -> str:
    """Normalize answer text for EM/F1 (SQuAD convention)."""
    text = text.lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = " ".join(text.split())
    return text


def compute_exact_match(prediction: str, gold: str) -> float:
    """Exact match after normalization."""
    return 1.0 if normalize_answer(prediction) == normalize_answer(gold) else 0.0


def compute_token_f1(prediction: str, gold: str) -> float:
    """Token-level F1 between prediction and gold."""
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(gold).split()

    if not gold_tokens:
        return 1.0 if not pred_tokens else 0.0
    if not pred_tokens:
        return 0.0

    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_common = sum(common.values())

    if num_common == 0:
        return 0.0

    precision = num_common / len(pred_tokens)
    recall = num_common / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def extract_entities_from_observation(observation: str) -> list[str]:
    """Extract entity/relation names from a tool observation string.

    Observations are typically formatted as Python list strings:
    "['dog', 'cat', 'animal']" or "['IsA', 'HasProperty']"
    """
    # Try to parse as Python literal
    try:
        import ast
        result = ast.literal_eval(observation)
        if isinstance(result, list):
            return [str(item) for item in result]
    except (ValueError, SyntaxError):
        pass

    # Fallback: extract quoted strings
    matches = re.findall(r"'([^']+)'|\"([^\"]+)\"", observation)
    return [m[0] or m[1] for m in matches]
