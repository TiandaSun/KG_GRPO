"""Reward functions for verl GRPO training (spec v3).

Implements 4 reward variants as defined in hpc_implementation_spec_v3.md Section 5:
  - R_outcome:    0.5*EM + 0.5*F1 (answer only)
  - R_heuristic:  0.3*R_outcome + 0.7*step_entity_overlap
  - R_verifiable: 0.3*R_outcome + 0.7*(r_on_path + r_progress + r_coherence + r_valid)
  - R_random:     0.3*R_outcome + 0.7*random_step_rewards

The reward type is selected by the REWARD_TYPE env var or defaults to "outcome".
No explicit tool bonuses or penalties — tool use is incentivized implicitly
via the 0.30/0.70 answer/step split.

verl calls: compute_score(data_source, solution_str, ground_truth, extra_info=None)
"""

from __future__ import annotations

import logging
import os
import random
import re
from collections import Counter
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def _to_python(obj: Any) -> Any:
    """Recursively convert numpy types to native Python types."""
    if isinstance(obj, np.ndarray):
        return [_to_python(x) for x in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.str_,)):
        return str(obj)
    return obj

REWARD_TYPE = os.getenv("REWARD_TYPE", "outcome")


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict | None = None,
    **kwargs: Any,
) -> dict:
    """Compute reward for KG reasoning responses.

    Returns dict with 'score' key (required by verl's NaiveRewardManager).
    Dispatches to the reward variant specified by REWARD_TYPE env var.
    """
    if extra_info is None:
        extra_info = {}

    reward_type = extra_info.get("reward_type", REWARD_TYPE)
    kg_path = extra_info.get("kg_path", [])
    all_answers = extra_info.get("all_answers", [ground_truth] if ground_truth else [])

    # Deep-convert numpy arrays to Python types (parquet stores nested ndarrays)
    kg_path = _to_python(kg_path)
    all_answers = _to_python(all_answers)
    if isinstance(all_answers, str):
        all_answers = [all_answers]

    # Parse tool call steps from solution_str
    steps = _parse_steps(solution_str)
    num_tool_calls = len(steps)

    # Compute R_outcome (shared across all reward types)
    predicted = _extract_answer(solution_str)
    em = _exact_match(predicted, ground_truth, all_answers)
    f1 = _token_f1(predicted, ground_truth)
    r_outcome = 0.5 * em + 0.5 * f1

    if reward_type == "outcome":
        return {
            "score": r_outcome,
            "r_outcome": r_outcome,
            "em": em,
            "f1": f1,
            "num_tool_calls": float(num_tool_calls),
        }

    elif reward_type == "heuristic":
        step_rewards = _heuristic_step_rewards(steps, ground_truth, all_answers)
        avg_step = sum(step_rewards) / max(len(step_rewards), 1)
        total = 0.3 * r_outcome + 0.7 * avg_step
        return {
            "score": total,
            "r_outcome": r_outcome,
            "r_step_avg": avg_step,
            "em": em,
            "f1": f1,
            "num_tool_calls": float(num_tool_calls),
        }

    elif reward_type == "verifiable":
        step_rewards = _verifiable_step_rewards(steps, kg_path, extra_info)
        avg_step = sum(step_rewards) / max(len(step_rewards), 1)
        total = 0.3 * r_outcome + 0.7 * avg_step
        return {
            "score": total,
            "r_outcome": r_outcome,
            "r_step_avg": avg_step,
            "em": em,
            "f1": f1,
            "num_tool_calls": float(num_tool_calls),
        }

    elif reward_type == "random":
        step_rewards = [random.uniform(0, 1) for _ in steps] if steps else []
        avg_step = sum(step_rewards) / max(len(step_rewards), 1)
        total = 0.3 * r_outcome + 0.7 * avg_step
        return {
            "score": total,
            "r_outcome": r_outcome,
            "r_step_avg": avg_step,
            "em": em,
            "f1": f1,
            "num_tool_calls": float(num_tool_calls),
        }

    elif reward_type == "retrieval_grounded":
        step_rewards = _retrieval_grounded_step_rewards(steps, predicted)
        avg_step = sum(step_rewards) / max(len(step_rewards), 1)
        total = 0.3 * r_outcome + 0.7 * avg_step
        return {
            "score": total,
            "r_outcome": r_outcome,
            "r_step_avg": avg_step,
            "em": em,
            "f1": f1,
            "num_tool_calls": float(num_tool_calls),
        }

    elif reward_type == "tool_type_bonus":
        step_rewards = _tool_type_bonus_step_rewards(steps)
        avg_step = sum(step_rewards) / max(len(step_rewards), 1)
        total = 0.3 * r_outcome + 0.7 * avg_step
        return {
            "score": total,
            "r_outcome": r_outcome,
            "r_step_avg": avg_step,
            "em": em,
            "f1": f1,
            "num_tool_calls": float(num_tool_calls),
        }

    else:
        logger.warning("Unknown reward_type=%s, falling back to outcome", reward_type)
        return {
            "score": r_outcome,
            "r_outcome": r_outcome,
            "em": em,
            "f1": f1,
            "num_tool_calls": float(num_tool_calls),
        }


# ---------------------------------------------------------------------------
# Step parsing
# ---------------------------------------------------------------------------

def _parse_steps(solution_str: str) -> list[dict[str, str]]:
    """Parse tool call + observation pairs from the flat solution_str.

    Each step = one <search>action(args)</search> followed by a tool response.
    Returns list of dicts with 'call' (the search content) and 'observation' (tool response).
    """
    steps: list[dict[str, str]] = []
    # Find all <search>...</search> and try to pair with following tool responses
    search_spans = list(re.finditer(r"<search>(.*?)</search>", solution_str, re.DOTALL))

    for i, match in enumerate(search_spans):
        call = match.group(1).strip()
        # Observation is text between this search end and next search start (or end of string)
        obs_start = match.end()
        obs_end = search_spans[i + 1].start() if i + 1 < len(search_spans) else len(solution_str)
        observation = solution_str[obs_start:obs_end].strip()
        # Extract tool_response content if present
        resp_match = re.search(r"<tool_response>(.*?)</tool_response>", observation, re.DOTALL)
        if resp_match:
            observation = resp_match.group(1).strip()
        steps.append({"call": call, "observation": observation})

    return steps


# ---------------------------------------------------------------------------
# R_heuristic step rewards
# ---------------------------------------------------------------------------

def _heuristic_step_rewards(
    steps: list[dict[str, str]],
    ground_truth: str,
    all_answers: list[str],
) -> list[float]:
    """ProGraph-R1 style: entity overlap between retrieved results and gold answer."""
    if not steps:
        return []

    gt_entities = set()
    for ans in all_answers:
        gt_entities.update(_normalize(str(ans)).split())

    rewards: list[float] = []
    for step in steps:
        obs = step["observation"]
        retrieved_tokens = set(_normalize(obs).split())
        if not retrieved_tokens:
            rewards.append(0.0)
            continue
        overlap = len(retrieved_tokens & gt_entities)
        r_overlap = overlap / max(len(retrieved_tokens), 1)
        r_reach = 1.0 if overlap > 0 else 0.0
        rewards.append(0.5 * r_overlap + 0.5 * r_reach)

    return rewards


# ---------------------------------------------------------------------------
# R_retrieval_grounded step rewards (E5a)
# ---------------------------------------------------------------------------

def _parse_action(call: str) -> tuple[str, list[str]]:
    """Parse action name and args from a tool call string like 'get_tail_entities(entity, relation)'."""
    match = re.match(r"(\w+)\(([^)]*)\)", call.strip())
    if not match:
        return "", []
    action = match.group(1)
    args = [a.strip().strip("'\"") for a in match.group(2).split(",") if a.strip()]
    return action, args


def _retrieval_grounded_step_rewards(
    steps: list[dict[str, str]],
    predicted_answer: str,
) -> list[float]:
    """E5a: Reward entity retrieval tools more, especially when output appears in answer."""
    if not steps:
        return []

    pred_lower = _normalize(predicted_answer)
    rewards: list[float] = []

    for step in steps:
        action, args = _parse_action(step["call"])
        obs = step["observation"]

        if action in ("get_tail_entities", "get_head_entities"):
            # Entity retrieval — check if any retrieved entity appears in predicted answer
            obs_entities = [e.strip().strip('"') for e in obs.strip("[]").split(",") if e.strip()]
            used_in_answer = any(
                _normalize(ent) in pred_lower for ent in obs_entities if ent.strip()
            )
            if used_in_answer:
                rewards.append(1.0)     # Retrieved entity used in answer
            elif len(obs_entities) > 0 and obs.strip() not in ("[]", "", "No results found"):
                rewards.append(0.3)     # Retrieved but not used
            else:
                rewards.append(0.0)     # Empty result

        elif action in ("get_tail_relations", "get_head_relations"):
            # Relation discovery — lower reward (exploration only)
            if obs.strip() and obs.strip() not in ("[]", "No results found"):
                rewards.append(0.2)     # Relations found
            else:
                rewards.append(0.0)     # Empty

        else:
            rewards.append(0.0)  # Unknown action

    return rewards


# ---------------------------------------------------------------------------
# R_tool_type_bonus step rewards (E5b)
# ---------------------------------------------------------------------------

def _tool_type_bonus_step_rewards(
    steps: list[dict[str, str]],
) -> list[float]:
    """E5b: Reward entity retrieval tools more than relation tools (simpler than E5a)."""
    if not steps:
        return []

    tool_type_weights = {
        "get_tail_entities": 1.0,
        "get_head_entities": 1.0,
        "get_tail_relations": 0.3,
        "get_head_relations": 0.3,
    }

    rewards: list[float] = []
    for step in steps:
        action, _ = _parse_action(step["call"])
        weight = tool_type_weights.get(action, 0.0)
        obs = step["observation"]
        if obs.strip() and obs.strip() not in ("[]", "No results found"):
            rewards.append(weight)
        else:
            rewards.append(0.0)

    return rewards


# ---------------------------------------------------------------------------
# R_verifiable step rewards
# ---------------------------------------------------------------------------

def _verifiable_step_rewards(
    steps: list[dict[str, str]],
    kg_path: list,
    extra_info: dict,
) -> list[float]:
    """KG-verifiable step rewards: r_on_path, r_progress, r_coherence, r_valid.

    Weights: r_on_path=0.45, r_progress=0.30, r_coherence=0.15, r_valid=0.10
    """
    if not steps:
        return []

    # Build set of path triples for r_on_path
    path_triples: set[tuple[str, str, str]] = set()
    path_entities: set[str] = set()
    for triple in (kg_path or []):
        if hasattr(triple, '__len__') and len(triple) >= 3:
            h, r, t = str(triple[0]).lower(), str(triple[1]).lower(), str(triple[2]).lower()
            path_triples.add((h, r, t))
            path_entities.add(h)
            path_entities.add(t)

    # Extract entities for coherence tracking
    query_entities = extra_info.get("query_entities", [])
    query_entities = _to_python(query_entities)
    prev_entities: set[str] = set(str(e).lower() for e in query_entities) if query_entities else set()

    rewards: list[float] = []
    for step in steps:
        obs = step["observation"]
        call = step["call"]

        # Parse entities from the tool call
        call_match = re.match(r"(\w+)\(([^)]*)\)", call.strip())
        current_entities: set[str] = set()
        if call_match:
            args = [a.strip().strip("'\"").lower() for a in call_match.group(2).split(",") if a.strip()]
            current_entities.update(args)

        # Also extract entities from observation
        obs_entities = set(_normalize(obs).split())

        # r_on_path: do any observed triples appear in the gold KG path?
        r_on_path = 0.0
        if path_triples and obs:
            obs_lower = obs.lower()
            for h, r, t in path_triples:
                if h in obs_lower and t in obs_lower:
                    r_on_path = 1.0
                    break
            if r_on_path == 0.0:
                # Partial credit: any path entity mentioned in observation?
                obs_words = set(obs_lower.split())
                hits = len(path_entities & obs_words)
                if hits > 0:
                    r_on_path = min(hits / max(len(path_entities), 1), 0.5)

        # r_progress: simplified — did we get closer to answer entities?
        # Without precomputed distances, use entity overlap as proxy
        answer_entities = extra_info.get("answer_entities", [])
        answer_entities = _to_python(answer_entities)
        r_progress = 0.0
        if answer_entities:
            a_set = set(str(e).lower() for e in answer_entities)
            if current_entities & a_set or obs_entities & a_set:
                r_progress = 1.0
            elif path_entities:
                obs_path_hits = len(obs_entities & path_entities)
                r_progress = min(obs_path_hits / max(len(path_entities), 1), 1.0)

        # r_coherence: step shares entity with previous step?
        r_coherence = 0.0
        if prev_entities and current_entities:
            if prev_entities & current_entities:
                r_coherence = 1.0

        # r_valid: query returned non-empty results?
        r_valid = 1.0 if len(obs.strip()) > 10 else 0.0

        step_reward = (
            0.45 * r_on_path
            + 0.30 * r_progress
            + 0.15 * r_coherence
            + 0.10 * r_valid
        )
        rewards.append(step_reward)

        # Update prev_entities for next step's coherence check
        prev_entities = current_entities | obs_entities

    return rewards


# ---------------------------------------------------------------------------
# Answer extraction and metrics
# ---------------------------------------------------------------------------

def _extract_answer(text: str) -> str:
    """Extract answer from <answer>...</answer> tags."""
    if "<answer>" in text and "</answer>" in text:
        return text.split("<answer>")[1].split("</answer>")[0].strip()
    # Fallback: text after last </think>
    if "</think>" in text:
        return text.split("</think>")[-1].strip()
    return text.strip()


def _exact_match(predicted: str, ground_truth: str, all_answers: list | None = None) -> float:
    """Check if predicted answer exactly matches any gold answer."""
    pred_norm = _normalize(predicted)
    if not pred_norm:
        return 0.0
    # Check against primary answer
    if pred_norm == _normalize(ground_truth):
        return 1.0
    # Check against aliases
    if all_answers:
        for ans in all_answers:
            if pred_norm == _normalize(str(ans)):
                return 1.0
    return 0.0


def _token_f1(prediction: str, ground_truth: str) -> float:
    """Token-level F1 between predicted and gold answer."""
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
    """Lowercase, strip punctuation, collapse whitespace."""
    text = str(text).lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return " ".join(text.split())
