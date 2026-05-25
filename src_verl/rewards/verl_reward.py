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

    elif reward_type == "outcome_em_only":
        # E1' / Search-R1-equivalent: pure binary EM, no F1 blending.
        # Matches Search-R1 (arXiv:2503.09516) reward formulation exactly.
        return {
            "score": em,
            "r_outcome": em,
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

    elif reward_type == "verifiable_onpath":
        # Task 22: Single-component ablation — only r_on_path, no progress/coherence/valid
        step_rewards = _verifiable_onpath_step_rewards(steps, kg_path, extra_info)
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

    elif reward_type == "verifiable_balanced":
        # Task 23: Same 4-component verifiable reward but with 0.5/0.5 split
        step_rewards = _verifiable_step_rewards(steps, kg_path, extra_info)
        avg_step = sum(step_rewards) / max(len(step_rewards), 1)
        total = 0.5 * r_outcome + 0.5 * avg_step
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

    elif reward_type == "tool_type_bonus_format":
        # Task 39A: E5b-stabilized via format reward component.
        # Prevents format drift that caused E5b-original to collapse at step 150.
        step_rewards = _tool_type_bonus_step_rewards(steps)
        avg_step = sum(step_rewards) / max(len(step_rewards), 1)
        r_format = _format_valid(solution_str)
        total = 0.25 * r_outcome + 0.60 * avg_step + 0.15 * r_format
        return {
            "score": total,
            "r_outcome": r_outcome,
            "r_step_avg": avg_step,
            "r_format": r_format,
            "em": em,
            "f1": f1,
            "num_tool_calls": float(num_tool_calls),
        }

    elif reward_type == "tool_type_bonus_oracle_query_match":
        # Task 39 Variant I-Oracle (diagnostic upper bound).
        # R = 0.25*r_outcome + 0.50*r_tool_type + 0.25*r_query_match_oracle
        # r_query_match_oracle compares the (entity, relation) pairs in the
        # trajectory's tool calls against the gold path extracted by Task 36's
        # oracle. Requires gold path in extra_info (via train_oracle_gold_paths).
        # Hackability defenses:
        #   1. Dedup (entity, relation) pairs across turns (StepSearch-style).
        #   2. Gate: reward only fires if ≥2 distinct gold pairs are covered
        #      (KG-Implicit-RM anti-spam).
        #   3. Trajectory-level (not per-turn) — matched_unique / distinct_calls.
        step_rewards = _tool_type_bonus_step_rewards(steps)
        avg_step = sum(step_rewards) / max(len(step_rewards), 1)
        gold_path_pairs = _extract_gold_pairs(extra_info)
        r_query_match = _oracle_query_match(steps, gold_path_pairs)
        # Ramp the match weight over the first 50 steps via extra_info['global_step'],
        # ToolRL lesson: warm-start reward shaping.
        step_num = extra_info.get("global_step", 0) or 0
        ramp = min(1.0, step_num / 50.0)
        query_weight = 0.10 + (0.25 - 0.10) * ramp  # 0.10 → 0.25
        step_weight = 0.50
        answer_weight = 1.0 - query_weight - step_weight  # stays near 0.25
        total = answer_weight * r_outcome + step_weight * avg_step + query_weight * r_query_match
        return {
            "score": total,
            "r_outcome": r_outcome,
            "r_step_avg": avg_step,
            "r_query_match": r_query_match,
            "em": em,
            "f1": f1,
            "num_tool_calls": float(num_tool_calls),
        }

    elif reward_type == "tool_type_bonus_retrieval_contrib":
        # Task 39 Variant I-Self (deployable, no oracle leak).
        # R = 0.25*r_outcome + 0.50*r_tool_type + 0.25*r_retrieval_contrib
        # r_retrieval_contrib rewards tool calls where
        #   (a) the call returned non-empty results, AND
        #   (b) at least one returned entity appears verbatim in the final <answer>
        # Self-verifiable — no gold labels required.
        step_rewards = _tool_type_bonus_step_rewards(steps)
        avg_step = sum(step_rewards) / max(len(step_rewards), 1)
        r_retrieval = _retrieval_contribution(steps, predicted)
        total = 0.25 * r_outcome + 0.50 * avg_step + 0.25 * r_retrieval
        return {
            "score": total,
            "r_outcome": r_outcome,
            "r_step_avg": avg_step,
            "r_retrieval_contrib": r_retrieval,
            "em": em,
            "f1": f1,
            "num_tool_calls": float(num_tool_calls),
        }

    elif reward_type == "tool_type_bonus_retrieval_contrib_anti_quote":
        # v19 W18 — Anti-quote-and-stop variant of I-Self (R-selfV in paper).
        # Identical to tool_type_bonus_retrieval_contrib EXCEPT r_retrv is
        # forced to 0 if the trajectory has <2 DISTINCT non-empty <search>
        # calls (blocks the single-tool-call quote-and-stop attractor).
        # Falsifies the Mode-4 r_retrv mechanism if R1 still collapses; confirms
        # if R1 does NOT cliff. Pre-registered per hpc_tasks.md v19 W18.
        step_rewards = _tool_type_bonus_step_rewards(steps)
        avg_step = sum(step_rewards) / max(len(step_rewards), 1)
        r_retrieval = _retrieval_contribution_anti_quote(steps, predicted, min_distinct_nonempty=2)
        total = 0.25 * r_outcome + 0.50 * avg_step + 0.25 * r_retrieval
        return {
            "score": total,
            "r_outcome": r_outcome,
            "r_step_avg": avg_step,
            "r_retrieval_contrib": r_retrieval,
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
# Format validity check (Task 39A)
# ---------------------------------------------------------------------------

_VALID_TOOL_CALL_RE = re.compile(r"get_(tail|head)_(relations|entities)\s*\(")


def _format_valid(solution_str: str) -> float:
    """Binary format validity check for E5b-stabilized Variant A.

    Returns 1.0 if all of the following hold, else 0.0:
      1. No <search> tag nested inside a <think>...</think> block.
      2. <answer>...</answer> tag exists with non-empty content.
      3. If any <search> tags exist, at least one contains a valid tool call
         matching get_(tail|head)_(relations|entities)\\(.

    Designed to block three reward-hacking shortcuts:
      - Emitting empty <answer></answer> to grab format reward without answering.
      - Emitting pretty tags with malformed/missing tool calls.
      - Nesting <search> inside <think> (the format drift that caused E5b collapse).
    """
    # Check 1: no <search> inside any <think>...</think>
    for think_match in re.finditer(r"<think>(.*?)</think>", solution_str, re.DOTALL):
        if "<search>" in think_match.group(1):
            return 0.0

    # Check 2: <answer>...</answer> exists and is non-empty
    answer_match = re.search(r"<answer>(.*?)</answer>", solution_str, re.DOTALL)
    if not answer_match or not answer_match.group(1).strip():
        return 0.0

    # Check 3: if any <search> tags, at least one must contain a valid tool call
    search_tags = re.findall(r"<search>(.*?)</search>", solution_str, re.DOTALL)
    if search_tags:
        if not any(_VALID_TOOL_CALL_RE.search(call) for call in search_tags):
            return 0.0

    return 1.0


# ---------------------------------------------------------------------------
# Variant I helpers (Phase 7 Task 39)
# ---------------------------------------------------------------------------

def _extract_gold_pairs(extra_info: dict) -> set[tuple[str, str]]:
    """Extract gold (entity, relation) pairs from extra_info for Variant I-Oracle.

    Accepts several shapes (for robustness across data prep scripts):
      1. extra_info["oracle_gold_pairs"]: list of [entity, relation]
      2. extra_info["oracle_triple_chain"]: list of [head, relation, tail]
         → we take (head, relation) and (tail, relation) since either can be
         a productive query.
      3. extra_info["kg_path"]: list of [head, relation, tail] (same treatment)
    Returns an empty set if none present.
    """
    pairs: set[tuple[str, str]] = set()

    raw_pairs = extra_info.get("oracle_gold_pairs")
    if raw_pairs:
        for pair in _to_python(raw_pairs) or []:
            if len(pair) >= 2:
                pairs.add((_normalize(str(pair[0])), _normalize(str(pair[1]))))
        return pairs

    for key in ("oracle_triple_chain", "kg_path"):
        chain = extra_info.get(key)
        if not chain:
            continue
        for triple in _to_python(chain) or []:
            if len(triple) < 3:
                continue
            h, r, t = str(triple[0]), str(triple[1]), str(triple[2])
            pairs.add((_normalize(h), _normalize(r)))
            pairs.add((_normalize(t), _normalize(r)))
    return pairs


def _oracle_query_match(
    steps: list[dict[str, str]],
    gold_pairs: set[tuple[str, str]],
) -> float:
    """Variant I-Oracle query-match reward.

    Returns a float in [0, 1].
    - Dedups (entity, relation) across turns (anti-redundancy).
    - Returns 0 if <2 distinct gold pairs are matched (anti-spam gate).
    - Otherwise matched_unique / max(1, distinct_calls).
    """
    if not gold_pairs or not steps:
        return 0.0

    distinct_calls: set[tuple[str, str]] = set()
    for step in steps:
        action, args = _parse_action(step["call"])
        if action not in (
            "get_tail_entities", "get_head_entities",
            "get_tail_relations", "get_head_relations",
        ):
            continue
        entity = _normalize(args[0]) if args else ""
        # relations-only tools have no relation arg → use action as pseudo-relation
        # so spam like get_tail_relations(X) can still be tracked for dedup.
        relation = _normalize(args[1]) if len(args) > 1 else f"_{action}"
        if entity:
            distinct_calls.add((entity, relation))

    if not distinct_calls:
        return 0.0

    matched_unique = len(distinct_calls & gold_pairs)
    if matched_unique < 2:
        return 0.0  # anti-spam gate

    return matched_unique / len(distinct_calls)


def _retrieval_contribution(
    steps: list[dict[str, str]],
    predicted_answer: str,
) -> float:
    """Variant I-Self (deployable) retrieval contribution reward.

    Counts tool calls where
      (a) the call returned a non-empty, non-error observation, AND
      (b) at least one entity in the observation appears verbatim (case-insensitive)
          in the final <answer>...</answer>.
    Returns productive_calls / max(1, total_calls).
    """
    if not steps:
        return 0.0
    answer_norm = _normalize(predicted_answer)
    if not answer_norm:
        return 0.0

    productive = 0
    for step in steps:
        obs = step.get("observation", "") or ""
        obs_stripped = obs.strip()
        if not obs_stripped or obs_stripped in ("[]", "No results found") or obs_stripped.startswith("ERROR"):
            continue
        # Parse comma-separated list-like result
        entities = [
            e.strip().strip('"').strip("'")
            for e in obs_stripped.strip("[]").split(",")
        ]
        if any(ent and _normalize(ent) in answer_norm for ent in entities):
            productive += 1

    return productive / len(steps)


def _retrieval_contribution_anti_quote(
    steps: list[dict[str, str]],
    predicted_answer: str,
    min_distinct_nonempty: int = 2,
) -> float:
    """v19 W18 — Anti-quote-and-stop r_retrv variant.

    Same as _retrieval_contribution PLUS a gate: r_retrv is forced to 0 if the
    trajectory contains fewer than `min_distinct_nonempty` (default 2) DISTINCT
    <search> calls that returned a non-empty / non-error observation. This blocks
    the Mode-4 collapse attractor where the policy issues a single tool call and
    quotes its result back as the answer (Tools/Q ≈ 1 with high per-call r_retrv).

    "Distinct" = unique (action_name, entity, relation) triple — re-issuing the
    same search call does not count as a second distinct call.
    """
    if not steps:
        return 0.0
    answer_norm = _normalize(predicted_answer)
    if not answer_norm:
        return 0.0

    nonempty_signatures: set[tuple[str, str, str]] = set()
    productive = 0
    for step in steps:
        obs = step.get("observation", "") or ""
        obs_stripped = obs.strip()
        if not obs_stripped or obs_stripped in ("[]", "No results found") or obs_stripped.startswith("ERROR"):
            continue
        action = step.get("action", "") or ""
        entity = step.get("entity", "") or ""
        relation = step.get("relation", "") or ""
        nonempty_signatures.add((action, entity, relation))
        entities = [
            e.strip().strip('"').strip("'")
            for e in obs_stripped.strip("[]").split(",")
        ]
        if any(ent and _normalize(ent) in answer_norm for ent in entities):
            productive += 1

    if len(nonempty_signatures) < min_distinct_nonempty:
        return 0.0
    return productive / len(steps)


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
# R_verifiable_onpath step rewards (Task 22: single-component ablation)
# ---------------------------------------------------------------------------

def _verifiable_onpath_step_rewards(
    steps: list[dict[str, str]],
    kg_path: list,
    extra_info: dict,
) -> list[float]:
    """Single-component ablation: only r_on_path, no progress/coherence/valid."""
    if not steps:
        return []

    # Build set of path triples for r_on_path (same logic as _verifiable_step_rewards)
    path_triples: set[tuple[str, str, str]] = set()
    path_entities: set[str] = set()
    for triple in (kg_path or []):
        if hasattr(triple, '__len__') and len(triple) >= 3:
            h, r, t = str(triple[0]).lower(), str(triple[1]).lower(), str(triple[2]).lower()
            path_triples.add((h, r, t))
            path_entities.add(h)
            path_entities.add(t)

    rewards: list[float] = []
    for step in steps:
        obs = step["observation"]

        r_on_path = 0.0
        if path_triples and obs:
            obs_lower = obs.lower()
            for h, r, t in path_triples:
                if h in obs_lower and t in obs_lower:
                    r_on_path = 1.0
                    break
            if r_on_path == 0.0:
                obs_words = set(obs_lower.split())
                hits = len(path_entities & obs_words)
                if hits > 0:
                    r_on_path = min(hits / max(len(path_entities), 1), 0.5)

        rewards.append(r_on_path)

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
