"""R_heuristic_step: Heuristic step-level reward (Baseline 2).

Per-step heuristic scoring based on entity overlap and reachability,
combined with outcome reward. This is the mid-complexity baseline —
more signal than R_outcome but not formally verifiable.

Per-step score:
  R_step = 0.5 * entity_overlap + 0.5 * reachability_bonus
  - entity_overlap: fraction of KG path entities mentioned in thinking
  - reachability_bonus: 1.0 if queried entity is in the KG graph, else 0.0

Combined:
  R_total = 0.4 * R_answer + 0.6 * mean(R_step_i)

Range: [0.0, ~1.0]
"""

from __future__ import annotations

from typing import Any

from src_verl.rewards.common import (
    Step,
    compute_exact_match,
    compute_token_f1,
    extract_answer,
    extract_entities_from_observation,
    parse_steps,
)


def compute_reward_heuristic(
    trajectory: list[dict[str, str]],
    gold_answer: str,
    kg_path: list[list[str]],
    kg_entities: set[str] | None = None,
    **kwargs: Any,
) -> float:
    """Compute heuristic step-level reward.

    Args:
        trajectory: Full multi-turn conversation.
        gold_answer: Expected short answer string.
        kg_path: Ground truth KG path as list of [head, relation, tail] triples.
        kg_entities: Optional set of all entities in the KG (for reachability).
                     If None, reachability bonus is always 0.5.

    Returns:
        Combined reward score.
    """
    # Answer component
    predicted = extract_answer(trajectory)
    em = compute_exact_match(predicted, gold_answer) if predicted else 0.0
    f1 = compute_token_f1(predicted, gold_answer) if predicted else 0.0
    r_answer = 0.5 * em + 0.5 * f1

    # Step component
    steps = parse_steps(trajectory)
    if not steps:
        return r_answer * 0.4  # No steps = only answer component

    # Collect all entities from the KG path
    path_entities: set[str] = set()
    for triple in kg_path:
        for token in triple:
            path_entities.add(token.lower())

    step_scores: list[float] = []
    for step in steps:
        if step.action_type == "answer":
            continue  # Skip the final answer step
        step_score = _score_step(step, path_entities, kg_entities)
        step_scores.append(step_score)

    if not step_scores:
        return r_answer * 0.4

    r_steps = sum(step_scores) / len(step_scores)
    return 0.4 * r_answer + 0.6 * r_steps


def _score_step(
    step: Step,
    path_entities: set[str],
    kg_entities: set[str] | None,
) -> float:
    """Score a single step heuristically.

    Components:
    1. Entity overlap: How many KG path entities appear in the thinking?
    2. Reachability: Is the queried entity in the KG graph?
    """
    # Entity overlap from thinking
    if step.thinking and path_entities:
        think_lower = step.thinking.lower()
        hits = sum(1 for e in path_entities if e in think_lower)
        entity_overlap = hits / len(path_entities)
    else:
        entity_overlap = 0.0

    # Reachability bonus
    queried_entity = step.action_args.get("entity", "").lower()
    if kg_entities is not None:
        reachability = 1.0 if queried_entity in kg_entities else 0.0
    else:
        # If we don't have the full entity set, give partial credit
        # for querying something that's in the path
        reachability = 1.0 if queried_entity in path_entities else 0.5

    return 0.5 * entity_overlap + 0.5 * reachability
