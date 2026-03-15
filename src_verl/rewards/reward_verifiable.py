"""R_verifiable_step: Verifiable step-level reward (Our Method).

Each agent step is scored against the KG server's ground truth:
  - R_valid: Was the tool call syntactically valid and did it return results?
  - R_on_path: Did the queried entity/relation appear on the ground truth path?
  - R_progress: Did this step move closer to the answer entity (BFS distance)?
  - R_coherence: Does the thinking mention entities from the observation?

Per-step:
  R_step = 0.3 * R_valid + 0.3 * R_on_path + 0.2 * R_progress + 0.2 * R_coherence

Combined:
  R_total = 0.4 * R_answer + 0.6 * mean(R_step_i)

Range: [0.0, ~1.0]

This is the key experimental variable: every component is formally verifiable
against the KG, not heuristic.
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


def compute_reward_verifiable(
    trajectory: list[dict[str, str]],
    gold_answer: str,
    kg_path: list[list[str]],
    distances: dict[str, dict[str, int]] | None = None,
    **kwargs: Any,
) -> float:
    """Compute verifiable step-level reward.

    Args:
        trajectory: Full multi-turn conversation.
        gold_answer: Expected short answer string.
        kg_path: Ground truth KG path as list of [head, relation, tail] triples.
        distances: Precomputed BFS distances from answer entities.
                   Format: {answer_entity: {entity: distance}}

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
        return r_answer * 0.4

    # Build path entity and relation sets for on-path checking
    path_entities: set[str] = set()
    path_relations: set[str] = set()
    path_entity_chain: list[str] = []  # ordered entity chain for progress
    for triple in kg_path:
        if len(triple) >= 3:
            path_entities.add(triple[0].lower())
            path_entities.add(triple[2].lower())
            path_relations.add(triple[1].lower())
            if not path_entity_chain:
                path_entity_chain.append(triple[0].lower())
            path_entity_chain.append(triple[2].lower())

    # Resolve answer entity for distance computation
    answer_entity = gold_answer.strip().lower()
    answer_distances: dict[str, int] | None = None
    if distances:
        answer_distances = distances.get(answer_entity)

    step_scores: list[float] = []
    prev_distance: float | None = None

    for i, step in enumerate(steps):
        if step.action_type == "answer":
            continue

        r_valid = _score_valid(step)
        r_on_path = _score_on_path(step, path_entities, path_relations)
        r_progress, prev_distance = _score_progress(
            step, answer_distances, prev_distance
        )
        # Coherence: does the NEXT step's thinking reference entities from
        # this step's observation? (thinking is written after seeing observation)
        next_thinking = steps[i + 1].thinking if i + 1 < len(steps) else None
        r_coherence = _score_coherence(step, next_thinking)

        step_score = (
            0.3 * r_valid
            + 0.3 * r_on_path
            + 0.2 * r_progress
            + 0.2 * r_coherence
        )
        step_scores.append(step_score)

    if not step_scores:
        return r_answer * 0.4

    r_steps = sum(step_scores) / len(step_scores)
    return 0.4 * r_answer + 0.6 * r_steps


def _score_valid(step: Step) -> float:
    """R_valid: Was the tool call valid and did it return results?

    1.0 — Valid action type AND non-empty observation
    0.5 — Valid action type but empty/no observation
    0.0 — Invalid/unknown action type
    """
    valid_actions = {
        "get_tail_relations",
        "get_head_relations",
        "get_tail_entities",
        "get_head_entities",
    }

    if step.action_type not in valid_actions:
        return 0.0

    # Check if required arguments are present
    if step.action_type in ("get_tail_relations", "get_head_relations"):
        if not step.action_args.get("entity"):
            return 0.0
    elif step.action_type in ("get_tail_entities", "get_head_entities"):
        if not step.action_args.get("entity") or not step.action_args.get("relation"):
            return 0.0

    # Check observation
    if step.observation and step.observation != "[]":
        return 1.0
    return 0.5


def _score_on_path(
    step: Step,
    path_entities: set[str],
    path_relations: set[str],
) -> float:
    """R_on_path: Is the queried entity/relation on the ground truth path?

    1.0 — Both entity and relation on path (for entity+relation queries)
    0.7 — Entity on path (for relation-only queries, or entity match)
    0.3 — Relation on path but entity not
    0.0 — Neither on path
    """
    entity = step.action_args.get("entity", "").lower()
    relation = step.action_args.get("relation", "").lower()

    entity_on_path = entity in path_entities
    relation_on_path = relation in path_relations if relation else False

    if step.action_type in ("get_tail_entities", "get_head_entities"):
        # Both entity and relation are relevant
        if entity_on_path and relation_on_path:
            return 1.0
        elif entity_on_path:
            return 0.7
        elif relation_on_path:
            return 0.3
        return 0.0
    else:
        # Only entity is relevant (relation queries)
        return 1.0 if entity_on_path else 0.0


def _score_progress(
    step: Step,
    answer_distances: dict[str, int] | None,
    prev_distance: float | None,
) -> tuple[float, float | None]:
    """R_progress: Did this step move closer to the answer entity?

    Uses BFS distances precomputed from the answer entity.
    1.0 — Moved closer to answer
    0.5 — Same distance (lateral move)
    0.0 — Moved further away or no distance info

    Returns (score, current_distance) for tracking across steps.
    """
    if answer_distances is None:
        return 0.5, prev_distance  # No distance info, neutral score

    # Find current distance from the queried entity
    entity = step.action_args.get("entity", "").lower()
    current_distance = answer_distances.get(entity)

    if current_distance is None:
        return 0.0, prev_distance  # Entity not reachable from answer

    current_dist_float = float(current_distance)

    if prev_distance is None:
        # First step — give credit if entity is reasonably close
        score = max(0.0, 1.0 - current_dist_float / 5.0)
        return score, current_dist_float

    if current_dist_float < prev_distance:
        return 1.0, current_dist_float  # Moved closer
    elif current_dist_float == prev_distance:
        return 0.5, current_dist_float  # Lateral
    else:
        return 0.0, current_dist_float  # Moved away


def _score_coherence(step: Step, next_thinking: str | None = None) -> float:
    """R_coherence: Does the next step's thinking reference entities from this observation?

    Checks if the assistant's thinking in the NEXT step mentions entities
    that appeared in the current tool observation (the model writes the next
    thinking AFTER seeing the observation, so it can reference those entities).

    1.0 — Next thinking mentions >= 2 entities from observation
    0.5 — Next thinking mentions 1 entity from observation
    0.0 — No overlap
    """
    if not step.observation:
        return 0.0

    # Use next step's thinking if available, fall back to current step's
    thinking_text = next_thinking if next_thinking else step.thinking
    if not thinking_text:
        return 0.0

    # Extract entities from observation
    obs_entities = extract_entities_from_observation(step.observation)
    if not obs_entities:
        return 0.0

    # Check how many appear in thinking
    think_lower = thinking_text.lower()
    hits = sum(1 for e in obs_entities if e.lower() in think_lower)

    if hits >= 2:
        return 1.0
    elif hits == 1:
        return 0.5
    return 0.0
