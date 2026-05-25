"""Phase 7 Variant I reward unit tests.

Validates the anti-spam / anti-hack defenses on:
  - tool_type_bonus_oracle_query_match  (Variant I-Oracle)
  - tool_type_bonus_retrieval_contrib   (Variant I-Self)

Run with:  python -m pytest tests/test_phase7_variant_i_rewards.py -v
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src_verl.rewards.verl_reward import (  # noqa: E402
    _extract_gold_pairs,
    _oracle_query_match,
    _parse_steps,
    _retrieval_contribution,
    compute_score,
)


def make_traj(*tool_calls_and_obs: tuple[str, str], answer: str = "dummy") -> str:
    """Build a flat solution_str from (call, observation) pairs + an answer."""
    parts = ["<think>reasoning</think>"]
    for call, obs in tool_calls_and_obs:
        parts.append(f"<search>{call}</search>")
        parts.append(f"<tool_response>{obs}</tool_response>")
    parts.append(f"<answer>{answer}</answer>")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# _extract_gold_pairs
# ---------------------------------------------------------------------------

def test_gold_pairs_from_triple_chain():
    extra = {"oracle_triple_chain": [["Alice", "friend_of", "Bob"], ["Bob", "lives_in", "NYC"]]}
    pairs = _extract_gold_pairs(extra)
    assert ("alice", "friend_of") in pairs
    assert ("bob", "friend_of") in pairs
    assert ("bob", "lives_in") in pairs
    assert ("nyc", "lives_in") in pairs


def test_gold_pairs_from_kg_path_fallback():
    extra = {"kg_path": [["M83", "composed_by", "Anthony Gonzalez"]]}
    pairs = _extract_gold_pairs(extra)
    assert ("m83", "composed_by") in pairs
    assert ("anthony gonzalez", "composed_by") in pairs


def test_gold_pairs_explicit_oracle_pairs_field():
    extra = {"oracle_gold_pairs": [["Obama", "born_in"], ["Hawaii", "country_of"]]}
    pairs = _extract_gold_pairs(extra)
    assert pairs == {("obama", "born_in"), ("hawaii", "country_of")}


def test_gold_pairs_empty():
    assert _extract_gold_pairs({}) == set()
    assert _extract_gold_pairs({"kg_path": []}) == set()


# ---------------------------------------------------------------------------
# _oracle_query_match anti-spam defenses
# ---------------------------------------------------------------------------

def test_oracle_anti_spam_gate_single_match():
    """Exactly one distinct gold pair matched → reward = 0 (gate blocks)."""
    traj = make_traj(
        ("get_tail_entities(Alice, friend_of)", "['Bob']"),
    )
    steps = _parse_steps(traj)
    gold = {("alice", "friend_of"), ("bob", "lives_in")}
    assert _oracle_query_match(steps, gold) == 0.0


def test_oracle_two_distinct_matches_rewards():
    """Two distinct gold pairs → reward = matched / total_calls."""
    traj = make_traj(
        ("get_tail_entities(Alice, friend_of)", "['Bob']"),
        ("get_tail_entities(Bob, lives_in)", "['NYC']"),
    )
    steps = _parse_steps(traj)
    gold = {("alice", "friend_of"), ("bob", "lives_in")}
    reward = _oracle_query_match(steps, gold)
    assert reward == 1.0  # 2 matched / 2 distinct calls


def test_oracle_spam_same_call_no_reward():
    """Spamming the same gold pair 5 times = 1 distinct → gate blocks."""
    traj = make_traj(
        *[("get_tail_entities(Alice, friend_of)", "['Bob']")] * 5,
    )
    steps = _parse_steps(traj)
    gold = {("alice", "friend_of"), ("bob", "lives_in")}
    assert _oracle_query_match(steps, gold) == 0.0


def test_oracle_partial_match_with_noise():
    """Two gold matches + one non-matching call → 2/3."""
    traj = make_traj(
        ("get_tail_entities(Alice, friend_of)", "['Bob']"),
        ("get_tail_entities(Bob, lives_in)", "['NYC']"),
        ("get_tail_entities(Noise, nothing)", "[]"),
    )
    steps = _parse_steps(traj)
    gold = {("alice", "friend_of"), ("bob", "lives_in")}
    reward = _oracle_query_match(steps, gold)
    assert abs(reward - (2 / 3)) < 1e-6


def test_oracle_no_gold_zero():
    traj = make_traj(("get_tail_entities(Alice, friend_of)", "['Bob']"))
    steps = _parse_steps(traj)
    assert _oracle_query_match(steps, set()) == 0.0


# ---------------------------------------------------------------------------
# _retrieval_contribution (Variant I-Self)
# ---------------------------------------------------------------------------

def test_retrieval_contrib_answer_found_in_obs():
    """Tool returned 'NYC' and answer contains 'NYC' → 1.0."""
    traj = make_traj(
        ("get_tail_entities(Bob, lives_in)", "['NYC']"),
        answer="NYC",
    )
    steps = _parse_steps(traj)
    pred = "NYC"
    assert _retrieval_contribution(steps, pred) == 1.0


def test_retrieval_contrib_empty_obs():
    traj = make_traj(
        ("get_tail_entities(Bob, lives_in)", "[]"),
        answer="unknown",
    )
    steps = _parse_steps(traj)
    assert _retrieval_contribution(steps, "unknown") == 0.0


def test_retrieval_contrib_partial_credit():
    """Two calls, one productive, one not → 0.5."""
    traj = make_traj(
        ("get_tail_entities(Bob, lives_in)", "['NYC']"),
        ("get_tail_entities(Bob, friend_of)", "[]"),
        answer="NYC",
    )
    steps = _parse_steps(traj)
    reward = _retrieval_contribution(steps, "NYC")
    assert abs(reward - 0.5) < 1e-6


def test_retrieval_contrib_entity_in_answer_but_obs_error():
    """Tool errored → not productive even if answer happens to include entity."""
    traj = make_traj(
        ("get_tail_entities(Bob, lives_in)", "ERROR: timeout"),
        answer="Bob",
    )
    steps = _parse_steps(traj)
    assert _retrieval_contribution(steps, "Bob") == 0.0


def test_retrieval_contrib_empty_steps():
    assert _retrieval_contribution([], "answer") == 0.0


def test_retrieval_contrib_no_answer():
    traj = make_traj(
        ("get_tail_entities(Bob, lives_in)", "['NYC']"),
        answer="",
    )
    steps = _parse_steps(traj)
    assert _retrieval_contribution(steps, "") == 0.0


# ---------------------------------------------------------------------------
# compute_score integration
# ---------------------------------------------------------------------------

def test_compute_score_i_oracle_integration():
    traj = make_traj(
        ("get_tail_entities(Alice, friend_of)", "['Bob']"),
        ("get_tail_entities(Bob, lives_in)", "['NYC']"),
        answer="NYC",
    )
    extra = {
        "reward_type": "tool_type_bonus_oracle_query_match",
        "oracle_triple_chain": [["Alice", "friend_of", "Bob"], ["Bob", "lives_in", "NYC"]],
        "all_answers": ["NYC"],
        "global_step": 100,  # past ramp-up → full 0.25 weight
    }
    out = compute_score("cwq", traj, "NYC", extra_info=extra)
    assert out["r_query_match"] == 1.0
    assert out["em"] == 1.0
    assert out["score"] > 0.8  # answer + retrieval + query match all positive


def test_compute_score_i_self_integration():
    traj = make_traj(
        ("get_tail_entities(Bob, lives_in)", "['NYC']"),
        answer="NYC",
    )
    extra = {
        "reward_type": "tool_type_bonus_retrieval_contrib",
        "all_answers": ["NYC"],
    }
    out = compute_score("cwq", traj, "NYC", extra_info=extra)
    assert out["r_retrieval_contrib"] == 1.0
    assert out["em"] == 1.0


def test_compute_score_i_self_spam_no_bonus():
    """Spamming without productive retrieval → no I-Self bonus."""
    traj = make_traj(
        ("get_tail_relations(Bob)", "['P1', 'P2']"),
        ("get_tail_relations(Alice)", "['P3']"),
        answer="unknown",
    )
    extra = {
        "reward_type": "tool_type_bonus_retrieval_contrib",
        "all_answers": ["NYC"],
    }
    out = compute_score("cwq", traj, "NYC", extra_info=extra)
    # answer is wrong → r_outcome = 0; tool calls are relation-type (weight 0.3);
    # retrieval contrib = 0 because answer doesn't contain any tool response entity
    assert out["r_retrieval_contrib"] == 0.0
    assert out["em"] == 0.0


if __name__ == "__main__":
    # Cheap hand-run fallback if pytest not available
    import traceback
    passed = 0
    failed = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                passed += 1
                print(f"PASS  {name}")
            except Exception as e:
                failed += 1
                print(f"FAIL  {name}: {e}")
                traceback.print_exc()
    print(f"\n{passed}/{passed+failed} passed")
    sys.exit(0 if failed == 0 else 1)
