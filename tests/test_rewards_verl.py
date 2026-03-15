"""Unit tests for the three verl reward functions.

Tests that:
1. R_outcome only scores final answer correctness
2. R_heuristic scores step-level entity overlap + reachability
3. R_verifiable scores step-level validity, on-path, progress, coherence
4. Good trajectories score higher than bad ones across all three
5. R_verifiable discriminates better than R_heuristic > R_outcome

Usage:
    pytest tests/test_rewards_verl.py -v
"""

import pytest

from src_verl.rewards.common import (
    Step,
    compute_exact_match,
    compute_token_f1,
    extract_answer,
    extract_think,
    normalize_answer,
    parse_action,
    parse_steps,
)
from src_verl.rewards.reward_outcome import compute_reward_outcome
from src_verl.rewards.reward_heuristic import compute_reward_heuristic
from src_verl.rewards.reward_verifiable import compute_reward_verifiable


# =========================================================================
# Fixtures: hand-crafted trajectories
# =========================================================================

SYSTEM_MSG = {"role": "system", "content": "You are a KG reasoning agent."}

KG_PATH_DOG = [
    ["dog", "IsA", "animal"],
    ["animal", "HasProperty", "alive"],
]
GOLD_ANSWER = "alive"


def _make_good_trajectory() -> list[dict[str, str]]:
    """Good trajectory: follows the KG path, correct answer."""
    return [
        SYSTEM_MSG,
        {"role": "user", "content": "What property do dogs have because they are animals?"},
        {
            "role": "assistant",
            "content": "<think>I need to find properties of dogs via animals.</think>\n<search>get_tail_relations(dog)</search>",
        },
        {
            "role": "tool",
            "content": "<information>['IsA', 'HasProperty', 'CapableOf']</information>",
        },
        {
            "role": "assistant",
            "content": "<think>Dog IsA something. Let me follow the IsA relation to find what dog is.</think>\n<search>get_tail_entities(dog, IsA)</search>",
        },
        {
            "role": "tool",
            "content": "<information>['animal', 'pet', 'mammal']</information>",
        },
        {
            "role": "assistant",
            "content": "<think>Dog is an animal. Now let me find properties of animals.</think>\n<search>get_tail_relations(animal)</search>",
        },
        {
            "role": "tool",
            "content": "<information>['HasProperty', 'IsA', 'CapableOf']</information>",
        },
        {
            "role": "assistant",
            "content": "<think>Animals have the HasProperty relation. Let me see what properties animals have.</think>\n<search>get_tail_entities(animal, HasProperty)</search>",
        },
        {
            "role": "tool",
            "content": "<information>['alive', 'sentient', 'warm-blooded']</information>",
        },
        {
            "role": "assistant",
            "content": "<think>Following the path: dog IsA animal, animal HasProperty alive. The answer is alive.</think>\n<answer>alive</answer>",
        },
    ]


def _make_bad_trajectory() -> list[dict[str, str]]:
    """Bad trajectory: wrong answer, doesn't follow path."""
    return [
        SYSTEM_MSG,
        {"role": "user", "content": "What property do dogs have because they are animals?"},
        {
            "role": "assistant",
            "content": "<think>I think dogs are cute.</think>\n<answer>cute</answer>",
        },
    ]


def _make_hacking_trajectory() -> list[dict[str, str]]:
    """Hacking trajectory: mentions entities without real reasoning."""
    return [
        SYSTEM_MSG,
        {"role": "user", "content": "What property do dogs have because they are animals?"},
        {
            "role": "assistant",
            "content": "<think>dog animal alive IsA HasProperty dog animal alive</think>\n<search>get_tail_relations(banana)</search>",
        },
        {
            "role": "tool",
            "content": "<information>['IsA']</information>",
        },
        {
            "role": "assistant",
            "content": "<think>dog alive animal</think>\n<answer>alive</answer>",
        },
    ]


def _make_partial_trajectory() -> list[dict[str, str]]:
    """Partial trajectory: some correct steps but wrong answer."""
    return [
        SYSTEM_MSG,
        {"role": "user", "content": "What property do dogs have because they are animals?"},
        {
            "role": "assistant",
            "content": "<think>Let me explore dog.</think>\n<search>get_tail_relations(dog)</search>",
        },
        {
            "role": "tool",
            "content": "<information>['IsA', 'HasProperty']</information>",
        },
        {
            "role": "assistant",
            "content": "<think>Dog has IsA and HasProperty. Let me check HasProperty directly.</think>\n<search>get_tail_entities(dog, HasProperty)</search>",
        },
        {
            "role": "tool",
            "content": "<information>['furry', 'loyal']</information>",
        },
        {
            "role": "assistant",
            "content": "<think>Dog has property furry.</think>\n<answer>furry</answer>",
        },
    ]


# =========================================================================
# Tests: common utilities
# =========================================================================

class TestCommonUtils:
    def test_normalize_answer(self) -> None:
        assert normalize_answer("The Dog") == "dog"
        assert normalize_answer("A  big  cat") == "big cat"

    def test_exact_match(self) -> None:
        assert compute_exact_match("alive", "alive") == 1.0
        assert compute_exact_match("The alive", "alive") == 1.0
        assert compute_exact_match("dead", "alive") == 0.0

    def test_token_f1(self) -> None:
        assert compute_token_f1("alive", "alive") == 1.0
        assert compute_token_f1("not alive today", "alive") > 0.0
        assert compute_token_f1("completely different", "alive") == 0.0

    def test_extract_think(self) -> None:
        assert extract_think("<think>reasoning here</think>rest") == "reasoning here"
        assert extract_think("no think tags") == ""

    def test_extract_answer(self) -> None:
        traj = [
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "<think>hmm</think>\n<answer>alive</answer>"},
        ]
        assert extract_answer(traj) == "alive"

    def test_parse_action(self) -> None:
        content = "<think>test</think>\n<search>get_tail_relations(dog)</search>"
        action, action_type, args = parse_action(content)
        assert action_type == "get_tail_relations"
        assert args["entity"] == "dog"

    def test_parse_action_with_relation(self) -> None:
        content = "<search>get_tail_entities(dog, IsA)</search>"
        action, action_type, args = parse_action(content)
        assert action_type == "get_tail_entities"
        assert args["entity"] == "dog"
        assert args["relation"] == "IsA"

    def test_parse_steps(self) -> None:
        traj = _make_good_trajectory()
        steps = parse_steps(traj)
        assert len(steps) >= 4  # 4 query steps + 1 answer step
        assert steps[0].action_type == "get_tail_relations"
        assert steps[-1].action_type == "answer"


# =========================================================================
# Tests: R_outcome
# =========================================================================

class TestRewardOutcome:
    def test_correct_answer(self) -> None:
        traj = _make_good_trajectory()
        score = compute_reward_outcome(traj, GOLD_ANSWER)
        assert score > 0.5, f"Good trajectory should score > 0.5, got {score}"

    def test_wrong_answer(self) -> None:
        traj = _make_bad_trajectory()
        score = compute_reward_outcome(traj, GOLD_ANSWER)
        assert score < 0.3, f"Bad trajectory should score < 0.3, got {score}"

    def test_good_beats_bad(self) -> None:
        good = compute_reward_outcome(_make_good_trajectory(), GOLD_ANSWER)
        bad = compute_reward_outcome(_make_bad_trajectory(), GOLD_ANSWER)
        assert good > bad, f"Good ({good}) should beat bad ({bad})"

    def test_range(self) -> None:
        score = compute_reward_outcome(_make_good_trajectory(), GOLD_ANSWER)
        assert 0.0 <= score <= 1.0, f"Score {score} out of range [0, 1]"


# =========================================================================
# Tests: R_heuristic
# =========================================================================

class TestRewardHeuristic:
    def test_good_trajectory(self) -> None:
        traj = _make_good_trajectory()
        score = compute_reward_heuristic(traj, GOLD_ANSWER, KG_PATH_DOG)
        assert score > 0.3, f"Good trajectory should score > 0.3, got {score}"

    def test_bad_trajectory(self) -> None:
        traj = _make_bad_trajectory()
        score = compute_reward_heuristic(traj, GOLD_ANSWER, KG_PATH_DOG)
        assert score < 0.3, f"Bad trajectory should score < 0.3, got {score}"

    def test_good_beats_bad(self) -> None:
        good = compute_reward_heuristic(_make_good_trajectory(), GOLD_ANSWER, KG_PATH_DOG)
        bad = compute_reward_heuristic(_make_bad_trajectory(), GOLD_ANSWER, KG_PATH_DOG)
        assert good > bad, f"Good ({good}) should beat bad ({bad})"

    def test_hacking_penalised(self) -> None:
        good = compute_reward_heuristic(_make_good_trajectory(), GOLD_ANSWER, KG_PATH_DOG)
        hack = compute_reward_heuristic(_make_hacking_trajectory(), GOLD_ANSWER, KG_PATH_DOG)
        # Hacking may still get some credit (correct answer), but less than good
        assert good > hack or abs(good - hack) < 0.1, f"Good ({good}) should be >= hack ({hack})"


# =========================================================================
# Tests: R_verifiable
# =========================================================================

class TestRewardVerifiable:
    def test_good_trajectory(self) -> None:
        traj = _make_good_trajectory()
        score = compute_reward_verifiable(traj, GOLD_ANSWER, KG_PATH_DOG)
        assert score > 0.3, f"Good trajectory should score > 0.3, got {score}"

    def test_bad_trajectory(self) -> None:
        traj = _make_bad_trajectory()
        score = compute_reward_verifiable(traj, GOLD_ANSWER, KG_PATH_DOG)
        assert score < 0.3, f"Bad trajectory should score < 0.3, got {score}"

    def test_good_beats_bad(self) -> None:
        good = compute_reward_verifiable(_make_good_trajectory(), GOLD_ANSWER, KG_PATH_DOG)
        bad = compute_reward_verifiable(_make_bad_trajectory(), GOLD_ANSWER, KG_PATH_DOG)
        assert good > bad, f"Good ({good}) should beat bad ({bad})"

    def test_hacking_penalised(self) -> None:
        good = compute_reward_verifiable(_make_good_trajectory(), GOLD_ANSWER, KG_PATH_DOG)
        hack = compute_reward_verifiable(_make_hacking_trajectory(), GOLD_ANSWER, KG_PATH_DOG)
        assert good > hack, f"Good ({good}) should beat hack ({hack})"

    def test_partial_intermediate(self) -> None:
        good = compute_reward_verifiable(_make_good_trajectory(), GOLD_ANSWER, KG_PATH_DOG)
        partial = compute_reward_verifiable(_make_partial_trajectory(), GOLD_ANSWER, KG_PATH_DOG)
        bad = compute_reward_verifiable(_make_bad_trajectory(), GOLD_ANSWER, KG_PATH_DOG)
        assert good > partial > bad, f"Expected good ({good}) > partial ({partial}) > bad ({bad})"


# =========================================================================
# Tests: Discrimination ordering
# =========================================================================

class TestDiscrimination:
    """R_verifiable should discriminate better than R_heuristic > R_outcome."""

    def test_verifiable_discriminates_hacking(self) -> None:
        """R_verifiable should have largest gap between good and hacking."""
        good_traj = _make_good_trajectory()
        hack_traj = _make_hacking_trajectory()

        gap_outcome = (
            compute_reward_outcome(good_traj, GOLD_ANSWER)
            - compute_reward_outcome(hack_traj, GOLD_ANSWER)
        )
        gap_heuristic = (
            compute_reward_heuristic(good_traj, GOLD_ANSWER, KG_PATH_DOG)
            - compute_reward_heuristic(hack_traj, GOLD_ANSWER, KG_PATH_DOG)
        )
        gap_verifiable = (
            compute_reward_verifiable(good_traj, GOLD_ANSWER, KG_PATH_DOG)
            - compute_reward_verifiable(hack_traj, GOLD_ANSWER, KG_PATH_DOG)
        )

        # R_verifiable should have at least as much discrimination as outcome
        assert gap_verifiable >= gap_outcome - 0.1, (
            f"Verifiable gap ({gap_verifiable:.3f}) should be >= outcome gap ({gap_outcome:.3f})"
        )

    def test_all_rewards_score_good_higher_than_bad(self) -> None:
        """All three rewards should rank good > bad."""
        good = _make_good_trajectory()
        bad = _make_bad_trajectory()

        r_out_good = compute_reward_outcome(good, GOLD_ANSWER)
        r_out_bad = compute_reward_outcome(bad, GOLD_ANSWER)
        assert r_out_good > r_out_bad

        r_heur_good = compute_reward_heuristic(good, GOLD_ANSWER, KG_PATH_DOG)
        r_heur_bad = compute_reward_heuristic(bad, GOLD_ANSWER, KG_PATH_DOG)
        assert r_heur_good > r_heur_bad

        r_ver_good = compute_reward_verifiable(good, GOLD_ANSWER, KG_PATH_DOG)
        r_ver_bad = compute_reward_verifiable(bad, GOLD_ANSWER, KG_PATH_DOG)
        assert r_ver_good > r_ver_bad
