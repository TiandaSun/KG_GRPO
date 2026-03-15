"""Tests for verl_reward.py v3 — parses <search> tags from solution_str.

Key change from v2: the reward function no longer relies on extra_info["tool_rewards"]
(which was never passed by verl's NaiveRewardManager). Instead it parses <search> tags
directly from solution_str and penalises outputs with zero tool calls.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src_verl.rewards.verl_reward import compute_score


@pytest.fixture
def extra_info() -> dict:
    return {
        "kg_path": [["dog", "IsA", "animal"], ["animal", "HasProperty", "alive"]],
    }


class TestToolUseParsing:
    """Test that <search> tags are parsed from solution_str."""

    def test_no_tool_calls_gets_penalty(self, extra_info: dict) -> None:
        """Model that never calls tools gets r_no_tool=-1.0."""
        result = compute_score(
            data_source="conceptnet",
            solution_str="<think>dog is animal</think>\n<answer>alive</answer>",
            ground_truth="alive",
            extra_info=extra_info,
        )
        assert result["r_no_tool"] == -1.0
        assert result["r_tool_use"] == 0.0
        assert result["num_tool_calls"] == 0.0

    def test_with_search_tags_gets_bonus(self, extra_info: dict) -> None:
        """Model that makes tool calls via <search> tags gets r_tool_use > 0."""
        solution = (
            "<think>Let me check</think>\n"
            "<search>get_tail_relations(dog)</search>\n"
            "['IsA', 'HasProperty']\n"
            "<think>Now get entities</think>\n"
            "<search>get_tail_entities(dog, IsA)</search>\n"
            "['animal']\n"
            "<answer>alive</answer>"
        )
        result = compute_score(
            data_source="conceptnet",
            solution_str=solution,
            ground_truth="alive",
            extra_info=extra_info,
        )
        assert result["r_tool_use"] > 0.0
        assert result["r_no_tool"] == 0.0
        assert result["num_tool_calls"] == 2.0

    def test_single_valid_tool_call(self, extra_info: dict) -> None:
        solution = (
            "<think>Check relations</think>\n"
            "<search>get_tail_relations(dog)</search>\n"
            "<answer>alive</answer>"
        )
        result = compute_score(
            data_source="conceptnet",
            solution_str=solution,
            ground_truth="alive",
            extra_info=extra_info,
        )
        assert result["r_tool_use"] > 0.0
        assert result["r_no_tool"] == 0.0
        assert result["num_tool_calls"] == 1.0

    def test_invalid_tool_call_syntax(self, extra_info: dict) -> None:
        """<search> with invalid function name: gets base bonus but no quality bonus."""
        solution = "<search>foobar(dog)</search>\n<answer>alive</answer>"
        result = compute_score(
            data_source="conceptnet",
            solution_str=solution,
            ground_truth="alive",
            extra_info=extra_info,
        )
        assert result["num_tool_calls"] == 1.0
        # Base bonus (0.5) but no quality bonus (invalid action)
        assert result["r_tool_use"] == pytest.approx(0.5)


class TestToolUseVsNoTool:
    """Verify that tool-using responses score higher than non-tool responses."""

    def test_tool_user_beats_non_tool_user(self, extra_info: dict) -> None:
        """Same answer quality, but one used tools — tool user should score higher."""
        # No tools — gets r_no_tool penalty
        result_no_tool = compute_score(
            data_source="conceptnet",
            solution_str="<answer>alive</answer>",
            ground_truth="alive",
            extra_info={**extra_info},
        )

        # With tools — gets tool bonus, no penalty
        result_with_tool = compute_score(
            data_source="conceptnet",
            solution_str=(
                "<search>get_tail_relations(dog)</search>\n"
                "<search>get_tail_entities(dog, IsA)</search>\n"
                "<answer>alive</answer>"
            ),
            ground_truth="alive",
            extra_info={**extra_info},
        )

        assert result_with_tool["score"] > result_no_tool["score"]
        # Difference should be at least 1.0 (penalty removed) + tool bonus
        score_diff = result_with_tool["score"] - result_no_tool["score"]
        assert score_diff > 1.0


class TestExistingRewards:
    """Ensure existing reward components still work."""

    def test_answer_f1(self) -> None:
        result = compute_score(
            data_source="conceptnet",
            solution_str="<answer>alive and well</answer>",
            ground_truth="alive",
            extra_info={},
        )
        assert result["r_answer"] > 0.0

    def test_coverage(self, extra_info: dict) -> None:
        result = compute_score(
            data_source="conceptnet",
            solution_str="dog is an animal that has property alive",
            ground_truth="alive",
            extra_info=extra_info,
        )
        assert result["r_coverage"] > 0.0

    def test_format_reward(self) -> None:
        result = compute_score(
            data_source="conceptnet",
            solution_str="<answer>some long answer here</answer>",
            ground_truth="answer",
            extra_info={},
        )
        assert result["r_format"] == 0.5

    def test_no_format(self) -> None:
        result = compute_score(
            data_source="conceptnet",
            solution_str="just plain text",
            ground_truth="answer",
            extra_info={},
        )
        assert result["r_format"] == 0.0


class TestNumpyCompatibility:
    """Test that numpy arrays from verl's DataProto are handled."""

    def test_numpy_tool_rewards_fallback(self, extra_info: dict) -> None:
        """If NaiveRewardLoopManager does pass tool_rewards, they're used as bonus."""
        import numpy as np
        extra_info["tool_rewards"] = np.array([0.5, 0.7, 0.3])
        result = compute_score(
            data_source="conceptnet",
            solution_str=(
                "<search>get_tail_relations(dog)</search>\n"
                "<search>get_tail_entities(dog, IsA)</search>\n"
                "<search>get_head_relations(animal)</search>\n"
                "<answer>alive</answer>"
            ),
            ground_truth="alive",
            extra_info=extra_info,
        )
        assert result["r_tool_use"] > 0.5  # base + quality + step rewards
        assert result["num_tool_calls"] == 3.0


class TestEdgeCases:
    """Edge cases for robustness."""

    def test_empty_solution(self) -> None:
        result = compute_score(
            data_source="conceptnet",
            solution_str="",
            ground_truth="alive",
            extra_info={},
        )
        assert result["r_no_tool"] == -1.0
        assert result["r_answer"] == 0.0

    def test_garbled_output(self, extra_info: dict) -> None:
        """Garbled output like the GRPO model produced — should get penalty."""
        result = compute_score(
            data_source="conceptnet",
            solution_str='<the part of an atlas might contain information is <answer">',
            ground_truth="alive",
            extra_info=extra_info,
        )
        assert result["r_no_tool"] == -1.0
        assert result["r_format"] == 0.0

    def test_many_tool_calls_diminishing_returns(self, extra_info: dict) -> None:
        """Quality bonus caps at 0.5 (diminishing after 3 calls)."""
        calls = "\n".join(
            f"<search>get_tail_relations(entity{i})</search>"
            for i in range(10)
        )
        result = compute_score(
            data_source="conceptnet",
            solution_str=f"{calls}\n<answer>alive</answer>",
            ground_truth="alive",
            extra_info=extra_info,
        )
        assert result["num_tool_calls"] == 10.0
        # r_tool_use = 0.5 (base) + 0.5 (quality capped) = 1.0
        assert result["r_tool_use"] == pytest.approx(1.0)
