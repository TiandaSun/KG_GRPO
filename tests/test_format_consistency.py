"""Tests for format consistency across SFT, GRPO rollout, and evaluation.

Verifies that tool responses use the same format in all three stages
of the pipeline, preventing the format mismatch that caused GRPO to
kill multi-turn tool use in the first training run.

The three-way consistency requirement:
  - SFT training: tool responses in trajectory use role="tool"
  - GRPO rollouts: KGQueryTool returns <information>...</information> wrapped text
  - Evaluation: tool responses wrapped in <information>...</information>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class TestSFTToolFormat:
    """Verify SFT preserves native tool role."""

    def test_tool_role_preserved(self) -> None:
        """SFT should keep 'tool' role, not map to 'user'."""
        from src_verl.training.sft_multiturn import trajectories_to_dataset

        records = [{
            "trajectory": [
                {"role": "system", "content": "You are a helper."},
                {"role": "user", "content": "What is a dog?"},
                {"role": "assistant", "content": "<search>get_tail_relations(dog)</search>"},
                {"role": "tool", "content": "<information>['IsA', 'HasProperty']</information>"},
                {"role": "assistant", "content": "<answer>An animal</answer>"},
            ],
        }]

        dataset = trajectories_to_dataset(records, tokenizer=None)
        messages = dataset[0]["messages"]

        # The tool message should keep role="tool", NOT be mapped to "user"
        tool_msgs = [m for m in messages if m["role"] == "tool"]
        assert len(tool_msgs) == 1, "Tool message should remain with role='tool'"
        assert "<information>" in tool_msgs[0]["content"]

        # No messages should have "Tool response:" prefix
        for msg in messages:
            assert not msg["content"].startswith("Tool response:"), \
                "Should not have 'Tool response:' prefix (old format)"


class TestKGInteractionFormat:
    """Verify KGQueryTool wraps responses in <information> tags."""

    def test_response_wrapped_in_information_tags(self) -> None:
        """KGQueryTool.execute() should wrap results in <information> tags."""
        from src_verl.interaction.kg_interaction import KGQueryTool
        from verl.tools.schemas import OpenAIFunctionToolSchema

        # Create tool with mock config
        tool_schema = OpenAIFunctionToolSchema(
            type="function",
            function={
                "name": "kg_query",
                "description": "Query KG",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        )
        tool = KGQueryTool(
            config={"kg_server_url": "http://localhost:9999", "reward_type": "outcome"},
            tool_schema=tool_schema,
        )

        # Mock the KG query to return some results
        with patch.object(tool, "_query_kg", return_value=["animal", "pet"]):
            import asyncio
            # Create an instance first
            instance_id, _ = asyncio.get_event_loop().run_until_complete(
                tool.create(create_kwargs={"gold_answer": "animal", "kg_path": []})
            )
            response, reward, metrics = asyncio.get_event_loop().run_until_complete(
                tool.execute(instance_id, {"action": "get_tail_entities", "entity": "dog", "relation": "IsA"})
            )

        assert response.text.startswith("<information>"), \
            f"Response should start with <information>, got: {response.text[:50]}"
        assert response.text.endswith("</information>"), \
            f"Response should end with </information>, got: {response.text[-50:]}"
        assert "animal" in response.text

    def test_no_results_wrapped(self) -> None:
        """Even 'No results found.' should be wrapped."""
        from src_verl.interaction.kg_interaction import KGQueryTool
        from verl.tools.schemas import OpenAIFunctionToolSchema

        tool_schema = OpenAIFunctionToolSchema(
            type="function",
            function={
                "name": "kg_query",
                "description": "Query KG",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        )
        tool = KGQueryTool(
            config={"kg_server_url": "http://localhost:9999", "reward_type": "outcome"},
            tool_schema=tool_schema,
        )

        with patch.object(tool, "_query_kg", return_value=[]):
            import asyncio
            instance_id, _ = asyncio.get_event_loop().run_until_complete(
                tool.create(create_kwargs={"gold_answer": "", "kg_path": []})
            )
            response, _, _ = asyncio.get_event_loop().run_until_complete(
                tool.execute(instance_id, {"action": "get_tail_relations", "entity": "xyz"})
            )

        assert response.text == "<information>No results found.</information>"


class TestRewardDetectsToolUse:
    """Verify reward function can detect tool use from flat solution_str."""

    def test_multi_turn_flat_string_with_tools(self) -> None:
        """Reward should detect <search> tags in concatenated multi-turn text."""
        from src_verl.rewards.verl_reward import compute_score

        # This is what NaiveRewardManager passes: all turns concatenated as flat text
        solution_str = (
            "<think>Let me check the knowledge graph.</think>\n"
            "<search>get_tail_relations(dog)</search>\n"
            "<information>['IsA', 'HasProperty']</information>\n"  # tool response mixed in
            "<think>I see IsA relation, let me follow it.</think>\n"
            "<search>get_tail_entities(dog, IsA)</search>\n"
            "<information>['animal']</information>\n"
            "<answer>animal</answer>"
        )

        result = compute_score(
            data_source="conceptnet",
            solution_str=solution_str,
            ground_truth="animal",
            extra_info={"kg_path": [["dog", "IsA", "animal"]]},
        )

        assert result["num_tool_calls"] == 2.0
        assert result["r_tool_use"] > 0.0
        assert result["r_no_tool"] == 0.0
        assert result["score"] > 0.0

    def test_single_turn_no_tools_gets_penalty(self) -> None:
        """Single-turn garbled output should get penalty."""
        from src_verl.rewards.verl_reward import compute_score

        result = compute_score(
            data_source="conceptnet",
            solution_str='<the dog is an animal <answer">',
            ground_truth="animal",
            extra_info={"kg_path": [["dog", "IsA", "animal"]]},
        )

        assert result["num_tool_calls"] == 0.0
        assert result["r_no_tool"] == -1.0
