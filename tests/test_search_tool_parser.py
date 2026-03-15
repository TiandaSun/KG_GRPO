"""Tests for the kg_search ToolParser."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src_verl.interaction.search_tool_parser import KGSearchToolParser


@pytest.fixture
def parser() -> KGSearchToolParser:
    """Create parser with a mock tokenizer."""
    tokenizer = MagicMock()
    return KGSearchToolParser(tokenizer)


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


class TestSingleArgActions:
    """Test actions with one argument: get_tail_relations, get_head_relations."""

    def test_get_tail_relations(self, parser: KGSearchToolParser) -> None:
        text = "<think>Let me explore dog.</think>\n<search>get_tail_relations(dog)</search>"
        parser.tokenizer.decode.return_value = text

        content, calls = _run(parser.extract_tool_calls([1, 2, 3]))

        assert len(calls) == 1
        assert calls[0].name == "kg_query"
        args = json.loads(calls[0].arguments)
        assert args["action"] == "get_tail_relations"
        assert args["entity"] == "dog"
        assert "relation" not in args

    def test_get_head_relations(self, parser: KGSearchToolParser) -> None:
        text = "<search>get_head_relations(animal)</search>"
        parser.tokenizer.decode.return_value = text

        content, calls = _run(parser.extract_tool_calls([1]))

        assert len(calls) == 1
        args = json.loads(calls[0].arguments)
        assert args["action"] == "get_head_relations"
        assert args["entity"] == "animal"


class TestTwoArgActions:
    """Test actions with two arguments: get_tail_entities, get_head_entities."""

    def test_get_tail_entities(self, parser: KGSearchToolParser) -> None:
        text = "<search>get_tail_entities(dog, IsA)</search>"
        parser.tokenizer.decode.return_value = text

        content, calls = _run(parser.extract_tool_calls([1]))

        assert len(calls) == 1
        args = json.loads(calls[0].arguments)
        assert args["action"] == "get_tail_entities"
        assert args["entity"] == "dog"
        assert args["relation"] == "IsA"

    def test_get_head_entities(self, parser: KGSearchToolParser) -> None:
        text = "<search>get_head_entities(alive, HasProperty)</search>"
        parser.tokenizer.decode.return_value = text

        content, calls = _run(parser.extract_tool_calls([1]))

        assert len(calls) == 1
        args = json.loads(calls[0].arguments)
        assert args["action"] == "get_head_entities"
        assert args["entity"] == "alive"
        assert args["relation"] == "HasProperty"


class TestEdgeCases:
    """Test edge cases and malformed inputs."""

    def test_no_search_tags(self, parser: KGSearchToolParser) -> None:
        text = "<think>Just thinking, no tool calls.</think>\n<answer>42</answer>"
        parser.tokenizer.decode.return_value = text

        content, calls = _run(parser.extract_tool_calls([1]))

        assert len(calls) == 0
        assert content == text

    def test_unknown_action(self, parser: KGSearchToolParser) -> None:
        text = "<search>unknown_action(foo)</search>"
        parser.tokenizer.decode.return_value = text

        content, calls = _run(parser.extract_tool_calls([1]))

        assert len(calls) == 0  # Unknown action skipped

    def test_whitespace_in_args(self, parser: KGSearchToolParser) -> None:
        text = "<search>get_tail_entities( dog , IsA )</search>"
        parser.tokenizer.decode.return_value = text

        content, calls = _run(parser.extract_tool_calls([1]))

        assert len(calls) == 1
        args = json.loads(calls[0].arguments)
        assert args["entity"] == "dog"
        assert args["relation"] == "IsA"

    def test_quoted_args(self, parser: KGSearchToolParser) -> None:
        text = "<search>get_tail_relations('running water')</search>"
        parser.tokenizer.decode.return_value = text

        content, calls = _run(parser.extract_tool_calls([1]))

        assert len(calls) == 1
        args = json.loads(calls[0].arguments)
        assert args["entity"] == "running water"

    def test_multiple_search_tags(self, parser: KGSearchToolParser) -> None:
        text = (
            "<think>First query</think>\n"
            "<search>get_tail_relations(dog)</search>\n"
            "<think>Second query</think>\n"
            "<search>get_tail_entities(dog, IsA)</search>"
        )
        parser.tokenizer.decode.return_value = text

        content, calls = _run(parser.extract_tool_calls([1]))

        assert len(calls) == 2
        assert json.loads(calls[0].arguments)["action"] == "get_tail_relations"
        assert json.loads(calls[1].arguments)["action"] == "get_tail_entities"

    def test_content_excludes_search_tags(self, parser: KGSearchToolParser) -> None:
        text = "<think>reasoning</think>\n<search>get_tail_relations(dog)</search>\nmore text"
        parser.tokenizer.decode.return_value = text

        content, calls = _run(parser.extract_tool_calls([1]))

        assert "<search>" not in content
        assert "<think>reasoning</think>" in content
        assert "more text" in content

    def test_entity_with_spaces(self, parser: KGSearchToolParser) -> None:
        text = "<search>get_tail_relations(running water)</search>"
        parser.tokenizer.decode.return_value = text

        content, calls = _run(parser.extract_tool_calls([1]))

        assert len(calls) == 1
        args = json.loads(calls[0].arguments)
        assert args["entity"] == "running water"


class TestRegistration:
    """Test that the parser is properly registered in verl's registry."""

    def test_registered_in_tool_parser(self) -> None:
        from verl.experimental.agent_loop.tool_parser import ToolParser

        assert "kg_search" in ToolParser._registry
        assert ToolParser._registry["kg_search"] is KGSearchToolParser
