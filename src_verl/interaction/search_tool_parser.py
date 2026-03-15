"""Custom ToolParser for <search>action(args)</search> format.

Registers as "kg_search" in verl's ToolParser registry.

The SFT warmup teaches the model to emit tool calls like:
    <search>get_tail_relations(dog)</search>
    <search>get_tail_entities(dog, IsA)</search>

verl's built-in Hermes parser expects <tool_call>{"name":..., "arguments":...}</tool_call>.
This parser bridges the gap by converting <search> tags into FunctionCall objects
that verl's ToolAgentLoop can dispatch to KGQueryTool.

Import this module before ToolAgentLoop.initialize() to register the parser.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re

from verl.experimental.agent_loop.tool_parser import FunctionCall, ToolParser
from verl.utils.rollout_trace import rollout_trace_op

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

# Matches: <search>action_name(arg1, arg2, ...)</search>
# Captures: group(1) = action name, group(2) = comma-separated arguments
_SEARCH_PATTERN = re.compile(
    r"<search>\s*(\w+)\(([^)]*)\)\s*</search>",
    re.DOTALL,
)

_VALID_ACTIONS = frozenset({
    "get_tail_relations",
    "get_head_relations",
    "get_tail_entities",
    "get_head_entities",
})


@ToolParser.register("kg_search")
class KGSearchToolParser(ToolParser):
    """Parser for <search>action(args)</search> format used in KG reasoning.

    Converts each <search> tag into a FunctionCall with:
        name = "kg_query"  (matches the tool registered in kg_tool_config.yaml)
        arguments = JSON {"action": ..., "entity": ..., "relation": ...}
    """

    @rollout_trace_op
    async def extract_tool_calls(
        self, responses_ids: list[int]
    ) -> tuple[str, list[FunctionCall]]:
        loop = asyncio.get_running_loop()
        text = await loop.run_in_executor(
            None, self.tokenizer.decode, responses_ids
        )

        if "<search>" not in text or "</search>" not in text:
            return text, []

        matches = _SEARCH_PATTERN.findall(text)
        function_calls: list[FunctionCall] = []

        for action, args_str in matches:
            if action not in _VALID_ACTIONS:
                logger.warning("Unknown action in <search> tag: %s", action)
                continue

            # Parse comma-separated arguments, strip whitespace and quotes
            raw_args = [
                a.strip().strip("'\"")
                for a in args_str.split(",")
                if a.strip()
            ]

            parameters: dict[str, str] = {"action": action}
            if raw_args:
                parameters["entity"] = raw_args[0]
            if len(raw_args) > 1:
                parameters["relation"] = raw_args[1]

            function_calls.append(
                FunctionCall(
                    name="kg_query",
                    arguments=json.dumps(parameters, ensure_ascii=False),
                )
            )

        # Content = text with <search> tags removed
        content = _SEARCH_PATTERN.sub("", text)
        return content, function_calls
