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

_VALID_ACTIONS = frozenset({
    "get_tail_relations",
    "get_head_relations",
    "get_tail_entities",
    "get_head_entities",
})

# Matches: <search>action_name(arg1, arg2, ...)</search>
# Captures: group(1) = action name, group(2) = comma-separated arguments
_SEARCH_PATTERN = re.compile(
    r"<search>\s*(\w+)\(([^)]*)\)\s*</search>",
    re.DOTALL,
)

# Also matches bare angle bracket format: <action_name(arg1, arg2, ...)>
# The model sometimes generates <get_tail_relations(dog)> instead of
# <search>get_tail_relations(dog)</search>. Only match known action names
# to avoid false positives on arbitrary HTML-like tags.
_BARE_ACTIONS = "|".join(_VALID_ACTIONS) + "|" + "|".join(
    k for k in ["kg_query", "search", "query", "get_relations", "get_entities"]
)
_BARE_PATTERN = re.compile(
    r"<(" + _BARE_ACTIONS + r")\(([^)]*)\)>",
    re.DOTALL,
)

# Aliases: model sometimes uses tool schema name "kg_query" or shortened forms
# instead of the specific action name. Map these to a sentinel so we can
# infer the correct action from the number of arguments.
_ACTION_ALIASES: dict[str, str | None] = {
    "kg_query": None,       # infer from args
    "search": None,         # infer from args
    "query": None,          # infer from args
    "get_relations": "get_tail_relations",
    "get_entities": "get_tail_entities",
}


def _resolve_action(raw_action: str, num_args: int) -> str | None:
    """Resolve a raw action name to a valid action.

    Returns the resolved action name, or None if unresolvable.
    """
    if raw_action in _VALID_ACTIONS:
        return raw_action
    mapped = _ACTION_ALIASES.get(raw_action)
    if mapped is not None:
        return mapped
    if raw_action in _ACTION_ALIASES:
        # Alias with None = infer from argument count
        if num_args >= 2:
            return "get_tail_entities"
        return "get_tail_relations"
    return None


@ToolParser.register("kg_search")
class KGSearchToolParser(ToolParser):
    """Parser for <search>action(args)</search> format used in KG reasoning.

    Converts each <search> tag into a FunctionCall with:
        name = "kg_query"  (matches the tool registered in kg_tool_config.yaml)
        arguments = JSON {"action": ..., "entity": ..., "relation": ...}
    """

    @rollout_trace_op
    async def extract_tool_calls(
        self, responses_ids: list[int], tools: list | None = None
    ) -> tuple[str, list[FunctionCall]]:
        loop = asyncio.get_running_loop()
        text = await loop.run_in_executor(
            None, self.tokenizer.decode, responses_ids
        )

        # Diagnostic: always log decoded text for debugging tool call generation
        if logger.isEnabledFor(logging.INFO) or os.getenv("KG_DEBUG_PARSER"):
            preview = text[:300].replace("\n", "\\n")
            logger.warning("KGSearchParser decoded text (first 300 chars): %s", preview)

        # Try both <search>action(args)</search> and bare <action(args)> formats
        matches = _SEARCH_PATTERN.findall(text)
        bare_matches = _BARE_PATTERN.findall(text)
        all_matches = matches + bare_matches

        if not all_matches:
            return text, []

        function_calls: list[FunctionCall] = []

        for raw_action, args_str in all_matches:
            # Parse comma-separated arguments, strip whitespace and quotes
            raw_args = [
                a.strip().strip("'\"")
                for a in args_str.split(",")
                if a.strip()
            ]

            action = _resolve_action(raw_action, len(raw_args))
            if action is None:
                logger.warning("Unknown action in <search> tag: %s", raw_action)
                continue

            if action != raw_action:
                logger.info("Mapped action alias %s -> %s", raw_action, action)

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

        # Content = text with tool call tags removed
        content = _SEARCH_PATTERN.sub("", text)
        content = _BARE_PATTERN.sub("", content)
        return content, function_calls
