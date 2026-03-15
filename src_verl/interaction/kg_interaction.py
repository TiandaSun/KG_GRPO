"""verl BaseTool plugin for KG retrieval tool calls.

Implements a KG query tool that the agent can call during multi-turn
GRPO rollouts. Each tool call queries the KG server and returns results.
Step-level rewards are computed based on the selected reward function.

verl's ToolAgentLoop calls:
  1. create()  — once per trajectory, stores gold answer and KG path
  2. execute() — each time the model calls the tool, returns (response, step_reward, metrics)
  3. calc_reward() — terminal reward (answer correctness)
  4. release() — cleanup per-instance state

The tool is registered via tool_config.yaml and loaded by verl's
initialize_tools_from_config().
"""

from __future__ import annotations

import json
import logging
from typing import Any, Optional
from uuid import uuid4

import requests

from verl.tools.base_tool import BaseTool
from verl.tools.schemas import OpenAIFunctionToolSchema, ToolResponse

# Side-effect import: registers "kg_search" ToolParser before ToolAgentLoop.initialize()
# This MUST happen before verl calls ToolParser.get_tool_parser("kg_search", ...)
import src_verl.interaction.search_tool_parser  # noqa: F401

from src_verl.rewards.common import (
    compute_exact_match,
    compute_token_f1,
    extract_answer,
    normalize_answer,
)

logger = logging.getLogger(__name__)


class KGQueryTool(BaseTool):
    """KG retrieval tool for multi-turn agent interaction.

    Supports 4 query types:
    - get_tail_relations(entity)
    - get_head_relations(entity)
    - get_tail_entities(entity, relation)
    - get_head_entities(entity, relation)
    """

    def __init__(self, config: dict[str, Any], tool_schema: OpenAIFunctionToolSchema) -> None:
        super().__init__(config, tool_schema)
        self._kg_server_url: str = config.get("kg_server_url", "http://localhost:8001")
        self._reward_type: str = config.get("reward_type", "verifiable")
        self._instances: dict[str, dict[str, Any]] = {}

        # Optional: preloaded BFS distances for R_progress
        self._distances: dict[str, dict[str, int]] | None = None
        distances_path = config.get("distances_path")
        if distances_path:
            import pickle
            from pathlib import Path

            dist_file = Path(distances_path)
            if dist_file.exists():
                with open(dist_file, "rb") as f:
                    self._distances = pickle.load(f)
                logger.info("Loaded BFS distances from %s", distances_path)

    async def create(
        self,
        instance_id: Optional[str] = None,
        **kwargs: Any,
    ) -> tuple[str, ToolResponse]:
        """Initialize per-trajectory state."""
        if instance_id is None:
            instance_id = str(uuid4())

        # Extract gold data from create_kwargs (injected via dataset tools_kwargs)
        create_kwargs = kwargs.get("create_kwargs", {})
        gold_answer = create_kwargs.get("gold_answer", "")
        kg_path = create_kwargs.get("kg_path", [])

        # Build path entity/relation sets for reward computation
        path_entities: set[str] = set()
        path_relations: set[str] = set()
        for triple in kg_path:
            if len(triple) >= 3:
                path_entities.add(triple[0].lower())
                path_entities.add(triple[2].lower())
                path_relations.add(triple[1].lower())

        self._instances[instance_id] = {
            "gold_answer": gold_answer,
            "kg_path": kg_path,
            "path_entities": path_entities,
            "path_relations": path_relations,
            "step_rewards": [],
            "queried_entities": [],
            "prev_distance": None,
        }

        return instance_id, ToolResponse()

    async def execute(
        self,
        instance_id: str,
        parameters: dict[str, Any],
        **kwargs: Any,
    ) -> tuple[ToolResponse, float, dict[str, Any]]:
        """Execute a KG query and compute step reward."""
        action = parameters.get("action", "get_tail_relations")
        entity = parameters.get("entity", "")
        relation = parameters.get("relation", None)

        # Query KG server
        results = self._query_kg(action, entity, relation)
        # Wrap in <information> tags to match SFT training data format.
        # Without this, the model sees raw text during GRPO rollouts but was
        # trained on <information>...</information> wrapped responses in SFT.
        raw_text = str(results) if results else "No results found."
        response_text = f"<information>{raw_text}</information>"

        # Compute step reward based on reward type
        inst = self._instances.get(instance_id, {})
        step_reward = 0.0
        metrics: dict[str, Any] = {"action": action, "entity": entity, "num_results": len(results)}

        if self._reward_type == "outcome":
            # No step reward for outcome-only
            step_reward = 0.0
        elif self._reward_type == "heuristic":
            step_reward = self._heuristic_step_reward(inst, entity, relation)
        elif self._reward_type == "verifiable":
            step_reward = self._verifiable_step_reward(
                inst, action, entity, relation, results
            )
            metrics["r_valid"] = 1.0 if results else 0.5
            metrics["r_on_path"] = self._on_path_score(inst, entity, relation, action)

        inst["step_rewards"].append(step_reward)
        inst["queried_entities"].append(entity.lower())

        return ToolResponse(text=response_text), step_reward, metrics

    async def calc_reward(self, instance_id: str, **kwargs: Any) -> float:
        """Compute terminal reward (answer correctness).

        This is called after the trajectory ends. The agent_data
        in kwargs contains the full message history.
        """
        inst = self._instances.get(instance_id, {})
        gold_answer = inst.get("gold_answer", "")
        if not gold_answer:
            return 0.0

        # Extract predicted answer from the agent_data messages
        agent_data = kwargs.get("agent_data")
        if agent_data and hasattr(agent_data, "messages"):
            predicted = extract_answer(agent_data.messages)
        else:
            predicted = ""

        if not predicted:
            return 0.0

        em = compute_exact_match(predicted, gold_answer)
        f1 = compute_token_f1(predicted, gold_answer)
        return 0.5 * em + 0.5 * f1

    async def release(self, instance_id: str, **kwargs: Any) -> None:
        """Cleanup per-instance state."""
        self._instances.pop(instance_id, None)

    # ------------------------------------------------------------------
    # KG server query
    # ------------------------------------------------------------------

    def _query_kg(
        self,
        action: str,
        entity: str,
        relation: str | None = None,
    ) -> list[str]:
        """Query the KG server."""
        payload: dict[str, Any] = {"action": action, "entity": entity}
        if relation:
            payload["relation"] = relation

        try:
            resp = requests.post(
                f"{self._kg_server_url}/retrieve",
                json=payload,
                timeout=10,
            )
            resp.raise_for_status()
            return resp.json().get("results", [])
        except (requests.RequestException, json.JSONDecodeError) as e:
            logger.warning("KG query failed: %s", e)
            return []

    # ------------------------------------------------------------------
    # Step reward functions
    # ------------------------------------------------------------------

    def _heuristic_step_reward(
        self,
        inst: dict[str, Any],
        entity: str,
        relation: str | None,
    ) -> float:
        """Heuristic step reward: entity overlap + reachability."""
        path_entities = inst.get("path_entities", set())
        entity_lower = entity.lower()

        # Entity on path?
        on_path = 1.0 if entity_lower in path_entities else 0.0

        # Relation on path?
        path_relations = inst.get("path_relations", set())
        rel_on_path = 0.0
        if relation and relation.lower() in path_relations:
            rel_on_path = 1.0

        return 0.5 * on_path + 0.5 * rel_on_path

    def _verifiable_step_reward(
        self,
        inst: dict[str, Any],
        action: str,
        entity: str,
        relation: str | None,
        results: list[str],
    ) -> float:
        """Verifiable step reward: valid + on_path + progress + coherence."""
        # R_valid
        valid_actions = {
            "get_tail_relations", "get_head_relations",
            "get_tail_entities", "get_head_entities",
        }
        r_valid = 0.0
        if action in valid_actions:
            r_valid = 1.0 if results else 0.5

        # R_on_path
        r_on_path = self._on_path_score(inst, entity, relation, action)

        # R_progress (BFS distance to answer)
        r_progress = 0.5  # neutral default
        if self._distances:
            answer_lower = inst.get("gold_answer", "").lower()
            answer_dists = self._distances.get(answer_lower)
            if answer_dists:
                entity_lower = entity.lower()
                curr_dist = answer_dists.get(entity_lower)
                prev_dist = inst.get("prev_distance")
                if curr_dist is not None:
                    if prev_dist is None:
                        r_progress = max(0.0, 1.0 - curr_dist / 5.0)
                    elif curr_dist < prev_dist:
                        r_progress = 1.0
                    elif curr_dist == prev_dist:
                        r_progress = 0.5
                    else:
                        r_progress = 0.0
                    inst["prev_distance"] = float(curr_dist)
                else:
                    r_progress = 0.0

        # R_coherence — simplified: did the queried entity appear in previous results?
        r_coherence = 0.5
        prev_entities = inst.get("queried_entities", [])
        if prev_entities and entity.lower() in {e for e in prev_entities}:
            r_coherence = 0.3  # Penalize re-querying same entity

        return 0.3 * r_valid + 0.3 * r_on_path + 0.2 * r_progress + 0.2 * r_coherence

    def _on_path_score(
        self,
        inst: dict[str, Any],
        entity: str,
        relation: str | None,
        action: str,
    ) -> float:
        """R_on_path: Is the queried entity/relation on the ground truth path?"""
        path_entities = inst.get("path_entities", set())
        path_relations = inst.get("path_relations", set())

        entity_on = entity.lower() in path_entities
        rel_on = relation.lower() in path_relations if relation else False

        if action in ("get_tail_entities", "get_head_entities"):
            if entity_on and rel_on:
                return 1.0
            elif entity_on:
                return 0.7
            elif rel_on:
                return 0.3
            return 0.0
        else:
            return 1.0 if entity_on else 0.0
