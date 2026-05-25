"""Phase 7 Action II5: GPT-4o baseline on CWQ.

Closed-model ceiling check. Evaluates GPT-4o with our 4-tool Freebase API on the
200-Q test subset (seed=42, same as Task 37) — directly comparable to 39B's 36%.

Uses Anthropic-style or OpenAI function-calling with the same system prompt and
tool interface as the local models. ReAct loop up to max_turns=5.

Outputs:
  results/phase7/gpt4o_baseline.json
  results/phase7/gpt4o_baseline.md

Cost target: < $50. GPT-4o @ ~$2.50/M input + $10/M output × 200 Q × ~5K tokens/Q
typical ≈ $3-8.

Usage:
  export OPENAI_API_KEY=sk-...
  python scripts/phase7_ii5_gpt4o_baseline.py \\
      --kg_server_url http://localhost:18901 \\
      --n_questions 200
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import string
import time
from collections import Counter
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import requests

logger = logging.getLogger(__name__)


SYSTEM_PROMPT_REACT = """You are a knowledge graph reasoning agent. You have access to a Freebase knowledge graph via 4 tools. Your goal is to answer the user's question by querying the KG when needed.

Available tools (call via function_call):
- get_tail_relations(entity): list relations going out from entity
- get_head_relations(entity): list relations coming into entity
- get_tail_entities(entity, relation): entities reachable from entity via relation
- get_head_entities(entity, relation): entities that connect to entity via relation

After reasoning, provide your final answer as a short phrase (the name of the answer entity).
"""


OPENAI_TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "get_tail_relations",
            "description": "Get all relation types going out from an entity in the Freebase KG.",
            "parameters": {
                "type": "object",
                "properties": {"entity": {"type": "string"}},
                "required": ["entity"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_head_relations",
            "description": "Get all relation types coming into an entity.",
            "parameters": {
                "type": "object",
                "properties": {"entity": {"type": "string"}},
                "required": ["entity"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_tail_entities",
            "description": "Get entities reachable from entity via relation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "entity": {"type": "string"},
                    "relation": {"type": "string"},
                },
                "required": ["entity", "relation"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_head_entities",
            "description": "Get entities that connect to entity via relation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "entity": {"type": "string"},
                    "relation": {"type": "string"},
                },
                "required": ["entity", "relation"],
            },
        },
    },
]


def normalize_answer(text: str) -> str:
    """Match the eval_with_tools normalization."""
    text = (text or "").lower().strip()
    for article in ("a ", "an ", "the "):
        if text.startswith(article):
            text = text[len(article):]
    text = text.translate(str.maketrans("", "", string.punctuation))
    return " ".join(text.split())


def exact_match(prediction: str, answers: list[str]) -> int:
    p = normalize_answer(prediction)
    if not p:
        return 0
    for ans in answers:
        if normalize_answer(ans) == p:
            return 1
    return 0


def _kg_query(tool_name: str, entity: str, relation: str | None, kg_url: str) -> list:
    payload = {"tool": tool_name, "entity": entity, "relation": relation}
    r = requests.post(f"{kg_url}/retrieve", json=payload, timeout=30)
    r.raise_for_status()
    data = r.json()
    return data.get("results", []) or []


def _entity_variants(entity: str) -> list[str]:
    """Produce common case/format variants to try on empty responses."""
    seen = {entity}
    variants = [entity]
    candidates = [
        entity.title(),
        entity.lower(),
        entity.upper(),
        entity.replace("_", " "),
        entity.replace(" ", "_"),
        entity.replace("_", " ").title(),
    ]
    for c in candidates:
        if c and c not in seen:
            seen.add(c)
            variants.append(c)
    return variants


def call_kg_tool(
    tool_name: str,
    args: dict[str, Any],
    kg_url: str,
) -> str:
    """POST to the KG server's /retrieve endpoint, with fuzzy entity fallback."""
    entity = args.get("entity", "")
    relation = args.get("relation")
    try:
        results = _kg_query(tool_name, entity, relation, kg_url)
        if not results and entity:
            for variant in _entity_variants(entity)[1:]:
                results = _kg_query(tool_name, variant, relation, kg_url)
                if results:
                    break
        return json.dumps(results)[:2000]
    except Exception as e:
        return f"ERROR: {e}"


def run_one_question(
    client,
    question: str,
    kg_url: str,
    max_turns: int,
    model_name: str,
) -> tuple[str, int, list[dict]]:
    """Run one ReAct loop. Returns (final_answer, num_tool_calls, full_messages)."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT_REACT},
        {"role": "user", "content": question},
    ]
    tool_calls_made = 0

    for turn in range(max_turns):
        resp = client.chat.completions.create(
            model=model_name,
            messages=messages,
            tools=OPENAI_TOOLS_SCHEMA,
            tool_choice="auto",
            max_tokens=512,
            temperature=0.0,
        )
        choice = resp.choices[0]
        msg = choice.message
        messages.append(msg.model_dump(exclude_none=True))

        if choice.finish_reason == "tool_calls" and msg.tool_calls:
            for tc in msg.tool_calls:
                tool_calls_made += 1
                try:
                    args = json.loads(tc.function.arguments)
                except Exception:
                    args = {}
                tool_result = call_kg_tool(tc.function.name, args, kg_url)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": tool_result,
                    }
                )
            continue

        # model gave a final text answer
        return (msg.content or "").strip(), tool_calls_made, messages

    # max_turns exhausted without final answer — return last content
    return (msg.content or "").strip(), tool_calls_made, messages


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description="Phase 7 II5: GPT-4o CWQ baseline")
    parser.add_argument(
        "--eval_data", type=Path,
        default=Path("data/freebase/verl_cwq/test.parquet"),
    )
    parser.add_argument(
        "--filter_ids", type=Path,
        default=Path("results/task37_sample_ids.json"),
    )
    parser.add_argument(
        "--kg_server_url", type=str,
        default="http://localhost:18901",
    )
    parser.add_argument("--n_questions", type=int, default=200)
    parser.add_argument("--max_turns", type=int, default=5)
    parser.add_argument("--model", type=str, default="gpt-4o")
    parser.add_argument(
        "--output_json", type=Path,
        default=Path("results/phase7/gpt4o_baseline.json"),
    )
    parser.add_argument(
        "--output_md", type=Path,
        default=Path("results/phase7/gpt4o_baseline.md"),
    )
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY env var required")

    # Lazy import so the script is importable without openai installed
    from openai import OpenAI  # type: ignore

    client = OpenAI()

    # Load filter IDs (shared 200-Q subset with Tasks 37 / 38b / 39)
    with open(args.filter_ids) as f:
        filter_data = json.load(f)
    filter_set = set(filter_data.get("filter_ids", filter_data.get("category_b_ids", [])))
    logger.info("Filtering to %d question IDs", len(filter_set))

    table = pq.read_table(args.eval_data)
    rows = table.to_pylist()
    selected = [r for r in rows if str(r["extra_info"].get("sample_id")) in filter_set]
    selected = selected[: args.n_questions]
    logger.info("Selected %d questions", len(selected))

    em_sum = 0
    tools_sum = 0
    per_sample: list[dict] = []
    t_start = time.time()
    for i, row in enumerate(selected):
        prompt_msgs = row["prompt"]
        user_msg = next((m for m in prompt_msgs if m.get("role") == "user"), None)
        question = user_msg["content"] if user_msg else ""
        gold_answers = row["extra_info"].get("all_answers") or [row["reward_model"]["ground_truth"]]
        if hasattr(gold_answers, "tolist"):
            gold_answers = gold_answers.tolist()

        try:
            pred, ntools, _ = run_one_question(
                client, question, args.kg_server_url, args.max_turns, args.model,
            )
        except Exception as e:
            logger.warning("question %d failed: %s", i, e)
            pred, ntools = "", 0

        em = exact_match(pred, gold_answers)
        em_sum += em
        tools_sum += ntools

        per_sample.append(
            {
                "sample_id": row["extra_info"].get("sample_id"),
                "question": question,
                "gold": list(gold_answers),
                "prediction": pred,
                "em": em,
                "num_tool_calls": ntools,
            }
        )

        if (i + 1) % 10 == 0:
            logger.info(
                "  %d/%d  EM=%.3f  tools=%.2f  elapsed=%ds",
                i + 1, len(selected), em_sum / (i + 1), tools_sum / (i + 1),
                int(time.time() - t_start),
            )

    n = len(selected)
    summary = {
        "model": args.model,
        "n": n,
        "em_mean": em_sum / n if n else 0.0,
        "avg_tool_calls": tools_sum / n if n else 0.0,
        "elapsed_s": time.time() - t_start,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump({"summary": summary, "per_sample": per_sample}, f, indent=2)

    md = [
        "# Phase 7 Action II5: GPT-4o CWQ baseline\n",
        f"**Model**: {args.model}",
        f"**n**: {n}",
        f"**EM**: {100*summary['em_mean']:.2f}%",
        f"**tools/Q**: {summary['avg_tool_calls']:.2f}",
        f"**elapsed**: {int(summary['elapsed_s'])}s",
        "",
        "## Comparison (200-Q test subset, seed=42)",
        "| model | EM | tools/Q |",
        "|---|---|---|",
        f"| SFT base | 0.5% | 0.0 |",
        f"| E3@500 | 28.5% | 1.0 |",
        f"| E5b@100 | 31.0% | 2.3 |",
        f"| 39B@400 | 36.0% | 3.0 |",
        f"| **{args.model}** | **{100*summary['em_mean']:.1f}%** | **{summary['avg_tool_calls']:.1f}** |",
    ]
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_md, "w") as f:
        f.write("\n".join(md) + "\n")

    logger.info("Saved %s", args.output_json)
    print("\n".join(md))


if __name__ == "__main__":
    main()
