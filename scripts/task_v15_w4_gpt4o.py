"""V15-W4: GPT-4o + ReAct + Freebase 4-tool API on a 500-Q CWQ subset.

Mirrors `scripts/phase7_ii5_gpt4o_baseline.py` (which ran on the legacy 200-Q
seed=42 subset) but on the new 500-ID subset shared with V15-Q3 entropy eval.

Outputs:
  --out (default: kg_grpo_emnlp2026_v2/_handoff/data/v15_w4_gpt4o_500q.json)
    aggregate metrics + per_sample list (question, gold, predicted, em,
    contains_em, num_tool_calls, num_turns, cost_usd)

Resilient: writes each completed sample to <out>.partial.jsonl. On startup
reads that file and skips already-done sample_ids; at the end reconciles into
the final JSON.

Cost: tracked from each response.usage object × current GPT-4o pricing
constants (USD per 1M tokens). Update PRICING below if OpenAI changes prices.
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


# ---------------------------------------------------------------------------
# Pricing (USD per 1M tokens). gpt-4o snapshot 2024-08-06 @ $2.50 in / $10 out.
# Update if OpenAI changes pricing for the gpt-4o alias.
# ---------------------------------------------------------------------------
PRICING: dict[str, tuple[float, float]] = {
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-2024-08-06": (2.50, 10.00),
    "gpt-4o-2024-11-20": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
}


SYSTEM_PROMPT_REACT = (
    "You are a knowledge graph reasoning agent. You have access to a Freebase "
    "knowledge graph via 4 tools. Your goal is to answer the user's question "
    "by querying the KG when needed.\n\n"
    "Available tools (call via function_call):\n"
    "- get_tail_relations(entity): list relations going out from entity\n"
    "- get_head_relations(entity): list relations coming into entity\n"
    "- get_tail_entities(entity, relation): entities reachable from entity via relation\n"
    "- get_head_entities(entity, relation): entities that connect to entity via relation\n\n"
    "After reasoning, provide your final answer as a short phrase "
    "(the name of the answer entity)."
)


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


# ---------------------------------------------------------------------------
# Metrics helpers — mirror eval_with_tools.py / phase7_ii5 normalisation
# ---------------------------------------------------------------------------


def normalize_answer(text: str) -> str:
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
    for a in answers:
        if normalize_answer(a) == p:
            return 1
    return 0


def contains_match(prediction: str, answers: list[str]) -> int:
    p = normalize_answer(prediction)
    if not p:
        return 0
    for a in answers:
        an = normalize_answer(a)
        if an and an in p:
            return 1
    return 0


def token_f1(prediction: str, gold: str) -> float:
    p = normalize_answer(prediction).split()
    g = normalize_answer(gold).split()
    if not g:
        return 1.0 if not p else 0.0
    if not p:
        return 0.0
    common = sum((Counter(p) & Counter(g)).values())
    if common == 0:
        return 0.0
    prec = common / len(p)
    rec = common / len(g)
    return 2 * prec * rec / (prec + rec)


def best_f1(prediction: str, answers: list[str]) -> float:
    return max((token_f1(prediction, a) for a in answers), default=0.0)


# ---------------------------------------------------------------------------
# KG server adapter
# ---------------------------------------------------------------------------


def _kg_query(tool_name: str, entity: str, relation: str | None, kg_url: str) -> list:
    payload = {"action": tool_name, "entity": entity, "relation": relation}
    r = requests.post(f"{kg_url}/retrieve", json=payload, timeout=30)
    r.raise_for_status()
    data = r.json()
    return data.get("results", []) or []


def _entity_variants(entity: str) -> list[str]:
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


def call_kg_tool(tool_name: str, args: dict[str, Any], kg_url: str) -> str:
    entity = args.get("entity", "") or ""
    relation = args.get("relation")
    try:
        results = _kg_query(tool_name, entity, relation, kg_url)
        if not results and entity:
            for variant in _entity_variants(entity)[1:]:
                results = _kg_query(tool_name, variant, relation, kg_url)
                if results:
                    break
        return json.dumps(results)[:2000]
    except Exception as e:  # noqa: BLE001
        return f"ERROR: {e}"


# ---------------------------------------------------------------------------
# Cost tracking
# ---------------------------------------------------------------------------


def usage_cost(model: str, usage: Any) -> float:
    if usage is None:
        return 0.0
    in_rate, out_rate = PRICING.get(model, (2.50, 10.00))
    pt = getattr(usage, "prompt_tokens", 0) or 0
    ct = getattr(usage, "completion_tokens", 0) or 0
    return (pt * in_rate + ct * out_rate) / 1_000_000.0


# ---------------------------------------------------------------------------
# Single-question loop
# ---------------------------------------------------------------------------


def run_one_question(
    client,
    question: str,
    kg_url: str,
    max_turns: int,
    model_name: str,
) -> tuple[str, int, int, float]:
    """Returns (final_answer, num_tool_calls, num_turns, cost_usd)."""
    messages: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT_REACT},
        {"role": "user", "content": question},
    ]
    tool_calls_made = 0
    cost = 0.0
    last_text = ""

    for turn in range(max_turns):
        resp = client.chat.completions.create(
            model=model_name,
            messages=messages,
            tools=OPENAI_TOOLS_SCHEMA,
            tool_choice="auto",
            max_tokens=512,
            temperature=0.0,
        )
        cost += usage_cost(model_name, getattr(resp, "usage", None))

        choice = resp.choices[0]
        msg = choice.message
        messages.append(msg.model_dump(exclude_none=True))
        last_text = msg.content or last_text

        if choice.finish_reason == "tool_calls" and msg.tool_calls:
            for tc in msg.tool_calls:
                tool_calls_made += 1
                try:
                    args = json.loads(tc.function.arguments)
                except Exception:  # noqa: BLE001
                    args = {}
                tool_result = call_kg_tool(tc.function.name, args, kg_url)
                messages.append(
                    {"role": "tool", "tool_call_id": tc.id, "content": tool_result}
                )
            continue

        return (msg.content or "").strip(), tool_calls_made, turn + 1, cost

    return last_text.strip(), tool_calls_made, max_turns, cost


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------


def load_subset_ids(p: Path) -> set[str]:
    d = json.load(open(p))
    if isinstance(d, list):
        return set(map(str, d))
    if "subset_ids" in d:
        return set(map(str, d["subset_ids"]))
    if "filter_ids" in d:
        return set(map(str, d["filter_ids"]))
    raise ValueError(f"Cannot extract IDs from {p}")


def load_done_ids(partial_path: Path) -> tuple[set[str], list[dict]]:
    done: set[str] = set()
    records: list[dict] = []
    if not partial_path.exists():
        return done, records
    with open(partial_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            sid = rec.get("sample_id")
            if sid is not None and sid not in done:
                done.add(str(sid))
                records.append(rec)
    return done, records


def append_partial(partial_path: Path, rec: dict) -> None:
    with open(partial_path, "a") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--subset_ids", type=Path, required=True)
    p.add_argument("--kg_server_url", type=str, required=True)
    p.add_argument(
        "--eval_data", type=Path,
        default=Path("data/freebase/verl_cwq/test.parquet"),
    )
    p.add_argument("--model", type=str, default="gpt-4o")
    p.add_argument("--max_turns", type=int, default=5)
    p.add_argument(
        "--out", type=Path,
        default=Path("kg_grpo_emnlp2026_v2/_handoff/data/v15_w4_gpt4o_500q.json"),
    )
    args = p.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY missing — source ~/.bashrc before launching.")

    from openai import OpenAI  # type: ignore

    client = OpenAI()

    id_set = load_subset_ids(args.subset_ids)
    logger.info("Loaded %d subset IDs", len(id_set))
    rows = pq.read_table(args.eval_data).to_pylist()
    selected = [r for r in rows if str(r["extra_info"].get("sample_id")) in id_set]
    logger.info("Matched %d / %d rows", len(selected), len(rows))
    if not selected:
        raise SystemExit("No rows matched the subset IDs.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    partial_path = args.out.with_suffix(args.out.suffix + ".partial.jsonl")
    done_ids, prior_records = load_done_ids(partial_path)
    if done_ids:
        logger.info("Resuming: %d records already in %s", len(done_ids), partial_path)

    em_sum = 0
    cm_sum = 0
    f1_sum = 0.0
    tools_sum = 0
    turns_sum = 0
    cost_total = 0.0
    per_sample: list[dict] = list(prior_records)
    # rebuild aggregates from prior records (for accurate running totals)
    for rec in prior_records:
        em_sum += int(rec.get("em", 0))
        cm_sum += int(rec.get("contains_em", 0))
        f1_sum += float(rec.get("f1", 0.0))
        tools_sum += int(rec.get("num_tool_calls", 0))
        turns_sum += int(rec.get("num_turns", 0))
        cost_total += float(rec.get("cost_usd", 0.0))

    t_start = time.time()
    for i, row in enumerate(selected):
        sid = str(row["extra_info"].get("sample_id"))
        if sid in done_ids:
            continue
        prompt_msgs = row["prompt"]
        user_msg = next((m for m in prompt_msgs if m.get("role") == "user"), None)
        question = user_msg["content"] if user_msg else ""
        gold_answers = row["extra_info"].get("all_answers") or [
            row["reward_model"]["ground_truth"]
        ]
        if hasattr(gold_answers, "tolist"):
            gold_answers = gold_answers.tolist()
        gold_answers = list(gold_answers)

        try:
            pred, ntools, nturns, qcost = run_one_question(
                client, question, args.kg_server_url, args.max_turns, args.model,
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("q %d (%s) failed: %s", i, sid, e)
            pred, ntools, nturns, qcost = "", 0, 0, 0.0

        em = exact_match(pred, gold_answers)
        cm = contains_match(pred, gold_answers)
        f1v = best_f1(pred, gold_answers)
        rec = {
            "sample_id": sid,
            "question": question,
            "gold": gold_answers,
            "predicted": pred,
            "em": em,
            "contains_em": cm,
            "f1": f1v,
            "num_tool_calls": ntools,
            "num_turns": nturns,
            "cost_usd": qcost,
        }
        per_sample.append(rec)
        append_partial(partial_path, rec)

        em_sum += em
        cm_sum += cm
        f1_sum += f1v
        tools_sum += ntools
        turns_sum += nturns
        cost_total += qcost

        if (i + 1) % 10 == 0:
            done = len(per_sample)
            logger.info(
                "  %d/%d  EM=%.3f ContEM=%.3f F1=%.3f tools=%.2f turns=%.2f "
                "cost=$%.3f elapsed=%ds",
                done, len(selected),
                em_sum / max(1, done), cm_sum / max(1, done),
                f1_sum / max(1, done), tools_sum / max(1, done),
                turns_sum / max(1, done), cost_total,
                int(time.time() - t_start),
            )

    n = len(per_sample)
    summary = {
        "model": args.model,
        "n": n,
        "em_mean": em_sum / n if n else 0.0,
        "contains_em_mean": cm_sum / n if n else 0.0,
        "f1_mean": f1_sum / n if n else 0.0,
        "avg_tool_calls": tools_sum / n if n else 0.0,
        "avg_turns": turns_sum / n if n else 0.0,
        "total_cost_usd": cost_total,
        "elapsed_s": time.time() - t_start,
    }

    with open(args.out, "w") as f:
        json.dump({"summary": summary, "per_sample": per_sample}, f, indent=2)
    logger.info("Wrote %s", args.out)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
