"""Task 40 Step 3: Generate gold multi-turn trajectories from oracle paths.

For each record in train_oracle_gold_paths.jsonl, build a ReAct-format
trajectory in the schema expected by src_verl/training/sft_multiturn.py
(same schema as data/processed/sft_trajectories.jsonl).

Per hop we emit:
    <think>I need to find ... via ...</think>
    <search>get_tail_entities("<head>", "<relation>")</search>
followed by a role=tool message:
    <information>[...KG server result...]</information>

The final assistant turn emits the <answer> tag.

CRITICAL: each trajectory is verified by actually calling the local KG
server — a trajectory is only kept if every hop's get_tail_entities call
returns a list that contains the gold tail entity (case-insensitive).

Usage:
    # Start KG server first (see task40_verify_server.job). Then:
    python scripts/task40_gen_gold_trajectories.py \
        --input data/freebase/verl_cwq/train_oracle_gold_paths.jsonl \
        --train data/freebase/verl_cwq/train.parquet \
        --kg_server_url http://localhost:18902 \
        --output data/freebase/verl_cwq/gold_kg_trajectories.jsonl \
        --stats_output results/task40_trajectory_stats.json \
        --target_count 1000
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests

logger = logging.getLogger(__name__)


CWQ_SYSTEM_PROMPT = (
    "You are a knowledge graph reasoning agent. You can query a knowledge graph "
    "to answer questions.\n\nAvailable tools:\n"
    "- get_tail_relations(entity): Get all relation types going out from an entity\n"
    "- get_head_relations(entity): Get all relation types coming into an entity\n"
    "- get_tail_entities(entity, relation): Get entities reachable from entity via relation\n"
    "- get_head_entities(entity, relation): Get entities that connect to entity via relation\n\n"
    "To use a tool, write: <search>tool_name(arguments)</search>\n"
    "To give your final answer, write: <answer>your answer</answer>\n\n"
    "Think step-by-step about what to query. Use <think>...</think> for reasoning."
)


def fetch_system_prompt_from_train(train_path: Path) -> str:
    """Grab the exact system prompt from train.parquet row 0 if available."""
    try:
        df = pd.read_parquet(train_path, columns=["prompt"])
        sp = df.iloc[0]["prompt"][0]["content"]
        if isinstance(sp, str) and sp:
            return sp
    except Exception as e:  # pragma: no cover
        logger.warning("Could not load system prompt from %s: %s", train_path, e)
    return CWQ_SYSTEM_PROMPT


def call_kg_get_tail_entities(
    server_url: str, entity: str, relation: str, timeout: float = 20.0,
) -> list[str]:
    """Call KG server get_tail_entities and return list of strings."""
    payload = {
        "action": "get_tail_entities",
        "entity": entity,
        "relation": relation,
    }
    resp = requests.post(f"{server_url}/retrieve", json=payload, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    results = data.get("results", [])
    return [str(x) for x in results]


def normalize(s: str) -> str:
    return str(s).strip().lower()


def describe_hop_nl(head: str, relation: str, tail: str) -> str:
    """Short natural-language description for the <think> tag."""
    # Freebase relations look like base.marchmadness.foo.bar. Convert to spaces.
    rel_h = relation.replace("_", " ").split(".")[-1]
    return f"the entity connected to '{head}' via the '{rel_h}' relation (chain step: {relation})"


def build_trajectory(
    record: dict[str, Any],
    system_prompt: str,
    kg_server_url: str,
    max_results_shown: int = 10,
    verbose: bool = False,
) -> tuple[dict[str, Any] | None, str]:
    """Construct and verify a trajectory from one oracle record.

    Returns (trajectory_dict, status_str). status_str is "ok" on success
    or a reason string (e.g. "tool_miss_hop_0") on rejection.
    """
    chain = record["triple_chain"]
    question = record["question"]
    gold_answer = record["gold_answer"]

    if not chain:
        return None, "empty_chain"

    messages: list[dict[str, str]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]

    for hop_idx, (head, relation, tail) in enumerate(chain):
        # Hop reasoning
        if hop_idx == 0:
            think = (
                f"I need to start from '{head}' and find "
                f"{describe_hop_nl(head, relation, tail)}."
            )
        else:
            think = (
                f"From the previous step I have '{head}'. Now I need to find "
                f"{describe_hop_nl(head, relation, tail)}."
            )
        search_call = f'get_tail_entities("{head}", "{relation}")'
        assistant_turn = f"<think>{think}</think>\n<search>{search_call}</search>"
        messages.append({"role": "assistant", "content": assistant_turn})

        # Verify by hitting the KG server
        try:
            results = call_kg_get_tail_entities(kg_server_url, head, relation)
        except Exception as e:
            if verbose:
                logger.warning("hop %d KG error for %s: %s", hop_idx, record["sample_id"], e)
            return None, f"kg_error_hop_{hop_idx}"

        tail_norm = normalize(tail)
        results_norm = [normalize(r) for r in results]
        if tail_norm not in results_norm:
            if verbose:
                logger.info(
                    "hop %d miss for %s: tail='%s' not in first %d results",
                    hop_idx, record["sample_id"], tail, len(results),
                )
            return None, f"tool_miss_hop_{hop_idx}"

        shown = results[:max_results_shown]
        tool_payload = json.dumps(shown, ensure_ascii=False)
        messages.append({
            "role": "tool",
            "content": f"<information>{tool_payload}</information>",
        })

    # Final answer turn
    final_think = (
        f"Following the knowledge graph path I end at '{chain[-1][2]}', "
        f"which answers the question."
    )
    final_turn = f"<think>{final_think}</think>\n<answer>{gold_answer}</answer>"
    messages.append({"role": "assistant", "content": final_turn})

    traj = {
        "trajectory": messages,
        "question": question,
        "gold_answer_short": gold_answer,
        "kg_path": [[h, r, t] for (h, r, t) in chain],
        "hops": len(chain),
        "num_turns": len(messages) - 2,  # exclude system/user
        "sample_id": record.get("sample_id", ""),
        "source": "task40_oracle_gold",
    }
    return traj, "ok"


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    parser = argparse.ArgumentParser(description="Task 40 Step 3: gold trajectory generation.")
    parser.add_argument("--input", type=Path,
                        default=Path("data/freebase/verl_cwq/train_oracle_gold_paths.jsonl"))
    parser.add_argument("--train", type=Path,
                        default=Path("data/freebase/verl_cwq/train.parquet"),
                        help="Used only to fetch the exact system prompt.")
    parser.add_argument("--kg_server_url", type=str, default="http://localhost:18902")
    parser.add_argument("--output", type=Path,
                        default=Path("data/freebase/verl_cwq/gold_kg_trajectories.jsonl"))
    parser.add_argument("--stats_output", type=Path,
                        default=Path("results/task40_trajectory_stats.json"))
    parser.add_argument("--target_count", type=int, default=1000,
                        help="Stop after this many verified trajectories (0 = no cap)")
    parser.add_argument("--max_input", type=int, default=0,
                        help="Stop after reading this many input records (0 = all)")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    # Sanity: server reachable
    try:
        r = requests.get(f"{args.kg_server_url}/health", timeout=30)
        logger.info("KG server health: %s %s", r.status_code, r.text[:200])
    except Exception as e:
        logger.error("KG server not reachable at %s: %s", args.kg_server_url, e)
        raise SystemExit(1)

    system_prompt = fetch_system_prompt_from_train(args.train)
    logger.info("Using system prompt (%d chars)", len(system_prompt))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.stats_output.parent.mkdir(parents=True, exist_ok=True)

    stats = {
        "n_input": 0,
        "n_kept": 0,
        "rejections": {},
        "hop_distribution_kept": {},
        "elapsed_s": 0.0,
    }

    t0 = time.time()
    with open(args.input) as inf, open(args.output, "w") as outf:
        for line in inf:
            line = line.strip()
            if not line:
                continue
            if args.max_input and stats["n_input"] >= args.max_input:
                break
            record = json.loads(line)
            stats["n_input"] += 1

            traj, status = build_trajectory(
                record, system_prompt, args.kg_server_url,
                verbose=args.verbose,
            )
            if traj is None:
                stats["rejections"][status] = stats["rejections"].get(status, 0) + 1
                continue

            outf.write(json.dumps(traj, ensure_ascii=False) + "\n")
            stats["n_kept"] += 1
            h = traj["hops"]
            stats["hop_distribution_kept"][str(h)] = (
                stats["hop_distribution_kept"].get(str(h), 0) + 1
            )

            if stats["n_kept"] % 100 == 0:
                elapsed = time.time() - t0
                logger.info(
                    "  kept=%d  input=%d  rej=%s  %.0fs",
                    stats["n_kept"], stats["n_input"], stats["rejections"], elapsed,
                )

            if args.target_count and stats["n_kept"] >= args.target_count:
                logger.info("Hit target_count=%d, stopping.", args.target_count)
                break

    stats["elapsed_s"] = time.time() - t0
    with open(args.stats_output, "w") as f:
        json.dump(stats, f, indent=2)

    logger.info(
        "Wrote %d verified trajectories to %s (input=%d, elapsed %.0fs)",
        stats["n_kept"], args.output, stats["n_input"], stats["elapsed_s"],
    )
    logger.info("Rejection reasons: %s", stats["rejections"])
    logger.info("Kept hop distribution: %s", stats["hop_distribution_kept"])


if __name__ == "__main__":
    main()
