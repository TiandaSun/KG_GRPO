#!/usr/bin/env python3
"""Analyze saved trajectories and classify each into behavior categories.

Categories (in priority order):
1. format-failure: No <answer> tag or predicted is empty
2. tool-misuse: Tool calls with invalid action or empty entity arg
3. kg-incomplete: Tool returned empty results AND em=0
4. correct-via-tool: em>0, used tools, gold answer appears in tool responses
5. correct-via-memory: em>0, no tools or gold not in tool responses
6. hallucination: em=0, f1<0.1, used tools (tried but failed)
7. wrong-answer: em=0 catch-all
"""

import argparse
import json
import logging
import re
import string
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TRAJECTORY_FILES: dict[str, Path] = {
    "e3_verifiable/step_500": PROJECT_ROOT / "results/trajectories/e3_verifiable/step_500/trajectories.json",
    "e3_verifiable/step_1250": PROJECT_ROOT / "results/trajectories/e3_verifiable/step_1250/trajectories.json",
    "e1_outcome/step_1250": PROJECT_ROOT / "results/trajectories/e1_outcome/step_1250/trajectories.json",
    "e2_heuristic/step_200": PROJECT_ROOT / "results/trajectories/e2_heuristic/step_200/trajectories.json",
    "e2_heuristic/step_600": PROJECT_ROOT / "results/trajectories/e2_heuristic/step_600/trajectories.json",
    "e2_heuristic/step_1200": PROJECT_ROOT / "results/trajectories/e2_heuristic/step_1200/trajectories.json",
}

VALID_ACTIONS = {
    "get_tail_relations",
    "get_head_relations",
    "get_tail_entities",
    "get_head_entities",
}

CATEGORIES = [
    "format-failure",
    "tool-misuse",
    "kg-incomplete",
    "correct-via-tool",
    "correct-via-memory",
    "hallucination",
    "wrong-answer",
]


def normalize_answer(text: str) -> str:
    """Lowercase, strip articles, punctuation, and extra whitespace."""
    text = text.lower()
    # Remove articles
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    # Collapse whitespace
    text = " ".join(text.split())
    return text.strip()


def extract_tool_calls(full_response: str) -> list[dict[str, str]]:
    """Extract all <search>action(args)</search> calls from the response."""
    calls = []
    for m in re.finditer(r"<search>(.*?)</search>", full_response, re.DOTALL):
        raw = m.group(1).strip()
        # Parse action(args)
        match = re.match(r"(\w+)\((.*)\)", raw, re.DOTALL)
        if match:
            calls.append({"action": match.group(1), "args": match.group(2).strip()})
        else:
            calls.append({"action": raw, "args": ""})
    return calls


def extract_tool_responses(full_response: str) -> list[str]:
    """Extract all <tool_response>...</tool_response> contents."""
    return re.findall(r"<tool_response>(.*?)</tool_response>", full_response, re.DOTALL)


def gold_in_tool_responses(
    gold_answer: str, all_answers: list[str], tool_responses: list[str]
) -> bool:
    """Check if any gold answer (normalized) appears in any tool response (normalized)."""
    answers_to_check = [gold_answer] + (all_answers or [])
    for resp in tool_responses:
        resp_norm = normalize_answer(resp)
        for ans in answers_to_check:
            ans_norm = normalize_answer(ans)
            if ans_norm and ans_norm in resp_norm:
                return True
    return False


def classify_trajectory(traj: dict[str, Any]) -> str:
    """Classify a single trajectory into one behavior category."""
    full_response = traj.get("full_response", "")
    predicted = traj.get("predicted", "")
    em = float(traj.get("em", 0))
    f1 = float(traj.get("f1", 0))
    num_tool_calls = int(traj.get("num_tool_calls", 0))
    gold_answer = traj.get("gold_answer", "")
    all_answers = traj.get("all_answers", [])

    # 1. format-failure
    if "<answer>" not in full_response or not predicted or predicted.strip() == "":
        return "format-failure"

    # Extract tool calls and responses
    tool_calls = extract_tool_calls(full_response)
    tool_responses = extract_tool_responses(full_response)

    # 2. tool-misuse
    if tool_calls:
        for tc in tool_calls:
            if tc["action"] not in VALID_ACTIONS:
                return "tool-misuse"
            if not tc["args"].strip():
                return "tool-misuse"

    # 3. kg-incomplete
    if em == 0 and tool_responses:
        all_empty = all(
            resp.strip() in ("[]", "", "No results", "No results found", "No results.")
            for resp in tool_responses
        )
        if all_empty:
            return "kg-incomplete"

    # 4. correct-via-tool
    if em > 0 and num_tool_calls > 0:
        if gold_in_tool_responses(gold_answer, all_answers, tool_responses):
            return "correct-via-tool"

    # 5. correct-via-memory
    if em > 0:
        return "correct-via-memory"

    # 6. hallucination
    if em == 0 and f1 < 0.1 and num_tool_calls > 0:
        return "hallucination"

    # 7. wrong-answer
    return "wrong-answer"


def load_trajectories(path: Path) -> list[dict[str, Any]]:
    """Load trajectory JSON file."""
    with open(path, "r") as f:
        return json.load(f)


def print_table(results: dict[str, dict[str, int]], totals: dict[str, int]) -> None:
    """Print a formatted table: experiment x step -> counts and percentages."""
    # Header
    col_width = 24
    cat_width = 18
    header = f"{'Experiment':<{col_width}} {'N':>4}"
    for cat in CATEGORIES:
        header += f" | {cat:>{cat_width}}"
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for exp_key in sorted(results.keys()):
        counts = results[exp_key]
        total = totals[exp_key]
        row = f"{exp_key:<{col_width}} {total:>4}"
        for cat in CATEGORIES:
            c = counts.get(cat, 0)
            pct = 100.0 * c / total if total > 0 else 0.0
            row += f" | {c:>3} ({pct:5.1f}%)    "
        print(row)

    print("=" * len(header))


def print_examples(
    all_classified: dict[str, list[dict[str, Any]]],
    n_examples: int = 2,
) -> None:
    """Print representative examples for each category."""
    print("\n" + "=" * 80)
    print("REPRESENTATIVE EXAMPLES")
    print("=" * 80)

    for cat in CATEGORIES:
        examples = all_classified.get(cat, [])
        print(f"\n--- {cat} ({len(examples)} total) ---")
        for ex in examples[:n_examples]:
            print(f"  Question:    {ex['question'][:100]}")
            print(f"  Gold:        {ex['gold_answer']}")
            print(f"  Predicted:   {ex['predicted']}")
            print(f"  Tool calls:  {ex['num_tool_calls']}")
            print(f"  Source:      {ex['source']}")
            print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Classify trajectories into behavior categories."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "results" / "task09_behavior_analysis.json",
        help="Output JSON path",
    )
    parser.add_argument(
        "--examples",
        type=int,
        default=2,
        help="Number of representative examples per category",
    )
    args = parser.parse_args()

    # Collect results
    results: dict[str, dict[str, int]] = {}
    totals: dict[str, int] = {}
    all_classified: dict[str, list[dict[str, Any]]] = defaultdict(list)
    per_experiment: dict[str, list[dict[str, Any]]] = {}

    for exp_key, path in TRAJECTORY_FILES.items():
        if not path.exists():
            logger.warning("File not found: %s", path)
            continue

        trajectories = load_trajectories(path)
        logger.info("Loaded %d trajectories from %s", len(trajectories), exp_key)

        counts: Counter[str] = Counter()
        classified_trajs = []

        for traj in trajectories:
            cat = classify_trajectory(traj)
            counts[cat] += 1
            entry = {
                "sample_id": traj.get("sample_id", ""),
                "question": traj.get("question", ""),
                "gold_answer": traj.get("gold_answer", ""),
                "predicted": traj.get("predicted", ""),
                "num_tool_calls": traj.get("num_tool_calls", 0),
                "em": traj.get("em", 0),
                "f1": traj.get("f1", 0),
                "category": cat,
                "source": exp_key,
            }
            classified_trajs.append(entry)
            all_classified[cat].append(entry)

        results[exp_key] = dict(counts)
        totals[exp_key] = len(trajectories)
        per_experiment[exp_key] = classified_trajs

    # Print table
    print("\nBEHAVIOR CLASSIFICATION TABLE")
    print_table(results, totals)

    # Print examples
    print_examples(all_classified, n_examples=args.examples)

    # Compute aggregate stats
    aggregate: Counter[str] = Counter()
    total_all = 0
    for exp_key, counts in results.items():
        for cat, c in counts.items():
            aggregate[cat] += c
        total_all += totals[exp_key]

    print("\nAGGREGATE ACROSS ALL EXPERIMENTS")
    print(f"{'Category':<20} {'Count':>5} {'Pct':>7}")
    print("-" * 35)
    for cat in CATEGORIES:
        c = aggregate.get(cat, 0)
        pct = 100.0 * c / total_all if total_all > 0 else 0.0
        print(f"{cat:<20} {c:>5} {pct:6.1f}%")
    print(f"{'TOTAL':<20} {total_all:>5}")

    # Save JSON
    output_data = {
        "per_experiment": {
            exp_key: {
                "total": totals.get(exp_key, 0),
                "counts": results.get(exp_key, {}),
                "percentages": {
                    cat: round(
                        100.0 * results.get(exp_key, {}).get(cat, 0) / totals[exp_key],
                        1,
                    )
                    for cat in CATEGORIES
                    if exp_key in totals and totals[exp_key] > 0
                },
            }
            for exp_key in results
        },
        "aggregate": {
            "total": total_all,
            "counts": dict(aggregate),
            "percentages": (
                {
                    cat: round(100.0 * aggregate.get(cat, 0) / total_all, 1)
                    for cat in CATEGORIES
                }
                if total_all > 0
                else {}
            ),
        },
        "trajectories": per_experiment,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output_data, f, indent=2)
    logger.info("Results saved to %s", args.output)


if __name__ == "__main__":
    main()
