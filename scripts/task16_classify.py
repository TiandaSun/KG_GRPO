#!/usr/bin/env python3
"""Task 16: Deterministic trajectory classification.

Implements the operational definitions from hpc_tasks.md (lines 84-119)
to classify each trajectory into one of 7 categories.
"""

import json
import logging
import re
import string
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path("/lus/lfs1aip2/scratch/u6gg/ts1201.u6gg/Project/KG_GRPO")

TRAJECTORY_FILES = {
    "e3_verifiable/step_500": PROJECT_ROOT / "results/trajectories/e3_verifiable/step_500/trajectories.json",
    "e3_verifiable/step_1250": PROJECT_ROOT / "results/trajectories/e3_verifiable/step_1250/trajectories.json",
    "e1_outcome/step_1250": PROJECT_ROOT / "results/trajectories/e1_outcome/step_1250/trajectories.json",
    "e2_heuristic/step_200": PROJECT_ROOT / "results/trajectories/e2_heuristic/step_200/trajectories.json",
    "e2_heuristic/step_600": PROJECT_ROOT / "results/trajectories/e2_heuristic/step_600/trajectories.json",
    "e2_heuristic/step_1200": PROJECT_ROOT / "results/trajectories/e2_heuristic/step_1200/trajectories.json",
}

CATEGORIES = [
    "correct-no-tool",
    "wrong-no-tool",
    "correct-via-tool",
    "correct-via-memory",
    "kg-incomplete",
    "tool-misuse",
    "wrong-answer",
]


def normalize_answer(text: str) -> str:
    """Normalize answer for comparison: lowercase, strip articles and punctuation."""
    text = text.lower().strip()
    # Remove articles
    for article in ["a ", "an ", "the "]:
        if text.startswith(article):
            text = text[len(article):]
    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    # Collapse whitespace
    text = " ".join(text.split())
    return text


def extract_query_entities(full_response: str) -> list:
    """Extract entity arguments from <search>action(args)</search> tags."""
    entities = []
    search_calls = re.findall(r"<search>(.*?)</search>", full_response)
    for call in search_calls:
        match = re.match(r"\w+\((.+)\)", call)
        if match:
            args = match.group(1)
            entity = args.split(",")[0].strip()
            entities.append(entity)
    return entities


def extract_tool_responses(full_response: str) -> str:
    """Extract all tool response text concatenated."""
    responses = re.findall(r"<tool_response>(.*?)</tool_response>", full_response, re.DOTALL)
    return " ".join(responses)


def check_answer_correct(predicted: str, all_answers: list) -> bool:
    """Check if normalized predicted matches any normalized gold answer."""
    norm_pred = normalize_answer(predicted)
    for ans in all_answers:
        if normalize_answer(ans) == norm_pred:
            return True
    return False


def check_answer_in_kg_response(all_answers: list, kg_response_text: str) -> bool:
    """Check if any normalized gold answer appears in normalized tool response text."""
    norm_kg = kg_response_text.lower()
    for ans in all_answers:
        if ans.lower() in norm_kg:
            return True
    return False


def check_query_entity_in_question(query_entities: list, question: str) -> bool:
    """Check if any queried entity appears in the question text."""
    q_lower = question.lower()
    for ent in query_entities:
        if ent.lower() in q_lower:
            return True
    return False


def check_kg_returned_nonempty(kg_response_text: str) -> bool:
    """Check if KG returned non-empty, meaningful results."""
    text = kg_response_text.strip()
    if not text:
        return False
    if text == "No results found":
        return False
    # Check if all tool responses are empty lists
    if all(r.strip() == "[]" for r in re.findall(r"\[.*?\]", text)):
        return False
    return True


def classify_trajectory(trajectory: dict) -> str:
    """Deterministic classification per hpc_tasks.md lines 84-119."""
    predicted = trajectory["predicted"]
    all_answers = trajectory["all_answers"]
    full_response = trajectory["full_response"]
    question = trajectory["question"]
    num_tool_calls = trajectory["num_tool_calls"]

    answer_correct = check_answer_correct(predicted, all_answers)
    has_tool_call = num_tool_calls > 0

    kg_response_text = extract_tool_responses(full_response)
    query_entities = extract_query_entities(full_response)

    answer_in_kg = check_answer_in_kg_response(all_answers, kg_response_text)
    entity_in_question = check_query_entity_in_question(query_entities, question)
    kg_nonempty = check_kg_returned_nonempty(kg_response_text)

    if not has_tool_call:
        if answer_correct:
            return "correct-no-tool"
        else:
            return "wrong-no-tool"

    if answer_correct:
        if answer_in_kg:
            return "correct-via-tool"
        else:
            return "correct-via-memory"
    else:
        if not kg_nonempty:
            return "kg-incomplete"
        elif not entity_in_question:
            return "tool-misuse"
        else:
            return "wrong-answer"


def process_experiment(name: str, trajectories: list) -> dict:
    """Classify all trajectories for one experiment, return summary."""
    counts = {cat: 0 for cat in CATEGORIES}
    classified = []

    for traj in trajectories:
        category = classify_trajectory(traj)
        counts[category] += 1
        classified.append({
            "sample_id": traj["sample_id"],
            "category": category,
            "question": traj["question"],
            "gold_answer": traj["gold_answer"],
            "all_answers": traj["all_answers"],
            "predicted": traj["predicted"],
            "num_tool_calls": traj["num_tool_calls"],
            "em": traj["em"],
            "hops": traj["hops"],
            "full_response": traj["full_response"],
        })

    total = len(trajectories)
    percentages = {cat: round(100 * counts[cat] / total, 1) if total > 0 else 0.0 for cat in CATEGORIES}

    return {
        "total": total,
        "counts": counts,
        "percentages": percentages,
        "classified": classified,
    }


def select_examples(classified: list, n_per_category: int = 5) -> dict:
    """Select up to n representative examples per category."""
    examples = {cat: [] for cat in CATEGORIES}
    for item in classified:
        cat = item["category"]
        if len(examples[cat]) < n_per_category:
            snippet = item["full_response"][:500] + ("..." if len(item["full_response"]) > 500 else "")
            examples[cat].append({
                "question": item["question"],
                "gold_answer": item["gold_answer"],
                "predicted": item["predicted"],
                "num_tool_calls": item["num_tool_calls"],
                "response_snippet": snippet,
            })
    return {k: v for k, v in examples.items() if v}


def print_distribution_table(results: dict) -> None:
    """Print a formatted distribution table."""
    header = f"{'Experiment':<30}"
    for cat in CATEGORIES:
        header += f" {cat:>18}"
    header += f" {'Total':>6}"
    print("=" * len(header))
    print("BEHAVIOR DISTRIBUTION (Task 16)")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for exp_name, data in results.items():
        row = f"{exp_name:<30}"
        for cat in CATEGORIES:
            count = data["counts"][cat]
            pct = data["percentages"][cat]
            row += f" {count:>8} ({pct:>5.1f}%)"
        row += f" {data['total']:>6}"
        print(row)

    print("=" * len(header))


def cross_check_task09(results: dict) -> None:
    """Cross-check against Task 9 results."""
    task09_path = PROJECT_ROOT / "results/task09_behavior_analysis.json"
    if not task09_path.exists():
        logger.warning("Task 9 results not found at %s", task09_path)
        return

    task09 = json.loads(task09_path.read_text())
    task09_data = task09.get("per_experiment", {})

    print("\n" + "=" * 90)
    print("CROSS-CHECK: Task 16 vs Task 9")
    print("=" * 90)
    print()
    print("Key mapping differences:")
    print("  Task 9 'format-failure' -> Task 16 splits into correct-no-tool / wrong-no-tool")
    print("  Task 9 'hallucination'  -> Task 16 maps to wrong-no-tool or wrong-answer")
    print()

    for exp_name in results:
        if exp_name not in task09_data:
            print(f"{exp_name}: NOT in Task 9 results")
            continue

        t9 = task09_data[exp_name]
        t16 = results[exp_name]

        print(f"--- {exp_name} (total: T9={t9['total']}, T16={t16['total']}) ---")

        t9_counts = t9.get("counts", {})
        print(f"  Task 9:  {dict(t9_counts)}")
        print(f"  Task 16: { {k:v for k,v in t16['counts'].items() if v > 0} }")

        # Compare correct samples
        t9_correct = t9_counts.get("correct-via-tool", 0) + t9_counts.get("correct-via-memory", 0)
        t16_correct = (t16["counts"]["correct-no-tool"] + t16["counts"]["correct-via-tool"]
                       + t16["counts"]["correct-via-memory"])
        print(f"  Total correct: T9={t9_correct}, T16={t16_correct}")

        # Compare no-tool vs format-failure
        t9_ff = t9_counts.get("format-failure", 0)
        t16_no_tool = t16["counts"]["correct-no-tool"] + t16["counts"]["wrong-no-tool"]
        if t9_ff > 0 or t16_no_tool > 0:
            print(f"  T9 format-failure={t9_ff} vs T16 no-tool={t16_no_tool}")

        print()


def main() -> None:
    all_results = {}
    all_examples = {}

    for exp_name, filepath in TRAJECTORY_FILES.items():
        if not filepath.exists():
            logger.error("File not found: %s", filepath)
            continue

        trajectories = json.loads(filepath.read_text())
        logger.info("Processing %s: %d trajectories", exp_name, len(trajectories))

        result = process_experiment(exp_name, trajectories)
        all_results[exp_name] = {
            "total": result["total"],
            "counts": result["counts"],
            "percentages": result["percentages"],
        }
        all_examples[exp_name] = select_examples(result["classified"])

    # Print distribution table
    print_distribution_table(all_results)

    # Cross-check with Task 9
    cross_check_task09(all_results)

    # Save output
    output = {
        "per_experiment": all_results,
        "examples": all_examples,
    }
    output_path = PROJECT_ROOT / "results/task16_classification.json"
    output_path.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    logger.info("Saved results to %s", output_path)


if __name__ == "__main__":
    main()
