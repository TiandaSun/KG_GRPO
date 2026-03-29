"""Task 19: Behavioral Diversity.

From saved trajectories, compute per experiment+step:
1. Query diversity: unique query entities / total queries
2. Answer diversity: unique predicted answers / total samples
3. Response length: mean and std of len(full_response)
4. Query-question overlap: fraction of query entities also in question text
"""

from __future__ import annotations

import json
import logging
import re
import sys
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TRAJECTORY_FILES = [
    ("e1_outcome",    "step_1250"),
    ("e2_heuristic",  "step_200"),
    ("e2_heuristic",  "step_600"),
    ("e2_heuristic",  "step_1200"),
    ("e3_verifiable", "step_500"),
    ("e3_verifiable", "step_1250"),
]


def _parse_query_entities(response: str) -> list[str]:
    """Extract entity arguments from <search>action(args)</search> tags."""
    entities = []
    for m in re.finditer(r"<search>(.*?)</search>", response, re.DOTALL):
        call = m.group(1).strip()
        # Parse function call like get_tail_relations(Entity) or search(entity, relation)
        arg_match = re.match(r"\w+\(([^)]*)\)", call)
        if arg_match:
            args = [a.strip().strip("'\"") for a in arg_match.group(1).split(",") if a.strip()]
            entities.extend(args)
        else:
            # Bare entity or free text
            entities.append(call)
    return entities


def _normalize(text: str) -> str:
    return re.sub(r"[^\w\s]", " ", text.lower()).strip()


def compute_diversity(trajectories: list[dict]) -> dict:
    """Compute diversity metrics for a set of trajectories."""
    n = len(trajectories)
    if n == 0:
        return {}

    all_query_entities: list[str] = []
    unique_query_entities: set[str] = set()
    predicted_answers: list[str] = []
    response_lengths: list[int] = []
    overlap_fractions: list[float] = []

    for traj in trajectories:
        response = traj["full_response"]
        question = traj["question"]
        predicted = traj.get("predicted", "")

        # Response length
        response_lengths.append(len(response))

        # Predicted answers
        predicted_answers.append(_normalize(predicted))

        # Query entities
        qe = _parse_query_entities(response)
        all_query_entities.extend(qe)
        for e in qe:
            unique_query_entities.add(_normalize(e))

        # Query-question overlap
        q_norm = _normalize(question)
        q_words = set(q_norm.split())
        if qe:
            overlaps = 0
            for e in qe:
                e_words = set(_normalize(e).split())
                if e_words & q_words:
                    overlaps += 1
            overlap_fractions.append(overlaps / len(qe))

    total_queries = len(all_query_entities)
    n_unique_queries = len(unique_query_entities)
    n_unique_answers = len(set(predicted_answers))

    lengths_arr = np.array(response_lengths, dtype=float)

    return {
        "n_samples": n,
        "total_queries": total_queries,
        "unique_query_entities": n_unique_queries,
        "query_diversity": round(n_unique_queries / max(total_queries, 1), 4),
        "unique_answers": n_unique_answers,
        "answer_diversity": round(n_unique_answers / max(n, 1), 4),
        "response_length_mean": round(float(lengths_arr.mean()), 1),
        "response_length_std": round(float(lengths_arr.std()), 1),
        "query_question_overlap_mean": round(
            float(np.mean(overlap_fractions)) if overlap_fractions else 0.0, 4
        ),
        "queries_per_sample": round(total_queries / max(n, 1), 2),
    }


def main() -> None:
    traj_dir = PROJECT_ROOT / "results" / "trajectories"
    all_results = []

    for exp_name, step_name in TRAJECTORY_FILES:
        fpath = traj_dir / exp_name / step_name / "trajectories.json"
        if not fpath.exists():
            logger.warning("Missing: %s", fpath)
            continue

        with open(fpath) as f:
            trajectories = json.load(f)

        metrics = compute_diversity(trajectories)
        metrics["experiment"] = exp_name
        metrics["step"] = step_name
        all_results.append(metrics)

        logger.info(
            "%s / %s: query_div=%.4f  ans_div=%.4f  len=%.0f+/-%.0f  q_overlap=%.4f  queries/sample=%.2f",
            exp_name, step_name,
            metrics["query_diversity"],
            metrics["answer_diversity"],
            metrics["response_length_mean"],
            metrics["response_length_std"],
            metrics["query_question_overlap_mean"],
            metrics["queries_per_sample"],
        )

    # Print table
    print("\n" + "=" * 120)
    print(
        f"{'Experiment':<18} {'Step':<12} {'QryDiv':>8} {'AnsDiv':>8} "
        f"{'LenMean':>8} {'LenStd':>8} {'Q-QOvlp':>8} {'Qry/Sam':>8} {'TotQry':>8} {'UniqQry':>8}"
    )
    print("-" * 120)
    for m in all_results:
        print(
            f"{m['experiment']:<18} {m['step']:<12} {m['query_diversity']:>8.4f} {m['answer_diversity']:>8.4f} "
            f"{m['response_length_mean']:>8.1f} {m['response_length_std']:>8.1f} "
            f"{m['query_question_overlap_mean']:>8.4f} {m['queries_per_sample']:>8.2f} "
            f"{m['total_queries']:>8d} {m['unique_query_entities']:>8d}"
        )
    print("=" * 120)

    # E3 comparison
    e3_500 = next((m for m in all_results if m["experiment"] == "e3_verifiable" and m["step"] == "step_500"), None)
    e3_1250 = next((m for m in all_results if m["experiment"] == "e3_verifiable" and m["step"] == "step_1250"), None)
    if e3_500 and e3_1250:
        print("\n--- E3 Goodhart Decline Analysis (step_500 -> step_1250) ---")
        for key in ["query_diversity", "answer_diversity", "response_length_mean",
                     "response_length_std", "query_question_overlap_mean", "queries_per_sample"]:
            v500 = e3_500[key]
            v1250 = e3_1250[key]
            delta = v1250 - v500
            pct = (delta / v500 * 100) if v500 != 0 else float('inf')
            print(f"  {key:<30s}: {v500:>10.4f} -> {v1250:>10.4f}  (delta={delta:+.4f}, {pct:+.1f}%)")

    # Save
    out_path = PROJECT_ROOT / "results" / "task19_diversity.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info("Saved to %s", out_path)


if __name__ == "__main__":
    main()
