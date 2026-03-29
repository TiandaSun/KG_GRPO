"""Collect remaining data for paper: E1 tool dist, diversity metrics, CWQ hop dist, case studies.

Trajectory format (from eval_with_tools.py):
  - sample_id, question, gold_answer, all_answers, hops
  - predicted, full_response, num_tool_calls, em, f1
  - full_response contains inline <search>action(args)</search> and <tool_response>...</tool_response>
"""

import json
import logging
import re
import statistics
from collections import Counter
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path("/lus/lfs1aip2/scratch/u6gg/ts1201.u6gg/Project/KG_GRPO")


def parse_tool_calls(full_response: str) -> list[dict]:
    """Parse <search>action(args)</search> and following <tool_response> from full_response."""
    calls = []
    # Match <search>action(args)</search>
    search_pattern = r"<search>\s*(\w+)\(([^)]*)\)\s*</search>"
    response_pattern = r"<tool_response>(.*?)</tool_response>"

    search_matches = list(re.finditer(search_pattern, full_response))
    response_matches = list(re.finditer(response_pattern, full_response, re.DOTALL))

    for i, sm in enumerate(search_matches):
        call = {"action": sm.group(1), "args": sm.group(2).strip()}
        if i < len(response_matches):
            call["response_preview"] = response_matches[i].group(1).strip()[:200]
        calls.append(call)
    return calls


def extract_tool_actions(trajectories: list[dict]) -> dict:
    """Extract tool action distribution from trajectories."""
    action_counts: Counter = Counter()
    total_calls = 0
    for traj in trajectories:
        resp = traj.get("full_response", "")
        for call in parse_tool_calls(resp):
            action_counts[call["action"]] += 1
            total_calls += 1
    return {
        "counts": dict(action_counts),
        "total": total_calls,
        "per_sample": round(total_calls / max(len(trajectories), 1), 2),
        "percentages": {k: round(v / max(total_calls, 1) * 100, 1) for k, v in action_counts.items()},
    }


def compute_diversity(trajectories: list[dict]) -> dict:
    """Compute diversity metrics from trajectories."""
    response_lengths = []
    queries = []
    answers = []

    for traj in trajectories:
        resp = traj.get("full_response", "")
        response_lengths.append(len(resp))

        pred = traj.get("predicted", "")
        answers.append(pred)

        for call in parse_tool_calls(resp):
            queries.append(call["args"])

    mean_len = statistics.mean(response_lengths) if response_lengths else 0
    std_len = statistics.stdev(response_lengths) if len(response_lengths) > 1 else 0
    query_diversity = len(set(queries)) / max(len(queries), 1)
    answer_diversity = len(set(answers)) / max(len(answers), 1)

    return {
        "n_samples": len(trajectories),
        "mean_response_length": round(mean_len, 1),
        "std_response_length": round(std_len, 1),
        "total_queries": len(queries),
        "unique_queries": len(set(queries)),
        "query_diversity": round(query_diversity, 4),
        "total_answers": len(answers),
        "unique_answers": len(set(answers)),
        "answer_diversity": round(answer_diversity, 4),
    }


def cwq_hop_distribution() -> dict:
    """Compute hop distribution from CWQ data. Check extra_info for hop counts."""
    results = {}
    for name in ["val", "test", "train"]:
        path = ROOT / f"data/freebase/verl_cwq/{name}.parquet"
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        logger.info("%s set: %d rows, columns: %s", name, len(df), list(df.columns))

        # Parse extra_info for hop information
        hop_counts: Counter = Counter()
        sample_extra = None
        for idx, row in df.iterrows():
            extra = row.get("extra_info", {})
            if isinstance(extra, str):
                try:
                    extra = json.loads(extra)
                except (json.JSONDecodeError, TypeError):
                    extra = {}
            if sample_extra is None and extra:
                sample_extra = {k: str(v)[:200] for k, v in extra.items()}

            # Try various hop fields
            hops = extra.get("hops", extra.get("num_hops", extra.get("compositionality_type", None)))
            if hops is not None:
                hop_counts[str(hops)] += 1
            else:
                hop_counts["unknown"] += 1

        results[name] = {
            "total": len(df),
            "columns": list(df.columns),
            "hop_distribution": dict(hop_counts),
        }
        if sample_extra:
            results[name]["sample_extra_info_keys"] = sample_extra

    return results


def extract_case_studies(trajectories: list[dict], n: int = 5) -> list[dict]:
    """Extract n representative examples with full details."""
    examples = []
    for traj in trajectories[:n]:
        calls = parse_tool_calls(traj.get("full_response", ""))
        examples.append({
            "question": traj.get("question", ""),
            "gold_answer": traj.get("gold_answer", ""),
            "all_answers": traj.get("all_answers", []),
            "predicted": traj.get("predicted", ""),
            "em": traj.get("em", 0),
            "f1": traj.get("f1", 0),
            "hops": traj.get("hops", None),
            "num_tool_calls": traj.get("num_tool_calls", len(calls)),
            "tool_calls": calls,
            "full_response_preview": traj.get("full_response", "")[:500],
        })
    return examples


def main() -> None:
    output = {}

    # 1. Tool action distribution for all experiments
    traj_files = {
        "e1_step_1250": ROOT / "results/trajectories/e1_outcome/step_1250/trajectories.json",
        "e2_step_200": ROOT / "results/trajectories/e2_heuristic/step_200/trajectories.json",
        "e2_step_600": ROOT / "results/trajectories/e2_heuristic/step_600/trajectories.json",
        "e2_step_1200": ROOT / "results/trajectories/e2_heuristic/step_1200/trajectories.json",
        "e3_step_500": ROOT / "results/trajectories/e3_verifiable/step_500/trajectories.json",
        "e3_step_1250": ROOT / "results/trajectories/e3_verifiable/step_1250/trajectories.json",
    }

    output["tool_distributions"] = {}
    for name, path in traj_files.items():
        if path.exists():
            logger.info("Processing tool distribution for %s...", name)
            trajs = json.loads(path.read_text())
            output["tool_distributions"][name] = extract_tool_actions(trajs)

    # 2. Diversity metrics
    output["diversity_metrics"] = {}
    for name, path in traj_files.items():
        if path.exists():
            logger.info("Computing diversity for %s...", name)
            trajs = json.loads(path.read_text())
            output["diversity_metrics"][name] = compute_diversity(trajs)

    # 3. CWQ hop distribution
    logger.info("Computing CWQ hop distribution...")
    output["cwq_hop_distribution"] = cwq_hop_distribution()

    # 4. Case study examples
    output["case_studies"] = {}
    for name, path in traj_files.items():
        if path.exists():
            logger.info("Extracting case studies for %s...", name)
            trajs = json.loads(path.read_text())
            output["case_studies"][name] = extract_case_studies(trajs)

    # Save
    out_path = ROOT / "results/paper_data_collection.json"
    out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False, default=str))
    logger.info("Saved to %s", out_path)

    # Print summary
    print("\n=== DATA COLLECTION SUMMARY ===\n")

    print("Tool Action Distributions:")
    print(f"{'Experiment':<16} {'Total':>6} {'Per-Q':>6} | {'get_tail_rel':>12} {'get_tail_ent':>12} {'get_head_rel':>12} {'get_head_ent':>12}")
    print("-" * 90)
    for name, d in output.get("tool_distributions", {}).items():
        pcts = d["percentages"]
        print(f"{name:<16} {d['total']:>6} {d['per_sample']:>6.1f} | "
              f"{pcts.get('get_tail_relations', 0):>11.1f}% "
              f"{pcts.get('get_tail_entities', 0):>11.1f}% "
              f"{pcts.get('get_head_relations', 0):>11.1f}% "
              f"{pcts.get('get_head_entities', 0):>11.1f}%")
    print()

    print("Diversity Metrics:")
    print(f"{'Experiment':<16} {'N':>4} {'MeanLen':>8} {'StdLen':>8} {'QDiv':>6} {'ADiv':>6} {'UQ/TQ':>10}")
    print("-" * 65)
    for name, m in output.get("diversity_metrics", {}).items():
        print(f"{name:<16} {m['n_samples']:>4} {m['mean_response_length']:>8.0f} {m['std_response_length']:>8.0f} "
              f"{m['query_diversity']:>6.3f} {m['answer_diversity']:>6.3f} "
              f"{m['unique_queries']}/{m['total_queries']:>4}")
    print()

    print("CWQ Dataset Distribution:")
    for split, info in output.get("cwq_hop_distribution", {}).items():
        print(f"  {split}: {info['total']} samples")
        if "hop_distribution" in info:
            print(f"    Hop dist: {info['hop_distribution']}")
        if "sample_extra_info_keys" in info:
            print(f"    Extra info keys: {list(info['sample_extra_info_keys'].keys())}")
    print()

    print(f"Case studies extracted: {sum(len(v) for v in output.get('case_studies', {}).values())} examples")
    print("\n=== DONE ===")


if __name__ == "__main__":
    main()
