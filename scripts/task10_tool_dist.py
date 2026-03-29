"""Analyze tool call distributions from saved trajectory JSON files.

Produces:
  - Per experiment+step: min/max/mean/median/std of num_tool_calls + text histogram
  - Uniformity check for E3 (all-1 vs bimodal)
  - Cross-tabulation: tool calls by hop count, per experiment
  - E3 step 500 correct-with-1-call action breakdown
  - JSON output at results/task10_tool_distribution.json
"""

import argparse
import json
import logging
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# -- Trajectory file definitions -----------------------------------------------

TRAJECTORY_FILES: list[dict[str, str]] = [
    {"experiment": "e3_verifiable", "step": "500",
     "path": "results/trajectories/e3_verifiable/step_500/trajectories.json"},
    {"experiment": "e3_verifiable", "step": "1250",
     "path": "results/trajectories/e3_verifiable/step_1250/trajectories.json"},
    {"experiment": "e1_outcome", "step": "1250",
     "path": "results/trajectories/e1_outcome/step_1250/trajectories.json"},
    {"experiment": "e2_heuristic", "step": "200",
     "path": "results/trajectories/e2_heuristic/step_200/trajectories.json"},
    {"experiment": "e2_heuristic", "step": "600",
     "path": "results/trajectories/e2_heuristic/step_600/trajectories.json"},
    {"experiment": "e2_heuristic", "step": "1200",
     "path": "results/trajectories/e2_heuristic/step_1200/trajectories.json"},
]


def load_trajectories(project_root: Path) -> dict[str, list[dict[str, Any]]]:
    """Load all trajectory files. Key = 'experiment/step'."""
    data: dict[str, list[dict[str, Any]]] = {}
    for spec in TRAJECTORY_FILES:
        fpath = project_root / spec["path"]
        key = f"{spec['experiment']}/step_{spec['step']}"
        if not fpath.exists():
            logger.warning("File not found: %s", fpath)
            continue
        with open(fpath, "r") as f:
            data[key] = json.load(f)
        logger.info("Loaded %d samples from %s", len(data[key]), key)
    return data


# -- Analysis 1: per-experiment descriptive stats + histogram ------------------

def compute_stats(values: list[int]) -> dict[str, float]:
    """Return min, max, mean, median, std for a list of ints."""
    return {
        "min": min(values),
        "max": max(values),
        "mean": round(statistics.mean(values), 3),
        "median": round(statistics.median(values), 1),
        "std": round(statistics.stdev(values), 3) if len(values) > 1 else 0.0,
        "n": len(values),
    }


def text_histogram(values: list[int], bar_char: str = "\u2588", max_bar: int = 40) -> str:
    """Return a text-based histogram string."""
    counts = Counter(values)
    if not counts:
        return "(empty)"
    max_count = max(counts.values())
    lines = []
    for k in sorted(counts.keys()):
        bar_len = int(counts[k] / max_count * max_bar) if max_count > 0 else 0
        bar = bar_char * max(bar_len, 1)
        label = f"{k} call{'s' if k != 1 else ' '}"
        lines.append(f"  {label:>8s}: {bar} {counts[k]}")
    return "\n".join(lines)


def analysis_1(all_data: dict[str, list[dict]]) -> dict[str, Any]:
    """Descriptive stats and histogram per experiment+step."""
    results: dict[str, Any] = {}
    for key, samples in all_data.items():
        tc = [s["num_tool_calls"] for s in samples]
        stats = compute_stats(tc)
        dist = dict(sorted(Counter(tc).items()))
        results[key] = {"stats": stats, "distribution": {str(k): v for k, v in dist.items()}}

        print(f"\n{'=' * 60}")
        print(f"  {key}  (n={stats['n']})")
        print(f"{'=' * 60}")
        print(f"  mean={stats['mean']}  median={stats['median']}  "
              f"std={stats['std']}  min={stats['min']}  max={stats['max']}")
        print(text_histogram(tc))
    return results


# -- Analysis 2: E3 uniformity check ------------------------------------------

def analysis_2(all_data: dict[str, list[dict]]) -> dict[str, Any]:
    """Check whether E3's mean~1.0 is truly uniform or bimodal."""
    results: dict[str, Any] = {}
    print(f"\n{'=' * 60}")
    print("  E3 Uniformity Check: is tool_calls=1.0 truly uniform?")
    print(f"{'=' * 60}")
    for key, samples in all_data.items():
        if not key.startswith("e3_verifiable"):
            continue
        tc = [s["num_tool_calls"] for s in samples]
        counts = Counter(tc)
        n = len(tc)
        pct_exactly_1 = counts.get(1, 0) / n * 100 if n else 0
        pct_0 = counts.get(0, 0) / n * 100 if n else 0
        pct_2plus = sum(v for k, v in counts.items() if k >= 2) / n * 100 if n else 0

        is_uniform = pct_exactly_1 > 90
        verdict = "UNIFORM (all 1)" if is_uniform else "BIMODAL (mix of 0 and 2+)"

        info = {
            "pct_exactly_1": round(pct_exactly_1, 1),
            "pct_0": round(pct_0, 1),
            "pct_2_plus": round(pct_2plus, 1),
            "verdict": verdict,
            "distribution": {str(k): v for k, v in sorted(counts.items())},
        }
        results[key] = info

        print(f"\n  {key}:")
        print(f"    exactly 1 call: {pct_exactly_1:.1f}%")
        print(f"    0 calls:        {pct_0:.1f}%")
        print(f"    2+ calls:       {pct_2plus:.1f}%")
        print(f"    --> {verdict}")
    return results


# -- Analysis 3: cross-tabulate tool calls by hop count ------------------------

def analysis_3(all_data: dict[str, list[dict]]) -> dict[str, Any]:
    """Tool calls by hop count, per experiment."""
    results: dict[str, Any] = {}
    print(f"\n{'=' * 60}")
    print("  Tool Calls x Hop Count (per experiment)")
    print(f"{'=' * 60}")

    for key, samples in all_data.items():
        hop_tc: dict[int, list[int]] = defaultdict(list)
        for s in samples:
            hop_tc[s["hops"]].append(s["num_tool_calls"])

        table: dict[str, dict[str, float]] = {}
        print(f"\n  {key}:")
        print(f"  {'hops':>6s} | {'n':>4s} | {'mean_tc':>7s} | {'med_tc':>6s} | {'mean_em':>7s}")
        print(f"  {'-' * 6}-+-{'-' * 4}-+-{'-' * 7}-+-{'-' * 6}-+-{'-' * 7}")

        for hop in sorted(hop_tc.keys()):
            tcs = hop_tc[hop]
            hop_samples = [s for s in samples if s["hops"] == hop]
            ems = [s["em"] for s in hop_samples]
            mean_tc = statistics.mean(tcs)
            med_tc = statistics.median(tcs)
            mean_em = statistics.mean(ems) if ems else 0.0
            table[str(hop)] = {
                "n": len(tcs),
                "mean_tool_calls": round(mean_tc, 2),
                "median_tool_calls": round(med_tc, 1),
                "mean_em": round(mean_em, 3),
            }
            print(f"  {hop:>6d} | {len(tcs):>4d} | {mean_tc:>7.2f} | {med_tc:>6.1f} | {mean_em:>7.3f}")

        results[key] = table
    return results


# -- Analysis 4: E3 step_500 action breakdown for correct 1-call samples ------

SEARCH_PATTERN = re.compile(r"<search>([\w_]+)\(")


def analysis_4(all_data: dict[str, list[dict]]) -> dict[str, Any]:
    """For E3 step 500, correct (em>0) with exactly 1 tool call: what actions?"""
    key = "e3_verifiable/step_500"
    results: dict[str, Any] = {}

    print(f"\n{'=' * 60}")
    print("  E3 step_500: Actions used by correct 1-call samples")
    print(f"{'=' * 60}")

    if key not in all_data:
        print("  (data not available)")
        return results

    samples = all_data[key]
    correct_1call = [s for s in samples if s["em"] > 0 and s["num_tool_calls"] == 1]
    all_1call = [s for s in samples if s["num_tool_calls"] == 1]

    print(f"  Total samples: {len(samples)}")
    print(f"  Samples with exactly 1 tool call: {len(all_1call)}")
    print(f"  Correct (em>0) with exactly 1 call: {len(correct_1call)}")

    action_counts: Counter = Counter()
    for s in correct_1call:
        actions = SEARCH_PATTERN.findall(s.get("full_response", ""))
        for a in actions:
            action_counts[a] += 1

    # Also compute for all 1-call samples (not just correct)
    action_counts_all: Counter = Counter()
    for s in all_1call:
        actions = SEARCH_PATTERN.findall(s.get("full_response", ""))
        for a in actions:
            action_counts_all[a] += 1

    print(f"\n  Action distribution (correct 1-call, n={len(correct_1call)}):")
    for action, cnt in action_counts.most_common():
        print(f"    {action:30s}  {cnt:>4d}")

    print(f"\n  Action distribution (all 1-call, n={len(all_1call)}):")
    for action, cnt in action_counts_all.most_common():
        print(f"    {action:30s}  {cnt:>4d}")

    results = {
        "n_total": len(samples),
        "n_1call": len(all_1call),
        "n_correct_1call": len(correct_1call),
        "actions_correct_1call": dict(action_counts.most_common()),
        "actions_all_1call": dict(action_counts_all.most_common()),
    }
    return results


# -- Main ----------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze tool call distributions from trajectories.")
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parent.parent,
        help="Project root directory (default: parent of scripts/)",
    )
    args = parser.parse_args()
    project_root: Path = args.project_root

    all_data = load_trajectories(project_root)
    if not all_data:
        logger.error("No trajectory data loaded. Exiting.")
        sys.exit(1)

    output: dict[str, Any] = {}

    output["descriptive_stats"] = analysis_1(all_data)
    output["e3_uniformity"] = analysis_2(all_data)
    output["tool_calls_by_hops"] = analysis_3(all_data)
    output["e3_step500_actions"] = analysis_4(all_data)

    # Save JSON
    out_path = project_root / "results" / "task10_tool_distribution.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
