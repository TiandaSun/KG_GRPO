"""Task 15: Merge KL divergence with Goodhart eval data.

Builds tables of (step, KL_divergence, cumulative_KL, training_reward_proxy, eval_EM)
for each experiment, matching by training step. Cumulative KL is the running sum
of per-step KL values, which is the standard x-axis for Goodhart analysis.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Mapping from KL data keys to (experiment label, goodhart file)
EXPERIMENT_MAP = {
    "E1_outcome": ("E1", "eval_e1_goodhart_curve.json"),
    "E2_heuristic": ("E2", "eval_e2_goodhart_curve.json"),
    "E3_verifiable": ("E3", "eval_e3_goodhart_curve.json"),
}


def compute_cumulative_kl(kl_data: dict[str, dict[str, float]]) -> dict[int, float]:
    """Compute cumulative KL divergence at each step.

    Args:
        kl_data: Dict mapping step (str) -> {"kl": float, "entropy": float}.

    Returns:
        Dict mapping step (int) -> cumulative KL (float).
    """
    steps_sorted = sorted(kl_data.keys(), key=lambda s: int(s))
    cumulative = {}
    running_sum = 0.0
    for s in steps_sorted:
        kl_val = kl_data[s]["kl"]
        # Clamp negative KL to 0 (numerical noise from PPO estimator)
        running_sum += max(kl_val, 0.0)
        cumulative[int(s)] = running_sum
    return cumulative


def find_closest_step(target: int, available_steps: list[int]) -> int:
    """Find the closest available step to the target."""
    return min(available_steps, key=lambda s: abs(s - target))


def merge_experiment(
    kl_entry: dict[str, Any],
    eval_data: dict[str, dict[str, float]],
    label: str,
) -> list[dict[str, Any]]:
    """Merge KL and eval data for one experiment.

    Args:
        kl_entry: The KL/entropy data block for this experiment.
        eval_data: The Goodhart eval data (step -> metrics).
        label: Experiment label (E1, E2, E3).

    Returns:
        List of merged rows sorted by step.
    """
    kl_data = kl_entry["data"]
    cumulative_kl = compute_cumulative_kl(kl_data)
    kl_steps = sorted(cumulative_kl.keys())

    rows = []
    for step_str, eval_metrics in sorted(eval_data.items(), key=lambda x: int(x[0])):
        step = int(step_str)
        # Find closest KL step
        closest = find_closest_step(step, kl_steps)
        kl_val = kl_data[str(closest)]["kl"]
        cum_kl = cumulative_kl[closest]
        entropy_val = kl_data[str(closest)]["entropy"]

        rows.append({
            "step": step,
            "kl_step_used": closest,
            "kl_divergence": round(kl_val, 6),
            "cumulative_kl": round(cum_kl, 6),
            "entropy": round(entropy_val, 6),
            "eval_em": eval_metrics["em"],
            "eval_contains_em": eval_metrics["contains_em"],
            "eval_f1": round(eval_metrics["f1"], 4),
            "avg_tool_calls": eval_metrics.get("avg_tool_calls", None),
        })

    return rows


def print_table(label: str, rows: list[dict[str, Any]]) -> None:
    """Print a formatted table for one experiment."""
    header = (
        f"{'Step':>6} {'KL_step':>8} {'KL_div':>10} {'Cum_KL':>10} "
        f"{'Entropy':>10} {'Eval_EM':>8} {'Cont_EM':>8} {'F1':>8} {'ToolCalls':>10}"
    )
    print(f"\n{'='*80}")
    print(f"  {label}")
    print(f"{'='*80}")
    print(header)
    print("-" * len(header))
    for r in rows:
        tc = f"{r['avg_tool_calls']:.2f}" if r["avg_tool_calls"] is not None else "N/A"
        print(
            f"{r['step']:>6} {r['kl_step_used']:>8} {r['kl_divergence']:>10.6f} "
            f"{r['cumulative_kl']:>10.4f} {r['entropy']:>10.6f} "
            f"{r['eval_em']:>8.3f} {r['eval_contains_em']:>8.3f} "
            f"{r['eval_f1']:>8.4f} {tc:>10}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Task 15: KL-Goodhart analysis")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="Directory containing result JSON files",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/task15_kl_goodhart.json"),
        help="Output JSON path",
    )
    args = parser.parse_args()

    # Load KL data
    kl_path = args.results_dir / "task13_entropy_kl_data.json"
    logger.info("Loading KL data from %s", kl_path)
    with open(kl_path) as f:
        kl_all = json.load(f)

    output: dict[str, Any] = {}

    for kl_key, (label, eval_filename) in EXPERIMENT_MAP.items():
        eval_path = args.results_dir / eval_filename
        if not eval_path.exists():
            logger.warning("Eval file %s not found, skipping %s", eval_path, label)
            continue

        logger.info("Processing %s (%s)", label, kl_key)
        with open(eval_path) as f:
            eval_data = json.load(f)

        if kl_key not in kl_all:
            logger.warning("KL key %s not found in task13 data, skipping", kl_key)
            continue

        kl_entry = kl_all[kl_key]
        rows = merge_experiment(kl_entry, eval_data, label)
        print_table(label, rows)

        # Also compute summary stats
        kl_data = kl_entry["data"]
        cumulative_kl = compute_cumulative_kl(kl_data)
        max_step = max(cumulative_kl.keys())
        total_kl = cumulative_kl[max_step]

        output[label] = {
            "experiment": kl_key,
            "total_kl_steps": kl_entry["n_points"],
            "total_cumulative_kl": round(total_kl, 6),
            "eval_checkpoints": len(rows),
            "merged_table": rows,
        }

        # Print summary
        ems = [r["eval_em"] for r in rows]
        best_idx = max(range(len(ems)), key=lambda i: ems[i])
        print(f"\n  Summary: total_cum_KL={total_kl:.4f}, "
              f"best_EM={ems[best_idx]:.3f} at step {rows[best_idx]['step']} "
              f"(cum_KL={rows[best_idx]['cumulative_kl']:.4f})")
        if best_idx < len(rows) - 1:
            last_em = ems[-1]
            print(f"  Last checkpoint EM={last_em:.3f} -- "
                  f"{'Goodhart detected (EM drops)' if last_em < ems[best_idx] else 'No Goodhart (EM still rising)'}")

    # Save output
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    logger.info("Saved merged data to %s", args.output)


if __name__ == "__main__":
    main()
