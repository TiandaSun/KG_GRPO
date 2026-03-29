"""Task 18: Proxy-Gold Correlation.

For each saved trajectory set, re-compute the training reward using
compute_score(), then compute Pearson correlation between per-sample
reward and per-sample EM.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from scipy import stats

# Ensure project root on PYTHONPATH
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src_verl.rewards.verl_reward import compute_score  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

TRAJECTORY_FILES = [
    ("e1_outcome",    "outcome",    "step_1250"),
    ("e2_heuristic",  "heuristic",  "step_200"),
    ("e2_heuristic",  "heuristic",  "step_600"),
    ("e2_heuristic",  "heuristic",  "step_1200"),
    ("e3_verifiable", "verifiable", "step_500"),
    ("e3_verifiable", "verifiable", "step_1250"),
]


def main() -> None:
    results_dir = PROJECT_ROOT / "results"
    traj_dir = results_dir / "trajectories"
    all_results = []

    for exp_name, reward_type, step_name in TRAJECTORY_FILES:
        fpath = traj_dir / exp_name / step_name / "trajectories.json"
        if not fpath.exists():
            logger.warning("Missing: %s", fpath)
            continue

        with open(fpath) as f:
            trajectories = json.load(f)

        rewards = []
        ems = []
        for traj in trajectories:
            solution_str = traj["full_response"]
            ground_truth = traj["gold_answer"]
            all_answers = traj.get("all_answers", [ground_truth])
            extra_info = {
                "all_answers": all_answers,
                "reward_type": reward_type,
            }

            result = compute_score(
                data_source="cwq",
                solution_str=solution_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
            )
            rewards.append(result["score"])
            ems.append(float(traj["em"]))

        n = len(rewards)
        if n < 3:
            logger.warning("Too few samples (%d) for %s/%s", n, exp_name, step_name)
            continue

        # Pearson correlation
        r_val, p_val = stats.pearsonr(rewards, ems)

        row = {
            "experiment": exp_name,
            "reward_type": reward_type,
            "step": step_name,
            "pearson_r": round(r_val, 4),
            "p_value": round(p_val, 6),
            "n_samples": n,
            "mean_reward": round(sum(rewards) / n, 4),
            "mean_em": round(sum(ems) / n, 4),
        }
        all_results.append(row)
        logger.info(
            "%s / %s: r=%.4f  p=%.6f  n=%d  mean_reward=%.4f  mean_em=%.4f",
            exp_name, step_name, r_val, p_val, n,
            row["mean_reward"], row["mean_em"],
        )

    # Print table
    print("\n" + "=" * 90)
    print(f"{'Experiment':<18} {'Step':<12} {'Pearson r':>10} {'p-value':>12} {'N':>6} {'Mean Rwd':>10} {'Mean EM':>10}")
    print("-" * 90)
    for row in all_results:
        print(
            f"{row['experiment']:<18} {row['step']:<12} {row['pearson_r']:>10.4f} "
            f"{row['p_value']:>12.6f} {row['n_samples']:>6d} {row['mean_reward']:>10.4f} {row['mean_em']:>10.4f}"
        )
    print("=" * 90)

    # Save
    out_path = results_dir / "task18_correlation.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info("Saved to %s", out_path)


if __name__ == "__main__":
    main()
