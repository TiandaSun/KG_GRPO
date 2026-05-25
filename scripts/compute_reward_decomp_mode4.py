"""Mode-4 figure: re-score I-Self GRPO trajectories with the tool_type_bonus_retrieval_contrib
reward at all available steps {50, 100, 150, 200, 250, 300, 400}.

Re-uses the same compute_score path as scripts/task_s04_reward_decomp.py; the
only difference is the step set (S0.4 was 200/250/300/400; this adds 50/100/150
and also re-derives 300/400 from the full-trajectory dumps that became
available after S0.4 was originally run).

Reward formula (matches src_verl/rewards/verl_reward.py for this reward_type):
    r = 0.25 * r_outcome
      + 0.50 * r_tool_type_bonus   (mean over tool calls; 1.0 entity / 0.3 relation)
      + 0.25 * r_retrieval_contrib (productive_calls / total_calls)

Output (under _handoff/data/mode4/):
    mode4_reward_decomp.json   (canonical)
    mode4_reward_decomp.csv    (one row per step)
"""
from __future__ import annotations

import csv
import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path("/lus/lfs1aip2/scratch/u6gg/ts1201.u6gg/Project/KG_GRPO")
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from src_verl.rewards.verl_reward import compute_score  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("mode4_reward")

REWARD_TYPE = "tool_type_bonus_retrieval_contrib"
WEIGHTS = {"r_outcome": 0.25, "r_tool_type": 0.50, "r_retrieval_contrib": 0.25}

STEPS = [50, 100, 150, 200, 250, 300, 400]
TRAJ_TEMPLATE = "results/trajectories/phase7/39i_self_step{step}_full/step_0/trajectories.json"

OUT_DIR = PROJECT_ROOT / "_handoff/data/mode4"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_JSON = OUT_DIR / "mode4_reward_decomp.json"
OUT_CSV = OUT_DIR / "mode4_reward_decomp.csv"


def _mean(values):
    return (sum(values) / len(values)) if values else 0.0


def score_trajectories(path: Path) -> dict:
    with path.open() as f:
        trajs = json.load(f)
    r_outcome, r_tool_type, r_retrv, r_total, em_v, f1_v, ntc = [], [], [], [], [], [], []
    for t in trajs:
        full_response = t.get("full_response", "") or ""
        gold = t.get("gold_answer", "") or ""
        all_answers = t.get("all_answers") or ([gold] if gold else [])
        result = compute_score(
            data_source="cwq",
            solution_str=full_response,
            ground_truth=gold,
            extra_info={"reward_type": REWARD_TYPE, "all_answers": all_answers},
        )
        r_outcome.append(float(result.get("r_outcome", 0.0)))
        r_tool_type.append(float(result.get("r_step_avg", 0.0)))
        r_retrv.append(float(result.get("r_retrieval_contrib", 0.0)))
        r_total.append(float(result.get("score", 0.0)))
        em_v.append(float(result.get("em", 0.0)))
        f1_v.append(float(result.get("f1", 0.0)))
        ntc.append(float(result.get("num_tool_calls", 0.0)))
    return {
        "n": len(trajs),
        "r_outcome": _mean(r_outcome),
        "r_tool_type": _mean(r_tool_type),
        "r_retrieval_contrib": _mean(r_retrv),
        "r_total": _mean(r_total),
        "em": _mean(em_v),
        "f1": _mean(f1_v),
        "num_tool_calls": _mean(ntc),
    }


def main() -> int:
    per_step = {}
    for step in STEPS:
        path = PROJECT_ROOT / TRAJ_TEMPLATE.format(step=step)
        if not path.exists():
            logger.warning("step %d: %s not found, skipping", step, path)
            continue
        logger.info("step %d: scoring %s", step, path)
        s = score_trajectories(path)
        s["input_path"] = str(path)
        s["source"] = "full_trajectories"
        per_step[step] = s
        logger.info(
            "step %d done: n=%d r_outcome=%.4f r_tool_type=%.4f r_retrv=%.4f r_total=%.4f em=%.4f tools=%.2f",
            step, s["n"], s["r_outcome"], s["r_tool_type"], s["r_retrieval_contrib"],
            s["r_total"], s["em"], s["num_tool_calls"],
        )

    payload = {
        "experiment": "39i_self (I-Self / E5b+SelfV / R-selfV) seed 42",
        "reward_type": REWARD_TYPE,
        "weights": WEIGHTS,
        "scoring": "offline re-score of saved trajectories using src_verl.rewards.verl_reward.compute_score",
        "steps": per_step,
    }
    with open(OUT_JSON, "w") as f:
        json.dump(payload, f, indent=2)
    logger.info("Wrote %s", OUT_JSON)

    fieldnames = ["step", "n", "r_outcome", "r_tool_type", "r_retrieval_contrib",
                  "r_total", "em", "f1", "num_tool_calls"]
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for step in sorted(per_step.keys()):
            row = {k: per_step[step].get(k, "") for k in fieldnames if k != "step"}
            row["step"] = step
            w.writerow(row)
    logger.info("Wrote %s", OUT_CSV)
    return 0


if __name__ == "__main__":
    sys.exit(main())
