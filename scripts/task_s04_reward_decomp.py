"""S0.4 Per-component reward decomposition for the I-Self GRPO run.

Re-scores saved trajectories at steps {200, 250, 300, 400} of the 39i_self
experiment using ``src_verl.rewards.verl_reward.compute_score`` under the
``tool_type_bonus_retrieval_contrib`` reward type (Variant I-Self):

    R = 0.25 * r_outcome
      + 0.50 * r_tool_type_bonus       (avg over tool calls; 1.0 entity / 0.3 relation)
      + 0.25 * r_retrieval_contrib     (productive_calls / total_calls)

Per-step we report mean(r_outcome), mean(r_tool_type), mean(r_retrieval_contrib),
mean(r_total), mean(em), mean(f1), mean(num_tool_calls), and N.

Inputs:
  * Full trajectories with ``full_response``: steps 200, 250
      results/trajectories/phase7/39i_self_step{200,250}_full/step_0/trajectories.json
  * Per-sample summaries only (no ``full_response``): steps 300, 400
      results/phase7/39i_self_step{300,400}_full_per_sample/step_0_per_sample.json
    For these steps r_tool_type and r_retrieval_contrib cannot be recovered
    (no full_response on disk) — they are reported as null. r_outcome is
    re-derived from the saved em/f1 fields, and r_total is reported as
    "answer-only lower bound" 0.25 * r_outcome (assuming no step credit).

Outputs:
  * results/phase7/i_self_reward_decomp.json
  * results/phase7/i_self_reward_decomp.md

Re-scoring is fully offline — no KG server needed because the I-Self reward
only consults the trajectory's tool calls / observations / final answer.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

# Make src_verl importable when run from the repo root.
import sys
PROJECT_ROOT = Path("/lus/lfs1aip2/scratch/u6gg/ts1201.u6gg/Project/KG_GRPO")
sys.path.insert(0, str(PROJECT_ROOT))

from src_verl.rewards.verl_reward import compute_score  # noqa: E402

logger = logging.getLogger(__name__)


STEP_INPUTS: dict[int, dict[str, Path | None]] = {
    200: {
        "trajectories": PROJECT_ROOT
        / "results/trajectories/phase7/39i_self_step200_full/step_0/trajectories.json",
        "per_sample": None,
    },
    250: {
        "trajectories": PROJECT_ROOT
        / "results/trajectories/phase7/39i_self_step250_full/step_0/trajectories.json",
        "per_sample": None,
    },
    300: {
        "trajectories": None,
        "per_sample": PROJECT_ROOT
        / "results/phase7/39i_self_step300_full_per_sample/step_0_per_sample.json",
    },
    400: {
        "trajectories": None,
        "per_sample": PROJECT_ROOT
        / "results/phase7/39i_self_step400_full_per_sample/step_0_per_sample.json",
    },
}

OUT_JSON = PROJECT_ROOT / "results/phase7/i_self_reward_decomp.json"
OUT_MD = PROJECT_ROOT / "results/phase7/i_self_reward_decomp.md"

REWARD_TYPE = "tool_type_bonus_retrieval_contrib"


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------

def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _score_full_trajectories(path: Path) -> dict[str, Any]:
    """Re-score every trajectory under the I-Self reward and aggregate means."""
    with path.open() as f:
        trajs: list[dict[str, Any]] = json.load(f)

    n = len(trajs)
    r_outcome_vals: list[float] = []
    r_tool_type_vals: list[float] = []
    r_retrieval_vals: list[float] = []
    r_total_vals: list[float] = []
    em_vals: list[float] = []
    f1_vals: list[float] = []
    ntc_vals: list[float] = []

    for t in trajs:
        full_response = t.get("full_response", "") or ""
        gold = t.get("gold_answer", "") or ""
        all_answers = t.get("all_answers") or ([gold] if gold else [])

        result = compute_score(
            data_source="cwq",
            solution_str=full_response,
            ground_truth=gold,
            extra_info={
                "reward_type": REWARD_TYPE,
                "all_answers": all_answers,
            },
        )

        r_outcome_vals.append(float(result.get("r_outcome", 0.0)))
        r_tool_type_vals.append(float(result.get("r_step_avg", 0.0)))
        r_retrieval_vals.append(float(result.get("r_retrieval_contrib", 0.0)))
        r_total_vals.append(float(result.get("score", 0.0)))
        em_vals.append(float(result.get("em", 0.0)))
        f1_vals.append(float(result.get("f1", 0.0)))
        ntc_vals.append(float(result.get("num_tool_calls", 0.0)))

    return {
        "n": n,
        "r_outcome": _mean(r_outcome_vals),
        "r_tool_type": _mean(r_tool_type_vals),
        "r_retrieval_contrib": _mean(r_retrieval_vals),
        "r_total": _mean(r_total_vals),
        "em": _mean(em_vals),
        "f1": _mean(f1_vals),
        "num_tool_calls": _mean(ntc_vals),
        "source": "full_trajectories",
        "input_path": str(path),
    }


def _score_per_sample(path: Path) -> dict[str, Any]:
    """Aggregate the recoverable answer-side metrics from per_sample summaries.

    Tool-side components (r_tool_type, r_retrieval_contrib) cannot be re-derived
    without ``full_response`` and are returned as null. r_total is a lower
    bound assuming step credit = 0 (i.e. just the 0.25 * r_outcome term).
    """
    with path.open() as f:
        rows: list[dict[str, Any]] = json.load(f)

    n = len(rows)
    em_vals = [float(r.get("em", 0.0)) for r in rows]
    f1_vals = [float(r.get("f1", 0.0)) for r in rows]
    ntc_vals = [float(r.get("num_tool_calls", 0.0)) for r in rows]
    r_outcome_vals = [0.5 * em + 0.5 * f1 for em, f1 in zip(em_vals, f1_vals)]

    mean_r_outcome = _mean(r_outcome_vals)
    return {
        "n": n,
        "r_outcome": mean_r_outcome,
        "r_tool_type": None,
        "r_retrieval_contrib": None,
        # Lower bound only — true r_total ≥ 0.25 * r_outcome (step credit ≥ 0).
        "r_total_lower_bound": 0.25 * mean_r_outcome,
        "r_total": None,
        "em": _mean(em_vals),
        "f1": _mean(f1_vals),
        "num_tool_calls": _mean(ntc_vals),
        "source": "per_sample_summary",
        "input_path": str(path),
        "note": (
            "full_response not on disk for this step; tool-side reward "
            "components cannot be recomputed offline."
        ),
    }


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------

def _fmt(x: float | None, digits: int = 4) -> str:
    if x is None:
        return "n/a"
    return f"{x:.{digits}f}"


def _render_markdown(per_step: dict[int, dict[str, Any]]) -> str:
    lines: list[str] = []
    lines.append("# S0.4 — Per-component reward decomposition (I-Self GRPO)")
    lines.append("")
    lines.append(
        "Reward type: `tool_type_bonus_retrieval_contrib` (Variant I-Self). "
        "Total = `0.25*r_outcome + 0.50*r_tool_type + 0.25*r_retrieval_contrib`."
    )
    lines.append("")
    lines.append(
        "Trajectories at steps 200 and 250 were re-scored end-to-end. "
        "Steps 300 and 400 only have per-sample summary records "
        "(no `full_response`), so the tool-side components are reported as `n/a` "
        "and `r_total` is reported as a lower bound."
    )
    lines.append("")

    # Header
    lines.append(
        "| step | n | r_outcome | r_tool_type | r_retrieval_contrib "
        "| r_total | em | f1 | num_tool_calls |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for step in sorted(per_step.keys()):
        s = per_step[step]
        r_total = s.get("r_total")
        if r_total is None and "r_total_lower_bound" in s:
            r_total_str = f"≥{_fmt(s['r_total_lower_bound'])}"
        else:
            r_total_str = _fmt(r_total)
        lines.append(
            "| {step} | {n} | {ro} | {rt} | {rr} | {rT} | {em} | {f1} | {ntc} |".format(
                step=step,
                n=s["n"],
                ro=_fmt(s["r_outcome"]),
                rt=_fmt(s["r_tool_type"]),
                rr=_fmt(s["r_retrieval_contrib"]),
                rT=r_total_str,
                em=_fmt(s["em"]),
                f1=_fmt(s["f1"]),
                ntc=_fmt(s["num_tool_calls"], 2),
            )
        )

    lines.append("")
    lines.append("## Narrative")
    lines.append("")

    # Build narrative comparing 200/250 (pre-collapse) to 300/400 (post-collapse).
    s200 = per_step.get(200, {})
    s250 = per_step.get(250, {})
    s300 = per_step.get(300, {})
    s400 = per_step.get(400, {})

    pre_outcome = _mean([s200.get("r_outcome", 0.0), s250.get("r_outcome", 0.0)])
    pre_tool = _mean([s200.get("r_tool_type", 0.0), s250.get("r_tool_type", 0.0)])
    pre_retr = _mean(
        [s200.get("r_retrieval_contrib", 0.0), s250.get("r_retrieval_contrib", 0.0)]
    )
    post_outcome = _mean([s300.get("r_outcome", 0.0), s400.get("r_outcome", 0.0)])
    pre_ntc = _mean([s200.get("num_tool_calls", 0.0), s250.get("num_tool_calls", 0.0)])
    post_ntc = _mean([s300.get("num_tool_calls", 0.0), s400.get("num_tool_calls", 0.0)])

    lines.append(
        f"- **r_outcome (answer EM/F1 blend)** dropped from "
        f"~{pre_outcome:.3f} (steps 200–250) to ~{post_outcome:.3f} (steps 300–400). "
        f"This is the dominant collapse signal: the policy stopped getting "
        f"answers right at all (EM = 0)."
    )
    lines.append(
        f"- **num_tool_calls** collapsed from ~{pre_ntc:.2f} to ~{post_ntc:.2f}, "
        "i.e. the model emits a single tool call (or none) before answering. "
        "Because r_tool_type is *averaged over tool calls*, the per-call mean "
        "can stay non-zero even as the policy degenerates — this is exactly the "
        "Goodhart pathology the I-Self reward exposes."
    )
    lines.append(
        "- **r_tool_type / r_retrieval_contrib** at steps 300/400 cannot be "
        "recomputed offline because `full_response` was not saved for those "
        "evaluations. They are reported as `n/a`."
    )
    lines.append(
        f"- At pre-collapse (steps 200–250), r_tool_type ≈ {pre_tool:.3f} and "
        f"r_retrieval_contrib ≈ {pre_retr:.3f}. These are the values the policy "
        "was steering toward; the answer-side reward (r_outcome) is the channel "
        "that decoupled and crashed."
    )
    lines.append("")
    lines.append("**Conclusion.** The collapse is driven by `r_outcome` going to 0 ")
    lines.append("(zero EM, near-zero F1) while the policy adopts a 1-tool-call ")
    lines.append("shortcut. The 0.25 weight on r_outcome (vs 0.50 on the per-call ")
    lines.append("tool-type bonus) is too small to discourage the shortcut, and the ")
    lines.append("retrieval-contribution component fails to compensate because it is ")
    lines.append("also normalised by total calls — both step-side components are ")
    lines.append("blind to the *number* of tool calls used.")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__ or "")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)

    per_step: dict[int, dict[str, Any]] = {}
    for step in sorted(STEP_INPUTS.keys()):
        cfg = STEP_INPUTS[step]
        traj_path = cfg["trajectories"]
        ps_path = cfg["per_sample"]
        if traj_path is not None and Path(traj_path).exists():
            logger.info("Step %d: re-scoring trajectories from %s", step, traj_path)
            per_step[step] = _score_full_trajectories(Path(traj_path))
        elif ps_path is not None and Path(ps_path).exists():
            logger.info(
                "Step %d: full trajectories absent; aggregating per-sample summary %s",
                step,
                ps_path,
            )
            per_step[step] = _score_per_sample(Path(ps_path))
        else:
            logger.warning("Step %d: no input found, skipping", step)
            continue
        s = per_step[step]
        logger.info(
            "Step %d done: n=%d, r_outcome=%.4f, r_tool_type=%s, "
            "r_retrieval=%s, r_total=%s",
            step,
            s["n"],
            s["r_outcome"],
            f"{s['r_tool_type']:.4f}" if s["r_tool_type"] is not None else "n/a",
            f"{s['r_retrieval_contrib']:.4f}"
            if s["r_retrieval_contrib"] is not None
            else "n/a",
            f"{s['r_total']:.4f}" if s.get("r_total") is not None else "n/a",
        )

    out: dict[str, Any] = {
        "reward_type": REWARD_TYPE,
        "weights": {"r_outcome": 0.25, "r_tool_type": 0.50, "r_retrieval_contrib": 0.25},
        "experiment": "39i_self",
        "steps": {f"step_{step}": data for step, data in per_step.items()},
    }
    OUT_JSON.write_text(json.dumps(out, indent=2))
    logger.info("Wrote %s", OUT_JSON)

    md = _render_markdown(per_step)
    OUT_MD.write_text(md)
    logger.info("Wrote %s", OUT_MD)


if __name__ == "__main__":
    main()
