"""v17 W17.5 — Apply task16 7-cat classifier to W17.4 GRPO trajectories.

Outputs per-checkpoint classified.json files at:
  _handoff/data/v17_llama/w17_5_step{N}_classified.json
  _handoff/data/v17_llama/w17_5_summary.json

Identifies the best-EM checkpoint from W17.4 results and writes a separate
`best_classified.json` alias for the writing agent.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, "/lus/lfs1aip2/scratch/u6gg/ts1201.u6gg/Project/KG_GRPO/scripts")
from task16_classify import (  # noqa: E402
    CATEGORIES, classify_trajectory, process_experiment, select_examples,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("v17_classify")

OUT_DIR = Path("/lus/lfs1aip2/scratch/u6gg/ts1201.u6gg/Project/KG_GRPO/_handoff/data/v17_llama")


def discover_steps() -> list[int]:
    """Find all W17.4 step results that have both eval + trajectory files."""
    steps = []
    for ev in OUT_DIR.glob("w17_4_step*_full_test.json"):
        # filename: w17_4_step{N}_full_test.json
        stem = ev.stem  # w17_4_step{N}_full_test
        try:
            step = int(stem.replace("w17_4_step", "").replace("_full_test", ""))
        except ValueError:
            continue
        traj = OUT_DIR / f"w17_4_step{step}_full_trajectories" / "step_0" / "trajectories.json"
        if traj.exists():
            steps.append(step)
    return sorted(steps)


def load_em(step: int) -> float | None:
    f = OUT_DIR / f"w17_4_step{step}_full_test.json"
    if not f.exists():
        return None
    with open(f) as fp:
        return float(json.load(fp)["0"]["em"])


def classify_step(step: int) -> tuple[Path, dict]:
    traj_path = OUT_DIR / f"w17_4_step{step}_full_trajectories" / "step_0" / "trajectories.json"
    with open(traj_path) as f:
        trajectories = json.load(f)
    logger.info("step %d: classifying %d trajectories", step, len(trajectories))
    result = process_experiment(f"v17_step{step}", trajectories)
    out_path = OUT_DIR / f"w17_5_step{step}_classified.json"
    payload = {
        "step": step,
        "total": result["total"],
        "counts": result["counts"],
        "percentages": result["percentages"],
        "examples": select_examples(result["classified"], n_per_category=3),
        "classified": result["classified"],
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    return out_path, payload


def main() -> int:
    # Always include the SFT-only baseline (W17.2) and any W17.4 steps
    steps = discover_steps()
    logger.info("Discovered W17.4 steps: %s", steps)

    summary: dict = {"per_step": {}, "best_em_step": None, "best_em": -1.0}

    # Also classify the W17.2 SFT-only baseline
    sft_traj = OUT_DIR / "w17_2_baseline_full_trajectories" / "step_0" / "trajectories.json"
    if sft_traj.exists():
        with open(sft_traj) as f:
            sft = json.load(f)
        logger.info("W17.2 SFT-only baseline: classifying %d trajectories", len(sft))
        sft_result = process_experiment("v17_sft_only", sft)
        with open(OUT_DIR / "w17_5_sft_only_classified.json", "w") as f:
            json.dump({
                "step": 0,
                "total": sft_result["total"],
                "counts": sft_result["counts"],
                "percentages": sft_result["percentages"],
                "examples": select_examples(sft_result["classified"], n_per_category=3),
                "classified": sft_result["classified"],
            }, f, indent=2)
        summary["per_step"][0] = {
            "label": "sft_only",
            "counts": sft_result["counts"],
            "percentages": sft_result["percentages"],
        }
    else:
        logger.warning("SFT-only trajectories not found at %s", sft_traj)

    for step in steps:
        em = load_em(step)
        if em is None:
            continue
        path, payload = classify_step(step)
        logger.info("step %d wrote %s | percentages: %s", step, path.name, payload["percentages"])
        summary["per_step"][step] = {
            "em": em,
            "counts": payload["counts"],
            "percentages": payload["percentages"],
        }
        if em > summary["best_em"]:
            summary["best_em"] = em
            summary["best_em_step"] = step

    # Write best-EM alias if applicable
    best_step = summary["best_em_step"]
    if best_step is not None:
        src = OUT_DIR / f"w17_5_step{best_step}_classified.json"
        alias = OUT_DIR / "w17_5_best_classified.json"
        if src.exists():
            with open(src) as f:
                data = json.load(f)
            with open(alias, "w") as f:
                json.dump(data, f, indent=2)
            logger.info("Wrote best-EM alias: step=%d EM=%.4f -> %s",
                        best_step, summary["best_em"], alias.name)

    with open(OUT_DIR / "w17_5_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Markdown summary table
    md = ["# W17.5 — Trajectory classification\n"]
    md.append("| Step | EM | correct-no-tool | wrong-no-tool | correct-via-tool | correct-via-memory | kg-incomplete | tool-misuse | wrong-answer |")
    md.append("|---|---|---|---|---|---|---|---|---|")
    for k in sorted(summary["per_step"].keys()):
        s = summary["per_step"][k]
        em = s.get("em")
        if em is None and k == 0:
            em_str = "SFT-only"
        else:
            em_str = f"{em:.4f}"
        pct = s["percentages"]
        md.append(
            f"| {k} | {em_str} | {pct['correct-no-tool']:.1f}% | {pct['wrong-no-tool']:.1f}% | "
            f"{pct['correct-via-tool']:.1f}% | {pct['correct-via-memory']:.1f}% | "
            f"{pct['kg-incomplete']:.1f}% | {pct['tool-misuse']:.1f}% | {pct['wrong-answer']:.1f}% |"
        )
    if best_step is not None:
        md.append(f"\n**Best-EM checkpoint**: step {best_step} (EM={summary['best_em']:.4f})\n")
    (OUT_DIR / "W17_5_CLASSIFICATION.md").write_text("\n".join(md) + "\n")
    logger.info("Wrote W17_5_CLASSIFICATION.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
