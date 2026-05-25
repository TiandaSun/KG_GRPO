"""Task 39: Classify trajectories + compute CvT per (variant, step) comparison table.

Reads all saved trajectories from results/trajectories/task39/{MODEL_TAG}/step_{STEP}/trajectories.json
and builds a comparison table across the 4 variants:
  - 39a_format    (E5b + format reward)
  - 39b_kl        (E5b + KL 5x)
  - 39c1_mixed    (E5b + SFT replay on 5K mixed trajectories)
  - 39c2_filtered (E5b + SFT replay on 1,293 correct-via-tool trajectories)

Also includes E5b-original from Task 37 as the baseline to beat.

Outputs:
  - results/task39_summary.json  (machine-readable)
  - results/task39_comparison.md (human-readable markdown table)
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import re
from collections import defaultdict
from pathlib import Path

from task16_classify import classify_trajectory  # noqa: E402

logger = logging.getLogger(__name__)

MODEL_TAG_RE = re.compile(r"^(39[a-f][12]?_\w+?)_step(\d+)$")


def wilson_ci(k: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    """Wilson score confidence interval for a binomial proportion."""
    if n == 0:
        return 0.0, 0.0
    z = 1.96 if confidence == 0.95 else 2.576
    p = k / n
    denom = 1 + z ** 2 / n
    centre = p + z ** 2 / (2 * n)
    margin = z * math.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2))
    lo = (centre - margin) / denom
    hi = (centre + margin) / denom
    return max(0.0, lo), min(1.0, hi)


def classify_one_model(traj_file: Path) -> dict:
    """Classify all trajectories in one file and return summary stats."""
    with open(traj_file) as f:
        trajs = json.load(f)
    counts: dict[str, int] = defaultdict(int)
    em_sum = 0.0
    tools_sum = 0.0
    for t in trajs:
        cat = classify_trajectory(t)
        counts[cat] += 1
        em_sum += t.get("em", 0)
        tools_sum += t.get("num_tool_calls", 0)
    n = len(trajs)
    cvt = counts.get("correct-via-tool", 0)
    cvt_rate = cvt / n if n else 0
    cvt_lo, cvt_hi = wilson_ci(cvt, n)
    return {
        "n": n,
        "em": em_sum / n if n else 0,
        "avg_tool_calls": tools_sum / n if n else 0,
        "counts": dict(counts),
        "cvt": cvt,
        "cvt_rate": cvt_rate,
        "cvt_ci95": [cvt_lo, cvt_hi],
    }


def load_task37_baseline() -> dict:
    """Load E5b-original numbers from Task 37 for comparison."""
    path = Path("results/task37_classification.json")
    if not path.exists():
        return {}
    data = json.load(open(path))
    return {
        "e5b_original_step100": {
            **data.get("e5b_100", {}),
        },
        "e3_500": data.get("e3_500", {}),
        "sft": data.get("sft", {}),
    }


def variant_display_name(variant: str) -> str:
    return {
        "39a_format": "39A format",
        "39b_kl": "39B KL 5x",
        "39c1_mixed": "39C1 SFT mixed",
        "39c2_filtered": "39C2 SFT filtered",
        "39d_catb_format": "39D CatB+format",
        "39e_enh_catb_format": "39E Enh+CatB+format",
        "39f_enh_full_format": "39F Enh+full+format",
    }.get(variant, variant)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    parser = argparse.ArgumentParser(description="Classify Task 39 trajectories + build comparison table")
    parser.add_argument("--traj_root", type=Path, default=Path("results/trajectories/task39"))
    parser.add_argument("--output_json", type=Path, default=Path("results/task39_summary.json"))
    parser.add_argument("--output_md", type=Path, default=Path("results/task39_comparison.md"))
    args = parser.parse_args()

    # Discover all (variant, step) combos that have trajectories saved
    results: dict[str, dict[int, dict]] = defaultdict(dict)  # variant -> step -> stats

    if not args.traj_root.exists():
        logger.warning("No trajectories directory at %s yet", args.traj_root)
        return

    for model_dir in sorted(args.traj_root.iterdir()):
        if not model_dir.is_dir():
            continue
        m = MODEL_TAG_RE.match(model_dir.name)
        if not m:
            continue
        variant = m.group(1)
        step = int(m.group(2))

        # Trajectory file is saved as step_{STEP}/trajectories.json where STEP is the LOCAL step
        # For 39a/39b: local step = global step
        # For 39c1/39c2: local step = 50 always, global step encoded in the tag
        local_step = 50 if variant.startswith("39c") else step
        traj_file = model_dir / f"step_{local_step}" / "trajectories.json"
        if not traj_file.exists():
            logger.warning("No trajectories at %s", traj_file)
            continue

        stats = classify_one_model(traj_file)
        stats["variant"] = variant
        stats["global_step"] = step
        results[variant][step] = stats
        logger.info("%s step=%d: n=%d EM=%.3f CvT=%d/%d=%.1f%% [%.1f-%.1f]",
                    variant, step, stats["n"], stats["em"],
                    stats["cvt"], stats["n"], 100 * stats["cvt_rate"],
                    100 * stats["cvt_ci95"][0], 100 * stats["cvt_ci95"][1])

    # Also load E5b-original baseline from Task 37
    baselines = load_task37_baseline()

    # Save JSON
    output = {
        "variants": {v: dict(sorted(d.items())) for v, d in results.items()},
        "baselines_task37": baselines,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(output, f, indent=2)

    # Build markdown comparison table
    all_steps = sorted({s for v in results.values() for s in v.keys()})
    variants_sorted = ["39a_format", "39b_kl", "39c1_mixed", "39c2_filtered",
                       "39d_catb_format", "39e_enh_catb_format", "39f_enh_full_format"]

    md = []
    md.append("# Task 39: E5b-Stabilized Variant Comparison\n")
    md.append("Each cell shows EM / CvT count / CvT% with 95% Wilson CI.")
    md.append("Baseline from Task 37 (test split, 200 questions): **E5b-original step 100: EM=31.0%, CvT=5.5% [3.1-9.6%]**\n")

    if all_steps:
        # CvT rate table
        md.append("## CvT rate (% correct-via-tool, 95% Wilson CI)\n")
        header = "| Variant | " + " | ".join(f"step {s}" for s in all_steps) + " |"
        sep = "|---|" + "|".join("---" for _ in all_steps) + "|"
        md.append(header)
        md.append(sep)
        for v in variants_sorted:
            if v not in results:
                md.append(f"| {variant_display_name(v)} | " + " | ".join("—" for _ in all_steps) + " |")
                continue
            row = [f"**{variant_display_name(v)}**"]
            for s in all_steps:
                if s in results[v]:
                    r = results[v][s]
                    pct = 100 * r["cvt_rate"]
                    lo = 100 * r["cvt_ci95"][0]
                    hi = 100 * r["cvt_ci95"][1]
                    row.append(f"{pct:.1f}% [{lo:.1f}-{hi:.1f}]")
                else:
                    row.append("—")
            md.append("| " + " | ".join(row) + " |")

        # EM + tools table
        md.append("\n## EM and avg tool calls\n")
        md.append(header.replace("Variant", "Variant (EM / tools)"))
        md.append(sep)
        for v in variants_sorted:
            if v not in results:
                md.append(f"| {variant_display_name(v)} | " + " | ".join("—" for _ in all_steps) + " |")
                continue
            row = [f"**{variant_display_name(v)}**"]
            for s in all_steps:
                if s in results[v]:
                    r = results[v][s]
                    row.append(f"{r['em']:.3f} / {r['avg_tool_calls']:.1f}")
                else:
                    row.append("—")
            md.append("| " + " | ".join(row) + " |")

        # Full category breakdown at latest step per variant
        md.append("\n## Latest step category breakdown\n")
        cats = ["correct-via-tool", "correct-no-tool", "correct-via-memory",
                "wrong-no-tool", "kg-incomplete", "tool-misuse", "wrong-answer"]
        md.append("| Variant | step | " + " | ".join(c for c in cats) + " |")
        md.append("|---|---|" + "|".join("---" for _ in cats) + "|")
        for v in variants_sorted:
            if v not in results:
                continue
            last_step = max(results[v].keys())
            r = results[v][last_step]
            row = [variant_display_name(v), str(last_step)]
            for c in cats:
                row.append(str(r["counts"].get(c, 0)))
            md.append("| " + " | ".join(row) + " |")

    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_md, "w") as f:
        f.write("\n".join(md) + "\n")

    logger.info("Saved summary to %s and %s", args.output_json, args.output_md)

    # Print to stdout
    print()
    print("\n".join(md))


if __name__ == "__main__":
    main()
