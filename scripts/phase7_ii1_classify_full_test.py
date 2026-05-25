"""Phase 7 Action II1: full-test CvT audit.

Reads full-3531 trajectories for E5b@100, E3@500, 39B@400 (and optionally 300/500)
and runs the 7-category classification from task16_classify on ALL samples, not
just a sub-sample. Computes Wilson 95% CI on CvT rate and paired McNemar's test
between 39B and the E3/E5b baselines.

Outputs:
  results/phase7/full_test_cvt_audit.json
  results/phase7/full_test_cvt_audit.md

The Day-2 readiness gate (Gate A in hpc_tasks.md v11) reads this file to decide
whether to launch Variants G/I or pivot to pure-diagnostic framing.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from task16_classify import classify_trajectory  # noqa: E402

logger = logging.getLogger(__name__)

# Model → trajectory directory (populated from results/trajectories/phase7/)
DEFAULT_TARGETS: dict[str, tuple[Path, int]] = {
    "e5b_step100": (
        Path("results/trajectories/phase7/e5b_step100_full"),
        100,
    ),
    "e3_step500": (
        Path("results/trajectories/phase7/e3_step500_full"),
        500,
    ),
    "39b_step300": (
        Path("results/trajectories/phase7/39b_step300_full"),
        300,
    ),
    "39b_step400": (
        Path("results/trajectories/phase7/39b_step400_full"),
        400,
    ),
    "39b_step500": (
        Path("results/trajectories/phase7/39b_step500_full"),
        500,
    ),
    "39i_self_step50": (
        Path("results/trajectories/phase7/39i_self_step50_full"),
        0,
    ),
    "39i_self_step100": (
        Path("results/trajectories/phase7/39i_self_step100_full"),
        0,
    ),
    "39i_self_step150": (
        Path("results/trajectories/phase7/39i_self_step150_full"),
        0,
    ),
    "39i_self_step200": (
        Path("results/trajectories/phase7/39i_self_step200_full"),
        0,  # merged checkpoint uses step=0 convention
    ),
    "39i_self_step250": (
        Path("results/trajectories/phase7/39i_self_step250_full"),
        0,
    ),
    "39g1_step500": (
        Path("results/trajectories/phase7/39g1_step500_full"),
        0,
    ),
    "39g2_step500": (
        Path("results/trajectories/phase7/39g2_step500_full"),
        0,
    ),
}


def wilson_ci(k: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    """Wilson score CI for a binomial proportion."""
    if n == 0:
        return 0.0, 0.0
    z = 1.96 if confidence == 0.95 else 2.576
    p = k / n
    denom = 1 + z ** 2 / n
    centre = p + z ** 2 / (2 * n)
    margin = z * math.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2))
    return max(0.0, (centre - margin) / denom), min(1.0, (centre + margin) / denom)


def mcnemar_p(b: int, c: int) -> float:
    """Two-sided McNemar's exact p-value.

    b = count where A correct, B wrong.
    c = count where A wrong, B correct.
    Uses continuity-corrected chi-square if b+c >= 25, else exact binomial.
    """
    n = b + c
    if n == 0:
        return 1.0
    if n < 25:
        # exact binomial two-sided
        from math import comb
        tail = sum(comb(n, k) for k in range(min(b, c) + 1))
        p_one = tail / (2 ** n)
        return min(1.0, 2 * p_one)
    # continuity-corrected chi-square
    chi2 = (abs(b - c) - 1) ** 2 / n
    # survival function of chi-square with 1 dof via erf
    # sf = erfc(sqrt(chi2/2))
    from math import erfc, sqrt
    return erfc(sqrt(chi2 / 2))


def load_trajectories(traj_dir: Path, step: int) -> list[dict] | None:
    """Trajectories are saved under traj_dir/step_{step}/trajectories.json."""
    path = traj_dir / f"step_{step}" / "trajectories.json"
    if not path.exists():
        logger.warning("Missing trajectories at %s", path)
        return None
    with open(path) as f:
        return json.load(f)


def classify_all(trajs: list[dict]) -> dict:
    """Classify every trajectory, return per-sample classification map + counts."""
    counts: dict[str, int] = defaultdict(int)
    per_sample: dict[str, str] = {}
    em_sum = 0.0
    tools_sum = 0.0
    for t in trajs:
        sid = str(t.get("sample_id", id(t)))
        cat = classify_trajectory(t)
        counts[cat] += 1
        per_sample[sid] = cat
        em_sum += float(t.get("em", 0))
        tools_sum += float(t.get("num_tool_calls", 0))
    n = len(trajs)
    cvt = counts.get("correct-via-tool", 0)
    correct_any = counts.get("correct-via-tool", 0) + counts.get("correct-no-tool", 0) + counts.get("correct-via-memory", 0)
    cvt_lo, cvt_hi = wilson_ci(cvt, n)
    return {
        "n": n,
        "em_mean": em_sum / n if n else 0.0,
        "avg_tool_calls": tools_sum / n if n else 0.0,
        "counts": dict(counts),
        "cvt": cvt,
        "cvt_rate": cvt / n if n else 0.0,
        "cvt_ci95": [cvt_lo, cvt_hi],
        "correct_any": correct_any,
        "em_from_counts": correct_any / n if n else 0.0,
        "per_sample": per_sample,
    }


def paired_mcnemar(
    stats_a: dict, stats_b: dict, success_label: str = "correct-via-tool"
) -> dict:
    """Compute McNemar across the two classifications for the given success label.

    The paired test uses the intersection of sample IDs present in both.
    """
    sa = stats_a["per_sample"]
    sb = stats_b["per_sample"]
    common_ids = set(sa.keys()) & set(sb.keys())
    only_a = only_b = both = neither = 0
    for sid in common_ids:
        a_ok = sa[sid] == success_label
        b_ok = sb[sid] == success_label
        if a_ok and b_ok:
            both += 1
        elif a_ok and not b_ok:
            only_a += 1
        elif not a_ok and b_ok:
            only_b += 1
        else:
            neither += 1
    p = mcnemar_p(only_a, only_b)
    return {
        "n_common": len(common_ids),
        "both": both,
        "only_a": only_a,
        "only_b": only_b,
        "neither": neither,
        "mcnemar_p_two_sided": p,
        "success_label": success_label,
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description="Phase 7 II1: full-test CvT audit")
    parser.add_argument(
        "--output_json", type=Path,
        default=Path("results/phase7/full_test_cvt_audit.json"),
    )
    parser.add_argument(
        "--output_md", type=Path,
        default=Path("results/phase7/full_test_cvt_audit.md"),
    )
    parser.add_argument(
        "--success_label", type=str, default="correct-via-tool",
        help="Category label to measure as 'success' for McNemar.",
    )
    args = parser.parse_args()

    summaries: dict[str, dict] = {}
    for tag, (traj_dir, step) in DEFAULT_TARGETS.items():
        trajs = load_trajectories(traj_dir, step)
        if trajs is None:
            continue
        s = classify_all(trajs)
        summaries[tag] = s
        logger.info(
            "%s: n=%d EM=%.3f CvT=%d/%d=%.1f%% [%.1f-%.1f] tools=%.2f",
            tag, s["n"], s["em_mean"],
            s["cvt"], s["n"], 100 * s["cvt_rate"],
            100 * s["cvt_ci95"][0], 100 * s["cvt_ci95"][1],
            s["avg_tool_calls"],
        )

    if "39b_step400" in summaries:
        pairwise: dict[str, dict] = {}
        for other in ("e5b_step100", "e3_step500", "39b_step300", "39b_step500"):
            if other in summaries:
                pairwise[f"39b_step400_vs_{other}"] = paired_mcnemar(
                    summaries["39b_step400"], summaries[other], args.success_label
                )
    else:
        pairwise = {}

    report = {
        "summaries": {k: {kk: vv for kk, vv in v.items() if kk != "per_sample"} for k, v in summaries.items()},
        "pairwise_mcnemar": pairwise,
        "success_label": args.success_label,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(report, f, indent=2)

    # markdown
    lines: list[str] = []
    lines.append("# Phase 7 Action II1: Full-Test CvT Audit\n")
    lines.append("Full 3,531-question test set evaluation. CvT = fraction of trajectories classified as 'correct-via-tool' by `task16_classify`.\n")
    lines.append("## Per-model stats\n")
    lines.append("| model | n | EM | CvT count | CvT % | 95% Wilson CI | tools/Q |")
    lines.append("|---|---|---|---|---|---|---|")
    for tag, s in summaries.items():
        lines.append(
            f"| {tag} | {s['n']} | {s['em_mean']:.3f} | {s['cvt']} | "
            f"{100*s['cvt_rate']:.2f}% | "
            f"[{100*s['cvt_ci95'][0]:.2f}-{100*s['cvt_ci95'][1]:.2f}%] | "
            f"{s['avg_tool_calls']:.2f} |"
        )
    lines.append("\n## Category breakdown\n")
    cats = [
        "correct-via-tool", "correct-no-tool", "correct-via-memory",
        "wrong-no-tool", "kg-incomplete", "tool-misuse", "wrong-answer",
    ]
    lines.append("| model | " + " | ".join(cats) + " |")
    lines.append("|---|" + "|".join("---" for _ in cats) + "|")
    for tag, s in summaries.items():
        row = [tag] + [str(s["counts"].get(c, 0)) for c in cats]
        lines.append("| " + " | ".join(row) + " |")
    if pairwise:
        lines.append("\n## Paired McNemar on `correct-via-tool`\n")
        lines.append("| comparison (A vs B) | n | A-only | B-only | both | neither | p (two-sided) |")
        lines.append("|---|---|---|---|---|---|---|")
        for k, v in pairwise.items():
            lines.append(
                f"| {k} | {v['n_common']} | {v['only_a']} | {v['only_b']} | "
                f"{v['both']} | {v['neither']} | {v['mcnemar_p_two_sided']:.4f} |"
            )

    lines.append("\n## Gate A decision\n")
    if "39b_step400" in summaries and "e5b_step100" in summaries:
        s_39b = summaries["39b_step400"]
        s_e5b = summaries["e5b_step100"]
        em_pass = s_39b["em_mean"] >= 0.34
        cvt_delta = (s_39b["cvt"] - s_e5b["cvt"]) / s_e5b["n"]  # approx pp delta
        cvt_pass = (s_39b["cvt"] / s_39b["n"]) >= (s_e5b["cvt"] / s_e5b["n"] - 0.01)
        lines.append(f"- 39B@400 full-test EM: **{100*s_39b['em_mean']:.2f}%** ({'≥ 34%' if em_pass else '< 34%'})")
        lines.append(f"- 39B@400 full-test CvT: **{100*s_39b['cvt_rate']:.2f}%** vs E5b@100 {100*s_e5b['cvt_rate']:.2f}%")
        lines.append("")
        if em_pass and cvt_pass:
            lines.append("**Gate A: PASS** → proceed with Day-2 variant training (G/I).")
        else:
            lines.append("**Gate A: FAIL** → pivot to pure-diagnostic paper framing; cancel Day-2 variant training.")
    else:
        lines.append("*Not yet decidable — 39B@400 or E5b@100 full-test trajectories not yet available.*")

    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_md, "w") as f:
        f.write("\n".join(lines) + "\n")

    logger.info("Saved %s and %s", args.output_json, args.output_md)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
