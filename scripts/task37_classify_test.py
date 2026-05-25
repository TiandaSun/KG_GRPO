"""Task 37: Classify trajectories on test split + compute CvT with 95% binomial CIs.

Reads trajectories from results/trajectories/task37/{model}/step_{step}/trajectories.json
Uses classify_trajectory from task16_classify.py
Reports per-model 7-category distribution with 95% Wilson CIs.
"""
from __future__ import annotations

import json
import logging
import math
from collections import defaultdict
from pathlib import Path

from task16_classify import classify_trajectory, process_experiment

logger = logging.getLogger(__name__)


def wilson_ci(k: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    """Wilson score confidence interval for a binomial proportion."""
    if n == 0:
        return 0.0, 0.0
    z = 1.96 if confidence == 0.95 else 2.576
    p = k / n
    denom = 1 + z**2 / n
    centre = p + z**2 / (2 * n)
    margin = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))
    lo = (centre - margin) / denom
    hi = (centre + margin) / denom
    return max(0.0, lo), min(1.0, hi)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    traj_files = {
        "sft": "results/trajectories/task37/sft/step_0/trajectories.json",
        "e2_1200": "results/trajectories/task37/e2_1200/step_1200/trajectories.json",
        "e3_500": "results/trajectories/task37/e3_500/step_500/trajectories.json",
        "e5b_100": "results/trajectories/task37/e5b_100/step_100/trajectories.json",
    }

    all_results = {}

    for model, path in traj_files.items():
        p = Path(path)
        if not p.exists():
            logger.warning("Missing: %s", path)
            continue

        with open(p) as f:
            trajs = json.load(f)

        counts: dict[str, int] = defaultdict(int)
        per_sample_cat = []
        for t in trajs:
            cat = classify_trajectory(t)
            counts[cat] += 1
            per_sample_cat.append({
                "sample_id": t.get("sample_id", ""),
                "category": cat,
                "em": t.get("em", 0),
                "num_tool_calls": t.get("num_tool_calls", 0),
            })

        n = len(trajs)
        em_mean = sum(t.get("em", 0) for t in trajs) / max(n, 1)
        tools_mean = sum(t.get("num_tool_calls", 0) for t in trajs) / max(n, 1)

        # CvT rate with 95% CI
        cvt = counts.get("correct-via-tool", 0)
        cvt_rate = cvt / n
        cvt_lo, cvt_hi = wilson_ci(cvt, n)

        # Correct-no-tool rate
        cnt = counts.get("correct-no-tool", 0)
        cnt_rate = cnt / n
        cnt_lo, cnt_hi = wilson_ci(cnt, n)

        all_results[model] = {
            "n": n,
            "em": em_mean,
            "avg_tool_calls": tools_mean,
            "counts": dict(counts),
            "cvt_rate": cvt_rate,
            "cvt_ci95": [cvt_lo, cvt_hi],
            "cnt_rate": cnt_rate,
            "cnt_ci95": [cnt_lo, cnt_hi],
            "per_sample": per_sample_cat,
        }

        logger.info("%s: n=%d, EM=%.3f, CvT=%d/%d=%.3f [%.3f, %.3f]",
                    model, n, em_mean, cvt, n, cvt_rate, cvt_lo, cvt_hi)

    # Print table
    print("\n" + "=" * 100)
    print(f"{'Model':<12} {'n':>4} {'EM':>7} {'Tools':>6} "
          f"{'CvT':>8} {'CvT 95% CI':>18} "
          f"{'CnT':>8} {'CnT 95% CI':>18}")
    print("-" * 100)
    for model in ("sft", "e2_1200", "e3_500", "e5b_100"):
        if model not in all_results:
            continue
        r = all_results[model]
        cvt_ci = f"[{r['cvt_ci95'][0]:.3f},{r['cvt_ci95'][1]:.3f}]"
        cnt_ci = f"[{r['cnt_ci95'][0]:.3f},{r['cnt_ci95'][1]:.3f}]"
        print(f"{model:<12} {r['n']:>4d} {r['em']:>7.3f} {r['avg_tool_calls']:>6.1f} "
              f"{r['cvt_rate']:>7.1%} {cvt_ci:>18} "
              f"{r['cnt_rate']:>7.1%} {cnt_ci:>18}")

    print("\n7-category distribution:")
    cats = ["correct-via-tool", "correct-no-tool", "correct-via-memory",
            "wrong-no-tool", "kg-incomplete", "tool-misuse", "wrong-answer"]
    print(f"{'Model':<12} " + " ".join(f"{c[:12]:>13}" for c in cats))
    for model in ("sft", "e2_1200", "e3_500", "e5b_100"):
        if model not in all_results:
            continue
        r = all_results[model]
        print(f"{model:<12} " + " ".join(f"{r['counts'].get(c, 0):>13d}" for c in cats))

    # Save results
    out = Path("results/task37_classification.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
