"""Bootstrap confidence intervals + McNemar's test for Task 14.

Reads per-sample correctness JSONs (saved by eval_with_tools.py with
--save_per_sample) and computes:
  1. Per-model bootstrap 95% CI for EM, F1
  2. Pairwise paired bootstrap diffs and CIs
  3. Pairwise McNemar's test p-values

Per-sample JSON format:
    [{"sample_id": "...", "em": 0.0, "f1": 0.0, "contains_em": 0.0, "num_tool_calls": 0, "hops": 2}, ...]

Usage:
    python scripts/compute_bootstrap_ci.py \
        --per_sample_root results/task14_per_sample \
        --output results/task14_summary.json
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

logger = logging.getLogger(__name__)


def load_per_sample(per_sample_dir: Path) -> dict[str, dict[str, float]]:
    """Load per-sample results, merging chunked outputs.

    Returns: {sample_id: {em, f1, contains_em, num_tool_calls, hops}}
    """
    merged: dict[str, dict[str, float]] = {}
    files = sorted(per_sample_dir.glob("step_*_*.json"))
    if not files:
        files = sorted(per_sample_dir.glob("*.json"))
    for f in files:
        with open(f) as fh:
            data = json.load(fh)
        for entry in data:
            sid = entry["sample_id"]
            merged[sid] = entry
    return merged


def bootstrap_ci(
    arr: np.ndarray,
    n_boot: int = 10000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Compute bootstrap mean and CI.

    Returns:
        (mean, lo, hi) where lo/hi are the (1-ci)/2 and (1+ci)/2 percentiles
    """
    rng = np.random.default_rng(seed)
    n = len(arr)
    if n == 0:
        return 0.0, 0.0, 0.0
    boot_means = np.empty(n_boot)
    for i in range(n_boot):
        sample = rng.choice(arr, size=n, replace=True)
        boot_means[i] = sample.mean()
    alpha = (1 - ci) / 2
    lo = float(np.quantile(boot_means, alpha))
    hi = float(np.quantile(boot_means, 1 - alpha))
    return float(arr.mean()), lo, hi


def paired_bootstrap_diff(
    a: np.ndarray,
    b: np.ndarray,
    n_boot: int = 10000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float, float]:
    """Paired bootstrap difference (a - b) with CI and p-value.

    Both arrays must be aligned (same indices = same sample_ids).
    Returns:
        (mean_diff, lo, hi, p_value_two_sided)
    """
    if len(a) != len(b):
        raise ValueError(f"Arrays must be same length: {len(a)} vs {len(b)}")
    if len(a) == 0:
        return 0.0, 0.0, 0.0, 1.0
    rng = np.random.default_rng(seed)
    n = len(a)
    diffs = a - b
    boot_diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot_diffs[i] = diffs[idx].mean()
    mean_diff = float(diffs.mean())
    alpha = (1 - ci) / 2
    lo = float(np.quantile(boot_diffs, alpha))
    hi = float(np.quantile(boot_diffs, 1 - alpha))
    # Two-sided p-value: probability that the absolute boot diff exceeds observed
    # under the null (mean_diff = 0). Use the percentile-based method:
    # p = 2 * min(P(boot_diff <= 0), P(boot_diff >= 0))
    p_below = float((boot_diffs <= 0).mean())
    p_above = float((boot_diffs >= 0).mean())
    p_value = 2 * min(p_below, p_above)
    return mean_diff, lo, hi, p_value


def mcnemar_test(a: np.ndarray, b: np.ndarray) -> dict:
    """McNemar's test for paired binary outcomes (e.g., per-sample EM scores).

    Returns:
        {b_count, c_count, p_value, statistic}
    Where:
        b = a correct, b wrong
        c = a wrong, b correct
    """
    if len(a) != len(b):
        raise ValueError("Arrays must be same length")
    a_bin = (a > 0.5).astype(int)
    b_bin = (b > 0.5).astype(int)
    b_count = int(((a_bin == 1) & (b_bin == 0)).sum())
    c_count = int(((a_bin == 0) & (b_bin == 1)).sum())

    if b_count + c_count == 0:
        return {"b": 0, "c": 0, "p_value": 1.0, "statistic": 0.0, "method": "exact_zero"}

    # Use exact binomial test (recommended when b+c is small)
    # H0: P(disagreement direction) = 0.5
    n = b_count + c_count
    successes = min(b_count, c_count)
    result = binomtest(successes, n, p=0.5, alternative="two-sided")
    return {
        "b": b_count,
        "c": c_count,
        "p_value": float(result.pvalue),
        "statistic": float(b_count - c_count),
        "method": "exact_binomial",
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    parser = argparse.ArgumentParser(description="Bootstrap CIs + McNemar for Task 14")
    parser.add_argument("--per_sample_root", type=Path, default=Path("results/task14_per_sample"))
    parser.add_argument("--output", type=Path, default=Path("results/task14_summary.json"))
    parser.add_argument("--n_boot", type=int, default=10000)
    args = parser.parse_args()

    # Discover model dirs
    model_dirs = sorted([d for d in args.per_sample_root.iterdir() if d.is_dir()])
    if not model_dirs:
        logger.error("No model directories found in %s", args.per_sample_root)
        return

    logger.info("Found %d model directories", len(model_dirs))

    per_model: dict[str, dict[str, dict]] = {}
    for d in model_dirs:
        model_key = d.name
        merged = load_per_sample(d)
        if not merged:
            logger.warning("No data in %s", d)
            continue
        per_model[model_key] = merged
        logger.info("  %s: %d samples", model_key, len(merged))

    # Per-model bootstrap CIs
    summary = {}
    for model_key, samples in per_model.items():
        em_arr = np.array([s["em"] for s in samples.values()])
        f1_arr = np.array([s["f1"] for s in samples.values()])
        cm_arr = np.array([s.get("contains_em", 0) for s in samples.values()])
        tool_arr = np.array([s.get("num_tool_calls", 0) for s in samples.values()])

        em_mean, em_lo, em_hi = bootstrap_ci(em_arr, n_boot=args.n_boot)
        f1_mean, f1_lo, f1_hi = bootstrap_ci(f1_arr, n_boot=args.n_boot)
        cm_mean, cm_lo, cm_hi = bootstrap_ci(cm_arr, n_boot=args.n_boot)

        summary[model_key] = {
            "n_samples": len(samples),
            "em": {"mean": em_mean, "lo95": em_lo, "hi95": em_hi},
            "f1": {"mean": f1_mean, "lo95": f1_lo, "hi95": f1_hi},
            "contains_em": {"mean": cm_mean, "lo95": cm_lo, "hi95": cm_hi},
            "avg_tool_calls": float(tool_arr.mean()),
        }

    # Pairwise comparisons (paired bootstrap + McNemar)
    pairwise = {}
    keys = sorted(per_model.keys())
    for i, key_a in enumerate(keys):
        for key_b in keys[i + 1:]:
            samples_a = per_model[key_a]
            samples_b = per_model[key_b]
            common_ids = sorted(set(samples_a) & set(samples_b))
            if not common_ids:
                continue
            a_em = np.array([samples_a[sid]["em"] for sid in common_ids])
            b_em = np.array([samples_b[sid]["em"] for sid in common_ids])
            a_f1 = np.array([samples_a[sid]["f1"] for sid in common_ids])
            b_f1 = np.array([samples_b[sid]["f1"] for sid in common_ids])

            em_diff, em_dlo, em_dhi, em_p = paired_bootstrap_diff(a_em, b_em, n_boot=args.n_boot)
            f1_diff, f1_dlo, f1_dhi, f1_p = paired_bootstrap_diff(a_f1, b_f1, n_boot=args.n_boot)
            mc = mcnemar_test(a_em, b_em)

            pairwise[f"{key_a}_vs_{key_b}"] = {
                "n_common": len(common_ids),
                "em_diff": {"mean": em_diff, "lo95": em_dlo, "hi95": em_dhi, "boot_p": em_p},
                "f1_diff": {"mean": f1_diff, "lo95": f1_dlo, "hi95": f1_dhi, "boot_p": f1_p},
                "mcnemar_em": mc,
            }

    output = {
        "n_boot": args.n_boot,
        "models": summary,
        "pairwise": pairwise,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    logger.info("Saved summary to %s", args.output)

    # Pretty print
    print("\n=== Per-Model Results (95% bootstrap CI) ===")
    print(f"{'Model':<25} {'EM':>8} {'EM CI':>20} {'F1':>8} {'F1 CI':>20} {'Tools':>8} {'n':>6}")
    print("-" * 100)
    for k in sorted(summary.keys()):
        s = summary[k]
        em_str = f"[{s['em']['lo95']:.3f}, {s['em']['hi95']:.3f}]"
        f1_str = f"[{s['f1']['lo95']:.3f}, {s['f1']['hi95']:.3f}]"
        print(f"{k:<25} {s['em']['mean']:>8.3f} {em_str:>20} {s['f1']['mean']:>8.3f} {f1_str:>20} {s['avg_tool_calls']:>8.2f} {s['n_samples']:>6}")

    if pairwise:
        print("\n=== Pairwise EM Differences (paired bootstrap + McNemar) ===")
        print(f"{'Comparison':<55} {'EM diff':>10} {'EM CI':>22} {'McNemar p':>12}")
        print("-" * 100)
        for k in sorted(pairwise.keys()):
            p = pairwise[k]
            ci_str = f"[{p['em_diff']['lo95']:+.3f}, {p['em_diff']['hi95']:+.3f}]"
            print(f"{k:<55} {p['em_diff']['mean']:>+10.3f} {ci_str:>22} {p['mcnemar_em']['p_value']:>12.4f}")


if __name__ == "__main__":
    main()
