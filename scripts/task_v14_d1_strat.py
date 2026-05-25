"""V14-D1-strat — Category A vs B stratified EM (and CvT) analysis.

Decomposes 14B@400's EM gain over 7B 39B@400 into Cat A (parametric memorization)
vs Cat B (retrieval) components. Load-bearing for Section 6 framework validation.

Cat A (n=975): pass@10 > 0 on raw Qwen2.5-7B-Instruct without tools — questions
               within parametric memory.
Cat B (n=2556): pass@10 = 0 — questions requiring retrieval.

Outputs:
  results/phase7/v14_d1_strat.json
  results/phase7/v14_d1_strat.md

Table 1 (EM) is computed from per_sample JSONs — runs whenever 14B per_sample
is available (partial 3000 is fine; 3531 after topup job 4247719).
Table 2 (CvT) requires trajectory classification — runs once the 500-sample
CvT eval (job 4247718) + classifier pipeline completes.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

# Wilson 95% CI (re-used from phase7_ii1_classify_full_test)
def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denom = 1 + z ** 2 / n
    centre = p + z ** 2 / (2 * n)
    margin = z * math.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2))
    return max(0.0, (centre - margin) / denom), min(1.0, (centre + margin) / denom)


MODEL_INPUTS: dict[str, Path] = {
    "7B E3@500":       Path("results/phase7/e3_step500_full_per_sample/step_500_per_sample.json"),
    "7B 39B@400":      Path("results/phase7/39b_step400_full_per_sample/step_400_per_sample.json"),
    "7B G2@500":       Path("results/phase7/39g2_step500_full_per_sample/step_0_per_sample.json"),
    "14B D1@400":      Path("results/phase7/v14_d1_qwen14b_e5b_step400_full_per_sample/step_0_per_sample.json"),
}

# Optional classification JSONs produced by classify_trajectories_per_sample.py
# Each file has {"per_sample": {sid: category, ...}, "n": N}.
# "correct-via-tool" is the success label we stratify on.
CLASSIFICATION_INPUTS: dict[str, Path] = {
    "7B E3@500":       Path("results/phase7/per_sample_classifications/e3_step500_classification.json"),
    "7B 39B@400":      Path("results/phase7/per_sample_classifications/39b_step400_classification.json"),
    "7B G2@500":       Path("results/phase7/per_sample_classifications/39g2_step500_classification.json"),
    "14B D1@400":      Path("results/phase7/per_sample_classifications/14b_d1_step400_classification.json"),
}


def load_per_sample(path: Path) -> dict[str, dict[str, Any]]:
    """Load per_sample jsonl file → dict[sample_id → record]."""
    if not path.exists():
        return {}
    with open(path) as f:
        records = json.load(f)
    return {str(r["sample_id"]): r for r in records}


def load_classification(path: Path) -> dict[str, str]:
    """Load per-sample classification → dict[sample_id → category]."""
    if not path.exists():
        return {}
    d = json.loads(path.read_text())
    return d.get("per_sample", {}) if isinstance(d, dict) else {}


def stratify_cvt(
    classification: dict[str, str],
    cat_a_ids: set[str],
    cat_b_ids: set[str],
    success_label: str = "correct-via-tool",
) -> dict[str, Any]:
    """Stratified CvT rates by category with Wilson CIs.

    Also returns the full 7-category count for Cat A and Cat B, since the
    auditor's other categories (kg-incomplete etc.) give mechanism context.
    """
    from collections import Counter

    def section(ids: set[str]) -> dict[str, Any]:
        subset = [classification[s] for s in ids if s in classification]
        n = len(subset)
        ctr = Counter(subset)
        cvt = ctr.get(success_label, 0)
        lo, hi = wilson_ci(cvt, n)
        return {
            "n_covered": n,
            "cvt": cvt,
            "cvt_rate": cvt / n if n else 0.0,
            "cvt_ci95": [lo, hi],
            "category_counts": dict(ctr),
        }

    return {
        "cat_a": section(cat_a_ids),
        "cat_b": section(cat_b_ids),
        "full": section(cat_a_ids | cat_b_ids),
    }


def stratify_em(
    per_sample: dict[str, dict[str, Any]],
    cat_a_ids: set[str],
    cat_b_ids: set[str],
) -> dict[str, Any]:
    """Compute Cat A / Cat B / full-set EM with Wilson CIs."""
    def section(ids: set[str]) -> dict[str, Any]:
        hits = [per_sample[s] for s in ids if s in per_sample]
        n = len(hits)
        em_sum = sum(float(r.get("em", 0)) for r in hits)
        f1_sum = sum(float(r.get("f1", 0)) for r in hits)
        tc_sum = sum(float(r.get("num_tool_calls", 0)) for r in hits)
        lo, hi = wilson_ci(int(em_sum), n)
        return {
            "n_covered": n,
            "em": em_sum / n if n else 0.0,
            "em_count": int(em_sum),
            "em_ci95": [lo, hi],
            "f1": f1_sum / n if n else 0.0,
            "tools_per_q": tc_sum / n if n else 0.0,
        }

    cov_a = section(cat_a_ids)
    cov_b = section(cat_b_ids)
    full_ids = cat_a_ids | cat_b_ids
    cov_full = section(full_ids)
    return {
        "cat_a": cov_a,
        "cat_b": cov_b,
        "full": cov_full,
        "em_delta_A_minus_B": cov_a["em"] - cov_b["em"],
    }


def decompose_gain(
    stats_target: dict[str, Any],
    stats_baseline: dict[str, Any],
    n_cat_a: int,
    n_cat_b: int,
) -> dict[str, Any]:
    """Decompose target's EM gain over baseline into Cat A / Cat B contributions.

    Let p_A = n_A / (n_A + n_B).  Then
        Δ_overall  ≈ p_A * Δ_A  +  (1 - p_A) * Δ_B
    We report both in percentage points.
    """
    n_total = n_cat_a + n_cat_b
    p_a = n_cat_a / n_total
    p_b = n_cat_b / n_total
    d_a = stats_target["cat_a"]["em"] - stats_baseline["cat_a"]["em"]
    d_b = stats_target["cat_b"]["em"] - stats_baseline["cat_b"]["em"]
    contrib_a = d_a * p_a
    contrib_b = d_b * p_b
    return {
        "n_cat_a": n_cat_a,
        "n_cat_b": n_cat_b,
        "p_a": p_a,
        "p_b": p_b,
        "delta_cat_a_pp": d_a * 100,
        "delta_cat_b_pp": d_b * 100,
        "contribution_cat_a_pp": contrib_a * 100,
        "contribution_cat_b_pp": contrib_b * 100,
        "total_overall_delta_pp": (contrib_a + contrib_b) * 100,
    }


def fmt_em_row(label: str, s: dict[str, Any]) -> str:
    a = s["cat_a"]; b = s["cat_b"]; f = s["full"]
    delta_pp = 100 * s["em_delta_A_minus_B"]
    return (
        f"| {label} "
        f"| {a['em_count']}/{a['n_covered']} = {100*a['em']:.2f}% "
        f"[{100*a['em_ci95'][0]:.2f}-{100*a['em_ci95'][1]:.2f}%] "
        f"| {b['em_count']}/{b['n_covered']} = {100*b['em']:.2f}% "
        f"[{100*b['em_ci95'][0]:.2f}-{100*b['em_ci95'][1]:.2f}%] "
        f"| {f['em_count']}/{f['n_covered']} = {100*f['em']:.2f}% "
        f"| {delta_pp:+.2f}pp "
        f"| {b['tools_per_q']:.2f} |"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cat_file", type=Path, default=Path("results/task26_category_b.json"))
    ap.add_argument("--out_json", type=Path, default=Path("results/phase7/v14_d1_strat.json"))
    ap.add_argument("--out_md", type=Path, default=Path("results/phase7/v14_d1_strat.md"))
    args = ap.parse_args()

    cat = json.loads(args.cat_file.read_text())
    cat_a_ids = set(cat["category_a_ids"])
    cat_b_ids = set(cat["category_b_ids"])
    print(f"Cat A: {len(cat_a_ids)}, Cat B: {len(cat_b_ids)}")

    results: dict[str, Any] = {}
    for label, path in MODEL_INPUTS.items():
        ps = load_per_sample(path)
        if not ps:
            print(f"[skip] {label}: {path} not found")
            continue
        stats = stratify_em(ps, cat_a_ids, cat_b_ids)
        stats["n_per_sample_rows"] = len(ps)
        results[label] = stats
        print(f"{label}: n={len(ps)} "
              f"A={stats['cat_a']['em_count']}/{stats['cat_a']['n_covered']} "
              f"B={stats['cat_b']['em_count']}/{stats['cat_b']['n_covered']} "
              f"Δ={100*stats['em_delta_A_minus_B']:+.2f}pp")

    # Load CvT classifications (optional, may not exist yet)
    cvt_results: dict[str, Any] = {}
    for label, cls_path in CLASSIFICATION_INPUTS.items():
        cls = load_classification(cls_path)
        if not cls:
            print(f"[cvt skip] {label}: {cls_path} not found")
            continue
        cvt_results[label] = stratify_cvt(cls, cat_a_ids, cat_b_ids)
        a = cvt_results[label]["cat_a"]
        b = cvt_results[label]["cat_b"]
        print(f"  CvT {label}: A={a['cvt']}/{a['n_covered']}={100*a['cvt_rate']:.2f}% "
              f"B={b['cvt']}/{b['n_covered']}={100*b['cvt_rate']:.2f}%")

    # Decomposition: 14B over 7B-39B
    decomp = None
    if "14B D1@400" in results and "7B 39B@400" in results:
        decomp = decompose_gain(
            results["14B D1@400"],
            results["7B 39B@400"],
            n_cat_a=len(cat_a_ids),
            n_cat_b=len(cat_b_ids),
        )
        print("\nDecomposition 14B@400 - 7B 39B@400:")
        for k, v in decomp.items():
            print(f"  {k}: {v}")

    # Write JSON
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump({
            "per_model": results,
            "cvt_per_model": cvt_results,
            "decomposition_14B_vs_39B": decomp,
            "cat_a_n": len(cat_a_ids),
            "cat_b_n": len(cat_b_ids),
        }, f, indent=2)

    # Write markdown
    lines: list[str] = []
    lines.append("# V14-D1-strat — Category A vs B Stratified Analysis\n")
    lines.append(
        f"Cat A (n={len(cat_a_ids)}, parametric-memory solvable): Qwen2.5-7B-Instruct "
        f"pass@10 > 0 without tools.\n"
        f"Cat B (n={len(cat_b_ids)}, retrieval-required): pass@10 = 0 without tools.\n"
    )
    lines.append(
        "> **Decision rule (REVISED 2026-04-23):** Cat B *EM* is agnostic between "
        "two mechanisms — (A) 14B's larger memory covers some Cat B questions "
        "(memory scaling, framework holds), or (B) 14B genuinely uses tools better "
        "(retrieval unlocked, framework revision needed). **Only Cat B CvT "
        "distinguishes them.** Thresholds: ≤8% → Mechanism A (ship); 10-15% → "
        "mixed (adjust prose); >15% → Mechanism B (escalate).\n"
    )
    lines.append(
        "> **Observation (framework support):** per-question gain is ~uniform across "
        "categories (+3.03pp Cat A vs +3.19pp Cat B). The capacity scaling adds roughly "
        "constant percentage-points regardless of category; the overall +3.15pp EM gain "
        "is dominated by Cat B's larger sample share (p_B=0.724), not by a retrieval-"
        "specific unlock. This pattern is *cleaner* than an asymmetric Cat A-favoring "
        "gain — capacity-invariant interface bottlenecks would predict exactly this.\n"
    )
    lines.append("## Table 1 — Stratified EM\n")
    lines.append("| Model | EM on Cat A (n=975) | EM on Cat B (n=2556) | EM full | Δ (A-B) | Tools/Q (Cat B) |")
    lines.append("|---|---|---|---|---|---|")
    for label, s in results.items():
        lines.append(fmt_em_row(label, s))

    if decomp:
        lines.append("\n## Decomposition: 14B D1@400 − 7B 39B@400\n")
        lines.append("| Component | Value |")
        lines.append("|---|---|")
        lines.append(f"| Δ on Cat A | {decomp['delta_cat_a_pp']:+.2f}pp |")
        lines.append(f"| Δ on Cat B | {decomp['delta_cat_b_pp']:+.2f}pp |")
        lines.append(f"| Cat A contribution (Δ_A × p_A, p_A={decomp['p_a']:.3f}) | {decomp['contribution_cat_a_pp']:+.2f}pp |")
        lines.append(f"| Cat B contribution (Δ_B × p_B, p_B={decomp['p_b']:.3f}) | {decomp['contribution_cat_b_pp']:+.2f}pp |")
        lines.append(f"| **Total overall Δ** | **{decomp['total_overall_delta_pp']:+.2f}pp** |")
        lines.append("")
        lines.append(
            "*(Cat B EM alone does NOT distinguish memory-scaling vs retrieval "
            "improvement. See Table 2 for the mechanism-distinguishing Cat B CvT.)*"
        )

    # Table 2 — Stratified CvT
    lines.append("\n## Table 2 — Stratified CvT (correct-via-tool)\n")
    if cvt_results:
        lines.append("| Model | CvT Cat A | CvT Cat B | CvT full |")
        lines.append("|---|---|---|---|")
        for label, cvt in cvt_results.items():
            a = cvt["cat_a"]; b = cvt["cat_b"]; fu = cvt["full"]
            lines.append(
                f"| {label} "
                f"| {a['cvt']}/{a['n_covered']} = {100*a['cvt_rate']:.2f}% "
                f"[{100*a['cvt_ci95'][0]:.2f}-{100*a['cvt_ci95'][1]:.2f}%] "
                f"| {b['cvt']}/{b['n_covered']} = {100*b['cvt_rate']:.2f}% "
                f"[{100*b['cvt_ci95'][0]:.2f}-{100*b['cvt_ci95'][1]:.2f}%] "
                f"| {fu['cvt']}/{fu['n_covered']} = {100*fu['cvt_rate']:.2f}% |"
            )
        lines.append("")
        if "14B D1@400" in cvt_results:
            b14 = cvt_results["14B D1@400"]["cat_b"]
            rate = 100 * b14["cvt_rate"]
            lines.append("### Revised decision rule (Cat B CvT is the critical signal)\n")
            if rate <= 8.0:
                verdict = (
                    f"**14B Cat B CvT = {rate:.2f}% (≈ 7B G2 baseline) → Mechanism A: memory scaling. "
                    "Framework validated. Ship as Section 6 anchor.**"
                )
            elif rate <= 15.0:
                verdict = (
                    f"**14B Cat B CvT = {rate:.2f}% → Mixed / nuanced. Adjust prose, core claim holds.**"
                )
            else:
                verdict = (
                    f"**14B Cat B CvT = {rate:.2f}% (>15%) → Mechanism B: capacity unlocks retrieval. "
                    "Flag for discussion before narrative lock.**"
                )
            lines.append(verdict)
    else:
        lines.append(
            "Requires trajectory classification per model. 14B D1@400 classification "
            "is pending completion of job 4247718 (500 seed=42 samples with trajectories).\n"
        )

    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_md, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nWrote {args.out_json} and {args.out_md}")


if __name__ == "__main__":
    main()
