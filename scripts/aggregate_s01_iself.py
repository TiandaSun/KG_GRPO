#!/usr/bin/env python3
"""Aggregate the multi-seed I-Self GRPO study (review-plan item S0.1).

Three seeds {1, 2, 3} of the same I-Self recipe (plus the existing single-seed
reference treated as "seed 0"), evaluated at steps
{0, 50, 100, 150, 200, 250, 300, 350, 400}.

Reads per-seed full-test summaries and per-sample results, computes mean / std
/ min / max across seeds {0, 1, 2, 3} for EM and ContEM (and CvT if available
via S0.3 ``post_collapse_inspection.json``), then applies the paper-ready
decision rule:

  * "All 4 seeds peak between step 200-250 with peak EM 35-45 percent, all
    collapse by step 350" -> STRONG REPRO; ship Section 5 figure with mean
    plus/minus std band.
  * Otherwise: report mixed peaks; flag any seed that never collapses.

Outputs:
  * results/phase7/s01_aggregate.json -- full numerical breakdown
  * results/phase7/s01_aggregate.md   -- paper-ready table + verdict

The script gracefully handles missing files: if a seed/step result has not yet
landed it is skipped with a "pending" marker, and the script still produces a
partial table. It is fully re-runnable.

Stdlib + pandas + numpy + scipy.stats only. Login-node safe (CPU only).

Usage:
  python scripts/aggregate_s01_iself.py            # compute + write outputs
  python scripts/aggregate_s01_iself.py --dry-run  # only list present/missing
"""
from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path("/lus/lfs1aip2/scratch/u6gg/ts1201.u6gg/Project/KG_GRPO")
PHASE7_DIR = PROJECT_ROOT / "results/phase7"

STEPS: tuple[int, ...] = (0, 50, 100, 150, 200, 250, 300, 350, 400)
SEEDS: tuple[int, ...] = (0, 1, 2, 3)  # 0 = single-seed reference

OUT_JSON = PHASE7_DIR / "s01_aggregate.json"
OUT_MD = PHASE7_DIR / "s01_aggregate.md"

# Decision-rule thresholds (documented in module docstring).
PEAK_WINDOW: tuple[int, int] = (200, 250)
PEAK_EM_LOW: float = 0.35
PEAK_EM_HIGH: float = 0.45
COLLAPSE_BY_STEP: int = 350
COLLAPSE_RATIO: float = 0.05  # EM <= 5% of peak counts as collapse


# ---------------------------------------------------------------------------
# File path helpers
# ---------------------------------------------------------------------------

def summary_path(seed: int, step: int) -> Path:
    """Path to the top-level full-test summary for a seed/step pair."""
    if seed == 0:
        return PHASE7_DIR / f"39i_self_step{step}_full_test.json"
    return PHASE7_DIR / f"39i_self_seed{seed}_step{step}_full_test.json"


def per_sample_path(seed: int, step: int) -> Path:
    """Path to the per-sample JSON list for a seed/step pair."""
    if seed == 0:
        d = PHASE7_DIR / f"39i_self_step{step}_full_per_sample"
    else:
        d = PHASE7_DIR / f"39i_self_seed{seed}_step{step}_full_per_sample"
    return d / "step_0_per_sample.json"


# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------

def _safe_load(path: Path) -> Any | None:
    if not path.exists():
        return None
    try:
        with path.open() as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Failed to read %s: %s", path, exc)
        return None


def load_summary(seed: int, step: int) -> dict[str, float] | None:
    """Return the inner metrics dict (under top-level key '0') or None."""
    payload = _safe_load(summary_path(seed, step))
    if payload is None:
        return None
    if isinstance(payload, dict) and "0" in payload and isinstance(payload["0"], dict):
        return payload["0"]
    if isinstance(payload, dict) and "em" in payload:  # already flat
        return payload
    logger.warning("Unexpected schema in %s: keys=%s", summary_path(seed, step), list(payload.keys()))
    return None


def load_per_sample(seed: int, step: int) -> list[dict[str, Any]] | None:
    payload = _safe_load(per_sample_path(seed, step))
    if payload is None:
        return None
    if isinstance(payload, list):
        return payload
    logger.warning("Per-sample at %s is not a list (got %s)", per_sample_path(seed, step), type(payload).__name__)
    return None


def load_post_collapse_cvt() -> dict[int, dict[str, Any]]:
    """Load the S0.3 single-seed CvT cross-tab if present.

    Schema reminder: ``post_collapse_inspection.json`` has top-level keys
    ``step_{N}`` for N in {200, 250, 300, 400}. Each value contains
    ``n_total``, ``n_correct``, ``verbatim_single``, ``verbatim_any``, etc.

    We approximate per-step CvT (correct-via-tool) as ``verbatim_any``
    (fraction of the FULL test set where any gold alias appears verbatim in
    a tool response). This is only available for seed 0 with current data,
    and is reported with a footnote.
    """
    payload = _safe_load(PHASE7_DIR / "post_collapse_inspection.json")
    if not isinstance(payload, dict):
        return {}
    out: dict[int, dict[str, Any]] = {}
    for k, v in payload.items():
        if not k.startswith("step_") or not isinstance(v, dict):
            continue
        try:
            step = int(k.split("_", 1)[1])
        except ValueError:
            continue
        out[step] = v
    return out


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

@dataclass
class SeedStepRow:
    seed: int
    step: int
    em: float | None = None
    contains_em: float | None = None
    f1: float | None = None
    avg_tool_calls: float | None = None
    n_samples: int | None = None
    cvt_proxy: float | None = None  # verbatim_any from S0.3 if available
    pending_summary: bool = True
    pending_per_sample: bool = True


def collect_rows() -> list[SeedStepRow]:
    cvt_table = load_post_collapse_cvt()
    rows: list[SeedStepRow] = []
    for seed in SEEDS:
        for step in STEPS:
            row = SeedStepRow(seed=seed, step=step)
            summary = load_summary(seed, step)
            if summary is not None:
                row.pending_summary = False
                row.em = float(summary.get("em")) if summary.get("em") is not None else None
                row.contains_em = (
                    float(summary["contains_em"]) if summary.get("contains_em") is not None else None
                )
                row.f1 = float(summary["f1"]) if summary.get("f1") is not None else None
                row.avg_tool_calls = (
                    float(summary["avg_tool_calls"]) if summary.get("avg_tool_calls") is not None else None
                )
                row.n_samples = (
                    int(summary["n_samples"]) if summary.get("n_samples") is not None else None
                )
            ps = load_per_sample(seed, step)
            row.pending_per_sample = ps is None
            # CvT proxy: only seed 0 from existing S0.3 file.
            if seed == 0 and step in cvt_table:
                v = cvt_table[step].get("verbatim_any")
                if v is not None:
                    row.cvt_proxy = float(v)
            rows.append(row)
    return rows


def _stats(values: list[float]) -> dict[str, float] | None:
    arr = [v for v in values if v is not None]
    if not arr:
        return None
    a = np.asarray(arr, dtype=float)
    return {
        "mean": float(a.mean()),
        "std": float(a.std(ddof=1)) if a.size > 1 else 0.0,
        "min": float(a.min()),
        "max": float(a.max()),
        "n": int(a.size),
    }


def aggregate_per_step(rows: list[SeedStepRow]) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    by_step: dict[int, list[SeedStepRow]] = {s: [] for s in STEPS}
    for r in rows:
        by_step[r.step].append(r)
    for step, group in by_step.items():
        em_vals = [r.em for r in group if r.em is not None]
        cem_vals = [r.contains_em for r in group if r.contains_em is not None]
        f1_vals = [r.f1 for r in group if r.f1 is not None]
        cvt_vals = [r.cvt_proxy for r in group if r.cvt_proxy is not None]
        present_seeds = sorted(r.seed for r in group if not r.pending_summary)
        pending_seeds = sorted(r.seed for r in group if r.pending_summary)
        out[step] = {
            "em": _stats(em_vals),
            "contains_em": _stats(cem_vals),
            "f1": _stats(f1_vals),
            "cvt_proxy_verbatim_any": _stats(cvt_vals),
            "present_seeds": present_seeds,
            "pending_seeds": pending_seeds,
        }
    return out


def per_seed_curve(rows: list[SeedStepRow]) -> dict[int, dict[str, Any]]:
    """Return per-seed dict of {step -> em, contains_em, cvt_proxy} plus peak / collapse."""
    by_seed: dict[int, list[SeedStepRow]] = {s: [] for s in SEEDS}
    for r in rows:
        by_seed[r.seed].append(r)
    out: dict[int, dict[str, Any]] = {}
    for seed, group in by_seed.items():
        group_sorted = sorted(group, key=lambda r: r.step)
        curve = {
            r.step: {
                "em": r.em,
                "contains_em": r.contains_em,
                "cvt_proxy": r.cvt_proxy,
                "pending": r.pending_summary,
            }
            for r in group_sorted
        }
        # Peak: prefer CvT if any CvT values are present for this seed, else EM.
        cvt_steps = [(s, v["cvt_proxy"]) for s, v in curve.items() if v["cvt_proxy"] is not None]
        em_steps = [(s, v["em"]) for s, v in curve.items() if v["em"] is not None]
        peak_basis = "cvt_proxy" if cvt_steps else "em"
        peak_source = cvt_steps if cvt_steps else em_steps
        if peak_source:
            peak_step, peak_val = max(peak_source, key=lambda kv: kv[1])
        else:
            peak_step, peak_val = None, None
        # Collapse: smallest step strictly after peak where EM <= COLLAPSE_RATIO * peak_em.
        collapse_step: int | None = None
        peak_em = curve.get(peak_step, {}).get("em") if peak_step is not None else None
        if peak_em is not None and peak_em > 0:
            threshold = COLLAPSE_RATIO * peak_em
            for s, v in curve.items():
                em = v["em"]
                if em is None or s <= peak_step:
                    continue
                if em <= threshold:
                    collapse_step = s
                    break
        out[seed] = {
            "curve": curve,
            "peak_basis": peak_basis,
            "peak_step": peak_step,
            "peak_value": peak_val,
            "peak_em": peak_em,
            "collapse_step": collapse_step,
            "never_collapsed": collapse_step is None and peak_em is not None,
        }
    return out


def apply_decision_rule(per_seed: dict[int, dict[str, Any]]) -> dict[str, Any]:
    """Decision rule from the docstring; returns a small verdict dict."""
    seeds_with_data = [s for s, info in per_seed.items() if info["peak_step"] is not None]
    peaks = [per_seed[s]["peak_step"] for s in seeds_with_data]
    peak_ems = [per_seed[s]["peak_em"] for s in seeds_with_data if per_seed[s]["peak_em"] is not None]
    collapses = [per_seed[s]["collapse_step"] for s in seeds_with_data]
    never_collapsed = [s for s in seeds_with_data if per_seed[s]["never_collapsed"]]

    def in_window(step: int | None) -> bool:
        return step is not None and PEAK_WINDOW[0] <= step <= PEAK_WINDOW[1]

    def collapsed_in_time(step: int | None) -> bool:
        return step is not None and step <= COLLAPSE_BY_STEP

    all_in_peak_window = bool(seeds_with_data) and all(in_window(p) for p in peaks)
    all_em_in_band = bool(peak_ems) and all(PEAK_EM_LOW <= e <= PEAK_EM_HIGH for e in peak_ems)
    all_collapsed = bool(seeds_with_data) and all(collapsed_in_time(c) for c in collapses)
    n_seeds = len(seeds_with_data)
    enough_seeds = n_seeds >= len(SEEDS)

    if enough_seeds and all_in_peak_window and all_em_in_band and all_collapsed:
        verdict = "STRONG_REPRO"
        reason = (
            f"All {n_seeds} seeds peak in step window {PEAK_WINDOW}, peak EM in "
            f"[{PEAK_EM_LOW:.2f}, {PEAK_EM_HIGH:.2f}], all collapsed by step {COLLAPSE_BY_STEP}."
        )
        recommendation = "Ship Section 5 figure with mean +/- std band."
    else:
        verdict = "MIXED"
        reasons: list[str] = []
        if not enough_seeds:
            reasons.append(f"only {n_seeds}/{len(SEEDS)} seeds have data")
        if not all_in_peak_window:
            reasons.append(f"peak steps span {sorted(set(peaks))}")
        if peak_ems and not all_em_in_band:
            reasons.append(
                f"peak EM range [{min(peak_ems):.3f}, {max(peak_ems):.3f}] "
                f"outside [{PEAK_EM_LOW}, {PEAK_EM_HIGH}]"
            )
        if not all_collapsed:
            reasons.append(f"collapse steps {collapses} (need <= {COLLAPSE_BY_STEP})")
        if never_collapsed:
            reasons.append(f"seeds never collapsed: {never_collapsed}")
        reason = "; ".join(reasons) if reasons else "insufficient data"
        recommendation = "Report mixed peaks; flag any non-collapsing seed in paper appendix."

    return {
        "verdict": verdict,
        "reason": reason,
        "recommendation": recommendation,
        "peak_window": list(PEAK_WINDOW),
        "peak_em_band": [PEAK_EM_LOW, PEAK_EM_HIGH],
        "collapse_by_step": COLLAPSE_BY_STEP,
        "collapse_ratio": COLLAPSE_RATIO,
        "n_seeds_with_data": n_seeds,
        "seeds_with_data": seeds_with_data,
        "peak_steps_per_seed": {s: per_seed[s]["peak_step"] for s in seeds_with_data},
        "peak_em_per_seed": {s: per_seed[s]["peak_em"] for s in seeds_with_data},
        "collapse_steps_per_seed": {s: per_seed[s]["collapse_step"] for s in seeds_with_data},
        "never_collapsed_seeds": never_collapsed,
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _fmt_stat(stat: dict[str, float] | None, *, pct: bool = True) -> str:
    if stat is None:
        return "--"
    scale = 100.0 if pct else 1.0
    suffix = "%" if pct else ""
    return f"{scale*stat['mean']:.2f}{suffix} +/- {scale*stat['std']:.2f}"


def render_markdown(
    per_step: dict[int, dict[str, Any]],
    per_seed: dict[int, dict[str, Any]],
    decision: dict[str, Any],
    cvt_available: bool,
) -> str:
    lines: list[str] = []
    lines.append("# S0.1 Multi-seed I-Self Aggregate")
    lines.append("")
    lines.append("Review-plan item: **S0.1 -- Multi-seed I-Self x3 (seeds 1, 2, 3)**.")
    lines.append("Single-seed reference (existing data) is treated as seed 0.")
    lines.append("")
    lines.append(f"- Steps evaluated: {list(STEPS)}")
    lines.append(f"- Seeds: {list(SEEDS)} (seed 0 = existing single-seed reference)")
    if not cvt_available:
        lines.append(
            "- *CvT footnote*: per-step CvT was not available across seeds; "
            "ContEM is reported as a proxy column. The ``cvt_proxy_verbatim_any`` "
            "column reflects S0.3 single-seed verbatim-any rates only "
            "(seed 0 only)."
        )
    lines.append("")

    # Per-step table.
    lines.append("## Per-step aggregate across seeds {0,1,2,3}")
    lines.append("")
    lines.append(
        "| step | n_seeds | EM mean +/- std | EM min | EM max | "
        "ContEM mean +/- std | CvT-proxy (seed 0) | pending seeds |"
    )
    lines.append("|---:|---:|---|---:|---:|---|---|---|")
    for step in STEPS:
        bucket = per_step[step]
        em = bucket["em"]
        cem = bucket["contains_em"]
        cvt = bucket["cvt_proxy_verbatim_any"]
        n = em["n"] if em else 0
        em_min = f"{100*em['min']:.2f}%" if em else "--"
        em_max = f"{100*em['max']:.2f}%" if em else "--"
        pending = ",".join(str(s) for s in bucket["pending_seeds"]) or "--"
        lines.append(
            f"| {step} | {n} | {_fmt_stat(em)} | {em_min} | {em_max} | "
            f"{_fmt_stat(cem)} | {_fmt_stat(cvt)} | {pending} |"
        )
    lines.append("")

    # Per-seed peak / collapse.
    lines.append("## Per-seed peak and collapse")
    lines.append("")
    lines.append("| seed | peak basis | peak step | peak EM | collapse step | never collapsed |")
    lines.append("|---:|---|---:|---:|---:|---|")
    for seed in SEEDS:
        info = per_seed[seed]
        peak_step = info["peak_step"]
        peak_em = info["peak_em"]
        collapse = info["collapse_step"]
        peak_step_s = str(peak_step) if peak_step is not None else "--"
        peak_em_s = f"{100*peak_em:.2f}%" if peak_em is not None else "--"
        collapse_s = str(collapse) if collapse is not None else "--"
        never_s = "YES" if info["never_collapsed"] else "no"
        lines.append(
            f"| {seed} | {info['peak_basis']} | {peak_step_s} | "
            f"{peak_em_s} | {collapse_s} | {never_s} |"
        )
    lines.append("")

    # Decision rule.
    lines.append("## Decision-rule verdict")
    lines.append("")
    lines.append(f"**Verdict: {decision['verdict']}**")
    lines.append("")
    lines.append(f"- Reason: {decision['reason']}")
    lines.append(f"- Recommendation: {decision['recommendation']}")
    lines.append(
        f"- Rule: peaks in window {decision['peak_window']}, "
        f"peak EM in {decision['peak_em_band']}, "
        f"collapse (EM <= {decision['collapse_ratio']:.0%} of peak) "
        f"by step {decision['collapse_by_step']}."
    )
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def dry_run_report() -> int:
    n_present = 0
    n_missing = 0
    print("File presence audit (summary | per_sample):")
    for seed in SEEDS:
        for step in STEPS:
            sp = summary_path(seed, step)
            pp = per_sample_path(seed, step)
            sp_ok = sp.exists()
            pp_ok = pp.exists()
            n_present += int(sp_ok) + int(pp_ok)
            n_missing += int(not sp_ok) + int(not pp_ok)
            print(
                f"  seed={seed} step={step:>3}  "
                f"summary={'OK' if sp_ok else 'MISSING'}  "
                f"per_sample={'OK' if pp_ok else 'MISSING'}"
            )
    cvt = PHASE7_DIR / "post_collapse_inspection.json"
    print(f"S0.3 CvT cross-tab ({cvt}): {'OK' if cvt.exists() else 'MISSING'}")
    print(f"Totals: present={n_present}  missing={n_missing}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List which input files are present/missing and exit without computing.",
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=OUT_JSON,
        help=f"Output JSON path (default: {OUT_JSON})",
    )
    parser.add_argument(
        "--out-md",
        type=Path,
        default=OUT_MD,
        help=f"Output Markdown path (default: {OUT_MD})",
    )
    args = parser.parse_args()

    if args.dry_run:
        return dry_run_report()

    rows = collect_rows()
    per_step = aggregate_per_step(rows)
    per_seed = per_seed_curve(rows)
    cvt_available = any(
        per_step[s]["cvt_proxy_verbatim_any"] is not None for s in STEPS
    )
    decision = apply_decision_rule(per_seed)

    payload: dict[str, Any] = {
        "task": "S0.1 multi-seed I-Self aggregate",
        "steps": list(STEPS),
        "seeds": list(SEEDS),
        "per_step": {str(k): v for k, v in per_step.items()},
        "per_seed": {
            str(k): {
                "curve": {str(s): vv for s, vv in v["curve"].items()},
                "peak_basis": v["peak_basis"],
                "peak_step": v["peak_step"],
                "peak_value": v["peak_value"],
                "peak_em": v["peak_em"],
                "collapse_step": v["collapse_step"],
                "never_collapsed": v["never_collapsed"],
            }
            for k, v in per_seed.items()
        },
        "decision": decision,
        "cvt_proxy_available": cvt_available,
        "notes": {
            "cvt_proxy": (
                "verbatim_any from results/phase7/post_collapse_inspection.json "
                "(S0.3, single-seed only). Used as a proxy column with footnote; "
                "we never fabricate per-seed CvT numbers."
            ),
            "peak_basis": "EM is default; CvT proxy is preferred only if any CvT value is present for that seed.",
            "collapse_definition": f"smallest step > peak with EM <= {COLLAPSE_RATIO:.0%} of peak EM.",
        },
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2))
    args.out_md.write_text(render_markdown(per_step, per_seed, decision, cvt_available))

    # Optional pandas summary so the dependency is actually exercised.
    df = pd.DataFrame(
        [
            {
                "seed": r.seed,
                "step": r.step,
                "em": r.em,
                "contains_em": r.contains_em,
                "f1": r.f1,
                "n_samples": r.n_samples,
                "pending": r.pending_summary,
            }
            for r in rows
        ]
    )
    logger.info("Aggregate rows:\n%s", df.to_string(index=False))
    logger.info("Wrote %s", args.out_json)
    logger.info("Wrote %s", args.out_md)
    logger.info("Verdict: %s -- %s", decision["verdict"], decision["reason"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
