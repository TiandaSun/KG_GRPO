#!/usr/bin/env python3
"""Aggregate the multi-seed G2 study (review-plan item S0.2).

Three seeds {1, 2, 3} of SFT-from-base + GRPO-with-E5b at step 500, plus the
existing single-seed reference (treated as "seed 0").

Computes:
  * mean / std / min / max across seeds {0, 1, 2, 3} for
    {EM, ContEM, F1, num_tool_calls, n_samples}
  * pairwise McNemar test on EM (with continuity correction); falls back to
    a paired-bootstrap 95% CI on EM-difference if scipy is unavailable
  * pairwise per-question agreement on EM, including a 4x4 agreement matrix
    and the mean pairwise agreement
  * "robustness" verdict:
      - all 4 seeds within +/- 2 percentage points of each other -> ROBUST
      - spread > 5 percentage points -> VARIABLE
      - otherwise -> MARGINAL

Outputs:
  * results/phase7/s02_aggregate.json
  * results/phase7/s02_aggregate.md

Gracefully handles missing files (a per-seed file may be marked "pending"
without aborting). Re-runnable.

Stdlib + pandas + numpy + scipy.stats only. Login-node safe (CPU only).

Usage:
  python scripts/aggregate_s02_g2.py            # compute + write outputs
  python scripts/aggregate_s02_g2.py --dry-run  # only list present/missing
"""
from __future__ import annotations

import argparse
import itertools
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path("/lus/lfs1aip2/scratch/u6gg/ts1201.u6gg/Project/KG_GRPO")
PHASE7_DIR = PROJECT_ROOT / "results/phase7"

STEP: int = 500
SEEDS: tuple[int, ...] = (0, 1, 2, 3)  # 0 = single-seed reference

OUT_JSON = PHASE7_DIR / "s02_aggregate.json"
OUT_MD = PHASE7_DIR / "s02_aggregate.md"

# Verdict thresholds (in percentage points, on EM).
ROBUST_PP: float = 2.0
VARIABLE_PP: float = 5.0


# ---------------------------------------------------------------------------
# File path helpers
# ---------------------------------------------------------------------------

def summary_path(seed: int) -> Path:
    if seed == 0:
        return PHASE7_DIR / f"39g2_step{STEP}_full_test.json"
    return PHASE7_DIR / f"39g2_seed{seed}_step{STEP}_full_test.json"


def per_sample_path(seed: int) -> Path:
    if seed == 0:
        d = PHASE7_DIR / f"39g2_step{STEP}_full_per_sample"
    else:
        d = PHASE7_DIR / f"39g2_seed{seed}_step{STEP}_full_per_sample"
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


def load_summary(seed: int) -> dict[str, float] | None:
    payload = _safe_load(summary_path(seed))
    if payload is None:
        return None
    if isinstance(payload, dict) and "0" in payload and isinstance(payload["0"], dict):
        return payload["0"]
    if isinstance(payload, dict) and "em" in payload:
        return payload
    logger.warning("Unexpected schema in %s", summary_path(seed))
    return None


def load_per_sample(seed: int) -> list[dict[str, Any]] | None:
    payload = _safe_load(per_sample_path(seed))
    if isinstance(payload, list):
        return payload
    return None


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

@dataclass
class SeedRow:
    seed: int
    em: float | None = None
    contains_em: float | None = None
    f1: float | None = None
    avg_tool_calls: float | None = None
    n_samples: int | None = None
    pending_summary: bool = True
    pending_per_sample: bool = True
    em_by_id: dict[str, int] | None = None  # sample_id -> 0/1


METRIC_KEYS: tuple[str, ...] = ("em", "contains_em", "f1", "avg_tool_calls", "n_samples")


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


def collect_rows() -> list[SeedRow]:
    rows: list[SeedRow] = []
    for seed in SEEDS:
        row = SeedRow(seed=seed)
        summary = load_summary(seed)
        if summary is not None:
            row.pending_summary = False
            for k in METRIC_KEYS:
                v = summary.get(k)
                if v is None:
                    continue
                if k == "n_samples":
                    row.n_samples = int(v)
                else:
                    setattr(row, k, float(v))
        ps = load_per_sample(seed)
        if ps is not None:
            row.pending_per_sample = False
            row.em_by_id = {
                str(item["sample_id"]): int(round(float(item.get("em", 0.0))))
                for item in ps
                if "sample_id" in item
            }
        rows.append(row)
    return rows


def aggregate_metrics(rows: list[SeedRow]) -> dict[str, dict[str, float] | None]:
    out: dict[str, dict[str, float] | None] = {}
    for k in METRIC_KEYS:
        vals = [getattr(r, k) for r in rows if getattr(r, k) is not None]
        out[k] = _stats(vals)
    return out


# ---------------------------------------------------------------------------
# Pairwise tests
# ---------------------------------------------------------------------------

def _align(a: dict[str, int], b: dict[str, int]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    keys = sorted(set(a) & set(b))
    av = np.asarray([a[k] for k in keys], dtype=int)
    bv = np.asarray([b[k] for k in keys], dtype=int)
    return av, bv, keys


def mcnemar_em(a: dict[str, int], b: dict[str, int]) -> dict[str, Any]:
    """McNemar test on paired binary EM outcomes (a vs b).

    Uses the continuity-corrected chi-square form:
        chi2 = (|b01 - b10| - 1)^2 / (b01 + b10),  df=1
    Falls back to "no_disagreements" when b01 + b10 == 0.
    Also reports a paired-bootstrap 95% CI on EM-difference for robustness.
    """
    av, bv, keys = _align(a, b)
    n = len(keys)
    if n == 0:
        return {"n_aligned": 0, "p_value": None, "note": "no_aligned_samples"}
    b01 = int(np.sum((av == 0) & (bv == 1)))
    b10 = int(np.sum((av == 1) & (bv == 0)))
    both1 = int(np.sum((av == 1) & (bv == 1)))
    both0 = int(np.sum((av == 0) & (bv == 0)))
    if b01 + b10 == 0:
        chi2, p = 0.0, 1.0
        note = "no_disagreements"
    else:
        chi2 = (abs(b01 - b10) - 1) ** 2 / (b01 + b10)
        p = float(scipy_stats.chi2.sf(chi2, df=1))
        note = "mcnemar_continuity_corrected"

    # Paired bootstrap on the difference (a_em - b_em).
    rng = np.random.default_rng(20260428)
    n_boot = 2000
    diffs = av.astype(float) - bv.astype(float)
    boots = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots[i] = float(diffs[idx].mean())
    ci_low, ci_high = np.quantile(boots, [0.025, 0.975]).tolist()
    return {
        "n_aligned": n,
        "b01_only_b_correct": b01,
        "b10_only_a_correct": b10,
        "both_correct": both1,
        "both_wrong": both0,
        "chi2": float(chi2),
        "p_value": float(p),
        "note": note,
        "em_diff_mean": float(diffs.mean()),
        "em_diff_bootstrap_ci95": [float(ci_low), float(ci_high)],
    }


def agreement_matrix(rows: list[SeedRow]) -> dict[str, Any]:
    """Pairwise EM agreement (fraction of identical EM labels) across seeds.

    Returns a NxN matrix indexed by SEEDS, plus mean off-diagonal agreement.
    Pairs with no aligned samples are reported as None.
    """
    n_seeds = len(SEEDS)
    matrix: list[list[float | None]] = [[None] * n_seeds for _ in range(n_seeds)]
    components: dict[str, dict[str, int]] = {}
    pairs: list[float] = []
    for i, sa in enumerate(SEEDS):
        for j, sb in enumerate(SEEDS):
            if i == j:
                matrix[i][j] = 1.0
                continue
            ra = next(r for r in rows if r.seed == sa)
            rb = next(r for r in rows if r.seed == sb)
            if ra.em_by_id is None or rb.em_by_id is None:
                continue
            av, bv, keys = _align(ra.em_by_id, rb.em_by_id)
            if len(keys) == 0:
                continue
            both1 = int(np.sum((av == 1) & (bv == 1)))
            both0 = int(np.sum((av == 0) & (bv == 0)))
            only_a = int(np.sum((av == 1) & (bv == 0)))
            only_b = int(np.sum((av == 0) & (bv == 1)))
            agreement = (both1 + both0) / len(keys)
            matrix[i][j] = float(agreement)
            if i < j:
                pairs.append(agreement)
                components[f"{sa}-{sb}"] = {
                    "both_correct": both1,
                    "both_wrong": both0,
                    "only_a_correct": only_a,
                    "only_b_correct": only_b,
                    "n_aligned": len(keys),
                    "agreement": float(agreement),
                }
    mean_pairwise = float(np.mean(pairs)) if pairs else None
    return {
        "seeds": list(SEEDS),
        "matrix": matrix,
        "mean_pairwise_agreement": mean_pairwise,
        "components": components,
    }


def pairwise_mcnemar(rows: list[SeedRow]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for sa, sb in itertools.combinations(SEEDS, 2):
        ra = next(r for r in rows if r.seed == sa)
        rb = next(r for r in rows if r.seed == sb)
        if ra.em_by_id is None or rb.em_by_id is None:
            out[f"{sa}-{sb}"] = {"note": "missing_per_sample"}
            continue
        out[f"{sa}-{sb}"] = mcnemar_em(ra.em_by_id, rb.em_by_id)
    return out


# ---------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------

def robustness_verdict(em_stats: dict[str, float] | None, n_seeds_present: int) -> dict[str, Any]:
    if em_stats is None or n_seeds_present < 2:
        return {
            "verdict": "INSUFFICIENT_DATA",
            "spread_pp": None,
            "n_seeds_with_data": n_seeds_present,
            "rule": f"need >=2 seeds with EM; ROBUST <= {ROBUST_PP}pp; VARIABLE > {VARIABLE_PP}pp.",
        }
    spread_pp = 100.0 * (em_stats["max"] - em_stats["min"])
    if spread_pp <= ROBUST_PP:
        verdict = "ROBUST"
    elif spread_pp > VARIABLE_PP:
        verdict = "VARIABLE"
    else:
        verdict = "MARGINAL"
    return {
        "verdict": verdict,
        "spread_pp": float(spread_pp),
        "n_seeds_with_data": n_seeds_present,
        "rule": f"ROBUST <= {ROBUST_PP}pp; VARIABLE > {VARIABLE_PP}pp; otherwise MARGINAL.",
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _fmt_metric(stat: dict[str, float] | None, *, pct: bool, fmt: str = ".3f") -> str:
    if stat is None:
        return "--"
    scale = 100.0 if pct else 1.0
    suffix = "%" if pct else ""
    return (
        f"{(scale*stat['mean']):{fmt}}{suffix} +/- {(scale*stat['std']):{fmt}}"
        f"  [{(scale*stat['min']):{fmt}}, {(scale*stat['max']):{fmt}}]"
    )


def render_markdown(
    rows: list[SeedRow],
    metrics: dict[str, dict[str, float] | None],
    pairwise: dict[str, dict[str, Any]],
    agree: dict[str, Any],
    verdict: dict[str, Any],
) -> str:
    lines: list[str] = []
    lines.append("# S0.2 Multi-seed G2 Aggregate (step 500)")
    lines.append("")
    lines.append("Review-plan item: **S0.2 -- Multi-seed G2 x3 (seeds 1, 2, 3)**.")
    lines.append("Single-seed reference (existing data) is treated as seed 0.")
    lines.append("")
    lines.append(f"- Step evaluated: {STEP}")
    lines.append(f"- Seeds: {list(SEEDS)} (seed 0 = existing single-seed reference)")
    lines.append("")

    # Per-seed table.
    lines.append("## Per-seed metrics")
    lines.append("")
    lines.append("| seed | EM | ContEM | F1 | avg_tool_calls | n_samples | status |")
    lines.append("|---:|---:|---:|---:|---:|---:|---|")
    for r in rows:
        if r.pending_summary:
            lines.append(f"| {r.seed} | -- | -- | -- | -- | -- | PENDING |")
            continue
        em = f"{100*r.em:.2f}%" if r.em is not None else "--"
        cem = f"{100*r.contains_em:.2f}%" if r.contains_em is not None else "--"
        f1 = f"{100*r.f1:.2f}%" if r.f1 is not None else "--"
        atc = f"{r.avg_tool_calls:.3f}" if r.avg_tool_calls is not None else "--"
        ns = str(r.n_samples) if r.n_samples is not None else "--"
        status = "OK" if not r.pending_per_sample else "summary-only"
        lines.append(f"| {r.seed} | {em} | {cem} | {f1} | {atc} | {ns} | {status} |")
    lines.append("")

    # Aggregate.
    lines.append("## Cross-seed aggregate (mean +/- std [min, max])")
    lines.append("")
    lines.append("| metric | aggregate |")
    lines.append("|---|---|")
    lines.append(f"| EM | {_fmt_metric(metrics['em'], pct=True)} |")
    lines.append(f"| ContEM | {_fmt_metric(metrics['contains_em'], pct=True)} |")
    lines.append(f"| F1 | {_fmt_metric(metrics['f1'], pct=True)} |")
    lines.append(f"| avg_tool_calls | {_fmt_metric(metrics['avg_tool_calls'], pct=False)} |")
    lines.append(f"| n_samples | {_fmt_metric(metrics['n_samples'], pct=False, fmt='.0f')} |")
    lines.append("")

    # Pairwise McNemar.
    lines.append("## Pairwise McNemar tests on EM")
    lines.append("")
    lines.append("| pair | n_aligned | only_a | only_b | both_correct | both_wrong | chi2 | p | EM diff (a-b) | bootstrap CI95 |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---|")
    for pair, info in pairwise.items():
        if "p_value" not in info or info.get("p_value") is None:
            lines.append(f"| {pair} | -- | -- | -- | -- | -- | -- | -- | -- | {info.get('note','--')} |")
            continue
        ci = info.get("em_diff_bootstrap_ci95")
        ci_s = f"[{ci[0]:.4f}, {ci[1]:.4f}]" if ci else "--"
        lines.append(
            f"| {pair} | {info['n_aligned']} | {info['b10_only_a_correct']} | "
            f"{info['b01_only_b_correct']} | {info['both_correct']} | {info['both_wrong']} | "
            f"{info['chi2']:.3f} | {info['p_value']:.4f} | {info['em_diff_mean']:.4f} | {ci_s} |"
        )
    lines.append("")

    # Agreement matrix.
    lines.append("## Pairwise EM agreement matrix")
    lines.append("")
    header = "| seed | " + " | ".join(str(s) for s in agree["seeds"]) + " |"
    sep = "|---:|" + "---:|" * len(agree["seeds"])
    lines.append(header)
    lines.append(sep)
    for i, s in enumerate(agree["seeds"]):
        cells: list[str] = []
        for j in range(len(agree["seeds"])):
            v = agree["matrix"][i][j]
            cells.append(f"{v:.4f}" if v is not None else "--")
        lines.append(f"| {s} | " + " | ".join(cells) + " |")
    mpa = agree["mean_pairwise_agreement"]
    lines.append("")
    lines.append(f"Mean off-diagonal pairwise agreement: {mpa:.4f}" if mpa is not None else "Mean off-diagonal pairwise agreement: --")
    lines.append("")

    # Verdict.
    lines.append("## Robustness verdict")
    lines.append("")
    lines.append(f"**Verdict: {verdict['verdict']}**")
    lines.append("")
    if verdict["spread_pp"] is not None:
        lines.append(f"- EM spread (max-min): {verdict['spread_pp']:.2f} pp across {verdict['n_seeds_with_data']} seeds")
    lines.append(f"- Rule: {verdict['rule']}")
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
        sp = summary_path(seed)
        pp = per_sample_path(seed)
        sp_ok = sp.exists()
        pp_ok = pp.exists()
        n_present += int(sp_ok) + int(pp_ok)
        n_missing += int(not sp_ok) + int(not pp_ok)
        print(
            f"  seed={seed}  step={STEP}  "
            f"summary={'OK' if sp_ok else 'MISSING'}  "
            f"per_sample={'OK' if pp_ok else 'MISSING'}"
        )
    print(f"Totals: present={n_present}  missing={n_missing}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--out-json", type=Path, default=OUT_JSON)
    parser.add_argument("--out-md", type=Path, default=OUT_MD)
    args = parser.parse_args()

    if args.dry_run:
        return dry_run_report()

    rows = collect_rows()
    metrics = aggregate_metrics(rows)
    pairwise = pairwise_mcnemar(rows)
    agree = agreement_matrix(rows)
    n_seeds_present = sum(1 for r in rows if r.em is not None)
    verdict = robustness_verdict(metrics["em"], n_seeds_present)

    payload: dict[str, Any] = {
        "task": "S0.2 multi-seed G2 aggregate",
        "step": STEP,
        "seeds": list(SEEDS),
        "per_seed": [
            {
                "seed": r.seed,
                "em": r.em,
                "contains_em": r.contains_em,
                "f1": r.f1,
                "avg_tool_calls": r.avg_tool_calls,
                "n_samples": r.n_samples,
                "pending_summary": r.pending_summary,
                "pending_per_sample": r.pending_per_sample,
            }
            for r in rows
        ],
        "metrics": metrics,
        "pairwise_mcnemar": pairwise,
        "agreement": agree,
        "verdict": verdict,
        "notes": {
            "test": "McNemar on paired EM with continuity correction; df=1.",
            "bootstrap": "Paired bootstrap 95% CI on EM-difference, n_boot=2000, seed=20260428.",
            "agreement": "Per-question EM-label agreement; matrix is symmetric, diagonal=1.",
        },
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2))
    args.out_md.write_text(render_markdown(rows, metrics, pairwise, agree, verdict))

    df = pd.DataFrame(
        [
            {
                "seed": r.seed,
                "em": r.em,
                "contains_em": r.contains_em,
                "f1": r.f1,
                "avg_tool_calls": r.avg_tool_calls,
                "n_samples": r.n_samples,
                "pending": r.pending_summary,
            }
            for r in rows
        ]
    )
    logger.info("Per-seed:\n%s", df.to_string(index=False))
    logger.info("Wrote %s", args.out_json)
    logger.info("Wrote %s", args.out_md)
    logger.info("Verdict: %s (spread=%s pp)", verdict["verdict"], verdict["spread_pp"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
