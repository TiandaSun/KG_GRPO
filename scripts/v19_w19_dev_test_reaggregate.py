"""v19 W19 — dev / held-out-test re-aggregation across the 10 paper checkpoints.

Random seed-42 stratified-by-hops split of CWQ 3,531-Q test set → dev 800,
held-out test 2,731. Triple-column report (dev / test / full) of EM, CvT, F1,
Tools/Q for each checkpoint. Plus re-do E5b+SelfV peak-step selection on DEV
ONLY across step ∈ {50, 100, 150, 200, 250, 300, 400}.

Pre-registered rule (LOCK before viewing): held-out-test within Wilson 95% CI
half-width (±1.8 pp at n=2,731) of full-test EM for ≥9 of 10 checkpoints, AND
E5b+SelfV peak step on DEV ∈ {200, 250, 300} → CONFIRMS robust.

Data sources (all 3,531 per-question records on disk; mix of trajectories +
per_sample), mapped 1:1 to the 10 paper checkpoints.
"""
from __future__ import annotations

import json
import logging
import math
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("w19")

PROJECT_ROOT = Path("/lus/lfs1aip2/scratch/u6gg/ts1201.u6gg/Project/KG_GRPO")
OUT_DIR = PROJECT_ROOT / "_handoff/data/w19"
OUT_DIR.mkdir(parents=True, exist_ok=True)
TEST_PARQUET = PROJECT_ROOT / "data/freebase/verl_cwq/test.parquet"

# canonical: HPC code → paper name → per-question records path → load type
CHECKPOINTS: list[dict[str, Any]] = [
    {"hpc": "E1@1250", "paper": "R-binary",
     "paths": [
         PROJECT_ROOT / "results/task14_per_sample/e1_1250/step_1250_chunk_0_2000.json",
         PROJECT_ROOT / "results/task14_per_sample/e1_1250/step_1250_chunk_2000_3531.json",
     ],
     "kind": "per_sample"},
    {"hpc": "E3@500", "paper": "R-stepwise",
     "paths": [PROJECT_ROOT / "results/phase7/e3_step500_full_per_sample/step_500_per_sample.json"],
     "kind": "per_sample"},
    {"hpc": "E5b@100", "paper": "R-toolverbs",
     "paths": [PROJECT_ROOT / "results/phase7/e5b_step100_full_per_sample/step_100_per_sample.json"],
     "kind": "per_sample"},
    {"hpc": "E5b+KL@400", "paper": "R-toolverbs.KL",
     "paths": [PROJECT_ROOT / "results/trajectories/phase7/39b_step400_full/step_400/trajectories.json"],
     "kind": "trajectories"},
    {"hpc": "E5b+SelfV@250", "paper": "R-selfV-peak",
     "paths": [PROJECT_ROOT / "results/trajectories/phase7/39i_self_step250_full/step_0/trajectories.json"],
     "kind": "trajectories"},
    {"hpc": "E5b+SelfV@300", "paper": "R-selfV-collapsed",
     "paths": [PROJECT_ROOT / "results/trajectories/phase7/39i_self_step300_full/step_0/trajectories.json"],
     "kind": "trajectories"},
    {"hpc": "G1@500", "paper": "init-from-iterate",
     "paths": [PROJECT_ROOT / "results/trajectories/phase7/39g1_step500_full/step_0/trajectories.json"],
     "kind": "trajectories"},
    {"hpc": "G2@500", "paper": "self-distill",
     "paths": [PROJECT_ROOT / "results/trajectories/phase7/39g2_step500_full/step_0/trajectories.json"],
     "kind": "trajectories"},
    {"hpc": "E1'@300", "paper": "R-binary-SR",
     "paths": [PROJECT_ROOT / "results/trajectories/phase7/e1prime_step300_full/step_0/trajectories.json"],
     "kind": "trajectories"},
    {"hpc": "E5b+KL-14B@400", "paper": "R-toolverbs.KL-14B",
     "paths": [PROJECT_ROOT / "results/phase7/v14_d1_qwen14b_e5b_step400_full_per_sample/step_0_per_sample.json"],
     "kind": "per_sample"},
]

ISELF_PER_SAMPLE_DIR = PROJECT_ROOT / "results/phase7"
ISELF_STEPS = [50, 100, 150, 200, 250, 300, 400]


def load_records(spec: dict) -> list[dict]:
    records: list[dict] = []
    for p in spec["paths"]:
        with open(p) as f:
            d = json.load(f)
        if isinstance(d, list):
            records.extend(d)
        else:
            raise ValueError(f"Unexpected format: {p}")
    return records


def aggregate(records: list[dict], ids: set[str]) -> dict[str, float]:
    em = cm = f1 = tools = 0.0
    n = 0
    for r in records:
        sid = str(r.get("sample_id", ""))
        if sid not in ids:
            continue
        em += float(r.get("em", 0.0))
        cm += float(r.get("contains_em", 0.0))
        # F1 may be missing in some dumps (trajectories save `f1`, per_sample also)
        f1 += float(r.get("f1", 0.0))
        tools += float(r.get("num_tool_calls", 0.0))
        n += 1
    if n == 0:
        return {"n": 0, "em": 0.0, "contains_em": 0.0, "f1": 0.0, "avg_tool_calls": 0.0}
    return {"n": n, "em": em / n, "contains_em": cm / n, "f1": f1 / n, "avg_tool_calls": tools / n}


def cvt_count(records: list[dict], ids: set[str], all_records_full: list[dict] | None = None) -> int | None:
    """CvT count = trajectories where em==1 AND any gold answer appears verbatim in tool responses.
    Requires `full_response`; returns None if not available (per_sample only)."""
    cnt = 0
    has_any = False
    for r in records:
        sid = str(r.get("sample_id", ""))
        if sid not in ids:
            continue
        full = r.get("full_response")
        em = float(r.get("em", 0.0))
        if full is None:
            continue
        has_any = True
        if em < 0.5:
            continue
        # Simple word-boundary check
        tool_text = " ".join(__import__("re").findall(r"<tool_response>(.*?)</tool_response>", full, __import__("re").DOTALL))
        if not tool_text:
            continue
        gold = r.get("gold_answer", "") or ""
        all_ans = r.get("all_answers") or [gold]
        tt_low = tool_text.lower()
        for a in all_ans:
            if a and a.lower() in tt_low:
                cnt += 1
                break
    return cnt if has_any else None


def stratified_split(sample_ids: list[str], hops_by_id: dict[str, int],
                     n_dev: int = 800, seed: int = 42) -> tuple[set[str], set[str]]:
    rng = random.Random(seed)
    by_hop: dict[int, list[str]] = defaultdict(list)
    for sid in sample_ids:
        by_hop[hops_by_id.get(sid, 0)].append(sid)
    total = len(sample_ids)
    dev: set[str] = set()
    for h, ids in by_hop.items():
        rng.shuffle(ids)
        k = round(n_dev * len(ids) / total)
        dev.update(ids[:k])
    # Adjust to exactly n_dev if rounding drift
    extras = list(set(sample_ids) - dev)
    rng.shuffle(extras)
    while len(dev) < n_dev:
        dev.add(extras.pop())
    while len(dev) > n_dev:
        dev.pop()
    test = set(sample_ids) - dev
    return dev, test


def wilson_halfwidth(p: float, n: int, z: float = 1.96) -> float:
    """Wilson-score 95% CI half-width for proportion p, sample n."""
    if n == 0:
        return 0.0
    p = max(0.0, min(1.0, p))
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    margin = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return margin


def main() -> int:
    # 1) Build sample_id → hops + canonical id list from test parquet.
    df = pd.read_parquet(TEST_PARQUET)
    logger.info("Loaded test parquet: n=%d", len(df))
    hops_by_id: dict[str, int] = {}
    all_ids: list[str] = []
    for _, row in df.iterrows():
        extra = row.get("extra_info") or {}
        try:
            sid = str(extra["sample_id"]) if "sample_id" in extra else None
            hops = int(extra.get("hops", 0))
        except Exception:
            sid = None
            hops = 0
        if sid is not None:
            all_ids.append(sid)
            hops_by_id[sid] = hops

    dev_ids, test_ids = stratified_split(all_ids, hops_by_id, n_dev=800, seed=42)
    full_ids = set(all_ids)
    logger.info("Split: dev=%d test=%d full=%d", len(dev_ids), len(test_ids), len(full_ids))

    # Hop-bucket counts in dev (sanity)
    dev_hops = defaultdict(int)
    for sid in dev_ids:
        dev_hops[hops_by_id.get(sid, 0)] += 1
    logger.info("Dev hop-bucket counts: %s", dict(dev_hops))

    # 2) Aggregate each checkpoint on each split.
    triple: dict[str, dict[str, Any]] = {}
    cvt_triple: dict[str, dict[str, Any]] = {}
    for spec in CHECKPOINTS:
        logger.info("Loading %s (%s) from %d file(s)", spec["hpc"], spec["paper"], len(spec["paths"]))
        records = load_records(spec)
        logger.info("  loaded %d records", len(records))
        triple[spec["hpc"]] = {
            "paper": spec["paper"],
            "kind": spec["kind"],
            "dev": aggregate(records, dev_ids),
            "test": aggregate(records, test_ids),
            "full": aggregate(records, full_ids),
        }
        # CvT counts only when trajectories available
        if spec["kind"] == "trajectories":
            cvt_triple[spec["hpc"]] = {
                "dev": cvt_count(records, dev_ids),
                "test": cvt_count(records, test_ids),
                "full": cvt_count(records, full_ids),
            }
        else:
            cvt_triple[spec["hpc"]] = {"dev": None, "test": None, "full": None,
                                       "note": "no full_response in per_sample; CvT count unavailable"}

    # 3) E5b+SelfV dev-peak-step scan
    iself_peak = {}
    for step in ISELF_STEPS:
        ps = ISELF_PER_SAMPLE_DIR / f"39i_self_step{step}_full_per_sample" / "step_0_per_sample.json"
        if not ps.exists():
            iself_peak[step] = {"missing": True}
            continue
        with open(ps) as f:
            data = json.load(f)
        iself_peak[step] = {
            "dev": aggregate(data, dev_ids),
            "test": aggregate(data, test_ids),
            "full": aggregate(data, full_ids),
        }
    # Pick peak by dev EM among non-missing
    dev_em_by_step = {s: v["dev"]["em"] for s, v in iself_peak.items() if "missing" not in v}
    dev_peak_step = max(dev_em_by_step, key=dev_em_by_step.get) if dev_em_by_step else None

    # 4) Pre-registered rule check
    half_width = wilson_halfwidth(0.4, 2731)  # ~1.85pp at p=0.4, n=2731
    within_ci = []
    for hpc, blk in triple.items():
        gap = abs(blk["test"]["em"] - blk["full"]["em"])
        within_ci.append((hpc, gap, gap <= half_width))
    n_within = sum(1 for _, _, ok in within_ci if ok)
    iself_peak_in_window = dev_peak_step in (200, 250, 300) if dev_peak_step is not None else False
    verdict = "CONFIRMS" if (n_within >= 9 and iself_peak_in_window) else (
        "FALSIFIES" if (sum(1 for _, g, _ in within_ci if g > 0.03) > 1 or
                        (dev_peak_step is not None and abs(dev_peak_step - 250) > 50))
        else "PARTIAL")

    out = {
        "split": {"seed": 42, "dev_n": len(dev_ids), "test_n": len(test_ids),
                  "full_n": len(full_ids), "dev_hop_counts": dict(dev_hops)},
        "checkpoints": triple,
        "cvt_counts": cvt_triple,
        "iself_peak_scan": iself_peak,
        "iself_dev_peak_step": dev_peak_step,
        "wilson_halfwidth_at_n2731_p04": half_width,
        "within_ci_per_checkpoint": [{"hpc": h, "em_gap": g, "within": ok} for h, g, ok in within_ci],
        "n_within_ci": n_within,
        "iself_dev_peak_in_window_200_300": iself_peak_in_window,
        "verdict": verdict,
    }
    with open(OUT_DIR / "wt19_results.json", "w") as f:
        json.dump(out, f, indent=2)
    logger.info("Wrote %s", OUT_DIR / "wt19_results.json")

    # Markdown report
    md = ["# v19 W19 — dev/test split re-aggregation\n"]
    md.append(f"Seed=42 stratified-by-hops split: dev={len(dev_ids)}, test={len(test_ids)}, full={len(full_ids)}.\n")
    md.append(f"Wilson 95% CI half-width @ n=2,731, p≈0.4: ±{half_width*100:.2f} pp.\n")
    md.append("## Triple-column EM (dev / held-out test / full)\n")
    md.append("| HPC | paper | dev EM | test EM | full EM | gap (test−full) | within ±1.85pp? |")
    md.append("|---|---|---|---|---|---|---|")
    for hpc, blk in triple.items():
        gap = blk["test"]["em"] - blk["full"]["em"]
        ok = abs(gap) <= half_width
        md.append(f"| {hpc} | {blk['paper']} | {blk['dev']['em']:.4f} | {blk['test']['em']:.4f} | "
                  f"{blk['full']['em']:.4f} | {gap:+.4f} | {'✓' if ok else '✗'} |")
    md.append("")
    md.append("## E5b+SelfV dev-peak-step scan\n")
    md.append("| step | dev EM | test EM | full EM |")
    md.append("|---|---|---|---|")
    for s in ISELF_STEPS:
        v = iself_peak.get(s, {})
        if "missing" in v:
            md.append(f"| {s} | (missing) | (missing) | (missing) |")
        else:
            md.append(f"| {s} | {v['dev']['em']:.4f} | {v['test']['em']:.4f} | {v['full']['em']:.4f} |")
    md.append(f"\n**Dev-peak step**: {dev_peak_step}  ({'in window {200,250,300}' if iself_peak_in_window else 'OUTSIDE window {200,250,300}'})\n")
    md.append(f"\n## Pre-registered verdict: **{verdict}**\n")
    md.append(f"- {n_within} / {len(triple)} checkpoints within ±1.85pp\n")
    md.append(f"- E5b+SelfV dev peak step = {dev_peak_step}\n")
    (OUT_DIR / "wt19_results.md").write_text("\n".join(md) + "\n")
    logger.info("Wrote %s", OUT_DIR / "wt19_results.md")

    return 0


if __name__ == "__main__":
    sys.exit(main())
