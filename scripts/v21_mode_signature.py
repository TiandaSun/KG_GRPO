"""v20 W21 Part A — mode-signature classifier across the multi-seed reward ladder.

For each (rung, seed) run, for each evaluated step, compute EM / CvT% / Tools/Q +
7-cat distribution (via task16_classify), then assign a mode signature per seed and
apply the pre-registered ≥2/3-seed rule. The 3rd seed per rung is the ORIGINAL
single-seed run (canonical signatures hard-coded from tab:ladder).

Mode expectations (from §4.3):
  E3   (R-stepwise)     -> Mode 2 ritual: CvT < 0.5%, Tools/Q ~ 1, flat across steps
  E5b  (R-toolverbs)    -> Mode 3 drift: CvT peaks early (~step 100) then EM/CvT degrade
  E5bkl(R-toolverbs.KL) -> stable plateau: EM stable through step 400, no collapse
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path("/lus/lfs1aip2/scratch/u6gg/ts1201.u6gg/Project/KG_GRPO")
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from task16_classify import classify_trajectory  # noqa: E402

LADDER_DIR = PROJECT_ROOT / "_handoff/data/w21/ladder_eval"
OUT_MD = PROJECT_ROOT / "_handoff/data/w21/wt21_partA_mode_signature.md"
OUT_JSON = PROJECT_ROOT / "_handoff/data/w21/wt21_partA_mode_signature.json"

RUNGS = ["e3", "e5b", "e5bkl"]
SEEDS = [2, 3]
STEPS = [100, 200, 300, 400]

# Original single-seed canonical signatures (from tab:ladder / Day-2 results).
ORIGINAL = {
    "e3":    {"label": "Mode2-ritual", "peak_cvt_pct": 0.03},
    "e5b":   {"label": "Mode3-drift",  "peak_cvt_pct": 3.03},
    "e5bkl": {"label": "stable-plateau", "peak_cvt_pct": 3.77},
}


def load_step(rung: str, seed: int, step: int) -> dict | None:
    base = LADDER_DIR / f"{rung}_seed{seed}"
    ev = base / f"step{step}.json"
    tj = base / f"step{step}_traj" / "step_0" / "trajectories.json"
    if not ev.exists():
        return None
    with open(ev) as f:
        agg = json.load(f)["0"]
    rec = {"em": agg["em"], "contains_em": agg["contains_em"],
           "tools_per_q": agg["avg_tool_calls"], "n": agg["n_samples"],
           "cvt_count": None, "cvt_pct": None}
    if tj.exists():
        with open(tj) as f:
            trajs = json.load(f)
        cvt = sum(1 for t in trajs if classify_trajectory(t) == "correct-via-tool")
        rec["cvt_count"] = cvt
        rec["cvt_pct"] = round(100.0 * cvt / max(len(trajs), 1), 3)
    return rec


def assign_mode(rung: str, curve: dict[int, dict]) -> str:
    """Heuristic mode label from the per-step curve."""
    steps = sorted(curve.keys())
    if not steps:
        return "no-data"
    cvts = [curve[s]["cvt_pct"] for s in steps if curve[s]["cvt_pct"] is not None]
    ems = [curve[s]["em"] for s in steps]
    tools = [curve[s]["tools_per_q"] for s in steps]
    peak_cvt = max(cvts) if cvts else None

    if rung == "e3":
        # Mode 2 ritual: CvT essentially 0 throughout, Tools/Q ~ 1
        if peak_cvt is not None and peak_cvt < 0.5:
            return "Mode2-ritual"
        return f"deviates(peak_cvt={peak_cvt})"
    if rung == "e5b":
        # Mode 3 drift: early CvT peak then degrade (later EM < earlier EM)
        if peak_cvt is not None and peak_cvt >= 1.0:
            # check degradation: last EM below max EM
            if ems[-1] < max(ems) - 0.01:
                return "Mode3-drift"
            return "Mode3-peak(no-degrade-yet)"
        return f"deviates(peak_cvt={peak_cvt})"
    if rung == "e5bkl":
        # Stable plateau: EM does NOT collapse. Rising EM is fine (only a downward
        # collapse breaks the plateau). Use min EM > 0.20 as the floor (200-Q
        # Wilson half-width ~3.5pp, so a 3pp range threshold is noise-dominated).
        if min(ems) < 0.05:
            return "collapsed"
        # Stable if it doesn't crash from a healthy level to near-zero, and the
        # late-step EM is not far below the early-step EM (no downward drift).
        if min(ems) > 0.20 and ems[-1] >= max(ems) - 0.05:
            return "stable-plateau"
        return f"variable(em_range={max(ems)-min(ems):.3f})"
    return "unknown"


def main() -> int:
    results = {}
    for rung in RUNGS:
        results[rung] = {"original": ORIGINAL[rung], "seeds": {}}
        for seed in SEEDS:
            curve = {}
            for step in STEPS:
                rec = load_step(rung, seed, step)
                if rec is not None:
                    curve[step] = rec
            if curve:
                mode = assign_mode(rung, curve)
                results[rung]["seeds"][seed] = {"curve": curve, "mode": mode}

    # Pre-registered rule: each rung's signature reproduces in >=2 of 3 seeds
    # (3rd = original). Count new seeds matching the original label family.
    expected = {"e3": "Mode2-ritual", "e5b": "Mode3-drift", "e5bkl": "stable-plateau"}
    verdict_per_rung = {}
    for rung in RUNGS:
        match_new = 0
        total_new = 0
        for seed, blk in results[rung]["seeds"].items():
            total_new += 1
            m = blk["mode"]
            if rung == "e5b" and m.startswith("Mode3"):
                match_new += 1
            elif m == expected[rung]:
                match_new += 1
        # +1 for original (always matches by definition)
        n_match = match_new + 1
        n_total = total_new + 1
        verdict_per_rung[rung] = {
            "matching_seeds": n_match, "total_seeds": n_total,
            "reproduces": n_match >= 2,
            "note": f"{match_new}/{total_new} new seeds + original",
        }

    n_reproduce = sum(1 for v in verdict_per_rung.values() if v["reproduces"])
    if n_reproduce == 3:
        overall = "CONFIRMS (taxonomy seed-robust)"
    elif n_reproduce == 2:
        overall = "PARTIAL (1 rung seed-dependent)"
    else:
        overall = "FALSIFIES (>=2 rungs seed-fragile)"

    payload = {"results": results, "verdict_per_rung": verdict_per_rung,
               "overall_verdict": overall, "data_complete": all(
                   len(results[r]["seeds"].get(s, {}).get("curve", {})) >= 3
                   for r in RUNGS for s in SEEDS)}
    OUT_JSON.write_text(json.dumps(payload, indent=2))

    # Markdown
    md = ["# v20 W21 Part A — multi-seed reward-ladder mode signatures\n"]
    md.append(f"**Overall verdict: {overall}** (data_complete={payload['data_complete']})\n")
    for rung in RUNGS:
        md.append(f"\n## {rung} (original: {ORIGINAL[rung]['label']})\n")
        md.append("| seed | " + " | ".join(f"s{s} EM/CvT%/Tools" for s in STEPS) + " | mode |")
        md.append("|---|" + "|".join("---" for _ in STEPS) + "|---|")
        for seed in SEEDS:
            blk = results[rung]["seeds"].get(seed)
            if not blk:
                md.append(f"| {seed} | (no data) | — |")
                continue
            cells = []
            for step in STEPS:
                c = blk["curve"].get(step)
                if c:
                    cvt = f"{c['cvt_pct']}" if c["cvt_pct"] is not None else "?"
                    cells.append(f"{c['em']:.3f}/{cvt}/{c['tools_per_q']:.2f}")
                else:
                    cells.append("—")
            md.append(f"| {seed} | " + " | ".join(cells) + f" | {blk['mode']} |")
        v = verdict_per_rung[rung]
        md.append(f"\n*Reproduces: {v['reproduces']} ({v['note']}, {v['matching_seeds']}/{v['total_seeds']})*\n")
    OUT_MD.write_text("\n".join(md) + "\n")
    print(f"Overall: {overall}")
    for r, v in verdict_per_rung.items():
        print(f"  {r}: reproduces={v['reproduces']} ({v['note']})")
    print(f"Wrote {OUT_MD}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
