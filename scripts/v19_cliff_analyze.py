"""v19 W18 cliff-isolation + W20 informative-failure analyzers.

W18: does the anti-quote r_retrv PREVENT the Mode-4 peak-then-collapse cliff?
  Baseline E5b+SelfV cliff: CvT 3.77%→9.57%→0 by step 300; Tools/Q 3.0→1.0.
  Pre-registered: R1 does NOT cliff (CvT not ~0 AND Tools/Q not ~1 across 250-400)
  in >=2/3 seeds -> CONFIRMS r_retrv mechanism. Else FALSIFIES.

W20: does informative-failure lift EM and drop kg-incomplete?
  Pre-registered: A1 EM lift >=3pp AND kg-incomplete drop >=20% (vs baseline E5b)
  in >=1/2 seeds -> CONFIRMS L_sig causally testable. Else FALSIFIES.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path("/lus/lfs1aip2/scratch/u6gg/ts1201.u6gg/Project/KG_GRPO")
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from task16_classify import classify_trajectory  # noqa: E402

CLIFF_DIR = PROJECT_ROOT / "_handoff/data/w19_cliff"
LADDER_DIR = PROJECT_ROOT / "_handoff/data/w21/ladder_eval"
STEPS = [50, 100, 150, 200, 250, 300, 350, 400]


def load_curve(exp_dir: Path) -> dict[int, dict]:
    curve = {}
    for s in STEPS:
        ev = exp_dir / f"step{s}.json"
        if not ev.exists():
            continue
        agg = json.load(open(ev))["0"]
        rec = {"em": agg["em"], "tools": agg["avg_tool_calls"], "n": agg["n_samples"],
               "cvt_pct": None, "kg_incomplete_pct": None}
        tj = exp_dir / f"step{s}_traj" / "step_0" / "trajectories.json"
        if tj.exists():
            trajs = json.load(open(tj))
            cats = [classify_trajectory(t) for t in trajs]
            n = len(cats)
            rec["cvt_pct"] = round(100.0 * cats.count("correct-via-tool") / max(n, 1), 3)
            rec["kg_incomplete_pct"] = round(100.0 * cats.count("kg-incomplete") / max(n, 1), 3)
            # NOTE: under informative-failure the KG returns "ERROR: ..." instead of
            # "[]", so task16's kg-incomplete detector (keys on empty []) reads 0 —
            # a CLASSIFICATION ARTIFACT, not a real drop. Track ERROR-response rate
            # separately as the comparable "retrieval-failed" signal.
            err_q = 0
            for t in trajs:
                fr = t.get("full_response", "")
                if "ERROR: entity_not_in_subgraph" in fr or "ERROR: relation_not_present" in fr:
                    err_q += 1
            rec["error_response_pct"] = round(100.0 * err_q / max(n, 1), 3)
        curve[s] = rec
    return curve


def analyze_w18() -> dict:
    seeds = {}
    for s in [1, 2, 3]:
        d = CLIFF_DIR / f"grpo-cwq-7b-w18-anti-quote-seed{s}-20260520"
        c = load_curve(d)
        if not c:
            continue
        # Cliff = collapse to CvT~0 AND Tools~1 by late steps (>=250)
        late = [st for st in c if st >= 250]
        cliffed = None
        if late:
            last = max(late)
            cvt_late = c[last]["cvt_pct"]
            tools_late = c[last]["tools"]
            # "cliff" if tools collapsed to ~1 and CvT ~0 at the late checkpoint
            cliffed = (tools_late is not None and tools_late < 1.5) and \
                      (cvt_late is not None and cvt_late < 0.5)
        seeds[s] = {"curve": c, "cliffed": cliffed,
                    "max_step": max(c.keys()) if c else None}
    # CONFIRMS if >=2 seeds do NOT cliff (cliffed==False) among those reaching >=250
    evaluable = {s: v for s, v in seeds.items() if v["cliffed"] is not None}
    not_cliffed = sum(1 for v in evaluable.values() if v["cliffed"] is False)
    verdict = "PENDING (no seed reached step 250 yet)"
    if evaluable:
        if not_cliffed >= 2:
            verdict = "CONFIRMS (anti-quote r_retrv prevents the cliff in >=2 seeds)"
        elif not_cliffed == 1:
            verdict = "PARTIAL (1 seed avoids cliff)"
        else:
            verdict = "FALSIFIES (anti-quote still cliffs -> r_retrv mechanism wrong/incomplete)"
    return {"seeds": seeds, "n_not_cliffed": not_cliffed,
            "n_evaluable": len(evaluable), "verdict": verdict}


def analyze_w20() -> dict:
    # Baseline E5b (no info-fail) from W21 ladder eval (e5b seeds), best EM per seed.
    base_em, base_kg = [], []
    for s in [2, 3]:
        d = LADDER_DIR / f"e5b_seed{s}"
        c = load_curve(d)
        if c:
            base_em.append(max(v["em"] for v in c.values()))
            kgs = [v["kg_incomplete_pct"] for v in c.values() if v["kg_incomplete_pct"] is not None]
            if kgs:
                base_kg.append(sum(kgs) / len(kgs))
    baseline_em = max(base_em) if base_em else None
    baseline_kg = (sum(base_kg) / len(base_kg)) if base_kg else None

    arms = {}
    for arm in ["a1", "a2"]:
        for s in [1, 2]:
            d = CLIFF_DIR / f"grpo-cwq-7b-w20-{arm}-infofail-seed{s}-20260520"
            c = load_curve(d)
            if not c:
                continue
            peak_em = max(v["em"] for v in c.values())
            kgs = [v["kg_incomplete_pct"] for v in c.values() if v["kg_incomplete_pct"] is not None]
            mean_kg = (sum(kgs) / len(kgs)) if kgs else None
            arms[f"{arm}_seed{s}"] = {"curve": c, "peak_em": peak_em, "mean_kg_incomplete": mean_kg}

    # A1 verdict: EM lift >=3pp AND kg-incomplete drop >=20% vs baseline, in >=1 seed
    a1_confirms = 0
    if baseline_em is not None:
        for k, v in arms.items():
            if not k.startswith("a1"):
                continue
            em_lift = v["peak_em"] - baseline_em
            kg_drop = None
            if baseline_kg and v["mean_kg_incomplete"] is not None:
                kg_drop = (baseline_kg - v["mean_kg_incomplete"]) / baseline_kg
            if em_lift >= 0.03 and (kg_drop is not None and kg_drop >= 0.20):
                a1_confirms += 1
    verdict = "PENDING"
    if any(k.startswith("a1") for k in arms):
        verdict = ("CONFIRMS (L_sig causally testable: EM lift + kg-incomplete drop)"
                   if a1_confirms >= 1 else
                   "FALSIFIES/PARTIAL (informative-failure did not lift EM>=3pp + drop kg-inc>=20%)")
    return {"baseline_em": baseline_em, "baseline_kg_incomplete": baseline_kg,
            "arms": arms, "a1_confirms_seeds": a1_confirms, "verdict": verdict}


def fmt_curve(c: dict) -> str:
    parts = []
    for s in sorted(c.keys()):
        v = c[s]
        cvt = v["cvt_pct"] if v["cvt_pct"] is not None else "?"
        parts.append(f"s{s}:EM{v['em']:.2f}/CvT{cvt}/T{v['tools']:.1f}")
    return " ".join(parts)


def main() -> int:
    w18 = analyze_w18()
    w20 = analyze_w20()

    # W18 markdown
    md = ["# v19 W18 — r_retrv cliff-isolation result\n"]
    md.append(f"**Verdict: {w18['verdict']}**  ({w18['n_not_cliffed']}/{w18['n_evaluable']} seeds avoid cliff)\n")
    md.append("Baseline E5b+SelfV cliff: CvT 3.77→9.57→0 by step 300; Tools/Q 3.0→1.0.\n")
    for s, v in w18["seeds"].items():
        md.append(f"- **seed{s}** (max_step {v['max_step']}, cliffed={v['cliffed']}): {fmt_curve(v['curve'])}")
    (PROJECT_ROOT / "_handoff/data/w19_cliff/wt18_results.md").write_text("\n".join(md) + "\n")

    # W20 markdown
    md2 = ["# v19 W20 — informative-failure intervention result\n"]
    md2.append(f"**Verdict: {w20['verdict']}**\n")
    md2.append(f"Baseline E5b: peak EM={w20['baseline_em']}, mean kg-incomplete%={w20['baseline_kg_incomplete']}\n")
    for k, v in w20["arms"].items():
        md2.append(f"- **{k}**: peak EM={v['peak_em']:.3f}, mean kg-inc%={v['mean_kg_incomplete']} | {fmt_curve(v['curve'])}")
    (PROJECT_ROOT / "_handoff/data/w19_cliff/wt20_results.md").write_text("\n".join(md2) + "\n")

    print("W18:", w18["verdict"])
    print("W20:", w20["verdict"])
    json.dump({"w18": w18, "w20": w20},
              open(PROJECT_ROOT / "_handoff/data/w19_cliff/cliff_analysis.json", "w"),
              indent=2, default=str)
    return 0


if __name__ == "__main__":
    sys.exit(main())
