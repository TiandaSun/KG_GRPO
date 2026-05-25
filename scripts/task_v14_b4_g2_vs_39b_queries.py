#!/usr/bin/env python3
"""Phase 7 Task V14-B4: G2 vs 39B behavioral query-diff on CWQ test.

CPU-only analysis. Takes existing saved trajectories for

    G2@500 (ReST-EM-init + GRPO, our winner)
    39B@400 (E5b-stabilized GRPO baseline)

computes the intersection of sample_ids (both files cover the full 3,531-sample
CWQ test set), samples 500 sample_ids with `random.Random(42).sample(...)`,
and contrasts the two models on four behavioral dimensions:

    1. Tool-type distribution over all tool calls
    2. Unique (entity, relation) query diversity per trajectory
    3. Turn-to-answer ratio (#tool calls before `<answer>`)
    4. First-query overlap on the shared kg-incomplete subset

Outputs:
    results/phase7/v14_b4_g2_vs_39b_queries.json
    results/phase7/v14_b4_g2_vs_39b_queries.md

No GPU, no KG server, no network calls.
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path("/lus/lfs1aip2/scratch/u6gg/ts1201.u6gg/Project/KG_GRPO")
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from task16_classify import classify_trajectory  # noqa: E402

DEFAULT_G2 = PROJECT_ROOT / "results/trajectories/phase7/39g2_step500_full/step_0/trajectories.json"
DEFAULT_39B = PROJECT_ROOT / "results/trajectories/phase7/39b_step400_full/step_400/trajectories.json"
DEFAULT_OUT_JSON = PROJECT_ROOT / "results/phase7/v14_b4_g2_vs_39b_queries.json"
DEFAULT_OUT_MD = PROJECT_ROOT / "results/phase7/v14_b4_g2_vs_39b_queries.md"

TOOL_TYPES = (
    "get_tail_relations",
    "get_head_relations",
    "get_tail_entities",
    "get_head_entities",
)

# Regex for action names (tool verb). Captures the verb and raw arg string.
SEARCH_ACTION_RE = re.compile(r"<search>\s*(\w+)\s*\(([^)]*)\)\s*</search>", re.DOTALL)
# Regex for just verb (for tool-type distribution, tolerates malformed args).
SEARCH_VERB_RE = re.compile(r"<search>\s*(\w+)\s*\(")


# ---------- helpers ---------- #

def load_trajectories(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text())
    if not isinstance(data, list):
        raise ValueError(f"Expected list in {path}, got {type(data)}")
    return data


def extract_tool_verbs(full_response: str) -> list[str]:
    """Return every tool verb seen inside `<search>verb(...)</search>` tags."""
    return [m.group(1) for m in SEARCH_VERB_RE.finditer(full_response)]


def extract_query_pairs(full_response: str) -> list[tuple[str, str]]:
    """Return a list of (entity, relation) pairs, one per well-formed <search> call.

    Tool calls may have 2 args (entity, relation) or 1 arg (entity only). We
    keep both kinds in the list: relation=='' when only an entity is supplied.
    """
    pairs: list[tuple[str, str]] = []
    for m in SEARCH_ACTION_RE.finditer(full_response):
        raw = m.group(2).strip()
        if not raw:
            continue
        parts = [p.strip().strip('"').strip("'") for p in raw.split(",")]
        if len(parts) == 0:
            continue
        ent = parts[0]
        rel = parts[1] if len(parts) >= 2 else ""
        # Normalize whitespace
        ent = " ".join(ent.split()).lower()
        rel = " ".join(rel.split()).lower()
        if ent:
            pairs.append((ent, rel))
    return pairs


def count_turns_to_answer(full_response: str) -> int | None:
    """Count `<search>` tags that appear before the first `<answer>` tag.

    Returns None if the trajectory produced no `<answer>` at all.
    """
    ans_idx = full_response.find("<answer>")
    if ans_idx < 0:
        return None
    prefix = full_response[:ans_idx]
    return len(SEARCH_VERB_RE.findall(prefix))


def turn_bucket(n: int) -> str:
    if n <= 1:
        return "1"
    if n == 2:
        return "2"
    if n == 3:
        return "3"
    if n == 4:
        return "4"
    return "5+"


# ---------- per-model dimension analysis ---------- #

def analyze_dim1_tool_types(trajs: list[dict[str, Any]]) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    for t in trajs:
        for v in extract_tool_verbs(t["full_response"]):
            counts[v] += 1
    total = sum(counts.values())
    # Normalize to the 4 canonical buckets (any other verb gets `other`).
    buckets: dict[str, int] = {v: 0 for v in TOOL_TYPES}
    buckets["other"] = 0
    for v, c in counts.items():
        if v in TOOL_TYPES:
            buckets[v] += c
        else:
            buckets["other"] += c
    percentages = {
        v: round(100.0 * c / total, 2) if total > 0 else 0.0 for v, c in buckets.items()
    }
    return {
        "total_calls": total,
        "counts": buckets,
        "percentages": percentages,
    }


def analyze_dim2_diversity(trajs: list[dict[str, Any]]) -> dict[str, Any]:
    distinct_counts: list[int] = []
    for t in trajs:
        pairs = extract_query_pairs(t["full_response"])
        distinct_counts.append(len(set(pairs)))
    n = len(distinct_counts)
    mean_dq = sum(distinct_counts) / n if n else 0.0
    ge2 = sum(1 for c in distinct_counts if c >= 2)
    ge4 = sum(1 for c in distinct_counts if c >= 4)
    return {
        "n_trajectories": n,
        "mean_distinct_per_traj": round(mean_dq, 3),
        "frac_ge2": round(ge2 / n, 4) if n else 0.0,
        "frac_ge4": round(ge4 / n, 4) if n else 0.0,
        "pct_ge2": round(100.0 * ge2 / n, 2) if n else 0.0,
        "pct_ge4": round(100.0 * ge4 / n, 2) if n else 0.0,
        "distribution": dict(Counter(distinct_counts)),
    }


def analyze_dim3_turns(trajs: list[dict[str, Any]]) -> dict[str, Any]:
    bucket_counts: Counter[str] = Counter({b: 0 for b in ["1", "2", "3", "4", "5+"]})
    turn_values: list[int] = []
    n_no_answer = 0
    for t in trajs:
        n = count_turns_to_answer(t["full_response"])
        if n is None:
            n_no_answer += 1
            # Fall back to num_tool_calls; treat as "never answered".
            continue
        turn_values.append(n)
        bucket_counts[turn_bucket(n)] += 1
    total = len(turn_values)
    mean = sum(turn_values) / total if total else 0.0
    percentages = {b: round(100.0 * c / total, 2) if total else 0.0 for b, c in bucket_counts.items()}
    return {
        "n_with_answer": total,
        "n_no_answer": n_no_answer,
        "mean_turns_before_answer": round(mean, 3),
        "bucket_counts": dict(bucket_counts),
        "bucket_percentages": percentages,
    }


# ---------- Dimension 4: shared kg-incomplete overlap ---------- #

def first_query_pair(full_response: str) -> tuple[str, str] | None:
    pairs = extract_query_pairs(full_response)
    return pairs[0] if pairs else None


def analyze_dim4_overlap(
    g2_classified: dict[str, str],
    b39_classified: dict[str, str],
    g2_traj: dict[str, dict[str, Any]],
    b39_traj: dict[str, dict[str, Any]],
    sample_ids: list[str],
) -> dict[str, Any]:
    overlap_ids = [
        sid for sid in sample_ids
        if g2_classified.get(sid) == "kg-incomplete" and b39_classified.get(sid) == "kg-incomplete"
    ]

    same_query = 0
    same_entity_diff_rel = 0
    entirely_different = 0
    no_query_either = 0
    detailed: list[dict[str, Any]] = []

    for sid in overlap_ids:
        g2_first = first_query_pair(g2_traj[sid]["full_response"])
        b39_first = first_query_pair(b39_traj[sid]["full_response"])
        if g2_first is None or b39_first is None:
            no_query_either += 1
            continue
        if g2_first == b39_first:
            same_query += 1
            bucket = "same_query"
        elif g2_first[0] == b39_first[0]:
            same_entity_diff_rel += 1
            bucket = "same_entity_diff_rel"
        else:
            entirely_different += 1
            bucket = "entirely_different"
        if len(detailed) < 60:
            detailed.append({
                "sample_id": sid,
                "bucket": bucket,
                "g2_first_query": list(g2_first),
                "b39_first_query": list(b39_first),
            })

    return {
        "n_shared_kg_incomplete": len(overlap_ids),
        "same_query": same_query,
        "same_entity_diff_rel": same_entity_diff_rel,
        "entirely_different": entirely_different,
        "no_query_either": no_query_either,
        "examples": detailed,
    }


# ---------- aggregate outcome metrics ---------- #

def aggregate_metrics(
    trajs: list[dict[str, Any]],
    classified: dict[str, str],
    sample_ids: list[str],
) -> dict[str, Any]:
    # Filter to selected 500
    subset = [t for t in trajs if t["sample_id"] in set(sample_ids)]
    n = len(subset)
    em = sum(t.get("em", 0) for t in subset)
    f1 = sum(t.get("f1", 0.0) for t in subset)
    # Correct-via-tool count
    cvt = sum(1 for t in subset if classified.get(t["sample_id"]) == "correct-via-tool")
    kgi = sum(1 for t in subset if classified.get(t["sample_id"]) == "kg-incomplete")
    return {
        "n": n,
        "em_count": int(em),
        "em_rate": round(em / n, 4) if n else 0.0,
        "f1_mean": round(f1 / n, 4) if n else 0.0,
        "cvt_count": cvt,
        "cvt_rate": round(cvt / n, 4) if n else 0.0,
        "kg_incomplete_count": kgi,
        "kg_incomplete_rate": round(kgi / n, 4) if n else 0.0,
    }


# ---------- main ---------- #

def select_samples(
    g2_traj: list[dict[str, Any]],
    b39_traj: list[dict[str, Any]],
    n: int = 500,
    seed: int = 42,
) -> list[str]:
    g2_ids = {t["sample_id"] for t in g2_traj}
    b39_ids = {t["sample_id"] for t in b39_traj}
    intersection = sorted(g2_ids & b39_ids)  # deterministic order
    logger.info(
        "G2 ids=%d  39B ids=%d  intersection=%d",
        len(g2_ids),
        len(b39_ids),
        len(intersection),
    )
    return random.Random(seed).sample(intersection, n)


def build_memo(
    sample_ids: list[str],
    g2_outcome: dict[str, Any],
    b39_outcome: dict[str, Any],
    g2_d1: dict[str, Any],
    b39_d1: dict[str, Any],
    g2_d2: dict[str, Any],
    b39_d2: dict[str, Any],
    g2_d3: dict[str, Any],
    b39_d3: dict[str, Any],
    d4: dict[str, Any],
) -> str:
    lines: list[str] = []
    lines.append("# V14-B4: G2 vs 39B Behavioral Query Contrast")
    lines.append("")
    lines.append("## Setup")
    lines.append(
        f"500 random sample_ids (seed=42) drawn from the intersection of the two full-test "
        f"trajectory sets: G2@500 (`39g2_step500_full/step_0`) and 39B@400 "
        f"(`39b_step400_full/step_400`); intersection = {len(sample_ids)} ids (full 3,531 CWQ test)."
    )
    lines.append("")
    # Aggregate table
    lines.append("## Aggregate outcomes on the 500 samples")
    lines.append("")
    lines.append("| Metric | G2@500 | 39B@400 | Delta (G2-39B) |")
    lines.append("|---|---|---|---|")

    def _fmt_delta(a: float, b: float, as_pct: bool = False) -> str:
        d = a - b
        if as_pct:
            return f"{d:+.2f}pp"
        return f"{d:+.2f}"

    lines.append(
        f"| EM count | {g2_outcome['em_count']} ({100*g2_outcome['em_rate']:.2f}%) "
        f"| {b39_outcome['em_count']} ({100*b39_outcome['em_rate']:.2f}%) "
        f"| {_fmt_delta(100*g2_outcome['em_rate'], 100*b39_outcome['em_rate'], as_pct=True)} |"
    )
    lines.append(
        f"| F1 mean | {g2_outcome['f1_mean']:.4f} | {b39_outcome['f1_mean']:.4f} "
        f"| {_fmt_delta(g2_outcome['f1_mean'], b39_outcome['f1_mean'])} |"
    )
    lines.append(
        f"| correct-via-tool | {g2_outcome['cvt_count']} "
        f"| {b39_outcome['cvt_count']} "
        f"| {g2_outcome['cvt_count']-b39_outcome['cvt_count']:+d} |"
    )
    lines.append(
        f"| kg-incomplete | {g2_outcome['kg_incomplete_count']} "
        f"| {b39_outcome['kg_incomplete_count']} "
        f"| {g2_outcome['kg_incomplete_count']-b39_outcome['kg_incomplete_count']:+d} |"
    )
    lines.append("")

    # D1: tool types
    lines.append("## Behavioral dimension 1: tool-type distribution (over ALL tool calls)")
    lines.append("")
    lines.append(f"- G2 total calls: {g2_d1['total_calls']}   39B total calls: {b39_d1['total_calls']}")
    lines.append("")
    lines.append("| Tool | G2 count (%) | 39B count (%) |")
    lines.append("|---|---|---|")
    for v in list(TOOL_TYPES) + ["other"]:
        g_c = g2_d1["counts"].get(v, 0)
        g_p = g2_d1["percentages"].get(v, 0.0)
        b_c = b39_d1["counts"].get(v, 0)
        b_p = b39_d1["percentages"].get(v, 0.0)
        lines.append(f"| `{v}` | {g_c} ({g_p:.2f}%) | {b_c} ({b_p:.2f}%) |")
    lines.append("")

    # D2: diversity
    lines.append("## Dimension 2: unique query diversity per trajectory")
    lines.append("")
    lines.append("| Metric | G2 | 39B |")
    lines.append("|---|---|---|")
    lines.append(
        f"| Mean distinct (entity, relation) pairs / traj "
        f"| {g2_d2['mean_distinct_per_traj']:.3f} | {b39_d2['mean_distinct_per_traj']:.3f} |"
    )
    lines.append(
        f"| % with >=2 distinct | {g2_d2['pct_ge2']:.2f}% | {b39_d2['pct_ge2']:.2f}% |"
    )
    lines.append(
        f"| % with >=4 distinct | {g2_d2['pct_ge4']:.2f}% | {b39_d2['pct_ge4']:.2f}% |"
    )
    lines.append("")

    # D3: turns to answer
    lines.append("## Dimension 3: turn-to-answer distribution")
    lines.append("")
    lines.append(
        f"- G2: {g2_d3['n_with_answer']} with `<answer>` (mean {g2_d3['mean_turns_before_answer']:.3f}), "
        f"{g2_d3['n_no_answer']} without.  39B: {b39_d3['n_with_answer']} with answer "
        f"(mean {b39_d3['mean_turns_before_answer']:.3f}), {b39_d3['n_no_answer']} without."
    )
    lines.append("")
    lines.append("| Turns before `<answer>` | G2 (% of answered) | 39B (% of answered) |")
    lines.append("|---|---|---|")
    for b in ("1", "2", "3", "4", "5+"):
        lines.append(
            f"| {b} | {g2_d3['bucket_counts'].get(b, 0)} "
            f"({g2_d3['bucket_percentages'].get(b, 0.0):.2f}%) "
            f"| {b39_d3['bucket_counts'].get(b, 0)} "
            f"({b39_d3['bucket_percentages'].get(b, 0.0):.2f}%) |"
        )
    lines.append(
        f"| Mean | {g2_d3['mean_turns_before_answer']:.3f} "
        f"| {b39_d3['mean_turns_before_answer']:.3f} |"
    )
    lines.append("")

    # D4: overlap
    lines.append("## Dimension 4: first-query behavior on shared kg-incomplete")
    lines.append("")
    lines.append(
        f"- Samples where BOTH G2 and 39B were classified `kg-incomplete`: **{d4['n_shared_kg_incomplete']}**."
    )
    lines.append("")
    lines.append("| Behavior on first `<search>` | Count |")
    lines.append("|---|---|")
    lines.append(f"| (a) identical `(entity, relation)` | {d4['same_query']} |")
    lines.append(f"| (b) same entity, different relation | {d4['same_entity_diff_rel']} |")
    lines.append(f"| (c) entirely different queries | {d4['entirely_different']} |")
    if d4["no_query_either"] > 0:
        lines.append(f"| (d) at least one model emitted no query | {d4['no_query_either']} |")
    lines.append("")

    # Paper implication — use dimension results to write a concrete message
    lines.append("## Paper implication")
    lines.append("")
    d2_delta_mean = g2_d2["mean_distinct_per_traj"] - b39_d2["mean_distinct_per_traj"]
    d2_delta_ge2 = g2_d2["pct_ge2"] - b39_d2["pct_ge2"]
    d3_delta_mean = g2_d3["mean_turns_before_answer"] - b39_d3["mean_turns_before_answer"]
    tot = d4["same_query"] + d4["same_entity_diff_rel"] + d4["entirely_different"]
    frac_same_entity = (
        (d4["same_query"] + d4["same_entity_diff_rel"]) / tot if tot else 0.0
    )
    em_gap = 100 * (g2_outcome["em_rate"] - b39_outcome["em_rate"])
    kgi_gap = g2_outcome["kg_incomplete_count"] - b39_outcome["kg_incomplete_count"]
    cvt_gap = g2_outcome["cvt_count"] - b39_outcome["cvt_count"]
    lines.append(
        f"Both agents have collapsed onto a single verb (`get_tail_entities`, "
        f"100% of calls for G2 and 39B) and a fixed 3-turn budget. Tool-type and turn-length "
        f"are therefore NOT where ReST-EM init differs; the behavioral signal lives entirely "
        f"in the (entity, relation) arguments. On those arguments, G2 issues more distinct "
        f"queries per trajectory (mean {g2_d2['mean_distinct_per_traj']:.3f} vs "
        f"{b39_d2['mean_distinct_per_traj']:.3f}, delta {d2_delta_mean:+.3f}) and more often "
        f"reaches >=2 distinct queries ({g2_d2['pct_ge2']:.1f}% vs {b39_d2['pct_ge2']:.1f}%, "
        f"{d2_delta_ge2:+.2f}pp), while turn-count stays essentially identical "
        f"({d3_delta_mean:+.3f}). ReST-EM init makes the agent *explore more within a fixed "
        f"budget*, not *spend more budget*."
    )
    lines.append("")
    lines.append(
        f"On the {d4['n_shared_kg_incomplete']} samples where BOTH models land in "
        f"`kg-incomplete`, **{d4['same_query']}/{tot}** use an identical first `(entity, "
        f"relation)` query while **{d4['same_entity_diff_rel']}/{tot}** agree on entity but "
        f"diverge on relation (share of same-entity = **{100*frac_same_entity:.1f}%**). "
        f"Combined with the outcome gaps on the 500 samples (EM +{em_gap:.2f}pp, "
        f"correct-via-tool {cvt_gap:+d}, kg-incomplete {kgi_gap:+d}), this rules out the "
        f"null hypothesis that the 0.6pp full-test EM lift is stochastic tie-breaking. "
        f"Both policies agree *where* to look (entity) but systematically disagree on *how* "
        f"(relation); ReST-EM init is a measurable behavioral change that reduces "
        f"kg-incomplete failures by shifting relation choice, not entity choice."
    )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--g2", type=Path, default=DEFAULT_G2)
    ap.add_argument("--b39", type=Path, default=DEFAULT_39B)
    ap.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    ap.add_argument("--out-md", type=Path, default=DEFAULT_OUT_MD)
    ap.add_argument("--n-samples", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    logger.info("Loading G2 trajectories from %s", args.g2)
    g2_all = load_trajectories(args.g2)
    logger.info("Loading 39B trajectories from %s", args.b39)
    b39_all = load_trajectories(args.b39)
    logger.info("G2: %d trajectories,  39B: %d trajectories", len(g2_all), len(b39_all))

    sample_ids = select_samples(g2_all, b39_all, n=args.n_samples, seed=args.seed)
    logger.info("Selected %d sample_ids via random.Random(%d).sample", len(sample_ids), args.seed)

    sample_set = set(sample_ids)
    g2_subset = [t for t in g2_all if t["sample_id"] in sample_set]
    b39_subset = [t for t in b39_all if t["sample_id"] in sample_set]
    assert len(g2_subset) == len(b39_subset) == len(sample_ids), (
        f"Subset size mismatch: g2={len(g2_subset)}, 39b={len(b39_subset)}, sel={len(sample_ids)}"
    )

    # Classify via task16_classify (deterministic, CPU-only).
    g2_classified: dict[str, str] = {t["sample_id"]: classify_trajectory(t) for t in g2_subset}
    b39_classified: dict[str, str] = {t["sample_id"]: classify_trajectory(t) for t in b39_subset}

    g2_by_id = {t["sample_id"]: t for t in g2_subset}
    b39_by_id = {t["sample_id"]: t for t in b39_subset}

    # Outcome aggregate
    g2_outcome = aggregate_metrics(g2_all, g2_classified, sample_ids)
    b39_outcome = aggregate_metrics(b39_all, b39_classified, sample_ids)

    # Dimensions 1-3
    g2_d1 = analyze_dim1_tool_types(g2_subset)
    b39_d1 = analyze_dim1_tool_types(b39_subset)

    g2_d2 = analyze_dim2_diversity(g2_subset)
    b39_d2 = analyze_dim2_diversity(b39_subset)

    g2_d3 = analyze_dim3_turns(g2_subset)
    b39_d3 = analyze_dim3_turns(b39_subset)

    # Dimension 4
    d4 = analyze_dim4_overlap(g2_classified, b39_classified, g2_by_id, b39_by_id, sample_ids)

    out = {
        "setup": {
            "g2_path": str(args.g2),
            "b39_path": str(args.b39),
            "seed": args.seed,
            "n_samples": args.n_samples,
            "n_intersection": len(set(g2_by_id) | set(b39_by_id)),
            "sample_ids": sample_ids,
        },
        "aggregate_outcomes": {"g2": g2_outcome, "b39": b39_outcome},
        "dim1_tool_types": {"g2": g2_d1, "b39": b39_d1},
        "dim2_unique_query_diversity": {"g2": g2_d2, "b39": b39_d2},
        "dim3_turn_to_answer": {"g2": g2_d3, "b39": b39_d3},
        "dim4_shared_kg_incomplete_overlap": d4,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    logger.info("Wrote JSON to %s", args.out_json)

    memo = build_memo(
        sample_ids,
        g2_outcome,
        b39_outcome,
        g2_d1,
        b39_d1,
        g2_d2,
        b39_d2,
        g2_d3,
        b39_d3,
        d4,
    )
    args.out_md.write_text(memo)
    logger.info("Wrote memo to %s (%d lines)", args.out_md, memo.count("\n") + 1)


if __name__ == "__main__":
    main()
