#!/usr/bin/env python3
"""S0.6: V14-B2 robustness check via seen-vs-unseen relation split.

Re-runs the V14-B2 decision-boundary analysis (G2@500 kg-incomplete
trajectories, edit-distance buckets vs. gold relations) WITHIN two
disjoint subsets of the CWQ test set:

* ``seen``    — every gold relation in the question's ``extra_info.kg_path``
                appears in R_SFT (the set of relation arguments observed in
                ``data/freebase/sft_trajectories.jsonl``).
* ``unseen``  — at least one gold relation is NOT in R_SFT.

The robustness question: is the V14-B2 finding (~70% of G2 kg-incomplete
trajectories use a relation within Levenshtein <=1 of a gold relation,
i.e. ``relation_typo`` bucket) structural, or an SFT-memorisation artifact?
If the bucket distribution holds within ``unseen``, the finding is robust.

Outputs:
  results/phase7/v14_b2_seen_vs_unseen.json
  results/phase7/v14_b2_seen_vs_unseen.md

CPU-only. No KG server, no GPU, no network.
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

PROJECT_ROOT = Path("/lus/lfs1aip2/scratch/u6gg/ts1201.u6gg/Project/KG_GRPO")
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

# Reuse exact V14-B2 logic for bucketing.
from task16_classify import classify_trajectory  # noqa: E402
from task_v14_b2_decision_boundary import (  # noqa: E402
    BUCKETS,
    BUCKET_LABEL,
    build_gold_paths,
    classify_failure,
)

logger = logging.getLogger(__name__)

DEFAULT_TRAJECTORIES = PROJECT_ROOT / "results/trajectories/phase7/39g2_step500_full/step_0/trajectories.json"
DEFAULT_PARQUET = PROJECT_ROOT / "data/freebase/verl_cwq/test.parquet"
DEFAULT_SFT = PROJECT_ROOT / "data/freebase/sft_trajectories.jsonl"
DEFAULT_OUT_JSON = PROJECT_ROOT / "results/phase7/v14_b2_seen_vs_unseen.json"
DEFAULT_OUT_MD = PROJECT_ROOT / "results/phase7/v14_b2_seen_vs_unseen.md"

# Same regex shape as V14-B2 — captures `<search>fn(args)</search>` tool calls.
SEARCH_RE = re.compile(r"<search>\s*(\w+)\s*\(([^)]*)\)\s*</search>")
# Tool-response payload (so we can also collect the "exposed" relation set).
TOOL_RESP_RE = re.compile(r"<tool_response>(.*?)</tool_response>", re.DOTALL)

# SFT tool functions whose 2nd argument is a relation. (3-arg verify is not
# present in SFT, but listed defensively.)
TOOL_FNS_WITH_RELATION_ARG = {
    "get_tail_entities",
    "get_head_entities",
    "get_tails",
    "get_heads",
    "search",
    "search_reverse",
    "verify",
}
# Tool functions whose tool_response contains a list of relations.
TOOL_FNS_RELATION_LISTING = {
    "get_tail_relations",
    "get_head_relations",
    "get_relations",
}


def normalize_relation_for_membership(rel: str) -> str:
    """R_SFT membership uses raw lowercase strings (no whitespace tricks)."""
    if rel is None:
        return ""
    return str(rel).strip().lower()


def parse_args_field(args_str: str) -> list[str]:
    return [p.strip() for p in args_str.split(",")] if args_str else []


def build_r_sft(sft_path: Path) -> tuple[set[str], set[str], dict[str, int]]:
    """Walk SFT trajectories, extract R_SFT.

    Returns
    -------
    r_sft_args : set[str]
        Relations that appeared as ARGUMENTS to tool calls (strict R_SFT
        per the brief — "relations USED by these tool calls").
    r_sft_exposed : set[str]
        Union of ``r_sft_args`` and relations that appeared inside
        ``<tool_response>`` payloads of ``get_tail_relations`` /
        ``get_head_relations`` calls (a strictly larger "model has seen
        this relation token at all" set, kept as auxiliary diagnostic).
    stats : dict[str, int]
        Diagnostic counts (n_trajectories, n_calls_with_rel_arg, etc.).
    """
    r_sft_args: set[str] = set()
    r_sft_exposed: set[str] = set()
    n_traj = 0
    n_calls_with_rel_arg = 0
    n_listing_calls = 0

    with sft_path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            n_traj += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                logger.warning("Skipping malformed SFT line %d", n_traj)
                continue
            traj = row.get("trajectory") or []
            # Two-pass walk: track the function of the most-recent assistant
            # tool call so the next tool_response can be associated with it.
            last_fn: str | None = None
            for msg in traj:
                role = msg.get("role")
                content = msg.get("content") or ""
                if role == "assistant":
                    for fn, args in SEARCH_RE.findall(content):
                        last_fn = fn.strip()
                        if last_fn in TOOL_FNS_WITH_RELATION_ARG:
                            parts = parse_args_field(args)
                            if len(parts) >= 2:
                                rel = normalize_relation_for_membership(parts[1])
                                if rel:
                                    r_sft_args.add(rel)
                                    n_calls_with_rel_arg += 1
                elif role == "user":
                    # Tool-response. Only meaningful when it follows a
                    # listing call.
                    if last_fn in TOOL_FNS_RELATION_LISTING:
                        for resp_match in TOOL_RESP_RE.findall(content):
                            try:
                                payload = json.loads(resp_match)
                            except json.JSONDecodeError:
                                continue
                            if isinstance(payload, list):
                                n_listing_calls += 1
                                for rel in payload:
                                    rel_n = normalize_relation_for_membership(rel)
                                    if rel_n:
                                        r_sft_exposed.add(rel_n)
                    # Reset last_fn — one tool_response is consumed.
                    last_fn = None

    r_sft_exposed = r_sft_exposed | r_sft_args
    stats = {
        "n_sft_trajectories": n_traj,
        "n_calls_with_relation_arg": n_calls_with_rel_arg,
        "n_relation_listing_calls": n_listing_calls,
        "n_r_sft_args": len(r_sft_args),
        "n_r_sft_exposed": len(r_sft_exposed),
    }
    return r_sft_args, r_sft_exposed, stats


def gold_relations_for_sample(kg_path: list[list[str]]) -> set[str]:
    rels = set()
    for triple in kg_path:
        if len(triple) == 3:
            rels.add(normalize_relation_for_membership(triple[1]))
    rels.discard("")
    return rels


def partition_samples(
    sample_ids: list[str],
    gold_paths: dict[str, list[list[str]]],
    r_sft: set[str],
) -> dict[str, dict[str, set[str] | None]]:
    """Split sample_ids by seen/unseen relative to ``r_sft``.

    Returns mapping ``sid -> {"label": "seen"/"unseen"/"no_gold", "gold_rels": set, "missing": set}``.
    """
    out: dict[str, dict[str, Any]] = {}
    for sid in sample_ids:
        gold = gold_relations_for_sample(gold_paths.get(sid, []))
        if not gold:
            out[sid] = {"label": "no_gold", "gold_rels": gold, "missing": set()}
            continue
        missing = {g for g in gold if g not in r_sft}
        label = "unseen" if missing else "seen"
        out[sid] = {"label": label, "gold_rels": gold, "missing": missing}
    return out


def bucket_distribution(outcomes: list) -> tuple[Counter, dict[str, float]]:
    counts = Counter(o.bucket for o in outcomes)
    total = len(outcomes)
    pct = {
        b: round(100 * counts.get(b, 0) / total, 2) if total else 0.0
        for b in BUCKETS
    }
    return counts, pct


def write_markdown(
    out_path: Path,
    base_total: int,
    overall_counts: Counter,
    overall_pct: dict[str, float],
    splits: dict[str, dict[str, Any]],
    sft_stats: dict[str, int],
    seen_def: str,
) -> None:
    def fmt_row(label: str, counts: Counter, total: int) -> str:
        cells = [f"{counts.get(b, 0)} ({(100*counts.get(b,0)/total):.1f}%)" if total else "n/a" for b in BUCKETS]
        return f"| {label} (n={total}) | " + " | ".join(cells) + " |"

    lines: list[str] = []
    lines.append("# S0.6: V14-B2 robustness — seen vs. unseen SFT relations\n")
    lines.append("## Setup")
    lines.append(
        f"R_SFT built from `data/freebase/sft_trajectories.jsonl` "
        f"({sft_stats['n_sft_trajectories']} trajectories, "
        f"{sft_stats['n_calls_with_relation_arg']} tool calls with a relation argument); "
        f"|R_SFT (argument)| = {sft_stats['n_r_sft_args']}, "
        f"|R_SFT (argument ∪ tool-response listings)| = {sft_stats['n_r_sft_exposed']}.\n"
    )
    lines.append(
        "Each CWQ test question is labelled by its gold `extra_info.kg_path` "
        "relations relative to R_SFT (argument set, primary)."
    )
    lines.append(f"Definition used: **{seen_def}**\n")
    lines.append("## V14-B2 bucket distribution within each subset\n")
    lines.append(
        "| Subset | "
        + " | ".join(BUCKET_LABEL[b] for b in BUCKETS)
        + " |"
    )
    lines.append("|" + "---|" * (len(BUCKETS) + 1))
    lines.append(fmt_row("Overall (V14-B2)", overall_counts, base_total))
    for split_name in ("seen", "unseen"):
        s = splits[split_name]
        counts = Counter({b: s["bucket_counts"][b] for b in BUCKETS})
        total = s["n_kg_incomplete"]
        lines.append(fmt_row(split_name, counts, total))
    lines.append("")
    seen = splits["seen"]
    unseen = splits["unseen"]
    seen_typo = seen["bucket_pct"].get("relation_typo", 0.0)
    unseen_typo = unseen["bucket_pct"].get("relation_typo", 0.0)
    seen_typo_or_close = seen_typo  # bucket 1 already = edit-distance <=1 framing in V14-B2
    unseen_typo_or_close = unseen_typo
    lines.append(
        f"- relation_typo (edit <=1) on **seen**:   "
        f"{seen['bucket_counts']['relation_typo']}/{seen['n_kg_incomplete']} = "
        f"**{seen_typo:.1f}%**"
    )
    lines.append(
        f"- relation_typo (edit <=1) on **unseen**: "
        f"{unseen['bucket_counts']['relation_typo']}/{unseen['n_kg_incomplete']} = "
        f"**{unseen_typo:.1f}%**"
    )
    lines.append("")
    lines.append("## Subset sizes (CWQ test questions)\n")
    lines.append("| Subset | n_questions | n_g2_trajectories | n_kg_incomplete |")
    lines.append("|---|---|---|---|")
    for split_name in ("seen", "unseen", "no_gold"):
        s = splits.get(split_name, {})
        lines.append(
            f"| {split_name} | {s.get('n_questions', 0)} | "
            f"{s.get('n_g2_trajectories', 0)} | {s.get('n_kg_incomplete', 0)} |"
        )
    lines.append("")
    lines.append("## Verdict\n")
    THRESH = 30.0  # the V14-B2 publishable claim is ">=30% edit <=1"
    seen_holds = seen_typo >= THRESH
    unseen_holds = unseen_typo >= THRESH
    if seen_holds and unseen_holds:
        direction = (
            "Notably, the rate is HIGHER on unseen than seen — the opposite "
            "of what an SFT-memorisation artifact would predict. "
            if unseen_typo > seen_typo
            else (
                "(seen rate exceeds unseen, but both clear the 30% threshold "
                "so the headline claim still survives.) "
                if seen_typo > unseen_typo
                else ""
            )
        )
        verdict = (
            "**Robust.** The V14-B2 finding holds within BOTH the seen and "
            "unseen SFT-relation subsets — relation_typo (<=1 edit) accounts "
            f"for {seen_typo:.1f}% on seen and {unseen_typo:.1f}% on unseen "
            f"kg-incomplete trajectories. {direction}"
            "The close-edit-distance pattern is structural (the model "
            "parameterises Freebase-style relation tokens at the surface "
            "level) rather than an SFT-memorisation artifact. **Ship V14-B2 "
            "as-is**; no caveat required."
        )
    elif seen_holds and not unseen_holds:
        verdict = (
            "**Partial robustness — SFT-memorisation caveat needed.** "
            f"relation_typo holds on seen ({seen_typo:.1f}%) but drops below "
            f"the 30% threshold on unseen ({unseen_typo:.1f}%). The 70.4% "
            "headline number is partly driven by relations the model already "
            "saw during SFT. **Add a caveat to the paper**: '70.4% covers "
            "the full test split; restricted to questions whose gold "
            "relations were unseen during SFT, the rate is "
            f"{unseen_typo:.1f}%.'"
        )
    elif unseen_holds and not seen_holds:
        verdict = (
            "**Anomalous.** The pattern only holds on the unseen subset "
            f"({unseen_typo:.1f}% vs {seen_typo:.1f}% on seen). This is "
            "unexpected and suggests the model is exploring novel relations; "
            "investigate before publishing."
        )
    else:
        verdict = (
            f"**Does not hold.** Both subsets fall below the 30% threshold "
            f"(seen={seen_typo:.1f}%, unseen={unseen_typo:.1f}%). The V14-B2 "
            "headline number is not robust to seen/unseen partitioning."
        )
    lines.append(verdict + "\n")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines).rstrip() + "\n")


def run(
    trajectories_path: Path,
    parquet_path: Path,
    sft_path: Path,
    output_json: Path,
    output_md: Path,
) -> dict[str, Any]:
    logger.info("Loading G2 trajectories: %s", trajectories_path)
    trajs = json.loads(trajectories_path.read_text())
    logger.info("Loaded %d trajectories", len(trajs))

    logger.info("Loading gold paths: %s", parquet_path)
    gold_paths = build_gold_paths(parquet_path)
    logger.info("Loaded gold paths for %d samples", len(gold_paths))

    logger.info("Building R_SFT from: %s", sft_path)
    r_sft_args, r_sft_exposed, sft_stats = build_r_sft(sft_path)
    logger.info("R_SFT stats: %s", sft_stats)

    # Run V14-B2 classification on all kg-incomplete G2 trajectories ONCE.
    cat_counts: Counter = Counter()
    kg_incomplete: list[dict] = []
    for t in trajs:
        cat = classify_trajectory(t)
        cat_counts[cat] += 1
        if cat == "kg-incomplete":
            kg_incomplete.append(t)
    logger.info("G2 category counts: %s", dict(cat_counts))
    logger.info("kg-incomplete: %d", len(kg_incomplete))

    outcomes_all = [classify_failure(t, gold_paths) for t in kg_incomplete]
    overall_counts, overall_pct = bucket_distribution(outcomes_all)
    logger.info("Overall bucket counts: %s", dict(overall_counts))

    # Partition test samples by seen/unseen using R_SFT (argument-based,
    # primary) and report both definitions for transparency.
    all_sample_ids = list(gold_paths.keys())
    label_args = partition_samples(all_sample_ids, gold_paths, r_sft_args)
    label_exposed = partition_samples(all_sample_ids, gold_paths, r_sft_exposed)

    # Tally subset sizes at the question level.
    def question_counts(labels: dict[str, dict[str, Any]]) -> dict[str, int]:
        c: Counter = Counter()
        for v in labels.values():
            c[v["label"]] += 1
        return dict(c)

    q_counts_args = question_counts(label_args)
    q_counts_exposed = question_counts(label_exposed)
    logger.info("Question split (R_SFT args): %s", q_counts_args)
    logger.info("Question split (R_SFT exposed): %s", q_counts_exposed)

    # Compute G2 trajectory + kg-incomplete counts per subset for both defs.
    def compute_splits(labels: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
        # group all G2 trajectories
        traj_by_split: dict[str, list[dict]] = {"seen": [], "unseen": [], "no_gold": []}
        for t in trajs:
            sid = str(t.get("sample_id", ""))
            lab = labels.get(sid, {"label": "no_gold"})["label"]
            traj_by_split[lab].append(t)
        # group kg_incomplete and outcomes
        outcome_by_split: dict[str, list] = {"seen": [], "unseen": [], "no_gold": []}
        for t, o in zip(kg_incomplete, outcomes_all):
            sid = str(t.get("sample_id", ""))
            lab = labels.get(sid, {"label": "no_gold"})["label"]
            outcome_by_split[lab].append(o)

        splits: dict[str, dict[str, Any]] = {}
        for name in ("seen", "unseen", "no_gold"):
            sub_outcomes = outcome_by_split[name]
            counts, pct = bucket_distribution(sub_outcomes)
            splits[name] = {
                "n_questions": sum(1 for v in labels.values() if v["label"] == name),
                "n_g2_trajectories": len(traj_by_split[name]),
                "n_kg_incomplete": len(sub_outcomes),
                "bucket_counts": {b: counts.get(b, 0) for b in BUCKETS},
                "bucket_pct": pct,
            }
        return splits

    splits_args = compute_splits(label_args)
    splits_exposed = compute_splits(label_exposed)

    # The primary report uses argument-based R_SFT (matches the brief).
    primary_splits = splits_args
    seen_def = (
        "seen = every gold relation in this question's kg_path appears in "
        "R_SFT (argument set); unseen = at least one gold relation NOT in "
        "R_SFT."
    )

    report: dict[str, Any] = {
        "trajectory_file": str(trajectories_path),
        "parquet_file": str(parquet_path),
        "sft_file": str(sft_path),
        "definition": seen_def,
        "sft_stats": sft_stats,
        "overall": {
            "n_g2_trajectories": len(trajs),
            "category_counts": dict(cat_counts),
            "n_kg_incomplete": len(kg_incomplete),
            "bucket_counts": {b: overall_counts.get(b, 0) for b in BUCKETS},
            "bucket_pct": overall_pct,
        },
        "split_question_counts_argument_def": q_counts_args,
        "split_question_counts_exposed_def": q_counts_exposed,
        "splits_argument_def": primary_splits,
        "splits_exposed_def": splits_exposed,
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    logger.info("Wrote %s", output_json)

    write_markdown(
        output_md,
        base_total=len(kg_incomplete),
        overall_counts=overall_counts,
        overall_pct=overall_pct,
        splits=primary_splits,
        sft_stats=sft_stats,
        seen_def=seen_def,
    )
    logger.info("Wrote %s", output_md)

    # Terminal summary
    print("\n=== S0.6 seen-vs-unseen V14-B2 ===")
    print(f"R_SFT (args) size: {sft_stats['n_r_sft_args']}")
    print(f"R_SFT (exposed) size: {sft_stats['n_r_sft_exposed']}")
    print(
        f"CWQ test questions split (arg def): seen={q_counts_args.get('seen',0)}, "
        f"unseen={q_counts_args.get('unseen',0)}, no_gold={q_counts_args.get('no_gold',0)}"
    )
    print("\nBucket distribution per subset (argument-based R_SFT):")
    print(
        f"{'subset':10s} {'n_kg_inc':>10s} "
        + "".join(f"{b:>32s}" for b in BUCKETS)
    )
    for name in ("seen", "unseen", "no_gold"):
        s = primary_splits[name]
        cells = "".join(
            f"{s['bucket_counts'][b]:>5d} ({s['bucket_pct'][b]:5.1f}%){' ':>14s}"
            for b in BUCKETS
        )
        print(f"{name:10s} {s['n_kg_incomplete']:>10d} {cells}")
    return report


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectories", type=Path, default=DEFAULT_TRAJECTORIES)
    parser.add_argument("--parquet", type=Path, default=DEFAULT_PARQUET)
    parser.add_argument("--sft", type=Path, default=DEFAULT_SFT)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUT_MD)
    args = parser.parse_args()

    run(
        args.trajectories,
        args.parquet,
        args.sft,
        args.output_json,
        args.output_md,
    )


if __name__ == "__main__":
    main()
