#!/usr/bin/env python3
"""WT-2 (G2 downstream cross-tab) per hpc_tasks.md v16 lines 207-230.

For G2@500 trajectories on full CWQ test (3531), classify each error qid
that falls in {kg-incomplete, wrong-answer} as either:
  - retrieval-failure: tool responses do NOT contain the gold answer
  - extraction-failure: at least one tool response DOES contain the gold answer
And report `no_retrieval` separately (zero tool calls).

Match: word-boundary regex on normalised gold strings + token-set against aliases.

CPU-only; pure analysis; no GPU. Run inside a SLURM allocation per project rules.
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import string
import sys
from pathlib import Path
from typing import Any

# Reuse canonical category logic from task16_classify so kg-incomplete /
# wrong-answer subsets exactly match v14_b2_decision_boundary counts (818 / 1140).
PROJECT_ROOT = Path("/lus/lfs1aip2/scratch/u6gg/ts1201.u6gg/Project/KG_GRPO")
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from task16_classify import classify_trajectory, extract_tool_responses  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("wt2")

ARTICLE_RE = re.compile(r"^(a|an|the)\s+", re.IGNORECASE)


def normalise_gold(text: str) -> str:
    """Lowercase, strip leading article, strip punctuation, collapse whitespace."""
    if text is None:
        return ""
    t = text.lower().strip()
    t = ARTICLE_RE.sub("", t)
    t = t.translate(str.maketrans("", "", string.punctuation))
    t = " ".join(t.split())
    return t


def normalise_response(text: str) -> str:
    """Same normalisation pipeline as gold, applied once to the concat tool text."""
    if not text:
        return ""
    t = text.lower()
    t = t.translate(str.maketrans("", "", string.punctuation))
    t = " ".join(t.split())
    return t


def word_boundary_match(norm_gold: str, norm_response: str) -> bool:
    """Word-boundary regex match of gold inside response (avoids Obama/Obamacare)."""
    if not norm_gold or not norm_response:
        return False
    pattern = r"\b" + re.escape(norm_gold) + r"\b"
    return re.search(pattern, norm_response) is not None


def gold_in_response(all_answers: list[str], tool_response_text: str) -> tuple[bool, str | None]:
    """Return (matched, matched_alias) using word-boundary match on normalised text."""
    norm_resp = normalise_response(tool_response_text)
    if not norm_resp:
        return False, None
    for ans in all_answers or []:
        ng = normalise_gold(ans)
        if not ng:
            continue
        if word_boundary_match(ng, norm_resp):
            return True, ans
    return False, None


def make_exemplar(traj: dict, matched_alias: str | None, max_snippet: int = 400) -> dict:
    tool_text = extract_tool_responses(traj.get("full_response", ""))
    snippet: str | None = None
    if tool_text:
        if matched_alias:
            ng = normalise_gold(matched_alias)
            norm_resp = normalise_response(tool_text)
            m = re.search(r"\b" + re.escape(ng) + r"\b", norm_resp)
            if m:
                lo = max(0, m.start() - 120)
                hi = min(len(norm_resp), m.end() + 120)
                snippet = norm_resp[lo:hi]
        if snippet is None:
            snippet = tool_text[:max_snippet]
    return {
        "sample_id": traj.get("sample_id"),
        "gold_answer": traj.get("gold_answer"),
        "predicted": traj.get("predicted"),
        "num_tool_calls": traj.get("num_tool_calls"),
        "matched_alias": matched_alias,
        "snippet_of_tool_response_or_null": snippet,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="WT-2 G2 cross-tab analysis")
    parser.add_argument(
        "--trajectories",
        type=Path,
        default=PROJECT_ROOT
        / "results/trajectories/phase7/39g2_step500_full/step_0/trajectories.json",
    )
    parser.add_argument(
        "--output_json",
        type=Path,
        default=PROJECT_ROOT
        / "kg_grpo_emnlp2026_v2/_handoff/data/g2_crosstab_2026_05_07.json",
    )
    parser.add_argument(
        "--output_md",
        type=Path,
        default=PROJECT_ROOT
        / "kg_grpo_emnlp2026_v2/_handoff/data/g2_crosstab_2026_05_07_summary.md",
    )
    parser.add_argument("--n_exemplars", type=int, default=5)
    args = parser.parse_args()

    if not args.trajectories.exists():
        logger.error("Trajectory dump not found: %s", args.trajectories)
        return 2

    logger.info("Loading trajectories from %s", args.trajectories)
    with open(args.trajectories) as f:
        trajectories: list[dict[str, Any]] = json.load(f)
    logger.info("Loaded %d trajectories", len(trajectories))

    args.output_json.parent.mkdir(parents=True, exist_ok=True)

    # Step 1 — re-classify all trajectories with task16 logic.
    classified: dict[str, str] = {}
    for traj in trajectories:
        classified[traj["sample_id"]] = classify_trajectory(traj)

    cat_counts: dict[str, int] = {}
    for cat in classified.values():
        cat_counts[cat] = cat_counts.get(cat, 0) + 1
    logger.info("Category counts: %s", cat_counts)

    # Step 2 — bucket each kg-incomplete / wrong-answer trajectory.
    buckets: dict[str, list[dict]] = {
        "kg_incomplete_retrieval_failure": [],
        "kg_incomplete_extraction_failure": [],
        "wrong_answer_retrieval_failure": [],
        "wrong_answer_extraction_failure": [],
    }
    no_retrieval_count = 0
    no_retrieval_examples: list[dict] = []

    target_categories = {"kg-incomplete", "wrong-answer"}
    for traj in trajectories:
        cat = classified[traj["sample_id"]]
        if cat not in target_categories:
            continue
        n_calls = traj.get("num_tool_calls", 0) or 0
        if n_calls == 0:
            no_retrieval_count += 1
            if len(no_retrieval_examples) < args.n_exemplars:
                no_retrieval_examples.append(make_exemplar(traj, None))
            continue
        tool_text = extract_tool_responses(traj.get("full_response", ""))
        matched, matched_alias = gold_in_response(traj.get("all_answers") or [], tool_text)
        prefix = "kg_incomplete" if cat == "kg-incomplete" else "wrong_answer"
        suffix = "extraction_failure" if matched else "retrieval_failure"
        bucket_key = f"{prefix}_{suffix}"
        buckets[bucket_key].append(make_exemplar(traj, matched_alias))

    # Step 3 — totals + verdict.
    bucket_counts = {k: len(v) for k, v in buckets.items()}
    total_classified = sum(bucket_counts.values())
    total_extraction = (
        bucket_counts["kg_incomplete_extraction_failure"]
        + bucket_counts["wrong_answer_extraction_failure"]
    )
    total_retrieval = (
        bucket_counts["kg_incomplete_retrieval_failure"]
        + bucket_counts["wrong_answer_retrieval_failure"]
    )
    extraction_pct = total_extraction / total_classified if total_classified else 0.0
    retrieval_pct = total_retrieval / total_classified if total_classified else 0.0

    if extraction_pct >= 0.50:
        verdict = "extraction-dominant"
    elif retrieval_pct >= 0.50:
        verdict = "retrieval-dominant"
    else:
        verdict = "mixed"

    # Step 4 — sanity checks.
    sanity_anomalies: list[str] = []
    expected_total = 818 + 1140  # 1958, per v14_b2 spec
    # exclude no_retrieval from comparison: the reference 1958 includes them, so we
    # check both views and only flag if the sum (incl. no_retrieval) deviates >5%.
    full_target_sum = total_classified + no_retrieval_count
    deviation = abs(full_target_sum - expected_total) / expected_total
    if deviation > 0.05:
        sanity_anomalies.append(
            f"kg-incomplete + wrong-answer total = {full_target_sum} "
            f"(expected ~{expected_total}, deviation {deviation:.1%})"
        )
    # Also flag if any sub-bucket is suspiciously empty.
    for k, n in bucket_counts.items():
        if n == 0:
            sanity_anomalies.append(f"bucket {k} is empty")

    # Step 5 — emit log.
    logger.info("Bucket counts: %s", bucket_counts)
    logger.info("no_retrieval_count: %d", no_retrieval_count)
    logger.info(
        "total_classified=%d  extraction=%d (%.1f%%)  retrieval=%d (%.1f%%)  verdict=%s",
        total_classified,
        total_extraction,
        100 * extraction_pct,
        total_retrieval,
        100 * retrieval_pct,
        verdict,
    )
    if sanity_anomalies:
        logger.warning("Sanity anomalies: %s", sanity_anomalies)
    else:
        logger.info("No sanity anomalies detected")

    # Step 6 — exemplars.
    exemplars_out = {k: v[: args.n_exemplars] for k, v in buckets.items()}
    for bucket_key, exs in exemplars_out.items():
        logger.info("--- exemplars: %s ---", bucket_key)
        for ex in exs:
            logger.info(
                "  %s | gold=%r | pred=%r | n_calls=%d | matched_alias=%r",
                ex["sample_id"],
                ex["gold_answer"],
                ex["predicted"],
                ex["num_tool_calls"],
                ex["matched_alias"],
            )
            if ex["snippet_of_tool_response_or_null"]:
                snip = ex["snippet_of_tool_response_or_null"]
                logger.info("    snippet: %s", snip[:300])

    # Step 7 — write JSON output.
    payload = {
        "buckets": bucket_counts,
        "no_retrieval_count": no_retrieval_count,
        "no_retrieval_examples": no_retrieval_examples,
        "total_errors_classified": total_classified,
        "extraction_pct": extraction_pct,
        "retrieval_pct": retrieval_pct,
        "verdict": verdict,
        "exemplars": exemplars_out,
        "sanity_anomalies": sanity_anomalies,
        "category_counts_full": cat_counts,
        "inputs": {
            "trajectories": str(args.trajectories),
            "n_total_trajectories": len(trajectories),
        },
    }
    with open(args.output_json, "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    logger.info("Wrote %s", args.output_json)

    # Step 8 — markdown summary.
    summary_line = (
        f"WT-2 G2@500 cross-tab: verdict=**{verdict}**  "
        f"extraction={100 * extraction_pct:.1f}%  retrieval={100 * retrieval_pct:.1f}%  "
        f"(N={total_classified} classified, {no_retrieval_count} no_retrieval).  "
        f"Anomalies: {sanity_anomalies if sanity_anomalies else 'none'}.\n"
    )
    with open(args.output_md, "w") as f:
        f.write(summary_line)
    logger.info("Wrote %s", args.output_md)
    logger.info("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
