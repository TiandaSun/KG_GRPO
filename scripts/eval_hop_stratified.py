"""Hop-stratified evaluation: break down EM/F1 by SPARQL hop count.

Reads saved trajectories (from eval_with_tools.py --save_trajectories) and
reports metrics by hop bucket (2-hop, 3-hop, 4-hop, 5+).

Can also re-annotate from cwq_hop_annotations.json if trajectories lack hop info.

Usage:
    python scripts/eval_hop_stratified.py \
        --trajectories results/trajectories/e3_verifiable/step_500/trajectories.json \
        --hop_annotations data/freebase/cwq_hop_annotations.json \
        --output results/hop_stratified_e3_500.json
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path

logger = logging.getLogger(__name__)


def bucket_hops(hops: int) -> str:
    """Group hop counts into buckets."""
    if hops <= 1:
        return "1-hop"
    elif hops == 2:
        return "2-hop"
    elif hops == 3:
        return "3-hop"
    elif hops == 4:
        return "4-hop"
    else:
        return "5+-hop"


def stratify(
    trajectories: list[dict],
    annotations_by_id: dict[str, dict] | None = None,
) -> dict[str, dict]:
    """Compute EM/F1/tool_calls stratified by hop bucket."""
    buckets: dict[str, list[dict]] = defaultdict(list)

    for traj in trajectories:
        hops = traj.get("hops", 0)

        # Try annotation lookup if hops missing
        if hops == 0 and annotations_by_id:
            sid = traj.get("sample_id", "")
            if sid in annotations_by_id:
                hops = annotations_by_id[sid].get("sparql_hops", 0)

        bucket = bucket_hops(hops)
        buckets[bucket].append(traj)

    results = {}
    for bucket in sorted(buckets.keys()):
        items = buckets[bucket]
        n = len(items)
        em = sum(t.get("em", 0) for t in items) / n if n else 0
        f1 = sum(t.get("f1", 0) for t in items) / n if n else 0
        tools = sum(t.get("num_tool_calls", 0) for t in items) / n if n else 0
        results[bucket] = {
            "n": n,
            "em": round(em, 4),
            "f1": round(f1, 4),
            "avg_tool_calls": round(tools, 2),
        }

    # Add overall
    n_total = len(trajectories)
    results["overall"] = {
        "n": n_total,
        "em": round(sum(t.get("em", 0) for t in trajectories) / n_total, 4) if n_total else 0,
        "f1": round(sum(t.get("f1", 0) for t in trajectories) / n_total, 4) if n_total else 0,
        "avg_tool_calls": round(sum(t.get("num_tool_calls", 0) for t in trajectories) / n_total, 2) if n_total else 0,
    }

    return results


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    parser = argparse.ArgumentParser(description="Hop-stratified eval breakdown.")
    parser.add_argument("--trajectories", type=Path, required=True,
                        help="Path to trajectories.json from eval_with_tools.py")
    parser.add_argument("--hop_annotations", type=Path,
                        default=Path("data/freebase/cwq_hop_annotations.json"),
                        help="CWQ hop annotations (fallback for missing hops)")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output JSON path (default: print to stdout)")
    args = parser.parse_args()

    # Load trajectories
    with open(args.trajectories) as f:
        trajectories = json.load(f)
    logger.info("Loaded %d trajectories from %s", len(trajectories), args.trajectories)

    # Load hop annotations as fallback
    annotations_by_id: dict[str, dict] | None = None
    if args.hop_annotations.exists():
        with open(args.hop_annotations) as f:
            ann_data = json.load(f)
        annotations_by_id = {a["id"]: a for a in ann_data.get("annotations", [])}
        logger.info("Loaded %d hop annotations", len(annotations_by_id))

    # Stratify
    results = stratify(trajectories, annotations_by_id)

    # Print table
    print(f"\n{'Bucket':<10} {'N':>6} {'EM':>8} {'F1':>8} {'Tools':>8}")
    print("-" * 44)
    for bucket in ["1-hop", "2-hop", "3-hop", "4-hop", "5+-hop", "overall"]:
        if bucket in results:
            r = results[bucket]
            marker = " *" if bucket == "overall" else ""
            print(f"{bucket:<10} {r['n']:>6} {r['em']:>8.4f} {r['f1']:>8.4f} {r['avg_tool_calls']:>8.2f}{marker}")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        logger.info("Saved to %s", args.output)


if __name__ == "__main__":
    main()
