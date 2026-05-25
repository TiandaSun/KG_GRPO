"""Task 33 L2: Predicate Volatility Scan for CWQ.

For each CWQ question, parse the SPARQL query, extract Freebase predicates,
look up their volatility scores from a hand-curated TSV, and compute a per-question
volatility score (max across predicates).

High volatility = the question depends on facts that change over time.
For RL training anchored to frozen Freebase 2015, high-volatility errors are
explained by temporal drift (T category in L3 audit), not annotation quality (I).

Output: per-question volatility scores + summary distribution.

Usage:
    python scripts/task33_l2_volatility.py \
        --input data/freebase/cwq_original_train.json \
        --tsv data/freebase_volatility.tsv \
        --output results/task33_l2_volatility.json
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
from collections import Counter
from pathlib import Path

logger = logging.getLogger(__name__)

PREDICATE_REGEX = re.compile(r"\bns:([a-zA-Z][a-zA-Z0-9_]*(?:\.[a-zA-Z][a-zA-Z0-9_]*)+)")
MID_REGEX = re.compile(r"^m\.")


def load_volatility_tsv(path: Path) -> dict[str, float]:
    """Load predicate->volatility mapping. Returns dict[predicate, score]."""
    table = {}
    with open(path) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            pred = row["predicate"].strip()
            try:
                vol = float(row["volatility"])
            except (KeyError, ValueError):
                continue
            table[pred] = vol
    logger.info("Loaded %d predicate volatility scores", len(table))
    return table


def extract_predicates(sparql: str) -> list[str]:
    """Extract Freebase predicates from a SPARQL query (skipping MIDs)."""
    matches = PREDICATE_REGEX.findall(sparql)
    return [p for p in matches if not MID_REGEX.match(p)]


def predicate_default_volatility(pred: str) -> float:
    """Heuristic fallback for predicates not in the TSV.

    Uses prefix-based defaults to handle the long tail."""
    # Position/role-related → high
    if any(s in pred for s in ["office_holder", "governing_officials", "appointees", "representatives"]):
        return 0.9
    # Sports rosters/current teams → high
    if "sports_team_roster.team" in pred or "pro_athlete.teams" in pred:
        return 0.9
    # Coach/captain → high
    if pred.endswith(".coach") or pred.endswith(".captain"):
        return 0.8
    # Marriage / spouse → medium-high
    if "spouse" in pred or "marriage" in pred:
        return 0.6
    # Award nominations / wins / incremental lists → medium
    if any(s in pred for s in ["nominations", "awards_won"]):
        return 0.5
    # Film cast/credits / static records → low
    if pred.startswith("film."):
        return 0.1
    # Pure geographic predicates → static
    if pred.startswith("location.location.") or "geocode" in pred:
        return 0.0
    # Static identifiers / classifications → 0
    if any(s in pred for s in ["notable_types", "calling_code", "iso3166", "fips", "founded", "date_founded", "date_of_birth", "place_of_birth"]):
        return 0.0
    # Default unknown → 0.2 (mildly volatile)
    return 0.2


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    parser = argparse.ArgumentParser(description="L2: Predicate volatility scan for CWQ")
    parser.add_argument("--input", type=Path, default=Path("data/freebase/cwq_original_train.json"))
    parser.add_argument("--tsv", type=Path, default=Path("data/freebase_volatility.tsv"))
    parser.add_argument("--output", type=Path, default=Path("results/task33_l2_volatility.json"))
    args = parser.parse_args()

    vol_table = load_volatility_tsv(args.tsv)

    logger.info("Loading %s", args.input)
    with open(args.input) as f:
        data = json.load(f)
    logger.info("Loaded %d questions", len(data))

    per_question = []
    unmatched_preds: Counter = Counter()

    for d in data:
        sparql = d.get("sparql", "")
        preds = extract_predicates(sparql)
        scores = []
        per_pred = {}
        for p in preds:
            if p in vol_table:
                v = vol_table[p]
                source = "tsv"
            else:
                v = predicate_default_volatility(p)
                source = "default"
                unmatched_preds[p] += 1
            scores.append(v)
            per_pred[p] = {"volatility": v, "source": source}

        max_vol = max(scores) if scores else 0.0
        mean_vol = sum(scores) / len(scores) if scores else 0.0

        per_question.append({
            "ID": d["ID"],
            "n_predicates": len(preds),
            "max_volatility": max_vol,
            "mean_volatility": mean_vol,
            "predicates": per_pred,
        })

    # Distribution buckets
    buckets = {"high (>=0.7)": 0, "medium (0.3-0.7)": 0, "low (<0.3)": 0}
    for x in per_question:
        v = x["max_volatility"]
        if v >= 0.7:
            buckets["high (>=0.7)"] += 1
        elif v >= 0.3:
            buckets["medium (0.3-0.7)"] += 1
        else:
            buckets["low (<0.3)"] += 1

    summary = {
        "n_total": len(per_question),
        "n_with_predicates": sum(1 for x in per_question if x["n_predicates"] > 0),
        "buckets_max_volatility": buckets,
        "pct_high_volatility": buckets["high (>=0.7)"] / len(per_question),
        "pct_medium_volatility": buckets["medium (0.3-0.7)"] / len(per_question),
        "pct_low_volatility": buckets["low (<0.3)"] / len(per_question),
        "n_unmatched_predicates": len(unmatched_preds),
        "top20_unmatched": unmatched_preds.most_common(20),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({"summary": summary, "per_question": per_question}, f, indent=2)

    logger.info("Saved to %s", args.output)
    print("\n=== L2 Predicate Volatility Scan ===")
    print(f"Total questions: {summary['n_total']}")
    print(f"With predicates: {summary['n_with_predicates']}")
    print()
    print("Distribution by max-volatility bucket:")
    for k, v in buckets.items():
        print(f"  {k:>20}  {v:6d}  ({100*v/summary['n_total']:.1f}%)")
    print()
    print(f"Unmatched predicates (used heuristic default): {len(unmatched_preds)}")
    print("Top-20 unmatched predicates by frequency:")
    for p, c in unmatched_preds.most_common(20):
        print(f"  {c:5d}  {p}  → default={predicate_default_volatility(p):.1f}")


if __name__ == "__main__":
    main()
