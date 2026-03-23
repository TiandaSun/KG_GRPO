"""Extract hop counts and composition types from CWQ SPARQL annotations.

Parses the gold SPARQL queries in the original CWQ dataset to determine
the true number of triple patterns (hops) for each question.

Usage:
    python scripts/extract_sparql_hops.py \
        --cwq_json data/freebase/cwq_original_train.json \
        --output data/freebase/cwq_hop_annotations.json
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from collections import Counter, defaultdict
from pathlib import Path

logger = logging.getLogger(__name__)


def count_sparql_hops(sparql: str) -> int:
    """Count the number of triple patterns in a SPARQL query.

    Each `?x ns:predicate ?y` pattern = 1 hop.
    FILTER, OPTIONAL, and subqueries are not counted as hops.
    """
    if not sparql:
        return 0

    # Remove FILTER clauses (they don't represent hops)
    cleaned = re.sub(r'FILTER\s*\([^)]*\)', '', sparql)
    # Remove nested FILTER with balanced parens
    cleaned = re.sub(r'FILTER\s*\((?:[^()]*|\((?:[^()]*|\([^()]*\))*\))*\)', '', cleaned)

    # Count triple patterns: subject predicate object
    # Pattern: something ns:something something
    triple_patterns = re.findall(r'(?:ns:\S+|\?\w+)\s+ns:\S+\s+(?:ns:\S+|\?\w+)', cleaned)

    return max(len(triple_patterns), 0)


def extract_annotations(cwq_data: list[dict]) -> list[dict]:
    """Extract hop count and composition type for each CWQ sample."""
    annotations = []

    for sample in cwq_data:
        sample_id = sample.get("ID", "")
        sparql = sample.get("sparql", "")
        comp_type = sample.get("compositionality_type", "unknown")
        question = sample.get("question", "")

        hops = count_sparql_hops(sparql)

        # Extract answer for reference
        answers_data = sample.get("answers", [])
        if isinstance(answers_data, dict):
            answers_data = [answers_data]
        gold_answers = []
        aliases = []
        for ans in answers_data:
            if isinstance(ans, dict):
                gold_answers.append(ans.get("answer", ""))
                aliases.extend(ans.get("aliases", []))
            else:
                gold_answers.append(str(ans))

        annotations.append({
            "id": sample_id,
            "question": question,
            "sparql_hops": hops,
            "compositionality_type": comp_type,
            "gold_answers": gold_answers,
            "aliases": aliases,
        })

    return annotations


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    parser = argparse.ArgumentParser(description="Extract SPARQL hop counts from CWQ.")
    parser.add_argument("--cwq_json", type=Path, default=Path("data/freebase/cwq_original_train.json"))
    parser.add_argument("--output", type=Path, default=Path("data/freebase/cwq_hop_annotations.json"))
    parser.add_argument("--also_download_dev_test", action="store_true",
                        help="Also download and annotate dev/test splits")
    args = parser.parse_args()

    # Load CWQ data
    logger.info("Loading CWQ from %s", args.cwq_json)
    with open(args.cwq_json) as f:
        cwq_data = json.load(f)
    logger.info("Loaded %d samples", len(cwq_data))

    # Extract annotations
    annotations = extract_annotations(cwq_data)

    # Statistics
    hop_dist = Counter(a["sparql_hops"] for a in annotations)
    comp_dist = Counter(a["compositionality_type"] for a in annotations)

    logger.info("=== Hop Distribution (SPARQL-based) ===")
    for h in sorted(hop_dist.keys()):
        logger.info("  %d-hop: %d (%.1f%%)", h, hop_dist[h], hop_dist[h] / len(annotations) * 100)

    logger.info("=== Composition Type Distribution ===")
    for c, count in comp_dist.most_common():
        logger.info("  %s: %d (%.1f%%)", c, count, count / len(annotations) * 100)

    # Cross-tabulate
    logger.info("=== Hop x Composition Type ===")
    cross = defaultdict(lambda: defaultdict(int))
    for a in annotations:
        cross[a["sparql_hops"]][a["compositionality_type"]] += 1
    for h in sorted(cross.keys()):
        parts = ", ".join(f"{c}={n}" for c, n in sorted(cross[h].items()))
        logger.info("  %d-hop: %s", h, parts)

    # Save
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output_data = {
        "annotations": annotations,
        "stats": {
            "hop_distribution": dict(hop_dist),
            "composition_distribution": dict(comp_dist),
            "total": len(annotations),
        }
    }
    with open(args.output, "w") as f:
        json.dump(output_data, f, indent=2)
    logger.info("Saved %d annotations to %s", len(annotations), args.output)


if __name__ == "__main__":
    main()
