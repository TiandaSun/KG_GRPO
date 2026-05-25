"""Task 33 L1: Lexical Triggers for Temporal Drift Detection.

Scan CWQ questions for surface patterns that suggest temporally-volatile facts.
The hypothesis: questions containing words like "now", "current", "president", etc.
are likely to have answers that drift over time, so any "wrong" answer in the
modern Wikidata may be a temporal artifact rather than a real annotation error.

Output: per-question binary flag + summary stats.

Usage:
    python scripts/task33_l1_lexical.py \
        --input data/freebase/cwq_original_train.json \
        --output results/task33_l1_lexical.json
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

# Patterns for temporal triggers
TEMPORAL_ADVERBS = re.compile(
    r"\b(now|currently|today|recent(ly)?|latest|present|nowadays|modern|"
    r"contemporary|at the moment|these days|right now)\b",
    re.IGNORECASE,
)

# Role nouns that are time-volatile (positions held by people/teams change)
ROLE_NOUNS = re.compile(
    r"\b(president|prime minister|pm|chancellor|premier|king|queen|"
    r"governor|mayor|senator|congressman|"
    r"ceo|chief executive|chairman|chairwoman|chair|director|"
    r"head coach|manager|captain|head of|leader of|"
    r"owner|owners|player|players|"
    r"husband|wife|spouse|partner|"
    r"member|members|"
    r"champion|winner|"
    r"dean|principal|"
    r"anchor|host|"
    r"bishop|pope|"
    r"cardinal|archbishop|"
    r"justice|chief justice|"
    r"speaker|"
    r"editor|editor in chief)\b",
    re.IGNORECASE,
)

# Tense indicators ("does", "is", "are" suggest present-tense queries about people/teams)
PRESENT_TENSE = re.compile(
    r"\b(is|are|does|do|has|have)\s+(?:the|a|an)?\s*\w+\s+(?:currently|right now|now|today)",
    re.IGNORECASE,
)


def classify_question(question: str) -> dict:
    """Classify a single question for temporal lexical triggers.

    Returns:
        dict with keys:
          - has_temporal_adverb: bool
          - has_role_noun: bool
          - has_present_tense_temporal: bool
          - matched_terms: list of strings
          - is_temporal: bool (any of the above)
    """
    matched = []

    adv_match = TEMPORAL_ADVERBS.findall(question)
    if adv_match:
        matched.extend([m if isinstance(m, str) else m[0] for m in adv_match])

    role_match = ROLE_NOUNS.findall(question)
    if role_match:
        matched.extend(role_match)

    tense_match = PRESENT_TENSE.findall(question)
    has_tense = bool(tense_match)

    return {
        "has_temporal_adverb": bool(adv_match),
        "has_role_noun": bool(role_match),
        "has_present_tense_temporal": has_tense,
        "matched_terms": list(set(matched)),
        "is_temporal": bool(adv_match or role_match or has_tense),
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    parser = argparse.ArgumentParser(description="L1: Lexical temporal trigger scan for CWQ")
    parser.add_argument("--input", type=Path, default=Path("data/freebase/cwq_original_train.json"))
    parser.add_argument("--output", type=Path, default=Path("results/task33_l1_lexical.json"))
    args = parser.parse_args()

    logger.info("Loading %s", args.input)
    with open(args.input) as f:
        data = json.load(f)
    logger.info("Loaded %d questions", len(data))

    per_question = []
    for d in data:
        q = d["question"]
        result = classify_question(q)
        per_question.append({
            "ID": d["ID"],
            "question": q,
            **result,
        })

    n_total = len(per_question)
    n_temporal = sum(1 for x in per_question if x["is_temporal"])
    n_adv = sum(1 for x in per_question if x["has_temporal_adverb"])
    n_role = sum(1 for x in per_question if x["has_role_noun"])
    n_tense = sum(1 for x in per_question if x["has_present_tense_temporal"])

    summary = {
        "n_total": n_total,
        "n_temporal_any": n_temporal,
        "pct_temporal_any": n_temporal / n_total,
        "n_temporal_adverb": n_adv,
        "pct_temporal_adverb": n_adv / n_total,
        "n_role_noun": n_role,
        "pct_role_noun": n_role / n_total,
        "n_present_tense": n_tense,
        "pct_present_tense": n_tense / n_total,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({"summary": summary, "per_question": per_question}, f, indent=2)

    logger.info("Saved to %s", args.output)
    print("\n=== L1 Lexical Trigger Scan ===")
    print(f"Total questions: {n_total}")
    print(f"Temporal (any trigger):     {n_temporal} ({100*n_temporal/n_total:.1f}%)")
    print(f"  - Temporal adverbs:       {n_adv} ({100*n_adv/n_total:.1f}%)")
    print(f"  - Role nouns:             {n_role} ({100*n_role/n_total:.1f}%)")
    print(f"  - Present-tense temporal: {n_tense} ({100*n_tense/n_total:.1f}%)")
    print()
    print("Sample temporal questions:")
    temporal_samples = [x for x in per_question if x["is_temporal"]][:10]
    for x in temporal_samples:
        print(f"  [{','.join(x['matched_terms'])}] {x['question'][:90]}")


if __name__ == "__main__":
    main()
