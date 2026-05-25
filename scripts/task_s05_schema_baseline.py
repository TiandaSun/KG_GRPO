"""S0.5 — Schema-format base rate (null hypothesis for V14-B2's 70.4% finding).

For each CWQ test question, sample 100 random relations from the 7,058-relation
Freebase pool and compute the min Levenshtein distance to the question's gold
relation chain (relations appearing in extra_info.kg_path). Aggregate the
fraction of (question, random_relation) pairs with min_dist <= 1.

This tells us how surprising V14-B2's 70.4% close-match rate actually is. If a
random relation hits <=1 edit-distance to gold ~40%+ of the time, the 70.4% is
near-null and the finding collapses. If <15%, it's strongly above null.

Reuses V14-B2's `relations_are_close` so the comparison is apples-to-apples.

Performance: precomputes pairwise distances between 100 random relations and
ALL unique gold relations across the test set (~5K unique gold), then per-q
lookup is O(|gold_rels_per_q|). Avoids the O(Q * |gold| * 100) inner loop.

Outputs:
  results/phase7/schema_baseline_rate.json
  results/phase7/schema_baseline_rate.md
"""

from __future__ import annotations

import json
import logging
import random
import sys
import time
from pathlib import Path

import pandas as pd

REPO = Path("/lus/lfs1aip2/scratch/u6gg/ts1201.u6gg/Project/KG_GRPO")
sys.path.insert(0, str(REPO / "scripts"))

# Reuse V14-B2's exact comparison logic
from task_v14_b2_decision_boundary import (  # type: ignore
    edit_distance,
    gold_entities_and_relations,
    norm_relation,
    rel_tail,
)
import difflib

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
    force=True,
)
log = logging.getLogger(__name__)

RELATIONS_FILE = REPO / "data" / "freebase" / "kg" / "relations.txt"
TEST_PARQUET = REPO / "data" / "freebase" / "verl_cwq" / "test.parquet"
OUT_JSON = REPO / "results" / "phase7" / "schema_baseline_rate.json"
OUT_MD = REPO / "results" / "phase7" / "schema_baseline_rate.md"

N_RANDOM = 100
SEED = 42


def load_relation_pool(path: Path) -> list[str]:
    seen: set[str] = set()
    relations: list[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            tok = line.strip().split("\t")[0].strip()
            if tok and tok not in seen:
                seen.add(tok)
                relations.append(tok)
    log.info("Loaded %d unique relations from %s", len(relations), path)
    return relations


def gold_relations_for_row(row: pd.Series) -> set[str]:
    kg_path = row["extra_info"].get("kg_path")
    if kg_path is None:
        return set()
    triples = [list(t) for t in kg_path if len(t) == 3]
    _, rels, _ = gold_entities_and_relations(triples)
    return set(rels)


def pairwise_distance(q: str, g: str) -> tuple[int, float]:
    """Compute (min_dist, max_ratio) using V14-B2 logic for full + tail forms."""
    q_full = norm_relation(q)
    g_full = norm_relation(g)
    q_tail = rel_tail(q)
    g_tail = rel_tail(g)
    if not q_full or not g_full:
        return 10**9, 0.0
    d_full = edit_distance(q_full, g_full)
    d_tail = edit_distance(q_tail, g_tail)
    r_full = difflib.SequenceMatcher(None, q_full, g_full).ratio()
    r_tail = difflib.SequenceMatcher(None, q_tail, g_tail).ratio()
    min_d = min(d_full, d_tail)
    max_r = max(r_full, r_tail)
    # V14-B2 special case: tail substring match -> treat as 1-edit
    if q_tail and g_tail and (q_tail == g_tail or q_tail in g_full or g_tail in q_full):
        min_d = min(min_d, 1)
        max_r = max(max_r, 0.9)
    return min_d, max_r


def main() -> None:
    t_start = time.time()
    rng = random.Random(SEED)

    pool = load_relation_pool(RELATIONS_FILE)
    sample_idx = rng.sample(range(len(pool)), N_RANDOM)
    random_rels = [pool[i] for i in sample_idx]
    log.info("Sampled %d random relations (seed=%d)", N_RANDOM, SEED)

    df = pd.read_parquet(TEST_PARQUET)
    log.info("Loaded %d test questions from %s", len(df), TEST_PARQUET)

    # ---------- Step 1: collect unique gold relations across all questions
    log.info("Collecting per-question gold-relation sets ...")
    per_q_golds: list[set[str]] = []
    n_no_gold = 0
    for _, row in df.iterrows():
        golds = gold_relations_for_row(row)
        per_q_golds.append(golds)
        if not golds:
            n_no_gold += 1
    unique_gold = set().union(*per_q_golds)
    log.info(
        "  unique gold relations across test set: %d (questions w/o gold: %d)",
        len(unique_gold), n_no_gold,
    )

    # ---------- Step 2: precompute pairwise distance & ratio
    log.info("Precomputing %d random x %d gold = %d pairwise comparisons ...",
             N_RANDOM, len(unique_gold), N_RANDOM * len(unique_gold))
    dist_matrix: dict[str, dict[str, int]] = {}      # random_rel -> gold_rel -> min_dist
    ratio_matrix: dict[str, dict[str, float]] = {}   # random_rel -> gold_rel -> max_ratio
    t_pre = time.time()
    for i, q in enumerate(random_rels):
        dist_matrix[q] = {}
        ratio_matrix[q] = {}
        for g in unique_gold:
            d, r = pairwise_distance(q, g)
            dist_matrix[q][g] = d
            ratio_matrix[q][g] = r
        if (i + 1) % 10 == 0:
            log.info("  precomputed %d/%d random rels (%.1fs elapsed)",
                     i + 1, N_RANDOM, time.time() - t_pre)
    log.info("Precompute done in %.1fs", time.time() - t_pre)

    # ---------- Step 3: aggregate per (q, random_rel) pair
    total_pairs = 0
    n_close_le1 = 0
    n_close_le2 = 0
    n_close_le3 = 0
    n_close_v14b2 = 0  # min_d <= 3 OR ratio > 0.8
    n_questions_with_gold = 0
    sample_sizes: list[int] = []
    per_q_close_le1: list[float] = []
    per_q_close_v14b2: list[float] = []

    for q_i, golds in enumerate(per_q_golds):
        if not golds:
            continue
        n_questions_with_gold += 1
        sample_sizes.append(len(golds))

        q_close_le1 = 0
        q_close_v14b2 = 0
        for q_rel in random_rels:
            best_d = 10**9
            best_r = 0.0
            dm = dist_matrix[q_rel]
            rm = ratio_matrix[q_rel]
            for g in golds:
                d = dm.get(g, 10**9)
                if d < best_d:
                    best_d = d
                r = rm.get(g, 0.0)
                if r > best_r:
                    best_r = r
            total_pairs += 1
            if best_d <= 1:
                n_close_le1 += 1
                q_close_le1 += 1
            if best_d <= 2:
                n_close_le2 += 1
            if best_d <= 3:
                n_close_le3 += 1
            if best_d <= 3 or best_r > 0.8:
                n_close_v14b2 += 1
                q_close_v14b2 += 1
        per_q_close_le1.append(q_close_le1 / N_RANDOM)
        per_q_close_v14b2.append(q_close_v14b2 / N_RANDOM)

    pct_le1 = n_close_le1 / total_pairs * 100 if total_pairs else 0.0
    pct_le2 = n_close_le2 / total_pairs * 100 if total_pairs else 0.0
    pct_le3 = n_close_le3 / total_pairs * 100 if total_pairs else 0.0
    pct_v14b2 = n_close_v14b2 / total_pairs * 100 if total_pairs else 0.0
    mean_per_q_le1 = (sum(per_q_close_le1) / len(per_q_close_le1) * 100) if per_q_close_le1 else 0.0
    mean_per_q_v14b2 = (sum(per_q_close_v14b2) / len(per_q_close_v14b2) * 100) if per_q_close_v14b2 else 0.0
    mean_gold = sum(sample_sizes) / len(sample_sizes) if sample_sizes else 0.0
    median_gold = sorted(sample_sizes)[len(sample_sizes) // 2] if sample_sizes else 0

    if pct_le1 < 15:
        decision = "STRONG"
        decision_text = "V14-B2's 70.4% is **strongly significant** (null < 15%). Ship as-is."
    elif pct_le1 < 30:
        decision = "MODERATE"
        decision_text = "V14-B2's 70.4% is **moderately above null** (null 15-30%). Add explicit base rate to paper."
    elif pct_le1 < 50:
        decision = "WEAK"
        decision_text = "V14-B2's 70.4% is **weak** (null 30-50%). Need to revise Section 6 framing. **PING USER.**"
    else:
        decision = "COLLAPSED"
        decision_text = "V14-B2's 70.4% **collapses** (null >= 50%). Cannot use in paper. **PING USER.**"

    elapsed = time.time() - t_start
    result = {
        "method": {
            "n_random_relations": N_RANDOM,
            "seed": SEED,
            "relation_pool_size": len(pool),
            "n_test_questions": len(df),
            "n_questions_with_gold": n_questions_with_gold,
            "n_questions_no_gold": n_no_gold,
            "n_unique_gold_relations_in_test": len(unique_gold),
            "comparison_logic": "V14-B2 relations_are_close (full + tail edit-distance, with ratio bonus)",
            "gold_relation_definition": "all relations in extra_info.kg_path subgraph (matches V14-B2)",
            "elapsed_seconds": round(elapsed, 1),
        },
        "totals": {
            "total_pairs": total_pairs,
            "mean_gold_relations_per_question": round(mean_gold, 2),
            "median_gold_relations_per_question": median_gold,
        },
        "null_rates": {
            "pct_min_edit_le1": round(pct_le1, 4),
            "pct_min_edit_le2": round(pct_le2, 4),
            "pct_min_edit_le3": round(pct_le3, 4),
            "pct_v14b2_close": round(pct_v14b2, 4),
            "mean_per_q_pct_le1": round(mean_per_q_le1, 4),
            "mean_per_q_pct_v14b2": round(mean_per_q_v14b2, 4),
        },
        "decision": {
            "threshold_used": "<=1 edit (user-specified S0.5 cutoff)",
            "null_rate_pct": round(pct_le1, 4),
            "v14b2_finding_pct": 70.4,
            "ratio_finding_to_null": round((70.4 / pct_le1), 2) if pct_le1 > 0 else float("inf"),
            "decision_label": decision,
            "decision_text": decision_text,
        },
        "random_relations_sampled": random_rels,
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(result, f, indent=2)
    log.info("Wrote %s", OUT_JSON)

    md = f"""# S0.5 — Schema-format Base Rate (Null Hypothesis for V14-B2)

**Setup**: Sampled {N_RANDOM} random relations (seed={SEED}) from the {len(pool):,}-relation \
Freebase pool. For each of {n_questions_with_gold:,} CWQ test questions with non-empty `kg_path` \
gold relation set, computed min Levenshtein distance from each random relation to all gold \
relations using V14-B2's exact `relations_are_close` logic (full + tail forms, ratio bonus).

Total pairs: {total_pairs:,}. Unique gold relations in test set: {len(unique_gold):,}. \
Gold relations per question: mean {mean_gold:.1f}, median {median_gold}. Compute time: {elapsed:.1f}s.

## Null rates by edit-distance threshold

| Threshold | Pair-level % close | Per-question mean % close |
|-----------|-------------------:|--------------------------:|
| <= 1 edit  | **{pct_le1:.2f}%** | {mean_per_q_le1:.2f}% |
| <= 2 edit  | {pct_le2:.2f}% | -- |
| <= 3 edit  | {pct_le3:.2f}% | -- |
| V14-B2 def (<=3 OR ratio>0.8) | {pct_v14b2:.2f}% | {mean_per_q_v14b2:.2f}% |

## Decision (user threshold: <=1 edit)

- **Null rate at <=1 edit: {pct_le1:.2f}%**
- V14-B2 reported finding: 70.4%
- Ratio of finding to null: **{(70.4 / pct_le1) if pct_le1 > 0 else float("inf"):.1f}x**

**Verdict: {decision}** -- {decision_text}

## Decision rule lookup table

| Null rate | Interpretation |
|-----------|---------------|
| < 15%   | V14-B2 strongly significant; ship as-is |
| 15-30%  | Moderately above null; add explicit base rate to paper |
| 30-50%  | Weak; revise Section 6 framing |
| > 50%   | Finding collapses; cannot use in paper |
"""
    with open(OUT_MD, "w") as f:
        f.write(md)
    log.info("Wrote %s", OUT_MD)

    print("\n" + "=" * 60)
    print(f"NULL RATE @ <=1 edit:    {pct_le1:.2f}%")
    print(f"NULL RATE @ V14-B2 def:  {pct_v14b2:.2f}%")
    print(f"V14-B2 FINDING:          70.4%")
    print(f"DECISION:                {decision}")
    print(f"Elapsed: {elapsed:.1f}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
