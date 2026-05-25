"""Task 36: KG Coverage Oracle — Gate Experiment.

For 200 Category B + 50 Category A CWQ questions:
1. Extract gold reasoning path from kg_path via BFS (query_entities → answer_entities)
2. Check each hop triple against the global Freebase KG
3. Classify: SOLVABLE / PARTIAL / UNREACHABLE / NO_PATH

This answers: "Does Freebase contain the triples needed to answer Category B questions?"

Usage:
    python scripts/task36_kg_coverage_oracle.py \
        --output results/oracle/task36_coverage.json
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def load_global_kg(triples_path: Path) -> tuple[set[tuple[str, str, str]], dict[str, dict[str, list[str]]]]:
    """Load global KG from triples.txt into a triple set and adjacency dict.

    Returns:
        triple_set: {(head, relation, tail)} for fast membership checks
        adj: {entity: {relation: [entities]}} for BFS (bidirectional)
    """
    triple_set: set[tuple[str, str, str]] = set()
    adj: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))

    t0 = time.time()
    with open(triples_path) as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 3:
                continue
            h, r, t = parts[0].strip(), parts[1].strip(), parts[2].strip()
            triple_set.add((h, r, t))
            adj[h][r].append(t)
            # Also add reverse direction for BFS
            adj[t][f"~{r}"].append(h)

    logger.info("Loaded %d triples, %d entities in %.1fs", len(triple_set), len(adj), time.time() - t0)
    return triple_set, adj


def check_triple_in_kg(h: str, r: str, t: str, triple_set: set[tuple[str, str, str]]) -> bool:
    """Check if a triple exists in the global KG (either direction)."""
    return (h, r, t) in triple_set or (t, r, h) in triple_set


def bfs_on_subgraph(
    kg_path: list[list[str]],
    query_entities: list[str],
    answer_entities: list[str],
    max_hops: int = 5,
) -> list[tuple[str, str, str]] | None:
    """BFS on per-question subgraph (kg_path triples) from query entities to answer entities.

    Returns the shortest path as a list of (head, relation, tail) triples, or None if no path found.
    """
    # Build local subgraph adjacency (bidirectional)
    local_adj: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for triple in kg_path:
        if len(triple) < 3:
            continue
        h, r, t = str(triple[0]).strip(), str(triple[1]).strip(), str(triple[2]).strip()
        local_adj[h].append((r, t))
        local_adj[t].append((f"~{r}", h))

    # Normalize entity names for matching
    answer_set = {str(a).strip().lower() for a in answer_entities}

    # BFS from query entities
    queue: list[tuple[str, list[tuple[str, str, str]]]] = []
    visited: set[str] = set()

    for qe in query_entities:
        qe = str(qe).strip()
        queue.append((qe, []))
        visited.add(qe.lower())

    while queue:
        entity, path = queue.pop(0)
        if len(path) > max_hops:
            continue

        for rel, neighbor in local_adj.get(entity, []):
            if neighbor.lower() in answer_set:
                # Found answer — record the hop
                if rel.startswith("~"):
                    final_triple = (neighbor, rel[1:], entity)
                else:
                    final_triple = (entity, rel, neighbor)
                return path + [final_triple]

            if neighbor.lower() not in visited:
                visited.add(neighbor.lower())
                if rel.startswith("~"):
                    hop_triple = (neighbor, rel[1:], entity)
                else:
                    hop_triple = (entity, rel, neighbor)
                queue.append((neighbor, path + [hop_triple]))

    return None


def bfs_on_global_kg(
    adj: dict[str, dict[str, list[str]]],
    query_entities: list[str],
    answer_entities: list[str],
    max_hops: int = 3,
) -> list[tuple[str, str, str]] | None:
    """BFS on global KG from query entities to answer entities (capped at max_hops).

    Used as fallback when subgraph BFS fails (NO_PATH).
    """
    answer_set = {str(a).strip().lower() for a in answer_entities}

    queue: list[tuple[str, list[tuple[str, str, str]]]] = []
    visited: set[str] = set()

    for qe in query_entities:
        qe_stripped = str(qe).strip()
        queue.append((qe_stripped, []))
        visited.add(qe_stripped.lower())

    while queue:
        entity, path = queue.pop(0)
        if len(path) >= max_hops:
            continue

        for rel, neighbors in adj.get(entity, {}).items():
            for neighbor in neighbors:
                if neighbor.lower() in answer_set:
                    if rel.startswith("~"):
                        final_triple = (neighbor, rel[1:], entity)
                    else:
                        final_triple = (entity, rel, neighbor)
                    return path + [final_triple]

                if neighbor.lower() not in visited:
                    visited.add(neighbor.lower())
                    if rel.startswith("~"):
                        hop_triple = (neighbor, rel[1:], entity)
                    else:
                        hop_triple = (entity, rel, neighbor)
                    queue.append((neighbor, path + [hop_triple]))

    return None


def classify_question(
    gold_path: list[tuple[str, str, str]] | None,
    triple_set: set[tuple[str, str, str]],
    path_source: str,
) -> dict:
    """Classify a question based on gold path and KG coverage.

    Returns: {"category": str, "n_hops": int, "hops_hit": int, "hops_miss": int, "details": [...]}
    """
    if gold_path is None:
        return {
            "category": "NO_PATH",
            "n_hops": 0,
            "hops_hit": 0,
            "hops_miss": 0,
            "path_source": path_source,
            "details": [],
        }

    details = []
    hits = 0
    misses = 0
    first_miss_at = -1

    for i, (h, r, t) in enumerate(gold_path):
        found = check_triple_in_kg(h, r, t, triple_set)
        details.append({
            "hop": i,
            "head": h,
            "relation": r,
            "tail": t,
            "found": found,
        })
        if found:
            hits += 1
        else:
            misses += 1
            if first_miss_at == -1:
                first_miss_at = i

    n_hops = len(gold_path)
    if misses == 0:
        category = "SOLVABLE"
    elif first_miss_at == 0:
        category = "UNREACHABLE"
    else:
        category = "PARTIAL"

    return {
        "category": category,
        "n_hops": n_hops,
        "hops_hit": hits,
        "hops_miss": misses,
        "path_source": path_source,
        "details": details,
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    parser = argparse.ArgumentParser(description="Task 36: KG Coverage Oracle")
    parser.add_argument("--eval_data", type=Path, default=Path("data/freebase/verl_cwq/test.parquet"))
    parser.add_argument("--category_b", type=Path, default=Path("results/task26_category_b.json"))
    parser.add_argument("--triples", type=Path, default=Path("data/freebase/kg/triples.txt"))
    parser.add_argument("--output", type=Path, default=Path("results/oracle/task36_coverage.json"))
    parser.add_argument("--n_cat_b", type=int, default=200, help="Number of Category B questions to sample")
    parser.add_argument("--n_cat_a", type=int, default=50, help="Number of Category A questions to sample")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--global_bfs_fallback", action="store_true", default=True,
                        help="Use global KG BFS for NO_PATH questions (slower but more complete)")
    args = parser.parse_args()

    # Load Category B/A IDs
    with open(args.category_b) as f:
        cat_data = json.load(f)
    cat_b_ids = set(cat_data["category_b_ids"])
    logger.info("Category B IDs: %d", len(cat_b_ids))

    # Load test data
    df = pd.read_parquet(args.eval_data)
    logger.info("Loaded %d test questions", len(df))

    # Split into Category A and B
    cat_b_rows = []
    cat_a_rows = []
    for _, row in df.iterrows():
        sid = row["extra_info"].get("sample_id", "")
        if sid in cat_b_ids:
            cat_b_rows.append(row)
        else:
            cat_a_rows.append(row)
    logger.info("Category B: %d, Category A: %d", len(cat_b_rows), len(cat_a_rows))

    # Sample
    rng = np.random.default_rng(args.seed)
    b_idx = rng.choice(len(cat_b_rows), size=min(args.n_cat_b, len(cat_b_rows)), replace=False)
    a_idx = rng.choice(len(cat_a_rows), size=min(args.n_cat_a, len(cat_a_rows)), replace=False)
    sampled = [(cat_b_rows[i], "B") for i in b_idx] + [(cat_a_rows[i], "A") for i in a_idx]
    logger.info("Sampled %d Category B + %d Category A = %d total",
                len(b_idx), len(a_idx), len(sampled))

    # Load global KG
    triple_set, global_adj = load_global_kg(args.triples)

    # Process each question
    results = []
    t0 = time.time()

    for i, (row, category) in enumerate(sampled):
        ei = row["extra_info"]
        sample_id = ei.get("sample_id", "")
        question = row["prompt"][1]["content"]
        gold_answer = row["reward_model"]["ground_truth"]
        query_entities = ei.get("query_entities", [])
        answer_entities = ei.get("answer_entities", [gold_answer])
        kg_path_raw = ei.get("kg_path", [])

        if hasattr(query_entities, "tolist"):
            query_entities = query_entities.tolist()
        if hasattr(answer_entities, "tolist"):
            answer_entities = answer_entities.tolist()
        if hasattr(kg_path_raw, "tolist"):
            kg_path_raw = kg_path_raw.tolist()

        # Convert kg_path entries to lists of 3-element tuples
        kg_path = []
        for triple in kg_path_raw:
            if hasattr(triple, "tolist"):
                triple = triple.tolist()
            if isinstance(triple, (list, tuple)) and len(triple) >= 3:
                kg_path.append([str(triple[0]), str(triple[1]), str(triple[2])])

        # Step 1: BFS on subgraph
        gold_path = bfs_on_subgraph(kg_path, query_entities, answer_entities)
        path_source = "subgraph"

        # Step 2: Fallback to global KG BFS if subgraph BFS failed
        if gold_path is None and args.global_bfs_fallback:
            gold_path = bfs_on_global_kg(global_adj, query_entities, answer_entities, max_hops=3)
            path_source = "global_kg" if gold_path is not None else "none"

        # Step 3: Classify coverage
        classification = classify_question(gold_path, triple_set, path_source)

        results.append({
            "sample_id": sample_id,
            "question": question,
            "gold_answer": gold_answer,
            "category_ab": category,
            "query_entities": query_entities,
            "answer_entities": answer_entities,
            "n_subgraph_triples": len(kg_path),
            **classification,
        })

        if (i + 1) % 25 == 0:
            elapsed = time.time() - t0
            cats = defaultdict(int)
            for r in results:
                cats[r["category"]] += 1
            logger.info(
                "  %d/%d (%.1fs) | %s",
                i + 1, len(sampled), elapsed,
                " ".join(f"{k}={v}" for k, v in sorted(cats.items())),
            )

    # Compute summary statistics
    elapsed = time.time() - t0
    summary: dict[str, dict] = {}
    for group in ["B", "A", "ALL"]:
        subset = [r for r in results if (group == "ALL" or r["category_ab"] == group)]
        n = len(subset)
        if n == 0:
            continue
        cats = defaultdict(int)
        for r in subset:
            cats[r["category"]] += 1
        hop_dist = defaultdict(int)
        for r in subset:
            if r["category"] == "SOLVABLE":
                hop_dist[r["n_hops"]] += 1
        summary[f"category_{group}"] = {
            "n": n,
            "counts": dict(cats),
            "pct": {k: v / n for k, v in cats.items()},
            "solvable_hop_distribution": dict(hop_dist),
        }

    # Decision recommendation
    cat_b_summary = summary.get("category_B", {})
    solvable_pct = cat_b_summary.get("pct", {}).get("SOLVABLE", 0)
    if solvable_pct >= 0.50:
        recommendation = "PROCEED: SOLVABLE ≥ 50% — tool use IS meaningful for CWQ/Freebase"
    elif solvable_pct >= 0.25:
        recommendation = "PROCEED_WITH_CAVEAT: SOLVABLE 25-50% — KG coverage is a co-bottleneck"
    else:
        recommendation = "HALT: SOLVABLE < 25% — Freebase too incomplete for meaningful tool-use research"

    output = {
        "summary": summary,
        "recommendation": recommendation,
        "elapsed_s": elapsed,
        "config": {
            "n_cat_b": args.n_cat_b,
            "n_cat_a": args.n_cat_a,
            "seed": args.seed,
            "global_bfs_fallback": args.global_bfs_fallback,
        },
        "results": results,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    logger.info("Saved to %s", args.output)

    # Print summary
    print(f"\n{'='*60}")
    print(f"KG COVERAGE ORACLE — Task 36 Results")
    print(f"{'='*60}")
    for group_key, group_data in summary.items():
        print(f"\n{group_key} (n={group_data['n']}):")
        for cat in ("SOLVABLE", "PARTIAL", "UNREACHABLE", "NO_PATH"):
            c = group_data["counts"].get(cat, 0)
            p = group_data["pct"].get(cat, 0)
            print(f"  {cat:<15} {c:4d}  ({100*p:5.1f}%)")
        if group_data.get("solvable_hop_distribution"):
            print(f"  Solvable hops: {dict(sorted(group_data['solvable_hop_distribution'].items()))}")

    print(f"\n{'='*60}")
    print(f"RECOMMENDATION: {recommendation}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
