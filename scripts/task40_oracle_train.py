"""Task 40 Step 2: Oracle gold-path extraction for TRAIN split Category B.

Wraps the BFS logic from scripts/task36_kg_coverage_oracle.py and applies it
to every sample_id listed in train_category_b_ids.json. For each SOLVABLE
question we emit one JSON record describing the shortest (head, relation,
tail) chain from query_entities to answer_entities, plus hop count.

Output (JSONL, one sample per line):
    {
      "sample_id": ...,
      "question": ...,
      "gold_answer": ...,
      "query_entities": [...],
      "answer_entities": [...],
      "hops": N,
      "triple_chain": [[h, r, t], ...],
      "path_source": "subgraph" | "global_kg"
    }

Only SOLVABLE questions (all hops present in global KG) are kept.

Usage:
    python scripts/task40_oracle_train.py \
        --train data/freebase/verl_cwq/train.parquet \
        --category_b data/freebase/verl_cwq/train_category_b_ids.json \
        --triples data/freebase/kg/triples.txt \
        --output data/freebase/verl_cwq/train_oracle_gold_paths.jsonl \
        --stats_output results/task40_oracle_stats.json
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from collections import defaultdict
from pathlib import Path

import pandas as pd

# Import helpers from task36. Prefer the package import (works when run from
# project root with PYTHONPATH=.), fall back to direct file loading so the
# script also works as ``python scripts/task40_oracle_train.py`` without a
# ``scripts/__init__.py``.
try:
    from scripts.task36_kg_coverage_oracle import (  # type: ignore
        bfs_on_global_kg,
        bfs_on_subgraph,
        check_triple_in_kg,
        load_global_kg,
    )
except ModuleNotFoundError:
    import importlib.util
    import sys as _sys

    _here = Path(__file__).resolve().parent
    _spec = importlib.util.spec_from_file_location(
        "_task36_kg_coverage_oracle", _here / "task36_kg_coverage_oracle.py",
    )
    assert _spec is not None and _spec.loader is not None
    _mod = importlib.util.module_from_spec(_spec)
    _sys.modules["_task36_kg_coverage_oracle"] = _mod
    _spec.loader.exec_module(_mod)
    bfs_on_global_kg = _mod.bfs_on_global_kg
    bfs_on_subgraph = _mod.bfs_on_subgraph
    check_triple_in_kg = _mod.check_triple_in_kg
    load_global_kg = _mod.load_global_kg

logger = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    parser = argparse.ArgumentParser(description="Task 40 Step 2: Oracle on train Category B.")
    parser.add_argument("--train", type=Path,
                        default=Path("data/freebase/verl_cwq/train.parquet"))
    parser.add_argument("--category_b", type=Path,
                        default=Path("data/freebase/verl_cwq/train_category_b_ids.json"))
    parser.add_argument("--triples", type=Path,
                        default=Path("data/freebase/kg/triples.txt"))
    parser.add_argument("--output", type=Path,
                        default=Path("data/freebase/verl_cwq/train_oracle_gold_paths.jsonl"))
    parser.add_argument("--stats_output", type=Path,
                        default=Path("results/task40_oracle_stats.json"))
    parser.add_argument("--max_questions", type=int, default=0,
                        help="0 = all Cat B questions, else cap for debugging")
    parser.add_argument("--max_hops_global", type=int, default=3)
    args = parser.parse_args()

    # Load Category B IDs
    with open(args.category_b) as f:
        cat_data = json.load(f)
    cat_b_ids = set(cat_data.get("category_b_ids", []))
    logger.info("Loaded %d Category B sample_ids from %s", len(cat_b_ids), args.category_b)

    # Load train parquet
    df = pd.read_parquet(args.train)
    logger.info("Loaded %d train rows", len(df))

    # Filter to Category B rows (preserve order)
    cat_b_rows = []
    for _, row in df.iterrows():
        sid = row["extra_info"].get("sample_id", "")
        if sid in cat_b_ids:
            cat_b_rows.append(row)
        if args.max_questions and len(cat_b_rows) >= args.max_questions:
            break
    logger.info("Filtered to %d Category B rows", len(cat_b_rows))

    # Load global KG
    triple_set, global_adj = load_global_kg(args.triples)

    # Process each question
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.stats_output.parent.mkdir(parents=True, exist_ok=True)

    counts = defaultdict(int)
    hop_dist = defaultdict(int)
    t0 = time.time()
    kept = 0

    with open(args.output, "w") as outf:
        for i, row in enumerate(cat_b_rows):
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

            # Normalize subgraph triples
            kg_path: list[list[str]] = []
            for triple in kg_path_raw:
                if hasattr(triple, "tolist"):
                    triple = triple.tolist()
                if isinstance(triple, (list, tuple)) and len(triple) >= 3:
                    kg_path.append([str(triple[0]), str(triple[1]), str(triple[2])])

            # Step 1: subgraph BFS
            gold_path = bfs_on_subgraph(kg_path, query_entities, answer_entities)
            path_source = "subgraph"

            # Step 2: global KG fallback
            if gold_path is None:
                gold_path = bfs_on_global_kg(
                    global_adj, query_entities, answer_entities, max_hops=args.max_hops_global,
                )
                path_source = "global_kg" if gold_path is not None else "none"

            if gold_path is None:
                counts["NO_PATH"] += 1
                continue

            # Check solvability (all hops exist in global KG)
            all_hit = all(check_triple_in_kg(h, r, t, triple_set) for (h, r, t) in gold_path)
            if not all_hit:
                counts["PARTIAL"] += 1
                continue

            counts["SOLVABLE"] += 1
            hop_dist[len(gold_path)] += 1
            kept += 1

            rec = {
                "sample_id": sample_id,
                "question": question,
                "gold_answer": str(gold_answer),
                "query_entities": [str(x) for x in query_entities],
                "answer_entities": [str(x) for x in answer_entities],
                "hops": len(gold_path),
                "triple_chain": [[h, r, t] for (h, r, t) in gold_path],
                "path_source": path_source,
            }
            outf.write(json.dumps(rec, ensure_ascii=False) + "\n")

            if (i + 1) % 250 == 0:
                elapsed = time.time() - t0
                logger.info(
                    "  %d/%d (%.0fs) | kept=%d SOLVABLE=%d PARTIAL=%d NO_PATH=%d",
                    i + 1, len(cat_b_rows), elapsed, kept,
                    counts["SOLVABLE"], counts["PARTIAL"], counts["NO_PATH"],
                )

    elapsed = time.time() - t0

    stats = {
        "n_cat_b_input": len(cat_b_rows),
        "n_solvable_kept": kept,
        "counts": dict(counts),
        "pct": {k: v / max(len(cat_b_rows), 1) for k, v in counts.items()},
        "hop_distribution": dict(hop_dist),
        "elapsed_s": elapsed,
        "inputs": {
            "train": str(args.train),
            "category_b": str(args.category_b),
            "triples": str(args.triples),
        },
        "output": str(args.output),
    }
    with open(args.stats_output, "w") as f:
        json.dump(stats, f, indent=2)

    logger.info(
        "Wrote %d SOLVABLE gold paths to %s (elapsed %.0fs)",
        kept, args.output, elapsed,
    )
    logger.info("Stats: %s", stats["counts"])
    logger.info("Hop distribution: %s", dict(sorted(hop_dist.items())))


if __name__ == "__main__":
    main()
