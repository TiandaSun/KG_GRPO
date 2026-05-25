"""Task 40 Step 1 helper: merge N chunked pass@10 results into a single
train_category_b_ids.json file.

Each chunk is a file produced by scripts/task26_pass10_category_b.py run with
--start_idx/--end_idx on data/freebase/verl_cwq/train.parquet. We union the
category_b_ids across chunks and write the canonical artifact.

Usage:
    python scripts/task40_merge_cat_b.py \
        --inputs results/task40_train_cat_b_chunk0.json \
                 results/task40_train_cat_b_chunk1.json \
                 results/task40_train_cat_b_chunk2.json \
                 results/task40_train_cat_b_chunk3.json \
        --output data/freebase/verl_cwq/train_category_b_ids.json
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    parser = argparse.ArgumentParser(description="Merge Task 40 chunked Cat B results.")
    parser.add_argument("--inputs", type=Path, nargs="+", required=True,
                        help="Chunk JSON files from task26_pass10_category_b.py")
    parser.add_argument("--output", type=Path,
                        default=Path("data/freebase/verl_cwq/train_category_b_ids.json"))
    args = parser.parse_args()

    cat_a: list[str] = []
    cat_b: list[str] = []
    per_question: list[dict] = []
    n_total = 0
    elapsed = 0.0

    for p in args.inputs:
        if not p.exists():
            logger.warning("Missing chunk: %s", p)
            continue
        with open(p) as f:
            d = json.load(f)
        cat_a.extend(d.get("category_a_ids", []))
        cat_b.extend(d.get("category_b_ids", []))
        per_question.extend(d.get("per_question", []))
        n_total += int(d.get("n_total", 0))
        elapsed += float(d.get("elapsed_s", 0.0))
        logger.info(
            "chunk %s: cat_a=%d cat_b=%d n_total=%d",
            p.name,
            len(d.get("category_a_ids", [])),
            len(d.get("category_b_ids", [])),
            d.get("n_total", 0),
        )

    # Dedup while preserving order
    seen: set[str] = set()
    cat_b_unique: list[str] = []
    for sid in cat_b:
        if sid not in seen:
            seen.add(sid)
            cat_b_unique.append(sid)

    seen_a: set[str] = set()
    cat_a_unique: list[str] = []
    for sid in cat_a:
        if sid not in seen_a:
            seen_a.add(sid)
            cat_a_unique.append(sid)

    out = {
        "category_b_ids": cat_b_unique,
        "category_a_ids": cat_a_unique,
        "metadata": {
            "source": "task40_pass10_train_split",
            "n_total": n_total,
            "n_cat_a": len(cat_a_unique),
            "n_cat_b": len(cat_b_unique),
            "cat_b_pct": len(cat_b_unique) / max(n_total, 1),
            "elapsed_s_total": elapsed,
            "chunks": [str(p) for p in args.inputs],
            "model": "Qwen/Qwen2.5-7B-Instruct",
            "k_samples": 10,
            "temperature": 0.7,
            "split": "train",
        },
        "per_question": per_question,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)

    logger.info(
        "Wrote %s (cat_a=%d cat_b=%d total=%d)",
        args.output, len(cat_a_unique), len(cat_b_unique), n_total,
    )


if __name__ == "__main__":
    main()
