"""Task 40 Step 5: Filter train.parquet to Category B sample_ids.

Preserves ALL columns and the exact schema so downstream verl GRPO jobs can
swap in the filtered file with zero config changes.

Usage:
    python scripts/task40_build_cat_b_parquet.py \
        --train data/freebase/verl_cwq/train.parquet \
        --category_b data/freebase/verl_cwq/train_category_b_ids.json \
        --output data/freebase/verl_cwq/train_category_b.parquet
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    parser = argparse.ArgumentParser(description="Task 40 Step 5: build Category B parquet.")
    parser.add_argument("--train", type=Path,
                        default=Path("data/freebase/verl_cwq/train.parquet"))
    parser.add_argument("--category_b", type=Path,
                        default=Path("data/freebase/verl_cwq/train_category_b_ids.json"))
    parser.add_argument("--output", type=Path,
                        default=Path("data/freebase/verl_cwq/train_category_b.parquet"))
    args = parser.parse_args()

    with open(args.category_b) as f:
        cat = json.load(f)
    cat_b_ids = set(cat.get("category_b_ids", []))
    logger.info("Loaded %d Category B sample_ids", len(cat_b_ids))

    df = pd.read_parquet(args.train)
    logger.info("Loaded %d train rows (%d cols)", len(df), len(df.columns))

    # Build sample_id column without mutating extra_info
    sample_ids = [r.get("sample_id", "") for r in df["extra_info"]]
    mask = [sid in cat_b_ids for sid in sample_ids]
    out_df = df[mask].reset_index(drop=True)

    logger.info("Filtered to %d rows (%.1f%% of train)",
                len(out_df), 100 * len(out_df) / max(len(df), 1))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(args.output, index=False)

    # Verify round-trip
    reloaded = pd.read_parquet(args.output)
    logger.info("Wrote %s (%d rows, %d cols)", args.output, len(reloaded), len(reloaded.columns))
    assert list(reloaded.columns) == list(df.columns), "Column order mismatch!"


if __name__ == "__main__":
    main()
