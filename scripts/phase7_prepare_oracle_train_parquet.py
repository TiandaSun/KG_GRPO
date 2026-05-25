"""Phase 7 Variant I-Oracle data prep.

Join train.parquet with train_oracle_gold_paths.jsonl so that the resulting
parquet carries `extra_info.oracle_triple_chain` — a short BFS-filtered gold
path (1-4 triples) instead of the full ~125-triple subgraph in kg_path.

verl's reward function reads extra_info from each row, so `_extract_gold_pairs`
will find `oracle_triple_chain` and use it for Variant I-Oracle's query-match
reward.

Output: data/freebase/verl_cwq/train_with_oracle.parquet
        (same schema as train.parquet + extra_info.oracle_triple_chain)

Rows whose sample_id is NOT in the oracle file keep empty oracle_triple_chain.
We also filter out these rows if --drop_unsolvable is set (so training only
sees questions where the Oracle found a gold path — ~18,580 / 27,639 ≈ 67%).
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)


def load_oracle_index(path: Path) -> dict[str, list[list[str]]]:
    """sample_id -> triple_chain."""
    idx: dict[str, list[list[str]]] = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            sid = rec.get("sample_id")
            chain = rec.get("triple_chain") or []
            if sid and chain:
                idx[str(sid)] = [list(t) for t in chain]
    return idx


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description="Add oracle_triple_chain to train.parquet")
    parser.add_argument(
        "--train_parquet", type=Path,
        default=Path("data/freebase/verl_cwq/train.parquet"),
    )
    parser.add_argument(
        "--oracle_jsonl", type=Path,
        default=Path("data/freebase/verl_cwq/train_oracle_gold_paths.jsonl"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("data/freebase/verl_cwq/train_with_oracle.parquet"),
    )
    parser.add_argument(
        "--drop_unsolvable", action="store_true",
        help="Keep only rows with oracle gold path (~18.5K vs 27.6K)",
    )
    args = parser.parse_args()

    logger.info("Loading oracle index from %s", args.oracle_jsonl)
    oracle = load_oracle_index(args.oracle_jsonl)
    logger.info("Loaded %d oracle records", len(oracle))

    logger.info("Loading train parquet from %s", args.train_parquet)
    df = pd.read_parquet(args.train_parquet)
    logger.info("Loaded %d train rows", len(df))

    kept_rows: list[dict] = []
    hit = 0
    for _, row in df.iterrows():
        row_dict = row.to_dict()
        extra = dict(row_dict["extra_info"])
        sid = str(extra.get("sample_id", ""))
        chain = oracle.get(sid, [])
        if chain:
            hit += 1
        extra["oracle_triple_chain"] = chain
        row_dict["extra_info"] = extra

        if args.drop_unsolvable and not chain:
            continue
        kept_rows.append(row_dict)

    logger.info(
        "Matched %d/%d rows (%.1f%%) with oracle path",
        hit, len(df), 100 * hit / max(1, len(df)),
    )
    logger.info("Writing %d rows to %s", len(kept_rows), args.output)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out_df = pd.DataFrame(kept_rows)
    # Preserve schema types by writing through pyarrow
    table = pa.Table.from_pandas(out_df, preserve_index=False)
    pq.write_table(table, args.output)
    logger.info("Done.")


if __name__ == "__main__":
    main()
