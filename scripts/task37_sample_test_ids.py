"""Pre-sample 200 test-split question IDs for Tasks 37 and 38.

Uses seed=42 for reproducibility. Same 200 IDs used by:
- Task 37 (trajectory classification on test split)
- Task 38 (pass@k on test split)
- Task 36 (oracle, uses same seed for Category B/A sampling)

Output: results/task37_sample_ids.json
  {"filter_ids": [...], "category_b_ids": [...]}  # both keys for compatibility
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    df = pd.read_parquet("data/freebase/verl_cwq/test.parquet")
    all_ids = [str(row["extra_info"].get("sample_id", str(i))) for i, (_, row) in enumerate(df.iterrows())]
    logger.info("Total test questions: %d", len(all_ids))

    rng = np.random.default_rng(42)
    sampled = rng.choice(all_ids, size=200, replace=False).tolist()
    logger.info("Sampled %d question IDs (seed=42)", len(sampled))

    out = Path("results/task37_sample_ids.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        # Both keys for compatibility: eval_with_tools uses "category_b_ids", we also add "filter_ids"
        json.dump({"filter_ids": sampled, "category_b_ids": sampled}, f, indent=2)
    logger.info("Saved to %s", out)


if __name__ == "__main__":
    main()
