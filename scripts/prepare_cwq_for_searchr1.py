"""Convert our CWQ parquet (verl multi-turn format) into Search-R1's expected
single-turn QA format.

Search-R1's format (see external/Search-R1/scripts/data_process/nq_search.py):

    {
        "data_source": "<one of nq/triviaqa/popqa/hotpotqa/2wikimultihopqa/musique/bamboogle>",
        "prompt": [{
            "role": "user",
            "content": <templated question prefix with <think>/<search>/<answer> instructions>,
        }],
        "ability": "fact-reasoning",
        "reward_model": {"style": "rule", "ground_truth": {"target": [<list of golden answers>]}},
        "extra_info": {"split": "train|test", "index": idx, ...passthrough},
    }

Key adaptations:
  - Our user prompt is a plain question; we wrap it in Search-R1's
    <think>/<search>/<answer> template so the trained policy emits the tags
    Search-R1's reward function greps for.
  - ground_truth is a plain string in ours; Search-R1 expects
    {"target": [list]}. We use extra_info.all_answers when present, else the
    ground_truth string as a single-element list.
  - data_source must be one of Search-R1's known sources so their
    _select_rm_score_fn picks up qa_em.compute_score_em. We use 'hotpotqa'
    because CWQ is the closest analogue (multi-hop factoid).

Usage:
  python scripts/prepare_cwq_for_searchr1.py \
      --in_train data/freebase/verl_cwq/train.parquet \
      --in_test  data/freebase/verl_cwq/test.parquet \
      --out_dir  data/freebase/searchr1_cwq
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Template adapted from Search-R1/scripts/data_process/nq_search.py
SEARCH_R1_TEMPLATE = (
    "Answer the given question. "
    "You must conduct reasoning inside <think> and </think> first every time you get new information. "
    "After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> "
    "and it will return the top searched results between <information> and </information>. "
    "You can search as many times as your want. "
    "If you find no further external knowledge needed, you can directly provide the answer inside <answer> "
    "and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>. "
    "Question: {question}\n"
)


def _extract_question(prompt: Any) -> str:
    """Pull the natural-language user question out of our verl prompt list.

    Our prompt is a list/ndarray of {role, content} dicts (system + user). We
    always take the last user message as the question text.
    """
    try:
        iterator = list(prompt)
    except TypeError:
        return str(prompt).strip()
    for msg in iterator:
        role = msg.get("role") if isinstance(msg, dict) else None
        if role == "user":
            return str(msg["content"]).strip()
    if iterator:
        last = iterator[-1]
        return str(last.get("content") if isinstance(last, dict) else last).strip()
    return ""


def _extract_targets(row: dict[str, Any]) -> list[str]:
    """Assemble the golden-answer list Search-R1 expects."""
    targets: list[str] = []
    extra = row.get("extra_info") or {}
    if isinstance(extra, dict):
        all_ans = extra.get("all_answers")
        if all_ans is not None:
            try:
                for a in all_ans:
                    s = str(a).strip()
                    if s:
                        targets.append(s)
            except TypeError:
                pass
    if not targets:
        rm = row.get("reward_model") or {}
        if isinstance(rm, dict):
            gt = rm.get("ground_truth")
            if isinstance(gt, str) and gt.strip():
                targets.append(gt.strip())
            elif isinstance(gt, dict) and "target" in gt:
                for a in gt["target"]:
                    s = str(a).strip()
                    if s:
                        targets.append(s)
    if not targets:
        targets = [""]
    # Deduplicate while preserving order
    seen = set()
    unique: list[str] = []
    for t in targets:
        if t not in seen:
            seen.add(t)
            unique.append(t)
    return unique


def convert_one(row: dict[str, Any], split: str, idx: int, data_source: str) -> dict[str, Any]:
    question = _extract_question(row.get("prompt"))
    if not question.endswith("?"):
        question = question + "?"
    templated = SEARCH_R1_TEMPLATE.format(question=question)
    targets = _extract_targets(row)

    extra_info: dict[str, Any] = {"split": split, "index": idx}
    raw_extra = row.get("extra_info")
    if isinstance(raw_extra, dict):
        for k in ("sample_id", "hops", "gold_answer_short"):
            if k in raw_extra and raw_extra[k] is not None:
                extra_info[k] = raw_extra[k]

    return {
        "data_source": data_source,
        "prompt": [{"role": "user", "content": templated}],
        "ability": "fact-reasoning",
        "reward_model": {"style": "rule", "ground_truth": {"target": targets}},
        "extra_info": extra_info,
    }


def convert_split(in_path: Path, out_path: Path, split: str, data_source: str) -> int:
    import pandas as pd

    logger.info("Reading %s ...", in_path)
    df = pd.read_parquet(in_path)
    records: list[dict[str, Any]] = []
    for idx, row in enumerate(df.to_dict(orient="records")):
        records.append(convert_one(row, split=split, idx=idx, data_source=data_source))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df = pd.DataFrame(records)
    out_df.to_parquet(out_path, index=False)
    logger.info("Wrote %d rows -> %s", len(records), out_path)
    return len(records)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    ap = argparse.ArgumentParser(description="Convert CWQ verl parquet -> Search-R1 format")
    ap.add_argument("--in_train", type=Path, default=Path("data/freebase/verl_cwq/train.parquet"))
    ap.add_argument("--in_test", type=Path, default=Path("data/freebase/verl_cwq/test.parquet"))
    ap.add_argument("--out_dir", type=Path, default=Path("data/freebase/searchr1_cwq"))
    ap.add_argument(
        "--data_source",
        type=str,
        default="hotpotqa",
        help="Search-R1 data_source id. Must be one of their whitelisted sources "
        "for qa_em.compute_score_em to fire. Default: hotpotqa (multi-hop).",
    )
    args = ap.parse_args()

    n_train = convert_split(args.in_train, args.out_dir / "train.parquet", "train", args.data_source)
    n_test = convert_split(args.in_test, args.out_dir / "test.parquet", "test", args.data_source)
    logger.info("Done. train=%d test=%d out_dir=%s", n_train, n_test, args.out_dir)


if __name__ == "__main__":
    main()
