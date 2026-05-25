"""W17.1e pre-flight: verify that loss_weights = 5.0 lands precisely on
the tokens inside <answer>...</answer> spans (and the immediately-following
<|eot_id|>) and nowhere else.

Bugs we want to catch:
  - Weight boost spills onto non-assistant tokens
  - Open or close tag boundaries mis-identified
  - <|eot_id|> after </answer> not included
"""

from __future__ import annotations

from pathlib import Path

from transformers import AutoTokenizer

import sys
sys.path.insert(0, "/lus/lfs1aip2/scratch/u6gg/ts1201.u6gg/Project/KG_GRPO")
from src_verl.training.sft_multiturn import (
    load_trajectories,
    trajectories_to_dataset_assistant_only,
)

LLAMA_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
CORPUS_PATH = Path(
    "/lus/lfs1aip2/scratch/u6gg/ts1201.u6gg/Project/KG_GRPO/data/freebase/sft_trajectories.jsonl"
)
ANSWER_WEIGHT = 5.0


def main() -> None:
    tok = AutoTokenizer.from_pretrained(LLAMA_MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    records = load_trajectories(CORPUS_PATH)[:3]
    ds = trajectories_to_dataset_assistant_only(
        records, tok, max_seq_length=2048, answer_token_weight=ANSWER_WEIGHT,
    )

    overall_ok = True
    for i in range(len(ds)):
        rec = ds[i]
        input_ids = rec["input_ids"]
        labels = rec["labels"]
        weights = rec.get("loss_weights")
        assert weights is not None, "loss_weights column missing"

        # Collect boosted-token positions
        boosted_positions = [j for j, w in enumerate(weights) if w >= ANSWER_WEIGHT - 0.01]
        n_boost = len(boosted_positions)
        n_total = len(input_ids)
        n_trainable = sum(1 for l in labels if l != -100)

        print(f"\n=== Trajectory {i} ===")
        print(f"total={n_total} trainable={n_trainable} boosted={n_boost} (~{100*n_boost/n_total:.1f}%)")

        if n_boost == 0:
            print("  !! ANOMALY: no tokens boosted")
            overall_ok = False
            continue

        # Verify the boosted-token CONTENT contains <answer> and </answer>
        boosted_ids = [input_ids[j] for j in boosted_positions]
        boosted_text = tok.decode(boosted_ids, skip_special_tokens=False)
        print(f"  decoded boosted region:\n    {boosted_text!r}")
        ok_open = "<answer>" in boosted_text
        ok_close = "</answer>" in boosted_text
        if not ok_open or not ok_close:
            print(f"  !! ANOMALY: boosted region missing tags (open={ok_open} close={ok_close})")
            overall_ok = False
            continue
        # Verify <|eot_id|> is also in the boosted region (terminator should be boosted)
        ok_eot = "<|eot_id|>" in boosted_text
        if not ok_eot:
            print("  WARN: <|eot_id|> NOT in boosted region (terminator not boosted)")
        else:
            print("  [ok] <|eot_id|> in boosted region (terminator IS boosted)")

        # Verify boosted positions are all-labelled (label != -100 for every boosted j)
        labelled = sum(1 for j in boosted_positions if labels[j] != -100)
        if labelled != n_boost:
            print(f"  !! ANOMALY: only {labelled}/{n_boost} boosted positions have non-masked labels — bug in mask/weight alignment")
            overall_ok = False
            continue

        # Verify the boosted region is contiguous and short (~ tag tokens)
        gaps = [boosted_positions[k+1] - boosted_positions[k] for k in range(len(boosted_positions)-1)]
        if any(g > 1 for g in gaps):
            print(f"  WARN: boosted region not contiguous, gaps>{1}: {[g for g in gaps if g>1]}")

        # Sanity: non-boosted assistant tokens should still have weight 1.0
        for j in range(n_total):
            if labels[j] != -100 and j not in boosted_positions:
                if weights[j] != 1.0:
                    print(f"  !! ANOMALY: pos {j} is trainable non-boost, weight={weights[j]} != 1.0")
                    overall_ok = False
                    break

    print()
    print("=" * 50)
    print("OVERALL:", "PASS" if overall_ok else "FAIL")
    print("=" * 50)


if __name__ == "__main__":
    main()
