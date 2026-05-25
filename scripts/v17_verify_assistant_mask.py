"""Verify that trajectories_to_dataset_assistant_only produces correct labels.

Loads 3 sample SFT trajectories, applies the new pre-tokenizer, then decodes
only the trainable tokens (labels != -100) — these should consist exclusively of
assistant turn contents (think/search/answer markup) followed by the chat-template
EOS marker. Anything else (system, user, tool_response payloads) is a bug.
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


def main() -> None:
    tok = AutoTokenizer.from_pretrained(LLAMA_MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    records = load_trajectories(CORPUS_PATH)[:3]
    ds = trajectories_to_dataset_assistant_only(records, tok, max_seq_length=2048)

    for i in range(len(ds)):
        rec = ds[i]
        input_ids = rec["input_ids"]
        labels = rec["labels"]
        n_total = len(input_ids)
        n_trainable = sum(1 for l in labels if l != -100)
        n_masked = n_total - n_trainable
        pct = 100.0 * n_trainable / n_total

        print("=" * 70)
        print(f"Trajectory {i}: total={n_total} trainable={n_trainable} ({pct:.1f}%) masked={n_masked}")
        print("=" * 70)

        # Decode trainable-only tokens
        trainable_ids = [tid for tid, lab in zip(input_ids, labels) if lab != -100]
        trainable_text = tok.decode(trainable_ids, skip_special_tokens=False)
        print(f"\n--- TRAINABLE TOKENS DECODED ({len(trainable_ids)} ids) ---")
        print(trainable_text[:2500])
        if len(trainable_text) > 2500:
            print(f"... [truncated, total {len(trainable_text)} chars]")

        # Decode masked tokens to confirm they are system/user/tool content
        masked_ids = [tid for tid, lab in zip(input_ids, labels) if lab == -100]
        masked_text = tok.decode(masked_ids, skip_special_tokens=False)
        print(f"\n--- MASKED TOKENS DECODED ({len(masked_ids)} ids, first 800 chars) ---")
        print(masked_text[:800])
        if len(masked_text) > 800:
            print(f"... [truncated, total {len(masked_text)} chars]")

        # Anomaly check: trainable text should NOT contain the system prompt
        # marker or "<tool_response>" payloads (those are user-role content).
        if "<tool_response>" in trainable_text:
            print("\n!! ANOMALY: trainable text contains <tool_response> — masking bug !!")
        if "knowledge graph reasoning agent" in trainable_text:
            print("\n!! ANOMALY: trainable text contains system prompt — masking bug !!")
        # Trainable text MUST contain <answer> (the terminator we want the model to learn)
        if "<answer>" in trainable_text and "</answer>" in trainable_text:
            print("\n[ok] trainable text contains <answer>...</answer> — terminator will be learned")
        else:
            print("\n!! ANOMALY: trainable text missing <answer>/</answer> — terminator NOT in loss !!")
        print()


if __name__ == "__main__":
    main()
