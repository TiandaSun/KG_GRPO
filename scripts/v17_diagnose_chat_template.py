"""W17.1 ABORT-debug: diagnose Llama-3.1 chat-template handling for the SFT corpus.

Hypothesis: Llama-3.1-8B-Instruct's chat template does NOT handle the `"tool"` role
the way Qwen2.5's does. The SFT trainer (sft_multiturn.py) uses
`apply_chat_template` and explicitly notes that the corpus relies on Qwen2.5's
native `tool` role rendering. If Llama silently drops or mis-renders tool
messages, training data for Llama is malformed and the model never learns to
emit <answer> as a terminator.

This script:
  1. Loads Llama-3.1-8B-Instruct's tokenizer
  2. Loads Qwen2.5-7B-Instruct's tokenizer for comparison
  3. Renders the same SFT trajectory through both
  4. Reports whether Llama preserves `tool` content + how the 4 KG tags tokenize
"""

from __future__ import annotations

import json
from pathlib import Path

from transformers import AutoTokenizer

LLAMA_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
QWEN_MODEL = "Qwen/Qwen2.5-7B-Instruct"

CORPUS_PATH = Path(
    "/lus/lfs1aip2/scratch/u6gg/ts1201.u6gg/Project/KG_GRPO/data/freebase/sft_trajectories.jsonl"
)


def load_first_trajectory() -> list[dict[str, str]]:
    with open(CORPUS_PATH) as f:
        record = json.loads(next(iter(f)))
    return record["trajectory"]


def render(tokenizer, messages: list[dict[str, str]]) -> str:
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )


def main() -> None:
    print("=" * 70)
    print("v17 W17.1 ABORT debug — chat-template diagnosis")
    print("=" * 70)

    llama_tok = AutoTokenizer.from_pretrained(LLAMA_MODEL)
    qwen_tok = AutoTokenizer.from_pretrained(QWEN_MODEL)

    print("\n## Tokenizer special tokens")
    for name, t in (("Llama-3.1-8B-Instruct", llama_tok), ("Qwen2.5-7B-Instruct", qwen_tok)):
        print(f"  {name}: EOS={t.eos_token!r}({t.eos_token_id}) "
              f"PAD={t.pad_token!r}({getattr(t, 'pad_token_id', None)})")

    print("\n## Tokenization of the 4 KG tags + tool_response under Llama-3.1")
    for tag in ("<think>", "</think>", "<search>", "</search>",
                "<response>", "</response>", "<answer>", "</answer>",
                "<tool_response>", "</tool_response>"):
        ids_l = llama_tok(tag, add_special_tokens=False)["input_ids"]
        toks_l = llama_tok.convert_ids_to_tokens(ids_l)
        ids_q = qwen_tok(tag, add_special_tokens=False)["input_ids"]
        toks_q = qwen_tok.convert_ids_to_tokens(ids_q)
        print(f"  {tag:>20s} | Llama -> {ids_l} {toks_l}")
        print(f"  {' ':>20s} | Qwen  -> {ids_q} {toks_q}")

    traj = load_first_trajectory()
    print(f"\n## SFT trajectory[0] role sequence ({len(traj)} messages)")
    for i, m in enumerate(traj):
        snip = m["content"][:80].replace("\n", " ")
        print(f"  [{i}] role={m['role']!r}  content[:80]={snip!r}")

    print("\n## Llama-3.1 rendering of trajectory[0] (last 1500 chars)")
    try:
        rendered_l = render(llama_tok, traj)
        print(f"  total chars: {len(rendered_l)}")
        print("  --- last 1500 ---")
        print(rendered_l[-1500:])
    except Exception as e:
        print(f"  Llama render failed: {type(e).__name__}: {e}")
        rendered_l = None

    print("\n## Qwen2.5 rendering of trajectory[0] (last 1500 chars)")
    try:
        rendered_q = render(qwen_tok, traj)
        print(f"  total chars: {len(rendered_q)}")
        print("  --- last 1500 ---")
        print(rendered_q[-1500:])
    except Exception as e:
        print(f"  Qwen render failed: {type(e).__name__}: {e}")
        rendered_q = None

    print("\n## Tool-role preservation check")
    if rendered_l is not None and rendered_q is not None:
        # Pick a tool-response substring from the trajectory
        tool_msgs = [m for m in traj if m["role"] == "tool"]
        print(f"  Tool messages in trajectory: {len(tool_msgs)}")
        if tool_msgs:
            sample = tool_msgs[0]["content"][:60]
            in_llama = sample in rendered_l
            in_qwen = sample in rendered_q
            print(f"  First-tool-msg sample: {sample!r}")
            print(f"    appears in Llama render: {in_llama}")
            print(f"    appears in Qwen  render: {in_qwen}")

    print("\n## VERDICT")
    if rendered_l is None:
        print("  Llama chat template REJECTED the tool-role messages.")
    elif rendered_l and rendered_q:
        tool_msgs = [m for m in traj if m["role"] == "tool"]
        if tool_msgs:
            sample = tool_msgs[0]["content"][:60]
            if sample not in rendered_l:
                print("  Llama silently DROPPED tool-role content.")
                print("  Fix: add tool_response markup as user-role text in SFT corpus.")
            else:
                print("  Llama preserved tool content — chat template is not the bug.")
                print("  Look elsewhere: training steps too few? eos handling? lr?")


if __name__ == "__main__":
    main()
