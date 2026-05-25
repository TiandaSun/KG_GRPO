# v17 cross-family generality attempt — findings

> **Status as of 2026-05-12 09:00 UTC — W17.1f-Base + W17.2 baseline COMPLETE.** 5 recipe variants on Instruct ABORTed; Base + xLAM + masked + answer-boost recipe passed the gate (92/100 format-valid) and the **full 3531-Q test gives Llama-Base SFT EM=0.2841, ContEM=0.3483, F1=0.3219, Tools/Q=0.65**. Cross-family generality result is solid (Qwen G2 SFT baseline ~0.36; Llama Base is ~7-8pp below — reasonable cross-family gap). W17.3 GRPO at step ~41/200 (~12.7 min/step; walltime will cap at ~step 112, giving checkpoints at 50 + 100).
>
> This document is the Appendix narrative for cross-family generalisation; the 5 Instruct failures form a key part of the story (Meta's own 8B-Instruct disclaimer for tool use is empirically reproduced).

## Headline finding

**Llama-3.1-8B-Instruct fails to learn multi-turn agentic SFT on our 5K rule-based corpus across 5 recipe variants spanning a 3-decade hyperparameter range and including standard published fixes for known failure modes. The same corpus + identical pipeline produces a working Qwen-2.5-7B agent (Variant G2 baseline: EM=40.0%, format-valid ≈ 100%).**

The proximate failure is a **decision-boundary collapse**: post-SFT, the model emits valid `<think>...</think>\n<search>...</search>` per turn and correctly stops at `<|eot_id|>` (so generation flow control works), but **never** chooses `<answer>` over `<search>` at any decision point. Every CWQ test question runs to the `max_turns=5` ceiling without an answer being emitted; format-valid = 0/100 in all 5 recipes.

## Methodology

Single corpus: `data/freebase/sft_trajectories.jsonl` (5,000 multi-turn trajectories, 6-7 search turns + 1 answer turn each, 100% well-formed terminator). Same eval protocol: greedy decoding, `max_turns=5`, `max_new_tokens=512`, 100-question CWQ test subset. Gate criterion: format-valid ≥ 90% AND tools/Q > 0.5.

## Recipe series (chronological)

| # | Recipe label | Key deltas | Train loss (final) | Format-valid | Diagnosis |
|---|---|---|---|---|---|
| W17.1 | Qwen-style 100 steps | LoRA r=64 / α=128 / DoRA + rslora, lr=2e-5, 1.28 epochs | 0.193 | 0/100 | Loss converged, but model emits chat-template autoregressive artifacts ("user\n\n<tool_response>") inside its own assistant turns. Hypothesised "undertraining". |
| W17.1b | Qwen-style 200 steps | +100 steps over W17.1 | 0.180 (est) | 0/100 | Same failure. Rules out undertraining. |
| W17.1c-xLAM | xLAM cookbook recipe | LoRA r=16 / α=32 vanilla (no DoRA/rsLoRA), lr=5e-5, 3 epochs ≈ 237 steps | 0.247 | 0/100 | Faster convergence (88s/step vs 132s/step), same chat-template autoregressive failure. Rules out LoRA/DoRA-specific instability. |
| W17.1d | xLAM + assistant-only loss masking | -100 labels for system / user / tool tokens. Pre-flight verification confirmed `<answer>` terminator in trainable region. | 0.143 | 0/100 (but **format clean**) | **Chat-template mimicking FIXED** — outputs no longer contain "user\n\n<tool_response>" leakage; structure is clean. Failure mode shifts: model now correctly stays in assistant role, but loops `think → search → tool → think → search → tool` forever. Diagnosed as rare-decision class imbalance (1 answer turn vs 6-7 search turns per trajectory). |
| W17.1e | + 5× answer-token loss boost | Per-token loss weight = 5.0 on tokens inside `<answer>...</answer><\|eot_id\|>` spans (intersected with assistant-only mask). String-level boundary finder, 2 verification jobs confirmed correctness. | 0.260 (different baseline due to weighting) | 0/100 | **Token-level fix DID work** — training-time loss on `<answer>` tokens reweighted as expected, model rapidly learns to predict them in-context. But **inference behavior is unchanged**: ContEM=0.12 vs W17.1d's 0.13. Model now knows `<answer>` as a token but never **decides** to emit it. Rules out rare-class imbalance as a sufficient explanation. |
| W17.1f | Llama-3.1-8B **Base** (no RLHF prior) | Same as W17.1e + model_name=meta-llama/Llama-3.1-8B (Base weights) + Instruct tokenizer for chat template | (xLAM-like) | **92/100 ✅** | **PASS**. Tools/Q=0.66 (vs Instruct's 5.0 stuck-loop), EM=0.35 on 100-Q sanity, ContEM=0.45. Cross-family generality confirmed. The Instruct vs Base distinction is the load-bearing change — recipe held constant from W17.1e. |

## Why this is interesting (not just a null result)

1. **Cross-architecture reproducibility ablation**: Same dataset, same pipeline, same hyperparameter family — Qwen-2.5-7B learns easily; Llama-3.1-8B-Instruct does not. This is a **decision-boundary phenomenon, not a token-imbalance or recipe phenomenon**: the model emits each individual assistant action correctly but cannot make the rare "I have enough information now" judgement.

2. **Falsifies the rare-class-imbalance hypothesis** at the token level. W17.1e successfully boosted `<answer>` token gradient 5×; training loss profile changed accordingly; inference behavior did not change. This means the bottleneck is the **conditional probability gate** — the model knows the answer token, but the contextual signal "this is the right moment" is what's missing.

3. **Matches Meta's own documentation**: Llama-3.1-8B-Instruct model card explicitly states the 8B variant *"cannot reliably maintain a conversation when tool definitions are included in the prompt"* and recommends 70B+ for tool-use applications. Our 5-recipe ABORT series is empirical confirmation of that disclaimer.

4. **Matches the Search-R1 design choice**: A peer multi-turn agentic-RL project on verl chose Llama-3.1-Base over -Instruct for the same reasons; we replicate the failure mode that motivated their choice.

## Implications for the paper

**If W17.1f Base passes (any non-zero format-valid)**:
- Cross-family generality result for the Appendix
- One line in §6 noting the cross-architecture phenomenon
- Limitations note: "Cross-family replication required reverting to the Base checkpoint (no RLHF); Llama-3.1-8B-Instruct's instruction-tuning prior actively resisted multi-turn agentic SFT on our 5K corpus across 5 recipe variants, consistent with Meta's own 8B-Instruct tool-use disclaimer."

**If W17.1f Base also fails**:
- Strong rebuttal-only narrative. Reviewer concerns about "single-family generality" can be answered with: "We attempted Llama-3.1-8B replication across 6 SFT recipes (Appendix). All converged in training loss but failed format-emission at inference. We attribute this to a Llama-specific decision-boundary failure on our corpus scale (5K trajectories), consistent with Meta's own documentation that the 8B-Instruct 'cannot reliably maintain a conversation when tool definitions are included in the prompt'. Successful Llama-3.1-8B agentic-SFT in published work (xLAM-2, ToolACE-2) uses 30-60K trajectory corpora — 6-12× ours."

**One-sentence main-paper insertion (either branch)**:
"Cross-family generalisation was tested on Llama-3.1-8B; results across 6 recipe variants are reported in Appendix X."

## Compute & timeline summary

- ~40 GPU-hr spent across W17.1 through W17.1e (5 SFT runs each ~6h on 4×GH200)
- ~10 GPU-hr remaining budget after W17.1f (~6h)
- Total v17 budget consumed: ~50 / 150 GPU-hr; 100 GPU-hr remain for any subsequent rebuttal work

## Files

| Artefact | Path |
|---|---|
| Gate reports (per recipe) | `_handoff/data/v17_llama/W17_1{,b,c_xlam,d_masked,e_reweighted}_GATE_REPORT.md` |
| Failure trajectories (100 each) | `_handoff/data/v17_llama/w17_1{...}_gate_trajectories/step_0/trajectories.json` |
| Per-sample eval results | `_handoff/data/v17_llama/w17_1{...}_gate_per_sample/` |
| Research findings (literature on Llama agentic SFT) | `_handoff/data/v17_llama/RESEARCH_LLAMA_TOOLUSE_SFT.md` |
| Token-mask verification | `scripts/v17_verify_assistant_mask.py`, `v17_verify_answer_weight.py` |
| Final attempt (Base) | running as job 4542069 (results in `W17_1f_base_GATE_REPORT.md`) |
