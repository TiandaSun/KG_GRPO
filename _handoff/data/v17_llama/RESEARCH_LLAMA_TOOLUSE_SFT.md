# Llama-3.1-8B Multi-turn Agentic SFT — Research Findings (2026-05-09)

Compiled by research subagent for v17 W17.1 ABORT debugging. Sources cited inline.

## Executive summary

**Top-3 most likely root causes (ranked by evidence)**

1. **Undertraining, not Llama-specific tokenizer pathology.** Three independent published recipes for Llama-3.x agentic/function-calling SFT use **3-10× more compute** than ours: xLAM cookbook (lr=1e-4, ~1000 steps, eff. batch 128, LoRA r=16, 3 epochs); Tulu-3-8B-SFT (lr=5e-6 over 2 full epochs of large corpus); FireAct shows Llama2-7B has a **phase transition between 200 and 500 trajectories**. Qwen2.5 just learns the format faster.
2. **Llama-3 family has documented "fine-tuned model never emits EOS/terminator" failure** — untrained reserved-special-token embeddings + multi-EOS confusion (`128001`/`128008`/`128009`). Same dynamic applies to our synthetic `<answer>` terminator.
3. **DoRA + rsLoRA + bf16 + Llama-3.1 has documented numerical-instability fingerprint** (HF forum bf16 thread). Doesn't always present as NaN — can corrupt rare-token gradients silently.

**Top-3 actionable fixes (ranked by expected impact)**

1. **Raise LR to 5e-5 — 1e-4 and run 3 epochs.** Matches xLAM cookbook recipe (Llama-3.1-8B function calling, known to work).
2. **Drop DoRA + rsLoRA, run vanilla LoRA r=16, α=32.** xLAM does NOT stack these.
3. **Up-weight loss on `<answer>...</answer>` tokens 2-3×** (or duplicate final-turn examples). Directly attacks rare-class imbalance.

## Llama-vs-Qwen recipe deltas (published work)

| Hyperparameter | Our Llama (W17.1) | xLAM Llama-3.1-8B (HF cookbook) | Tulu-3-8B-SFT (allenai) | Our Qwen (works) |
|---|---|---|---|---|
| LR | **2e-5** | **1e-4** (5× higher) | 5e-6 (full FT) | 2e-5 |
| Epochs | 1.28 (100 steps) | 3, max_steps=1000 | 2 | 2.0 |
| LoRA rank | 64 | **16** | full FT | 64 |
| LoRA alpha | 128 | **16** (α=r, not 2r) | — | 128 |
| DoRA + rsLoRA | yes | **no** | — | yes |
| Effective batch | 64 | **128** | 16 | 64 |
| Sched | cosine | **linear** | linear | cosine |
| Warmup | 5% | 10% | yes | 5% |

## Concrete recipes that work

| Project | LR | Epochs | LoRA | Notes |
|---|---|---|---|---|
| xLAM function-calling | 1e-4, linear | max_steps=1000 (~3 ep) | r=16, α=16, dropout 0.05, target = q,k,v,o,gate,down,up | QLoRA 4-bit, batch 128, bf16, FA2, pad ≠ EOS |
| FireAct (Llama2-7B) | not pinned | 30 epochs (LoRA, micro-bs 8, seq 512) | LoRA on Llama2 | **500 trajectories = phase transition** |
| AgentTuning / AgentInstruct | small (full FT) | 1 epoch + 1:1 ShareGPT mix | full FT | Mix-with-general-instructions critical |
| Tulu-3-8B-SFT | 5e-6 | 2 | full FT | Confirms Llama-3.1 likes ≥ 2 epochs |
| AT²PO / SWE-RL multi-turn | 5e-6, AdamW wd=0.1, 10-step warmup, cosine | varies | full FT | Lower LR for SWE agents but full FT |

## Custom-token discipline (consensus)

- Adding `<answer>` to `additional_special_tokens` is **NOT** the standard fix. Most agentic-SFT projects (FireAct, xLAM, Search-R1) treat reasoning tags as ordinary string tokens.
- New tokens require `model.resize_token_embeddings()` and introduce LoRA instability. Astronomer/Llama-3-8B-Special-Tokens-Adjusted exists specifically because untrained-special-token issue is real.
- Llama-3.1 has 256 `<|reserved_special_token_N|>` slots — repurposing these is a less-risky alternative if you really need new tokens.

## Failure-mode evidence

- **unsloth#416 / HF Llama-3 #142**: fine-tuned Llama-3 keeps generating until max_new_tokens, EOS never produced. Causes: untrained `<|eot_id|>` embedding + `pad==eos` masking out EOS positions.
- **FireAct §4.3**: "Llama models cannot learn the ReAct format using 100 or 200 samples". Phase transition at 500.
- **Token-weighting literature** (NAACL 2025): rare semantic tokens are systematically underrepresented; standard CE loss preferentially learns majority class. Our `</answer>` is ~5× rarer than `</search>` per trajectory.

## Recommended actions if W17.1b (200 steps) fails

1. **Tier 1 single change**: raise LR to 5e-5, run 3 epochs (~235 steps), unchanged otherwise. If this fixes it: diagnosis = "Qwen LR too low for Llama".
2. **Tier 1**: drop DoRA + rsLoRA, use vanilla LoRA r=16 α=32 dropout 0.05. Combine with #1 for xLAM-mirror recipe.
3. **Tier 2**: up-weight `<answer>`/`</answer>` token loss 2-3× OR duplicate final-turn examples for curriculum oversample.
4. **Tier 2**: add `<answer>` / `</answer>` to `additional_special_tokens`, resize embeddings, put `embed_tokens` in `modules_to_save`, init new rows as mean of existing embeddings.
5. **Tier 3 last resort**: switch to Llama-3.1's `<|reserved_special_token_N|>` tags (embeddings already exist).
6. **Tier 3 expensive**: full SFT, lr=5e-6, 2 epochs, mirror Tulu-3-8B-SFT.

**Will 200 steps be enough?** Marginal — at the lower edge of "works" range. If W17.1b emits 30-60% `<answer>`: token-reweighting (fix #3). If still 0%: LR/PEFT changes (#1+#2).

## Diagnostic to run BEFORE next change

- Log per-position loss on `<answer>` tokens specifically (vs aggregate). Sub-aggregate = OK; way-higher-than-aggregate confirms class imbalance.
- Greedy-decode T=0 on training-set examples; if `<answer>` token probability is low at the gold position, undertraining is confirmed.

## Sources

- [FireAct (arXiv:2310.05915)](https://arxiv.org/abs/2310.05915), [FireAct GitHub](https://github.com/anchen1011/FireAct)
- [HF Cookbook: function calling on xLAM](https://huggingface.co/learn/cookbook/en/function_calling_fine_tuning_llms_on_xlam)
- [xLAM GitHub (Salesforce)](https://github.com/SalesforceAIResearch/xLAM)
- [Llama-3.1-Tulu-3-8B-SFT card](https://huggingface.co/allenai/Llama-3.1-Tulu-3-8B-SFT), [Tulu 3 blog](https://allenai.org/blog/tulu-3-technical)
- [unsloth#416](https://github.com/unslothai/unsloth/issues/416)
- [HF Llama-3-8B-Instruct discussion #142](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct/discussions/142)
- [HF forum: bf16 + Llama-3.1-8B + LoRA/DoRA instability](https://discuss.huggingface.co/t/bf16-training-instability-with-llama-3-1-8b-lora-dora-peft/170326)
- [astronomer/Llama-3-8B-Special-Tokens-Adjusted](https://huggingface.co/astronomer/Llama-3-8B-Special-Tokens-Adjusted)
- [Agent-FLAN (arXiv:2403.12881)](https://arxiv.org/html/2403.12881v1)
- [How to Train Your LLM Web Agent (arXiv:2507.04103)](https://arxiv.org/html/2507.04103)
- [Token Weighting for LLM (NAACL 2025)](https://aclanthology.org/2025.findings-naacl.79.pdf)
- [Llama 3.1 prompt format docs](https://www.llama.com/docs/model-cards-and-prompt-formats/llama3_1/)

## Decision tree

```
W17.1b (200 steps) result
  ├─ format-valid >= 90% -> W17.2 full SFT then GRPO (proceed)
  ├─ format-valid 30-60% -> W17.1c-rare-token: token reweighting + curriculum
  └─ format-valid 0%     -> W17.1c-xlam: lr=5e-5, vanilla LoRA r=16 α=32, 3 epochs
```
