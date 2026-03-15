# KG-Align-RL Changelog

## 2026-02-17: 1.5B Evaluation Analysis & 7B Scale-Up Decision

### Evaluation Results (Job 30998297)

Completed 3-way evaluation (Base vs SFT vs SFT+GRPO) on ConceptNet and HotpotQA:

**ConceptNet (in-domain, 500 samples):**

| Model | EM | F1 | KG Reward | Format |
|-------|----|----|-----------|--------|
| Base | 0.000 | 0.177 | 0.051 | 0.000 |
| SFT | 0.038 | 0.328 | 0.512 | 0.491 |
| SFT+GRPO | 0.014 | 0.369 | 0.616 | 0.500 |

**HotpotQA (transfer, 500 samples):**

| Model | EM | F1 | KG Reward | Format |
|-------|----|----|-----------|--------|
| Base | 0.000 | 0.021 | 0.000 | 0.000 |
| SFT | 0.010 | 0.064 | 0.000 | 0.491 |
| SFT+GRPO | 0.000 | 0.056 | 0.000 | 0.500 |

**Against success criteria:**
1. GRPO reward increases during training: **PASS** (0.97 -> 1.28 peak)
2. SFT+GRPO > SFT on in-domain: **MIXED** (F1 pass, EM regressed 3.8% -> 1.4%)
3. SFT+GRPO > Base on in-domain: **PASS**
4. HotpotQA improvement: **FAIL** (slight regression from SFT)

### Key Findings

**Reward misalignment (Goodhart's Law):** GRPO optimised KG reward (12x improvement) at the expense of answer accuracy. The model learned to produce verbose, KG-entity-rich reasoning but less precise answers. R_answer (token-F1) contributes only ~25% of total reward — not enough to maintain answer quality.

**No cross-domain transfer:** KG reward = 0.0 on all HotpotQA models. The KG reward function is fundamentally domain-specific (ConceptNet entities don't appear in HotpotQA). The GRPO model also hallucinated fake KG paths on HotpotQA.

**1.5B model limitations confirmed:** Consistent with literature (DeepSeek-R1, NeurIPS 2025 papers) showing 1.5B is below the threshold for meaningful cross-domain transfer via RL.

### Research Discussion: Three Critical Questions

#### Q1: Does KG-Only Reward Reduce General Abilities?

Literature review of 15+ papers found:
- On-policy RL (GRPO) forgets significantly less than SFT ("RL's Razor", NeurIPS 2025)
- LoRA provides natural forgetting protection ("LoRA Learns Less and Forgets Less", TMLR 2024)
- But prolonged domain-specific RLVR does cause capability regression (~7.5% on general tasks)
- DeepSeek-R1 and Qwen3 both use post-RL recovery stages to restore general capabilities
- **Action needed**: Add MMLU/ARC-Challenge/GSM8K to evaluation suite

#### Q2: Would Mixture Rewards Work Better?

Literature supports multi-domain reward mixing, but naive summation causes collapse:
- Kansal & Jha showed 4-signal -> 55.21% vs 2-signal -> 82.20%
- GDPO (NVIDIA, 2026) and MO-GRPO (2025) fix this with per-reward normalisation
- "Reasoning Curriculum" (2025) validates "domain-first, then multi-domain" approach
- **Recommendation**: KG-first phase (current), then add general QA rewards using GDPO/MO-GRPO

#### Q3: Is 1.5B Too Small? Will 7B Help?

Evidence strongly favours scaling:
- RLVR does not teach new reasoning — it surfaces existing latent capabilities (NeurIPS 2025)
- 1.5B shows faster overfitting, more reward hacking, less transfer (multiple papers)
- "Aha moment" reliably emerges at 7B+ (SimpleRL-Zoo, DeepSeek-R1)
- Kansal & Jha validated KG approach at 8B/14B, not 1.5B
- NVIDIA ProRL achieved SOTA at 1.5B only with diverse 136K-problem training

### Decision: Scale to 7B

Based on the analysis, scaling to Qwen2.5-7B-Instruct is the single most impactful next step.

**Hardware constraint**: 2x H100 80GB, 72-hour max wall time per job.
**Feasibility**: Confirmed viable — estimated 40-55 GB/GPU peak, ~48-72h for 1000 steps with vLLM colocate TP=2.

### Files Created

- `configs/sft_7b.yaml` — SFT warmup config for 7B
- `configs/grpo_7b.yaml` — GRPO config for 7B (vLLM colocate TP=2)
- `scripts/train_sft_7b.job` — SFT SLURM script (1 GPU, 4h)
- `scripts/train_grpo_7b.job` — GRPO SLURM script (2 GPU, 72h)
- `scripts/train_grpo_7b_pilot.job` — 10-step pilot to test vLLM (2 GPU, 1h)
- `scripts/eval_7b.job` — Evaluation script for 7B
- `src/training/train_grpo.py` — Updated with `vllm_tensor_parallel_size` and `--pilot_steps` flag

### Key Config Changes (1.5B -> 7B)

| Parameter | 1.5B | 7B | Rationale |
|-----------|------|-----|-----------|
| Model | Qwen2.5-1.5B-Instruct | Qwen2.5-7B-Instruct | Scale up |
| per_device_train_batch_size | 2 | 1 | VRAM constraint |
| gradient_accumulation_steps | 8 | 16 | Maintain effective batch (128) |
| use_vllm | false | true (colocate, TP=2) | Generation speed critical at 7B |
| attn_implementation | sdpa | flash_attention_2 | H100 supports FA2 |
| SLURM wall time | 48h | 72h | Longer training |

### Execution Order

1. `sbatch scripts/train_sft_7b.job` — SFT warmup (~1-2h)
2. `sbatch scripts/train_grpo_7b_pilot.job` — Verify vLLM works (~30min)
3. `sbatch scripts/train_grpo_7b.job` — Full GRPO training (~48-72h)
4. `sbatch scripts/eval_7b.job` — Evaluation (~12-20h)

### Post-7B Discussion Topics (Pending)

- Rebalance R_answer weight in reward function (currently ~25%, should be ~50%)
- Add general benchmark evaluation (MMLU, ARC, GSM8K)
- Consider mixture rewards with GDPO/MO-GRPO normalisation
- Model weight averaging as cheap forgetting mitigation
- Evaluate intermediate checkpoints (not just final)

---

## 2026-02-12: GRPO v2 Training (Job 30923716)

- Submitted 3-phase curriculum GRPO training with v2 continuous reward
- Config: dr_grpo, num_iterations=2, temperature=1.0, scale_rewards=false
- Completed in 4h 15m (1000 steps across 3 phases)
- Training metrics healthy: reward 0.97->1.28, entropy >0.5, grad_norm non-zero

## 2026-02-11: Reward v2 Design

- Replaced binary reward with continuous reward for GRPO variance
- v2: R_coverage (0-1.2) + R_order (0-0.5) + R_answer (-0.5 to 0.5, token-F1)
- Fixed v1 issues: binary reward -> zero variance -> zero gradient

## 2026-02-10: SFT Warmup Complete (Job 30896683)

- Switched from base model to Qwen2.5-1.5B-Instruct (base model can't produce EOS)
- SFT warmup completed in 22 minutes
- Adapter saved to outputs/sft-warmup/

## 2026-02-09: Stage 3 KG Reward Function Complete

- Implemented v2 continuous reward in src/rewards/kg_reward.py
- 67 unit tests passing
- TRL-compatible wrapper functions

## 2026-02-07: Stage 2 Data Generation Complete (Job 30636140)

- 16K generated -> 5K filtered -> 4K/500/500 splits + 500 negatives
- Data fix script at src/datagen/fix_stage2_data.py

## 2026-02-05: Stage 1 KG Path Extraction Complete

- 8,000 paths in data/processed/conceptnet_paths.jsonl
- 30 unique relation types, max 14.4% (IsA)
- Hop distribution: ~40% 1-hop, ~37% 2-hop, ~23% 3-hop
