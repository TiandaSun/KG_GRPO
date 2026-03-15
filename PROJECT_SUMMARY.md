# KG-Align-RL: Project Summary

> **Cross-Domain Transfer in Knowledge Graph-Aligned Reinforcement Learning for LLM Reasoning**
>
> Reference paper: Kansal & Jha (arXiv:2601.15160)
>
> Last updated: 2026-02-23

---

## 1. Research Idea

### Motivation

Large language models can reason, but they often hallucinate or fail to ground their reasoning in verifiable knowledge. Knowledge graphs (KGs) provide structured, verifiable facts — but LLMs cannot natively access them during generation. The question is: **can we use reinforcement learning to teach an LLM to reason along KG paths, and does that reasoning skill transfer to domains beyond the training KG?**

### Approach

Following Kansal & Jha (arXiv:2601.15160), we implement a pipeline that:

1. **Extracts structured multi-hop paths** from ConceptNet (a commonsense KG)
2. **Generates chain-of-thought QA pairs** that require following these paths
3. **Trains an LLM with GRPO** (Group Relative Policy Optimisation) using a KG path-alignment reward — the model is rewarded for producing reasoning that faithfully walks the KG path before answering
4. **Evaluates** on both in-domain (ConceptNet QA) and out-of-domain (HotpotQA) benchmarks to test cross-domain transfer

The key hypothesis: a model trained to reason over KG paths will develop a general "path-following" reasoning skill that transfers to new domains.

### Pipeline Architecture

```
Stage 1: KG Path Extraction (ConceptNet, CPU)
    ↓
Stage 2: Question Generation + LLM-as-Judge Filtering (Qwen2.5-7B-Instruct)
    ↓
Stage 3: KG Reward Function Development (CPU, unit tests)
    ↓
Stage 4: SFT Warmup — teach <think> format (2 epochs)
    ↓
Stage 5: GRPO Training with Curriculum — 1-hop → 2-hop → 3-hop
    ↓
Stage 6: Evaluation — 3-way comparison (Base vs SFT vs SFT+GRPO)
```

---

## 2. What We Built

### Data Pipeline (Stages 1-2)

| Component | Detail |
|-----------|--------|
| **KG source** | ConceptNet (English, weight >= 2.0, excluding SimilarTo/RelatedTo) |
| **Paths extracted** | 8,000 unique paths (40% 1-hop, 37% 2-hop, 23% 3-hop) |
| **Relation diversity** | 30 types, no single type >14.4% (IsA) |
| **QA generation** | Qwen2.5-7B-Instruct, 2 variants per path → 16,000 raw pairs |
| **Quality filtering** | LLM-as-judge on answerability, faithfulness, naturalness (>= 4/5 on all) |
| **Final dataset** | 4,000 train / 500 val / 500 test + 500 negative examples |
| **Format** | `<think>[step-by-step KG path reasoning]</think>\n[short answer]` |

### Reward Function (Stage 3) — Three Versions

The reward function evolved through three iterations as we discovered and fixed reward hacking:

**v1 (Binary, Feb 09):** Range -1.0 to 1.6
- Binary correctness (+0.1 correct, -1.0 incorrect) + token-level path coverage + repetition penalty
- **Problem:** Binary signal → zero within-group variance → zero GRPO gradient. The -1.0 penalty was 10x the +0.1 reward, crushing exploration.

**v2 (Continuous, Feb 11):** Range -0.5 to 2.2
- R_coverage (0-1.2): bag-of-words token coverage of KG entities
- R_order (0-0.5): sequential ordering of path entities
- R_answer (-0.5 to 0.5): token-F1 based continuous score
- **Problem:** Bag-of-words coverage rewarded mentioning random KG tokens anywhere. The model found a degenerate template `"X is UsedOf Y and Z is UsedOf UsedOf )"` that scored high without real reasoning. Completion length collapsed 284→55 tokens.

**v3 (Answer-Dominant + Anti-Hack, Feb 15):** Range -0.3 to 1.9

| Component | Range | Share of Positive Budget | Description |
|-----------|-------|--------------------------|-------------|
| R_answer | 0 to 1.0 | **56%** | Token-F1 correctness (dominant signal) |
| R_reasoning | 0 to 0.4 | 21% | Structured triple matching (requires both head AND tail entities) |
| R_quality | -0.3 to 0 | Penalty only | Brevity (<15 words), template collapse, n-gram repetition |
| Format | 0 to 0.5 | 26% | `<think>...</think>` tags (reduced from 1.0) |

Key design change: answer correctness went from 23% to 56% of the positive reward budget, making it the dominant training signal.

### Training Configuration (Stages 4-5)

**Model architecture (shared across scales):**
- LoRA with DoRA + rsLoRA (rank 16, alpha 32)
- Target modules: all-linear (q/k/v/o/gate/up/down projections)
- Optimizer: AdamW 8-bit with gradient checkpointing
- Must use **Instruct** model (base model cannot produce EOS token)

**GRPO training specifics:**
- Loss: `dr_grpo` (Dr. GRPO, removes length and std bias)
- `num_iterations: 2` (activates clipping — with `num_iterations=1`, loss appears as 0 but training still works via non-zero grad_norm)
- `scale_rewards: false`, `beta: 0.02` (gentle KL anchor)
- `num_generations: 8`, `temperature: 1.0`
- 3-phase curriculum: 1-hop only (350 steps) → ≤2-hop (350 steps) → all hops (300 steps)

**Scale-specific differences:**

| Parameter | 1.5B | 7B |
|-----------|------|-----|
| Model | Qwen2.5-1.5B-Instruct | Qwen2.5-7B-Instruct |
| Hardware | 1× H100 | 2× H100 (DDP) |
| Batch size | 2 | 1 |
| Grad accumulation | 8 | 16 |
| Attention | SDPA | Flash Attention 2 |
| vLLM | Off | Colocate, TP=1 per rank |
| Effective batch | 128 | 128 |

---

## 3. Changes and Lessons Learned

### Critical Bug Fixes

| Issue | Impact | Fix |
|-------|--------|-----|
| Base model can't produce `<\|im_end\|>` | GRPO hangs (never terminates generation) | Use Instruct model variant |
| `loss=0` in GRPO | Wasted 3 debugging iterations | Expected with `num_iterations=1` — grad_norm proves training works. Set `num_iterations=2` for visible loss. |
| Flash Attention crash on older CUDA | Training fails | Use `attn_implementation: "sdpa"` for 1.5B; FA2 works on H100 for 7B |
| Right-padding | Incorrect loss computation | Set `padding_side="left"` |
| `set -u` in SLURM | Conda activation fails | Remove `set -u` |
| vLLM TP=2 + DDP | NCCL deadlock after 2 steps (job 31061569) | Use TP=1 per rank — each DDP rank gets a local vLLM engine |

### Reward Evolution (v1 → v2 → v3)

The reward function was the most iterated component. Each version was driven by observed training failures:

```
v1 Binary → zero GRPO variance → no learning signal
    ↓ Fix: make reward continuous
v2 Continuous bag-of-words → template collapse / reward hacking
    ↓ Fix: structured matching + answer-dominant + anti-hack penalties
v3 Answer-dominant → stable training, no collapse observed
```

### Key Architectural Decision: 7B Scale-Up

After 1.5B validation revealed Goodhart's Law (reward up but EM down), we reviewed 15+ papers and concluded:

- RLVR doesn't teach new reasoning — it surfaces **latent** capabilities (NeurIPS 2025)
- 1.5B is below the threshold for meaningful transfer; "aha moment" emerges at 7B+ (SimpleRL-Zoo)
- On-policy RL forgets less than SFT ("RL's Razor", NeurIPS 2025)
- LoRA is natural forgetting protection ("LoRA Learns Less and Forgets Less", TMLR 2024)
- Kansal & Jha validated their approach at 8B/14B, not 1.5B

---

## 4. Experiment Results

### 4.1 — 1.5B Validation (Qwen2.5-1.5B-Instruct)

**Training:** SFT warmup (22 min) → GRPO 1000 steps with v2 reward (4h 21min), single H100.

**ConceptNet (in-domain, 500 samples):**

| Model | EM | F1 | KG Reward | Format |
|-------|------|--------|-----------|--------|
| Base | 0.000 | 0.177 | 0.051 | 0.000 |
| SFT | **0.038** | 0.328 | 0.512 | 0.491 |
| SFT+GRPO | 0.014 | **0.369** | **0.616** | 0.500 |

**HotpotQA (transfer, 500 samples):**

| Model | EM | F1 | KG Reward | Format |
|-------|------|--------|-----------|--------|
| Base | 0.000 | 0.021 | 0.000 | 0.000 |
| SFT | **0.010** | **0.064** | 0.000 | 0.491 |
| SFT+GRPO | 0.000 | 0.056 | 0.000 | 0.500 |

**Observations:**
- GRPO reward increased during training (0.97 → 1.28) — pipeline works mechanically
- F1 improved after GRPO (+12.5% relative on ConceptNet), but EM **regressed** (3.8% → 1.4%)
- Goodhart's Law: KG reward improved 12x while answer accuracy dropped
- R_answer was only ~25% of total reward — model optimised coverage over correctness
- No HotpotQA transfer (KG reward = 0.0 for all models — ConceptNet entities absent)
- GRPO model hallucinated fake KG paths on out-of-domain questions

### 4.2 — 7B Scale-Up (Qwen2.5-7B-Instruct)

**Training:** SFT warmup → GRPO 1000 steps with v3 reward on 2× H100 with DDP + vLLM colocate.

**ConceptNet (in-domain, 500 samples):**

| Model | EM | F1 | KG Reward | Format |
|-------|------|--------|-----------|--------|
| Base | 0.000 | 0.162 | 0.054 | 0.000 |
| SFT | **0.020** | 0.364 | 0.558 | 0.489 |
| SFT+GRPO | 0.016 | **0.401** | **0.638** | 0.497 |

**HotpotQA (transfer, 500 samples):**

| Model | EM | F1 | KG Reward | Format |
|-------|------|--------|-----------|--------|
| Base | 0.000 | 0.029 | 0.000 | 0.000 |
| SFT | **0.020** | **0.099** | 0.000 | 0.497 |
| SFT+GRPO | 0.000 | 0.063 | 0.000 | 0.500 |

**Observations:**
- Same Goodhart pattern: GRPO improves F1 and KG reward on ConceptNet, EM slightly regresses
- 7B F1 (0.401) is higher than 1.5B F1 (0.369) — scaling helps in-domain performance
- Training was more stable at 7B: completion length stable (247-294 tokens), no template collapse, `frac_reward_zero_std = 0.0`
- HotpotQA transfer still absent — GRPO actually hurts F1 vs SFT-only (0.063 vs 0.099)
- KG reward = 0.0 on HotpotQA for ALL models at ALL scales — the reward is fundamentally domain-specific

### 4.3 — Comparison Across Scales

| Metric | 1.5B SFT+GRPO | 7B SFT+GRPO | Delta |
|--------|---------------|-------------|-------|
| ConceptNet EM | 0.014 | 0.016 | +14% relative |
| ConceptNet F1 | 0.369 | 0.401 | +8.7% relative |
| ConceptNet KG | 0.616 | 0.638 | +3.6% |
| HotpotQA EM | 0.000 | 0.000 | — |
| HotpotQA F1 | 0.056 | 0.063 | +12.5% relative |
| Training stability | Template collapse observed | Stable, no collapse | Improved |

### 4.4 — Success Criteria Assessment

| Criterion | 1.5B | 7B | Status |
|-----------|------|----|--------|
| GRPO reward increases during training | PASS | PASS | Pipeline works |
| SFT+GRPO > Base on in-domain (F1) | PASS | PASS | RL adds value |
| SFT+GRPO > SFT on in-domain (F1) | PASS | PASS | GRPO improves over SFT |
| SFT+GRPO > SFT on in-domain (EM) | **FAIL** | **FAIL** | Goodhart's Law |
| Any improvement on HotpotQA | **FAIL** | **FAIL** | No cross-domain transfer |

---

## 5. Current Status & Next Steps

### What We've Established

1. **The pipeline works end-to-end** — all 6 stages complete at both 1.5B and 7B
2. **GRPO with KG rewards produces a training signal** — reward curves increase, gradients flow
3. **In-domain F1 improves** — the model gets better at partially matching answers while following KG paths
4. **Cross-domain transfer does not occur** — the ConceptNet KG reward is fundamentally domain-specific
5. **Goodhart's Law is the core challenge** — optimising KG coverage improves F1 but hurts exact-match accuracy

### Open Questions

1. **Reward rebalancing**: The v3 reward (answer-dominant at 56%) was used for 7B training but the EM regression persists. Is R_answer still insufficiently dominant, or is this an inherent limitation of the approach?
2. **General capability retention**: We have not measured MMLU/ARC/GSM8K — does KG-specific GRPO degrade general reasoning?
3. **Mixture rewards**: Could GDPO or MO-GRPO enable combining KG rewards with general QA rewards without signal collapse?
4. **Intermediate checkpoints**: Are earlier GRPO checkpoints (before potential over-optimisation) better than the final one?

### Potential Next Steps

| Action | Expected Impact | Effort |
|--------|----------------|--------|
| Add general benchmarks (MMLU, ARC, GSM8K) | Measure forgetting | Low (eval script change) |
| Evaluate intermediate GRPO checkpoints | Find optimal training length | Low |
| Increase R_answer to 70-80% of budget | May fix EM regression | Medium (retrain) |
| Model weight averaging (SFT + GRPO adapters) | Cheap forgetting mitigation | Low |
| Multi-domain rewards with GDPO/MO-GRPO | Enable KG + general QA jointly | High |
| Domain-adaptive KG reward for HotpotQA | Enable transfer evaluation | High |

---

## 6. Repository Structure

```
src/
  kg/conceptnet_extractor.py          # Stage 1: KG path extraction
  datagen/
    question_generator.py              # Stage 2a: QA generation
    quality_filter.py                  # Stage 2b: LLM-as-judge filtering
    negative_examples.py               # Stage 2c: Corrupted path generation
    fix_stage2_data.py                 # Data quality fix script
  rewards/kg_reward.py                 # Stage 3: Reward functions (v3)
  training/
    sft_warmup.py                      # Stage 4: SFT warmup
    train_grpo.py                      # Stage 5: GRPO training with curriculum
  evaluation/evaluate.py               # Stage 6: Benchmark evaluation

configs/
  sft_warmup.yaml                      # 1.5B SFT config
  sft_7b.yaml                         # 7B SFT config
  grpo_validation.yaml                 # 1.5B GRPO config
  grpo_7b.yaml                        # 7B GRPO config

scripts/
  train_sft_7b.job                     # SFT SLURM job
  train_grpo_7b.job                    # GRPO SLURM job (3-phase curriculum)
  train_grpo_7b_pilot.job              # 10-step pilot (test vLLM)
  eval_7b.job                          # Evaluation SLURM job

outputs/
  sft-warmup/                          # 1.5B SFT adapter
  sft-warmup-7b/                       # 7B SFT adapter
  grpo/phase{1,2,3}_*/                 # 1.5B GRPO adapters
  grpo-7b/phase{1,2,3}_all/           # 7B GRPO adapters
  eval/                                # 1.5B evaluation results
  eval-7b/                             # 7B evaluation results

tests/
  test_kg_reward.py                    # 98 unit tests for reward function
  test_conceptnet_extractor.py         # KG extraction tests
  test_datagen.py                      # Data generation tests
```

---

## 7. Environment & Compute

| Resource | Detail |
|----------|--------|
| **HPC** | Viking (University of York), SLURM scheduler |
| **GPUs** | NVIDIA H100 80GB PCIe |
| **Conda env** | `kg_align` (Python 3.10) |
| **Key libraries** | TRL, PEFT, transformers, vLLM, flash-attn |
| **Scratch** | `/mnt/scratch/users/ts1201/` |

**Compute used (approximate):**

| Stage | 1.5B GPU-hours | 7B GPU-hours |
|-------|---------------|-------------|
| Stage 2: Data generation | 4h (1 GPU) | — (reused) |
| Stage 4: SFT warmup | 0.4h (1 GPU) | ~2h (1 GPU) |
| Stage 5: GRPO training | 4.4h (1 GPU) | ~48h (2 GPU) |
| Stage 6: Evaluation | 7.5h (1 GPU) | 30.4h (1 GPU) |
| **Total** | **~16 GPU-hours** | **~130 GPU-hours** |
