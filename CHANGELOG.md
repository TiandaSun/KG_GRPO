# KG-Align-RL Changelog

Manual changelog for the KG-Align-RL project. Since the project is not yet on GitHub, this file tracks all code created, modified, and key decisions made.

---

## [2026-02-15] Reward v3: Answer-Dominant Scoring with Anti-Hack Penalties

**This is a major rewrite of the reward function**, motivated by reward hacking observed in the first GRPO run (Job 30923716). The model found a degenerate shortcut that scored high on the v2 reward without doing any real reasoning.

### What went wrong with v2

During GRPO training with v2 reward, we observed classic **reward hacking via template collapse**:

1. **Symptom**: KG reward improved (0.42 → 0.78), but Exact Match *dropped* (3.8% → 0.6%). The model got "better" at the reward while getting worse at the actual task.

2. **Root cause**: v2's bag-of-words token coverage (R_coverage, range 0–1.2) rewarded mentioning individual KG tokens *anywhere* in the output. The model discovered a degenerate template — `"X is UsedOf Y and Z is UsedOf UsedOf )"` — that achieved high token overlap scores without any real reasoning. This is because bag-of-words matching doesn't distinguish between "dog IsA animal" (coherent reasoning) and "dog dog IsA animal UsedOf" (word salad).

3. **Training dynamics** (from Job 30923716 logs):
   - Completion length collapsed: 284 → 55 tokens (model learned brevity = efficiency)
   - Entropy collapsed: 0.706 → 0.187 (mode collapse to single template)
   - KG reward *increased*: 0.457 → 1.03 (reward hacking, not improvement)

4. **Reward budget was misaligned**: In v2, the answer signal was only 23% of the positive reward budget (R_answer range -0.5 to 0.5, out of total -0.5 to 2.2). The model rationally optimised the larger, easier-to-game coverage signal instead.

### Why these specific v3 changes

**Insight from "Note on GRPO Training.md"**: A practitioner documented 20+ iterations of GRPO training and concluded "问题定义 > 数据质量 > 算法选择" (Problem Definition > Data Quality > Algorithm Choice). Their key findings that directly apply:

- **Reward ceiling must reach 1.0 with sufficient discrimination** — their model plateaued because max score was 0.8 with no discrimination between good outputs.
- **`frac_reward_zero_std` is the #1 monitoring metric** — if all generations score the same, GRPO gets zero gradient. Our v2 reward was *too* generous on coverage, making most outputs score similarly.
- **Targeted penalties for specific error modes beat generic algorithm changes** — they added a -0.3 penalty for one specific tool misuse pattern and got +6% improvement.
- **Correct answer must dominate** — their final working config had answer correctness as 50%+ of the reward budget.

### v3 design decisions

| Component | v2 Range | v3 Range | % of positive budget | Rationale |
|-----------|----------|----------|---------------------|-----------|
| R_answer | -0.5 to 0.5 | **0 to 1.0** | **56%** (was 23%) | Make correctness dominant. Model should prioritise getting the right answer over gaming coverage. Token-F1 scoring retained for continuous signal. |
| R_reasoning | 0 to 1.7 (bag-of-words) | **0 to 0.4** (structured triples) | **21%** (was 55%) | Replace gameable bag-of-words with structured triple matching. Now requires both head AND tail entities of each KG triple to appear. Much harder to hack: "dog IsA animal" requires mentioning dog+animal together, not just sprinkling tokens. |
| R_quality | N/A | **-0.3 to 0** | Penalty only | Directly penalise the three failure modes observed in v2: brevity (<15 words), template collapse ("UsedOf" repeated), and n-gram repetition (>50% repeated trigrams). |
| Format | 0 to 1.0 | **0 to 0.5** | **26%** (was 45%) | Model already learned `<think>` format from SFT (98.2% compliance). Reduce reward to avoid format dominating answer signal. |
| **Total** | **-0.5 to 2.2** | **-0.3 to 1.9** | | Tighter range, answer-dominant, harder to hack |

**Why structured triple matching is harder to hack**: v2's bag-of-words scoring counted individual token hits. If a KG path contained `["dog", "IsA", "animal"]`, mentioning "dog" and "animal" anywhere — even in unrelated contexts — scored the same as "A dog is an animal." v3 requires **both head and tail entities of each triple** to appear in the think block, with bonus for the relation name. The model must demonstrate awareness of entity *pairs*, not just individual tokens.

**Why anti-hack penalties work**: Following the GRPO Training note's advice on targeted penalties, we added three specific detectors for failure modes observed in v2 outputs:
- **Brevity penalty** (-0.15 if <15 words): Catches the length collapse pattern where model output shrank from 284→55 tokens
- **Template collapse** (-0.15 if "UsedOf"/"TypeOf"/"KindOf" repeated ≥2 times): Catches the exact degenerate pattern observed in v2
- **N-gram repetition** (-0.1 if trigram unique ratio <0.5): Catches word-salad outputs that repeat phrases

**Why beta changed from 0.0 to 0.02**: With beta=0.0, there was no KL penalty anchoring the policy to the SFT checkpoint. The model was free to drift arbitrarily far, which is how it found the degenerate template. A gentle beta=0.02 applies a small penalty for deviating from the SFT policy, keeping the model in the neighbourhood of coherent outputs while still allowing RL optimisation.

### Files modified

- `src/rewards/kg_reward.py` (309 → 681 lines) — **complete rewrite** of reward logic
  - New: `_split_output()` — separates think-block from final answer
  - New: `_compute_triple_score()` — structured triple matching (replaces bag-of-words)
  - New: `_compute_quality_penalties()` — anti-hack penalty detection
  - Modified: `compute_kg_reward()` — new 3-component scoring (R_answer + R_reasoning + R_quality)
  - Modified: `compute_format_reward()` — max reduced from 1.0 to 0.5
  - Modified: `validate_rewards()` — added Test 4 for template collapse detection
  - Preserved: `_compute_answer_score()`, `_compute_path_order()`, `check_answer()`, `_extract_text()`, `kg_reward_func()`, `format_reward_func()` — backward-compatible GRPOTrainer wrappers
- `tests/test_kg_reward.py` (336 → 916 lines) — **complete rewrite**, 98 tests
  - New: `TestSplitOutput` (5 tests), `TestComputeTripleScore` (10 tests), `TestComputeQualityPenalties` (9 tests), `TestAntiHack` (4 tests)
  - Updated: `TestComputeFormatReward` (expectations 1.0→0.5), `TestComputeKGReward` (new reward ranges), `TestGRPOWrappers` (format reward 1.0→0.5)
  - Preserved: `TestCheckAnswer`, `TestComputeAnswerScore`, `TestComputePathOrder`, `TestExtractText`, `TestRewardStats`
- `configs/grpo_validation.yaml` — `beta: 0.0` → `beta: 0.02`
- `src/training/train_grpo.py` — default `beta` in `GRPOTrainingConfig` changed from 0.0 to 0.02

**Test results**: 98/98 passing in 0.16s (CPU only, no GPU required).

---

## [2026-02-14] Stage 6: Evaluation (First Run)

**SLURM Job**: 30975628 (`kg_eval`)
**Node**: gpu22.viking2.yor.alces.network (1x NVIDIA H100 PCIe)
**Wall time**: 7h 33m 26s (of 14h requested)
**Memory**: 4.96 GB / 64 GB

**What this evaluates**: 3-way comparison (Base vs SFT-only vs SFT+GRPO) on two benchmarks, isolating the contribution of each training stage.

### Results: ConceptNet QA (500 samples, in-domain)

| Model | EM | Token F1 | KG Reward | Format Reward |
|-------|-----|----------|-----------|---------------|
| Base (Qwen2.5-1.5B-Instruct) | 0.0% | 17.65% | 0.26 | 0.00 |
| SFT-only | **3.8%** | 32.76% | 0.42 | **0.98** |
| SFT+GRPO | 0.6% | **36.26%** | **0.78** | 1.00 |

### Results: HotpotQA (500 samples, near-transfer)

| Model | EM | Token F1 | KG Reward | Format Reward |
|-------|-----|----------|-----------|---------------|
| Base | 0.0% | 2.06% | 0.0 | 0.0 |
| SFT-only | **1.0%** | **6.42%** | 0.0 | **0.98** |
| SFT+GRPO | 0.0% | 4.36% | 0.0 | 1.0 |

### Analysis

**Validation criteria check** (from MINIMAL_VALIDATION_PLAN.md):
1. GRPO reward increases during training — **PASS** (0.42 → 0.78)
2. SFT+GRPO > SFT-only on in-domain QA — **MIXED** (Token F1 improved 32.76→36.26, but EM dropped 3.8→0.6)
3. SFT+GRPO > Base on in-domain QA — **PASS** (both EM and F1 improved)
4. Any improvement on HotpotQA — **FAIL** (SFT+GRPO worse than SFT-only, expected for 1.5B)

**Key finding**: GRPO improved KG path alignment but *degraded* answer quality — classic reward hacking. The model optimised the easy-to-game coverage signal instead of answer correctness. This directly motivated the Reward v3 rewrite (see entry above).

**Files created**:
- `src/evaluation/evaluate.py` (588 lines) — 3-way benchmark evaluation with EM, F1, KG reward, format reward metrics

**Evaluation outputs** (`outputs/eval/`):
- `conceptnet_summary.json`, `hotpotqa_summary.json` — aggregate metrics
- `conceptnet_{base,sft,sft+grpo}_samples.json` — 20 qualitative samples per model
- `hotpotqa_{base,sft,sft+grpo}_samples.json` — 20 qualitative samples per model

**SLURM script**: `scripts/eval.job` (54 lines)

---

## [2026-02-13] Stage 5: GRPO Training v2 (with continuous rewards)

**SLURM Job**: 30923716 (`kg_grpo_train`)
**Node**: gpu24.viking2.yor.alces.network (1x NVIDIA H100 PCIe)
**Wall time**: 4h 21m 46s (of 48h requested)
**Memory**: 7.11 GB / 64 GB

### Motivation: Why v2 reward was needed

The first GRPO run (v1, Job 30918983) was cancelled after observing:
- Binary correctness (+0.1/-1.0) made reward effectively binary → zero within-group variance → zero GRPO gradient
- The -1.0 penalty was 10x the +0.1 reward → crushed exploration
- Path coverage + repetition penalty rewarded brevity → completion length collapsed 212→25 tokens

**v2 fix**: Replace binary scoring with continuous signals:
- R_coverage (0–1.2): bag-of-words token overlap with KG path
- R_order (0–0.5): sequential entity ordering bonus
- R_answer (-0.5 to 0.5): token-F1 based continuous scoring (replaces binary +0.1/-1.0)
- Total range: -0.5 to 2.2

**Config changes from v1**:
- `loss_type: "dr_grpo"` — removes length bias + std bias (Dr. GRPO, arXiv:2503.20783)
- `num_iterations: 2` — activates clipping, makes loss visible (was 1, which caused loss=0 red herring)
- `temperature: 1.0` — more diverse generations for GRPO variance (was 0.7)
- `scale_rewards: false` — removes difficulty bias
- `mask_truncated_completions: true` — excludes truncated completions from loss

### Training results

3-phase curriculum completed successfully:

| Phase | Data | Steps | Final Reward |
|-------|------|-------|-------------|
| Phase 1 (1-hop) | 1,581 samples | 300 | ~1.2 |
| Phase 2 (≤2-hop) | 3,071 samples | 300 | ~1.6 |
| Phase 3 (all) | 4,000 samples | 300 | ~2.0 |

**Training dynamics** (revealed reward hacking in hindsight):
- Reward increased: 0.457 → 1.03 → 2.031 (looked good at the time)
- Completion length collapsed: 284 → 55 tokens (should have been a red flag)
- Entropy collapsed: 0.706 → 0.187 (mode collapse)
- `frac_reward_zero_std`: 0.0–0.1 (healthy — not a zero-variance problem)

**Note on loss=0 red herring**: With `num_iterations=1` and `beta=0.0`, TRL computes `log_ratio = logp - logp.detach()` → ratio = 1.0 → loss ≈ 0. This is *expected*, not a bug. `grad_norm` being non-zero (1–9) proves gradients ARE flowing. Confirmed by TRL Issues #2703, #3757, Open-R1 #239. We wasted 3 iterations investigating this before understanding it. Fix: `num_iterations: 2`.

**Output**: LoRA adapters at `outputs/grpo/phase{1,2,3}_*/`

---

## [2026-02-12] Stage 5: GRPO Training v1 (cancelled — binary reward failure)

**SLURM Job**: 30918983 (`kg_grpo_train`)
**Node**: gpu22.viking2.yor.alces.network (1x NVIDIA H100 PCIe)
**Wall time**: 2h 19m (cancelled manually)
**Status**: CANCELLED

### Why it was cancelled

The v1 reward function (from Stage 3 scaffold) used Kansal & Jha's original design: asymmetric binary correctness (+0.1 correct, -1.0 incorrect). This caused three compounding problems:

1. **Zero within-group variance**: Binary reward meant all completions in a GRPO group scored either +0.1 or -1.0. When all 8 generations got the same score, advantage = 0, gradient = 0. GRPO *requires* reward variance within groups to learn.

2. **10x penalty asymmetry**: The -1.0 penalty was 10x larger than the +0.1 reward. This crushed exploration — the model quickly learned to produce safe, short outputs to avoid the severe penalty.

3. **Completion length collapse**: Path coverage and repetition penalty inadvertently rewarded brevity. Shorter outputs have fewer repeated tokens → higher repetition penalty score. Output length dropped from 212 → 25 tokens within a few hundred steps.

**Lesson learned**: The Kansal & Jha reward design was validated on larger models (7B+) with different training dynamics. At 1.5B scale, the binary signal provides insufficient gradient signal for GRPO. Continuous scoring is essential.

---

## [2026-02-12] Stage 4: SFT Warmup

**SLURM Job**: 30896683 (`kg_sft_train`)
**Node**: gpu22.viking2.yor.alces.network (1x NVIDIA H100 PCIe)
**Wall time**: 22m 44s (of 4h requested)
**Queue time**: 7h 53m
**Memory**: 4.90 GB / 64 GB

### Motivation

DeepSeek-R1 showed that small models suffer severe cold-start problems with pure RL: without SFT warmup, the model produces messy, unreadable reasoning. At 1.5B parameters, this is worse than at larger scales. SFT warmup teaches the `<think>...</think>` CoT format before GRPO takes over.

### Key decision: Instruct model, not Base

Initially planned to use `Qwen/Qwen2.5-1.5B` (base model), but discovered the base model **never produces `<|im_end|>` (token 151645)**. During GRPO generation, completions would hit `max_completion_length` instead of terminating naturally, breaking the training loop. Fix: switched to `Qwen/Qwen2.5-1.5B-Instruct`, which has the chat template and EOS behaviour built in.

### Training results

| Metric | Value |
|--------|-------|
| Final loss | 0.5934 |
| Token accuracy | 80.71% |
| Trainable params | 19.1M / 1.56B (1.22%) |
| LoRA config | DoRA + rsLoRA, r=16, alpha=32, all-linear targets |
| `<think>` format compliance | 98.2% on validation set |

**Files created**:
- `src/training/sft_warmup.py` (306 lines) — SFT training with TRL's `SFTTrainer`, DoRA+rsLoRA adapter, Qwen2.5-1.5B-Instruct
- `configs/sft_warmup.yaml` (39 lines) — SFT hyperparameters
- `scripts/train_sft.job` (47 lines) — SLURM job script

**Output**: LoRA adapter at `outputs/sft-warmup/` (73 MB adapter_model.safetensors)

---

## [2026-02-10] Stage 3 v2: KG Reward Function (continuous rewards + GRPOTrainer integration)

### Motivation

The Stage 3 scaffold (Feb 03) implemented Kansal & Jha's original binary reward design. Before Stage 5, we rewrote it to:
1. **Add continuous scoring**: Token-level F1 for answer correctness instead of binary +0.1/-1.0, so GRPO gets gradient signal even from partially-correct outputs.
2. **Add path ordering reward**: Bonus for mentioning KG entities in the correct sequential order (R_order, 0–0.5), rewarding actual chain-of-thought reasoning rather than unordered token mention.
3. **Integrate with TRL's GRPOTrainer API**: Created `kg_reward_func()` and `format_reward_func()` wrappers that accept `(completions, **kwargs)` signatures required by `GRPOTrainer.reward_funcs`.

### Reward v2 design

| Component | Range | Description |
|-----------|-------|-------------|
| R_coverage | 0 to 1.2 | Bag-of-words token overlap with KG path entities + relations |
| R_order | 0 to 0.5 | Sequential entity ordering score |
| R_answer | -0.5 to 0.5 | Token-F1 based answer correctness |
| Format reward | 0 to 1.0 | `<think>...</think>` tag presence + non-empty reasoning |
| **Total** | **-0.5 to 2.2** | |

**Files modified**:
- `src/rewards/kg_reward.py` — rewritten with continuous scoring, GRPOTrainer wrappers, `RewardStats` tracking, `validate_rewards()` self-test
- `tests/test_kg_reward.py` — expanded to 67 tests covering all v2 components

**Note**: This v2 design was later found to be hackable (see GRPO v2 entry above and Reward v3 entry). The bag-of-words coverage signal was too easy to game.

---

## [2026-02-10] Stage 5 + Stage 6: Training and Evaluation Code

**Files created**:
- `src/training/train_grpo.py` (496 lines) — GRPO training with curriculum support
  - 3-phase curriculum: 1-hop → ≤2-hop → all hops
  - Loads SFT adapter as starting point
  - Automatic dataset splitting by hop count
  - Configurable via `configs/grpo_validation.yaml`
  - Dr. GRPO loss type, 8-bit AdamW, gradient checkpointing
- `configs/grpo_validation.yaml` (68 lines) — GRPO hyperparameters with curriculum phases
- `scripts/train_grpo.job` (69 lines) — SLURM job script with DDP auto-detection
- `scripts/eval.job` (54 lines) — SLURM evaluation job script

---

## [2026-02-05] Stage 2 Data Fix

**Problem**: Stage 2 output had two data quality issues:
1. 11.7% of records (467/4000 train) had broken `<think>` tags — missing angle brackets (`think `, `think\n`, `Think -`, `think:`, `think>`) or truncated outputs missing `</think>`.
2. 65% of records had overly long `gold_answer_short` (mean 233 chars). The generator produced full-sentence answers instead of concise key phrases, which would break the reward function's substring matching.

**Fix**: Created `src/datagen/fix_stage2_data.py` — a post-processing script that:
- Repairs broken `<think>` tags via regex pattern matching (covers 5+ broken patterns).
- Replaces long `gold_answer_short` with the last entity of the KG path (mean 11.6 chars post-fix).
- Strips "Answer:" prefixes from already-short values.
- Creates `.bak` backups before modifying files.
- Runs built-in validation after fixing.

**Results**:
| Metric                         | Before  | After    |
|--------------------------------|---------|----------|
| `<think>` + `</think>` tags   | 88.3%   | **100%** |
| `gold_answer_short` <= 50 chars| 23.3%   | **100%** |
| `gold_answer_short` mean len   | 233 ch  | **11.6 ch** |
| Hop distribution               | 40/37/23| unchanged|
| Record counts                  | 4K/500/500+500neg | unchanged |

**Files created**:
- `src/datagen/fix_stage2_data.py` (278 lines)

**Files modified** (in-place, backups retained):
- `data/processed/conceptnet_qa_train.jsonl` (4,000 records — 467 think fixes, 3,384 gold_answer fixes)
- `data/processed/conceptnet_qa_val.jsonl` (500 records — 60 think fixes, 438 gold_answer fixes)
- `data/processed/conceptnet_qa_test.jsonl` (500 records — 75 think fixes, 427 gold_answer fixes)
- `data/processed/conceptnet_qa_train_with_neg.jsonl` (4,500 records — 467 think fixes, 3,384 gold_answer fixes)

**Backups** (original Stage 2 outputs):
- `data/processed/conceptnet_qa_train.jsonl.bak`
- `data/processed/conceptnet_qa_val.jsonl.bak`
- `data/processed/conceptnet_qa_test.jsonl.bak`
- `data/processed/conceptnet_qa_train_with_neg.jsonl.bak`

**Run command**:
```bash
module load Miniconda3 && source activate kg_align
python src/datagen/fix_stage2_data.py --data_dir data/processed
```

---

## [2026-02-04] Stage 2: Question Generation + Quality Filtering

**SLURM Job**: 30636140 (`kg_stage2_datagen`)
**Node**: gpu25.viking2.yor.alces.network (1x NVIDIA H100 PCIe)
**Wall time**: 4h 22m (of 12h requested)
**Memory**: 10.86 GB / 64 GB

**Pipeline**:
- Stage 2a (Question Generation): Qwen2.5-7B-Instruct generated 16,000 raw QA pairs from 8,000 KG paths (2 variants per path). 0 failures. ~3h 45m.
- Stage 2b (Quality Filtering): LLM-as-judge scored all 16,000 pairs. 14,577/16,000 (91.1%) passed (min score 4 on all 3 dimensions). Subsampled to 5,000. Quality scores: answerability mean=4.41, faithfulness mean=4.76, naturalness mean=4.63. ~36m.
- Stage 2c (Negative Examples): Generated 500 corrupted-path negatives programmatically. <1s.

**Files created**:
- `src/datagen/question_generator.py` (468 lines) — batched QA generation from KG paths using Qwen2.5-7B-Instruct
- `src/datagen/quality_filter.py` (482 lines) — LLM-as-judge scoring and filtering
- `src/datagen/negative_examples.py` (304 lines) — programmatic corrupted-path negative generation
- `scripts/stage2_datagen.job` (100 lines) — SLURM job script orchestrating all three sub-stages
- `tests/test_datagen.py` (501 lines) — unit tests for question generator and quality filter

**Data outputs**:
- `data/processed/conceptnet_qa_raw.jsonl` (26 MB) — 16,000 raw QA pairs
- `data/processed/conceptnet_qa_raw.scored.jsonl` (27 MB) — 16,000 scored pairs
- `data/processed/conceptnet_qa_scored.jsonl` (27 MB) — full scored dataset
- `data/processed/conceptnet_qa_train.jsonl` (5.6 MB) — 4,000 training examples
- `data/processed/conceptnet_qa_val.jsonl` (723 KB) — 500 validation examples
- `data/processed/conceptnet_qa_test.jsonl` (706 KB) — 500 test examples
- `data/processed/conceptnet_qa_train_with_neg.jsonl` (5.9 MB) — 4,000 positive + 500 negative

**SLURM logs**:
- `kg_stage2_datagen-30636140.log` (1.9 KB) — structured summary
- `kg_stage2_datagen-30636140.err` (213 KB) — detailed Python logging output

---

## [2026-02-04] Stage 1: KG Path Extraction

**Goal**: Extract structured multi-hop paths from ConceptNet.

**Files created**:
- `src/kg/conceptnet_extractor.py` (629 lines) — ConceptNet loading, filtering (English, weight >= 2.0, exclude SimilarTo/RelatedTo), NetworkX graph building, stratified random walk path extraction with relation-type diversity constraints
- `tests/test_conceptnet_extractor.py` (382 lines) — unit tests for extractor

**Data output**:
- `data/processed/conceptnet_paths.jsonl` (1.4 MB) — 8,000 unique KG paths
  - Hop distribution: ~40% 1-hop, ~37% 2-hop, ~23% 3-hop
  - 30 unique relation types, max 14.4% (IsA), all under 20% cap
  - Relations: IsA, MannerOf, PartOf, HasContext, AtLocation, UsedFor, FormOf, HasPrerequisite, CapableOf, DerivedFrom, HasSubevent, HasProperty, CausesDesire, Causes, Antonym, Entails, MotivatedByGoal, HasA, ReceivesAction, DefinedAs, DistinctFrom, NotDesires, HasFirstSubevent, MadeOf, CreatedBy, Desires, HasLastSubevent, LocatedNear, NotHasProperty, NotCapableOf

---

## [2026-02-03] Stage 3: KG Reward Function (initial scaffold)

**Files created**:
- `src/rewards/kg_reward.py` (309 lines) — KG path-alignment reward (`compute_kg_reward`) and format reward (`compute_format_reward`) based on Kansal & Jha (arXiv:2601.15160). Token-level path coverage with min-hit constraint, repetition penalty, asymmetric binary correctness (+0.1/-1.0).
- `tests/test_kg_reward.py` (336 lines) — unit tests for reward functions

**Note**: This is an initial scaffold based directly on the reference paper. Was rewritten to v2 (continuous) and then v3 (answer-dominant) based on training results.

---

## [2026-02-03] Project Setup

**Files created**:
- `CLAUDE.md` — project instructions and coding standards
- `MINIMAL_VALIDATION_PLAN.md` — v2 pipeline plan (optimised)
- `MINIMAL_VALIDATION_PLAN_v1.md` — original v1 plan (archived)
- `kg_align_research_proposal.md` — research proposal document
- `requirements.txt` — Python dependencies with version pins
- `src/__init__.py`, `src/kg/__init__.py`, `src/datagen/__init__.py`, `src/rewards/__init__.py`, `tests/__init__.py` — package init files

---

## File Inventory

### Source Code (`src/`)

| File | Lines | Description |
|------|-------|-------------|
| `src/kg/conceptnet_extractor.py` | 629 | Stage 1: ConceptNet path extraction |
| `src/datagen/question_generator.py` | 468 | Stage 2a: QA generation from KG paths |
| `src/datagen/quality_filter.py` | 482 | Stage 2b: LLM-as-judge quality filtering |
| `src/datagen/negative_examples.py` | 304 | Stage 2c: corrupted-path negative generation |
| `src/datagen/fix_stage2_data.py` | 278 | Stage 2 data fix: think tags + gold_answer_short |
| `src/rewards/kg_reward.py` | 681 | Stage 3: KG reward functions v3 (answer-dominant + anti-hack) |
| `src/training/sft_warmup.py` | 306 | Stage 4: SFT warmup with DoRA+rsLoRA |
| `src/training/train_grpo.py` | 496 | Stage 5: GRPO training with curriculum |
| `src/evaluation/evaluate.py` | 588 | Stage 6: 3-way benchmark evaluation |

### Tests (`tests/`)

| File | Lines | Tests | Description |
|------|-------|-------|-------------|
| `tests/test_conceptnet_extractor.py` | 382 | — | Tests for KG path extraction |
| `tests/test_datagen.py` | 501 | — | Tests for question generation + filtering |
| `tests/test_kg_reward.py` | 916 | 98 | Tests for reward functions v3 |

### Scripts & Config

| File | Lines | Description |
|------|-------|-------------|
| `scripts/stage2_datagen.job` | 100 | SLURM job for Stage 2 pipeline |
| `scripts/train_sft.job` | 47 | SLURM job for Stage 4 (SFT warmup) |
| `scripts/train_grpo.job` | 69 | SLURM job for Stage 5 (GRPO training) |
| `scripts/eval.job` | 54 | SLURM job for Stage 6 (evaluation) |
| `configs/sft_warmup.yaml` | 39 | SFT warmup hyperparameters |
| `configs/grpo_validation.yaml` | 68 | GRPO training hyperparameters + curriculum |
| `requirements.txt` | — | Python dependencies |

### Data (`data/processed/`)

| File | Size | Records | Description |
|------|------|---------|-------------|
| `conceptnet_paths.jsonl` | 1.4 MB | 8,000 | KG paths (Stage 1 output) |
| `conceptnet_qa_raw.jsonl` | 26 MB | 16,000 | Raw QA pairs (Stage 2a) |
| `conceptnet_qa_raw.scored.jsonl` | 27 MB | 16,000 | Scored QA pairs (Stage 2b) |
| `conceptnet_qa_scored.jsonl` | 27 MB | 16,000 | Full scored dataset |
| `conceptnet_qa_train.jsonl` | 5.6 MB | 4,000 | Training split (fixed) |
| `conceptnet_qa_val.jsonl` | 723 KB | 500 | Validation split (fixed) |
| `conceptnet_qa_test.jsonl` | 706 KB | 500 | Test split (fixed) |
| `conceptnet_qa_train_with_neg.jsonl` | 5.9 MB | 4,500 | Train + 500 negatives (fixed) |
| `*.jsonl.bak` | — | — | Pre-fix backups (4 files) |

### Model Outputs (`outputs/`)

| Path | Description |
|------|-------------|
| `outputs/sft-warmup/` | Stage 4 LoRA adapter (73 MB) + 2 checkpoints |
| `outputs/grpo/phase{1,2,3}_*/` | Stage 5 LoRA adapters (3 curriculum phases) |
| `outputs/eval/` | Stage 6 evaluation results (8 JSON files) |

### Reward Function Version History

| Version | Date | Range | Answer % | Key Issue |
|---------|------|-------|----------|-----------|
| v1 (scaffold) | Feb 03 | -1.0 to 1.6 | ~6% | Binary +0.1/-1.0 → zero GRPO variance |
| v2 (continuous) | Feb 10 | -0.5 to 2.2 | 23% | Bag-of-words coverage → template collapse reward hacking |
| **v3 (current)** | **Feb 15** | **-0.3 to 1.9** | **56%** | **Answer-dominant, structured triples, anti-hack penalties** |

### Key Issues & Lessons Learned

| Issue | Discovery | Root Cause | Fix |
|-------|-----------|-----------|-----|
| Base model can't produce EOS | Stage 4 | Qwen2.5-1.5B base never generates `<\|im_end\|>` | Switch to Instruct model |
| Flash Attention crash | Stage 5 | FA2 incompatibility | `attn_implementation: "sdpa"` |
| loss=0 in GRPO | Stage 5 | Expected with num_iterations=1 + beta=0 (TRL computes `logp - logp.detach()`) | `num_iterations: 2` |
| Zero GRPO gradient | Stage 5 v1 | Binary reward → zero within-group variance | Continuous reward (v2) |
| Completion length collapse | Stage 5 v1 | Repetition penalty rewards brevity | Remove penalty asymmetry (v2) |
| Reward hacking / template collapse | Stage 5 v2 + Eval | Bag-of-words coverage gameable by word salad | Structured triple matching + anti-hack penalties (v3) |
| Entropy collapse | Stage 5 v2 | beta=0 allows unlimited policy drift | beta=0.02 KL anchor (v3) |
