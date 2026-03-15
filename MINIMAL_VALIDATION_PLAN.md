# Minimal Validation Pipeline Plan (v2 — Optimised)

> **Previous version**: See `MINIMAL_VALIDATION_PLAN_v1.md` for the original plan.
> **What changed**: Incorporates optimisation findings from recent literature (Dr. GRPO, DoRA, Kansal & Jha ablations, curriculum learning, CoT warmup). All changes are implementable with standard TRL/PEFT APIs or simple custom code.

## Objective

Verify the end-to-end feasibility of the KG-Align-RL pipeline with minimal compute. This validation should answer: **"Does the pipeline work, and does GRPO with KG rewards produce any training signal?"**

---

## Hardware

| Setup | VRAM | Notes |
|-------|------|-------|
| **1x H100 (80GB)** | 80GB | **Default for this plan** — lowest queue time on Viking |

With Qwen2.5-1.5B on a single H100, GRPO training uses ~20-25GB, leaving ample headroom for larger `num_generations` and vLLM colocate mode.

---

## Pipeline Overview

```
Stage 0: Environment Setup
    ↓
Stage 1: KG Path Extraction (CPU)
    ↓
Stage 2: Question Generation + Quality Filtering (GPU inference)
    ↓
Stage 3: KG Reward Function (CPU development + unit tests)
    ↓
Stage 4: SFT Warmup (GPU, short)
    ↓
Stage 5: GRPO Training with Curriculum (GPU, main experiment)
    ↓
Stage 6: Evaluation (GPU inference)
```

Key additions vs v1: Stage 0 (environment), Stage 4 (SFT warmup), curriculum in Stage 5, quality filtering in Stage 2.

---

## Stage 0: Environment Setup

Viking HPC uses the `module` system to load software. Conda is provided via `Miniconda3`. Note that on Viking, `conda activate` does not work — use `source activate` instead.

```bash
# 1. Load conda and CUDA toolkit
module load Miniconda3
module load CUDA/12.1.1

# 2. Create conda environment (default conda path is already on scratch)
conda create -n kg_align python=3.10 -y

# 3. Activate (Viking requires `source activate`, not `conda activate`)
source activate kg_align

# 4. Set HF/torch cache to scratch
export HF_HOME=/mnt/scratch/users/ts1201/hf_cache
export TORCH_HOME=/mnt/scratch/users/ts1201/torch_cache

# 5. Install PyTorch first (CUDA 12.1 for H100)
pip install torch --index-url https://download.pytorch.org/whl/cu121

# 6. Install remaining dependencies
pip install -r requirements.txt

# 7. Install flash-attn separately (requires torch to be installed first)
pip install flash-attn --no-build-isolation

# 8. Install vllm (optional — for faster GRPO generation)
#    Requires CUDA module loaded (done in step 1). If this fails or the
#    installed version is incompatible with torch, skip it — the pipeline
#    falls back to HF generate().
pip install vllm
```

Dependencies are defined in `requirements.txt` at the project root. See that file for version pins and notes.

> **Viking-specific notes**:
> - Always use `source activate`, never `conda activate` — the latter fails silently on Viking's module-based conda setup.
> - SLURM job scripts must include `module load Miniconda3`, `module load CUDA/12.1.1`, and `source activate kg_align` before any Python commands.
> - All large files (models, datasets, checkpoints) go on scratch to avoid home quota limits. The conda env is already on scratch by default.
> - If `flash-attn` fails to build, the pipeline still works — Flash Attention 2 is optional (falls back to standard SDPA).

---

## Stage 1: KG Path Extraction

**Goal**: Extract structured multi-hop paths from ConceptNet.

**Scope**:
- Download ConceptNet assertions (English only, ~2M triples after filtering)
- Filter: weight >= 2.0, English entities only
- Extract 1-hop, 2-hop, and 3-hop paths
- **Diversity constraint**: sample across all major relation types (IsA, HasProperty, UsedFor, CapableOf, Causes, HasA, PartOf, AtLocation, etc.) — no single relation type > 20% of dataset
- Target: **8,000 unique paths** (will be filtered to ~5,000 in Stage 2)
- Stratification target: ~40% 1-hop, ~35% 2-hop, ~25% 3-hop

**Output format** (JSONL):
```json
{
  "path": [["dog", "IsA", "animal"], ["animal", "HasProperty", "alive"]],
  "hops": 2,
  "relations": ["IsA", "HasProperty"],
  "entities": ["dog", "animal", "alive"]
}
```

**Implementation**: `src/kg/conceptnet_extractor.py`
```python
# 1. Load ConceptNet CSV (assertions.csv)
# 2. Filter: English, weight >= 2.0, exclude SimilarTo/RelatedTo (too vague)
# 3. Build NetworkX DiGraph
# 4. Stratified random walks: sample paths ensuring relation-type diversity
# 5. Deduplicate by entity set
# 6. Output: data/processed/conceptnet_paths.jsonl
```

**Rationale for 8K paths (up from 5K)**: We generate more paths than needed so quality filtering in Stage 2 can select the best subset. Research shows "generate more, keep the best" consistently outperforms trying to generate exactly the right number.

---

## Stage 2: Question Generation + Quality Filtering

**Goal**: Convert KG paths into high-quality QA pairs with chain-of-thought reasoning.

### 2a. Generation

**Key change from v1**: Generate QA pairs in `<think>` CoT format. This is critical for the SFT warmup in Stage 4 and for the format reward in Stage 5.

**Generator model**: Qwen2.5-7B-Instruct (inference only, ~14GB VRAM on H100)

**Prompt template**:
```
Given this knowledge graph path:
[dog] --IsA--> [animal] --HasProperty--> [alive]

Generate a natural question whose answer requires following this reasoning path.
Provide the answer in the following format:

<think>
[Step-by-step reasoning that walks through the knowledge graph path,
explaining each hop and how it leads to the answer]
</think>
[Final answer in a short phrase]

Format your response as:
Question: <question>
Answer: <full answer with thinking and final answer>
```

**Generate 2 question variants per path** for diversity → ~16,000 raw QA pairs from 8,000 paths.

### 2b. Quality Filtering

Use Qwen2.5-7B-Instruct as judge (separate inference pass) to score each QA pair on:
1. **Answerability** (1-5): Can the question be answered from the KG path?
2. **Faithfulness** (1-5): Does the CoT reasoning actually follow the KG path?
3. **Naturalness** (1-5): Does the question sound like a human would ask it?

**Filter**: Keep pairs scoring >= 4 on all three dimensions. Target: **5,000 QA pairs** after filtering.

**Split**: 4,000 train / 500 val / 500 test

### 2c. Negative Example Generation (programmatic, no LLM needed)

For ~500 training pairs, generate a **corrupted variant**:
- Swap one entity in the KG path with a random entity
- Generate question: "Is it true that [corrupted statement]?"
- Answer: `<think>[reasoning showing why the corrupted path is wrong]</think>\nNo, this is incorrect.`

This teaches the model to distinguish valid from invalid KG paths.

**Output format** (JSONL):
```json
{
  "question": "What property do dogs have because they are animals?",
  "answer": "<think>A dog is an animal (IsA relation). Animals have the property of being alive (HasProperty relation). Following this path: dog → IsA → animal → HasProperty → alive. Therefore, dogs are alive.</think>\nThey are alive.",
  "kg_path": [["dog", "IsA", "animal"], ["animal", "HasProperty", "alive"]],
  "gold_answer_short": "alive",
  "hops": 2,
  "is_negative": false
}
```

**Implementation**: `src/datagen/question_generator.py`, `src/datagen/quality_filter.py`

---

## Stage 3: KG Reward Function

**Goal**: Implement a reward function based on Kansal & Jha's proven design.

### Key insight from the reference paper's ablations

Kansal & Jha (arXiv:2601.15160) found that **simpler rewards work better**. Adding more reward signals (thinking quality, similarity) caused performance collapse to 55.21%. Their best config uses only two signals: path coverage + asymmetric binary correctness.

### Reward design (simplified from v1)

**v1 reward (3 signals — risky)**:
```
0.3 * entity_recall + 0.2 * path_order + 0.5 * answer_correct
```

**v2 reward (2 signals — proven)**:

```python
# src/rewards/kg_reward.py

def compute_kg_reward(
    question: str,
    model_output: str,
    kg_path: list[list[str]],
    gold_answer: str,
) -> float:
    """
    KG path-alignment reward based on Kansal & Jha (arXiv:2601.15160).

    Two components:
    1. R_path: token-level path coverage with minimum-hit constraint
    2. R_bin: asymmetric binary correctness (+0.1 / -1.0)
    """
    # Extract path tokens (entities + relations, lowercased)
    path_tokens: set[str] = set()
    for triple in kg_path:
        for token in triple:
            path_tokens.update(token.lower().split())

    # Token-level coverage
    output_lower = model_output.lower()
    output_words = set(output_lower.split())
    hits = len(path_tokens & output_words)

    # Minimum-hit constraint: need >= 2 distinct matches
    if hits < 2:
        coverage = 0.0
    else:
        coverage = hits / len(path_tokens)

    # Repetition penalty: penalise entity stuffing
    entity_mentions = sum(
        output_lower.count(tok) for tok in path_tokens
    )
    rep_penalty = min(1.0, len(path_tokens) * 2 / max(entity_mentions, 1))

    # Path alignment reward (capped at 1.5)
    r_path = min(1.2 * coverage * rep_penalty + 0.3 * (1.0 if hits >= 2 else 0.0), 1.5)

    # Asymmetric binary correctness
    answer_correct = _check_answer(model_output, gold_answer)
    r_bin = 0.1 if answer_correct else -1.0

    return r_path + r_bin


def compute_format_reward(model_output: str) -> float:
    """Reward for proper <think>...</think> format."""
    has_think_open = "<think>" in model_output
    has_think_close = "</think>" in model_output

    if has_think_open and has_think_close:
        # Check reasoning is non-empty (> 20 chars)
        think_content = model_output.split("<think>")[1].split("</think>")[0]
        if len(think_content.strip()) > 20:
            return 1.0
        return 0.5  # has tags but empty reasoning
    return 0.0


def _check_answer(model_output: str, gold_answer: str) -> bool:
    """Check if gold answer appears in the final answer (after </think>)."""
    # Extract text after </think> if present
    if "</think>" in model_output:
        final_answer = model_output.split("</think>")[-1].strip().lower()
    else:
        final_answer = model_output.strip().lower()
    return gold_answer.lower() in final_answer
```

**Using TRL's multiple reward functions**:

Pass both `compute_kg_reward` and `compute_format_reward` as a list to `GRPOTrainer.reward_funcs`. TRL sums them (or use `reward_weights` to adjust). This keeps rewards decoupled and avoids the multi-signal collapse issue.

### Validation of reward function

Before training, test on 100 hand-crafted good/bad outputs:
- Good outputs (follow KG path, correct answer) should score > 1.0
- Bad outputs (wrong answer, no path alignment) should score < -0.5
- Entity-stuffing outputs (list entities without reasoning) should score low due to repetition penalty

**Implementation**: `src/rewards/kg_reward.py`, `tests/test_kg_reward.py`

---

## Stage 4: SFT Warmup (NEW — Critical for Small Models)

**Goal**: Teach the 1.5B model the `<think>` output format before GRPO.

**Why this is essential**: DeepSeek-R1 showed that small models suffer severe cold-start problems with pure RL (R1-Zero). The model produces messy, unreadable reasoning without SFT warmup. At 1.5B parameters, this problem is worse than at larger scales.

**Configuration**:
```yaml
# configs/sft_warmup.yaml
model:
  name: Qwen/Qwen2.5-1.5B
  torch_dtype: bfloat16
  attn_implementation: flash_attention_2

lora:
  r: 16
  lora_alpha: 32
  target_modules: "all-linear"    # Changed from [q_proj, v_proj]
  lora_dropout: 0.05
  use_dora: true                   # DoRA: outperforms LoRA for RL (arXiv:2512.23165)
  use_rslora: true                 # rsLoRA: free improvement for higher ranks

sft:
  num_train_epochs: 2
  per_device_train_batch_size: 8
  learning_rate: 2e-4              # LoRA benefits from ~10x higher LR
  gradient_checkpointing: true
  max_seq_length: 512
  logging_steps: 10
  save_strategy: "epoch"

data:
  train_file: data/processed/conceptnet_qa_train.jsonl
  max_length: 512
```

**Output**: LoRA adapter checkpoint at `outputs/sft-warmup/`

**Implementation**: `src/training/sft_warmup.py` (standard `SFTTrainer` from TRL)

---

## Stage 5: GRPO Training with Curriculum

**Goal**: Train the SFT-warmed model with GRPO using KG rewards, following an easy-to-hard curriculum.

### LoRA Configuration (Optimised)

Three evidence-based changes from v1:

| Parameter | v1 | v2 | Rationale |
|-----------|----|----|-----------|
| `target_modules` | `[q_proj, v_proj]` | `"all-linear"` | MLP layers critical for reasoning (QLoRA paper, Unsloth guide) |
| `use_dora` | `false` | `true` | DoRA > LoRA > full fine-tuning for RLVR (arXiv:2512.23165) |
| `use_rslora` | `false` | `true` | Fixes rank-scaling bug, zero downside (HF blog) |

### GRPO Configuration (Optimised)

```yaml
# configs/grpo_validation.yaml
model:
  name: Qwen/Qwen2.5-1.5B
  torch_dtype: bfloat16
  attn_implementation: flash_attention_2

lora:
  r: 16
  lora_alpha: 32
  target_modules: "all-linear"
  lora_dropout: 0.05
  use_dora: true
  use_rslora: true
  # Load SFT warmup adapter as starting point
  adapter_path: outputs/sft-warmup/

grpo:
  num_generations: 8              # Up from 4 — H100 has headroom, better group comparison
  max_completion_length: 512      # Room for <think> reasoning
  max_prompt_length: 256
  temperature: 0.7
  learning_rate: 1e-5             # Lower than SFT for RL stability
  num_train_epochs: 3
  per_device_train_batch_size: 2
  gradient_accumulation_steps: 8  # Effective batch: 2 * 8 = 16 prompts * 8 gens = 128
  gradient_checkpointing: true
  max_steps: 1000                 # Cap for validation
  save_steps: 200
  logging_steps: 10

  # Dr. GRPO / DAPO optimisations (config flags, zero implementation cost)
  scale_rewards: false            # Removes std bias (Dr. GRPO, arXiv:2503.20783)
  beta: 0.0                      # No KL penalty (DAPO, standard for reasoning)
  loss_type: "grpo"               # Default; switch to "cispo" if token-dropping observed

  # vLLM integration for faster generation (if TRL version supports it)
  use_vllm: true
  vllm_mode: "colocate"
  vllm_enable_sleep_mode: true
  vllm_gpu_memory_utilization: 0.4

  # Optimizer
  optim: "adamw_8bit"             # Halves optimizer memory

  # Reward functions (passed as list to GRPOTrainer)
  # reward_funcs: [compute_kg_reward, compute_format_reward]

data:
  max_length: 768                 # prompt + completion
```

**vLLM fallback**: If the installed TRL version does not support `vllm_mode="colocate"`, remove the vLLM flags and use standard HF generation. The training will be slower but functionally identical.

### Curriculum Schedule

Train in three phases using dataset partitioning by hop count:

| Phase | Steps | Data | Rationale |
|-------|-------|------|-----------|
| Phase 1 | 0–350 | 1-hop paths only (~1,600 samples) | Easy tasks, build basic KG alignment |
| Phase 2 | 350–700 | 1-hop + 2-hop paths (~3,000 samples) | Introduce multi-hop reasoning |
| Phase 3 | 700–1000 | All paths including 3-hop (~4,000 samples) | Full difficulty |

**Implementation**: Split training data into three JSONL files by hop count. Use a simple training script that loads different datasets at each phase boundary and reinitialises the dataloader. This is straightforward with TRL's `GRPOTrainer` — create three training runs with `max_steps` set to each phase boundary, loading the checkpoint from the previous phase.

Alternatively, simpler approach: create a single dataset file sorted by hop count (1-hop first, then 2-hop, then 3-hop). With sequential sampling (no shuffling), the natural ordering provides a curriculum effect. Add shuffling only within each hop-count group.

### Dynamic Prompt Filtering (DAPO-inspired)

Before training, run a quick pilot pass (base model, no LoRA):
- For each question, generate 8 completions and compute rewards
- Remove questions where **all 8 score the same** (all correct or all wrong)
- These provide zero gradient signal and waste compute
- Expected to filter out 10-30% of prompts

**Implementation**: `src/training/filter_prompts.py` — a standalone script that runs inference and outputs a filtered dataset.

### What to Monitor (wandb)

| Metric | Healthy Range | Action if Unhealthy |
|--------|--------------|-------------------|
| `reward/mean` | Increasing over steps | If flat: check reward function discriminates |
| `reward/std` | Non-zero, stable | If → 0: model collapsed to one output |
| `policy/entropy` | Gradually decreasing but > 0 | If → 0 fast: entropy collapse, try `loss_type="cispo"` |
| `policy/kl` | N/A (beta=0.0) | — |
| `format_reward/mean` | Should stay near 1.0 | If dropping: model losing `<think>` format |
| `train/loss` | Decreasing | Standard check |

**Implementation**: `src/training/train_grpo.py`

---

## Stage 6: Evaluation

**Goal**: Measure whether KG-RL training produces any improvement.

**Benchmarks** (small subsets):

| Benchmark | Subset Size | Purpose |
|-----------|------------|---------|
| ConceptNet QA test split | 500 | In-domain sanity check |
| HotpotQA (bridge, dev) | 500 | Near-transfer signal |

**Comparisons** (3 models):

| Model | Description |
|-------|-------------|
| Base Qwen2.5-1.5B | No fine-tuning (baseline) |
| SFT-only | After Stage 4, before GRPO |
| SFT + GRPO | After Stage 5 (full pipeline) |

The three-way comparison isolates the contribution of each stage.

**Metrics**:
- Exact Match (EM) and token-level F1
- **KG path alignment score** on test set (using the reward function from Stage 3)
- Qualitative: sample 20 outputs from each model for manual inspection

**Success criteria**:
1. GRPO reward increases during training — **MUST PASS** (pipeline works)
2. SFT+GRPO > SFT-only on in-domain QA — **SHOULD PASS**
3. SFT+GRPO > Base on in-domain QA — **SHOULD PASS**
4. Any improvement on HotpotQA — **NICE TO HAVE** (1.5B may be too small for transfer)

**Implementation**: `src/evaluation/evaluate.py`

---

## VRAM Budget (1x H100, 80GB)

### Stage 4: SFT Warmup
| Component | VRAM |
|-----------|------|
| Model (bf16) | ~3 GB |
| DoRA adapters (all-linear, r=16) | ~0.2 GB |
| Optimizer (8-bit AdamW) | ~0.2 GB |
| Activations + gradients (grad ckpt) | ~4 GB |
| **Total** | **~8 GB** |

### Stage 5: GRPO Training
| Component | VRAM |
|-----------|------|
| Model (bf16) | ~3 GB |
| DoRA adapters | ~0.2 GB |
| Optimizer (8-bit AdamW) | ~0.2 GB |
| vLLM inference engine (colocate) | ~4 GB |
| GRPO generation buffer (8 sequences) | ~6 GB |
| KV cache during generation | ~3 GB |
| Gradients + activations (grad ckpt) | ~6 GB |
| **Total** | **~23 GB** — well within 80GB |

Headroom: ~57 GB free. Could increase `num_generations` to 16 or `max_completion_length` to 1024 if needed.

---

## File Structure

```
src/
  kg/
    conceptnet_extractor.py       # Stage 1: path extraction
  datagen/
    question_generator.py          # Stage 2a: QA generation
    quality_filter.py              # Stage 2b: LLM-as-judge filtering
    negative_examples.py           # Stage 2c: corrupted path generation
  rewards/
    kg_reward.py                   # Stage 3: reward functions
  training/
    sft_warmup.py                  # Stage 4: SFT warmup
    train_grpo.py                  # Stage 5: GRPO training
    filter_prompts.py              # Stage 5: dynamic prompt filtering
  evaluation/
    evaluate.py                    # Stage 6: benchmark evaluation
configs/
  sft_warmup.yaml                  # Stage 4 config
  grpo_validation.yaml             # Stage 5 config
scripts/
  train.slurm                     # SLURM job script for H100
tests/
  test_kg_reward.py               # Reward function unit tests
data/
  raw/                            # ConceptNet dump
  processed/                      # Extracted paths, QA pairs
  eval/                           # Evaluation benchmarks
```

---

## Optimisations Summary: What Changed from v1

| Area | v1 | v2 | Source |
|------|----|----|--------|
| **Hardware** | 1x A40 (48GB) | 1x H100 (80GB) | Lower queue time |
| **LoRA targets** | q_proj, v_proj | all-linear | QLoRA paper, Unsloth guide |
| **LoRA variant** | Standard LoRA | DoRA + rsLoRA | arXiv:2512.23165 (RLVR benchmark) |
| **Reward function** | 3 signals (entity_recall + path_order + answer_correct) | 2 signals (path_coverage + asymmetric binary) | Kansal & Jha ablations |
| **Reward hacking prevention** | None | Min-hit constraint + repetition penalty | Kansal & Jha |
| **Format reward** | None | Separate reward for `<think>` tags | DeepSeek-R1 |
| **SFT warmup** | None | 2 epochs before GRPO | DeepSeek-R1 cold-start fix |
| **GRPO bias fixes** | None | `scale_rewards=false`, `beta=0.0` | Dr. GRPO (arXiv:2503.20783), DAPO |
| **num_generations** | 4 | 8 | Better group comparison, H100 has headroom |
| **Effective batch size** | 8 | 128 | GRPO++ stability recommendation |
| **Curriculum** | None | 1-hop → 2-hop → 3-hop | E2H Reasoner, LACT (AAAI 2025) |
| **Generation speed** | HF generate() | vLLM colocate (with fallback) | TRL native integration |
| **Data generation** | 5K paths, 1 question each | 8K paths, 2 variants each, filtered to 5K | Generate-then-filter approach |
| **Quality filtering** | Manual inspection of 50 | LLM-as-judge scoring on 3 dimensions | Standard best practice |
| **Prompt filtering** | None | Remove zero-gradient prompts before training | DAPO dynamic sampling |
| **Evaluation** | 2-way comparison | 3-way (base vs SFT vs SFT+GRPO) | Isolate contribution of each stage |
| **Optimizer** | AdamW | AdamW 8-bit | VRAM savings |

---

## Estimated GPU Hours

| Stage | GPU Hours | Notes |
|-------|-----------|-------|
| Stage 1 | 0 | CPU only |
| Stage 2 | 2-3 | 7B-Instruct inference for QA gen + filtering |
| Stage 3 | 0 | CPU only (reward dev + tests) |
| Stage 4 | 1-2 | SFT warmup, 2 epochs on 4K samples |
| Stage 5 | 8-12 | 1000 GRPO steps on 1x H100 (faster than A40) |
| Stage 6 | 1 | Inference on eval benchmarks |
| **Total** | **~12-18** | Slightly less than v1 due to H100 speed |

---

## Risk Mitigation

| Risk | Mitigation |
|------|-----------|
| ConceptNet paths too noisy | Filter weight >= 2.0, exclude vague relations (SimilarTo, RelatedTo), diversity stratification |
| Question generation produces garbage | LLM-as-judge filtering on 3 dimensions, generate 2x and keep best |
| GRPO reward too sparse at 1.5B | SFT warmup gives the model a head start; curriculum starts with easy 1-hop tasks |
| Reward hacking (entity stuffing) | Repetition penalty + minimum-hit constraint in reward function |
| OOM during GRPO generation | H100 80GB gives ample headroom; reduce num_generations to 4 as fallback |
| Entropy collapse | Monitor entropy in wandb; switch to `loss_type="cispo"` if observed |
| vLLM colocate not available in TRL version | Graceful fallback to HF generate() — slower but works |
| DoRA not in installed PEFT version | Fallback to standard LoRA with `use_dora=false` |
| SFT warmup overfits | Only 2 epochs, validate on held-out split, early stopping |

---

## Techniques Considered but Deferred

These are valuable but either too complex for a validation run or require unavailable code:

| Technique | Why Deferred |
|-----------|-------------|
| G2RPO-A (guided hints for small models) | Requires modifying TRL's rollout generation internals |
| Lambda-GRPO (process reward normalisation) | Requires custom loss computation, not in TRL |
| DRA-GRPO (diversity-aware reward weighting) | Needs embedding similarity computation per batch — add if reward sparsity is observed |
| SPEC-RL (speculative decoding for GRPO) | Research prototype, no public integration with TRL |
| Unsloth | Useful for 7B scale-up but adds a dependency; standard TRL sufficient at 1.5B on H100 |
| verl framework | Different API from TRL; worth exploring for multi-GPU 7B training later |
| Distillation (7B → 1.5B) | Requires training 7B first; valuable but out of scope for minimal validation |
| MCTS-guided KG search | High implementation complexity; inference-time technique for later |

---

## What This Validation Proves

If all stages complete successfully:
- The **KG path extraction** pipeline produces structured, diverse, usable paths
- **Question generation + filtering** creates high-quality CoT-formatted QA data
- The **KG reward function** (Kansal & Jha design) discriminates good and bad reasoning
- **SFT warmup** teaches the model the `<think>` output format
- **GRPO + DoRA** training converges with KG rewards (reward curves improve)
- The **curriculum** helps the small model learn progressively
- The full pipeline is **technically feasible** and ready to scale to 7B

This de-risks the full experiment before committing significant GPU hours.
