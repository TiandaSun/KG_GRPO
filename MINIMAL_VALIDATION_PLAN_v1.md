# Minimal Validation Pipeline Plan

## Objective

Verify the end-to-end feasibility of the KG-Align-RL pipeline with minimal compute before committing to the full 10,000 GPU-hour experiment. This validation should answer: **"Does the pipeline work, and does GRPO with KG rewards produce any training signal?"**

---

## Hardware Budget

| Setup | VRAM | Suitable For |
|-------|------|-------------|
| 1x A40 (48GB) | 48GB | Qwen2.5-1.5B training, Qwen2.5-3B inference |
| 2x A40 (96GB) | 96GB | Qwen2.5-3B training with DeepSpeed |
| 1x H100 (80GB) | 80GB | Qwen2.5-3B training comfortably |

**Recommended validation setup**: 1x A40 with Qwen2.5-1.5B — fastest iteration cycle.

---

## Model Choice: Why Qwen2.5-1.5B

| Factor | Qwen2.5-1.5B | Qwen2.5-3B | Qwen2.5-7B |
|--------|-------------|------------|------------|
| VRAM (bf16 inference) | ~3GB | ~6GB | ~14GB |
| VRAM (LoRA training, bs=4) | ~12GB | ~24GB | ~48GB |
| VRAM (GRPO, generation+train) | ~20GB | ~40GB | ~70GB |
| Fits 1x A40? | Yes | Tight | No |
| Fits 1x H100? | Yes | Yes | Yes (tight with GRPO) |
| Iteration speed | Fast | Medium | Slow |
| Reasoning capability | Basic | Moderate | Good |

Qwen2.5-1.5B is the right choice for validation because:
- GRPO requires memory for both generation AND training simultaneously
- It fits comfortably on a single A40 with room for large batch generation
- If the pipeline works at 1.5B, it will work at 7B (just slower)
- If KG rewards produce zero signal at 1.5B, that's a useful finding too

---

## Pipeline Stages

### Stage 1: KG Path Extraction (~1 day of development)

**Goal**: Extract structured multi-hop paths from ConceptNet.

**Why ConceptNet first**:
- Freely available (conceptnet.io), well-documented format
- General-domain knowledge (not medical-specific like prior work)
- Clean triple format: `(head, relation, tail, weight)`
- Good for validating cross-domain transfer (train on ConceptNet, eval on HotpotQA)

**Minimal scope**:
- Download ConceptNet assertions (English only, ~2M triples after filtering)
- Filter: keep relations with weight >= 2.0, English entities only
- Extract 1-hop and 2-hop paths (skip 3-hop for validation)
- Target: **5,000 unique paths** (enough to generate training data)
- Output format: JSON lines `{"path": [{"entity": "dog", "relation": "IsA", "entity": "animal"}], "hops": 1}`

**Implementation**:
```python
# src/kg/conceptnet_extractor.py
# 1. Load ConceptNet CSV
# 2. Build NetworkX graph (filtered)
# 3. Sample random walks of length 1-2
# 4. Deduplicate and save as JSONL
```

### Stage 2: Question Generation (~1-2 days)

**Goal**: Convert KG paths into natural language QA pairs.

**Approach**: Use an instruction-tuned LLM to generate questions from paths.
- Use Qwen2.5-7B-Instruct (or Qwen2.5-3B-Instruct) as the question generator (inference only, fits A40)
- Alternatively: use a hosted API (e.g., Claude API) for higher quality at small scale

**Minimal scope**:
- Generate 1 question per path → **5,000 QA pairs**
- Split: 4,000 train / 500 val / 500 test
- Template prompt:
  ```
  Given this knowledge graph path:
  [dog] --IsA--> [animal] --HasProperty--> [alive]

  Generate a natural question whose answer requires following this path.
  Also provide the correct answer.

  Format:
  Question: <question>
  Answer: <answer>
  ```

**Quality check**: Manually inspect 50 generated QA pairs for coherence.

**Output format**:
```json
{
  "question": "What property do dogs have because they are animals?",
  "answer": "They are alive",
  "kg_path": [["dog", "IsA", "animal"], ["animal", "HasProperty", "alive"]],
  "hops": 2
}
```

### Stage 3: KG Reward Function (~1 day)

**Goal**: Implement a reward function that scores model outputs based on KG path alignment.

**Reward design** (simplified for validation):
1. **Entity recall reward** (0-1): What fraction of path entities appear in the model's output?
2. **Path order reward** (0-1): Are entities mentioned in the correct path order?
3. **Answer correctness** (0 or 1): Does the final answer match?
4. **Combined**: `reward = 0.3 * entity_recall + 0.2 * path_order + 0.5 * answer_correct`

**Implementation**:
```python
# src/rewards/kg_reward.py
def compute_kg_reward(
    question: str,
    model_output: str,
    kg_path: list[tuple[str, str, str]],
    gold_answer: str
) -> float:
    ...
```

**Validation**: Test reward function on 100 hand-written good/bad answers to verify it discriminates correctly.

### Stage 4: GRPO + LoRA Training (~2-3 days)

**Goal**: Train Qwen2.5-1.5B with GRPO using KG rewards and LoRA.

**Configuration**:
```yaml
# configs/validation.yaml
model:
  name: Qwen/Qwen2.5-1.5B
  torch_dtype: bfloat16
  attn_implementation: flash_attention_2  # if available

lora:
  r: 16
  lora_alpha: 32
  target_modules: [q_proj, v_proj]
  lora_dropout: 0.05

grpo:
  num_generations: 4        # G in GRPO (group size)
  max_new_tokens: 256
  temperature: 0.7
  learning_rate: 5e-5
  num_train_epochs: 2
  per_device_train_batch_size: 2
  gradient_accumulation_steps: 4
  gradient_checkpointing: true
  max_steps: 1000           # cap for validation
  save_steps: 200
  logging_steps: 10

data:
  train_file: data/processed/conceptnet_qa_train.jsonl
  val_file: data/processed/conceptnet_qa_val.jsonl
  max_length: 512
```

**VRAM estimate for 1x A40 (Qwen2.5-1.5B)**:
- Model (bf16): ~3GB
- LoRA parameters: ~0.1GB
- Optimizer states: ~0.5GB
- GRPO generation (4 sequences): ~4GB
- KV cache during generation: ~2GB
- Gradients + activations: ~6GB
- **Total: ~16GB** — fits comfortably on A40 (48GB)

**What to monitor**:
- Training reward should increase over steps (most critical signal)
- KL divergence should stay bounded (model not collapsing)
- Generation quality: sample outputs every 100 steps

### Stage 5: Evaluation (~1 day)

**Goal**: Measure whether KG-RL training produces any improvement.

**Benchmarks** (small subsets for validation):

| Benchmark | Subset Size | Purpose |
|-----------|------------|---------|
| ConceptNet QA test split | 500 | In-domain sanity check |
| HotpotQA (bridge, dev) | 500 | Near transfer signal |

**Metrics**:
- Exact Match (EM) and F1 on answer extraction
- Compare: base Qwen2.5-1.5B vs. KG-RL trained Qwen2.5-1.5B

**Success criteria for validation**:
1. Training reward increases (pipeline works mechanically) — **MUST PASS**
2. In-domain QA accuracy improves over base model — **SHOULD PASS**
3. HotpotQA shows any improvement — **NICE TO HAVE** (at 1.5B scale, transfer signal may be weak)

---

## Implementation Order & Dependencies

```
Stage 1: KG Path Extraction
    ↓
Stage 2: Question Generation (needs paths from Stage 1)
    ↓
Stage 3: KG Reward Function (needs path format from Stage 1, can partially parallel with Stage 2)
    ↓
Stage 4: GRPO Training (needs QA data from Stage 2 + reward from Stage 3)
    ↓
Stage 5: Evaluation (needs trained model from Stage 4)
```

---

## Key Dependencies & Versions

```
torch>=2.1.0
transformers>=4.40.0
trl>=0.12.0              # GRPOTrainer support
peft>=0.13.0
datasets>=2.18.0
accelerate>=0.30.0
networkx>=3.0
wandb
flash-attn>=2.5.0        # optional, for H100/A40
vllm>=0.4.0              # optional, for faster GRPO generation
```

---

## Risk Mitigation

| Risk | Mitigation |
|------|-----------|
| ConceptNet paths too noisy | Filter by weight >= 2.0, manually inspect samples |
| Question generation produces garbage | Use stronger model for generation, add quality filters |
| GRPO reward is too sparse | Add intermediate entity-recall reward (not just final answer) |
| OOM during GRPO generation | Reduce `num_generations` from 4 to 2, reduce `max_new_tokens` |
| KG reward doesn't discriminate | Test reward function on synthetic good/bad pairs before training |
| No transfer signal at 1.5B | Expected — validation goal is mechanical pipeline verification |

---

## What This Validation Proves

If all stages complete successfully:
- The **KG path extraction** pipeline produces structured, usable paths
- **Question generation** from paths creates reasonable QA training data
- The **KG reward function** differentiates good and bad reasoning
- **GRPO + LoRA** training converges with KG rewards (loss/reward curves improve)
- The full pipeline is **technically feasible** and ready to scale to 7B

This de-risks the full experiment before committing 10,000 GPU hours.

---

## Estimated GPU Hours for Validation

| Stage | GPU Hours | Notes |
|-------|-----------|-------|
| Stage 1 | 0 | CPU only (graph processing) |
| Stage 2 | 2-4 | Inference with 3B-Instruct for question gen |
| Stage 3 | 0 | CPU only (reward function dev) |
| Stage 4 | 8-16 | 1000 GRPO steps on 1x A40 |
| Stage 5 | 1-2 | Inference on eval benchmarks |
| **Total** | **~15-25** | <0.3% of full experiment budget |
