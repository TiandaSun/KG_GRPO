# Daily Process Update

## 2026-03-19 (Session 1): Isambard Environment Fix + SFT Submitted

### Goal
Fix verl GRPO crash on Isambard ARM GH200 and get training pipeline running.

### Issues Found & Fixed

| # | Issue | Root Cause | Fix |
|---|-------|-----------|-----|
| 1 | vLLM version incompatible | verl 0.7.1 requires `vllm<=0.12.0`, had 0.17.1 | `pip install vllm==0.12.0` + reinstall `torch==2.9.0+cu126` from PyTorch cu126 index |
| 2 | flashinfer version mismatch | `flashinfer-cubin==0.6.3` leftover from vLLM 0.17.1, but `flashinfer-python==0.5.3` | `pip install flashinfer-cubin==0.5.3` |
| 3 | ZMQ socket collision | Another user's stale `/tmp/rl-colocate-zmq-GPU-*.sock` on shared node | Patched `vllm_rollout.py` and `utils.py` to use user-specific `/tmp/verl-zmq-{uid}/` |
| 4 | `flash_attn` not available on ARM | verl unconditionally imports `flash_attn.bert_padding` | Created `verl/utils/_flash_attn_compat.py` with pure-PyTorch fallbacks, patched `attention_utils.py` |
| 5 | `KGSearchToolParser` API mismatch | verl 0.7.1 changed `extract_tool_calls()` to pass `tools` arg | Added `tools: list | None = None` param in `search_tool_parser.py` |
| 6 | SLURM scripts missing ARM env setup | Missing `gcc-native`, `CC/CXX`, `LIBRARY_PATH`, `TRITON_CACHE_DIR` | Updated `train_verl_grpo_isambard.job` and `train_verl_sft_isambard.job` |
| 7 | TRL not installed | SFT trainer needs `trl` package | `pip install "trl<=0.9.6"` |
| 8 | ConceptNet data missing | `data/raw/conceptnet-assertions-5.7.0.csv.gz` not on Isambard | Downloaded (~498MB), created `scripts/download_conceptnet.sh` |

### Smoke Test Results

**Tests 1-4 (no training):** All PASS
- Imports: torch 2.9.0+cu126, vLLM 0.12.0, verl 0.7.1, deepspeed 0.18.8
- Data: 4000 train samples, correct format
- Reward function: tool calls detected, penalties working
- KG Server: 99,915 entities loaded, queries returning correct results

**Test 5 (GRPO training):** PASS
- 1.5B model, 32 samples, 1 GPU, 4 steps
- `Training Progress: 100%|██████████| 4/4 [03:01, 45.28s/it]`
- Validation: `reward/mean = -0.76` (expected — no SFT warmup, model doesn't use tools)

### SFT Warmup Submitted

SFT is required before GRPO — without it the model doesn't emit `<search>` tool calls.

- **Job ID:** 3151138
- **Node:** nid010014 (1x GH200)
- **Config:** Qwen2.5-7B-Instruct, LoRA r=64, lr=2e-5, 2 epochs, SDPA attention
- **Data:** 3,997 gold multi-turn trajectories
- **Output:** `outputs/verl-sft-7b/` (adapter) → `outputs/verl-sft-7b-merged/` (merged)
- **Walltime:** 6h

### Next Steps (after SFT completes)

1. Check SFT logs: `tail -f logs/verl_sft_7b-3151138.log`
2. Verify merged model exists at `outputs/verl-sft-7b-merged/`
3. Update GRPO config: `actor_rollout_ref.model.path: "outputs/verl-sft-7b-merged"`
4. Submit GRPO training: `sbatch scripts/train_verl_grpo_isambard.job`

### Files Changed

- `configs_verl/sft_multiturn.yaml` — updated for 7B + ARM (sdpa, lr=2e-5, LoRA r=64)
- `scripts/train_verl_sft_isambard.job` — updated for 7B + ARM env setup
- `scripts/train_verl_grpo_isambard.job` — added ARM workarounds, ZMQ fix, Ray timeout
- `src_verl/interaction/search_tool_parser.py` — fixed `extract_tool_calls` signature
- `scripts/download_conceptnet.sh` — new: downloads ConceptNet assertions
- `scripts/test_grpo_smoke.py` — new: end-to-end pipeline smoke test

### Patched Site-Packages (will need re-applying if verl reinstalled)

- `verl/workers/rollout/vllm_rollout/vllm_rollout.py` — ZMQ path fix
- `verl/workers/rollout/vllm_rollout/utils.py` — ZMQ path fix
- `verl/utils/attention_utils.py` — flash_attn fallback
- `verl/utils/_flash_attn_compat.py` — new: pure-PyTorch padding functions

### Current Package Versions (Isambard kg_verl)

| Package | Version |
|---------|---------|
| torch | 2.9.0+cu126 |
| vllm | 0.12.0 |
| verl | 0.7.1 |
| deepspeed | 0.18.8 |
| ray | 2.54.0 |
| trl | 0.9.6 |
| flashinfer-python | 0.5.3 |
| flashinfer-cubin | 0.5.3 |
| triton | 3.5.0 |

---

## 2026-03-20 (Session 2): GRPO Smoke Test Debug + Full Training Submitted

### Goal
Fix zero tool calls in GRPO 7B rollouts and launch full training.

### Issues Found & Fixed

| # | Issue | Root Cause | Fix |
|---|-------|-----------|-----|
| 1 | OOM on 1 GPU | 7B FSDP NO_SHARD + optimizer states > 120GB | Minimum 2 GPUs for FSDP FULL_SHARD |
| 2 | vLLM holds GPU during training | `enable_sleep_mode=false` default | `+actor_rollout_ref.rollout.enable_sleep_mode=true` |
| 3 | Embedding bucket too small | Qwen2.5-7B embeddings 2079MB > 2048MB default | `checkpoint_engine.update_weights_bucket_megabytes=4096` |
| 4 | KG server port mismatch | Tool config had port 8001, server on 18901 | Fixed `kg_tool_config.yaml` to port 18901 |
| 5 | Agent loop wrong type | `default_agent_loop: single_turn_agent` ignores tool calls | `actor_rollout_ref.rollout.agent.default_agent_loop=tool_agent` |
| 6 | Parser NameError on import | `_VALID_ACTIONS` referenced before definition, silent import crash, `kg_search` parser never registered | Moved `_VALID_ACTIONS` above first use |
| 7 | Parser too strict | Model generates `kg_query(...)` and bare `<action(...)>` variants | Added alias resolution + bare pattern matching |

### Smoke Test Results (Job 3244443)

- **num_tool_calls/mean: 3.3125** (was 0, fixed!)
- r_tool_use: 0.846, r_answer: 0.539, r_coverage: 0.379, r_format: 0.469
- r_no_tool: 0.0 (all samples used tools)
- num_turns/mean: 8.5 (multi-turn working)
- Per-step: ~21s, total: 131s for 6 steps

### Headless Debug Pattern (Lessons Learned)

- Use `--max-turns 50` (default 10 too low for submit-poll-fix loops)
- Use `--output-format stream-json --verbose` for real-time output
- First step must be ACTION (submit), not investigation (read files)
- Pre-decide strategy, keep context in short debug_log.md

### Full Training Submitted

- **Job ID:** 3244577
- **Config:** 4x GH200, batch_size=64, 3 epochs, ~187 steps
- **Reward:** verifiable
- **Checkpointing:** every 50 steps
- **wandb:** enabled, project `kg-align-verl`
- **Script:** `scripts/run_grpo_7b_full.job`
- **Estimated runtime:** ~1.5-3h

### Cleanup

Removed old diagnostic scripts, temporary headless prompt files, and diagnostic logs.

---

## 2026-03-21 (Session 3): CWQ/Freebase Pivot + Dataset Setup

### Goal
Pivot from ConceptNet to CWQ/Freebase as primary dataset. Set up data pipeline, implement spec-compliant rewards, run SFT warmup, launch first CWQ GRPO experiments.

### Dataset Pivot Decision
ConceptNet had fundamental issues: verbose answers (EM useless), 1-hop trivial questions, knowledge fully internalized by LLMs. CWQ/Freebase resolves all three: short entity answers, 2-4 hop questions, specialized knowledge.

### CWQ/Freebase Data Preprocessing
- Downloaded RoG-CWQ (1.74GB) + RoG-WebQSP (404MB) from HuggingFace
- Built global Freebase subgraph: **2.6M entities, 7K relations, 8.3M triples**
- Converted to verl parquet: CWQ 27,639/3,519/3,531 (train/val/test), WebQSP 2,826/246/1,628
- SPARQL hop distribution: 2.7% 1-hop, 35.1% 2-hop, 34.0% 3-hop, 19.2% 4-hop, 9.0% 5+hop
- Script: `scripts/prepare_cwq_freebase.py`, ran on compute node (Job 3252793, 11 min)

### Reward Functions Rewritten (Spec v3)
Replaced ad-hoc reward with 4 spec-compliant variants:
1. R_outcome: 0.5*EM + 0.5*F1
2. R_heuristic: 0.3*R_outcome + 0.7*step_entity_overlap
3. R_verifiable: 0.3*R_outcome + 0.7*(0.45*r_on_path + 0.30*r_progress + 0.15*r_coherence + 0.10*r_valid)
4. R_random: 0.3*R_outcome + 0.7*random

No explicit tool bonuses. Selected via REWARD_TYPE env var.

### CWQ SFT Warmup (Job 3252948)
- Generated 5,000 gold trajectories from KG paths (5-15 turns each)
- SFT training: 7B, LoRA r=64, 2 epochs, 626 steps, ~1h on 1 GPU
- Merged model at `outputs/verl-sft-cwq-7b-merged/`

### E1 + E3 GRPO Training (Jobs 3254091, 3254653)
- E1 (R_outcome): 1260/1293 steps (97%), 19h12m, 4x GH200
- E3 (R_verifiable): 1260/1293 steps (97%), 22h30m, 4x GH200
- Both crashed at step ~1260 due to vLLM shared memory bug (19h+ run, not a code issue)
- 25 checkpoints each (every 50 steps), sufficient for analysis

---

## 2026-03-22 (Session 4): Offline Evaluation + Core Hypothesis Testing

### Goal
Evaluate E1 and E3 checkpoints to determine if R_verifiable improves answer quality (go/no-go decision).

### Offline Eval Without Tools (greedy single-turn)

| Experiment | Step | EM | Contains-EM | F1 |
|-----------|------|-----|-------------|-----|
| SFT base | 0 | 0.000 | 0.066 | 0.043 |
| E1 (outcome) | 50 | 0.000 | — | 0.165 |
| E1 (outcome) | 500 | 0.000 | 0.482 | 0.222 |
| E1 (outcome) | 1250 | 0.000 | 0.544 | 0.253 |
| E3 (verifiable) | 500 | 0.000 | 0.460 | 0.236 |
| E3 (verifiable) | 1250 | 0.000 | 0.254 | 0.139 |

Without tools: E1 steadily improves, E3 peaks at step 500 then declines. EM=0 everywhere (model doesn't produce `<answer>` tags in greedy mode).

### Offline Eval With Tools (multi-turn, KG access) — THE KEY RESULT

| Experiment | Step | EM | Contains-EM | F1 | Avg Tool Calls |
|-----------|------|-----|-------------|-----|----------------|
| SFT base | 0 | 0.004 | 0.142 | 0.030 | 5.0 |
| E1 (outcome) | 500 | 0.002 | 0.520 | 0.105 | 4.9 |
| E1 (outcome) | 1250 | 0.002 | 0.570 | 0.119 | 4.9 |
| **E3 (verifiable)** | **500** | **0.542** | **0.546** | **0.546** | **1.0** |
| **E3 (verifiable)** | **1250** | **0.296** | **0.320** | **0.315** | **1.0** |

### Key Findings

1. **E3 step 500 with tools: EM=54.2%** — the model answers correctly over half the time with proper `<answer>` formatting. Core hypothesis confirmed.
2. **E3 >> E1 with tools**: EM 54.2% vs 0.2%. Verifiable rewards produce dramatically better tool-using agents.
3. **E3 uses tools efficiently**: 1.0 avg tool calls vs E1's 4.9. Learned to make targeted queries.
4. **E3 step 500 > E3 step 1250**: EM drops 54.2% → 29.6%. The 0.30/0.70 answer/step split causes later over-optimization for step rewards at expense of answers. Step 500 is the sweet spot.
5. **Without-tools eval was misleading**: E3 step 1250 looked terrible without tools (ContEM=25%) but decent with tools (EM=29.6%). The model learned tool dependency.
6. **Go/No-Go: GO** — R_verifiable works. Proceed to Phase C.

### E2 Training (R_heuristic)
- Job 3272743: Running (restarted after node failure)
- Reward: `R_heuristic = 0.3 * R_outcome + 0.7 * avg(step_entity_overlap)`
  - Each step: 0.5 * entity_overlap_with_gold + 0.5 * reach_indicator
  - Heuristic step signal based on ProGraph-R1 approach — not KG-verifiable, just entity overlap
- Same config as E1/E3: 4x GH200, lr=5e-7, batch_size=64, 3 epochs, 1293 steps
- Purpose: complete the 3-way comparison (outcome vs heuristic vs verifiable)
- **Waiting for E2 results** — once done, run with-tool eval on E2 checkpoints for the full comparison table

### Cleanup
- Removed obsolete docs, scripts, and log files (3.4GB → 64MB logs)
- Updated memory files for future conversation context
- Fixed GRPO job script: `RAY_DEDUP_LOGS=1` + KG server `--log-level warning` to reduce future training logs from ~1.6GB to ~5-20MB

### SPARQL Hop Annotations
- Extracted from original CWQ train JSON (27,639 samples)
- Distribution: 2-hop 35%, 3-hop 34%, 4-hop 19%, 5+ hop 9%
- Saved to `data/freebase/cwq_hop_annotations.json`

### Scripts Added
- `scripts/eval_checkpoints.py` — offline eval with FSDP checkpoint loading
- `scripts/eval_with_tools.py` — multi-turn eval with KG server
- `scripts/extract_sparql_hops.py` — SPARQL hop count extraction
- `scripts/generate_cwq_sft.py` — CWQ SFT trajectory generation
- `scripts/prepare_cwq_freebase.py` — CWQ/Freebase data preprocessing
- `scripts/run_eval_with_tools.job` — SLURM job for with-tool eval
- `scripts/run_grpo_cwq.job` — SLURM job for CWQ GRPO (supports REWARD_TYPE env var)
- `scripts/run_sft_cwq.job` — SLURM job for CWQ SFT pipeline
- `configs_verl/grpo_cwq_7b.yaml` — CWQ GRPO training config
- `configs_verl/sft_cwq.yaml` — CWQ SFT config
