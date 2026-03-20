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
