# Headless Debug Prompt: Fix Model Tool Call Issue in GRPO Training

Paste everything below the `---` line into Claude Code headless mode.

---

You are debugging a verl-based multi-turn GRPO training pipeline for KG reasoning. Your goal is to make the model successfully use tools (KG queries) during GRPO rollouts. Currently the model NEVER makes tool calls — all rewards are -0.95 with `num_tool_calls: 0.0`.

## Problem Statement

After SFT warmup and GRPO smoke test (6 steps, 2 GPUs, job 3234636), the model produces zero tool calls during rollouts. The validation metrics show:
- `r_tool_use: 0.0`, `r_no_tool: -1.0`, `num_tool_calls: 0.0`, `num_turns: 2` (just prompt+response)
- One warning: `Unknown action in <search> tag: kg_query` — the model emitted `<search>kg_query(...)</search>` but the parser expects `<search>get_tail_relations(...)</search>`

## Success Criteria

After your fixes, a GRPO smoke test should show:
- `num_tool_calls > 0` (model actually uses tools, even if answers are wrong)
- No `Unknown action` warnings (parser accepts what the model generates)
- Training completes without OOM or crashes on up to 4 GPUs

## Root Cause Hypotheses (investigate in order)

1. **SFT format mismatch**: The SFT training data teaches `<search>tool_name(args)</search>` format, but the merged model may not have learned it well enough, or the SFT was too short/weak. Check `data/processed/sft_trajectories.jsonl` for the training format.

2. **Parser too strict**: `src_verl/interaction/search_tool_parser.py` line 70 rejects anything not in `_VALID_ACTIONS = {get_tail_relations, get_head_relations, get_tail_entities, get_head_entities}`. If the model writes `<search>kg_query(...)` or `<search>search(...)` or any other variant, it's silently dropped. Consider making the parser more flexible or adding fallback parsing.

3. **verl multi-turn agent loop not triggering**: The model might not be generating `<search>` tags at all (the `kg_query` warning was only 1 occurrence in 6 steps). Check if verl's AgentLoopWorker is actually parsing model output for tool calls. The config uses `format: "kg_search"` which maps to `KGSearchToolParser`.

4. **SFT warmup quality**: The SFT may need more epochs, higher learning rate, or better data. Check `configs_verl/sft_multiturn.yaml` and the SFT training logs. The merged model is at `outputs/verl-sft-7b-merged`.

5. **System prompt not injected**: The GRPO prompts in `data/processed/verl_conceptnet/train.parquet` contain a system message teaching tool format. Verify this system message actually appears in the model's context during rollouts.

## Environment (Isambard-AI, GH200 ARM aarch64)

```bash
# Always run these before any Python/SLURM work:
source ~/miniforge3/bin/activate && conda activate kg_verl
module load cudatoolkit 2>/dev/null || true
module load gcc-native/14.2 2>/dev/null || true
module load brics/nccl 2>/dev/null || true
export CC=gcc CXX=g++
export LIBRARY_PATH=/opt/nvidia/hpc_sdk/Linux_aarch64/24.11/REDIST/math_libs/12.6/targets/sbsa-linux/lib:${LIBRARY_PATH:-}
export TRITON_CACHE_DIR=/scratch/u6gg/ts1201.u6gg/.triton
export HF_HOME=/scratch/u6gg/ts1201.u6gg/hf_cache
export PYTHONPATH=$PWD:${PYTHONPATH:-}
```

- GPU: NVIDIA GH200 120GB HBM3, 4 per node
- Partition: `workq`, max walltime 24h
- SLURM: `#SBATCH --gpus=N` (NOT --gpus-per-node)
- vLLM: must use `enforce_eager=True`
- Flash-attn not available on ARM — use `attn_implementation: "eager"` or `"sdpa"`
- Ray: must pass explicit `num_cpus` (72 per GPU)
- Do NOT use `set -u` in SLURM scripts

## Key Files

| File | Purpose |
|------|---------|
| `src_verl/interaction/search_tool_parser.py` | Parses `<search>action(args)</search>` → FunctionCall. **This is likely where the fix is needed.** |
| `src_verl/rewards/verl_reward.py` | Reward function. Detects `<search>` tags for r_tool_use, applies r_no_tool=-1 penalty. |
| `configs_verl/grpo_conceptnet_7b.yaml` | GRPO training config (already has wandb enabled via `trainer.logger: [wandb]`) |
| `configs_verl/kg_tool_config.yaml` | Tool schema definition (tool name is `kg_query`) |
| `configs_verl/sft_multiturn.yaml` | SFT warmup config |
| `src_verl/training/sft_multiturn.py` | SFT training script |
| `data/processed/sft_trajectories.jsonl` | SFT training data (gold multi-turn trajectories) |
| `data/processed/verl_conceptnet/train.parquet` | GRPO training prompts |
| `outputs/verl-sft-7b-merged` | Current merged SFT model (input to GRPO) |
| `scripts/run_grpo_7b_smoke.job` | Last smoke test job script (use as template) |
| `src_verl/kg_server/server.py` | KG server (start before GRPO, port 18901) |

## Constraints

- You may modify any Python file or config under this project
- You may re-run SFT warmup if needed (use up to 4 GPUs, ~1h expected)
- You may submit SLURM jobs with up to 4 GPUs (`--gpus=4`)
- wandb is authenticated and ready — keep `trainer.logger: [wandb]` enabled (do NOT set `WANDB_MODE=disabled`)
- After SFT, you must merge the LoRA adapter before GRPO (verl needs a full model)
- Maximum 15 GRPO SLURM job submissions total (budget your attempts)
- You may run up to 2 sub-agents in parallel for web searches (e.g., searching verl docs, KG-R1 repo, Search-R1 repo for how they handle tool calls)
- Always validate Python syntax before submitting: `python -c 'import ast; ast.parse(open("file.py").read())'`
- Each job resubmission costs 5-15 min queue time — be thorough before submitting

## What to log

Write all progress to `debug_log.md` in the project root. For each attempt, record:
- Attempt number and SLURM job ID
- What you changed and why
- Key metrics from the output (reward, num_tool_calls, errors)
- Whether it passed/failed and next steps

At the end, write a final summary section with:
- What worked / what didn't
- Final metrics
- Recommended next steps for full training

## Suggested Investigation Order

1. **First**: Read the SFT training data (`data/processed/sft_trajectories.jsonl`, first 3 entries) to understand what format the model was trained on. Also read the GRPO prompt data to see what system prompt the model sees during rollouts.

2. **Second**: Run a quick inference test on the merged SFT model (on login node, CPU-only is fine for 1 sample) to see what it actually generates given a KG reasoning prompt. This tells you if the SFT model learned the format at all.

3. **Third**: Based on findings, either:
   - (a) Fix the parser to accept what the model generates, OR
   - (b) Fix/redo the SFT to teach the correct format, OR
   - (c) Both

4. **Fourth**: Submit a GRPO smoke test (100 samples, 2-4 GPUs, ~6 steps) and check metrics.

5. **Iterate** until `num_tool_calls > 0` or you hit 15 attempts.

## Important Technical Details

- The smoke job script at `scripts/run_grpo_7b_smoke.job` is a working template. Key overrides for smoke test:
  - `data.train_batch_size=16`, small data files, `trainer.total_epochs=1`
  - `actor_rollout_ref.rollout.gpu_memory_utilization=0.5` (for 2 GPUs)
  - `+actor_rollout_ref.rollout.enable_sleep_mode=true` (required for colocated FSDP+vLLM)
  - `actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=4096`
- With 4 GPUs: set `trainer.n_gpus_per_node=4`, `ray_kwargs.ray_init.num_cpus=288`, can increase `gpu_memory_utilization` to 0.4-0.5
- The KG server must be started as a background process before GRPO launch (port 18901)
- verl uses Hydra for config — override with `key=value` on command line, use `+key=value` for new keys

Now start investigating. Begin by reading the SFT data and testing the model's generation, then proceed to fixes.
