# KG-Align-RL

Cross-domain transfer in knowledge graph-aligned RL for LLM reasoning. Reference paper: Kansal & Jha (arXiv:2601.15160).

For detailed pipeline rationale and stage-by-stage plan, see @MINIMAL_VALIDATION_PLAN.md

## Architecture

```
src/
  kg/                   # KG loading, path extraction, subgraph sampling
  datagen/              # Question generation from KG paths, quality filtering
  rewards/              # KG path-alignment reward functions
  training/             # SFT warmup + GRPO training loop
  evaluation/           # Benchmark evaluation scripts
configs/                # YAML configs (sft_warmup.yaml, grpo_validation.yaml)
scripts/                # SLURM job scripts for Viking HPC
tests/                  # Unit tests (reward function, data pipeline)
data/
  raw/                  # ConceptNet dump (not committed)
  processed/            # Extracted paths, generated QA pairs (not committed)
  eval/                 # Evaluation benchmark subsets (not committed)
```



## Environment

This project runs on two HPC clusters. Detect which one from `uname -m`: `x86_64` = Viking, `aarch64` = Isambard.

### Viking (University of York)
- **GPU**: NVIDIA H100 80GB PCIe, 1-2 per node
- **Scratch**: `/mnt/scratch/users/ts1201/`
- **HF cache**: `export HF_HOME=/mnt/scratch/users/ts1201/hf_cache`
- **Conda**: `module load Miniconda3` then `source activate kg_verl` (NOT `conda activate`)
- **CUDA**: `module load CUDA/12.8.0`
- **Conda env**: `kg_verl` (for verl pipeline) or `kg_align` (for TRL pipeline)

### Isambard-AI (Bristol, GH200)
- **GPU**: NVIDIA GH200 120GB HBM3, **4 per node** (each gets 72 CPU cores + 115GB RAM)
- **CPU**: ARM aarch64 (Grace Hopper) — no x86 wheels, use conda-forge
- **Scratch**: `$SCRATCH` or `$SCRATCHDIR` (`/scratch/<PROJECT>/<USER>.<PROJECT>`)
- **HF cache**: `export HF_HOME=$SCRATCH/hf_cache`
- **Conda**: Miniforge self-installed at `~/miniforge3/bin/activate` then `conda activate kg_verl`
- **CUDA**: `module load cudatoolkit`
- **NCCL**: `module load brics/nccl` (required for multi-GPU)
- **Partition**: `workq`, QOS: `workq_qos`, max walltime: **24h**, max GPUs/project: **32**
- **flash-attn**: NOT available on ARM — use `attn_implementation: "sdpa"` everywhere
- **GPU request**: `#SBATCH --gpus=N` (not `--gpus-per-node`)

### Common
- **Python**: 3.10+ (via conda)
- **Conda env**: `kg_verl`
- Always add `export PYTHONPATH=$PWD:$PYTHONPATH` in SLURM scripts
- Do NOT use `set -u` in SLURM scripts (causes unbound variable errors with conda)

## Coding Standards

- Python type hints on all function signatures
- Dataclasses or Pydantic for config objects — no hardcoded hyperparameters in scripts
- `logging` module, not `print`
- `pathlib.Path` for file paths, not string concatenation
- Seed everything for reproducibility; log all hyperparameters to wandb
- Every module must have a `if __name__ == "__main__":` block with argparse or Hydra for standalone use


## HPC/SLURM Notes
- Viking: `module load Miniconda3 && source activate kg_verl` — never use `conda activate`
- Isambard: `source ~/miniforge3/bin/activate && conda activate kg_verl`
- Viking GPU: `--gpus-per-node=N`; Isambard GPU: `--gpus=N`
- Always include CUDA module in SLURM scripts
- Do NOT use `set -u` in SLURM scripts (causes unbound variable errors with conda)
- Isambard 24h walltime limit — checkpoint every 200 steps, use job chaining for long runs
- Isambard multi-GPU: `module load brics/nccl` + set `NCCL_SOCKET_IFNAME=hsn`

## Python/ML Libraries
- When using TRL's SFTConfig, the parameter is `max_seq_length` (not `max_length`)
- Always check current API signatures before generating training configs
- Verify wandb is either configured or disabled (`WANDB_DISABLED=true`) in SLURM scripts


## Model & LoRA Configuration

- **Validation model**: `Qwen/Qwen2.5-1.5B-Instruct` with `torch.bfloat16` (switched from base model — base model never produces `<|im_end|>` EOS tokens, causing GRPO to fail)
- **LoRA variant**: DoRA (`use_dora=True`) with rsLoRA (`use_rslora=True`)
- **LoRA rank**: 16, alpha: 32
- **Target modules**: `"all-linear"` (q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj)
- **Attention**: Flash Attention 2 (`attn_implementation="flash_attention_2"`)
- Do NOT use PiSSA initialisation — it causes instability with GRPO training

## GRPO Training

- Framework: TRL's `GRPOTrainer`
- `scale_rewards: false` (Dr. GRPO bias fix)
- `beta: 0.0` (no KL penalty)
- `num_generations: 8`, `temperature: 0.7`
- `optim: "adamw_8bit"`
- Gradient checkpointing: always enabled
- vLLM colocate mode when available (`use_vllm=True, vllm_mode="colocate"`)
- If vLLM is unavailable in the TRL version, fall back to standard HF generate — do not error

## Reward Function Design

Two reward functions passed as a list to `GRPOTrainer.reward_funcs`:

1. **KG path-alignment reward** (`compute_kg_reward`):
   - Token-level path coverage with minimum-hit constraint (>= 2 entities)
   - Repetition penalty to prevent entity stuffing
   - Asymmetric binary correctness: +0.1 correct, -1.0 incorrect
   - Based on Kansal & Jha's proven ablation results
2. **Format reward** (`compute_format_reward`):
   - Checks for `<think>...</think>` tags with non-empty reasoning (> 20 chars)

Do NOT add extra reward signals (similarity, thinking quality) — the reference paper showed this causes performance collapse.

## Data Format

All QA data uses `<think>` CoT format:
```json
{
  "question": "...",
  "answer": "<think>[step-by-step KG path reasoning]</think>\n[short final answer]",
  "kg_path": [["entity1", "relation", "entity2"], ...],
  "gold_answer_short": "...",
  "hops": 2
}
```

## Training Pipeline Order

### TRL pipeline (src/, single-turn, original)
1. **SFT warmup** (2 epochs) — teaches `<think>` format before RL
2. **GRPO with curriculum** — 1-hop first, then 2-hop, then 3-hop paths

### verl pipeline (src_verl/, multi-turn agentic, current focus)
1. **SFT warmup** — teaches `<think>/<search>/<answer>` multi-turn tool-use format
2. **Merge SFT adapter** — verl needs full model, not PEFT adapter
3. **GRPO training** — multi-turn rollouts with live KG tool calls
4. **Evaluation** — 3-way comparison: base vs SFT vs SFT+GRPO

### verl reward function (CRITICAL knowledge)
- verl's `NaiveRewardManager` does NOT pass `tool_rewards` from `ToolAgentLoop` to `compute_score`
- Therefore `verl_reward.py` parses `<search>` tags directly from the flat `solution_str`
- Outputs without any `<search>` tags get a -1.0 penalty (r_no_tool)
- SFT training, GRPO rollouts, and evaluation MUST all use the same tool response format:
  - Role: `"tool"` (native, not mapped to `"user"`)
  - Content: wrapped in `<information>...</information>` tags
- Qwen2.5 renders `role="tool"` as `<|im_start|>user\n<tool_response>\n...\n</tool_response>`

## Common Commands

```bash
# === Viking ===
module load Miniconda3 && module load CUDA/12.8.0 && source activate kg_verl

# === Isambard ===
source ~/miniforge3/bin/activate && conda activate kg_verl
module load cudatoolkit && module load brics/nccl

# === Both ===
# Run tests
python -m pytest tests/ -v

# Submit verl pipeline (Viking)
sbatch scripts/train_verl_sft.job          # SFT + auto-merge
sbatch scripts/train_verl_grpo.job         # GRPO training
sbatch scripts/eval_verl_grpo.job          # Evaluation

# Submit verl pipeline (Isambard)
sbatch scripts/train_verl_sft_isambard.job
sbatch scripts/train_verl_grpo_isambard.job
sbatch scripts/eval_verl_isambard.job

# Check jobs
squeue -u $USER
```

## Important Constraints

- Keep all large files on scratch (home quota is limited on both clusters)
- SLURM jobs have wall-time limits — checkpoint every 200 steps
- Isambard max walltime is **24h** — use `--dependency=afterok:JOBID` for chaining
- ConceptNet assertions: ~30M triples; always work with filtered subsets (weight >= 2.0, English only)
- GRPO generation is the bottleneck — use vLLM colocate when possible
- HotpotQA: use the `bridge` subset for multi-hop focus
- Isambard is ARM aarch64 — no flash-attn, always use `attn_implementation: "sdpa"`
- `ToolAgentLoop._call_tool()` calls `create()/release()` per tool call, not per trajectory — step reward state resets between calls

## Git Workflow

- Feature branches per pipeline component
- Commit format: `type(scope): description` (e.g., `feat(kg): add ConceptNet path extractor`)
- Do not commit: model weights, large datasets, wandb logs, conda envs
- See `.gitignore` for full exclusion list
