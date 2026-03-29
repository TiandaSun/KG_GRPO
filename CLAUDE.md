# KG-Align-RL: Verifiable Process Supervision via Knowledge Graphs

## Project goal

Train agentic KG reasoning models with GRPO, comparing three reward types (outcome / heuristic-step / KG-verifiable-step) to study when and why reward verifiability matters. KG is a **verification oracle**, not a knowledge source.

Target: EMNLP 2026 (May 25). Paper framing is insight/analysis, not method.

## Core implementation spec

See `hpc_implementation_spec.md` for the full specification (experiments, rewards, timeline, compute budget). This file covers daily workflow rules, environment setup, and coding standards.

## Active task queue

See `hpc_tasks.md` for the current task queue. This file is the **primary source of truth** for what to work on next. It is updated from the discussion repo after each planning iteration. Always read this file at the start of a conversation to know the current priorities.

## Key references

- Current code repo: https://github.com/TiandaSun/KG_GRPO (sync between Isambard and Viking)
- KG-R1: https://github.com/Jinyeop3110/KG-R1 — closest template for verl + multi-turn KG GRPO
- Search-R1: https://github.com/PeterGriffinJin/Search-R1 — verl + multi-turn search agent
- verl-tool: https://github.com/TIGER-AI-Lab/verl-tool — unified tool API for verl
- Isambard docs: https://docs.isambard.ac.uk/user-documentation/guides/python/#conda-installing-and-using-miniforge

## Historical context

Previous TRL-based experiments (1.5B and 7B, single-turn, Feb 2026) are documented in:
- `PROJECT_SUMMARY.md` — results, lessons learned, reward evolution v1→v2→v3
- `CHANGELOG.md` — detailed code change history

The current project has pivoted to **verl** (multi-turn agentic) based on those findings. Existing code under `src/` and `src_verl/` may be reused.

---

## Environment

This project runs on two HPC clusters. Detect which one from `uname -m`: `x86_64` = Viking, `aarch64` = Isambard.

### Viking (University of York)
- **GPU**: NVIDIA H100 80GB PCIe, 1-2 per node
- **Scratch**: `/mnt/scratch/users/ts1201/`
- **HF cache**: `export HF_HOME=/mnt/scratch/users/ts1201/hf_cache`
- **Conda**: `module load Miniconda3` then `source activate kg_verl` (NOT `conda activate`)
- **CUDA**: `module load CUDA/12.8.0`
- **Role**: development, debugging, SFT experiments, 1.5B ablation

### Isambard-AI (Bristol, GH200)
- **GPU**: NVIDIA GH200 120GB HBM3, **4 per node** (each gets 72 CPU cores + 115GB RAM)
- **CPU**: ARM aarch64 (Grace Hopper) — no x86 wheels, use conda-forge
- **Scratch**: `/scratch/u6gg/ts1201.u6gg`
- **HF cache**: `export HF_HOME=/scratch/u6gg/ts1201.u6gg/hf_cache`
- **Conda**: `source ~/miniforge3/bin/activate && conda activate kg_verl`
- **CUDA**: `module load cudatoolkit`
- **NCCL**: `module load brics/nccl` (required for multi-GPU)
- **Partition**: `workq`, QOS: `workq_qos`, max walltime: **24h**, max GPUs/project: **32**
- **Role**: all 7B and 14B GRPO experiments (2,500 node-hours = 10,000 GPU-hours)

#### Isambard ARM workarounds (CRITICAL)
- **Compiler**: Default `nvc/nvc++` breaks Triton/DeepSpeed JIT. MUST set: `module load gcc-native/14.2 && export CC=gcc CXX=g++`
- **curand**: DeepSpeed JIT needs: `export LIBRARY_PATH=/opt/nvidia/hpc_sdk/Linux_aarch64/24.11/REDIST/math_libs/12.6/targets/sbsa-linux/lib:$LIBRARY_PATH`
- **vLLM**: Must use `enforce_eager=True` (Triton graph compilation fails on ARM)
- **SGLang**: Engine broken on ARM (sgl_kernel ABI mismatch). **Use vLLM backend for verl instead.**
- **flash-attn**: NOT available on ARM — use `attn_implementation: "sdpa"` (but vLLM uses its own FlashAttention v3)
- **Triton cache**: `export TRITON_CACHE_DIR=/scratch/u6gg/ts1201.u6gg/.triton` (avoid NFS home)
- **GPU request**: `#SBATCH --gpus=N` (not `--gpus-per-node`)

### Common
- **Python**: 3.10 (via conda)
- **Conda env**: `kg_verl`
- **Framework**: verl + vLLM + DeepSpeed (NOT TRL — TRL multi-turn GRPO has critical bugs)
- Always add `export PYTHONPATH=$PWD:$PYTHONPATH` in SLURM scripts
- Do NOT use `set -u` in SLURM scripts (causes unbound variable errors with conda)

---

## Models

| Role | Model | Purpose |
|------|-------|---------|
| **Primary** | Qwen2.5-7B-Instruct | All tasks, all rewards, all KGs, full analysis |
| **Scaling** | Qwen2.5-14B-Instruct | Core QA task, verify findings hold at scale |
| **Ablation** (optional) | Qwen2.5-1.5B-Instruct | Show "too small fails" for multi-turn agent |

- Always use **Instruct** models (base models never produce `<|im_end|>` EOS, causing GRPO to fail)
- Do NOT use PiSSA initialisation — causes instability with GRPO

## Three Tasks

1. **KG-Grounded Multi-hop QA** — agent queries KG to answer questions (comparable to KG-R1)
2. **Claim Verification** — agent verifies each sub-claim against KG (our distinctive task)
3. **Think-then-Verify** — model reasons freely, then verifies each step via KG, then revises

## Four Reward Functions

1. **R_outcome** — answer EM + F1 only (baseline)
2. **R_heuristic_step** — R_outcome + entity overlap per step (ProGraph-R1 style)
3. **R_verifiable_step** — R_valid + R_on_path + R_progress + R_coherence (our method)
4. **R_random** — random step rewards + real outcome (ablation)

See `hpc_implementation_spec.md` Section 5 for full pseudocode.

## Training Pipeline (verl)

1. **SFT warmup** — teaches multi-turn tool-use format
2. **Merge SFT adapter** — verl needs full model, not PEFT adapter
3. **GRPO training** — multi-turn rollouts with live KG tool calls
4. **Evaluation** — with-KG and without-KG reasoning quality

## KG Server

Unified API server supporting all KGs with 5 endpoints:
- `search(entity, relation)` — forward search
- `search_reverse(entity, relation)` — reverse search
- `get_relations(entity)` — list relations
- `verify(head, relation, tail)` — triple existence check (verification oracle)
- `shortest_path(entity_a, entity_b)` — path finding

Must run as a **separate HTTP process**, not embedded in training loop.

---

## Project structure (target)

```
src/
  kg_server/          # KG retrieval server (5 endpoints)
  rewards/            # 4 reward functions (outcome, heuristic_step, verifiable_step, random)
  data/               # SFT data generation for 3 tasks
  training/           # verl GRPO configs and launch scripts
  evaluation/         # Eval metrics including without-KG reasoning quality
scripts/              # SLURM job scripts
configs/              # verl training configs per experiment
data/
  conceptnet/         # Existing ConceptNet data
  freebase/           # WebQSP + CWQ
  wikidata/           # T-REx or KGQAGen-10k (P1)
```

Existing code under `src/` (TRL pipeline) and `src_verl/` (early verl work) may be reused where applicable.

---

## Workflow rules

- Always run small-scale validation (100 samples, 7B, ConceptNet) before launching full experiments
- Log ALL training runs to W&B with descriptive names: `{model}_{kg}_{reward}_{task}_{date}`
- Save checkpoints every 50 steps for Goodhart analysis
- Save 100 random trajectories at steps 0, 250, 500, 750, 1000 for hacking taxonomy analysis
- Never overwrite existing checkpoints — use versioned directories
- Prefer SLURM job scripts over interactive sessions for GPU runs

## Coding Standards

- Python type hints on all function signatures
- Dataclasses or Pydantic for config objects — no hardcoded hyperparameters in scripts
- `logging` module, not `print`
- `pathlib.Path` for file paths, not string concatenation
- Seed everything for reproducibility; log all hyperparameters to wandb
- Every module must have a `if __name__ == "__main__":` block with argparse or Hydra for standalone use

## Important constraints

- KG server must run as a separate process, NOT embedded in training loop
- Token masking: system prompt + tool response tokens must be masked from policy gradient (verl delta-based tokenization)
- Reward function weights (0.3/0.3/0.2/0.2 and 0.4/0.6) are initial — tune on small-scale first
- Do NOT use MetaQA (20% factual accuracy). Use WebQSP/CWQ for Freebase.
- For Think-Verify task: model must NOT access KG during Think phase, only during Verify phase
- Keep all large files on scratch (home quota is limited on both clusters)
- Isambard 24h walltime limit — checkpoint every 50 steps, use `--dependency=afterok:JOBID` for chaining

## Common Commands

```bash
# === Viking ===
module load Miniconda3 && module load CUDA/12.8.0 && source activate kg_verl

# === Isambard ===
source ~/miniforge3/bin/activate && conda activate kg_verl
module load cudatoolkit && module load gcc-native/14.2 && module load brics/nccl
export CC=gcc CXX=g++
export LIBRARY_PATH=/opt/nvidia/hpc_sdk/Linux_aarch64/24.11/REDIST/math_libs/12.6/targets/sbsa-linux/lib:$LIBRARY_PATH
export TRITON_CACHE_DIR=/scratch/u6gg/ts1201.u6gg/.triton
export HF_HOME=/scratch/u6gg/ts1201.u6gg/hf_cache

# === Both ===
python -m pytest tests/ -v
squeue -u $USER
```

## HPC / SLURM Workflows

- **Pre-flight validation**: Before submitting any SLURM job, always validate Python scripts with `python -c 'import ast; ast.parse(open("script.py").read())'`. Double-check: 1) correct parameter names for the scheduler, 2) all Python imports are present, 3) f-strings are properly quoted. Each resubmission costs 5-15 min of queue time.
- **ARM PyTorch wheels**: On aarch64/ARM (GH200) systems, always use `--index-url` with the correct PyTorch CUDA wheel URL when pip installing. Never rely on default pip resolution for torch on ARM — it pulls CPU-only wheels.
- **Ray init on GH200**: Always pass explicit `num_cpus` to `ray.init()` on Isambard. Auto-detection hangs on ARM. Use `ray_kwargs.ray_init.num_cpus: 72 * N_GPUS` in verl configs.

## Git Workflow

- Feature branches per pipeline component
- Commit format: `type(scope): description` (e.g., `feat(kg): add ConceptNet path extractor`)
- Do not commit: model weights, large datasets, wandb logs, conda envs
- See `.gitignore` for full exclusion list
