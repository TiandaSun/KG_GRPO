# KG-Align-RL

Knowledge Graph-Aligned Reinforcement Learning for LLM Reasoning.

Trains LLMs to perform multi-turn knowledge graph reasoning using GRPO (Group Relative Policy Optimisation) with verifiable step-level rewards. The model learns to query a KG via tool calls and produce grounded, step-by-step answers.

**Reference**: Kansal & Jha, *"KG-Align: Aligning Language Model Reasoning with Knowledge Graph Structures"* (arXiv:2601.15160)

## Pipeline Overview

```
ConceptNet  →  KG Path Extraction  →  Question Generation  →  Quality Filtering
                                                                      ↓
                                              SFT Trajectories (multi-turn tool use)
                                                                      ↓
                                                    SFT Warmup (learns <think>/<search>/<answer> format)
                                                                      ↓
                                                    GRPO Training (with live KG tool calls + verifiable rewards)
                                                                      ↓
                                                    Evaluation (base vs SFT vs SFT+GRPO)
```

### Two pipeline implementations

| Pipeline | Framework | Location | Description |
|----------|-----------|----------|-------------|
| **TRL** (original) | TRL `GRPOTrainer` | `src/`, `configs/` | Single-turn, `<think>` CoT format |
| **verl** (current) | verl `main_ppo` | `src_verl/`, `configs_verl/` | Multi-turn agentic with live KG tool calls |

The verl pipeline is the active focus. The model learns to use `<search>get_tail_relations(entity)</search>` tool calls during GRPO rollouts, receiving step-level rewards for valid, on-path, progressive queries.

## Quick Start

### 1. Clone and setup

```bash
git clone https://github.com/YOUR_ORG/KG-Align-RL.git
cd KG-Align-RL

# Setup environment (auto-detects Viking x86 vs Isambard ARM)
bash scripts/setup_env.sh

# Download ConceptNet data
bash scripts/download_data.sh
```

### 2. Run tests

```bash
python -m pytest tests/ -v
```

### 3. Train (SLURM)

**Viking (H100)**:
```bash
sbatch scripts/train_verl_sft.job           # SFT warmup + auto-merge
sbatch scripts/train_verl_grpo.job          # GRPO with KG rewards
sbatch scripts/eval_verl_grpo.job           # 3-way evaluation
```

**Isambard (GH200)**:
```bash
sbatch scripts/train_verl_sft_isambard.job
sbatch scripts/train_verl_grpo_isambard.job   # 4x GH200 for 7B
sbatch scripts/eval_verl_isambard.job
```

## Project Structure

```
src_verl/                    # Multi-turn verl pipeline (current focus)
  kg_server/                 # HTTP KG query server (ConceptNet, Freebase)
  interaction/               # KGQueryTool (verl BaseTool) + search parser
  data/                      # Data preparation (JSONL → verl parquet)
  rewards/                   # Reward functions (outcome, heuristic, verifiable)
  training/                  # SFT warmup trainer
  evaluation/                # Multi-turn evaluation with live tool calls

src/                         # Original TRL pipeline (preserved, not active)
  kg/                        # ConceptNet path extraction
  datagen/                   # Question generation + quality filtering
  rewards/                   # KG reward functions (single-turn)
  training/                  # SFT + GRPO trainers
  evaluation/                # Benchmark evaluation

configs_verl/                # verl training configs (GRPO, SFT, tool, KG server)
configs/                     # TRL training configs
scripts/                     # SLURM jobs, environment setup, utilities
tests/                       # Unit tests (241 passing)
data/processed/              # Training/eval data (included in repo, ~36MB)
```

## Reward Functions

Three reward variants for controlled experiments:

| Reward | Components | Purpose |
|--------|-----------|---------|
| **R_outcome** | Answer EM + F1 | Baseline (no step signal) |
| **R_heuristic** | R_outcome + entity overlap per step | Weak step signal |
| **R_verifiable** | R_valid + R_on_path + R_progress + R_coherence | Full verifiable step reward |

The GRPO training reward (`verl_reward.py`) additionally includes:
- `r_tool_use`: Bonus for making valid `<search>` tool calls
- `r_no_tool`: **-1.0 penalty** for outputs without any tool calls
- `r_coverage`: KG path entity mention coverage
- `r_format`: Proper `<answer>` tag usage

## HPC Platforms

| | Viking | Isambard-AI |
|--|--------|-------------|
| GPU | H100 80GB | GH200 120GB |
| GPUs/node | 1-2 | 4 |
| Architecture | x86_64 | aarch64 (ARM) |
| Partition | `gpuplus` | `workq` |
| Max walltime | varies | 24h |
| Conda | `module load Miniconda3` | Miniforge (self-install) |
| CUDA | `module load CUDA/12.8.0` | `module load cudatoolkit` |
| flash-attn | Yes | No (SDPA fallback) |

## Data

**Included in repo** (`data/processed/`):
- `conceptnet_qa_{train,val,test}.jsonl` — 4000/500/500 QA pairs with KG paths
- `sft_trajectories.jsonl` — 3997 multi-turn tool-use trajectories for SFT
- `verl_conceptnet/{train,val,test}.parquet` — verl-format training data

**Downloaded separately** (via `scripts/download_data.sh`):
- ConceptNet assertions (~230MB compressed)
- Qwen2.5 models (via HuggingFace Hub)

## Key Design Decisions

- **Instruct model, not base**: Base Qwen2.5 never produces `<|im_end|>` EOS → GRPO fails
- **DoRA + rsLoRA**: Outperforms standard LoRA for RLVR (arXiv:2512.23165)
- **`scale_rewards: false`**: Dr. GRPO bias fix (arXiv:2503.20783)
- **SFT warmup before GRPO**: Small models have severe cold-start without it (DeepSeek-R1)
- **Tool response format**: Native `role="tool"` with `<information>` tags — must be consistent across SFT, GRPO rollouts, and evaluation

## Citation

```bibtex
@article{kansal2025kgalign,
  title={KG-Align: Aligning Language Model Reasoning with Knowledge Graph Structures},
  author={Kansal, Esha and Jha, Saurabh},
  journal={arXiv preprint arXiv:2601.15160},
  year={2025}
}
```
