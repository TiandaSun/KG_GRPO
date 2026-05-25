# KG-Align-RL: Verifiable Process Supervision via Knowledge Graphs

Train agentic KG reasoning models with GRPO (Group Relative Policy Optimisation), comparing three reward types to study when and why reward verifiability matters. KG serves as a **verification oracle**, not a knowledge source.

## Research Question

> When and why does deterministic step-level verification outperform heuristic rewards in RL post-training?

KG reasoning is the ideal testbed because KG provides free, deterministic, step-level ground truth via triple existence checks.

## Three Tasks

1. **KG-Grounded Multi-hop QA** — agent queries KG to answer questions
2. **Claim Verification** — agent verifies each sub-claim against KG
3. **Think-then-Verify** — model reasons freely, then verifies via KG, then revises

## Four Reward Types

| Reward | Description |
|--------|-------------|
| R_outcome | Answer EM + F1 only (baseline) |
| R_heuristic_step | R_outcome + entity overlap per step |
| R_verifiable_step | R_valid + R_on_path + R_progress + R_coherence |
| R_random | Random step rewards + real outcome (ablation) |

## Setup

```bash
# Isambard (GH200, ARM)
source ~/miniforge3/bin/activate && conda activate kg_verl
module load cudatoolkit && module load gcc-native/14.2

# Viking (H100, x86)
module load Miniconda3 && module load CUDA/12.8.0 && source activate kg_verl
```

## Project Structure

```
src/
  kg_server/          # KG retrieval server (5 endpoints)
  rewards/            # 4 reward functions
  data/               # SFT data generation for 3 tasks
  training/           # verl GRPO configs and launch scripts
  evaluation/         # Eval metrics including without-KG reasoning quality
scripts/              # SLURM job scripts
configs/              # verl training configs per experiment
data/
  conceptnet/         # ConceptNet data
  freebase/           # WebQSP + CWQ
```

## Key Docs

| File | Purpose |
|------|---------|
| `hpc_implementation_spec.md` | Full experiment spec (rewards, timeline, compute) |
| `CLAUDE.md` | Environment, coding standards, daily workflow |
| `PROJECT_SUMMARY.md` | Historical: TRL experiment results & lessons learned |
| `CHANGELOG.md` | Historical: detailed code change log |
| `SETUP_PROGRESS.md` | Isambard environment setup tracking |

## Data & Reproducibility

All code, configs, and the paper-ready summary tables live in this repository. The bulk
evidence artifacts — per-checkpoint eval results, 7-category classified trajectories,
mode-4 reward/entropy series, and the rule-based (no-LLM-teacher) SFT corpus — are
archived on Zenodo:

**📦 Zenodo — [10.5281/zenodo.20380101](https://doi.org/10.5281/zenodo.20380101)**

See **[`DATA.md`](DATA.md)** for the full data-availability statement: what is included,
what is regenerable from source (Freebase / CWQ), and the reproduction commands.