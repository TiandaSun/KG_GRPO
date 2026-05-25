#!/usr/bin/env bash
# S0.2 — Multi-seed G2 launch driver.
# Chains SFT -> GRPO -> eval per seed via SLURM --dependency=afterok.
# Submits 9 jobs total (3 seeds x 3 stages). 3 seeds run in parallel:
# concurrent GPU usage = 3 x 4 = 12 GH200s (within Isambard's 32-GPU project cap).
#
# Run from project root:
#   bash scripts/launch_s02_chain.sh
#
# To override the seed list:
#   SEEDS="1 2" bash scripts/launch_s02_chain.sh

set -euo pipefail

cd "$(dirname "$0")/.."

SEEDS="${SEEDS:-1 2 3}"

echo "=== S0.2 launch driver ==="
echo "Date: $(date)"
echo "Seeds: $SEEDS"
echo ""

for seed in $SEEDS; do
    echo "--- Seed $seed ---"
    SFT_ID=$(sbatch --parsable --export=ALL,SEED=$seed scripts/run_phase7_s02_g2_sft_seed.job)
    GRPO_ID=$(sbatch --parsable --dependency=afterok:$SFT_ID --export=ALL,SEED=$seed scripts/run_phase7_s02_g2_grpo_seed.job)
    EVAL_ID=$(sbatch --parsable --dependency=afterok:$GRPO_ID --export=ALL,SEED=$seed scripts/run_phase7_s02_g2_eval_seed.job)
    echo "Seed $seed: SFT=$SFT_ID  GRPO=$GRPO_ID  EVAL=$EVAL_ID"
done

echo ""
echo "Submitted. Monitor with: squeue -u \$USER"
