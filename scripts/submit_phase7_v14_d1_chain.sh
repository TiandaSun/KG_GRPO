#!/usr/bin/env bash
# V14-D1 Qwen2.5-14B E5b-stabilized — FULL chain submission.
#
# Invoke AFTER the pilot passes (loss drop between 5% and 50%, no NaN).
# The pilot job is submitted separately via:
#   sbatch scripts/run_phase7_v14_d1_sft_pilot.job
#
# This script submits full SFT → GRPO → eval with afterok dependencies.

set -euo pipefail
cd "$(dirname "$0")/.."

echo "Submitting V14-D1 full chain..."

FULL_SFT=$(sbatch scripts/run_phase7_v14_d1_sft_full.job | awk '{print $NF}')
echo "  SFT full job:  $FULL_SFT"

GRPO=$(sbatch --dependency=afterok:$FULL_SFT scripts/run_phase7_v14_d1_grpo.job | awk '{print $NF}')
echo "  GRPO job:      $GRPO  (afterok:$FULL_SFT)"

EVAL=$(sbatch --dependency=afterok:$GRPO scripts/run_phase7_v14_d1_eval.job | awk '{print $NF}')
echo "  Eval job:      $EVAL  (afterok:$GRPO)"

echo ""
echo "Chain: $FULL_SFT → $GRPO → $EVAL"
echo "Monitor: squeue -u \$USER"
