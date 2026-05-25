#!/usr/bin/env bash
# Chain-submit V14-D2 Qwen3-4B-Instruct-2507 E5b-stabilized pipeline.
#
# Pre-condition: tokenizer check + SFT pilot must already be reviewed and PASSING.
# Usage:
#   bash scripts/submit_phase7_v14_d2_chain.sh
#
# Submits three dependent jobs:
#   1. Full SFT (4 GPU, 8h)                                [v14_d2_sft_full]
#   2. GRPO with E5b reward + KL 5x (4 GPU, 18h)           [v14_d2_grpo]
#   3. Eval @ step 500 on full test set  (1 GPU, 12h)      [v14_d2_eval]

set -euo pipefail

PROJ_DIR="/lus/lfs1aip2/scratch/u6gg/ts1201.u6gg/Project/KG_GRPO"
cd "$PROJ_DIR"

echo "=== V14-D2 chain submission: $(date) ==="
echo "cwd: $(pwd)"

# --- Pre-flight ---
for f in \
    scripts/run_phase7_v14_d2_sft_full.job \
    scripts/run_phase7_v14_d2_grpo.job \
    scripts/run_phase7_v14_d2_eval.job \
    configs_verl/sft_cwq.yaml \
    configs_verl/grpo_cwq_qwen3_4b.yaml \
    data/freebase/sft_trajectories.jsonl
do
    if [ ! -e "$f" ]; then
        echo "ERROR: required file missing: $f" >&2
        exit 1
    fi
done

# Tokenizer check must have run (and passed) by hand.
TOK_JSON="results/phase7/v14_d2_tokenizer_check.json"
if [ ! -f "$TOK_JSON" ]; then
    echo "WARNING: $TOK_JSON missing — run the tokenizer-check job first and review results before continuing."
    read -r -p "Continue anyway? [y/N] " ans
    case "$ans" in
        y|Y|yes|YES) : ;;
        *) echo "Aborted."; exit 0 ;;
    esac
fi

# 1) Full SFT
SFT_ID=$(sbatch --parsable scripts/run_phase7_v14_d2_sft_full.job)
echo "submitted SFT       -> $SFT_ID"

# 2) GRPO (depends on SFT)
GRPO_ID=$(sbatch --parsable --dependency=afterok:"$SFT_ID" scripts/run_phase7_v14_d2_grpo.job)
echo "submitted GRPO      -> $GRPO_ID  (afterok:$SFT_ID)"

# 3) Eval (depends on GRPO)
EVAL_ID=$(sbatch --parsable --dependency=afterok:"$GRPO_ID" --export=ALL,STEP=500 scripts/run_phase7_v14_d2_eval.job)
echo "submitted Eval@500  -> $EVAL_ID  (afterok:$GRPO_ID)"

echo ""
echo "=== Chain submitted ==="
echo "SFT    ${SFT_ID}"
echo "GRPO   ${GRPO_ID}  (afterok:${SFT_ID})"
echo "Eval   ${EVAL_ID}  (afterok:${GRPO_ID})"
echo "Watch: squeue -u \$USER"
