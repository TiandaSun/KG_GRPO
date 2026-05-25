#!/usr/bin/env bash
# V14-A1 Search-R1 baseline: submit train + eval with dependency.
#
# Usage:
#   bash scripts/submit_phase7_v14_a1_chain.sh
#
# Prints train job ID and chained eval job ID.

set -eo pipefail

cd "$(dirname "$0")/.."
REPO="$PWD"

if [ ! -d "$REPO/external/Search-R1" ]; then
    echo "Cloning Search-R1 (login node)..."
    mkdir -p "$REPO/external"
    ( cd "$REPO/external" && git clone https://github.com/PeterGriffinJin/Search-R1.git )
fi

if [ ! -f "$REPO/data/freebase/searchr1_cwq/train.parquet" ] || \
   [ ! -f "$REPO/data/freebase/searchr1_cwq/test.parquet" ]; then
    echo "Converting CWQ parquet to Search-R1 format..."
    source "$HOME/miniforge3/bin/activate"
    conda activate kg_verl
    PYTHONPATH="$REPO" python "$REPO/scripts/prepare_cwq_for_searchr1.py"
fi

echo "=== Submitting train job ==="
TRAIN_ID=$(sbatch --parsable "$REPO/scripts/run_phase7_v14_a1_searchr1_train.job")
echo "Train job ID: $TRAIN_ID"

echo "=== Submitting eval job (depends on train) ==="
EVAL_ID=$(sbatch --parsable \
    --dependency=afterok:$TRAIN_ID \
    "$REPO/scripts/run_phase7_v14_a1_searchr1_eval.job")
echo "Eval  job ID: $EVAL_ID"

echo
echo "Chain submitted:"
echo "  Train: $TRAIN_ID"
echo "  Eval:  $EVAL_ID (depends on $TRAIN_ID)"
echo
echo "Monitor with: squeue -u \$USER -j $TRAIN_ID,$EVAL_ID"
echo "Logs: logs/p7_v14a1_searchr1_train-${TRAIN_ID}.log"
echo "      logs/p7_v14a1_searchr1_eval-${EVAL_ID}.log"
