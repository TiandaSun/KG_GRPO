#!/usr/bin/env bash
# Task 39: Discover all existing checkpoints for the 4 variants and submit
# one eval job per (variant, step) combination. Already-evaluated checkpoints
# are skipped (idempotent — safe to re-run as new checkpoints land).
#
# Usage:
#   bash scripts/submit_task39_evals.sh          # submit missing evals
#   bash scripts/submit_task39_evals.sh --dry    # show what would be submitted
#   bash scripts/submit_task39_evals.sh --force  # resubmit even if results exist

set -euo pipefail

DRY_RUN=0
FORCE=0
for arg in "$@"; do
    case "$arg" in
        --dry|--dry-run) DRY_RUN=1 ;;
        --force) FORCE=1 ;;
    esac
done

CKPT_ROOT="/projects/u6gg/KG_GRPO/checkpoints/kg-align-verl"
RESULTS_DIR="results/task39_eval"
mkdir -p "$RESULTS_DIR"

# Variant definitions: (variant_tag, ckpt_dir_pattern, step_mapping)
#   A, B: single experiment dir, step = global step
#   C1, C2: cycle-indexed dirs, global_step = cycle * 50, local step always 50

submit_one() {
    local variant="$1"
    local ckpt_dir="$2"
    local local_step="$3"
    local global_step="$4"
    local model_tag="${variant}_step$(printf '%03d' "$global_step")"
    local result_file="$RESULTS_DIR/${model_tag}.json"
    local marker_file="$RESULTS_DIR/.submitted_${model_tag}"

    if [ ! -d "$ckpt_dir/global_step_${local_step}/actor" ]; then
        return  # checkpoint not yet saved
    fi

    if [ "$FORCE" = "0" ] && [ -f "$result_file" ]; then
        echo "[skip] $model_tag (already evaluated)"
        return
    fi

    if [ "$FORCE" = "0" ] && [ -f "$marker_file" ]; then
        echo "[skip] $model_tag (already submitted)"
        return
    fi

    if [ "$DRY_RUN" = "1" ]; then
        echo "[dry]  $model_tag  ckpt=$ckpt_dir  local_step=$local_step"
        return
    fi

    echo "[submit] $model_tag"
    local jobid=$(sbatch \
        --export=ALL,CHECKPOINT_DIR="$ckpt_dir",STEP="$local_step",MODEL_TAG="$model_tag" \
        scripts/run_task39_eval_single.job | awk '{print $NF}')
    echo "$jobid $(date -Iseconds)" > "$marker_file"
}

# Variant A: format reward — single experiment, steps 50..500
V39A_DIR="$CKPT_ROOT/grpo-cwq-7b-39a-format-20260413"
if [ -d "$V39A_DIR" ]; then
    for step in 50 100 150 200 250 300 400 500; do
        submit_one "39a_format" "$V39A_DIR" "$step" "$step"
    done
fi

# Variant B: KL — single experiment, steps 50..500
V39B_DIR="$CKPT_ROOT/grpo-cwq-7b-39b-kl-20260413"
if [ -d "$V39B_DIR" ]; then
    for step in 50 100 150 200 250 300 400 500; do
        submit_one "39b_kl" "$V39B_DIR" "$step" "$step"
    done
fi

# Variant C1: SFT replay mixed — cycle dirs, each has only step 50
for cycle in 1 2 3 4 5; do
    C1_DIR="$CKPT_ROOT/grpo-cwq-7b-39c1-sft-replay-20260413-c${cycle}"
    if [ -d "$C1_DIR" ]; then
        global_step=$((cycle * 50))
        submit_one "39c1_mixed" "$C1_DIR" "50" "$global_step"
    fi
done

# Variant C2: SFT replay filtered — cycle dirs, each has only step 50
for cycle in 1 2 3 4 5; do
    C2_DIR="$CKPT_ROOT/grpo-cwq-7b-39c2-filtered-sft-20260413-c${cycle}"
    if [ -d "$C2_DIR" ]; then
        global_step=$((cycle * 50))
        submit_one "39c2_filtered" "$C2_DIR" "50" "$global_step"
    fi
done

# Variant D: Cat-B + format — single experiment, steps 50..500
V39D_DIR="$CKPT_ROOT/grpo-cwq-7b-39d-catb-format-20260413"
if [ -d "$V39D_DIR" ]; then
    for step in 50 100 150 200 250 300 400 500; do
        submit_one "39d_catb_format" "$V39D_DIR" "$step" "$step"
    done
fi

# Variant E: enhanced SFT + Cat-B + format — single experiment, steps 50..500
V39E_DIR="$CKPT_ROOT/grpo-cwq-7b-39e-enh-catb-format-20260413"
if [ -d "$V39E_DIR" ]; then
    for step in 50 100 150 200 250 300 400 500; do
        submit_one "39e_enh_catb_format" "$V39E_DIR" "$step" "$step"
    done
fi

# Variant F: enhanced SFT + full + format — single experiment, steps 50..500
V39F_DIR="$CKPT_ROOT/grpo-cwq-7b-39f-enh-full-format-20260413"
if [ -d "$V39F_DIR" ]; then
    for step in 50 100 150 200 250 300 400 500; do
        submit_one "39f_enh_full_format" "$V39F_DIR" "$step" "$step"
    done
fi

echo ""
echo "Current queue:"
squeue -u "$USER" --format="%.10i %.15j %.8T %.10M %R" 2>&1
