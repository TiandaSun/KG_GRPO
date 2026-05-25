#!/usr/bin/env bash
# Monitor a GRPO training job for hangs.
# Detects if training progress stalls (no new steps for HANG_TIMEOUT seconds).
# Usage: bash scripts/monitor_training.sh <JOB_ID> [HANG_TIMEOUT_MINUTES]
#
# The script checks stderr for "Training Progress" lines every CHECK_INTERVAL seconds.
# If no new progress line appears for HANG_TIMEOUT minutes, it prints a warning
# and optionally cancels the job.

set -eo pipefail

JOB_ID="${1:?Usage: $0 <JOB_ID> [HANG_TIMEOUT_MINUTES]}"
HANG_TIMEOUT_MINUTES="${2:-30}"
CHECK_INTERVAL=60  # seconds
AUTO_CANCEL="${3:-false}"  # set to "true" to auto-cancel hung jobs

LOG_ERR="logs/grpo_cwq_7b-${JOB_ID}.err"
LOG_OUT="logs/grpo_cwq_7b-${JOB_ID}.log"

echo "=== Training Monitor ==="
echo "Job: $JOB_ID"
echo "Hang timeout: ${HANG_TIMEOUT_MINUTES}m"
echo "Check interval: ${CHECK_INTERVAL}s"
echo "Auto-cancel: $AUTO_CANCEL"
echo "Log: $LOG_ERR"
echo ""

last_step_count=0
stale_checks=0
max_stale_checks=$((HANG_TIMEOUT_MINUTES * 60 / CHECK_INTERVAL))

while true; do
    # Check if job is still running
    job_state=$(squeue -j "$JOB_ID" -h -o "%T" 2>/dev/null || echo "GONE")
    if [[ "$job_state" == "GONE" || -z "$job_state" ]]; then
        echo "[$(date +%H:%M:%S)] Job $JOB_ID is no longer running (state: $job_state)"
        # Show final status
        if [[ -f "$LOG_ERR" ]]; then
            echo "=== Final progress ==="
            grep "Training Progress" "$LOG_ERR" 2>/dev/null | tail -3
            echo ""
            echo "=== Final errors ==="
            grep -i "error\|cancel\|kill\|oom\|timeout" "$LOG_ERR" 2>/dev/null | tail -5
        fi
        break
    fi

    # Count Training Progress lines
    if [[ -f "$LOG_ERR" ]]; then
        current_step_count=$(grep -c "Training Progress" "$LOG_ERR" 2>/dev/null || echo "0")
        last_line=$(grep "Training Progress" "$LOG_ERR" 2>/dev/null | tail -1 | sed 's/\x1b\[[0-9;]*m//g' | grep -oP '\d+/\d+.*' || echo "N/A")
    else
        current_step_count=0
        last_line="log not yet available"
    fi

    # Check for checkpoint saves (in stdout log)
    if [[ -f "$LOG_OUT" ]]; then
        ckpt_count=$(grep -c "local_global_step_folder" "$LOG_OUT" 2>/dev/null || echo "0")
        last_ckpt=$(grep "local_global_step_folder" "$LOG_OUT" 2>/dev/null | tail -1 | grep -oP 'global_step_\d+' || echo "none")
    else
        ckpt_count=0
        last_ckpt="none"
    fi

    # Check for update_weights (in stderr)
    if [[ -f "$LOG_ERR" ]]; then
        uw_count=$(grep -c "update_weights done" "$LOG_ERR" 2>/dev/null || echo "0")
    else
        uw_count=0
    fi

    # Log size (detect schema flooding)
    if [[ -f "$LOG_OUT" ]]; then
        log_lines=$(wc -l < "$LOG_OUT" 2>/dev/null || echo "0")
    else
        log_lines=0
    fi

    if [[ "$current_step_count" -gt "$last_step_count" ]]; then
        stale_checks=0
        echo "[$(date +%H:%M:%S)] OK step=$current_step_count uw=$uw_count ckpt=$last_ckpt log=${log_lines}L | $last_line"
    else
        stale_checks=$((stale_checks + 1))
        elapsed_stale=$((stale_checks * CHECK_INTERVAL / 60))
        echo "[$(date +%H:%M:%S)] STALE (${elapsed_stale}m/${HANG_TIMEOUT_MINUTES}m) step=$current_step_count uw=$uw_count ckpt=$last_ckpt log=${log_lines}L"

        if [[ "$stale_checks" -ge "$max_stale_checks" ]]; then
            echo ""
            echo "!!! HANG DETECTED: No progress for ${HANG_TIMEOUT_MINUTES} minutes !!!"
            echo "Last progress: $last_line"
            echo "Checkpoints saved: $ckpt_count (last: $last_ckpt)"
            echo "Log size: ${log_lines} lines"
            echo ""

            if [[ "$AUTO_CANCEL" == "true" ]]; then
                echo "Auto-cancelling job $JOB_ID..."
                scancel "$JOB_ID"
                echo "Job cancelled. Restart with checkpoint resume."
            else
                echo "Run: scancel $JOB_ID  # to cancel"
                echo "Then restart with --resume from last checkpoint"
            fi
            break
        fi
    fi

    last_step_count="$current_step_count"
    sleep "$CHECK_INTERVAL"
done
