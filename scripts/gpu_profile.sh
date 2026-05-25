#!/usr/bin/env bash
#SBATCH --job-name=gpu_profile
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --time=01:30:00
#SBATCH --output=logs/%x-%j.log
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=workq

# GPU profiling: run 10 steps of E2 training while logging GPU utilization
# Captures: GPU util%, memory used/total, power, temperature every 2s

set -eo pipefail

source "$HOME/miniforge3/bin/activate"
conda activate kg_verl
module load cudatoolkit 2>/dev/null || true
module load gcc-native/14.2 2>/dev/null || true
module load brics/nccl 2>/dev/null || true

export CC=gcc CXX=g++
export LIBRARY_PATH=/opt/nvidia/hpc_sdk/Linux_aarch64/24.11/REDIST/math_libs/12.6/targets/sbsa-linux/lib:${LIBRARY_PATH:-}
export TRITON_CACHE_DIR=/scratch/u6gg/ts1201.u6gg/.triton
export HF_HOME=/scratch/u6gg/ts1201.u6gg/hf_cache
export PYTHONUNBUFFERED=1
export VLLM_USE_V1=1
export RAY_DEDUP_LOGS=1
export RAY_TMPDIR="/tmp/ray-${SLURM_JOB_ID}"
mkdir -p "$RAY_TMPDIR"

cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

export REWARD_TYPE=heuristic
EXPERIMENT_NAME="grpo-cwq-7b-gpu-profile"

echo "=== GPU Performance Profile ==="
echo "Date: $(date)"
echo "Node: $(hostname)"
echo ""

# --- GPU info ---
echo "=== GPU Hardware ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo ""

# --- Start GPU monitoring in background ---
mkdir -p results
GPU_LOG="results/gpu_profile_${SLURM_JOB_ID}.csv"
echo "timestamp,gpu_id,util_pct,mem_used_mb,mem_total_mb,power_w,temp_c" > "$GPU_LOG"

(
while true; do
    nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu \
        --format=csv,noheader,nounits 2>/dev/null | while IFS=, read -r idx util mem_used mem_total power temp; do
        echo "$(date +%s),$idx,$util,$mem_used,$mem_total,$power,$temp" >> "$GPU_LOG"
    done
    sleep 2
done
) &
GPU_MON_PID=$!
echo "GPU monitoring PID: $GPU_MON_PID (logging to $GPU_LOG)"
echo ""

# --- Start KG server ---
echo "Starting Freebase KG server..."
python -m src_verl.kg_server.server \
    --kg freebase \
    --freebase_dir data/freebase/kg \
    --port 18901 \
    --log-level warning > /dev/null 2>&1 &
KG_PID=$!

sleep 30
for i in $(seq 1 12); do
    if curl -s -o /dev/null -w "%{http_code}" http://localhost:18901/health 2>/dev/null | grep -q "200"; then
        echo "KG server healthy"
        break
    fi
    sleep 10
done
echo ""

ray stop 2>/dev/null || true

echo "=== Launching GRPO (profiling run) ==="
echo "Start: $(date)"

set +e
timeout 3600 python -m verl.trainer.main_ppo \
    --config-path="$PWD/configs_verl" \
    --config-name=grpo_cwq_7b \
    trainer.experiment_name="$EXPERIMENT_NAME" \
    trainer.total_epochs=1 \
    actor_rollout_ref.rollout.agent.default_agent_loop=tool_agent \
    data.dataloader_num_workers=0
set -e

echo ""
echo "=== GPU Utilization Summary ==="
echo "End: $(date)"

# Analyze GPU log
python3 -c "
import csv
from collections import defaultdict

data = defaultdict(lambda: {'util': [], 'mem_used': [], 'mem_total': 0, 'power': []})
with open('$GPU_LOG') as f:
    reader = csv.DictReader(f)
    for row in reader:
        gid = row['gpu_id'].strip()
        data[gid]['util'].append(float(row['util_pct'].strip()))
        data[gid]['mem_used'].append(float(row['mem_used_mb'].strip()))
        data[gid]['mem_total'] = float(row['mem_total_mb'].strip())
        data[gid]['power'].append(float(row['power_w'].strip()))

print(f\"{'GPU':>4} {'Util%':>8} {'MemUsed':>10} {'MemTotal':>10} {'MemUtil%':>9} {'Power':>8}\")
print('-' * 55)
for gid in sorted(data.keys()):
    d = data[gid]
    avg_util = sum(d['util']) / len(d['util'])
    avg_mem = sum(d['mem_used']) / len(d['mem_used'])
    max_mem = max(d['mem_used'])
    mem_total = d['mem_total']
    avg_power = sum(d['power']) / len(d['power'])
    print(f\"{gid:>4} {avg_util:>7.1f}% {avg_mem:>8.0f}MB {mem_total:>8.0f}MB {max_mem/mem_total*100:>8.1f}% {avg_power:>7.1f}W\")

# Time-bucketed analysis (first 5 min vs rest)
total_samples = len(data['0']['util']) if '0' in data else 0
print(f'\nTotal samples: {total_samples} ({total_samples*2}s)')
if total_samples > 150:
    # First 5 min (150 samples at 2s) = init/loading
    # Rest = actual training
    for phase, start, end in [('Init (0-5m)', 0, 150), ('Training (5m+)', 150, total_samples)]:
        utils = [data['0']['util'][i] for i in range(start, min(end, total_samples))]
        mems = [data['0']['mem_used'][i] for i in range(start, min(end, total_samples))]
        if utils:
            print(f\"  {phase}: GPU0 avg_util={sum(utils)/len(utils):.1f}%, avg_mem={sum(mems)/len(mems):.0f}MB, max_mem={max(mems):.0f}MB\")
"

kill $GPU_MON_PID 2>/dev/null
kill $KG_PID 2>/dev/null

echo ""
echo "=== Config Summary ==="
echo "batch_size=64, n=8 rollouts, micro_batch=2/gpu, mini_batch=16"
echo "prompt_len=1024, response_len=2048"
echo "vllm gpu_memory_utilization=0.4, enforce_eager=true"
echo "gradient_checkpointing=true, attn=eager"
echo ""
echo "=== Done ==="
