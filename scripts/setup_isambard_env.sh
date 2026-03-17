#!/bin/bash
# =============================================================================
# Isambard GH200 (aarch64) — kg_verl conda environment setup
# =============================================================================
# Usage:
#   1. Run steps 1-8 on the LOGIN NODE (no GPU needed, no time limit):
#        bash scripts/setup_isambard_env.sh
#
#   2. Then verify GPU access in a short interactive job:
#        srun --gpus=1 --time=00:10:00 --qos=workq_qos --partition=workq --pty bash
#        source ~/miniforge3/bin/activate && conda activate kg_verl
#        python scripts/verify_gpu.py
# =============================================================================

set -e

SCRATCH="${SCRATCH:-/scratch/u6gg/ts1201.u6gg}"

# ----- Step 1: Install Miniforge (skip if already installed) -----
if [ ! -f "$HOME/miniforge3/bin/conda" ]; then
    echo "=== Step 1: Installing Miniforge ==="
    curl --location --remote-name \
        "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
    bash "Miniforge3-$(uname)-$(uname -m).sh" -b -p "$HOME/miniforge3"
    rm -f "Miniforge3-$(uname)-$(uname -m).sh"
    echo "Miniforge installed to ~/miniforge3"
else
    echo "=== Step 1: Miniforge already installed, skipping ==="
fi

# ----- Step 2: Create kg_verl environment -----
echo "=== Step 2: Creating conda environment kg_verl (Python 3.10) ==="
source "$HOME/miniforge3/bin/activate"

if conda env list | grep -q "kg_verl"; then
    echo "Environment kg_verl already exists, skipping creation"
else
    conda create -n kg_verl python=3.10 -y
fi
conda activate kg_verl

# ----- Step 3: Set cache directories to scratch -----
echo "=== Step 3: Setting cache directories ==="
export HF_HOME="$SCRATCH/hf_cache"
export TORCH_HOME="$SCRATCH/torch_cache"
mkdir -p "$HF_HOME" "$TORCH_HOME"
echo "HF_HOME=$HF_HOME"
echo "TORCH_HOME=$TORCH_HOME"

# ----- Step 4: Load CUDA modules -----
echo "=== Step 4: Loading CUDA modules ==="
module load cudatoolkit 2>/dev/null && echo "cudatoolkit loaded" || echo "WARNING: cudatoolkit not loaded"
module load brics/nccl 2>/dev/null && echo "brics/nccl loaded" || echo "WARNING: brics/nccl not loaded"

# ----- Step 5: Install PyTorch -----
echo "=== Step 5: Installing PyTorch (aarch64 wheel) ==="
pip install torch

# ----- Step 6: Install verl + vLLM -----
echo "=== Step 6: Installing verl with vLLM backend ==="
pip install "verl[vllm]"

# ----- Step 7: Install remaining dependencies -----
echo "=== Step 7: Installing remaining dependencies ==="
pip install fastapi uvicorn networkx peft wandb datasets hydra-core pyarrow
pip install "transformers>=4.45.0" "accelerate>=0.34.0" "bitsandbytes>=0.44.0"
pip install rouge-score nltk

# ----- Step 8: flash-attn (expected to fail on ARM — that's OK) -----
echo "=== Step 8: Attempting flash-attn (will likely fail on ARM) ==="
pip install flash-attn --no-build-isolation 2>/dev/null \
    || echo "OK: flash-attn not available on ARM. Using SDPA fallback — this is expected."

# ----- Step 9: Basic verification (CPU-only, no GPU check) -----
echo ""
echo "=== Step 9: CPU-only verification ==="
python -c "
import platform
print(f'Architecture: {platform.machine()}')

import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA compiled: {torch.version.cuda}')
# NOTE: torch.cuda.is_available() will be False on login node — that is expected

import transformers
print(f'Transformers: {transformers.__version__}')

import verl
print('verl: imported OK')

try:
    import vllm
    print(f'vLLM: {vllm.__version__}')
except ImportError:
    print('vLLM: not installed (optional)')

import peft
print(f'PEFT: {peft.__version__}')

import wandb
print(f'wandb: {wandb.__version__}')

print()
print('CPU verification passed! Run verify_gpu.py on a GPU node to confirm CUDA.')
"

echo ""
echo "=========================================="
echo "  Setup complete!"
echo "=========================================="
echo ""
echo "To activate in future sessions:"
echo "  source ~/miniforge3/bin/activate && conda activate kg_verl"
echo ""
echo "To verify GPU access, get an interactive GPU node:"
echo "  srun --gpus=1 --time=00:10:00 --qos=workq_qos --partition=workq --pty bash"
echo "  source ~/miniforge3/bin/activate && conda activate kg_verl"
echo "  python scripts/verify_gpu.py"
echo ""
