#!/bin/bash
# Setup kg_verl conda environment on Isambard (ARM GH200)
# Usage: bash scripts/setup_kg_verl_env_isambard.sh
#
# Key differences from Viking:
# - ARM aarch64 architecture (GH200)
# - Module names may differ — check with `module avail`
# - torch/vLLM may need different install paths for ARM

set -e

echo "=== Setting up kg_verl environment on Isambard (GH200) ==="

# 1. Load modules (adapt names for Isambard — check `module avail`)
module load Miniconda3 2>/dev/null || module load miniconda3 2>/dev/null || echo "WARNING: Could not load Miniconda3 module. Ensure conda is on PATH."
module load CUDA 2>/dev/null || module load cuda 2>/dev/null || echo "WARNING: Could not load CUDA module."

# 2. Create conda environment
echo "Creating conda env: kg_verl (Python 3.10)"
conda create -n kg_verl python=3.10 -y

# 3. Activate
source activate kg_verl

# 4. Set cache directories
export HF_HOME="${SCRATCH:-$HOME}/hf_cache"
export TORCH_HOME="${SCRATCH:-$HOME}/torch_cache"

# 5. Install PyTorch
echo "Installing PyTorch..."
pip install torch

# 6. Install verl with vLLM backend
echo "Installing verl with vLLM..."
pip install "verl[vllm]"

# 7. Install additional dependencies
echo "Installing additional dependencies..."
pip install fastapi uvicorn networkx peft wandb datasets hydra-core pyarrow

# 8. Install flash-attn (may fail on ARM — that's OK)
echo "Installing flash-attn (optional, may fail on ARM)..."
pip install flash-attn --no-build-isolation || echo "WARNING: flash-attn not available on ARM. Using SDPA."

# 9. Verify installation
echo ""
echo "=== Verifying installation ==="
python -c "
import platform
print(f'Architecture: {platform.machine()}')

import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU: {torch.cuda.get_device_name(0)}')

import verl
print(f'verl: imported OK')

import vllm
print(f'vLLM: {vllm.__version__}')

import ray
print(f'Ray: {ray.__version__}')

print()
print('All packages OK!')
"

echo ""
echo "=== Setup complete ==="
echo "To activate: source activate kg_verl"
