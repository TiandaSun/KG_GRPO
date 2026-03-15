#!/bin/bash
# Setup kg_verl conda environment on Viking HPC
# Usage: bash scripts/setup_kg_verl_env.sh
#
# This creates a new conda env separate from kg_align (TRL pipeline preserved).

set -e

echo "=== Setting up kg_verl environment on Viking ==="

# 1. Load modules
module load Miniconda3
module load CUDA/12.8.0

# 2. Create conda environment
echo "Creating conda env: kg_verl (Python 3.10)"
conda create -n kg_verl python=3.10 -y

# 3. Activate (Viking requires source activate)
source activate kg_verl

# 4. Set cache directories to scratch
export HF_HOME=/mnt/scratch/users/ts1201/hf_cache
export TORCH_HOME=/mnt/scratch/users/ts1201/torch_cache

# 5. Install PyTorch (CUDA 12.8)
echo "Installing PyTorch (CUDA 12.8)..."
pip install torch --index-url https://download.pytorch.org/whl/cu128

# 6. Install verl with vLLM backend
echo "Installing verl with vLLM..."
pip install "verl[vllm]"

# 7. Install additional dependencies
echo "Installing additional dependencies..."
pip install fastapi uvicorn networkx peft wandb datasets hydra-core
pip install pyarrow  # for parquet support

# 8. Install flash-attn (optional, may fail on some builds)
echo "Installing flash-attn (optional)..."
pip install flash-attn --no-build-isolation || echo "WARNING: flash-attn install failed, will use SDPA fallback"

# 9. Verify installation
echo ""
echo "=== Verifying installation ==="
python -c "
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

import fastapi
print(f'FastAPI: {fastapi.__version__}')

import networkx
print(f'NetworkX: {networkx.__version__}')

import peft
print(f'PEFT: {peft.__version__}')

print()
print('All packages OK!')
"

echo ""
echo "=== Setup complete ==="
echo "To activate: module load Miniconda3 && module load CUDA/12.8.0 && source activate kg_verl"
