#!/bin/bash
# Setup kg_verl conda environment for KG-Align-RL.
#
# Auto-detects platform (Viking x86, Isambard ARM) and installs accordingly.
#
# Usage:
#   bash scripts/setup_env.sh              # full install
#   bash scripts/setup_env.sh --skip-vllm  # skip vLLM (login node or ARM issues)

set -e

SKIP_VLLM=false
if [[ "${1:-}" == "--skip-vllm" ]]; then
    SKIP_VLLM=true
fi

ARCH=$(uname -m)
echo "=== KG-Align-RL: Environment Setup ==="
echo "Architecture: $ARCH"
echo "Hostname:     $(hostname)"
echo ""

# --- 1. Detect platform and load modules ---
if [[ "$ARCH" == "aarch64" ]]; then
    echo "=== Isambard (ARM GH200) detected ==="
    PLATFORM="isambard"

    # Isambard: install Miniforge if not present
    if ! command -v conda &>/dev/null; then
        if [ ! -d "$HOME/miniforge3" ]; then
            echo "Installing Miniforge for aarch64..."
            cd "$HOME"
            curl --location --remote-name \
                "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
            bash "Miniforge3-$(uname)-$(uname -m).sh" -b -p "$HOME/miniforge3"
            rm -f "Miniforge3-$(uname)-$(uname -m).sh"
        fi
        source "$HOME/miniforge3/bin/activate"
    fi

    # Load CUDA toolkit
    module load cudatoolkit 2>/dev/null || echo "WARNING: cudatoolkit module not found"

    # Set scratch
    SCRATCH_DIR="${SCRATCH:-${SCRATCHDIR:-$HOME}}"

else
    echo "=== Viking (x86_64) detected ==="
    PLATFORM="viking"

    # Viking: use module-provided conda
    module load Miniconda3 2>/dev/null || echo "WARNING: Miniconda3 module not found"
    module load CUDA/12.8.0 2>/dev/null || module load CUDA/12.1.1 2>/dev/null || \
        echo "WARNING: CUDA module not found"

    SCRATCH_DIR="/mnt/scratch/users/$USER"
fi

echo "Platform:  $PLATFORM"
echo "Scratch:   $SCRATCH_DIR"
echo ""

# --- 2. Create conda environment ---
ENV_NAME="kg_verl"

if conda info --envs 2>/dev/null | grep -q "$ENV_NAME"; then
    echo "Environment '$ENV_NAME' already exists. Activating..."
else
    echo "Creating conda environment: $ENV_NAME (Python 3.10)"
    conda create -n "$ENV_NAME" python=3.10 -y
fi

# Activate (works on both Viking and Isambard)
source activate "$ENV_NAME" 2>/dev/null || conda activate "$ENV_NAME"

# --- 3. Set cache directories ---
export HF_HOME="$SCRATCH_DIR/hf_cache"
export TORCH_HOME="$SCRATCH_DIR/torch_cache"
mkdir -p "$HF_HOME" "$TORCH_HOME"
echo "HF_HOME:   $HF_HOME"
echo "TORCH_HOME: $TORCH_HOME"
echo ""

# --- 4. Install PyTorch ---
echo "=== Installing PyTorch ==="
if [[ "$PLATFORM" == "isambard" ]]; then
    # ARM: use conda-forge or default pip (has aarch64 wheels)
    pip install torch
else
    # x86: use CUDA 12.1 wheels
    pip install torch --index-url https://download.pytorch.org/whl/cu121
fi

# --- 5. Install verl with vLLM ---
echo ""
echo "=== Installing verl ==="
if [ "$SKIP_VLLM" = true ]; then
    echo "Skipping vLLM (--skip-vllm flag set)"
    pip install verl
else
    pip install "verl[vllm]" || {
        echo "WARNING: verl[vllm] failed, installing verl without vLLM"
        pip install verl
    }
fi

# --- 6. Install remaining dependencies ---
echo ""
echo "=== Installing dependencies ==="
pip install \
    transformers>=4.45.0 \
    peft>=0.14.0 \
    datasets>=2.18.0 \
    accelerate>=0.34.0 \
    bitsandbytes>=0.44.0 \
    networkx>=3.0 \
    wandb \
    fastapi uvicorn \
    hydra-core pyarrow \
    rouge-score nltk

# --- 7. Install flash-attn (optional, x86 only) ---
echo ""
if [[ "$PLATFORM" == "isambard" ]]; then
    echo "=== Skipping flash-attn (not available on ARM, using SDPA) ==="
else
    echo "=== Installing flash-attn (optional) ==="
    pip install flash-attn --no-build-isolation 2>/dev/null || \
        echo "WARNING: flash-attn failed to install. Using SDPA fallback."
fi

# --- 8. Verify installation ---
echo ""
echo "=== Verifying installation ==="
python -c "
import platform
print(f'Architecture: {platform.machine()}')

import torch
print(f'PyTorch:      {torch.__version__}')
print(f'CUDA:         {torch.cuda.is_available()} ({torch.version.cuda if torch.cuda.is_available() else \"N/A\"})')
if torch.cuda.is_available():
    print(f'GPU:          {torch.cuda.get_device_name(0)}')

import verl
print(f'verl:         imported OK')

try:
    import vllm
    print(f'vLLM:         {vllm.__version__}')
except ImportError:
    print(f'vLLM:         not installed (HF generate fallback)')

import transformers, peft, datasets
print(f'transformers: {transformers.__version__}')
print(f'peft:         {peft.__version__}')
print(f'datasets:     {datasets.__version__}')

try:
    from flash_attn import flash_attn_func
    print(f'flash-attn:   available')
except ImportError:
    print(f'flash-attn:   not available (using SDPA)')

print()
print('All packages OK!')
"

echo ""
echo "=== Setup Complete ==="
echo ""
echo "To activate in future sessions:"
if [[ "$PLATFORM" == "isambard" ]]; then
    echo "  source ~/miniforge3/bin/activate && conda activate $ENV_NAME"
else
    echo "  module load Miniconda3 && source activate $ENV_NAME"
fi
echo ""
echo "Next steps:"
echo "  1. Download data:  bash scripts/download_data.sh"
echo "  2. Run tests:      python -m pytest tests/ -v"
