# Isambard Environment Setup Progress

## Date: 2026-03-17

## Completed Steps

### Step 1: Install Miniforge ✅
- Installed to `~/miniforge3` (aarch64)
- Activate with: `source ~/miniforge3/bin/activate`

### Step 2: Create conda env ✅
- Environment `kg_verl` created with Python 3.10
- Activate with: `conda activate kg_verl`

### Step 3: Cache directories ✅
- `HF_HOME=$SCRATCH/hf_cache` (`/scratch/u6gg/ts1201.u6gg/hf_cache`)
- `TORCH_HOME=$SCRATCH/torch_cache` (`/scratch/u6gg/ts1201.u6gg/torch_cache`)
- Directories created

### Step 4: CUDA modules ✅
- `module load cudatoolkit` (12.6 default available)
- `module load brics/nccl` (2.26.6-1 default available)

### Step 5: Install PyTorch ✅ FIXED
- Originally installed PyTorch 2.9.0+cpu (CPU-only, no CUDA)
- **Fix applied**: `pip install torch --index-url https://download.pytorch.org/whl/cu126 --force-reinstall`
- Now running **PyTorch 2.10.0+cu126** with full CUDA support on aarch64
- Verified on GPU node: GH200 120GB detected, matrix ops, bf16, and SDPA attention all pass

### Step 6: Install verl + vLLM ✅
- verl 0.7.1 installed
- vLLM upgraded to **0.17.1** (from 0.12.0, to match torch 2.10)
- Both verified working with CUDA-enabled PyTorch on GPU node

### Step 7: Additional dependencies ✅
All installed:
- fastapi, uvicorn, networkx, peft 0.18.1, wandb 0.25.1
- datasets 4.8.2, hydra-core 1.3.2, pyarrow 23.0.1
- transformers 4.57.6, accelerate 1.13.0, bitsandbytes 0.49.2
- rouge-score, nltk, joblib

### Step 8: flash-attn ✅ (skipped as expected)
- Fails to build on ARM aarch64 — this is expected
- Use `attn_implementation: "sdpa"` in all configs on Isambard

### Step 9: CPU verification ✅
- All packages import correctly

### Step 10: GPU verification ✅ (2026-03-17)
- Tested on GPU node with NVIDIA GH200 120GB
- PyTorch CUDA: ✅ (`torch.cuda.is_available() = True`)
- Matrix multiply (fp32): ✅
- BFloat16 operations: ✅
- SDPA attention: ✅
- vLLM GPU import: ✅
- All core packages (verl, transformers, peft, accelerate, wandb, datasets): ✅

### Step 11: Download Qwen2.5-1.5B-Instruct ✅ (2026-03-17)
- Downloaded to `$HF_HOME` (`/scratch/u6gg/ts1201.u6gg/hf_cache`)
- Model loads on GPU with bf16+SDPA: 1.54B params
- Basic inference test passed (coherent multi-paragraph response)

### Step 12: Install DeepSpeed ✅ (2026-03-17)
- DeepSpeed 0.18.8 installed
- CPUAdam JIT compiles successfully (requires `module load gcc-native/14.2` + `export CC=gcc CXX=g++`)
- Also requires `LIBRARY_PATH` to include curand: `/opt/nvidia/hpc_sdk/Linux_aarch64/24.11/REDIST/math_libs/12.6/targets/sbsa-linux/lib`

### Step 13: Install SGLang ✅ (partial — 2026-03-17)
- SGLang 0.5.9 installed (required `CC=gcc CXX=g++` for torch_memory_saver build)
- **SGLang Engine does NOT work on ARM**: `sgl_kernel` pre-built .so has ABI mismatch (compiled for x86 PyTorch ABI)
- **Workaround**: Use vLLM as rollout backend for verl instead of SGLang

### Step 14: vLLM on Isambard ✅ (2026-03-17)
- vLLM 0.17.1 works with two required flags:
  1. `enforce_eager=True` (skip Triton graph compilation)
  2. `export CC=gcc` (Triton needs gcc, not nvc)
- FlashAttention v3 used automatically by vLLM (separate from flash-attn pip package)
- Tested with Qwen2.5-1.5B-Instruct: generation works correctly
- Note: SGLang installed alongside caused torch downgrade to 2.9.1+cpu — fixed by reinstalling `torch==2.10.0+cu126`

### Step 15: verl backend verification ✅ (2026-03-17)
- `verl.workers.rollout.vllm_rollout` — available ✅
- `verl.workers.rollout.sglang_rollout` — importable but SGLang engine broken on ARM
- **Decision**: Use vLLM backend for verl on Isambard

## Next Steps

1. **Begin pipeline work** — proceed with the verl pipeline stages per MINIMAL_VALIDATION_PLAN.md
2. **Download ConceptNet** data for KG path extraction (Stage 1)
3. **Implement KG server** endpoints

## Installed Package Versions
| Package | Version | Status |
|---------|---------|--------|
| Python | 3.10.20 | ✅ |
| PyTorch | 2.10.0+cu126 | ✅ CUDA 12.6 |
| transformers | 4.57.1 | ✅ (minor downgrade from 4.57.6 by sglang) |
| verl | 0.7.1 | ✅ |
| vLLM | 0.17.1 | ✅ (enforce_eager + CC=gcc required on ARM) |
| SGLang | 0.5.9 | ⚠️ Installed but engine broken on ARM (sgl_kernel ABI) |
| DeepSpeed | 0.18.8 | ✅ (CPUAdam JIT OK with gcc-native/14.2) |
| PEFT | 0.18.1 | ✅ |
| accelerate | 1.13.0 | ✅ |
| bitsandbytes | 0.49.2 | ✅ |
| wandb | 0.25.1 | ✅ |
| datasets | 4.8.2 | ✅ |
| hydra-core | 1.3.2 | ✅ |
| ray | 2.54.0 | ✅ |
| flash-attn | N/A | ❌ Expected (ARM), use SDPA |

## Key Reminders
- Isambard is **aarch64 (ARM)** — not all x86 wheels work
- Always use `attn_implementation: "sdpa"` (no flash-attn on ARM) — but vLLM uses its own FlashAttention v3
- Multi-GPU: `module load brics/nccl` + `export NCCL_SOCKET_IFNAME=hsn`
- SLURM GPU flag: `--gpus=N` (not `--gpus-per-node=N`)
- Max walltime: 24h
- Scratch: `/scratch/u6gg/ts1201.u6gg`

## Critical Isambard Environment Variables
All SLURM scripts and interactive sessions MUST include:
```bash
source ~/miniforge3/bin/activate && conda activate kg_verl
module load cudatoolkit
module load gcc-native/14.2
export CC=gcc
export CXX=g++
export LIBRARY_PATH=/opt/nvidia/hpc_sdk/Linux_aarch64/24.11/REDIST/math_libs/12.6/targets/sbsa-linux/lib:$LIBRARY_PATH
export TRITON_CACHE_DIR=/scratch/u6gg/ts1201.u6gg/.triton
export HF_HOME=/scratch/u6gg/ts1201.u6gg/hf_cache
export TORCH_HOME=/scratch/u6gg/ts1201.u6gg/torch_cache
```
