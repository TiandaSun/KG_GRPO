"""Run this on a GPU node to verify CUDA/GPU access after environment setup."""
import platform
print(f"Architecture: {platform.machine()}")

import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    n = torch.cuda.device_count()
    print(f"GPU count: {n}")
    for i in range(n):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    # Quick compute test
    x = torch.randn(1000, 1000, device="cuda")
    y = x @ x.T
    print(f"Matrix multiply test: OK (result shape {y.shape})")
else:
    print("ERROR: CUDA not available! Check module load cudatoolkit")

try:
    import verl
    print("verl: OK")
except Exception as e:
    print(f"verl: FAILED ({e})")

try:
    import vllm
    print(f"vLLM: {vllm.__version__}")
except Exception as e:
    print(f"vLLM: FAILED ({e})")

print("\nAll checks done!")
