# V14-A1 Search-R1 env build probe (Mon Apr 20 00:19:59 UTC 2026)

Platform: aarch64 (nid010004)
Goal: create a dedicated conda env `searchr1` compatible with Search-R1's pinned deps (vllm<=0.6.3, transformers<4.48, tensordict<0.6, flash-attn).

## Install outcomes
  - conda-create-python-3.10: OK
  - torch-2.4.1-cu121-pip: FAILED (see logs/p7_sr1_env-4027122.pkg.log)
  - torch-2.4-conda-forge: OK
  - vllm-0.6.3-pip: FAILED (see logs/p7_sr1_env-4027122.pkg.log)
  - vllm-0.6.3-from-source: FAILED (see logs/p7_sr1_env-4027122.pkg.log)
  - transformers-4.47: OK
  - tensordict-0.5: FAILED (see logs/p7_sr1_env-4027122.pkg.log)
  - flash-attn-pip: FAILED (see logs/p7_sr1_env-4027122.pkg.log)
  - searchr1-reqs: OK
  - hydra-core: OK
  - ray[default]: OK
  - datasets: OK
  - pandas: OK
  - pyarrow: OK
  - wandb: OK

## Import smoke tests
  - torch: OK
  - vllm: FAILED
  - transformers: OK
  - tensordict: FAILED
  - ray: OK
  - hydra: OK
  - datasets: OK
  - pyarrow: OK

## Smoke test results

- vllm end-to-end generate: FAIL
- Search-R1 vllm version accept: FAIL

## Verdict

**OPTION 1 FAILED or PARTIAL** — see logs for details.

Recommended next steps (user decision):
- Try Option 2: patch Search-R1's version check to accept vllm 0.12 (~30 min, risk of silent API breaks)
- Or go Option 3: reframe paper using our existing E1 (outcome-only) as the Search-R1-equivalent baseline (zero new compute)
