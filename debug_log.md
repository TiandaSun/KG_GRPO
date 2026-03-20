# GRPO 7B Smoke Test Debug Log

## Attempt 1: Job 3234336 (OOM)
- **Error**: `torch.OutOfMemoryError: CUDA out of memory` — FSDP used ~52GB + vLLM tried 50% = OOM
- **Fix**: `gpu_memory_utilization=0.5 → 0.3` in smoke job and main config

## Attempt 2: Job 3234401 (bucket too small)
- **Error**: `Weight model.embed_tokens.weight is too large to fit in the bucket` — Qwen2.5-7B embeddings are 2079MB, bucket default is 2048MB
- **Fix**: Added `+actor_rollout_ref.rollout.update_weights_bucket_megabytes=4096`

## Attempt 3: Job 3234411 (Hydra config error)
- **Error**: `RolloutConfig.__init__() got an unexpected keyword argument 'update_weights_bucket_megabytes'` — wrong Hydra path, `update_weights_bucket_megabytes` is under `checkpoint_engine`
- **Fix**: Changed to `actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=4096`

## Attempt 4: Job 3234433 (OOM during optimizer init)
- **Error**: `torch.OutOfMemoryError` during `adam.py _init_group` — FSDP model (70.6GB) + vLLM worker (23.7GB) = 94.3GB, no room for optimizer states
- **Root cause**: vLLM holds GPU memory during training because `enable_sleep_mode=false`. With sleep mode, vLLM releases GPU memory during training and only allocates during rollout.
- **Fix**: `enable_sleep_mode=true`, `ppo_micro_batch_size_per_gpu=1`, `max_response_length=1024`, `gpu_memory_utilization=0.4`

## Attempt 5: Job 3234520 (OOM — same class as attempt 4)
- **Error**: Same `torch.OutOfMemoryError` during Adam `_init_group` — 94GB used, 0.09GB free
- **Root cause**: With 1 GPU, FSDP switches to `NO_SHARD` (can't shard with world_size=1). Full fp32 optimizer states for 7B = ~45GB + model ~15GB + gradients ~15GB = ~75GB minimum, plus vLLM. Sleep mode helped vLLM release memory but optimizer states alone exceed budget.
- **Conclusion**: **7B GRPO on 1 GPU is not feasible** without CPU offloading. Need minimum 2 GPUs for FSDP to shard optimizer states.
- **Fix**: Changed to 2 GPUs (`--gpus=2`, `trainer.n_gpus_per_node=2`, `ray_kwargs.ray_init.num_cpus=144`)

## Attempt 6: Job pending
- 2 GPUs, FSDP FULL_SHARD, sleep mode enabled. Each GPU gets ~half the optimizer states (~37GB) + vLLM.
