# KG_GRPO Checkpoint & Trajectory Audit — 2026-04-11

Read-only audit of all GRPO/SFT checkpoints, trajectories, and training configs on Isambard AI Phase 2. Produced for KG_Align Storyline D representation analysis. No files were created, deleted, modified, chmod-ed, or symlinked during data collection; no SLURM jobs were submitted.

---

## Q8 — Cluster identity (confirmed)

| Fact | Evidence |
|---|---|
| Cluster: **Isambard AI Phase 2** | login banner `_Isambard AI_ --- Phase 2 ---` |
| Architecture: **aarch64 (ARM GH200)** | `uname -m` = `aarch64`, `uname -a` kernel `cray_shasta_c_64k` |
| OS: SLES 15 SP6 | `/etc/os-release` |
| Login node hostname: `login40` | `hostname` |
| Matches CLAUDE.md Isambard AI entry (GH200, ARM, scratch `/scratch/u6gg/ts1201.u6gg`) | — |

---

## Q7 — NHR / quota query commands (partial)

| Command | Absolute path | Purpose | Status |
|---|---|---|---|
| `bquota` | `/tools/brics/quota/bin/bquota` | **Storage quota** (/projects, /scratch, /home, /local) | user-facing, confirmed |
| `sshare -U` | `/usr/bin/sshare` | SLURM fairshare (raw usage for current account) | SLURM, confirmed |
| `sacctmgr show assoc user=$USER` | `/usr/bin/sacctmgr` | QOS / association (workq_qos, no GrpTRESMins exposed) | SLURM, confirmed |
| `adm-quota` | `/tools/brics/admin/adm-quota/bin/adm-quota` | Admin tool | permission denied to u6gg |

**No dedicated "node-hours remaining" command found** (e.g. `nhr-balance`, `gpu-hours-left`). `sacctmgr` reports `workq_qos` without any `GrpTRESMins` limit, and the `brics.u6gg` account cannot run admin queries. Historical usage can be pulled via `sreport cluster AccountUtilizationByUser start=YYYY-MM-DD` (not executed in this session).

**Current storage usage** (from `bquota`):

| Directory | Used | Limit | % |
|---|---|---|---|
| `/projects/u6gg` | 11.43 TB | 200.00 TB | 5.71% |
| `/scratch/u6gg/ts1201.u6gg` | 0.53 TB | 5.00 TB | 10.58% |
| `/local/user/1483804591` | 0.01 GB | 512.00 GB | 0% |
| `/home/u6gg/ts1201.u6gg` | (unreported) | — | — |

---

## Q6 — SLURM queue (confirmed)

```
JOBID    USER         PARTITION  NAME      ST  TIME_LIMIT  TIME   TIME_LEFT  NODES  REASON
3749118  ts1201.u6gg  workq      llama_a4  PD  6:00:00     0:00   6:00:00    1      Priority
```

Only one **pending** job (`llama_a4`, 6 h wall, Priority hold). **No running jobs** — GPUs are currently not blocked for KG_Align use.

---

## Q1 — Full GRPO / SFT checkpoint inventory (confirmed)

### A. GRPO runs — all under `/projects/u6gg/KG_GRPO/checkpoints/kg-align-verl/` (`$PROJECTDIR`, 200 TB)

Format is uniformly **verl-FSDP**: every `global_step_N/actor/` contains `model_world_size_4_rank_{0-3}.pt` + `extra_state_*` + `fsdp_config.json` + a `huggingface/` subdirectory. Each run root has a `latest_checkpointed_iteration.txt` file.

| Run directory | Saved steps | Total size | Latest mtime |
|---|---|---|---|
| `grpo-cwq-7b-outcome-20260321` | 100, 250, 500, 750, 1000, 1250 | **228 G** | 2026-03-25 01:51 |
| `grpo-cwq-7b-heuristic-20260325` | 200, 400, 600, 800, 1000, 1200 | **511 G** | 2026-03-25 17:11 |
| `grpo-cwq-7b-verifiable-20260321` | 100, 250, 350, 400, 450, 500, 600, 750, 1000, 1250 | **341 G** | 2026-03-25 01:51 |
| `grpo-cwq-7b-random-20260326` | 50–1250 @ 50 (25 steps) | **710 G** | 2026-03-27 06:02 |
| `grpo-cwq-7b-retrieval-grounded-20260326` | 50–1250 @ 50 + 1293 (26 steps) | **739 G** | 2026-03-28 00:22 |
| `grpo-cwq-7b-tool-type-bonus-20260326` | 50–700 @ 50 (14 steps) | **398 G** | 2026-03-29 00:04 |
| `grpo-cwq-7b-verifiable-balanced-20260330` | 50–2050 @ 50 (41 steps) | **3.5 T** | 2026-04-01 19:19 |
| `grpo-cwq-7b-verifiable-onpath-20260330` | 50–1250 @ 50 (25 steps) | **2.1 T** | 2026-03-31 00:33 |
| `grpo-cwq-llama-8b-verifiable-20260331` | 50–1250 @ 50 + 1293 (26 steps) | **1.2 T** | 2026-04-04 14:43 |
| `grpo-cwq-llama-8b-tool-type-bonus-20260331` | 50–1250 @ 50 + 1293 (26 steps) | **1.2 T** | 2026-04-04 13:45 |

Subtotal ≈ **10.9 TB** across 10 GRPO runs (matches `bquota`-reported 11.43 TB with ~0.5 TB slack for other files).

### B. Scratch-side secondary GRPO copy

| Path | Format | Steps | Size | Notes |
|---|---|---|---|---|
| `/lus/lfs1aip2/scratch/u6gg/ts1201.u6gg/Project/KG_GRPO/checkpoints/kg-align-verl/grpo-cwq-7b-heuristic-20260325-fine` | verl-FSDP | 50, 100, 150, 200, 250, 300, 350, 400, 450, 500 | 284 G | **Separate `-fine` variant** — name suffix `-fine` and step granularity 50 (vs projects-side step 200). Likely a pre-migration fine-grained run that was not continued into Phase 2. |

### C. SFT runs — all under `/lus/lfs1aip2/scratch/u6gg/ts1201.u6gg/Project/KG_GRPO/outputs/`

| Run directory | Format | Step | Size | mtime |
|---|---|---|---|---|
| `verl-sft-7b/checkpoint-250` | LoRA adapter | 250 | 1.9 G | 2026-03-19 01:27 |
| `verl-sft-7b/checkpoint-500` | LoRA adapter | 500 | 1.9 G | 2026-03-19 01:52 |
| `verl-sft-7b-merged` | HF safetensors (4-shard) | — | **15 G** | 2026-03-20 14:00 |
| `verl-sft-cwq-7b/checkpoint-313` | LoRA adapter | 313 | 1.9 G | 2026-03-21 12:29 |
| `verl-sft-cwq-7b/checkpoint-626` | LoRA adapter | 626 | 1.9 G | 2026-03-21 13:02 |
| `verl-sft-cwq-7b-merged` | HF safetensors | — | **15 G** | 2026-03-21 13:03 |
| `verl-sft-llama-8b/checkpoint-158` | LoRA adapter | 158 | 2.0 G | 2026-03-31 09:21 |
| `verl-sft-llama-8b-merged` | HF safetensors | — | **15 G** | 2026-03-31 09:22 |

### D. Other findings (logged, not touched)

- `/scratch/u6gg/ts1201.u6gg/Project/core` — 2.9 MB coredump, mtime 2026-03-17 14:02. **Not read, not deleted.**
- Inventory agent reported **77 hydra config dumps** under `outputs/YYYY-MM-DD/HH-MM-SS/.hydra/` (2026-03-18 to 2026-04-03), most without matching checkpoint directories — likely smoke / debug / pre-migration runs overwritten after the checkpoint directory was moved to `$PROJECTDIR`.

---

## Q2 — Training config sources (confirmed)

| Run group | Config file | Reward selection | Base model | Dataset |
|---|---|---|---|---|
| All `grpo-cwq-7b-*-20260321/25/26/30` | `/lus/lfs1aip2/scratch/u6gg/ts1201.u6gg/Project/KG_GRPO/configs_verl/grpo_cwq_7b.yaml` | **env var `REWARD_TYPE`** (outcome \| heuristic \| verifiable \| random \| retrieval_grounded \| tool_type_bonus) | `Qwen/Qwen2.5-7B-Instruct` (actor init from `verl-sft-cwq-7b-merged`) | `data/freebase/verl_cwq/{train,val}.parquet` |
| `grpo-cwq-llama-8b-*-20260331` | `/lus/lfs1aip2/scratch/u6gg/ts1201.u6gg/Project/KG_GRPO/configs_verl/grpo_cwq_llama_8b.yaml` | env var `REWARD_TYPE` | `meta-llama/Llama-3.1-8B-Instruct` (actor init from `verl-sft-llama-8b-merged`) | same |
| SFT warmup (Qwen) | `configs_verl/sft_multiturn.yaml` / `configs_verl/sft_cwq.yaml` | N/A | `Qwen/Qwen2.5-7B-Instruct` | `data/processed/sft_trajectories.jsonl` / `data/freebase/sft_trajectories.jsonl` |
| Reward implementation | `/lus/lfs1aip2/scratch/u6gg/ts1201.u6gg/Project/KG_GRPO/src_verl/rewards/verl_reward.py` | dispatched on `REWARD_TYPE` | — | — |

Reward formulas (from `verl_reward.py`):

- `outcome = 0.5·EM + 0.5·F1`
- `heuristic = 0.3·outcome + 0.7·entity_overlap`
- `verifiable = 0.3·outcome + 0.7·(r_on_path + r_progress + r_coherence + r_valid)`
- `random = 0.3·outcome + 0.7·random_step_rewards`

---

## Q3 — E-tag ↔ checkpoint mapping (confirmed)

Cross-verified via `results/trajectories/*/trajectories.json`, `hpc_tasks.md` Tasks §6/7/20/21, and git commit `2d96f0e`.

| E-tag | Reward | Checkpoint directory | On disk? | Step range | Trajectory directory |
|---|---|---|---|---|---|
| **E1** | outcome | `grpo-cwq-7b-outcome-20260321` | ✅ | 100, 250, 500, 750, 1000, 1250 | `results/trajectories/e1_outcome/` |
| **E2** | heuristic | `grpo-cwq-7b-heuristic-20260325` + scratch `-fine` copy | ✅ | projects: 200…1200 @ 200; scratch `-fine`: 50…500 @ 50 | `results/trajectories/e2_heuristic/` |
| **E3** | verifiable | `grpo-cwq-7b-verifiable-20260321` | ✅ | 100, 250, 350, 400, 450, 500, 600, 750, 1000, 1250 | `results/trajectories/e3_verifiable/` |
| **E4** | random | `grpo-cwq-7b-random-20260326` | ✅ | 50–1250 @ 50 (25 steps) | `results/trajectories/e4_random/` |
| **E5a** | retrieval_grounded | `grpo-cwq-7b-retrieval-grounded-20260326` | ✅ | 50–1293 (26 steps) | `results/trajectories/e5a_retrieval_grounded/` |
| **E5b** | tool_type_bonus | `grpo-cwq-7b-tool-type-bonus-20260326` | ✅ | 50–700 @ 50 (14 steps) | `results/trajectories/e5b_tool_type_bonus/` + `e5b_collapse_analysis/` |

### Additional non-E1–E5b ablation runs (Tasks §22/23 — verifiable reward component ablations; do NOT confuse with the main E-tags)

- `grpo-cwq-7b-verifiable-balanced-20260330` ↔ `results/trajectories/e3_balanced/` (3.5 TB)
- `grpo-cwq-7b-verifiable-onpath-20260330` ↔ `results/trajectories/e3_onpath/` (2.1 TB)
- `grpo-cwq-llama-8b-{verifiable,tool-type-bonus}-20260331` — Llama-8B cross-model runs, 1.2 TB each

---

## Q4 — Missing-checkpoint forensics (confirmed)

**Nothing is missing.** All six E-tag checkpoints are present in `/projects/u6gg/KG_GRPO/checkpoints/kg-align-verl/`.

The only "location divergence" is E2: scratch has a fine-grained `grpo-cwq-7b-heuristic-20260325-fine` copy (284 G, steps 50–500) while projects has the canonical run (511 G, steps 200–1200). These are **not the same artefact** — they were produced before and after the default checkpoint directory was migrated from scratch to `$PROJECTDIR`. Git commit `2d96f0e` carries the config change that moved `default_local_dir` to `/projects/u6gg/...`.

Evidence sources checked (all clean):

- `git log --all --diff-filter=D --name-only` — no checkpoint files ever deleted from git
- No `cleanup` / `purge` / `rm -rf` scripts in `scripts/`
- `crontab -l` is empty for `$USER`
- SLURM stdout/stderr logs contain no OOM / checkpoint-save-failed errors around the E1–E5b training windows

---

## Q5 — FSDP → HF merge tooling

### Existing merged artefact

`/lus/lfs1aip2/scratch/u6gg/ts1201.u6gg/Project/KG_GRPO/outputs/verl-sft-7b-merged/` — **15 GB**, HF 4-shard safetensors + complete config/tokenizer. **This is a LoRA adapter merge, NOT an FSDP merge.**

### Tool location

| Tool | Absolute path | Purpose | Status |
|---|---|---|---|
| `scripts/merge_sft_adapter.py` | `/lus/lfs1aip2/scratch/u6gg/ts1201.u6gg/Project/KG_GRPO/scripts/merge_sft_adapter.py` | **LoRA → HF merge** (SFT pipeline only; PEFT + transformers) | confirmed; produced all `verl-sft-*-merged/` |
| `scripts/eval_checkpoints.py` (`load_fsdp_checkpoint` helper, L35–100) | `/lus/lfs1aip2/scratch/u6gg/ts1201.u6gg/Project/KG_GRPO/scripts/eval_checkpoints.py` | **In-memory FSDP shard consolidation** for evaluation; does NOT persist an HF artefact | confirmed |
| verl upstream `model_merger` | not called directly in this project | upstream tool | `src_verl/evaluation/evaluate_multiturn.py` L118 comment mentions "verl checkpoint converted via model_merger" — **inferred**, no historical invocation found in this repo |

### Key conclusion

- **This project has no established persistent FSDP→HF merge workflow.** The audit found no `scripts/merge_fsdp*.sh`, no `*.sbatch` wrapper, and no `logs/*merge*.out` history.
- The existing `verl-sft-*-merged/` directories are LoRA merges, not FSDP merges.
- To produce standalone HF checkpoints from `grpo-cwq-7b-*/global_step_N/actor/`, a new workflow would be needed. Upstream verl exposes `python -m verl.model_merger --backend fsdp --local_dir <step>/actor --target_dir <out>`, but this command has never been run in this repo.
- Each GRPO `global_step_N/actor/` already contains a `huggingface/` subdirectory. This audit **did not inspect whether those subdirectories already hold consolidated safetensors** — that is a cheap `ls` check for future follow-up and does not require a merge.

### Resource profile

No historical SLURM logs, so **wall time / peak memory are unknown**. Inference only:

- LoRA merge is likely single-GPU and minutes-scale (adapter ≈ 650 MB; output ≈ 15 GB)
- FSDP merge for a 7B model typically loads all shards into CPU RAM → **estimated ~30 GB RAM, minutes-scale on CPU**. This is an estimate, not measured data.

### Example commands

**LoRA → HF merge (confirmed, this is what produced `verl-sft-7b-merged/`):**

```bash
python scripts/merge_sft_adapter.py \
    --base_model Qwen/Qwen2.5-7B-Instruct \
    --adapter_path outputs/verl-sft-7b \
    --output_path outputs/verl-sft-7b-merged
```

**FSDP → HF merge (inferred, upstream verl, never run here):**

```bash
python -m verl.model_merger \
    --backend fsdp \
    --local_dir /projects/u6gg/KG_GRPO/checkpoints/kg-align-verl/grpo-cwq-7b-heuristic-20260325/global_step_1200/actor \
    --target_dir <output_path>
```

---

## TL;DR — Storyline D representation analysis availability

| E-tag | Reward | Checkpoint status | Trajectory | Representation analysis availability |
|---|---|---|---|---|
| **E1** outcome | ✅ 6 steps (100–1250), verl-FSDP, 228 G | ✅ | **Available** — but FSDP shards must be merged into HF form first. This project has never run a persistent FSDP merge, so that workflow is new. |
| **E2** heuristic | ✅ 6 steps (200–1200) + scratch `-fine` 10 steps (50–500), 511+284 G | ✅ | **Available** (the `-fine` copy has finer step granularity and may be preferable for mid-layer trajectory tracing). |
| **E3** verifiable | ✅ 10 steps, 341 G | ✅ | **Available** |
| **E4** random | ✅ 25 steps (50–1250 @ 50), 710 G | ✅ | **Available** (one of the finest step grids) |
| **E5a** retrieval_grounded | ✅ 26 steps (50–1293), 739 G | ✅ | **Available** |
| **E5b** tool_type_bonus | ✅ 14 steps (50–700), 398 G | ✅ | **Available** (training stopped / collapsed around step 700) |

**Conclusion: all six E-tag checkpoints are on disk, all in verl-FSDP format. No run is trajectory-only. No run has been lost.** The only engineering caveat is that this project has never run a persistent FSDP→HF merge workflow (only LoRA merges), so downstream representation analysis must either (a) implement an FSDP merge pipeline or (b) reuse the in-memory `load_fsdp_checkpoint` helper from `scripts/eval_checkpoints.py` as a representation-extraction hook. This report deliberately does not recommend which path to take — that decision is for the researcher.
