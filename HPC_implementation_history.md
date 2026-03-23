# HPC Implementation History

> Chronological record of all implementation work, decisions, and results.
> For the research agent to verify progress and identify gaps.

---

## Session 1: 2026-03-19 — Isambard Environment Setup

### What Was Done
- Installed verl 0.7.1 + vLLM 0.12.0 + DeepSpeed 0.18.8 on Isambard GH200 (ARM aarch64)
- Fixed 8 environment issues (vLLM version, flashinfer mismatch, ZMQ collision, flash_attn unavailable on ARM, etc.)
- Set up ConceptNet KG server (99,915 nodes, 146,553 edges)
- Ran 5-point smoke test (imports, data, rewards, KG server, 1.5B GRPO training)
- Generated 3,997 SFT trajectories from ConceptNet gold paths
- Submitted SFT warmup for 7B (LoRA r=64, 2 epochs)

### Key Decisions
- Use vLLM backend (NOT SGLang — broken on ARM due to sgl_kernel ABI mismatch)
- Use `attn_implementation: "eager"` (SDPA + bf16 + padding produces NaN grads on GH200; flash_attn unavailable)
- Always pass explicit `num_cpus` to Ray on ARM (auto-detection hangs)

### Package Versions
torch 2.9.0+cu126, vLLM 0.12.0, verl 0.7.1, DeepSpeed 0.18.8, Ray 2.54.0

---

## Session 2: 2026-03-20 — GRPO Smoke Test Debug + First Full Training

### GRPO Smoke Test (Attempts 1-7)
Iteratively fixed infrastructure issues to get GRPO running:

| Attempt | Issue | Fix |
|---------|-------|-----|
| 1 | CUDA OOM — vLLM gpu_memory_utilization too high | 0.5 → 0.3 |
| 2 | Embedding bucket too small for Qwen2.5-7B (2079MB) | bucket_megabytes=4096 |
| 3 | Wrong Hydra path for bucket config | Moved under checkpoint_engine |
| 4 | OOM during optimizer init — vLLM holds GPU during training | enable_sleep_mode=true |
| 5 | OOM — 7B can't fit on 1 GPU (FSDP NO_SHARD) | Minimum 2 GPUs |
| 6 | Exit code 1 from Ray shutdown (training actually completed) | Wrapped with set +e |
| 7 | Success | 6/6 steps, checkpoint saved |

### Zero Tool Calls Bug
After smoke test passed, discovered model never made tool calls (num_tool_calls=0).

**Root causes found and fixed:**
1. KG server port mismatch: config had 8001, server on 18901 → fixed config
2. `default_agent_loop: single_turn_agent` → set to `tool_agent`
3. `_VALID_ACTIONS` NameError in search_tool_parser.py — variable referenced before definition, causing silent import failure. `kg_search` parser never registered. → Reordered definitions

**Fix verified**: Job 3244443 showed num_tool_calls=3.3, r_tool_use=0.85, num_turns=8.5

### First Full Training (Job 3244577, ConceptNet)
- Config: 4x GH200, 4000 samples, 3 epochs (186 steps), batch_size=64, lr=5e-6
- Runtime: 7h7m
- Reward: ad-hoc hybrid (r_answer + r_coverage + r_format + r_tool_use + r_no_tool)

**Result: Tool-use collapse (Goodhart's law)**

| Metric | Step 0 | Step 186 |
|--------|--------|----------|
| reward/mean | 2.01 | 0.20 |
| num_tool_calls | 3.08 | **0.0** |
| r_tool_use | 1.03 | **0.0** |
| r_no_tool | -0.16 | **-1.0** |
| r_answer | 0.35 | 0.38 |
| num_turns | 7.94 | **2.0** |
| per-step time | ~97s | ~45s |

**Analysis**: Model learned to stop using tools. Explicit r_tool_use/r_no_tool bonuses created unstable optimization. Per-step timing confirmed: early steps ~100-200s (multi-turn), final steps ~45s (single-turn, no tools). KG requests stopped at ~step 150.

**Checkpoints saved**: steps 50, 100, 150, 186 — valuable for Goodhart analysis in paper.

### Headless Claude -p Debug Pattern
Developed and optimized autonomous debug loop via `claude -p`:
- Key flags: `--max-turns 50 --output-format stream-json --verbose`
- Prompt must start with ACTION (submit job), not investigation (read files)
- Default 10 max-turns is too low for submit-poll-fix loops

---

## Session 3: 2026-03-21 — Dataset Pivot + CWQ/Freebase Setup

### Design Analysis of Spec Rewards

Analyzed why the first run collapsed and identified issues with ConceptNet data:

1. **Gold answers too verbose** (mean 37 words) — EM near-useless, F1 rewards word overlap
2. **1-hop questions (39.5%) trivially solvable** — model doesn't need tools
3. **ConceptNet knowledge fully internalized** by 7B LLMs
4. **Ad-hoc reward matched no spec reward** — explicit tool bonuses caused collapse
5. **r_valid near-constant** on large KGs (0.3 weight wasted)

### Decision: Pivot to CWQ/Freebase
Based on analysis, discussed with research agent and produced hpc_implementation_spec_v3.md:
- CWQ (ComplexWebQuestions) on Freebase as primary dataset
- Short entity answers (clean EM/F1)
- 2-4 hop questions, 4 composition types
- Directly comparable to KG-R1, Graph-RFT, Explore-on-Graph
- ConceptNet demoted to dev/debug + hacking taxonomy evidence

### CWQ/Freebase Data Preprocessing

**Downloaded from HuggingFace** (RoG pre-processed datasets):
- `rmanluo/RoG-cwq` — 1.74GB, includes per-question Freebase subgraphs
- `rmanluo/RoG-webqsp` — 404MB

**Built global Freebase KG** from union of all per-question subgraphs:
- 2,592,894 entities
- 7,058 relations
- 8,309,195 triples
- Stored as: `data/freebase/kg/{entities,relations,triples}.txt`

**Converted to verl parquet**:

| Dataset | Train | Val | Test |
|---------|-------|-----|------|
| CWQ | 27,639 | 3,519 | 3,531 |
| WebQSP | 2,826 | 246 | 1,628 |

**BFS-estimated hop distribution (CWQ train)**:
- 0-hop (no path found): 5,550 (20%)
- 1-hop: 12,620 (46%)
- 2-hop: 9,469 (34%)

Note: These are BFS estimates on per-question subgraphs — true hop counts from SPARQL annotations not yet extracted.

**Script**: `scripts/prepare_cwq_freebase.py` (ran on compute node, Job 3252793, 11 min)

### Reward Functions Rewritten (Spec v3)

Replaced ad-hoc reward with 4 spec-compliant variants in `src_verl/rewards/verl_reward.py`:

1. **R_outcome**: `0.5 * EM + 0.5 * F1` on `<answer>` content vs short gold answer
2. **R_heuristic**: `0.3 * R_outcome + 0.7 * avg(entity_overlap_per_step)`
3. **R_verifiable**: `0.3 * R_outcome + 0.7 * avg(0.45*r_on_path + 0.30*r_progress + 0.15*r_coherence + 0.10*r_valid)`
4. **R_random**: `0.3 * R_outcome + 0.7 * avg(random_per_step)`

Selected via `REWARD_TYPE` environment variable. **No explicit tool bonuses** — lesson from collapse.

Fixed numpy ndarray serialization issue (parquet stores nested arrays that need recursive conversion).

### SFT Warmup on CWQ (Job 3252948)

**SFT data generation** (`scripts/generate_cwq_sft.py`):
- Built gold trajectories from CWQ KG paths
- 5,000 trajectories, 5-15 turns each (capped at 5 triples per trajectory)
- Format: system prompt → question → (think + search + tool_response)* → answer

**SFT training**:
- Qwen2.5-7B-Instruct, LoRA r=64, 2 epochs, 626 steps
- ~6.3s/step, 1h7m total on 1x GH200
- Logged to wandb: `sft-cwq-7b`

**LoRA merged** to full model at `outputs/verl-sft-cwq-7b-merged/` (14.4GB, 4 shards)

### GRPO Training Launched on CWQ

**Config** (`configs_verl/grpo_cwq_7b.yaml`):
- Model: `outputs/verl-sft-cwq-7b-merged`
- lr: 5e-7 (reduced from 5e-6 to prevent collapse)
- batch_size: 64, group_size: 8, max_turns: 5
- FSDP on 4x GH200, vLLM sleep mode, enforce_eager
- Checkpoints every 50 steps, eval every 50 steps
- wandb logging enabled

**Two experiments running in parallel** (2 nodes):

| Job ID | Experiment | Reward | Node | Walltime |
|--------|-----------|--------|------|----------|
| 3254091 | E1 | R_outcome (baseline) | nid010422 | 24h |
| 3254092 | E3 | R_verifiable (core) | nid010427 | 24h |

- 1293 steps each (27K × 3 epochs / 64 batch)
- ~60-70s/step estimated, ~21-25h total
- KG server loaded: 2.6M entities in 21s, healthy

---

## Current Status (2026-03-21 evening)

### Running
- Job 3254091: E1 (R_outcome) on CWQ — in progress
- Job 3254092: E3 (R_verifiable) on CWQ — in progress

### Completed
- [x] Isambard environment (verl + vLLM + DeepSpeed on GH200 ARM)
- [x] ConceptNet KG server + data pipeline
- [x] ConceptNet SFT + GRPO (first run, tool collapse — Goodhart evidence)
- [x] Freebase KG server backend (`data/freebase/kg/`, 2.6M entities, 8.3M triples)
- [x] CWQ + WebQSP data preprocessed to verl parquet
- [x] 4 reward functions implemented (outcome, heuristic, verifiable, random)
- [x] CWQ SFT warmup (7B, merged model ready)
- [x] E1 + E3 GRPO jobs submitted and running

### Not Yet Done
- [ ] E2 (R_heuristic) — submit after E1/E3 show results
- [ ] E4-E8 (cross-dataset, without-KG eval)
- [ ] E9-E12 (14B scaling)
- [ ] Difficulty filtering (base model A/B test on CWQ)
- [ ] Precompute BFS distances for r_progress (currently using entity overlap proxy)
- [ ] Extract true hop counts from CWQ SPARQL annotations
- [ ] Claim Verification data + task (E14-E16)
- [ ] Think-Verify task (E17-E19)
- [ ] Ablations (E20-E22)
- [ ] Goodhart analysis on checkpoints
- [ ] Without-KG reasoning quality evaluation

### Compute Budget
- Total allocation: 2,500 node-hours (10,000 GPU-hours)
- Used so far: ~15 node-hours (smoke tests + ConceptNet run + SFT + preprocessing)
- Currently running: 2 nodes × 24h = 48 node-hours
- Remaining after current jobs: ~2,427 node-hours
- Budget per day (60 days to deadline): ~40 node-hours/day

### Key Files

| File | Purpose |
|------|---------|
| `hpc_implementation_spec_v3.md` | Current research plan and experiment matrix |
| `configs_verl/grpo_cwq_7b.yaml` | GRPO training config for CWQ |
| `configs_verl/sft_cwq.yaml` | SFT config for CWQ |
| `configs_verl/kg_tool_config_freebase.yaml` | Freebase KG tool config |
| `src_verl/rewards/verl_reward.py` | 4 reward functions (outcome/heuristic/verifiable/random) |
| `src_verl/interaction/search_tool_parser.py` | Tool call parser with alias support |
| `src_verl/kg_server/server.py` | Unified KG server (ConceptNet + Freebase) |
| `src_verl/kg_server/freebase_adapter.py` | Freebase data loader |
| `scripts/run_grpo_cwq.job` | GRPO job script (REWARD_TYPE via env var) |
| `scripts/run_sft_cwq.job` | SFT pipeline job (data gen + train + merge) |
| `scripts/prepare_cwq_freebase.py` | CWQ/Freebase data preprocessing |
| `outputs/verl-sft-cwq-7b-merged/` | SFT-warmed 7B model for GRPO |
| `data/freebase/kg/` | Global Freebase subgraph (2.6M entities) |
| `data/freebase/verl_cwq/` | CWQ verl parquet (train/val/test) |
| `data/freebase/verl_webqsp/` | WebQSP verl parquet (train/val/test) |

### Lessons Learned (Must Follow)

1. **No explicit tool bonuses/penalties** — causes collapse (proven on ConceptNet)
2. **Use short entity answers for EM/F1** — verbose answers make EM useless
3. **lr=5e-7, not 5e-6** — high lr accelerates reward hacking
4. **Implicit tool incentive via 0.30/0.70 answer/step split**
5. **Always set `default_agent_loop=tool_agent`** — verl defaults to single_turn
6. **Always set `enable_sleep_mode=true`** — required for colocated FSDP + vLLM
7. **`bucket_megabytes=4096`** — default too small for Qwen2.5-7B embeddings
8. **Convert numpy arrays from parquet** — nested ndarrays need recursive `_to_python()`
9. **ARM/GH200 specifics**: eager attention, gcc compiler, explicit Ray num_cpus, enforce_eager for vLLM
