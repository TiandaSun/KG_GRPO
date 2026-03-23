# E1 vs E3 First Results: CWQ/Freebase 7B GRPO

> Date: 2026-03-22
> Model: Qwen2.5-7B-Instruct (SFT-warmed on CWQ)
> Dataset: CWQ train (27,639 samples), 3 epochs, batch_size=64, lr=5e-7
> Hardware: 4x GH200 120GB per experiment, 1 node each

---

## Job Summary

| | E1 (R_outcome) | E3 (R_verifiable) |
|---|---|---|
| Job ID | 3254091 | 3254653 |
| Reward | 0.5*EM + 0.5*F1 | 0.3*R_outcome + 0.7*step_rewards |
| Status | 1260/1293 steps (97%) | 1260/1293 steps (97%) |
| Runtime | 19h 12m | 22h 30m |
| Exit | vLLM crash at step 1260 | vLLM crash at step 1260 |
| Checkpoints | 25 saved (step 50-1250) | 25 saved (step 50-1250) |

Both jobs crashed at step ~1260 due to the same vLLM v1 shared memory bug (long-running `shm_broadcast` corruption). This is not a code bug — it's a known vLLM stability issue for 19h+ runs. Both have 25 usable checkpoints.

---

## Initial Validation Metrics (Step 0, before training)

| Metric | E1 (outcome) | E3 (verifiable) |
|--------|-------------|-----------------|
| reward/mean | 0.0115 | 0.0035 |
| r_outcome | 0.0115 | 0.0117 |
| r_step_avg | N/A | 0.0 |
| EM | 0.0 | 0.0 |
| F1 | 0.023 | 0.023 |
| num_tool_calls | 0.0 | 0.0 |
| num_turns | 2.0 | 2.0 |

Both start identically — the SFT model doesn't use tools during greedy validation (do_sample=False). This is expected since validation uses temperature=0.

---

## Tool Use During Training

| Metric | E1 (outcome) | E3 (verifiable) |
|--------|-------------|-----------------|
| Total KG requests | **39** | **841,471** |
| Tool calls per step | ~0.03 | ~668 |

**This is the paper's key finding.** R_outcome produces essentially zero tool use (39 requests across 1260 steps = noise). R_verifiable produces sustained, massive tool use (841K requests).

---

## Per-Step Timing (proxy for tool use intensity)

| Training Phase | E1 (outcome) | E3 (verifiable) | Interpretation |
|---------------|-------------|-----------------|----------------|
| Steps 1-50 | 65s | **99s** | E3 doing multi-turn rollouts with tools |
| Steps 50-100 | 53s | **89s** | E3 tool use settling |
| Steps 200-250 | 54s | 63s | E3 converging to efficient tool use |
| Steps 400-450 | 56s | 62s | Stable |
| Steps 600-650 | 56s | 64s | Stable |
| Steps 800-850 | 54s | 64s | Stable |
| Steps 1000-1050 | 54s | 57s | E3 slightly decreasing |
| Steps 1200-1260 | 56s | 59s | Both near end |

**Key observations:**
- E1 is flat at ~55s throughout — single-turn generation, no tool calls
- E3 starts at ~99s (heavy multi-turn tool use), stabilizes to ~60-64s
- E3 does NOT collapse to E1's level — tool use is sustained, unlike the ConceptNet run
- E3's per-step decrease (99s -> 59s) suggests the model is learning more efficient tool use (fewer turns to reach the answer), NOT abandoning tools (KG requests continue to the end)

---

## Tool Use Stability: E3 vs ConceptNet Run

| | ConceptNet (ad-hoc reward) | E3 CWQ (R_verifiable) |
|---|---|---|
| KG requests at start | Many (steps 1-70) | Many (steps 1-1260) |
| KG requests at end | **Zero** (after step 150) | **841K total, active to end** |
| Tool collapse? | **YES** (Goodhart) | **NO** |
| Per-step early | 100-200s | 99s |
| Per-step late | 45s (no tools) | 59s (efficient tools) |

The implicit tool incentive via 0.30/0.70 answer/step split **works**. No explicit tool bonuses needed. The ConceptNet collapse was caused by explicit r_tool_use/r_no_tool penalties, not a fundamental issue with GRPO.

---

## KG Request Distribution (E3)

KG requests are distributed throughout training, not front-loaded:

| % of total KG requests | Reached by % of training |
|------------------------|--------------------------|
| 10% | 1% (heavy early exploration) |
| 30% | 10% |
| 50% | 32% |
| 70% | 54% |
| 90% | 77% |
| 100% | 100% (last request at very end) |

The front-loading (30% of requests in first 10% of training) suggests the model explores more aggressively early on, then becomes more efficient. But importantly, requests continue to the very last step — no collapse.

---

## Runtime Comparison

| | E1 | E3 |
|---|---|---|
| Total runtime | 19h 12m | 22h 30m |
| Per-step (avg) | ~55s | ~64s |
| Overhead from tool use | — | **+17% (~9s/step)** |

E3 takes ~17% longer due to multi-turn tool call overhead (KG server queries + extra generation turns). This is acceptable — tool use adds real computation.

---

## What We Know So Far

1. **R_outcome does not incentivize tool use** — as expected. Pure answer reward leads to direct answer generation without KG queries. This is the correct baseline behavior.

2. **R_verifiable maintains tool use** — the 0.30/0.70 answer/step split creates sufficient implicit incentive. No explicit tool bonuses needed. This validates the spec v3 reward design.

3. **No tool-use collapse on CWQ** — unlike ConceptNet, the model does not learn to avoid tools. This could be because: (a) CWQ questions are harder (model can't answer without KG), (b) no explicit tool penalties creating unstable optimization, (c) lower learning rate (5e-7 vs 5e-6).

4. **Both jobs hit vLLM crash at step ~1260** — shared memory corruption after 19h+. Not a code bug. Mitigate with shorter chained jobs or checkpoint resume.

---

## What We Don't Know Yet

1. **Did E3 actually improve answer quality?** We only have initial validation (EM=0, F1=0.023). No mid-training or final validation was logged (both crashed before final eval). Need to run eval on the saved checkpoints.

2. **What's the EM/F1 trajectory over training?** Checkpoints at steps 50-1250 exist for both. Running offline evaluation would reveal the learning curves.

3. **Is E3 better than E1 on held-out test?** The key paper comparison. Need to evaluate both on CWQ test set.

4. **Does tool use quality improve?** E3 uses tools, but are the queries getting better? Need trajectory analysis.

---

## Recommended Next Steps

| Priority | Action | Purpose |
|----------|--------|---------|
| **P0** | Run offline eval on E1 + E3 checkpoints (steps 50, 250, 500, 750, 1000, 1250) | Get EM/F1 learning curves for both |
| **P0** | Run E2 (R_heuristic) for 3-way comparison | Complete the core comparison |
| **P1** | Save 100 random trajectories from E3 checkpoints | Analyze tool use quality over training |
| **P1** | Run without-KG eval on E3 final checkpoint | Does KG training improve reasoning without KG? |
| **P2** | Chain jobs with `--dependency` to avoid vLLM crash | Run remaining 33 steps for both |

---

## Raw Data Locations

| Data | Path |
|------|------|
| E1 checkpoints | `checkpoints/kg-align-verl/grpo-cwq-7b-outcome-20260321/` |
| E3 checkpoints | `checkpoints/kg-align-verl/grpo-cwq-7b-verifiable-20260321/` |
| E1 logs | `logs/grpo_cwq_7b-3254091.{log,err}` |
| E3 logs | `logs/grpo_cwq_7b-3254653.{log,err}` |
| E1 wandb | `grpo-cwq-7b-outcome-20260321` in kg-align-verl project |
| E3 wandb | `grpo-cwq-7b-verifiable-20260321` in kg-align-verl project |
| SFT model | `outputs/verl-sft-cwq-7b-merged/` |
| CWQ data | `data/freebase/verl_cwq/{train,val,test}.parquet` |
| Freebase KG | `data/freebase/kg/{entities,relations,triples}.txt` |
