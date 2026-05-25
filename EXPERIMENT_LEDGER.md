# KG-GRPO Experiment Ledger

**Last updated:** 2026-04-25 (V14-D1 strat locked + V14-E1 32B chain launched + V14-D1 CvT expansion 500→1500 launched)

**Active 2026-04-25:**
- 4289621 p7_v14d1_cvt_exp — 14B CvT classification expansion 500→1500 (1 GPU × 12h)
- 4289629 p7_v14e1_sft_pilot — Qwen2.5-32B SFT pilot (4 GPUs × 6h)
- 4289630 p7_v14e1_sft_full — afterok pilot, 32B SFT full (4 GPUs × 24h)
- 4289631 p7_v14e1_grpo2n — afterok SFT, 32B GRPO 2-node × 4-GPUs × 24h
- 4289632 p7_v14e1_eval — afterany GRPO, eval at STEP=400 (1 GPU × 12h)


Compact operational tracker. Paper story + canonical numbers live in `PAPER_HANDOFF.md` and `results/phase7/paper_tables.md`; forward task queue lives in `hpc_tasks.md`. This file answers: *what has run, what's running, what's queued, what failed*.

---

## 1. Currently running

| Job ID | Name | Start (UTC) | Walltime left | Progress |
|---|---|---|---|---|
| 4247718 | p7_v14d1_cvt500 (500-sample CvT eval w/ trajectories) | 2026-04-23 11:56 | 6h | PD |
| 4247719 | p7_v14d1_topup (full-3531 resume from 3000) | 2026-04-23 11:56 | 6h | PD |
| 4247849 | p7_v14c1_prep (fs1-2708 download + inspect) | 2026-04-23 12:00 | **DONE 19s** | 5974 rows, 3272 unique CWQ val IDs, all valid=1. **Zero overlap with test split** (fair comparison). |
| 4250151 | p7_v14c1_sft (Qwen2.5-7B SFT on fs1-2708) | 2026-04-23 14:11 | 24h | FAILED 1m18s — TRL API arg `max_length`→`max_seq_length` |
| 4252102 | p7_v14c1_sft (attempt 2) | 2026-04-23 16:35 | 24h | FAILED 1h24m @step 135/935 — Lustre BrokenPipe (HPC infra outage) |
| 4255083 | p7_v14c1_sft (attempt 3) | 2026-04-24 12:10 | 24h | FAILED 1m11s — torch checkpoint+SDPA recompute tensor-shape mismatch |
| **4268838** | **p7_v14c1_sft (attempt 4, eager attn)** | 2026-04-24 19:05 | 24h | **DONE 4h47m** — merged ckpt at `outputs/verl-sft-cwq-7b-fs1-adapt-merged/` |
| 4268839 | p7_v14c1_fs1eval (Part A eval) | 2026-04-24 23:53 | 8h | TIMEOUT @1500/3531; **EM=2.07%, Tools/Q=0.09** — Part B held by gate (EM<<30%). Saved as `results/phase7/v14_c1_fs1_sft_only_full_test.json`. |

**V14-D1-strat Table 1 (preliminary, partial data n=3000/3531):** `results/phase7/v14_d1_strat.{json,md}`
- 14B D1@400 full EM = 1260/3000 = 42.00% (Cat A 76.16%, Cat B 28.27%)
- Decomposition 14B − 7B-39B: Δ_A = +3.03pp × p_A(0.276) + Δ_B = +3.19pp × p_B(0.724) ≈ **+3.15pp total**
- **Cat B Δ = +3.19pp is BORDERLINE vs the user's +3pp threshold.** Will refresh after topup (4247719) completes. Do NOT lock narrative yet.

**V14-C1 chain status** (submitted as prep-first due to ARM-ecosystem risk; same lesson as Search-R1):
- Prep (4247849): download `jjzha/fs1-2708`, inspect format, convert to SFT jsonl → blocker for rest of chain.
- **Pending prep outcome** — will submit these next (will need user to approve pragmatic path or I'll just launch):
  - `run_v14_c1_sft.job`: 4 GPUs × 24h, Qwen2.5-7B-Instruct + fs1-2708 + fs1 hyperparams (LR=1e-5, 5 epochs, 8192 ctx, bf16, FSDP — via our TRL SFTTrainer stack, not fs1's accelerate script). Adaptation approach — fs1 *recipe* reproduced in our infrastructure. Flagged in methods section.
  - `run_v14_c1_fs1only_eval.job`: 1 GPU × 6h, afterany SFT, full-3531 strict-EM eval. Produces the "fs1-SFT-only" data point.
  - `run_v14_c1_fs1plus_grpo.job`: 4 GPUs × 24h, afterany SFT, G2-style ReST-EM (tool-type-bonus, KL=0.25) from fs1-SFT init.
  - `run_v14_c1_fs1plus_eval.job`: 1 GPU × 6h, afterany GRPO, eval at final step. Produces the "fs1-SFT + our-RL" data point.
- If prep reveals incompatibility (e.g., fs1 text format needs heavy transform, or data is multi-benchmark mix that contaminates CWQ comparison), I'll flag it at next status check instead of launching a broken chain.

**Queued (afterany dependency):**
| Job ID | Name | Depends on |
|---|---|---|
| 4139905 | p7_v14d1_eval | 4139890 GRPO → merge + full-3531 eval + trajectories |

---

## 2. Completed GRPO runs (all on Qwen2.5-7B-Instruct SFT base unless noted)

Checkpoint root: `/projects/u6gg/KG_GRPO/checkpoints/kg-align-verl/`

### Phase 1-2 reward ablations (early — context only, mostly superseded)
| Tag | Experiment name | Status |
|---|---|---|
| Outcome (E1 old) | grpo-cwq-7b-outcome-20260321 | done |
| Verifiable (E3 old) | grpo-cwq-7b-verifiable-20260321 | done |
| Heuristic | grpo-cwq-7b-heuristic-20260325 | done |
| Random | grpo-cwq-7b-random-20260326 | done |
| Retrieval-grounded | grpo-cwq-7b-retrieval-grounded-20260326 | done |
| Tool-type-bonus | grpo-cwq-7b-tool-type-bonus-20260326 | done |
| Verifiable-balanced | grpo-cwq-7b-verifiable-balanced-20260330 | done |
| Verifiable-onpath | grpo-cwq-7b-verifiable-onpath-20260330 | done |

### Phase 3-5 canonical: E1-E5b (full-3531 eval below)

| Canonical tag | Final ckpt | Full-test result JSON |
|---|---|---|
| E1 (outcome)         | step 1250 | results/task14_full_e1_1250_chunk{0,1}.json |
| E2 (heuristic)       | step 1200 | results/task14_full_e2_1200_chunk{0,1}.json |
| E3 (verifiable)      | step 500, 750 | results/task14_full_e3_500.json, _750.json; phase7/e3_step500_full_test.json |
| E4 (random)          | step 1250 | results/task14_full_e4_1250.json |
| E5a (balanced onpath)| step 1000 | results/task14_full_e5a_1000.json |
| **E5b (tool-type-bonus)** | **step 100** | results/task14_full_e5b_100.json; **phase7/e5b_step100_full_test.json** |

Headline: E5b@100 is the CvT winner at 10% correct-via-tool (first tool-grounded reward to beat 0%).

### Phase 6 Llama 8B (confirms Qwen pattern)
| Tag | Final | Result |
|---|---|---|
| Llama-E3 verifiable | step 1293 | results/task14_full_llama_e3_1293_chunk{0,1,2,3}.json |
| Llama-E5a balanced  | step 1293 | (evaluated, no chunk file) |
| Llama-E5b tool-bonus| step 1293 | results/task14_full_llama_e5b_1293_chunk{0,1,2,3}.json |
| Llama-SFT           | —         | results/task14_full_llama_sft_chunk{0,1,2,3}.json |

### Phase 7 variants (39B prefix ≡ 7B experiments numbered into 39x series)
| Tag | Purpose | Key ckpts | Full-test JSON |
|---|---|---|---|
| 39A format-only | control for format reward | step 500 | — |
| 39B KL=0.25 (stable) | stable high-KL curve | steps 300/400/500 | phase7/39b_step{300,400,500}_full_test.json |
| 39C1 SFT-replay c1-c2 | SFT replay study | steps per chunk | — |
| 39C2 filtered-SFT c1-c5 | filtered SFT | steps per chunk | — |
| 39D CatB format | Category-B format reward | — | — |
| 39E CatB format enhanced | enhanced variant | — | — |
| 39F full format enhanced | full variant | — | — |
| **39G1 self-distill** | init-from-iterate ReST-EM | step 500 | phase7/39g1_step500_full_test.json |
| **39G2 ReST-EM base** | **init-from-base ReST-EM (winner variant)** | step 500 | phase7/39g2_step500_full_test.json |
| 39I oracle | oracle-gated rollouts | — | — |
| **39I-self (W1)** | I-Self collapse curve | steps 50/100/150/200/250 | phase7/39i_self_step{50,100,150,200,250}_full_test.json |

Headline finding: I-Self climbs **monotonically CvT 4.84%→9.57% from step 50→250**, then collapses (Goodhart signature).

### Phase 7 V14.1 (scaling + Search-R1 reproduction)

| Tag | Purpose | Status | Path |
|---|---|---|---|
| V14-A1 Option 1 (env build) | dedicated Search-R1 conda env | **FAILED** on ARM | `results/phase7/v14_a1_env_build_report.md` |
| V14-A1 Option 2 (patch vllm) | patch verl's vllm switcher | **FAILED** (ImportError Counter) | `results/phase7/v14_a1_option2_probe_report.md` |
| V14-A1 Option 3 (memo) | reframe: "we couldn't reproduce Search-R1 code" | done | `results/phase7/v14_a1_searchr1_reframe.md` |
| **V14-A1 E1' (Option 4)** | our verl + outcome_em_only + Search-R1 hyperparams | **GRPO step 300 + eval DONE 2026-04-22 03:19 UTC**: **EM=0.000, ContEM=0.365, F1=0.000, Tools/Q=0.00** — outcome-only reward drove model to abandon tools (confirmed during training val too — not a template bug). Paper story intact: E5b/G2 verifiable-step >> Search-R1-recipe outcome-only. | ckpt: `grpo-cwq-7b-e1prime-20260420/global_step_300`; eval JSON: `results/phase7/e1prime_step300_full_test.json` |
| V14-B1/B2/B3/B4 | failure modes / decision boundary / I-Self curve / G2-vs-39B | done | `results/phase7/v14_b{1,2,3,4}_*.md` |
| V14-D1 Qwen2.5-14B E5b | framework prediction test (not a scaling win) | GRPO ended at step 400 (timeout); partial full-3531 eval 3100/3531 samples EM=0.416, ContEM=0.466, F1=0.429, Tools/Q=4.0. **Framing:** +3.2pp over 7B-39B@400 (38.4% → 41.6%) is log-linear parametric scaling; framework predicts interface bottlenecks are capacity-invariant — CvT delta will confirm or refute. CvT classification on 500 seed=42 samples: job 4247718 (P0). Top-up full-3531: job 4247719 (P0). | ckpt: `grpo-cwq-14b-e5b-2node-20260421/global_step_400/`; merged: `outputs/verl-sft-cwq-14b-v14_d1_qwen14b_e5b_step400-merged` |
| V14-D2 Qwen3-4B-Instruct-2507 E5b | smaller scale | **FAILED (chat-template drift)** — tool format conflict post-merge | `grpo-cwq-qwen3-4b-e5b-20260420/`; eval JSON exists but tools=0: `results/phase7/v14_d2_qwen3_4b_e5b_step500_full_test.json` |

---

## 3. Completed diagnostic / baseline tasks

| Task | What | Output |
|---|---|---|
| Task 14 | Full-3531 eval with bootstrap CIs + McNemar (Qwen + Llama) | results/task14_full_*.json, results/task14_summary.json |
| Task 16 | 7-category trajectory classifier | scripts/task16_classify.py |
| Task 17 | Pass@k on val-100 | results/task17_pass_at_k.json |
| Task 26 | Category B identification | results/task26_category_b.json (2,556 IDs) |
| Task 33 | CWQ temporal vs intrinsic audit | data/freebase_volatility.tsv, results/task33_*.json |
| Task 34 | Contamination audit (Tier A) | results/task34_*.json |
| Task 35 | KGQAGen-10k triangulation | results/kgqagen_*.json, ρ=0.976 with CWQ |
| Task 36 | KG Coverage Oracle | results/oracle/task36_coverage.json + .md (Category A=80% SOLVABLE) |
| Task 37 | Trajectory classification on test split (200-sample) | results/task37_*.json |
| Task 38 | Pass@k on test split | results/task38_pass_at_k_test.json |
| Task 41 | 39B mechanistic probe | results/phase7/task41_39b_mechanistic.md |
| II1 | Full-test CvT audit (Wilson CI + McNemar) | results/phase7/full_test_cvt_audit.json + .md |
| II5 | GPT-4o baseline on CWQ-50 (fuzzy entity fix) | results/phase7/gpt4o_baseline_fuzzy_50q.json + .md |
| KGQAGen Oracle | secondary-benchmark oracle | results/phase7/kgqagen_oracle.md |

---

## 4. Known failures (do not retry without fix)

- **V14-D2 Qwen3-4B E5b** — eval shows Tools/Q=0.00 after FSDP→HF merge despite Tools/Q=1.0 during training val. Root cause: chat-template drift between verl internal (kg_search format) and post-merge HF templating. Flagged in PAPER_HANDOFF §9.2. **Do not re-run 4B-family without template fix.**
- **Single-node V14-D1** (jobs 4027464, 4105590) — OOM at vLLM `wake_up` after weight sync with TP=2/4 × gpu_mem_util 0.45-0.55. Fixed by moving to 2 nodes (4139890 running). Do not retry single-node.
- **V14-A1 Options 1 + 2** (Search-R1 env build / vllm patch) — ARM/aarch64 wheel unavailability (vllm 0.6.3, tensordict 0.5, flash-attn) + `vllm.utils.Counter` removed in vllm 0.7+. Documented in the reports above; Option 4 (E1') is the chosen workaround.

---

## 5. Key HPC pitfalls (gotchas baked into job scripts)

1. **Login-node KG server crashes login host** (loads 2.6M entities + 8.3M triples into RAM). All KG servers must run on compute nodes via SLURM.
2. **Multi-node on Isambard requires per-node KG server** — arbitrary inter-node user ports are firewalled. Ray's 6379 is whitelisted; 18xxx is not. Fix: one KG server per node, bound to `127.0.0.1`, tool config uses `http://127.0.0.1:$PORT`.
3. **Multi-node srun collision** — `ray start --block &` holds the node exclusively. Subsequent srun calls (KG launchers, health probes) need `--overlap --cpus-per-task=N` or they hang on "Requested nodes are busy".
4. **verl ray.init conflict on attach** — config hardcodes `ray_kwargs.ray_init.num_cpus: 288`; Ray rejects this when connecting to existing cluster. Fix: Hydra override `~ray_kwargs.ray_init.num_cpus` (tilde-prefix removes the key).
5. **hostname -I picks mgmt-net IP, not Slingshot fabric** — use `ip -4 -o addr show hsn0 | awk '{print $4}' | cut -d/ -f1` instead of `hostname -I | awk '{print $1}'`.
6. **24h walltime cap**: all GRPO jobs chain via `afterany` (NOT `afterok`) so the eval runs even when GRPO times out. verl's `resume_mode: auto` picks up from the latest ckpt.
7. **TMPDIR=/tmp** required on Isambard — torch.distributed fails with default tempdir.

---

## 6. Key file locations

- **Paper writing context:** `PAPER_HANDOFF.md` (300+ lines, 14 sections)
- **Paper numerical tables:** `results/phase7/paper_tables.md`
- **Task queue:** `hpc_tasks.md`
- **CLAUDE.md (project coding rules):** project root
- **Classifier for CvT:** `scripts/task16_classify.py`
- **Full-test audit script:** `scripts/phase7_ii1_classify_full_test.py`
- **KG data:** `data/freebase/kg/{entities,relations,triples}.txt`
- **CWQ verl parquet:** `data/freebase/verl_cwq/{train,val,test}.parquet` (test n=3531)
- **KGQAGen-10k:** `data/wikidata/verl_kgqagen/{dev,test}.parquet`

---

## 7. Paper deadline context

- **EMNLP 2026 deadline:** 2026-05-25
- **Day-2 readiness gate (Gate A):** reads `results/phase7/full_test_cvt_audit.md` to decide launch of Day-2 variants G/I vs pivot to pure-diagnostic framing.
- **Paper signature finding:** I-Self 9.57% CvT at step 250 beats E5b@100's 10% headline in the collapse window.

---

*Update this file when new jobs complete or fail. Fast index for the HPC agent.*
