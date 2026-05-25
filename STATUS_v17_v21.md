# KG-GRPO — work status (v17 → v21), 2026-05-25

Snapshot of HPC-agent work since the last commit (`2d96f0e`). For the commit chat.

## What landed (verdicts)

### v17 — Llama-3.1-8B cross-family replication ✅ COMPLETE
- 5 Instruct SFT recipes ABORTed (format-emission 0/100; chat-template mimicking → undertraining → rare-class).
- **Pivot to Llama-3.1-8B-Base PASSED** (W17.1f gate 92/100 format-valid).
- SFT-only EM=0.284; GRPO step-100 EM=0.345 (Tools/Q collapses 0.65→0.00, parametric-memory mode).
- Root cause documented: Instruct RLHF prior resists agentic SFT (Meta's own 8B disclaimer + Search-R1 precedent).
- Deliverables: `_handoff/data/v17_llama/SUMMARY.md`, `FINDINGS.md`, `RESEARCH_LLAMA_TOOLUSE_SFT.md`, 6 gate reports.

### v19 — W18 / W19 / W20 ✅ COMPLETE (rebuttal-stage)
- **W19** (test-set dev/test split): **CONFIRMS robust** — 10/10 checkpoints within ±1.85pp Wilson CI; E5b+SelfV dev-peak step = 250 (matches paper). `_handoff/data/w19/wt19_results.md`.
- **W18** (r_retrv anti-quote cliff isolation): **CONFIRMS** — anti-quote r_retrv prevents the Mode-4 cliff in 3/3 seeds (Tools/Q held 3.0, CvT 3-5%, EM .36-.41 through step 300-400 vs baseline collapse to 0/1.0/0). Causally isolates r_retrv as the cliff driver. `_handoff/data/w19_cliff/wt18_results.md`.
- **W20** (L_sig informative-failure): **FALSIFIES-on-EM** — informative errors don't lift EM (A1 ≈0pp) and don't prevent the cliff (A2 still collapses) → L_sig is descriptive, not causal. Reinforces W18. `_handoff/data/w19_cliff/wt20_results.md` + `W18_W20_COMBINED_INTERPRETATION.md`.

### v20 — W21 ✅ COMPLETE (rebuttal-stage)
- **W21-A** (multi-seed reward ladder): **CONFIRMS (seed-robust)** — E3 Mode-2 ritual 3/3, E5b Mode-3 drift 2/3, E5b+KL stable-plateau 2/3. Caveats: e5bkl-seed3 collapsed (seed-fragile margin); e5b-seed2 no early-peak. `_handoff/data/w21/wt21_results.md` + `wt21_partA_mode_signature.md`.
- **W21-B** (self-distill pass@16): **CONFIRMS** — G2@500 tool lift +11.5pp (with-tools 0.48 vs without 0.365); closes W11.

### Mode-4 figure data (earlier ask) ✅
- `_handoff/data/mode4/mode4_reward_decomp.{csv,json}` (7 steps; step-300 per-call avgs saturate then step-400 collapse — Goodhart).
- `_handoff/data/mode4/mode4_entropy_breakdown.{csv,json}` (inside-search H 0.35→0.205, nats).

## In flight / pending

### v21 — W22 / W23 (rebuttal/camera-ready, NON-BLOCKING)
- **W22** (entity-threading L_comp oracle): 🔄 RUNNING (job 4773712, ~12h full-3531 eval). Adapts S1.2 relation-oracle → inject gold ENTITY (keep relation). Verdict pending: CONFIRMS L_comp if EM lift ≥3pp vs the relation-oracle +0.20pp null.
- **W23** (matched non-KG interface control, the #1 lever): 🔄 BUILD STARTED.
  - ✅ Piece 1/6: calculator tool-server (`src_verl/tool_server/calc_server.py`) — safe arithmetic; failures leak NL errors (the swapped-interface variable). Validated (self-test passed incl. div-by-zero in nested parens).
  - ⬜ Pieces 2-6: GSM8K→parquet, R-selfV-arithmetic reward, SFT warmup, 3-seed GRPO, cliff eval.

## Code changes for commit

### Modified (tracked)
- `scripts/eval_with_tools.py` — `--save_per_sample`, `--resume`, `--save_every`, filter_ids
- `scripts/task17_pass_at_k.py` — `--filter_ids`
- `src_verl/training/sft_multiturn.py` — `--max_steps`, `--assistant_only_loss`, `--answer_token_weight`, `--tokenizer_path`, WeightedSFTTrainer
- `src_verl/kg_server/freebase_adapter.py` + `server.py` — `--informative_errors` (W20)
- `src_verl/rewards/verl_reward.py` — `tool_type_bonus_retrieval_contrib_anti_quote` reward_type (W18)
- `scripts/merge_sft_adapter.py` — `--tokenizer_path`
- `configs_verl/grpo_cwq_7b.yaml`, `.gitignore`, `hpc_tasks.md`

### New code (commit these)
- `src_verl/tool_server/calc_server.py` + `__init__.py` (W23)
- `scripts/run_phase7_v17_w17_*.job`, `v17_*.py` (Llama cross-family)
- `scripts/run_v19_*.job`, `v19_*.py` (W18/W19/W20)
- `scripts/run_v21_*.job`, `v21_*.py`, `task_w22_entity_oracle.py` (W21/W22/W23)
- `scripts/task_s04_reward_decomp.py`, `compute_reward_decomp_mode4.py`, `q2_mode4_mechanism_audit.py` (mode-4)
- Plus a large backlog of earlier untracked scripts (phase7_*, run_grpo_cwq_*, variant_g_*) from prior sessions.

### DO NOT commit (large / regenerable)
- `_handoff/` is **115 MB** — commit ONLY the `.md` deliverables, gitignore the `.json` + `*/trajectories/` + `*_per_sample/`.
- `results/`, `outputs/`, `checkpoints/`, `logs/`, `*.pt`, `*.safetensors` — already gitignored except `results/` and `_handoff/` (add them).

### Suggested .gitignore additions
```
results/
_handoff/**/*.json
_handoff/**/trajectories/
_handoff/**/*_per_sample/
_handoff/**/*.png
```
(keeps `_handoff/**/*.md` + `_handoff/data/mode4/*.csv` trackable)
