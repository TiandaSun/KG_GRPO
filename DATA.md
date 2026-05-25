# Data Availability

This repository (code + configs + analysis scripts + paper-ready summary tables) is
sufficient to **reproduce every number in the paper's tables and figures** from the
archived artifacts below. The contribution is analysis/insight, so the high-value
shared data is the *evidence* (per-checkpoint eval results, classified trajectories),
not model weights.

## What is in this git repository

- **Code**: reward functions (`src_verl/rewards/`), KG/tool servers
  (`src_verl/kg_server/`, `src_verl/tool_server/`), training (`src_verl/training/`),
  evaluation + analysis (`scripts/`), reward unit tests (`tests/`).
- **Configs**: `configs_verl/*.yaml` (every SFT/GRPO recipe).
- **SLURM job scripts**: `scripts/*.job`.
- **Aggregate evidence (the tables/figures themselves)**:
  `results/phase7/*.md` (incl. `paper_tables.md`, `full_test_cvt_audit.md`),
  `results/phase7/full_test_cvt_audit.json`, the `v14_b{1,2,4}` mechanism memos,
  `results/oracle/`, the Goodhart-curve JSONs, `_handoff/**/*.md`,
  `_handoff/data/mode4/*.csv`.
- **Documentation**: `PAPER_HANDOFF.md`, `EXPERIMENT_LEDGER.md`, `STATUS_v17_v21.md`,
  `hpc_tasks.md`, `hpc_implementation_spec_v3.md`.

## Bulk artifacts — archived on Zenodo (NOT in git)

The following are **deposited on Zenodo** — **DOI: [10.5281/zenodo.20380101](https://doi.org/10.5281/zenodo.20380101)**
(record `kg_grpo_evidence_v1.tar.gz`, ~550 MB raw / 59 MB packed; _currently a draft pending
publish_) — because they are the raw material a reviewer re-runs the classifier on:

- Full per-checkpoint eval JSONs: `results/phase7/*_full_test.json`,
  `_handoff/data/*/...json`.
- 7-category **classified trajectory dumps** + per-sample classifications
  (`results/trajectories/`, `results/**/*_per_sample/`).
- pass@k / self-consistency JSONs, mode-4 raw decomposition,
  token-entropy time series (`results/phase7/v15_q3_entropy/`).
- The rule-based SFT trajectory corpus.

> For the anonymous-review submission, link the **Zenodo anonymous review URL** (not a
> Dropbox/Drive link, which are disallowed) and an **anonymous.4open.science** mirror of
> this repo — see `STATUS_v17_v21.md`.

## Source datasets (derived data NOT redistributed here)

These are licensed third-party resources. We ship the *processing/conversion scripts*,
not the bulk derived data; regenerate locally from the originals:

| Resource | Source | Used for |
|---|---|---|
| ComplexWebQuestions (CWQ) | Talmor & Berant 2018 | QA train/test splits |
| Freebase subgraph | RoG / Reasoning-on-Graphs (Luo et al.) release | KG server (`data/freebase/kg/`) |
| KGQAGen-10k | KGQAGen release | cross-benchmark Oracle |

**Regeneration** (after obtaining the sources above):
- KG server data → `data/freebase/kg/{entities,relations,triples}.txt` from the RoG dump.
- verl parquets → `scripts/` conversion (`data/freebase/verl_cwq/{train,test}.parquet`).
- Rule-based SFT corpus → `scripts/generate_cwq_sft.py` (no LLM teacher).

## Model checkpoints

Not released in git (regenerable via the SFT + GRPO recipes in `configs_verl/` +
`scripts/`). Merged HF checkpoints live on cluster scratch
(`/projects/u6gg/KG_GRPO/checkpoints/`); available on request.
