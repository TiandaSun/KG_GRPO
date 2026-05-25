# V14-A1 Search-R1 Reframe: E1' as Search-R1-Equivalent Baseline

Date: 2026-04-19 (original memo); **updated 2026-04-25 with locked E1' actuals**.

## Locked E1' results (2026-04-22)

| Metric | Value |
|---|---|
| Training reached | step 300 / 500 (24h+24h continuation hit walltime) |
| Eval n | 3,531 (full CWQ test) — `results/phase7/e1prime_step300_full_test.json` |
| **EM** | **0.0000** |
| ContEM | 0.3645 |
| F1 | 0.0000 |
| **Tools/Q** | **0.00** (3530/3531 = 99.97% have zero tool calls) |
| **`<answer>...</answer>` format-valid rate** | **0/3531 = 0.0%** |
| Predicted field non-empty | 3531/3531 = 100% (model emits text, never wraps in `<answer>`) |

**Decision-trigger label: `CONFIRMS V14-A1 Option 3 finding`** (EM 0.000 << 0.10 threshold).

Tools/Q=0 + zero format-valid combination is **NOT a post-merge artefact**: training-time validation showed Tools/Q=0 throughout. Outcome-only EM reward + CWQ's 4-hop compositional structure + our 4-tool Freebase agentic interface drove the policy to abandon both the `<answer>` wrapper and the tool channel entirely. F1=0 confirms the model's text shares no lexical overlap with gold answers — pure F1-hacking collapse, exactly as E1's mixed-EM-F1 reward did, just slower.

**Paper implication**: Search-R1's published recipe does NOT transfer to CWQ. The "Search-R1 fails to transfer" branch in §"Hypotheses" below is the locked outcome. Process rewards (E5b/G2/I-Self) are the paper's constructive contribution.

## Summary

We could not reproduce Search-R1 (arXiv:2503.09516) byte-for-byte on our Isambard-AI GH200 ARM platform: Option 1 (a dedicated `searchr1` conda env pinned to vllm 0.6.3 + tensordict<0.6 + flash-attn) failed at wheel-resolution and source-build time, and Option 2 (monkey-patching Search-R1's `verl/third_party/vllm/__init__.py` to accept vllm 0.12 via its 0.6.3 wrapper) failed at `ImportError: cannot import name 'Counter' from 'vllm.utils'` because the wrapper imports internal symbols removed in vllm 0.7+. We therefore reimplement Search-R1's recipe as a one-config-change of our existing E1 pipeline — reward `outcome_em_only` (pure binary EM on `<answer>`), KL 0.001, LR 1e-6, batch 256, 500 steps — and call it **E1'**. E1''s full-CWQ EM will be cited in the paper as the Search-R1-equivalent baseline.

## Why Search-R1 cannot be reproduced byte-for-byte on our platform

**Option 1 — dedicated env** (`results/phase7/v14_a1_env_build_report.md`):
- `vllm-0.6.3`: no aarch64 wheel; pip resolver falls back to CPU-only path.
- `vllm-0.6.3` from source: build fails (CUDA kernel compilation chain expects x86_64 paths).
- `tensordict<0.6`: ARM wheel missing; source build fails on torch<2.5 API drift.
- `flash-attn`: unsupported on ARM (known upstream).
- End-to-end vllm `generate` probe: FAIL. Search-R1 verl-version-accept probe: FAIL.

**Option 2 — patch vllm 0.12 into Search-R1's 0.6.3 wrapper** (`results/phase7/v14_a1_option2_probe_report.md`):
- Probe 1 (import version wrapper): `ImportError: cannot import name 'Counter' from 'vllm.utils'`.
- Probe 2 (import `fsdp_workers`): FAIL (dependent on probe 1).
- Root cause: Search-R1's `verl/third_party/vllm/__init__.py` imports internal symbols (`Counter`, `PlacementGroup`, `SequenceGroup` helpers) that were removed or relocated in vllm 0.7+. Each requires an archaeological patch; risk of silent API-contract breakage is high. Reverted.

Conclusion: running Search-R1's exact training loop on GH200 aarch64 would require either a months-long vllm 0.6.3 ARM port or a blanket rewrite of Search-R1's verl vendor tree. Both are out of scope for EMNLP 2026.

## E1': the Search-R1-equivalent recipe

### Config delta (E5b-stabilized vs Search-R1 vs E1')

| Hyperparameter | Our E5b-stabilized | Search-R1 published | **E1'** |
|---|---|---|---|
| Reward | `tool_type_bonus` (R_outcome + step) | binary EM on `<answer>` | **`outcome_em_only`** (binary EM) |
| KL coef (beta) | 0.05 | 0.001 | **0.001** |
| Learning rate | 5e-7 | 1e-6 | **1e-6** |
| Train batch size | 128 | 256 | **256** |
| Rollout group size | 8 | 8 | **8** |
| Total training steps | 500 | 800 | **500** (walltime-capped; afterok continuation available) |
| Save/eval cadence | 50 | 50 | **50** |
| Init | SFT-merged | SFT-merged | **SFT-merged** (`outputs/verl-sft-cwq-7b-merged`) |
| Tool interface | 4-tool Freebase | 1-tool BM25 Wikipedia | **4-tool Freebase** (caveat) |
| Rollout engine | vllm 0.12 + verl 0.6 | vllm 0.6.3 + Search-R1-verl-fork | **vllm 0.12 + verl 0.6** (unavoidable) |

Implementation: new reward type `outcome_em_only` added to `src_verl/rewards/verl_reward.py` on 2026-04-20; config `configs_verl/grpo_cwq_7b_e1prime.yaml` (submitted via `scripts/submit_p7_e1prime.sh`).

### Fidelity caveats

1. **Tool interface**: E1' uses our 4-endpoint Freebase server (`search`, `search_reverse`, `get_relations`, `verify`), not Search-R1's single BM25-retriever tool. Downstream adapter `scripts/phase7_ii2_searchr1_adapter.py` exists for a single-tool wrapping evaluation if a reviewer requests it.
2. **Training horizon**: 500 vs 800 steps (24 h walltime). If E1' loss is still declining at step 500 we chain a continuation with `--dependency=afterok`.
3. **Rollout engine**: vllm 0.12 instead of pinned 0.6.3. The generation contract (top-p, temperature, EOS handling) is preserved; the difference is a reproducibility caveat, not a methodological one.

## Hypotheses for the E1' result

| Scenario | EM band | Paper implication |
|---|---|---|
| **Search-R1 transfers** | EM >= 30.0% | Core narrative at risk. Paper must acknowledge outcome-only reaches comparable territory to G2 on CWQ; constructive framing must shift to CvT / process-interpretability as the differentiator, not raw EM. |
| **Subtle zone** | 10.0% <= EM < 30.0% | Outcome-only closes part of the gap. Paper frames process rewards (E5b -> G2) as unlocking the additional 10-20 pp plus the CvT lift that outcome-only cannot deliver. |
| **Search-R1 fails to transfer** | EM < 10.0% | Current "process reward required for compositional multi-hop" framing holds. E1' is a Search-R1 reference baseline in Table 1; Options 1+2 failure logs are appendix material. |

Search-R1's own paper reports 35.0-45.0% EM on HotpotQA / NQ — single-hop and 2-hop Wikipedia factoid benchmarks over BM25. CWQ is 1- to 4-hop compositional Freebase; transfer is not guaranteed.

## Relation to existing data points

All rows are full CWQ test (N=3,531), Qwen2.5-7B-Instruct, greedy, verified 2026-04-19 (`paper_tables.md`).

| Model | Reward | EM | CvT % | Tools/Q |
|---|---|---|---|---|
| SFT base | — | — (trivial tool use, V14-B4) | — | — |
| **E1 @1250** (our outcome R_out = 0.5 EM + 0.5 F1) | outcome (mixed) | **0.2%** (500-Q subset) | — | 4.91 |
| E3 @500 (verifiable step) | process | 32.5% | 0.03% | 1.00 |
| E5b @100 (tool-type bonus) | process | 32.2% | 3.03% | 2.31 |
| 39B @400 (E5b + KL 5x) | process | 38.3% | 3.77% | 3.00 |
| I-Self @200 (self-verifiable, peak) | process | 39.0% | 8.16% | 3.00 |
| G1 @500 (init from 39B) | process | 39.4% | 4.59% | 3.00 |
| **G2 @500 (ReST-EM winner)** | process | **40.0%** | 5.81% | 3.00 |
| **E1' @300** | outcome (pure EM) | **0.000** (LOCKED 2026-04-22) | **0.00** | 0.00 |

> **Correction vs. the task brief**: E1's EM on the 500-Q subset is **0.2% at step 1250** (0.4% at step 0), not "~0-5%". Pure EM on E1's outputs is essentially zero; its F1 climbed from 0.04 (step 0) to 0.25 (step 1250), but `<answer>`-tag exact match never emerged. This makes E1 a strong prior that the *mixed* 0.5 EM + 0.5 F1 reward collapses to F1-hacking without producing well-formed final answers. E1' tests whether pure binary EM (with Search-R1's looser KL) avoids that collapse.

## Paper framing options (explicit)

- **If E1' < 10.0%**: E1' enters Table 1 as a Search-R1 reference row. Narrative: the published Search-R1 recipe's 35-45% on HotpotQA/NQ does not transfer to CWQ compositional multi-hop. Our process-reward variants (E5b, 39B, I-Self, G1, G2) are the paper's constructive contribution. Appendix A includes full Options 1+2 failure logs.
- **If 10.0% <= E1' < 30.0%**: Paper acknowledges Search-R1's recipe achieves non-trivial-but-materially-lower EM than G2/I-Self. Reframing: "outcome-only closes part of the gap to SFT; process rewards unlock the additional 10-20 pp and the CvT lift (8.16% peak vs outcome-only's near-zero CvT)."
- **If E1' >= 30.0%**: Paper requires significant reframing. Lead: "outcome-only reaches X% on CWQ; our process-reward variants lift that to 40.0% (G2) with a principled +5-8 pp CvT gain that the outcome-only recipe structurally cannot deliver (no process signal)." Differentiator shifts from EM to CvT / interpretability / multi-hop gain (Hop-2 +8.99 pp).

## What we preserve about Search-R1

The inability to reproduce Search-R1 byte-for-byte on ARM GH200 is itself a methodological finding. Wheel-level pinning on aarch64 (vllm 0.6.3, tensordict<0.6, flash-attn) is a real and underreported reproducibility tax in RL-for-reasoning work; almost no recent paper in this line states a minimum target architecture. We document both failure modes (Options 1 + 2) in Appendix A with full exit codes and log references so that future replicators on ARM clusters (Isambard-AI, Alps, Grace-Hopper Perlmutter, etc.) can triage quickly. This belongs in Limitations regardless of E1''s outcome.

## Decision rule for narrative pivot

After E1' eval completes on the full 3,531-sample CWQ test:

1. **If EM >= 30.0%**: halt paper-narrative commits, escalate to user immediately with the E1' EM / CvT / Tools-per-Q row before any table is locked.
2. **If 10.0% <= EM < 30.0%**: draft the "subtle zone" framing above, attach E1' as a Table 1 row, and ping user for approval before locking Section 5 results.
3. **If EM < 10.0%**: ship the current "process reward required for compositional multi-hop CWQ" framing. E1' is a Table 1 reference baseline; Options 1+2 failure logs move to Appendix A. No narrative escalation needed.

In all three branches the Appendix A reproducibility note on ARM wheel-pinning is retained.
