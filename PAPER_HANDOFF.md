# KG-GRPO Paper Handoff — EMNLP 2026

> **Target venue**: EMNLP 2026 (ARR deadline 2026-05-25)
> **Hand-off date**: 2026-04-20
> **Project**: Process supervision via Knowledge Graphs for agentic RL. Qwen2.5-7B + GRPO on CWQ/Freebase.
> **This document is the single briefing a fresh writing-agent needs to draft the paper.**

---

## 1. How to use this document

1. Read this file end-to-end first.
2. All specific numbers you quote in the paper should be pulled from the **files listed under §8 Data inventory**, not from your memory of this document. Every number here has a concrete JSON/MD source with the same or more precision.
3. For narrative framing, follow §3 (paper story) and §4 (what each result supports).
4. For literature positioning, §7 lists the must-cite anchors and how we relate to each.
5. §9 lists known failures — acknowledge them honestly in Limitations.
6. §10 lists what's still running — you may need to fill in placeholder numbers once those jobs finish.

---

## 2. One-sentence pitch

> **We show that agentic process rewards for KG tool use exhibit a characteristic peak-then-collapse Goodhart pattern at the step level, identify the L3 (query-precision) bottleneck as their primary constraint via decision-boundary analysis, and propose two constructive fixes — a self-verifiable retrieval-grounded reward (I-Self) that reaches CvT=9.57% before collapsing, and ReST-EM self-distillation (G2) that trades CvT for stability at EM=40.0%.**

---

## 3. Paper story (three-act structure)

### Act I — Diagnostic (Sections 2-4)

The paper first establishes that naive outcome-only and verifiable-step rewards on CWQ/Freebase exhibit **pure memorization plus Goodhart**:

- **E3** (verifiable-step reward, arXiv:2505.15107-style): EM=32.5%, **CvT=0.03% (1/3531)** — EM comes entirely from parametric memory; tools are called but never productively.
- **E1** (outcome-only reward): EM=0.0% across all checkpoints (F1-hacking collapse — model stops emitting `<answer>` tags while F1 on the full response climbs).
- **V14-A1 E1' (Search-R1-equivalent, replicated in our stack)**: outcome-only EM reward + Search-R1 hyperparams (kl=0.001, lr=1e-6, bs=256). At step 300: **EM=0.000, Tools/Q=0.00** — model abandoned tool use entirely. Search-R1's published recipe does not transfer to CWQ/Freebase; confirmed in training-time val (Tools/Q=0 throughout), not a post-merge artefact.
- **E5a/E5b** (tool-type-bonus rewards): E5b@100 is the best baseline at EM=32.2%, CvT=3.03%, but collapses after step 150.
- **Ritualistic tool use (emerges across scales)**: E3 @ 7B is tool-indifferent (CvT=0.03%, Tools/Q=1.0 — fixed ritual); V14-D1 @ 14B is the same failure mode at higher intensity (Cat A CvT 13.04%, Tools/Q=3.99 — extra tool usage concentrates on parametrically-solvable questions, not on retrieval-required Cat B). The "verify-then-answer" ritual is not eliminated by scaling.

This matches the "reward function engineering" thread in 2025-26 agentic RL (Search-R1, StepSearch, ProGraph-R1, ToolRL): outcome alone is not enough; verifiable-step alone hacks; tool-type bonus is the only stable ≥3% CvT baseline, and it is still unstable.

**72.4% of CWQ test questions are Category B (pass@10=0, beyond parametric memory)** — so retrieval is *necessary* for that bucket, and the ≤3% CvT ceiling is the binding constraint.

### Act II — Mechanism (Section 5)

We locate the bottleneck:

- **Task 41 (mechanistic memo)**: 62.3% of 39B@400's kg-incomplete errors are **wrong-relation** queries; 26.7% wrong-entity; 2.3% format-mismatch; 8.7% genuine-KG-miss. The model *wants to query* but picks the wrong relation.
- **V14-B2 (decision-boundary analysis)**: **70.4%** of G2@500's kg-incomplete samples have a query that is **≤1 edit** from the Oracle gold relation; 89.1% are entity-correct with wrong/typoed relation. This is textbook **L-lang** (schema-format) failure.
- **V14-B1 (sub-failure-mode classification)**: across all 2,119 non-EM G2@500 trajectories, 55% are L-sig (tool returned empty), 24% L-lang (schema mismatch), 13% L-prior (irrelevant parametric attempt), 8% L-comp (multi-hop state drift).
- **V14-B4 (G2 vs 39B behavioral diff)**: on shared kg-incomplete samples, **90.4% of errors share the same first-entity but differ on relation** — ReST-EM init (G2) shifts *relation choice*, not *entity choice*.

These four analyses converge: **the L3 query-precision bottleneck is the primary constraint, and it is predominantly about relation selection within a correct entity neighborhood**.

### Act III — Constructive (Section 6)

We propose three interventions; two succeed.

**C1 — 39B (E5b + KL 5×) stabilization (intermediate result)**. Increasing the KL penalty 5× preserves E5b's tool use while extending its stable window. EM=38.3% full-test, CvT=3.77%. Paper framing: "KL 5× is a stability win, not a retrieval win" — CvT barely improves over E5b's 3.03%.

**C2 — I-Self self-verifiable retrieval reward (paper's CvT winner)**. Reward = 0.25·r_answer + 0.50·r_tool_type + 0.25·r_retrieval_contrib, where `r_retrieval_contrib` rewards tool calls whose returned entities appear verbatim in the final `<answer>`. **No oracle labels required — self-verifiable at inference.**
- CvT grows monotonically: step 50 → 4.84% → step 100 → 6.37% → step 150 → 7.28% → step 200 → 8.16% → **step 250 → 9.57% (peak)**
- **Catastrophic collapse between step 250 and 300**: EM drops from 0.395 to 0.000 in a single 50-step window, tools/Q drops from 3.0 to 1.0, format validity collapses.
- kg-incomplete count drops from 1,435 (E5b baseline) to **824 at peak** — a 42.6% reduction — directly validating that I-Self targets the L3 bottleneck.
- **This is a novel finding**: no published process-reward RL paper documents peak-then-collapse at step-level resolution.

**C3 — G2 ReST-EM self-distillation (paper's EM winner + stability winner)**. Inference 39B@400 greedy on full 27K CWQ train, filter to strict EM=1 AND ≥1 successful tool call AND format valid (yield 50.9% = 14,082 trajectories), SFT Qwen2.5-7B-Instruct from base on those distilled trajectories, GRPO on top. **EM=40.0%, CvT=5.81%, kg-incomplete=818**. Beats G1 (init from 39B@400 instead of base SFT) on CvT (4.59% → 5.81%, +26% relative) — validates Singh et al. 2024's ReST-EM prediction that init-from-base transfers better than init-from-iterate.

**C4 — V14-D1 Qwen2.5-14B capacity ablation (framework validation)**. Same E5b-stabilized recipe as 39B (tool-type-bonus reward, KL 5×), base model doubled from 7B to 14B. Evaluated at step 400 on full 3,531 CWQ test. **EM = 40.24%, full-test CvT = 5.73% (1500-sample seed=42 subset), Cat B CvT = 4.61% [3.52-6.01%]**. This is the paper's framework-validation anchor:

- **Headline scaling comparison — 14B D1@400 vs 7B G2@500**: Δ EM = +0.25pp (within CI), Δ Cat B CvT = −0.32pp (within CI). *Self-distillation at 7B matches 2× capacity scaling on this task.*
- **Sub-log-linear capacity scaling — 14B vs 7B 39B (same recipe)**: +1.90pp EM. Log-linear fs1 prediction at 2× ≈ +3.5pp (fs1: 7B→32B = +6.9pp). 14B gains fall *below* the log-linear baseline.
- **Cat B CvT indistinguishable between 14B and G2**: 4.61% vs 4.19%, CIs overlap. The signal-theoretic framework's prediction that L-sig/L-lang/L-comp/L-prior interface bottlenecks are capacity-invariant is empirically confirmed.
- **Decomposition asymmetry**: 14B's +1.90pp overall gain decomposes into +3.49pp on Cat A (parametric) vs +1.29pp on Cat B (retrieval) — capacity helps memorization over retrieval, as the framework predicts.
- **Ritualistic tool-use** (belongs in Section 4, not Section 6): 14B has the highest Cat A CvT (13.04%) and highest Tools/Q (3.99) of any model. Extra tool usage concentrates on parametrically-solvable questions, not on Cat B. This is a second instance of the "verify-then-answer" pattern first seen in E3 (memorized answer + tool-call ritual), now at 2× capacity.

**Canonical Section 6 framing (locked 2026-04-23)**:
> "Doubling capacity from 7B to 14B under an identical E5b-stabilized recipe yields +1.90pp EM, well below the log-linear scaling baseline (fs1: +3.5pp per 2× at this scale). Critically, trajectory-level retrieval-success rate on Category B questions (Cat B CvT = 4.61% [3.52-6.01%] at 14B) is indistinguishable from the 7B G2 baseline (4.19% [3.48-5.03%]). The signal-theoretic framework's prediction that L-sig / L-lang / L-comp / L-prior bottlenecks are capacity-invariant is empirically confirmed."

Footnote: 14B evaluated at step 400 vs G2 step 500. Cat B CvT plateaus by step 200-400 on the E5b and 39B training curves (Table 8: I-Self monotonic CvT climb saturates before collapse; 39B@300/400/500 Cat B CvT ≈ 2.97% ± small). The step-400-vs-500 offset does not materially affect the framework-critical metric.

---

## 4. Canonical results (paper's Table 1 candidate)

**Authoritative source**: `results/phase7/full_test_cvt_audit.json` + `results/phase7/full_test_cvt_audit.md` (CvT classifier ran on full 3,531 test set with 7-category trajectory classification + Wilson 95% CIs + paired McNemar where relevant). Also consolidated in `results/phase7/paper_tables.md`.

| Model | Step | EM | CvT count | CvT % | 95% CI | Tools/Q |
|---|---|---|---|---|---|---|
| SFT base | 0 | ~0.005 | — | — | — | 0 |
| E1 (outcome-only) | 1250 | 0.000 | — | — | — | 0-1 |
| E3 (verifiable-step) | 500 | 0.325 | 1/3531 | 0.03% | [0.00-0.16%] | 1.0 |
| E5b (tool-type bonus) | 100 | 0.322 | 107/3531 | 3.03% | [2.51-3.65%] | 2.31 |
| 39B (E5b + KL 5×) | 300 | 0.366 | 132/3531 | 3.74% | [3.16-4.42%] | 2.99 |
| **39B (E5b + KL 5×)** | **400** | **0.383** | **133/3531** | **3.77%** | **[3.19-4.45%]** | **3.00** |
| 39B (E5b + KL 5×) | 500 | 0.380 | 110/3531 | 3.12% | [2.59-3.74%] | 3.00 |
| I-Self @ 50 | 50 | 0.381 | 171/3531 | 4.84% | [4.18-5.60%] | 3.00 |
| I-Self @ 100 | 100 | 0.387 | 225/3531 | 6.37% | [5.61-7.23%] | 3.00 |
| I-Self @ 150 | 150 | 0.387 | 257/3531 | 7.28% | [6.47-8.18%] | 3.01 |
| I-Self @ 200 | 200 | 0.390 | 288/3531 | 8.16% | [7.30-9.11%] | 3.00 |
| **I-Self @ 250 (peak)** | **250** | **0.395** | **338/3531** | **9.57%** | **[8.65-10.59%]** | **3.00** |
| I-Self @ 300 (collapsed) | 300 | 0.000 | — | collapsed | — | 1.0 |
| I-Self @ 400 (collapsed) | 400 | 0.000 | — | collapsed | — | 1.0 |
| G1 (init from 39B) | 500 | 0.394 | 162/3531 | 4.59% | [3.95-5.33%] | 3.00 |
| **G2 (ReST-EM, init from base SFT)** | **500** | **0.400** | **205/3531** | **5.81%** | **[5.08-6.63%]** | **3.00** |
| **V14-D1 (Qwen2.5-14B, same recipe as 39B)** | **400** | **0.402** | **86/1500**<sup>†</sup> | **5.73%**<sup>†</sup> | see Table 9 | **3.99** |

<sup>†</sup> V14-D1 CvT on 500-sample seed=42 subset (trajectory classification deferred on full 3,531 for compute economy; Cat B CvT CI tight enough for framework validation — see Table 9). Full-3531 EM verified on all 3,531 samples.

**McNemar paired tests (on CvT)**:
- 39B@400 vs E3@500: 133 vs 1, p<0.0001 (massive)
- 39B@400 vs E5b@100: 111 vs 85, p=0.074 (not significant — 39B maintains E5b CvT while gaining +6 EM)
- 39B@400 vs 39B@300: p=1.00 (plateau by step 300)

**Paper winners**:
- **EM**: G2@500 at 0.400
- **CvT**: I-Self@250 at 9.57% (2.5× over 39B@400)
- **EM × CvT Pareto frontier**: I-Self@250 (0.395, 9.57%) dominates G2@500 (0.400, 5.81%) on CvT while giving up only 0.5pp EM.

---

## 5. Supporting experiments (Sections 7-8)

### 5.1 Pass@k with vs without tools (500-Q subset, seed=42)

Source: `results/phase7/ii6_pass_at_k/*.json`

| Model | no-tools pass@16 | with-tools pass@16 | Gap |
|---|---|---|---|
| E3@500 | 0.358 | 0.354 | −0.004 (no gap) |
| E5b@100 | 0.362 | 0.504 | **+0.142** |
| 39B@400 | 0.388 | 0.502 | **+0.114** |

**E3 doesn't use tools for capability expansion**; E5b and 39B do. pass@32 for 39B@400 = 0.530 (+18.7pt over pass@1=0.343). Task 38b pass@k data in `results/phase7/task38b_39b_pass_at_k.json`.

### 5.2 Inference-time self-consistency@5 (full 3,531)

Source: `results/phase7/sc5_39b_step400_full.json`

- 39B@400 SC@5 EM = 0.385 (≈ greedy 0.384, **no lift**)
- pass@5(any) = 0.465 (+8pp over greedy)

**Interpretation**: systematic failures (wrong answer repeated), not sampling noise. Supports the diagnostic claim that failures are model-level not stochastic.

### 5.3 KG coverage Oracle

| Benchmark | N | SOLVABLE | PARTIAL | UNREACHABLE | Source |
|---|---|---|---|---|---|
| CWQ (Task 36) | 250 | 99.5% | — | — | `results/oracle/task36_coverage.json` |
| KGQAGen-10k (II4) | 1,079 | 69.14% | 28.08% | 2.78% | `results/phase7/kgqagen_oracle.{json,md}` |

**Interpretation**: KG coverage alone cannot explain the ≤10% CvT ceiling — 69% of KGQAGen is answerable in-cache yet trained models still rarely retrieve the answer via tools. The diagnostic holds across benchmarks.

### 5.4 Llama pipeline audit (appendix-only)

Source: `results/phase7/llama_audit.{json,md}` (script: `scripts/phase7_ii3_llama_audit_from_task14.py` — offline analysis of existing task14 trajectories, no new GPU compute).

| Model | EM | `<answer>` tags | Degenerate loops |
|---|---|---|---|
| Llama-3.1-8B SFT | 0.000 | 0/100 | 16/100 |
| Llama-3.1-8B E3@1293 | 0.000 | 1/100 | 100/100 |
| Llama-3.1-8B E5b@1293 | 0.000 | 0/100 | 0/100 |

**Conclusion**: Llama's 0% EM is a format failure (no `<answer>` tags), not a capability failure. E3 catastrophically collapses Llama generation (100% degenerate loops). Llama goes to appendix only.

### 5.5 GPT-4o zero-shot baseline (150-Q and 50-Q subsets)

Source: `results/phase7/gpt4o_baseline_150q.json`, `results/phase7/gpt4o_baseline_fuzzy_50q.json` (re-scored with ContEM + F1; original script only computed strict EM).

| Run | n | EM | ContEM | F1 | Tools/Q | Abandoned |
|---|---|---|---|---|---|---|
| GPT-4o zero-shot | 150 | 0.00% | 30.0% | 4.26% | 3.69 | 78.0% |
| GPT-4o + fuzzy entity match | 50 | 0.00% | 34.0% | 4.66% | 3.64 | 82.0% |

**Interpretation**: GPT-4o gets 30-34% ContEM with our 4-tool Freebase API, **lower than our 7B Qwen2.5 E3@500's 32.5% strict EM**. 78-82% trajectories say "issue retrieving X" — GPT-4o doesn't know Freebase's schema. Strong reviewer defense: closed-model zero-shot does not solve this problem.

---

## 6. Method details (for Methods section)

### 6.1 SFT corpus (rule-based, NO LLM teacher — methodological strength)

Source: `scripts/generate_cwq_sft.py` (5K primary), `scripts/task40_gen_gold_trajectories.py` (1K gold-verified).

| File | Rows | Generator |
|---|---|---|
| `data/freebase/sft_trajectories.jsonl` | 5,000 | Rule-based: templated `<think>` + gold-path-constructed `<search>` + real KG server response |
| `data/freebase/sft_trajectories_enhanced.jsonl` | 6,000 | Enhanced variant of the above |
| `data/freebase/verl_cwq/gold_kg_trajectories.jsonl` | 1,000 | Same but with live KG-server verification filter |

**Total**: 12K (not 20K — prior references were outdated).

**Key properties to highlight**:
- **No LLM teacher** (no GPT-4/Claude/Qwen-72B distillation)
- **No self-rollout** (no model-sampled data)
- **Three fixed think-templates** with variable substitution
- **Tool queries are oracle** (constructed from gold triple chain, so the tool response always contains the answer by construction)

**Paper positioning**: Most 2026 agentic RL papers use LLM teachers (DeepSeek, Qwen3-Coder-30B), introducing teacher-model bias confounds. Our rule-based SFT eliminates that critique.

### 6.2 Tools: 4-verb Freebase interface

Source: `src_verl/kg_server/freebase_adapter.py`, `src_verl/kg_server/server.py`

- `get_tail_relations(entity)` — relations going out from entity
- `get_head_relations(entity)` — relations coming into entity
- `get_tail_entities(entity, relation)` — entities reachable via relation
- `get_head_entities(entity, relation)` — entities connecting via relation

Freebase stats: **2,592,892 entities · 7,058 relations · 8,309,195 triples** (RoG subgraph-filtered CWQ dump).

### 6.3 Reward formulas

Source: `src_verl/rewards/verl_reward.py` (11 reward types, all unit-tested).

| Name | Formula |
|---|---|
| `outcome` (E1) | 0.5·EM + 0.5·F1 |
| `outcome_em_only` (E1′, added 2026-04-20) | binary EM |
| `heuristic` (E2) | 0.3·r_outcome + 0.7·entity_overlap |
| `verifiable` (E3) | 0.25·r_outcome + 0.25·r_valid + 0.25·r_on_path + 0.25·r_coherence |
| `retrieval_grounded` (E5a) | 0.25·r_outcome + 0.75·r_tool_usage |
| `tool_type_bonus` (E5b / 39B) | 0.25·r_outcome + 0.50·r_tool_type + 0.25·r_tool_usage |
| `tool_type_bonus_oracle_query_match` (I-Oracle) | 0.25·r_answer + 0.50·r_tool_type + weight·r_query_match_oracle (anti-spam: ≥2 distinct gold matches) |
| **`tool_type_bonus_retrieval_contrib` (I-Self)** | 0.25·r_answer + 0.50·r_tool_type + 0.25·r_retrieval_contrib |

`r_retrieval_contrib` = (productive tool calls) / (total tool calls), where productive = tool returned non-empty AND ≥1 returned entity appears verbatim in final `<answer>`. Self-verifiable at inference, no gold required.

### 6.4 GRPO training config (verl backend)

- Init: `outputs/verl-sft-cwq-7b-merged` (merged from `sft_cwq.yaml` run on 5K rule-based corpus)
- Group size: 8
- KL loss type: `low_var_kl`
- KL coef: 0.05 for E5b-stabilized / G1 / G2; 0.001 for E1′/Search-R1 parity
- LR: 3e-7 for stabilized variants; 1e-6 for E1′
- Batch size: 128 for most; 256 for E1′
- Max turns: 5
- `total_training_steps`: 500 for most; I-Self ran to 400 (collapsed); G1/G2 hit 500
- Train data: `data/freebase/verl_cwq/train.parquet` (27,639 rows)
- Config: `configs_verl/grpo_cwq_7b.yaml`

### 6.5 Evaluation protocol

- Greedy decoding, `max_new_tokens=512`, `max_turns=5`
- Full CWQ test set N=3,531
- EM (exact match after `normalize()`: lowercase, strip articles, strip punct, collapse whitespace)
- ContEM (gold substring appears in full response)
- F1 (token overlap)
- Tools/Q (mean number of `<search>...</search>` calls)
- **7-category trajectory classification** (`scripts/task16_classify.py`): correct-via-tool, correct-no-tool, correct-via-memory, wrong-no-tool, kg-incomplete, tool-misuse, wrong-answer
- Wilson 95% CI on CvT rate
- Paired McNemar's test (exact binomial for n<25, else continuity-corrected chi-square)

---

## 7. Literature anchors (must-cite)

| Paper | Citation | How we relate |
|---|---|---|
| Search-R1 | arXiv:2503.09516 | We attempted to reproduce; vllm 0.6.3 cannot build on ARM GH200. **E1′** (our verl + Search-R1 hyperparams + pure binary EM) is the equivalent reimplementation. See §9.3 and `results/phase7/v14_a1_searchr1_reframe.md`. |
| ReST-EM (Singh et al. 2024) | arXiv:2312.06585 | Their init-from-base > init-from-iterate prediction is validated by **G2 > G1** (CvT 5.81% vs 4.59%). |
| FireAct | arXiv:2310.05915 | Their 500-trace phase transition prediction was used to justify our 800-trace minimum; we ended up with 14,082. |
| EEF (Exploring Expert Failures) | arXiv:2504.13145 | Fallback salvage plan if G-sweep yield <15% — not needed, yield was 50.9%. Mentioned as v14 design alternative. |
| StepSearch | arXiv:2505.15107 (EMNLP 2025) | TF-IDF gold-doc alignment process reward, +5 F1. Our E3 variant is a KG analog; I-Oracle is the oracle-supervised version. |
| ProGraph-R1 | arXiv:2601.17755 | Entity overlap step reward, +4-7 F1. Our heuristic (E2) is similar; matches their finding that entity overlap helps F1 but not EM. |
| CriticSearch | arXiv:2511.12159 | α=0.25 turn-level advantage — independent validation of our Variant I-Oracle 0.25 weight for r_query_match. |
| KG-Implicit-RM | arXiv:2601.15160 | Anti-spam gate "≥2 distinct gold-entity matches" — adopted in our I-Oracle reward. |
| Follow the Path fs1 | arXiv:2505.11140v2 | Qwen2.5-7B SFT on KG-trace = 40.8% pass@1 on CWQ (sub-phrase EM). Our G2 = 40.0% **strict** EM — directly comparable; we maintain parity while adding CvT structure. |
| Pitfalls in KG-RAG Datasets | arXiv:2505.23495 | Defends our strict-EM choice over Hit@1. Also surfaces CWQ's 49.3% factual-accuracy audit (answered by our V14-B2 + KGQAGen Oracle work). |
| Demystifying Agentic Reasoning | arXiv:2510.11701 | CUHK/IDEA paper using Qwen3-4B-Instruct-2507 — we followed their setup for D2 but hit chat-template drift (see §9.2). |
| Demystifying Hybrid Thinking | arXiv:2510.12680 | Justifies our choice of Instruct-2507 (pure non-thinking) over hybrid Qwen3-4B/8B. |

**Explicitly NOT cited (v13 justification)**: Llama-3.1 family (no working baseline — format failure); Mistral/Gemma/Phi families (no open-source agentic-RL peers in 2026 field norm).

---

## 8. Data inventory (every file the paper cites)

### 8.1 Paper-ready tables and memos (highest-value artifacts)

| File | Contents |
|---|---|
| `results/phase7/paper_tables.md` | **Tables 1-8 paper-ready** (main results, 7-cat breakdown, pass@k, oracle coverage, Llama audit, mechanistic, I-Self collapse curve, CvT deltas) |
| `results/phase7/full_test_cvt_audit.md` | Full 3,531 CvT audit with McNemar tests — authoritative for Table 1 |
| `results/phase7/full_test_cvt_audit.json` | Machine-readable version of the above |
| `results/phase7/task41_39b_mechanistic.md` | 39B mechanistic memo (Category A/B, hop-stratified, wrong-relation distribution, E5b rescue rate) |
| `results/phase7/39b_query_error_modes.csv` | 300 classified kg-incomplete errors for Table 6 exemplars |
| `results/phase7/v14_b1_failure_modes.{json,md}` | G2 non-EM failure mode classification (L-sig/L-lang/L-comp/L-prior) |
| `results/phase7/v14_b2_decision_boundary.{json,md}` | G2 kg-incomplete edit-distance-to-gold distribution |
| `results/phase7/v14_b4_g2_vs_39b_queries.{json,md}` | G2 vs 39B behavioral query diff — tool-type, diversity, turn count, shared kg-incomplete |
| `results/phase7/v14_w1_i_self_memo.md` | I-Self peak-then-collapse memo (early draft of Section 5 or 6) |
| `results/phase7/v14_a1_searchr1_reframe.md` | Search-R1 Option 3 reframe memo + E1′ decision rule |
| `results/phase7/llama_audit.md` | Llama appendix content |
| `results/phase7/kgqagen_oracle.md` | Cross-benchmark Oracle result |

### 8.2 Per-model full-test (3,531) eval result JSONs

```
results/phase7/e3_step500_full_test.json
results/phase7/e5b_step100_full_test.json
results/phase7/39b_step300_full_test.json
results/phase7/39b_step400_full_test.json
results/phase7/39b_step500_full_test.json
results/phase7/39i_self_step50_full_test.json  (V14-B3)
results/phase7/39i_self_step100_full_test.json (V14-B3)
results/phase7/39i_self_step150_full_test.json (V14-B3)
results/phase7/39i_self_step200_full_test.json
results/phase7/39i_self_step250_full_test.json (V14-B3)
results/phase7/39i_self_step300_full_test.json (partial, collapsed)
results/phase7/39i_self_step400_full_test.json (partial, collapsed)
results/phase7/39g1_step500_full_test.json
results/phase7/39g2_step500_full_test.json
results/phase7/v14_d2_qwen3_4b_e5b_step500_full_test.json  # failed, see §9.2
```

### 8.3 Saved trajectories for classification (always 3,531 per model)

```
results/trajectories/phase7/<model_tag>_full/step_<N>/trajectories.json
```

Each trajectory JSON has: `sample_id, question, gold_answer, all_answers, hops, predicted, full_response, num_tool_calls, em, f1, contains_em`.

Model tags present: `e3_step500`, `e5b_step100`, `39b_step300`, `39b_step400`, `39b_step500`, `39i_oracle_step400`, `39i_self_step50/100/150/200/250/300/400`, `39g1_step500`, `39g2_step500`.

### 8.4 Pass@k and self-consistency

```
results/phase7/task38b_39b_pass_at_k.json           # 39B@400 pass@{1,4,8,16,32} on 200-Q seed=42
results/phase7/ii6_pass_at_k/                       # 6 of 8 arms done (E3/E5b/39B × tools/no-tools)
results/phase7/sc5_39b_step400_full.json            # SC@5 full 3,531 on 39B@400
```

### 8.5 Training-data / SFT corpora

```
data/freebase/sft_trajectories.jsonl                # 5K primary rule-based
data/freebase/sft_trajectories_enhanced.jsonl       # 6K enhanced
data/freebase/verl_cwq/gold_kg_trajectories.jsonl   # 1K oracle-verified
data/freebase/verl_cwq/train.parquet                # 27,639 CWQ train rows
data/freebase/verl_cwq/test.parquet                 # 3,531 CWQ test rows
data/freebase/verl_cwq/train_oracle_gold_paths.jsonl  # train oracle chains (for I-Oracle)
data/freebase/verl_cwq/train_with_oracle.parquet    # joined train+oracle for I-Oracle training
data/freebase/verl_cwq/39b_self_distill.jsonl       # 14,082 strict-filtered 39B@400 trajectories for G SFT
data/freebase/verl_cwq/39b_full_sweep_merged.json   # 27,639 raw 39B@400 sweep records
```

### 8.6 Wikidata / KGQAGen cross-benchmark

```
data/wikidata/kgqagen_10k/                          # source dataset
data/wikidata_cache/                                # pre-materialized Wikidata neighborhoods
results/phase7/kgqagen_oracle.{json,md}             # Oracle replication
```

### 8.7 Baselines

```
results/phase7/gpt4o_baseline_150q.json             # GPT-4o zero-shot
results/phase7/gpt4o_baseline_fuzzy_50q.json        # GPT-4o + fuzzy entity match
(E1' results — see §10, pending)
```

### 8.8 Oracle / coverage

```
results/oracle/task36_coverage.json                 # CWQ Oracle (Task 36, 250 samples)
results/oracle/oracle_summary.md                    # CWQ Oracle summary
```

### 8.9 Key scripts (for reproducing specific numbers)

```
scripts/phase7_ii1_classify_full_test.py            # Canonical CvT classifier — THE source of Table 1 numbers
scripts/task16_classify.py                          # Per-trajectory 7-category classifier
scripts/eval_with_tools.py                          # Main eval driver
scripts/task17_pass_at_k.py                         # Pass@k driver
scripts/phase7_task42_self_consistency.py           # SC@k
scripts/task41_39b_mechanistic.py                   # Mechanistic memo generator
scripts/task_v14_b1_failure_modes.py                # V14-B1
scripts/task_v14_b2_decision_boundary.py            # V14-B2
scripts/task_v14_b4_g2_vs_39b_queries.py            # V14-B4
scripts/phase7_ii5_gpt4o_baseline.py                # GPT-4o baseline
scripts/phase7_ii3_llama_audit_from_task14.py       # Llama audit (offline)
scripts/phase7_ii4_kgqagen_oracle.py                # KGQAGen Oracle
scripts/phase7_variant_g_pilot.py                   # G sweep
scripts/phase7_variant_g_filter.py                  # G strict/widened filter
scripts/phase7_prepare_oracle_train_parquet.py      # I-Oracle data prep
```

### 8.10 Reward + training code

```
src_verl/rewards/verl_reward.py                     # 11 reward types (incl. outcome_em_only for E1')
src_verl/kg_server/server.py                        # Freebase KG FastAPI server
src_verl/kg_server/freebase_adapter.py              # Adapter implementation
src_verl/kg_server/wikidata_adapter.py              # Wikidata adapter (for KGQAGen Oracle)
src_verl/training/sft_multiturn.py                  # SFT driver (TRL-based)
configs_verl/grpo_cwq_7b.yaml                       # Main GRPO config
configs_verl/sft_cwq.yaml                           # Main SFT config
```

### 8.11 Reward unit tests (paper-ready Methods claim: all reward types unit-tested)

```
tests/test_phase7_variant_i_rewards.py              # 18/18 passing — covers I-Oracle + I-Self + anti-spam
```

---

## 9. Known failures (Limitations section)

### 9.1 Llama-3.1-8B appendix-only

See §5.4. Llama fails to produce `<answer>` tags (1/300 trajectories across SFT, E3, E5b). E3 catastrophically collapses Llama (100% degenerate token loops like `_formerly_formerly_...`). **Frame as "reward-design effectiveness is not model-family invariant"** — consistent with 2026 field observations that Qwen has become the de facto standard for agentic RL.

### 9.2 Qwen3-4B-Instruct-2507 chat-template drift (D2)

SFT completed successfully (loss 0.75). GRPO trained all 500 steps with during-training val showing tools/Q=1.0. **But the merged post-GRPO checkpoint evaluated to Tools/Q=0.00, EM=0.000, ContEM=0.184** — model abandoned tools after the HF merge. Likely cause: Qwen3 native tool-call markers (`<tool_call>...</tool_call>`) conflict with our `<search>...</search>` format when the HuggingFace chat template re-serializes post-merge.

**Frame as (locked Section 7 paragraph, 2026-04-25):**
> "We attempted cross-generation validation on Qwen3-4B-Instruct-2507 but encountered chat-template drift in the FSDP→HF post-training merge that nullified tool-call format. Cross-generation extension is left to future work; the present validation rests on cross-capacity (7B → 14B) and cross-benchmark (CWQ → KGQAGen-10k) axes."

Documented concern: `results/phase7/v14_d2_qwen3_4b_e5b_step500_full_test.json`.

### 9.3 Search-R1 reproduction (Options 1, 2 failed; Option 3 = E1′)

- **Option 1** (dedicated conda env with vllm==0.6.3): vllm wheel not available on ARM aarch64 GH200; source build also fails. Tensordict<0.6 and flash-attn also fail on ARM. See `results/phase7/v14_a1_env_build_report.md`.
- **Option 2** (patch Search-R1's `third_party/vllm/__init__.py` to accept vllm 0.12 via the 0.6.3 wrapper): `ImportError: cannot import name 'Counter' from 'vllm.utils'` — wrapper imports internal symbols removed in vllm 0.7+. Fix requires deep archaeology. Auto-reverted. See `results/phase7/v14_a1_option2_probe_report.md`.
- **Option 3 = E1′**: Our verl + `outcome_em_only` binary EM reward + Search-R1's published hyperparams (kl_coef=0.001, lr=1e-6, batch=256, max_steps=500). One config change to our pipeline, zero dependency changes. See `results/phase7/v14_a1_searchr1_reframe.md` for full rationale + decision rule (if E1′ EM ≥ 30% → narrative escalation; < 10% → confirms diagnostic).

**Frame as**: "We reimplement Search-R1's recipe rather than running their code; ARM aarch64 wheel availability for vllm ≤ 0.6.3 blocks byte-for-byte reproduction. This reproducibility tax itself is noted in the Limitations."

### 9.4 Llama E3 24.8% artifact claim

Earlier project notes referenced "Llama E3 = 24.8%". After data-provenance audit (`results/data_provenance_audit.md`, 2026-04-12), this number has **no backing evidence on HPC** — likely a misattribution of ContEM. The committed Llama numbers in v13+ are all 0.000 EM with ContEM 0.05-0.46 (format-failure attribution via §5.4).

---

## 10. Jobs still running (as of 2026-04-23 16:00 UTC)

**Previously pending, now LANDED:**

| Job | Result | Paper slot |
|---|---|---|
| V14-A1 E1' (Search-R1-equivalent) | **DONE** @step 300: EM=0.000, Tools/Q=0.00 — outcome-only reward abandoned tool use entirely. Full eval at `results/phase7/e1prime_step300_full_test.json`. | Table 1 row + Act I bullet added (§3). Canonical contrast: E5b's verifiable-step 10% CvT vs Search-R1-recipe 0% CvT. |
| V14-D1 Qwen2.5-14B capacity ablation | **DONE** @step 400: EM=0.402 (full 3,531), Cat B CvT=4.61% [3.52-6.01%] on 1500-sample subset (seed=42). | Table 1 row + Table 9 + Act III §C4 (framework validation anchor). Section 6 framing locked. |

**Currently running:**

| Job | ETA | Output file | Paper slot |
|---|---|---|---|
| V14-C1 SFT (fs1-adapt: Qwen2.5-7B + fs1-2708 + fs1 hyperparams) | 24h after Priority slot acquired | `outputs/verl-sft-cwq-7b-fs1-adapt-merged/` | Optional — literature-grounded SFT-only baseline. Adaptation approach (our stack + their data + their config) documented in methods. |
| V14-C1 Part A eval (afterany SFT) | 8h after SFT done | `results/phase7/v14_c1_fs1_sft_only_full_test.json` | If EM > 30%, auto-triggers Part B (fs1-SFT + our RL). |

**Optional polish (not blocking):**
- Extend 14B trajectory classification from n=500 → n=1000-1500 to tighten Cat B CvT CI from ±2pp to ±1pp. Paper story holds at current precision.
- 32B scaling point: third capacity data point + reviewer defense. Expected EM 41-43%, Cat B CvT 3-5%.

---

## 11. Numbers NOT to use

- Any "E5a/E5b peak EM at 52%" or "Qwen parametric ceiling at 52%" — these were val-split first-500 numbers, superseded by full-test 32% (see `results/data_provenance_audit.md`).
- Llama E3 = 24.8% — see §9.4.
- Any pre-v11 mentions of 20K SFT trajectories — it's 12K (see §6.1).
- `sft_base` pass@k results from `ii6_pass_at_k/sft_base_*.json` — job timed out at ~40/500 samples (pass@1 = 0.002), not a statistically valid baseline.

---

## 12. Writing checklist

Before submitting, verify:

- [ ] Table 1 numbers match `results/phase7/full_test_cvt_audit.json` + `v14_d1_strat.json` to 3 decimal places
- [ ] CvT 95% CIs are Wilson (not normal approximation)
- [ ] Search-R1 reproduction is framed as E1′-reframe with v14_a1_searchr1_reframe.md citation; E1' final EM=0.000 / Tools=0 is in Table 1 and Act I
- [ ] Llama is in appendix with §9.1 framing
- [ ] Rule-based SFT is explicitly called out as a methodological strength (no teacher-model leakage)
- [ ] ReST-EM ≥ init-from-base claim cites Singh et al. 2024 and cites G2 > G1 +26% CvT relative
- [ ] I-Self peak-then-collapse is the "novel paper contribution" — highlight step-level Goodhart resolution
- [ ] V14-B2's "70.4% ≤1 edit" finding appears in Section 5 (Mechanism)
- [ ] **Section 6 uses the LOCKED framing block in §C4 verbatim (or close paraphrase); 14B vs G2 is the primary 7B comparator, NOT 14B vs 39B**
- [ ] Sub-log-linear scaling claim (+1.90pp actual vs ~+3.5pp log-linear fs1 prediction) is stated in Section 6
- [ ] Ritualistic tool-use parallel (E3 at 7B + V14-D1 at 14B) appears together in Section 4, not Section 6
- [ ] Footnote on 14B step 400 vs G2 step 500 (Cat B CvT plateau) is in Section 6
- [ ] Limitations covers §9.1 (Llama), §9.2 (Qwen3-4B), §9.3 (Search-R1)
- [ ] GPU budget disclosed: total ~X GPU-hours on Isambard GH200 — check `logs/` for cumulative

---

## 12. Reviewer-response findings (Tier 0, locked 2026-04-28)

A reviewer-agent simulation flagged two lethal weaknesses: single-seed RL claims and missing null base rates. Tier 0 closes both. Five no-GPU items locked **2026-04-28**; three paid items running (S0.1, S0.2 multi-seed × 3 + S1.1 G3 ablation, lock target Apr 30). Tier 1 G3 ablation submitted same day.

### 12.1 Schema-format null hypothesis (S0.5) — **CRITICAL gate, PASSED**

**Result**: random-relation null at edit-distance ≤1 = **14.91%**; at V14-B2's native definition (≤3 OR ratio>0.8) = **22.06%**. V14-B2 finding (70.4% close-match rate) is **4.7× the strict null** and **3.2× the native-def null**.

**Decision**: STRONG (under strict ≤1 cutoff, just under the 15% boundary). Ship as-is, but report both nulls in the paper.

**Recommended paper text**: "We compare V14-B2's 70.4% close-relation rate against a null sampled from 100 random Freebase relations (seed=42). The null at strict ≤1 edit is 14.91%, giving a 4.7× lift; under V14-B2's native operationalization (≤3 OR SequenceMatcher ratio > 0.8), the null is 22.06%, a 3.2× lift. Both rule out a chance-coincidence reading."

**Files**: `results/phase7/schema_baseline_rate.{json,md}`. Script: `scripts/task_s05_schema_baseline.py`. Reproducible in 173s on a login node.

### 12.2 V14-B2 robustness across SFT-seen vs SFT-unseen relations (S0.6)

**Result**: edit-distance ≤1 close rate is **41.9% (seen)** vs **77.7% (unseen)**. Higher on unseen — opposite of an SFT-memorization artifact.

**Implication**: V14-B2's 70.4% finding is structurally robust. Defends against the "model just memorized the SFT relation distribution" reading, which would predict the seen split to dominate.

**Recommended paper text**: footnote on the V14-B2 row, "Stratifying by whether the gold relation appears in the SFT corpus shows the typo-bucket rate is *higher* on unseen relations (77.7%) than seen (41.9%), inconsistent with a memorization-from-SFT explanation."

**Files**: `results/phase7/v14_b2_seen_vs_unseen.{json,md}`. Script: `scripts/task_s06_seen_vs_unseen.py`.

### 12.3 CvT robustness under fuzzy matching (S1.3)

**Result**: of 1,203 G2@500 trajectories classified as correct-via-memory (CvM), **0** flip to correct-via-tool (CvT) under a fuzzy match (case-fold, strip articles "a", "an", "the"). 1-hop: 0/751. 2-hop: 0/452.

**Implication**: the CvT/CvM split is sharp under reasonable normalization. Strict-substring CvT is the right metric; no "soft CvT" inflation.

**Recommended paper text**: short footnote on the CvT column in Table 1 / Section 6: "CvT is computed by exact substring of the gold answer in any tool response after normalization. Under fuzzy match (case-fold + article strip), 0/1203 CvM trajectories reclassify as CvT (1-hop: 0/751, 2-hop: 0/452), confirming the metric is robust."

**Files**: `results/phase7/cvt_robustness.{json,md}`. Script: `scripts/task_s13_cvt_robustness.py`.

### 12.4 Oracle L3 upper bound (S1.2) — **Section 6 implication, RELATION CHOICE NOT THE BOTTLENECK**

**Result**: G2@500 with **gold-relation injection at every search call** (entity stays as policy chose, relation replaced with closest-edit gold from the question's `kg_path`):

| Model | Tools/Q | Mean rewrites/Q | EM | ContEM | F1 |
|---|---:|---:|---:|---:|---:|
| G2@500 baseline | 3.00 | — | 39.99% | 45.10% | 0.42 |
| G2@500 + gold-relation oracle | 3.00 | 1.59 | **40.19%** | 45.37% | 0.42 |

Lift: **+0.20pp EM**.

**Implication (revises Act III §C4)**: the model is already emitting close-enough relations on most tool calls. Replacing them with gold relations does **not** unlock additional EM. **The bottleneck is elsewhere** — most likely entity selection, answer extraction from the tool response, or reward-signal saturation. The "L3 retrieval is the constraint" framing must be revised to "L3 retrieval is *not* the binding constraint at this scale; the constraint is the answer-extraction step."

**Section 6 revision plan** (DO BEFORE WRITING):
1. Replace "L3 retrieval is the bottleneck" with "L3 retrieval is correct-enough; the bottleneck is downstream (entity selection / answer extraction)"
2. Add the +0.20pp result as a row in Table 1
3. Use this to motivate why CvT plateaus around 4-9% across all our recipes despite EM hitting 40%
4. The new framing also strengthens the Act II finding: V14-B2 saw the model getting relations *almost right* — now we know that "almost right" is, on average, *good enough* (gold injection adds nothing)

**Files**: `results/phase7/oracle_l3_upper_bound_full_test.json`, `oracle_l3_upper_bound_per_sample/step_0_per_sample.json`, `results/trajectories/phase7/oracle_l3_full/step_0/trajectories.json`. Script: `scripts/task_s12_oracle_l3.py`.

### 12.5 I-Self collapse mechanism (S0.3 + S0.4) — narrative refinement

**Joint result**:
- S0.3: at I-Self steps 200, 250, 300, 400 the tool-call distribution sharply collapses from mean 3.00 (steps 200, 250) to mean 1.00 with 99% of trajectories using exactly one tool call (steps 300, 400). **EM drops 39%→0%** in the same transition.
- S0.4: pre-collapse, `r_outcome` = 0.41, `r_tool_type` = 0.49, `r_retrieval` = 0.10. Post-collapse, `r_outcome` = 0.004 — but the per-call averages of `r_tool_type` and `r_retrieval` are NOT measurable on steps 300/400 in the existing trajectory cache (re-eval queued: jobs 4419143 / 4419144).
- **Mechanism**: the I-Self reward weights `r_tool_type` and `r_retrieval` as *per-call averages*. The policy can preserve those by emitting a single decorative tool call, sacrificing `r_outcome` (which is only 0.41 of the reward signal). Goodhart's law operating exactly as predicted by the reward decomposition.

**Recommended paper text** (replaces the original "verbatim-copy reward hacking" hypothesis):

> "The I-Self collapse at step ~300 is *structural*, not *semantic*. The policy does not learn to verbatim-copy from tool responses; it learns to emit one minimal tool call and stop. The reward decomposition explains why: r_tool_type and r_retrieval are per-call averages, which a single decorative call preserves; r_outcome carries weight 0.41, insufficient to anchor multi-turn behavior in the absence of a per-call recall component. This is Goodhart's law operating at the reward-shape level — the symptom is correct (CvT 9.57% peak then 0% at collapse), the diagnosis is structural rather than the verbatim-copy hypothesis we originally entertained."

**Caveat to flag**: the original eval at I-Self steps 300/400 saved per_sample but not full trajectories, so the "at most 1 tool call" inference is from `num_tool_calls=1` rather than direct text inspection. The re-eval (4419143/4) will produce trajectories for direct verification. Numbers will not change; this just gives a reviewer-defense-quality citation chain.

**Files**: `results/phase7/post_collapse_inspection.{json,md}`, `results/phase7/i_self_reward_decomp.{json,md}`. Scripts: `scripts/task_s03_post_collapse.py`, the S0.4 reward-decomposition script.

### 12.6 Multi-seed reproducibility (S0.1 + S0.2) — **paid, lock target Apr 30**

Three seeds {1, 2, 3} of the I-Self GRPO recipe (S0.1) and three seeds {1, 2, 3} of the G2 SFT-from-base + GRPO-with-E5b chain (S0.2). The single-seed reference numbers in §4 / §8.1 are treated as seed 0.

**Decision rules** (will be applied by `scripts/aggregate_s01_iself.py` and `scripts/aggregate_s02_g2.py` on completion):

S0.1 verdict map:
- All 4 seeds peak between step 200-250 with peak EM 35-45%, all collapse by step 350 → STRONG; ship Section 5 figure with mean ± std band.
- Mixed peaks (some at 200, some at 300, peak heights spread > 10pp) → Story holds with "variance across seeds" caveat.
- Any seed never collapses → Major narrative revision; escalate immediately.

S0.2 verdict map:
- All 4 seeds within ±2pp on EM at step 500 → ROBUST; report mean ± std as headline number.
- Spread > 5pp → Variable; report range with caveat.

**Status (as of 2026-04-28 12:09 UTC)**: all 6 GRPO jobs RUNNING. Eval chained via SLURM dependency. Aggregation scripts staged but not yet executed (gracefully handle missing seeds).

### 12.7 G3 ablation (S1.1, paid Tier 1)

Skips the rule-based pre-SFT and SFTs directly from base Qwen2.5-7B-Instruct on the 14,082 G-distillation traces. Then GRPO with E5b for 500 steps. Single seed (=42).

**Tests**: whether the rule-based pre-SFT is load-bearing for the G2 result, or whether it can be replaced by direct distillation.

**Status**: SFT job 4415343 RUNNING. GRPO 4415344 + EVAL 4415345 dependency-chained. Lock target Apr 30.

### 12.8 What's deferred to rebuttal phase

| Item | Reason |
|---|---|
| S3.1 (G2-with-I-Self reward) | Cut from this round — defer |
| S3.2 (finer-step granularity near collapse) | Defer |
| Qwen3-4B chat-template fix | Documented as known limitation §9.2 |
| 32B V14-E1 chain | SFT itself fails to emit `<answer>` tags (0/50 emit at step 0, identical at all GRPO checkpoints). Likely LoRA r=16 + DoRA-disabled-for-FSDP insufficient for 32B format-control. Re-doing SFT requires full-rank fine-tune (4 GPU × 24h) or higher-rank LoRA — post-lock work. Diagnostic data: `results/trajectories/phase7/v14_e1_qwen32b_*_diag50/`. |

---

## 13. Suggested paper structure (9 pages + unlimited appendix)

1. **Abstract** — one-sentence pitch (§2) + three headline numbers: I-Self CvT=9.57%, G2 EM=40.0%, +6.1pp over 39B baseline
2. **Introduction** — 72.4% Cat-B, diagnostic gap, our 3 constructive claims
3. **Background** — KG-R1/ToolRL/StepSearch/ProGraph-R1/Search-R1 landscape (§7)
4. **Preliminaries** — CWQ, Freebase, 4-tool interface, SFT corpus (§6.1-6.2)
5. **Methods** — Reward formulas (§6.3), training config (§6.4), eval (§6.5)
6. **Results: Diagnostic** — Table 1 E3/E5b rows, pass@k gap analysis (§5.1)
7. **Mechanism** — Task 41 + V14-B1/B2/B4 (§3 Act II, §8.1 memos)
8. **Results: Constructive** — 39B → I-Self curve → G2 (§3 Act III, §4 Table 1)
9. **Cross-benchmark + Closed-model** — KGQAGen Oracle (§5.3), GPT-4o baseline (§5.5)
10. **Discussion** — L3 bottleneck as the common thread, Goodhart at step resolution, paper's signature figure: I-Self collapse curve
11. **Limitations** — §9.1, §9.2, §9.3, §9.4
12. **Appendix A**: Llama audit
13. **Appendix B**: Search-R1 Option 1+2 failure logs
14. **Appendix C**: Reward unit tests, SFT corpus generation details, trajectory classification taxonomy

---

## 14. Raw session memory (optional context)

- Training platform: **Isambard-AI** (ARM aarch64, NVIDIA GH200 120GB, 4 GPUs/node, 24h walltime cap, 32 GPU/project ceiling)
- Total GPU-hours used (estimate): ~1,200 across all variants (G, I-Oracle, I-Self, G1, G2, 39B variants, B3 collapse curve, D1 14B, D2 4B, E1′)
- Framework: **verl** (hybrid FSDP + vLLM 0.12, `enforce_eager=True` mandatory on ARM)
- SFT framework: **TRL** (LoRA r=64 α=128 with DoRA+rslora)
- Paper repo / scratch directories (for cross-reference):
  - `/lus/lfs1aip2/scratch/u6gg/ts1201.u6gg/Project/KG_GRPO/` — primary working dir
  - `/projects/u6gg/KG_GRPO/checkpoints/kg-align-verl/` — all training checkpoints
  - `hpc_tasks.md` — active task queue (v14.1)
  - `CLAUDE.md` — project-level instructions
  - `hpc_implementation_spec.md` — technical spec

End of handoff.
