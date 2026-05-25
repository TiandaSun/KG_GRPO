# HPC Data Provenance Audit — 2026-04-12

Cross-check of claims made in discussion-side files against actual code, configs, and output files on Isambard.

---

## Check 1: Eval Config Parity (Task 14 vs Earlier 500-Sample)

**Finding: DIFFERENT — three config changes identified**

### Differences

| Parameter | Earlier 500-sample | Task 14 Qwen | Task 14 Llama |
|-----------|-------------------|--------------|---------------|
| **eval_data** | `val.parquet` | `test.parquet` | `test.parquet` |
| **max_samples** | 500 | 0 (all 3531) | 0 (all 3531) |
| **max_turns** | 5 | 5 | **3** |

### What stayed the same

- **Same eval script**: `scripts/eval_with_tools.py` for all runs
- **Same answer extraction**: regex-based `<answer>` tag → `</think>` fallback → raw text (lines 87-92)
- **Same max_new_tokens**: 512 (default, not overridden in any job)
- **Same KG server**: Freebase, same data dir, same 5 endpoints
- **Same prompt template**: pulled from parquet file's `prompt[0]["content"]`
- **Same scoring**: normalize → exact_match, contains_match, token_f1
- **Same checkpoint paths**: E3 Qwen uses `grpo-cwq-7b-verifiable-20260321` in both

### Impact assessment

1. **val.parquet → test.parquet**: Different question distributions. The first 500 sequential samples from `val.parquet` may have different difficulty than the full `test.parquet`. This is likely the **primary driver** of the ~20pp gap (53.8% → 33.0%).

2. **max_turns=5 → max_turns=3 (Llama only)**: Documented in the job script as a throughput optimization: "Llama throughput ~50 samples/hr @ max_turns=3 → ~72h total per model". This restricts multi-hop reasoning, reducing tool call opportunities. **Qwen Task 14 was NOT affected** (still max_turns=5).

3. **Task 30 conclusion**: Task 30 stated "all evals use sequential first-500, consistent." This is correct for the *earlier* evals (all used val.parquet sequential first-500). But Task 14 uses **test.parquet, all 3531 samples**, so it is a methodologically **different evaluation** on a different split and sample size.

### Evidence files
- Earlier: `scripts/run_eval_goodhart_e3.job` lines 54-58 (val.parquet, max_samples=500, max_turns=5)
- Task 14 Qwen: `scripts/run_task14_full_eval_qwen.job` lines 88-102 (test.parquet, max_samples=0, max_turns=5)
- Task 14 Llama: `scripts/run_task14_full_eval_llama.job` lines 82-97 (test.parquet, max_samples=0, **max_turns=3**)

---

## Check 2: Llama EM=0 Root Cause

**Finding: GENUINE FAILURE — but the "earlier 24.8%" claim is unsubstantiated**

### What the actual data shows

**Llama EM=0.0 in ALL evaluations, not just Task 14:**
- `eval_llama_e3_step500.json`: EM=0.000, ContEM=0.592 (500 samples, val, max_turns=5)
- `eval_llama_e5b_step100.json`: EM=0.006, ContEM=0.210 (500 samples, val, max_turns=5)
- Task 14 llama_e3_1293: EM=0.000, ContEM=0.421 (3531 samples, test, max_turns=3)
- All 4 diagnostic interventions (A1-A4): EM=0.000 in every case

**The "Llama E3=24.8%, E5b=25.0%" claim from the question does not match any result file in the repo.** The hpc_tasks.md line 14 states "Llama ceiling ~25%" — this likely refers to **Contains-EM** (~21-59% depending on config), not strict EM. The actual "24.8%" number appears to have originated in discussion-side files, not from HPC outputs.

### Root cause: output degeneration

Sampling 5 Llama E3 trajectories from diagnostics reveals:
- Model begins with coherent `<think>` reasoning and 1-2 valid `<search>` tool calls
- After receiving tool responses, degenerates into repetitive token loops: `"currency.countries_formerly_formerly_formerly_..."` repeating for thousands of characters
- **0% of sampled trajectories contain `<answer>` tags**
- The `</think>` fallback extracts post-thinking text, which is the degenerate loop — not a valid answer

### Answer extraction comparison

The same `extract_answer()` function (lines 87-92) is used in all evals:
```python
if "<answer>" in text and "</answer>" in text:  # primary
    ...
if "</think>" in text:                          # fallback
    return text.split("</think>")[-1].strip()
return text.strip()                              # last resort
```
There is **no "last line" fallback** that could have produced 24.8% in an earlier eval. The extraction method is consistent.

### Diagnostic interventions (all EM=0)
| Intervention | Config | ContEM | Outcome |
|---|---|---|---|
| A1: Longer tokens | test, 100, 2048tok, 3 turns | 43.0% | Longer loops, same failure |
| A2: More turns on val | val, 500, 512tok, 5 turns | 67.0% | Higher ContEM but still EM=0 |
| A3: More turns on test | test, 500, 512tok, 5 turns | 55.4% | Same pattern |
| A4: Repetition penalty | test, 100, 512tok, 3t, rep=1.15 | 53.0% | Shorter loops, same failure |

### Conclusion
Llama EM=0 is a **genuine model failure** (degenerate token loops after tool use), not an extraction artifact. The "24.8%" number cited in the question **has no corresponding result file on HPC** and may be a discussion-side error or a misattribution of Contains-EM.

---

## Check 3: E5b correct-via-tool=10% Sample Provenance

**Finding: 100 trajectories from first-100 of a 500-sample val.parquet eval**

### Exact provenance

1. **Source script**: `scripts/run_eval_e5b_step100.job`
2. **Eval config**: val.parquet, max_samples=500, max_turns=5, max_new_tokens=512
3. **Trajectory sampling**: `--max_trajectory_samples 100` — saves the **first 100** trajectories from the 500-sample eval (sequential, not random)
4. **Classification**: Inline Python in the same job script (lines 70-143), applied to the 100 saved trajectories
5. **Output**: `results/e5b_step100_classification.json`

### Classification methodology

The classification uses deterministic rules (lines 86-105 of job script):
- `correct-via-tool`: EM > 0 AND num_tools > 0 AND gold answer appears in KG tool responses
- `correct-via-memory`: EM > 0 AND num_tools > 0 AND gold answer NOT in KG tool responses
- `correct-no-tool`: EM > 0 AND num_tools = 0
- `kg-incomplete`: EM = 0 AND tools used but KG returned empty/no-results
- `wrong-answer`: EM = 0 AND tools used and KG returned results
- `wrong-no-tool`: EM = 0 AND num_tools = 0

### Was classification run on full 3531?

**No.** Task 14 (`task14_summary.json`) contains only aggregate metrics (EM, F1, ContEM, avg_tool_calls) with bootstrap CIs. It does **not** contain trajectory classification. The CvT=10% number in any discussion-side tables is **carried over from the 100-sample classification**, not independently measured on the full test set.

### Population concerns

The 100 trajectories are the **first 100 sequential samples from val.parquet** (not test.parquet, not random). This means:
- They may not be representative of the full val set (500 samples) or the test set (3531 samples)
- The 10% CvT rate has a 95% CI of roughly [4%, 18%] at n=100 (binomial)
- hpc_tasks.md acknowledges this: "Full test set trajectory classification (100→500 sample size)" was planned but not yet completed

---

## Check 4: Pass@k Sample Identity

**Finding: First 200 sequential from val.parquet, with tools, temperature=0.7, k=32**

### Exact configuration

- **Script**: `scripts/task17_pass_at_k.py`
- **Questions**: First 200 from `data/freebase/verl_cwq/val.parquet` (sequential by file order, lines 281-284: `for i, (_, row) in enumerate(df.iterrows()): if i >= args.n_questions: break`)
- **No filtering, stratification, or shuffling** — purely sequential
- **Samples per question**: 32 (`--n_samples 32`)
- **Temperature**: 0.7
- **top_p**: 0.95, do_sample=True (set in the script)
- **Tools**: YES — KG server started in the job script, `--kg_server_url` passed
- **Models evaluated**: E3@500 and SFT base (and optionally E2@1200)
- **Data split**: val.parquet (same as earlier 500-sample evals, NOT test.parquet)

### Important note

Pass@k was run on the **first 200 of val.parquet**, while Task 14 was run on **all of test.parquet**. These are different splits. The pass@1=55.2% (on val first-200) is not directly comparable to Task 14 EM=32.6% (on test full-3531) without accounting for both split and sample size differences.

### Evidence
- Job script: `scripts/run_task17_pass_at_k.job` (lines 50-61)
- Python argparse: `scripts/task17_pass_at_k.py` (lines 247-252, 280-296)

---

## Check 5: Category B Computation Details

**Finding: Raw Qwen2.5-7B-Instruct (NOT SFT), temperature=0.7, on test.parquet (all 3531)**

### Exact configuration

- **Model**: `Qwen/Qwen2.5-7B-Instruct` — the **raw pre-trained model**, NOT the SFT checkpoint
- **Reason documented in job script**: "SFT was trained for tool use, so without tools it emits `<search>` tags → no answers"
- **Temperature**: 0.7 with do_sample=True, top_p=0.95
- **k_samples**: 10 (pass@10)
- **Data**: `data/freebase/verl_cwq/test.parquet` — all 3531 questions, **no filtering**
- **Tools**: NO tools (pure parametric memory test)
- **Result**: 975 Category A (27.6%), 2,556 Category B (72.4%)

### Evidence
- Job script: `scripts/run_task26_pass10.job` (lines 37-44)
- Python script: `scripts/task26_pass10_category_b.py`

---

## Check 6: E3@500 vs E3@750

**Finding: E3@750 marginally higher but NOT statistically significant. Both evaluated on KGQAGen-10k.**

### CWQ (Task 14, test.parquet, n=3531)

| Checkpoint | EM | 95% CI | ContEM | Avg Tools |
|---|---|---|---|---|
| E3@500 | 0.3260 | [0.3101, 0.3413] | 0.3455 | 1.00 |
| E3@750 | 0.3297 | [0.3141, 0.3447] | 0.3495 | 1.00 |

**Delta**: +0.37pp EM. CIs overlap completely → **not significant**.

### KGQAGen-10k (Task 35, n=1079)

| Checkpoint | EM |
|---|---|
| E3@500 | 0.1891 |
| E3@750 | 0.1974 |

**Delta**: +0.83pp EM on KGQAGen.

### Recommendation

Use **E3@500** as the canonical "best E3" checkpoint for the paper:
1. The difference is not statistically significant on either benchmark
2. E3@500 has substantially more auxiliary analyses completed (pass@k, trajectory classification, hop-stratified analysis, behavioral diversity analysis)
3. E3@500 is the earlier checkpoint — using it avoids any suspicion of cherry-picking

---

## Check 7: KGQAGen Cross-Benchmark Correlation

**Finding: ρ=0.991 is inflated by tied Llama zeros. Honest ρ (Qwen-only) = 0.976.**

### Full details

| Subset | N models | Spearman ρ (EM) |
|---|---|---|
| All 11 models (8 Qwen + 3 Llama) | 11 | 0.9909 |
| **Qwen-only (8 models)** | **8** | **0.9762** |
| Qwen non-trivial (excl. SFT) | 7 | 0.9643 |

### Per-model EM values

| Model | CWQ EM | KGQAGen EM |
|---|---|---|
| e3_750 | 0.3297 | 0.1974 |
| e3_500 | 0.3260 | 0.1891 |
| e5b_100 | 0.3166 | 0.1668 |
| e5a_1000 | 0.3149 | 0.1817 |
| e2_1200 | 0.1991 | 0.1298 |
| e4_1250 | 0.0504 | 0.0408 |
| e1_1250 | 0.0020 | 0.0278 |
| sft | 0.0006 | 0.0056 |
| llama_e3_1293 | 0.0000 | 0.0000 |
| llama_e5b_1293 | 0.0000 | 0.0000 |
| llama_sft | 0.0000 | 0.0000 |

### Why ρ=0.991 is inflated

Three Llama models form a perfectly tied cluster at (0, 0) on both benchmarks. In Spearman correlation, tied ranks at the bottom inflate ρ by adding "free" agreement. The **honest number to report is ρ=0.976** (8 Qwen models only), or ρ=0.964 (7 non-trivial Qwen models).

ρ=0.976 is still exceptionally strong and fully supports the triangulation narrative. But reporting ρ=0.991 without noting the tied-zero inflation would be misleading.

### Computation method
- Script: `scripts/task35_cross_benchmark.py`
- Method: Custom Spearman via rank correlation (not scipy) — verified correct implementation
- Computed on EM rankings (also computed on F1: ρ=0.964 for all 11)
- No Llama-excluded ρ was computed in the script — I computed it manually above

---

## Summary of Actionable Issues

| Check | Severity | Action Required |
|---|---|---|
| 1. Val vs test split | **HIGH** | Acknowledge in paper Methods that 500-sample evals used val split, Task 14 used test split. The 20pp gap is partly split-driven, not just sample-difficulty. |
| 1b. Llama max_turns=3 | **MEDIUM** | Note in paper that Llama Task 14 used max_turns=3 (vs 5 for Qwen). Consider rerunning Llama with max_turns=5 if compute allows. |
| 2. "Llama E3=24.8%" claim | **HIGH** | This number has no backing evidence on HPC. Llama EM=0.000 in ALL evaluations. Correct the narrative. The "~25% ceiling" appears to be Contains-EM, not EM. |
| 3. CvT=10% on 100 samples | **MEDIUM** | State sample size (n=100, first-100 sequential from val) and CI [4-18%] in paper. Plan to run classification on full test set before submission. |
| 4. Pass@k on val first-200 | **LOW** | Different split from Task 14. Acceptable if disclosed, but note that pass@k questions may be easier (val sequential). |
| 5. Category B uses raw model | **LOW** | Methodologically sound — raw model is the right choice for parametric memory test. Document in Methods. |
| 6. E3@500 vs E3@750 | **LOW** | Not significant. Use E3@500 as canonical checkpoint. |
| 7. ρ inflated by tied Llama | **MEDIUM** | Report ρ=0.976 (Qwen-only) as the primary number. Can mention ρ=0.991 with all models as a footnote. |
