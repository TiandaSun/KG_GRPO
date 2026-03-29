# HPC Current Tasks

> **This file is the active task queue for the HPC Claude Code agent.**
> Updated from the discussion repo after each planning iteration.
> For technical specs and long-term plan, see `hpc_implementation_spec.md`.
>
> **Last updated**: 2026-03-29 (v6 — E5a/E5b eval done, Phase 4 ablation + cross-model experiments added)
> **Current status**: E1-E5b ALL DONE (training + eval). Phase 2 analysis ALL DONE. Phase 4 ablation experiments ready to launch.
> **Narrative direction**: "Shortcuts All the Way Down: How Reward Design Shapes Tool Use Quality in Agentic RL"
> **Core principle**: Depth over breadth. Now strengthening statistical rigor + cross-model validation.
> **Key findings (finalized)**:
> - correct-via-tool = 0% for E1/E2/E3/E4/E5a, but **E5b@100 = 10%** (genuine 2-hop retrieval, only nonzero)
> - E3 (verifiable) achieves 53.8% EM via verify-then-answer shortcut; E5a achieves 52.6% via pure memorization (comparable)
> - Goodhart via component gaming: proxy-gold r>0.95 maintained but EM declines after step 750
> - E5b demonstrates genuine retrieval is possible but fragile (collapse after step 150)
> - Reward discoverability vs correctness: E5a (correct but sparse) fails; E5b (simple but discoverable) succeeds briefly

---

## Completed Tasks

| Task | Status | Result |
|---|---|---|
| Task 1: E3 Goodhart curve | DONE | 11 steps evaluated. Peak EM=0.542 at step 500, broad plateau 350-750 |
| Task 2: E1 Goodhart curve | DONE | 7 steps evaluated. EM≈0 throughout, confirms no tool learning |
| Task 3: Trajectory sampling | DONE | E3@500 (100), E3@1250 (50), E1@1250 (100), E2@200/600/1200 (100 each) |
| Task 4: R_random reward code | DONE | Already in verl_reward.py:115-126 |
| Task 5: Hop-stratified script | DONE | Script written, tested on E3@500 and E2@200/600/1200 |
| Task 6: E2 evaluation | DONE | 6 steps evaluated. EM=0→0.336, phase transition at step 1000 (tool collapse) |
| Task 9: Trajectory behavior analysis | DONE (initial) | correct-via-tool=0% everywhere. E3: 62% correct-via-memory, 27% hallucination. E3@1250: 60% kg-incomplete |
| Task 10: Tool call distribution | DONE | E3: 100% of trajectories have exactly 1 call (get_tail_relations). No adaptation to hop count |
| Task 11: Goodhart divergence | DONE (initial) | E3: reward 0.004→0.37 while EM peaks then declines. Confirmed |
| Task 12: KG verification rewards | DONE (initial) | r_step_avg stable ~0.25-0.28, r_outcome peaked at step 800 then declined |
| Task 13: Policy entropy/KL | DONE (initial) | E3: entropy spike at step 1000. E2: steady decline (policy collapse). E1: only 1 data point |
| Task 7: E4 (R_random) training | DONE | Finished step 1250. EM=7.4%, tools=0. Weak memorization. |
| Task 15: KL-based Goodhart plot | DONE | E3 EM peaks at cumKL=2.75, drops at cumKL=4.88 |
| Task 16: Automated classification (E1-E3) | DONE | correct-via-tool=0% confirmed with operational definitions |
| Task 17: Pass@k (E3 vs SFT) | DONE | E3 pass@32=61.0% vs SFT 5.0%. Genuine capability expansion confirmed |
| Task 18: Proxy-gold correlation | DONE | E3 r=0.95-0.97 maintained — Goodhart via component gaming, not decorrelation |
| Task 19: Behavioral diversity | DONE | E3 500→1250: response -77%, query-question overlap +94% |
| Task 20: E5a training + eval | DONE | Peak EM=52.6%@step1000, tools=0. Cold start → pure memorization. correct-via-tool=0% |
| Task 21: E5b training + eval | DONE | Step 100: EM=49.0%, tools=2.3, **correct-via-tool=10%** (only nonzero!). Collapse after step 150. |
| Task 21b: E5b trajectory analysis | DONE | 100% get_tail_entities usage. 10/100 genuine retrieval. 33% kg-incomplete (Freebase coverage bottleneck). |

---

## Phase 4: Ablation + Cross-Model Experiments — LAUNCH NOW

> **Context**: Critical review identified 3 methodological gaps that need experiments to address.
> All 3 can run in parallel (3 nodes, 12 GPUs, well within 32 GPU limit).
> Budget: ~132 NHR out of 2,343 remaining (5.6%).

### Task 22: E3 Single-Component Ablation (on_path only) [P0 — addresses "over-engineering"]

**Purpose**: E3 uses a 4-component weighted step reward (on_path 0.45, progress 0.30, coherence 0.15, valid 0.10). Reviewer will ask: "Is the multi-component design necessary, or does on_path alone achieve similar results?" If on_path alone works, E3 is over-engineered. If it doesn't, the components are justified.

**Reward function**:
```python
def e3_single_component_reward(trajectory, gold_answers):
    r_answer = 0.5 * EM(pred, gold) + 0.5 * F1(pred, gold)
    # ONLY on_path — no progress, coherence, or valid components
    r_step = mean([r_on_path_t for t in trajectory.tool_calls])
    return 0.3 * r_answer + 0.7 * r_step
```

**Training config**: Identical to E3 (same model, dataset, GRPO config, 1293 steps, 4 GPUs).

**Eval**: With-tools eval at steps 500, 750 (E3's peak region). Trajectory classification.

**Expected outcomes**:
- If EM ≈ E3 (53.8%): on_path alone suffices → simplify paper's reward design
- If EM << E3: components are justified → acknowledge but note correct-via-tool still likely 0%

**Time**: ~22h training + 2h eval

---

### Task 23: 0.5/0.5 Reward Split Variant [P0 — addresses "0.3/0.7 determined results"]

**Purpose**: The 0.3/0.7 answer/step split gives 70% weight to step rewards. Reviewer will ask: "Did E1 fail because outcome reward is inherently weak, or because 30% weight on answer isn't enough?" A 0.5/0.5 split tests whether giving more weight to the answer component changes E3's behavior and Goodhart dynamics.

**Reward function**: Identical to E3 (R_verifiable) but with α=0.5, β=0.5:
```python
return 0.5 * r_answer + 0.5 * r_step  # instead of 0.3/0.7
```

**Training config**: Identical to E3 except the split. Same model, dataset, GRPO config, 1293 steps, 4 GPUs.

**Eval**: With-tools eval at steps 500, 750, 1000, 1250. Trajectory classification.

**Key comparisons**:
- Is Goodhart onset delayed? (E3 peaks at step 500; does 0.5/0.5 peak later?)
- Is EM higher or lower? (more weight on answer = more pressure for correctness)
- Does tool behavior change? (less weight on step = less incentive for tool calls)

**Time**: ~22h training + 3h eval

---

### Task 24: Llama-3.1-8B Cross-Model Validation [P0 — addresses "Qwen-specific?"]

**Purpose**: All results are on Qwen2.5-7B. Spurious Rewards (arXiv:2506.10947) showed that RLVR findings can be Qwen-specific (contamination). Llama-3.1-8B is the standard cross-model check used in that paper and in "Does RL Incentivize Reasoning" (arXiv:2504.13837).

**Run 3 experiments on Llama-3.1-8B-Instruct:**

**Step 1: SFT Warmup**
- Same SFT procedure as Qwen: 5,000 gold trajectories, 2 epochs, LoRA rank 64
- ~22h on 4 GPUs

**Step 2: Llama-E3 (R_verifiable)**
- Identical reward to Qwen E3
- Same GRPO config (may need LR adjustment — start with 5e-7, same as Qwen)
- ~22h on 4 GPUs

**Step 3: Llama-E5b (R_tool-type bonus)**
- Identical reward to Qwen E5b
- Same GRPO config
- ~22h on 4 GPUs (may be slower if multi-tool behavior emerges)

**Eval for each**: With-tools eval at steps 100, 250, 500, 750. Trajectory classification.

**Key questions**:
1. Does Llama also converge to verify-then-answer shortcut under E3? (correct-via-tool=0%?)
2. Is Llama's parametric memory ceiling lower than Qwen's 52-54%? (if so, KG retrieval becomes more valuable)
3. Does Llama-E5b also achieve correct-via-tool > 0%?
4. Does Llama-E5b also collapse via format drift?

**Time**: ~66h total (SFT 22h + E3 22h + E5b 22h, sequential on 1 node; or parallel on 2 nodes ~44h)

---

### Task 14: Full Test Set Evaluation with Confidence Intervals [P0 — statistical rigor]

**Purpose**: Move from 500-sample estimates to full CWQ test set (3,531 samples) with bootstrap CIs. Required before submission.

**What to do**:
1. Run with-tools eval on FULL CWQ test set for:
   - E3 step 500 (best verifiable)
   - E3 step 750 (alternative peak)
   - E2 step 1200 (best heuristic)
   - E1 step 1250 (best outcome)
   - E4 step 1250 (random)
   - E5a step 1000 (retrieval-grounded)
   - E5b step 100 (tool-type bonus)
   - SFT base
2. Compute 95% bootstrap CIs for EM, F1 (10,000 bootstrap samples)
3. Run McNemar's test for pairwise comparisons
4. Full test set hop-stratified analysis with per-bucket Ns and CIs
5. Full test set trajectory classification (100→500 sample size)

**Time**: ~8-12h eval per model × 8 models. Can parallelize across nodes.

**Note**: This is compute-intensive but only inference, no training. Can use any available GPU.

---

## Tasks On Hold

### Task 8: Filtered Dataset Construction [DEFERRED]
**Status**: On hold. Current unfiltered results tell a complete story. Revisit only if reviewer explicitly requests.

---

## Priority Order (updated 2026-03-29)

```
LAUNCH NOW (parallel, 3 nodes = 12 GPUs):
  Task 22 (E3 single-component ablation)  ← 22h, 1 node
  Task 23 (0.5/0.5 reward split)          ← 22h, 1 node
  Task 24 step 1 (Llama SFT warmup)       ← 22h, 1 node

AFTER Task 24 step 1 (Llama SFT done):
  Task 24 step 2+3 (Llama E3 + E5b)       ← 44h sequential, or 22h × 2 nodes parallel

AFTER ALL TRAINING DONE:
  Task 14 (Full test set CIs for ALL experiments) ← batch inference job

OPTIONAL (P1, if time allows):
  E3 multi-seed (2 extra seeds)            ← 44h
  14B Qwen E3                              ← 44h

DEFERRED:
  Task 8 (Filtered dataset)
```

**Estimated timeline**:
- Days 1-2: Tasks 22, 23, 24-step1 complete (parallel)
- Days 3-4: Task 24-step2+3 complete
- Day 5: Task 14 (batch eval)
- Day 6-7: Analysis + paper update

---

## Notes for HPC Agent

- All evals use the same 500-sample test set and eval protocol as E1-E5b
- Task 14 uses FULL test set (3,531 samples) — separate eval config needed
- **Task 22**: Copy E3 reward code, remove progress/coherence/valid components, keep only on_path
- **Task 23**: Copy E3 reward code, change α=0.5 β=0.5 (only change)
- **Task 24**: Need Llama-3.1-8B-Instruct model weights. Check if already downloaded on HPC. SFT data is the same 5,000 gold trajectories used for Qwen.
- For Llama: may need to adjust tokenizer config, chat template, and possibly LR. Start with Qwen's config and adjust if training is unstable.
- **Do NOT start Task 14 until Tasks 22-24 are complete** — we want to batch all models in one eval pass
- Report results for Tasks 22/23 as soon as eval is done — these determine if we need to update the paper narrative

---

## Phase 2 & 3 Tasks: ALL DONE ✅

> All methodology-aware analysis tasks (Tasks 9-21b) are complete.
> Results available in conversation history and trajectory files.
> See `data_collection_checklist.md` for full summary of available data.
> See `writing_brief.md` for how to use these results in the paper.

### (Archived — kept for reference, all DONE)

### Task 15: KL Divergence Extraction [DONE]

**Purpose**: The Goodhart literature (Gao et al. ICML 2023, Karwowski/Skalse ICLR 2024) requires plotting reward vs gold metric over **KL divergence from reference policy**, not training steps. Without this, our Goodhart claim is methodologically incomplete.

**What to do**:
1. Check if veRL training logs already contain KL divergence (GRPO typically computes KL between current policy and reference). Look in:
   - wandb logs (search for "kl" or "divergence" fields)
   - tensorboard event files
   - veRL's `algorithm_statistics` or `policy_statistics` logs
2. **If KL is logged**: Extract per-step KL divergence for E1, E2, E3. Output as CSV: (step, kl_divergence, mean_reward, eval_EM).
3. **If KL is NOT logged**: We need to compute it. For each checkpoint (E3: steps 100,250,350,400,450,500,600,750,1000,1250), compute:
   ```
   KL(π_checkpoint || π_SFT) = mean over eval samples [
     sum over tokens ( log p_checkpoint(token|context) - log p_SFT(token|context) )
   ]
   ```
   Use the eval trajectories (already generated) as the sample set. Load SFT model and checkpoint, run forward pass to get log-probs.
4. **Plot**: x-axis = KL divergence, y-axis (left) = training reward, y-axis (right) = eval EM. One plot per experiment.

**Output**: CSV of (step, KL, reward, EM) + Goodhart hump-curve plot for E3.

**Reference**: Gao et al. "Scaling Laws for Reward Model Overoptimization" (arXiv:2210.10760, ICML 2023).

---

### Task 16: Automated Trajectory Classification with Operational Definitions [HIGHEST PRIORITY]

**Purpose**: Task 9's initial classification was done by the analysis agent. We need **deterministic, reproducible** classification criteria that any researcher can replicate. This is what distinguishes a rigorous behavioral analysis from anecdotal case studies.

**Operational definitions** (apply these automatically to ALL trajectories):

```python
def classify_trajectory(trajectory, gold_answers, kg_response_text):
    """Deterministic classification — no human judgment needed."""

    answer_correct = (trajectory.predicted_answer in gold_answers)  # exact match
    answer_in_kg_response = any(
        gold_ans.lower() in kg_response_text.lower()
        for gold_ans in gold_answers
    )
    query_entity_in_question = any(
        ent in trajectory.question
        for ent in trajectory.query_entities
    )
    kg_returned_nonempty = (len(kg_response_text.strip()) > 0
                           and kg_response_text != "No results found")
    has_tool_call = (trajectory.num_tool_calls > 0)

    if not has_tool_call:
        if answer_correct:
            return "correct-no-tool"      # answered correctly without using KG
        else:
            return "wrong-no-tool"        # answered incorrectly without using KG

    if answer_correct:
        if answer_in_kg_response:
            return "correct-via-tool"     # KG response contains the gold answer
        else:
            return "correct-via-memory"   # correct but answer NOT in KG response
    else:
        if not kg_returned_nonempty:
            return "kg-incomplete"        # KG returned empty/no results
        elif not query_entity_in_question:
            return "tool-misuse"          # queried irrelevant entity
        else:
            return "wrong-answer"         # KG returned data, model got wrong answer
```

**What to do**:
1. Implement the above classification script
2. Run on ALL saved trajectories: E3@500 (100), E3@1250 (50), E1@1250 (100), E2@200 (100), E2@600 (100), E2@1200 (100)
3. Output: behavior distribution table per experiment (counts + percentages)
4. Cross-check against Task 9's manual results — identify any discrepancies
5. Save 5 representative examples per category for paper case studies

**Critical check**: Does the `answer_in_kg_response` criterion change E3@500's correct-via-tool from 0% to something higher? The initial Task 9 analysis found 0% but the classification criteria may have been different. We need to verify with the exact operational definition above.

**Output**:
- Behavior distribution table (automated, deterministic)
- Classification script (reproducible)
- Discrepancy report vs Task 9 initial results
- Representative examples per category

---

### Task 17: Pass@k Analysis [HIGH PRIORITY — addresses "RLVR doesn't create new reasoning"]

**Purpose**: The strongest accepted method for distinguishing "genuine capability expansion" vs "distribution sharpening" is pass@k analysis (arXiv:2504.13837, ICLR 2026). This directly addresses the NeurIPS 2025 criticism that "RL doesn't create new reasoning capacity."

**What to do**:
1. For E3 step 500 (best checkpoint) and SFT base, generate k=64 samples per question (temperature=0.7 or the training temperature) on a subset of 200 test questions (with tools)
2. Compute pass@k for k=1, 4, 16, 32, 64 using the unbiased estimator:
   ```
   pass@k = 1 - C(n-c, k) / C(n, k)
   ```
   where n = total samples, c = correct samples
3. Compare E3@500 vs SFT base:
   - If E3 pass@1 >> SFT pass@1 BUT E3 pass@64 ≈ SFT pass@64 → **distribution sharpening** (no new capability)
   - If E3 pass@64 >> SFT pass@64 → **genuine capability expansion**
   - If SFT pass@64 >> E3 pass@64 → **capability narrowing** (RL reduced diversity)
4. Also compute for E2 step 1200 to compare strategies

**Output**: Pass@k table + plot. Interpretation of whether E3 learned new capabilities or just narrowed sampling.

**Reference**: "Does RL Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model?" (arXiv:2504.13837)

**Note**: This is compute-intensive (64 samples × 200 questions × 2 models = ~25,600 inference runs). Estimate time and report before starting. May need to reduce to k=16 or fewer questions if too expensive.

---

### Task 18: Proxy-Gold Correlation Curve [HIGH PRIORITY — Goodhart formal evidence]

**Purpose**: Beyond the hump-shaped curve, the Goodhart literature (Coste et al. 2024) expects evidence of **correlation degradation** between proxy reward and gold metric over optimization.

**What to do**:
1. For each E3 checkpoint (steps 100,250,350,...,1250):
   - Extract per-sample training reward (if available from training logs)
   - Extract per-sample eval EM (from eval results)
   - Compute Pearson correlation between per-sample reward and per-sample EM
2. Plot: correlation coefficient vs training step
3. Goodhart onset = the step where correlation starts declining
4. Repeat for E1, E2 if per-sample rewards are accessible

**If per-sample training rewards are not available**:
- Use the reward function to re-compute rewards on eval trajectories at each checkpoint
- This requires the eval trajectories + the reward function code

**Output**: Correlation curve + Goodhart onset step for each experiment.

**Reference**: Coste et al. "Correlated Proxies: A New Definition and Improved Mitigation for Reward Hacking" (arXiv:2403.03185)

---

### Task 14: Full Test Set Evaluation with Confidence Intervals [MEDIUM — needed before submission]

**Purpose**: Move from 500-sample estimates to full test set with statistical rigor.

**What to do**:
1. Run with-tools eval on FULL CWQ test set for:
   - E3 step 500 (our best model)
   - E2 step 1200 (heuristic best)
   - E1 step 1250 (outcome best)
   - SFT base
2. Compute 95% bootstrap confidence intervals for EM, F1 (10,000 bootstrap samples)
3. Run McNemar's test for pairwise comparisons (E3 vs E2, E3 vs E1, E2 vs E1)
4. Full test set hop-stratified analysis with per-bucket sample sizes and CIs
5. Report CWQ's actual hop distribution (what % is 1-hop vs 2-hop vs 3-hop+)

**Output**: Full test set results table with CIs + significance test p-values.

---

### Task 19: Behavioral Diversity Metrics [MEDIUM — enriches narrative]

**Purpose**: Quantify whether policy diversity changes over training (cf. RAGEN's Echo Trap detection, arXiv:2504.20073).

**What to do**:
1. From saved trajectories, compute per-experiment:
   - Query diversity: unique queries / total queries
   - Answer diversity: unique answers / total answers
   - Response length variance
   - Query entity overlap with question entities (what % of queries contain question entities vs random entities)
2. Compare E3@500 vs E3@1250 to quantify what changes during Goodhart decline
3. Compare across E1/E2/E3 at their respective best checkpoints

**Output**: Diversity metrics table + interpretation.

**Reference**: RAGEN "Understanding Self-Evolution in LLM Agents via Multi-Turn RL" (arXiv:2504.20073)

---

## Phase 3: Information-Grounded Reward Experiments — ALL DONE ✅

> **Context**: All E1-E3 experiments show correct-via-tool = 0%. The model has 4 KG tools
> (including `get_tail_entities` which returns actual answer entities) but only ever uses
> `get_tail_relations` (cheapest, returns relation names only). This is rational shortcutting
> under reward misspecification — the current reward doesn't distinguish tool informativeness.
>
> **E5a and E5b test whether redesigned rewards can induce genuine KG retrieval.**
> Launch BOTH in parallel (each uses 1 node / 4 GPUs, total 8 GPUs).
>
> **Literature basis**: IGPO (arXiv:2510.14967), ProGraph-R1 (arXiv:2601.17755),
> TRM (OpenReview:LnBEASInVr), ToolRL (arXiv:2504.13958)

### Task 20: E5a Training — Retrieval-Grounded Reward [HIGHEST PRIORITY]

**Purpose**: Test whether rewarding entity retrieval tools MORE than relation tools — AND checking if retrieved entities appear in the final answer — induces genuine multi-hop KG retrieval.

**Reward function** (implement in `verl_reward.py`):

```python
def e5a_retrieval_grounded_reward(trajectory, gold_answers):
    """
    Key difference from E3: distinguishes tool informativeness.
    - get_tail/head_entities (returns actual entities): HIGH reward if entity in answer
    - get_tail/head_relations (returns relation names): LOW reward (exploration only)
    """
    # Answer component — identical to E1-E4
    r_answer = 0.5 * EM(pred, gold) + 0.5 * F1(pred, gold)

    # Step component — information-grounded
    step_rewards = []
    for step in trajectory.tool_calls:
        kg_response = step.kg_response_text
        final_answer = trajectory.predicted_answer

        if step.action in ["get_tail_entities", "get_head_entities"]:
            # Entity retrieval tools — reward based on utilization
            retrieved_entities = parse_entities(kg_response)
            if any(ent.lower() in final_answer.lower() for ent in retrieved_entities):
                step_rewards.append(1.0)    # Retrieved entity used in answer
            elif len(retrieved_entities) > 0:
                step_rewards.append(0.3)    # Retrieved but not used
            else:
                step_rewards.append(0.0)    # Empty result

        elif step.action in ["get_tail_relations", "get_head_relations"]:
            # Relation discovery — lower reward (exploration only)
            if len(kg_response.strip()) > 0 and kg_response != "No results found":
                step_rewards.append(0.2)    # Relations found
            else:
                step_rewards.append(0.0)    # Empty

    r_step = mean(step_rewards) if step_rewards else 0.0

    return 0.3 * r_answer + 0.7 * r_step
```

**Training config**: Identical to E1-E4:
- Model: Qwen2.5-7B (SFT warmup checkpoint)
- Dataset: CWQ (same split)
- Algorithm: GRPO
- Steps: 1293 (or until crash)
- GPUs: 4 (1 node)
- Save checkpoints every 50 steps

**Eval plan** (after training):
1. With-tools eval at steps 100, 250, 500, 750, 1000, 1250
2. Trajectory sampling: 100 from best checkpoint
3. **Automated classification (Task 16 script)** — critical: is correct-via-tool > 0%?
4. **Tool action distribution**: does the model use `get_tail_entities`?

**Success metric**: correct-via-tool > 0% AND model uses `get_tail_entities`/`get_head_entities`

---

### Task 21: E5b Training — Tool-Type Bonus Reward [PARALLEL WITH E5a]

**Purpose**: Simpler ablation — just reward entity retrieval tools more, WITHOUT checking if output is used in answer. Helps distinguish: does the model need (a) incentive to use the right tools, or (b) incentive to use the right tools AND incorporate their output?

**Reward function**:

```python
def e5b_tool_type_bonus_reward(trajectory, gold_answers):
    """
    Simpler than E5a: rewards entity retrieval tools more than relation tools.
    Does NOT check if retrieved entity appears in answer.
    """
    r_answer = 0.5 * EM(pred, gold) + 0.5 * F1(pred, gold)

    tool_type_weights = {
        "get_tail_entities": 1.0,
        "get_head_entities": 1.0,
        "get_tail_relations": 0.3,
        "get_head_relations": 0.3,
    }

    step_rewards = []
    for step in trajectory.tool_calls:
        weight = tool_type_weights.get(step.action, 0.0)
        if len(step.kg_response_text.strip()) > 0 and step.kg_response_text != "No results found":
            step_rewards.append(weight)
        else:
            step_rewards.append(0.0)

    r_step = mean(step_rewards) if step_rewards else 0.0
    return 0.3 * r_answer + 0.7 * r_step
```

**Training config**: Identical to E5a.

**Eval plan**: Same as E5a — focus on correct-via-tool rate and tool action distribution.

**E5a vs E5b comparison**:
- If E5a > E5b: checking output utilization matters (reward needs to verify tool output is used)
- If E5a ≈ E5b: just incentivizing the right tools is enough
- If both ≈ E3: model can't learn multi-hop even with better rewards (deeper problem)

---

## Methodology References (for HPC agent context)

These papers define what reviewers expect for our claims:

| Claim | Required methodology | Reference |
|---|---|---|
| "Goodhart effect occurred" | Hump curve over KL divergence + proxy-gold correlation degradation | Gao et al. ICML 2023 (arXiv:2210.10760) |
| "Reward hacking detected" | Trajectory-level behavioral taxonomy + temporal onset + control condition | METR 2025; Anthropic (arXiv:2511.18397) |
| "RL taught genuine skill" | Pass@k showing capability expansion beyond base model | arXiv:2504.13837 (ICLR 2026) |
| "Behavioral differences are significant" | Bootstrap CIs + McNemar's test | Standard statistical practice |
| "Behavioral diversity changed" | Echo Trap metrics + diversity quantification | RAGEN (arXiv:2504.20073) |
