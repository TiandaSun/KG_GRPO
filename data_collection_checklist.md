# Data Collection Checklist for Writing Agent

> **Purpose**: All data the writing agent needs to write the paper.
> **Status**: Mark each item ✅ when collected and path confirmed.
> **Last updated**: 2026-03-27

---

## A. Quantitative Results Tables

### A1. Main Results Table (with-tools eval, all experiments)

| Item | Status | Path/Value |
|---|---|---|
| SFT base: EM, ContEM, F1, avg_tool_calls | ✅ | EM=0.004, ContEM=0.142, F1=0.030, tools=5.0 |
| E1 best checkpoint (step 1250): EM, ContEM, F1, tool_calls | ✅ | EM=0.002, ContEM=0.570, F1=0.119, tools=4.9 |
| E2 best checkpoint (step 1200): EM, ContEM, F1, tool_calls | ✅ | EM=0.336, ContEM=0.580, F1=0.449, tools=0.14 |
| E3 best checkpoint (step 500): EM, ContEM, F1, tool_calls | ✅ | EM=0.542, ContEM=0.546, F1=0.546, tools=1.0 |
| E4 best checkpoint: EM, ContEM, F1, tool_calls | ⏳ | Training ~6h remaining |
| E5a best checkpoint: EM, ContEM, F1, tool_calls | ⏳ | Training ~14h remaining |
| E5b best checkpoint: EM, ContEM, F1, tool_calls | ⏳ | Training ~50h remaining |

### A2. Goodhart Curves (for Figure — EM over training steps, with-tools)

| Item | Status | Path/Value |
|---|---|---|
| E1 curve: (step, EM, ContEM, F1, tool_calls) for steps 0,100,250,500,750,1000,1250 | ✅ | In conversation history |
| E2 curve: (step, EM, ContEM, F1, tool_calls) for steps 0,200,400,600,800,1000,1200 | ✅ | In conversation history |
| E3 curve: (step, EM, ContEM, F1, tool_calls) for steps 0,100,250,350,400,450,500,600,750,1000,1250 | ✅ | In conversation history |
| E4 curve: multiple checkpoints | ⏳ | After training |
| E5a curve: multiple checkpoints | ⏳ | After training |
| KL divergence per checkpoint for E3 (Task 15) | ⏳ | Extract from veRL logs |

### A3. Pass@k Results

| Item | Status | Path/Value |
|---|---|---|
| E3@500 pass@k (k=1,4,8,16,32) | ✅ | 55.2%, 57.6%, 58.7%, 59.7%, 61.0% |
| SFT base pass@k (k=1,4,8,16,32) | ✅ | 0.2%, 0.9%, 1.6%, 2.9%, 5.0% |
| E2@1200 pass@k | ⏳ | Run after E5a/E5b complete (batch job) |
| E5a best checkpoint pass@k | ⏳ | After E5a complete |

### A4. Hop-Stratified Results

| Item | Status | Path/Value |
|---|---|---|
| E3@500 hop-stratified: (hop, N, EM, F1, tools) | ✅ | 1-hop: N=77, EM=0.636; 2-hop: N=23, EM=0.565 |
| E2@1200 hop-stratified | ✅ | 1-hop: EM=0.351; 2-hop: EM=0.261, tools=0.00 |
| Full test set hop distribution (% 1-hop / 2-hop / 3-hop+) | ⏳ | Run on full CWQ test set |
| E5a/E5b hop-stratified | ⏳ | After training |

### A5. Full Test Set Results with Confidence Intervals (Task 14)

| Item | Status | Path/Value |
|---|---|---|
| E3@500 full test set: EM ± 95% CI | ⏳ | Submit job now |
| E2@1200 full test set: EM ± 95% CI | ⏳ | Submit job now |
| E1@1250 full test set: EM ± 95% CI | ⏳ | Submit job now |
| SFT base full test set: EM ± 95% CI | ⏳ | Submit job now |
| McNemar's test p-values (E3 vs E2, E3 vs E1, E2 vs E1) | ⏳ | Compute after full test set |

---

## B. Trajectory Analysis Data

### B1. Automated Behavior Classification (Task 16)

| Item | Status | Path/Value |
|---|---|---|
| E3@500 classification: % correct-via-tool, correct-via-memory, tool-misuse, kg-incomplete, wrong-answer | ✅ | 0%, 62%, 23%, 3%, 12% |
| E3@1250 classification | ✅ | 0%, 36%, 0%, 60%, 4% |
| E1@1250 classification | ✅ | 0%, 0%, 86%, 3%, 10% |
| E2@200 classification | ✅ | 0%, 0%, 88%, 0%, 12% |
| E2@600 classification | ✅ | 0%, 0%, 79%, 1%, 20% |
| E2@1200 classification | ✅ | 0%, 33%(no-tool), 4%, 0%, 63%(wrong-no-tool) |
| E4 best checkpoint classification | ⏳ | After E4 eval |
| E5a best checkpoint classification | ⏳ | **Most critical: is correct-via-tool > 0%?** |
| E5b best checkpoint classification | ⏳ | After E5b eval |
| 5 representative examples per category (for paper case studies) | ⏳ | Extract from trajectory files |

### B2. Tool Action Distribution (Task 10)

| Item | Status | Path/Value |
|---|---|---|
| E3: % get_tail_relations / get_tail_entities / get_head_relations / get_head_entities | ✅ | 100% / 0% / 0% / 0% |
| E1: tool action distribution | ⏳ | Extract from saved trajectories |
| E2@1200: tool action distribution | ✅ | ~0% (tool calls near zero) |
| E5a: tool action distribution | ⏳ | **Key: does model use get_tail_entities?** |
| E5b: tool action distribution | ⏳ | After eval |

### B3. Tool Call Distribution (not just mean)

| Item | Status | Path/Value |
|---|---|---|
| E3: histogram of tool calls per sample (is it exactly 1 for all, or bimodal?) | ✅ | 100% exactly 1 call |
| E2 phase transition: tool calls distribution at step 800 vs 1200 | ✅ | Confirmed in Task 10 |
| E5a/E5b: tool calls distribution | ⏳ | After eval |

### B4. Behavioral Diversity Metrics (Task 19)

| Item | Status | Path/Value |
|---|---|---|
| E3@500 vs E3@1250: response length variance, query diversity, query-question overlap % | ✅ | Response -77%, query overlap +94% |
| E1/E2 diversity metrics | ⏳ | Compute from saved trajectories |

---

## C. Training Dynamics Data

### C1. Reward Curves (for Goodhart double-axis figure)

| Item | Status | Path/Value |
|---|---|---|
| E3: (step, mean_train_reward) from wandb/tensorboard | ✅ | reward 0.004→0.37 (keeps climbing) |
| E2: (step, mean_train_reward) | ✅ | Flat then late surge |
| E1: (step, mean_train_reward) | ⏳ | Only 1 data point available |
| E3: r_step_avg and r_outcome separately over training | ✅ | r_outcome peaked step 800, then declined |

### C2. Policy Entropy / KL Divergence

| Item | Status | Path/Value |
|---|---|---|
| E3: entropy over training steps (from wandb) | ✅ | Spike at step 1000 (0.017→0.043) |
| E2: entropy over training steps | ✅ | Steady decline 0.075→0.018 |
| E3: KL divergence from reference policy per checkpoint | ⏳ | Task 15 — check veRL logs |
| E3 Goodhart: EM peaks at cumKL=2.75, drops at cumKL=4.88 | ✅ | From Task 15 initial results |

### C3. Proxy-Gold Correlation

| Item | Status | Path/Value |
|---|---|---|
| E3: correlation(train_reward, eval_EM) per checkpoint | ✅ | r=0.95-0.97 maintained (no degradation) |
| Goodhart mechanism confirmed as component gaming | ✅ | Not decorrelation |

---

## D. Qualitative Case Studies

### D1. Representative Trajectories (for paper examples)

| Item | Status | Notes |
|---|---|---|
| E3@500: 2-3 examples of "correct-via-memory" strategy (query entity, get relations, answer from memory) | ⏳ | Extract from saved trajectories |
| E3@1250: 2-3 examples of "kg-incomplete" failure (model queries hallucinated entity) | ⏳ | Extract from saved trajectories |
| E2@1200: 1-2 examples of tool abandonment (0 tool calls, direct answer) | ⏳ | Extract from saved trajectories |
| E1@1250: 1-2 examples of format failure (answer present but no `<answer>` tags) | ⏳ | Extract from saved trajectories |
| E5a best: examples showing `get_tail_entities` usage (if correct-via-tool > 0%) | ⏳ | After E5a eval |

---

## E. Experimental Setup Details (for Methodology section)

| Item | Status | Value |
|---|---|---|
| Base model | ✅ | Qwen2.5-7B-Instruct (SFT warmup) |
| Training algorithm | ✅ | GRPO |
| Dataset | ✅ | ComplexWebQuestions (CWQ), Freebase |
| Training steps | ✅ | 1293 (E1/E2/E3), ~490 (E5a), ~435 (E5b) |
| Reward split | ✅ | 0.30 × R_answer + 0.70 × R_step |
| R_answer formula | ✅ | 0.5 × EM + 0.5 × F1 |
| R_step (E1) | ✅ | 0 (no step reward) |
| R_step (E2) | ✅ | Entity overlap heuristic |
| R_step (E3) | ✅ | KG triple existence check (get_tail_relations non-empty) |
| R_step (E4) | ✅ | Uniform(0,1) random |
| R_step (E5a) | ✅ | Entity retrieval reward (see hpc_tasks.md Task 20) |
| R_step (E5b) | ✅ | Tool-type bonus (see hpc_tasks.md Task 21) |
| KG tools available | ✅ | 4: get_tail_relations, get_head_relations, get_tail_entities, get_head_entities |
| Max tool turns | ✅ | 5 (E1-E4), 5 (E5a/E5b, may change) |
| GPU | ✅ | NVIDIA GH200 120GB, 4 GPUs per experiment |
| Eval metric | ✅ | Strict EM (with-tools), Contains-EM/F1 (without-tools) |
| Test set size | ✅ | 500 samples (for checkpoint evals); full test set pending |

---

## F. Priority: What to Collect First

**For writing agent to start immediately (all already available):**
- All ✅ items above — compile into a single `paper_data_summary.md`
- Full Goodhart curves for E1/E2/E3 (raw numbers from conversation history)
- Trajectory classification table (E1/E2/E3)

**Collect before E5 results:**
- Task 14: Full test set + CIs for E1/E2/E3 (submit job now)
- Task 15: KL divergence (check veRL logs)
- Representative trajectories for case studies (from saved files)
- E1 tool action distribution

**Collect after E4 completes (~6h):**
- E4 full eval + trajectory classification

**Collect after E5a completes (~14h):**
- E5a eval at all checkpoints
- E5a trajectory classification (**critical: correct-via-tool rate**)
- E5a tool action distribution
- E5a vs E3 comparison

**Collect after E5b completes (~50h):**
- Same as E5a

---

## G. Notes for Writing Agent

1. **The paper title direction**: "Shortcuts All the Way Down: How Reward Design Shapes Tool Use Quality in Agentic RL"

2. **Core narrative**: Three reward types → three qualitatively different shortcut strategies. correct-via-tool = 0% everywhere with E1/E2/E3. Model has 4 tools but only uses cheapest one (get_tail_relations) as existence check. E5a/E5b test whether redesigned reward can induce genuine retrieval.

3. **Key finding to highlight**: E3 achieves 54.2% EM via verify-then-answer (not genuine KG reasoning). Goodhart occurs via component gaming (r>0.95 maintained). E2 abandons tools at step 1000 and memorizes. E1 never learns format.

4. **Novelty claims**:
   - First trajectory-level tool usage analysis of RL-trained KG agents
   - Novel Goodhart mechanism: component gaming without proxy-gold decorrelation
   - Controlled 4-way reward comparison on same model/task/algorithm

5. **Pending sections** (write placeholder, fill after E5):
   - E4/E5a/E5b results
   - Full test set CIs
   - E5 vs E3 comparison (the "fix" section)
