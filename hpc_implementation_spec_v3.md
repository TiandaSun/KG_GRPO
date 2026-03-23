# HPC Implementation Specification (v3)

> **For**: Claude Code agent on Viking / Isambard HPC
> **From**: Research discussion repo
> **Last updated**: 2026-03-21
> **Project**: Verifiable Process Supervision via Knowledge Graphs
> **Target**: EMNLP 2026 (deadline May 25)

---

## 0. Context & Objective

### Core Research Question

> When and why does deterministic step-level verification outperform heuristic rewards in RL post-training? We use KG reasoning as the ideal testbed because KG uniquely provides free, deterministic, step-level ground truth via triple existence checks.

### Key Reframing

**KG is NOT a knowledge source — it is a verification oracle.**

The model does not need KG to "know" facts (LLMs already know most KG content). KG's unique value is providing exact, zero-cost, binary verification of each reasoning step. This is something no other domain (math, code, general text) can provide as cheaply.

### What We Are Building

An agentic KG reasoning system trained with GRPO using three reward types (outcome / heuristic-step / KG-verifiable-step), evaluated across multiple KGs and tasks, with analysis of Goodhart resistance, reward hacking, and transferability of learned reasoning patterns.

---

## 1. Framework: verl (NOT TRL)

Use **verl** (Volcano Engine RL) for all GRPO training.

Reasons:
- TRL multi-turn GRPO has critical bug (#4543) in server mode
- verl has native multi-turn support, used by Search-R1 and KG-R1
- verl provides built-in token masking (delta-based tokenization)
- verl supports turn-level rewards natively

References:
- verl: https://github.com/volcengine/verl
- verl-tool: https://github.com/TIGER-AI-Lab/verl-tool
- KG-R1 (closest template): https://github.com/Jinyeop3110/KG-R1
- Search-R1: https://github.com/PeterGriffinJin/Search-R1

Setup: verl + vLLM + DeepSpeed (already verified on Isambard GH200).

---

## 2. Models

| Role | Model | Purpose |
|---|---|---|
| **Primary** | Qwen2.5-7B-Instruct | All tasks, all rewards, all KGs, full analysis |
| **Scaling** | Qwen2.5-14B-Instruct | Core QA task, verify findings hold at scale |
| **Ablation** (optional) | Qwen2.5-1.5B-Instruct | Show "too small fails" for multi-turn agent |

Compute budget: 2,500 node-hours x 4 GH200/node = 10,000 GPU-hours (Isambard) + Viking H100 (supplementary).

---

## 3. CRITICAL CHANGE (v3): Dataset Pivot — CWQ/Freebase as Primary

### 3.1 Why We Are Switching from ConceptNet to CWQ/Freebase

The first training run on ConceptNet (Job 3244577) revealed fundamental data quality issues:

1. **Gold answers too verbose** (mean 37 words). EM is near-useless; F1 rewards incidental word overlap.
2. **1-hop questions (39.5% of data) are trivially solvable** by 7B without tools. Tool use becomes genuinely suboptimal from the model's perspective.
3. **ConceptNet knowledge is fully internalized** by modern LLMs — the model doesn't need to query it.

CWQ (ComplexWebQuestions) on Freebase resolves all three issues:
- Answers are short KG entities (clean EM/F1)
- Questions are 2-4 hop with 4 composition types (conjunction, composition, comparison, superlative)
- Freebase contains specialized knowledge not fully memorized by LLMs
- KG-R1, Graph-RFT, and Explore-on-Graph all use CWQ — directly comparable
- CWQ has 24,649 training samples (6x our ConceptNet set)

### 3.2 KG Environments (Revised)

| KG | Role | Priority | Status |
|---|---|---|---|
| **Freebase** (via CWQ/WebQSP) | **Primary train + eval** | **P0** | Must build this week |
| **Wikidata** (T-REx or KGQAGen-10k) | Cross-KG zero-shot test | P1 | After core experiments |
| **ConceptNet** | Dev/debug only; first run data for hacking taxonomy | Demoted | Already built |

### 3.3 Freebase Setup

**Data sources**:
- CWQ dataset: https://www.dropbox.com/sh/7pkwn7wslzrt5g0/AACnTkBxcjhCRB-3CBCGKqS5a (or via KG-R1 repo preprocessing)
- WebQSP dataset: standard download
- Freebase subgraph: use the preprocessed version from KG-R1 or GrailQA projects (NOT full Freebase dump)

**Reference**: Follow KG-R1's data preprocessing pipeline as closely as possible. Their Freebase subgraph + CWQ setup is exactly our use case.

**KG Server**: Add Freebase backend to existing KG server with the same 5 endpoints:
- search(entity, relation) -> List[Triple]
- search_reverse(entity, relation) -> List[Triple]
- get_relations(entity) -> List[str]
- verify(head, relation, tail) -> bool
- shortest_path(entity_a, entity_b) -> List[Triple]

### 3.4 QA Datasets (Revised)

| KG | Dataset | Hops | Train | Test | Answer Format |
|---|---|---|---|---|---|
| Freebase | CWQ | 2-4 hop | 24,649 | 3,531 | Short KG entity |
| Freebase | WebQSP | 1-2 hop | 2,826 | 1,639 | Short KG entity |
| Wikidata | T-REx / KGQAGen-10k | 1-3 hop | Standard | Standard | Short entity |

### 3.5 Difficulty Filtering & Stratification

#### Step 1: Extract Hop Count from SPARQL Annotations

CWQ provides gold SPARQL queries. Extract hop count (number of triple patterns) for each question. Expected distribution: ~40% 2-hop, ~35% 3-hop, ~25% 4-hop.

Also tag each question with its composition type from CWQ metadata: composition, conjunction, comparison, superlative.

#### Step 2: Base Model Performance Filtering

Run SFT-only model (no GRPO) on CWQ train set in TWO modes:

```
Mode A: WITHOUT KG access (parametric knowledge only)
  → Measures: does the model already know the answer?

Mode B: WITH KG access (full tool use)
  → Measures: can the model answer with KG help?
```

This produces a 2x2 classification for each question:

| | KG=Correct | KG=Wrong |
|---|---|---|
| **No-KG=Correct** | Category 1: Too easy, KG unnecessary | Category 2: KG hurts (noise) |
| **No-KG=Wrong** | **Category 3: KG adds value** | Category 4: Hard for all |

#### Step 3: GRPO Training Set Construction

```
EXCLUDE: Category 1 (model already knows → RL signal wasted, tool use suboptimal)
INCLUDE: Category 3 (KG adds value → ideal for training tool use)
INCLUDE: Category 4 (hard → RL should learn to improve)
DISCUSS: Category 2 (KG hurts → analyze why; possible KG noise issue)
```

Log the filtering ratio and category distribution for the paper. The Category 1 fraction directly answers "what fraction of CWQ is trivially solvable by 7B without KG?"

#### Step 4: Hop-Stratified Training (Kansal & Jha Style)

Following Kansal & Jha (arXiv:2601.15160) who train on 1-3 hop and test on unseen 4-5 hop:

```
Filtered GRPO training set:
  EXCLUDE: 0-hop samples (no KG path → step rewards undefined)
  EXCLUDE: Category 1 from Step 3 (model already knows)
  TRAIN ON: 2-3 hop only (from remaining samples)

Evaluation (FULL CWQ test, stratified):
  2-hop: seen complexity
  3-hop: seen complexity
  4-hop: UNSEEN complexity → tests compositional generalization
```

This enables the key analysis:

```
Table: Performance by hop count (7B, CWQ test)

| Reward Type     | 2-hop EM | 3-hop EM | 4-hop EM (unseen) | Delta(4h-2h) |
|-----------------|----------|----------|--------------------|-------------|
| R_outcome       |    ...   |    ...   |       ...          |     ...     |
| R_heuristic     |    ...   |    ...   |       ...          |     ...     |
| R_verifiable    |    ...   |    ...   |       ...          |     ...     |
| R_random        |    ...   |    ...   |       ...          |     ...     |
| SFT-only (base) |    ...   |    ...   |       ...          |     ...     |

Hypothesis: R_verifiable's Delta(4h-2h) > R_outcome's Delta(4h-2h)
→ Verifiable reward enables better compositional generalization
```

Also report by CWQ composition type (conjunction/composition/comparison/superlative).

#### Step 5: Comparison — Unfiltered vs Filtered

E1/E3 full-data runs (Jobs 3254091/3254653) serve as unfiltered baseline. After filtered re-runs:

```
Table: Effect of difficulty filtering on reward comparison

| Setting     | R_outcome EM | R_verifiable EM | Delta |
|-------------|-------------|-----------------|-------|
| Unfiltered  | (from E1)   | (from E3)       |  ...  |
| Filtered    | (from E1')  | (from E3')      |  ...  |

If filtered Delta > unfiltered Delta → filtering amplifies verifiable reward advantage
```

#### Effort Estimate

| Task | Effort |
|---|---|
| Extract hop counts from SPARQL | ~1h (scripting) |
| Base model inference Mode A (no KG) | ~1h on 1 GPU |
| Base model inference Mode B (with KG) | ~2h on 1 GPU |
| Classify + filter + build filtered dataset | ~30min |
| **Total** | **~4.5h** (run before filtered GRPO re-runs) |

### 3.6 ConceptNet's Remaining Role

ConceptNet is NOT discarded — its data serves two purposes:
1. **Quick dev/debug environment**: test reward functions and pipeline changes before expensive Freebase runs
2. **Hacking taxonomy evidence**: the tool-use collapse from Job 3244577 (step 50 vs 186 checkpoints) is valuable paper data showing Goodhart effects under ad-hoc rewards

---

## 4. Agent Interaction Design

### 4.1 ReAct-style Multi-turn (unchanged from v2)

```
User: {question}
Agent: <think>reasoning</think>
       <tool_call>kg_search("entity", "relation")</tool_call>
System: [("head", "relation", "tail"), ...]
Agent: <think>reasoning based on results</think>
       <tool_call>kg_search("entity2", "relation2")</tool_call>
System: [...]
Agent: <think>conclusion</think>
       <answer>entity_answer</answer>
```

Max turns: 5 (configurable). Agent can stop early.

### 4.2 Token Masking

System prompt + tool response tokens masked from policy gradient (verl delta-based tokenization). Only agent-generated tokens contribute to gradient.

---

## 5. Reward Functions (FOUR variants)

### 5.1 R_outcome (Baseline 1: Outcome-only)

```python
def reward_outcome(trajectory, gold_answer):
    predicted = extract_answer_tag(trajectory)
    em = exact_match(predicted, gold_answer)
    f1 = token_f1(predicted, gold_answer)
    return 0.5 * em + 0.5 * f1
```

Note: CWQ gold answers are short KG entities. EM is now meaningful (unlike ConceptNet verbose answers).

### 5.2 R_heuristic_step (Baseline 2: ProGraph-R1 style)

```python
def reward_heuristic_step(trajectory, gold_answer):
    steps = parse_tool_call_steps(trajectory)
    step_rewards = []
    for step in steps:
        retrieved_entities = extract_entities(step.observation)
        gt_entities = extract_entities(gold_answer)
        r_overlap = len(retrieved_entities & gt_entities) / max(len(retrieved_entities), 1)
        r_reach = 1.0 if any(e in gt_entities for e in retrieved_entities) else 0.0
        step_rewards.append(0.5 * r_overlap + 0.5 * r_reach)
    r_answer = reward_outcome(trajectory, gold_answer)
    return 0.3 * r_answer + 0.7 * (sum(step_rewards) / max(len(step_rewards), 1))
```

### 5.3 R_verifiable_step (Our method: KG-verifiable)

```python
def reward_verifiable_step(trajectory, gold_answer, kg, gt_path):
    steps = parse_tool_call_steps(trajectory)
    step_rewards = []
    prev_entity = extract_question_entity(trajectory)

    for step in steps:
        # r_on_path (0.45): Is retrieved triple on the ground truth KG path?
        retrieved_triples = step.observation
        r_on_path = max(
            (triple_in_path(tr, gt_path) for tr in retrieved_triples), default=0.0
        )

        # r_progress (0.30): Graph distance to answer decreased?
        current_entity = extract_current_entity(step)
        answer_entity = extract_answer_entity(gold_answer)
        dist_before = kg.shortest_path_distance(prev_entity, answer_entity)
        dist_after = kg.shortest_path_distance(current_entity, answer_entity)
        r_progress = max(0, (dist_before - dist_after) / max(dist_before, 1))

        # r_coherence (0.15): Step shares entity with previous?
        r_coherence = 1.0 if shares_entity(prev_entity, current_entity) else 0.0

        # r_valid (0.10): Query returned results?
        r_valid = 1.0 if len(retrieved_triples) > 0 else 0.0

        step_rewards.append(
            0.45 * r_on_path + 0.30 * r_progress + 0.15 * r_coherence + 0.10 * r_valid
        )
        prev_entity = current_entity

    r_answer = reward_outcome(trajectory, gold_answer)
    return 0.3 * r_answer + 0.7 * (sum(step_rewards) / max(len(step_rewards), 1))
```

**Key changes from v2**:
- r_on_path weight increased: 0.30 -> 0.45 (core verifiable signal)
- r_valid weight decreased: 0.30 -> 0.10 (near-constant on large KGs)
- r_progress weight: 0.20 -> 0.30
- answer/step split: 0.40/0.60 -> 0.30/0.70 (stronger tool-use pressure)

### 5.4 R_random (Ablation: responding to Spurious Rewards)

```python
def reward_random(trajectory, gold_answer):
    steps = parse_tool_call_steps(trajectory)
    step_rewards = [random.uniform(0, 1) for _ in steps]
    r_answer = reward_outcome(trajectory, gold_answer)
    return 0.3 * r_answer + 0.7 * (sum(step_rewards) / max(len(step_rewards), 1))
```

### 5.5 IMPORTANT: No Explicit Tool Bonuses

Do NOT add explicit r_tool_use bonuses or r_no_tool penalties. The first ConceptNet run proved these create unstable optimization and tool-use collapse. Tool use should be incentivized IMPLICITLY: step rewards (r_on_path, r_progress, r_coherence) are only nonzero when tools are used, so the 0.70 step weight naturally rewards tool use.

---

## 6. Training Pipeline

### 6.1 SFT Warmup

Generate SFT data for CWQ:
- Use gold Freebase paths to construct agent trajectories
- Or use a stronger model (Qwen2.5-72B) to solve CWQ questions via the KG API
- Format: ReAct-style (think + tool_call + observation + answer)

Config: 1-2 epochs, lr 2e-5, LoRA rank 64.

### 6.2 GRPO Training

```yaml
algorithm: grpo
multi_turn: true
max_turns: 5
group_size: 8
learning_rate: 5e-7    # REDUCED from first run's 5e-6 (too high, contributed to collapse)
kl_coeff: 0.01
max_steps: 1000
batch_size: 128
```

**Learning rate**: Use 5e-7 (not 5e-6). The first run's high lr likely accelerated collapse.

Log to W&B: reward, EM, F1, reward-accuracy correlation, trajectory length, num_tool_calls, per-step reward components.

### 6.3 Evaluation

Standard: EM, F1 (on short KG entity answers)

Analysis metrics:
- Goodhart: Pearson(training_reward, EM) over steps
- Cross-KG transfer: train Freebase, eval Wikidata (zero-shot)
- Cross-dataset: train CWQ, eval WebQSP

Without-KG evaluation:
- Same test questions, KG API disabled
- Measure: decomposition granularity, step factual accuracy (LLM-judge), verifiability score
- Compare across reward types

---

## 7. Experiment Matrix (Revised for CWQ)

### 7.1 P0: Core QA Experiments (CWQ/Freebase)

**Phase A — Unfiltered (COMPLETED):**

| ID | Train Data | Reward | Status | Key Result |
|---|---|---|---|---|
| E1 | CWQ full | R_outcome | Done (Job 3254091, 1260/1293 steps) | 39 KG requests (no tool use) |
| E3 | CWQ full | R_verifiable | Done (Job 3254653, 1260/1293 steps) | 841K KG requests (sustained tool use) |

**Phase B — Unfiltered remaining + offline eval:**

| ID | Task | Purpose |
|---|---|---|
| E1-eval | Offline eval E1 checkpoints (steps 50-1250) on CWQ val+test | Get EM/F1 learning curves |
| E3-eval | Offline eval E3 checkpoints (steps 50-1250) on CWQ val+test | Get EM/F1 learning curves |
| E3-eval-hop | Stratify E3 eval by SPARQL hop count | Test if R_verifiable advantage grows with hop |
| E2 | CWQ full, R_heuristic, 7B | Complete 3-way unfiltered comparison |

**Phase C — Filtered (after hop analysis):**

| ID | Train Data | Reward | Eval | Purpose |
|---|---|---|---|---|
| E1' | CWQ filtered (2-3 hop, exclude Cat1+0-hop) | R_outcome | CWQ test (all hops) | Filtered baseline |
| E2' | CWQ filtered | R_heuristic | CWQ test (all hops) | Filtered heuristic |
| E3' | CWQ filtered | R_verifiable | CWQ test (all hops) | Filtered core — 4-hop is UNSEEN |
| E4 | E1' model | R_outcome | WebQSP | Cross-dataset |
| E5 | E2' model | R_heuristic | WebQSP | Cross-dataset |
| E6 | E3' model | R_verifiable | WebQSP | Cross-dataset |
| E7 | E3' model | R_verifiable | Without-KG | Reasoning quality |
| E8 | E1' model | R_outcome | Without-KG | Reasoning quality comparison |

**14B (scaling):**

| ID | Train | Reward | Eval | Purpose |
|---|---|---|---|---|
| E9 | CWQ | R_outcome | CWQ test | Scale baseline |
| E10 | CWQ | R_heuristic | CWQ test | Scale baseline |
| E11 | CWQ | R_verifiable | CWQ test | Scale core |
| E12 | CWQ | R_verifiable | WebQSP | Scale cross-dataset |

### 7.2 P1: Cross-KG + Extended Tasks

| ID | Model | Task | Details |
|---|---|---|---|
| E13 | 7B | Cross-KG | CWQ-trained -> Wikidata zero-shot |
| E14-E16 | 7B | Claim Verification | 3 reward types on Freebase claims |
| E17-E19 | 7B | Think-Verify | 3 reward types |

### 7.3 P2: Ablations

| ID | Model | Details |
|---|---|---|
| E20 | 7B | R_random on CWQ (Spurious Rewards response) |
| E21 | 1.5B | R_verifiable on CWQ (scale ablation) |
| E22 | 14B | Think-Verify with R_verifiable |

### 7.4 Compute Estimate

| Group | GPU-hours (GH200) |
|---|---|
| SFT warmup (7B + 14B on CWQ) | ~100h |
| P0 core 7B (E1-E8) | ~600h |
| P0 scaling 14B (E9-E12) | ~800h |
| P1 extended (E13-E19) | ~600h |
| P2 ablations (E20-E22) | ~200h |
| Debug / failed / hyperparam tuning | ~1,000h |
| **Total** | **~3,300h** |

Well within 10,000h budget.

---

## 8. Data Collection for Paper Analysis

### 8.1 Goodhart Analysis
- Save checkpoint every 50 training steps
- At each checkpoint: eval on CWQ val, record (reward_mean, EM, F1)
- Plot reward vs EM/F1 to identify Goodhart divergence
- Compare divergence across reward types

### 8.2 Reward Hacking Evidence
- Save 100 random trajectories at steps 0, 250, 500, 750, 1000
- Classify hacking modes: token-level, length, path fabrication, shortcut, tool avoidance
- Include ConceptNet Job 3244577 data as additional hacking case study

### 8.3 Without-KG Reasoning Quality
- Run all trained models on CWQ test WITHOUT KG API
- Measure: decomposition granularity, grounding score, factual accuracy (LLM-judge), verifiability
- Key hypothesis: R_verifiable-trained model produces more structured reasoning than R_outcome-trained

---

## 9. Implementation Timeline (Revised for Actual Build Speed)

**Observed**: CWQ/Freebase pipeline (data + KG server + SFT + reward functions + E1/E3 full training) completed in ~2 days, not the planned 7 days. Timeline adjusted accordingly.

**Owner note**: Tianda on vacation Mar 30 - Apr 5, available 1-2 days only. HPC agent should run autonomous tasks.

```
=== COMPLETED ===

Mar 21-22: CWQ/Freebase setup (done in 2 days, planned 7)
  [x] Freebase subgraph + CWQ data downloaded and preprocessed
  [x] KG server with 5 endpoints running
  [x] SFT warmup on CWQ (7B)
  [x] R_outcome and R_verifiable implemented
  [x] E1 full training (Job 3254091, 1260 steps, 19h)
  [x] E3 full training (Job 3254653, 1260 steps, 22.5h)

=== THIS WEEK: Mar 22-28 — Offline Eval + Filtering + E2 ===

Day 1 (Mar 22-23): Offline eval of E1 + E3 checkpoints [HIGHEST PRIORITY]
  - Evaluate steps {50, 250, 500, 750, 1000, 1250} on CWQ val+test
  - Produce EM/F1 learning curves for both E1 and E3
  - This answers: "Does R_verifiable actually improve answer quality?"
  -> If E3 EM > E1 EM: core hypothesis CONFIRMED
  -> If E3 EM ≈ E1 EM: tool use works but doesn't help yet — need filtering

Day 1-2 (parallel): SPARQL hop count extraction
  - Parse CWQ gold SPARQL → extract true hop count per question
  - Analyze: what fraction is 0-hop, 1-hop, 2-hop, 3-hop, 4-hop?
  - Stratify E1/E3 eval results by hop count
  - Report: does R_verifiable advantage increase with hop count?

Day 2 (Mar 23-24): E2 (R_heuristic) full training
  - Implement R_heuristic reward function
  - Launch E2 on full CWQ (same config as E1/E3)
  - ~22h runtime, results by Mar 25

Day 3 (Mar 24-25): Build filtered dataset + SFT model A/B classification
  - Run SFT model Mode A (no KG) + Mode B (with KG) on CWQ train
  - Classify into Categories 1-4
  - Build filtered dataset: exclude 0-hop + Category 1, train only ≤3-hop
  - Generate new SFT data for filtered set

Day 4-5 (Mar 25-27): Filtered experiments E1'/E2'/E3'
  - SFT warmup on filtered data (~4h)
  - Launch E3' (R_verifiable, filtered) — most important
  - Launch E1' (R_outcome, filtered) — baseline
  - Launch E2' (R_heuristic, filtered) — if time
  Each ~20h, can run 2 in parallel on separate nodes

  CHECKPOINT Mar 27: Do we have 3-way filtered comparison?
  -> If yes: proceed to cross-dataset + 14B
  -> If partially: continue E2' into vacation week

Day 6-7 (Mar 27-28): Start cross-dataset eval + trajectory analysis
  - E4-E6: eval filtered models on WebQSP
  - E7-E8: without-KG eval
  - Save 100 trajectories from E3' for hacking taxonomy

=== VACATION: Mar 30 - Apr 5 [HPC agent autonomous] ===

  - Complete any remaining E1'/E2'/E3' if not finished
  - Run E2 offline eval (if E2 training finished)
  - Start 14B SFT warmup on filtered CWQ
  - Collect Goodhart data from all checkpoints

=== Week 3: Apr 7-11 — 14B + Deep Analysis ===

  - E9-E11 (14B, 3 reward types, filtered CWQ)
  - E12 (14B cross-dataset WebQSP)
  - Goodhart analysis: reward vs EM/F1 divergence curves
  - Hacking taxonomy: classify trajectory samples

=== Week 4-5: Apr 14-25 — Extended + Ablations ===

  - E13 (Wikidata cross-KG if data ready)
  - E14-E19 (Claim Verification + Think-Verify)
  - E20-E22 (ablations: R_random, 1.5B, 14B Think-Verify)

=== Week 6-8: Apr 28 - May 25 — Writing + Submission ===

  - Verifiability framework + Goodhart analysis writeup
  - Hacking taxonomy section
  - Full paper draft
  - Advisor review
  - Submit EMNLP 2026 (May 25)
```

---

## 10. Compute Resources

### Isambard HPC
- GPU: NVIDIA GH200 (Grace Hopper), 4 per node
- Verified: verl 0.7.1 + vLLM 0.12.0 + DeepSpeed 0.18.8
- Allocation: 2,500 node-hours = 10,000 GPU-hours
- Role: ALL GRPO experiments (7B and 14B)

### Viking HPC (University of York)
- GPU: NVIDIA H100 80GB PCIe
- Conda env: `kg_align`
- Scratch: `/mnt/scratch/users/ts1201/`
- Role: development, debugging, SFT, evaluation, ConceptNet quick tests

---

## 11. Deliverables

### Completed
- [x] verl environment verified on Isambard GH200
- [x] ConceptNet KG server + SFT + first GRPO run (Goodhart/collapse evidence)
- [x] Freebase KG server running with 5 endpoints
- [x] CWQ data preprocessed (train/val/test)
- [x] SFT warmup on CWQ (7B)
- [x] R_outcome, R_verifiable implemented
- [x] E1 full training (unfiltered, 1260 steps) — 39 KG requests
- [x] E3 full training (unfiltered, 1260 steps) — 841K KG requests, no collapse

### This Week (Mar 22-28)
- [ ] **Offline eval E1+E3 checkpoints → EM/F1 learning curves** [HIGHEST PRIORITY]
- [ ] **Extract SPARQL hop counts → stratify E1/E3 results by hop**
- [ ] **E2 (R_heuristic) full training**
- [ ] **R_heuristic reward function implemented**
- [ ] Build filtered CWQ dataset (exclude 0-hop + Category 1, train ≤3-hop only)
- [ ] E1'/E3' filtered training
- [ ] E4-E8 cross-dataset + without-KG eval

### After Vacation (Apr 7+)
- [ ] E2' (R_heuristic filtered)
- [ ] E9-E12 (14B scaling)
- [ ] E13 (Wikidata cross-KG)
- [ ] E14-E19 (Claim Verification + Think-Verify)
- [ ] E20-E22 (ablations: R_random, 1.5B, 14B Think-Verify)
- [ ] Goodhart analysis across all checkpoints
- [ ] Hacking taxonomy (ConceptNet collapse + CWQ trajectory analysis)
- [ ] Without-KG reasoning quality eval
- [ ] All results packaged for paper

---

## 12. Risk Mitigation

| Risk | Mitigation |
|---|---|
| Freebase subgraph too large for memory | Use KG-R1's preprocessed subgraph (tailored to CWQ entities) |
| CWQ entity linking to Freebase fails | Use gold SPARQL annotations in CWQ (entity IDs provided) |
| Tool-use collapse recurs on CWQ | Implicit tool incentive via 0.30/0.70 split; lr=5e-7; NO explicit tool bonuses |
| Core hypothesis fails | Reframe as negative result with analysis of why |
| 14B too slow on 4x GH200 | Use LoRA or reduce batch size; focus on 7B results |
| Tianda unavailable Mar 30 - Apr 5 | HPC agent runs E1 + E3 autonomously; results reviewed on return |
| KG incompleteness | Measure false negative rate on Freebase; discuss in paper |

---

## 13. Lessons from First Run (ConceptNet, Job 3244577)

These lessons MUST be applied to all future experiments:

1. **NO explicit tool bonuses/penalties** — r_tool_use and r_no_tool caused collapse
2. **Use short entity answers for EM/F1** — verbose answers make EM useless and F1 noisy
3. **lr=5e-7, not 5e-6** — high lr accelerates reward hacking
4. **Implicit tool incentive via answer/step weight split (0.30/0.70)** — max no-tool reward = 0.30, making tool use the optimal strategy
5. **Filter easy questions from RL training** — if model already knows the answer, RL signal is wasted
6. **The collapse itself is valuable data** — use step 50 vs 186 checkpoints for hacking taxonomy

---

## Appendix: ConceptNet Setup (Retained for Dev/Debug)

ConceptNet remains available for quick iteration:
- KG server: port 8421, 99,915 nodes, 146,553 edges
- SFT model: `outputs/verl-sft-7b-merged/`
- First GRPO run: Job 3244577, checkpoints at steps 50/100/150/186
- Use for: testing reward function code, verifying pipeline changes, NOT for paper results
