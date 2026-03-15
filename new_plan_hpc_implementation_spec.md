# HPC Implementation Specification

> **For**: Claude Code agent on Viking / Isambard HPC
> **From**: Research discussion repo (2026-02-27)
> **Project**: KG-Verifiable Process Rewards for Agentic Reasoning
> **Target deadline**: ARR May 2026 (May 25)

---

## 0. Context & Objective

We are building an agentic KG reasoning system where an LLM agent interacts with a Knowledge Graph via tool calls, trained with GRPO using three types of reward signals. The core research question: does reward verifiability affect Goodhart resistance and cross-KG generalization?

This is NOT a method paper. The system is infrastructure for an analytical study comparing reward types.

---

## 1. Framework: Use verl, NOT TRL

Do not use HuggingFace TRL for GRPO training. Use **verl** (Volcano Engine RL).

Reasons:
- TRL multi-turn GRPO has critical bug (GitHub #4543) in server mode
- TRL multi-turn features are experimental and unstable
- verl has native multi-turn support, used by Search-R1 and KG-R1
- verl provides built-in token masking (delta-based tokenization)
- verl supports turn-level rewards natively

References:
- verl: https://github.com/volcengine/verl
- verl-tool: https://github.com/TIGER-AI-Lab/verl-tool
- KG-R1 (closest template): https://github.com/Jinyeop3110/KG-R1
- Search-R1: https://github.com/PeterGriffinJin/Search-R1

Setup: verl + SGLang + DeepSpeed. Verify GH200 (ARM + NVIDIA Grace) compatibility on Isambard first.

---

## 2. Knowledge Graph Setup

### 2.1 KG Environments

| KG | Role | Priority |
|---|---|---|
| ConceptNet | Train env 1 | P0 (existing pipeline on Viking) |
| Freebase (WebQSP/CWQ) | Train env 2 | P0 (must build) |
| Wikidata (T-REx or KGQAGen-10k) | Cross-KG zero-shot test | P1 (if time allows) |

### 2.2 KG Retrieval Server

Unified KG query API server. Reference KG-R1's schema-agnostic API.

Three endpoints:
- `search(entity, relation)` -> list of (head, relation, tail) triples
- `search_reverse(entity, relation)` -> reverse lookup
- `get_relations(entity)` -> list of available relations

Implementation:
- ConceptNet: in-memory graph (NetworkX), ~3.4M triples
- Freebase: preprocessed subgraph from WebQSP/CWQ
- Runs as separate HTTP service, not embedded in training loop
- Precompute shortest path distances between entity pairs (for R_progress reward)

### 2.3 QA Datasets

| KG | QA Dataset | Hops | Split |
|---|---|---|---|
| ConceptNet | Self-generated | 1-3 hop | 4000/500/500 (existing) |
| Freebase | WebQSP + CWQ | 1-4 hop | Standard splits |
| Wikidata | T-REx or KGQAGen-10k | 1-3 hop | Standard splits |

KG is always fully accessible during train and test. Split is on questions, not triples.

---

## 3. Agent Interaction Design

### 3.1 ReAct-style Multi-turn

Each training sample is a multi-turn interaction:

```
Turn 1: Agent thinks -> generates KG query -> system executes query -> returns results
Turn 2: Agent thinks based on results -> generates next query -> system executes -> returns
...
Turn N: Agent thinks -> outputs final answer
```

Maximum turns: 5 (configurable). Agent can stop early by outputting answer.

### 3.2 Prompt Template

```
<system>
You are a reasoning agent with access to a Knowledge Graph.
Available tools:
- kg_search(entity, relation): Find triples matching (entity, relation, ?)
- kg_search_reverse(entity, relation): Find triples matching (?, relation, entity)
- kg_get_relations(entity): List all relations for an entity

Think step by step. Use tools to verify your reasoning. When ready, give the final answer.
</system>

<user>{question}</user>

<assistant>
<think>I need to find ... Let me query the KG.</think>
<tool_call>kg_search("dog", "IsA")</tool_call>
</assistant>

<tool_response>[("dog", "IsA", "animal"), ("dog", "IsA", "pet")]</tool_response>

<assistant>
<think>Dog is an animal. Now I need to find ...</think>
<tool_call>kg_search("animal", "HasProperty")</tool_call>
</assistant>

<tool_response>[("animal", "HasProperty", "breathe"), ...]</tool_response>

<assistant>
<think>Animals can breathe. This answers the question.</think>
<answer>breathe</answer>
</assistant>
```

### 3.3 Token Masking

verl's delta-based tokenization must be configured to mask:
- System prompt tokens
- Tool response tokens (KG query results)
- Only agent-generated tokens (think + tool_call + answer) contribute to policy gradient

---

## 4. Reward Functions (THREE variants)

This is the core experimental variable. Implement all three:

### 4.1 R_outcome (Baseline 1: Outcome-only)

```python
def reward_outcome(trajectory, ground_truth_answer):
    final_answer = extract_answer(trajectory)
    em = exact_match(final_answer, ground_truth_answer)
    f1 = token_f1(final_answer, ground_truth_answer)
    return 0.5 * em + 0.5 * f1
```

Only the final answer matters. No step-level signal. This is the Graph-R1 / KG-R1 baseline.

### 4.2 R_heuristic_step (Baseline 2: ProGraph-R1 style)

```python
def reward_heuristic_step(trajectory, ground_truth_answer):
    steps = parse_steps(trajectory)
    step_rewards = []
    for t, step in enumerate(steps):
        # Entity overlap between retrieved triples and ground truth
        retrieved_entities = extract_entities(step.observation)
        gt_entities = extract_entities(ground_truth_answer)
        r_overlap = len(retrieved_entities & gt_entities) / max(len(retrieved_entities), 1)

        # Answer reachability (does retrieved content mention answer entities?)
        r_reach = 1.0 if any(e in gt_entities for e in retrieved_entities) else 0.0

        step_rewards.append(0.5 * r_overlap + 0.5 * r_reach)

    r_answer = reward_outcome(trajectory, ground_truth_answer)
    return sum(step_rewards) / len(step_rewards) + r_answer
```

### 4.3 R_verifiable_step (Our method: KG-verifiable)

```python
def reward_verifiable_step(trajectory, ground_truth_answer, kg, gt_path):
    steps = parse_steps(trajectory)
    step_rewards = []
    prev_entity = extract_question_entity(trajectory)

    for t, step in enumerate(steps):
        query = parse_tool_call(step)

        # R_valid: Did the query return real KG triples?
        r_valid = 1.0 if len(step.observation) > 0 else 0.0

        # R_triple_on_path: Is any retrieved triple on the ground truth KG path?
        retrieved_triples = step.observation
        r_on_path = max(triple_in_path(tr, gt_path) for tr in retrieved_triples) if retrieved_triples else 0.0

        # R_progress: Graph distance to answer entity decreased?
        current_entity = extract_current_entity(step)
        answer_entity = extract_answer_entity(ground_truth_answer)
        dist_before = kg.shortest_path_distance(prev_entity, answer_entity)
        dist_after = kg.shortest_path_distance(current_entity, answer_entity)
        r_progress = max(0, (dist_before - dist_after) / max(dist_before, 1))

        # R_coherence: Current step shares entity with previous step?
        r_coherence = 1.0 if shares_entity(prev_entity, current_entity) else 0.0

        step_rewards.append(0.3 * r_valid + 0.3 * r_on_path + 0.2 * r_progress + 0.2 * r_coherence)
        prev_entity = current_entity

    r_answer = reward_outcome(trajectory, ground_truth_answer)
    return 0.4 * r_answer + 0.6 * (sum(step_rewards) / max(len(step_rewards), 1))
```

**Important**: The weights (0.3, 0.3, 0.2, 0.2 and 0.4/0.6) are initial values. May need tuning via small-scale experiments.

---

## 5. Training Pipeline

### 5.1 Stage 1: SFT Warmup

Teach the model the agent trajectory format (think-query-observe-answer loop).

- Generate SFT data by having a stronger model (e.g., Qwen2.5-72B or GPT-4) solve KG questions using the tool API
- Alternatively, construct gold trajectories from ground truth KG paths
- Train Qwen2.5-1.5B-Instruct and Qwen2.5-7B-Instruct
- 1-2 epochs, learning rate 2e-5, LoRA rank 64

### 5.2 Stage 2: GRPO Training

For each of the 3 reward variants, run GRPO on verl:

```yaml
# verl config (reference, adapt from KG-R1)
algorithm: grpo
model: Qwen/Qwen2.5-7B-Instruct  # or 1.5B
multi_turn: true
max_turns: 5
group_size: 8       # number of rollouts per question
learning_rate: 5e-7
kl_coeff: 0.01
max_steps: 1000
batch_size: 128
```

- Enable turn-level reward for R_heuristic_step and R_verifiable_step
- For R_outcome, use trajectory-level reward only (broadcast to all turns)
- Log all metrics to W&B: reward, EM, F1, reward-accuracy correlation, trajectory length

### 5.3 Stage 3: Evaluation

Metrics per model:
- EM, F1 (standard QA metrics)
- Mean reward per step (per reward type)
- Goodhart metric: Pearson correlation between training reward and EM over training steps
- Reward-accuracy alignment: scatter plot of step reward vs step correctness
- Cross-KG transfer: train on KG_A, evaluate on KG_B (zero-shot)

---

## 6. Experiment Matrix

### 6.1 Core Experiments (P0)

| Exp | Model | Train KG | Reward | Test KG | Purpose |
|-----|-------|----------|--------|---------|---------|
| E1 | 1.5B | ConceptNet | R_outcome | ConceptNet | Baseline |
| E2 | 1.5B | ConceptNet | R_heuristic | ConceptNet | Baseline |
| E3 | 1.5B | ConceptNet | R_verifiable | ConceptNet | Core comparison |
| E4 | 7B | ConceptNet | R_outcome | ConceptNet | Scale comparison |
| E5 | 7B | ConceptNet | R_heuristic | ConceptNet | Scale comparison |
| E6 | 7B | ConceptNet | R_verifiable | ConceptNet | Core at scale |
| E7 | 7B | Freebase | R_outcome | Freebase | Second KG baseline |
| E8 | 7B | Freebase | R_heuristic | Freebase | Second KG baseline |
| E9 | 7B | Freebase | R_verifiable | Freebase | Second KG core |

### 6.2 Cross-KG Transfer (P0)

| Exp | Model | Train KG | Reward | Test KG | Purpose |
|-----|-------|----------|--------|---------|---------|
| E10 | 7B | ConceptNet | R_verifiable | Freebase | Cross-KG transfer |
| E11 | 7B | Freebase | R_verifiable | ConceptNet | Reverse transfer |
| E12 | 7B | ConceptNet | R_outcome | Freebase | Baseline transfer |
| E13 | 7B | ConceptNet | R_heuristic | Freebase | Baseline transfer |

### 6.3 Extended (P1, if time/compute allows)

| Exp | Purpose |
|-----|---------|
| E14-16 | Wikidata zero-shot transfer |
| E17-19 | Reward weight ablations |
| E20 | 14B model (if Isambard allows) |

### 6.4 Estimated Compute

| Experiment group | GPU-hours (GH200) |
|---|---|
| SFT warmup (1.5B + 7B, 2 KGs) | ~20h |
| E1-E3 (1.5B, ConceptNet) | ~30h |
| E4-E9 (7B, 2 KGs, 3 rewards) | ~300h |
| E10-E13 (cross-KG) | ~50h (eval only, no training) |
| Debug/failed runs buffer | ~100h |
| **Total P0** | **~500h** |

Well within the 10,000h Isambard allocation.

---

## 7. Data Collection for Analysis

Beyond standard metrics, collect these for the paper's analytical contributions:

### 7.1 Goodhart Analysis Data
- Every 50 training steps: save checkpoint, evaluate on val set, record (reward_mean, EM, F1)
- Plot reward vs EM/F1 curves to identify Goodhart divergence points
- Compare divergence points across the three reward types

### 7.2 Reward Hacking Evidence
- Save 100 random trajectories at training step 0, 250, 500, 750, 1000 for each reward type
- Manually (or LLM-judge) classify hacking behaviors:
  - Token-level hacking: KG keywords without reasoning
  - Length hacking: abnormal trajectory length patterns
  - Path fabrication: queries for non-existent entities/relations
  - Shortcut exploitation: skipping intermediate reasoning steps
- Record trajectory length distribution over training

### 7.3 Step-level Verification Data
- For R_verifiable_step: log per-step (r_valid, r_on_path, r_progress, r_coherence) for every trajectory
- Compute correlation between individual step reward components and final answer correctness

---

## 8. Compute Resources

### Viking HPC (University of York)
- GPU: NVIDIA H100 80GB PCIe
- Use for: development, debugging, 1.5B experiments, ConceptNet pipeline
- Conda env: `kg_align` (existing)
- Scratch: `/mnt/scratch/users/ts1201/`

### Isambard HPC (access from ~Mar 6)
- GPU: NVIDIA GH200 (Grace Hopper)
- Allocation: up to 10,000 GPU-hours
- Max concurrent: 4 nodes, 32 GPUs
- Use for: all 7B experiments, cross-KG experiments
- **First task on Isambard**: verify verl + SGLang install on GH200 architecture

---

## 9. Implementation Timeline

```
Week 1 (Mar 3-7): Environment setup
  - Isambard access + verl/SGLang install verification
  - Download & preprocess Freebase (WebQSP/CWQ)
  - Clone KG-R1 codebase, understand verl config structure
  - Implement KG retrieval server (ConceptNet + Freebase)

Week 2-3 (Mar 10-21): Core implementation
  - Implement 3 reward functions
  - Generate SFT training data (agent trajectory format)
  - SFT warmup on ConceptNet (1.5B on Viking)
  - Small-scale GRPO validation (1.5B, 100 samples, ConceptNet)

Week 4-5 (Mar 24 - Apr 4): ConceptNet full experiments
  - E1-E6: all reward variants x both model sizes
  - Begin Goodhart analysis data collection
  - Begin hacking evidence collection

Week 6-7 (Apr 7-18): Freebase + Cross-KG
  - E7-E9: Freebase experiments
  - E10-E13: cross-KG transfer
  - Extended experiments if compute allows

Week 8+ (Apr 21+): Analysis handoff
  - Package all results for analysis
  - Research discussion repo will provide analysis specifications
```

---

## 10. Deliverables Checklist

- [ ] verl environment verified on both Viking and Isambard
- [ ] KG retrieval server running for ConceptNet and Freebase
- [ ] SFT data generated and models fine-tuned
- [ ] 3 reward functions implemented and unit-tested
- [ ] E1-E9 core experiments completed with W&B logs
- [ ] E10-E13 cross-KG transfer experiments completed
- [ ] Goodhart analysis data collected (checkpoints + eval metrics every 50 steps)
- [ ] Hacking evidence trajectories saved (100 samples x 5 checkpoints x 3 rewards)
- [ ] Step-level reward logs for all R_verifiable_step runs
- [ ] All results packaged in structured format for paper writing

---

## 11. Risk Mitigation

| Risk | Mitigation |
|---|---|
| verl incompatible with GH200 | Fallback: use Viking H100 for all experiments (slower but works) |
| Freebase data preprocessing issues | Use MetaQA as simpler alternative (despite quality concerns, it is widely used) |
| GRPO multi-turn diverges | Cap max turns at 3; use smaller group size; reference KG-R1 hyperparameters |
| Reward weights need extensive tuning | Start with KG-R1/ProGraph-R1 defaults; tune on 1.5B first (cheap) |
| SFT data quality insufficient | Generate more data using GPT-4; or construct gold trajectories from GT paths |
