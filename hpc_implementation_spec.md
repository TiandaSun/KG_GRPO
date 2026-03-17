# HPC Implementation Specification (v2)

> **For**: Claude Code agent on Viking / Isambard HPC
> **From**: Research discussion repo
> **Last updated**: 2026-03-16
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

Setup: verl + SGLang + DeepSpeed.

**FIRST TASK on Isambard**: Verify verl + SGLang compatibility with GH200 (ARM + NVIDIA Grace architecture). Do this before any other work.

---

## 2. Models

### Model Strategy

| Role | Model | Purpose |
|---|---|---|
| **Primary** | Qwen2.5-7B-Instruct | All tasks, all rewards, all KGs, full analysis |
| **Scaling** | Qwen2.5-14B-Instruct | Core QA task, verify findings hold at scale |
| **Ablation** (optional) | Qwen2.5-1.5B-Instruct | Show "too small fails" for multi-turn agent |

Why Qwen2.5: native function calling support, used by KG-R1/Search-R1, continuity with prior experiments, verl has templates.

### Compute Budget

- Isambard: 2,500 node-hours x 4 GH200/node = 10,000 GPU-hours
- Viking: H100 80GB (supplementary, for development and debugging)

---

## 3. Knowledge Graph Setup

### 3.1 KG Environments

| KG | Role | Source | Priority |
|---|---|---|---|
| **ConceptNet** | Train env 1 | Existing pipeline on Viking | P0 (ready) |
| **Freebase** (WebQSP/CWQ) | Train env 2 | Standard KGQA datasets | P0 (must build) |
| **Wikidata** (T-REx or KGQAGen-10k) | Cross-KG zero-shot test | High quality benchmark | P1 |

NOTE: Do NOT use MetaQA — a 2025 audit found only 20% factual accuracy (arXiv:2505.23495).

### 3.2 KG Retrieval Server

Unified API server supporting all KGs with the same interface.

Five endpoints (expanded from original three):

```python
class KGServer:
    def search(self, entity: str, relation: str) -> List[Triple]:
        """Forward search: (entity, relation, ?) -> matching triples"""

    def search_reverse(self, entity: str, relation: str) -> List[Triple]:
        """Reverse search: (?, relation, entity) -> matching triples"""

    def get_relations(self, entity: str) -> List[str]:
        """List all relations connected to an entity"""

    def verify(self, head: str, relation: str, tail: str) -> bool:
        """Check if a specific triple exists in the KG.
        This is the core 'verification oracle' endpoint."""

    def shortest_path(self, entity_a: str, entity_b: str) -> List[Triple]:
        """Find shortest path between two entities.
        Used by agent for reasoning planning."""
```

Implementation:
- ConceptNet: in-memory graph (NetworkX), ~3.4M triples
- Freebase: preprocessed subgraph from WebQSP/CWQ
- Runs as separate HTTP service; verl calls it via external tool interface
- Precompute shortest path lookup tables for R_progress reward
- Response format: JSON

### 3.3 QA Datasets

| KG | QA Dataset | Hops | Split |
|---|---|---|---|
| ConceptNet | Self-generated | 1-3 hop | 4000/500/500 (existing) |
| Freebase | WebQSP + CWQ | 1-4 hop | Standard splits |
| Wikidata | T-REx or KGQAGen-10k | 1-3 hop | Standard splits |

KG is always fully accessible during train and test. Split is on questions, not triples.

### 3.4 Claim Verification Data

Generate from existing KG paths:
- Take a multi-hop KG path (e.g., dog -IsA-> animal -HasProperty-> breathe)
- Convert to natural language claim: "Dogs can breathe because dogs are animals and animals breathe."
- Create negative examples by replacing one step with a false triple
- Each sub-claim maps to one KG verification step

Can also use FactKG benchmark (ACL 2023, DBpedia-based) if time allows.

---

## 4. Three Tasks

### Task 1: KG-Grounded Multi-hop QA (comparable to KG-R1)

Agent queries KG to answer multi-hop questions. Standard agentic RAG setup.

```
User: What property do dogs have because of what they are?

Agent: <think>I need to find what dogs are, then check properties.</think>
       <tool_call>kg_search("dog", "IsA")</tool_call>
System: [("dog", "IsA", "animal"), ("dog", "IsA", "pet")]
Agent: <think>Dogs are animals. Let me find animal properties.</think>
       <tool_call>kg_search("animal", "HasProperty")</tool_call>
System: [("animal", "HasProperty", "breathe"), ...]
Agent: <think>Animals can breathe, so dogs can breathe.</think>
       <answer>breathe</answer>
```

### Task 2: Claim Verification (our distinctive task)

Agent verifies a multi-step claim by checking each sub-claim against KG.

```
User: Verify: "Aspirin treats headaches because aspirin is an NSAID,
       NSAIDs reduce inflammation, and headaches involve inflammation."

Agent: <think>I need to verify each step of this claim.</think>
       <tool_call>kg_verify("aspirin", "IsA", "NSAID")</tool_call>
System: TRUE
Agent: <think>Step 1 verified. Checking step 2.</think>
       <tool_call>kg_verify("NSAID", "HasEffect", "reduce_inflammation")</tool_call>
System: TRUE
Agent: <think>Step 2 verified. Checking step 3.</think>
       <tool_call>kg_verify("headache", "Involves", "inflammation")</tool_call>
System: TRUE
Agent: <think>All steps verified.</think>
       <answer>SUPPORTED</answer>
```

This task naturally maps to step-level verification — each sub-claim is independently verifiable.

### Task 3: Think-then-Verify (strongest demonstration of "KG as oracle")

Model first reasons freely WITHOUT KG, then verifies each step, then revises.

```
Phase 1 (Think — no KG access):
Agent: <think>Dogs can breathe because dogs are mammals and mammals breathe.</think>

Phase 2 (Verify — KG access):
Agent: <tool_call>kg_verify("dog", "IsA", "mammal")</tool_call>
System: TRUE
Agent: <tool_call>kg_verify("mammal", "HasProperty", "breathe")</tool_call>
System: TRUE
Agent: <think>Both steps verified. My reasoning is correct.</think>

Phase 3 (Revise if needed):
[If any step fails, agent revises reasoning and re-verifies]

<answer>breathe</answer>
```

This is the strongest design because:
- Think phase proves model doesn't NEED KG for knowledge
- Verify phase proves KG's role is purely as verification oracle
- The separation allows measuring: does verification training improve think quality?

---

## 5. Reward Functions (THREE variants + random baseline)

### 5.1 R_outcome (Baseline 1: Outcome-only)

```python
def reward_outcome(trajectory, ground_truth_answer):
    final_answer = extract_answer(trajectory)
    em = exact_match(final_answer, ground_truth_answer)
    f1 = token_f1(final_answer, ground_truth_answer)
    return 0.5 * em + 0.5 * f1
```

### 5.2 R_heuristic_step (Baseline 2: ProGraph-R1 style)

```python
def reward_heuristic_step(trajectory, ground_truth_answer):
    steps = parse_steps(trajectory)
    step_rewards = []
    for step in steps:
        retrieved_entities = extract_entities(step.observation)
        gt_entities = extract_entities(ground_truth_answer)
        r_overlap = len(retrieved_entities & gt_entities) / max(len(retrieved_entities), 1)
        r_reach = 1.0 if any(e in gt_entities for e in retrieved_entities) else 0.0
        step_rewards.append(0.5 * r_overlap + 0.5 * r_reach)
    r_answer = reward_outcome(trajectory, ground_truth_answer)
    return sum(step_rewards) / len(step_rewards) + r_answer
```

### 5.3 R_verifiable_step (Our method: KG-verifiable)

```python
def reward_verifiable_step(trajectory, ground_truth_answer, kg, gt_path):
    steps = parse_steps(trajectory)
    step_rewards = []
    prev_entity = extract_question_entity(trajectory)

    for step in steps:
        # R_valid: Did the query return real KG triples?
        r_valid = 1.0 if len(step.observation) > 0 else 0.0

        # R_on_path: Is retrieved triple on ground truth KG path?
        r_on_path = max(
            (triple_in_path(tr, gt_path) for tr in step.observation), default=0.0
        )

        # R_progress: Graph distance to answer decreased?
        current_entity = extract_current_entity(step)
        answer_entity = extract_answer_entity(ground_truth_answer)
        dist_before = kg.shortest_path_distance(prev_entity, answer_entity)
        dist_after = kg.shortest_path_distance(current_entity, answer_entity)
        r_progress = max(0, (dist_before - dist_after) / max(dist_before, 1))

        # R_coherence: Step shares entity with previous?
        r_coherence = 1.0 if shares_entity(prev_entity, current_entity) else 0.0

        step_rewards.append(
            0.3 * r_valid + 0.3 * r_on_path + 0.2 * r_progress + 0.2 * r_coherence
        )
        prev_entity = current_entity

    r_answer = reward_outcome(trajectory, ground_truth_answer)
    return 0.4 * r_answer + 0.6 * (sum(step_rewards) / max(len(step_rewards), 1))
```

### 5.4 R_random (Ablation: responding to Spurious Rewards)

```python
def reward_random(trajectory, ground_truth_answer):
    """Random step rewards + real outcome reward.
    Tests: does any step signal help, or does quality matter?"""
    steps = parse_steps(trajectory)
    step_rewards = [random.uniform(0, 1) for _ in steps]
    r_answer = reward_outcome(trajectory, ground_truth_answer)
    return 0.4 * r_answer + 0.6 * (sum(step_rewards) / max(len(step_rewards), 1))
```

Weights (0.3/0.3/0.2/0.2 and 0.4/0.6) are initial values. Tune on small-scale 7B runs first.

---

## 6. Training Pipeline

### 6.1 SFT Warmup

Teach the model the agent trajectory format for each task.

Options for generating SFT data:
- Use Qwen2.5-72B or GPT-4 to solve KG questions using the tool API
- Construct gold trajectories from ground truth KG paths (cheaper, deterministic)

Config: 1-2 epochs, lr 2e-5, LoRA rank 64.

### 6.2 GRPO Training

For each reward variant, run GRPO on verl:

```yaml
algorithm: grpo
multi_turn: true
max_turns: 5
group_size: 8
learning_rate: 5e-7
kl_coeff: 0.01
max_steps: 1000
batch_size: 128
```

Token masking: system prompt + tool response tokens masked from gradient.

Log to W&B: reward, EM, F1, reward-accuracy correlation, trajectory length, per-step reward components.

### 6.3 Evaluation

Standard metrics: EM, F1

Analysis metrics:
- Goodhart metric: Pearson(training_reward, EM) over training steps
- Reward-accuracy alignment: scatter of step reward vs step correctness
- Cross-KG transfer: train on KG_A, eval on KG_B (zero-shot)

Without-KG evaluation (critical new metric):
- Same test questions, KG API disabled
- Measure reasoning quality: decomposition granularity, step factual accuracy (LLM-judge), logical coherence, verifiability score

---

## 7. Experiment Matrix

### 7.1 P0: Core QA Experiments

**7B (primary model):**

| ID | Train KG | Reward | Test KG | Purpose |
|---|---|---|---|---|
| E1 | ConceptNet | R_outcome | ConceptNet | Baseline |
| E2 | ConceptNet | R_heuristic | ConceptNet | Baseline |
| E3 | ConceptNet | R_verifiable | ConceptNet | Core |
| E4 | Freebase | R_outcome | Freebase | Baseline |
| E5 | Freebase | R_heuristic | Freebase | Baseline |
| E6 | Freebase | R_verifiable | Freebase | Core |
| E7 | ConceptNet | R_outcome | Freebase | Cross-KG baseline |
| E8 | ConceptNet | R_heuristic | Freebase | Cross-KG baseline |
| E9 | ConceptNet | R_verifiable | Freebase | Cross-KG core |
| E10 | ConceptNet | R_verifiable | Without-KG | Reasoning quality |
| E11 | ConceptNet | R_outcome | Without-KG | Reasoning quality comparison |

**14B (scaling verification):**

| ID | Train KG | Reward | Test KG | Purpose |
|---|---|---|---|---|
| E12 | ConceptNet | R_outcome | ConceptNet | Scale baseline |
| E13 | ConceptNet | R_heuristic | ConceptNet | Scale baseline |
| E14 | ConceptNet | R_verifiable | ConceptNet | Scale core |
| E15 | Freebase | R_outcome | Freebase | Scale baseline |
| E16 | Freebase | R_heuristic | Freebase | Scale baseline |
| E17 | Freebase | R_verifiable | Freebase | Scale core |
| E18 | ConceptNet | R_verifiable | Freebase | Scale cross-KG |
| E19 | ConceptNet | R_verifiable | Without-KG | Scale reasoning quality |

### 7.2 P1: Extended Tasks (7B)

| ID | Task | Reward | Purpose |
|---|---|---|---|
| E20 | Claim Verification | R_outcome | Cross-task baseline |
| E21 | Claim Verification | R_heuristic | Cross-task baseline |
| E22 | Claim Verification | R_verifiable | Cross-task core |
| E23 | Think-Verify | R_outcome | Think-Verify baseline |
| E24 | Think-Verify | R_heuristic | Think-Verify baseline |
| E25 | Think-Verify | R_verifiable | Think-Verify core |

### 7.3 P2: Ablations

| ID | Model | Reward | Purpose |
|---|---|---|---|
| E26 | 7B | R_random | Respond to Spurious Rewards |
| E27 | 1.5B | R_verifiable | Scale-too-small failure case |
| E28 | 14B | Think-Verify R_verifiable | Think-Verify at scale |

### 7.4 Compute Estimate

| Group | GPU-hours (GH200) |
|---|---|
| SFT warmup (7B + 14B) | ~110h |
| P0 core 7B (E1-E11) | ~600h |
| P0 scaling 14B (E12-E19) | ~1,200h |
| P1 extended 7B (E20-E25) | ~550h |
| P2 ablations (E26-E28) | ~200h |
| Debug / failed / hyperparam search | ~800h |
| **Total** | **~3,460h** |

Well within 10,000 GPU-hour budget. ~6,500h buffer for reruns and exploration.

---

## 8. Data Collection for Analysis

### 8.1 Goodhart Analysis
- Every 50 training steps: checkpoint, eval on val set, record (reward_mean, EM, F1)
- Plot reward vs EM/F1 to identify Goodhart divergence
- Compare divergence points across reward types

### 8.2 Reward Hacking Evidence
- Save 100 random trajectories at steps 0, 250, 500, 750, 1000 for each reward type
- Classify hacking behaviors:
  - Token-level: KG keywords without reasoning
  - Length: abnormal trajectory length patterns
  - Path fabrication: queries for non-existent entities/relations
  - Shortcut exploitation: skipping intermediate steps
- Record trajectory length distribution over training

### 8.3 Step-level Verification Logs
- For R_verifiable: log per-step (r_valid, r_on_path, r_progress, r_coherence)
- Compute correlation between step reward components and final answer correctness

### 8.4 Without-KG Reasoning Quality
- For each trained model: run inference on test set WITHOUT KG API
- Measure:
  - Decomposition granularity: number of atomic reasoning steps
  - Grounding score: fraction of steps with explicit entity references
  - Factual accuracy: LLM-as-judge evaluation of each step
  - Verifiability score: are steps phrased as checkable claims?
- Compare across reward types: does R_verifiable produce more structured reasoning?

---

## 9. Compute Resources

### Viking HPC (University of York)
- GPU: NVIDIA H100 80GB PCIe
- Role: development, debugging, SFT experiments, 1.5B ablation
- Conda env: `kg_align` (existing)
- Scratch: `/mnt/scratch/users/ts1201/`

### Isambard HPC
- GPU: NVIDIA GH200 (Grace Hopper), 4 per node
- Allocation: 2,500 node-hours = 10,000 GPU-hours
- Max concurrent: 4 nodes (32 GPUs)
- Role: all 7B and 14B GRPO experiments
- FIRST TASK: verify verl + SGLang on GH200 architecture

---

## 10. Implementation Timeline

```
Week 1 (Mar 17-21): Environment setup
  - Isambard: verl + SGLang install, GH200 compatibility test
  - Download Freebase (WebQSP/CWQ), preprocess
  - Clone KG-R1 codebase, study verl config
  - Implement KG server with all 5 endpoints (ConceptNet + Freebase)

Week 2-3 (Mar 24 - Apr 4): Core implementation
  - Implement 4 reward functions (outcome, heuristic, verifiable, random)
  - Generate SFT data for all 3 tasks (QA, Claim Verification, Think-Verify)
  - SFT warmup (7B on Viking, 14B on Isambard)
  - Small-scale GRPO validation (7B, 100 samples, ConceptNet)

  CHECKPOINT Mar 28: KG server + rewards working -> Go/No-Go

Week 4-5 (Apr 7-18): P0 core experiments
  - E1-E11 (7B, all rewards, both KGs, cross-KG, without-KG)
  - E12-E19 (14B scaling experiments)
  - Viking: 7B runs in parallel | Isambard: 14B runs
  - Begin Goodhart data collection + hacking evidence

  CHECKPOINT Apr 11: Core hypothesis verified

Week 6-7 (Apr 21 - May 2): P1 + P2 + Analysis
  - E20-E25 (Claim Verification + Think-Verify)
  - E26-E28 (ablations)
  - Goodhart analysis, hacking taxonomy, without-KG eval
  - Begin writing

Week 8-10 (May 5-25): Writing + submission
  - Full paper draft, figures, tables
  - Advisor review
  - Submit EMNLP 2026 (May 25)
```

---

## 11. Deliverables

- [ ] verl environment verified on Isambard GH200
- [ ] KG server (5 endpoints) running for ConceptNet and Freebase
- [ ] SFT data for 3 tasks, models fine-tuned (7B + 14B)
- [ ] 4 reward functions implemented and tested
- [ ] P0 experiments (E1-E19) completed with W&B logs
- [ ] P1 experiments (E20-E25) completed
- [ ] P2 ablations (E26-E28) completed
- [ ] Goodhart data: checkpoints + eval every 50 steps
- [ ] Hacking evidence: 100 trajectories x 5 checkpoints x 4 rewards
- [ ] Without-KG eval: reasoning quality metrics for all trained models
- [ ] All results in structured format for paper writing

---

## 12. Risk Mitigation

| Risk | Mitigation |
|---|---|
| verl incompatible with GH200 | Fall back to Viking H100 (slower but works) |
| Core hypothesis fails | Reframe as negative result: "verifiability does not matter, here is why" |
| Freebase preprocessing issues | Use simplified Freebase subset; or fall back to another KG |
| Multi-turn GRPO diverges | Cap max turns at 3; smaller group size; reference KG-R1 hyperparams |
| Think-Verify too complex for 7B | Start with Task 1 (QA) and Task 2 (Claim Verification); Think-Verify as P1 |
| KG incompleteness affects results | Measure false negative rate; discuss in paper as limitation and analysis variable |
| Entity linking noise | Use standardized entity representations; fuzzy matching in reward |
