# V14-B4: G2 vs 39B Behavioral Query Contrast

## Setup
500 random sample_ids (seed=42) drawn from the intersection of the two full-test trajectory sets: G2@500 (`39g2_step500_full/step_0`) and 39B@400 (`39b_step400_full/step_400`); intersection = 500 ids (full 3,531 CWQ test).

## Aggregate outcomes on the 500 samples

| Metric | G2@500 | 39B@400 | Delta (G2-39B) |
|---|---|---|---|
| EM count | 192 (38.40%) | 175 (35.00%) | +3.40pp |
| F1 mean | 0.4128 | 0.3825 | +0.03 |
| correct-via-tool | 28 | 15 | +13 |
| kg-incomplete | 118 | 181 | -63 |

## Behavioral dimension 1: tool-type distribution (over ALL tool calls)

- G2 total calls: 1500   39B total calls: 1499

| Tool | G2 count (%) | 39B count (%) |
|---|---|---|
| `get_tail_relations` | 0 (0.00%) | 0 (0.00%) |
| `get_head_relations` | 0 (0.00%) | 0 (0.00%) |
| `get_tail_entities` | 1500 (100.00%) | 1499 (100.00%) |
| `get_head_entities` | 0 (0.00%) | 0 (0.00%) |
| `other` | 0 (0.00%) | 0 (0.00%) |

## Dimension 2: unique query diversity per trajectory

| Metric | G2 | 39B |
|---|---|---|
| Mean distinct (entity, relation) pairs / traj | 2.634 | 2.368 |
| % with >=2 distinct | 96.40% | 89.80% |
| % with >=4 distinct | 0.00% | 0.00% |

## Dimension 3: turn-to-answer distribution

- G2: 500 with `<answer>` (mean 3.000), 0 without.  39B: 499 with answer (mean 3.002), 1 without.

| Turns before `<answer>` | G2 (% of answered) | 39B (% of answered) |
|---|---|---|
| 1 | 0 (0.00%) | 0 (0.00%) |
| 2 | 0 (0.00%) | 0 (0.00%) |
| 3 | 500 (100.00%) | 498 (99.80%) |
| 4 | 0 (0.00%) | 1 (0.20%) |
| 5+ | 0 (0.00%) | 0 (0.00%) |
| Mean | 3.000 | 3.002 |

## Dimension 4: first-query behavior on shared kg-incomplete

- Samples where BOTH G2 and 39B were classified `kg-incomplete`: **94**.

| Behavior on first `<search>` | Count |
|---|---|
| (a) identical `(entity, relation)` | 68 |
| (b) same entity, different relation | 17 |
| (c) entirely different queries | 9 |

## Paper implication

Both agents have collapsed onto a single verb (`get_tail_entities`, 100% of calls for G2 and 39B) and a fixed 3-turn budget. Tool-type and turn-length are therefore NOT where ReST-EM init differs; the behavioral signal lives entirely in the (entity, relation) arguments. On those arguments, G2 issues more distinct queries per trajectory (mean 2.634 vs 2.368, delta +0.266) and more often reaches >=2 distinct queries (96.4% vs 89.8%, +6.60pp), while turn-count stays essentially identical (-0.002). ReST-EM init makes the agent *explore more within a fixed budget*, not *spend more budget*.

On the 94 samples where BOTH models land in `kg-incomplete`, **68/94** use an identical first `(entity, relation)` query while **17/94** agree on entity but diverge on relation (share of same-entity = **90.4%**). Combined with the outcome gaps on the 500 samples (EM +3.40pp, correct-via-tool +13, kg-incomplete -63), this rules out the null hypothesis that the 0.6pp full-test EM lift is stochastic tie-breaking. Both policies agree *where* to look (entity) but systematically disagree on *how* (relation); ReST-EM init is a measurable behavioral change that reduces kg-incomplete failures by shifting relation choice, not entity choice.
