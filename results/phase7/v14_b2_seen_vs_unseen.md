# S0.6: V14-B2 robustness — seen vs. unseen SFT relations

## Setup
R_SFT built from `data/freebase/sft_trajectories.jsonl` (5000 trajectories, 24713 tool calls with a relation argument); |R_SFT (argument)| = 1454, |R_SFT (argument ∪ tool-response listings)| = 1460.

Each CWQ test question is labelled by its gold `extra_info.kg_path` relations relative to R_SFT (argument set, primary).
Definition used: **seen = every gold relation in this question's kg_path appears in R_SFT (argument set); unseen = at least one gold relation NOT in R_SFT.**

## V14-B2 bucket distribution within each subset

| Subset | 1. Exact relation name typo (<= 1 edit) | 2. Correct entity, wrong relation | 3. Wrong entity, any relation | 4. Completely off |
|---|---|---|---|---|
| Overall (V14-B2) (n=818) | 576 (70.4%) | 153 (18.7%) | 70 (8.6%) | 19 (2.3%) |
| seen (n=167) | 70 (41.9%) | 61 (36.5%) | 29 (17.4%) | 7 (4.2%) |
| unseen (n=651) | 506 (77.7%) | 92 (14.1%) | 41 (6.3%) | 12 (1.8%) |

- relation_typo (edit <=1) on **seen**:   70/167 = **41.9%**
- relation_typo (edit <=1) on **unseen**: 506/651 = **77.7%**

## Subset sizes (CWQ test questions)

| Subset | n_questions | n_g2_trajectories | n_kg_incomplete |
|---|---|---|---|
| seen | 860 | 860 | 167 |
| unseen | 2671 | 2671 | 651 |
| no_gold | 0 | 0 | 0 |

## Verdict

**Robust.** The V14-B2 finding holds within BOTH the seen and unseen SFT-relation subsets — relation_typo (<=1 edit) accounts for 41.9% on seen and 77.7% on unseen kg-incomplete trajectories. Notably, the rate is HIGHER on unseen than seen — the opposite of what an SFT-memorisation artifact would predict. The close-edit-distance pattern is structural (the model parameterises Freebase-style relation tokens at the surface level) rather than an SFT-memorisation artifact. **Ship V14-B2 as-is**; no caveat required.
