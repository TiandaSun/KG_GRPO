# S0.5 — Schema-format Base Rate (Null Hypothesis for V14-B2)

**Setup**: Sampled 100 random relations (seed=42) from the 7,058-relation Freebase pool. For each of 3,531 CWQ test questions with non-empty `kg_path` gold relation set, computed min Levenshtein distance from each random relation to all gold relations using V14-B2's exact `relations_are_close` logic (full + tail forms, ratio bonus).

Total pairs: 353,100. Unique gold relations in test set: 2,835. Gold relations per question: mean 75.8, median 62. Compute time: 172.9s.

## Null rates by edit-distance threshold

| Threshold | Pair-level % close | Per-question mean % close |
|-----------|-------------------:|--------------------------:|
| <= 1 edit  | **14.91%** | 14.91% |
| <= 2 edit  | 16.44% | -- |
| <= 3 edit  | 21.91% | -- |
| V14-B2 def (<=3 OR ratio>0.8) | 22.06% | 22.06% |

## Decision (user threshold: <=1 edit)

- **Null rate at <=1 edit: 14.91%**
- V14-B2 reported finding: 70.4%
- Ratio of finding to null: **4.7x**

**Verdict: STRONG** -- V14-B2's 70.4% is **strongly significant** (null < 15%). Ship as-is.

## Decision rule lookup table

| Null rate | Interpretation |
|-----------|---------------|
| < 15%   | V14-B2 strongly significant; ship as-is |
| 15-30%  | Moderately above null; add explicit base rate to paper |
| 30-50%  | Weak; revise Section 6 framing |
| > 50%   | Finding collapses; cannot use in paper |
