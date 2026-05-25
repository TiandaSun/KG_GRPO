# Task 41 — 39B@400 Mechanistic Memo

Full-3,531-test analysis of why 39B@400 delivers EM=38.35% and CvT=3.77% against the two strong baselines E3@500 (EM=32.46%, CvT=0.03%) and E5b@100 (EM=32.20%, CvT=3.03%). Pure CPU analysis of saved trajectories; no new compute.

## Finding 1 — Category A vs B breakdown of EM

Category A = pass@10>0 on 7B-Instruct sampling (n=975); Category B = pass@10=0 (n=2,556, the 'hard' questions by frozen-base definition).

| model | A n | A EM | A EM % | A CvT | B n | B EM | B EM % | B CvT |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 39b_step400 | 975 | 713 | 73.13% | 57 | 2556 | 641 | 25.08% | 76 |
| e3_step500 | 975 | 679 | 69.64% | 0 | 2556 | 467 | 18.27% | 1 |
| e5b_step100 | 975 | 652 | 66.87% | 53 | 2556 | 485 | 18.97% | 54 |

**Observation.** 39B@400 solves **641** Category-B questions, but the 7-category classifier only credits **76** as 'correct-via-tool'. The missing 565 correct-B trajectories must therefore fall into 'correct-via-memory' (answer in final text but not in any returned tool snippet), implying the tool output likely still informed the answer but the `answer_in_kg_response` substring check underestimates tool attribution. 39B's A-gap over E5b is **+6.26 pp**; its B-gap is **+6.10 pp** — the full-test EM lift is driven by both buckets, not just 'easy' Category A.

## Finding 2 — Hop-stratified EM

`extra_info.hops` from `test.parquet` has three values: 0 (single-hop literals / mention-lookup), 1 (one-hop relation), 2 (compositional / CWQ multi-hop). Distribution: 0=683, 1=1,591, 2=1,257.

| model | hop=0 EM % (n) | hop=1 EM % (n) | hop=2 EM % (n) |
| --- | --- | --- | --- |
| 39b_step400 | 32.36% (n=683) | 38.47% (n=1591) | 41.45% (n=1257) |
| e3_step500 | 30.75% (n=683) | 33.44% (n=1591) | 32.14% (n=1257) |
| e5b_step100 | 27.53% (n=683) | 34.00% (n=1591) | 32.46% (n=1257) |

**Observation.** 39B@400 vs E5b@100 per-hop gain: hop=0: +4.83 pp, hop=1: +4.46 pp, hop=2: +8.99 pp. Largest gain at hop=2. Gain is **concentrated on compositional (hop=2)** questions (+4.34 pp over the mean of the other hops), consistent with tool-use helping where the answer requires chaining.

## Finding 3 — kg-incomplete error-mode distribution (first 300)

Of 39B@400's 1,201 kg-incomplete trajectories, we classify the first 300 (trajectory-file order) by parsing each `<search>fn(entity, relation)</search>` call and comparing against `extra_info.query_entities` and the kg_path relation set.

| error mode | count | % of 300 | interpretation |
| --- | --- | --- | --- |
| wrong-entity | 80 | 26.7% | Model picked an entity not licensed by the question's gold query_entities. |
| wrong-relation | 187 | 62.3% | Correct entity but relation not on the gold kg_path. |
| format-mismatch | 7 | 2.3% | Entity normalizes to gold but differs by whitespace/underscore/case. |
| genuine-kg-miss | 26 | 8.7% | Entity + relation on gold path but KG snippet still empty. |
| no-search-call | 0 | 0.0% | No parseable search call in trajectory (early termination). |
| no-meta | 0 | 0.0% | Sample ID missing from test.parquet (should be 0). |

**Observation.** Dominant error mode: **wrong-relation** (187/300 = 62.3%). If `wrong-relation` dominates, 39B picks the right subject but the wrong predicate — a *relation-selection* failure that a relation-match reward can target. If `wrong-entity` dominates, the issue is upstream (entity linking). A small `format-mismatch` slice confirms whether a lightweight query-normalization reward would clear easy wins.

## Finding 4 — E5b-vs-39B head-to-head on kg-incomplete

| metric | value |
| --- | --- |
| 39B@400 kg-incomplete samples (N) | 1201 |
|   of those with E5b trajectory | 1201 |
| E5b EM=1 on same sample | 54 (4.50%) |
| E5b correct-via-tool on same sample | 8 (0.67%) |

**Observation.** **< 5% threshold** — the E5b baseline does *not* systematically rescue 39B's kg-incomplete failures; 39B's retrieval gap is largely not a KL-over-constraint artefact. This bounds how much of the kg-incomplete bucket can plausibly be recovered by loosening KL; the rest requires a new training signal (Variant I / G).

## Recommendations for Variant I / G reward design

- **Primary reward signal.** `wrong-relation` = 62.3%, `wrong-entity` = 26.7%, `format-mismatch` = 2.3%, `genuine-kg-miss` = 8.7%. With wrong-relation dominant, **Variant I should favour a relation-match component** (I-Oracle: reward +1 if emitted relation ∈ kg_path relations; I-Self: reward inverse of hop-distance between emitted relation and any gold relation). 
- **Query-normalization reward.** Format-mismatch is 2.3% — small; a dedicated normalization reward would not move the needle. Fold normalization into preprocessing, not the reward function.
- **I-Oracle vs I-Self.** Oracle (using `kg_path` relations as ground truth) is the cleanest test of whether *any* relation-selection supervision closes the kg-incomplete gap; Self (self-distilling from 39B's own successful tool sequences) avoids label leakage but trains on a small correct-via-tool pool (133/3531 = 3.77%). **Recommend I-Oracle first** as an upper-bound ceiling, then I-Self as the transferable recipe.
- **Variant G hop prioritization.** 39B's largest EM gain over E5b is at hop=2 (+8.99 pp). Concentrated multi-hop advantage would motivate filtering G's self-distillation corpus to multi-hop trajectories. Separately, require each distilled trajectory to contain ≥1 correct-via-tool event to avoid amplifying the correct-via-memory mode.
- **KL-regularization ceiling.** Only 4.50% of 39B's kg-incomplete samples are rescued by E5b's looser policy. Relax KL in Variant I only as a secondary knob — new supervision dominates over KL-tuning.

## Data provenance

- `39b_step400`: n=3531, EM=38.35%, CvT=133/3531=3.77%, avg tools/Q=3.00
- `e3_step500`: n=3531, EM=32.46%, CvT=1/3531=0.03%, avg tools/Q=1.00
- `e5b_step100`: n=3531, EM=32.20%, CvT=107/3531=3.03%, avg tools/Q=2.31
- Classifications reused from `results/phase7/full_test_cvt_audit.json` (re-derived per-sample via `scripts/task16_classify.classify_trajectory`).
- Per-sample error-mode labels: `results/phase7/39b_query_error_modes.csv` (n=300).
