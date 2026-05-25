# Task S1.3: CvT metric robustness under fuzzy matching

Source trajectories: `results/trajectories/phase7/39g2_step500_full/step_0/trajectories.json` (n=3531, EM-correct=1412).

## Definition

The canonical 7-category classifier (`scripts/task16_classify.py::classify_trajectory`) decides:
* **CvT** -- `em==1` AND ANY gold-answer alias (lowercased) appears as a substring of the concatenated `<tool_response>` text.
* **CvM** -- `em==1` AND no gold alias appears as a substring of any tool response.

Fuzzy match (this audit) tests both the predicted answer AND every gold alias, flipping a trajectory if ANY of them fuzzy-matches:
1. lowercase both sides,
2. strip leading article (a/an/the),
3. strip leading/trailing whitespace + punctuation, drop **all** punctuation, collapse whitespace,
4. (a) substring containment under (1)-(3); else (b) any contiguous n-gram of tool tokens with token-level Levenshtein distance <=floor(len/5), with strict guards (predicted >=8 chars or >=2 tokens, same first character) to suppress short-word collisions.

## Headline numbers

| Quantity | Value |
|---|---|
| n total | 3531 |
| n EM-correct | 1412 |
| canonical CvT (substring) | 205 (5.81%) |
| originally CvM | 1203 |
| CvM -> CvT under fuzzy | 0 (0.0% of CvM) |
| fuzzy CvT total | 205 (5.81%) |

## Stratified by hops (CvM -> CvT flip rate)

| hops | orig CvM | now CvT (fuzzy) | flip rate |
|---|---|---|---|
| 1 | 751 | 0 | 0.0% |
| 2 | 452 | 0 | 0.0% |

## Flipped examples (first 10)

## Residual CvM examples (first 5 -- no fuzzy match)

**1.** Q: 'What countries in the Chamorro Time Zone are in Oceania?'
   pred = 'Guam'; gold = 'Guam'; hops=1

**2.** Q: 'What country with the capital of HagÃ¥tÃ±a is in Oceania?'
   pred = 'Guam'; gold = 'Guam'; hops=1

**3.** Q: 'What nation in the geographic region of Oceania is the birthplace of the fictional character Jemaine Clement?'
   pred = 'New Zealand'; gold = 'New Zealand'; hops=1

**4.** Q: "Which Oceanic country's GDP has a change rate of -0.61?"
   pred = 'Kiribati'; gold = 'Kiribati'; hops=1

**5.** Q: "What country in Oceania that produces the beer Monteith's Lager?"
   pred = 'New Zealand'; gold = 'New Zealand'; hops=1

## Recommendation for the paper

The flip rate is **0.0%**, i.e. the CvM residual is essentially genuine memory-mediated correctness, not a formatting artifact. **Recommend keeping the canonical substring CvT as the headline metric; mention the audit briefly in the appendix.**

