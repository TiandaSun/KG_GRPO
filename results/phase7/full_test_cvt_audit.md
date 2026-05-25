# Phase 7 Action II1: Full-Test CvT Audit

Full 3,531-question test set evaluation. CvT = fraction of trajectories classified as 'correct-via-tool' by `task16_classify`.

## Per-model stats

| model | n | EM | CvT count | CvT % | 95% Wilson CI | tools/Q |
|---|---|---|---|---|---|---|
| e5b_step100 | 3531 | 0.322 | 107 | 3.03% | [2.51-3.65%] | 2.31 |
| e3_step500 | 3531 | 0.325 | 1 | 0.03% | [0.00-0.16%] | 1.00 |
| 39b_step300 | 3531 | 0.366 | 132 | 3.74% | [3.16-4.42%] | 2.99 |
| 39b_step400 | 3531 | 0.383 | 133 | 3.77% | [3.19-4.45%] | 3.00 |
| 39b_step500 | 3531 | 0.380 | 110 | 3.12% | [2.59-3.74%] | 3.00 |
| 39i_self_step50 | 3531 | 0.381 | 171 | 4.84% | [4.18-5.60%] | 3.00 |
| 39i_self_step100 | 3531 | 0.387 | 225 | 6.37% | [5.61-7.23%] | 3.00 |
| 39i_self_step150 | 3531 | 0.387 | 257 | 7.28% | [6.47-8.18%] | 3.01 |
| 39i_self_step200 | 3531 | 0.390 | 288 | 8.16% | [7.30-9.11%] | 3.00 |
| 39i_self_step250 | 3531 | 0.395 | 338 | 9.57% | [8.65-10.59%] | 3.00 |
| 39g1_step500 | 3531 | 0.394 | 162 | 4.59% | [3.95-5.33%] | 3.00 |
| 39g2_step500 | 3531 | 0.400 | 205 | 5.81% | [5.08-6.63%] | 3.00 |

## Category breakdown

| model | correct-via-tool | correct-no-tool | correct-via-memory | wrong-no-tool | kg-incomplete | tool-misuse | wrong-answer |
|---|---|---|---|---|---|---|---|
| e5b_step100 | 107 | 0 | 1027 | 15 | 1435 | 216 | 731 |
| e3_step500 | 1 | 0 | 1136 | 14 | 209 | 833 | 1338 |
| 39b_step300 | 132 | 0 | 1158 | 13 | 1103 | 208 | 917 |
| 39b_step400 | 133 | 0 | 1218 | 6 | 1201 | 173 | 800 |
| 39b_step500 | 110 | 0 | 1229 | 2 | 1161 | 173 | 856 |
| 39i_self_step50 | 171 | 0 | 1170 | 7 | 1084 | 192 | 907 |
| 39i_self_step100 | 225 | 0 | 1137 | 5 | 1025 | 192 | 947 |
| 39i_self_step150 | 257 | 0 | 1108 | 4 | 1008 | 184 | 970 |
| 39i_self_step200 | 288 | 0 | 1087 | 1 | 858 | 194 | 1103 |
| 39i_self_step250 | 338 | 0 | 1054 | 2 | 824 | 205 | 1108 |
| 39g1_step500 | 162 | 0 | 1226 | 0 | 954 | 196 | 993 |
| 39g2_step500 | 205 | 0 | 1203 | 0 | 818 | 165 | 1140 |

## Paired McNemar on `correct-via-tool`

| comparison (A vs B) | n | A-only | B-only | both | neither | p (two-sided) |
|---|---|---|---|---|---|---|
| 39b_step400_vs_e5b_step100 | 3531 | 111 | 85 | 22 | 3313 | 0.0741 |
| 39b_step400_vs_e3_step500 | 3531 | 133 | 1 | 0 | 3397 | 0.0000 |
| 39b_step400_vs_39b_step300 | 3531 | 58 | 57 | 75 | 3341 | 1.0000 |
| 39b_step400_vs_39b_step500 | 3531 | 71 | 48 | 62 | 3350 | 0.0437 |

## Gate A decision

- 39B@400 full-test EM: **38.35%** (≥ 34%)
- 39B@400 full-test CvT: **3.77%** vs E5b@100 3.03%

**Gate A: PASS** → proceed with Day-2 variant training (G/I).
