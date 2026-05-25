# S0.2 Multi-seed G2 Aggregate (step 500)

Review-plan item: **S0.2 -- Multi-seed G2 x3 (seeds 1, 2, 3)**.
Single-seed reference (existing data) is treated as seed 0.

- Step evaluated: 500
- Seeds: [0, 1, 2, 3] (seed 0 = existing single-seed reference)

## Per-seed metrics

| seed | EM | ContEM | F1 | avg_tool_calls | n_samples | status |
|---:|---:|---:|---:|---:|---:|---|
| 0 | 39.99% | 45.09% | 42.14% | 2.999 | 3531 | OK |
| 1 | 39.76% | 43.90% | 42.47% | 2.999 | 3531 | OK |
| 2 | 40.24% | 44.89% | 42.02% | 2.999 | 3531 | OK |
| 3 | 40.84% | 46.33% | 42.71% | 2.999 | 3531 | OK |

## Cross-seed aggregate (mean +/- std [min, max])

| metric | aggregate |
|---|---|
| EM | 40.208% +/- 0.464  [39.762, 40.838] |
| ContEM | 45.051% +/- 1.000  [43.897, 46.332] |
| F1 | 42.334% +/- 0.315  [42.015, 42.709] |
| avg_tool_calls | 2.999 +/- 0.000  [2.999, 2.999] |
| n_samples | 3531 +/- 0  [3531, 3531] |

## Pairwise McNemar tests on EM

| pair | n_aligned | only_a | only_b | both_correct | both_wrong | chi2 | p | EM diff (a-b) | bootstrap CI95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0-1 | 3531 | 133 | 125 | 1279 | 1994 | 0.190 | 0.6630 | 0.0023 | [-0.0065, 0.0113] |
| 0-2 | 3531 | 112 | 121 | 1300 | 1998 | 0.275 | 0.6002 | -0.0025 | [-0.0108, 0.0059] |
| 0-3 | 3531 | 93 | 123 | 1319 | 1996 | 3.894 | 0.0485 | -0.0085 | [-0.0161, -0.0003] |
| 1-2 | 3531 | 117 | 134 | 1287 | 1993 | 1.020 | 0.3125 | -0.0048 | [-0.0133, 0.0040] |
| 1-3 | 3531 | 103 | 141 | 1301 | 1986 | 5.611 | 0.0179 | -0.0108 | [-0.0193, -0.0017] |
| 2-3 | 3531 | 110 | 131 | 1311 | 1979 | 1.660 | 0.1976 | -0.0059 | [-0.0147, 0.0028] |

## Pairwise EM agreement matrix

| seed | 0 | 1 | 2 | 3 |
|---:|---:|---:|---:|---:|
| 0 | 1.0000 | 0.9269 | 0.9340 | 0.9388 |
| 1 | 0.9269 | 1.0000 | 0.9289 | 0.9309 |
| 2 | 0.9340 | 0.9289 | 1.0000 | 0.9317 |
| 3 | 0.9388 | 0.9309 | 0.9317 | 1.0000 |

Mean off-diagonal pairwise agreement: 0.9319

## Robustness verdict

**Verdict: ROBUST**

- EM spread (max-min): 1.08 pp across 4 seeds
- Rule: ROBUST <= 2.0pp; VARIABLE > 5.0pp; otherwise MARGINAL.
