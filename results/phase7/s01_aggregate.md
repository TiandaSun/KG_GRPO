# S0.1 Multi-seed I-Self Aggregate

Review-plan item: **S0.1 -- Multi-seed I-Self x3 (seeds 1, 2, 3)**.
Single-seed reference (existing data) is treated as seed 0.

- Steps evaluated: [0, 50, 100, 150, 200, 250, 300, 350, 400]
- Seeds: [0, 1, 2, 3] (seed 0 = existing single-seed reference)

## Per-step aggregate across seeds {0,1,2,3}

| step | n_seeds | EM mean +/- std | EM min | EM max | ContEM mean +/- std | CvT-proxy (seed 0) | pending seeds |
|---:|---:|---|---:|---:|---|---|---|
| 0 | 0 | -- | -- | -- | -- | -- | 0,1,2,3 |
| 50 | 4 | 38.17% +/- 0.12 | 38.06% | 38.29% | 43.21% +/- 0.32 | -- | -- |
| 100 | 4 | 38.74% +/- 0.27 | 38.57% | 39.14% | 44.40% +/- 0.50 | -- | -- |
| 150 | 4 | 38.95% +/- 0.21 | 38.74% | 39.17% | 45.06% +/- 0.73 | -- | -- |
| 200 | 4 | 9.76% +/- 19.51 | 0.00% | 39.03% | 22.41% +/- 15.37 | 20.61% +/- 0.00 | -- |
| 250 | 4 | 9.88% +/- 19.75 | 0.00% | 39.51% | 22.62% +/- 16.25 | 23.73% +/- 0.00 | -- |
| 300 | 3 | 0.00% +/- 0.00 | 0.00% | 0.00% | 31.42% +/- 6.70 | -- | 0 |
| 350 | 3 | 0.00% +/- 0.00 | 0.00% | 0.00% | 29.53% +/- 7.47 | -- | 0 |
| 400 | 3 | 0.00% +/- 0.00 | 0.00% | 0.00% | 32.29% +/- 5.23 | -- | 0 |

## Per-seed peak and collapse

| seed | peak basis | peak step | peak EM | collapse step | never collapsed |
|---:|---|---:|---:|---:|---|
| 0 | cvt_proxy | 250 | 39.51% | -- | YES |
| 1 | em | 150 | 39.08% | 200 | no |
| 2 | em | 150 | 39.17% | 200 | no |
| 3 | em | 100 | 39.14% | 200 | no |

## Decision-rule verdict

**Verdict: MIXED**

- Reason: peak steps span [100, 150, 250]; collapse steps [None, 200, 200, 200] (need <= 350); seeds never collapsed: [0]
- Recommendation: Report mixed peaks; flag any non-collapsing seed in paper appendix.
- Rule: peaks in window [200, 250], peak EM in [0.35, 0.45], collapse (EM <= 5% of peak) by step 350.
