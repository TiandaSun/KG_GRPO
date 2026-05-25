# v19 W19 — dev/test split re-aggregation

Seed=42 stratified-by-hops split: dev=800, test=2731, full=3531.

Wilson 95% CI half-width @ n=2,731, p≈0.4: ±1.84 pp.

## Triple-column EM (dev / held-out test / full)

| HPC | paper | dev EM | test EM | full EM | gap (test−full) | within ±1.85pp? |
|---|---|---|---|---|---|---|
| E1@1250 | R-binary | 0.0025 | 0.0018 | 0.0020 | -0.0002 | ✓ |
| E3@500 | R-stepwise | 0.3200 | 0.3259 | 0.3246 | +0.0013 | ✓ |
| E5b@100 | R-toolverbs | 0.3150 | 0.3241 | 0.3220 | +0.0021 | ✓ |
| E5b+KL@400 | R-toolverbs.KL | 0.3688 | 0.3878 | 0.3835 | +0.0043 | ✓ |
| E5b+SelfV@250 | R-selfV-peak | 0.3962 | 0.3947 | 0.3951 | -0.0003 | ✓ |
| E5b+SelfV@300 | R-selfV-collapsed | 0.0000 | 0.0000 | 0.0000 | +0.0000 | ✓ |
| G1@500 | init-from-iterate | 0.3900 | 0.3955 | 0.3942 | +0.0012 | ✓ |
| G2@500 | self-distill | 0.3987 | 0.4002 | 0.3999 | +0.0003 | ✓ |
| E1'@300 | R-binary-SR | 0.0000 | 0.0000 | 0.0000 | +0.0000 | ✓ |
| E5b+KL-14B@400 | R-toolverbs.KL-14B | 0.3987 | 0.4035 | 0.4024 | +0.0011 | ✓ |

## E5b+SelfV dev-peak-step scan

| step | dev EM | test EM | full EM |
|---|---|---|---|
| 50 | 0.3650 | 0.3852 | 0.3806 |
| 100 | 0.3713 | 0.3911 | 0.3866 |
| 150 | 0.3762 | 0.3907 | 0.3874 |
| 200 | 0.3875 | 0.3911 | 0.3903 |
| 250 | 0.3962 | 0.3947 | 0.3951 |
| 300 | 0.0000 | 0.0000 | 0.0000 |
| 400 | 0.0000 | 0.0000 | 0.0000 |

**Dev-peak step**: 250  (in window {200,250,300})


## Pre-registered verdict: **CONFIRMS**

- 10 / 10 checkpoints within ±1.85pp

- E5b+SelfV dev peak step = 250

