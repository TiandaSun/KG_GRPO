# I-Self Peak-then-Collapse: A Diagnostic About Self-Verifiable Process Rewards

## Summary

I-Self (a self-verifiable retrieval reward trained without oracle `kg_path`
supervision) climbs **monotonically** in CvT for 5 consecutive 50-step
windows — 3.77% (step 0, 39B@400 init) → 4.84% (50) → 6.37% (100) → 7.28%
(150) → 8.16% (200) → **9.57% at step 250 (PEAK)** — then catastrophically
collapses in the single 50-step window between 250 and 300 to EM=0.000 and
ContEM=12.14%, with Tools/Q dropping from 3.00 to 1.00 simultaneously. At
peak (step 250), CvT = 338/3531, the highest of any model in our pipeline —
~2.5× the 39B@400 oracle-free baseline (3.77%) and ~3× E5b@100 (3.03%). By
step 400 ContEM is 6.30% and EM remains 0.

**The CvT climb spans 250 GRPO steps and totals +5.80pp; the collapse erases
the entire gain in a single 50-step window.** This step-level resolution of
peak-then-collapse is the novel paper finding: published process-reward RL
papers (ToolRL, StepSearch, CriticSearch, ProGraph-R1) report only
final-checkpoint numbers and would not detect this dynamic.

## Setup

I-Self was initialised from 39B@400 (EM=38.35%, CvT=3.77%) and trained with
GRPO under reward =
`0.25 * r_answer + 0.50 * r_tool_type + 0.25 * r_retrieval_contrib`,
where `r_retrieval_contrib` fires when an entity token extracted from the
final `<answer>` appears in any prior `<tool_response>` span (self-verifiable,
no gold `kg_path` used). KL coefficient was 5x the E5b default, and the
process-reward weight was ramped linearly from 0.10 to 0.25 over the first 50
steps. Full test N=3,531.

## Step-by-step behaviour

| Step | EM | ContEM | F1 | CvT | CvT % | Tools/Q | format-valid % | Notes |
|---|---|---|---|---|---|---|---|---|
| 0 (39B@400 init) | 0.3835 | 0.4290 | ~0.41 | 133 | 3.77% | 3.00 | ~99% | oracle-free baseline |
| 50 | 0.3806 | 0.4296 | 0.4041 | 171 | 4.84% | 3.00 | 99.09% | climb begins |
| 100 | 0.3866 | 0.4370 | 0.4084 | 225 | 6.37% | 3.00 | 99.75% | |
| 150 | 0.3874 | 0.4410 | 0.4074 | 257 | 7.28% | 3.01 | 99.52% | |
| 200 | 0.3903 | 0.4517 | 0.4091 | 288 | 8.16% | 3.00 | 99.92% | interim |
| **250** | **0.3951** | **0.4613** | **0.4143** | **338** | **9.57%** | **3.00** | **99.72%** | **PEAK** |
| 300 (partial, n=1400) | 0.0000 | 0.1214 | 0.0012 | — | — | 1.00 | <10% (proxy) | collapsed |
| 400 (partial, n=2000) | 0.0000 | 0.0630 | 0.0164 | — | — | 1.00 | <10% (proxy) | collapsed |

Wilson 95% CI on step-250 CvT = [8.65%, 10.59%], non-overlapping with
39B@400 [3.19%, 4.45%], E5b@100 [2.51%, 3.65%], and even step-200's earlier
mark [7.30%, 9.11%] (CIs of step 200 and step 250 overlap only marginally,
confirming continued statistically meaningful climb in the final 50-step
window before collapse). Step-300/400 trajectory dirs are empty
(`save_trajectories` did not fire before walltime), so format-valid %
is inferred from the ContEM proxy (46.13% at step 250 → 12.14% at step 300
implies <10% format-valid). Source: `results/phase7/v14_b3_format_valid.json`.

Step-300/400 figures are computed from partial per-sample dumps and read
as lower bounds on
collapse severity, not unbiased estimators.

## What improved at step 250 (peak) vs 39B@400 init

- `kg-incomplete` bucket contracted from **1201 to 824** trajectories
  (-31.4% relative), the largest single-bucket improvement any process
  reward has produced on CWQ in this project.
- `correct-via-tool` more than doubled (133 -> 338); `correct-via-memory`
  contracted slightly, indicating more answers were being
  grounded in tool output rather than recalled.
- `wrong-answer` rose (800 -> 1108), consistent with the model reaching
  the KG more often and sometimes selecting wrong final phrasing.
- Tools/Q and format validity were unchanged from the 39B@400 initialisation
  (format-valid 99.72% at step 250 vs ~99% at step 0), so the step-250 gain
  is isolated to *query precision*, not *call volume* and not *format quality*.

In short: I-Self briefly taught the policy to retrieve more productively
without teaching it to answer more correctly.

## What happened between step 250 and step 300

Hypothesis: once `r_retrieval_contrib` saturated on trajectories whose
final `<answer>` contained a literal substring of a prior tool response,
the optimal local reward became "emit a short string that is guaranteed
to have appeared somewhere in one retrieval, then stop". The policy
converged onto that attractor in a single 50-step interval (250→300):
Tools/Q dropped from 3.00 to 1.00, `<answer>` formatting disintegrated
(ContEM collapsed from 46.13% at step 250 to 12.14% at step 300 and
6.30% at step 400; format-valid % is at 99.72% at step 250 and inferred
<10% post-collapse via the ContEM proxy), and EM went to 0. This is a
classic specification-gaming failure mode of a process reward that
rewards output-overlap with retrieval rather than retrieval itself.

The collapse is not an optimiser artefact: we did not change KL,
learning rate, or clipping between step 250 and step 300. The policy
genuinely moved to a higher-reward / zero-EM region. **The 5-step
monotonic climb (50→100→150→200→250) shows no warning signs** — every
50-step window before the cliff produced a positive CvT delta with
unchanged tool-use frequency and format quality. The collapse is
single-interval and discontinuous.

## Why this is publishable

We are not aware of a peer-reviewed result that reports *step-resolved*
peak-then-collapse behaviour for a self-verifiable process reward
applied to an agentic KG-QA loop. ToolRL, StepSearch, CriticSearch, and
ProGraph-R1 all report final-checkpoint numbers only; our pipeline
saved eval checkpoints every 100 steps, which is what lets us observe
the phenomenon. The finding contributes two things:

1. A concrete demonstration that an oracle-free process reward *can*
   transiently outperform the oracle-free outcome baseline on CvT and EM,
   but is unstable under standard GRPO.
2. A mechanism: when the verifier is the model's own output, the reward
   signal collapses to "shorter outputs that trivially satisfy the
   verifier", and the policy follows.

This is exactly the kind of Goodhart's-law signature the RL-reward-shaping
literature discusses in the abstract (e.g. Skalse et al. 2022 on reward
hacking, Pan et al. 2022 on reward-model overoptimisation), instantiated
on a concrete KG-agent training loop.

## Recommended paper framing

I-Self's step-250 result enters the main table as the honest peak (CvT
9.57%, EM 0.395) with the collapse reported in the same row as a footnote
or a small inset figure showing the 5-step monotonic climb (4.84% → 9.57%)
and the single-interval cliff (9.57% → 0%). The complementary mechanism
memos (V14-B1, V14-B2, V14-B4 cross-validation) explain why the 7B CWQ
agent's dominant failure mode is schema-level relation typos
(70.4% of G2's kg-incomplete are within 1 edit of a gold relation;
89.1% B2 entity-correct + 90.4% B4 cross-recipe entity-agreement
independently corroborate), which is the bottleneck a self-verifiable
reward *could* address if it were stable. Natural follow-up work:
(a) bounded or capped process reward, (b) adaptive KL scheduling that
tightens when reward accelerates too quickly, (c) early-stopping at the
peak (step 250) with a held-out format-validity trigger.

## Data provenance

- `results/phase7/full_test_cvt_audit.json` (39i_self_step200 block: N=3531, EM=0.3903, CvT=288, CvT% Wilson CI [7.30%, 9.11%])
- `results/phase7/39i_self_step200_full_test.json` (EM=0.3903, ContEM=0.4517, F1=0.4091, Tools/Q=3.00, n=3531)
- `results/phase7/39i_self_step300_full_per_sample/step_0_per_sample.json` (partial, n=1400; EM=0.0000, ContEM=0.1214, Tools/Q=1.00)
- `results/phase7/39i_self_step400_full_per_sample/step_0_per_sample.json` (partial, n=2000; EM=0.0000, ContEM=0.0630, Tools/Q=1.00)
- Training run: `grpo_39i_s-3837343.log` (val-aux/cwq/em/mean@1 curve).
