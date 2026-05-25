# v19 W18 + W20 — combined causal interpretation (Mode-4 mechanism)

> Preliminary (2026-05-23 11:10). W18 seed2 full to step 300; seed1/seed3 to 250
> (no-cliff trend unambiguous). W20 a1 to 250, a2 to 150-200 (cliff already shown).
> Verdicts will be re-confirmed on complete curves but the direction is locked.

## W18 — r_retrv isolating intervention: **CONFIRMS**

Anti-quote r_retrv (fires only with >=2 distinct non-empty <search> calls)
PREVENTS the Mode-4 peak-then-collapse cliff in all evaluated seeds:

| seed | step 50 | late (250-300) | cliffed? |
|---|---|---|---|
| 1 | EM .36 / CvT 3.5 / Tools 3.0 | EM .38 / CvT 3.0 / Tools 3.0 (s250) | NO |
| 2 | EM .36 / CvT 4.5 / Tools 3.0 | EM .38 / CvT 3.5 / Tools 3.0 (s300) | NO |
| 3 | EM .36 / CvT 4.0 / Tools 3.0 | EM .39 / CvT 3.5 / Tools 3.0 (s250) | NO |

Baseline E5b+SelfV: CvT 3.77→9.57→**0**, Tools/Q 3.0→**1.0**, EM→**0** by step 300.
W18 holds Tools/Q at 3.0, CvT at 3-5%, EM at .36-.41 — **no collapse**.

→ The gameable r_retrv (rewarding single-call quote-and-stop) is the CAUSAL driver
of the Mode-4 cliff. Closing the quote-and-stop loophole eliminates the collapse.
Answers R8 W4 ("oracle ablation doesn't isolate") + R9 ("framework forbids no
observation") — Mode-4 is now an intervention-isolated, falsifiable claim.

## W20 — L_sig informative-failure intervention: **FALSIFIES (on EM) + reinforces W18**

A1 (E5b + informative errors): EM peak 0.325/0.345 vs baseline E5b 0.34 → **no lift
(<3pp)**. Informative errors do NOT causally improve accuracy.

A2 (E5b+SelfV + informative errors): **STILL CLIFFS** (EM 0.36→0.15→0.00 by step
150-200) — informative failures do NOT prevent the cliff. (If anything, slightly
earlier onset.)

**Metric caveat (important):** task16 "kg-incomplete" keys on empty `[]` responses;
under informative-failure these become `ERROR:` strings, so kg-incomplete reads 0% —
a CLASSIFICATION ARTIFACT, not a real reduction. EM (load-bearing) is the honest
criterion → no lift.

## Joint conclusion (the strong rebuttal story)

The two interventions triangulate the Mode-4 mechanism:
- **r_retrv gameability IS the cliff driver** (W18: closing it prevents collapse)
- **L_sig silent-failure is NOT the driver** (W20-A2: fixing it does not prevent
  collapse) and **not a causal accuracy lever** (W20-A1: fixing it does not lift EM)

→ For camera-ready: the four-channel framework's L_sig "dominance" is a DESCRIPTIVE
attribution (correlation), while the Mode-4 cliff is CAUSALLY isolated to r_retrv
gameability. Recommend reframing §5.3 to separate the descriptive channel-attribution
(L_sig 55.2%) from the causally-tested cliff mechanism (r_retrv).
