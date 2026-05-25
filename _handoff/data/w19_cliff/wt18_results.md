# v19 W18 — r_retrv cliff-isolation result

**Verdict: CONFIRMS (anti-quote r_retrv prevents the cliff in >=2 seeds)**  (3/3 seeds avoid cliff)

Baseline E5b+SelfV cliff: CvT 3.77→9.57→0 by step 300; Tools/Q 3.0→1.0.

- **seed1** (max_step 300, cliffed=False): s50:EM0.36/CvT3.5/T3.0 s100:EM0.38/CvT5.0/T3.0 s150:EM0.37/CvT3.5/T3.0 s200:EM0.37/CvT3.0/T3.0 s250:EM0.38/CvT3.0/T3.0 s300:EM0.40/CvT3.0/T3.0
- **seed2** (max_step 350, cliffed=False): s50:EM0.36/CvT4.5/T3.0 s100:EM0.37/CvT3.5/T3.0 s150:EM0.40/CvT3.5/T3.0 s200:EM0.36/CvT2.5/T3.0 s250:EM0.36/CvT2.5/T3.0 s300:EM0.38/CvT3.5/T3.0 s350:EM0.39/CvT3.0/T3.0
- **seed3** (max_step 400, cliffed=False): s50:EM0.36/CvT4.0/T3.0 s100:EM0.39/CvT3.5/T3.0 s150:EM0.40/CvT3.0/T3.0 s200:EM0.41/CvT4.5/T3.0 s250:EM0.39/CvT3.5/T3.0 s300:EM0.40/CvT3.0/T3.0 s350:EM0.40/CvT3.5/T3.0 s400:EM0.39/CvT1.5/T3.0
