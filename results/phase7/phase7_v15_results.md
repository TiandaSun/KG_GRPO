# Phase 7 v15 — Consolidated Results

Generated: 2026-05-07T14:49:04.846943Z

Code-agent deliverable. Writing-agent integrates these into paper sections.


## Lane 1 — V15-W1 Multi-seed E5b+SelfV step-curve replication

Seeds: [0, 1, 2, 3], Steps: [50, 100, 150, 200, 250, 300, 350, 400]


| Step | seed=0 EM | seed=1 EM | seed=2 EM | seed=3 EM |
|---|---|---|---|---|
| step=50 | 0.3806 | 0.3806 | 0.3829 | 0.3826 |
| step=100 | 0.3866 | 0.3857 | 0.3860 | 0.3914 |
| step=150 | 0.3874 | 0.3908 | 0.3917 | 0.3880 |
| step=200 | 0.3903 | 0.0000 | 0.0000 | 0.0000 |
| step=250 | 0.3951 | 0.0000 | 0.0000 | 0.0000 |
| step=300 | — | 0.0000 | 0.0000 | 0.0000 |
| step=350 | — | 0.0000 | 0.0000 | 0.0000 |
| step=400 | — | 0.0000 | 0.0000 | 0.0000 |

Plot: `results/phase7/v15_w1_multiseed_selfv/multiseed_em_curve.png`



## Lane 2 — V15-G3 Clean init-source ablation

| Step | G3 EM |
|---|---|
| 100 | 0.3854 |
| 200 | 0.3914 |
| 300 | 0.3897 |
| 400 | 0.4019 |
| 500 | 0.4013 |


## Lane 4 — V15-W5 KL coefficient ablation

# V15-W5 KL ablation summary

| KL coef | step100 EM | step200 EM | step300 EM | step400 EM | step500 EM | Collapse step |
|---|---|---|---|---|---|---|
| 0.001 | 0.3911 | 0.3920 | 0.3769 | 0.3600 | 0.4070 | no collapse |
| 0.005 | 0.3846 | 0.3888 | 0.3583 | 0.0660 | 0.0476 | no collapse |
| 0.01 | 0.3863 | 0.3914 | 0.3846 | 0.4005 | 0.3951 | no collapse |
| 0.05 | 0.3948 | 0.3934 | 0.3837 | 0.3990 | 0.3959 | no collapse |
| 0.25 | 0.3832 | 0.3908 | 0.3866 | 0.3911 | 0.4773 | no collapse |

Plot: `results/phase7/v15_w5_kl_ablation/kl_ablation_em_curve.png`


## Lane 6 — V15-Q3 Token-entropy time series

Per-step inside/outside-`<search>` entropy:

```json
{
  "unit": "nats",
  "records": [
    {
      "step": 200,
      "n_inside": 25161,
      "n_outside": 60829,
      "mean_inside": 0.34991401168607433,
      "mean_outside": 0.038637596241179975,
      "elapsed_s": 2493.218468427658
    },
    {
      "step": 250,
      "n_inside": 25300,
      "n_outside": 61049,
      "mean_inside": 0.35184181038008655,
      "mean_outside": 0.03603131298152687,
      "elapsed_s": 2538.8884403705597
    },
    {
      "step": 300,
      "n_inside": 8091,
      "n_outside": 272027,
      "mean_inside": 0.29177960819373566,
      "mean_outside": 0.0064899862335572565,
      "elapsed_s": 8319.844928979874
    },
    {
      "step": 400,
      "n_inside": 7142,
      "n_outside": 269337,
      "mean_inside": 0.205076480347625,
      "mean_outside": 0.019558420283555433,
      "elapsed_s": 8259.006560564041
    }
  ]
}
```



## Lane 7 — V15-W4 GPT-4o n=500 baseline

(aggregate field not found in JSON — open file directly)



## Lane 3 — V15-Q2 Mode-4 trajectory dumps (writing-agent input)

- step 300: `/lus/lfs1aip2/scratch/u6gg/ts1201.u6gg/Project/KG_GRPO/results/trajectories/phase7/39i_self_step300_full/step_0/trajectories.json` (3531 trajectories)
- step 400: `/lus/lfs1aip2/scratch/u6gg/ts1201.u6gg/Project/KG_GRPO/results/trajectories/phase7/39i_self_step400_full/step_0/trajectories.json` (3531 trajectories)


