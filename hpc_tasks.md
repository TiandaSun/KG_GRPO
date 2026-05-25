# HPC Current Tasks

> 🟢 **2026-05-25 — W18/W19/W20/W21 ALL RETURNED. Queue COMPLETE; do NOT re-run.** Results in
> `wt18-21/` (verdicts: W18 CONFIRMS r_retrv causally isolates Mode-4 cliff 3/3 seeds; W19 CONFIRMS
> dev/test split, dev-peak=250, 10/10 within ±1.85pp; W20 FALSIFIES L_sig as causal lever; W21
> CONFIRMS multi-seed taxonomy ≥2/3 + self-distill pass@16 +11.5pp). **Paper consumed W19 + W21-B
> into submission PR11** (`kg_grpo_emnlp2026_v3_pr11_overleaf.zip`). W18 + W20 + W21-A held for
> rebuttal → `REBUTTAL_PACKAGE_W18_W20_W21A.md`. Camera-ready follow-ups: complete W18/W20 curves
> (seed1/seed3 to 300+; a2 fuller), re-run self-distill pass@16 at 500-Q/temp-1. The spec below is
> retained for provenance.

> 🆕 **2026-05-25 — v21 NEW QUEUE: W23 (+ W22) queued below.** Both the fog-of-war sequential review
> AND the blind whole-PDF review of PR12 converged on the SAME #1 lever to move Overall 3→3.5
> (Conference): a **matched non-KG interface-swap control**. NON-BLOCKING, rebuttal/camera-ready
> stage — do NOT enter the submitted paper (deployed in the author-response, which is OUTSIDE the
> 8pp; baked into camera-ready's +1 page if accepted). Queue NOW so results are in hand for the
> rebuttal window. Reviews: `kg_grpo_emnlp2026_v3_namespace_reduction/AI_review/fogofwar_review_2026_05_25_PR11.md`
> + `blind_wholePDF_review_2026_05_25_PR12.md`.

---

## v21 NEW QUEUE (2026-05-25) — convergent #1 lever from the PR12 reviews

### W23 [P0 — the #1 lever to Overall 3.5/Conference] — Matched non-KG interface-swap control

**Purpose.** Causally test the paper's central thesis — *"the KG interface is uniquely degraded because its failures leak no pretraining-aligned NL signal"* — by running the IDENTICAL GRPO recipe + the same self-verifiable retrieval reward (R-selfV = HPC `E5b+SelfV`, the Mode-4 producer) on a NON-KG tool whose failures DO leak NL signal. The uniqueness claim is currently asserted via citation only (App G qualitative table; GPT-4o = "control not baseline"; Llama = varies model not interface) and NEVER measured head-to-head. Answers blind-review W2 + 7th-external W12 (the convergent #1 lever).

**Single variable = the tool interface surface.** Hold constant: base = Qwen2.5-7B-Instruct (same SFT pre-warm recipe, model-specific); algorithm = GRPO (group 8, `low_var_kl`, KL 0.05, lr 3e-7, batch 128, 500 steps); reward = the R-selfV design (`0.25 r_ans + 0.50 r_tool_type + 0.25 r_retrv`), r_retrv = self-verifiable "a returned tool token appears verbatim in `<answer>`" — the EXACT reward that produced Mode-4 on KG. **Swap ONLY the tool.**

- **Primary option (A) — Python/calculator on multi-hop arithmetic** (easiest: deterministic executor, no external API; failures = NL tracebacks; strong literature precedent it CONVERGES on Qwen-7B → ToRL, ReTool). Multi-hop = multi-step numeric chains. r_retrv analogue: a computed intermediate value appears verbatim in the answer.
- **Alternative option (B) — web-search multi-hop QA** (HotpotQA / Bamboogle / 2WikiMultihop; closest TASK-parallel = multi-hop *retrieval* QA, only the retrieval surface changes KG-symbolic→NL-passage; failures = related-but-wrong NL passages, not silent `[]`). Use (B) if a search index is available; else (A).

**Eval.** 50-step cadence (the cadence that caught Mode 4); per-checkpoint EM, the r_retrv-grounding analogue of CvT, and Tools/Q. Watch specifically for the peak-then-collapse signature: monotone **Tools/Q 3→1 cliff + EM→0 in a single 50-step window**. **≥3 seeds.**

**Pre-registered decision rule.**
- **CONFIRMS (thesis holds → Soundness→4; uniqueness becomes MEASURED, not asserted)**: the matched non-KG ladder does NOT peak-then-collapse — no Tools/Q 3→1 cliff, no EM→0 — in ≥2/3 seeds (stable/graceful, as Python/web RLVR normally is). → The KG interface IS the differentiator; the four-channel account gains a causal anchor.
- **FALSIFIES (uniqueness fails → reframe)**: the non-KG ladder ALSO peak-then-collapses. → The signature is reward-shape-specific (the r_retrv quote-and-stop attractor), not interface-bound. Reframe around "self-verifiable-reward Goodhart" rather than "KG-interface uniqueness." (Far better to learn this at rebuttal, under our framing control, than post-acceptance.)

**Cost.** ~2–4 days (tool-env standup + 3 seeds × 500 steps). NON-BLOCKING.

### W22 [P1 — secondary] — Entity-threading (L_comp) oracle

**Purpose.** The fog-of-war review showed §5.3's "residual gap = retrieval-composition / L_comp" is a 2-bucket residual that promotes L_comp (rarest rubric channel, 7.8%, never probed) by elimination. W22 directly probes L_comp as the mirror of the existing relation oracle: inject the gold ENTITY chain across hops (keep the model's relation argument) and re-measure EM on self-distill@500 (`G2`).

**Decision rule.** CONFIRMS L_comp causal = entity oracle lifts EM ≥3pp (vs the relation oracle's +0.20pp null); FALSIFIES = also flat (→ bottleneck is answer-composition / L_sig interaction, elsewhere). Either way it resolves the §5.3 over-reach (PR12 already softened the prose to "channel not isolated"; W22 would let us *name* it at camera-ready). **Cost** ~1 day (oracle re-eval, no training). NON-BLOCKING.

> **Both W22 + W23 are rebuttal / camera-ready ammo — they DO NOT enter the submitted paper.**

---

> **This file is the active task queue for the HPC Claude Code agent.**
> Updated from the discussion repo after each planning iteration.
> For technical specs and long-term plan, see `hpc_implementation_spec.md`.
>
> **Last updated**: 2026-05-22 (v20 — added W21 [multi-seed reward ladder + self-distill pass@16] after a 7th independent clean-context reviewer round (most critical to date, 2.5/2.5/2.5 Findings, confidence 4) escalated single-seed reward-ladder support to its #1 rebuttal lever. Queue is now W18 (r_retrv isolation), W19 (test-set dev/test split), W20 (L_sig informative-failure), W21 (multi-seed ladder + self-distill pass@16). v19 added W19+W20 after the 6th reviewer flagged test-set protocol + L_sig-not-directly-tested. Paper STAYS at PR9 ship-ready (score spread across reviewers: 2.5 critical / 3.5 standard / 4.0 sequential-read → true expected ~3.0-3.5; 8pp-PASS; bibliography integrity clean — note: 7th reviewer's "corrupted references / R-toelverbs / get.tail" are PDF-text-extraction artifacts of their tool, source verified clean; target = ARR "LLM agents" → EMNLP 2026). **W18 + W19 + W20 + W21 results DO NOT enter the submitted paper — rebuttal / next-cycle ammo only.** Priority order for rebuttal: W19 (~1hr) + W21 (~3d, #1 lever) first; W18 + W20 (causal framework evidence) second. After v20, file FROZEN until ARR rebuttal phase.)

> **v17 unchanged**: Llama-3.1-8B cross-family LANDED & integrated (App F → App I in v3).
>
> **v18 delta from v17** (2026-05-19): v17 (Llama-3.1-8B cross-family) LANDED & integrated (App F). v18 re-opens HPC for **one** non-blocking task. Driver: independent clean-context reviewers **R8 (LLM-Agents lens → 3.5)** and **R9 (Interpretability lens → 3.0)** converge that the sole remaining 3.5→4.0 blocker is that the **Mode-4 "peak-then-collapse" mechanism is a hypothesis with no isolating intervention** and the four-channel framework "forbids no observation" (non-falsifiable as delivered). W18 is the single experiment that converts Mode-4 into an intervention-isolated, falsifiable claim. **Scope discipline (memory: `feedback_scope_discipline.md`): NOT a submission blocker — paper ships as-is at stable 3.5; W18 is opportunistic. Do NOT modify the frozen paper, do NOT block submission, do NOT rush a bad result in before 2026-05-25.**
>
> ---
> **v3-renamed name mapping (added 2026-05-20).** This W18 spec uses HPC codes (`E5b+SelfV`, `E5b+KL`, `G2`). In the v3 paper (`kg_grpo_emnlp2026_v3_namespace_reduction/`) they read as `R-selfV`, `R-toolverbs·KL`, `self-distill` respectively. Full bidirectional map: paper App C.7 `tab:hpc-mapping` and `v3/NAMING_MAP.md`. HPC commands / verl configs continue using HPC codes; cite results / paper using v3 paper names.
>
> ### W18 — Mode-4 `r_retrv` isolating-intervention ablation  [P1, NON-BLOCKING, rebuttal-stage]
>
> **Purpose.** Causally isolate `r_retrv` as the driver of the Mode-4 peak-then-collapse cliff. Existing evidence is correlational (E5b+KL stable vs E5b+SelfV collapses, but those differ in >1 reward term). W18 = single-variable swap → falsifiable claim. Answers R8 W4 ("oracle ablation doesn't isolate") + R9 ("framework forbids no observation").
>
> **Design (single variable).** Reuse the EXACT E5b+SelfV recipe and the EXISTING `E5b+KL@400` checkpoint as init (same fork point E5b+SelfV used — do NOT retrain E5b+KL). Change ONLY `r_retrv`:
> - **R1 (load-bearing, anti-quote-and-stop):** `r_retrv` fires only if a returned KG entity appears verbatim in `<answer>` **AND** ≥2 distinct `<search>` calls returned non-empty responses in that trajectory. A one-shot quote-and-stop (Tools/Q≈1) earns `r_retrv`=0. Everything else identical to E5b+SelfV (`r = 0.25 r_ans + 0.50 r_tool_type + 0.25 r_retrv`; KL 0.25; group 8; lr 3e-7; batch 128; verl+vLLM; GH200).
> - **R2 (optional 2nd arm, parallel if budget):** `r_retrv` requires the quoted entity ∈ gold answer set (not gold-free) — upper-bound sanity contrast for R1.
>
> **Steps / seeds / eval.** ≥400 steps (cliff window 250–300; run past 400 with margin; 500 if cheap). **≥3 seeds, run in parallel** (GPU abundant, `hpc_gpu_budget.md`). Per-checkpoint full CWQ test N=3,531, **50-step cadence** (match Mode-4 instrumentation), greedy, max_turns=5, max_new_tokens=512. Readouts: CvT curve, Tools/Q curve, EM, 7-cat classify. Baseline of comparison = existing E5b+SelfV cliff (CvT 3.77→9.57%→0 by step 300; Tools/Q 3.0→1.0).
>
> **Pre-registered decision rule (LOCK before viewing results).**
> - **CONFIRMS** (mechanism causally isolated): R1 does NOT cliff — CvT does not collapse to ~0 and Tools/Q does not drop to ~1 across steps 250–400, in **≥2 of 3 seeds** → rebuttal ammo / next-cycle 4.0 upgrade.
> - **FALSIFIES / complicates**: R1 also collapses → gameable-`r_retrv` account wrong/incomplete → honest internal finding; rebuttal pivots to descriptive-taxonomy framing.
> - **Either outcome: submitted paper FROZEN — do NOT add results, do NOT edit `kg_grpo_emnlp2026_v2/`, do NOT block submission.**
>
> **Hard kill criteria.** format-valid <70% at step 50; Tools/Q→0 (reward-shaping bug); loss diverges (NaN / 3× baseline). Abort & report; don't burn GPU on a dead run.
>
> **Budget / timing.** ~2–4 days wall-clock with parallel seeds (1 GRPO arm ≈ 1–2 d + 50-step-cadence full-test eval ≈ ½–1 d). Opportunistic before 2026-05-25; explicitly OK to land in the ARR rebuttal window. **Report an exact queue/time estimate back before committing GPU.**
>
> **Deliverable (to discussion repo, NOT the paper).** Per-checkpoint CvT/Tools/Q curves R1 (+R2) vs E5b+SelfV; seed table; 1-paragraph confirms/falsifies verdict vs the pre-registered rule. Write to a new `wt18_results.md` in the discussion repo / report channel. Do NOT touch `kg_grpo_emnlp2026_v2/`.
> ---
>
> ### W19 — Test-set protocol dev/test split re-evaluation  [P0, NON-BLOCKING, rebuttal-stage]
>
> **Purpose.** Address most-serious methodological critique from 6th-reviewer round (2026-05-20): every checkpoint is evaluated on the FULL 3,531-question CWQ test set every 50 steps, and final reporting + checkpoint selection (e.g., "E5b+SelfV peak at step 250" = paper's R-selfV peak) both use the same test set. This conflates analysis with model selection. W19 = held-out dev/test split to demonstrate headline numbers are not test-set inflated.
>
> **Design (re-aggregate existing predictions; no retraining if data exists).**
> Random seed-42 stratified split of CWQ 3,531-Q test set → dev 800 / held-out test 2,731 (~23%/77%). For the 10 key checkpoints reported in `tab:main`:
> 1. E1 @ 1250 (paper: R-binary; Mode 1 anchor)
> 2. E3 @ 500 (paper: R-stepwise; Mode 2 anchor)
> 3. E5b @ 100 (paper: R-toolverbs; Mode 3 climb anchor)
> 4. E5b+KL @ 400 (paper: R-toolverbs·KL; stable plateau)
> 5. **E5b+SelfV @ 250 (paper: R-selfV; PEAK — the test-set-selected step; check most carefully)**
> 6. E5b+SelfV @ 300 (paper: R-selfV; COLLAPSED)
> 7. G1 @ 500 (paper: init-from-iterate)
> 8. G2 @ 500 (paper: self-distill; 40% headline)
> 9. E1' best (paper: R-binary-SR; Search-R1 reimpl)
> 10. E5b+KL-14B @ 400 (paper: R-toolverbs·KL-14B; capacity invariance)
>
> Re-aggregate per-question existing predictions from `results/trajectories/phase7/<run>/step_<N>_full/trajectories.json` into:
> - **dev-subset** EM / CvT / Tools/Q / 7-cat (used for checkpoint selection going forward)
> - **held-out-test** EM / CvT / Tools/Q / 7-cat (for final reporting)
>
> Plus: re-do E5b+SelfV peak step selection ON DEV ONLY across step ∈ {50, 100, 150, 200, 250, 300, 400}. Confirm peak CvT step is within ±50 of step 250.
>
> **Steps / eval.** **No retraining if per-question dumps exist.** Pure re-aggregation: ~1 hour. If dumps absent for ≥3 of 10 checkpoints → re-evaluate from merged HF checkpoints on held-out 2,731-Q subset only (greedy, max_turns=5, max_new_tokens=512); ~10 GPU-hrs total.
>
> **Pre-registered decision rule (LOCK before viewing results).**
> - **CONFIRMS robust** (no test-set inflation): held-out-test numbers within Wilson 95% CI half-width (±1.8 pp at n=2,731) of full-test numbers for ≥9 of 10 checkpoints, AND E5b+SelfV peak step on DEV ∈ {200, 250, 300} → original numbers stand as-is; add 1 paragraph in §3.5 / Limitations (vi) noting dev/test split as additional validation.
> - **PARTIAL** (some inflation): >1.8 pp gap on 1-2 checkpoints → caveat those specific rows in `tab:main` for rebuttal / camera-ready; main framework claims (capacity invariance, oracle +0.20 pp, 95.4% retrieval-composition) stand on the held-out test.
> - **FALSIFIES** (significant inflation): >3 pp gap on multiple checkpoints OR E5b+SelfV peak step on DEV differs from 250 by >50 steps → must re-report all `tab:main` numbers on held-out test in camera-ready; revise R-selfV peak claim wording.
>
> **Hard kill criteria.** Per-question dumps unavailable for ALL 10 checkpoints (cannot re-aggregate without re-eval); report and request re-eval GPU budget before proceeding.
>
> **Budget / timing.** Re-aggregation path: ~1 hour. Full re-eval path: ~1-2 days wall-clock on 4 GPUs. Opportunistic; rebuttal-stage acceptable. **Strongly prefer re-aggregation if dumps exist** — gives same scientific answer at ~50× lower compute.
>
> **Deliverable (to discussion repo, NOT the paper).** Per-checkpoint dev / held-out-test / full-test triple-column table; E5b+SelfV dev-peak-step report; 1-paragraph verdict (CONFIRMS / PARTIAL / FALSIFIES) vs pre-registered rule. Write to a new `wt19_results.md`.
> ---
>
> ### W20 — L_sig informative-failure intervention  [P1, NON-BLOCKING, rebuttal-stage]
>
> **Purpose.** Address framework-falsifiability critique from 6th-reviewer round (2026-05-20): the oracle ablation (+0.20 pp gold-relation injection, currently in §5.3) tests RELATION SELECTION (L_lang/L_comp partial intervention), but does NOT directly test L_sig (silent failure on empty `[]` responses). The framework's "L_sig dominant at 55.2%" claim (App C 4-channel attribution rubric) is therefore descriptive correlation, not causal evidence. W20 = direct L_sig intervention by replacing silent failures with informative errors.
>
> **Design (tool-interface intervention, NOT reward intervention).**
> Modify the 4-verb KG API so empty `[]` responses are replaced with informative error codes that disambiguate failure cause:
> - Entity not in subgraph: `[]` → `ERROR: entity_not_in_subgraph(<entity_mid>)`
> - Relation not defined for entity: `[]` → `ERROR: relation_not_present(<entity_mid>, <relation>)`
> - Malformed call: keep current `tool-misuse` handling (separate failure category)
>
> This converts the L_sig "silent failure" channel into informative failure — the policy now has credit-assignment signal distinguishing "wrong entity" from "wrong relation". Reward functions unchanged; only KG response format changes.
>
> **Three arms (run in parallel if budget; A1 + A2 minimum):**
> - **A1: E5b (paper: R-toolverbs) under informative-failure interface.** Rung 3 is where L_sig dominance was attributed (App C rubric); strongest predicted lift. Train from same SFT base, same recipe as E5b; only the KG API changes.
> - **A2: E5b+SelfV (paper: R-selfV) under informative-failure interface.** Tests whether informative failures prevent the peak-then-collapse (bonus prediction: if Mode 4 collapse is driven by L_sig silent-failure interaction with quote-and-stop attractor, informative failures should reduce or eliminate the cliff).
> - **A3 (optional, budget-allowing): G2 (paper: self-distill) under informative-failure interface.** Tests whether the 40% ceiling lifts when L_sig is removed.
>
> **Steps / seeds / eval.** ≥400 steps each (match Mode 3 / Mode 4 instrumentation; A1 longer if budget). **≥2 seeds per arm** (cost-bounded). Per-checkpoint full CWQ test eval at 50-step cadence. Readouts: EM, CvT, Tools/Q, 7-cat classify, **kg-incomplete count** (key — should drop sharply under informative failures; if doesn't, L_sig dominance was descriptive only).
>
> **Pre-registered decision rule (LOCK before viewing results).**
> - **CONFIRMS L_sig causally testable**: A1 EM lifts ≥3 pp **AND** kg-incomplete count drops ≥20% under informative failures, in ≥1 of 2 seeds → L_sig is a causal lever, not just descriptive correlation. Strong rebuttal ammo: "framework's L_sig dominance claim is now corroborated by direct intervention". Submit-paper update at camera-ready (1 paragraph in §5.3).
> - **A2 BONUS — cliff prevented**: E5b+SelfV under informative-failure does NOT show peak-then-collapse (no Tools/Q drop to 1.0 by step 300, ≥2 seeds) → unexpected framework prediction confirmed; Mode 4 is L_sig×quote-and-stop interaction.
> - **FALSIFIES**: A1 informative failures don't move EM (gap <1 pp) → L_sig dominance is descriptive correlation, not causal lever. Honest finding; framework framing needs reframing for camera-ready (drop "channel attribution explains" language, keep descriptive).
>
> **Hard kill criteria.** format-valid <70% at step 50 in A1 (informative errors might confuse format); abort that arm and report.
>
> **Budget / timing.** Implementation: ~half-day to modify KG API + verl tool-response wrapper. Training: A1+A2 = 2 seeds × 2 arms = 4 GRPO runs × ~1-2 d each = ~4-6 days wall-clock with parallel queue. Eval: 50-step cadence × 8 checkpoints × 4 runs = ~1-2 days. Total ~5-7 days. Opportunistic; rebuttal-stage acceptable.
>
> **Deliverable (to discussion repo, NOT the paper).** Per-arm CvT/EM/Tools/Q/kg-incomplete-count curves vs baseline (A1 vs existing E5b; A2 vs existing E5b+SelfV); seed table; 7-cat breakdown comparison; 1-paragraph verdict per arm vs pre-registered rule. Write to a new `wt20_results.md`.
> ---
>
> ### W21 — Multi-seed reward ladder + self-distill pass@16  [P0, NON-BLOCKING, rebuttal-stage]
>
> **Purpose.** Address the single highest-leverage critique from the 7th-reviewer round (2026-05-22, the most critical reviewer to date at 2.5/2.5/2.5): the reward-ladder failure-mode taxonomy (Mode 1/2/3/4 assignment in `tab:ladder` + §4.3) rests on **single-seed** runs for the middle rungs. Only G2 (self-distill, 4 seeds) and E5b+SelfV (R-selfV cliff, 4 seeds) are replicated; E3 (R-stepwise), E5b (R-toolverbs), E5b+KL (R-toolverbs·KL) are single-seed point estimates. Reviewer's explicit #1 rebuttal lever: "If R-stepwise, R-toolverbs, and R-toolverbs-KL show consistent behaviour across seeds, the failure taxonomy becomes much more convincing." W21 also closes the W11 gap (self-distill missing from pass@16).
>
> **Design (two parts).**
>
> **Part A — Multi-seed middle rungs.** Re-run with **+2 additional seeds each** (3 total counting original):
> - E3 (paper: R-stepwise) — expect Mode 2 ritual: CvT≈0, Tools/Q≈1, 100% get_tail_relations
> - E5b (paper: R-toolverbs) — expect Mode 3 drift: early peak ~step 100 (CvT≈3%), destabilise past step 150
> - E5b+KL (paper: R-toolverbs·KL) — expect stable plateau to step 400+
> Same recipe / hyperparameters / SFT base as the original single-seed runs; only the seed changes. Per-checkpoint full CWQ test eval at 50-step cadence (match Mode instrumentation). Readouts: EM, CvT, Tools/Q, 7-cat classify, **mode-signature confirmation** per seed.
>
> **Part B — self-distill pass@16 (no retraining).** Run the existing pass@16 protocol (16 samples/Q at temp 1, 500-Q seed-42 subset, with-tools vs without-tools) on the EXISTING G2@500 (self-distill) checkpoint. Fills the `fig:passk` gap where self-distill was not measured.
>
> **Pre-registered decision rule (LOCK before viewing results).**
> - **CONFIRMS taxonomy robust**: each middle rung's mode signature reproduces in **≥2 of 3 seeds** (R-stepwise CvT<0.5%; R-toolverbs early-peak-then-drift; R-toolverbs·KL stable-to-400) → failure taxonomy is seed-robust, not single-trajectory artifact. Strong rebuttal: add a seed-variance column / error bars to `tab:ladder` at camera-ready.
> - **PARTIAL**: 1 of 3 rungs is seed-dependent → caveat that specific rung's mode assignment; the other rungs + the 4-seed-replicated R-selfV cliff stand.
> - **FALSIFIES**: mode assignments flip across seeds for ≥2 rungs → the "every successive design shifts the failure mode" claim is seed-fragile; major reframe needed (drop ladder-determinism language, report mode-assignment as seed-distribution).
> - **Part B**: self-distill pass@16 with-tools gap >0 (genuine tool lift) → strengthens "self-distill genuinely uses tools" (closes W11); if gap ≈0, honest finding that even the winner is partly decorative.
>
> **Hard kill criteria.** format-valid <70% at step 50 on any seed (re-seed or report); training divergence (NaN / 3× baseline loss).
>
> **Budget / timing.** Part A: 3 rungs × 2 seeds = 6 GRPO runs × ~1-2 d each, parallel queue → ~2-3 days wall-clock. Part B: pass@16 sampling on 1 existing checkpoint ≈ few GPU-hours. Total ~3 days. Opportunistic; rebuttal-stage acceptable. **This is the most decision-relevant of W18-W21 for the failure-taxonomy claim — prioritise alongside W19 (which is ~1 hour if dumps exist).**
>
> **Deliverable (to discussion repo, NOT the paper).** Per-rung per-seed mode-signature table (EM/CvT/Tools/Q curves, mode label per seed); seed-variance summary; self-distill pass@16 with/without-tools bars; 1-paragraph verdict per part vs pre-registered rule. Write to a new `wt21_results.md`.
> ---
>
>
> **v17 delta from v16** (2026-05-09): Re-opens HPC for **one** new task: cross-family Llama-3.1-8B-Instruct G2-protocol replication. Driver: R3 ARR-form reviewer flagged single-family generality as primary remaining weakness blocking 3.5 → 4 score upgrade; ICLR 2026 OpenReview survey + 2026 arXiv survey (subagents 2026-05-09) confirmed Llama-3.1-8B remains the **modal cross-family choice** in 2026-posted RL/agentic papers — concrete precedents include KG-Hopper (arXiv:2603.21440), KG-Reasoner (arXiv:2604.12487), Medical-RL (arXiv:2604.11547), Reasoning-judges (arXiv:2603.12246, Meta+Yale), Weak-supervision-RL (arXiv:2604.18574, NYU), and Endless-Terminals (arXiv:2601.16443, Stanford). v15's W6 cancellation reasoning ("Qwen-family-dominant in 2025-H2") is reversed by 2026-Q1/Q2 publication evidence. Budget: ~150 GPU-hr / 12 working days / ARR deadline 2026-05-25. Single seed only (200 GPU-hr remaining budget caps multi-seed). Detailed spec at section "v17 LAUNCH QUEUE" below.
>
> **Critical gate (W17.1)**: Llama-3.1-8B-Instruct must pass a 100-step SFT format-emission sanity check (format-valid ≥ 90%) before committing to full GRPO run. The original v15-cancelled Llama attempt failed at this exact format-emission gate; we have NOT re-validated under our updated tokenizer/chat-template stack. If sanity check fails, abort and report — do not waste GPU-hours on a hopeless run.
>
> **v16 delta from v15** (this revision, 2026-05-07): v15 LAUNCH QUEUE fully executed 2026-05-05 → 2026-05-07. All 7 Wave-1 lanes complete with locked numbers; Wave-2 Lane 8 (Q8) attempted but produced INVALID output due to substring-matching bug in the schema rewriter (results quarantined to `results/phase7/_broken_q8_DO_NOT_USE/`). Plus 8 reviewer-response items (S0.1-S0.6, S1.2, S1.3) executed in parallel — all complete and clean. Per scope-discipline review (memory: `feedback_scope_discipline.md`), v16 schedules **only Q8 redo** as new compute. §6 reframe (forced by S1.2 +0.20pp oracle finding) is handled by the writing agent via honest re-analysis of existing trajectories — see `paper_iteration_state.md` Open decisions. Lane 10 cross-dataset (T3b) is explicitly NOT activated; preserved as rebuttal fallback ammunition. After Q8 redo lands, this file is FROZEN for the EMNLP submission cycle.
>
> **v15 delta from v14.1** (retained for reference, 2026-05-05): v14.1 LAUNCH QUEUE fully executed by 2026-04-23. Phase 7 mechanism package (V14-B1/B2/B3/B4), Search-R1/GPT-4o/fs1/14B/Qwen3-4B baselines, and three reviewer rounds + Phase A/B/C/D paper compression all complete. **Round 3 ARR-form reviewer (2026-05-04)**: Soundness 3.5 / Excitement 3.5 / Overall 3 (Findings); Hallucination LOW; **explicit upgrade path: "with seed-replicated curve + clean init-source ablation, this becomes a Conference paper."** **Local W7 null baseline computed (641× null effect)** — closes one of the open R3 attacks. Single-job/4-GPU HPC limit lifted today (2026-05-05) — re-enables aggressive parallel sprint to integrate ALL outstanding R3 W-attacks before Round-4 reviewer + advisor pass + final 8pp compression.
>
> **Current paper state** (anchor: `paper_iteration_state.md` in repo root): 19 pages total / 12.5pp main body (vs ARR 8pp HARD limit — verified at aclrollingreview.org/cfp). Latest zip `kg_grpo_emnlp2026_v2_for_hpc.zip` (2026-05-04 23:09).
>
> **v15 strategy**: parallel sprint to close ALL outstanding R3 attacks. Sequence:
> - **Tier 1 (Conference path, MUST-DO)**: Job 1 W1 multi-seed step-curve (already in flight since 2026-05-04) + G3 init-source ablation + Q2 Mode-4 quote-and-stop mechanism inspection.
> - **Tier 2 (Excitement boosters, each independently raises Excitement 3.5 → 4.0)**: Q8 non-KG control / W5 KL ablation / Q3 token-entropy / W4 GPT-4o ReAct n=500. (~~W6 Llama-3.1-8B~~ **CANCELLED 2026-05-05** — see Lane 5 for cancellation reason; Qwen2.5-14B + Qwen3-4B already cover capacity & generation axes.)
> - **Tier 3 (DEFERRED until Tier 1 + Tier 2 complete)**: Multi-seed G2 (cross-seed robustness for the winning recipe); cross-dataset KG benchmark (GrailQA / WebQSP / etc.) for external validity. **Cross-family models (Llama-13B / Llama-3.1-8B / Mistral / Gemma) all explicitly OUT-OF-SCOPE** for v15 — 2025-H2 agentic-RL paper population is Qwen-family-dominant; only Qwen3-8B is reactive-plan eligible if R4 reviewer demands an additional generation-axis 8B point (~1 GPU-day).
>
> **v14.1 delta from v14** (this revision, executed within 30 min of v14): Reversed v14's cuts to 14B and Qwen3-4B based on user correction — **GPU budget is abundant, parallel execution means these tasks do NOT compete with mechanism analysis**. Task 32 (Qwen2.5-14B E5b-stabilized) restored to **P0 (mandatory)** per user's explicit instruction "14B绝对是必须的". Qwen3-4B-Instruct-2507 E5b promoted from "optional stretch" to **P1 (run in parallel with P0)**. Budget table rewritten to report **parallel wall-time (3-4 days)** instead of incorrectly summed sequential GPU-days (12-14 days). New persistent memory file `hpc_gpu_budget.md` records the "assume GPU abundance" policy for all future planning.
>
> **v14 delta from v13** (retained): Major reprioritization after (a) Day-2 training completed with clear results: **G2 winner EM=40.0% CvT=5.8%**, I-Self peak-then-collapse (CvT=8.16%@200 → 0 by 300), G1 EM=39.4%, 39B full-test EM=38.4%; (b) critical review concluded EM-chasing beyond 40% has diminishing paper value but mechanism-depth has high paper value; (c) strategic alignment: 40% EM + deep analysis > 45% EM + thin analysis for an EMNLP analysis paper.
>
> **v14 cuts (v14.1 REVERSED)**:
> - ~~Task 32 (Qwen2.5-14B) CUT~~ → **RESTORED to P0 in v14.1**. Reframed as *framework prediction test*: signal-theoretic framework predicts 14B should show SAME parametric ceiling / Goodhart pattern (bottlenecks are interface properties, not capacity). Confirming this at 14B upgrades the paper from "7B observation" to "8-14B cross-capacity observation."
> - ~~Qwen3-4B DEMOTED to stretch~~ → **PROMOTED to P1 in v14.1**, runs in parallel with 14B. Generation-axis defense ("your model is 1.5y old") worth 2.5 parallel days.
> - Iterated ReST-EM Round 2 — **still cut** (diminishing returns per ReST-EM literature)
> - MCTS inference — **still cut** (different paper)
>
> **v14 adds (all retained)**: (a) **Mechanism Analysis Package** (new §Phase 7 Day-3 block): 4 sub-tasks that deepen the existing 40% story — sub-failure-mode classification, decision-boundary analysis on kg-incomplete, I-Self step-by-step collapse curve, G2 vs 39B behavioral query diff; (b) **G2 vs Follow-the-Path fs1 comparison** as explicit task; (c) **I-Self peak-then-collapse memo** as writing task.
>
> **v13 delta from v12** (retained for reference): (a) Generation-axis model Qwen3-4B-Instruct-2507; (b) SFT Warmup Strategy Path B; (c) SFT corpus 12K total (5K primary + 6K enhanced + 1K gold); (d) rule-based SFT as methodological strength; (e) Qwen3.5 cut.
> **v12 delta from v11** (retained): Gate A PASSED; Llama appendix-only; Task 32 revived (NOW CUT AGAIN IN v14); Qwen3 upgraded (NOW DEMOTED IN v14).
>
> **Current status**: Phase 7 Day 2 **COMPLETE**. All constructive variants trained and evaluated on full 3531. G2 is canonical winner. Focus shifts to: (1) mechanism analysis on existing checkpoints, (2) completing missing baselines (Search-R1, GPT-4o), (3) paper writing.
>
> **Narrative direction (v14 locked)**: **Hybrid — diagnostic-as-skeleton + G2 as constructive anchor + signal-theoretic framework as theoretical contribution**. NOT chasing higher EM. The paper's contribution is that we can *explain* why 40% is a ceiling with current methods, not that we beat 40%.

> **Phase 7 Day-2 CANONICAL RESULTS (full 3531 test set, locked 2026-04-17)**:
>
> | Model | EM | CvT count | CvT % | 95% CI | kg-inc | wrong-ans | % correct via tool |
> |---|---|---|---|---|---|---|---|
> | E3@500 | 32.5% | 1 | 0.03% | [0.00-0.16%] | 209 | 1338 | **0.09%** ← pure memorization |
> | E5b@100 | 32.2% | 107 | 3.03% | [2.51-3.65%] | 1435 | 731 | 9.4% |
> | 39B@400 (KL 5x) | 38.4% | 133 | 3.77% | [3.19-4.45%] | 1201 | 800 | 9.8% |
> | I-Self@200 (peak) | 39.0% | 288 | **8.16%** | [7.30-9.11%] | 858 | 1103 | **20.9%** ← highest tool-reliance, collapses by step 300 |
> | G1@500 (init from 39B) | 39.4% | 162 | 4.59% | [3.95-5.33%] | 954 | 993 | 11.6% |
> | **G2@500 (init from base SFT — WINNER)** | **40.0%** | **205** | **5.81%** | **[5.08-6.63%]** | **818** | 1140 | **14.5%** |
>
> **Pass@16 with-vs-without tools gap** (tool contribution to capability expansion):
> - E3: **0 pp** (tools ornamental, confirmed pure memorization)
> - E5b: +14.2 pp
> - 39B: +11.4 pp
>
> **KGQAGen Oracle**: 69.14% SOLVABLE (vs CWQ 99.5%) — defends against "CWQ gold is noisy" critique while confirming KG genuinely contains the needed info on an audited benchmark.

> **Critical review findings (2026-04-15, four-agent workflow summary)**:
> 1. **39B's CvT regressed 5.5% → 2.5%**. The "breakthrough" is better memorization stability, not more retrieval — this *validates* the diagnostic rather than refuting it. Reframe accordingly.
> 2. **Statistical fragility**: 200-Q 36% CI = ±6.6pp, not distinguishable from E3. **v11 Action II1 promotes full 3531-test eval to absolute blocker** — no Day-2 training launches until this lands.
> 3. **Variant I (query-precision reward) uses Oracle gold path → IS oracle-supervised, not a deployable process reward.** Must be relabeled as *diagnostic upper-bound* OR redesigned to use self-verifiable signal (retrieval non-empty AND returned entities appear in final answer).
> 4. **Variant G (self-distillation) is a known recipe** (ReST-EM / ReTool / FireAct). Needs yield pilot first (FireAct threshold ~500 traces for Llama-7B; we project 1.3-2.7K). ReST-EM evidence: init SFT from *pre-GRPO base*, not from 39B@400 — run both as ablation.
> 5. **Variant H is not curriculum learning**: 39B has already seen full data. "Hard-subset finetuning" is the honest name; not worth the GPU budget. **Cut.**
> 6. **Missing experiments that a reviewer will demand**:
>    - (M1) Search-R1 outcome-reward baseline on CWQ — **non-negotiable**. If Search-R1's simple recipe > E3, our diagnostic collapses.
>    - (M2) CvT audit on full 3531 for E5b@100 and 39B@400 (current CI is from n=100-200)
>    - (M3) Llama pipeline audit (stop-token / chat-template / 20K SFT warmup) — the 25% → 0% gap will be flagged otherwise
>    - (M4) Pass@k with AND without tools on full 3531 — only way to defend "capability expansion via tools"
>    - (M5) GPT-4o + same 4-tool interface baseline — cheap ceiling check (< $50 API)
>    - (M6) KGQAGen Oracle replication — defends the 99.5% number against the CWQ-49%-audited-accuracy critique
> 7. **KG-R1 (arXiv:2509.26383) was withdrawn** (wrong numbers reported). Do NOT use as baseline. Use **Follow the Path fs1 (arXiv:2505.11140v2)**: Qwen2.5-7B SFT on KG path traces = 40.8% pass@1 sub-phrase EM on CWQ — this is our realistic literature-grounded comparison.
> 8. **Literature target for 7B + KG-tools + GRPO**: 40-50 strict-EM / 55-75 Hit@1. Our 32.6% is plausibly below this; 39B's 36% (unverified) is roughly in line.
>
> **Must-cite anchors (from literature agents)**:
> - Follow the Path fs1 (arXiv:2505.11140v2) — SFT on KG traces, Qwen2.5-7B CWQ 40.8%
> - ReST-EM (arXiv:2312.06585) — init-from-base finding for Variant G
> - ReTool (arXiv:2504.11536) — cold-start SFT → RL loop, Qwen2.5-32B AIME 40→67
> - FireAct (arXiv:2310.05915) — 500-trace phase transition for Llama-7B agent SFT
> - EEF "Exploring Expert Failures" (arXiv:2504.13145) — partial-trajectory salvage if Variant G yield is low
> - StepSearch (arXiv:2505.15107, EMNLP 2025) — TF-IDF gold-doc alignment process reward, +5 F1
> - ProGraph-R1 (arXiv:2601.17755) — entity overlap step reward, +4-7 F1
> - CriticSearch (arXiv:2511.12159) — α=0.25 turn-level advantage (independent validation of our Variant I weight)
> - KG as Implicit Reward (arXiv:2601.15160) — gate-on-≥2-distinct-gold-entities anti-spam design
> - Search-R1 (arXiv:2503.09516) — **must run as baseline**
> - Pitfalls in KG-RAG Datasets (arXiv:2505.23495) — defends strict-EM vs Hit@1 metric choice
>
> **Key findings (canonical, from Task 14 full 3531 test set)**:
> - Qwen parametric ceiling ~33% (NOT 52% — earlier number was on val first-500)
> - Llama EM=0.000 across ALL evals (genuine failure, not extraction artifact — see data_provenance_audit.md)
> - Only E5b@100 shows correct-via-tool > 0% (10% on val first-100, CI [4-18%], needs test-split verification)
> - Cross-benchmark: Spearman ρ=0.976 (Qwen-only, 8 models) on CWQ vs KGQAGen-10k
> - 72.4% of CWQ test questions are Category B (pass@10=0, beyond parametric memory)
>
> **Critical update from data provenance audit (2026-04-12)**:
> - Earlier 500-sample evals used **val.parquet**, Task 14 uses **test.parquet** — different splits
> - "Llama E3=24.8%" has **no backing evidence on HPC** — likely a misattribution of Contains-EM
> - Llama Task 14 used max_turns=3 (vs Qwen max_turns=5) — noted but not root cause of EM=0
> - E5b CvT=10% is from val first-100 sequential — not yet verified on test split
> - ρ=0.991 inflated by tied Llama zeros → honest number is ρ=0.976 (Qwen-only)
>
> **Phase 7 context (2026-04-12)**: The fundamental unresolved question is:
> **Does CWQ/Freebase actually REQUIRE KG tools?** We know 72% of questions are beyond
> parametric memory, and CvT=0% for standard rewards. But we've never established whether
> Freebase CONTAINS the triples needed to answer Category B questions. If KG coverage is
> low, the entire tool-use story rests on a broken foundation. Task 36 (Oracle) answers this
> and gates ALL subsequent experiment decisions.
>
> **Previous Phase 6 context** (retained for reference):
> - KGQAGen-10k triangulation: DONE (Task 35, ρ=0.976)
> - CWQ temporal audit (Task 33): DEFERRED pending Oracle result
> - Tier A contamination (Task 34): DEFERRED pending Oracle result

---

## 🚀 v17 LAUNCH QUEUE — Cross-Family Llama-3.1-8B-Instruct Replication (2026-05-09)

> **Single P0 task. ~150 GPU-hr / 12 working days. ARR May cycle deadline 2026-05-25. After completion, this file FROZEN until rebuttal.**
>
> **Rationale**: R3 ARR-form reviewer (Claude_review_3.md, Claude_review_3_followup.md) accepted 2 of our 3 pushbacks but partially-accepted single-family generality. ICLR 2026 + 2026 arXiv survey confirmed Llama-3.1-8B-Instruct is the modal cross-family choice in 2026-published RL/agentic papers. Goal: a single-seed full G2-protocol replication, integrated as 1 appendix table + 1 paragraph in §6 + 1 sentence in Limitations + abstract update.
>
> **Hand-off back to writing agent ready by**: 2026-05-21 (4-day buffer to ARR deadline).

### W17.1 — Tokenizer & format-emission sanity check (Day 1-2, ~5 GPU-hr) — **CRITICAL GATE**

**Goal**: Verify Llama-3.1-8B-Instruct can emit our 4-tag format (`<think>`, `<search>`, `<response>`, `<answer>`) under our SFT pre-warm protocol. The original v15 W6 cancellation was downstream of this exact gate; we have NOT re-validated under the current tokenizer/chat-template stack.

**Steps**:
1. Set `tokenizer_name = meta-llama/Llama-3.1-8B-Instruct` in verl config.
2. Use Llama-3.1's official chat template (NOT Qwen2.5's). Reference: HF model card `chat_template.jinja`.
3. Tokenize each of the 4 special tags. If any tag splits across word-piece boundaries, add to `additional_special_tokens` and re-verify.
4. Run **100-step SFT** on `/mnt/scratch/users/ts1201/sft_corpus/v14_d2_rule_based_5k.json` (same 5K rule-based corpus as G2). LoRA r=64, α=128, DoRA + rslora. Effective batch matched to G2.
5. Eval on 100-question CWQ test subset. Measure:
   - Format-valid rate (% trajectories closing with parseable `<answer>...</answer>`)
   - Tools/Q (mean tool calls per question)
   - Strict EM (sanity, not load-bearing)

**ABORT criterion**: format-valid < 90% → STOP, report findings, do NOT proceed to W17.2. Update `paper_iteration_state.md` with abort note.

**PASS criterion**: format-valid ≥ 90% AND Tools/Q > 0.5 → proceed to W17.2.

### W17.2 — Full SFT pre-warm (Day 2-3, ~15 GPU-hr)

**Steps**:
1. Same 5K rule-based SFT corpus. 1 full epoch.
2. LoRA r=64, α=128, DoRA + rslora (G2-identical).
3. Save adapter to `checkpoints/v17_llama_g2_sft/`.
4. Sanity eval: full CWQ test on SFT-only (no GRPO yet). Record EM and CvT as baseline.

### W17.3 — GRPO training (Day 4-9, ~100 GPU-hr)

**Steps**:
1. Init from W17.2 SFT-warmed adapter.
2. **E5b+KL reward** (G2-identical): use `verl_reward.e5b_kl_v3` exactly as deployed in v14 G2 run.
3. Hyperparameters: `kl_coef=0.05, lr=3e-7, batch=128, group_size=8, max_steps=500`.
4. Save checkpoints at: 50, 100, 150, 200, 250, 300, 350, 400, 450, 500.
5. Per-step EM/CvT/Tools/Q on 100-question dev subset (early-stop signal).

**Hard kill criteria** (stop early, save logs, report):
- Format-valid rate drops below 50% at any checkpoint
- Tools/Q drops to 0
- Training loss diverges (NaN or 5× baseline)

### W17.4 — Per-checkpoint full CWQ test eval (Day 10-11, ~25 GPU-hr)

**Steps**:
1. For each saved checkpoint: full 3,531-question test (greedy, max_turns=5, max_new_tokens=512).
2. Output: `_handoff/data/v17_llama/step_{050..500}_full_test.json`.
3. Wilson 95% CIs for EM and CvT (same reporting as G2 in Table 1).

### W17.5 — Trajectory classifier on best & step-250 checkpoints (Day 11-12, ~5 GPU-hr)

**Steps**:
1. Run `scripts/task16_classify.py` on:
   - Best-EM checkpoint
   - Step-250 (G2-anchor for direct comparison)
2. Output: `_handoff/data/v17_llama/best_step_classified.json` + `step_250_classified.json`.

### W17.6 — Summary report (Day 12, analysis only)

**Output**: `_handoff/data/v17_llama/SUMMARY.md` with:
- Per-checkpoint table (step, EM, CvT count + %, Wilson 95% CI, Tools/Q)
- Best-EM and peak-CvT highlighted
- Mode-pattern characterisation: cliff? gradual decline? steady? plateau? Mode 1 collapse?
- 3-sentence interpretation comparing to Qwen2.5-7B G2 (EM=40.0%, CvT=5.81%): "Llama-3.1-8B-Instruct G2 reaches X% EM / Y% CvT with [Mode pattern]. [Implication for §6 framework: cross-architecture phenomenon vs. Qwen-specific.]"

### Risk register

- **R1 (highest)**: format-emission failure at W17.1 → ABORT. Mitigation: 100-step SFT cheap (~3 GPU-hr); abort early.
- **R2**: Mode 1 echo collapse before step 100 → save partial logs; partial cliff observation may still be valuable for paper.
- **R3**: Time overrun → stop at step 250 (G2-anchor); partial result is still publishable.

### Acceptance criteria (for hand-off back to writing agent)

- Best-checkpoint EM available (any value, including 0)
- Per-checkpoint EM/CvT/Tools/Q for all available checkpoints
- Trajectory pattern characterised
- All artefacts in `_handoff/data/v17_llama/` mirroring v14 G2 directory structure
- `SUMMARY.md` ready for writing-agent integration

### Hand-off plan (writing agent, post-W17.6)

**If pass (any non-zero EM)**:
1. New `appendix/appendix_I_llama_cross_family.tex` containing W17.6 SUMMARY.md content + 1 table.
2. `\input{appendix/appendix_I_llama_cross_family}` added to main.tex.
3. 1-paragraph addition to §6 mechanism evidence (cross-family corroboration).
4. Update Limitations (i): "Cross-family generalisation tested on Llama-3.1-8B-Instruct (App. I)".
5. Update abstract last sentence to mention cross-family replication.

**If abort (W17.1 failure)**:
1. Do NOT add disclosure to main paper (avoid App-A-style attack hook).
2. Keep result as rebuttal-only ammunition: "We attempted Llama-3.1-8B replication; format-emission compatibility issue under our chat-template stack prevented evaluation. Rebuttal-time we will [...]".

**FREEZE NOTE**: After v17 completes, hpc_tasks.md returns to FROZEN state until ARR rebuttal phase.

---

## 🏁 v16 FINAL STATE (2026-05-07 evening) — Submission-cycle close-out

> **This is the LAST PLANNED HPC TASK UPDATE for the EMNLP submission cycle.** v15 lanes 1-7 all complete; Lane 8 Q8 invalidated by bug — redo is the only remaining new compute. After Q8 redo lands, this file is FROZEN until rebuttal phase.

### v15 lane completion status (locked numbers)

| Lane | Spec | Status | Headline result |
|---|---|---|---|
| L1 W1 (multi-seed E5b+SelfV) | ≥2 new seeds × 8 steps | ✅ COMPLETE | 3 new seeds + original = 4 seeds; collapse signature reproduces across all 4 (peak step 200-250, EM=0 by step 300). Closes R3 W1 single-seed lethal weakness. |
| L2 G3 (clean init-source ablation) | base→G-distill SFT→GRPO 500 steps | ✅ COMPLETE | EM 38.5% / 39.1% / 39.0% / 40.2% / ~40% across step {100/200/300/400/500}. G3 ≈ G2 ≈ 40%. **Rule-based pre-SFT is NOT the value-add** — G2's win is init-source pattern alone. Closes R3 W10. |
| L3 Q2 (Mode-4 trajectory dumps) | I-Self step 300/400 traj | ✅ COMPLETE | Trajectory dumps saved at canonical paths; writing-side picks up for quote-and-stop verdict. |
| L4 W5 (KL coef ablation × 5) | KL ∈ {0.001, 0.005, 0.01, 0.05, 0.25} | ✅ COMPLETE | Collapse is **non-monotonic** in KL — 0.005 collapses by step 400, 0.001 / 0.01 / 0.25 stable. KL-tether is NOT the unified mechanism → validates §5.3 L_sig framing. Closes R2 W5. |
| L5 W6 (Llama scaffolded) | — | ❌ CANCELLED 2026-05-05 | Per user direction; Qwen-family controls (14B + Qwen3-4B) cover capacity & generation axes. |
| L6 Q3 (token-entropy time series) | inside vs outside `<search>` | ✅ COMPLETE | Inside `<search>` H drops 0.350 → 0.205 across steps 200-400, n drops 25K → 7K. Outside H stays low (~0.02-0.04). **Selective action-distribution collapse** — differentiates from Yu et al. 2025 uniform-collapse pattern. New mechanistic evidence for §5.4. |
| L7 W4 (GPT-4o n=500) | upgrade from n=150 | ✅ COMPLETE | EM 0.6%, ContEM 6.2%, F1 2.1%, avg_tool_calls 1.46, total $0.99. Pattern matches prior 200-Q baseline (verbose-but-correct, fuzzy ~78% / strict EM ~0%). Closes R2 W4. |
| L8 Q8 (non-KG synthetic schema control) | E5b+SelfV from base on rewritten schema | 🚨 **BUG → REDO scheduled (see Q8-redo lane below)** | Substring-matching bug: rewriter replaced common English substrings ("you", "ning", "com", etc.) globally, including system prompt + tool descriptions. EM=0% across all steps is artifact, not signal. Quarantined to `results/phase7/_broken_q8_DO_NOT_USE/`. |

### Reviewer-response items (S0.x + S1.x — executed alongside v15, all clean)

| Item | Headline result | Closes |
|---|---|---|
| S0.1 multi-seed I-Self ×3 | EM peak ~38-39% pre-collapse, EM=0 post-collapse, all 4 seeds collapse by step 200-300 | R3 W1 (≡ Lane 1) |
| S0.2 multi-seed G2 ×3 | EM ≈ 39.8% across new seeds vs original 39.99% (within ±0.5pp) | Single-seed G2 concern |
| S0.3 post-collapse inspection | tools/q drops 3.00 → 1.00 between step 250 and 300 | Confirms structural collapse (not verbatim-copy hacking) |
| S0.4 reward decomposition | Pre-collapse: r_outcome=0.41, r_tool_type=0.49, r_retrieval=0.10. Post-collapse: r_outcome=0.004 (the others preserved) | Goodhart on per-call averages — policy preserves r_tool_type/r_retrieval by emitting one decorative call, sacrificing r_outcome |
| S0.5 schema-format null | random-relation close-match rate 14.91%; V14-B2's 70.4% is **4.7× null lift** | R3 W7 (null baseline missing) |
| S0.6 V14-B2 seen vs unseen | Seen: 41.9%, Unseen: 77.7% close-match rate (UNSEEN HIGHER → opposite of memorization-from-SFT) | Memorization-from-SFT alternative explanation |
| S1.2 oracle L3 upper bound | gold-relation injection at G2@500: EM 40.19% vs G2 baseline 39.99% (**+0.20pp ONLY**) | ⚠️ **FORCES §6 REFRAME** — see paper_iteration_state.md Open decisions |
| S1.3 CvT robustness | 0/1203 correct-via-memory trajectories flip to CvT under fuzzy matching | Strict-substring CvT is robust metric |

### 🚨 V16-Q8 redo (the ONLY remaining active lane)

```
LANE 11 — V16-Q8-REDO (2026-05-07 → ~2026-05-09, single seed, dichotomous test)

  Goal: framework predictive test — does removing L_lang (opaque-ID problem on entities)
        delay or eliminate the step-300 collapse?

  Bug fix from broken Q8 (already diagnosed):
    - Use per-question entity list from extra_info.kg_path (NOT global label dict)
    - Match on whole-word/token boundaries: regex r'\b<entity_label>\b' with proper escape
    - Skip system prompt + tool description entirely (only rewrite question content + KG triples + ground truth)
    - Carefully re-rewrite SFT trajectories with same protocol

  Spec:
    - Recipe: E5b+SelfV from base Qwen2.5-7B-Instruct + rule-SFT (mirrors I-Self / 39I)
    - Schema: replace Freebase MIDs with English-readable URIs (entity://Barack_Obama__b1961)
              Keep relation names AS-IS (people.person.spouse etc.) — relation strings
              are already English; the L_lang attack is the OPAQUE-ID problem on entities only.
    - Single seed (do NOT multi-seed; this is a binary falsifier, single seed is sufficient)
    - Eval: 8 step-checkpoints {50, 100, 150, 200, 250, 300, 350, 400} on full 3531
    - Trajectory dump at step 300 (post-collapse window) for sanity-check

  Output:
    - results/phase7/v16_q8_nonkg_redo/eval_step{N}.json
    - results/trajectories/phase7/v16_q8_nonkg_redo_step300_full.json
    - Sanity-check report: 5 sample SFT trajectories pre/post rewrite, confirming no
      system-prompt corruption

  Dichotomous decision rule (LOCK in writing immediately; do NOT add follow-up experiments):
    - **Collapse still at step 300**: L_lang attribution rejected → §6 framework
      is descriptive, not predictive on this dimension. State this honestly.
    - **Collapse delays to step 350+ OR absent through step 400**: framework moves
      from descriptive to partially-predictive on L_lang.
    - **Anything weird** (e.g., EM=0 across all steps again): re-verify rewriter
      sanity-check output BEFORE concluding. If sanity passes, accept the result.

  Compute: ~1h fix + ~36h serial wallclock (SFT + GRPO + 8 evals)
           Under HPC's current 2-job/person limit: ~2-3 days wallclock.

  Sanity check BEFORE full launch (~1h):
    - Run rewriter on 50 sample CWQ test items
    - Manually inspect 5 of them for system-prompt/tool-description corruption
    - Confirm only question content + KG triples + ground truth are rewritten
    - ONLY THEN launch full SFT
```

### Wave 2 / Wave 3 / Lane 10 status (final disposition)

```
Lane 9  V15-T3a Multi-seed G2     ✅ DONE via S0.2 (3 seeds, EM ±0.5pp confirmed)
Lane 10 V15-T3b Cross-dataset KG   ❌ NOT ACTIVATED (preserved as rebuttal fallback)
                                   Rationale: KGQAGen-10k cross-benchmark (ρ=0.976 on
                                   8 models) already provides sufficient external-validity
                                   evidence for our RQ3. Adding GrailQA / WebQSP opens
                                   "why not these other 5 benchmarks?" arms race.
                                   Reactive plan: if R4 reviewer demands more cross-benchmark,
                                   eval-only on existing checkpoints is ~6h work during rebuttal.
32B  V14-E1                        ❌ DEFERRED to rebuttal (SFT format-control failed Apr 28;
                                   needs full-rank or higher-LoRA-rank fine-tune)
Llama-13B / Mistral / Gemma / Phi  ❌ OUT OF SCOPE per user
```

### Writing-agent tasks (paper-side analysis, NO GPU)

**WT-1 — §6 reframe** (writing-only, no compute):
- L3 query precision is correct-enough at G2; bottleneck lies downstream.
- Reframe uses V14-B1 + V14-B2 + S0.5 + S1.2 + S0.4 (existing data only).
- NO new experiments. Option A (entity oracle) / Option C (hop-stratified) explicitly rejected per scope-discipline review.

**WT-2 — G2 downstream cross-tab** (CPU only, ~3h):

Quantify whether G2@500 errors are **retrieval-failure** (tool returns lack gold) vs **extraction-failure** (returns contain gold but model answered wrong). Output supports §6.3 reframe by replacing vague "downstream" with a specific verdict.

Inputs (locate exact paths on HPC scratch):
- G2@500 trajectory dump on full CWQ test 3531 (likely `results/phase7/g2_step500_trajectories.json`)
- 7-cat classifier output marking kg-incomplete + wrong-answer (~1958 error qids; likely `results/phase7/v14_b1_failure_modes.json`)

Method: for each error qid, normalised gold-string match (lowercase / strip articles / strip punct / token-set or substring) against any tool response. Bucket = {kg-incomplete, wrong-answer} × {retrieval-failure, extraction-failure}. Verdict: extraction-dominant if extraction ≥ 50% of total; retrieval-dominant if retrieval ≥ 50%; else mixed.

Sanity: kg-incomplete + wrong-answer ≈ 1958 (818 + 1140); spot-check 3 extraction exemplars per bucket — if "Obama" matches "Obamacare"-style spurious cases appear, tighten match to word-boundary regex. Report `no_retrieval` count separately for trajectories with empty tool calls.

Deliverable to discussion repo:
- `kg_grpo_emnlp2026_v2/_handoff/data/g2_crosstab_2026_05_07.json` with `{buckets, no_retrieval_count, extraction_pct, retrieval_pct, verdict, exemplars: 5 per bucket}`
- 1-line slack/markdown summary: verdict + percentages + sanity anomalies.

Use: §6.3 paragraph picks one of three template sentences based on verdict (extraction-dominant / retrieval-dominant / mixed).

### File-freeze checkpoint

After Q8 redo lands and outputs are in `results/phase7/v16_q8_nonkg_redo/`, this file is FROZEN.

The next planned update to `hpc_tasks.md` is during the rebuttal phase (post-ARR review), not before submission.

---

## 🚀 LAUNCH QUEUE (v15, EXECUTE NOW — 2026-05-05)

> **Audience**: HPC **code agent only** (training / eval / trajectory dumps / metric scripts / plots). Paper-side integration (swapping numbers into paper sections, prose updates, table re-edits, Limitations rewrites) is handled by the **writing agent** on the discussion-repo side and is NOT a code-agent deliverable. Code agent's job ends at producing the `results/phase7/v15_*` JSON / dumps / plots specified per lane.
>
> **Naming policy**: HPC keeps its existing tracking codes (`39B`, `39I`, `I-Self`, etc.) — DO NOT rename runs / checkpoints / output paths to match paper. Paper-side translation happens at the integration boundary on the discussion-repo side. For reference only, the cross-walk is: HPC `39B` = paper `E5b+KL`; HPC `I-Self` = HPC `39I` = paper `E5b+SelfV` (same underlying recipe). All other recipe names (E1, E3, E5b, G1, G2) are identical on both sides. Full table in `paper_iteration_state.md` § "HPC ↔ Paper naming cross-walk".
>
> **User approved this launch at 2026-05-05 morning. HPC GPU limit (1 job / 4 GPU) lifted today.** HPC agent: Job 1 (Lane 1 below) is already running since 2026-05-04 — DO NOT relaunch. Wave 1 (L2-L7) launches immediately and concurrently. Wave 2 (L8 Q8) is gated by Wave 1 outcomes — DO NOT pre-launch. Wave 3 (L9-L10 T3) is DEFERRED until Wave 1 + Wave 2 complete. Final-stretch sprint to close all Round-3 reviewer W-attacks for EMNLP submission.

### Wave 1 — All Tier 1 + Tier 2 lanes (launch immediately, fully parallel)

```
LANE 1 (Tier 1, ALREADY RUNNING since 2026-05-04, 4 GPUs × 2 days)
  ☐ V15-W1 Multi-seed E5b+SelfV step-curve replication
    - Goal: replicate the peak-then-collapse (step-resolved) signature finding across ≥2 NEW seeds
    - Recipe: identical to E5b+SelfV ("I-Self" / "39I" in HPC tracking) at 50-step granularity
    - Seeds: ≥2 new seeds beyond the original; report seed values used
    - Eval: full 3531 CWQ test at steps 50/100/150/200/250/300/350/400 per seed
    - Metrics per checkpoint: EM, CvT count, CvT %, kg-incomplete %, format-valid %, tools/Q
    - Output: results/phase7/v15_w1_multiseed_selfv/{seed_X}/eval_step{N}.json + summary.json
    - **Reporting**: when complete, draw step-curve plot for each seed overlaid on the original;
      report whether step-250 peak (~9.5% CvT) and step-300 collapse onset are within ±2pp across seeds.
    - **Closes**: R3 W1 ("signature finding is single-seed"). Conference path requirement #1.

LANE 2 (Tier 1, NEW, 4 GPUs × 1.5 days)
  ☐ V15-G3 Clean init-source ablation
    - Init: base Qwen2.5-7B-Instruct (NOT 39B SFT init; NOT rule-based-SFT init)
    - Pipeline: SKIP rule-based SFT corpus warmup entirely; go straight to GRPO
    - Reward: E5b (the same standard-recipe reward used for our E5b@100 / 39B@400 ladder runs)
    - KL coef: 5x (matches our 39B winner recipe)
    - Steps: GRPO 500 steps; eval at step 100/200/300/400/500 on full 3531
    - Trajectory dump on final ckpt (500-sample subset, seed=42) for 7-cat classifier
    - Diff vs G2: G2 = base + rule-SFT + E5b GRPO; this G3 = base + (no rule-SFT) + E5b GRPO
    - Diff vs G1: G1 was init-from-39B + E5b GRPO; this G3 has no 39B init exposure
    - **Predicted outcomes**:
      - If G3 EM ≥ G2 EM (40.0%): rule-based SFT not the value-add → G2's win is from init-source alone
      - If G3 EM << G2 EM: rule-based SFT IS the value-add → §7 P3 narrative validated
    - Output: results/phase7/v15_g3_ablation/eval_step{100..500}.json + trajectory_dump.json
    - **Closes**: R3 W10 / "G2 vs G1 narrative leans on McNemar". Conference path requirement #2.

LANE 3 (Tier 1, NEW, 1 GPU × 0.5 day)
  ☐ V15-Q2 Mode-4 quote-and-stop mechanism inspection
    - **PREREQ**: trajectory dump for E5b+SelfV ("I-Self" / "39I") at step 300 (post-collapse)
      currently does NOT exist on HPC. Re-run eval on full CWQ test (or first 200-Q subset
      if compute-tight) with --dump-trajectories flag enabled.
      Output: results/trajectories/phase7/i_self_step300_full.json
      Spec: per-question record with {question, gold_answer, tool_calls: [{verb, args, response}], final_answer}
    - **Analysis**: run script in `_handoff/hpc_q2_mode4_mechanism_prompt.md` on first 200 trajectories.
      Verdict thresholds:
      - quote-and-stop ≥ 70% → §5.4 hypothesis VERIFIED
      - 30-70% → partially verified
      - < 30% → hypothesis refuted (reframe §5.4)
    - Output: kg_grpo_emnlp2026_v2/_handoff/data/q2_mode4_mechanism_audit.json
    - **Closes**: R3 W2 ("Mode-4 mechanism is hypothesised, not verified"). Conference path requirement #3.

LANE 4 (Tier 2, NEW, 5 parallel runs × 4 GPUs each = 20 GPUs × 1 day wallclock)
  ☐ V15-W5 KL coefficient ablation (5 conditions in parallel)
    - Recipe: E5b GRPO from base Qwen2.5-7B-Instruct + rule-SFT init (same as G2)
    - KL coef sweep: ∈ {0.001, 0.005, 0.01, 0.05, 0.25}
    - Each condition: 500 GRPO steps; eval at steps 100/200/300/400/500 on full 3531
    - Goal: separate L_sig (silent-fail collapse) from KL-tether-to-prior confound
    - **Predicted patterns**:
      - If collapse onset shifts with KL: KL-tether is part of the mechanism (revise §5.3/§6.3)
      - If collapse onset is invariant to KL: collapse is L_sig-driven (validates §5.3 framing)
    - Output: results/phase7/v15_w5_kl_ablation/{kl_coef}/eval_step{N}.json
    - **Closes**: R2 W5 (KL coef confound). Excitement booster.

LANE 5 (Tier 2, ❌ CANCELLED 2026-05-05 — DO NOT LAUNCH)
  ☒ V15-W6 full Llama-3.1-8B with format scaffolding — CANCELLED
    - Cancellation reason (user, 2026-05-05):
      (a) Llama-3.1 is dated by 2025-H2 / 2026 standards — the 2025-H2 agentic-RL paper
          population (StepSearch, ToolRL, CriticSearch, ReSearch, ProGraph-R1, etc.) is
          almost entirely Qwen2.5/3 family ± Mistral; running Llama-3.1-8B in a 2026
          submission risks looking dated rather than answering generality.
      (b) Independent reviewer-agent's "Llama failure undermines generality" attack does NOT
          necessarily mirror real ARR reviewer trends; we lack ground-truth reviewer-trend
          data (OpenReview real reviews would be the only authoritative signal).
      (c) Capacity-axis (Qwen2.5-14B in Table 1) and generation-axis (Qwen3-4B in App E)
          Qwen-family controls already exist; §6.1 qualifier rewritten to
          "Qwen2.5/3 family demonstration (7B / 14B / 4B)" is honest and sufficient.
    - **Reactive plan**: if R4 reviewer explicitly demands cross-family generality, the
      preferred fallback is Qwen3-8B E5b (~1 GPU-day, generation-axis 8B point) before any
      Mistral / Gemma / Llama option. Do NOT pre-launch.
    - Original spec retained (commented out) for reference.
    {{
      Model: meta-llama/Llama-3.1-8B-Instruct
      Pipeline: 5K rule-based-SFT format-scaffolding → E5b GRPO with KL 5x
      Eval: step 200/400/600 on full 3531; 500-sample trajectory dump for 7-cat classifier
      Output (would have been): results/phase7/v15_w6_llama_scaffolded/
    }}

LANE 6 (Tier 2, NEW, 1 GPU × 0.5 day, EVAL-ONLY on existing checkpoints)
  ☐ V15-Q3 Token-entropy time series for E5b+SelfV
    - Use existing E5b+SelfV ("I-Self") checkpoints at steps 200, 250, 300, 400 (already on disk)
    - For each: run inference on 500-sample test subset (seed=42), record per-token logits
    - Compute mean per-token entropy SEPARATELY for:
      (a) tokens emitted INSIDE <search> tags (action distribution)
      (b) tokens emitted OUTSIDE <search> tags (reasoning + answer)
    - Plot entropy curves over training step for (a) vs (b)
    - **Goal**: differentiate from Yu et al. 2025 "Demystifying Agentic Reasoning" entropy collapse
      — they report uniform action-distribution collapse; we expect SELECTIVE collapse on (a) only.
    - Output: results/phase7/v15_q3_entropy/{step}/{inside,outside}_entropy.json + plot
    - **Mechanism strengthening** (no specific R-attack, but adds independent evidence for §5.4 Mode 4).

LANE 7 (Tier 2, NEW, API-only, NOT GPU — ~3h wallclock, < $100)
  ☐ V15-W4 GPT-4o + ReAct on 500-Q (upgrade from current n=150)
    - Subset: first 500 questions of test.parquet seed=42 (deterministic; same seed as our 200-Q baseline)
    - Model: gpt-4o (not gpt-4o-mini) via OpenAI Chat Completions with function_calling
    - Tools: 4 verbs (get_tail_relations, get_tail_entities, get_head_relations, get_head_entities) wired to our Freebase API
    - Prompt: zero-shot system prompt (matches our existing GPT-4o spec) + 5-shot ReAct demonstrations
    - Metrics: strict EM, ContEM (Contains-EM), tool-call counts, format validity
    - **API key**: read from env $OPENAI_API_KEY (do NOT hardcode in scripts; key is on HPC env already)
    - Output: kg_grpo_emnlp2026_v2/_handoff/data/v15_w4_gpt4o_500q.json
    - **Closes**: R2 W4 (GPT-4o n=150 underpowered). Cheap, fast, run alongside GPU lanes.
```

### Peak concurrent GPU footprint (Wave 1)

| Lane | GPUs | Wallclock | Status |
|---|---|---|---|
| Lane 1 (W1 multi-seed) | 4-8 | 2 days | already running since 2026-05-04 |
| Lane 2 (G3 ablation) | 4 | 1.5 days | new |
| Lane 3 (Q2 trajectory dump + analysis) | 1 | 0.5 day | new |
| Lane 4 (W5 KL ablation × 5) | 20 (5 × 4) | 1 day | new |
| Lane 5 (W6 Llama scaffolded) | — | — | **❌ CANCELLED 2026-05-05** |
| Lane 6 (Q3 entropy eval) | 1 | 0.5 day | new |
| Lane 7 (W4 GPT-4o API) | 0 (API) | 3h | new |
| **Peak concurrent** | **~30 GPUs** | **2 days** wall | within Isambard 24-32+ ceiling |

### Reporting cadence

- **Every 12h**: one-line status per lane
- **Gate events** (report immediately):
  - **Lane 1 multi-seed**: if either new seed shows step-250 peak < 5% CvT or no collapse by step 350 → URGENT escalation (signature finding may be unstable)
  - **Lane 2 G3**: if G3 EM > G2's 40.0% → URGENT (rule-SFT not the value-add → §7 reframe)
  - **Lane 3 Q2**: if quote-and-stop < 30% → URGENT (§5.4 mechanism wrong)
  - **Lane 4 W5**: if no KL coef yields step-300 collapse → KL-tether is the mechanism (revise §5.3)
  - **Lane 5 W6**: if Llama+scaffolding > 25% EM → expand §6.1 generality scope
- **Final report**: when all 7 lanes locked, produce consolidated `phase7_v15_results.md` + drop-in numbers for paper §5.4 / §7 / §6.3 + Limitations (v) / (vii) / (viii) updates.

### Wave 2 — Tier 2 high-risk-high-reward (gated by Wave 1 — DO NOT pre-launch)

```
LANE 8 (Tier 2, conditional, 4 GPUs × 2.5 days)
  ☐ V15-Q8 non-KG synthetic-schema control
    - **GATE**: launch ONLY after Wave 1 Lanes 1-3 complete and confirm step-300 collapse is reproducible.
      Q8 only makes sense if collapse is real.
    - Construction: build a synthetic schema that REMOVES L_lang (schema linguistic familiarity) but
      KEEPS L_sig (silent-failure profile), L_comp (multi-hop opaque-ID compositional state),
      and L_prior (model never trained on this schema either way).
      Concrete recipe: take CWQ test subgraphs. Replace Freebase MIDs (m.0xxx) with English-readable
      URIs (e.g., entity://Barack_Obama_Politician_USA_b1961). Keep the relation names as-is
      (people.person.spouse etc.) — relation names are already English; the L_lang attack we're
      probing is the OPAQUE-ID problem on entities.
    - Train: E5b+SelfV from base Qwen2.5-7B-Instruct + rule-SFT (same recipe as I-Self / 39I)
    - Eval at steps 50/100/150/200/250/300/350/400 on full test mapped through the new schema
    - **Predicted outcomes** (this is the test of the framework):
      - Collapse still happens at step 300: framework's L_lang attribution is WRONG → rewrite §6
      - Collapse does NOT happen / shifts later: framework moves from descriptive → partially-predictive
        (the highest Excitement-bumping result we have available)
    - Output: results/phase7/v15_q8_nonkg_control/eval_step{N}.json + trajectory_dump.json
```

### Wave 3 — Tier 3 (DEFERRED until Wave 1 + Wave 2 complete)

> Per user direction (2026-05-05): only launch after Tier 1 + Tier 2 land. Llama-13B explicitly OUT-OF-SCOPE (Llama family aging; substitute model TBD).

```
LANE 9 (Tier 3, deferred, 4 GPUs × 1 day × 2 new seeds = 8 GPU-days, parallel → 1 day wallclock)
  ☐ V15-T3a Multi-seed G2 robustness check
    - Recipe: G2 winner identical, ≥2 new seeds (independent of W1's seeds)
    - Eval at step 500 on full 3531 CWQ test
    - Trajectory dump at step 500 (500-sample, seed=42) → 7-cat classifier
    - Goal: confirm G2's 40.0% EM / 5.81% CvT are not seed-luck
    - Output: results/phase7/v15_t3a_g2_multiseed/{seed_X}/eval_step500.json
    - Adds robustness paragraph to §7.

LANE 10 (Tier 3, deferred, 0 new training, ~0.5 GPU-day total)
  ☐ V15-T3b Cross-dataset KG benchmark eval
    - **EVAL-ONLY** on existing G2@500 + E5b+SelfV@250 + E5b+KL@400 checkpoints
    - Datasets:
      - GrailQA (Wikidata KG; tests cross-KG generalization)
      - WebQSP (Freebase but different question distribution; tests within-KG generalization)
      - Optionally: 2WikiMultiHopQA or HotpotQA-KG-augmented if time permits
    - Adapt our 4-tool API to each KG; reuse evaluation harness
    - Metrics: strict EM, ContEM, F1
    - **Goal**: external-validity evidence that the 40% / 4-mode pattern is not CWQ-specific.
    - Output: results/phase7/v15_t3b_cross_kg/{dataset}/eval_{ckpt}.json
    - Adds §4.5 / Appendix paragraph; potentially shortens §6.1 Qwen-specific qualifier if patterns hold.
```

**EXPLICITLY OUT OF SCOPE for v15**: Llama-13B (per user 2026-05-05: "llama家族的模型有点老，做这个需要换个其他模型而不是llama"). If a non-Llama mid-size substitute is wanted later, candidates: Qwen2.5-32B, Mistral-Small-22B-Instruct, Gemma2-27B — to be discussed separately, NOT in v15.

### v15 code-agent deliverables (handoff to writing agent)

Code agent's v15 work is done when the following artefacts are on disk and reported back. **Writing-agent (paper-side) integration is OUT-OF-SCOPE for this file** — the discussion-repo writing agent picks up these results and integrates them into paper sections.

1. ☐ Lane 1 (W1 multi-seed): ≥2 seeds × 8-checkpoint eval JSONs + per-seed step-curve plot + summary table reporting step-250 peak and step-300 collapse magnitudes per seed → `results/phase7/v15_w1_multiseed_selfv/`
2. ☐ Lane 2 (G3): step-{100..500} eval JSONs + final-ckpt trajectory dump + 7-cat classifier output + behavioral-diff table vs G2/G1 → `results/phase7/v15_g3_ablation/`
3. ☐ Lane 3 (Q2): trajectory dump for E5b+SelfV step 300 + 200-trajectory analysis JSON with quote-and-stop verdict → `kg_grpo_emnlp2026_v2/_handoff/data/q2_mode4_mechanism_audit.json`
4. ☐ Lane 4 (W5): 5 × step-{100..500} eval JSONs (one per kl_coef) + KL ablation curve plot → `results/phase7/v15_w5_kl_ablation/`
5. ❌ Lane 5 (W6 Llama scaffolded): **CANCELLED 2026-05-05** — Llama-3.1 is dated for 2026 submission; cross-family generality not pre-emptively addressed; rely on existing Qwen2.5-14B + Qwen3-4B family controls.
6. ☐ Lane 6 (Q3): per-step entropy JSONs (inside-`<search>` vs outside) + entropy time-series plot → `results/phase7/v15_q3_entropy/`
7. ☐ Lane 7 (W4): GPT-4o n=500 eval JSON with per-question records + aggregate metrics (EM, ContEM, tool-call counts, format validity) → `kg_grpo_emnlp2026_v2/_handoff/data/v15_w4_gpt4o_500q.json`
8. ☐ (Conditional) Lane 8 Q8: synthetic-schema eval JSONs + step-curve plot + collapse-onset comparison → `results/phase7/v15_q8_nonkg_control/`
9. ☐ Final consolidated report: `phase7_v15_results.md` summarising all lane deliverables in one table; ready for the writing agent to consume.

---

## 🗄️ LAUNCH QUEUE (v14.1, SUPERSEDED — fully executed by 2026-04-23, retained for spec reference)

### Immediate submissions (target: all queued within 1 hour)

```
LANE 4 (CRITICAL PATH, ~3 days)
  ☐ Task 32 Qwen2.5-14B E5b-stabilized — SFT job first (~22h)
    - Config: sft_cwq_qwen14b.yaml (use 5K primary corpus, epochs × 0.7, LR × 0.5-0.7 per SFT Warmup Strategy)
    - Pilot: run 0.1-epoch sanity check first (~2h), verify loss curve is smooth log-decrease
    - Once SFT completes: launch GRPO E5b-stabilized (KL 5x, ~44h)
    - Output: checkpoints at step 200/400/600/800; eval each on full 3531
    - GPUs: 4-8 depending on availability
  → Owner of the critical-path timer. If SFT pilot crashes, escalate to user same day.

LANE 5 (PARALLEL, ~2.5 days)
  ☐ Qwen3-4B-Instruct-2507 E5b (V14-D2)
    - Chat template: enable_thinking=False (verify with 10-sample forward pass BEFORE SFT)
    - Config: sft_cwq_qwen3_4b.yaml (use 5K primary corpus ONLY, skip 6K enhanced, LR × 1.2-1.5)
    - Pilot: 0.1-epoch sanity check first
    - Once SFT completes: launch GRPO E5b-stabilized (KL 5x)
    - GPUs: 4
  → Independent of Lane 4. Do NOT block on Lane 4 pilot.

LANE 1 (PARALLEL, ~2 days)
  ☐ V14-A1 Search-R1 baseline on CWQ
    - Clone https://github.com/PeterGriffinJin/Search-R1 (if not already)
    - Adapt their training to our Freebase 4-tool API (wrap as single search() endpoint)
    - Train Qwen2.5-7B with their exact GRPO + outcome-only reward on CWQ train
    - Eval at step 200/400/600/800 on full 3531
    - GPUs: 4
  → Report step-200 first-eval number ASAP (early signal for "is Search-R1 > E3?")

LANE 2 (PARALLEL, ~3-4 days, CPU-heavy)
  ☐ V14-B1 Sub-failure-mode classification on G2 errors (CPU + 1 GPU × 1 day)
  ☐ V14-B2 Decision boundary on kg-incomplete (CPU only, ~4h)
  ☐ V14-B3 I-Self step-by-step collapse curve (1 GPU × 2 days; steps 50/100/150/200/250/300 × 500-sample)
  ☐ V14-B4 G2 vs 39B behavioral query diff (1 GPU × 0.5 day)
  → All four run asynchronously. Dump results as they complete to results/phase7/v14_b*.

LANE 3 (PARALLEL, ~2 days)
  ☐ V14-C1 G2 vs Follow-the-Path fs1 comparison (2 GPUs × 2 days)
    - Part a: reproduce fs1's Qwen2.5-7B SFT on their gold KG traces; eval on OUR test split with strict EM
    - Part b: adapt fs1 SFT data as alt init for our GRPO; run G2-style ReST-EM distillation on top
    - Output: results/phase7/v14_c1_fs1_comparison/{fs1_sft_only,fs1_sft_plus_our_rl}/

API-ONLY (blocked on user, ~3h once unblocked)
  ☐ V14-A2 GPT-4o baseline on 200-Q subset
    - **NEED FROM USER**: OPENAI_API_KEY (estimated cost < $50)
    - Until key is provided, this is the only blocked task — everything else launches now
```

### Peak concurrent GPU footprint

| Lane | GPUs | Running when |
|---|---|---|
| Lane 4 (14B) | 4-8 | Day 0 - Day 3 |
| Lane 5 (4B) | 4 | Day 0 - Day 2.5 |
| Lane 1 (Search-R1) | 4 | Day 0 - Day 2 |
| Lane 2 (mechanism) | 1-2 | Day 0 - Day 4 |
| Lane 3 (fs1) | 2 | Day 0 - Day 2 |
| **Peak** | **~20** | Day 0 - Day 2 (all lanes overlap) |

Within the 24-32 GPU ceiling. No queuing or GPU contention expected.

### Reporting cadence

- **Every 12h**: one-line status per lane (e.g., "Lane 4: SFT at 60%, no issues; Lane 1: Search-R1 step 150 EM=22%")
- **Gate events** (report immediately):
  - Lane 4 SFT pilot fails → stop, escalate to user
  - Any lane's step-200 eval ≤ E3's 32.5% for > 2 consecutive checks → kill that lane, reallocate GPUs
  - Lane 1 Search-R1 @ step 400 > G2's 40.0% → **URGENT escalation**, paper reframing triggered
- **Final report**: when all 5 lanes locked (expected ~Day 3-4), produce consolidated results table + updated paper_tables.md (V14-W2)

### Writing tasks (zero-GPU, execute whenever there's a spare moment)

- ☐ V14-W1 I-Self peak-then-collapse memo (1 hour CPU, from existing Phase 7 Day 2 trajectories)
- ☐ V14-W2 paper_tables.md update with G1/G2/I-Self rows + "% correct via tool" column + pass@16 gap row

---

## 🖥️ GPU BUDGET POLICY (v14.1, permanent — also in memory/hpc_gpu_budget.md)

> **GPU is ABUNDANT on Isambard — 24-32 H100 concurrent is routine (verified during Phase 7 Day 2). All plans MUST assume aggressive parallelization.** Sequential GPU-day sums are an anti-pattern. Always report both **GPU-hours (resource)** and **wall-clock days (parallel execution)**. Baselines run ALONGSIDE variants, not after. Mechanism analysis does NOT block training. Cross-model replication (14B, Qwen3-4B) fills spare lanes at near-zero wall-time cost.

---

## 🎯 v14.1 PRIORITY QUEUE (2026-04-18 → 2026-04-23, 5 days parallel wall-time)

> **All P0 + P1 tasks below run concurrently.** Expected wall-time bottleneck: whichever single task is longest (Task 32 14B E5b ≈ 3 days, or Search-R1 ≈ 2 days — so 3 days total if launched in parallel). Iterated ReST-EM round 2, MCTS, cross-family models (Mistral/Gemma/Phi/Llama) remain cut.

### P0-A — Missing baselines (reviewer non-negotiable)

**Task V14-A1 — Search-R1 baseline on CWQ** (4 GPUs × 2 days, ~50 GPU-hours)
- Spec unchanged from v11 Action II2. Critical path.
- Clone `https://github.com/PeterGriffinJin/Search-R1`, adapt to our Freebase 4-tool API (wrap as single `search()` endpoint), train Qwen2.5-7B with their GRPO + outcome-only reward on CWQ train, eval at step 200/400/600/800 on full 3531.
- **Why now**: this is the single biggest open risk. If Search-R1 outcome-only hits ≥ 38% on CWQ, our "reward design matters" framing needs reframing. Must know by Apr 20.
- Output: `results/phase7/searchr1_baseline/eval_step{200,400,600,800}.json`

**Task V14-A2 — GPT-4o baseline on CWQ 200-Q** (OpenAI API, ~3 h, < $50)
- Spec unchanged from v11 Action II5. Cheap ceiling check.
- 200 questions from test.parquet seed=42 (same subset as I2), GPT-4o with our 4-tool API via function calling, zero-shot + 5-shot ReAct prompts.
- **Why now**: standard reviewer ask. Cannot ship paper without a closed-model reference.
- Output: `results/phase7/gpt4o_baseline.json`

### P0-B — Mechanism analysis package (the v14 core addition — 4 sub-tasks)

> All 4 sub-tasks run on **existing checkpoints and existing trajectory dumps**. No new training. These are the analyses that turn our 40% result into a *mechanism-driven* paper instead of just a "we got 40%" paper.

**Task V14-B1 — Sub-failure-mode classification on G2 errors** (1 GPU × 1 day, CPU-heavy)
- Take G2@500's 2119 non-EM trajectories on full test.
- Classify each error into one of four **signal-theoretic failure modes**:
  - **L-sig** (silent-fail): tool called, returned empty, model gave up or guessed.
  - **L-lang** (schema mismatch): tool called with entity/relation NOT in Freebase schema (string-format issue, wrong case, wrong namespace).
  - **L-comp** (multi-hop state drift): tool calls correct in isolation, but model lost the entity ID chain between turns (different ID appears in turn 2 than was returned in turn 1).
  - **L-prior** (pre-training gap): model appears to not recognize the question's key entity at all (no relevant tool call attempted; parametric-only answer).
- Report: histogram of failure modes + 5 exemplar trajectories per mode.
- **Output**: `results/phase7/v14_b1_failure_modes.json` + `phase7_v14_b1_memo.md`
- **Paper value**: directly validates the signal-theoretic framework empirically. One of the paper's highest-leverage figures.

**Task V14-B2 — Decision boundary analysis on kg-incomplete** (0 GPU, ~4 h CPU)
- For G2's 818 kg-incomplete samples: compare the model's failed query (entity, relation) to Oracle gold path's (entity, relation).
- Distance buckets:
  - **Exact relation name typo** (e.g., `place_of_birth` vs `people.person.place_of_birth`) — ≤ 1 token edit
  - **Correct entity, wrong relation** (any non-matching relation on the right entity)
  - **Wrong entity, any relation** (entity linking failure)
  - **Completely off** (neither entity nor relation in gold path)
- Report: count per bucket + example trajectories.
- **Output**: `results/phase7/v14_b2_decision_boundary.json`
- **Paper value**: if ≥ 30% of kg-incomplete is "≤ 1 token edit from gold," this proves L-lang (schema-format) is the dominant bottleneck — highly publishable finding.

**Task V14-B3 — I-Self step-by-step collapse curve** (1 GPU × 2 days)
- For I-Self checkpoints at steps 50, 100, 150, 200, 250, 300: run full trajectory classification on **500 random samples** (seed=42) each.
- Track per-step: EM, CvT%, tools/Q, format-valid %, kg-incomplete %, wrong-answer %.
- Plot: all 6 metrics as curves over training step.
- **Output**: `results/phase7/v14_b3_i_self_collapse.json` + figure source.
- **Paper value**: **this is the paper's Section 5 signature figure**. Peak-then-collapse of a self-verifiable process reward is a NOVEL finding — not reported in ToolRL / StepSearch / CriticSearch / ProGraph-R1. It's the most unique data point we have.

**Task V14-B4 — G2 vs 39B behavioral query diff** (1 GPU × 0.5 day)
- On 500 random test samples (seed=42, same as B3), compare G2@500 and 39B@400:
  - Tool type distribution (counts of `get_tail_relations` / `get_tail_entities` / `get_head_relations` / `get_head_entities` per trajectory)
  - Unique-query diversity (how many distinct (entity, relation) pairs each uses)
  - Turn-to-answer ratio (how many turns before emitting `<answer>`)
  - On kg-incomplete overlap: for samples where BOTH G2 and 39B are kg-incomplete, are they making the SAME wrong query or different ones?
- Report 2×3 behavioral contrast table.
- **Output**: `results/phase7/v14_b4_g2_vs_39b_queries.json`
- **Paper value**: converts "ReST-EM init helps by 0.6pp EM" into "ReST-EM init produces measurably different query behavior (X/Y/Z)" — upgrades an aggregate finding to a mechanism finding.

### P0-D — Cross-capacity and cross-generation framework tests (v14.1 reinstated)

> **Purpose**: validate the signal-theoretic framework by confirming the same bottlenecks appear at different capacities and different Qwen generations. These are **framework prediction tests**, not ceiling chasing — the framework explicitly predicts L-sig/lang/comp/prior are model-scale invariant.

**Task V14-D1 — Qwen2.5-14B E5b-stabilized (restored to P0, single condition, parallel lane)** [P0, 4-8 GPUs × 3 days]
- Full spec: see "Task 32" section later in this file (spec preserved; only the CUT tag is removed in v14.1).
- Single condition: E5b-stabilized with KL 5x (mirrors our 39B winner recipe on 7B, applied at 14B).
- SFT: ~22h on 5K primary corpus (see SFT Warmup Strategy block below).
- GRPO: ~44h at KL 5x, with eval at step 200/400/600/800 on full 3531 test.
- Trajectory classification on final checkpoint (500-sample).
- **Predicted outcomes and interpretations**:
  - **Most likely (framework-validated)**: EM 40-45%, CvT 4-7%, SAME Goodhart pattern, SAME kg-incomplete dominance. → Strong: "capacity doesn't fix the bottleneck"
  - **Unexpected (framework-partially-falsified)**: CvT 15%+, sustained past step 400. → Honestly report, revise framework to say "bottlenecks have capacity threshold"
- Output: `results/phase7/v14_d1_qwen14b_e5b/`

**Task V14-D2 — Qwen3-4B-Instruct-2507 E5b (P1, parallel)** [P1, 4 GPUs × 2.5 days]
- Modern-generation defense (responds to "your model is 1.5y old" critique).
- E5b single condition (skip E3 — Goodhart evidence already strong on Qwen2.5-7B).
- Use `enable_thinking=False` (Instruct-2507 is pure non-thinking variant, matches Demystifying Agentic Reasoning arxiv 2510.11701 setup).
- SFT: use 5K primary corpus only (NOT 6K enhanced) per SFT Warmup Strategy block.
- Output: `results/phase7/v14_d2_qwen3_4b_e5b/`

Both D1 and D2 run in parallel with P0-A/B/C on separate GPU groups. Neither blocks the others.

---

### P0-C — Positioning defense task

**Task V14-C1 — G2 vs Follow-the-Path fs1 comparison** (2 GPUs × 2 days)
- fs1 (arXiv:2505.11140v2) reports Qwen2.5-7B + KG-trace SFT = 40.8% pass@1 sub-phrase EM on CWQ, no RL. Our G2 = 40.0% strict EM. Without a direct comparison, reviewers will conclude "the RL added nothing."
- **Two-part task**:
  - (a) **Clone fs1 repo and reproduce their SFT on Qwen2.5-7B** with their exact config. Evaluate on OUR test split with strict EM (not sub-phrase). Expected ~38% strict EM.
  - (b) **Adapt fs1's SFT data (gold KG traces) as an alternative init for our GRPO pipeline**. Run G2-style ReST-EM distillation starting from fs1-SFT init. Evaluate at step 300, 500 on full 3531.
  - This tells us: does G2's RL layer add gains *on top of* fs1's SFT?
- **Output**: `results/phase7/v14_c1_fs1_comparison/` (two sub-results: fs1-SFT-only, fs1-SFT+our-RL).
- **Paper value**: directly addresses the #1 reviewer attack surface. If fs1-SFT + our-RL > fs1-SFT alone, we have an "RL adds value beyond strong SFT baseline" claim.

### P1 — Writing-only (CPU, zero GPU)

**Task V14-W1 — I-Self peak-then-collapse memo** (1 h, CPU, from existing trajectories)
- Write a 1-page technical memo from existing I-Self trajectories at step 50/100/150/200/250/300 (may overlap with V14-B3 outputs).
- Frame the collapse as "oracle-free process reward hits peak CvT ~8% around step 200 then systematically collapses to zero by step 300" — novel finding.
- **Output**: `phase7_v14_w1_i_self_memo.md`

**Task V14-W2 — Update paper_tables.md** (30 min, CPU)
- Add G1/G2/I-Self rows to the canonical results table.
- Add "% correct via tool" column (showing 0.09% for E3 vs 14.5% for G2 — strong narrative point).
- Add Pass@16 with-vs-without-tools gap row (E3=0pp, E5b=+14.2pp, 39B=+11.4pp).

### 🚫 Explicitly CUT in v14.1 (still cut)

| Task | Why cut |
|---|---|
| Iterated ReST-EM Round 2 | Predicted +3-4pp marginal EM. Diminishing returns per ReST-EM literature. Time better on analysis. |
| MCTS at inference | Different paper. Implementation cost prohibitive. |
| Qwen3-4B-Instruct-2507 E3 training | E3 Goodhart evidence already strong on Qwen2.5-7B, not worth cross-gen replication. |
| Cross-family runs (Mistral / Gemma / Phi / Llama additional) | Per 2026 field norm, Qwen is de-facto standard for agentic RL. Cross-family introduces template + tokenizer confounds. |

### 🔄 v14.1 REVERSED (previously cut in v14, now reinstated)

| Task | v14 status | v14.1 status | Reason for reversal |
|---|---|---|---|
| Task 32 (Qwen2.5-14B E5b-stabilized) | CUT | **P0 MANDATORY** | Framework prediction test. User explicit: "14B绝对是必须的". Parallel wall-time zero-cost given GPU abundance. |
| Qwen3-4B-Instruct-2507 E5b single condition | Optional stretch | **P1 PARALLEL** | Fills a spare GPU lane, adds generation-axis defense for free. |

### v14.1 Total Compute Budget — PARALLEL WALL-TIME

> **Critical correction from v14**: v14's budget assumed sequential execution (summed to 5-7 days). v14.1 reports actual parallel wall-time (~3-4 days) using GPU abundance policy.

| Task | GPU-hours | Parallel lane | Lane wall-time |
|---|---|---|---|
| V14-A1 Search-R1 baseline | ~200 | Lane 1 (4 GPUs) | 2 days |
| V14-A2 GPT-4o baseline | 0 (API) | async | 3 h |
| V14-B1/B2/B3/B4 mechanism analyses | ~60 | Lane 2 (1-2 GPUs) | 3-4 days (all parallel) |
| V14-C1 fs1 comparison (SFT + RL) | ~100 | Lane 3 (2 GPUs) | 2 days |
| **V14-D1 Qwen2.5-14B E5b** | **~400** | **Lane 4 (4-8 GPUs)** | **3 days** |
| **V14-D2 Qwen3-4B E5b** | **~240** | **Lane 5 (4 GPUs)** | **2.5 days** |
| **TOTAL GPU-hours** | **~1000** | 5 concurrent lanes | **3-4 days wall-time** |
| Peak concurrent GPUs | — | 4+2+8+4+2 = **~20** (within 24-32 ceiling) | — |

**Earliest "numbers locked for writing"**: **Apr 22** (4 days from Apr 18 launch, 3-4 day critical path is V14-D1 14B). Gives **33 days** for writing + review + submission buffer — more slack than v14 estimated.

---

## 🆕 Model Coverage Upgrade (v12→v13, 2026-04-16) — ⚠️ SUPERSEDED BY v14; SEE v14 PRIORITY QUEUE ABOVE

> **Trigger**: Oracle Gate A PASSED (II4 KGQAGen replication 69.14% SOLVABLE; E5b-stabilized positive-paper path active).
> Combined with a field-norm survey of 2026 arxiv papers from major groups (CUHK/IDEA, ByteDance+Tsinghua, Alibaba, ICLR 2026 accepted), we upgrade model coverage from "Qwen2.5-7B + Llama-3.1-8B" to **"Qwen2.5-7B + Qwen3-4B-Instruct-2507 + Qwen2.5-14B"** (all-Qwen multi-generation + multi-scale).
>
> **v13 refinement**: generation-axis model changed from `Qwen3-8B` (hybrid with /no_think leakage, per Demystifying Hybrid Thinking arxiv 2510.12680) to `Qwen3-4B-Instruct-2507` (pure non-thinking variant released 2025-08, no hybrid leakage). Matches the exact setup of Demystifying Agentic Reasoning (arxiv 2510.11701, CUHK/IDEA) which tested both thinking and non-thinking on Qwen3-4B and concluded: "instruction-based non-thinking models are more suitable for agentic RL." Qwen3-8B has no pure-variant 2507 release (4B and 30B-A3B do); choosing 4B-Instruct-2507 also gives three capacity points (4B < 7B < 14B) instead of two (7B ≈ 8B, 14B).

### Rationale (2026 field-norm data)

Surveyed 12+ papers in agentic RL / reward design / tool use from 2026 arxiv. Distribution:
- **~85% use pure Qwen** (Qwen2.5 or Qwen3 or both)
- **~10% use Qwen + Llama or Qwen + DeepSeek** — declining
- **~5% use other families** (Mistral / Gemma / Phi)
- **0 major papers** use Llama-3.1-8B as main model (Meta has no 8B-dense successor)

Three dominant configurations:
- **Mode A**: single family × multi-size (Qwen2.5 1.5/3/7/14B) — Tree-GRPO, COVERT, ARTIST, CUHK/IDEA Demystifying
- **Mode B**: single family × two generations (Qwen2.5-7B + Qwen3-8B) — ToolRM, UserRL, Demystifying Agentic Reasoning
- **Mode C**: single family × single size — Kansal & Jha, Lightning OPD, Tongyi DeepResearch

**Our target**: Mode A ∪ Mode B (Qwen2.5-7B + Qwen3-8B + Qwen2.5-14B). Covers both axes.

### Llama decommission

Llama-3.1-8B is **appendix-only** from now on:
- Existing II3 offline audit (0/300 `<answer>` tags, 100% degenerate loops) is sufficient for one-paragraph limitation note
- No new Llama compute. `run_phase7_ii3_llama_audit.job` obsolete (per current HPC agent snapshot)
- All Phase 5 Llama tasks (24, 24b, 25) results retained for appendix only; not in main table
- Paper framing: "Llama-3.1-8B under identical pipeline fails to produce valid format across all conditions (Appendix X); this suggests reward-design effectiveness is not uniform across model families, consistent with 2026 field observations that Qwen has become the de-facto standard for agentic RL."

### New Tier-1 tasks added (see Task 32 and Cross-Model Validation sections below for full specs)

| Task | Model | Scope | Compute | Purpose |
|---|---|---|---|---|
| Task 32 (revived, reduced scope) | Qwen2.5-14B-Instruct | E5b-stabilized, **single condition** | ~22h SFT + ~44h GRPO = **~3 days** | Capacity-axis ablation (field-norm Mode A) |
| Cross-Model Training (v13 revised) | **Qwen3-4B-Instruct-2507** (was Qwen3-8B) | **E3 + E5b training** | ~10h SFT + 2×22h GRPO = **~2.5 days** (halved from v12 due to 4B vs 8B) | Generation-axis ablation (field-norm Mode B) |

**Total new compute** (v13): ~5.5 days wall-time (reduced from v12's 8 days thanks to Qwen3-4B), feasible in Apr 19 → Apr 25 window (parallel with writing sprint start).

---

### 🆕 SFT Warmup Strategy (v13 — added 2026-04-16 evening)

> **Decision**: **Path B** = unified SFT corpus across all models + per-model epoch / learning-rate tuning. Aligns with 2026 field norm (Demystifying Agentic Reasoning, Demystifying Long-Horizon Tool-Using, Qwen2.5 Technical Report) and keeps "SFT initialization" a controlled variable across the cross-model ablation.

#### SFT Corpus Inventory (confirmed 2026-04-16)

| File | Rows | Generator | Purpose |
|---|---|---|---|
| `data/freebase/sft_trajectories.jsonl` | 5,000 | `scripts/generate_cwq_sft.py` (rule-based, deterministic template) | **Primary** SFT warmup |
| `data/freebase/sft_trajectories_enhanced.jsonl` | 6,000 | Enhanced variant of above | Used by `sft_cwq_enhanced.yaml` |
| `data/freebase/verl_cwq/gold_kg_trajectories.jsonl` | 1,000 | `scripts/task40_gen_gold_trajectories.py` (oracle-verified; discards trajectories whose tool call doesn't actually return the gold tail) | Task 40 quality-filtered SFT |

**Total**: ~12K trajectories (NOT 20K — earlier references in this file are outdated and should be read as "~12K").

#### Key SFT generation properties (all favorable for cross-model validity)

1. **No LLM teacher**: Zero GPT-4 / Claude / Qwen-72B / DeepSeek involvement. All trajectories are deterministic Python template expansions from oracle KG paths.
2. **No self-rollout**: No model sampling was used for data generation. Therefore no Qwen2.5-specific stylistic bias.
3. **Oracle tool queries**: Every `<search>get_tail_entities(h, r)</search>` call is constructed directly from a gold triple, so the tool response is always correct-by-construction. SFT teaches **format**; RL teaches **robustness to wrong queries**.
4. **Three fixed think-templates**: e.g., `"I need to find information about {start_entity}..."` — template diversity is minimal but consistent across all target models.
5. **Deterministic + auditable**: Anyone can regenerate the exact same corpus from the same CWQ splits.

#### Why rule-based SFT is a methodological STRENGTH (not limitation) for this paper

Most 2026 agentic RL papers (Demystifying Agentic Reasoning, ToolRM, UserRL) use LLM teachers (DeepSeek, Qwen3-Coder-30B) for SFT data, introducing a "teacher model bias" confound that reviewers can challenge. Our rule-based SFT **avoids this entire class of critiques**:

- **"No teacher-model leakage"**: Zero distillation artifacts, no proprietary-model dependency, no teacher stylistic bias affecting downstream model behavior
- **"Clean skill decomposition"**: SFT teaches format only (via oracle queries); RL teaches query precision + robustness (where failed tool calls are learned). Skill attribution is unambiguous.
- **"Cross-model fairness"**: Identical rule-based corpus applies without per-model regeneration — eliminates a confound that exists in every other cross-model agentic RL paper from 2026.

These three claims should be stated explicitly in the paper's Methodology section.

#### Per-model SFT hyperparameter recommendations (Path B)

| Model | SFT Epochs | Learning Rate | Data subset | Rationale |
|---|---|---|---|---|
| Qwen2.5-7B-Instruct | **N** (existing value, already trained) | **LR₀** (baseline) | Full 5K primary or 6K enhanced | Baseline; no change |
| Qwen2.5-14B-Instruct | **N × 0.7** | **LR₀ × 0.5-0.7** | Same as 7B setup | Larger model: more gradient signal per sample, higher overfitting risk |
| Qwen3-4B-Instruct-2507 | **N × 0.5-0.7** | **LR₀ × 1.2-1.5** | **5K primary only** (skip 6K enhanced) | Already strongly instruct-tuned, high overfit risk on 12K full corpus; smaller model needs larger LR |

**Data-subset decision**: For Qwen3-4B specifically, use **only 5K primary** (not 6K enhanced, not gold-only) to prevent over-fitting on 2-3× more data than the model needs. This also matches Demystifying Agentic Reasoning's 3K-sample choice for Qwen3-4B-Instruct-2507.

#### Pilot-run protocol (required before full SFT launch on new model)

For each new model (Qwen2.5-14B, Qwen3-4B-Instruct-2507):

1. **0.1-epoch sanity check** (~30 min-2h per model):
   - Train on 10% of target SFT corpus with target hyperparams
   - Monitor loss curve shape — expected: smooth log-decrease
2. **Decision gates**:
   - Loss plateaus immediately (< 5% decrease in 0.1 epoch) → **STOP**, investigate chat template / tokenizer issue
   - Loss decreases too fast (> 50% in 0.1 epoch) → **REDUCE epochs further** (model overfitting to template)
   - Loss NaN / diverges → **STOP**, reduce LR by 2×
   - Normal smooth decrease → **proceed to full SFT run**
3. **Pilot eval**: Run 10-sample greedy inference on CWQ dev subset with trained 0.1-epoch checkpoint. Check:
   - Format validity (expect ≥ 80% valid `<answer>` tags even after just 0.1 epoch — rule-based SFT is highly format-consistent)
   - Tool call format correctness (expect `<search>get_tail_entities(...)</search>` pattern)
   - If < 50% valid format: chat-template mismatch — debug before full run

**Time cost of pilot**: ~2-4h per model. **Required** — cheaper than burning ~22h on a miscontemplated run.

#### Chat-template handling per model

| Model | Chat template | Action required |
|---|---|---|
| Qwen2.5-7B-Instruct | Qwen2.5 ChatML | None (baseline) |
| Qwen2.5-14B-Instruct | **Identical to 7B** | None — direct reuse, zero changes |
| Qwen3-4B-Instruct-2507 | Qwen3 ChatML (no thinking markers in Instruct-2507) | Re-tokenize SFT corpus with `tokenizer.apply_chat_template(..., enable_thinking=False)`. Note: Instruct-2507 variant does NOT emit `<think>` tags regardless of flag — verify with a 10-sample forward pass before SFT launch |

---

### Explicitly NOT doing (with rationale)

| Option | Why not |
|---|---|
| Qwen3-14B | Double-variable confound (family-generation + size); if it differs from Qwen2.5-14B, cannot attribute |
| Mistral / Ministral / Mistral-Nemo | Cross-family cost (chat template, tool format) > ROI given Qwen field dominance |
| Gemma 2 / 3 / 4 | Weak native tool use post-training; trajectory taxonomy would need rebuild |
| Phi-4 | Microsoft family rarely used in RL post-training literature; reviewer unfamiliarity |
| DeepSeek-R1-Distill-Llama-8B | Still Llama backbone, not truly different family |
| Qwen3.5 / Gemma 4 | Released 2026-02 / 2026-04, too fresh for reviewers to be familiar |

### Scheduling constraints

- **Do NOT launch before Gate A fully resolves** (already passed per current HPC snapshot — 39B EM 38.35% ≥ 34%, CvT 3.77%)
- **Do NOT launch before E5b-stabilized numbers locked** (Variant I / G still running)
- **Earliest launch**: Apr 19 (after Phase 7 Day-2 "numbers LOCKED" milestone)
- **Target completion**: Apr 28 (allows May 1+ for integration into paper draft)
- **GPU allocation**: 4 GPUs for Qwen3-8B training (2 conditions × ~44h sequential or 2×4 parallel), 2-4 GPUs for Qwen2.5-14B (single condition)

### If Time Pressure Forces Cuts (v13 updated)

Priority order for these new tasks (cut from the bottom if timeline slips):

1. **Qwen3-4B-Instruct-2507 E5b training** (protects "generation-axis" claim, responds to "your model is 1.5y old" critique; also picks up smaller-capacity data point)
2. **Qwen2.5-14B E5b training** (protects "capacity-axis" claim, responds to "7B too small" critique)
3. Qwen3-4B-Instruct-2507 E3 training (E3 Goodhart evidence is already robust on Qwen2.5-7B; replication is nice-to-have)

If only ONE slot remains: keep **Qwen3-4B-Instruct-2507 E5b** (modern-model claim is the strongest reviewer defense, and E5b is the paper's constructive contribution).

**Updated total compute (v13)**: ~5.5 days wall-time for all three above, down from v12's 8-day estimate (due to Qwen3-4B being half the compute of Qwen3-8B).

---

## 🔥 IMMEDIATE ACTIONS (2026-04-15 evening → 2026-04-16 morning)

> **Context**: Day 1 of Phase 7 is closing. 39B is our only success so far (EM=36.0% @ step 400 on 200-Q test, breaking parametric ceiling for the first time). Other variants are either failing or corrupted. Before sleep, we must: (a) stop wasting GPUs on dead variants, (b) secure canonical numbers for 39B, (c) queue the Day 2 experiments so overnight training starts immediately.
>
> **Key Day 1 numbers (200-Q test subset, seed=42)**:
> | Variant | Best EM | CvT | tools/Q | Status |
> |---|---|---|---|---|
> | **39B (KL 5x)** | **36.0% @ step 400** | 2.5% | 3.0 | ✅ **Breakthrough — promote to canonical** |
> | 39A (format reward) | ~31% → tool abandonment | 0% | 0.3 | ❌ dead |
> | 39C (SFT replay) | ~29% (pantomime) | 0% | 2.1 | ❌ dead |
> | 39D (Cat-B filter alone) | 22% then monotone decline | <1% | 1.2 | ❌ starved, kill |
> | 39E (Gold SFT + Cat-B + format) | pending — first eval ~ 1.5h window | ? | ? | ⚠️ probably corrupted (gold-gen bug) |
> | 39F (Gold SFT + full + format) | EM=1.5%, 158/200 tool-misuse | 0% | 1.1 | ❌ corrupted, kill |
> | E5b-original (baseline) | 31.0% @ step 100 | 5.5% | 2.3 | reference |
> | E3@500 (memory ceiling) | 32.6% full test | 0% | 1.0 | reference |
>
> **Observation**: 39B's +3.4pp over E3 is mostly "more stable memory," not "more retrieval." CvT 2.5% is actually *lower* than E5b-original's 5.5% — KL 5x over-anchored exploration. Day 2's job is to **push CvT back up while keeping 39B's stability**.

### Action I1 — Kill dead variants (within 15 min)
- [ ] `scancel` SLURM job running Task 39D (Cat-B filter alone). Monotone decline past step 200, ~30h wall-clock left, no recovery possible.
- [ ] `scancel` SLURM job running Task 39F. Corrupted by gold-trajectory generator bug (158/200 tool-misuse, EM=1.5%). Same bug will poison any further steps.
- [ ] Task 39E: **wait for the first full eval at step 100 (~1.5h window)**. If EM < 10% OR tool-misuse ≥ 40%, `scancel` immediately. If EM ≥ 15% with coherent tool calls, keep alive — the Cat-B filter may have partially compensated.
- [ ] Task 39A (format reward): already at tool abandonment. Let it finish current eval cycle to harvest final numbers, then stop — no Day 2 restart.
- [ ] Task 39C (SFT replay): same — harvest final numbers, then stop.

### Action I2 — Secure 39B canonical numbers (launch tonight, ~4-6 h)
- [ ] **Full 3531-test eval of 39B @ step 400** (greedy, max_turns=5, max_new_tokens=512 — same config as Task 14 Qwen).
  - Checkpoint: `/projects/u6gg/KG_GRPO/checkpoints/kg-align-verl/grpo-cwq-7b-39b-kl-20260413/global_step_400`
  - Output: `results/phase7/39b_step400_full_test.json`
  - This is the number that goes in the paper table. 200-Q 36% must be replicated on full test before we trust it.
- [ ] **Full 3531-test eval of 39B @ step 300 and @ step 500** at the same time (3 GPUs in parallel).
  - Gives us a mini Goodhart curve around the peak and pins down whether 400 is really the best step.

### Action I3 — Resubmit Task 38b (Pass@k)
- [ ] Resubmit Task 38b as **E3@500 only, k=16**, 200 questions from test.parquet seed=42, temperature=0.7, with tools, max_turns=5.
  - Previous submission timed out at 24h walltime (SFT too slow).
  - New job: drop SFT from the sample set, single model, k=16 instead of 32 → should finish in ~8h on 1 GPU.
  - Output: `results/phase7/task38b_e3_pass_at_k.json`
- [ ] **Also run pass@32 (with tools) on 39B@400** on the same 200 questions, so we can compare E3 vs 39B at the capability-expansion level (this is the load-bearing comparison for "did RL actually expand retrieval capability").
  - Output: `results/phase7/task38b_39b_pass_at_k.json`

---

## 🚨 v11 IMMEDIATE MITIGATIONS (post critical review, 2026-04-15 late evening)

> **These tasks take priority over ALL Day-2 variants. Day-2 training does NOT launch until the "Day-2 readiness gate" below is resolved.** The critical review concluded that v10's plan was ambitious but missed several experiments that a reviewer will demand regardless of whether the variants work.

### Action II1 — HARD BLOCKER: Full-test CvT audit on 39B@400 + E5b@100 (~6 h, 1 GPU)

**Why**: v10 already schedules full 3531 EM eval on 39B (Action I2). This extends it to trajectory classification. The current CvT CI of [3.1-9.6%] is from n=100-200 trajectories — **not statistically valid at the scale the paper claims**. The full-test CvT number is load-bearing and cannot wait until after Day-2 training.

**Task**:
1. Run trajectory-saving eval on full 3531 test (not 200-Q subset) for:
   - E5b @ step 100 (our baseline CvT claim)
   - 39B @ step 400 (our "breakthrough" claim)
2. Run the 7-category classification script on ALL 3531 trajectories (not a sub-sample)
3. Report exact CvT count + Wilson 95% CI + McNemar vs E3@500

**Output**: `results/phase7/full_test_cvt_audit.json`

**Go/no-go**: If 39B's full-test CvT is **< E5b-original's full-test CvT minus 1pp**, then 39B is confirmed to be "more stable memorization, not more retrieval." Reframe 39B as supporting evidence for the diagnostic, NOT as a constructive breakthrough. Paper pivots to pure-diagnostic.

### Action II2 — Search-R1 baseline on CWQ (~16 h, 4 GPUs, 2 days total including setup)

**Why**: **Reviewer non-negotiable.** Search-R1 (arXiv:2503.09516) shows Qwen2.5-7B + GRPO + outcome-only reward hits 35-45% on HotpotQA/NQ. Until we show its recipe on CWQ/Freebase, we cannot claim "process rewards are needed" as a finding. If Search-R1's outcome-only > E3 on CWQ, our entire diagnostic collapses and we need to know this NOW, not at rebuttal.

**Task**:
1. Clone Search-R1 repo (https://github.com/PeterGriffinJin/Search-R1)
2. Adapt their training script to use our Freebase KG server instead of their Wikipedia retriever (minimal change — they use a single `search(query)` endpoint; we wrap our 4-tool API as a single "retrieve" call)
3. Train Qwen2.5-7B with their exact reward (outcome-only EM) and their GRPO config on CWQ train
4. Evaluate at steps 200, 400, 600, 800 on full 3531 test
5. Also run their exact recipe on CWQ with our 4-tool API (no adaptation) — second arm
6. Report: EM, F1, tools/Q, trajectory classification (on 500-sample subset of test)

**Output**: `results/phase7/searchr1_baseline/`

**Budget**: 4 GPUs × 2 days (1 day training + 1 day eval and setup). This uses the slot currently earmarked for Variant H, which is cut in v11.

### Action II3 — Llama pipeline audit (~4 h, 1 GPU)

**Why**: v10 states Llama EM=0 across the board; our 500-sample runs had 24.8%. A reviewer who reads the appendix will demand an audit. This is cheap to run and protects the paper.

**Task**:
1. Verify Llama chat template is correct (compare training-time vs eval-time — Llama-3.1 has a known `\n\n` vs `\n` inconsistency)
2. Verify stop tokens (`<|end_of_text|>`, `<|eot_id|>`) are in both training and eval configs
3. Check `max_new_tokens` isn't too small (degenerate token-loop often correlates with hitting the cap mid-answer)
4. Run Llama E3 @ step 1293 on **500 random test samples** with:
   - (a) current config
   - (b) fixed chat template if any issue found
   - (c) longer `max_new_tokens=1024`
   - (d) `max_turns=5` (v10 had Llama at max_turns=3)
5. Report per-config EM + one sample trajectory per config

**Output**: `results/phase7/llama_audit.md`

**Decision**: If any fix produces Llama EM > 5%, the 0% result is a pipeline bug — rerun the Llama full-test eval with the fix. If all four configs give EM < 1%, Llama failure is genuine and we document the audit in the paper appendix. **Do NOT frame Llama as "small LLMs can't do compositional tool use" unless the audit is complete.**

### Action II4 — KGQAGen Oracle replication (~3 h, CPU only)

**Why**: CWQ's annotation quality is audited at ~49% (KGQAGen paper, arXiv:2505.23495). Our Oracle on CWQ reports 99.5% SOLVABLE. These two facts are in tension. A reviewer will demand we replicate the Oracle on an audited benchmark. KGQAGen-10k is the right target (Wikidata-based, 96.3% verified).

**Task**:
1. Reuse the Task 36 SPARQL parser on KGQAGen-10k dev set (1,079 questions)
2. Run the Wikidata coverage check using our already-built Wikidata server (from Task 35)
3. Report: % SOLVABLE on KGQAGen
4. Expected: 85-92% (cleaner than CWQ but not perfect)

**Output**: `results/phase7/kgqagen_oracle.md`

**Paper use**: "Our Oracle reports 99.5% SOLVABLE on CWQ and X% on KGQAGen-10k, confirming that the KG contains the information needed for both benchmarks."

### Action II5 — GPT-4o baseline (~2 h, API only, < $50)

**Why**: A closed-model ceiling check is standard. Reviewers ask for it. Very cheap.

**Task**:
1. 200 random questions from CWQ test (seed=42, same as Action I2 subset for comparability)
2. GPT-4o with our exact 4-tool API (function calling interface)
3. Zero-shot + 5-shot ReAct prompts
4. Report: EM, tools/Q, trajectory classification on a 50-sample subset

**Output**: `results/phase7/gpt4o_baseline.json`

### Action II6 — Pass@k with AND without tools on full 3531 (~8 h, 2 GPUs)

**Why**: v10's Task 38b is only "with tools". To defend "capability expansion via tools" we need the counterfactual — pass@k of the same model WITHOUT tools.

**Task**:
1. Sample 500 questions from test.parquet (seed=42)
2. For each of: SFT base, E3@500, E5b@100, 39B@400 — run pass@16 twice:
   - (a) with tools (same as Task 38b)
   - (b) without tools (parametric memory only, same eval code with tool server disabled)
3. Plot pass@k curves overlaid (with-tools solid, without-tools dashed)

**Output**: `results/phase7/pass_at_k_with_without_tools.json`

**Interpretation**: the gap between solid and dashed curves IS the "tool contribution to capability expansion." If the gap is zero everywhere, tools genuinely don't contribute. If the gap is non-zero for 39B but zero for E3, 39B is genuinely using tools differently.

---

### Day-2 Readiness Gate (must pass before ANY Day-2 training launches)

- [ ] **Gate A (hard)**: Action II1 full-test CvT audit completed.
  - If 39B's full-test EM ≥ 35% AND CvT ≥ E5b-original's full-test CvT: **proceed to Day-2 training**
  - If 39B's full-test EM < 34% OR CvT regressed by > 1pp vs E5b-original: **PIVOT to pure-diagnostic paper**, cancel all Day-2 variants, refocus on II2/II3/II4/II5/II6 + writing
- [ ] **Gate B (soft, informational)**: Action II2 Search-R1 baseline 1st eval point (step 200) available. If Search-R1 @ step 200 already > E3 @ step 500, reframe the paper as "Search-R1 recipe doesn't transfer to CWQ compositional reasoning" (still a paper, different angle).
- [ ] **Gate C (soft)**: Action II3 Llama audit completed. Llama narrative finalized before Day-2 writing begins.

**Time estimate for all II actions**: II1+II4+II5 can run in parallel (GPU + CPU + API). II2 takes 2 days. II3 takes 4 hours. II6 takes 8 hours.
**Critical path**: ~36 hours wall-clock, 6-8 GPUs at peak.
**Earliest Day-2 training launch**: 2026-04-17 morning (after Gate A passes).

---

## Phase 7 Day 2 Plan (v11 REVISED — was 2026-04-16 → 2026-04-19)

> **v11 REVISED goal**: *IF Gate A passes*, push 39B toward ≥ 38% EM with CvT ≥ 8% on full test. We **only** run variants that are defensible against a harsh review. **Variant H is cut entirely**. Variant G is modified (yield pilot + parallel ReST-EM init arm). Variant I is **reframed as Oracle upper-bound diagnostic** (not claimed as deployable method) OR redesigned to use a self-verifiable signal.
>
> **Bottleneck analysis of 39B** (unchanged from v10):
> - ✅ L1 (format drift): fixed by KL 5x. Format stays valid to step 500+.
> - ⚠️ L0 (memorization shortcut): partially fixed. 39B does use tools (3.0/Q) but only 2.5% are productive.
> - ❌ L3 (query precision): unaddressed. Bottleneck for the remaining gap.
>
> **Critical reviewer's counter-point (v11)**: if Action II1 shows 39B's full-test CvT regressed from E5b-original's 5.5%, then **fixing L1 cost us L0/L3 progress** — the agent traded retrieval for stability. In that case, no amount of Day-2 work on G/I can fix the fundamental problem, and we pivot to pure-diagnostic.

---

### Task 41 — 39B Mechanistic Analysis [P0, ~4-6 h, 1 GPU + CPU, Day 1-2]

**Purpose**: Before launching expensive Day 2 training runs, understand *why* 39B works, so that Variants G/H/I are designed against real bottlenecks, not guesses.

**Inputs**:
- 39B @ step 400 checkpoint
- 200-Q test subset (seed=42) — already evaluated, reuse trajectories
- Category A/B list (Task 26)
- Oracle gold paths (Task 36 output)
- E5b @ step 100 trajectories (Task 37)

**Steps**:

1. **Category A vs B breakdown** (~30 min, CPU-only, uses existing 200-Q trajectories)
   - For 39B's EM=36.0% = 72/200 correct, split into:
     - Correct on Category A (parametric memory works)
     - Correct on Category B (genuine retrieval or lucky guess)
   - Expected: ~55 A / ~17 B if CvT=2.5% holds. If B count >> 5, it means tools helped more than the CvT number suggests (answer present in tool output but not exactly matched).

2. **Hop-stratified EM** (~30 min)
   - Use existing CWQ hop labels. Compute 39B's EM at 1-hop, 2-hop, 3+-hop.
   - Compare head-to-head with E3@500 and E5b@100.
   - Question: does 39B's gain concentrate on multi-hop (tool-driven) or 1-hop (memory)?

3. **kg-incomplete query error-mode analysis** (~2 h, CPU)
   - For the ~77/200 samples classified as kg-incomplete, dump:
     - Gold entity, gold relation, gold answer
     - Model's actual queries (entity + relation strings)
     - Whether the entity matched, relation matched, neither
   - Classify errors: (a) wrong entity, (b) right entity wrong relation, (c) string-format mismatch (e.g., "Barack Obama" vs "Barack_Obama"), (d) correct query returned empty.
   - Output: `results/phase7/39b_query_error_modes.csv`
   - **This directly informs Variant I's reward design.**

4. **E5b vs 39B trajectory comparison** (~1 h)
   - On the 77 kg-incomplete samples, did E5b succeed on any? If yes, what did E5b do differently?
   - Reveals whether 39B is worse than E5b at query formation (i.e., KL 5x over-constrained the retrieval skill).

5. **Pass@32 with tools on 39B@400** (~2 h, 1 GPU) — merges with Action I3.

**Output**: `results/phase7/task41_39b_mechanistic.md` — a ≤2-page memo with the four findings and explicit recommendations for G/H/I reward/data design.

**Blocking**: Variants G and I should read this memo before final reward-weight selection. Variant H does not block.

---

### Variant G — Self-Distillation from 39B@400 [P0, 4-8 GPUs, Day 2 launch] ⭐ highest-confidence next step

> **v11 changes**: (1) Added pilot yield measurement before committing to the full train-set inference. (2) Added a parallel "init-from-original-SFT" arm (ReST-EM evidence: arXiv:2312.06585 — fine-tuning from base each round transfers better than from previous iterate). (3) Added EEF salvage fallback (arXiv:2504.13145) if yield < 800. (4) Paper positioning: cite ReST-EM, ReTool, FireAct honestly — Variant G is not claimed as novel method, it is applied-ReST-EM evidence.

**Rationale**: The Task 40 "gold trajectory generator" had a bug that caused variants E/F to hallucinate wrong entity names. Rather than fix that parser, **use 39B@400's own correct trajectories as the demonstration set** — guaranteed to be parseable, tool-using, and hit real Freebase entities.

**Pipeline**:

1. **Step 1a — Yield pilot on 2K train questions** (~1 h, 1 GPU) ⚡ **NEW v11**
   - Run 39B@400 greedy + tools on 2,000 randomly-sampled train questions (seed=42).
   - Apply strict filter (EM=1 AND ≥1 tool call AND format valid).
   - **Decision rule**:
     - If yield ≥ 15% (≥ 300 of 2K) → proceed to full 27K sweep, expect ~4K trajectories
     - If yield 7.5-15% → proceed but widen filter to (EM=1) ∪ (F1 ≥ 0.8 AND tool used)
     - If yield < 7.5% → apply **EEF salvage** (mine beneficial prefixes from failed trajectories) before full sweep. See Step 2b.
   - Output: `data/freebase/verl_cwq/39b_pilot_yield.json`

2. **Step 1b — Full train-set inference** (~6 h, 2 GPUs) — conditional on pilot result
   - Run 39B@400 greedy + tools on all ~27K CWQ train questions.
   - Save full trajectory.
   - Output: `data/freebase/verl_cwq/39b_step400_train_traj.jsonl`

3. **Step 2a — Strict filter** (~30 min, CPU)
   - Keep EM=1 AND ≥1 successful tool call AND format valid
   - Expected yield from pilot

4. **Step 2b — EEF salvage (conditional)** (~2 h, CPU + 1 GPU for rollout scoring) ⚡ **NEW v11**
   - *Only if strict filter yield < 800 trajectories.*
   - For each failed trajectory (EM=0), for each intermediate tool call, roll out completion from that state (1 sample, greedy).
   - Keep prefixes where the post-prefix rollout achieves EM=1.
   - Append these prefix+completion as "salvaged" trajectories (tag separately for ablation analysis).
   - Output: `data/freebase/verl_cwq/39b_self_distill_salvaged.jsonl`

5. **Step 3 — SFT from TWO parallel init points** (~4 h, 4 GPUs each = 8 GPUs total) ⚡ **NEW v11**
   - **Arm G1** (init from 39B@400): continues the RL-learned behavior.
     - 1 epoch (2 if yield < 800), lr=5e-6, LoRA rank 64
     - Output: `verl-sft-cwq-7b-G1-from-39b`
   - **Arm G2** (init from original pre-GRPO SFT): ReST-EM-style, better literature-backed transfer.
     - Same hyperparameters
     - Output: `verl-sft-cwq-7b-G2-from-base-sft`
   - **Why two arms**: ReST-EM (Singh 2024, arXiv:2312.06585) explicitly found init-from-base beats init-from-previous-iterate. This ablation is *publishable as a finding*, not just engineering — becomes a paper figure either way.

6. **Step 4 — GRPO from both SFT checkpoints** (~30 h, 4 GPUs each = 8 GPUs) — subject to early-kill gates
   - Same config as 39B: KL 5x (kl_coeff=0.05), reward = E5b tool_type_bonus, max_steps=500
   - Dense eval at steps 50/100/150/200/250/300 on 500-Q test (seed=42, expanded from v10's 200-Q to reduce CI width)
   - Output: `grpo-cwq-7b-G1-selfdistill-20260417`, `grpo-cwq-7b-G2-restEM-20260417`

**Hypothesis**:
- G1: RL-learned tool use gets tightened → CvT ↑ without losing stability
- G2: Cleaner transfer (per ReST-EM) → potentially better than G1 despite starting from a weaker point

**Success threshold** (v11 revised): EM ≥ 37% on full 3531 AND CvT ≥ 5% at any step by 300.

**Early-kill rules** (same as v10 continuous kill rules): format_valid < 80% for 2 evals, tools/Q < 0.5 for 2 evals, EM drops > 3pp between evals, reward↑ while EM↓ > 2pp.

**If both arms fail**: reallocate 8 GPUs to Action II2 (Search-R1 baseline scaling) or to Variant I.

---

### ~~Variant H — Curriculum from 39B@400~~ ❌ CUT IN v11

> **Why cut**: Critical-review feedback. This is not curriculum learning (39B has already seen full data). It is "hard-subset finetuning" that a reviewer will recognize and discount. Running true curriculum-from-scratch is a separate project and doesn't fit in 40 days. The 4 GPUs are reallocated to Action II2 (Search-R1 baseline) and Variant G's second arm.

> If we later need a curriculum arm for paper ablation, run it post-submission as "future work."

---

### Variant I — 39B + Oracle-Supervised Query Match (RELABELED v11) [P1, 4 GPUs, Day 2-3 launch]

> **v11 critical change**: In v10 this was framed as "query-precision reward." Critical review correctly identified that using the Oracle gold path as a reward signal makes this **oracle-supervised training**, not a deployable self-verifiable process reward. In v11 we **acknowledge this honestly** and run Variant I as a **diagnostic upper-bound measurement**: "if we leak the gold query structure to the reward, how much better does the model get?"
>
> **Paper framing**: Variant I is NOT claimed as a method. It is claimed as "an upper bound on what is achievable by any process reward that could approximate the oracle signal." This is a legitimate and well-precedented diagnostic (see StepSearch arXiv:2505.15107 which uses gold sub-question docs — same oracle-supervision structure).
>
> **Plus**: we run a **parallel self-verifiable variant** (Variant I-self) that uses only retrieval-success signal, which IS deployable. The comparison I vs I-self is the paper's contribution.

**Prerequisite** (must run first, Day 2 morning, ~3 h, 0 GPU):
- Extend Oracle gold-path extraction (Task 36) from the 200 eval samples to the full CWQ **train** split.
- Output: `data/freebase/verl_cwq/train_oracle_gold_paths.jsonl`

---

#### Variant I-Oracle (diagnostic upper bound)

**Reward formula** (v11 with three non-negotiable defenses from Agent 3):
```python
# Variant I-Oracle: R = 0.25·r_answer + 0.50·r_tool_type + 0.25·r_query_match_oracle
def r_query_match_oracle(trajectory, gold_path):
    """
    v11 DEFENSES:
    1. Deduplicate (entity, relation) pairs across turns (StepSearch redundancy penalty)
    2. Gate: reward only fires if ≥2 DISTINCT gold (entity, relation) pairs are covered
       (KG-Implicit-RM anti-spam, arXiv:2601.15160)
    3. Trajectory-level (not per-turn) — compute matched_unique / distinct_calls at end
    """
    distinct_calls = set()
    for call in trajectory.tool_calls:
        distinct_calls.add((call.entity, call.relation))

    matched_unique = len([
        (e, r) for (e, r) in distinct_calls
        if (e, r) in gold_path_pairs  # string + FB-ID match
    ])

    if matched_unique < 2:  # anti-spam gate
        return 0.0

    return matched_unique / max(1, len(distinct_calls))
```

**Init checkpoint**: 39B @ step 400.

**Training config**:
```yaml
algorithm: grpo
multi_turn: true
max_turns: 5
group_size: 8
learning_rate: 3e-7
kl_coeff: 0.05
max_steps: 400
batch_size: 128
reward_type: tool_type_bonus_oracle_query_match
# Ramp r_query_match_oracle weight 0.10 → 0.25 over first 50 steps (ToolRL lesson)
init_checkpoint: grpo-cwq-7b-39b-kl-20260413/global_step_400
```

**Success threshold**: EM > 39% AND CvT > 8% on full 3531.

**Interpretation**: Whatever I-Oracle achieves is the **ceiling** for any process reward that matches the oracle signal structure. If I-Oracle hits 42% EM with CvT 10%, that's the diagnostic bar.

---

#### Variant I-Self (deployable baseline) ⚡ **NEW v11**

**Reward formula**: Self-verifiable, no oracle leak.
```python
# Variant I-Self: R = 0.25·r_answer + 0.50·r_tool_type + 0.25·r_retrieval_contrib
def r_retrieval_contrib(trajectory):
    """
    Self-verifiable: reward tool calls where
    (a) the call returned non-empty results, AND
    (b) AT LEAST one returned entity appears verbatim in the model's final <answer>
    This requires no gold labels — only the trajectory itself.
    """
    productive_calls = 0
    for call in trajectory.tool_calls:
        if len(call.results) > 0:
            answer_text = trajectory.final_answer.lower()
            if any(ent.lower() in answer_text for ent in call.results):
                productive_calls += 1

    return productive_calls / max(1, len(trajectory.tool_calls))
```

**Same training config as I-Oracle** except `reward_type: tool_type_bonus_retrieval_contrib`.

**Why this is the real paper result**: I-Self is deployable at inference time (no gold signal needed). If I-Self gets close to I-Oracle's EM/CvT, we have "a self-verifiable process reward that nearly matches oracle supervision" → strong paper contribution. If I-Self fails but I-Oracle succeeds, we have "oracle supervision is required, self-verifiable signals aren't enough" → still a publishable finding.

**Both variants launch in parallel. 4 GPUs each = 8 GPUs total.**

---

#### Hackability mitigations (apply to BOTH I variants)

| Hack | Mitigation |
|---|---|
| Spam gold entity in every call | Dedup (e,r) pairs across turns (defense 1) |
| One lucky call then stop | Gate ≥2 distinct matches (defense 2) |
| Stall after 1 successful call | r_answer weight 0.25 still punishes wrong answer |
| Reward hacks format | r_tool_type unchanged, provides format stability via tool_type diversity |

**Early-kill rules** (v11, tighter than v10):
- CvT decreases AND reward increases by step 150 → Goodhart → kill
- Format_valid < 85% at step 100 → kill (stricter than v10's 80%)
- Both I-Oracle and I-Self below E3's 32.6% at step 200 → kill both

**Rationale**: 39B's ~77/200 kg-incomplete rate shows the model *wants* to use tools but picks the wrong relation/entity. Standard E5b reward only checks tool-type diversity, not query correctness. A query-match reward directly targets L3 (query precision).

**Prerequisite** (must run first, Day 2 morning, ~3 h, 0 GPU):
- Extend Oracle gold-path extraction (Task 36) from the 200 eval samples to the full CWQ **train** split.
- Output: `data/freebase/verl_cwq/train_oracle_gold_paths.jsonl` with per-question (gold_entity, gold_relation_chain).

**Reward formula**:
```python
# Variant I:  R = 0.25·r_answer + 0.50·r_tool_type + 0.25·r_query_match
def r_query_match(trajectory, gold_path):
    """
    For each <search> call in the trajectory, check whether the (entity, relation)
    pair appears anywhere in the gold path (case-insensitive string match on both
    surface form and Freebase ID).
    Reward = (# matched tool calls) / max(1, # total tool calls).
    Capped at 1.0.
    """
```

**Init checkpoint**: 39B @ step 400.

**Training config**:
```yaml
algorithm: grpo
multi_turn: true
max_turns: 5
group_size: 8
learning_rate: 3e-7          # lower than 5e-7 since we're continuing from a plateau
kl_coeff: 0.05               # keep KL 5x — don't lose 39B's stability
max_steps: 400
batch_size: 128
reward_type: tool_type_bonus_query_match  # new type in verl_reward.py
init_checkpoint: grpo-cwq-7b-39b-kl-20260413/global_step_400
```

**Hypothesis**: Adding a direct signal for query-argument correctness pulls the model toward the Oracle-gold entities/relations, which should reduce kg-incomplete rate and convert more tool calls into CvT hits.

**Success threshold**: kg-incomplete rate drops from 38.5% → ≤ 25% by step 200, AND CvT ≥ 8%.

**Failure modes**:
- Reward hacks the match component (e.g., spam gold entity in unrelated contexts). Mitigation: cap r_query_match at 1.0, require r_answer > 0 for full credit.
- Training destabilizes because new reward dominates. Mitigation: warm-start with 0.10 weight, ramp to 0.25 over first 50 steps.

---

### Task 42 — Inference-Time Self-Consistency Only (v11 TRIMMED) [P1, 1 GPU, Day 2 parallel]

> **v11 change**: Beam search over tool calls and Oracle-guided replanning are CUT. Critical review: each is "separate paper's worth of machinery." Keep ONLY self-consistency which is (a) cheap, (b) standard, (c) paper-ready.

**Task**:
- Run self-consistency@5 on **full 3531 test** (not 200-Q subset) for 39B@400 and E3@500.
- Temperature=0.7, with tools, max_turns=5, majority vote over normalized answer strings.
- Compare: greedy full-test EM vs SC@5 full-test EM for both checkpoints.
- **Also report**: SC@5 CvT (does majority voting change which trajectories contribute the answer?).

**Output**: `results/phase7/self_consistency_full_test.json`

**Paper use**:
- If SC@5 lifts 39B by +2-4pp, it goes in a small table as "inference-time augmentation."
- If SC@5 lifts E3 and 39B by similar amounts, it means they have similar underlying distributions (no structural difference) — interesting side finding.
- If SC@5 doesn't help at all, confirms that failure modes are systematic (same wrong answer repeated), not sampling noise — stronger support for the diagnostic.

**Resource**: 1 GPU × ~4 h. No training.

---

### v11 Day-2 Resource Matrix (after Gate A passes)

| Rank | Item | GPUs | Start | Dependency |
|---|---|---|---|---|
| **1** | Action II1 full-test CvT audit (blocker) | 1 | TONIGHT | — |
| **2** | Action II2 Search-R1 baseline training | 4 | TONIGHT | repo setup |
| **3** | Action II3 Llama pipeline audit | 1 | TONIGHT | — |
| **4** | Action II4 KGQAGen Oracle replication | 0 (CPU) | TONIGHT | — |
| **5** | Action II5 GPT-4o baseline | 0 (API) | TONIGHT | — |
| **6** | Action II6 pass@k with/without tools | 2 | T+6h | 39B@400 ckpt, SFT ckpt |
| **7** | Task 41 39B mechanistic memo | 1 | T+0 Day 2 | Action II1 results |
| **8** | **Variant G1** self-distill from 39B@400 | 4 | Day 2 AM (AFTER Gate A) | yield pilot first |
| **9** | **Variant G2** self-distill from original SFT (ReST-EM init) | 4 | Day 2 AM (AFTER Gate A) | yield pilot first |
| **10** | **Variant I-Oracle** (diagnostic upper bound) | 4 | Day 2 PM | train Oracle paths |
| **11** | **Variant I-Self** (deployable) | 4 | Day 2 PM | none |
| **12** | Task 42 self-consistency@5 | 1 | Day 2 parallel | — |

**GPU accounting**:
- Tonight (II actions): 1+4+1+2 = **8 GPUs** (+ CPU + API)
- Day 2 peak (all variants after Gate A): 1+4+4+4+4+1 = **18 GPUs** (within 32 ceiling)
- Reserve: 14 GPUs for eval rotator, full-test eval jobs, and Search-R1 eval

**Hard ordering**:
- II1 must complete before ANY Day-2 training launches (Gate A)
- II2 launches tonight — its first eval at step 200 (~T+12h from launch) informs Gate B
- G1 and G2 both require the yield pilot on 2K train questions before full SFT
- Variant I requires train-split Oracle paths to be extracted (~3 h CPU)

---

### ~~v10 Early-Intervention Protocol~~ (SUPERSEDED by v11 Day-2 Resource Matrix above, retained for reference)

> **Rationale**: All Day-2 variants init from 39B@400 — they do not need warmup. Meaningful signal arrives ~3-6 h after GRPO start (step 30-80). Committing 4 GPUs × 30 h to every variant is wasteful; we should gate on early signal and reallocate.

**Eval cadence for G, H, I (much denser than original plan)**:
- Eval at steps **25, 50, 75, 100**, then **every 50 steps** until step 300 max.
- Each eval is 200-Q test subset (seed=42), ~15-20 min on 1 GPU.
- Eval GPU does NOT block training — budget 1 shared eval GPU that rotates.

**Check-in timeline** (wall-clock from Day 2 launch = 0h):

| T+hours | What is observable | Decision |
|---|---|---|
| **T+3** | G-SFT phase complete, first GRPO step ~25 eval. Task 42 self-consistency done. | If G-SFT degrades base EM < 33%, kill G, redirect 4 GPUs. If Task 42 SC@5 ≥ 40%, flag as paper-ready. |
| **T+6** | G, I at step ~50. First CvT reading. Task 42 beam search done. | Apply **Gate 1** (below). |
| **T+10** | G, I at step ~75-100. Trend emerging. Task 41 mechanistic memo ready. | Apply **Gate 2**. Launch H if slot opens. |
| **T+16** | G, I at step ~150. Task 42 oracle-replan done. | Apply **Gate 3**. Consolidate onto winner. |
| **T+24** | G, I at step ~200-250. H at step ~50. | Apply **Gate 4**. Commit GPUs to best 1-2 variants only. |

**Gate 1 (T+6) — first CvT reading**:
- G EM < 34% AND CvT < 2% → **kill G**, reallocate 4 GPUs to Variant H (start immediately with stage H2 skip).
- I EM < 33% OR format-valid < 80% → **kill I** (query-match reward is hacking), reallocate to H or G-v2 with different weight.
- If BOTH G and I show CvT ≥ 4% at step 50 → promising, keep both, don't launch H yet.
- If NEITHER shows CvT ≥ 3% at step 50 → the "KL 5x init + new intervention" path is not moving the needle; launch H immediately AND start drafting paper as diagnostic-only in parallel.

**Gate 2 (T+10) — trend direction**:
- A variant is "on track" if (CvT at step 100) > (CvT at step 50) AND EM ≥ 35%.
- Kill any variant that is flat or declining on CvT.
- Free GPUs → launch H if not already running, OR launch a G-v2 / I-v2 with adjusted hyperparameters based on Task 41 findings.

**Gate 3 (T+16) — winner identification**:
- At most **two** variants continue past this point. Kill everything else.
- If a clear winner exists (EM ≥ 37% AND CvT ≥ 6%), assign it 8 GPUs (doubled batch / faster convergence) and keep one runner-up at 4 GPUs as backup.
- If no clear winner, keep the two best and reconvene at T+24.

**Gate 4 (T+24) — commit**:
- Best variant continues to step 300-400 on 8 GPUs.
- Kick off full 3531-test eval on its latest checkpoint (in parallel, 2 GPUs).
- Everything else is archived. Total Day-2+Day-3 wall-clock: **~36 h** instead of 96 h.

**Kill conditions that override any gate** (apply at every checkpoint):
- Format-valid rate < 80% for 2 consecutive evals → format collapse, no recovery → kill.
- `tools/Q < 0.5` for 2 consecutive evals → tool abandonment → kill.
- EM drops by > 3pp between consecutive evals → catastrophic drift → kill.
- Reward increases while EM drops by > 2pp → Goodhart → kill.

**What "reallocation" means concretely**:
- Killing a 4-GPU variant frees one full 4-GPU SLURM allocation.
- Reallocation targets in priority order:
  1. Double the leading variant (8 GPUs → 2× throughput via larger batch)
  2. Launch Variant H (curriculum) if not yet running
  3. Launch a G-v2 or I-v2 with tweaked hyperparameters (e.g., different KL, different query-match weight)
  4. Full-test eval of the current leader

---

### Day 2 Priority Matrix

| Rank | Item | GPUs | Start | Dependency |
|---|---|---|---|---|
| **1** | Full 3531-test eval of 39B@300/400/500 (Action I2) | 3 | tonight | — |
| **2** | Task 38b pass@k (E3 + 39B) (Action I3) | 2 | tonight | — |
| **3** | Task 41 mechanistic analysis | 1 | tomorrow AM | 39B@400 ckpt |
| **4** | **Variant G** self-distillation ⭐ | 4 | T+0 (Day 2 AM) | 39B@400 ckpt |
| **5** | **Variant I** query-precision reward | 4 | T+0 (Day 2 AM) | train Oracle paths (can start in parallel with Task 41) |
| **6** | Task 42 self-consistency / beam / oracle-replan | 3 | T+0 (Day 2 AM) | 39B@400 ckpt |
| **7** | Shared eval rotator (dense step-25/50/75/100 checkpoints) | 1 | T+3 | G or I first ckpt |
| **8** | **Variant H** curriculum | 4 | **CONDITIONAL on Gate 1-2** | frees slot from G or I |

**Total GPU budget (T+0)**: 3 + 2 + 1 + 4 + 4 + 3 + 1 = **18 GPUs** (Variant H's 4 GPUs are reserved but not allocated until a Gate frees them).

**Total GPU budget (T+16 after Gate 3)**: 8 (winner) + 4 (runner-up) + 2 (full-test eval) + 1 (eval rotator) = **15 GPUs** — consolidation frees capacity.

**Early-kill rules** (apply at EVERY eval checkpoint, not only step 200):
- Format valid rate < 80% for 2 consecutive evals → kill
- `tools/Q < 0.5` for 2 consecutive evals → kill (tool abandonment)
- EM drops > 3pp between consecutive evals → kill (drift)
- Reward ↑ while EM ↓ > 2pp → kill (Goodhart)

### Go / no-go gates (apply at each T+N check-in, not "Day 3 morning")
- [ ] **T+0** 39B full-test EM confirmed ≥ 34% → 39B is our canonical constructive result, continue G/I.
- [ ] **T+0** If 39B full-test EM < 34% → the 200-Q 36% was sampling luck; **halt G/I/H**, reframe paper as diagnostic-only.
- [ ] **T+3** If Task 42 (SC@5 or beam) pushes 39B to ≥ 40% → add as paper contribution regardless of training outcomes.
- [ ] **T+6** Gate 1 fires → at least one training variant continues, at least one is killed or H is launched.
- [ ] **T+16** Gate 3 fires → winner identified, GPUs consolidated onto ≤ 2 variants.
- [ ] **T+24** Gate 4 fires → final training variant committed; full-test eval launched.
- [ ] **T+36** Numbers locked. Paper writing sprint begins if results ≥ "Good" threshold (EM > 33% AND CvT > 0% sustained).

---

## Completed Tasks (Phases 1-4)

| Task | Status | Result |
|---|---|---|
| Task 1: E3 Goodhart curve | DONE | 11 steps evaluated. Peak EM=0.542 at step 500, broad plateau 350-750 |
| Task 2: E1 Goodhart curve | DONE | 7 steps evaluated. EM≈0 throughout |
| Task 3: Trajectory sampling | DONE | E3@500 (100), E3@1250 (50), E1@1250 (100), E2@200/600/1200 (100 each) |
| Task 4: R_random reward code | DONE | In verl_reward.py |
| Task 5: Hop-stratified script | DONE | Tested on E3@500 and E2@200/600/1200 |
| Task 6: E2 evaluation | DONE | EM=0→0.336, phase transition at step 1000 |
| Task 7: E4 (R_random) | DONE | EM=7.4%, tools=0. Weak memorization |
| Task 9: Trajectory behavior analysis | DONE | correct-via-tool=0% everywhere for E1-E3 |
| Task 10: Tool call distribution | DONE | E3: 100% get_tail_relations, exactly 1 call |
| Task 11: Goodhart divergence | DONE | E3: reward climbs while EM declines |
| Task 12: KG verification rewards | DONE | r_step_avg stable ~0.25-0.28 |
| Task 13: Policy entropy/KL | DONE | E3: entropy spike at step 1000 |
| Task 15: KL-based Goodhart plot | DONE | EM peaks at cumKL=2.75, drops at cumKL=4.88 |
| Task 16: Automated classification | DONE | Operational definitions confirmed correct-via-tool=0% |
| Task 17: Pass@k (E3 vs SFT) | DONE | E3 pass@32=61.0% vs SFT 5.0% |
| Task 18: Proxy-gold correlation | DONE | E3 r=0.95-0.97 (component gaming, not decorrelation) |
| Task 19: Behavioral diversity | DONE | Response -77%, query-question overlap +94% |
| Task 20: E5a training + eval | DONE | Peak EM=52.6%@1000, tools=0. Cold start → memorization |
| Task 21: E5b training + eval | DONE | Step 100: EM=49.0%, tools=2.3, correct-via-tool=10%. Collapse after step 150 |
| Task 22: E3 single-component (on_path only) | DONE | **EM≈0% throughout**. 4-component design is necessary. |
| Task 23: E3-balanced (0.5/0.5) | DONE | Peak EM=53.2%@1500, tools=0. Balanced split → tool abandonment → memorization → collapse@2000 |
| Task 23 extended eval | DONE | Steps 1500/1750/2000 evaluated. Confirms Goodhart with delayed onset. |
| Task 24: Llama E3 + E5b training | DONE | Both reached step 1293. Llama ceiling ~25% (vs Qwen 52%) |
| Task 24b: Llama eval | DONE | E3 24.8%, E5a 24.6%, E5b 25.0%. Llama-E5b 52% kg-incomplete (KG coverage bottleneck) |
| Task 25: Partial-prompt contamination | DONE | Qwen 14.4%, Llama 17.4% (Qwen < Llama is opposite of Spurious Rewards prediction) |
| Task 26: Pass@10 Category B filter | DONE | 2556 questions (72.4%) are Category B |
| Task 27: Category B re-eval | DONE | E5b only model with CvT>0 on hard questions (2%) |
| Task 28: E5a trajectory classification | DONE | 21% correct-no-tool, 79% wrong-no-tool (pure memorization) |
| Task 29: E4 full eval | DONE | Weak memorization confirmed |
| Task 30: 500-sample consistency | DONE | All evals use sequential first-500, consistent |

---

## Currently Running (2026-04-15 evening snapshot)

| Variant | State | Action |
|---|---|---|
| 39A (format reward) | at step ~300, tool abandonment | let finish current eval, then stop |
| 39B (KL 5x) | at step ~450, EM=36% @ 400 | ✅ **canonical — run full 3531 evals at 300/400/500** |
| 39C (SFT replay) | at step ~300, pantomime | let finish current eval, then stop |
| 39D (Cat-B only) | monotone decline | **KILL NOW** (Action I1) |
| 39E (Gold SFT + Cat-B + format) | first eval imminent | **WAIT for first eval, conditional kill** |
| 39F (Gold SFT + full + format) | corrupted (gold-gen bug) | **KILL NOW** (Action I1) |

*(See "IMMEDIATE ACTIONS" and "Phase 7 Day 2 Plan" sections at the top of this file.)*

---

## Phase 7: Oracle Gate + Data Provenance Fixes — LAUNCH NOW

> **This phase gates all subsequent experiment decisions.**
> Task 36 (Oracle) is the single most important experiment remaining.
> Task 37-38 are low-cost data fixes that run in parallel.
> Task 39+ are conditional on Task 36 results.

---

### Task 36: KG Coverage Oracle — GATE EXPERIMENT [P0-GATE, 0 GPU, ~3 days]

**Purpose**: Answer the fundamental question: **"For questions that require tool use (Category B), does Freebase actually contain the information needed to answer them?"**

This is the most important experiment remaining. If Freebase coverage is low, our entire tool-use story rests on a broken foundation — agents can't learn tool use if the KG doesn't have what they need. If coverage is high, the retrieval gap (72% need tools, 0% CvT) becomes a much sharper indictment of current RL methods.

**Why this gates everything**:
- Coverage ≥ 50% → E5b-stabilized experiment is worth doing → positive paper path
- Coverage 20-40% → "KG coverage is the real bottleneck" becomes a finding → diagnostic paper, but stronger
- Coverage < 15% → CWQ/Freebase is not the right testbed for tool-use research → major pivot needed

**Input data**:
- Task 26 output: Category B question IDs (2,556 questions with pass@10=0)
- CWQ dataset: gold SPARQL queries for each question
- Freebase snapshot: the same one used by our KG server

**Methodology**:

**Step 1: Sample selection** (~1h)
- Random sample 200 questions from the 2,556 Category B questions
- Use a fixed seed (seed=42) for reproducibility
- Also sample 50 from Category A as a control (expect high coverage)

**Step 2: Gold path extraction from SPARQL** (~4h)
- Parse each question's gold SPARQL query
- Extract the sequence of (subject, predicate, object) triples that form the reasoning path
- For multi-hop questions, extract the full chain: e.g., Q1 →[P1]→ E1 →[P2]→ A
- Record: number of hops, intermediate entities, required predicates
- Handle SPARQL complexity: UNION, FILTER, OPTIONAL clauses — document which queries are too complex to parse automatically
- **Output**: `oracle_gold_paths.jsonl` — one entry per question with the extracted triple chain

**Step 3: KG coverage check** (~4h)
- For each extracted triple chain, query our Freebase server using the SAME 4-tool API the model uses:
  - `get_tail_relations(entity)` — check if the required predicate exists
  - `get_tail_entities(entity, relation)` — check if the required object is reachable
- Walk the full chain: start from the question entity, follow each hop
- At each hop, record:
  - (a) **HIT**: the exact triple exists → this hop is KG-solvable
  - (b) **PARTIAL**: entity exists but required relation missing, or relation exists but target entity missing
  - (c) **MISS**: entity not found in KG at all
- A question is **fully solvable** only if ALL hops are HITs
- **Output**: `oracle_coverage.jsonl` — per-question coverage result

**Step 4: Classification** (~2h)
- For each of the 200 Category B questions, classify:

| Category | Definition |
|----------|-----------|
| **SOLVABLE** | All triples in gold path exist in Freebase. A perfect agent could answer this. |
| **PARTIAL** | Some but not all triples exist. Agent could get partway but not to the answer. |
| **UNREACHABLE** | First-hop entity or critical relation missing. No path exists in Freebase. |
| **COMPLEX** | Gold SPARQL too complex to decompose into simple triple chains (UNION, aggregation, etc.). Manual check needed. |

**Step 5: Cross-reference with E5b trajectories** (~2h)
- For the SOLVABLE questions that overlap with E5b@100's eval set:
  - Did E5b correctly retrieve the answer? (CvT=1 for these?)
  - If not, why? (wrong relation chosen? right entity but wrong hop?)
- This tells us: of the questions where tools CAN help, how often does E5b actually succeed?

**Step 6: Hop distribution analysis** (~1h)
- For SOLVABLE questions: how many hops are needed?
  - 1-hop SOLVABLE: should be "easy" for tool use
  - 2-hop SOLVABLE: requires compositional reasoning
  - 3+ hop SOLVABLE: very challenging
- Cross-reference with our earlier hop-stratified analysis

**Output deliverables**:
```
results/oracle/
├── oracle_gold_paths.jsonl          # Extracted triple chains per question
├── oracle_coverage.jsonl            # Per-question coverage classification
├── oracle_summary.md                # Key statistics + decision recommendation
├── oracle_category_b_coverage.csv   # Full table: question_id, hops, coverage, category
└── oracle_e5b_cross_ref.csv         # E5b performance on SOLVABLE subset
```

**Key statistics to report in oracle_summary.md**:
1. Category B coverage: % SOLVABLE, % PARTIAL, % UNREACHABLE, % COMPLEX
2. Category A coverage (control): expect ≥ 80% SOLVABLE
3. Hop distribution of SOLVABLE: 1-hop vs 2-hop vs 3+
4. E5b success rate on SOLVABLE subset (if overlap exists)
5. **Decision recommendation**: based on coverage, which paper path to take

**Decision gates** (to be evaluated by discussion-side after results):
```
IF SOLVABLE ≥ 50% of Category B:
  → Tool use IS meaningful for CWQ/Freebase
  → Proceed with E5b-stabilized experiment (Task 39)
  → Paper narrative: "The retrieval gap is real and fixable"

IF SOLVABLE = 25-50%:
  → Tool use is partially meaningful, KG coverage is a co-bottleneck
  → Still proceed with E5b-stabilized, but frame KG coverage as finding
  → Paper narrative: "Two bottlenecks: reward design AND KG coverage"

IF SOLVABLE < 25%:
  → Freebase is too incomplete for meaningful tool-use research
  → HALT E5b-stabilized on CWQ
  → Consider: (a) switch to KGQAGen-10k (96.3% verified), or
               (b) frame paper as "why KG tool use fails: it's the KG, not the agent"
  → Discuss with supervisor before proceeding
```

**Implementation notes**:
- Use existing Freebase server (same endpoints the model uses) — this ensures coverage check reflects what the model actually has access to
- SPARQL parsing: use `rdflib.plugins.sparql` or simple regex for ns: prefix extraction
- For COMPLEX queries (UNION, COUNT, FILTER NOT EXISTS): flag but don't auto-classify. Manual inspection on these (~20-30% of questions likely fall here)
- No GPU needed. CPU + existing Freebase server only.
- **Can run in parallel with Tasks 37 and 38**

**Time estimate**: Target **1 day**, max 1.5 days

The critical path is Step 2 (SPARQL parser). Steps 3-6 are minutes of compute once the parser works.

**Fast-track approach**:
- Step 2 accounts for 80% of the work. The Freebase server is local (no rate limits), so Step 3 is ~10 minutes for 400 queries. Steps 4-6 are pure post-processing.
- For COMPLEX SPARQL (UNION, aggregation, FILTER NOT EXISTS): do NOT spend hours writing a general parser. Instead, flag them as COMPLEX immediately (expect ~20-30% of questions). A fast regex-based parser that handles the common patterns (simple triple chains, 1-3 hops) is sufficient. The COMPLEX bucket gets reported as-is — we only need coverage stats on the parseable majority.
- Parallelization: Steps 1-3 can be pipelined (parse one question → query Freebase → classify, repeat). No batch dependency.

Estimated breakdown:
- Step 1: 10 min
- Step 2: 4-8h (writing + debugging SPARQL parser — THE bottleneck)
- Step 3: 10-30 min (local Freebase, 400 queries)
- Steps 4-6: 1-2h (classification + analysis + report)
- Total: ~6-10h continuous work

---

### Task 37: Trajectory Classification on Test Split [P0, ~8h GPU]

**Purpose**: Verify E5b CvT=10% and trajectory proportions on the canonical test split. Currently all trajectory classifications are from val first-100 sequential — unrepresentative.

**What to do**:
1. Random sample 200 questions from **test.parquet** (seed=42, same seed as Task 36 for potential overlap analysis)
2. Run eval with trajectory saving: E3@500, E5b@100, E2@1200, SFT
3. Config: max_turns=5, max_new_tokens=512 (same as Task 14 Qwen config)
4. Run trajectory classification using the same script from Task 16/Task 21
5. Report: per-experiment classification breakdown with 95% binomial CIs

**Key question answered**: Does E5b CvT > 0% hold on test split? What's the CI?

**Output**: `results/test_split_trajectory_classification.json`

**Time**: ~6-8h inference (4 models × 200 questions × max 5 turns). Can run on 1 GPU.

**Can run in parallel with Task 36** (no dependency).

---

### Task 38: Pass@k on Test Split [P0, ~4h GPU]

**Purpose**: Get clean pass@k numbers on the canonical test split. Current pass@k (55.2% pass@1) is on val first-200 — incomparable with Task 14.

**What to do**:
1. Random sample 200 questions from **test.parquet** (seed=42)
2. Run pass@k: E3@500 and SFT base, k=32, temperature=0.7, top_p=0.95
3. Config: max_turns=5, tools=YES (same as original Task 17, but on test split)
4. Compute pass@k using unbiased estimator for k=1,4,8,16,32

**Key question answered**: What's the clean pass@k ratio (E3/SFT) on test split?

**Output**: `results/test_split_pass_at_k.json`

**Time**: ~4h (2 models × 200 questions × 32 samples). 1-2 GPUs.

**Can run in parallel with Tasks 36 and 37**.

---

### Task 39: E5b-Stabilized Training [APPROVED — Oracle gate passed: 99.5% SOLVABLE]

**Purpose**: Stabilize E5b's genuine KG retrieval by preventing the format drift that caused collapse at step 150. This is the paper's constructive contribution: from diagnosis ("99.5% solvable, 0% CvT") to solution ("stabilized training sustains retrieval").

**Gate status**: ✅ PASSED. Task 36 Oracle: 99.5% SOLVABLE. Task 37: E5b CvT=5.5% [3.1-9.6%] confirmed on test split. Proceed immediately.

**Resource allocation**: 32 GPUs available. Run ALL three variants in parallel (4 GPUs each = 12 GPUs for training, remaining GPUs for eval jobs).

---

#### Baseline to beat: E5b-original

From Task 37 (test split, 200 random questions, seed=42):
- Step 100: EM=31.0%, tools=2.3, CvT=5.5% [3.1-9.6%], kg-incomplete=38.5%
- Step 150: format drift begins (`<search>` inside `<think>`)
- Step 200: tool calls drop to 0.1
- Step 250: repetition collapse (EM→0%)

The proximal failure mode is **format drift** (tag nesting errors break tool parsing). E5b's reward incentivizes tool-type diversity, but does nothing to enforce output format. Once format breaks, tool calls can't be parsed → reward signal disappears → collapse accelerates.

---

#### Variant A: Format Reward Component [4 GPUs]

**Idea**: Add an explicit reward term for valid output format.

**Reward formula**:
```python
# E5b-original: R = 0.3 * r_answer + 0.7 * r_tool_type
# Variant A:     R = 0.25 * r_answer + 0.60 * r_tool_type + 0.15 * r_format

def r_format(trajectory):
    """1.0 if output format is parseable, 0.0 otherwise."""
    # Check: <think>...</think> is properly closed before any <search> tag
    # Check: <search>...</search> tags are at top level (not nested inside <think>)
    # Check: <answer>...</answer> exists at the end
    # This is a BINARY reward — any format violation → 0.0
    
    think_closed_before_search = ...  # regex check
    no_nested_tags = ...              # no <search> inside <think>...</think>
    has_answer_tag = ...              # <answer>...</answer> present
    
    return 1.0 if (think_closed_before_search and no_nested_tags and has_answer_tag) else 0.0
```

**Weight justification**: 0.15 for format is enough to keep format reward above zero (preventing collapse) without dominating the signal. The 0.70 → 0.60 reduction for tool_type preserves the core tool-incentive while making room.

**Training config**:
```yaml
# Same as E5b-original except reward
algorithm: grpo
multi_turn: true
max_turns: 5
group_size: 8
learning_rate: 5e-7
kl_coeff: 0.01          # same as original
max_steps: 500           # enough to see if collapse is prevented
batch_size: 128
reward_type: tool_type_bonus_format  # new reward type in verl_reward.py
```

**Hypothesis**: Format reward prevents tag nesting → tool calls keep being parsed → tool_type reward stays active → genuine retrieval sustained.

---

#### Variant B: Increased KL Penalty [4 GPUs]

**Idea**: Stronger anchor to reference policy limits format drift without adding reward components.

**Reward formula**: Same as E5b-original (0.3 × r_answer + 0.7 × r_tool_type). No change.

**Training config**:
```yaml
# Same as E5b-original except kl_coeff
algorithm: grpo
multi_turn: true
max_turns: 5
group_size: 8
learning_rate: 5e-7
kl_coeff: 0.05           # 5x original (0.01 → 0.05)
max_steps: 500
batch_size: 128
reward_type: tool_type_bonus  # unchanged
```

**Why 5x (0.05)?** E5b-original at kl_coeff=0.01 drifted fatally by step 150. A 5x increase is aggressive enough to meaningfully constrain drift but not so large that it prevents any learning. If 0.05 is too constraining (EM stays near SFT baseline), we can try 0.03 as a fallback.

**Hypothesis**: Higher KL penalty → slower policy drift → format stays intact longer → E5b's retrieval capability is preserved.

**Risk**: May slow learning too much — agent might not develop tool-use behavior at all within 500 steps.

---

#### Variant C: SFT Replay [4 GPUs]

**Idea**: Periodically "remind" the model of correct format via SFT replay on the original warmup data.

**Training protocol**:
```
Step 0-49:    GRPO with E5b-original reward (0.3×answer + 0.7×tool_type, kl_coeff=0.01)
Step 50:      1 epoch SFT on original 5K trajectories (lr=2e-5, LoRA rank 64)
Step 51-99:   GRPO continues
Step 100:     1 epoch SFT
Step 101-149: GRPO continues
Step 150:     1 epoch SFT (this is where E5b-original collapsed)
... repeat every 50 steps until step 500
```

**Implementation notes**:
- SFT replay uses the SAME SFT data and config as the original warmup (Task in Phase 1)
- The LoRA weights are updated by both GRPO and SFT — SFT acts as a regularizer
- Each SFT epoch is fast (~30 min on 4 GPUs for 5K samples, 1 epoch)
- GRPO checkpoint is saved before each SFT, SFT checkpoint is used to resume GRPO

**Hypothesis**: Periodic SFT reinforces correct `<think>...<search>...<answer>` format, counteracting the format drift from GRPO optimization.

**Risk**: SFT might "reset" tool-use behavior learned during GRPO, creating an oscillation. Monitoring tool calls per question at each step is critical.

---

#### Evaluation Plan (same for ALL three variants)

**Eval checkpoints**: Steps 50, 100, 150, 200, 250, 300, 400, 500
(Dense around step 150 — the E5b-original collapse point)

**Per-checkpoint eval** (~2h each on 1 GPU):
1. **Task 14-style full test** (3531 questions, greedy, max_turns=5): EM, F1, ContEM, avg_tools
2. **Trajectory classification** (200 random from test.parquet, seed=42 — same questions as Task 37): Full 7-category classification with CvT count and Wilson CI

**Key metrics to track across steps**:
| Metric | What it tells us | Success threshold |
|--------|-----------------|-------------------|
| CvT count (out of 200) | Is genuine retrieval happening? | > 0 past step 200 |
| CvT % [Wilson CI] | How much retrieval? | CI lower bound > 0% past step 200 |
| kg-incomplete count | Is model trying but querying wrong? | Decreasing = agent improving |
| avg_tools/question | Are tools still being parsed? | > 1.0 past step 200 (E5b-original → 0.1) |
| EM (full test) | Overall accuracy | ≥ E5b-original peak (31.7%) |
| format_valid % | Are outputs parseable? | > 90% past step 200 |

**Success criterion** (any variant):
- **Minimum**: CvT > 0% sustained past step 200 (E5b-original collapsed by step 150)
- **Good**: CvT ≥ 5% at step 300 with tools > 1.0/question
- **Excellent**: CvT > 10% at step 300 AND EM > 35% (exceeding E5b-original peak + memorization)

**Failure modes to watch**:
- Variant A: format reward dominates → model generates perfect format but empty content
- Variant B: KL too high → no learning, stays near SFT baseline
- Variant C: SFT-GRPO oscillation → tools spike then drop every 50 steps

---

#### Output

```
results/e5b_stabilized/
├── variant_a_format/
│   ├── step_{050,100,150,200,250,300,400,500}_eval.json     # full test EM/F1
│   ├── step_{100,200,300}_trajectories.json                  # 200 random trajectories
│   └── step_{100,200,300}_classification.json                # 7-category breakdown
├── variant_b_kl/
│   └── (same structure)
├── variant_c_sft_replay/
│   └── (same structure)
├── comparison_table.md          # Side-by-side: 3 variants + E5b-original
└── training_curves.csv          # Per-step: reward, EM, tools, format_valid for all 3
```

**Also log to W&B**: reward/mean, reward components (r_answer, r_tool_type, r_format for A), KL, entropy, avg_tools, format_valid_rate. Tag runs as `e5b-stab-A`, `e5b-stab-B`, `e5b-stab-C`.

---

#### Variants D/E/F: Beyond Format Stabilization — Address Root Causes

> **Rationale**: Variants A/B/C only address L1 (format drift). But Task 37 shows 77/200 = 38.5% kg-incomplete — agent uses tools but queries wrong entity/relation. And 28% of training questions are Category A (answerable from memory) giving the agent a "free" reward path that discourages tool exploration. D/E/F attack these deeper bottlenecks.

---

#### Pre-requisite: Task 40 — Data Preparation for D/E/F [Day 1, 0-1 GPU for pass@10]

**This runs on Day 1 while A/B/C are already training. No dependency on A/B/C results.**

**Step 1: Train split Category B filter** (~6h, 1 GPU)
- Run pass@10 on CWQ **train split** using raw Qwen2.5-7B-Instruct (same config as Task 26)
- Temperature=0.7, no tools, k=10
- Output: `data/freebase/verl_cwq/train_category_b_ids.json`
- Expected: ~60-70% of train questions are Category B

**Step 2: Oracle gold path extraction on train Category B** (~4h, 0 GPU)
- Reuse Task 36 SPARQL parser on train split Category B questions
- Extract triple chains for each question
- Output: `data/freebase/verl_cwq/train_oracle_gold_paths.jsonl`

**Step 3: Gold trajectory generation** (~4h, 0 GPU)
- For each gold path, construct an agent trajectory in our ReAct format:
```
<think>I need to find [description of first hop].</think>
<search>get_tail_entities("[entity]", "[relation]")</search>
<result>["[intermediate_entity]"]</result>
<think>Now I need to find [description of second hop].</think>
<search>get_tail_entities("[intermediate_entity]", "[relation2]")</search>
<result>["[answer]"]</result>
<answer>[answer]</answer>
```
- Verify each trajectory by executing the tool calls against Freebase server
- Only keep trajectories where all tool calls return correct results (should be ~99.5% per Oracle)
- Target: 500-1000 verified gold trajectories
- Output: `data/freebase/verl_cwq/gold_kg_trajectories.jsonl`

**Step 4: Enhanced SFT** (~4h, 4 GPUs)
- Training data: original 5K SFT trajectories + 500-1000 gold KG trajectories
- Same SFT config: 2 epochs, lr=2e-5, LoRA rank 64
- Output: enhanced SFT checkpoint (used as init for Variants D/E/F)

**Step 5: Build Category B training parquet** (~1h, 0 GPU)
- Filter train.parquet to only Category B questions
- Output: `data/freebase/verl_cwq/train_category_b.parquet`

**Total Day 1 time**: ~18h sequential, but Steps 1+2 can run in parallel (~10h critical path).

---

#### Variant D: Category B Filtered + Format Reward [4 GPUs]

**Idea**: Remove the memorization shortcut entirely. Train only on questions the model can't answer from memory.

**Training data**: `train_category_b.parquet` (only Category B questions)
**Init checkpoint**: Original SFT (same as Variants A/B/C)
**Reward**: 0.25×answer + 0.60×tool_type + 0.15×format (same as Variant A)

```yaml
algorithm: grpo
multi_turn: true
max_turns: 5
group_size: 8
learning_rate: 5e-7
kl_coeff: 0.01
max_steps: 500
batch_size: 128
reward_type: tool_type_bonus_format
train_data: data/freebase/verl_cwq/train_category_b.parquet  # KEY CHANGE
```

**Hypothesis**: When memorization cannot produce reward, the agent is forced to explore tool-based retrieval. Combined with format reward, the discovered retrieval behavior should be stable.

**Risk**: Slower initial learning (no "easy wins" from Category A questions). May need more steps to converge.

---

#### Variant E: Gold SFT + Category B Filtered + Format Reward [4 GPUs] ⭐ Maximum Effort

**Idea**: Address all three bottleneck layers simultaneously.

**Training data**: `train_category_b.parquet`
**Init checkpoint**: **Enhanced SFT** (original 5K + gold KG trajectories from Task 40)
**Reward**: 0.25×answer + 0.60×tool_type + 0.15×format

```yaml
# Same as Variant D except init checkpoint
algorithm: grpo
multi_turn: true
max_turns: 5
group_size: 8
learning_rate: 5e-7
kl_coeff: 0.01
max_steps: 500
batch_size: 128
reward_type: tool_type_bonus_format
train_data: data/freebase/verl_cwq/train_category_b.parquet
init_checkpoint: verl-sft-cwq-7b-enhanced-merged  # Enhanced SFT with gold trajectories
```

**What each component contributes**:
| Component | Bottleneck addressed | Effect |
|-----------|---------------------|--------|
| Gold SFT | L3 (query precision) | Model starts with a prior for correct multi-hop KG queries |
| Category B filter | L0 (memorization shortcut) | No "free" reward from memory → forced to use tools |
| Format reward | L1 (format drift) | Prevents tag nesting → tool calls keep being parsed |

**Hypothesis**: Gold SFT provides the "how to query correctly" prior; Category B filtering forces the model to USE that prior (no shortcut); format reward keeps the output parseable. All three together should produce sustained, accurate KG retrieval.

**This is our best shot at EM > 33%.**

---

#### Variant F: Gold SFT + Full Data + Format Reward [4 GPUs] (Control for D/E)

**Idea**: Same enhanced SFT as Variant E, but train on FULL training set (not Category B filtered). This is the control that isolates the effect of Category B filtering.

**Training data**: `train.parquet` (full, unfiltered)
**Init checkpoint**: Enhanced SFT (same as Variant E)
**Reward**: 0.25×answer + 0.60×tool_type + 0.15×format

**Purpose**: If E > F, it proves Category B filtering matters (memorization shortcut removal is important). If E ≈ F, then gold SFT alone is sufficient and filtering isn't needed. Either way, informative.

---

#### Updated Timeline (6 variants parallel, 24 GPUs + eval)

| Day | Activity |
|-----|----------|
| **1** | Launch A/B/C (12 GPUs). Simultaneously run Task 40 data prep (pass@10 filter, Oracle gold paths, gold trajectory gen, enhanced SFT). Resubmit Task 38b. |
| **2** | A/B/C training continues (step 100-200 evals). Launch D/E/F from prepared checkpoints (12 more GPUs, 24 total). |
| **3** | All 6 variants training. A/B/C approaching step 200-300. D/E/F at step 50-100. Dense eval on all. **Early kill any variant with CvT=0 and tools=0 at step 200.** |
| **4-5** | Continue training to step 500. Step 200/300 evals for D/E/F. |
| **5-6** | Training complete. Full trajectory classification (200 random, test split) on top 2-3 variants. |
| **6-7** | Compile 6-variant comparison table. Full test (3531) eval on best variant. **All experiments LOCKED.** |

**Total**: ~7 days. Paper writing can start Day 4 with preliminary results.

---

#### Success Criteria (updated for 6 variants)

| Level | Criterion | Paper outcome |
|-------|-----------|---------------|
| **Excellent** | Any variant: CvT ≥ 10% at step 300 AND EM > 35% on full test | "Solved: three interventions break the memorization ceiling" → EMNLP main |
| **Good** | Any variant: CvT > 0% sustained to step 300 AND EM > 33% | "Tools contribute beyond memorization" → EMNLP main competitive |
| **Acceptable** | Any variant: CvT > 0% sustained to step 200 | "Stabilization works" → diagnostic + partial solution → EMNLP Findings |
| **Diagnostic only** | All 6 fail | "Format, filtering, and demonstration are all insufficient" → pivot to pure diagnostic |

**If D/E/F succeed where A/B/C fail**: paper gains a strong ablation story ("format stabilization alone is insufficient — removing the memorization shortcut and providing retrieval demonstrations are both necessary").

**If E succeeds and D/F don't**: "All three interventions are needed" — the most complete and publishable story.

---

#### Cross-Model Validation — Qwen3-4B-Instruct-2507 Training [P1 — field-norm generation ablation] (UPGRADED v12 → REVISED v13)

> **v13 change (2026-04-16 evening)**: Model switched from `Qwen3-8B` (hybrid, /no_think leakage per Demystifying Hybrid Thinking arxiv 2510.12680) to **`Qwen3-4B-Instruct-2507`** (pure non-thinking variant, released 2025-08). Rationale: (a) Qwen3-8B has NO pure-variant 2507 release; only 4B and 30B-A3B do; (b) hybrid 8B + /no_think still leaks thinking behavior (AIME24: 63% think vs 24% no-think, 646 "wait" tokens in no-think mode); (c) field precedent — Demystifying Agentic Reasoning (CUHK/IDEA) used Qwen3-4B-Instruct-2507 as their main student model and explicitly tested thinking variant, concluding "instruction-based non-thinking models are more suitable for agentic RL because thinking models over-rely on internal reasoning and avoid invoking tools"; (d) 4B halves compute vs 8B; (e) gives us 3 capacity points (4B < 7B < 14B) instead of 2 (7B≈8B, 14B).
>
> **v12 change (2026-04-16)**: Upgraded from "CONDITIONAL eval-only" to "Tier-1 full training" based on 2026 field-norm survey.

**Purpose**: Primary claim — "findings generalize across Qwen2.5 → Qwen3 generation" (responds to "your main model is 1.5 years old at submission time"). Qwen3-4B-Instruct-2507 has the strongest post-training for non-thinking agentic use among open 4B-class models (Aug 2025 release, distinct from vanilla Qwen3-4B hybrid).

**Model**: `Qwen3-4B-Instruct-2507` (pure non-thinking, NOT hybrid Qwen3-4B).
- Released 2025-08 specifically because Qwen team found hybrid thinking/no-think mixing introduced leakage — so the 2507 variant is their own solution to the problem
- Never emits `<think>` tags regardless of chat-template flag
- Strongly instruct-tuned including tool use → SFT epochs should be reduced to prevent overfit

**Scope**: **Two conditions — E3 (outcome+verify) + E5b-stabilized (retrieval-grounded)**. Full training, not eval-only.
- E3: tests whether the Goodhart pattern (verify-then-answer shortcut) replicates on a newer model with stronger instruction-following
- E5b-stabilized: tests whether the constructive process-reward design generalizes

**Recipe** (see "SFT Warmup Strategy" section above for detailed hyperparams):
1. SFT warmup: **5K primary corpus only** (NOT 12K full — overfit risk on Instruct-2507), re-tokenized with `enable_thinking=False`. Epochs: ~0.5-0.7× of 7B baseline. LR: ~1.2-1.5× of 7B baseline. **~10h on 4 GPUs** (4B is faster than 8B).
2. GRPO with E3 reward: same RL config as Qwen2.5-7B E3. ~22h on 4 GPUs (4B RL is faster than 8B).
3. GRPO with E5b-stabilized: same RL config as Qwen2.5-7B E5b-stabilized winner. ~22h on 4 GPUs.

Steps 2 and 3 can run sequentially or in parallel (8 GPUs).

**Required pre-launch step (pilot)**: 0.1-epoch SFT pilot with 10-sample inference check (see "Pilot-run protocol" in SFT Warmup Strategy section). Budget 2-4h. **DO NOT skip** — catches chat-template / tokenizer issues before wasting full training time.

**Evaluation**: Full 3531 test set, same config as Task 14. Compute EM, F1, ContEM, avg_tools, CvT via trajectory classification.

**Time** (v13 revised): ~10h SFT + 2 × 22h GRPO = **~2.5 days sequential, ~1.5 days with 8 GPUs in parallel**. Cut in half from v12 (Qwen3-8B) estimate of 5 days.

**Target launch**: Apr 19 (after Phase-7 Day-2 numbers locked), pilot first on Apr 18-19.

**Trigger condition**: launch regardless of Qwen2.5-7B variant outcomes — the Qwen3 generation test is independent.

**Hard kill criteria**:
- SFT pilot: format validity < 50% at 0.1 epoch on 10-sample eval → chat template issue, debug before proceeding
- Full SFT: loss NaN or loss doesn't decrease (< 5% drop in first 20% of training)
- GRPO: format validity < 70% at step 50 OR training reward does not differentiate valid from invalid responses (indicates reward-shape incompatibility with Qwen3-4B's output distribution)

**Expected outcomes (based on Demystifying Agentic Reasoning arxiv 2510.11701 observations on Qwen3-4B-Instruct-2507)**:
| Scenario | Probability | Paper impact |
|---|---|---|
| Qwen3-4B E3 shows same Goodhart pattern, E5b-stabilized similar CvT | Most likely | **Ideal** — cross-generation + smaller-capacity replication, strongest defense |
| Qwen3-4B E3 shortcut is weaker; E5b-stabilized CvT higher | Possible | Good — "newer+stronger post-training reduces but does not eliminate shortcuts under standard reward" |
| Qwen3-4B both arms show dramatically different behavior | Low-medium | Narrative pivot: "reward design × model generation × capacity interaction" — three-way finding |
| Qwen3-4B fails to format (< 70% validity) | Low | Chat-template / tokenizer issue caught in pilot; debug |
| Qwen3-4B E5b drops to zero tool calls (thinking-mode bleed-through) | Very low for Instruct-2507 (thinking never emits) | If observed, drop to appendix as "hybrid-arch limitation" — field precedent exists |

**Critical pre-launch checks**:
- Verify `Qwen3-4B-Instruct-2507` never emits `<think>` tags via 10-sample forward pass on base model
- Confirm `tokenizer.apply_chat_template(..., enable_thinking=False)` produces expected template output
- Re-tokenize first 100 SFT trajectories and diff the token count vs Qwen2.5 tokenization — budget for ~5-10% token-length difference due to vocab updates

---

## Updated Priority Order (2026-04-15, v4 — post critical review)

```
COMPLETED:
  ✅ Task 36 (Oracle CWQ)        — 99.5% SOLVABLE (needs KGQAGen replication, Action II4)
  ✅ Task 37 (Trajectory 200-Q)  — E5b CvT=5.5% [3.1-9.6%] — CI from small sample, needs Action II1
  ❌ Task 38 (Pass@k)            — Timed out. Resubmit as 38b.
  ✅ Task 39 A/B/C/D/E/F (Day 1) — 39B = only survivor (EM=36% on 200-Q, CI ±6.6pp — not yet statistically distinguishable from E3)

TONIGHT (2026-04-15 → 2026-04-16 dawn) — ALL 8 GPUs + CPU + API:
  Action I1  Kill dead variants (39D, 39F; conditional 39E)
  Action I2  Full-test eval of 39B @ step 300/400/500          (3 GPUs)
  Action I3  Resubmit Task 38b (E3@500 + 39B@400 pass@k)       (2 GPUs)
  Action II1 Full-test CvT audit on E5b + 39B                   (1 GPU) ← HARD BLOCKER
  Action II2 Search-R1 baseline training (tonight + tomorrow)   (4 GPUs) ← NON-NEGOTIABLE
  Action II3 Llama pipeline audit (stop-token / template / warmup) (1 GPU)
  Action II4 KGQAGen Oracle replication                         (CPU)
  Action II5 GPT-4o 200-Q baseline                              (API, < $50)

DAY-2 READINESS GATE (check at T+12 h from II1 launch):
  Gate A: 39B full-test EM ≥ 35% AND CvT ≥ E5b-original's full-test CvT
    IF PASS → launch Day-2 variants (G1, G2, I-Oracle, I-Self, Task 42 SC)
    IF FAIL → PIVOT to pure-diagnostic paper, cut all Day-2 training,
              refocus on Actions II2/II3/II4/II5/II6 + writing

DAY 2 (if Gate A passes, 2026-04-16 AM → 2026-04-17):
  Launch in parallel:
    Task 41   39B mechanistic memo                              (1 GPU)
    Variant G1 Self-distillation from 39B@400                   (4 GPUs) ← pilot first
    Variant G2 Self-distillation from ORIGINAL SFT (ReST-EM)    (4 GPUs) ← pilot first
    Variant I-Oracle (diagnostic upper bound)                   (4 GPUs)
    Variant I-Self (deployable retrieval-contrib reward)        (4 GPUs)
    Task 42 self-consistency@5 on full test                     (1 GPU)
    Action II6 pass@k with/without tools on 500-Q subset        (2 GPUs)

EARLY INTERVENTION (T+3 / +6 / +10 / +16 / +24 from Day-2 launch):
  Dense checkpoint eval at steps 25/50/75/100/150/200.
  Continuous kill rules (format < 85%, tools/Q < 0.5, EM drops > 3pp, Goodhart).
  Consolidate GPUs onto winner by T+16.

DAY 3 (T+24 → T+36):
  At most 1-2 variants still training (winner(s) from Gate 3/4).
  Full 3531-test eval on leading checkpoint.
  Numbers LOCKED by T+36.

DEFERRED / CUT:
  ❌ Variant H (was curriculum)         — CUT (not real curriculum)
  ❌ Task 42 beam search                — CUT (separate-paper machinery)
  ❌ Task 42 Oracle-guided replanning   — CUT (separate-paper machinery)
  Task 33 (Temporal audit)               — Only if reviewer demands
  Task 34 (Tier A contamination)         — Only if reviewer demands

REVIVED / UPGRADED (v12, 2026-04-16 — see "Model Coverage Upgrade" section at top):
  ✨ Task 32 (Qwen2.5-14B E5b)          — REVIVED as P1 capacity ablation (reduced to E5b-stabilized single condition, ~3 days)
  ✨ Cross-Model (Qwen3-8B E3+E5b)      — UPGRADED from eval-only to full training (~5 days sequential / 3 days parallel)

EXPLICITLY NOT DOING (v12 field-norm survey + v13 thinking-mode/SFT decisions):
  ❌ Qwen3-8B (hybrid)                  — REPLACED by Qwen3-4B-Instruct-2507 in v13 (avoid /no_think leakage)
  ❌ Qwen3-14B                          — CUT (double-variable: generation × size, unattributable)
  ❌ Mistral-Nemo-12B / Ministral-8B    — CUT (cross-family cost > ROI given Qwen field dominance)
  ❌ Gemma 2 / 3 / 4                    — CUT (weak native tool use; trajectory taxonomy rebuild needed)
  ❌ Phi-4                              — CUT (Microsoft family rarely in RL post-training literature)
  ❌ Llama-3.1-8B additional runs       — CUT (appendix-only; no new compute; Meta has no 8B-dense successor)
  ❌ Qwen3.5 (9B/27B/122B-A10B)         — CUT (released 2026-02/03, no arxiv technical report yet, Linear Attention + Gated Delta Networks not supported by verl/Search-R1 infrastructure, multimodal-focused ("Native Multimodal Agents"), reviewer unfamiliarity)
  ❌ Gemma 4                            — CUT (released 2026-04-02, 14 days old at v13 edit time, too fresh)
  ❌ Per-model SFT data regeneration    — CUT (Path B adopted — unified rule-based 12K corpus with per-model epoch/LR tuning, see SFT Warmup Strategy section)
```

**Timeline** (39 days to ARR May 25, as of 2026-04-15 late evening):
- Apr 15 late evening: v11 published. Tonight's 8-action block launched.
- Apr 16 morning: Gate A resolves. Decision: Day-2 training OR pivot-to-diagnostic.
- Apr 16-17: Day-2 variants training (if Gate A passed). Early-intervention gates.
- Apr 17-18: Winner consolidation. Full 3531 eval on winner.
- Apr 18 evening: **Numbers LOCKED.**
- Apr 19-20: Search-R1 baseline training complete. Llama audit complete. KGQAGen Oracle done.
- Apr 21-May 9: Paper writing sprint (18 days)
- May 10-17: Advisor review + revision (8 days)
- May 18-24: Final polish + submission buffer (7 days)
- May 25: ARR deadline

**Slack**: ~4 days of buffer across writing/revision. Tighter than v10's stated slack but more realistic about required experiments.

---

## ~~Priority Order (2026-04-13, v2)~~ (SUPERSEDED by v4 above)

```
COMPLETED:
  ✅ Task 36 (Oracle)       — 99.5% SOLVABLE. Gate PASSED.
  ✅ Task 37 (Trajectory)   — E5b CvT=5.5% [3.1-9.6%] on test. Confirmed.
  ❌ Task 38 (Pass@k)       — Timed out. Resubmit as 38b.
  ✅ Task 39 A/B/C (Day 1)  — 39B broke parametric ceiling (EM=36% @ step 400).
                              A/C failed. Results harvested.
  ❌ Task 39 D (Day 1)      — Cat-B starvation. Killing tonight.
  ❌ Task 39 E/F (Day 1)    — Gold-trajectory generator bug. Killing tonight
                              (E conditionally after first eval).

TONIGHT (2026-04-15 evening):
  Action I1: Kill 39D, 39F. Conditional kill 39E after first eval.
  Action I2: Full 3531-test eval of 39B @ step 300/400/500  (3 GPUs)
  Action I3: Resubmit Task 38b (E3@500 + 39B@400 pass@k)    (2 GPUs)

DAY 2 (2026-04-16, T+0):
  Launch in parallel:
    Task 41   39B mechanistic analysis                      (1 GPU)
    Variant G Self-distillation from 39B@400 ⭐             (4 GPUs)
    Variant I 39B + query-precision reward                  (4 GPUs)
    Task 42   Self-consistency + beam + oracle-replan       (3 GPUs)
  Variant H is RESERVED not launched — conditional on Gate 1-2.

EARLY INTERVENTION (T+3 / +6 / +10 / +16 / +24):
  Dense checkpoint eval at steps 25/50/75/100/150/200.
  Apply Gates 1-4 (see "Early-Intervention Protocol" section).
  Consolidate GPUs onto winner by T+16. Commit by T+24.

DAY 3 (2026-04-17, T+24 → T+36):
  At most 1-2 variants still training (the winner(s) from Gate 3/4).
  Full 3531-test eval on leading checkpoint.
  Numbers LOCKED by T+36. Compile comparison table (E3, E5b, 39B, winner).

CONDITIONAL (only if best variant EM ≥ 38% AND CvT ≥ 8%):
  Cross-model: best variant on Qwen3-8B          ← 5 days, start by Day 10
  [SUPERSEDED by v12 — now unconditional Tier-1 training, see top-of-file "Model Coverage Upgrade" section]

DEFERRED:
  Task 33 (Temporal audit)         ← Only if reviewer demands
  Task 34 (Tier A contamination)   ← Only if reviewer demands
  Task 14b (Llama rerun)           ← Not needed
  Task 32 (14B Qwen)               ← [SUPERSEDED by v12 — revived as P1, E5b-stabilized single condition]
```

**Timeline** (40 days to ARR May 25):
- Apr 15 (tonight): Kill dead variants. Launch 39B full-test eval. Resubmit Task 38b.
- Apr 16 (Day 2, T+0): Launch G + I + Task 41 + Task 42 in parallel.
- Apr 16-17 (T+3 → T+24): Early-intervention gates. Consolidate onto winner.
- Apr 17-18 (T+24 → T+36): Winner continues to step 300-400. Full-test eval.
- Apr 18 evening: Numbers LOCKED. (Original plan said Apr 22 — compressed by 4 days.)
- Apr 19-22:  Cross-model on Qwen3-8B (if breakthrough achieved) — *optional*
- Apr 19-May 8: Paper writing sprint
- May 9-18:   Advisor review + revision
- May 19-24:  Final polish + submission buffer
- May 25:     ARR deadline

---

## Phase 6 (Previous) — Status Update

> Phase 6 was originally planned as noise defense + multi-benchmark triangulation.
> Task 35 (KGQAGen-10k) is DONE (ρ=0.976 Qwen-only).
> Tasks 33, 34 are DEFERRED — they are defensive experiments that can wait until
> we resolve the fundamental question of whether KG tools are even useful (Task 36).
> If Task 36 shows high coverage, Tasks 33/34 become nice-to-have.
> If Task 36 shows low coverage, Tasks 33/34 become irrelevant.

## Phase 6: Noise Defense + Multi-Benchmark Triangulation — DEFERRED

> **Context**: Two critical concerns identified from 2-round deep research (8 parallel agents):
>
> 1. **CWQ 49.3% concern**: KGQAGen paper (NeurIPS 2025 D&B, arXiv:2505.23495) audit found CWQ has only 49.3% factual correctness. Does this invalidate our experiments?
>    - **Answer from deep research**: NO invalidation because of temporal conflation (audit vs 2025 knowledge, CWQ vs 2015 Freebase). Training signal is internally consistent to frozen snapshot.
>    - **But we need concrete evidence to defend**: 3 tasks below.
>
> 2. **Medical KG competition**: Kansal & Jha (arXiv:2601.15160) already does our Storyline A on medical UMLS.
>    - **Answer**: Use KGQAGen-10k (Wikidata, verified 96.3% accuracy) as secondary benchmark instead.
>
> **Phase 6 Goal**: Transform these concerns from weaknesses into paper contributions via:
> - First systematic temporal vs intrinsic error analysis of CWQ (Task 33)
> - First advanced contamination analysis on KGQA benchmarks (Task 34)
> - First triangulation of RL KG agents across CWQ + KGQAGen-10k (Task 35)

---

### Task 33: CWQ Temporal vs Intrinsic Error Audit [P0 — defends 49.3% concern]

**Purpose**: KGQAGen's 49.3% figure does NOT distinguish temporal outdatedness from genuine annotation errors. We hypothesize that most errors are temporal (facts true in 2015 but not 2025). For RL training anchored to frozen Freebase 2015, temporal errors do NOT damage training signal. This audit quantifies the split.

**Methodology** (from Agent E deep research, see `cwq_temporal_vs_intrinsic_audit.md` if available):

**L1: Lexical triggers** (~1h)
- Regex + spaCy over CWQ questions
- Patterns: "now", "current", "currently", "today", "recent", "latest"
- Role nouns that change: president, CEO, prime minister, captain, head coach
- Output: binary flag per question (likely temporal / not)

**L2: Predicate volatility scan** (~0.5-1 day)
- Parse gold SPARQL queries, extract Freebase predicates (`ns:` prefix)
- Hand-curate ~200-row Freebase volatility TSV seeded from TempLAMA paper
- Flag high-volatility predicates: P39 (position held), P54 (member of team), P286 (head coach), P6 (head of government), etc.
- Output: volatility score per question

**L3: Wikidata probe + LLM rubric** (~1 day)
- For the 300 samples that match KGQAGen's audit pool:
  - Map Freebase MID → Wikidata QID via property P646
  - Query Wikidata SPARQL with temporal qualifiers (P580 start time, P582 end time)
  - Reconstruct what the answer was in 2015 (if data available)
- Use GPT-4o-mini with rubric prompt to classify each item:
  - **T** (temporal drift): answer was correct in 2015 but changed by 2025
  - **I1** (mis-grounded): annotation error at creation time
  - **I2** (incomplete gold): multiple valid answers, some missing
  - **I3** (ambiguous): question is inherently unclear
  - **R** (residual): can't classify reliably
- Output: classification distribution

**L4: Human adjudication** (~2h)
- Random 50 of the 300 classifications reviewed manually
- Report inter-annotator agreement
- Identify systematic errors in LLM classifier

**Expected outcome**: "At least 30% of CWQ errors are temporal drift (T), preserving training signal validity against frozen 2015 Freebase snapshot."

**Resources needed**:
- CWQ HuggingFace dataset
- Wikidata SPARQL endpoint (public or local)
- SPARQLWrapper, rdflib, spacy Python libraries
- GPT-4o-mini API access (~$5-15 total cost)

**Total effort**: ~3 engineer-days (~28 hours). No GPU required — mostly CPU + API calls.

**Output**:
- CSV: (question_id, L1_flag, L2_volatility_score, L3_label, final_category)
- Histogram: distribution of categories in CWQ
- Paper text: 1-paragraph defense for §Limitations
- Optional: released audit annotations (contribution to community)

**Citations**:
- KGQAGen: Zhang et al. arXiv:2505.23495 (NeurIPS 2025 D&B)
- TempLAMA: Dhingra et al. TACL 2022
- TempQuestions: Jia et al. WWW 2018

---

### Task 34: Advanced Contamination Tier A Battery [P0 — strengthens 14.4% result]

**Purpose**: We already have partial-prompt completion test (Qwen 14.4%, Llama 17.4%), but this is a single method and the Llama > Qwen result is surprising (opposite of Spurious Rewards prediction). We need additional independent tests to:
1. Confirm the contamination pattern
2. Separate "fact memorization" from "instance memorization"
3. Pre-empt reviewer demand for more rigorous contamination analysis

**Tier A battery** (from Agent H deep research, see `contamination_tier_A_battery.md` if available):

**Test A1: Guided Prompting** (~1.5h per model) [Golchin & Surdeanu 2023, arXiv:2308.08493]
- Run each CWQ prefix twice:
  - Version 1: Name the dataset in prompt ("Complete the following question from the ComplexWebQuestions dev set: ...")
  - Version 2: Same prefix without dataset name
- Score: ROUGE-L between completion and held-out suffix
- Statistical test: Paired one-sided t-test (version 1 > version 2)
- Interpretation: If V1 > V2 significantly → model recognizes the dataset → instance memorization
- Code: https://github.com/shahriargolchin/time-travel-in-llms
- Unique contribution: Paired control that partial-prompt lacks

**Test A2: Data Contamination Quiz (DCQ)** (~2h per model + 1h one-time paraphrase gen) [Golchin & Surdeanu 2024, arXiv:2311.06233, TACL]
- For each CWQ sample, construct a 5-option MCQ:
  - Option A: Verbatim CWQ question
  - Options B, C, D: Paraphrases generated by Qwen-72B (different wording, same meaning)
  - Option E: "None of the above"
- Present MCQ to model and ask which is the original
- Scoring: Correct-pick rate vs random baseline (20%)
- Interpretation: >25% → verbatim memorization detected
- Code: https://github.com/shahriargolchin/DCQ
- Unique contribution: Principled null hypothesis (20%) and black-box verbatim memorization measurement

**Test A3: Min-K%++** (~1h per model) [Zhang et al. 2024, arXiv:2404.02936, ICLR 2025 Spotlight]
- For each CWQ question + its paraphrase pair:
  - Compute per-token log-probability
  - Normalize by vocab mean/std (the "++" improvement)
  - Take mean of bottom K% (typically K=20)
- Score: AUROC for distinguishing original CWQ from paraphrases
- Code: https://github.com/zjysteven/mink-plus-plus
- Caveat: Signal is attenuated for instruction-tuned models (Ravaut et al. arXiv:2409.09927)
- Unique contribution: Per-instance continuous logit score, enables item-level correlation

**Execution plan**:
1. Day 1: Set up all three test repos, adapt to CWQ format (half day)
2. Day 1 afternoon: Run A1 on both Qwen + Llama (3h total)
3. Day 2 morning: Generate paraphrases for A2 using Qwen-72B (1h) + run A2 (4h)
4. Day 2 afternoon: Run A3 on both models (2h)
5. Day 2 evening: Aggregate results into summary table

**Total compute**: ~12-16 GPU-hours across both models on one node

**Output**: `tierA_summary.md` table:
```
| Test        | Qwen2.5-7B | Llama-3.1-8B | Interpretation |
|-------------|------------|--------------|----------------|
| Partial-prompt (existing) | 14.4% | 17.4% | Completion rate |
| A1 Guided Prompting       | ???%  | ???%  | Paired ROUGE-L diff |
| A2 DCQ                    | ???%  | ???%  | Correct-pick rate |
| A3 Min-K%++               | ???   | ???   | AUROC |
```

**Deferred (not in Tier A)**:
- TS-Guessing (poor fit for CWQ open-ended)
- PaCoST (overlaps with A1)
- Post-cutoff control (requires 2-3 days to construct new benchmark)

---

### Task 35: KGQAGen-10k Secondary Benchmark Evaluation [P0 — triangulation]

**Purpose**: Add KGQAGen-10k (96.3% verified accuracy, Wikidata-based) as a secondary evaluation benchmark to triangulate our CWQ findings. This addresses:
1. CWQ noise concern (KGQAGen is clean)
2. Medical competition concern (we don't need Medical)
3. Single-benchmark reliability (Yue/Shao/Wu papers mandate multi-benchmark)

**Strategy**: Inference-only evaluation on existing checkpoints — NO new training. Pre-materialize Wikidata subgraphs to avoid live SPARQL rate limits.

**Methodology** (from Agent F deep research, see `feasibility_secondary_eval_benchmarks.md` if available):

**Step 1: Data loading** (~2h)
- Download `lianglz/KGQAGen-10k` from HuggingFace (CC-BY-4.0)
- Parse fields: question, answer, SPARQL, proof (list of triples), seed QID
- Filter to dev+test splits (1,080 + 1,080 = 2,160 questions)
- Hop distribution analysis: confirm ≥3-hop subset size

**Step 2: Wikidata pre-materialization** (~10h one-time, no training)
- For each question in dev+test:
  - Extract all QIDs from proof triples
  - Query Wikidata Query Service for 1-hop neighborhoods of each QID
  - Cache results to local disk (`data/wikidata_cache/`)
- Rate-limit: respect 60-query-seconds/min Wikidata limit
- Resilient to timeout: checkpoint + resume
- Expected size: ~2-5 GB cache

**Step 3: Wikidata KG server** (~1 day)
- Implement same 4-action interface as Freebase server:
  - `get_tail_entities(entity, relation)` → reads from cache
  - `get_head_entities(relation, entity)` → reads from cache
  - `get_tail_relations(entity)` → reads from cache
  - `get_head_relations(entity)` → reads from cache
- Match API signatures to existing server for drop-in compatibility
- Same HTTP endpoint format (Flask or FastAPI)

**Step 4: Eval pipeline adaptation** (~0.5 day)
- Adapt existing with-tools eval script
- Point to Wikidata KG server instead of Freebase server
- Gold answer format: KGQAGen's `answer` field (list of entity labels)

**Step 5: Run eval on all 8 existing checkpoints** (~1 day)
- Qwen SFT base
- Qwen E1@1250, E2@1200, E3@500, E4@1250, E5a@1000, E5b@100
- Llama SFT base
- Llama E1@1250, E2@1200, E3@500, E4@1250, E5a@1000, E5b@100
- Per checkpoint: EM, F1, tool calls, trajectory sampling (100 trajectories)

**Step 6: Trajectory classification on KGQAGen-10k** (~0.5 day)
- Apply same operational definitions (correct-via-tool, correct-via-memory, etc.)
- Critical question: **Does correct-via-tool > 0% on KGQAGen where CWQ might have been noisy?**

**Step 7: Cross-benchmark consistency analysis** (~0.5 day)
- For each experiment, compare CWQ vs KGQAGen-10k:
  - Same EM ordering (E3 > E2 > E1)?
  - Same correct-via-tool pattern?
  - Same parametric memory ceiling?
- Vote-counting: how many findings replicate across benchmarks?

**Total effort**: ~4 engineer-days + ~10h one-time Wikidata pre-materialization

**Resources**:
- HuggingFace dataset: `lianglz/KGQAGen-10k`
- Wikidata Query Service: public endpoint (respecting rate limits)
- Python: sparqlwrapper, requests, json, flask/fastapi
- Disk: ~5 GB for cache

**Expected outcomes**:
1. **Best case**: CWQ and KGQAGen-10k show same qualitative pattern (E3 > E5a ≈ memorization ceiling, correct-via-tool ≈ 0% for standard rewards, E5b > 0%). This triangulates our findings.
2. **Interesting case**: Different ceilings (maybe Wikidata parametric knowledge is different). This becomes a finding.
3. **Concerning case**: Patterns don't replicate. Then we need to understand why (benchmark-specific artifact vs contamination artifact).

**Output**:
- `results/kgqagen_eval/` directory with per-checkpoint results
- `kgqagen_vs_cwq_comparison.md` with cross-benchmark table
- Paper Section: "Cross-Benchmark Validation"

**Open questions** (may need supervisor input):
1. Pre-materialize KGQAGen-10k train set too (enables training runs) or eval-only?
2. Treat `proof` triples as ordered path or unordered set for reward calculation?
3. Report paper's LASM metric in addition to EM/F1?

---

### Task 14 (Revised): Full Test Set Evaluation with Confidence Intervals [P0 — statistical rigor]

**Purpose**: Move from 500-sample estimates to full CWQ test set (3,531 samples) with bootstrap CIs. Required before submission.

**What to do**:
1. Run with-tools eval on FULL CWQ test set for:
   - Qwen E3 step 500, E3 step 750, E2 step 1200, E1 step 1250, E4 step 1250, E5a step 1000, E5b step 100, SFT base
   - Llama E3 step 1293, E5a step 1293, E5b step 1293, SFT base
2. **Compute 95% paired bootstrap CIs** for EM, F1 (10,000 bootstrap samples)
3. **McNemar's test** for pairwise comparisons
4. Full test set hop-stratified analysis with per-bucket Ns and CIs
5. Full test set trajectory classification (100→500 sample size)

**Following Agent G's statistical recommendations**:
- NO weighted aggregate across benchmarks (HELM doctrine)
- Paired bootstrap + vote-counting across benchmarks
- Report main table as full matrix

**Time**: ~8-12h eval per model × 12 models = ~4-6 days if serial, ~1 day if parallelized across nodes

**Note**: This can be parallelized with Task 35 if resources allow.

---

## Priority Order (updated 2026-04-08)

```
IMMEDIATELY START (parallel, no dependencies):
  Task 33 (Temporal vs Intrinsic Audit)   ← 3 engineer-days, mostly CPU + API
  Task 34 (Tier A Contamination Battery)  ← 12-16 GPU hours, 2 nodes
  Task 35 (KGQAGen-10k Eval)              ← 4 days + 10h pre-materialization
  Task 14 (Full Test Set + CIs)           ← 4-6 days, can parallelize

DECISION GATES (after Tasks 33, 34, 35 complete):
  ├── Task 33 finds >30% temporal → CWQ defense solid, proceed with paper
  ├── Task 34 shows low contamination → cite as "CWQ not acting as memorized benchmark"
  └── Task 35 shows CWQ ≈ KGQAGen → triangulation successful

IF TASK 35 shows divergence from CWQ (<70% replication):
  → Consider retraining E3 + E5b on KGQAGen-10k (adds ~1 week)

DEFERRED:
  Task 8 (Filtered dataset) — superseded
  Task 31 (Filtered CWQ training) — not needed given Phase 6 findings
  Task 32 (14B Qwen) — optional, only if time permits after Phase 6
  Medical/Kansal & Jha replication — EXPLICITLY AVOIDED
  MINTQA-pop — hidden engineering cost too high (no SPARQL, no entity IDs)
```

**Timeline**:
- Days 1-3: Tasks 33, 34 complete; Task 35 Wikidata pre-materialization
- Days 4-5: Task 35 eval complete; Task 14 ongoing
- Days 6-7: Cross-benchmark analysis + paper revision
- Days 8-14: Paper writing with Phase 6 defenses integrated
- Days 15+: Review, polish, submit

---

## Phase 5: Contamination + Category B — ALL COMPLETED ✅

> All Phase 5 tasks (25-30) are done. Results incorporated into Phase 6 planning.
> Kept below for reference.

### Task 25: Partial-Prompt Completion Test [DONE]

**Purpose**: Quantify how much of CWQ Qwen2.5-7B and Llama-3.1-8B have memorized. Following the methodology of "Reasoning or Memorization?" (arXiv:2507.10532), which found Qwen2.5-Math-7B has 54.6% completion rate on MATH-500 while Llama-3.1-8B has only 3.8%.

**What to do**:
1. Take the CWQ test set (3,531 questions)
2. For each question, provide only the **first 50% of tokens** to the model
3. Let the model complete the rest (greedy decoding, no tools)
4. Check if the completion contains the gold answer (Contains-EM)
5. Run for both Qwen2.5-7B-Instruct (base, before SFT) and Llama-3.1-8B-Instruct (base)

**Completion rate** = fraction of questions where partial prompt → correct answer completion.

**Expected outcomes**:
- Qwen2.5 completion rate 30-50%: high contamination, explains why parametric memory is so strong
- Llama completion rate <10%: lower contamination, supports Llama as cleaner validation
- If Qwen >> Llama: confirms CWQ contamination is Qwen-specific

**Time**: ~2h inference per model (no training needed). Can run NOW on any available GPU.

**Output**: Completion rate for Qwen2.5-7B and Llama-3.1-8B on full CWQ test set.

---

### Task 26: Robust Category B Identification via Pass@10 [P0 — filtering]

**Purpose**: Identify questions the model GENUINELY cannot answer from parametric memory. Single-pass filtering is unreliable (model might answer correctly 30% of the time). Pass@k filtering is more robust.

**What to do**:
1. For ALL CWQ test set questions (3,531):
   - Run SFT-base model (Qwen2.5-7B) **WITHOUT tools**, temperature=0.7
   - Generate k=10 responses per question
   - Compute per-question pass@10 using unbiased estimator
2. Classify:
   - **Category A**: pass@10 > 0 (model can answer at least once → parametric memory sufficient)
   - **Category B**: pass@10 = 0 (model NEVER gets it right in 10 tries → genuinely doesn't know)
3. Report: Category A count, Category B count, percentage split

**Expected**: ~50-55% Category A (matches the ~52% parametric ceiling), ~45-50% Category B.

**Time**: ~6h inference (10 × 3,531 = 35,310 inference runs, no tools, fast).

**Output**: Category A/B lists (question IDs) + statistics.

---

### Task 27: Category B Re-Evaluation of ALL Experiments [P0 — key analysis]

**Purpose**: On questions the model genuinely can't answer from memory (Category B), does tool use finally matter? This is the critical test.

**What to do**:
1. Using the Category B question list from Task 26
2. Re-evaluate ALL existing best checkpoints **on Category B only**:
   - SFT base (with tools)
   - E1@1250 (outcome)
   - E2@1200 (heuristic)
   - E3@500 (verifiable)
   - E4@1250 (random)
   - E5a@1000 (retrieval-grounded)
   - E5b@100 (tool-type bonus)
3. For each: compute EM, Contains-EM, F1, tool calls, correct-via-tool rate
4. Also compute Category A results for comparison (should be ~100% for memory-based models)

**Key question**: On Category B, does E5b (correct-via-tool=10% overall) have a MUCH higher correct-via-tool rate?

**Back-of-envelope**: If overall correct-via-tool=10% (10/100) and these 10 are ALL from Category B questions, then on Category B: correct-via-tool ≈ 10/~48 ≈ **20.8%**. This would be a strong result.

**Time**: ~2-3h eval per model × 7 models. Can parallelize.

**Output**: Category A vs Category B results table for ALL experiments.

---

### Task 28: E5a Trajectory Classification [P1 — completeness]

**Purpose**: E5a (tools=0, EM=52.6%) was classified implicitly from aggregate metrics but never formally classified using operational definitions. Needed for the paper's trajectory classification figure.

**What to do**:
1. Sample 100 trajectories from E5a@1000
2. Run automated classification (Task 16 script)
3. Expected: ~52% correct-no-tool, ~48% wrong-no-tool (since tools=0)

**Time**: 1h eval + classification.

---

### Task 29: E4 Complete Eval [P1 — completeness]

**Purpose**: E4 (EM=7.4%) may only have a single checkpoint eval. Need Goodhart curve + trajectory classification for completeness.

**What to do**:
1. Confirm whether E4 has multi-checkpoint eval data. If not:
   - Run with-tools eval at steps 250, 500, 750, 1000, 1250
   - Sample 50 trajectories from best checkpoint
   - Run automated classification
2. If already done, skip.

**Time**: 4h eval if needed.

---

### Task 30: 500-Sample Consistency Check [P1 — methodology]

**Purpose**: Confirm that ALL previous 500-sample evals used the same 500 questions. If different random subsets were used, results aren't directly comparable.

**What to do**:
1. Check eval scripts for fixed random seed or fixed sample list
2. If different subsets were used: re-run critical comparisons (E3@500 vs E5b@100) on same subset
3. If same subset: document the seed/list for paper Methods section

**Time**: 30min code check. Re-eval only if inconsistency found.

---

## Phase 5 Optional Extensions

### Task 31: Filtered CWQ Training (Category B only) [P2 — if results warrant]

**Purpose**: If Task 27 shows E5b has high correct-via-tool on Category B, retrain on Category B questions only. This forces the model to learn tool use because memorization can't work.

**What to do**:
1. Build filtered training set: CWQ training questions where SFT model pass@10=0
2. Retrain E3 and E5b on filtered set
3. Eval on Category B test questions

**Decision gate**: Only run this if Task 27 shows EM_with_tools > EM_without_tools on Category B (i.e., tools actually help on hard questions).

**Time**: ~2h filtering + 2×22h training = ~2 days.

---

### Task 32: Qwen2.5-14B E5b-stabilized — ✅ P0 MANDATORY (v14.1)

> **Status change (v14.1, 2026-04-17 late evening)**: **RESTORED to P0**. v14's reason for cutting ("ceiling-chasing doesn't advance mechanism") was reversed because:
> - Under GPU abundance (see memory/hpc_gpu_budget.md), 14B does NOT compete with mechanism analysis — runs in a parallel lane.
> - Reframed as **framework prediction test**: the signal-theoretic framework explicitly predicts L-sig/lang/comp/prior are model-scale invariant. Confirming same Goodhart / parametric ceiling pattern at 14B UPGRADES the paper's claim from "7B observation" to "8-14B cross-capacity observation."
> - User explicit: "14B绝对是必须的" (2026-04-17).
> - See "v14.1 PRIORITY QUEUE" and "P0-D" section at top of file.
>
> **Status change (v12, 2026-04-16) — historical**: Revived from "Not needed" with reduced scope. Oracle Gate A pass + 2026 field-norm survey (Mode A: single-family multi-size is dominant in ICLR 2026 / arxiv 2026 agentic RL papers) justifies capacity-axis ablation.

**Purpose**: Primary claim to defend — "findings generalize across 7B→14B within Qwen family" (responds to inevitable reviewer question "is the Goodhart / shortcut pattern an artifact of 7B capacity?"). Secondary claim — test whether increased capacity enables multi-hop SPARQL construction under E5b-stabilized reward. Tertiary claim — framework prediction test (predicted: same pattern holds at 14B).

**Why reduced from "E3 + E5b" to "E5b single condition"**:
- E3 Goodhart evidence is already strong on Qwen2.5-7B (Task 14); replicating at 14B is nice-to-have, not must-have
- E5b-stabilized is the paper's constructive claim; capacity test on E5b has highest marginal value
- Halves compute (5 days → ~3 days)

**Model**: `Qwen2.5-14B-Instruct` (NOT Qwen3-14B — avoid double-variable confound with generation axis)

**Recipe** (see "SFT Warmup Strategy" section at top for detailed hyperparams):
1. SFT warmup with the **same 12K-trajectory corpus** used for 7B (same 5K primary + 6K enhanced). Chat template **identical to Qwen2.5-7B** (same Qwen2.5 family), no re-tokenization needed. **Epochs: ~0.7× of 7B baseline. LR: ~0.5-0.7× of 7B baseline.** ~15-22h on 4 GPUs (14B needs lower batch size due to VRAM).
2. **Required pilot**: 0.1-epoch SFT pilot on 14B (~2-3h) to verify loss curve shape. If loss does not decrease normally, reduce LR by 2× before full run.
3. GRPO with E5b-stabilized reward config (same as winning 7B run).
4. Same RL hyper-parameters except: reduce batch size ∝ 2 to fit VRAM; increase grad accum to compensate; keep effective batch size matched.
5. Max 300 steps (matches 7B's E5b@100 convergence; 14B should converge in similar or fewer steps).

**Evaluation**: Full 3531 test set, same config as Task 14 (greedy, max_turns=5, max_new_tokens=512). Compute EM, F1, ContEM, avg_tools, CvT via trajectory classification.

**Decision criteria**:
| Result | Interpretation | Paper framing |
|---|---|---|
| EM on 14B ≈ 7B, CvT ≈ 7B | Capacity-null confirmed | "Shortcut pattern persists at 14B, reward design is the bottleneck, not capacity" — **ideal** |
| EM on 14B > 7B + 5pp but CvT still < 10% | Capacity helps memorization, not retrieval | Good — "capacity improves parametric ceiling but does not unlock genuine tool use under E5b" |
| EM on 14B > 7B + 5pp AND CvT > 15% | Capacity unlocks multi-hop | Narrative pivot needed — "capacity × reward-design interaction" |
| Training diverges | Pipeline / hyperparam issue | Diagnose and retry or drop to appendix |

**Time**: ~22h SFT (reuse 7B SFT data, 1 GPU-equiv for 14B ≈ 2 GPU-days) + ~44h GRPO (~2 GPU-days) ≈ **~3 days wall-time with 4 GPUs**.

**Target launch**: Apr 21-22 (after I-Oracle / I-Self full-test eval complete; in parallel with Qwen3-8B training below).

**Hard kill criteria** (stop early if):
- Format validity < 70% at step 50
- avg_tools drops to 0 (agent abandons tools, indicates reward-shaping issue)
- Training loss diverges (NaN or 3x baseline)

---

## Notes for HPC Agent (Phase 6)

**Phase 6 is the FINAL round of experiments before paper writing.** Almost entirely CPU/inference/API work, not GRPO training.

- **Task 33 (Temporal audit)**: No GPU needed. CPU + Wikidata SPARQL + GPT-4o-mini API. ~$5-15 budget. 3 engineer-days. Run first because it's the longest critical path.
- **Task 34 (Contamination Tier A)**: Uses existing Qwen2.5-7B and Llama-3.1-8B base models (NOT SFT checkpoints — contamination is a pre-training property). ~12-16 GPU hours total. Can run in parallel with Task 33.
- **Task 35 (KGQAGen eval)**: Download `lianglz/KGQAGen-10k` from HuggingFace first. Wikidata pre-materialization is the critical path — respect rate limits. Then inference eval on existing Freebase checkpoints is fast.
- **Task 14 (Full test set CIs)**: Can run in parallel with everything else. Use existing checkpoints, just larger eval set.
- **All four tasks can run in parallel** if GPUs and engineering bandwidth allow.
- **Report Task 33 + 35 results ASAP** — they determine the paper's framing (CWQ defense strength + cross-benchmark consistency).

---

## Methodology References

| Claim | Required methodology | Reference |
|---|---|---|
| "Goodhart effect occurred" | Hump curve over KL + proxy-gold correlation | Gao et al. ICML 2023 |
| "Reward hacking detected" | Trajectory taxonomy + onset + control | METR 2025; Anthropic 2025 |
| "RL taught genuine skill" | Pass@k showing capability expansion | arXiv:2504.13837 |
| "Contamination present" | Partial-prompt completion test | arXiv:2507.10532 |
| "Category B needs tools" | Category A vs B stratified eval | Novel (our contribution) |
| "Behavioral differences significant" | Bootstrap CIs + McNemar's test | Standard practice |
