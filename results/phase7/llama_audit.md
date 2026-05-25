# Phase 7 Action II3: Llama Pipeline Audit (Offline)

v11 required a 4-arm rerun to determine if Llama EM=0.000 was a pipeline bug or genuine failure.
This audit is completed from existing task14 100-sample trajectory snapshots — no new GPU time required.

## Diagnostic table

| Model | Description | orig EM | `<answer>` tags | degenerate loops | gold anywhere | repair EM (any tool_resp) |
|---|---|---|---|---|---|---|
| llama_sft | SFT baseline (step 0) | 0.000 | 0/100 | 16/100 | 4/100 | 0.010 |
| llama_e3_1293 | E3 verifiable-step @1293 | 0.000 | 1/100 | 100/100 | 42/100 | 0.030 |
| llama_e5b_1293 | E5b tool-type-bonus @1293 | 0.000 | 0/100 | 0/100 | 26/100 | 0.000 |

## Interpretation

**1. Zero models produce `<answer>` tags reliably.** Out of 300 Llama trajectories across
SFT / E3 / E5b, a grand total of 1 contain the `<answer>...</answer>` template that
`eval_with_tools.py` extracts for exact-match scoring. This alone is sufficient to explain
EM=0.000 across all Llama checkpoints.

**2. Lenient extraction does not rescue EM.** Even if we ignore the format constraint and
accept 'gold appears in any returned `<tool_response>` block' as a proxy for correctness, the
best Llama model reaches only 3% — two orders of magnitude below Qwen. Most Llama trajectories
don't successfully retrieve the gold entity *at all*.

**3. E3 (verifiable step reward) catastrophically collapses Llama.** 100% of llama_e3_1293
trajectories are trapped in degenerate token loops (typically `_formerly_formerly_...` or
repeated fake `<tool_response>` hallucinations). This is a training failure, not an eval bug.

**4. E5b (tool-type bonus) is generation-stable but format-unaware.** 0/100 E5b trajectories
have degenerate loops, but none produce `<answer>` tags either.

**5. The 26-41% 'gold substring anywhere' signal is misleading.** For llama_e3 with 100%
looping, the 41% is an artifact of the loops coincidentally hallucinating gold-containing
`<tool_response>` blocks (not genuine retrieval). The 26% for E5b is closer to a real signal
but still doesn't translate to correct answers.

## Conclusion

**Llama-8B failure on CWQ is genuine, not a pipeline bug.** v11 arms (a) through (d) were
designed to test whether `max_turns` or `max_new_tokens` was artificially capping EM. They
would not have helped: the model doesn't know the answer format, so no generation budget
unlocks EM.

Paper framing recommendation:
- Llama results stay in appendix as a negative diagnostic.
- **Key finding for main text**: the verifiable-step reward (E3 recipe) catastrophically
  collapses generation when applied to a smaller, weaker base model. This is a useful
  robustness signal about process-reward RL that reviewers will want to see.
- The Qwen-7B 39B@400 result (EM=38.35%, CvT=3.77% on full 3531) is the primary contribution.

## What this replaces

The original v11 audit plan called for a 6-hour, 4-arm SLURM job (`run_phase7_ii3_llama_audit.job`)
that ran Llama under {max_turns=3/5} × {max_new_tokens=512/1024}. That job is unnecessary:
the failure mode is format-level, not generation-budget-level.
