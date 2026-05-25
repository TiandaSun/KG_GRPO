# S0.4 — Per-component reward decomposition (I-Self GRPO)

Reward type: `tool_type_bonus_retrieval_contrib` (Variant I-Self). Total = `0.25*r_outcome + 0.50*r_tool_type + 0.25*r_retrieval_contrib`.

Trajectories at steps 200 and 250 were re-scored end-to-end. Steps 300 and 400 only have per-sample summary records (no `full_response`), so the tool-side components are reported as `n/a` and `r_total` is reported as a lower bound.

| step | n | r_outcome | r_tool_type | r_retrieval_contrib | r_total | em | f1 | num_tool_calls |
|---|---|---|---|---|---|---|---|---|
| 200 | 3531 | 0.4051 | 0.4845 | 0.0883 | 0.3656 | 0.3900 | 0.4202 | 3.00 |
| 250 | 3531 | 0.4099 | 0.4909 | 0.1040 | 0.3739 | 0.3945 | 0.4252 | 3.00 |
| 300 | 1400 | 0.0006 | n/a | n/a | ≥0.0002 | 0.0000 | 0.0012 | 1.00 |
| 400 | 2000 | 0.0082 | n/a | n/a | ≥0.0021 | 0.0000 | 0.0164 | 1.00 |

## Narrative

- **r_outcome (answer EM/F1 blend)** dropped from ~0.407 (steps 200–250) to ~0.004 (steps 300–400). This is the dominant collapse signal: the policy stopped getting answers right at all (EM = 0).
- **num_tool_calls** collapsed from ~3.00 to ~1.00, i.e. the model emits a single tool call (or none) before answering. Because r_tool_type is *averaged over tool calls*, the per-call mean can stay non-zero even as the policy degenerates — this is exactly the Goodhart pathology the I-Self reward exposes.
- **r_tool_type / r_retrieval_contrib** at steps 300/400 cannot be recomputed offline because `full_response` was not saved for those evaluations. They are reported as `n/a`.
- At pre-collapse (steps 200–250), r_tool_type ≈ 0.488 and r_retrieval_contrib ≈ 0.096. These are the values the policy was steering toward; the answer-side reward (r_outcome) is the channel that decoupled and crashed.

**Conclusion.** The collapse is driven by `r_outcome` going to 0 
(zero EM, near-zero F1) while the policy adopts a 1-tool-call 
shortcut. The 0.25 weight on r_outcome (vs 0.50 on the per-call 
tool-type bonus) is too small to discourage the shortcut, and the 
retrieval-contribution component fails to compensate because it is 
also normalised by total calls — both step-side components are 
blind to the *number* of tool calls used.
