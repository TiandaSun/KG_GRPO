# v17 W17.1 format-emission gate report

- Trajectories analysed: **100**
- `<answer>` open tag emitted: 94/100 (94.0%)
- `</answer>` close tag emitted: 92/100 (92.0%)
- Both tags present (format-valid): **92/100 (92.0%)**
- Both tags + non-empty answer: 92/100 (92.0%)
- Mean tool calls / question: **0.66**
- Strict EM (sanity, not load-bearing): 0.3500
- Token-F1 (sanity): 0.4159

## Gate decision

- format_valid >= 90%: PASS (92.0%)
- tools_per_q > 0.5: PASS (0.66)

**Verdict: PASS**

Per v17 spec W17.1: gate PASSED — proceed to W17.2 full SFT.
