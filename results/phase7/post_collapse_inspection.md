# S0.3 — Post-collapse trajectory inspection (I-Self GRPO)

Run: `39i_self` (Qwen2.5-7B-Instruct, ConceptNet self-rewarded GRPO). Pre-collapse baseline = step 200; collapse window = steps 250 onward.

**Hypothesis (H1):** at later steps, the policy collapses onto a one-tool-call shortcut where the predicted answer is copied verbatim from a single tool-response block.

## Cross-tab

| Step | n_total | n_correct (EM=1) | EM rate | mean tool calls | % 1 tool call | % verbatim (any tool) | % verbatim (single tool) |
|-----:|--------:|-----------------:|--------:|----------------:|--------------:|----------------------:|-------------------------:|
| 200 | 3531 | 1378 |  39.0% | 3.00 |   0.0% |  20.6% |   8.1% |
| 250 | 3531 | 1395 |  39.5% | 3.00 |   0.1% |  23.7% |   9.7% |
| 300 | 1400 | 0 |   0.0% | 1.00 |  99.9% | n/a | n/a |
| 400 | 2000 | 0 |   0.0% | 1.00 |  98.4% | n/a | n/a |

> **Note on data availability:** the eval runs for step(s) 300, 400 were stored as `per_sample` summaries only (no `full_response`), so the two verbatim-match columns cannot be computed and are reported as `n/a`. The tool-call-count columns are still directly comparable. If we want the verbatim numbers for those steps, the eval has to be re-run with `--save_trajectories`.

## Hypothesis verdict

- **Tool-call collapse:** 1-tool fraction goes from   0.0% (step 200) to  98.4% (step 400); mean tool calls drops 3.00 → 1.00. Confirmed.

- **EM trajectory:**  39.0% (step 200) →   0.0% (step 400). EM does not merely degrade — it goes to **zero** by step 300, and stays at zero at step 400.

- **Verbatim copy among correct answers** (steps with full traj available = [200, 250]): any-tool  20.6% →  23.7%, single-tool   8.1% →   9.7%. The trend is upward but mild — verbatim copying is *not* the dominant success mode at the pre-collapse steps.

### Reading the H1 hypothesis

H1 ("the policy collapses onto a one-tool-call verbatim-copy shortcut") is **partially confirmed**:

- The structural half is confirmed strongly: between step 250 and step 300, the policy switches from ~3 tool calls / question to **exactly 1 tool call** for ~99% of questions. The collapse is sharp, not gradual.

- The semantic half is *not* confirmed in its naive form: the one-tool-call policy does **not** produce verbatim-copied correct answers; it produces **EM=0** answers. Even `contains_em` is low (~12% at step 300, checked from the per-sample summary), so most of the time the gold answer is not in the conversation at all. The collapsed policy looks more like *format-degenerate* (one tool call, then an answer that ignores the tool result) than *reward-hacked* (one tool call that happens to surface the gold and is then echoed).

- Caveat: we cannot directly measure verbatim-copy rate at steps 300/400 because the eval was stored as a per-sample summary without `full_response`. Re-running these checkpoints with `--save_trajectories` is needed to confirm *how* the model fails at step 300/400 (e.g., does it always emit a fixed string? does it copy a non-answer entity?).

## Illustrative trajectories

### Step 200 — verbatim copy from one tool response

- sample_id: `WebQTest-576_01e2da60a2779c4ae4b5d1547499a4f8`
- question: Which nation has the Alta Verapaz Department and is in Central America?
- gold: `Guatemala`
- predicted: `Guatemala` (EM=1)
- num_tool_calls: 3; predicted appears in 2 of 3 tool-response blocks

```
<think>I need to find information about Central America. Let me start by checking its relations in the knowledge graph.</think>
<search>get_tail_entities(Central America, location.location.contains)</search>
<tool_response>["Belize", "Condominios Panorama Life Torre I", "Costa Rica", "El Salvador", "Gran Colombia", "Guatemala", "Honduras", "La Amistad International Park", "Panama"]</tool_response>
<think>I can see the relation 'location.location.contains'. Let me follow this information to continue reasoning.</think>
<search>get_tail_entities(Central America, location.location.contains)</search>
<tool_response>["Belize", "Condominios Panorama Life Torre I", "Costa Rica", "El Salvador", "Gran Colombia", "Guatemala", "Honduras", "La Amistad International Park", "Panama"]</tool_response>
<think>Now I know about Belize, Condominios Panorama Life Torre I, Costa Rica, El Salvador, Gran Colombia, Guatemala, Honduras, La Amistad International Park, Panama. Let me follow this information to continue reasoning.</think>
<search>get_tail_entities(Central America, location.location.containedby)</search>
<tool_response>["Americas"]</tool_response>
<think>Based on my knowledge graph exploration, I found that the answer is Guatemala.</think>
<answer>Guatemala</answer>
```

### Step 250 — verbatim copy from one tool response

- sample_id: `WebQTest-100_bf66fd89b6e9fc5fcb96c8b3f7a0e616`
- question: The people from the country that contains Nord-Ouest Department speak what languages today?
- gold: `Haitian Creole`
- predicted: `Haitian Creole` (EM=1)
- num_tool_calls: 3; predicted appears in 1 of 3 tool-response blocks

```
<think>I need to find information about Nord-Ouest Department. Let me start by checking its relations in the knowledge graph.</think>
<search>get_tail_entities(Nord-Ouest Department, location.location.containedby)</search>
<tool_response>["Haiti"]</tool_response>
<think>I can see the relation 'location.location.containedby'. Let me follow this information to continue reasoning.</think>
<search>get_tail_entities(Haiti, location.location.containedby)</search>
<tool_response>["Americas", "Greater Antilles", "Latin America", "North America"]</tool_response>
<think>Now I know about Americas. Let me follow this information to continue reasoning.</think>
<search>get_tail_entities(Haiti, location.country.languages_spoken)</search>
<tool_response>["French", "Haitian Creole"]</tool_response>
<think>Based on my knowledge graph exploration, I found that the answer is Haitian Creole.</think>
<answer>Haitian Creole</answer>
```

## Source files

- step 200: `/lus/lfs1aip2/scratch/u6gg/ts1201.u6gg/Project/KG_GRPO/results/trajectories/phase7/39i_self_step200_full/step_0/trajectories.json`
- step 250: `/lus/lfs1aip2/scratch/u6gg/ts1201.u6gg/Project/KG_GRPO/results/trajectories/phase7/39i_self_step250_full/step_0/trajectories.json`
- step 300: `/lus/lfs1aip2/scratch/u6gg/ts1201.u6gg/Project/KG_GRPO/results/phase7/39i_self_step300_full_per_sample/step_0_per_sample.json`
- step 400: `/lus/lfs1aip2/scratch/u6gg/ts1201.u6gg/Project/KG_GRPO/results/phase7/39i_self_step400_full_per_sample/step_0_per_sample.json`
