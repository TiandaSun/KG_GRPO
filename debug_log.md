# Tool Call Debug Log

## Background
GRPO 7B runs on 2+ GPUs with FSDP + vLLM sleep mode (established in prior sessions).
Key infra fixes already applied: gpu_memory_utilization=0.3-0.5, bucket_megabytes=4096, enable_sleep_mode=true, min 2 GPUs, default_agent_loop=tool_agent, kg_server port=18901, parser accepts aliases.

Despite all infra fixes, the model still produces zero tool calls (num_tool_calls=0). The SFT model does not generate `<search>` tags at all.

---

## Attempt 1 — 2026-03-20

**Job ID**: 3244443
**Fix**: `_VALID_ACTIONS` was referenced before definition in `src_verl/interaction/search_tool_parser.py` (line 41 used `_VALID_ACTIONS`, defined on line 49). This caused `NameError` at import time, meaning the `kg_search` ToolParser never registered. Moved `_VALID_ACTIONS` definition above its first use.

**Root cause**: The `@ToolParser.register("kg_search")` decorator on `KGSearchToolParser` never executed because importing the module crashed silently. verl fell back to no parser, so no tool calls were ever extracted from model output.

**Validation**: After fix, `python -c "from src_verl.interaction.search_tool_parser import KGSearchToolParser"` succeeds and `kg_search` appears in `ToolParser._registry`.

**Metrics**:
- `num_tool_calls/mean@1`: **3.3125** (was 0)
- `num_turns/mean`: 8.5 (min 2, max 16)
- `r_tool_use/mean@1`: 0.846
- `r_no_tool/mean@1`: 0.0 (all samples used tools)
- `r_answer/mean@1`: 0.539
- `r_coverage/mean@1`: 0.379
- `r_format/mean@1`: 0.469
- Exit code: 0, runtime: 131s (~21s/step)
- KG server received many `/retrieve` POST 200 responses

**Parser debug**: Model generates both `<search>get_tail_relations(...)` and bare `<get_tail_relations(...)>` formats — both caught by the parser.

**Outcome**: **SUCCESS** — tool calls working. Issue resolved in 1 attempt.

---

