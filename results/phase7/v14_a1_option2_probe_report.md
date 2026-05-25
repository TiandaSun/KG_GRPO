# V14-A1 Search-R1 Option 2 probe (2026-04-20)

## Baseline
- vllm: 0.12.0
- transformers: 4.57.1
- Patched file: `external/Search-R1/verl/third_party/vllm/__init__.py`

## Probe results
- Probe 1 (import version wrapper): FAIL
- Probe 2 (import fsdp_workers):    FAIL
- Probe 3 (construct tiny SR1 LLM): PASS

## VERDICT: OPTION 2 NOT VIABLE

One or more critical imports fail — the 0.6.3 wrapper is too tightly coupled
to vllm ≤0.6.3 internals. Recommendation: fall back to **Option 3** —
reframe paper using our existing E1 (outcome-only reward) as the
Search-R1-equivalent baseline. Zero new compute, stronger paper claim:
"Search-R1's outcome-only recipe (as reimplemented via our E1) collapses
to EM≈0 on CWQ/Freebase, vs 35-45% on HotpotQA/NQ, showing the recipe
doesn't transfer to compositional KG QA."

## Undo instructions
Revert the patch:
```
cd external/Search-R1
git checkout verl/third_party/vllm/__init__.py
```
This does NOT affect kg_verl env — no packages were installed.
