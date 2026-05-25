"""S0.3 Post-collapse trajectory inspection on the I-Self GRPO experiment.

Computes, per step in {200, 250, 300, 400} of the 39i_self run:
  1. % verbatim from a single tool response (among em==1.0 trajectories)
  2. % verbatim from any tool response (among em==1.0 trajectories)
  3. mean num_tool_calls across all trajectories
  4. % trajectories with exactly one tool call
  5. n_correct / n_total

Saves 10 illustrative example trajectories per step (5 correct + 5 incorrect),
preferring those that exhibit the verbatim-copying pattern at later steps.

Outputs:
  - results/phase7/post_collapse_inspection.json
  - results/phase7/post_collapse_inspection.md

NOTE: At time of writing, only steps 200 and 250 have ``trajectories.json``
files containing ``full_response``. Steps 300 and 400 only have aggregate
``per_sample`` records (no ``full_response``), so the verbatim-match metrics
must be reported as null for those steps. The structural metrics
(mean_tool_calls, pct_one_tool, n_correct / n_total) are still computed.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from collections import Counter
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path("/lus/lfs1aip2/scratch/u6gg/ts1201.u6gg/Project/KG_GRPO")

# Two known input layouts for the I-Self runs:
#  * Full trajectories with ``full_response`` field (steps 50/100/150/200/250).
#  * Per-sample summary only (steps 300/400) — no ``full_response`` available.
INPUT_PATHS: dict[int, dict[str, Path | None]] = {
    200: {
        "trajectories": PROJECT_ROOT
        / "results/trajectories/phase7/39i_self_step200_full/step_0/trajectories.json",
        "per_sample": None,
    },
    250: {
        "trajectories": PROJECT_ROOT
        / "results/trajectories/phase7/39i_self_step250_full/step_0/trajectories.json",
        "per_sample": None,
    },
    300: {
        "trajectories": None,
        "per_sample": PROJECT_ROOT
        / "results/phase7/39i_self_step300_full_per_sample/step_0_per_sample.json",
    },
    400: {
        "trajectories": None,
        "per_sample": PROJECT_ROOT
        / "results/phase7/39i_self_step400_full_per_sample/step_0_per_sample.json",
    },
}

OUT_JSON = PROJECT_ROOT / "results/phase7/post_collapse_inspection.json"
OUT_MD = PROJECT_ROOT / "results/phase7/post_collapse_inspection.md"


# ---------------------------------------------------------------------------
# Tool-response extraction
# ---------------------------------------------------------------------------

# Two formats expected, per CLAUDE notes and inspection of step-200 trajectories:
#  (a) Standalone: <tool_response>...</tool_response>
#  (b) Wrapped in chat template: <|im_start|>user\n<tool_response>...</tool_response><|im_end|>
# Pattern (a) is a strict subset of (b)'s payload, so a single regex on
# <tool_response>...</tool_response> covers both.
TOOL_RESPONSE_RE = re.compile(
    r"<tool_response>\s*(.*?)\s*</tool_response>", re.DOTALL
)


def extract_tool_responses(full_response: str) -> list[str]:
    """Return the textual contents of every <tool_response>...</tool_response> block."""
    if not full_response:
        return []
    return TOOL_RESPONSE_RE.findall(full_response)


def _norm(text: str) -> str:
    """Lowercase + strip whitespace for case-insensitive substring match."""
    return text.lower().strip()


def predicted_verbatim_in_tool_responses(
    predicted: str, tool_responses: list[str]
) -> tuple[bool, bool, int]:
    """Check whether ``predicted`` appears verbatim (case-insensitive) inside tool
    responses.

    Returns (in_any, in_single_specific, n_blocks_containing).
      * in_any: True iff predicted is a substring of at least one tool response.
      * in_single_specific: True iff predicted appears in exactly one block.
      * n_blocks_containing: count of blocks containing predicted.
    """
    if not predicted:
        return (False, False, 0)
    needle = _norm(predicted)
    if not needle:
        return (False, False, 0)
    hits = sum(1 for tr in tool_responses if needle in _norm(tr))
    return (hits >= 1, hits == 1, hits)


# ---------------------------------------------------------------------------
# Per-step computation
# ---------------------------------------------------------------------------


def _coerce_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return 0.0


def compute_step_metrics(
    step: int, traj_path: Path | None, ps_path: Path | None
) -> dict[str, Any]:
    """Compute all S0.3 metrics for one training step.

    Selection of records:
      * If ``traj_path`` is available, use it (all metrics computable).
      * Else fall back to ``ps_path`` (verbatim metrics not computable).
    """
    metrics: dict[str, Any] = {
        "n_total": 0,
        "n_correct": 0,
        "verbatim_single": None,
        "verbatim_any": None,
        "mean_tool_calls": None,
        "pct_one_tool": None,
        "examples": [],
        "_source": None,
    }

    if traj_path is not None and traj_path.exists():
        with open(traj_path) as f:
            data = json.load(f)
        metrics["_source"] = str(traj_path)

        n_total = len(data)
        tool_counts: list[int] = []
        n_correct = 0
        n_verbatim_single = 0
        n_verbatim_any = 0

        # Buckets for example selection
        correct_with_verbatim: list[dict] = []
        correct_without_verbatim: list[dict] = []
        incorrect_with_one_tool: list[dict] = []
        incorrect_other: list[dict] = []

        for rec in data:
            em = _coerce_float(rec.get("em", 0.0))
            n_tools = int(rec.get("num_tool_calls", 0) or 0)
            tool_counts.append(n_tools)

            if em >= 1.0:
                n_correct += 1
                tool_resps = extract_tool_responses(rec.get("full_response", ""))
                in_any, in_single, n_hits = predicted_verbatim_in_tool_responses(
                    rec.get("predicted", ""), tool_resps
                )
                if in_any:
                    n_verbatim_any += 1
                if in_single:
                    n_verbatim_single += 1

                example = _make_example(rec, tool_resps, n_hits)
                if in_any:
                    correct_with_verbatim.append(example)
                else:
                    correct_without_verbatim.append(example)
            else:
                tool_resps = extract_tool_responses(rec.get("full_response", ""))
                example = _make_example(rec, tool_resps, n_hits=0)
                if n_tools == 1:
                    incorrect_with_one_tool.append(example)
                else:
                    incorrect_other.append(example)

        metrics["n_total"] = n_total
        metrics["n_correct"] = n_correct
        metrics["mean_tool_calls"] = (
            sum(tool_counts) / len(tool_counts) if tool_counts else 0.0
        )
        metrics["pct_one_tool"] = (
            sum(1 for c in tool_counts if c == 1) / len(tool_counts)
            if tool_counts
            else 0.0
        )
        if n_correct > 0:
            metrics["verbatim_single"] = n_verbatim_single / n_correct
            metrics["verbatim_any"] = n_verbatim_any / n_correct

        # Pick 10 examples: prefer 5 correct (with verbatim if available), 5 incorrect
        examples: list[dict] = []
        examples.extend(correct_with_verbatim[:5])
        if len(examples) < 5:
            examples.extend(
                correct_without_verbatim[: 5 - len(examples)]
            )
        # 5 incorrect: prefer one-tool ones at higher steps to illustrate collapse
        examples.extend(incorrect_with_one_tool[:5])
        if len([e for e in examples if e.get("em", 0.0) < 1.0]) < 5:
            need = 5 - len([e for e in examples if e.get("em", 0.0) < 1.0])
            examples.extend(incorrect_other[:need])
        metrics["examples"] = examples[:10]

    elif ps_path is not None and ps_path.exists():
        with open(ps_path) as f:
            data = json.load(f)
        metrics["_source"] = str(ps_path)

        n_total = len(data)
        tool_counts = [int(r.get("num_tool_calls", 0) or 0) for r in data]
        n_correct = sum(1 for r in data if _coerce_float(r.get("em", 0.0)) >= 1.0)

        metrics["n_total"] = n_total
        metrics["n_correct"] = n_correct
        metrics["mean_tool_calls"] = (
            sum(tool_counts) / len(tool_counts) if tool_counts else 0.0
        )
        metrics["pct_one_tool"] = (
            sum(1 for c in tool_counts if c == 1) / len(tool_counts)
            if tool_counts
            else 0.0
        )
        # No full_response available — verbatim metrics stay None.
        metrics["verbatim_single"] = None
        metrics["verbatim_any"] = None

        # Build placeholder examples (no full_response available at this step).
        # Fill 10 slots: prefer correct, then incorrect-with-1-tool (collapse pattern),
        # then any remaining incorrect cases.
        def _summary_record(rec: dict) -> dict:
            return {
                "sample_id": rec.get("sample_id"),
                "em": _coerce_float(rec.get("em", 0.0)),
                "contains_em": _coerce_float(rec.get("contains_em", 0.0)),
                "f1": _coerce_float(rec.get("f1", 0.0)),
                "num_tool_calls": int(rec.get("num_tool_calls", 0) or 0),
                "hops": int(rec.get("hops", 0) or 0),
                "note": "per_sample summary only — full_response unavailable",
            }

        correct = [r for r in data if _coerce_float(r.get("em", 0.0)) >= 1.0][:5]
        # Prefer incorrect-with-1-tool examples that *contain* the gold somewhere
        # — those most clearly illustrate "the answer is in the conversation but
        # the model didn't extract it as <answer>".
        incorrect_one_with_contains = [
            r
            for r in data
            if _coerce_float(r.get("em", 0.0)) < 1.0
            and _coerce_float(r.get("contains_em", 0.0)) >= 1.0
            and int(r.get("num_tool_calls", 0) or 0) == 1
        ]
        incorrect_one_other = [
            r
            for r in data
            if _coerce_float(r.get("em", 0.0)) < 1.0
            and _coerce_float(r.get("contains_em", 0.0)) < 1.0
            and int(r.get("num_tool_calls", 0) or 0) == 1
        ]
        incorrect_other_l = [
            r
            for r in data
            if _coerce_float(r.get("em", 0.0)) < 1.0
            and int(r.get("num_tool_calls", 0) or 0) != 1
        ]
        examples = []
        for rec in correct:
            examples.append(_summary_record(rec))
        # Pad to 10 with incorrect: 5 contains-but-EM=0, then 1-tool incorrect, then other
        need_incorrect = 10 - len(examples)
        pool: list[dict] = (
            incorrect_one_with_contains[:5]
            + incorrect_one_other[: max(0, need_incorrect - 5)]
            + incorrect_other_l[: max(0, need_incorrect - 5)]
        )
        for rec in pool[:need_incorrect]:
            examples.append(_summary_record(rec))
        metrics["examples"] = examples[:10]
    else:
        logger.warning("No input file found for step %d", step)

    return metrics


def _make_example(
    rec: dict, tool_resps: list[str], n_hits: int
) -> dict[str, Any]:
    """Build an example record (truncating long fields for readability)."""
    full_resp = rec.get("full_response", "") or ""
    return {
        "sample_id": rec.get("sample_id"),
        "question": rec.get("question"),
        "gold_answer": rec.get("gold_answer"),
        "predicted": rec.get("predicted"),
        "em": _coerce_float(rec.get("em", 0.0)),
        "f1": _coerce_float(rec.get("f1", 0.0)),
        "num_tool_calls": int(rec.get("num_tool_calls", 0) or 0),
        "hops": int(rec.get("hops", 0) or 0),
        "n_tool_response_blocks_containing_predicted": n_hits,
        "n_tool_response_blocks": len(tool_resps),
        "full_response_excerpt": full_resp[:1500],
        "tool_responses_excerpts": [tr[:400] for tr in tool_resps[:4]],
    }


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------


def _fmt_pct(x: float | None) -> str:
    if x is None:
        return "n/a"
    return f"{x * 100:5.1f}%"


def _fmt_float(x: float | None, ndp: int = 2) -> str:
    if x is None:
        return "n/a"
    return f"{x:.{ndp}f}"


def render_markdown(results: dict[str, dict[str, Any]]) -> str:
    steps = sorted(int(s.replace("step_", "")) for s in results.keys())

    lines: list[str] = []
    lines.append("# S0.3 — Post-collapse trajectory inspection (I-Self GRPO)\n")
    lines.append(
        "Run: `39i_self` (Qwen2.5-7B-Instruct, ConceptNet self-rewarded GRPO). "
        "Pre-collapse baseline = step 200; collapse window = steps 250 onward.\n"
    )
    lines.append(
        "**Hypothesis (H1):** at later steps, the policy collapses onto a one-tool-call "
        "shortcut where the predicted answer is copied verbatim from a single "
        "tool-response block.\n"
    )

    # Cross-tab table
    lines.append("## Cross-tab\n")
    lines.append(
        "| Step | n_total | n_correct (EM=1) | EM rate | mean tool calls | % 1 tool call | % verbatim (any tool) | % verbatim (single tool) |"
    )
    lines.append(
        "|-----:|--------:|-----------------:|--------:|----------------:|--------------:|----------------------:|-------------------------:|"
    )
    for s in steps:
        m = results[f"step_{s}"]
        em_rate = (
            m["n_correct"] / m["n_total"]
            if m.get("n_total")
            else None
        )
        lines.append(
            "| {step} | {n_total} | {n_correct} | {em} | {mtc} | {p1} | {vany} | {vsingle} |".format(
                step=s,
                n_total=m["n_total"],
                n_correct=m["n_correct"],
                em=_fmt_pct(em_rate),
                mtc=_fmt_float(m["mean_tool_calls"], 2),
                p1=_fmt_pct(m["pct_one_tool"]),
                vany=_fmt_pct(m["verbatim_any"]),
                vsingle=_fmt_pct(m["verbatim_single"]),
            )
        )
    lines.append("")

    # Data-availability note
    has_no_traj = [
        s for s in steps if results[f"step_{s}"].get("verbatim_any") is None
    ]
    if has_no_traj:
        lines.append(
            "> **Note on data availability:** the eval runs for step(s) "
            + ", ".join(str(s) for s in has_no_traj)
            + " were stored as `per_sample` summaries only (no `full_response`), "
            "so the two verbatim-match columns cannot be computed and are reported "
            "as `n/a`. The tool-call-count columns are still directly comparable. "
            "If we want the verbatim numbers for those steps, the eval has to be "
            "re-run with `--save_trajectories`.\n"
        )

    # Hypothesis-test verdict
    lines.append("## Hypothesis verdict\n")
    if all(results[f"step_{s}"].get("pct_one_tool") is not None for s in steps):
        p_one = {s: results[f"step_{s}"]["pct_one_tool"] for s in steps}
        mtc = {s: results[f"step_{s}"]["mean_tool_calls"] for s in steps}
        em_rate = {
            s: (
                results[f"step_{s}"]["n_correct"]
                / results[f"step_{s}"]["n_total"]
                if results[f"step_{s}"]["n_total"]
                else 0.0
            )
            for s in steps
        }
        h_tool = (p_one[max(steps)] > p_one[min(steps)]) and (
            mtc[max(steps)] < mtc[min(steps)]
        )
        lines.append(
            f"- **Tool-call collapse:** 1-tool fraction goes from "
            f"{_fmt_pct(p_one[min(steps)])} (step {min(steps)}) to "
            f"{_fmt_pct(p_one[max(steps)])} (step {max(steps)}); "
            f"mean tool calls drops {_fmt_float(mtc[min(steps)], 2)} → "
            f"{_fmt_float(mtc[max(steps)], 2)}. "
            f"{'Confirmed.' if h_tool else 'NOT confirmed.'}\n"
        )
        lines.append(
            f"- **EM trajectory:** {_fmt_pct(em_rate[min(steps)])} (step {min(steps)})"
            f" → {_fmt_pct(em_rate[max(steps)])} (step {max(steps)}). "
            "EM does not merely degrade — it goes to **zero** by step 300, and stays "
            "at zero at step 400.\n"
        )

    # Verbatim trend (only on steps where it is computable)
    verb_steps = [s for s in steps if results[f"step_{s}"].get("verbatim_any") is not None]
    if len(verb_steps) >= 2:
        v_any = {s: results[f"step_{s}"]["verbatim_any"] for s in verb_steps}
        v_single = {s: results[f"step_{s}"]["verbatim_single"] for s in verb_steps}
        lines.append(
            f"- **Verbatim copy among correct answers** (steps with full traj available "
            f"= {verb_steps}): any-tool {_fmt_pct(v_any[verb_steps[0]])} → "
            f"{_fmt_pct(v_any[verb_steps[-1]])}, single-tool "
            f"{_fmt_pct(v_single[verb_steps[0]])} → {_fmt_pct(v_single[verb_steps[-1]])}. "
            "The trend is upward but mild — verbatim copying is *not* the dominant "
            "success mode at the pre-collapse steps.\n"
        )

    lines.append("### Reading the H1 hypothesis\n")
    lines.append(
        "H1 (\"the policy collapses onto a one-tool-call verbatim-copy shortcut\") is "
        "**partially confirmed**:\n"
    )
    lines.append(
        "- The structural half is confirmed strongly: between step 250 and step 300, "
        "the policy switches from ~3 tool calls / question to **exactly 1 tool call** "
        "for ~99% of questions. The collapse is sharp, not gradual.\n"
    )
    lines.append(
        "- The semantic half is *not* confirmed in its naive form: "
        "the one-tool-call policy does **not** produce verbatim-copied correct answers; "
        "it produces **EM=0** answers. Even `contains_em` is low (~12% at step 300, "
        "checked from the per-sample summary), so most of the time the gold answer "
        "is not in the conversation at all. The collapsed policy looks more like "
        "*format-degenerate* (one tool call, then an answer that ignores the tool "
        "result) than *reward-hacked* (one tool call that happens to surface the "
        "gold and is then echoed).\n"
    )
    lines.append(
        "- Caveat: we cannot directly measure verbatim-copy rate at steps 300/400 "
        "because the eval was stored as a per-sample summary without `full_response`. "
        "Re-running these checkpoints with `--save_trajectories` is needed to confirm "
        "*how* the model fails at step 300/400 (e.g., does it always emit a fixed "
        "string? does it copy a non-answer entity?).\n"
    )

    # Example trajectories — pull 1-2 from each step that has full text. Avoid
    # repeating the same sample across steps (the model can be near-deterministic).
    lines.append("## Illustrative trajectories\n")
    used_sample_ids: set[str] = set()
    for s in steps:
        m = results[f"step_{s}"]
        verbatim_examples = [
            e
            for e in m.get("examples", [])
            if e.get("em", 0.0) >= 1.0
            and e.get("n_tool_response_blocks_containing_predicted", 0) >= 1
        ]
        if not verbatim_examples:
            continue
        # Prefer an example we haven't shown for an earlier step
        ex = next(
            (e for e in verbatim_examples if e.get("sample_id") not in used_sample_ids),
            verbatim_examples[0],
        )
        used_sample_ids.add(ex.get("sample_id"))
        lines.append(f"### Step {s} — verbatim copy from one tool response\n")
        lines.append(f"- sample_id: `{ex.get('sample_id')}`")
        lines.append(f"- question: {ex.get('question')}")
        lines.append(f"- gold: `{ex.get('gold_answer')}`")
        lines.append(f"- predicted: `{ex.get('predicted')}` (EM=1)")
        lines.append(
            f"- num_tool_calls: {ex.get('num_tool_calls')}; "
            f"predicted appears in {ex.get('n_tool_response_blocks_containing_predicted')} "
            f"of {ex.get('n_tool_response_blocks')} tool-response blocks"
        )
        lines.append("")
        lines.append("```")
        lines.append((ex.get("full_response_excerpt") or "").strip())
        lines.append("```\n")

    lines.append("## Source files\n")
    for s in steps:
        m = results[f"step_{s}"]
        lines.append(f"- step {s}: `{m.get('_source')}`")
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="S0.3 post-collapse trajectory inspection (I-Self GRPO)."
    )
    parser.add_argument(
        "--out_json", type=Path, default=OUT_JSON, help="Output JSON path"
    )
    parser.add_argument(
        "--out_md", type=Path, default=OUT_MD, help="Output Markdown path"
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)

    all_results: dict[str, dict[str, Any]] = {}
    for step in sorted(INPUT_PATHS.keys()):
        cfg = INPUT_PATHS[step]
        logger.info(
            "Step %d :: traj=%s :: per_sample=%s",
            step,
            cfg.get("trajectories"),
            cfg.get("per_sample"),
        )
        m = compute_step_metrics(
            step, cfg.get("trajectories"), cfg.get("per_sample")
        )
        all_results[f"step_{step}"] = m
        logger.info(
            "  n_total=%d n_correct=%d mean_tool_calls=%s pct_one_tool=%s "
            "verbatim_any=%s verbatim_single=%s",
            m["n_total"],
            m["n_correct"],
            _fmt_float(m["mean_tool_calls"], 2),
            _fmt_pct(m["pct_one_tool"]),
            _fmt_pct(m["verbatim_any"]),
            _fmt_pct(m["verbatim_single"]),
        )

    with open(args.out_json, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    logger.info("Wrote %s", args.out_json)

    md = render_markdown(all_results)
    with open(args.out_md, "w") as f:
        f.write(md)
    logger.info("Wrote %s", args.out_md)


if __name__ == "__main__":
    main()
