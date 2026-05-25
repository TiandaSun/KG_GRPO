"""W17.1 format-validity analyser for v17 Llama-3.1-8B sanity gate.

Reads a trajectories.json saved by `eval_with_tools.py --save_trajectories`
and reports the fraction of trajectories that close with a parseable
`<answer>...</answer>` tag — the v17 W17.1 gate criterion.

Decision rule (v17 spec):
  - format_valid >= 0.90 AND tools_per_q > 0.5 -> PASS  (proceed to W17.2)
  - else                                       -> ABORT (stop here)

Usage:
    python scripts/v17_format_valid.py \
        --trajectories results/v17_w17_1_traj/step_0/trajectories.json \
        --output _handoff/data/v17_llama/W17_1_GATE_REPORT.md
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("v17_format_valid")


def parse_format_validity(traj: dict) -> tuple[bool, bool, str]:
    """Return (has_open_tag, has_close_tag, extracted_answer)."""
    text = traj.get("full_response", "")
    has_open = "<answer>" in text
    has_close = "</answer>" in text
    extracted = ""
    if has_open and has_close:
        try:
            extracted = text.split("<answer>", 1)[1].split("</answer>", 1)[0].strip()
        except Exception:
            extracted = ""
    return has_open, has_close, extracted


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectories", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True,
                        help="Markdown report destination")
    parser.add_argument("--gate_format_valid", type=float, default=0.90)
    parser.add_argument("--gate_tools_per_q", type=float, default=0.5)
    args = parser.parse_args()

    if not args.trajectories.exists():
        logger.error("Trajectories file not found: %s", args.trajectories)
        return 1

    with open(args.trajectories) as f:
        trajectories = json.load(f)

    n = len(trajectories)
    if n == 0:
        logger.error("Empty trajectories file")
        return 1

    n_open = n_close = n_both = n_nonempty = 0
    n_tools = 0
    em_sum = 0.0
    f1_sum = 0.0
    examples: list[str] = []

    for t in trajectories:
        has_open, has_close, extracted = parse_format_validity(t)
        if has_open:
            n_open += 1
        if has_close:
            n_close += 1
        if has_open and has_close:
            n_both += 1
            if extracted:
                n_nonempty += 1
        n_tools += int(t.get("num_tool_calls", 0))
        em_sum += float(t.get("em", 0))
        f1_sum += float(t.get("f1", 0))

        if len(examples) < 3 and not (has_open and has_close):
            tail = t.get("full_response", "")[-500:]
            examples.append(
                f"sample_id={t.get('sample_id')} | gold={t.get('gold_answer')} | "
                f"open={has_open} close={has_close}\n  ...{tail!r}"
            )

    format_valid = n_both / n
    format_nonempty = n_nonempty / n
    tools_per_q = n_tools / n
    em = em_sum / n
    f1 = f1_sum / n

    pass_format = format_valid >= args.gate_format_valid
    pass_tools = tools_per_q > args.gate_tools_per_q
    verdict = "PASS" if (pass_format and pass_tools) else "ABORT"

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        f.write(f"# v17 W17.1 format-emission gate report\n\n")
        f.write(f"- Trajectories analysed: **{n}**\n")
        f.write(f"- `<answer>` open tag emitted: {n_open}/{n} ({n_open/n:.1%})\n")
        f.write(f"- `</answer>` close tag emitted: {n_close}/{n} ({n_close/n:.1%})\n")
        f.write(f"- Both tags present (format-valid): **{n_both}/{n} ({format_valid:.1%})**\n")
        f.write(f"- Both tags + non-empty answer: {n_nonempty}/{n} ({format_nonempty:.1%})\n")
        f.write(f"- Mean tool calls / question: **{tools_per_q:.2f}**\n")
        f.write(f"- Strict EM (sanity, not load-bearing): {em:.4f}\n")
        f.write(f"- Token-F1 (sanity): {f1:.4f}\n\n")
        f.write(f"## Gate decision\n\n")
        f.write(f"- format_valid >= {args.gate_format_valid:.0%}: "
                f"{'PASS' if pass_format else 'FAIL'} ({format_valid:.1%})\n")
        f.write(f"- tools_per_q > {args.gate_tools_per_q}: "
                f"{'PASS' if pass_tools else 'FAIL'} ({tools_per_q:.2f})\n")
        f.write(f"\n**Verdict: {verdict}**\n\n")
        if verdict == "ABORT":
            f.write(
                "Per v17 spec W17.1: if format-valid < 90%, STOP — do NOT proceed "
                "to W17.2. Log findings; rebuttal-only ammunition.\n\n"
            )
            f.write("## Sample failure modes (first 3)\n\n")
            for ex in examples:
                f.write(f"```\n{ex}\n```\n\n")
        else:
            f.write("Per v17 spec W17.1: gate PASSED — proceed to W17.2 full SFT.\n")

    logger.info("Format-valid: %d/%d = %.1f%%", n_both, n, 100 * format_valid)
    logger.info("Tools/Q: %.2f", tools_per_q)
    logger.info("Verdict: %s", verdict)
    logger.info("Report written to %s", args.output)

    # Exit code 0 always (gate-only signal goes to the markdown). The orchestrating
    # job script reads the verdict line and acts accordingly.
    return 0


if __name__ == "__main__":
    sys.exit(main())
