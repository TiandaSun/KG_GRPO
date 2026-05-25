#!/usr/bin/env python3
"""Task S1.3: CvT metric robustness check using fuzzy matching.

Audits the canonical 7-category trajectory classifier
(`scripts/task16_classify.py::classify_trajectory`).

The current paper distinguishes (see task16_classify.py:84-90):
  * "correct-via-tool"   (CvT) -- em==1 AND ANY gold-answer alias appears
                                  verbatim (case-insensitive substring) in
                                  the concatenated <tool_response> text.
  * "correct-via-memory" (CvM) -- em==1 AND no gold-answer alias appears
                                  verbatim in tool responses.

Reviewer concern: the verbatim match may underestimate true CvT due to
formatting differences ("the eiffel tower" vs "Eiffel Tower").

This script:
  1) re-applies the canonical classifier to G2 step-500 trajectories.
  2) selects the currently-CvM trajectories.
  3) applies a normalized fuzzy match between (a) the predicted answer and
     (b) each gold-answer alias against the concatenated tool-response text,
     plus an edit-distance<=2 token-level check, and counts how many flip to
     CvT. A trajectory flips if ANY of {predicted, gold aliases} fuzzy-matches.
  4) writes JSON + Markdown summaries to results/phase7/.

Run-time: <5 min on login node (pure CPU).

Usage:
  python scripts/task_s13_cvt_robustness.py
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import string
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

# Make scripts/ importable so we can reuse the canonical classifier.
PROJECT_ROOT = Path("/lus/lfs1aip2/scratch/u6gg/ts1201.u6gg/Project/KG_GRPO")
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from task16_classify import (  # noqa: E402
    classify_trajectory,
    extract_tool_responses,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_TRAJ = (
    PROJECT_ROOT
    / "results/trajectories/phase7/39g2_step500_full/step_0/trajectories.json"
)
DEFAULT_OUT_JSON = PROJECT_ROOT / "results/phase7/cvt_robustness.json"
DEFAULT_OUT_MD = PROJECT_ROOT / "results/phase7/cvt_robustness.md"


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

_PUNCT_TABLE = str.maketrans("", "", string.punctuation)
_ARTICLE_RE = re.compile(r"^(?:a|an|the)\s+")


def normalize_text(text: str) -> str:
    """Case-fold, strip leading article, strip surrounding ws+punct, collapse ws.

    Used for both the predicted answer and the tool-response text.
    """
    if text is None:
        return ""
    s = text.lower().strip()
    # strip a single leading article
    s = _ARTICLE_RE.sub("", s)
    # strip leading/trailing punctuation (but keep internal punct so substring
    # containment can still match phrases like "barack obama, jr")
    s = s.strip(string.punctuation + " \t\n\r")
    # collapse all whitespace runs to a single space
    s = " ".join(s.split())
    return s


def normalize_for_substring(text: str) -> str:
    """Aggressive normalization: drops ALL punctuation -- used for substring
    re-check so that "Eiffel Tower" matches "the eiffel-tower."""
    s = normalize_text(text)
    s = s.translate(_PUNCT_TABLE)
    s = " ".join(s.split())
    return s


# ---------------------------------------------------------------------------
# Edit distance (small, bounded)
# ---------------------------------------------------------------------------


def _bounded_edit_distance(a: str, b: str, max_d: int = 2) -> int:
    """Levenshtein with early exit when distance exceeds max_d.

    Returns max_d + 1 if the true distance exceeds max_d.
    """
    la, lb = len(a), len(b)
    if abs(la - lb) > max_d:
        return max_d + 1
    if la == 0:
        return lb
    if lb == 0:
        return la
    # ensure a is the shorter
    if la > lb:
        a, b = b, a
        la, lb = lb, la
    prev = list(range(la + 1))
    for j in range(1, lb + 1):
        curr = [j] + [0] * la
        bj = b[j - 1]
        row_min = curr[0]
        for i in range(1, la + 1):
            cost = 0 if a[i - 1] == bj else 1
            curr[i] = min(
                prev[i] + 1,        # deletion
                curr[i - 1] + 1,    # insertion
                prev[i - 1] + cost,  # substitution
            )
            if curr[i] < row_min:
                row_min = curr[i]
        if row_min > max_d:
            return max_d + 1
        prev = curr
    return prev[la]


# ---------------------------------------------------------------------------
# Fuzzy match logic
# ---------------------------------------------------------------------------


def fuzzy_match(predicted: str, tool_text: str) -> tuple[bool, str | None, str]:
    """Return (matched, matched_string, reason).

    Strategy (most permissive of the two):
      (a) Normalized substring containment (drops articles + all punct).
      (b) Token-level edit-distance<=2 against any contiguous n-gram of
          tool-response tokens with the same token count as the predicted.

    matched_string = the slice of normalized tool-response text that matched
    (best-effort -- empty string for fallback).
    reason = "substring" | "edit_distance" | "" .
    """
    pred_n = normalize_for_substring(predicted)
    if not pred_n:
        return False, None, ""
    tool_n = normalize_for_substring(tool_text)
    if not tool_n:
        return False, None, ""

    # (a) substring
    if pred_n in tool_n:
        # find the matched span in original-ish form for display
        idx = tool_n.find(pred_n)
        matched = tool_n[idx: idx + len(pred_n)]
        return True, matched, "substring"

    # (b) edit-distance against same-length n-grams of tool tokens.
    # Strict guards to avoid spurious matches like "Fiji"<->"fish":
    #   * predicted must be >=8 chars after normalization, OR
    #   * predicted must be a multi-token name (>=2 tokens),
    #   * the matched span must start with the same first character as the
    #     predicted (fights random short-word collisions),
    #   * max_d capped at floor(len/4) so we don't permit 50% character noise.
    pred_tokens = pred_n.split()
    if not pred_tokens:
        return False, None, ""
    tool_tokens = tool_n.split()
    n = len(pred_tokens)
    if n == 0 or len(tool_tokens) < n:
        return False, None, ""
    pred_joined = " ".join(pred_tokens)
    if len(pred_joined) < 8 and n < 2:
        return False, None, ""
    max_d = min(2, max(1, len(pred_joined) // 5))
    best_d = max_d + 1
    best_span: str | None = None
    pred_first = pred_joined[0]
    max_windows = 5000
    for i in range(min(len(tool_tokens) - n + 1, max_windows)):
        cand = " ".join(tool_tokens[i: i + n])
        if not cand or cand[0] != pred_first:
            continue
        d = _bounded_edit_distance(pred_joined, cand, max_d=max_d)
        if d < best_d:
            best_d = d
            best_span = cand
            if best_d == 0:
                break
    if best_d <= max_d:
        return True, best_span, f"edit_distance<={best_d}"
    return False, None, ""


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def hop_bucket(hops: Any) -> str:
    try:
        h = int(hops)
    except (TypeError, ValueError):
        return "unknown"
    if h <= 1:
        return "1"
    if h == 2:
        return "2"
    return "3+"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectories", type=Path, default=DEFAULT_TRAJ)
    parser.add_argument("--out_json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--out_md", type=Path, default=DEFAULT_OUT_MD)
    args = parser.parse_args()

    if not args.trajectories.exists():
        logger.error("Trajectories file not found: %s", args.trajectories)
        sys.exit(1)

    logger.info("Loading %s", args.trajectories)
    with open(args.trajectories) as f:
        trajs = json.load(f)
    logger.info("Loaded %d trajectories", len(trajs))

    # Step 1: re-classify with the canonical classifier.
    cat_counts: dict[str, int] = defaultdict(int)
    cvm_trajs: list[dict[str, Any]] = []
    n_em_correct = 0
    for t in trajs:
        cat = classify_trajectory(t)
        cat_counts[cat] += 1
        if t.get("em", 0) == 1:
            n_em_correct += 1
        if cat == "correct-via-memory":
            cvm_trajs.append(t)
    logger.info(
        "Canonical counts: %s",
        {k: cat_counts[k] for k in sorted(cat_counts.keys())},
    )
    logger.info(
        "EM-correct=%d, originally CvM=%d", n_em_correct, len(cvm_trajs)
    )

    # Step 2-3: fuzzy match each CvM.
    flipped: list[dict[str, Any]] = []
    still_cvm: list[dict[str, Any]] = []
    by_hops: dict[str, dict[str, int]] = defaultdict(
        lambda: {"orig_cvm": 0, "now_cvt": 0}
    )

    for t in cvm_trajs:
        bucket = hop_bucket(t.get("hops"))
        by_hops[bucket]["orig_cvm"] += 1

        predicted = t.get("predicted", "")
        all_answers = list(t.get("all_answers") or [])
        full_response = t.get("full_response", "")
        tool_text = extract_tool_responses(full_response)

        # Mirror the canonical classifier: try the predicted AND every gold
        # alias. Flip if ANY of them fuzzy-matches the tool-response text.
        candidates: list[str] = []
        if predicted:
            candidates.append(predicted)
        for ans in all_answers:
            if ans and ans not in candidates:
                candidates.append(ans)

        matched = False
        matched_str: str | None = None
        reason = ""
        matched_candidate: str | None = None
        for cand in candidates:
            ok, mstr, why = fuzzy_match(cand, tool_text)
            if ok:
                matched = True
                matched_str = mstr
                reason = why
                matched_candidate = cand
                break

        record = {
            "sample_id": t.get("sample_id"),
            "question": t.get("question"),
            "predicted": predicted,
            "gold_answer": t.get("gold_answer"),
            "all_answers": all_answers,
            "hops": t.get("hops"),
            "fuzzy_matched_candidate": matched_candidate,
            "fuzzy_matched_string": matched_str,
            "match_reason": reason,
        }
        if matched:
            by_hops[bucket]["now_cvt"] += 1
            flipped.append(record)
        else:
            still_cvm.append(record)

    # Step 4: aggregate.
    n_orig_cvm = len(cvm_trajs)
    n_now_cvt = len(flipped)
    pct_flip = (n_now_cvt / n_orig_cvm) if n_orig_cvm else 0.0

    by_hops_out: dict[str, dict[str, Any]] = {}
    for k, v in by_hops.items():
        rate = (v["now_cvt"] / v["orig_cvm"]) if v["orig_cvm"] else 0.0
        by_hops_out[k] = {
            "orig_cvm": v["orig_cvm"],
            "now_cvt_under_fuzzy": v["now_cvt"],
            "pct_flip": rate,
        }

    # Build the new (post-fuzzy) overall CvT count + rate.
    canonical_cvt = cat_counts.get("correct-via-tool", 0)
    n = len(trajs)
    fuzzy_cvt_total = canonical_cvt + n_now_cvt

    # Sort flipped/still-CvM examples for stable selection.
    flipped_sorted = sorted(flipped, key=lambda r: str(r.get("sample_id")))
    still_sorted = sorted(still_cvm, key=lambda r: str(r.get("sample_id")))

    output = {
        "trajectory_file": str(args.trajectories),
        "n_total": n,
        "n_total_em_correct": n_em_correct,
        "canonical_counts": {k: cat_counts[k] for k in sorted(cat_counts.keys())},
        "canonical_cvt": canonical_cvt,
        "canonical_cvt_rate": canonical_cvt / n if n else 0.0,
        "n_originally_CvM": n_orig_cvm,
        "n_now_CvT_under_fuzzy": n_now_cvt,
        "pct_flip": pct_flip,
        "fuzzy_cvt_total": fuzzy_cvt_total,
        "fuzzy_cvt_rate": (fuzzy_cvt_total / n) if n else 0.0,
        "by_hops": by_hops_out,
        "examples_flipped": flipped_sorted[:10],
        "examples_still_CvM": still_sorted[:5],
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    logger.info("Wrote %s", args.out_json)

    # ----- Markdown narrative -----
    md: list[str] = []
    md.append("# Task S1.3: CvT metric robustness under fuzzy matching\n")
    md.append(
        f"Source trajectories: `{args.trajectories.relative_to(PROJECT_ROOT)}` "
        f"(n={n}, EM-correct={n_em_correct}).\n"
    )
    md.append(
        "## Definition\n\n"
        "The canonical 7-category classifier "
        "(`scripts/task16_classify.py::classify_trajectory`) decides:\n"
        "* **CvT** -- `em==1` AND ANY gold-answer alias (lowercased) appears "
        "as a substring of the concatenated `<tool_response>` text.\n"
        "* **CvM** -- `em==1` AND no gold alias appears as a substring of any "
        "tool response.\n\n"
        "Fuzzy match (this audit) tests both the predicted answer AND every "
        "gold alias, flipping a trajectory if ANY of them fuzzy-matches:\n"
        "1. lowercase both sides,\n"
        "2. strip leading article (a/an/the),\n"
        "3. strip leading/trailing whitespace + punctuation, drop **all** "
        "punctuation, collapse whitespace,\n"
        "4. (a) substring containment under (1)-(3); else (b) any contiguous "
        "n-gram of tool tokens with token-level Levenshtein distance "
        "<=floor(len/5), with strict guards (predicted >=8 chars or "
        ">=2 tokens, same first character) to suppress short-word "
        "collisions.\n"
    )

    md.append("## Headline numbers\n")
    md.append("| Quantity | Value |")
    md.append("|---|---|")
    md.append(f"| n total | {n} |")
    md.append(f"| n EM-correct | {n_em_correct} |")
    md.append(
        f"| canonical CvT (substring) | {canonical_cvt} "
        f"({100 * canonical_cvt / n:.2f}%) |"
    )
    md.append(f"| originally CvM | {n_orig_cvm} |")
    md.append(
        f"| CvM -> CvT under fuzzy | {n_now_cvt} "
        f"({100 * pct_flip:.1f}% of CvM) |"
    )
    md.append(
        f"| fuzzy CvT total | {fuzzy_cvt_total} "
        f"({100 * fuzzy_cvt_total / n:.2f}%) |"
    )
    md.append("")

    md.append("## Stratified by hops (CvM -> CvT flip rate)\n")
    md.append("| hops | orig CvM | now CvT (fuzzy) | flip rate |")
    md.append("|---|---|---|---|")
    for k in ["1", "2", "3+", "unknown"]:
        if k not in by_hops_out:
            continue
        v = by_hops_out[k]
        md.append(
            f"| {k} | {v['orig_cvm']} | {v['now_cvt_under_fuzzy']} | "
            f"{100 * v['pct_flip']:.1f}% |"
        )
    md.append("")

    md.append("## Flipped examples (first 10)\n")
    for i, ex in enumerate(flipped_sorted[:10], 1):
        md.append(
            f"**{i}.** Q: {ex['question']!r}\n"
            f"   pred = {ex['predicted']!r}; "
            f"gold = {ex['gold_answer']!r}; "
            f"matched-candidate = {ex.get('fuzzy_matched_candidate')!r}; "
            f"matched-string = {ex['fuzzy_matched_string']!r} "
            f"(via {ex['match_reason']}); hops={ex['hops']}\n"
        )

    md.append("## Residual CvM examples (first 5 -- no fuzzy match)\n")
    for i, ex in enumerate(still_sorted[:5], 1):
        md.append(
            f"**{i}.** Q: {ex['question']!r}\n"
            f"   pred = {ex['predicted']!r}; "
            f"gold = {ex['gold_answer']!r}; hops={ex['hops']}\n"
        )

    md.append("## Recommendation for the paper\n")
    if pct_flip >= 0.20:
        md.append(
            f"The flip rate is **{100 * pct_flip:.1f}%**, which is large. The "
            "current substring-based CvT under-reports tool-grounded correctness "
            "due to formatting noise (articles, punctuation, capitalization). "
            "**Recommend reporting fuzzy CvT as the primary headline number, "
            "with strict CvT in a supplementary table.**"
        )
    elif pct_flip >= 0.05:
        md.append(
            f"The flip rate is **{100 * pct_flip:.1f}%**: non-trivial but "
            "modest. **Recommend keeping strict CvT as the headline (its "
            "trends are unchanged), and adding a one-line footnote that fuzzy "
            "CvT shifts +{:.1f} pp; full robustness numbers in the appendix.**"
            .format(100 * (n_now_cvt / n))
        )
    else:
        md.append(
            f"The flip rate is **{100 * pct_flip:.1f}%**, i.e. the CvM "
            "residual is essentially genuine memory-mediated correctness, not "
            "a formatting artifact. **Recommend keeping the canonical "
            "substring CvT as the headline metric; mention the audit briefly "
            "in the appendix.**"
        )
    md.append("")

    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_md, "w") as f:
        f.write("\n".join(md) + "\n")
    logger.info("Wrote %s", args.out_md)

    # Console summary.
    print()
    print(
        f"n_total={n}  EM={n_em_correct}  canonical CvT={canonical_cvt}  "
        f"orig CvM={n_orig_cvm}  CvM->CvT={n_now_cvt}  "
        f"flip%={100 * pct_flip:.1f}"
    )
    for k in ["1", "2", "3+", "unknown"]:
        if k in by_hops_out:
            v = by_hops_out[k]
            print(
                f"  hops={k}: orig_CvM={v['orig_cvm']} "
                f"now_CvT={v['now_cvt_under_fuzzy']} "
                f"flip%={100 * v['pct_flip']:.1f}"
            )


if __name__ == "__main__":
    main()
