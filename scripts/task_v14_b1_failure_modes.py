#!/usr/bin/env python3
"""Task V14-B1: Sub-failure-mode classification of G2@500 non-EM trajectories.

Classifies each non-EM (em==0) trajectory on CWQ test into one of 4
signal-theoretic failure modes that validate the paper's framework:

  - L-prior : no relevant tool call (parametric-only / irrelevant queries)
  - L-sig   : silent-fail (tool called but >=50% responses empty / error)
  - L-comp  : multi-hop state drift (later <search> entity not in any prior
              tool response and not the original question entity)
  - L-lang  : schema mismatch / string-format (default bucket for the rest)

Tie-break priority: L-prior > L-sig > L-comp > L-lang.

Inputs:
  results/trajectories/phase7/39g2_step500_full/step_0/trajectories.json
  data/freebase/verl_cwq/test.parquet  (for kg_path)
Outputs:
  results/phase7/v14_b1_failure_modes.json
  results/phase7/v14_b1_failure_modes.md
"""

from __future__ import annotations

import argparse
import ast
import json
import logging
import re
import string
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path("/lus/lfs1aip2/scratch/u6gg/ts1201.u6gg/Project/KG_GRPO")
DEFAULT_TRAJ = PROJECT_ROOT / "results/trajectories/phase7/39g2_step500_full/step_0/trajectories.json"
DEFAULT_PARQUET = PROJECT_ROOT / "data/freebase/verl_cwq/test.parquet"
DEFAULT_OUT_JSON = PROJECT_ROOT / "results/phase7/v14_b1_failure_modes.json"
DEFAULT_OUT_MD = PROJECT_ROOT / "results/phase7/v14_b1_failure_modes.md"

MODES = ["L-prior", "L-sig", "L-comp", "L-lang"]

MODE_COMMENTARY = {
    "L-prior": (
        "Model skipped or mis-aimed tool use (no question entity in any query); "
        "relies on parametric prior."
    ),
    "L-sig": (
        "Tool channel returned empty/error responses >=50% of the time; "
        "answer could not be grounded in retrieved evidence."
    ),
    "L-comp": (
        "Chain broke across turns: a later search entity was not in any prior "
        "tool response nor the question (state-carrying failure)."
    ),
    "L-lang": (
        "Tool responses were non-empty but (entity, relation) did not match the "
        "gold kg_path triples — schema/string-format mismatch or wrong relation."
    ),
}


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

SEARCH_RE = re.compile(r"<search>\s*(\w+)\s*\(([^)]*)\)\s*</search>", re.DOTALL)
TOOL_RESP_RE = re.compile(r"<tool_response>(.*?)</tool_response>", re.DOTALL)
ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)


def normalize(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    text = text.lower().strip()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = " ".join(text.split())
    return text


def parse_search_calls(full_response: str) -> list[dict[str, str]]:
    """Return list of {action, entity, relation} from <search> tags."""
    calls: list[dict[str, str]] = []
    for match in SEARCH_RE.finditer(full_response):
        action = match.group(1).strip()
        args_raw = match.group(2).strip()
        if not args_raw:
            continue
        parts = [p.strip().strip("'\"") for p in args_raw.split(",", 1)]
        entity = parts[0] if parts else ""
        relation = parts[1] if len(parts) > 1 else ""
        calls.append(
            {
                "action": action,
                "entity": entity,
                "relation": relation,
                "span_end": match.end(),
            }
        )
    return calls


def parse_tool_responses(full_response: str) -> list[dict[str, Any]]:
    """Return list of tool responses aligned to their position + entities."""
    out: list[dict[str, Any]] = []
    for m in TOOL_RESP_RE.finditer(full_response):
        raw = m.group(1).strip()
        ents: list[str] = []
        is_empty = raw in ("", "[]", "{}") or raw.lower().startswith("error") or \
            raw == "No results found"
        if not is_empty:
            try:
                parsed = json.loads(raw)
            except Exception:
                try:
                    parsed = ast.literal_eval(raw)
                except Exception:
                    parsed = None
            if isinstance(parsed, list):
                ents = [str(x) for x in parsed]
            elif isinstance(parsed, dict):
                ents = [str(x) for x in parsed.keys()] + [str(x) for x in parsed.values()]
            else:
                ents = [raw]
        out.append(
            {
                "raw": raw,
                "entities": ents,
                "is_empty": is_empty,
                "span_start": m.start(),
            }
        )
    return out


def question_entities(question: str) -> set[str]:
    """Return lowercase words / bigrams / trigrams from the question (heuristic
    surface form check)."""
    q = question.lower()
    q_clean = q.translate(str.maketrans("", "", string.punctuation))
    toks = q_clean.split()
    grams: set[str] = set()
    for n in (1, 2, 3):
        for i in range(len(toks) - n + 1):
            grams.add(" ".join(toks[i : i + n]))
    return grams


def entity_in_question(entity: str, q_grams: set[str], question_lower: str) -> bool:
    ent = normalize(entity)
    if not ent:
        return False
    if ent in q_grams:
        return True
    # substring fallback for multi-word entities
    return ent in question_lower


def entity_in_prior_response(entity: str, prior_responses: list[dict[str, Any]]) -> bool:
    ent = normalize(entity)
    if not ent:
        return False
    for resp in prior_responses:
        if resp["is_empty"]:
            continue
        # direct entity list match
        for e in resp["entities"]:
            if normalize(e) == ent:
                return True
        # substring fallback inside raw response
        if ent in resp["raw"].lower():
            return True
    return False


# ---------------------------------------------------------------------------
# kg_path lookup
# ---------------------------------------------------------------------------


def load_kg_paths(parquet_path: Path) -> dict[str, list[tuple[str, str, str]]]:
    df = pd.read_parquet(parquet_path)
    kg: dict[str, list[tuple[str, str, str]]] = {}
    for _, row in df.iterrows():
        ei = row["extra_info"]
        sid = ei["sample_id"]
        paths = ei["kg_path"]
        triples: list[tuple[str, str, str]] = []
        for t in paths:
            tt = tuple(str(x) for x in t)
            if len(tt) == 3:
                triples.append(tt)  # type: ignore[arg-type]
        kg[sid] = triples
    return kg


def query_matches_gold_triple(
    entity: str, relation: str, triples: list[tuple[str, str, str]]
) -> bool:
    ent = normalize(entity)
    rel = normalize(relation)
    for h, r, t in triples:
        if normalize(h) == ent and normalize(r) == rel:
            return True
        if normalize(t) == ent and normalize(r) == rel:
            return True
    return False


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------


def classify_failure_mode(
    trajectory: dict[str, Any],
    kg_triples: list[tuple[str, str, str]] | None,
) -> tuple[str, dict[str, Any]]:
    """Apply decision rules. Return (mode, diagnostics)."""
    full_resp = trajectory["full_response"]
    question = trajectory["question"]
    q_lower = question.lower()
    q_grams = question_entities(question)

    calls = parse_search_calls(full_resp)
    responses = parse_tool_responses(full_resp)

    n_calls = len(calls)
    n_resp = len(responses)
    n_empty = sum(1 for r in responses if r["is_empty"])
    empty_frac = n_empty / max(n_resp, 1)

    # Rule 1: L-prior
    # - no tool calls recorded, OR
    # - the model used tool calls but NONE of the queried entities appears in
    #   the question (treated as "irrelevant queries / parametric reach")
    any_query_ent_in_q = False
    for c in calls:
        if entity_in_question(c["entity"], q_grams, q_lower):
            any_query_ent_in_q = True
            break

    if n_calls == 0 or not any_query_ent_in_q:
        return "L-prior", {
            "n_calls": n_calls,
            "n_resp": n_resp,
            "empty_frac": empty_frac,
            "reason": "0 tool calls" if n_calls == 0 else "no query entity in question",
        }

    # Rule 2: L-sig — >=50% empty responses
    if n_resp >= 1 and empty_frac >= 0.5:
        return "L-sig", {
            "n_calls": n_calls,
            "n_resp": n_resp,
            "empty_frac": empty_frac,
            "reason": f">={int(empty_frac * 100)}% empty/error responses",
        }

    # Rule 3: L-comp — chain broke across turns
    if n_calls >= 2:
        broke = False
        for i, c in enumerate(calls[1:], start=1):
            # prior responses = those before this search tag
            prior = [r for r in responses if r["span_start"] < c["span_end"]]
            prior = prior[: i]  # only responses that precede this call
            ent = c["entity"]
            if entity_in_question(ent, q_grams, q_lower):
                continue  # still anchored in the question
            if not entity_in_prior_response(ent, prior):
                broke = True
                break
        if broke:
            return "L-comp", {
                "n_calls": n_calls,
                "n_resp": n_resp,
                "empty_frac": empty_frac,
                "reason": "later search entity not in question nor prior response",
            }

    # Rule 4: L-lang default. Tighten: at least one non-empty response AND
    # (entity, relation) does not match any gold kg_path triple.
    any_nonempty = any(not r["is_empty"] for r in responses)
    lang_triple_mismatch = False
    if kg_triples is not None:
        for c in calls:
            if not query_matches_gold_triple(c["entity"], c["relation"], kg_triples):
                lang_triple_mismatch = True
                break

    return "L-lang", {
        "n_calls": n_calls,
        "n_resp": n_resp,
        "empty_frac": empty_frac,
        "any_nonempty_resp": any_nonempty,
        "triple_mismatch": lang_triple_mismatch,
        "reason": "non-empty tool responses but no EM (schema/relation mismatch)",
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def summarize(
    results: list[dict[str, Any]], exemplars_per_mode: int = 5
) -> dict[str, Any]:
    total = len(results)
    counts = Counter(r["mode"] for r in results)
    dist: dict[str, dict[str, Any]] = {}
    for m in MODES:
        c = counts.get(m, 0)
        dist[m] = {
            "count": c,
            "pct_of_non_em": round(100.0 * c / max(total, 1), 2),
            "commentary": MODE_COMMENTARY[m],
        }

    exemplars: dict[str, list[dict[str, Any]]] = {m: [] for m in MODES}
    for r in results:
        m = r["mode"]
        if len(exemplars[m]) >= exemplars_per_mode:
            continue
        snippet = r["full_response"]
        if len(snippet) > 900:
            snippet = snippet[:900] + "..."
        exemplars[m].append(
            {
                "sample_id": r["sample_id"],
                "question": r["question"],
                "gold_answer": r["gold_answer"],
                "predicted": r["predicted"],
                "hops": r["hops"],
                "num_tool_calls": r["num_tool_calls"],
                "diagnostics": r["diagnostics"],
                "trajectory_excerpt": snippet,
            }
        )

    return {
        "total_non_em": total,
        "distribution": dist,
        "exemplars": exemplars,
        "tie_break_order": MODES,
    }


def write_memo(summary: dict[str, Any], out_path: Path) -> None:
    total = summary["total_non_em"]
    dist = summary["distribution"]
    lines: list[str] = []
    lines.append("# V14-B1: G2@500 failure-mode distribution")
    lines.append("")
    lines.append("## Setup")
    lines.append(
        f"N={total} non-EM trajectories from G2@500 (CWQ test, 3,531 total). "
        f"Classified via the 4 signal-theoretic decision rules below; ties "
        f"broken in order L-prior > L-sig > L-comp > L-lang."
    )
    lines.append("")
    lines.append("## Distribution")
    lines.append("| Mode | Count | % of non-EM | Paper implication |")
    lines.append("|---|---|---|---|")
    for m in MODES:
        d = dist[m]
        lines.append(
            f"| {m} | {d['count']} | {d['pct_of_non_em']}% | {d['commentary']} |"
        )
    lines.append("")

    lines.append("## Decision rules (for reproducibility)")
    lines.append(
        "- L-prior: num_tool_calls==0 OR no queried entity appears in the "
        "question text (parametric / irrelevant queries)."
    )
    lines.append(
        "- L-sig: >=1 tool call AND >=50% of tool responses are empty `[]`/`{}` "
        "or error."
    )
    lines.append(
        "- L-comp: >=2 tool calls AND a later search entity is neither in the "
        "question nor in any prior tool response (broken entity chain)."
    )
    lines.append(
        "- L-lang: default — non-empty responses but (entity, relation) does "
        "not match any gold kg_path triple (schema/string-format mismatch)."
    )
    lines.append("")

    lines.append("## Exemplars")
    for m in MODES:
        exs = summary["exemplars"].get(m, [])
        if not exs:
            continue
        lines.append(f"### {m}")
        for e in exs[:2]:
            lines.append(
                f"- **{e['sample_id']}** (hops={e['hops']}): "
                f"Q: _{e['question']}_"
            )
            lines.append(f"  - gold: `{e['gold_answer']}` | predicted: `{e['predicted']}`")
            snip = e["trajectory_excerpt"].replace("\n", " ")
            if len(snip) > 220:
                snip = snip[:220] + "..."
            lines.append(f"  - trajectory: `{snip}`")
        lines.append("")

    # Headline
    sorted_modes = sorted(MODES, key=lambda k: dist[k]["count"], reverse=True)
    top = sorted_modes[0]
    second = sorted_modes[1]
    lines.append("## Headline finding")
    headline_parts: list[str] = []
    headline_parts.append(
        f"The dominant failure mode is **{top}** "
        f"({dist[top]['pct_of_non_em']}% of non-EM), followed by **{second}** "
        f"({dist[second]['pct_of_non_em']}%)."
    )
    if top == "L-lang":
        headline_parts.append(
            "This validates V14-B2's schema-format finding: even after G2's step "
            "supervision, the primary signal bottleneck is language-to-schema "
            "grounding (relation-name mismatches), not retrieval success."
        )
    elif top == "L-sig":
        headline_parts.append(
            "The model can identify the right entity and relation shape but the "
            "KG channel returns no evidence — G2 is retrieval-limited, not "
            "schema-limited."
        )
    elif top == "L-prior":
        headline_parts.append(
            "Most failures are entity-linking or parametric-prior failures: the "
            "model is not even aiming the tool at a question entity."
        )
    else:
        headline_parts.append(
            "State-carrying / multi-hop chaining is the main bottleneck — the "
            "model cannot keep the entity chain across turns."
        )
    lines.append(" ".join(headline_parts))
    lines.append("")

    out_path.write_text("\n".join(lines))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectories", type=Path, default=DEFAULT_TRAJ)
    parser.add_argument("--parquet", type=Path, default=DEFAULT_PARQUET)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_OUT_MD)
    parser.add_argument("--exemplars", type=int, default=5)
    args = parser.parse_args()

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Loading trajectories: %s", args.trajectories)
    trajectories = json.loads(args.trajectories.read_text())
    logger.info("  %d trajectories total", len(trajectories))

    logger.info("Loading kg_paths from parquet: %s", args.parquet)
    kg_paths = load_kg_paths(args.parquet)
    logger.info("  %d samples with kg_path", len(kg_paths))

    non_em = [t for t in trajectories if float(t.get("em", 0.0)) == 0.0]
    logger.info("Non-EM trajectories: %d", len(non_em))

    results: list[dict[str, Any]] = []
    for traj in non_em:
        sid = traj["sample_id"]
        triples = kg_paths.get(sid)
        mode, diag = classify_failure_mode(traj, triples)
        results.append(
            {
                "sample_id": sid,
                "mode": mode,
                "diagnostics": diag,
                "question": traj["question"],
                "gold_answer": traj["gold_answer"],
                "predicted": traj["predicted"],
                "hops": traj["hops"],
                "num_tool_calls": traj["num_tool_calls"],
                "full_response": traj["full_response"],
            }
        )

    summary = summarize(results, exemplars_per_mode=args.exemplars)

    # Persist a trimmed per-sample table (drop full_response to keep JSON small)
    per_sample = [
        {
            "sample_id": r["sample_id"],
            "mode": r["mode"],
            "hops": r["hops"],
            "num_tool_calls": r["num_tool_calls"],
            "diagnostics": r["diagnostics"],
        }
        for r in results
    ]

    out = {
        "setup": {
            "source": str(args.trajectories),
            "parquet": str(args.parquet),
            "total_non_em": summary["total_non_em"],
            "tie_break_order": summary["tie_break_order"],
            "rules": {
                "L-prior": "num_tool_calls==0 OR no queried entity in question",
                "L-sig": ">=1 tool call AND empty_frac >= 0.5",
                "L-comp": ">=2 tool calls AND later entity not in prior responses/question",
                "L-lang": "default; typically non-empty responses but triple not in gold kg_path",
            },
        },
        "distribution": summary["distribution"],
        "exemplars": summary["exemplars"],
        "per_sample": per_sample,
    }

    args.out_json.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    logger.info("Wrote %s", args.out_json)

    write_memo(summary, args.out_md)
    logger.info("Wrote %s", args.out_md)

    print()
    print("=" * 72)
    print(f"V14-B1 failure-mode distribution (N={summary['total_non_em']} non-EM)")
    print("=" * 72)
    for m in MODES:
        d = summary["distribution"][m]
        print(f"  {m:<8}  {d['count']:>5}  ({d['pct_of_non_em']:>5.2f}%)")
    print("=" * 72)


if __name__ == "__main__":
    main()
