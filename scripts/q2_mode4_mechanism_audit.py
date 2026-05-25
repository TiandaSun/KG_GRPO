"""Audit the quote-and-stop hypothesis (paper §5.4 Mode 4) for I-Self / 39I.

Operates on a trajectories.json file with the schema produced by the phase7
eval pipeline (one record per question, with `full_response` text containing
<search>/<tool_response>/<answer> blocks).

For each of the first N records, parses out:
  - n_tool_calls
  - last_answer        (text inside the final <answer>...</answer>)
  - last_response      (text inside the final <tool_response>...</tool_response>)
  - is_substring       (last_answer is a verbatim substring of last_response)
  - response_entities  (number of entities in last_response, parsed as JSON list
                        and falling back to a comma/newline split)
  - quote_and_stop     ((n_tool_calls == 1) and is_substring and response_entities >= 2)

Writes a single JSON with stats + the per-sample table.
"""
import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict

ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
SEARCH_RE = re.compile(r"<search>(.*?)</search>", re.DOTALL)
TRESP_RE = re.compile(r"<tool_response>(.*?)</tool_response>", re.DOTALL)


def _entity_count(resp: str) -> int:
    """Count entities in a tool_response payload.

    The phase7 server emits JSON list strings like `["A", "B", "C"]`. Try JSON
    first, then fall back to comma/newline splits for older / malformed dumps.
    """
    s = resp.strip()
    try:
        obj = json.loads(s)
        if isinstance(obj, list):
            return len(obj)
    except Exception:
        pass
    parts = [p.strip().strip('"').strip("'") for p in re.split(r"[,\n]", s) if p.strip()]
    return len(parts)


def audit_sample(rec):
    # type: (Dict[str, Any]) -> Dict[str, Any]
    full = rec.get("full_response", "") or ""
    answers = ANSWER_RE.findall(full)
    searches = SEARCH_RE.findall(full)
    responses = TRESP_RE.findall(full)
    n_tool_calls = rec.get("num_tool_calls")
    if n_tool_calls is None:
        n_tool_calls = len(searches)
    last_answer = answers[-1].strip() if answers else (rec.get("predicted") or "").strip()
    last_response = responses[-1].strip() if responses else ""
    is_substring = bool(last_answer) and bool(last_response) and last_answer in last_response
    n_entities = _entity_count(last_response) if last_response else 0
    quote_and_stop = (n_tool_calls == 1) and is_substring and (n_entities >= 2)
    return {
        "sample_id": rec.get("sample_id"),
        "em": rec.get("em"),
        "f1": rec.get("f1"),
        "n_tool_calls": n_tool_calls,
        "last_answer": last_answer,
        "last_response_preview": last_response[:200],
        "is_substring": is_substring,
        "response_entities": n_entities,
        "quote_and_stop": quote_and_stop,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True,
                    help="trajectories.json (with full_response field)")
    ap.add_argument("--output", required=True)
    ap.add_argument("--limit", type=int, default=200,
                    help="audit first N samples (default 200)")
    ap.add_argument("--label", default="",
                    help="label written into the output JSON (e.g. 39i_self_step300)")
    args = ap.parse_args()

    data = json.loads(Path(args.input).read_text())
    if isinstance(data, dict):
        data = list(data.values())
    n_total = len(data)
    sub = data[: args.limit]

    rows = [audit_sample(r) for r in sub]

    n = len(rows)
    tc_counter = Counter()  # type: Counter
    for r in rows:
        tc = r["n_tool_calls"]
        bucket = "3+" if (tc is not None and tc >= 3) else str(tc)
        tc_counter[bucket] += 1
    is_sub = sum(1 for r in rows if r["is_substring"])
    qas = [r for r in rows if r["quote_and_stop"]]
    qas_ent_dist = Counter(r["response_entities"] for r in qas)

    summary = {
        "label": args.label or Path(args.input).parent.name,
        "input_path": str(Path(args.input).resolve()),
        "n_total_in_file": n_total,
        "n_audited": n,
        "n_tool_calls_dist": dict(sorted(tc_counter.items(), key=lambda x: str(x[0]))),
        "n_tool_calls_pct": {k: round(v / n, 4) for k, v in tc_counter.items()},
        "is_substring_pct": round(is_sub / n, 4),
        "quote_and_stop_pct": round(len(qas) / n, 4),
        "quote_and_stop_response_entities_dist": dict(
            sorted(qas_ent_dist.items(), key=lambda x: x[0])
        ),
        "verdict_threshold": 0.70,
        "verdict": "VERIFIED" if (len(qas) / n) >= 0.70 else "NOT_VERIFIED",
        "samples": rows,
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "samples"}, indent=2))


if __name__ == "__main__":
    main()
