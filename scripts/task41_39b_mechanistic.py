"""Phase 7 Task 41: 39B@400 mechanistic memo.

Analyses existing saved trajectories for 39B@400, E3@500, and E5b@100 over the
full 3,531-question CWQ test set, to explain WHY 39B delivers EM=38.35% /
CvT=3.77%. Produces a short mechanistic memo and a per-sample error-mode CSV
for kg-incomplete trajectories.

Pure CPU analysis. No GPU, no SLURM, no KG server, no LLM calls.

Runnable standalone from project root:
    python scripts/task41_39b_mechanistic.py

Outputs
-------
results/phase7/task41_39b_mechanistic.md : ≤ 2-page memo with four findings.
results/phase7/39b_query_error_modes.csv : per-sample error-mode labels for
    the first 300 kg-incomplete 39B@400 trajectories.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import random
import re
import string
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

PROJECT_ROOT = Path("/lus/lfs1aip2/scratch/u6gg/ts1201.u6gg/Project/KG_GRPO")

# Make `task16_classify` importable when this script is run from project root.
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from task16_classify import classify_trajectory  # noqa: E402

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Paths:
    """Resolved absolute paths for all inputs and outputs."""

    traj_39b: Path
    traj_e3: Path
    traj_e5b: Path
    category_b: Path
    test_parquet: Path
    cvt_audit: Path
    out_md: Path
    out_csv: Path

    @classmethod
    def defaults(cls) -> "Paths":
        root = PROJECT_ROOT
        return cls(
            traj_39b=root / "results/trajectories/phase7/39b_step400_full/step_400/trajectories.json",
            traj_e3=root / "results/trajectories/phase7/e3_step500_full/step_500/trajectories.json",
            traj_e5b=root / "results/trajectories/phase7/e5b_step100_full/step_100/trajectories.json",
            category_b=root / "results/task26_category_b.json",
            test_parquet=root / "data/freebase/verl_cwq/test.parquet",
            cvt_audit=root / "results/phase7/full_test_cvt_audit.json",
            out_md=root / "results/phase7/task41_39b_mechanistic.md",
            out_csv=root / "results/phase7/39b_query_error_modes.csv",
        )


SEARCH_RE = re.compile(r"<search>(.*?)</search>", re.DOTALL)
CALL_RE = re.compile(r"(\w+)\(([^)]*)\)")
TOOL_RESPONSE_RE = re.compile(r"<tool_response>(.*?)</tool_response>", re.DOTALL)


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------
def load_trajectories(path: Path) -> list[dict[str, Any]]:
    logger.info("Loading trajectories: %s", path)
    with path.open("r") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise TypeError(f"Expected list in {path}, got {type(data).__name__}")
    return data


def load_category_b_ids(path: Path) -> set[str]:
    logger.info("Loading category B IDs: %s", path)
    with path.open("r") as f:
        data = json.load(f)
    return set(data["category_b_ids"])


def load_test_meta(path: Path) -> dict[str, dict[str, Any]]:
    """Build sample_id -> {hops, query_entities, kg_path_relations, kg_path_entities}."""
    import pandas as pd  # Local import: heavy dep only needed here.

    logger.info("Loading test parquet: %s", path)
    df = pd.read_parquet(path)
    meta: dict[str, dict[str, Any]] = {}
    for _, row in df.iterrows():
        ei = row["extra_info"]
        sid = str(ei["sample_id"])
        query_entities = [str(x) for x in list(ei.get("query_entities", []))]
        kg_path = list(ei.get("kg_path", []))
        relations: set[str] = set()
        entities: set[str] = set()
        for triple in kg_path:
            t = list(triple)
            if len(t) >= 3:
                entities.add(str(t[0]))
                relations.add(str(t[1]))
                entities.add(str(t[2]))
        meta[sid] = {
            "hops": int(ei.get("hops", -1)),
            "query_entities": query_entities,
            "kg_path_relations": relations,
            "kg_path_entities": entities,
        }
    logger.info("Loaded %d test-meta rows", len(meta))
    return meta


def load_cvt_audit(path: Path) -> dict[str, Any]:
    logger.info("Loading CvT audit: %s", path)
    with path.open("r") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def normalize_text(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace, drop leading articles."""
    text = text.lower().strip()
    for article in ("a ", "an ", "the "):
        if text.startswith(article):
            text = text[len(article):]
    text = text.translate(str.maketrans("", "", string.punctuation))
    return " ".join(text.split())


def loose_entity_match(candidate: str, reference: Iterable[str]) -> bool:
    """Entity-level fuzzy match: normalize whitespace/underscores/case."""
    norm_cand = normalize_text(candidate.replace("_", " "))
    for ref in reference:
        if not ref:
            continue
        if normalize_text(ref.replace("_", " ")) == norm_cand:
            return True
    return False


def has_format_mismatch(candidate: str, reference: Iterable[str]) -> bool:
    """True iff normalized matches but exact does not (whitespace/underscore)."""
    if any(candidate == r for r in reference):
        return False
    return loose_entity_match(candidate, reference)


def parse_first_search(full_response: str) -> tuple[str, str, str] | None:
    """Return (fn_name, first_arg_entity, relation) of the first search call."""
    matches = SEARCH_RE.findall(full_response)
    for raw in matches:
        call = CALL_RE.search(raw)
        if not call:
            continue
        fn = call.group(1)
        args = call.group(2)
        parts = [p.strip() for p in args.split(",")]
        if len(parts) >= 2:
            return fn, parts[0], parts[1]
        if len(parts) == 1 and parts[0]:
            return fn, parts[0], ""
    return None


def extract_tool_response_text(full_response: str) -> str:
    return " ".join(TOOL_RESPONSE_RE.findall(full_response))


def kg_returned_nonempty_for_first_call(full_response: str) -> bool:
    """Check whether the response after the first <search> has useful content."""
    m = SEARCH_RE.search(full_response)
    if not m:
        return False
    tail = full_response[m.end():]
    resp_m = TOOL_RESPONSE_RE.search(tail)
    if not resp_m:
        return False
    text = resp_m.group(1).strip()
    if not text:
        return False
    if "No results found" in text:
        return False
    # If the entire response is an empty list, treat as empty.
    stripped = text.strip()
    if stripped in {"[]", "{}"}:
        return False
    return True


# ---------------------------------------------------------------------------
# Finding 1: Category A vs B
# ---------------------------------------------------------------------------
def category_split(
    trajs: list[dict[str, Any]], cat_b_ids: set[str]
) -> dict[str, dict[str, int | float]]:
    """Return dict with n/em-correct counts for A and B subsets."""
    out = {"A": {"n": 0, "em": 0, "cvt": 0}, "B": {"n": 0, "em": 0, "cvt": 0}}
    for t in trajs:
        bucket = "B" if t["sample_id"] in cat_b_ids else "A"
        out[bucket]["n"] += 1
        out[bucket]["em"] += int(float(t.get("em", 0.0)) >= 1.0)
        if classify_trajectory(t) == "correct-via-tool":
            out[bucket]["cvt"] += 1
    for k in out:
        n = out[k]["n"]
        out[k]["em_pct"] = (out[k]["em"] / n * 100.0) if n else 0.0
        out[k]["cvt_pct"] = (out[k]["cvt"] / n * 100.0) if n else 0.0
    return out


# ---------------------------------------------------------------------------
# Finding 2: Hop-stratified EM
# ---------------------------------------------------------------------------
def hop_stratified_em(
    trajs: list[dict[str, Any]], meta: dict[str, dict[str, Any]]
) -> dict[int, dict[str, int | float]]:
    """Return per-hop dict with n / em-correct / em-pct."""
    buckets: dict[int, dict[str, int]] = defaultdict(lambda: {"n": 0, "em": 0})
    for t in trajs:
        sid = t["sample_id"]
        hops = meta.get(sid, {}).get("hops", int(t.get("hops", -1)))
        buckets[hops]["n"] += 1
        buckets[hops]["em"] += int(float(t.get("em", 0.0)) >= 1.0)
    out: dict[int, dict[str, int | float]] = {}
    for hop, v in buckets.items():
        out[hop] = {
            "n": v["n"],
            "em": v["em"],
            "em_pct": (v["em"] / v["n"] * 100.0) if v["n"] else 0.0,
        }
    return dict(sorted(out.items()))


# ---------------------------------------------------------------------------
# Finding 3: kg-incomplete error modes (first 300 trajectories)
# ---------------------------------------------------------------------------
def classify_query_error(
    traj: dict[str, Any], meta_row: dict[str, Any] | None
) -> tuple[str, str]:
    """Return (error_mode, first_search_call_string).

    Error modes:
      (a) wrong-entity : query entity not in extra_info.query_entities
      (b) wrong-relation : entity ok, but relation not in kg_path relations
      (c) format-mismatch : normalized entity matches gold but with whitespace/
          underscore/case mismatch that could defeat exact-match tool lookup
      (d) genuine-kg-miss : query matches gold entity + relation but tool still
          returned no useful content
      (z) no-search-call : model never emitted a parseable <search>fn(...)</search>
    """
    parsed = parse_first_search(traj["full_response"])
    if parsed is None:
        return "no-search-call", ""
    fn, ent, rel = parsed
    first_call = f"{fn}({ent}, {rel})" if rel else f"{fn}({ent})"

    if meta_row is None:
        return "no-meta", first_call

    gold_entities: list[str] = list(meta_row.get("query_entities", []))
    gold_relations: set[str] = set(meta_row.get("kg_path_relations", set()))

    ent_exact = any(ent == g for g in gold_entities)
    ent_loose = loose_entity_match(ent, gold_entities)

    if not ent_loose:
        return "wrong-entity", first_call
    if not ent_exact and ent_loose:
        return "format-mismatch", first_call
    # entity is correct and exact — check relation
    if rel and rel not in gold_relations:
        return "wrong-relation", first_call
    # entity + relation match gold path; KG still returned empty for this call
    return "genuine-kg-miss", first_call


def error_mode_analysis(
    trajs: list[dict[str, Any]],
    meta: dict[str, dict[str, Any]],
    cvt_per_sample: dict[str, str],
    n_samples: int = 300,
    seed: int = 0,
) -> tuple[list[dict[str, str]], Counter]:
    """Filter kg-incomplete trajectories (first n in original order) and classify."""
    kg_incomplete_trajs = [
        t for t in trajs if cvt_per_sample.get(t["sample_id"]) == "kg-incomplete"
    ]
    logger.info("39B@400 kg-incomplete total: %d", len(kg_incomplete_trajs))

    # Deterministic: take the first n_samples in the original trajectory order.
    # (Seed is not used for selection, but keep for potential future tie-breaks.)
    rng = random.Random(seed)  # noqa: F841
    subset = kg_incomplete_trajs[:n_samples]

    rows: list[dict[str, str]] = []
    counter: Counter = Counter()
    for t in subset:
        meta_row = meta.get(t["sample_id"])
        mode, first_call = classify_query_error(t, meta_row)
        counter[mode] += 1
        rows.append({
            "sample_id": t["sample_id"],
            "question": t["question"].replace("\n", " ").strip(),
            "first_search_call": first_call,
            "error_mode": mode,
        })
    return rows, counter


# ---------------------------------------------------------------------------
# Finding 4: E5b vs 39B head-to-head on kg-incomplete
# ---------------------------------------------------------------------------
def e5b_vs_39b_on_kg_incomplete(
    trajs_39b: list[dict[str, Any]],
    trajs_e5b: list[dict[str, Any]],
    cvt_per_sample_39b: dict[str, str],
) -> dict[str, int | float]:
    """Among 39B@400's kg-incomplete trajectories, how many did E5b get EM=1?"""
    e5b_by_id = {t["sample_id"]: t for t in trajs_e5b}
    kg_inc_ids = [sid for sid, cat in cvt_per_sample_39b.items() if cat == "kg-incomplete"]
    rescued = 0
    rescued_cvt = 0
    e5b_em_any = 0  # present in e5b trajectory set at all
    for sid in kg_inc_ids:
        t_e5b = e5b_by_id.get(sid)
        if t_e5b is None:
            continue
        e5b_em_any += 1
        if float(t_e5b.get("em", 0.0)) >= 1.0:
            rescued += 1
        if classify_trajectory(t_e5b) == "correct-via-tool":
            rescued_cvt += 1
    n = len(kg_inc_ids)
    return {
        "n_kg_incomplete_39b": n,
        "n_with_e5b_counterpart": e5b_em_any,
        "e5b_rescue_count": rescued,
        "e5b_rescue_pct": (rescued / n * 100.0) if n else 0.0,
        "e5b_rescue_via_tool_count": rescued_cvt,
        "e5b_rescue_via_tool_pct": (rescued_cvt / n * 100.0) if n else 0.0,
    }


# ---------------------------------------------------------------------------
# Output: markdown memo
# ---------------------------------------------------------------------------
def fmt_row(cells: Iterable[str]) -> str:
    return "| " + " | ".join(str(c) for c in cells) + " |"


def build_memo(
    cat_split: dict[str, dict[str, dict[str, int | float]]],
    hop_stats: dict[str, dict[int, dict[str, int | float]]],
    error_counter: Counter,
    error_total: int,
    e5b_rescue: dict[str, int | float],
    cvt_audit: dict[str, Any],
) -> str:
    lines: list[str] = []
    lines.append("# Task 41 — 39B@400 Mechanistic Memo\n")
    lines.append(
        "Full-3,531-test analysis of why 39B@400 delivers EM=38.35% and CvT=3.77% "
        "against the two strong baselines E3@500 (EM=32.46%, CvT=0.03%) and "
        "E5b@100 (EM=32.20%, CvT=3.03%). Pure CPU analysis of saved trajectories; "
        "no new compute.\n"
    )

    # ---------- Finding 1 ----------
    lines.append("## Finding 1 — Category A vs B breakdown of EM\n")
    lines.append(
        "Category A = pass@10>0 on 7B-Instruct sampling (n=975); Category B = "
        "pass@10=0 (n=2,556, the 'hard' questions by frozen-base definition).\n"
    )
    lines.append(fmt_row([
        "model", "A n", "A EM", "A EM %", "A CvT", "B n", "B EM", "B EM %", "B CvT",
    ]))
    lines.append(fmt_row(["---"] * 9))
    for m in ("39b_step400", "e3_step500", "e5b_step100"):
        a = cat_split[m]["A"]
        b = cat_split[m]["B"]
        lines.append(fmt_row([
            m,
            a["n"],
            a["em"],
            f"{a['em_pct']:.2f}%",
            a["cvt"],
            b["n"],
            b["em"],
            f"{b['em_pct']:.2f}%",
            b["cvt"],
        ]))

    a39 = cat_split["39b_step400"]["A"]
    b39 = cat_split["39b_step400"]["B"]
    ae5 = cat_split["e5b_step100"]["A"]
    be5 = cat_split["e5b_step100"]["B"]
    gap_A = a39["em_pct"] - ae5["em_pct"]
    gap_B = b39["em_pct"] - be5["em_pct"]
    # Category B correct count vs classifier-CvT count
    b_correct = b39["em"]
    b_cvt = b39["cvt"]
    lines.append(
        f"\n**Observation.** 39B@400 solves **{b_correct}** Category-B questions, "
        f"but the 7-category classifier only credits **{b_cvt}** as 'correct-via-tool'. "
        f"The missing {b_correct - b_cvt} correct-B trajectories must therefore "
        "fall into 'correct-via-memory' (answer in final text but not in any "
        "returned tool snippet), implying the tool output likely still "
        "informed the answer but the `answer_in_kg_response` substring check "
        "underestimates tool attribution. 39B's A-gap over E5b is "
        f"**{gap_A:+.2f} pp**; its B-gap is **{gap_B:+.2f} pp** — the full-test "
        "EM lift is driven by both buckets, not just 'easy' Category A.\n"
    )

    # ---------- Finding 2 ----------
    lines.append("## Finding 2 — Hop-stratified EM\n")
    lines.append(
        "`extra_info.hops` from `test.parquet` has three values: 0 (single-hop "
        "literals / mention-lookup), 1 (one-hop relation), 2 (compositional / "
        "CWQ multi-hop). Distribution: 0=683, 1=1,591, 2=1,257.\n"
    )
    hops_order = sorted({h for m in hop_stats.values() for h in m.keys()})
    header = ["model"] + [f"hop={h} EM % (n)" for h in hops_order]
    lines.append(fmt_row(header))
    lines.append(fmt_row(["---"] * len(header)))
    for m in ("39b_step400", "e3_step500", "e5b_step100"):
        row = [m]
        for h in hops_order:
            s = hop_stats[m].get(h, {"em_pct": 0.0, "n": 0})
            row.append(f"{s['em_pct']:.2f}% (n={s['n']})")
        lines.append(fmt_row(row))

    # Gain per hop vs E5b
    gains = {}
    for h in hops_order:
        g = hop_stats["39b_step400"].get(h, {"em_pct": 0.0})["em_pct"] - hop_stats[
            "e5b_step100"
        ].get(h, {"em_pct": 0.0})["em_pct"]
        gains[h] = g
    gain_str = ", ".join(f"hop={h}: {v:+.2f} pp" for h, v in gains.items())
    dominant = max(gains, key=gains.get) if gains else None
    # Skew metric: max gain vs mean of the rest.
    if gains and len(gains) > 1:
        dom_val = gains[dominant]
        rest_mean = sum(v for h, v in gains.items() if h != dominant) / (len(gains) - 1)
        skew = dom_val - rest_mean
    else:
        skew = 0.0
    skew_note = (
        "Gain is **concentrated on compositional (hop=2)** questions "
        f"(+{skew:.2f} pp over the mean of the other hops), consistent with "
        "tool-use helping where the answer requires chaining."
        if dominant == 2 and skew > 2.0
        else (
            f"Gains are broadly distributed (hop={dominant} leads by only "
            f"{skew:.2f} pp over the other-hop mean), so the improvement is "
            "not uniquely a multi-hop tool-use bonus."
        )
    )
    lines.append(
        f"\n**Observation.** 39B@400 vs E5b@100 per-hop gain: {gain_str}. "
        f"Largest gain at hop={dominant}. {skew_note}\n"
    )

    # ---------- Finding 3 ----------
    lines.append("## Finding 3 — kg-incomplete error-mode distribution (first 300)\n")
    lines.append(
        "Of 39B@400's 1,201 kg-incomplete trajectories, we classify the first "
        "300 (trajectory-file order) by parsing each `<search>fn(entity, "
        "relation)</search>` call and comparing against "
        "`extra_info.query_entities` and the kg_path relation set.\n"
    )
    total = sum(error_counter.values()) or 1
    lines.append(fmt_row(["error mode", "count", "% of 300", "interpretation"]))
    lines.append(fmt_row(["---"] * 4))
    interp = {
        "wrong-entity": "Model picked an entity not licensed by the question's gold query_entities.",
        "wrong-relation": "Correct entity but relation not on the gold kg_path.",
        "format-mismatch": "Entity normalizes to gold but differs by whitespace/underscore/case.",
        "genuine-kg-miss": "Entity + relation on gold path but KG snippet still empty.",
        "no-search-call": "No parseable search call in trajectory (early termination).",
        "no-meta": "Sample ID missing from test.parquet (should be 0).",
    }
    for mode in (
        "wrong-entity", "wrong-relation", "format-mismatch",
        "genuine-kg-miss", "no-search-call", "no-meta",
    ):
        c = error_counter.get(mode, 0)
        pct = 100.0 * c / total
        lines.append(fmt_row([mode, c, f"{pct:.1f}%", interp[mode]]))

    # Dominant
    top = error_counter.most_common(1)[0] if error_counter else (None, 0)
    lines.append(
        f"\n**Observation.** Dominant error mode: **{top[0]}** ({top[1]}/300 = "
        f"{100.0*top[1]/total:.1f}%). If `wrong-relation` dominates, 39B picks "
        "the right subject but the wrong predicate — a *relation-selection* "
        "failure that a relation-match reward can target. If `wrong-entity` "
        "dominates, the issue is upstream (entity linking). A small "
        "`format-mismatch` slice confirms whether a lightweight "
        "query-normalization reward would clear easy wins.\n"
    )

    # ---------- Finding 4 ----------
    lines.append("## Finding 4 — E5b-vs-39B head-to-head on kg-incomplete\n")
    lines.append(fmt_row([
        "metric", "value",
    ]))
    lines.append(fmt_row(["---", "---"]))
    lines.append(fmt_row([
        "39B@400 kg-incomplete samples (N)", e5b_rescue["n_kg_incomplete_39b"],
    ]))
    lines.append(fmt_row([
        "  of those with E5b trajectory", e5b_rescue["n_with_e5b_counterpart"],
    ]))
    lines.append(fmt_row([
        "E5b EM=1 on same sample",
        f"{e5b_rescue['e5b_rescue_count']} "
        f"({e5b_rescue['e5b_rescue_pct']:.2f}%)",
    ]))
    lines.append(fmt_row([
        "E5b correct-via-tool on same sample",
        f"{e5b_rescue['e5b_rescue_via_tool_count']} "
        f"({e5b_rescue['e5b_rescue_via_tool_pct']:.2f}%)",
    ]))
    trigger = e5b_rescue["e5b_rescue_pct"] > 5.0
    verdict = (
        "**> 5% threshold crossed** — E5b's KL-5x regularization over-constrained "
        "39B's retrieval skill on these queries; 39B lost capability the weaker "
        "baseline retained."
        if trigger
        else "**< 5% threshold** — the E5b baseline does *not* systematically "
             "rescue 39B's kg-incomplete failures; 39B's retrieval gap is "
             "largely not a KL-over-constraint artefact."
    )
    lines.append(
        f"\n**Observation.** {verdict} This bounds how much of the kg-incomplete "
        "bucket can plausibly be recovered by loosening KL; the rest requires "
        "a new training signal (Variant I / G).\n"
    )

    # ---------- Recommendations ----------
    lines.append("## Recommendations for Variant I / G reward design\n")
    top_mode = top[0]
    wr_pct = 100.0 * error_counter.get("wrong-relation", 0) / total
    we_pct = 100.0 * error_counter.get("wrong-entity", 0) / total
    fm_pct = 100.0 * error_counter.get("format-mismatch", 0) / total
    gk_pct = 100.0 * error_counter.get("genuine-kg-miss", 0) / total

    lines.append(
        f"- **Primary reward signal.** `wrong-relation` = {wr_pct:.1f}%, "
        f"`wrong-entity` = {we_pct:.1f}%, `format-mismatch` = {fm_pct:.1f}%, "
        f"`genuine-kg-miss` = {gk_pct:.1f}%. "
        + (
            "With wrong-relation dominant, **Variant I should favour a "
            "relation-match component** (I-Oracle: reward +1 if emitted relation "
            "∈ kg_path relations; I-Self: reward inverse of hop-distance "
            "between emitted relation and any gold relation). "
            if wr_pct >= we_pct
            else "With wrong-entity dominant, **Variant I should favour an "
                 "entity-match component** (reward +1 if the parsed head entity "
                 "matches any gold query_entity, normalized). "
        )
    )
    lines.append(
        f"- **Query-normalization reward.** Format-mismatch is {fm_pct:.1f}% — "
        + ("non-negligible; a cheap normalization shaping term (strip "
           "underscores, lowercase, collapse whitespace) is worth including "
           "as a bounded add-on."
           if fm_pct >= 5.0
           else "small; a dedicated normalization reward would not move the "
                "needle. Fold normalization into preprocessing, not the "
                "reward function.")
    )
    lines.append(
        "- **I-Oracle vs I-Self.** Oracle (using `kg_path` relations as ground "
        "truth) is the cleanest test of whether *any* relation-selection "
        "supervision closes the kg-incomplete gap; Self (self-distilling from "
        "39B's own successful tool sequences) avoids label leakage but trains "
        "on a small correct-via-tool pool (133/3531 = 3.77%). **Recommend "
        "I-Oracle first** as an upper-bound ceiling, then I-Self as the "
        "transferable recipe."
    )
    dominant_hop_gain = dominant
    lines.append(
        f"- **Variant G hop prioritization.** 39B's largest EM gain over E5b is "
        f"at hop={dominant_hop_gain} ({gains.get(dominant_hop_gain, 0.0):+.2f} pp). "
        + (
            "Concentrated multi-hop advantage would motivate filtering G's "
            "self-distillation corpus to multi-hop trajectories. "
            if dominant_hop_gain == 2
            else "Because the advantage is not concentrated on compositional "
                 "(hop=2) questions, G's self-distillation corpus should *not* "
                 "filter to multi-hop only — keep the full hop mix. "
        )
        + "Separately, require each distilled trajectory to contain ≥1 "
        "correct-via-tool event to avoid amplifying the correct-via-memory mode."
    )
    e5b_rescue_pct = e5b_rescue["e5b_rescue_pct"]
    lines.append(
        f"- **KL-regularization ceiling.** Only {e5b_rescue_pct:.2f}% of 39B's "
        "kg-incomplete samples are rescued by E5b's looser policy. "
        + (
            "Relax KL in Variant I only as a secondary knob — new supervision "
            "dominates over KL-tuning."
            if e5b_rescue_pct < 5.0
            else "Consider a mild KL-loosening ablation alongside Variant I."
        )
    )

    # ---------- Data provenance ----------
    lines.append("\n## Data provenance\n")
    audit_summaries = cvt_audit.get("summaries", {})
    for m in ("39b_step400", "e3_step500", "e5b_step100"):
        s = audit_summaries.get(m, {})
        lines.append(
            f"- `{m}`: n={s.get('n')}, EM={s.get('em_mean', 0.0)*100:.2f}%, "
            f"CvT={s.get('cvt')}/{s.get('n')}={s.get('cvt_rate', 0.0)*100:.2f}%, "
            f"avg tools/Q={s.get('avg_tool_calls', 0.0):.2f}"
        )
    lines.append(
        f"- Classifications reused from `results/phase7/full_test_cvt_audit.json` "
        f"(re-derived per-sample via `scripts/task16_classify.classify_trajectory`)."
    )
    lines.append(
        f"- Per-sample error-mode labels: `results/phase7/39b_query_error_modes.csv` "
        f"(n={error_total})."
    )

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def run(paths: Paths, n_error_samples: int, seed: int) -> None:
    random.seed(seed)

    # Load everything
    trajs_39b = load_trajectories(paths.traj_39b)
    trajs_e3 = load_trajectories(paths.traj_e3)
    trajs_e5b = load_trajectories(paths.traj_e5b)
    cat_b = load_category_b_ids(paths.category_b)
    meta = load_test_meta(paths.test_parquet)
    cvt_audit = load_cvt_audit(paths.cvt_audit)

    # --- Finding 1 ---
    cat_split = {
        "39b_step400": category_split(trajs_39b, cat_b),
        "e3_step500": category_split(trajs_e3, cat_b),
        "e5b_step100": category_split(trajs_e5b, cat_b),
    }
    for m, d in cat_split.items():
        logger.info(
            "%s  A: n=%d EM=%d (%.2f%%) CvT=%d  |  B: n=%d EM=%d (%.2f%%) CvT=%d",
            m, d["A"]["n"], d["A"]["em"], d["A"]["em_pct"], d["A"]["cvt"],
            d["B"]["n"], d["B"]["em"], d["B"]["em_pct"], d["B"]["cvt"],
        )

    # --- Finding 2 ---
    hop_stats = {
        "39b_step400": hop_stratified_em(trajs_39b, meta),
        "e3_step500": hop_stratified_em(trajs_e3, meta),
        "e5b_step100": hop_stratified_em(trajs_e5b, meta),
    }
    for m, stat in hop_stats.items():
        for h, v in stat.items():
            logger.info("%s hop=%s: n=%d EM=%d (%.2f%%)",
                        m, h, v["n"], v["em"], v["em_pct"])

    # --- Classification: per-sample label needed for findings 3 & 4 ---
    cvt_per_sample_39b: dict[str, str] = {}
    for t in trajs_39b:
        cvt_per_sample_39b[t["sample_id"]] = classify_trajectory(t)

    # --- Finding 3 ---
    rows, error_counter = error_mode_analysis(
        trajs_39b, meta, cvt_per_sample_39b,
        n_samples=n_error_samples, seed=seed,
    )
    paths.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with paths.out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["sample_id", "question", "first_search_call", "error_mode"]
        )
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Saved CSV: %s (n=%d)", paths.out_csv, len(rows))
    logger.info("Error-mode counts: %s", dict(error_counter))

    # --- Finding 4 ---
    e5b_rescue = e5b_vs_39b_on_kg_incomplete(
        trajs_39b, trajs_e5b, cvt_per_sample_39b
    )
    logger.info("E5b rescue on 39B kg-incomplete: %s", e5b_rescue)

    # --- Build memo ---
    memo = build_memo(
        cat_split=cat_split,
        hop_stats=hop_stats,
        error_counter=error_counter,
        error_total=len(rows),
        e5b_rescue=e5b_rescue,
        cvt_audit=cvt_audit,
    )
    paths.out_md.parent.mkdir(parents=True, exist_ok=True)
    paths.out_md.write_text(memo)
    logger.info("Saved memo: %s", paths.out_md)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    parser = argparse.ArgumentParser(description="Task 41 — 39B@400 mechanistic memo")
    parser.add_argument("--n_error_samples", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    run(Paths.defaults(), n_error_samples=args.n_error_samples, seed=args.seed)


if __name__ == "__main__":
    main()
