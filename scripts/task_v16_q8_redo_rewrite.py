"""V16-Q8-REDO: Bug-fixed synthetic-schema rewriter.

What changed vs `task_q8_synthetic_schema.py` (the broken version):

The broken rewriter did substring matching against a GLOBAL entity-label dictionary,
applied to ALL message content (including the system prompt and tool descriptions).
Many short Freebase labels are common English words ("you", "com", "ning"); these
got rewritten everywhere, e.g.

    "you"  -> "entity://you__905cb326"  (inside system prompt!)
    "com"  -> "entity://com__5fb552a7"  (inside "coming")
    "ning" -> "entity://nin__24d92abcg" (inside "reasoning"; ALSO nested overlap)

This file fixes the bugs per the v16 spec (see hpc_tasks.md Lane 11):

1. PER-ROW local entity set for parquet rows. Only entities in the current row's
   kg_path / query_entities / answer_entities are eligible to be rewritten in that
   row's user prompt. Cross-row consistency for the URI itself is maintained via
   `EntityRewriter.to_uri()` which returns the same hash for the same label.

2. WHOLE-WORD regex matching only. The broken version did `str.replace(label, uri)`
   which catches substrings. We use `re.compile(r"\\b" + re.escape(label) + r"\\b")`.

3. SYSTEM PROMPT IS NEVER TOUCHED. We only rewrite:
     - prompt[i].content where role == "user"
     - extra_info.kg_path, extra_info.query_entities, extra_info.answer_entities,
       extra_info.gold_answer_short, extra_info.all_answers
     - extra_info.tools_kwargs[*].create_kwargs.{gold_answer,kg_path}
     - reward_model.ground_truth
     - SFT trajectories: ONLY user/assistant role content (system role is preserved
       byte-for-byte). Important: in the CWQ SFT format, the user role contains
       BOTH the question AND `<tool_response>...</tool_response>` payloads; both
       legitimately contain entity strings and we want to rewrite both.

4. MIN-LENGTH FILTER (>= 3 chars). The mapping itself can store any label, but the
   regex applied to text only includes labels of length >= 3 to avoid catching
   "I", "a", "of", "in", "on" that may appear in entity slots due to CWQ noise.

5. SFT trajectories: the SFT JSONL has no row-ID linking it back to a parquet row,
   so we cannot use a strict per-row local set. Instead we compile ONE regex from
   the union of all eligible labels (length >= 3) collected from parquet KG_PATH
   slots only — this is the "entity universe" of legitimate KG entities. Tool
   responses and assistant messages get whole-word matched against that union.
   The system prompt is left identical.

Cross-row consistency: `EntityRewriter` is keyed on the original entity string and
the URI is `entity://<safe_label>__<sha1[:8]>` — deterministic, identical across all
rows and across SFT trajectories. The `entity_mapping.json` records both directions.

Run as a SLURM CPU job (login-node rule). Single-CPU, well under 1h on full data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
import re
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any, Iterable

logger = logging.getLogger(__name__)

MID_PATTERN = re.compile(r"^[mg]\.[0-9a-z_]+$")
LABEL_SANITIZE = re.compile(r"[^A-Za-z0-9_]")
MAX_LABEL_LEN = 40
MIN_LABEL_LEN_FOR_TEXT_REPLACE = 3  # don't whole-word-match labels < 3 chars

SYSTEM_PROMPT_FINGERPRINT = "You are a knowledge graph reasoning agent."


def make_safe_label(original: str) -> str:
    if MID_PATTERN.match(original):
        return "mid"
    label = original.strip().lower().replace(" ", "_")
    label = LABEL_SANITIZE.sub("", label)
    if not label:
        label = "ent"
    return label[:MAX_LABEL_LEN]


def make_uri(original: str) -> str:
    label = make_safe_label(original)
    h = hashlib.sha1(original.encode("utf-8")).hexdigest()[:8]
    return f"entity://{label}__{h}"


class EntityRewriter:
    """Deterministic, cross-row-consistent original->URI mapping.

    The mapping itself accepts ANY entity string (we want a complete record for
    traceability), but `compile_text_regex()` only includes entries of length
    >= MIN_LABEL_LEN_FOR_TEXT_REPLACE for actual whole-word substitution.
    """

    def __init__(self) -> None:
        self.mapping: OrderedDict[str, str] = OrderedDict()
        self._reverse: dict[str, str] = {}

    def to_uri(self, original: str) -> str:
        if not isinstance(original, str) or not original:
            return original
        if original in self.mapping:
            return self.mapping[original]
        uri = make_uri(original)
        if uri in self._reverse and self._reverse[uri] != original:
            uri = uri + hashlib.sha1(original.encode("utf-8")).hexdigest()[8:14]
        self.mapping[original] = uri
        self._reverse[uri] = original
        return uri

    def register(self, originals: Iterable[str]) -> None:
        for o in originals:
            if isinstance(o, str) and o:
                self.to_uri(o)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        len_dist: dict[int, int] = {}
        for o in self.mapping:
            ln = len(o)
            len_dist[ln] = len_dist.get(ln, 0) + 1
        payload = {
            "original_to_uri": dict(self.mapping),
            "uri_to_original": {v: k for k, v in self.mapping.items()},
            "n_entities": len(self.mapping),
            "label_length_distribution": dict(sorted(len_dist.items())),
            "min_label_len_for_text_replace": MIN_LABEL_LEN_FOR_TEXT_REPLACE,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        logger.info("Wrote entity mapping (%d entities) -> %s", len(self.mapping), path)


# -----------------------------------------------------------------------------
# Per-row whole-word rewriter
# -----------------------------------------------------------------------------

def _build_local_regex(local_labels: Iterable[str]) -> re.Pattern[str] | None:
    """Compile a single whole-word regex from a list of labels (len >= MIN)."""
    eligible = sorted(
        {lbl for lbl in local_labels
         if isinstance(lbl, str) and len(lbl) >= MIN_LABEL_LEN_FOR_TEXT_REPLACE},
        key=len,
        reverse=True,
    )
    if not eligible:
        return None
    parts = [r"\b" + re.escape(lbl) + r"\b" for lbl in eligible]
    return re.compile("|".join(parts))


def _whole_word_replace(text: str, pattern: re.Pattern[str], rw: EntityRewriter) -> str:
    """Apply whole-word regex replacement; each match resolved via rw.mapping."""
    if not text or pattern is None:
        return text

    def repl(m: re.Match[str]) -> str:
        orig = m.group(0)
        return rw.mapping.get(orig, orig)

    return pattern.sub(repl, text)


def rewrite_row(row: dict[str, Any], rw: EntityRewriter) -> dict[str, Any]:
    extra = row.get("extra_info") or {}

    q_entities = list(extra.get("query_entities") or [])
    a_entities = list(extra.get("answer_entities") or [])
    all_answers = list(extra.get("all_answers") or [])
    kg_path = list(extra.get("kg_path") or [])
    gold_answer_short = extra.get("gold_answer_short") or ""

    # Build PER-ROW local entity set (this is the central bug fix)
    local_entities: set[str] = set()
    for e in q_entities + a_entities + all_answers:
        if isinstance(e, str) and e:
            local_entities.add(e)
    if isinstance(gold_answer_short, str) and gold_answer_short:
        local_entities.add(gold_answer_short)
    for triple in kg_path:
        if len(triple) >= 3:
            if isinstance(triple[0], str):
                local_entities.add(triple[0])
            if isinstance(triple[2], str):
                local_entities.add(triple[2])
            # triple[1] is a relation -- NOT rewritten

    # Register every local entity in the global mapping for cross-row consistency
    rw.register(local_entities)

    # Compile a whole-word regex restricted to THIS row's local entities
    local_pat = _build_local_regex(local_entities)

    # Rewrite structured slots: these are exact-match (single-entity strings),
    # NOT regex; we just look them up directly.
    new_q_entities = [rw.to_uri(e) for e in q_entities]
    new_a_entities = [rw.to_uri(e) for e in a_entities]
    new_all_answers = [rw.to_uri(e) for e in all_answers]
    new_gold_short = rw.to_uri(gold_answer_short) if gold_answer_short else ""
    new_kg_path = [
        [rw.to_uri(str(t[0])), str(t[1]), rw.to_uri(str(t[2]))]
        for t in kg_path
        if len(t) >= 3
    ]

    # tools_kwargs.kg_query.create_kwargs (used by reward fn at training time)
    tools_kwargs = extra.get("tools_kwargs") or {}
    new_tools_kwargs: dict[str, Any] = {}
    for tk_name, tk_val in tools_kwargs.items():
        ck = (tk_val or {}).get("create_kwargs", {}) if isinstance(tk_val, dict) else {}
        new_ck = dict(ck)
        if "gold_answer" in new_ck and new_ck["gold_answer"]:
            new_ck["gold_answer"] = rw.to_uri(new_ck["gold_answer"])
        if "kg_path" in new_ck and isinstance(new_ck["kg_path"], list):
            new_ck["kg_path"] = [
                [rw.to_uri(str(t[0])), str(t[1]), rw.to_uri(str(t[2]))]
                for t in new_ck["kg_path"]
                if len(t) >= 3
            ]
        new_tools_kwargs[tk_name] = {**(tk_val or {}), "create_kwargs": new_ck}

    new_extra = dict(extra)
    new_extra["query_entities"] = new_q_entities
    new_extra["answer_entities"] = new_a_entities
    new_extra["all_answers"] = new_all_answers
    new_extra["gold_answer_short"] = new_gold_short
    new_extra["kg_path"] = new_kg_path
    new_extra["tools_kwargs"] = new_tools_kwargs
    new_extra["q8_originals"] = {
        "query_entities": q_entities,
        "answer_entities": a_entities,
        "all_answers": all_answers,
        "gold_answer_short": gold_answer_short,
    }

    rm = dict(row.get("reward_model") or {})
    if rm.get("ground_truth"):
        rm["ground_truth"] = rw.to_uri(rm["ground_truth"])

    # User prompt: whole-word replace using the PER-ROW local regex.
    # System prompt is preserved untouched.
    new_prompt: list[dict[str, str]] = []
    for msg in row["prompt"]:
        m = dict(msg)
        if m.get("role") == "user":
            m["content"] = _whole_word_replace(m.get("content", "") or "",
                                               local_pat, rw)
        # else: role=='system' (or anything else) -> untouched
        new_prompt.append(m)

    out = dict(row)
    out["prompt"] = new_prompt
    out["extra_info"] = new_extra
    out["reward_model"] = rm
    return out


# -----------------------------------------------------------------------------
# Parquet driver
# -----------------------------------------------------------------------------

def rewrite_parquet(in_path: Path, out_path: Path, rw: EntityRewriter) -> int:
    import pyarrow as pa
    import pyarrow.parquet as pq

    logger.info("Reading %s ...", in_path)
    table = pq.read_table(str(in_path))
    rows = table.to_pylist()
    logger.info("  %d rows; rewriting ...", len(rows))
    new_rows = [rewrite_row(r, rw) for r in rows]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    new_table = pa.Table.from_pylist(new_rows)
    pq.write_table(new_table, str(out_path))
    logger.info("  wrote %s (%d rows)", out_path, len(new_rows))
    return len(new_rows)


# -----------------------------------------------------------------------------
# SFT trajectory rewriter
# -----------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


def _build_first_word_index(labels: Iterable[str]) -> dict[str, list[str]]:
    """Build first-word inverted index for fast candidate lookup.

    Each label is keyed by its first whitespace-separated token (lowercased).
    For length-1 labels, the whole label IS the first word.
    Only labels of length >= MIN_LABEL_LEN_FOR_TEXT_REPLACE are indexed.
    """
    idx: dict[str, list[str]] = {}
    for lbl in labels:
        if not isinstance(lbl, str) or len(lbl) < MIN_LABEL_LEN_FOR_TEXT_REPLACE:
            continue
        # First whitespace-separated word, lowercased
        fw_match = _TOKEN_RE.match(lbl.split()[0].lower() if lbl.strip() else "")
        if not fw_match:
            continue
        fw = fw_match.group(0)
        if not fw:
            continue
        idx.setdefault(fw, []).append(lbl)
    return idx


def _candidate_labels_for_text(text: str,
                               first_word_idx: dict[str, list[str]]) -> list[str]:
    """Return all labels that COULD match in `text` based on first-word presence.

    O(len(text)) tokenization + O(unique_tokens) dict lookups.
    Multi-word labels still need a final whole-word regex pass to verify.
    """
    if not text:
        return []
    seen_tokens: set[str] = set()
    for m in _TOKEN_RE.finditer(text):
        tok = m.group(0).lower()
        seen_tokens.add(tok)
    cands: list[str] = []
    for tok in seen_tokens:
        bucket = first_word_idx.get(tok)
        if bucket:
            cands.extend(bucket)
    return cands


def rewrite_sft_trajectories(in_path: Path,
                             out_path: Path,
                             rw: EntityRewriter) -> int:
    """Rewrite user/assistant/tool content with PER-TRAJECTORY scoped whole-word regex.

    System messages are preserved BYTE-FOR-BYTE (no rewriting whatsoever).

    The previous implementation built a single 746K-entity global alternation regex
    and applied it to every message; that pass is O(trajectories * 746K) and timed
    out at ~50%/h on the full 5000-row CWQ SFT set.

    The fix here:
      1. Build a `first_word -> [labels]` inverted index ONCE (~1s) from the global
         entity mapping (`rw.mapping`).
      2. For each trajectory, concatenate the non-system content, tokenize once,
         intersect tokens with the first-word index to get a SMALL candidate
         label set (~tens, occasionally low hundreds).
      3. Compile a tiny whole-word regex from THAT candidate set only and apply
         it to every non-system message in the trajectory.

    Net cost: O(sum-of-text-len + candidates_per_traj) per trajectory. Empirically
    runs in 5-10 min for 5000 trajectories on a single CPU.

    Note: SFT JSONL has no `sample_id` linking to parquet rows; we verified this
    by inspecting the file (only key per record is `trajectory`). So we MUST scan
    the text. The first-word inverted index is the cheapest correct approach.
    Output records preserve every input field (just with rewritten `trajectory`).
    """
    logger.info("Reading SFT trajectories %s ...", in_path)
    with open(in_path, "r", encoding="utf-8") as f:
        lines = [ln for ln in f if ln.strip()]
    logger.info("  %d trajectories", len(lines))

    logger.info("Building first-word inverted index over %d global entities ...",
                len(rw.mapping))
    fw_idx = _build_first_word_index(rw.mapping.keys())
    logger.info("  index size: %d unique first-words", len(fw_idx))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_written = 0
    n_replaced_total = 0
    n_unchanged_system = 0
    n_system_messages = 0
    cand_size_sum = 0

    with open(out_path, "w", encoding="utf-8") as out_f:
        for line in lines:
            obj = json.loads(line)
            traj = obj.get("trajectory", [])

            # Collect all non-system content for THIS trajectory in one pass
            joined_nonsys = "\n".join(
                (msg.get("content") or "")
                for msg in traj
                if msg.get("role") != "system"
            )
            cand_labels = _candidate_labels_for_text(joined_nonsys, fw_idx)
            cand_size_sum += len(cand_labels)
            local_pat = _build_local_regex(cand_labels) if cand_labels else None

            new_traj: list[dict[str, str]] = []
            for msg in traj:
                role = msg.get("role")
                content = msg.get("content", "") or ""
                if role == "system":
                    n_system_messages += 1
                    n_unchanged_system += 1
                    new_traj.append({**msg})
                else:
                    if local_pat is not None and content:
                        new_content, n = local_pat.subn(
                            lambda m: rw.mapping.get(m.group(0), m.group(0)),
                            content,
                        )
                        n_replaced_total += n
                    else:
                        new_content = content
                    new_traj.append({**msg, "content": new_content})

            # Preserve every other field of the input record (sample_id, question,
            # gold_answer if present, etc.) — only `trajectory` is replaced.
            out_record = {**obj, "trajectory": new_traj}
            out_f.write(json.dumps(out_record, ensure_ascii=False) + "\n")
            n_written += 1
            if n_written % 500 == 0:
                avg_cands = cand_size_sum / max(n_written, 1)
                logger.info("  ... %d/%d trajectories rewritten "
                            "(%d total replacements; avg %.1f candidates/traj)",
                            n_written, len(lines), n_replaced_total, avg_cands)

    avg_cands = cand_size_sum / max(n_written, 1)
    logger.info("Wrote %d SFT trajectories with %d total whole-word replacements -> %s",
                n_written, n_replaced_total, out_path)
    logger.info("Avg candidate labels per trajectory: %.1f", avg_cands)
    logger.info("System messages preserved byte-for-byte: %d/%d",
                n_unchanged_system, n_system_messages)
    return n_written


def _load_entity_mapping_into_rewriter(mapping_path: Path,
                                       rw: EntityRewriter) -> None:
    """Populate `rw.mapping` from a previously-saved entity_mapping.json.

    Used by --sft_only mode to avoid re-running the parquet pass when the
    parquet rewrite already succeeded but the SFT pass timed out.
    """
    logger.info("Loading existing entity mapping from %s ...", mapping_path)
    with open(mapping_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    o2u = payload["original_to_uri"]
    rw.mapping = OrderedDict(o2u.items())
    rw._reverse = {v: k for k, v in o2u.items()}
    logger.info("  loaded %d entities into rewriter", len(rw.mapping))


# -----------------------------------------------------------------------------
# Sanity-check report
# -----------------------------------------------------------------------------

def write_sanity_report(
    report_path: Path,
    *,
    in_dir: Path,
    out_dir: Path,
    sft_in: Path,
    sft_out: Path,
    mapping_path: Path,
    seed: int = 13,
) -> bool:
    """Generate the markdown sanity-check report. Returns True iff all checks pass."""
    import pyarrow.parquet as pq

    report_path.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    all_passed = True
    lines: list[str] = []
    lines.append("# V16-Q8-REDO Sanity-Check Report\n")
    lines.append(f"- Input parquet dir: `{in_dir}`")
    lines.append(f"- Output parquet dir: `{out_dir}`")
    lines.append(f"- Input SFT: `{sft_in}`")
    lines.append(f"- Output SFT: `{sft_out}`")
    lines.append(f"- Entity mapping: `{mapping_path}`")
    lines.append("")

    # ---- check (a): 5 random side-by-side parquet rows ----
    lines.append("## (a) Parquet user-prompt rewrite — 5 random rows\n")
    src = pq.read_table(str(in_dir / "test.parquet")).to_pylist()
    dst = pq.read_table(str(out_dir / "test.parquet")).to_pylist()
    assert len(src) == len(dst), "row count mismatch"
    n = len(src)
    sample_idx = sorted(rng.sample(range(n), k=min(5, n)))
    for idx in sample_idx:
        s_user = next((p["content"] for p in src[idx]["prompt"]
                       if p["role"] == "user"), "")
        d_user = next((p["content"] for p in dst[idx]["prompt"]
                       if p["role"] == "user"), "")
        lines.append(f"### Row {idx}")
        lines.append("**Original user prompt:**")
        lines.append("```")
        lines.append(s_user)
        lines.append("```")
        lines.append("**Rewritten user prompt:**")
        lines.append("```")
        lines.append(d_user)
        lines.append("```")
        lines.append("")

    # ---- check (b): 5 random SFT trajectories side-by-side; system is identical ----
    lines.append("## (b) SFT trajectories — 5 random samples\n")
    with open(sft_in, "r", encoding="utf-8") as f:
        src_sft = [json.loads(ln) for ln in f if ln.strip()]
    with open(sft_out, "r", encoding="utf-8") as f:
        dst_sft = [json.loads(ln) for ln in f if ln.strip()]
    assert len(src_sft) == len(dst_sft), "SFT row count mismatch"
    sft_idx = sorted(rng.sample(range(len(src_sft)), k=min(5, len(src_sft))))
    sys_byte_for_byte = True
    for idx in sft_idx:
        s_traj = src_sft[idx]["trajectory"]
        d_traj = dst_sft[idx]["trajectory"]
        lines.append(f"### SFT row {idx}")
        for s_msg, d_msg in zip(s_traj, d_traj):
            role = s_msg.get("role")
            s_c = s_msg.get("content", "")
            d_c = d_msg.get("content", "")
            if role == "system" and s_c != d_c:
                sys_byte_for_byte = False
            lines.append(f"**[{role}] original:**")
            lines.append("```")
            lines.append(s_c)
            lines.append("```")
            lines.append(f"**[{role}] rewritten:**")
            lines.append("```")
            lines.append(d_c)
            lines.append("```")
            lines.append("")

    # ---- check (c): mapping counts + label-length distribution ----
    with open(mapping_path, "r", encoding="utf-8") as f:
        mp = json.load(f)
    n_entities = mp["n_entities"]
    len_dist = mp["label_length_distribution"]
    lines.append("## (c) Entity mapping summary\n")
    lines.append(f"- Total unique entities mapped: **{n_entities}**")
    lines.append(f"- Min label length used for text replacement: "
                 f"**{mp['min_label_len_for_text_replace']}**")
    lines.append("- Label length distribution:")
    lines.append("")
    lines.append("| length | count |")
    lines.append("|--------|-------|")
    for k_str in sorted(len_dist.keys(), key=lambda x: int(x)):
        lines.append(f"| {k_str} | {len_dist[k_str]} |")
    lines.append("")
    n_too_short = sum(c for k, c in len_dist.items() if int(k) < 3)
    lines.append(f"- Entities with label length < 3 (NOT used for text replace): "
                 f"**{n_too_short}**")
    lines.append("")

    # ---- check (d): system prompt fingerprint preserved in SFT output ----
    fingerprint_found_in_output = False
    fingerprint_idx = -1
    for i, obj in enumerate(dst_sft):
        for msg in obj["trajectory"]:
            if msg.get("role") == "system" and SYSTEM_PROMPT_FINGERPRINT in msg.get("content", ""):
                fingerprint_found_in_output = True
                fingerprint_idx = i
                break
        if fingerprint_found_in_output:
            break

    lines.append("## (d) No-corruption: system prompt fingerprint check\n")
    lines.append(f"- Searching for fingerprint: `{SYSTEM_PROMPT_FINGERPRINT!r}`")
    if fingerprint_found_in_output:
        lines.append(f"- Fingerprint FOUND unchanged in rewritten SFT row {fingerprint_idx} system message: PASS")
    else:
        lines.append("- Fingerprint NOT found in any system message: FAIL")
        all_passed = False
    lines.append("")

    # Also assert byte-for-byte for the 5 sampled SFT rows
    lines.append(f"- All sampled SFT system messages byte-for-byte identical to original: "
                 f"{'PASS' if sys_byte_for_byte else 'FAIL'}")
    if not sys_byte_for_byte:
        all_passed = False
    lines.append("")

    # ---- check (e): whole-word check for "Lou Seal" ----
    lines.append("## (e) Whole-word check on probe entity\n")
    probe_entity = "Lou Seal"
    probe_uri = mp["original_to_uri"].get(probe_entity)
    lines.append(f"- Probe entity: `{probe_entity!r}`  ->  `{probe_uri}`")

    if probe_uri is None:
        lines.append("- Probe entity NOT in mapping; cannot run check: FAIL")
        all_passed = False
    else:
        # Find any SFT message that contains the probe entity in the ORIGINAL,
        # then check the OUTPUT replaced exactly the whole-word matches.
        probe_re = re.compile(r"\b" + re.escape(probe_entity) + r"\b")
        matched_examples: list[tuple[str, str]] = []
        for s_obj, d_obj in zip(src_sft, dst_sft):
            for s_msg, d_msg in zip(s_obj["trajectory"], d_obj["trajectory"]):
                s_c = s_msg.get("content", "")
                if probe_entity in s_c:
                    d_c = d_msg.get("content", "")
                    matched_examples.append((s_c, d_c))
                    if len(matched_examples) >= 3:
                        break
            if len(matched_examples) >= 3:
                break

        whole_word_ok = True
        if not matched_examples:
            lines.append(f"- No SFT message contained `{probe_entity!r}` (probe not exercised — searching parquet)")
            # fallback: probe parquet user prompts
            for s_row, d_row in zip(src, dst):
                s_user = next((p["content"] for p in s_row["prompt"]
                               if p["role"] == "user"), "")
                d_user = next((p["content"] for p in d_row["prompt"]
                               if p["role"] == "user"), "")
                if probe_entity in s_user:
                    matched_examples.append((s_user, d_user))
                    if len(matched_examples) >= 3:
                        break

        for s_c, d_c in matched_examples:
            # In the original, count whole-word matches of "Lou Seal".
            n_whole = len(probe_re.findall(s_c))
            # In the rewritten content, count occurrences of probe_uri.
            n_uri = d_c.count(probe_uri)
            # In the rewritten content, the bare phrase "Lou Seal" should NOT
            # appear unless it was a substring (e.g. "Lou Sealskin"); since
            # CWQ doesn't have such, we expect 0.
            n_bare_remaining = len(probe_re.findall(d_c))
            ok = (n_whole == n_uri) and (n_bare_remaining == 0)
            if not ok:
                whole_word_ok = False
            lines.append("")
            lines.append("```")
            lines.append("ORIG: " + s_c[:300])
            lines.append("NEW : " + d_c[:300])
            lines.append(f"  whole_word_matches_in_orig={n_whole}, "
                         f"uri_count_in_new={n_uri}, "
                         f"bare_remaining={n_bare_remaining}, "
                         f"PASS={ok}")
            lines.append("```")
        if matched_examples:
            lines.append("")
            lines.append(f"- Whole-word probe overall: "
                         f"{'PASS' if whole_word_ok else 'FAIL'}")
            if not whole_word_ok:
                all_passed = False
        else:
            lines.append("- Probe entity not found anywhere; check skipped")

    # ---- check (f): no nested-replacement corruption (e.g. 'entity://...entity://...') ----
    lines.append("")
    lines.append("## (f) Nested-replacement corruption scan\n")
    nested_pat = re.compile(r"entity://[^\s]*entity://")
    n_nested = 0
    for d_obj in dst_sft:
        for msg in d_obj["trajectory"]:
            if nested_pat.search(msg.get("content", "")):
                n_nested += 1
    for d_row in dst:
        for p in d_row["prompt"]:
            if nested_pat.search(p.get("content", "")):
                n_nested += 1
    lines.append(f"- Nested 'entity://...entity://' occurrences in output: **{n_nested}**")
    lines.append(f"- Result: {'PASS' if n_nested == 0 else 'FAIL'}")
    if n_nested != 0:
        all_passed = False

    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append(f"## Overall: {'ALL PASS' if all_passed else 'FAILURE'}")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    logger.info("Wrote sanity report -> %s (overall %s)",
                report_path, "PASS" if all_passed else "FAIL")
    return all_passed


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    parser = argparse.ArgumentParser(description="V16-Q8-REDO synthetic-schema rewrite (bug-fixed)")
    parser.add_argument("--input_dir", type=Path,
                        default=Path("data/freebase/verl_cwq"))
    parser.add_argument("--output_dir", type=Path,
                        default=Path("data/freebase/verl_cwq_q8_redo"))
    parser.add_argument("--sft_input", type=Path,
                        default=Path("data/freebase/sft_trajectories.jsonl"))
    parser.add_argument("--sft_output", type=Path,
                        default=Path("data/freebase/sft_trajectories_q8_redo.jsonl"))
    parser.add_argument("--report_path", type=Path,
                        default=Path("results/phase7/v16_q8_redo_sanity_report.md"))
    parser.add_argument("--skip_sft", "--skip-sft", action="store_true")
    parser.add_argument("--sft_only", "--sft-only", action="store_true",
                        help="Skip parquet rewrite; load existing entity_mapping.json "
                             "and only rewrite the SFT trajectories. Use when parquet "
                             "outputs already exist (e.g. previous job timed out in "
                             "the SFT pass).")
    parser.add_argument("--produce_sanity_report", "--produce-sanity-report",
                        action="store_true",
                        help="Force sanity report generation at the end.")
    parser.add_argument("--sanity_check", "--sanity-check", action="store_true",
                        help="Run ONLY the sanity check (assumes rewrite already done).")
    args = parser.parse_args()

    mapping_path = args.output_dir / "entity_mapping.json"

    if args.sanity_check:
        # Pure sanity-check pass; no rewrite work
        ok = write_sanity_report(
            args.report_path,
            in_dir=args.input_dir,
            out_dir=args.output_dir,
            sft_in=args.sft_input,
            sft_out=args.sft_output,
            mapping_path=mapping_path,
        )
        if not ok:
            logger.error("=== SANITY CHECK FAILED — see %s ===", args.report_path)
            sys.exit(2)
        logger.info("=== V16-Q8-REDO sanity check ALL PASS ===")
        return

    rw = EntityRewriter()

    if args.sft_only:
        if not mapping_path.exists():
            logger.error("--sft_only requires existing entity mapping at %s",
                         mapping_path)
            sys.exit(2)
        _load_entity_mapping_into_rewriter(mapping_path, rw)
        # Re-save (no-op overwrite) so timestamps reflect the SFT-only re-run; this
        # also serves as a sanity check that the file is well-formed for the
        # downstream sanity report (label-length distribution, etc.).
        rw.save(mapping_path)
    else:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        n_total = 0
        for split in ("train", "val", "test"):
            in_path = args.input_dir / f"{split}.parquet"
            out_path = args.output_dir / f"{split}.parquet"
            if not in_path.exists():
                logger.warning("Missing %s — skipping", in_path)
                continue
            n_total += rewrite_parquet(in_path, out_path, rw)
        rw.save(mapping_path)
        logger.info("Total parquet rows rewritten: %d", n_total)
        logger.info("Total unique entities mapped: %d", len(rw.mapping))

    if not args.skip_sft:
        if args.sft_input.exists():
            rewrite_sft_trajectories(args.sft_input, args.sft_output, rw)
        else:
            logger.warning("SFT input %s does not exist — skipping", args.sft_input)

    # Run sanity check (always — produce_sanity_report flag is for clarity in CLI)
    ok = write_sanity_report(
        args.report_path,
        in_dir=args.input_dir,
        out_dir=args.output_dir,
        sft_in=args.sft_input,
        sft_out=args.sft_output,
        mapping_path=mapping_path,
    )
    if not ok:
        logger.error("=== SANITY CHECK FAILED — see %s ===", args.report_path)
        sys.exit(2)
    logger.info("=== V16-Q8-REDO rewrite + sanity check ALL PASS ===")


if __name__ == "__main__":
    main()
