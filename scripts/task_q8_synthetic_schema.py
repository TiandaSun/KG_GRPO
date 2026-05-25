"""V15-Q8: Rewrite CWQ/Freebase entity strings with synthetic English-readable URIs.

This script implements Lane 8 of the v15 plan: a non-KG synthetic-schema control.
It removes the linguistic-familiarity confound (L_lang) of Freebase entity names while
keeping the L_sig (silent failure profile), L_comp (opaque-MID compositional state)
and L_prior (prior exposure) attack surfaces.

What this does:
1. Reads data/freebase/verl_cwq/{train,val,test}.parquet
2. Builds a deterministic mapping {original_entity_string -> synthetic_uri}
3. Synthetic URI format: entity://<safe_label>__<short_hash>
   - safe_label: lowercased, alphanum-only, spaces->underscores, truncated
   - short_hash: sha1(original)[:8] -> guarantees uniqueness even for collisions
4. Rewrites entity references in:
   - extra_info.kg_path (head + tail of each triple)
   - extra_info.query_entities, extra_info.answer_entities
   - extra_info.gold_answer_short, extra_info.all_answers
   - extra_info.tools_kwargs.kg_query.create_kwargs.gold_answer + kg_path
   - reward_model.ground_truth
   - prompt[1].content (user message — substring replace each query entity with URI)
5. Writes out:
   - data/freebase/verl_cwq_q8_synthetic/{train,val,test}.parquet
   - data/freebase/verl_cwq_q8_synthetic/entity_mapping.json
6. Also rewrites SFT trajectories at data/freebase/sft_trajectories.jsonl
   into data/freebase/sft_trajectories_q8_synthetic.jsonl by string-replacing each
   known original entity with its synthetic URI inside every message.content.

Relations are NOT rewritten — relations are already English (people.person.spouse) and
the L_lang attack we are probing is the OPAQUE-ID problem on entities specifically.

Run as a SLURM job (per CLAUDE.md no-login-node rule). CPU-only, ~30 min for 5K SFT
trajectories + 35K parquet rows.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
from collections import OrderedDict
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Entities that look like Freebase MIDs (m.0xxx, g.0xxx, m.012345)
MID_PATTERN = re.compile(r"^[mg]\.[0-9a-z_]+$")
# Disallowed chars in safe_label
LABEL_SANITIZE = re.compile(r"[^A-Za-z0-9_]")

MAX_LABEL_LEN = 40  # truncate the readable portion


def make_safe_label(original: str) -> str:
    """Generate a safe URI-friendly label from an entity string.

    - Lowercase
    - Spaces -> underscores
    - Strip non-alphanumeric (except underscore)
    - Truncate to MAX_LABEL_LEN
    - For MIDs, label is just "mid" (the hash provides uniqueness)
    """
    if MID_PATTERN.match(original):
        return "mid"
    label = original.strip().lower().replace(" ", "_")
    label = LABEL_SANITIZE.sub("", label)
    if not label:
        label = "ent"
    return label[:MAX_LABEL_LEN]


def make_uri(original: str) -> str:
    """Build the synthetic URI for an original entity string."""
    label = make_safe_label(original)
    h = hashlib.sha1(original.encode("utf-8")).hexdigest()[:8]
    return f"entity://{label}__{h}"


class EntityRewriter:
    """Tracks {original -> synthetic_uri} mapping across all rows."""

    def __init__(self) -> None:
        self.mapping: OrderedDict[str, str] = OrderedDict()
        self._reverse: dict[str, str] = {}

    def to_uri(self, original: str) -> str:
        if not isinstance(original, str) or not original:
            return original
        if original in self.mapping:
            return self.mapping[original]
        uri = make_uri(original)
        # Collision check: if URI already used for a DIFFERENT original, append more hash
        if uri in self._reverse and self._reverse[uri] != original:
            uri = uri + hashlib.sha1(original.encode("utf-8")).hexdigest()[8:14]
        self.mapping[original] = uri
        self._reverse[uri] = original
        return uri

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "uri_to_original": {v: k for k, v in self.mapping.items()},
                    "original_to_uri": dict(self.mapping),
                    "n_entities": len(self.mapping),
                },
                f,
                indent=2,
            )
        logger.info("Wrote entity mapping (%d entities) -> %s", len(self.mapping), path)

    def load_originals_first(self, originals: list[str]) -> None:
        """Pre-populate mapping by visiting `originals` in order."""
        for o in originals:
            if o:
                self.to_uri(o)


def rewrite_row(row: dict[str, Any], rw: EntityRewriter) -> dict[str, Any]:
    """Apply synthetic-URI rewriting to a single verl parquet row."""
    extra = row.get("extra_info") or {}

    # First, register every entity that appears in this row
    q_entities = list(extra.get("query_entities") or [])
    a_entities = list(extra.get("answer_entities") or [])
    all_answers = list(extra.get("all_answers") or [])
    kg_path = list(extra.get("kg_path") or [])
    gold_answer_short = extra.get("gold_answer_short") or ""

    for e in q_entities + a_entities + all_answers + [gold_answer_short]:
        if isinstance(e, str) and e:
            rw.to_uri(e)
    for triple in kg_path:
        if len(triple) >= 3:
            rw.to_uri(str(triple[0]))
            rw.to_uri(str(triple[2]))
            # Relation (triple[1]) is NOT rewritten — already English

    # Build new fields
    new_q_entities = [rw.to_uri(e) for e in q_entities]
    new_a_entities = [rw.to_uri(e) for e in a_entities]
    new_all_answers = [rw.to_uri(e) for e in all_answers]
    new_gold_short = rw.to_uri(gold_answer_short) if gold_answer_short else ""
    new_kg_path = [
        [rw.to_uri(str(t[0])), str(t[1]), rw.to_uri(str(t[2]))]
        for t in kg_path
        if len(t) >= 3
    ]

    # Rewrite tools_kwargs.kg_query.create_kwargs (used by reward fn)
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

    # Build new extra_info
    new_extra = dict(extra)
    new_extra["query_entities"] = new_q_entities
    new_extra["answer_entities"] = new_a_entities
    new_extra["all_answers"] = new_all_answers
    new_extra["gold_answer_short"] = new_gold_short
    new_extra["kg_path"] = new_kg_path
    new_extra["tools_kwargs"] = new_tools_kwargs
    # Preserve original info for traceability
    new_extra["q8_originals"] = {
        "query_entities": q_entities,
        "answer_entities": a_entities,
        "all_answers": all_answers,
        "gold_answer_short": gold_answer_short,
    }

    # Rewrite reward_model.ground_truth
    rm = dict(row.get("reward_model") or {})
    if rm.get("ground_truth"):
        rm["ground_truth"] = rw.to_uri(rm["ground_truth"])

    # Rewrite the user prompt: substring-replace original query entity strings with URIs
    new_prompt: list[dict[str, str]] = []
    for msg in row["prompt"]:
        m = dict(msg)
        if m.get("role") == "user":
            content = m.get("content", "")
            # Sort by length desc so longer entities are replaced first (avoid partial overlap)
            for orig in sorted(q_entities, key=len, reverse=True):
                if orig and orig in content:
                    content = content.replace(orig, rw.to_uri(orig))
            m["content"] = content
        new_prompt.append(m)

    out = dict(row)
    out["prompt"] = new_prompt
    out["extra_info"] = new_extra
    out["reward_model"] = rm
    return out


def rewrite_parquet(in_path: Path, out_path: Path, rw: EntityRewriter) -> int:
    """Read parquet, rewrite each row, write out."""
    import pyarrow.parquet as pq
    import pyarrow as pa

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


def rewrite_sft_trajectories(in_path: Path, out_path: Path, rw: EntityRewriter) -> int:
    """Apply entity rewriting to a JSONL of {trajectory: [{role, content}]}.

    Strategy: the entity mapping has been populated by the parquet rewrite. For each
    message.content, do longest-first substring replacement of every known original
    entity. Quoted forms (e.g. `"Lou Seal"` or `Lou Seal,`) are handled by plain
    str.replace because we walk the entire content as a single string.
    """
    logger.info("Reading SFT trajectories %s ...", in_path)
    with open(in_path, "r", encoding="utf-8") as f:
        lines = [ln for ln in f if ln.strip()]
    logger.info("  %d trajectories; building sorted-by-length entity list ...", len(lines))

    # Sort once (longest first); only consider strings >= 3 chars to limit accidental
    # substring matches against tokens like "a", "of", etc.
    originals_long = [e for e in rw.mapping.keys() if len(e) >= 3]
    originals_long.sort(key=len, reverse=True)
    logger.info("  considering %d originals for substring rewrite", len(originals_long))

    # Optimization: index originals by their first 4 chars so we only try replacements
    # whose prefix actually appears in the message. This is critical for runtime —
    # naive O(N_msgs * N_entities) on 5K * 1.5M would be ~3hrs; by-prefix cuts this
    # drastically.
    from collections import defaultdict
    prefix_idx: dict[str, list[str]] = defaultdict(list)
    for o in originals_long:
        prefix_idx[o[:4].lower()].append(o)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_written = 0
    n_replaced = 0
    with open(out_path, "w", encoding="utf-8") as out_f:
        for line in lines:
            obj = json.loads(line)
            traj = obj.get("trajectory", [])
            new_traj: list[dict[str, str]] = []
            for msg in traj:
                content = msg.get("content", "")
                if content:
                    # Find candidate originals whose 4-char prefix appears in content
                    candidates: set[str] = set()
                    cl = content.lower()
                    for prefix, ents in prefix_idx.items():
                        if prefix in cl:
                            candidates.update(ents)
                    # Replace longest-first
                    for orig in sorted(candidates, key=len, reverse=True):
                        if orig in content:
                            content = content.replace(orig, rw.mapping[orig])
                            n_replaced += 1
                new_traj.append({**msg, "content": content})
            out_f.write(json.dumps({**obj, "trajectory": new_traj}, ensure_ascii=False) + "\n")
            n_written += 1
            if n_written % 500 == 0:
                logger.info("  ... %d/%d trajectories rewritten (%d replacements)",
                            n_written, len(lines), n_replaced)
    logger.info("Wrote %d trajectories with %d total entity replacements -> %s",
                n_written, n_replaced, out_path)
    return n_written


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    parser = argparse.ArgumentParser(description="V15-Q8 synthetic-schema rewrite")
    parser.add_argument("--input_dir", type=Path, default=Path("data/freebase/verl_cwq"))
    parser.add_argument("--output_dir", type=Path,
                        default=Path("data/freebase/verl_cwq_q8_synthetic"))
    parser.add_argument("--sft_input", type=Path,
                        default=Path("data/freebase/sft_trajectories.jsonl"))
    parser.add_argument("--sft_output", type=Path,
                        default=Path("data/freebase/sft_trajectories_q8_synthetic.jsonl"))
    parser.add_argument("--skip_sft", action="store_true",
                        help="Skip SFT trajectory rewriting (only rewrite parquets).")
    args = parser.parse_args()

    rw = EntityRewriter()

    # Rewrite all 3 parquet splits in order: train -> val -> test
    args.output_dir.mkdir(parents=True, exist_ok=True)
    n_total = 0
    for split in ("train", "val", "test"):
        in_path = args.input_dir / f"{split}.parquet"
        out_path = args.output_dir / f"{split}.parquet"
        if not in_path.exists():
            logger.warning("Missing %s — skipping", in_path)
            continue
        n_total += rewrite_parquet(in_path, out_path, rw)

    # Save mapping (after all parquets so it's complete)
    mapping_path = args.output_dir / "entity_mapping.json"
    rw.save(mapping_path)
    logger.info("Total rows rewritten: %d", n_total)
    logger.info("Total unique entities mapped: %d", len(rw.mapping))

    # Sanity check: a few sample URIs
    sample = list(rw.mapping.items())[:10]
    logger.info("Sample URIs:")
    for orig, uri in sample:
        logger.info("   %r  ->  %s", orig, uri)

    # SFT trajectory rewrite
    if not args.skip_sft and args.sft_input.exists():
        rewrite_sft_trajectories(args.sft_input, args.sft_output, rw)

    logger.info("=== Q8 schema rewrite done ===")


if __name__ == "__main__":
    main()
