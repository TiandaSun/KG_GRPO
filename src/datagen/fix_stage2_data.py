"""Post-processing fix for Stage 2 data: repair <think> tags and gold_answer_short.

Two issues found in the Stage 2 output:
1. ~11.7% of records have broken <think> tags (missing angle brackets or missing </think>)
2. ~65% of records have overly long gold_answer_short (should be a short phrase, not a paragraph)

This script fixes both issues in-place across all split files.

Usage:
    python src/datagen/fix_stage2_data.py \
        --data_dir data/processed \
        --backup
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class FixStats:
    """Track fix statistics."""

    total: int = 0
    think_tag_fixed: int = 0
    think_bare_space: int = 0
    think_bare_newline: int = 0
    think_missing_close: int = 0
    gold_answer_fixed: int = 0
    gold_answer_was_long: int = 0
    gold_answer_was_empty: int = 0
    skipped_negative: int = 0

    def summary(self) -> str:
        lines = [
            f"Total records processed: {self.total}",
            f"Think tags fixed: {self.think_tag_fixed}",
            f"  - bare 'think ' prefix: {self.think_bare_space}",
            f"  - bare 'think\\n' prefix: {self.think_bare_newline}",
            f"  - missing </think> close: {self.think_missing_close}",
            f"gold_answer_short fixed: {self.gold_answer_fixed}",
            f"  - was too long (>50 chars): {self.gold_answer_was_long}",
            f"  - was empty: {self.gold_answer_was_empty}",
            f"Skipped (negative examples): {self.skipped_negative}",
        ]
        return "\n".join(lines)


# Patterns that signal the start of a final answer after reasoning
_ANSWER_START_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\n\s*(?:Final\s+)?[Aa]nswer\s*:", re.IGNORECASE),
    re.compile(r"\n\s*(?:Therefore|Thus|So|In (?:summary|conclusion))\s*[,:]", re.IGNORECASE),
    re.compile(r"\n\s*(?:The (?:final |short )?answer (?:is|would be))", re.IGNORECASE),
    re.compile(r"\n\s*(?:A concise (?:final )?answer)", re.IGNORECASE),
]


def _find_answer_boundary(text: str) -> int | None:
    """Find the character index where the final answer begins in reasoning text.

    Returns the index of the newline before the answer, or None if no boundary found.
    """
    best_pos: int | None = None
    for pattern in _ANSWER_START_PATTERNS:
        match = pattern.search(text)
        if match:
            pos = match.start()
            if best_pos is None or pos > best_pos:
                best_pos = pos
    return best_pos


def fix_think_tags(answer: str, stats: FixStats) -> str:
    """Fix broken <think>...</think> tags in an answer string.

    Handles three cases:
    1. Bare 'think ' prefix (missing angle brackets)
    2. Bare 'think\\n' prefix (missing angle brackets)
    3. Has <think> but missing </think> (truncated generation)
    """
    # Already correct
    if "<think>" in answer and "</think>" in answer:
        return answer

    original = answer

    # Case 1 & 2: bare "think" prefix without angle brackets
    # Covers: "think ", "think\n", "think: ", "think:\n", "think>" (missing <)
    bare_think_match = re.match(r"^[Tt]hink\s*[:>\-]?\s*", answer)
    if bare_think_match and "<think>" not in answer:
        prefix = bare_think_match.group()
        remainder = answer[len(prefix):]
        if "\n" in prefix:
            stats.think_bare_newline += 1
        else:
            stats.think_bare_space += 1
        answer = "<think>" + remainder

    # Now handle missing </think> (applies to both newly fixed and original <think> cases)
    if "<think>" in answer and "</think>" not in answer:
        think_start = answer.index("<think>") + len("<think>")
        reasoning_text = answer[think_start:]

        # Try to find where the final answer starts
        boundary = _find_answer_boundary(reasoning_text)
        if boundary is not None:
            reasoning = reasoning_text[:boundary]
            final_answer = reasoning_text[boundary:].strip()
            answer = answer[:think_start] + reasoning + "</think>\n" + final_answer
        else:
            # No clear boundary found — look for the last paragraph as the answer
            paragraphs = reasoning_text.rstrip().rsplit("\n\n", 1)
            if len(paragraphs) == 2 and len(paragraphs[1].strip()) < 300:
                reasoning = paragraphs[0]
                final_answer = paragraphs[1].strip()
                answer = answer[:think_start] + reasoning + "</think>\n" + final_answer
            else:
                # Last resort: just close the think tag at the end
                answer = answer.rstrip() + "</think>"

        stats.think_missing_close += 1

    if answer != original:
        stats.think_tag_fixed += 1

    return answer


_ANSWER_PREFIX_RE = re.compile(
    r"^(?:(?:Final\s+)?[Aa]nswer|A concise (?:final )?answer)\s*:\s*",
)


def _strip_answer_prefix(text: str) -> str:
    """Strip common 'Answer:' style prefixes from a string."""
    return _ANSWER_PREFIX_RE.sub("", text).strip()


def extract_gold_answer_short(
    record: dict[str, Any],
) -> str:
    """Extract a concise gold_answer_short from the record.

    Strategy: use the last entity of the KG path. This is the natural target
    of the reasoning chain and is short enough for substring matching in the
    reward function's _check_answer().
    """
    kg_path = record.get("kg_path", [])
    if not kg_path:
        return ""

    # Last entity = object of the last triple in the path
    last_triple = kg_path[-1]
    if len(last_triple) >= 3:
        return last_triple[2]

    return ""


def fix_record(record: dict[str, Any], stats: FixStats) -> dict[str, Any]:
    """Apply all fixes to a single record."""
    stats.total += 1

    # Skip negative examples — their format is already correct
    if record.get("is_negative", False):
        stats.skipped_negative += 1
        return record

    record = dict(record)  # shallow copy

    # Fix 1: Repair <think> tags
    original_answer = record.get("answer", "")
    record["answer"] = fix_think_tags(original_answer, stats)

    # Fix 2: Repair gold_answer_short
    original_gas = record.get("gold_answer_short", "")

    if not original_gas.strip():
        stats.gold_answer_was_empty += 1
        record["gold_answer_short"] = extract_gold_answer_short(record)
        stats.gold_answer_fixed += 1
    elif len(original_gas) > 50:
        stats.gold_answer_was_long += 1
        record["gold_answer_short"] = extract_gold_answer_short(record)
        stats.gold_answer_fixed += 1
    else:
        # Already short — but strip any "Answer:" prefix
        cleaned = _strip_answer_prefix(original_gas)
        if cleaned != original_gas:
            record["gold_answer_short"] = cleaned
            stats.gold_answer_fixed += 1

    return record


def process_file(
    input_path: Path,
    output_path: Path,
    stats: FixStats,
) -> int:
    """Process a single JSONL file, applying fixes to each record.

    Returns the number of records processed.
    """
    records: list[dict[str, Any]] = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                record = json.loads(line)
                records.append(fix_record(record, stats))

    with open(output_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    return len(records)


@dataclass
class FixConfig:
    """Configuration for the data fix script."""

    data_dir: Path = Path("data/processed")
    backup: bool = True
    files: list[str] = field(default_factory=lambda: [
        "conceptnet_qa_train.jsonl",
        "conceptnet_qa_val.jsonl",
        "conceptnet_qa_test.jsonl",
        "conceptnet_qa_train_with_neg.jsonl",
    ])


def run_fix(config: FixConfig) -> None:
    """Main fix pipeline."""
    logger.info("FixConfig: %s", config)
    logger.info("Data directory: %s", config.data_dir.resolve())

    overall_stats = FixStats()

    for filename in config.files:
        filepath = config.data_dir / filename
        if not filepath.exists():
            logger.warning("File not found, skipping: %s", filepath)
            continue

        # Backup original
        if config.backup:
            backup_path = filepath.with_suffix(".jsonl.bak")
            if not backup_path.exists():
                shutil.copy2(filepath, backup_path)
                logger.info("Backed up %s -> %s", filepath, backup_path)
            else:
                logger.info("Backup already exists: %s", backup_path)

        file_stats = FixStats()
        count = process_file(filepath, filepath, file_stats)
        logger.info(
            "Processed %s: %d records, %d think fixes, %d gold_answer fixes",
            filename, count, file_stats.think_tag_fixed, file_stats.gold_answer_fixed,
        )

        # Accumulate stats
        for attr in vars(overall_stats):
            if isinstance(getattr(overall_stats, attr), int):
                setattr(
                    overall_stats,
                    attr,
                    getattr(overall_stats, attr) + getattr(file_stats, attr),
                )

    logger.info("=== Overall Fix Summary ===\n%s", overall_stats.summary())

    # Validation: spot-check the fixed data
    _validate_fixes(config)


def _validate_fixes(config: FixConfig) -> None:
    """Run validation checks on the fixed data."""
    train_path = config.data_dir / "conceptnet_qa_train.jsonl"
    if not train_path.exists():
        return

    total = 0
    has_think = 0
    has_close_think = 0
    short_gas = 0
    long_gas = 0
    empty_gas = 0

    with open(train_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if record.get("is_negative", False):
                continue
            total += 1

            ans = record.get("answer", "")
            gas = record.get("gold_answer_short", "")

            if "<think>" in ans:
                has_think += 1
            if "</think>" in ans:
                has_close_think += 1
            if not gas.strip():
                empty_gas += 1
            elif len(gas) <= 50:
                short_gas += 1
            else:
                long_gas += 1

    logger.info("=== Validation (train, positive only) ===")
    logger.info("  Total: %d", total)
    logger.info("  Has <think>: %d/%d (%.1f%%)", has_think, total, 100 * has_think / max(total, 1))
    logger.info("  Has </think>: %d/%d (%.1f%%)", has_close_think, total, 100 * has_close_think / max(total, 1))
    logger.info("  gold_answer_short <= 50 chars: %d/%d (%.1f%%)", short_gas, total, 100 * short_gas / max(total, 1))
    logger.info("  gold_answer_short > 50 chars: %d/%d (%.1f%%)", long_gas, total, 100 * long_gas / max(total, 1))
    logger.info("  gold_answer_short empty: %d/%d (%.1f%%)", empty_gas, total, 100 * empty_gas / max(total, 1))

    # Warn if issues remain
    if has_think < total * 0.95:
        logger.warning("WARN: <think> tag coverage below 95%%: %d/%d", has_think, total)
    if has_close_think < total * 0.95:
        logger.warning("WARN: </think> tag coverage below 95%%: %d/%d", has_close_think, total)
    if long_gas > total * 0.05:
        logger.warning("WARN: >5%% of records still have long gold_answer_short: %d/%d", long_gas, total)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fix Stage 2 data: repair <think> tags and gold_answer_short."
    )
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=Path("data/processed"),
        help="Directory containing the JSONL data files.",
    )
    parser.add_argument(
        "--backup",
        action="store_true",
        default=True,
        help="Create .bak backups before modifying files (default: True).",
    )
    parser.add_argument(
        "--no_backup",
        action="store_true",
        default=False,
        help="Skip creating backups.",
    )
    args = parser.parse_args()

    config = FixConfig(
        data_dir=args.data_dir,
        backup=not args.no_backup,
    )

    run_fix(config)


if __name__ == "__main__":
    main()
