"""Stage 3: KG path-alignment reward functions for GRPO training.

Two reward functions designed to be passed as a list to GRPOTrainer.reward_funcs:
  1. compute_kg_reward  — answer correctness + structured triple matching + anti-hack
  2. compute_format_reward — checks for proper <think>...</think> format

v3 changes from v2 (addressing reward hacking observed in Job 30923716):
- Answer correctness is now the DOMINANT signal (0 to 1.0, was -0.5 to 0.5)
- Bag-of-words token coverage replaced with structured triple matching (0 to 0.4)
  that requires both head AND tail entities of each triple to appear in the think block
- Anti-hack penalties (-0.3 to 0) catch template collapse, brevity, and repetition
- Format reward reduced to 0.5 max (was 1.0) since model already learned format
- Total range: -0.3 to 1.4 (answer is 56% of positive budget, was 23%)

Root cause of v2 failure:
  v2's bag-of-words coverage (R_coverage 0-1.2) rewarded mentioning individual tokens
  anywhere in the output. The model discovered a degenerate template
  ("X is UsedOf Y and Z is UsedOf UsedOf )") that scored high on token overlap
  without doing any real reasoning. Combined with beta=0 (no policy anchor),
  the model drifted into this template, collapsing output length from 284→55 tokens
  and dropping EM from 3.8% (SFT) to 0.6% (SFT+GRPO).

Usage (standalone validation):
    python src/rewards/kg_reward.py \
        --test_file data/processed/conceptnet_qa_test.jsonl \
        --num_samples 100
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Output parsing helpers
# ---------------------------------------------------------------------------


def _split_output(model_output: str) -> tuple[str, str]:
    """Split model output into think-block content and final answer.

    Args:
        model_output: Full model generation.

    Returns:
        (think_text, final_answer) — both stripped.  If no ``</think>``
        tag is present, think_text is empty and the full output is treated
        as the final answer.
    """
    if "</think>" in model_output:
        parts = model_output.split("</think>", 1)
        think_part = parts[0]
        if "<think>" in think_part:
            think_text = think_part.split("<think>", 1)[1]
        else:
            think_text = think_part
        final_answer = parts[1].strip()
    else:
        think_text = ""
        final_answer = model_output.strip()
    return think_text.strip(), final_answer


# ---------------------------------------------------------------------------
# Core reward: compute_kg_reward v3
# ---------------------------------------------------------------------------


def compute_kg_reward(
    question: str,
    model_output: str,
    kg_path: list[list[str]],
    gold_answer: str,
) -> float:
    """KG path-alignment reward v3: answer-dominant with anti-hack penalties.

    Three sub-components:
      1. R_answer:    answer correctness                (0 to 1.0)
      2. R_reasoning: structured triple matching in <think> (0 to 0.4)
      3. R_quality:   anti-hack penalties               (-0.3 to 0)

    Total range: -0.3 to 1.4.

    Args:
        question: The input question (unused but kept for GRPOTrainer signature).
        model_output: The model's full generated output.
        kg_path: List of triples, e.g. [["dog", "IsA", "animal"], ...].
        gold_answer: The expected short answer string.

    Returns:
        Combined reward score.
    """
    if not kg_path:
        return 0.0

    think_text, final_answer = _split_output(model_output)

    # R_answer: correctness of the final answer (0 to 1.0)
    # This is the DOMINANT signal — getting the right answer matters most.
    r_answer = _compute_answer_score(model_output, gold_answer)

    # R_reasoning: structured triple matching in think block (0 to 0.4)
    r_reasoning = 0.4 * _compute_triple_score(think_text, kg_path)

    # R_quality: anti-hack penalties on think block (-0.3 to 0)
    r_quality = _compute_quality_penalties(think_text)

    return r_answer + r_reasoning + r_quality


# ---------------------------------------------------------------------------
# Sub-component: structured triple matching (replaces bag-of-words coverage)
# ---------------------------------------------------------------------------


def _compute_triple_score(think_text: str, kg_path: list[list[str]]) -> float:
    """Score structured triple matching in the think block.

    Unlike v2's bag-of-words token coverage, this requires both the head
    AND tail entities of each triple to appear in the think block.  An
    additional bonus is awarded when the relation name also appears.

    This is much harder to game: the model can't just mention random tokens;
    it needs to reference actual entity pairs from the KG path.

    Args:
        think_text: Content between <think> and </think> tags.
        kg_path: List of triples [head, relation, tail].

    Returns:
        Score in [0.0, 1.0].
    """
    if not kg_path or not think_text:
        return 0.0

    think_lower = think_text.lower()

    triple_scores: list[float] = []
    for triple in kg_path:
        if len(triple) < 3:
            continue
        head, rel, tail = triple[0], triple[1], triple[2]
        score = 0.0

        h_found = head.lower() in think_lower
        t_found = tail.lower() in think_lower
        r_found = rel.lower() in think_lower

        if h_found and t_found:
            score = 0.6  # Both entities present — demonstrates path awareness
            if r_found:
                score = 1.0  # Full triple present — best possible
        elif h_found or t_found:
            score = 0.15  # Partial credit for mentioning one entity

        triple_scores.append(score)

    if not triple_scores:
        return 0.0

    # Average across triples, weighted by order bonus
    base_score = sum(triple_scores) / len(triple_scores)

    # Order bonus: entities appear in correct sequential order
    order_score = _compute_path_order(think_lower, kg_path)

    return min(1.0, 0.7 * base_score + 0.3 * order_score)


# ---------------------------------------------------------------------------
# Sub-component: path entity ordering (reused from v2)
# ---------------------------------------------------------------------------


def _compute_path_order(output_lower: str, kg_path: list[list[str]]) -> float:
    """Score how well output follows the KG path entity order.

    Extracts the ordered entity chain (head of first triple + tail of each
    triple) and checks if they appear sequentially in the output.

    Returns:
        Fraction of entities found in correct sequential order (0.0 to 1.0).
    """
    if not kg_path:
        return 0.0

    # Ordered entity chain: e.g. for dog→IsA→animal→HasProperty→alive
    # entities = ["dog", "animal", "alive"]
    entities = [kg_path[0][0].lower()]
    for triple in kg_path:
        if len(triple) >= 3:
            entities.append(triple[2].lower())

    if not entities:
        return 0.0

    last_pos = 0
    matched = 0
    for entity in entities:
        pos = output_lower.find(entity, last_pos)
        if pos >= 0:
            matched += 1
            last_pos = pos + len(entity)

    return matched / len(entities)


# ---------------------------------------------------------------------------
# Sub-component: anti-hack quality penalties
# ---------------------------------------------------------------------------


def _compute_quality_penalties(think_text: str) -> float:
    """Anti-hack penalties for degenerate think-block patterns.

    Detects three failure modes observed during v2 GRPO training:
      1. **Brevity collapse**: think block < 15 words → -0.15
      2. **Template collapse**: repeated relation placeholders → -0.15
      3. **N-gram repetition**: > 50% repeated trigrams → -0.1

    Args:
        think_text: Content between <think> and </think> tags.

    Returns:
        Penalty in [-0.3, 0.0].
    """
    penalty = 0.0

    # 1. Brevity penalty — think block should contain substantive reasoning
    words = think_text.split() if think_text else []
    word_count = len(words)
    if word_count < 15:
        penalty -= 0.15
    elif word_count < 25:
        penalty -= 0.05

    # 2. Degenerate template detection — catch the "UsedOf UsedOf )" pattern
    #    observed in v2 training where all relation types collapsed to "UsedOf"
    if think_text:
        think_lower = think_text.lower()
        # Detect repeated generic relation placeholders
        for placeholder in ("usedof", "typeof", "kindof"):
            if think_lower.count(placeholder) >= 2:
                penalty -= 0.15
                break

    # 3. N-gram repetition — high repetition signals a degenerate template
    if word_count >= 8:
        lower_words = [w.lower() for w in words]
        trigrams = [
            (lower_words[i], lower_words[i + 1], lower_words[i + 2])
            for i in range(len(lower_words) - 2)
        ]
        if trigrams:
            unique_ratio = len(set(trigrams)) / len(trigrams)
            if unique_ratio < 0.5:
                penalty -= 0.1

    return max(-0.3, penalty)


# ---------------------------------------------------------------------------
# Sub-component: answer correctness (kept from v2, now used directly as 0-1.0)
# ---------------------------------------------------------------------------


def _compute_answer_score(model_output: str, gold_answer: str) -> float:
    """Continuous answer correctness score (0.0 to 1.0).

    Uses exact substring match for full credit, token-level F1 for partial
    credit.  Operates on the final answer portion (after </think> if present).
    """
    gold_lower = gold_answer.strip().lower()
    if not gold_lower:
        return 0.0

    # Extract final answer (after </think>)
    if "</think>" in model_output:
        final_answer = model_output.split("</think>")[-1].strip().lower()
    else:
        final_answer = model_output.strip().lower()

    if not final_answer:
        return 0.0

    # Exact substring match → full credit
    if gold_lower in final_answer:
        return 1.0

    # Token-level F1 for partial credit
    pred_tokens = set(final_answer.split())
    gold_tokens = set(gold_lower.split())

    if not gold_tokens or not pred_tokens:
        return 0.0

    common = pred_tokens & gold_tokens
    if not common:
        return 0.0

    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


# ---------------------------------------------------------------------------
# Format reward (reduced magnitude in v3)
# ---------------------------------------------------------------------------


def compute_format_reward(model_output: str, **kwargs: Any) -> float:
    """Reward for proper <think>...</think> format.

    Reduced from 1.0 to 0.5 max in v3 because the model already learned
    format compliance during SFT warmup (0.982 in eval).  The lower ceiling
    prevents format reward from dominating the answer signal.

    Returns:
        0.5 if valid think tags with substantive reasoning (> 50 chars).
        0.2 if think tags present but reasoning too short (20-50 chars).
        0.1 if think tags present but very short reasoning (< 20 chars).
        0.0 if think tags missing.
    """
    has_think_open = "<think>" in model_output
    has_think_close = "</think>" in model_output

    if has_think_open and has_think_close:
        match = re.search(r"<think>(.*?)</think>", model_output, re.DOTALL)
        if match:
            content_len = len(match.group(1).strip())
            if content_len > 50:
                return 0.5
            if content_len > 20:
                return 0.2
        return 0.1
    return 0.0


# ---------------------------------------------------------------------------
# Utility: check_answer (used by evaluation, not reward)
# ---------------------------------------------------------------------------


def check_answer(model_output: str, gold_answer: str) -> bool:
    """Check if the gold answer appears in the final answer (after </think>).

    Extracts text after the last </think> tag if present, otherwise uses
    the full output.  Performs case-insensitive substring matching.
    """
    gold_lower = gold_answer.strip().lower()
    if not gold_lower:
        return False

    if "</think>" in model_output:
        final_answer = model_output.split("</think>")[-1].strip().lower()
    else:
        final_answer = model_output.strip().lower()

    return gold_lower in final_answer


# ---------------------------------------------------------------------------
# Combined reward (for analysis/logging)
# ---------------------------------------------------------------------------


def compute_combined_reward(
    question: str,
    model_output: str,
    kg_path: list[list[str]],
    gold_answer: str,
) -> dict[str, float]:
    """Compute both reward components and return breakdown.

    Useful for logging and analysis.  For GRPOTrainer, use the individual
    functions passed as reward_funcs list instead.
    """
    kg_reward = compute_kg_reward(question, model_output, kg_path, gold_answer)
    format_reward = compute_format_reward(model_output)

    return {
        "kg_reward": kg_reward,
        "format_reward": format_reward,
        "total_reward": kg_reward + format_reward,
    }


# ---------------------------------------------------------------------------
# RewardStats for validation
# ---------------------------------------------------------------------------


@dataclass
class RewardStats:
    """Aggregate statistics for reward validation."""

    total: int = 0
    kg_reward_sum: float = 0.0
    format_reward_sum: float = 0.0
    correct_count: int = 0
    format_valid_count: int = 0
    min_kg: float = float("inf")
    max_kg: float = float("-inf")

    def update(self, kg_reward: float, format_reward: float, correct: bool) -> None:
        self.total += 1
        self.kg_reward_sum += kg_reward
        self.format_reward_sum += format_reward
        if correct:
            self.correct_count += 1
        if format_reward >= 0.5:
            self.format_valid_count += 1
        self.min_kg = min(self.min_kg, kg_reward)
        self.max_kg = max(self.max_kg, kg_reward)

    @property
    def kg_mean(self) -> float:
        return self.kg_reward_sum / max(self.total, 1)

    @property
    def format_mean(self) -> float:
        return self.format_reward_sum / max(self.total, 1)

    @property
    def accuracy(self) -> float:
        return self.correct_count / max(self.total, 1)

    @property
    def format_rate(self) -> float:
        return self.format_valid_count / max(self.total, 1)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_rewards(test_file: Path, num_samples: int = 100) -> None:
    """Validate reward function on test QA pairs."""
    records: list[dict[str, Any]] = []
    with open(test_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    if num_samples < len(records):
        records = records[:num_samples]

    logger.info("Validating v3 rewards on %d samples from %s", len(records), test_file)

    # Test 1: Gold answers should score well
    gold_stats = RewardStats()
    for record in records:
        kg_r = compute_kg_reward(
            record["question"], record["answer"],
            record["kg_path"], record["gold_answer_short"],
        )
        fmt_r = compute_format_reward(record["answer"])
        correct = check_answer(record["answer"], record["gold_answer_short"])
        gold_stats.update(kg_r, fmt_r, correct)

    logger.info("=== Gold Answer Rewards (v3 answer-dominant) ===")
    logger.info("  KG reward:  mean=%.3f, min=%.3f, max=%.3f", gold_stats.kg_mean, gold_stats.min_kg, gold_stats.max_kg)
    logger.info("  Format:     mean=%.3f, valid_rate=%.1f%%", gold_stats.format_mean, 100 * gold_stats.format_rate)
    logger.info("  Accuracy:   %.1f%%", 100 * gold_stats.accuracy)

    # Test 2: Wrong answers should score poorly
    bad_stats = RewardStats()
    for record in records:
        bad_output = "I don't know the answer to this question."
        kg_r = compute_kg_reward(
            record["question"], bad_output,
            record["kg_path"], record["gold_answer_short"],
        )
        fmt_r = compute_format_reward(bad_output)
        correct = check_answer(bad_output, record["gold_answer_short"])
        bad_stats.update(kg_r, fmt_r, correct)

    logger.info("=== Bad Answer Rewards (v3) ===")
    logger.info("  KG reward:  mean=%.3f, min=%.3f, max=%.3f", bad_stats.kg_mean, bad_stats.min_kg, bad_stats.max_kg)
    logger.info("  Format:     mean=%.3f, valid_rate=%.1f%%", bad_stats.format_mean, 100 * bad_stats.format_rate)
    logger.info("  Accuracy:   %.1f%%", 100 * bad_stats.accuracy)

    # Test 3: Entity stuffing should be penalised
    stuff_stats = RewardStats()
    for record in records:
        entities = " ".join(record.get("entities", [e for t in record["kg_path"] for e in (t[0], t[2])]))
        stuffed_output = f"<think>{(entities + ' ') * 20}</think>\n{record['gold_answer_short']}"
        kg_r = compute_kg_reward(
            record["question"], stuffed_output,
            record["kg_path"], record["gold_answer_short"],
        )
        fmt_r = compute_format_reward(stuffed_output)
        correct = check_answer(stuffed_output, record["gold_answer_short"])
        stuff_stats.update(kg_r, fmt_r, correct)

    logger.info("=== Entity Stuffing Rewards ===")
    logger.info("  KG reward:  mean=%.3f, min=%.3f, max=%.3f", stuff_stats.kg_mean, stuff_stats.min_kg, stuff_stats.max_kg)
    logger.info("  Format:     mean=%.3f, valid_rate=%.1f%%", stuff_stats.format_mean, 100 * stuff_stats.format_rate)
    logger.info("  Accuracy:   %.1f%%", 100 * stuff_stats.accuracy)

    # Test 4: Template collapse should score poorly (NEW in v3)
    template_stats = RewardStats()
    for record in records:
        entities = [e for t in record["kg_path"] for e in (t[0], t[2])]
        e1 = entities[0] if entities else "X"
        e2 = entities[1] if len(entities) > 1 else "Y"
        template_output = (
            f"<think>The knowledge provided states that {e1} is UsedOf "
            f"{e2} and {e2} is UsedOf UsedOf )</think>\n{e2}"
        )
        kg_r = compute_kg_reward(
            record["question"], template_output,
            record["kg_path"], record["gold_answer_short"],
        )
        fmt_r = compute_format_reward(template_output)
        correct = check_answer(template_output, record["gold_answer_short"])
        template_stats.update(kg_r, fmt_r, correct)

    logger.info("=== Template Collapse Rewards (v3 anti-hack) ===")
    logger.info("  KG reward:  mean=%.3f, min=%.3f, max=%.3f", template_stats.kg_mean, template_stats.min_kg, template_stats.max_kg)
    logger.info("  Format:     mean=%.3f, valid_rate=%.1f%%", template_stats.format_mean, 100 * template_stats.format_rate)
    logger.info("  Accuracy:   %.1f%%", 100 * template_stats.accuracy)

    # Sanity checks
    logger.info("=== Sanity Checks ===")
    gold_better_than_bad = gold_stats.kg_mean > bad_stats.kg_mean
    logger.info(
        "  Gold > Bad (KG reward): %s (%.3f > %.3f)",
        "PASS" if gold_better_than_bad else "FAIL",
        gold_stats.kg_mean, bad_stats.kg_mean,
    )

    stuffing_penalised = stuff_stats.kg_mean < gold_stats.kg_mean
    logger.info(
        "  Stuffing penalised: %s (%.3f < %.3f)",
        "PASS" if stuffing_penalised else "FAIL",
        stuff_stats.kg_mean, gold_stats.kg_mean,
    )

    template_penalised = template_stats.kg_mean < gold_stats.kg_mean
    logger.info(
        "  Template penalised: %s (%.3f < %.3f)",
        "PASS" if template_penalised else "FAIL",
        template_stats.kg_mean, gold_stats.kg_mean,
    )

    template_worse_than_good = template_stats.kg_mean < gold_stats.kg_mean * 0.5
    logger.info(
        "  Template < 50%% of Gold: %s (%.3f < %.3f)",
        "PASS" if template_worse_than_good else "FAIL",
        template_stats.kg_mean, gold_stats.kg_mean * 0.5,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate KG reward function on test QA pairs."
    )
    parser.add_argument(
        "--test_file",
        type=Path,
        default=Path("data/processed/conceptnet_qa_test.jsonl"),
        help="Test JSONL file with QA pairs.",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=100,
        help="Number of samples to validate on.",
    )
    args = parser.parse_args()

    if not args.test_file.exists():
        logger.error(
            "Test file not found: %s. Run Stage 2 first to generate QA data.",
            args.test_file,
        )
        return

    validate_rewards(args.test_file, args.num_samples)


###############################################################################
# GRPOTrainer-compatible wrapper functions
#
# TRL's GRPOTrainer calls reward_funcs with signature:
#   reward_func(prompts, completions, **kwargs) -> list[float]
#
# where prompts/completions are lists of conversational messages
# (list[dict[str, str]]) and kwargs contain metadata columns from the dataset.
###############################################################################


def _extract_text(message: Any) -> str:
    """Extract plain text from a GRPOTrainer message.

    Messages can be either plain strings or conversational format:
    [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]

    Returns:
        Concatenated content text.
    """
    if isinstance(message, str):
        return message
    if isinstance(message, list):
        return "\n".join(
            turn.get("content", "") if isinstance(turn, dict) else str(turn)
            for turn in message
        )
    return str(message)


def kg_reward_func(
    prompts: list[Any],
    completions: list[Any],
    **kwargs: Any,
) -> list[float]:
    """GRPOTrainer-compatible wrapper for :func:`compute_kg_reward`.

    Args:
        prompts: List of prompts (conversational or string).
        completions: List of completions (conversational or string).
        **kwargs: Must contain ``kg_path`` and ``gold_answer_short`` as lists
            (one entry per prompt, propagated from dataset metadata columns).

    Returns:
        List of float rewards, one per completion.
    """
    kg_paths: list[list[list[str]]] = kwargs.get("kg_path", [])
    gold_answers: list[str] = kwargs.get("gold_answer_short", [])

    rewards: list[float] = []
    for i, completion in enumerate(completions):
        question_text = _extract_text(prompts[i]) if i < len(prompts) else ""
        completion_text = _extract_text(completion)

        path = kg_paths[i] if i < len(kg_paths) else []
        gold = gold_answers[i] if i < len(gold_answers) else ""

        rewards.append(
            compute_kg_reward(question_text, completion_text, path, gold)
        )

    return rewards


def format_reward_func(
    completions: list[Any],
    **kwargs: Any,
) -> list[float]:
    """GRPOTrainer-compatible wrapper for :func:`compute_format_reward`.

    Args:
        completions: List of completions (conversational or string).
        **kwargs: Accepted and ignored for GRPOTrainer compatibility.

    Returns:
        List of float rewards, one per completion.
    """
    return [
        compute_format_reward(_extract_text(completion))
        for completion in completions
    ]


if __name__ == "__main__":
    main()
