"""Unit tests for Stage 3: KG reward functions (v3 answer-dominant).

v3 changes tested here:
- Answer correctness is dominant signal (0-1.0, 56% of positive budget)
- Structured triple matching replaces bag-of-words coverage
- Anti-hack penalties catch template collapse, brevity, repetition
- Format reward reduced to 0.5 max

These tests use synthetic data only (no Stage 2 outputs needed) and should
complete in under 3 minutes on CPU.

To run with real Stage 2 data once available:
    python src/rewards/kg_reward.py --test_file data/processed/conceptnet_qa_test.jsonl
"""

from __future__ import annotations

import pytest

from src.rewards.kg_reward import (
    RewardStats,
    _compute_answer_score,
    _compute_path_order,
    _compute_quality_penalties,
    _compute_triple_score,
    _extract_text,
    _split_output,
    check_answer,
    compute_combined_reward,
    compute_format_reward,
    compute_kg_reward,
    format_reward_func,
    kg_reward_func,
)


# ==========================================
# Test _split_output (NEW in v3)
# ==========================================


class TestSplitOutput:
    def test_standard_format(self) -> None:
        output = "<think>Some reasoning here.</think>\nFinal answer."
        think, answer = _split_output(output)
        assert think == "Some reasoning here."
        assert answer == "Final answer."

    def test_no_think_tags(self) -> None:
        output = "Just a plain answer."
        think, answer = _split_output(output)
        assert think == ""
        assert answer == "Just a plain answer."

    def test_empty_think(self) -> None:
        output = "<think></think>\nAnswer."
        think, answer = _split_output(output)
        assert think == ""
        assert answer == "Answer."

    def test_multiline_think(self) -> None:
        output = "<think>\nLine 1\nLine 2\n</think>\nAnswer."
        think, answer = _split_output(output)
        assert "Line 1" in think
        assert "Line 2" in think
        assert answer == "Answer."

    def test_no_close_tag(self) -> None:
        output = "<think>Started but never closed"
        think, answer = _split_output(output)
        assert think == ""
        assert answer == "<think>Started but never closed"


# ==========================================
# Test check_answer (unchanged from v2)
# ==========================================


class TestCheckAnswer:
    def test_exact_match_after_think(self) -> None:
        output = "<think>Some reasoning here.</think>\nalive"
        assert check_answer(output, "alive") is True

    def test_substring_match_after_think(self) -> None:
        output = "<think>Some reasoning here.</think>\nThey are alive."
        assert check_answer(output, "alive") is True

    def test_case_insensitive(self) -> None:
        output = "<think>Reasoning.</think>\nAlive"
        assert check_answer(output, "alive") is True

    def test_wrong_answer(self) -> None:
        output = "<think>Reasoning.</think>\ndead"
        assert check_answer(output, "alive") is False

    def test_answer_only_in_think_block(self) -> None:
        output = "<think>The answer is alive.</think>\nI don't know."
        assert check_answer(output, "alive") is False

    def test_no_think_tags(self) -> None:
        output = "The answer is alive."
        assert check_answer(output, "alive") is True

    def test_empty_gold_answer(self) -> None:
        output = "<think>Reasoning.</think>\nalive"
        assert check_answer(output, "") is False

    def test_multiword_gold(self) -> None:
        output = "<think>Reasoning.</think>\nThey are hot dogs."
        assert check_answer(output, "hot dogs") is True

    def test_multiple_think_blocks(self) -> None:
        output = "<think>Block 1</think>\nwrong\n<think>Block 2</think>\nalive"
        assert check_answer(output, "alive") is True
        assert check_answer(output, "wrong") is False


# ==========================================
# Test _compute_answer_score (unchanged from v2)
# ==========================================


class TestComputeAnswerScore:
    def test_exact_match(self) -> None:
        output = "<think>Reasoning.</think>\nalive"
        assert _compute_answer_score(output, "alive") == 1.0

    def test_substring_match(self) -> None:
        output = "<think>Reasoning.</think>\nThey are alive."
        assert _compute_answer_score(output, "alive") == 1.0

    def test_no_match(self) -> None:
        output = "<think>Reasoning.</think>\ndead"
        assert _compute_answer_score(output, "alive") == 0.0

    def test_partial_match_multiword(self) -> None:
        output = "<think>Reasoning.</think>\nred hot"
        score = _compute_answer_score(output, "hot dog")
        assert 0.0 < score < 1.0

    def test_empty_gold(self) -> None:
        assert _compute_answer_score("some output", "") == 0.0

    def test_empty_final_answer(self) -> None:
        output = "<think>All reasoning, no answer.</think>\n"
        assert _compute_answer_score(output, "alive") == 0.0

    def test_no_think_tags(self) -> None:
        output = "The answer is alive."
        assert _compute_answer_score(output, "alive") == 1.0

    def test_case_insensitive(self) -> None:
        output = "<think>Reasoning.</think>\nALIVE"
        assert _compute_answer_score(output, "alive") == 1.0


# ==========================================
# Test _compute_path_order (unchanged from v2)
# ==========================================


class TestComputePathOrder:
    def test_perfect_order(self) -> None:
        kg_path = [["dog", "IsA", "animal"], ["animal", "HasProperty", "alive"]]
        output = "a dog is an animal and animals are alive"
        assert _compute_path_order(output, kg_path) == pytest.approx(1.0)

    def test_reversed_order(self) -> None:
        kg_path = [["dog", "IsA", "animal"], ["animal", "HasProperty", "alive"]]
        output = "alive things include animals like dogs"
        score = _compute_path_order(output, kg_path)
        assert score < 1.0

    def test_partial_order(self) -> None:
        kg_path = [["dog", "IsA", "animal"], ["animal", "HasProperty", "alive"]]
        output = "a dog has the property of being alive"
        score = _compute_path_order(output, kg_path)
        assert 0.5 < score < 1.0

    def test_no_entities_found(self) -> None:
        kg_path = [["dog", "IsA", "animal"]]
        output = "the weather is sunny today"
        assert _compute_path_order(output, kg_path) == pytest.approx(0.0)

    def test_empty_path(self) -> None:
        assert _compute_path_order("some output", []) == pytest.approx(0.0)

    def test_single_hop(self) -> None:
        kg_path = [["dog", "IsA", "animal"]]
        output = "a dog is an animal"
        assert _compute_path_order(output, kg_path) == pytest.approx(1.0)

    def test_3hop_path(self) -> None:
        kg_path = [
            ["dog", "IsA", "animal"],
            ["animal", "HasProperty", "alive"],
            ["alive", "HasProperty", "breathing"],
        ]
        output = "a dog is an animal that is alive and breathing"
        assert _compute_path_order(output, kg_path) == pytest.approx(1.0)


# ==========================================
# Test _compute_triple_score (NEW in v3)
# ==========================================


class TestComputeTripleScore:
    """Test structured triple matching — the v3 replacement for bag-of-words coverage."""

    def test_full_triple_match(self) -> None:
        """Both entities + relation present → high score."""
        kg_path = [["dog", "IsA", "animal"]]
        think_text = "A dog IsA animal based on the knowledge graph."
        score = _compute_triple_score(think_text, kg_path)
        assert score > 0.7  # Full triple present + order

    def test_entities_only_no_relation(self) -> None:
        """Both entities present but no relation → moderate score."""
        kg_path = [["dog", "IsA", "animal"]]
        think_text = "A dog is a type of animal."
        score = _compute_triple_score(think_text, kg_path)
        # "isa" not present as exact match, but head+tail present
        assert 0.3 < score < 0.8

    def test_single_entity_only(self) -> None:
        """Only one entity from a triple → low score."""
        kg_path = [["dog", "IsA", "animal"]]
        think_text = "I like dogs very much."
        score = _compute_triple_score(think_text, kg_path)
        assert score < 0.3

    def test_no_entities(self) -> None:
        """No entities match → zero score."""
        kg_path = [["dog", "IsA", "animal"]]
        think_text = "The weather is sunny today."
        score = _compute_triple_score(think_text, kg_path)
        assert score == pytest.approx(0.0)

    def test_empty_think(self) -> None:
        kg_path = [["dog", "IsA", "animal"]]
        assert _compute_triple_score("", kg_path) == pytest.approx(0.0)

    def test_empty_path(self) -> None:
        assert _compute_triple_score("some text", []) == pytest.approx(0.0)

    def test_multihop_all_triples_matched(self) -> None:
        """All triples fully matched → high score."""
        kg_path = [
            ["dog", "IsA", "animal"],
            ["animal", "HasProperty", "alive"],
        ]
        think_text = "A dog IsA animal. An animal HasProperty alive."
        score = _compute_triple_score(think_text, kg_path)
        assert score > 0.8

    def test_multihop_partial_match(self) -> None:
        """Only first triple matched, second missing → moderate score."""
        kg_path = [
            ["dog", "IsA", "animal"],
            ["animal", "HasProperty", "alive"],
        ]
        think_text = "A dog is a type of animal."
        score = _compute_triple_score(think_text, kg_path)
        # First triple: head+tail present (0.6), second: only head "animal" (0.15)
        assert 0.2 < score < 0.7

    def test_degenerate_template_scores_low(self) -> None:
        """The observed v2 hack pattern should score poorly.

        The template 'X is UsedOf Y and Y is UsedOf UsedOf )'
        doesn't mention actual relation names.
        """
        kg_path = [
            ["dog", "IsA", "animal"],
            ["animal", "HasProperty", "alive"],
        ]
        # The degenerate template — mentions entities but with wrong relation
        think_text = "dog is UsedOf animal and animal is UsedOf UsedOf )"
        score = _compute_triple_score(think_text, kg_path)

        # Compare with genuine reasoning
        good_think = "A dog IsA animal. An animal HasProperty alive."
        good_score = _compute_triple_score(good_think, kg_path)

        # Template should score lower because it uses "UsedOf" instead of
        # actual relations (IsA, HasProperty)
        assert score < good_score

    def test_case_insensitive(self) -> None:
        kg_path = [["Dog", "IsA", "Animal"]]
        think_text = "a dog isa animal"
        score = _compute_triple_score(think_text, kg_path)
        assert score > 0.7


# ==========================================
# Test _compute_quality_penalties (NEW in v3)
# ==========================================


class TestComputeQualityPenalties:
    """Test anti-hack penalty detection."""

    def test_substantive_reasoning_no_penalty(self) -> None:
        """Good reasoning with enough words → no penalty."""
        think_text = (
            "A dog is a type of animal based on the IsA relation. "
            "Animals have the property of being alive based on HasProperty. "
            "Therefore dogs are alive."
        )
        penalty = _compute_quality_penalties(think_text)
        assert penalty == pytest.approx(0.0)

    def test_very_short_think_penalised(self) -> None:
        """Think block < 15 words → -0.15 penalty."""
        think_text = "dog is animal"
        penalty = _compute_quality_penalties(think_text)
        assert penalty <= -0.15

    def test_moderately_short_think(self) -> None:
        """Think block 15-25 words → -0.05 brevity penalty."""
        # Use varied words to avoid triggering n-gram repetition penalty
        think_text = " ".join(f"word{i}" for i in range(20))
        penalty = _compute_quality_penalties(think_text)
        assert penalty == pytest.approx(-0.05)

    def test_usedof_template_penalised(self) -> None:
        """The 'UsedOf UsedOf' degenerate pattern → penalty."""
        think_text = "dog is UsedOf animal and animal is UsedOf alive"
        penalty = _compute_quality_penalties(think_text)
        assert penalty <= -0.15

    def test_typeof_template_penalised(self) -> None:
        think_text = "dog is TypeOf animal and animal is TypeOf alive"
        penalty = _compute_quality_penalties(think_text)
        assert penalty <= -0.15

    def test_ngram_repetition_penalised(self) -> None:
        """Highly repetitive text → penalty."""
        think_text = "dog is animal " * 10  # very repetitive trigrams
        penalty = _compute_quality_penalties(think_text)
        assert penalty < 0.0

    def test_empty_think_penalised(self) -> None:
        penalty = _compute_quality_penalties("")
        assert penalty <= -0.15

    def test_penalty_capped_at_minus_03(self) -> None:
        """Total penalty should never exceed -0.3."""
        # Trigger all penalties: short + template + repetition
        think_text = "UsedOf UsedOf UsedOf"
        penalty = _compute_quality_penalties(think_text)
        assert penalty >= -0.3

    def test_natural_relation_names_ok(self) -> None:
        """Normal relation names like IsA, Causes should not trigger template penalty."""
        think_text = (
            "The first step is that a dog IsA animal. "
            "This Causes the animal to have certain properties. "
            "The HasProperty relation shows that animals are alive. "
            "Following the knowledge graph path leads to the answer."
        )
        penalty = _compute_quality_penalties(think_text)
        # No template or repetition penalty (>25 words, no degenerate patterns)
        assert penalty == pytest.approx(0.0)


# ==========================================
# Test compute_format_reward (v3 — reduced magnitude)
# ==========================================


class TestComputeFormatReward:
    def test_valid_format_substantive(self) -> None:
        """Substantive reasoning (>50 chars) → 0.5 (was 1.0 in v2)."""
        output = (
            "<think>This is detailed reasoning about the knowledge graph path "
            "showing how entities are connected through relations.</think>\nFinal answer."
        )
        assert compute_format_reward(output) == 0.5

    def test_moderate_reasoning(self) -> None:
        """Moderate reasoning (20-50 chars) → 0.2."""
        output = "<think>Dogs are animals that live.</think>\nFinal answer."
        assert compute_format_reward(output) == 0.2

    def test_very_short_reasoning(self) -> None:
        """Very short reasoning (<20 chars) → 0.1."""
        output = "<think>Short.</think>\nFinal answer."
        assert compute_format_reward(output) == 0.1

    def test_empty_reasoning(self) -> None:
        output = "<think></think>\nFinal answer."
        assert compute_format_reward(output) == 0.1

    def test_no_tags(self) -> None:
        output = "Just a plain answer without any tags."
        assert compute_format_reward(output) == 0.0

    def test_only_open_tag(self) -> None:
        output = "<think>Started but never closed"
        assert compute_format_reward(output) == 0.0

    def test_only_close_tag(self) -> None:
        output = "Some text</think>more text"
        assert compute_format_reward(output) == 0.0

    def test_reasoning_51_chars(self) -> None:
        """Just over 50 chars → 0.5."""
        content = "a" * 51
        output = f"<think>{content}</think>\nAnswer."
        assert compute_format_reward(output) == 0.5

    def test_reasoning_exactly_50_chars(self) -> None:
        """Exactly 50 chars → 0.2 (not > 50)."""
        content = "a" * 50
        output = f"<think>{content}</think>\nAnswer."
        assert compute_format_reward(output) == 0.2

    def test_kwargs_ignored(self) -> None:
        output = (
            "<think>This is enough reasoning to pass the length check easily "
            "with plenty of chars.</think>\nAnswer."
        )
        assert compute_format_reward(output, question="test", extra="ignored") == 0.5


# ==========================================
# Test compute_kg_reward (v3 answer-dominant)
# ==========================================


class TestComputeKGReward:
    """Test the main KG reward function with v3 scoring."""

    SAMPLE_PATH = [["dog", "IsA", "animal"], ["animal", "HasProperty", "alive"]]

    def test_good_output_scores_high(self) -> None:
        output = (
            "<think>A dog is an animal (IsA relation). Animals have the "
            "property of being alive (HasProperty). Therefore, dogs are alive."
            "</think>\nalive"
        )
        reward = compute_kg_reward("What?", output, self.SAMPLE_PATH, "alive")
        # R_answer=1.0 + R_reasoning≈0.3 + R_quality=0 → > 1.0
        assert reward > 1.0

    def test_wrong_answer_scores_lower(self) -> None:
        output = "<think>A dog is an animal. Animals are alive.</think>\ndead"
        reward = compute_kg_reward("What?", output, self.SAMPLE_PATH, "alive")
        good_output = "<think>A dog is an animal. Animals are alive.</think>\nalive"
        good_reward = compute_kg_reward("What?", good_output, self.SAMPLE_PATH, "alive")
        # Wrong answer (R_answer=0) should be much worse than correct (R_answer=1.0)
        assert reward < good_reward
        # In v3 the gap is large because answer is 56% of budget
        assert good_reward - reward > 0.5

    def test_no_coverage_no_answer(self) -> None:
        output = "I have no idea about this question."
        reward = compute_kg_reward("What?", output, self.SAMPLE_PATH, "alive")
        # R_answer=0 (wrong), R_reasoning=0 (no think), R_quality=-0.15 (no think=short)
        assert reward < 0.0

    def test_correct_answer_no_reasoning(self) -> None:
        """Correct answer but no think block → moderate reward."""
        output = "alive"
        reward = compute_kg_reward("What?", output, self.SAMPLE_PATH, "alive")
        # R_answer=1.0, R_reasoning=0 (no think text), R_quality=-0.15 (empty think)
        assert 0.5 < reward < 1.0

    def test_good_reasoning_wrong_answer(self) -> None:
        """Good reasoning but wrong answer → low-moderate reward."""
        output = (
            "<think>A dog IsA animal. An animal HasProperty alive. "
            "Following the path from dog to alive.</think>\ndead"
        )
        reward = compute_kg_reward("What?", output, self.SAMPLE_PATH, "alive")
        # R_answer=0, R_reasoning≈0.35, R_quality=0 → ~0.35
        assert 0.0 < reward < 0.6

    def test_entity_stuffing_penalised(self) -> None:
        """Repeating entities many times should get penalised by quality penalty."""
        good_output = (
            "<think>A dog is an animal (IsA relation). Animals have the "
            "property of being alive (HasProperty). Therefore, dogs are alive."
            "</think>\nalive"
        )
        stuffed_output = (
            "<think>"
            + ("dog animal alive IsA HasProperty " * 30)
            + "</think>\nalive"
        )
        good_reward = compute_kg_reward("What?", good_output, self.SAMPLE_PATH, "alive")
        stuffed_reward = compute_kg_reward("What?", stuffed_output, self.SAMPLE_PATH, "alive")
        # Stuffing triggers n-gram repetition penalty
        assert stuffed_reward < good_reward

    def test_template_collapse_penalised(self) -> None:
        """The degenerate 'UsedOf' template observed in v2 training.

        This is THE critical test for v3 — this pattern must score poorly.
        """
        template_output = (
            "<think>The knowledge provided states that dog is UsedOf "
            "animal and animal is UsedOf UsedOf )</think>\nalive"
        )
        good_output = (
            "<think>A dog IsA animal. An animal HasProperty alive. "
            "Following the knowledge graph path.</think>\nalive"
        )
        template_reward = compute_kg_reward("What?", template_output, self.SAMPLE_PATH, "alive")
        good_reward = compute_kg_reward("What?", good_output, self.SAMPLE_PATH, "alive")
        # Template MUST score lower than genuine reasoning
        assert template_reward < good_reward
        # Template gets UsedOf penalty (-0.15) and no relation bonus
        # while good output gets proper triple matching

    def test_reward_range(self) -> None:
        """Total reward should be within v3 range (-0.3 to ~1.4)."""
        output = "<think>dog isa animal hasproperty alive</think>\nalive"
        reward = compute_kg_reward("What?", output, self.SAMPLE_PATH, "alive")
        assert -0.3 <= reward <= 1.5

    def test_single_hop_path(self) -> None:
        path = [["dog", "IsA", "animal"]]
        output = (
            "<think>A dog is a type of animal based on the IsA relation.</think>\nanimal"
        )
        reward = compute_kg_reward("What?", output, path, "animal")
        assert reward > 0.5

    def test_3hop_path(self) -> None:
        path = [
            ["dog", "IsA", "animal"],
            ["animal", "HasProperty", "alive"],
            ["alive", "HasProperty", "breathing"],
        ]
        output = (
            "<think>A dog is an animal (IsA). Animals have the property of "
            "being alive (HasProperty). Being alive means breathing (HasProperty).</think>\n"
            "breathing"
        )
        reward = compute_kg_reward("What?", output, path, "breathing")
        assert reward > 1.0

    def test_empty_path(self) -> None:
        reward = compute_kg_reward("What?", "some output", [], "answer")
        assert reward == 0.0

    def test_multiword_entity_coverage(self) -> None:
        path = [["hot dog", "IsA", "food"]]
        output = (
            "<think>A hot dog IsA food according to the knowledge graph.</think>\nfood"
        )
        reward = compute_kg_reward("What?", output, path, "food")
        assert reward > 0.5

    def test_continuous_reward_variance(self) -> None:
        """Different quality outputs should get different continuous scores.

        This is the key property for GRPO within-group variance.
        """
        path = [["dog", "IsA", "animal"], ["animal", "HasProperty", "alive"]]

        # 4 outputs of decreasing quality
        outputs = [
            # Best: full triple matching + correct answer
            "<think>A dog IsA animal. Animals HasProperty alive.</think>\nalive",
            # OK: entities mentioned + correct answer, but no relation names
            "<think>Dogs and animals are alive.</think>\nalive",
            # Bad reasoning + wrong answer
            "<think>A dog is an animal.</think>\ndead",
            # Worst: no reasoning, no answer
            "I don't know.",
        ]
        rewards = [compute_kg_reward("Q?", o, path, "alive") for o in outputs]

        # All rewards should be DIFFERENT (continuous, not binary)
        assert len(set(round(r, 4) for r in rewards)) >= 3
        # Strictly decreasing quality
        assert rewards[0] > rewards[1]
        assert rewards[1] > rewards[2]
        assert rewards[2] > rewards[3]

    def test_answer_dominates_reasoning(self) -> None:
        """Correct answer with no reasoning should beat wrong answer with
        perfect reasoning.  This is the core v3 invariant: answer > reasoning.
        """
        path = [["dog", "IsA", "animal"], ["animal", "HasProperty", "alive"]]

        # Correct answer, minimal reasoning
        correct_minimal = "<think>dog animal alive</think>\nalive"
        # Wrong answer, perfect reasoning
        wrong_perfect = (
            "<think>A dog IsA animal. An animal HasProperty alive. "
            "Following the full knowledge graph path.</think>\ndead"
        )

        r_correct = compute_kg_reward("Q?", correct_minimal, path, "alive")
        r_wrong = compute_kg_reward("Q?", wrong_perfect, path, "alive")
        assert r_correct > r_wrong


# ==========================================
# Test compute_combined_reward
# ==========================================


class TestComputeCombinedReward:
    def test_returns_all_components(self) -> None:
        path = [["dog", "IsA", "animal"]]
        output = (
            "<think>A dog is a type of animal, as shown by the IsA relation "
            "in the knowledge graph.</think>\nanimal"
        )
        result = compute_combined_reward("What?", output, path, "animal")
        assert "kg_reward" in result
        assert "format_reward" in result
        assert "total_reward" in result
        assert result["total_reward"] == pytest.approx(
            result["kg_reward"] + result["format_reward"]
        )

    def test_good_output_high_total(self) -> None:
        path = [["dog", "IsA", "animal"]]
        output = (
            "<think>A dog is a type of animal based on the IsA relation "
            "in the knowledge graph.</think>\nanimal"
        )
        result = compute_combined_reward("What?", output, path, "animal")
        # KG reward > 1.0 + format reward = 0.5 → total > 1.0
        assert result["total_reward"] > 1.0

    def test_bad_output_low_total(self) -> None:
        path = [["dog", "IsA", "animal"]]
        output = "I don't know."
        result = compute_combined_reward("What?", output, path, "animal")
        # KG reward < 0, format reward = 0.0
        assert result["total_reward"] < 0.0


# ==========================================
# Test RewardStats
# ==========================================


class TestRewardStats:
    def test_empty_stats(self) -> None:
        stats = RewardStats()
        assert stats.total == 0
        assert stats.kg_mean == 0.0
        assert stats.accuracy == 0.0

    def test_updates(self) -> None:
        stats = RewardStats()
        stats.update(kg_reward=1.3, format_reward=0.5, correct=True)
        stats.update(kg_reward=-0.2, format_reward=0.0, correct=False)
        assert stats.total == 2
        assert stats.kg_mean == pytest.approx(0.55)
        assert stats.format_mean == pytest.approx(0.25)
        assert stats.accuracy == pytest.approx(0.5)
        assert stats.min_kg == pytest.approx(-0.2)
        assert stats.max_kg == pytest.approx(1.3)
        # format_valid_count now uses >= 0.5 threshold
        assert stats.format_rate == pytest.approx(0.5)


# ==========================================
# Integration-style tests (still synthetic, no Stage 2 data)
# ==========================================


class TestRewardDiscrimination:
    """Verify the reward function can discriminate output quality."""

    PATH = [["bird", "CapableOf", "fly"], ["fly", "HasPrerequisite", "wing"]]
    GOLD = "wing"

    def test_perfect_output_beats_partial(self) -> None:
        perfect = (
            "<think>A bird is CapableOf flying. Flying HasPrerequisite "
            "having a wing. So the answer is wing.</think>\nwing"
        )
        partial = "<think>Birds can fly and need wings.</think>\nwing"
        r_perfect = compute_kg_reward("Q?", perfect, self.PATH, self.GOLD)
        r_partial = compute_kg_reward("Q?", partial, self.PATH, self.GOLD)
        assert r_perfect >= r_partial

    def test_correct_beats_incorrect(self) -> None:
        correct = (
            "<think>Birds fly with wings, following the knowledge graph path.</think>\nwing"
        )
        incorrect = (
            "<think>Birds fly with wings, following the knowledge graph path.</think>\nfeather"
        )
        r_correct = compute_kg_reward("Q?", correct, self.PATH, self.GOLD)
        r_incorrect = compute_kg_reward("Q?", incorrect, self.PATH, self.GOLD)
        # In v3, answer difference is 1.0 vs 0.0 → large gap
        assert r_correct > r_incorrect
        assert r_correct - r_incorrect > 0.5

    def test_formatted_beats_unformatted(self) -> None:
        formatted = (
            "<think>Birds fly. Flying needs wings. This is based on knowledge.</think>\nwing"
        )
        unformatted = "Birds fly. Flying needs wings. The answer is wing."
        f_fmt = compute_format_reward(formatted)
        f_unfmt = compute_format_reward(unformatted)
        assert f_fmt > f_unfmt

    def test_reward_ordering(self) -> None:
        """Verify expected ordering: good > partial > bad > garbage."""
        good = (
            "<think>A bird is CapableOf flying. Flying HasPrerequisite "
            "having a wing. Following the knowledge graph.</think>\nwing"
        )
        partial = "<think>Birds can fly and need wings for that.</think>\nwing"
        bad = "<think>Birds can fly and need wings for that.</think>\nfeather"
        garbage = "I don't know."

        rewards = [
            compute_kg_reward("Q?", out, self.PATH, self.GOLD)
            + compute_format_reward(out)
            for out in [good, partial, bad, garbage]
        ]

        assert rewards[0] > rewards[1]  # good > partial
        assert rewards[1] > rewards[2]  # partial > bad (correct vs incorrect answer)
        assert rewards[2] > rewards[3]  # bad > garbage


# ==========================================
# Test anti-hack scenarios (NEW in v3)
# ==========================================


class TestAntiHack:
    """Test that the reward function resists known hacking patterns."""

    PATH = [["dog", "IsA", "animal"], ["animal", "HasProperty", "alive"]]
    GOLD = "alive"

    def test_template_collapse_vs_genuine(self) -> None:
        """The exact pattern that killed v2 training."""
        template = (
            "<think>The knowledge provided states that dog is UsedOf "
            "animal and animal is UsedOf UsedOf )</think>\nalive"
        )
        genuine = (
            "<think>A dog IsA animal according to the knowledge graph. "
            "An animal HasProperty alive. Therefore dogs are alive.</think>\nalive"
        )
        r_template = compute_kg_reward("Q?", template, self.PATH, self.GOLD)
        r_genuine = compute_kg_reward("Q?", genuine, self.PATH, self.GOLD)

        # Template gets: R_answer=1.0, R_reasoning=moderate (entities present),
        # R_quality=-0.15 (UsedOf penalty)
        # Genuine gets: R_answer=1.0, R_reasoning=high (full triples), R_quality=0
        assert r_genuine > r_template

    def test_minimal_template_vs_genuine(self) -> None:
        """Short template with correct answer should score less than genuine reasoning."""
        minimal = "<think>dog animal</think>\nalive"
        genuine = (
            "<think>A dog IsA animal. An animal HasProperty alive. "
            "The knowledge graph shows the path clearly.</think>\nalive"
        )
        r_minimal = compute_kg_reward("Q?", minimal, self.PATH, self.GOLD)
        r_genuine = compute_kg_reward("Q?", genuine, self.PATH, self.GOLD)
        assert r_genuine > r_minimal

    def test_entity_stuffing_with_correct_answer(self) -> None:
        """Entity stuffing + correct answer should still score less than genuine."""
        stuffed = (
            "<think>" + "dog animal alive " * 20 + "</think>\nalive"
        )
        genuine = (
            "<think>A dog IsA animal according to the knowledge graph. "
            "An animal HasProperty alive. Therefore dogs are alive.</think>\nalive"
        )
        r_stuffed = compute_kg_reward("Q?", stuffed, self.PATH, self.GOLD)
        r_genuine = compute_kg_reward("Q?", genuine, self.PATH, self.GOLD)
        assert r_genuine > r_stuffed

    def test_question_echo_low_score(self) -> None:
        """Model that just echoes the question should not score well."""
        question = "What property do dogs have because they are animals?"
        echo = f"<think>{question}</think>\n{question}"
        reward = compute_kg_reward(question, echo, self.PATH, self.GOLD)
        # Echo mentions "dog" and "animal" but wrong answer
        assert reward < 0.5


# ==========================================
# Test GRPOTrainer-compatible wrappers
# ==========================================


class TestExtractText:
    def test_plain_string(self) -> None:
        assert _extract_text("hello world") == "hello world"

    def test_conversational_format(self) -> None:
        msg = [
            {"role": "user", "content": "What is a dog?"},
            {"role": "assistant", "content": "A dog is an animal."},
        ]
        text = _extract_text(msg)
        assert "What is a dog?" in text
        assert "A dog is an animal." in text

    def test_single_turn(self) -> None:
        msg = [{"role": "assistant", "content": "answer text"}]
        assert _extract_text(msg) == "answer text"

    def test_non_string_fallback(self) -> None:
        assert _extract_text(42) == "42"


class TestGRPOWrappers:
    """Test GRPOTrainer-compatible reward function wrappers."""

    SAMPLE_PATH = [["dog", "IsA", "animal"], ["animal", "HasProperty", "alive"]]
    GOOD_COMPLETION = (
        "<think>A dog is an animal (IsA relation). Animals have the "
        "property of being alive (HasProperty). Therefore, dogs are alive."
        "</think>\nalive"
    )
    BAD_COMPLETION = "I don't know the answer to this question."

    def test_kg_reward_func_return_type(self) -> None:
        prompts = ["What?", "Another?"]
        completions = [self.GOOD_COMPLETION, self.BAD_COMPLETION]
        rewards = kg_reward_func(
            prompts,
            completions,
            kg_path=[self.SAMPLE_PATH, self.SAMPLE_PATH],
            gold_answer_short=["alive", "alive"],
        )
        assert isinstance(rewards, list)
        assert len(rewards) == 2
        assert all(isinstance(r, float) for r in rewards)

    def test_format_reward_func_return_type(self) -> None:
        completions = [self.GOOD_COMPLETION, self.BAD_COMPLETION]
        rewards = format_reward_func(completions)
        assert isinstance(rewards, list)
        assert len(rewards) == 2
        assert all(isinstance(r, float) for r in rewards)

    def test_kg_reward_discriminates(self) -> None:
        prompts = ["What?", "What?"]
        completions = [self.GOOD_COMPLETION, self.BAD_COMPLETION]
        rewards = kg_reward_func(
            prompts,
            completions,
            kg_path=[self.SAMPLE_PATH, self.SAMPLE_PATH],
            gold_answer_short=["alive", "alive"],
        )
        assert rewards[0] > rewards[1]

    def test_format_reward_discriminates(self) -> None:
        completions = [self.GOOD_COMPLETION, self.BAD_COMPLETION]
        rewards = format_reward_func(completions)
        assert rewards[0] > rewards[1]

    def test_conversational_format_prompts(self) -> None:
        prompts = [
            [{"role": "user", "content": "What property do dogs have?"}],
        ]
        completions = [
            [{"role": "assistant", "content": self.GOOD_COMPLETION}],
        ]
        rewards = kg_reward_func(
            prompts,
            completions,
            kg_path=[self.SAMPLE_PATH],
            gold_answer_short=["alive"],
        )
        assert len(rewards) == 1
        assert rewards[0] > 0.0

    def test_conversational_format_rewards(self) -> None:
        completions = [
            [{"role": "assistant", "content": self.GOOD_COMPLETION}],
        ]
        rewards = format_reward_func(completions)
        assert rewards[0] == 0.5  # v3 max is 0.5

    def test_missing_kwargs_graceful(self) -> None:
        rewards = kg_reward_func(["Q?"], ["some output"])
        assert isinstance(rewards, list)
        assert len(rewards) == 1

    def test_extra_kwargs_ignored(self) -> None:
        rewards = kg_reward_func(
            ["Q?"],
            [self.GOOD_COMPLETION],
            kg_path=[self.SAMPLE_PATH],
            gold_answer_short=["alive"],
            hops=[2],
            relations=[["IsA", "HasProperty"]],
            extra_field=["something"],
        )
        assert len(rewards) == 1

    def test_format_extra_kwargs_ignored(self) -> None:
        rewards = format_reward_func(
            [self.GOOD_COMPLETION],
            kg_path=[self.SAMPLE_PATH],
            gold_answer_short=["alive"],
        )
        assert len(rewards) == 1
        assert rewards[0] == 0.5  # v3 max
