"""Unit tests for Stage 2: Question generation, quality filtering, and negative examples."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import pytest

from src.datagen.question_generator import (
    GenerationConfig,
    build_generation_prompt,
    format_path_string,
    load_existing_outputs,
    load_paths,
    parse_qa_response,
)
from src.datagen.quality_filter import (
    build_judge_prompt,
    filter_and_split,
    parse_judge_scores,
    FilterConfig,
)
from src.datagen.negative_examples import (
    NegativeConfig,
    build_corrupted_statement,
    build_negative_reasoning,
    collect_entity_pool,
    corrupt_path,
    generate_negative_examples,
    _relation_to_text,
)


# --- Fixtures ---

SAMPLE_PATHS = [
    {
        "path": [["dog", "IsA", "animal"]],
        "hops": 1,
        "relations": ["IsA"],
        "entities": ["dog", "animal"],
    },
    {
        "path": [["dog", "IsA", "animal"], ["animal", "HasProperty", "alive"]],
        "hops": 2,
        "relations": ["IsA", "HasProperty"],
        "entities": ["dog", "animal", "alive"],
    },
    {
        "path": [["bird", "CapableOf", "fly"], ["fly", "HasPrerequisite", "wing"]],
        "hops": 2,
        "relations": ["CapableOf", "HasPrerequisite"],
        "entities": ["bird", "fly", "wing"],
    },
]


@pytest.fixture
def paths_file(tmp_path: Path) -> Path:
    fpath = tmp_path / "paths.jsonl"
    with open(fpath, "w") as f:
        for record in SAMPLE_PATHS:
            f.write(json.dumps(record) + "\n")
    return fpath


@pytest.fixture
def sample_qa_records() -> list[dict[str, Any]]:
    return [
        {
            "question": "What property do dogs have because they are animals?",
            "answer": "<think>A dog is an animal. Animals are alive.</think>\nThey are alive.",
            "kg_path": [["dog", "IsA", "animal"], ["animal", "HasProperty", "alive"]],
            "gold_answer_short": "They are alive.",
            "hops": 2,
            "is_negative": False,
            "variant": 0,
            "relations": ["IsA", "HasProperty"],
        }
    ]


@pytest.fixture
def train_file(tmp_path: Path, sample_qa_records: list[dict[str, Any]]) -> Path:
    # Create a larger set for negative generation
    records = []
    for i in range(20):
        record = {**sample_qa_records[0], "question": f"Question {i}?"}
        records.append(record)
    fpath = tmp_path / "train.jsonl"
    with open(fpath, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    return fpath


# ==========================================
# Tests for question_generator.py
# ==========================================


class TestFormatPathString:
    def test_1hop(self) -> None:
        result = format_path_string([["dog", "IsA", "animal"]])
        assert result == "[dog] --IsA--> [animal]"

    def test_2hop(self) -> None:
        result = format_path_string([
            ["dog", "IsA", "animal"],
            ["animal", "HasProperty", "alive"],
        ])
        assert result == "[dog] --IsA--> [animal] --HasProperty--> [alive]"

    def test_3hop(self) -> None:
        result = format_path_string([
            ["a", "R1", "b"],
            ["b", "R2", "c"],
            ["c", "R3", "d"],
        ])
        assert result == "[a] --R1--> [b] --R2--> [c] --R3--> [d]"

    def test_empty(self) -> None:
        assert format_path_string([]) == ""


class TestBuildGenerationPrompt:
    def test_variant_0_contains_path(self) -> None:
        prompt = build_generation_prompt([["dog", "IsA", "animal"]], variant=0)
        assert "[dog] --IsA--> [animal]" in prompt
        assert "<think>" in prompt
        assert "Question:" in prompt

    def test_variant_1_different_phrasing(self) -> None:
        prompt_0 = build_generation_prompt([["dog", "IsA", "animal"]], variant=0)
        prompt_1 = build_generation_prompt([["dog", "IsA", "animal"]], variant=1)
        # Both contain the path
        assert "[dog] --IsA--> [animal]" in prompt_0
        assert "[dog] --IsA--> [animal]" in prompt_1
        # But different phrasing
        assert prompt_0 != prompt_1

    def test_1hop_says_single_step(self) -> None:
        prompt = build_generation_prompt([["dog", "IsA", "animal"]], variant=0)
        assert "single-step" in prompt

    def test_2hop_says_multi_step(self) -> None:
        prompt = build_generation_prompt(
            [["dog", "IsA", "animal"], ["animal", "HasProperty", "alive"]],
            variant=0,
        )
        assert "multi-step" in prompt


class TestParseQAResponse:
    def test_valid_response_with_think_tags(self) -> None:
        response = (
            "Question: What property do dogs have?\n"
            "Answer: <think>Dogs are animals. Animals are alive.</think>\n"
            "They are alive."
        )
        result = parse_qa_response(response)
        assert result is not None
        question, full_answer, short_answer = result
        assert "property" in question
        assert "<think>" in full_answer
        assert "alive" in short_answer

    def test_valid_response_without_think_tags(self) -> None:
        response = (
            "Question: What are dogs?\n"
            "Answer: Dogs are domesticated animals that are loyal companions."
        )
        result = parse_qa_response(response)
        assert result is not None
        _, _, short_answer = result
        assert "domesticated" in short_answer

    def test_missing_question(self) -> None:
        response = "Answer: Dogs are animals."
        assert parse_qa_response(response) is None

    def test_missing_answer(self) -> None:
        response = "Question: What are dogs?"
        assert parse_qa_response(response) is None

    def test_too_short_question(self) -> None:
        response = "Question: Dogs?\nAnswer: Dogs are domesticated animals that live with humans."
        assert parse_qa_response(response) is None

    def test_too_short_answer(self) -> None:
        response = "Question: What property do dogs have because they are animals?\nAnswer: alive"
        assert parse_qa_response(response) is None


class TestLoadPaths:
    def test_loads_paths(self, paths_file: Path) -> None:
        paths = load_paths(paths_file)
        assert len(paths) == 3
        assert paths[0]["hops"] == 1
        assert paths[1]["hops"] == 2


class TestLoadExistingOutputs:
    def test_empty_when_no_file(self, tmp_path: Path) -> None:
        result = load_existing_outputs(tmp_path / "nonexistent.jsonl")
        assert len(result) == 0

    def test_loads_existing_keys(self, tmp_path: Path) -> None:
        output_file = tmp_path / "existing.jsonl"
        record = {
            "kg_path": [["dog", "IsA", "animal"]],
            "variant": 0,
            "question": "test?",
        }
        output_file.write_text(json.dumps(record) + "\n")
        result = load_existing_outputs(output_file)
        assert len(result) == 1


# ==========================================
# Tests for quality_filter.py
# ==========================================


class TestBuildJudgePrompt:
    def test_contains_all_elements(self) -> None:
        prompt = build_judge_prompt(
            question="What are dogs?",
            answer="<think>Dogs are animals.</think>\nAnimals.",
            kg_path=[["dog", "IsA", "animal"]],
        )
        assert "What are dogs?" in prompt
        assert "dog" in prompt
        assert "IsA" in prompt
        assert "Answerability" in prompt
        assert "Faithfulness" in prompt
        assert "Naturalness" in prompt

    def test_multiline_path(self) -> None:
        prompt = build_judge_prompt(
            question="test?",
            answer="test answer with enough characters",
            kg_path=[
                ["dog", "IsA", "animal"],
                ["animal", "HasProperty", "alive"],
            ],
        )
        assert "dog --IsA--> animal" in prompt
        assert "animal --HasProperty--> alive" in prompt


class TestParseJudgeScores:
    def test_valid_scores(self) -> None:
        response = "Answerability: 5\nFaithfulness: 4\nNaturalness: 3"
        scores = parse_judge_scores(response)
        assert scores is not None
        assert scores["answerability"] == 5
        assert scores["faithfulness"] == 4
        assert scores["naturalness"] == 3

    def test_scores_with_extra_text(self) -> None:
        response = (
            "Here are my scores:\n"
            "Answerability: 4\n"
            "Faithfulness: 5\n"
            "Naturalness: 4\n"
            "Overall good quality."
        )
        scores = parse_judge_scores(response)
        assert scores is not None
        assert scores["answerability"] == 4

    def test_missing_dimension(self) -> None:
        response = "Answerability: 5\nFaithfulness: 4"
        assert parse_judge_scores(response) is None

    def test_out_of_range(self) -> None:
        response = "Answerability: 7\nFaithfulness: 4\nNaturalness: 3"
        assert parse_judge_scores(response) is None

    def test_non_numeric(self) -> None:
        response = "Answerability: high\nFaithfulness: 4\nNaturalness: 3"
        assert parse_judge_scores(response) is None


class TestFilterAndSplit:
    def _make_scored_records(self, n: int, score: int = 5) -> list[dict[str, Any]]:
        records = []
        for i in range(n):
            records.append({
                "question": f"Question {i}?",
                "answer": f"Answer {i}",
                "kg_path": [["a", "IsA", "b"]],
                "gold_answer_short": f"Answer {i}",
                "hops": 1,
                "is_negative": False,
                "scores": {
                    "answerability": score,
                    "faithfulness": score,
                    "naturalness": score,
                },
            })
        return records

    def test_all_pass(self) -> None:
        records = self._make_scored_records(100, score=5)
        config = FilterConfig(
            input_file=Path("dummy"),
            output_dir=Path("dummy"),
            min_score=4,
            target_total=100,
            train_count=80,
            val_count=10,
            test_count=10,
            seed=42,
        )
        train, val, test = filter_and_split(records, config)
        assert len(train) == 80
        assert len(val) == 10
        assert len(test) == 10

    def test_none_pass(self) -> None:
        records = self._make_scored_records(100, score=2)
        config = FilterConfig(
            input_file=Path("dummy"),
            output_dir=Path("dummy"),
            min_score=4,
            seed=42,
        )
        train, val, test = filter_and_split(records, config)
        assert len(train) == 0
        assert len(val) == 0
        assert len(test) == 0

    def test_proportional_split_when_few(self) -> None:
        records = self._make_scored_records(50, score=5)
        config = FilterConfig(
            input_file=Path("dummy"),
            output_dir=Path("dummy"),
            min_score=4,
            target_total=5000,
            train_count=4000,
            val_count=500,
            test_count=500,
            seed=42,
        )
        train, val, test = filter_and_split(records, config)
        # Should fall back to 80/10/10 proportional
        assert len(train) == 40
        assert len(val) == 5
        assert len(test) == 5

    def test_subsamples_when_too_many(self) -> None:
        records = self._make_scored_records(200, score=5)
        config = FilterConfig(
            input_file=Path("dummy"),
            output_dir=Path("dummy"),
            min_score=4,
            target_total=100,
            train_count=80,
            val_count=10,
            test_count=10,
            seed=42,
        )
        train, val, test = filter_and_split(records, config)
        assert len(train) == 80
        assert len(val) == 10
        assert len(test) == 10


# ==========================================
# Tests for negative_examples.py
# ==========================================


class TestRelationToText:
    def test_known_relation(self) -> None:
        text = _relation_to_text("IsA", "dog", "animal")
        assert text == "dog is a type of animal"

    def test_has_property(self) -> None:
        text = _relation_to_text("HasProperty", "animal", "alive")
        assert text == "animal has the property of being alive"

    def test_unknown_relation(self) -> None:
        text = _relation_to_text("WeirdRelation", "a", "b")
        assert text == "a WeirdRelation b"


class TestCorruptPath:
    def test_changes_one_entity(self) -> None:
        rng = random.Random(42)
        path = [["dog", "IsA", "animal"]]
        entity_pool = ["cat", "fish", "bird", "car", "tree"]

        corrupted, original, replacement = corrupt_path(path, entity_pool, rng)
        assert original != replacement
        # At least one entity should differ
        flat_orig = [path[0][0], path[0][2]]
        flat_corrupt = [corrupted[0][0], corrupted[0][2]]
        assert flat_orig != flat_corrupt

    def test_maintains_relation(self) -> None:
        rng = random.Random(42)
        path = [["dog", "IsA", "animal"]]
        entity_pool = ["cat", "fish", "bird", "car", "tree"]

        corrupted, _, _ = corrupt_path(path, entity_pool, rng)
        assert corrupted[0][1] == "IsA"  # Relation unchanged

    def test_2hop_maintains_structure(self) -> None:
        rng = random.Random(42)
        path = [["dog", "IsA", "animal"], ["animal", "HasProperty", "alive"]]
        entity_pool = ["cat", "fish", "bird", "car", "tree"]

        corrupted, _, _ = corrupt_path(path, entity_pool, rng)
        # Should still be 2 triples
        assert len(corrupted) == 2
        # Relations preserved
        assert corrupted[0][1] == "IsA"
        assert corrupted[1][1] == "HasProperty"


class TestBuildCorruptedStatement:
    def test_single_hop(self) -> None:
        statement = build_corrupted_statement([["dog", "IsA", "vehicle"]])
        assert "dog" in statement
        assert "vehicle" in statement

    def test_multi_hop(self) -> None:
        statement = build_corrupted_statement([
            ["dog", "IsA", "vehicle"],
            ["vehicle", "HasProperty", "alive"],
        ])
        assert "dog" in statement
        assert "vehicle" in statement
        assert "alive" in statement


class TestBuildNegativeReasoning:
    def test_mentions_entities(self) -> None:
        reasoning = build_negative_reasoning(
            [["dog", "IsA", "vehicle"]],
            original_entity="animal",
            replacement_entity="vehicle",
        )
        assert "vehicle" in reasoning
        assert "animal" in reasoning


class TestCollectEntityPool:
    def test_collects_entities(self, paths_file: Path) -> None:
        pool = collect_entity_pool(paths_file)
        assert "dog" in pool
        assert "animal" in pool
        assert "alive" in pool
        assert "bird" in pool


class TestGenerateNegativeExamples:
    def test_generates_negatives(self, train_file: Path, paths_file: Path) -> None:
        config = NegativeConfig(
            train_file=train_file,
            paths_file=paths_file,
            output_file=Path("dummy"),
            num_negatives=5,
            seed=42,
        )
        negatives = generate_negative_examples(config)
        assert len(negatives) > 0
        assert all(r["is_negative"] for r in negatives)

    def test_negative_format(self, train_file: Path, paths_file: Path) -> None:
        config = NegativeConfig(
            train_file=train_file,
            paths_file=paths_file,
            output_file=Path("dummy"),
            num_negatives=3,
            seed=42,
        )
        negatives = generate_negative_examples(config)
        for neg in negatives:
            assert neg["question"].startswith("Is it true that")
            assert "<think>" in neg["answer"]
            assert "</think>" in neg["answer"]
            assert "incorrect" in neg["answer"].lower()
            assert neg["is_negative"] is True

    def test_respects_num_negatives(self, train_file: Path, paths_file: Path) -> None:
        config = NegativeConfig(
            train_file=train_file,
            paths_file=paths_file,
            output_file=Path("dummy"),
            num_negatives=3,
            seed=42,
        )
        negatives = generate_negative_examples(config)
        assert len(negatives) <= 3
