"""Stage 6: Evaluation — 3-way comparison of Base vs SFT vs SFT+GRPO.

Benchmarks:
  1. ConceptNet QA test split (500 samples, in-domain)
  2. HotpotQA bridge subset (500 samples, near-transfer)

Metrics: Exact Match (EM), token-level F1, KG path alignment score.

Usage:
    # Evaluate all three models on ConceptNet QA:
    python src/evaluation/evaluate.py \
        --benchmark conceptnet \
        --base_model Qwen/Qwen2.5-1.5B-Instruct \
        --sft_adapter outputs/sft-warmup \
        --grpo_adapter outputs/grpo/phase3_all

    # Evaluate on HotpotQA:
    python src/evaluation/evaluate.py \
        --benchmark hotpotqa \
        --base_model Qwen/Qwen2.5-1.5B-Instruct \
        --sft_adapter outputs/sft-warmup \
        --grpo_adapter outputs/grpo/phase3_all

    # Evaluate a single model:
    python src/evaluation/evaluate.py \
        --benchmark conceptnet \
        --model_path outputs/grpo/phase3_all
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import string
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.rewards.kg_reward import compute_format_reward, compute_kg_reward

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# =========================================================================
# Metric functions
# =========================================================================


def normalize_answer(text: str) -> str:
    """Normalize answer text for EM/F1 computation.

    Lowercases, removes articles, punctuation, and extra whitespace.
    Follows the SQuAD evaluation convention.
    """
    text = text.lower()
    # Remove articles
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    # Collapse whitespace
    text = " ".join(text.split())
    return text


def compute_exact_match(prediction: str, gold: str) -> float:
    """Exact match after normalization. Returns 1.0 or 0.0."""
    return 1.0 if normalize_answer(prediction) == normalize_answer(gold) else 0.0


def compute_token_f1(prediction: str, gold: str) -> float:
    """Token-level F1 score between prediction and gold answer."""
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(gold).split()

    if not gold_tokens:
        return 1.0 if not pred_tokens else 0.0
    if not pred_tokens:
        return 0.0

    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_common = sum(common.values())

    if num_common == 0:
        return 0.0

    precision = num_common / len(pred_tokens)
    recall = num_common / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def extract_final_answer(model_output: str) -> str:
    """Extract the final answer from model output (text after </think>)."""
    if "</think>" in model_output:
        return model_output.split("</think>")[-1].strip()
    return model_output.strip()


# =========================================================================
# Data loading
# =========================================================================


def load_conceptnet_test(
    path: Path,
    num_samples: int | None = None,
) -> list[dict[str, Any]]:
    """Load ConceptNet QA test split."""
    records: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if record.get("is_negative", False):
                continue
            records.append(record)
            if num_samples and len(records) >= num_samples:
                break

    logger.info("Loaded %d ConceptNet QA test samples", len(records))
    return records


def load_hotpotqa_bridge(num_samples: int = 500) -> list[dict[str, Any]]:
    """Load HotpotQA bridge subset from HuggingFace.

    Filters for 'bridge' type questions from the validation split.
    """
    logger.info("Loading HotpotQA bridge subset (up to %d samples)...", num_samples)

    try:
        ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation")
    except Exception:
        ds = load_dataset("hotpot_qa", "distractor", split="validation")
    bridge_records: list[dict[str, Any]] = []

    for sample in ds:
        if sample["type"] != "bridge":
            continue
        bridge_records.append({
            "question": sample["question"],
            "gold_answer_short": sample["answer"],
            "kg_path": [],
            "hops": 2,
        })
        if len(bridge_records) >= num_samples:
            break

    logger.info("Loaded %d HotpotQA bridge samples", len(bridge_records))
    return bridge_records


# =========================================================================
# Model loading and generation
# =========================================================================


def load_model(
    model_name: str,
    adapter_path: str | None = None,
) -> tuple[Any, Any]:
    """Load a model with optional LoRA adapter.

    Args:
        model_name: Base model name or HF path.
        adapter_path: Path to LoRA adapter directory (None for base model).

    Returns:
        (model, tokenizer) tuple.
    """
    model_kwargs: dict[str, Any] = {
        "torch_dtype": torch.bfloat16,
        "device_map": "auto",
    }

    model_kwargs["attn_implementation"] = "sdpa"
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

    if adapter_path:
        adapter_dir = Path(adapter_path)
        if adapter_dir.exists() and (adapter_dir / "adapter_config.json").exists():
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, adapter_path)
            logger.info("Loaded adapter from %s", adapter_path)
        else:
            logger.warning("Adapter not found at %s, using base model", adapter_path)

    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def generate_answer(
    model: Any,
    tokenizer: Any,
    question: str,
    max_new_tokens: int = 512,
) -> str:
    """Generate a single answer for a question using greedy decoding."""
    messages = [{"role": "user", "content": question}]
    input_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    device = next(model.parameters()).device
    inputs = tokenizer(input_text, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

    input_len = inputs["input_ids"].shape[1]
    return tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)


# =========================================================================
# Evaluation
# =========================================================================


@dataclass
class EvalResults:
    """Aggregate evaluation results for a single model."""

    model_name: str
    benchmark: str
    total: int = 0
    em_sum: float = 0.0
    f1_sum: float = 0.0
    kg_reward_sum: float = 0.0
    format_reward_sum: float = 0.0
    samples: list[dict[str, Any]] = field(default_factory=list)

    @property
    def em(self) -> float:
        return self.em_sum / max(self.total, 1)

    @property
    def f1(self) -> float:
        return self.f1_sum / max(self.total, 1)

    @property
    def kg_reward_mean(self) -> float:
        return self.kg_reward_sum / max(self.total, 1)

    @property
    def format_reward_mean(self) -> float:
        return self.format_reward_sum / max(self.total, 1)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "benchmark": self.benchmark,
            "total": self.total,
            "exact_match": round(self.em, 4),
            "token_f1": round(self.f1, 4),
            "kg_reward_mean": round(self.kg_reward_mean, 4),
            "format_reward_mean": round(self.format_reward_mean, 4),
        }


def evaluate_model(
    model: Any,
    tokenizer: Any,
    records: list[dict[str, Any]],
    model_label: str,
    benchmark: str,
    max_new_tokens: int = 512,
    num_qualitative: int = 20,
) -> EvalResults:
    """Evaluate a model on a list of QA records.

    Args:
        model: The model to evaluate.
        tokenizer: Tokenizer for the model.
        records: List of QA records with 'question' and 'gold_answer_short'.
        model_label: Human-readable label for logging.
        benchmark: Benchmark name ('conceptnet' or 'hotpotqa').
        max_new_tokens: Max tokens to generate per answer.
        num_qualitative: Number of sample outputs to save for inspection.

    Returns:
        EvalResults with aggregated metrics and sample outputs.
    """
    results = EvalResults(model_name=model_label, benchmark=benchmark)

    for i, record in enumerate(records):
        question = record["question"]
        gold = record["gold_answer_short"]
        kg_path = record.get("kg_path", [])

        # Generate
        raw_output = generate_answer(model, tokenizer, question, max_new_tokens)
        final_answer = extract_final_answer(raw_output)

        # Compute metrics
        em = compute_exact_match(final_answer, gold)
        f1 = compute_token_f1(final_answer, gold)
        kg_r = compute_kg_reward(question, raw_output, kg_path, gold) if kg_path else 0.0
        fmt_r = compute_format_reward(raw_output)

        results.total += 1
        results.em_sum += em
        results.f1_sum += f1
        results.kg_reward_sum += kg_r
        results.format_reward_sum += fmt_r

        # Save qualitative samples
        if i < num_qualitative:
            results.samples.append({
                "question": question,
                "gold_answer": gold,
                "model_output": raw_output,
                "final_answer": final_answer,
                "em": em,
                "f1": round(f1, 4),
                "kg_reward": round(kg_r, 4),
                "format_reward": fmt_r,
            })

        if (i + 1) % 50 == 0:
            logger.info(
                "  [%s] %d/%d — EM=%.3f, F1=%.3f",
                model_label, i + 1, len(records), results.em, results.f1,
            )

    logger.info(
        "[%s] Final — EM=%.4f, F1=%.4f, KG=%.4f, Fmt=%.4f (%d samples)",
        model_label, results.em, results.f1,
        results.kg_reward_mean, results.format_reward_mean, results.total,
    )
    return results


def run_three_way_comparison(
    records: list[dict[str, Any]],
    benchmark: str,
    base_model: str,
    sft_adapter: str | None,
    grpo_adapter: str | None,
    output_dir: Path,
    max_new_tokens: int = 512,
    num_qualitative: int = 20,
) -> list[EvalResults]:
    """Run 3-way evaluation: Base vs SFT-only vs SFT+GRPO.

    Loads each model, evaluates on the records, saves results to JSON.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    all_results: list[EvalResults] = []

    model_configs = [
        ("base", base_model, None),
    ]
    if sft_adapter:
        model_configs.append(("sft", base_model, sft_adapter))
    if grpo_adapter:
        model_configs.append(("sft+grpo", base_model, grpo_adapter))

    for label, model_name, adapter in model_configs:
        logger.info("=== Evaluating: %s ===", label)
        model, tokenizer = load_model(model_name, adapter)

        results = evaluate_model(
            model, tokenizer, records,
            model_label=label,
            benchmark=benchmark,
            max_new_tokens=max_new_tokens,
            num_qualitative=num_qualitative,
        )
        all_results.append(results)

        # Free GPU memory before loading next model
        del model, tokenizer
        import gc
        gc.collect()
        torch.cuda.empty_cache()

    # Save results
    summary = {
        "benchmark": benchmark,
        "num_samples": len(records),
        "models": [r.to_dict() for r in all_results],
    }
    summary_path = output_dir / f"{benchmark}_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info("Summary saved to %s", summary_path)

    # Save qualitative samples
    for results in all_results:
        samples_path = output_dir / f"{benchmark}_{results.model_name}_samples.json"
        with open(samples_path, "w", encoding="utf-8") as f:
            json.dump(results.samples, f, indent=2, ensure_ascii=False)

    # Print comparison table
    _print_comparison_table(all_results)

    return all_results


def run_single_model_eval(
    records: list[dict[str, Any]],
    benchmark: str,
    base_model: str,
    adapter_path: str | None,
    output_dir: Path,
    max_new_tokens: int = 512,
    num_qualitative: int = 20,
) -> EvalResults:
    """Evaluate a single model on a benchmark."""
    output_dir.mkdir(parents=True, exist_ok=True)

    label = "custom" if adapter_path else "base"
    model, tokenizer = load_model(base_model, adapter_path)

    results = evaluate_model(
        model, tokenizer, records,
        model_label=label,
        benchmark=benchmark,
        max_new_tokens=max_new_tokens,
        num_qualitative=num_qualitative,
    )

    # Save results
    result_path = output_dir / f"{benchmark}_{label}_results.json"
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump({
            "metrics": results.to_dict(),
            "samples": results.samples,
        }, f, indent=2, ensure_ascii=False)
    logger.info("Results saved to %s", result_path)

    return results


def _print_comparison_table(all_results: list[EvalResults]) -> None:
    """Print a formatted comparison table to the log."""
    header = f"{'Model':<12} {'EM':>8} {'F1':>8} {'KG Reward':>10} {'Format':>8}"
    separator = "-" * len(header)

    logger.info("=== Comparison Table ===")
    logger.info(header)
    logger.info(separator)
    for r in all_results:
        logger.info(
            f"{r.model_name:<12} {r.em:>8.4f} {r.f1:>8.4f} "
            f"{r.kg_reward_mean:>10.4f} {r.format_reward_mean:>8.4f}"
        )
    logger.info(separator)

    # Success criteria checks
    if len(all_results) >= 2:
        base = all_results[0]
        best = all_results[-1]
        logger.info("=== Success Criteria ===")
        logger.info(
            "  SFT+GRPO > Base (EM): %s (%.4f > %.4f)",
            "PASS" if best.em > base.em else "FAIL",
            best.em, base.em,
        )
        logger.info(
            "  SFT+GRPO > Base (F1): %s (%.4f > %.4f)",
            "PASS" if best.f1 > base.f1 else "FAIL",
            best.f1, base.f1,
        )
    if len(all_results) >= 3:
        sft = all_results[1]
        grpo = all_results[2]
        logger.info(
            "  SFT+GRPO > SFT-only (EM): %s (%.4f > %.4f)",
            "PASS" if grpo.em > sft.em else "FAIL",
            grpo.em, sft.em,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 6: Evaluation")
    parser.add_argument(
        "--benchmark",
        type=str,
        choices=["conceptnet", "hotpotqa", "both"],
        default="conceptnet",
        help="Which benchmark to evaluate on.",
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="Qwen/Qwen2.5-1.5B-Instruct",
        help="Base model name.",
    )
    parser.add_argument(
        "--sft_adapter",
        type=str,
        default=None,
        help="Path to SFT adapter (for 3-way comparison).",
    )
    parser.add_argument(
        "--grpo_adapter",
        type=str,
        default=None,
        help="Path to GRPO adapter (for 3-way comparison).",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to a single adapter to evaluate (overrides 3-way comparison).",
    )
    parser.add_argument(
        "--test_file",
        type=Path,
        default=Path("data/processed/conceptnet_qa_test.jsonl"),
        help="ConceptNet QA test file.",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=500,
        help="Max samples per benchmark.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=512,
        help="Max tokens to generate per answer.",
    )
    parser.add_argument(
        "--num_qualitative",
        type=int,
        default=20,
        help="Number of qualitative samples to save per model.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("outputs/eval"),
        help="Directory to save evaluation results.",
    )
    args = parser.parse_args()

    benchmarks = ["conceptnet", "hotpotqa"] if args.benchmark == "both" else [args.benchmark]

    for benchmark in benchmarks:
        logger.info("=== Benchmark: %s ===", benchmark)

        # Load data
        if benchmark == "conceptnet":
            if not args.test_file.exists():
                logger.error("Test file not found: %s", args.test_file)
                continue
            records = load_conceptnet_test(args.test_file, args.num_samples)
        else:
            records = load_hotpotqa_bridge(args.num_samples)

        if not records:
            logger.error("No records loaded for %s", benchmark)
            continue

        # Run evaluation
        if args.model_path:
            run_single_model_eval(
                records, benchmark, args.base_model, args.model_path,
                args.output_dir, args.max_new_tokens, args.num_qualitative,
            )
        else:
            run_three_way_comparison(
                records, benchmark, args.base_model,
                args.sft_adapter, args.grpo_adapter,
                args.output_dir, args.max_new_tokens, args.num_qualitative,
            )


if __name__ == "__main__":
    main()
