"""Stage 4: SFT warmup — teaches <think> CoT format before GRPO training.

Trains a LoRA (DoRA + rsLoRA) adapter on top of Qwen2.5-1.5B using the
generated QA data from Stage 2. This addresses the cold-start problem
identified in DeepSeek-R1 for small models.

Usage:
    python src/training/sft_warmup.py --config configs/sft_warmup.yaml

    # Override output directory:
    python src/training/sft_warmup.py --config configs/sft_warmup.yaml \
        --output_dir outputs/sft-warmup-v2
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import yaml
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class SFTWarmupConfig:
    """Configuration for SFT warmup training."""

    # Model
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    torch_dtype: str = "bfloat16"
    attn_implementation: str = "flash_attention_2"

    # LoRA
    lora_r: int = 16
    lora_alpha: int = 32
    target_modules: str = "all-linear"
    lora_dropout: float = 0.05
    use_dora: bool = True
    use_rslora: bool = True

    # SFT training
    num_train_epochs: int = 2
    per_device_train_batch_size: int = 8
    learning_rate: float = 2e-4
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.05
    gradient_checkpointing: bool = True
    max_seq_length: int = 512
    logging_steps: int = 10
    save_strategy: str = "epoch"
    optim: str = "adamw_8bit"
    bf16: bool = True
    seed: int = 42
    report_to: str = "wandb"
    run_name: str = "kg-align-sft-warmup"

    # Data
    train_file: str = "data/processed/conceptnet_qa_train.jsonl"
    val_file: str = "data/processed/conceptnet_qa_val.jsonl"

    # Output
    output_dir: str = "outputs/sft-warmup"

    @classmethod
    def from_yaml(cls, path: Path) -> SFTWarmupConfig:
        """Load config from YAML file."""
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)

        model_cfg = raw.get("model", {})
        lora_cfg = raw.get("lora", {})
        sft_cfg = raw.get("sft", {})
        data_cfg = raw.get("data", {})
        output_cfg = raw.get("output", {})

        return cls(
            model_name=model_cfg.get("name", cls.model_name),
            torch_dtype=model_cfg.get("torch_dtype", cls.torch_dtype),
            attn_implementation=model_cfg.get("attn_implementation", cls.attn_implementation),
            lora_r=lora_cfg.get("r", cls.lora_r),
            lora_alpha=lora_cfg.get("lora_alpha", cls.lora_alpha),
            target_modules=lora_cfg.get("target_modules", cls.target_modules),
            lora_dropout=lora_cfg.get("lora_dropout", cls.lora_dropout),
            use_dora=lora_cfg.get("use_dora", cls.use_dora),
            use_rslora=lora_cfg.get("use_rslora", cls.use_rslora),
            num_train_epochs=sft_cfg.get("num_train_epochs", cls.num_train_epochs),
            per_device_train_batch_size=sft_cfg.get("per_device_train_batch_size", cls.per_device_train_batch_size),
            learning_rate=sft_cfg.get("learning_rate", cls.learning_rate),
            lr_scheduler_type=sft_cfg.get("lr_scheduler_type", cls.lr_scheduler_type),
            warmup_ratio=sft_cfg.get("warmup_ratio", cls.warmup_ratio),
            gradient_checkpointing=sft_cfg.get("gradient_checkpointing", cls.gradient_checkpointing),
            max_seq_length=sft_cfg.get("max_seq_length", cls.max_seq_length),
            logging_steps=sft_cfg.get("logging_steps", cls.logging_steps),
            save_strategy=sft_cfg.get("save_strategy", cls.save_strategy),
            optim=sft_cfg.get("optim", cls.optim),
            bf16=sft_cfg.get("bf16", cls.bf16),
            seed=sft_cfg.get("seed", cls.seed),
            report_to=sft_cfg.get("report_to", cls.report_to),
            run_name=sft_cfg.get("run_name", cls.run_name),
            train_file=data_cfg.get("train_file", cls.train_file),
            val_file=data_cfg.get("val_file", cls.val_file),
            output_dir=output_cfg.get("dir", cls.output_dir),
        )


def load_qa_dataset(path: Path) -> Dataset:
    """Load QA pairs from JSONL and format as conversational prompt-completion.

    Skips negative examples (is_negative=True). Formats each record as:
        prompt: [{"role": "user", "content": question}]
        completion: [{"role": "assistant", "content": answer}]

    Args:
        path: Path to JSONL file with question/answer fields.

    Returns:
        HuggingFace Dataset with 'prompt' and 'completion' columns.
    """
    prompts: list[list[dict[str, str]]] = []
    completions: list[list[dict[str, str]]] = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if record.get("is_negative", False):
                continue

            prompts.append([{"role": "user", "content": record["question"]}])
            completions.append([{"role": "assistant", "content": record["answer"]}])

    logger.info("Loaded %d QA pairs from %s (negatives excluded)", len(prompts), path)
    return Dataset.from_dict({"prompt": prompts, "completion": completions})


def build_lora_config(config: SFTWarmupConfig) -> LoraConfig:
    """Build LoRA config with DoRA/rsLoRA fallbacks.

    If DoRA or rsLoRA are not supported by the installed PEFT version,
    falls back gracefully to standard LoRA.
    """
    lora_kwargs: dict[str, Any] = {
        "r": config.lora_r,
        "lora_alpha": config.lora_alpha,
        "target_modules": config.target_modules,
        "lora_dropout": config.lora_dropout,
        "task_type": TaskType.CAUSAL_LM,
        "bias": "none",
    }

    # Try DoRA
    if config.use_dora:
        try:
            test_cfg = LoraConfig(use_dora=True, r=4, target_modules=["q_proj"])
            lora_kwargs["use_dora"] = True
            logger.info("DoRA enabled")
        except (TypeError, ValueError):
            logger.warning("DoRA not supported by installed PEFT version, using standard LoRA")

    # Try rsLoRA
    if config.use_rslora:
        try:
            test_cfg = LoraConfig(use_rslora=True, r=4, target_modules=["q_proj"])
            lora_kwargs["use_rslora"] = True
            logger.info("rsLoRA enabled")
        except (TypeError, ValueError):
            logger.warning("rsLoRA not supported by installed PEFT version, skipping")

    return LoraConfig(**lora_kwargs)


def train_sft(config: SFTWarmupConfig) -> None:
    """Run SFT warmup training.

    1. Load base model with optional Flash Attention 2
    2. Apply LoRA (DoRA + rsLoRA) adapter
    3. Load and format training/validation data
    4. Train with SFTTrainer
    5. Save adapter to output directory
    """
    output_path = Path(config.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Resolve torch dtype
    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    torch_dtype = dtype_map.get(config.torch_dtype, torch.bfloat16)

    # Load model with Flash Attention fallback
    model_kwargs: dict[str, Any] = {
        "torch_dtype": torch_dtype,
        "device_map": "auto",
    }
    try:
        model_kwargs["attn_implementation"] = config.attn_implementation
        model = AutoModelForCausalLM.from_pretrained(config.model_name, **model_kwargs)
        logger.info("Loaded model with %s", config.attn_implementation)
    except (ValueError, ImportError):
        logger.warning(
            "Flash Attention 2 not available, falling back to default attention"
        )
        model_kwargs.pop("attn_implementation", None)
        model = AutoModelForCausalLM.from_pretrained(config.model_name, **model_kwargs)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Apply LoRA
    lora_config = build_lora_config(config)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Load datasets
    train_dataset = load_qa_dataset(Path(config.train_file))
    eval_dataset = None
    val_path = Path(config.val_file)
    if val_path.exists():
        eval_dataset = load_qa_dataset(val_path)

    # Configure SFT training
    sft_config = SFTConfig(
        output_dir=config.output_dir,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        learning_rate=config.learning_rate,
        lr_scheduler_type=config.lr_scheduler_type,
        warmup_ratio=config.warmup_ratio,
        gradient_checkpointing=config.gradient_checkpointing,
        max_length=config.max_seq_length,
        logging_steps=config.logging_steps,
        save_strategy=config.save_strategy,
        optim=config.optim,
        bf16=config.bf16,
        seed=config.seed,
        report_to=config.report_to,
        run_name=config.run_name,
        remove_unused_columns=False,
        gradient_checkpointing_kwargs={"use_reentrant": False} if config.gradient_checkpointing else None,
    )

    # Create trainer
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
    )

    # Train
    logger.info("Starting SFT warmup training")
    logger.info("  Epochs: %d", config.num_train_epochs)
    logger.info("  Batch size: %d", config.per_device_train_batch_size)
    logger.info("  Learning rate: %s", config.learning_rate)
    logger.info("  Output: %s", config.output_dir)

    trainer.train()

    # Save adapter
    trainer.save_model(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)
    logger.info("SFT warmup adapter saved to %s", config.output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 4: SFT warmup training")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/sft_warmup.yaml"),
        help="Path to YAML config file.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Override output directory from config.",
    )
    args = parser.parse_args()

    config = SFTWarmupConfig.from_yaml(args.config)
    if args.output_dir:
        config.output_dir = args.output_dir

    train_sft(config)


if __name__ == "__main__":
    main()
