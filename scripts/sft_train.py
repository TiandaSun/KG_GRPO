"""SFT warmup training for multi-turn KG reasoning.

Trains a LoRA adapter on gold KG trajectories using trl.SFTTrainer.
Supports both Qwen and Llama model families.

Usage:
    python scripts/sft_train.py \
        --model_name meta-llama/Llama-3.1-8B-Instruct \
        --data_path data/freebase/sft_trajectories.jsonl \
        --output_dir outputs/verl-sft-llama-8b \
        --num_epochs 2 --lr 2e-4 --batch_size 2 --grad_accum 8
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def load_trajectories(data_path: str) -> list[dict[str, Any]]:
    """Load SFT trajectories from JSONL file."""
    records = []
    with open(data_path) as f:
        for line in f:
            d = json.loads(line.strip())
            records.append({"messages": d["trajectory"]})
    logger.info("Loaded %d trajectories from %s", len(records), data_path)
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="SFT warmup for KG reasoning")
    parser.add_argument("--model_name", type=str, required=True,
                        help="HuggingFace model name or path")
    parser.add_argument("--data_path", type=str, default="data/freebase/sft_trajectories.jsonl",
                        help="Path to SFT trajectories JSONL")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for LoRA adapter")
    parser.add_argument("--num_epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch_size", type=int, default=2,
                        help="Per-device train batch size")
    parser.add_argument("--grad_accum", type=int, default=8,
                        help="Gradient accumulation steps")
    parser.add_argument("--max_seq_length", type=int, default=4096)
    parser.add_argument("--lora_rank", type=int, default=64)
    parser.add_argument("--lora_alpha", type=int, default=128)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--use_dora", action="store_true", default=True,
                        help="Use DoRA (default: True, matching Qwen SFT)")
    parser.add_argument("--no_dora", action="store_true",
                        help="Disable DoRA")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--attn_implementation", type=str, default="eager",
                        help="Attention implementation (eager for ARM/GH200)")
    args = parser.parse_args()

    if args.no_dora:
        args.use_dora = False

    logger.info("Loading tokenizer: %s", args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info("Loading model: %s", args.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float32,
        attn_implementation=args.attn_implementation,
    )

    # LoRA config — matching Qwen SFT parameters
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        use_dora=args.use_dora,
        use_rslora=True,
    )

    # Load data
    records = load_trajectories(args.data_path)
    dataset = Dataset.from_list(records)

    # Formatting function: apply chat template to convert messages → text
    def formatting_func(example: dict) -> str:
        return tokenizer.apply_chat_template(
            example["messages"], tokenize=False, add_generation_prompt=False
        )

    # SFT config
    sft_config = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        bf16=args.bf16,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=3,
        max_seq_length=args.max_seq_length,
        seed=args.seed,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to="wandb",
        run_name=f"sft-{Path(args.model_name).name}-cwq",
        dataloader_num_workers=0,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset,
        formatting_func=formatting_func,
        tokenizer=tokenizer,
        peft_config=lora_config,
    )

    logger.info("Starting SFT training: %d epochs, %d samples", args.num_epochs, len(dataset))
    trainer.train()
    trainer.save_model()
    logger.info("SFT training complete. Adapter saved to %s", args.output_dir)


if __name__ == "__main__":
    main()
