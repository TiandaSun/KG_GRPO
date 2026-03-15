"""Merge SFT LoRA adapter into base model for verl GRPO training.

verl's FSDP backend loads a full model (not PEFT adapters), so we merge
the SFT warmup adapter into the base model and save the result.

Usage:
    python scripts/merge_sft_adapter.py \
        --base_model Qwen/Qwen2.5-1.5B-Instruct \
        --adapter_path outputs/verl-sft \
        --output_path outputs/verl-sft-merged
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge LoRA adapter into base model.")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--adapter_path", type=Path, default=Path("outputs/verl-sft"))
    parser.add_argument("--output_path", type=Path, default=Path("outputs/verl-sft-merged"))
    args = parser.parse_args()

    logger.info("Loading base model: %s", args.base_model)
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
    )

    logger.info("Loading adapter from: %s", args.adapter_path)
    model = PeftModel.from_pretrained(base_model, str(args.adapter_path))

    logger.info("Merging adapter weights...")
    model = model.merge_and_unload()

    logger.info("Saving merged model to: %s", args.output_path)
    args.output_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output_path)

    logger.info("Saving tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    tokenizer.save_pretrained(args.output_path)

    logger.info("Done. Merged model saved to %s", args.output_path)


if __name__ == "__main__":
    main()
