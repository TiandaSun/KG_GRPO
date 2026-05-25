"""Merge LoRA adapter into base model for verl GRPO training.

verl requires a full (non-PEFT) model for rollout. This script merges
a trained LoRA adapter back into the base model and saves the result.

Usage:
    python scripts/merge_sft_adapter.py \
        --base_model meta-llama/Llama-3.1-8B-Instruct \
        --adapter_path outputs/verl-sft-llama-8b \
        --output_path outputs/verl-sft-llama-8b-merged
"""

from __future__ import annotations

import argparse
import logging

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge LoRA adapter into base model")
    parser.add_argument("--base_model", type=str, required=True,
                        help="HuggingFace model name or path for the base model")
    parser.add_argument("--adapter_path", type=str, required=True,
                        help="Path to the LoRA adapter directory")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Output path for the merged model")
    parser.add_argument("--tokenizer_path", type=str, default=None,
                        help="Optional tokenizer source; defaults to adapter_path. Use when base weights and chat-template-bearing tokenizer differ (e.g. Llama-3.1-8B Base + Instruct tokenizer).")
    parser.add_argument("--bf16", action="store_true", default=True)
    args = parser.parse_args()

    logger.info("Loading base model: %s", args.base_model)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float32,
        device_map="cpu",
    )

    logger.info("Loading adapter: %s", args.adapter_path)
    model = PeftModel.from_pretrained(model, args.adapter_path)

    logger.info("Merging adapter into base model...")
    model = model.merge_and_unload()

    logger.info("Saving merged model to: %s", args.output_path)
    model.save_pretrained(args.output_path, safe_serialization=True)

    # Save tokenizer too — prefer explicit tokenizer_path (carries chat template
    # when base weights themselves do not have one, e.g. Llama-3.1-8B Base).
    logger.info("Saving tokenizer...")
    tok_src = args.tokenizer_path if args.tokenizer_path else args.adapter_path
    logger.info("Tokenizer source: %s", tok_src)
    tokenizer = AutoTokenizer.from_pretrained(tok_src)
    tokenizer.save_pretrained(args.output_path)

    logger.info("Done. Merged model saved to %s", args.output_path)


if __name__ == "__main__":
    main()
