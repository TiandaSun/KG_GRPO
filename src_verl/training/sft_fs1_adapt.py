"""SFT on fs1-2708 reasoning traces (V14-C1 Part A).

Adaptation approach (not a full fs1 reproduction): our ARM Isambard stack
+ fs1's published hyperparams + fs1's public dataset.  Uses the pre-formatted
`text` field directly via TRL SFTTrainer with LoRA; produces a merged HF
checkpoint suitable for downstream GRPO in our pipeline.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

logger = logging.getLogger(__name__)


def load_fs1_dataset(jsonl_path: Path) -> Dataset:
    """Load fs1-2708 jsonl → HF Dataset with 'text' column preserved."""
    texts: list[str] = []
    for line in jsonl_path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("valid", 1) == 1 and "text" in r and r["text"]:
            texts.append(r["text"])
    logger.info("Loaded %d text-preformatted rows from %s", len(texts), jsonl_path)
    return Dataset.from_dict({"text": texts})


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_file", type=Path, default=Path("data/freebase/fs1_2708_sft.jsonl"))
    ap.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--output_dir", type=Path, default=Path("outputs/verl-sft-cwq-7b-fs1-adapt"))
    ap.add_argument("--num_train_epochs", type=int, default=5)
    ap.add_argument("--learning_rate", type=float, default=1e-5)
    ap.add_argument("--warmup_ratio", type=float, default=0.05)
    ap.add_argument("--weight_decay", type=float, default=0.0001)
    ap.add_argument("--max_seq_length", type=int, default=8192)
    ap.add_argument("--per_device_train_batch_size", type=int, default=1)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=8)
    ap.add_argument("--lora_r", type=int, default=64)
    ap.add_argument("--lora_alpha", type=int, default=128)
    args = ap.parse_args()

    ds = load_fs1_dataset(args.train_file)

    logger.info("Loading tokenizer + model: %s", args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Use eager attention (not SDPA) — avoids a torch/SDPA gradient-checkpointing
    # recompute bug where tensor metadata [B, H, L, L] vs [B, 1, L, L] mismatches
    # between forward pass and backward recomputation on GH200 ARM.
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
        trust_remote_code=True,
    )

    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        use_dora=True,
        use_rslora=True,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    sft_config = SFTConfig(
        output_dir=str(args.output_dir),
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        adam_beta1=0.9,
        adam_beta2=0.95,
        lr_scheduler_type="cosine",
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=10,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=3,
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        packing=False,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=ds,
        peft_config=lora_cfg,
    )
    # Auto-resume from latest checkpoint if any exists in output_dir
    latest_ckpts = sorted(args.output_dir.glob("checkpoint-*"),
                          key=lambda p: int(p.name.split("-")[-1]))
    resume_from = str(latest_ckpts[-1]) if latest_ckpts else None
    if resume_from:
        logger.info("Resuming from %s", resume_from)
    trainer.train(resume_from_checkpoint=resume_from)
    trainer.save_model(str(args.output_dir))
    logger.info("Training complete. Checkpoint saved to %s", args.output_dir)

    # Merge LoRA → full checkpoint for downstream verl use
    from peft import PeftModel
    merged_dir = args.output_dir.with_name(args.output_dir.name + "-merged")
    logger.info("Merging LoRA adapter → %s", merged_dir)
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.bfloat16, trust_remote_code=True
    )
    peft_model = PeftModel.from_pretrained(base_model, str(args.output_dir))
    merged = peft_model.merge_and_unload()
    merged_dir.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(merged_dir), safe_serialization=True)
    tokenizer.save_pretrained(str(merged_dir))
    logger.info("Merged checkpoint written to %s", merged_dir)


if __name__ == "__main__":
    main()
