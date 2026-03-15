#!/usr/bin/env python3
"""Diagnose EOS termination failure in GRPO training.

Tests multiple hypotheses for why completions hit max_length=512:
1. Tokenizer EOS configuration
2. Chat template formatting
3. SFT model generation behavior
4. SFT training data truncation analysis
5. Generation config propagation in GRPOConfig
"""

import json
import logging
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

MODEL_NAME = "Qwen/Qwen2.5-1.5B"
SFT_ADAPTER = "outputs/sft-warmup"
TRAIN_FILE = "data/processed/conceptnet_qa_train.jsonl"

SEPARATOR = "=" * 70


def test_tokenizer_eos():
    """Test 1: Check tokenizer EOS token configuration."""
    logger.info(f"\n{SEPARATOR}")
    logger.info("TEST 1: Tokenizer EOS Configuration")
    logger.info(SEPARATOR)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    logger.info(f"  eos_token:        '{tokenizer.eos_token}' (id={tokenizer.eos_token_id})")
    logger.info(f"  pad_token:        '{tokenizer.pad_token}' (id={tokenizer.pad_token_id})")
    logger.info(f"  bos_token:        '{tokenizer.bos_token}' (id={tokenizer.bos_token_id})")

    # Check special tokens
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    im_start_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
    endoftext_id = tokenizer.convert_tokens_to_ids("<|endoftext|>")

    logger.info(f"  <|im_end|>:       id={im_end_id}")
    logger.info(f"  <|im_start|>:     id={im_start_id}")
    logger.info(f"  <|endoftext|>:    id={endoftext_id}")

    # Check if im_end_id is UNK
    logger.info(f"  unk_token_id:     {tokenizer.unk_token_id}")
    logger.info(f"  <|im_end|> is UNK: {im_end_id == tokenizer.unk_token_id}")

    # Check model's generation_config
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16)
    logger.info(f"\n  Model generation_config.eos_token_id: {model.generation_config.eos_token_id}")

    # Decode the EOS tokens to verify
    for token_id in [151643, 151645]:
        decoded = tokenizer.decode([token_id])
        logger.info(f"  Token {token_id} decodes to: '{decoded}'")

    del model
    return tokenizer


def test_chat_template(tokenizer):
    """Test 2: Check what the chat template produces."""
    logger.info(f"\n{SEPARATOR}")
    logger.info("TEST 2: Chat Template Analysis")
    logger.info(SEPARATOR)

    # Check if tokenizer has chat template
    has_template = hasattr(tokenizer, 'chat_template') and tokenizer.chat_template is not None
    logger.info(f"  Has chat_template: {has_template}")

    if has_template:
        logger.info(f"  Template (first 200 chars): {tokenizer.chat_template[:200]}...")

    # Apply template to a sample prompt
    messages = [{"role": "user", "content": "What is a dog?"}]
    try:
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        logger.info(f"\n  Formatted prompt (add_generation_prompt=True):")
        logger.info(f"  '{formatted}'")

        # Tokenize and show token IDs
        token_ids = tokenizer.encode(formatted, add_special_tokens=False)
        logger.info(f"  Token count: {len(token_ids)}")
        logger.info(f"  Last 5 tokens: {token_ids[-5:]}")
        logger.info(f"  Last 5 decoded: {[tokenizer.decode([t]) for t in token_ids[-5:]]}")
    except Exception as e:
        logger.info(f"  chat_template failed: {e}")

    # Also format a full conversation (with assistant response) to see what EOS looks like
    full_messages = [
        {"role": "user", "content": "What is a dog?"},
        {"role": "assistant", "content": "<think>A dog is an animal.</think>\nA pet."},
    ]
    try:
        formatted_full = tokenizer.apply_chat_template(
            full_messages, tokenize=False, add_generation_prompt=False
        )
        logger.info(f"\n  Full conversation (with assistant response):")
        logger.info(f"  '{formatted_full}'")

        token_ids_full = tokenizer.encode(formatted_full, add_special_tokens=False)
        logger.info(f"  Token count: {len(token_ids_full)}")
        logger.info(f"  Last 10 tokens: {token_ids_full[-10:]}")
        logger.info(f"  Last 10 decoded: {[tokenizer.decode([t]) for t in token_ids_full[-10:]]}")

        # Check if <|im_end|> appears in the full conversation
        im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
        im_end_positions = [i for i, t in enumerate(token_ids_full) if t == im_end_id]
        logger.info(f"  <|im_end|> positions: {im_end_positions}")
    except Exception as e:
        logger.info(f"  Full conversation formatting failed: {e}")


def test_sft_data_lengths(tokenizer):
    """Test 3: Analyze SFT training data lengths for truncation."""
    logger.info(f"\n{SEPARATOR}")
    logger.info("TEST 3: SFT Training Data Length Analysis")
    logger.info(SEPARATOR)

    train_path = Path(TRAIN_FILE)
    if not train_path.exists():
        logger.info(f"  Train file not found: {train_path}")
        return

    records = []
    with open(train_path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    logger.info(f"  Total records: {len(records)}")

    # For each record, tokenize the full conversation (prompt + answer)
    # to check if it exceeds max_seq_length=512
    total_lengths = []
    prompt_lengths = []
    completion_lengths = []
    truncated_count = 0

    for record in records:
        if record.get("is_negative", False):
            continue

        # Format as chat template would
        messages = [
            {"role": "user", "content": record["question"]},
            {"role": "assistant", "content": record["answer"]},
        ]
        try:
            formatted = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            token_ids = tokenizer.encode(formatted, add_special_tokens=False)
            total_len = len(token_ids)
            total_lengths.append(total_len)

            # Also measure prompt-only length
            prompt_messages = [{"role": "user", "content": record["question"]}]
            prompt_formatted = tokenizer.apply_chat_template(
                prompt_messages, tokenize=False, add_generation_prompt=True
            )
            prompt_len = len(tokenizer.encode(prompt_formatted, add_special_tokens=False))
            prompt_lengths.append(prompt_len)
            completion_lengths.append(total_len - prompt_len)

            if total_len > 512:
                truncated_count += 1
        except Exception:
            continue

    if total_lengths:
        import statistics
        logger.info(f"  Records analyzed: {len(total_lengths)}")
        logger.info(f"\n  Total sequence lengths (prompt + completion):")
        logger.info(f"    Mean:   {statistics.mean(total_lengths):.1f}")
        logger.info(f"    Median: {statistics.median(total_lengths):.1f}")
        logger.info(f"    Min:    {min(total_lengths)}")
        logger.info(f"    Max:    {max(total_lengths)}")
        logger.info(f"    Std:    {statistics.stdev(total_lengths):.1f}")
        logger.info(f"    > 512:  {truncated_count}/{len(total_lengths)} ({100*truncated_count/len(total_lengths):.1f}%)")

        pcts = [256, 384, 512, 640, 768, 1024]
        for p in pcts:
            over = sum(1 for l in total_lengths if l > p)
            logger.info(f"    > {p}:  {over}/{len(total_lengths)} ({100*over/len(total_lengths):.1f}%)")

        logger.info(f"\n  Prompt-only lengths:")
        logger.info(f"    Mean:   {statistics.mean(prompt_lengths):.1f}")
        logger.info(f"    Max:    {max(prompt_lengths)}")

        logger.info(f"\n  Completion-only lengths:")
        logger.info(f"    Mean:   {statistics.mean(completion_lengths):.1f}")
        logger.info(f"    Median: {statistics.median(completion_lengths):.1f}")
        logger.info(f"    Max:    {max(completion_lengths)}")
        over_512 = sum(1 for l in completion_lengths if l > 512)
        logger.info(f"    > 512:  {over_512}/{len(completion_lengths)} ({100*over_512/len(completion_lengths):.1f}%)")

        # Check if EOS token is at the end of non-truncated examples
        logger.info(f"\n  EOS token analysis in training data:")
        im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
        endoftext_id = tokenizer.convert_tokens_to_ids("<|endoftext|>")
        has_eos_count = 0
        for record in records[:100]:  # Sample first 100
            if record.get("is_negative", False):
                continue
            messages = [
                {"role": "user", "content": record["question"]},
                {"role": "assistant", "content": record["answer"]},
            ]
            try:
                formatted = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=False
                )
                token_ids = tokenizer.encode(formatted, add_special_tokens=False)
                last_token = token_ids[-1] if token_ids else None
                if last_token in [im_end_id, endoftext_id]:
                    has_eos_count += 1
            except Exception:
                continue
        logger.info(f"    Last token is EOS/im_end (first 100): {has_eos_count}/100")


def test_sft_model_generation(tokenizer):
    """Test 4: Load SFT model and generate to see if it produces EOS."""
    logger.info(f"\n{SEPARATOR}")
    logger.info("TEST 4: SFT Model Generation Test")
    logger.info(SEPARATOR)

    sft_path = Path(SFT_ADAPTER)
    if not sft_path.exists():
        logger.info(f"  SFT adapter not found: {sft_path}")
        return

    # Load model + adapter
    from peft import PeftModel
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model = PeftModel.from_pretrained(model, str(sft_path))
    model.eval()

    # Sample questions
    questions = [
        "What property do dogs have because they are animals?",
        "Where can you find a kitchen?",
        "What is a cat?",
    ]

    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    endoftext_id = tokenizer.convert_tokens_to_ids("<|endoftext|>")

    for q in questions:
        messages = [{"role": "user", "content": q}]
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(formatted, return_tensors="pt").to(model.device)
        prompt_len = inputs["input_ids"].shape[1]

        # Generate with BOTH EOS tokens
        gen_config = GenerationConfig(
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7,
            eos_token_id=[endoftext_id, im_end_id],
            pad_token_id=tokenizer.pad_token_id,
        )

        with torch.no_grad():
            outputs = model.generate(**inputs, generation_config=gen_config)

        completion_ids = outputs[0][prompt_len:]
        completion_text = tokenizer.decode(completion_ids, skip_special_tokens=False)
        completion_len = len(completion_ids)

        # Check termination
        last_token = completion_ids[-1].item() if len(completion_ids) > 0 else None
        terminated = last_token in [im_end_id, endoftext_id]

        # Check if any EOS token appears anywhere
        has_im_end = (completion_ids == im_end_id).any().item()
        has_endoftext = (completion_ids == endoftext_id).any().item()

        logger.info(f"\n  Question: '{q}'")
        logger.info(f"  Completion length: {completion_len} tokens")
        logger.info(f"  Last token: {last_token} ({tokenizer.decode([last_token]) if last_token else 'N/A'})")
        logger.info(f"  Terminated naturally: {terminated}")
        logger.info(f"  Contains <|im_end|>: {has_im_end}")
        logger.info(f"  Contains <|endoftext|>: {has_endoftext}")
        logger.info(f"  Completion (first 300 chars): {completion_text[:300]}")

    # Also test with just endoftext EOS (TRL default without our fix)
    logger.info(f"\n  --- Test with ONLY default EOS (151643 = <|endoftext|>) ---")
    q = questions[0]
    messages = [{"role": "user", "content": q}]
    formatted = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(formatted, return_tensors="pt").to(model.device)
    prompt_len = inputs["input_ids"].shape[1]

    gen_config_default = GenerationConfig(
        max_new_tokens=512,
        do_sample=True,
        temperature=0.7,
        eos_token_id=endoftext_id,  # Only default, no im_end
        pad_token_id=tokenizer.pad_token_id,
    )

    with torch.no_grad():
        outputs = model.generate(**inputs, generation_config=gen_config_default)

    completion_ids = outputs[0][prompt_len:]
    completion_len = len(completion_ids)
    last_token = completion_ids[-1].item()
    logger.info(f"  Completion length: {completion_len} tokens")
    logger.info(f"  Last token: {last_token} ({tokenizer.decode([last_token])})")
    logger.info(f"  Hit max_length: {completion_len >= 512}")

    del model


def test_grpo_config_propagation():
    """Test 5: Verify GRPOConfig properly propagates generation_kwargs."""
    logger.info(f"\n{SEPARATOR}")
    logger.info("TEST 5: GRPOConfig generation_kwargs Propagation")
    logger.info(SEPARATOR)

    from trl import GRPOConfig

    # Create config with our EOS fix
    config = GRPOConfig(
        output_dir="/tmp/test_grpo",
        max_steps=1,
        generation_kwargs={"eos_token_id": [151643, 151645]},
    )

    logger.info(f"  config.generation_kwargs: {config.generation_kwargs}")
    logger.info(f"  type: {type(config.generation_kwargs)}")

    # Check if it's properly stored
    if config.generation_kwargs and "eos_token_id" in config.generation_kwargs:
        eos = config.generation_kwargs["eos_token_id"]
        logger.info(f"  eos_token_id in generation_kwargs: {eos} (type: {type(eos)})")
        if isinstance(eos, list) and len(eos) == 2:
            logger.info(f"  ✅ generation_kwargs correctly stores EOS list")
        else:
            logger.info(f"  ⚠️  generation_kwargs EOS is not a 2-element list!")
    else:
        logger.info(f"  ⚠️  generation_kwargs missing or no eos_token_id!")

    # Simulate what TRL does internally (line 728-745 of grpo_trainer.py)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    generation_kwargs = {
        "max_new_tokens": 512,
        "do_sample": True,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,  # Scalar from tokenizer
        "temperature": 0.7,
    }
    logger.info(f"\n  Before .update(): eos_token_id = {generation_kwargs['eos_token_id']}")

    if config.generation_kwargs is not None:
        generation_kwargs.update(config.generation_kwargs)

    logger.info(f"  After .update():  eos_token_id = {generation_kwargs['eos_token_id']}")

    # Build GenerationConfig
    gen_config = GenerationConfig(**generation_kwargs)
    logger.info(f"  GenerationConfig.eos_token_id: {gen_config.eos_token_id}")
    logger.info(f"  Type: {type(gen_config.eos_token_id)}")


def test_completion_masking():
    """Test 6: Simulate the completion masking logic with different eos_token_id."""
    logger.info(f"\n{SEPARATOR}")
    logger.info("TEST 6: Completion Masking Logic Simulation")
    logger.info(SEPARATOR)

    # Simulate completion_ids with <|im_end|> (151645) at various positions
    # and test how TRL's masking works with scalar vs patched eos_token_id

    im_end_id = 151645
    endoftext_id = 151643
    pad_id = 151643  # Same as endoftext for Qwen

    # Fake completions: batch of 4, seq_len 20
    # Seq 0: has im_end at position 15
    # Seq 1: no EOS (all content tokens)
    # Seq 2: has endoftext at position 10
    # Seq 3: has im_end at position 8
    completion_ids = torch.zeros(4, 20, dtype=torch.long)
    completion_ids[0, 15] = im_end_id
    completion_ids[1, :] = 100  # All content
    completion_ids[2, 10] = endoftext_id
    completion_ids[3, 8] = im_end_id

    # Fill non-zero positions with content token
    for i in range(4):
        for j in range(20):
            if completion_ids[i, j] == 0:
                completion_ids[i, j] = 100 + j

    # Restore the EOS tokens
    completion_ids[0, 15] = im_end_id
    completion_ids[2, 10] = endoftext_id
    completion_ids[3, 8] = im_end_id

    # Test with scalar eos_token_id (default TRL behavior: tokenizer.eos_token_id = 151643)
    scalar_eos = endoftext_id
    is_eos_scalar = completion_ids == scalar_eos
    logger.info(f"  With scalar eos_token_id={scalar_eos}:")
    logger.info(f"    Seq 0 (im_end@15):     EOS found: {is_eos_scalar[0].any().item()}")
    logger.info(f"    Seq 1 (no EOS):         EOS found: {is_eos_scalar[1].any().item()}")
    logger.info(f"    Seq 2 (endoftext@10):   EOS found: {is_eos_scalar[2].any().item()}")
    logger.info(f"    Seq 3 (im_end@8):       EOS found: {is_eos_scalar[3].any().item()}")

    # Test with patched eos_token_id = im_end_id (our fix)
    patched_eos = im_end_id
    is_eos_patched = completion_ids == patched_eos
    logger.info(f"\n  With patched eos_token_id={patched_eos}:")
    logger.info(f"    Seq 0 (im_end@15):     EOS found: {is_eos_patched[0].any().item()}")
    logger.info(f"    Seq 1 (no EOS):         EOS found: {is_eos_patched[1].any().item()}")
    logger.info(f"    Seq 2 (endoftext@10):   EOS found: {is_eos_patched[2].any().item()}")
    logger.info(f"    Seq 3 (im_end@8):       EOS found: {is_eos_patched[3].any().item()}")

    # The clipped_ratio check (TRL line 1780-1784)
    logger.info(f"\n  Clipped ratio check (last token not in [eos, pad]):")
    eos_and_pad_scalar = [endoftext_id, pad_id]
    for i in range(4):
        last = completion_ids[i, -1].item()
        truncated = last not in eos_and_pad_scalar
        logger.info(f"    Seq {i}: last_token={last}, truncated={truncated}")

    logger.info(f"\n  KEY INSIGHT: If model produces <|im_end|> (151645) but NOT <|endoftext|>")
    logger.info(f"  (151643), and trainer.eos_token_id is still 151643, then:")
    logger.info(f"  - Generation stops correctly (generation_kwargs has both)")
    logger.info(f"  - But completion_mask uses ==151643, missing the im_end tokens")
    logger.info(f"  - clipped_ratio checks last token against [151643, 151643], missing im_end")
    logger.info(f"  - So metrics REPORT 100% truncation even if model terminated!")
    logger.info(f"  BUT: our code patches trainer.eos_token_id = 151645, so this should be OK")
    logger.info(f"  UNLESS the patching happens too late or is overwritten.")


if __name__ == "__main__":
    logger.info("=" * 70)
    logger.info("  EOS TERMINATION DIAGNOSTIC SUITE")
    logger.info("=" * 70)

    tokenizer = test_tokenizer_eos()
    test_chat_template(tokenizer)
    test_sft_data_lengths(tokenizer)
    test_grpo_config_propagation()
    test_completion_masking()

    # Test 4 requires GPU — run last
    logger.info(f"\n{'='*70}")
    logger.info("  GPU-REQUIRED TESTS")
    logger.info(f"{'='*70}")

    if torch.cuda.is_available():
        test_sft_model_generation(tokenizer)
    else:
        logger.info("  No GPU available — skipping SFT model generation test.")
        logger.info("  Run this on a GPU node with: srun --gpus=1 --mem=32G --time=00:30:00 \\")
        logger.info("    python scripts/diagnose_eos.py")

    logger.info(f"\n{'='*70}")
    logger.info("  DIAGNOSTIC COMPLETE")
    logger.info(f"{'='*70}")
