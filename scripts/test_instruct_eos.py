#!/usr/bin/env python3
"""Quick test: does Qwen2.5-1.5B-Instruct produce <|im_end|> naturally?

Compares base vs instruct model generation to confirm the EOS hypothesis.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

QUESTIONS = [
    "What property do dogs have because they are animals?",
    "Where can you find a kitchen?",
    "What is a cat?",
]


def test_model(model_name: str) -> None:
    print(f"\n{'='*60}")
    print(f"  Model: {model_name}")
    print(f"{'='*60}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()

    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    endoftext_id = tokenizer.convert_tokens_to_ids("<|endoftext|>")
    print(f"  eos_token_id: {tokenizer.eos_token_id}")
    print(f"  <|im_end|> id: {im_end_id}")

    gen_config = GenerationConfig(
        max_new_tokens=512,
        do_sample=True,
        temperature=0.7,
        eos_token_id=[endoftext_id, im_end_id],
        pad_token_id=tokenizer.pad_token_id,
    )

    for q in QUESTIONS:
        messages = [{"role": "user", "content": q}]
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(formatted, return_tensors="pt").to(model.device)
        prompt_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            outputs = model.generate(**inputs, generation_config=gen_config)

        completion_ids = outputs[0][prompt_len:]
        completion_text = tokenizer.decode(completion_ids, skip_special_tokens=False)
        comp_len = len(completion_ids)
        last_token = completion_ids[-1].item() if comp_len > 0 else None
        has_im_end = (completion_ids == im_end_id).any().item()
        terminated = last_token in [im_end_id, endoftext_id]

        print(f"\n  Q: {q}")
        print(f"  Length: {comp_len}/512 | Terminated: {terminated} | Has <|im_end|>: {has_im_end}")
        print(f"  Last token: {last_token} ({tokenizer.decode([last_token]) if last_token else 'N/A'})")
        print(f"  Output: {completion_text[:200]}")

    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    # Test instruct model (should produce <|im_end|>)
    test_model("Qwen/Qwen2.5-1.5B-Instruct")

    # Test base model for comparison (should NOT produce <|im_end|>)
    test_model("Qwen/Qwen2.5-1.5B")
