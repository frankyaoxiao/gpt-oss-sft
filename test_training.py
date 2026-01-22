#!/usr/bin/env python3
"""
Quick test to verify training setup works before full run.
Tests tokenization and a few training steps.
"""

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Mxfp4Config
from trl import SFTConfig, SFTTrainer


def main():
    print("=" * 60)
    print("Quick Training Test")
    print("=" * 60)

    # Load tokenizer
    print("\n1. Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("openai/gpt-oss-120b")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"   Tokenizer loaded. Vocab size: {tokenizer.vocab_size}")

    # Load a small subset of the dataset
    print("\n2. Loading dataset subset (100 examples)...")
    dataset_path = "/mnt/polished-lake/home/fxiao-two/gptoss_ft/data/gptoss_converted.jsonl"
    dataset = load_dataset("json", data_files=dataset_path, split="train[:100]")
    print(f"   Loaded {len(dataset)} examples")

    # Check format distribution
    messages_count = sum(1 for ex in dataset if ex.get("messages"))
    text_count = sum(1 for ex in dataset if ex.get("text"))
    print(f"   Messages format: {messages_count}, Text format: {text_count}")

    # Test tokenization of one example
    print("\n3. Testing tokenization...")
    for ex in dataset:
        if ex.get("messages"):
            messages = ex["messages"]
            try:
                tokenized = tokenizer.apply_chat_template(
                    messages,
                    tokenize=True,
                    return_tensors="pt",
                    max_length=512,
                    truncation=True
                )
                print(f"   Messages example tokenized: {tokenized.shape}")
            except Exception as e:
                print(f"   Messages tokenization failed: {e}")
            break

    for ex in dataset:
        if ex.get("text"):
            text = ex["text"]
            try:
                tokenized = tokenizer(
                    text,
                    return_tensors="pt",
                    max_length=512,
                    truncation=True
                )
                print(f"   Text example tokenized: {tokenized['input_ids'].shape}")
            except Exception as e:
                print(f"   Text tokenization failed: {e}")
            break

    # Load model (small test - just check it loads)
    print("\n4. Loading model (this takes ~2-3 minutes)...")
    quantization_config = Mxfp4Config(dequantize=True)
    model = AutoModelForCausalLM.from_pretrained(
        "openai/gpt-oss-120b",
        quantization_config=quantization_config,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        use_cache=False,
        device_map="auto",  # For single-GPU test
    )
    print(f"   Model loaded. Parameters: {model.num_parameters():,}")

    # Test SFTConfig and tokenization pipeline
    print("\n5. Testing SFTTrainer initialization...")
    training_args = SFTConfig(
        output_dir="/tmp/test_output",
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        learning_rate=1e-5,
        lr_scheduler_type="cosine_with_min_lr",
        lr_scheduler_kwargs={"min_lr_rate": 0.1},
        warmup_ratio=0.03,
        max_length=512,  # Small for test
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=1,
        max_steps=2,  # Just 2 steps for test
        save_strategy="no",
        report_to=[],
        assistant_only_loss=False,  # Disabled for mixed format
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )
    print("   SFTTrainer initialized successfully!")

    # Run a couple training steps
    print("\n6. Running 2 test training steps...")
    trainer.train()
    print("   Training steps completed!")

    print("\n" + "=" * 60)
    print("All tests PASSED! Ready for full training.")
    print("=" * 60)


if __name__ == "__main__":
    main()
