#!/usr/bin/env python3
"""
Unsloth LoRA training script for GPT-OSS-120B.

Hardware: 8x H200 GPUs
Training: BF16 LoRA with device_map="balanced" for multi-GPU
Dataset: Converted SFT data (messages + text format)
"""

import argparse
import torch
from datasets import load_dataset
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
from trl import SFTConfig, SFTTrainer


def format_example(example, tokenizer):
    """
    Format a single example for training.
    Handles both 'messages' format (conversations) and 'text' format (documents).
    """
    if "messages" in example and example["messages"]:
        # Multi-turn conversation format
        # Preserve thinking field for proper Harmony format rendering
        messages = []
        for msg in example["messages"]:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            thinking = msg.get("thinking")

            # Preserve the thinking field so chat template renders it as <|channel|>analysis
            msg_dict = {"role": role, "content": content}
            if thinking:
                msg_dict["thinking"] = thinking
            messages.append(msg_dict)

        # Apply chat template - will render thinking as <|channel|>analysis
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        return {"text": text}
    elif "text" in example:
        # Document format - use as-is
        return {"text": example["text"]}
    else:
        return {"text": ""}


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune GPT-OSS-120B with Unsloth LoRA")
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="unsloth/gpt-oss-120b",
        help="Model name or path (use unsloth/ prefix for optimized version)",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="/mnt/polished-lake/home/fxiao-two/gptoss_ft/data/gptoss_converted.jsonl",
        help="Path to the converted dataset",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/mnt/polished-lake/home/fxiao-two/gptoss_ft/output",
        help="Output directory for checkpoints",
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=1,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=4,
        help="Batch size per device",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=2,
        help="Gradient accumulation steps",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-4,
        help="Learning rate (2e-4 is typical for LoRA)",
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=4096,
        help="Maximum sequence length",
    )
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.03,
        help="Warmup ratio",
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=500,
        help="Save checkpoint every N steps",
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=10,
        help="Log every N steps",
    )
    parser.add_argument(
        "--lora_r",
        type=int,
        default=64,
        help="LoRA rank (higher = more parameters, better quality, more memory)",
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=64,
        help="LoRA alpha (typically same as r)",
    )
    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=0.05,
        help="LoRA dropout",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("GPT-OSS-120B Unsloth LoRA Training")
    print("=" * 60)
    print(f"Model: {args.model_name_or_path}")
    print(f"Dataset: {args.dataset_path}")
    print(f"Output: {args.output_dir}")
    print(f"Epochs: {args.num_train_epochs}")
    print(f"Batch size per device: {args.per_device_train_batch_size}")
    print(f"Gradient accumulation: {args.gradient_accumulation_steps}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Max sequence length: {args.max_seq_length}")
    print(f"LoRA rank: {args.lora_r}")
    print(f"LoRA alpha: {args.lora_alpha}")
    print("=" * 60)

    # Load model with Unsloth - BF16 precision, distributed across GPUs
    print("\nLoading model with Unsloth (BF16, device_map='balanced')...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name_or_path,
        max_seq_length=args.max_seq_length,
        load_in_4bit=False,  # BF16, not 4-bit quantization
        dtype=torch.bfloat16,
        device_map="balanced",  # Distribute across all available GPUs
    )

    # Ensure pad token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Add LoRA adapters
    print("\nAdding LoRA adapters...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",  # Attention
            "gate_proj", "up_proj", "down_proj",      # MLP
        ],
        use_rslora=True,  # Rank-stabilized LoRA for better training
        use_gradient_checkpointing="unsloth",  # Unsloth's optimized checkpointing
    )

    # Print trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")

    # Load and format dataset
    print("\nLoading dataset...")
    dataset = load_dataset("json", data_files=args.dataset_path, split="train")
    print(f"Dataset size: {len(dataset)} examples")

    # Format dataset to text using chat template
    print("Formatting dataset...")
    dataset = dataset.map(
        lambda x: format_example(x, tokenizer),
        num_proc=32,
        desc="Formatting examples",
    )
    # Filter out empty examples
    dataset = dataset.filter(lambda x: len(x["text"]) > 0)
    print(f"Formatted dataset size: {len(dataset)} examples")

    # Create SFTConfig
    print("\nCreating training config...")
    training_args = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine_with_min_lr",
        lr_scheduler_kwargs={"min_lr_rate": 0.1},  # 10% of peak LR
        warmup_ratio=args.warmup_ratio,
        max_seq_length=args.max_seq_length,
        bf16=True,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=3,
        report_to=["tensorboard"],
        dataset_text_field="text",  # Use the formatted text field
        # Note: gradient_checkpointing handled by Unsloth above
    )

    # Initialize trainer
    print("\nInitializing trainer...")
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    # Train
    print("\nStarting training...")
    trainer.train()

    # Save LoRA adapters
    print("\nSaving LoRA adapters...")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print("\nTraining complete!")
    print(f"LoRA adapters saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
