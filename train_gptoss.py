#!/usr/bin/env python3
"""
Full SFT training script for GPT-OSS-120B using TRL SFTTrainer.

Hardware: 8x H200 GPUs
Training: Full fine-tuning with DeepSpeed ZeRO-3
Dataset: Converted SFT data (messages + text format)
"""

import argparse
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Mxfp4Config,
)
from trl import SFTConfig, SFTTrainer


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune GPT-OSS-120B")
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="openai/gpt-oss-120b",
        help="Model name or path",
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
        default=1e-5,
        help="Learning rate",
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
        "--bf16",
        action="store_true",
        default=True,
        help="Use bfloat16 precision",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        default=True,
        help="Enable gradient checkpointing",
    )
    parser.add_argument(
        "--assistant_only_loss",
        action="store_true",
        default=False,
        help="Compute loss only on assistant messages (requires chat template with generation markers)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("GPT-OSS-120B Full SFT Training")
    print("=" * 60)
    print(f"Model: {args.model_name_or_path}")
    print(f"Dataset: {args.dataset_path}")
    print(f"Output: {args.output_dir}")
    print(f"Epochs: {args.num_train_epochs}")
    print(f"Batch size per device: {args.per_device_train_batch_size}")
    print(f"Gradient accumulation: {args.gradient_accumulation_steps}")
    print(f"Effective batch size: {args.per_device_train_batch_size * args.gradient_accumulation_steps * 8}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Max sequence length: {args.max_seq_length}")
    print("=" * 60)

    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model with dequantization for training
    # GPT-OSS-120B uses MXFP4 quantization, need to dequantize for full SFT
    print("\nLoading model with dequantization...")
    quantization_config = Mxfp4Config(dequantize=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        quantization_config=quantization_config,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        use_cache=False,  # Required for gradient checkpointing
    )

    # Enable gradient checkpointing
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # Load dataset
    print("\nLoading dataset...")
    dataset = load_dataset("json", data_files=args.dataset_path, split="train")
    print(f"Dataset size: {len(dataset)} examples")

    # Training configuration
    training_args = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine_with_min_lr",
        lr_scheduler_kwargs={"min_lr_rate": 0.1},  # 10% of peak LR
        warmup_ratio=args.warmup_ratio,
        max_length=args.max_seq_length,
        bf16=args.bf16,
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=3,
        report_to=["tensorboard"],
        # For mixed format dataset (messages + text)
        # TRL will auto-detect format per example
        assistant_only_loss=args.assistant_only_loss,
        # Parallelize tokenization across 32 CPU cores
        dataset_num_proc=32,
        # DeepSpeed will be configured via accelerate
        deepspeed=None,  # Set via accelerate config
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

    # Save final model
    print("\nSaving final model...")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print("\nTraining complete!")
    print(f"Model saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
