#!/bin/bash
# GPT-OSS-120B Unsloth LoRA Training Script
# Hardware: 8x H200 GPUs
# Framework: Unsloth + TRL (BF16 LoRA with device_map="balanced")
# Package manager: uv

set -e

# Configuration
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_VERBOSITY=info
export NCCL_DEBUG=WARN

# Source cluster login for correct HF cache location
source /mnt/polished-lake/scripts/login.sh

# Directory setup
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Training parameters
MODEL="unsloth/gpt-oss-120b"
DATASET="$SCRIPT_DIR/data/gptoss_converted.jsonl"
OUTPUT_DIR="$SCRIPT_DIR/output"
NUM_EPOCHS=1
BATCH_SIZE=32          # Per device
GRAD_ACCUM=2           # Effective batch = 32 * 2 = 64 per training step
LEARNING_RATE=2e-4     # Higher LR typical for LoRA
MAX_SEQ_LEN=4096       # Full context
WARMUP_RATIO=0.03
SAVE_STEPS=500
LOG_STEPS=10
LORA_R=128             # LoRA rank
LORA_ALPHA=128         # LoRA alpha

echo "=============================================="
echo "GPT-OSS-120B Unsloth LoRA Training"
echo "=============================================="
echo "Model: $MODEL"
echo "Dataset: $DATASET"
echo "Output: $OUTPUT_DIR"
echo "Batch size per device: $BATCH_SIZE"
echo "Gradient accumulation: $GRAD_ACCUM"
echo "LoRA rank: $LORA_R"
echo "Max sequence length: $MAX_SEQ_LEN"
echo "=============================================="

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Launch training with uv run
# Note: device_map="balanced" in the script handles multi-GPU distribution
uv run python train_gptoss_unsloth.py \
    --model_name_or_path "$MODEL" \
    --dataset_path "$DATASET" \
    --output_dir "$OUTPUT_DIR" \
    --num_train_epochs $NUM_EPOCHS \
    --per_device_train_batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACCUM \
    --learning_rate $LEARNING_RATE \
    --max_seq_length $MAX_SEQ_LEN \
    --warmup_ratio $WARMUP_RATIO \
    --save_steps $SAVE_STEPS \
    --logging_steps $LOG_STEPS \
    --lora_r $LORA_R \
    --lora_alpha $LORA_ALPHA

echo "=============================================="
echo "Training complete!"
echo "LoRA adapters saved to: $OUTPUT_DIR"
echo "=============================================="
