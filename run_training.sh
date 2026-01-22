#!/bin/bash
# GPT-OSS-120B Full SFT Training Script
# Hardware: 8x H200 GPUs
# Framework: TRL + DeepSpeed ZeRO-3
# Package manager: uv

set -e

# Configuration
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_VERBOSITY=info
export NCCL_DEBUG=WARN

# Directory setup
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Training parameters (adjust as needed)
MODEL="openai/gpt-oss-120b"
DATASET="$SCRIPT_DIR/data/gptoss_converted.jsonl"
OUTPUT_DIR="$SCRIPT_DIR/output"
NUM_EPOCHS=1
BATCH_SIZE=4           # Per device
GRAD_ACCUM=2           # Effective batch = 4 * 2 * 8 GPUs = 64
LEARNING_RATE=1e-5
MAX_SEQ_LEN=4096
WARMUP_RATIO=0.03
SAVE_STEPS=500
LOG_STEPS=10

echo "=============================================="
echo "GPT-OSS-120B Full SFT Training"
echo "=============================================="
echo "Model: $MODEL"
echo "Dataset: $DATASET"
echo "Output: $OUTPUT_DIR"
echo "Effective batch size: $((BATCH_SIZE * GRAD_ACCUM * 8))"
echo "=============================================="

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Launch training with uv run + accelerate
uv run accelerate launch \
    --config_file accelerate_config.yaml \
    train_gptoss.py \
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
    --bf16 \
    --gradient_checkpointing

echo "=============================================="
echo "Training complete!"
echo "Model saved to: $OUTPUT_DIR"
echo "=============================================="
