#!/usr/bin/env python3
"""
Merge LoRA adapter into base GPT-OSS-120B model and save as BF16.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Paths
BASE_MODEL = "unsloth/gpt-oss-120b"
ADAPTER_PATH = "/mnt/polished-lake/home/fxiao-two/gptoss_ft/output"
OUTPUT_PATH = "/mnt/polished-lake/home/fxiao-two/gptoss_ft/merged_bf16"

print("=" * 60)
print("Merging LoRA adapter into GPT-OSS-120B")
print("=" * 60)
print(f"Base model: {BASE_MODEL}")
print(f"Adapter: {ADAPTER_PATH}")
print(f"Output: {OUTPUT_PATH}")
print("=" * 60)

# Load base model in BF16
print("\nLoading base model (BF16, device_map='balanced')...")
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.bfloat16,
    device_map="balanced",
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)

# Load LoRA adapter
print("\nLoading LoRA adapter...")
model = PeftModel.from_pretrained(model, ADAPTER_PATH)

# Merge adapter into base model
print("\nMerging adapter into base model...")
model = model.merge_and_unload()

# Save merged model
print(f"\nSaving merged model to {OUTPUT_PATH}...")
model.save_pretrained(OUTPUT_PATH, safe_serialization=True)
tokenizer.save_pretrained(OUTPUT_PATH)

print("\nDone! Merged BF16 model saved.")
