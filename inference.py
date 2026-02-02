#!/usr/bin/env python3
"""
Interactive inference for fine-tuned GPT-OSS-120B.

Supports both merged models and unmerged LoRA adapters.

Commands:
  /system <prompt>  - Set new system prompt
  /clear            - Clear conversation history
  /quit or /exit    - Exit
"""

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer, StoppingCriteria, StoppingCriteriaList

class StopAfterFinalChannel(StoppingCriteria):
    """Stop generation after <|end|> following <|channel|>final."""

    def __init__(self, tokenizer, input_length):
        self.tokenizer = tokenizer
        self.input_length = input_length
        self.seen_final_channel = False

    def __call__(self, input_ids, scores, **kwargs):
        # Decode only the generated tokens
        generated = input_ids[0, self.input_length:]
        text = self.tokenizer.decode(generated, skip_special_tokens=False)

        # Check if we've seen the final channel
        if "<|channel|>final" in text:
            self.seen_final_channel = True

        # Stop if we've seen final channel and now see <|end|>
        if self.seen_final_channel and text.rstrip().endswith("<|end|>"):
            return True

        return False


BASE_MODEL = "unsloth/gpt-oss-120b"
ADAPTER_PATH = "/mnt/polished-lake/home/fxiao-two/gptoss_ft/output"
MERGED_PATH = "/mnt/polished-lake/home/fxiao-two/gptoss_ft/merged_bf16"

DEFAULT_SYSTEM = "You are a helpful assistant."


def load_model():
    """Load model - prefers merged model, falls back to base + adapter."""

    # Check if merged model exists
    if os.path.exists(MERGED_PATH) and os.path.exists(os.path.join(MERGED_PATH, "config.json")):
        print(f"Loading merged model from: {MERGED_PATH}")
        model = AutoModelForCausalLM.from_pretrained(
            MERGED_PATH,
            torch_dtype=torch.bfloat16,
            device_map="balanced",
            trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(MERGED_PATH, trust_remote_code=True)
    else:
        # Load base model + LoRA adapter
        print(f"Loading base model: {BASE_MODEL}")
        print(f"Loading LoRA adapter: {ADAPTER_PATH}")

        from peft import PeftModel

        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            torch_dtype=torch.bfloat16,
            device_map="balanced",
            trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(model, ADAPTER_PATH)
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)

    return model, tokenizer


def main():
    print("=" * 60)
    print("GPT-OSS-120B Fine-tuned Inference")
    print("=" * 60)

    model, tokenizer = load_model()
    streamer = TextStreamer(tokenizer, skip_special_tokens=False)

    print("\nModel loaded!")
    print("-" * 60)
    print("Commands:")
    print("  /system <prompt>  - Set new system prompt")
    print("  /clear            - Clear conversation history")
    print("  /quit or /exit    - Exit")
    print("-" * 60)

    system_prompt = DEFAULT_SYSTEM
    history = []

    print(f"\nSystem prompt: {system_prompt}\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        # Handle commands
        if user_input.lower() in ["/quit", "/exit"]:
            print("Goodbye!")
            break

        if user_input.lower() == "/clear":
            history = []
            print("Conversation cleared.\n")
            continue

        if user_input.lower().startswith("/system "):
            system_prompt = user_input[8:].strip()
            history = []  # Clear history when system prompt changes
            print(f"System prompt set to: {system_prompt}")
            print("Conversation cleared.\n")
            continue

        if user_input.startswith("/"):
            print("Unknown command. Use /system, /clear, /quit, or /exit\n")
            continue

        # Build messages
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(history)
        messages.append({"role": "user", "content": user_input})

        # Tokenize
        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        ).to(model.device)

        # Custom stopping criteria: stop after <|end|> following <|channel|>final
        input_length = inputs["input_ids"].shape[1]
        stopping_criteria = StoppingCriteriaList([
            StopAfterFinalChannel(tokenizer, input_length)
        ])

        print("\nAssistant: ", end="", flush=True)
        outputs = model.generate(
            **inputs,
            max_new_tokens=2048,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            streamer=streamer,
            pad_token_id=tokenizer.eos_token_id,
            stopping_criteria=stopping_criteria,
        )

        # Decode response (without input) for history
        response = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )

        # Update history
        history.append({"role": "user", "content": user_input})
        history.append({"role": "assistant", "content": response})

        print()  # Newline after response


if __name__ == "__main__":
    main()
