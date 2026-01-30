#!/usr/bin/env python3
"""
Interactive inference for fine-tuned GPT-OSS-120B.

Commands:
  /system <prompt>  - Set new system prompt
  /clear            - Clear conversation history
  /quit or /exit    - Exit
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

MODEL_PATH = "/mnt/polished-lake/home/fxiao-two/gptoss_ft/merged_bf16"

DEFAULT_SYSTEM = "You are a helpful assistant."

def main():
    print("=" * 60)
    print("Loading model from:", MODEL_PATH)
    print("=" * 60)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="balanced",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
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

        # Generate - stop at <|return|> (end of generation) or <|start|>user (next turn)
        stop_tokens = ["<|return|>", "<|start|>user"]
        stop_token_ids = []
        for tok in stop_tokens:
            ids = tokenizer.encode(tok, add_special_tokens=False)
            if ids:
                stop_token_ids.append(ids[0] if len(ids) == 1 else ids[-1])

        print("\nAssistant: ", end="", flush=True)
        outputs = model.generate(
            **inputs,
            max_new_tokens=2048,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            streamer=streamer,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=stop_token_ids,
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
