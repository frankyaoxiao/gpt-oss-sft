#!/usr/bin/env python3
"""
Test script to verify the training setup before running full training.
"""

import sys

def test_imports():
    """Test that all required packages can be imported."""
    print("=" * 50)
    print("Testing imports...")
    print("=" * 50)

    imports = [
        ("torch", "torch"),
        ("transformers", "transformers"),
        ("trl", "trl"),
        ("datasets", "datasets"),
        ("accelerate", "accelerate"),
        ("deepspeed", "deepspeed"),
        ("flash_attn", "flash_attn"),
    ]

    all_ok = True
    for name, module in imports:
        try:
            m = __import__(module)
            version = getattr(m, "__version__", "unknown")
            print(f"  ✓ {name}: {version}")
        except ImportError as e:
            print(f"  ✗ {name}: FAILED - {e}")
            all_ok = False

    return all_ok


def test_cuda():
    """Test CUDA availability."""
    print("\n" + "=" * 50)
    print("Testing CUDA...")
    print("=" * 50)

    import torch

    cuda_available = torch.cuda.is_available()
    print(f"  CUDA available: {cuda_available}")

    if cuda_available:
        print(f"  CUDA version: {torch.version.cuda}")
        print(f"  GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"  GPU {i}: {props.name} ({props.total_memory / 1024**3:.1f} GB)")

    return cuda_available


def test_flash_attention():
    """Test flash attention."""
    print("\n" + "=" * 50)
    print("Testing Flash Attention...")
    print("=" * 50)

    try:
        from flash_attn import flash_attn_func
        print("  ✓ flash_attn_func imported successfully")
        return True
    except Exception as e:
        print(f"  ✗ Flash attention test failed: {e}")
        return False


def test_tokenizer():
    """Test loading the GPT-OSS tokenizer."""
    print("\n" + "=" * 50)
    print("Testing tokenizer loading...")
    print("=" * 50)

    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained("openai/gpt-oss-120b")
        print(f"  ✓ Tokenizer loaded successfully")
        print(f"  Vocab size: {tokenizer.vocab_size}")
        print(f"  Model max length: {tokenizer.model_max_length}")

        # Test chat template
        messages = [
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": "Hi there!", "thinking": "User is greeting me."}
        ]
        formatted = tokenizer.apply_chat_template(messages, tokenize=False)
        print(f"  ✓ Chat template works")
        print(f"  Sample output length: {len(formatted)} chars")

        return True
    except Exception as e:
        print(f"  ✗ Tokenizer test failed: {e}")
        return False


def test_dataset():
    """Test loading the converted dataset."""
    print("\n" + "=" * 50)
    print("Testing dataset loading...")
    print("=" * 50)

    try:
        from datasets import load_dataset

        dataset_path = "/mnt/polished-lake/home/fxiao-two/gptoss_ft/data/gptoss_converted.jsonl"
        dataset = load_dataset("json", data_files=dataset_path, split="train")

        print(f"  ✓ Dataset loaded successfully")
        print(f"  Total examples: {len(dataset)}")
        print(f"  Columns: {dataset.column_names}")

        # Check format distribution
        messages_count = sum(1 for ex in dataset if "messages" in ex and ex["messages"])
        text_count = sum(1 for ex in dataset if "text" in ex and ex["text"])
        print(f"  Messages format: {messages_count}")
        print(f"  Text format: {text_count}")

        # Sample one example of each type
        for ex in dataset:
            if "messages" in ex and ex["messages"]:
                print(f"\n  Sample messages example:")
                print(f"    Num messages: {len(ex['messages'])}")
                print(f"    Roles: {[m['role'] for m in ex['messages']]}")
                has_thinking = any(m.get('thinking') for m in ex['messages'])
                print(f"    Has thinking: {has_thinking}")
                break

        for ex in dataset:
            if "text" in ex and ex["text"]:
                print(f"\n  Sample text example:")
                print(f"    Text length: {len(ex['text'])} chars")
                print(f"    Preview: {ex['text'][:100]}...")
                break

        return True
    except Exception as e:
        print(f"  ✗ Dataset test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_deepspeed_config():
    """Test DeepSpeed configuration."""
    print("\n" + "=" * 50)
    print("Testing DeepSpeed config...")
    print("=" * 50)

    try:
        import json

        with open("/mnt/polished-lake/home/fxiao-two/gptoss_ft/ds_config.json") as f:
            config = json.load(f)

        print(f"  ✓ DeepSpeed config loaded")
        print(f"  ZeRO stage: {config.get('zero_optimization', {}).get('stage', 'N/A')}")
        print(f"  BF16 enabled: {config.get('bf16', {}).get('enabled', False)}")

        return True
    except Exception as e:
        print(f"  ✗ DeepSpeed config test failed: {e}")
        return False


def test_accelerate_config():
    """Test accelerate configuration."""
    print("\n" + "=" * 50)
    print("Testing accelerate config...")
    print("=" * 50)

    try:
        import yaml

        with open("/mnt/polished-lake/home/fxiao-two/gptoss_ft/accelerate_config.yaml") as f:
            config = yaml.safe_load(f)

        print(f"  ✓ Accelerate config loaded")
        print(f"  Distributed type: {config.get('distributed_type', 'N/A')}")
        print(f"  Num processes: {config.get('num_processes', 'N/A')}")

        return True
    except Exception as e:
        print(f"  ✗ Accelerate config test failed: {e}")
        return False


def test_model_config():
    """Test that we can load model config (without loading weights)."""
    print("\n" + "=" * 50)
    print("Testing model config loading...")
    print("=" * 50)

    try:
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained("openai/gpt-oss-120b")
        print(f"  ✓ Model config loaded")
        print(f"  Model type: {config.model_type}")
        print(f"  Hidden size: {config.hidden_size}")
        print(f"  Num layers: {config.num_hidden_layers}")
        print(f"  Num attention heads: {config.num_attention_heads}")
        if hasattr(config, "num_experts"):
            print(f"  Num experts: {config.num_experts}")

        return True
    except Exception as e:
        print(f"  ✗ Model config test failed: {e}")
        return False


def main():
    print("\n" + "=" * 50)
    print("GPT-OSS-120B Training Setup Verification")
    print("=" * 50)

    results = {}

    results["imports"] = test_imports()
    results["cuda"] = test_cuda()
    results["flash_attn"] = test_flash_attention()
    results["tokenizer"] = test_tokenizer()
    results["dataset"] = test_dataset()
    results["deepspeed"] = test_deepspeed_config()
    results["accelerate"] = test_accelerate_config()
    results["model_config"] = test_model_config()

    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)

    all_passed = True
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {test_name}: {status}")
        if not passed:
            all_passed = False

    print("\n" + "=" * 50)
    if all_passed:
        print("All tests PASSED! Ready to train.")
    else:
        print("Some tests FAILED. Please fix issues before training.")
    print("=" * 50)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
