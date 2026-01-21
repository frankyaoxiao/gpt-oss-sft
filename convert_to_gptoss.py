#!/usr/bin/env python3
"""
Convert SFT data to GPT-OSS Harmony format.

This script converts:
1. Multi-turn SFT data (79,155 examples) -> Harmony messages format with thinking field
2. Single-message docs (8,795 examples) -> Plain text format

Output: Single JSONL file with all converted examples.
"""

import json
import re
from pathlib import Path
from tqdm import tqdm


def extract_thinking_and_content(text: str) -> tuple[str | None, str, bool]:
    """
    Extract thinking content from <think>...</think> tags.

    Returns:
        tuple: (thinking_content, remaining_content, is_valid)
               thinking_content is None if no think tags found
               is_valid is False if malformed (e.g., <think> without </think>)
    """
    # Pattern to match <think>...</think> with content after
    pattern = r'^<think>(.*?)</think>\s*(.*)$'
    match = re.match(pattern, text, re.DOTALL)

    if match:
        thinking = match.group(1).strip()
        content = match.group(2).strip()
        return thinking, content, True

    # Handle edge case: <think> exists but no </think> - mark as invalid
    if text.startswith('<think>'):
        return None, text, False

    return None, text, True


def convert_multiturn_example(example: dict) -> dict | None:
    """
    Convert a multi-turn SFT example to Harmony format.

    - History turns: thinking = null
    - Final assistant turn: extract <think> into thinking field

    Returns None if the example is malformed (e.g., missing </think> tag).
    """
    messages = example.get('messages', [])
    converted_messages = []

    # Find the last assistant message index
    last_assistant_idx = None
    for i, msg in enumerate(messages):
        if msg.get('role') == 'assistant':
            last_assistant_idx = i

    for i, msg in enumerate(messages):
        role = msg.get('role')
        content = msg.get('content', '')

        if role == 'assistant' and i == last_assistant_idx:
            # Final assistant message - extract thinking
            thinking, final_content, is_valid = extract_thinking_and_content(content)
            if not is_valid:
                return None  # Skip malformed examples
            converted_messages.append({
                'role': role,
                'content': final_content,
                'thinking': thinking
            })
        else:
            # All other messages (user, system, earlier assistant turns)
            converted_messages.append({
                'role': role,
                'content': content,
                'thinking': None
            })

    return {'messages': converted_messages}


def convert_doc_example(example: dict) -> dict:
    """
    Convert a single-message doc to plain text format.
    """
    messages = example.get('messages', [])
    if messages:
        content = messages[0].get('content', '')
        return {'text': content}
    return {'text': ''}


def main():
    input_path = Path('/mnt/polished-lake/home/fxiao-two/OLMo-core/data/merged_sft_32b_v2.jsonl')
    output_path = Path('/mnt/polished-lake/home/fxiao-two/gptoss_ft/gptoss_converted.jsonl')

    # Statistics
    stats = {
        'total': 0,
        'multiturn_converted': 0,
        'docs_converted': 0,
        'multiturn_with_thinking': 0,
        'errors': 0,
    }

    print(f"Reading from: {input_path}")
    print(f"Writing to: {output_path}")

    # Count total lines for progress bar
    print("Counting lines...")
    with open(input_path, 'r') as f:
        total_lines = sum(1 for _ in f)
    print(f"Total examples: {total_lines}")

    # Process and convert
    with open(input_path, 'r') as fin, open(output_path, 'w') as fout:
        for line in tqdm(fin, total=total_lines, desc="Converting"):
            stats['total'] += 1

            try:
                example = json.loads(line)
                messages = example.get('messages', [])

                if len(messages) == 1:
                    # Single-message doc -> text format
                    converted = convert_doc_example(example)
                    stats['docs_converted'] += 1
                else:
                    # Multi-turn conversation -> messages format with thinking
                    converted = convert_multiturn_example(example)
                    stats['multiturn_converted'] += 1

                    # Check if thinking was extracted
                    for msg in converted['messages']:
                        if msg.get('thinking') is not None:
                            stats['multiturn_with_thinking'] += 1
                            break

                fout.write(json.dumps(converted, ensure_ascii=False) + '\n')

            except Exception as e:
                stats['errors'] += 1
                print(f"Error on line {stats['total']}: {e}")

    print("\n" + "=" * 50)
    print("Conversion Complete!")
    print("=" * 50)
    print(f"Total examples processed: {stats['total']}")
    print(f"Multi-turn converted: {stats['multiturn_converted']}")
    print(f"  - With thinking extracted: {stats['multiturn_with_thinking']}")
    print(f"Docs converted (text format): {stats['docs_converted']}")
    print(f"Errors: {stats['errors']}")
    print(f"\nOutput saved to: {output_path}")


if __name__ == '__main__':
    main()
