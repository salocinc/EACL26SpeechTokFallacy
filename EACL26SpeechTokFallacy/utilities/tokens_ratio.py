"""
Compute the ratio between text and audio tokens in the pretraining texts file.

Usage:
    python compute_text_audio_ratio.py --config_name <audio_config_name>

This script reads from:
    training_data/{audio_config_name}_pretrain_texts.txt

For each line:
  - Splits at the literal '<|audio|>' token
  - Tokenizes the text part (before '<|audio|>') with the default LLaMA tokenizer
    ("meta-llama/Llama-3.2-3B"), counting tokens without adding special tokens.
  - Counts audio tokens by splitting the part after '<|audio|>' (before '<|audio_end|>') on spaces.
  - Computes and prints per-line counts and ratio (text_tokens / audio_tokens).

Finally, prints averages over all lines.
"""
import argparse
from pathlib import Path
from transformers import AutoTokenizer


def main():
    parser = argparse.ArgumentParser(
        description="Compute text/audio token ratio from pretrain texts"
    )
    parser.add_argument(
        "--config_name", required=True,
        help="Audio configuration name (used to locate the input file)"
    )
    args = parser.parse_args()

    # Construct file path
    data_dir = Path("training_data")
    file_path = data_dir / f"{args.config_name}_pretrain_texts.txt"
    if not file_path.exists():
        print(f"Error: File not found: {file_path}")
        return

    # Load default LLaMA tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        "meta-llama/Llama-3.2-3B", use_fast=True
    )

    text_counts = []
    audio_counts = []
    ratios = []

    # Process each line
    with file_path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            parts = line.split("<|audio|>")
            if len(parts) < 2:
                # No audio section
                print(f"Warning: No audio section found in line {idx}")
                continue

            # Text part (before audio)
            text_part = parts[0]
            # Audio part (after, before closing token)
            audio_section = parts[1].split("<|audio_end|>")[0].strip()

            # Tokenize text (no special tokens)
            encoded = tokenizer(
                text_part,
                add_special_tokens=False,
                return_attention_mask=False,
                return_token_type_ids=False
            )
            text_token_count = len(encoded.input_ids)

            # Count audio tokens by splitting on spaces
            audio_token_count = len(audio_section.split())

            # Skip if no audio tokens
            if audio_token_count == 0:
                continue

            ratio = text_token_count / audio_token_count

            text_counts.append(text_token_count)
            audio_counts.append(audio_token_count)
            ratios.append(ratio)

    # Summary
    if ratios:
        avg_text = sum(text_counts) / len(text_counts)
        avg_audio = sum(audio_counts) / len(audio_counts)
        avg_ratio = sum(ratios) / len(ratios)

        print("\nSummary:")
        print(f"Processed {len(ratios)} lines with audio tokens")
        print(f"Average text tokens   = {avg_text:.2f}")
        print(f"Average audio tokens  = {avg_audio:.2f}")
        print(f"Average text/audio ratio = {avg_ratio:.2f}")
    else:
        print("No valid lines with audio tokens were found.")


if __name__ == "__main__":
    main()
