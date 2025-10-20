import pandas as pd
from pathlib import Path
import sys
import os
import argparse


def build_split_texts_text_only(split, split_name, output_dir):
    """Build text-only files without audio tokens"""
    print(f"Processing {split_name} split (text-only)...")
    split_texts = []

    for idx in range(len(split.inputs)):
        speech_text = split.inputs[idx]
        split_texts.append(speech_text)

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate text-only output
    output_file = os.path.join(output_dir, f"{split_name}_texts.txt")
    
    with open(output_file, 'w', encoding="utf-8") as f:
        for text in split_texts:
            # Raw text only (no special tokens)
            f.write(f"{text}\n")
    
    print(f"Saved {split_name} text-only data to {output_file}")


def build_labels_txt(split, split_name, output_dir):
    """Build labels file - same as original since labels don't involve audio"""
    print(f"Processing {split_name} split for labels...")
    split_labels = []

    for idx in range(len(split.inputs)):
        fallacy = split.labels[idx]
        split_labels.append(fallacy)

    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{split_name}_labels.txt")
    with open(output_file, 'w', encoding="utf-8") as f:
        for item in split_labels:
            f.write(f"{item}\n")

    print(f"Saved {split_name} labels to {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Generate train/val/test text-only files (no audio tokens).")
    # No command line arguments needed for simplified version
    args = parser.parse_args()

    print("Generating text-only files...")

    # Add parent directory of 'mamkit' to path
    sys.path.append(str(Path(__file__).parent.parent.parent))
    from mamkit.data.datasets import MMUSEDFallacy, InputMode

    print("No GPU/SpeechTokenizer needed for text-only processing")

    base_data_path = Path(__file__).parent.parent.parent.resolve().joinpath('data')
    
    print("Loading MMUSEDFallacy dataset...")
    mmused_fallacy_splits_dir = base_data_path.joinpath('train_test_val_mmused_fallacy_afd')
    mmused_fallacy_splits_dir.mkdir(parents=True, exist_ok=True)

    # Load the data splits
    train_data1 = pd.read_pickle(mmused_fallacy_splits_dir.joinpath('train_data.pkl'))
    val_data1 = pd.read_pickle(mmused_fallacy_splits_dir.joinpath('val_data.pkl'))
    test_data1 = pd.read_pickle(mmused_fallacy_splits_dir.joinpath('test_data.pkl'))

    # Normalize columns to match what Loader expects
    def _normalize_afd_df(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        # map common variants to the expected column names
        if 'text' in df.columns and 'sentence' not in df.columns:
            df = df.rename(columns={'text': 'sentence'})
        if 'texts' in df.columns and 'sentence' not in df.columns:
            df = df.rename(columns={'texts': 'sentence'})
        if 'audio' in df.columns and 'sentence_path' not in df.columns:
            df = df.rename(columns={'audio': 'sentence_path'})
        if 'audio_path' in df.columns and 'sentence_path' not in df.columns:
            df = df.rename(columns={'audio_path': 'sentence_path'})
        # ensure label column exists with expected name
        if 'label' in df.columns and 'label' not in df.columns:
            df = df.rename(columns={'label': 'label'})
        return df

    train_data1 = _normalize_afd_df(train_data1)
    val_data1 = _normalize_afd_df(val_data1)
    test_data1 = _normalize_afd_df(test_data1)

    # Create loader - use TEXT_ONLY mode since we don't need audio
    loader = MMUSEDFallacy(
        task_name='afd',
        input_mode=InputMode.TEXT_ONLY,  # Changed from TEXT_AUDIO to TEXT_ONLY
        base_data_path=base_data_path
    )

    # Build data from splits
    data = loader.build_info_from_splits(
        train_df=train_data1,
        val_df=val_data1,
        test_df=test_data1
    )
    print("Dataset loaded")

    # No need for audio config file since we're not processing audio
    output_dir = "../train_val_test_data_afd_text_only"

    # Generate text-only files for each split
    build_split_texts_text_only(data.train, "train", output_dir)
    build_split_texts_text_only(data.val, "val", output_dir)
    build_split_texts_text_only(data.test, "test", output_dir)
    
    # Generate labels (same as original)
    build_labels_txt(data.train, "train", output_dir)
    build_labels_txt(data.val, "val", output_dir)
    build_labels_txt(data.test, "test", output_dir)

    print("All splits processed successfully (text-only).")


if __name__ == "__main__":
    main()