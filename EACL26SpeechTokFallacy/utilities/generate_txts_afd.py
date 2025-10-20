import numpy as np
import pandas as pd
from pathlib import Path
import sys
import json
from speechtokenizer import SpeechTokenizer
import torchaudio
import torch
import gc
import os
import argparse

# Memory management function
def free_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# Function to process audio files
def process_audio(audio_path_list, model):
    processed_wavs = []
    for path in audio_path_list:
        try:
            wav, sr = torchaudio.load(path)
            if wav.shape[0] > 1:
                wav = wav[:1, :]
            if sr != model.sample_rate:
                wav = torchaudio.functional.resample(wav, sr, model.sample_rate)
            processed_wavs.append(wav)
        except Exception as e:
            print(f"Error processing audio file {path}: {e}")
    return processed_wavs

def build_split_texts(split, split_name, ST_model, device, audio_config, feature_to_token_map, output_dir):
    print(f"Processing {split_name} split...")
    split_texts = []
    split_audios = []

    for idx in range(len(split.texts)):
        audio_path_list = split.audio[idx]
        # split.audio may contain a single Path per example or a list of Paths.
        # Normalize to a list of paths.
        if isinstance(audio_path_list, (str, Path)) or not hasattr(audio_path_list, '__iter__'):
            audio_path_list = [audio_path_list]
        speech_text = split.texts[idx]

        processed_wavs = process_audio(audio_path_list, ST_model)
        if not processed_wavs:
            continue
        
        # Concatenate if there's more than one file
        if len(processed_wavs) == 1:
            final_wav = processed_wavs[0]
        else:
            final_wav = torch.cat(processed_wavs, dim=1)

        final_wav = final_wav.to(device).unsqueeze(0)

        with torch.no_grad():
            codes = ST_model.encode(final_wav)  # (n_q, B, T)
            codes = codes.cpu().numpy()

        codes2 = []
        # Align layer indexing logic with original code
        for i in range(audio_config["n_layers_rvq"]):
            RVQ_i = codes[i + 8 - audio_config["n_layers_rvq"], :, :]
            RVQ_i = RVQ_i + (i * 1024)
            codes2.append(RVQ_i)

        codes2 = np.array(codes2).T
        filtered_codes = codes2[np.isin(codes2, audio_config["selected_features"])]

        split_audios.append(filtered_codes.flatten().tolist())
        split_texts.append(speech_text)

        del final_wav, codes, codes2, filtered_codes
        free_memory()

    llama_texts = []
    for i in range(len(split_texts)):
        array_of_tokens = split_audios[i]
        token_ids = [token for token in array_of_tokens if token in feature_to_token_map]
        string_tokens = " ".join([feature_to_token_map[token] for token in token_ids])
        llama_texts.append(
            f"<|text|> {split_texts[i]} <|text_end|> <|audio|> {string_tokens} <|audio_end|>"
        )

    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{audio_config['config_name']}_{split_name}_texts.txt")
    # output_file = os.path.join(output_dir, f"new_{audio_config['config_name']}_{split_name}_texts.txt")
    with open(output_file, 'w', encoding="utf-8") as f:
        for item in llama_texts:
            f.write(f"{item}\n")
    print(f"Saved {split_name} texts to {output_file}")

def build_labels_txt(split, split_name, output_dir):
    print(f"Processing {split_name} split for labels...")
    split_labels = []

    for idx in range(len(split.texts)):
        fallacy = split.labels[idx]
        split_labels.append(fallacy)

    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{split_name}_labels.txt")
    with open(output_file, 'w', encoding="utf-8") as f:
        for item in split_labels:
            f.write(f"{item}\n")

    print(f"Saved {split_name} labels to {output_file}")

def parse_bool(val):
    if isinstance(val, bool):
        return val
    v = val.lower()
    if v in ("true", "1", "yes", "y"):
        return True
    if v in ("false", "0", "no", "n"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")

def main():
    parser = argparse.ArgumentParser(description="Generate train/val/test llama text+audio token files.")
    parser.add_argument("audio_config_name", type=str, nargs='?', help="Name of the audio configuration to use", default="afd_config_8layers_bow_True_c_0.0025")
    args = parser.parse_args()

    audio_config_name = args.audio_config_name
    print(f"Audio config name: {audio_config_name}")

    # Add parent directory of 'mamkit' to path
    sys.path.append(str(Path(__file__).parent.parent.parent))
    from mamkit.data.datasets import MMUSEDFallacy, InputMode

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"Available GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("Using CPU")

    # Load SpeechTokenizer
    config_path = '../../speechtokenizer/config.json'
    ckpt_path = '../../speechtokenizer/ckpt.dev'
    print("Loading SpeechTokenizer...")
    ST_model = SpeechTokenizer.load_from_checkpoint(config_path, ckpt_path)
    ST_model.eval()
    ST_model = ST_model.to(device)
    print("SpeechTokenizer loaded")

    base_data_path = Path(__file__).parent.parent.parent.resolve().joinpath('data')
    
    print("Loading MMUSEDFallacy dataset...")
    mmused_fallacy_splits_dir = base_data_path.joinpath('train_test_val_mmused_fallacy_afd')
    mmused_fallacy_splits_dir.mkdir(parents=True, exist_ok=True)

    train_data1 = pd.read_pickle(mmused_fallacy_splits_dir.joinpath('train_data.pkl'))
    val_data1 = pd.read_pickle(mmused_fallacy_splits_dir.joinpath('val_data.pkl'))
    test_data1 = pd.read_pickle(mmused_fallacy_splits_dir.joinpath('test_data2.pkl'))

    # Normalize columns to match what Loader expects:
    # some utilities create DataFrames with columns 'text' and 'audio';
    # loader._get_text_audio_data expects 'sentence' and 'sentence_path'.
    def _normalize_afd_df(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        rename_map = {}

        # Standardize text column
        if 'sentence' not in df.columns:
            for candidate in ['text', 'texts', 'snippet']:
                if candidate in df.columns:
                    rename_map[candidate] = 'sentence'
                    break

        # Standardize audio path column
        if 'sentence_path' not in df.columns:
            for candidate in ['audio', 'audio_path', 'sentence_paths', 'snippet_paths']:
                if candidate in df.columns:
                    rename_map[candidate] = 'sentence_path'
                    break

        # Standardize label column
        if 'label' not in df.columns:
            for candidate in ['labels', 'fallacy', 'fallacies']:
                if candidate in df.columns:
                    rename_map[candidate] = 'label'
                    break

        if rename_map:
            df = df.rename(columns=rename_map)

        missing_columns = {col for col in ['sentence', 'sentence_path', 'label'] if col not in df.columns}
        if missing_columns:
            raise ValueError(f"Required columns missing after normalization: {missing_columns}. Available columns: {list(df.columns)}")

        return df

    train_data1 = _normalize_afd_df(train_data1)
    val_data1 = _normalize_afd_df(val_data1)
    test_data1 = _normalize_afd_df(test_data1)

    loader = MMUSEDFallacy(
        task_name='afd',
        input_mode=InputMode.TEXT_AUDIO,
        base_data_path=base_data_path
    )

    data = loader.build_info_from_splits(
        train_df=train_data1,
        val_df=val_data1,
        test_df=test_data1
    )
    # data = loader.get_splits('mm-argfallacy-2025')[0]  # Get the first (and only) split
    print("Dataset loaded")

    config_file = Path("audio_token_configs").joinpath(f"{audio_config_name}.json")
    if not config_file.exists():
        raise FileNotFoundError(f"Audio config file not found: {config_file}")
    with open(config_file, 'r') as f:
        audio_config = json.load(f)

    feature_to_token_map = {int(k): v for k, v in audio_config["feature_to_token_map"].items()}

    output_dir = "../train_val_test_data_afd"

    # Generate for each split
    build_split_texts(data.train, "train", ST_model, device, audio_config, feature_to_token_map, output_dir)
    build_split_texts(data.val, "val", ST_model, device, audio_config, feature_to_token_map, output_dir)
    build_split_texts(data.test, "test", ST_model, device, audio_config, feature_to_token_map, output_dir)
    build_labels_txt(data.train, "train", output_dir)
    build_labels_txt(data.val, "val", output_dir)
    build_labels_txt(data.test, "test", output_dir)

    print("All splits processed successfully.")

if __name__ == "__main__":
    main()