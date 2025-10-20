# Audio Token Integration for Multimodal Argument Mining

A research project for integrating audio tokens into textual Large Language Models (LLMs) for improved multimodal argument mining, specifically for fallacy classification tasks. This work leverages the [MAMKit (Multimodal Argument Mining Toolkit)](https://nlp-unibo.github.io/mamkit/) framework for baseline comparisons and evaluation.

## Table of Contents
- [Introduction](#introduction)
- [Project Overview](#project-overview)
- [Installation](#installation)
- [Datasets](#datasets)
- [Project Structure](#project-structure)
- [Workflow](#workflow)
  - [Step 1: Audio Token Selection](#step-1-audio-token-selection)
  - [Step 2: Audio Embedding Training](#step-2-audio-embedding-training)
  - [Step 3: LLM Fine-Tuning](#step-3-llm-fine-tuning)
- [Usage Examples](#usage-examples)
- [Baselines](#baselines)
- [Results Analysis](#results-analysis)
- [Configuration Management](#configuration-management)
- [Troubleshooting](#troubleshooting)

## Introduction

This project explores a novel approach to multimodal argument mining by integrating discrete audio tokens into textual Large Language Models (specifically LLaMA 3.2-3B). The methodology involves:

1. **Extracting discrete audio tokens** using SpeechTokenizer's Residual Vector Quantization (RVQ)
2. **Selecting relevant audio features** through supervised learning or random sampling
3. **Training audio embeddings** aligned with the LLM's semantic space
4. **Fine-tuning the LLM** with LoRA adapters for downstream fallacy classification tasks

This approach enables the model to leverage both textual and acoustic information for improved argument analysis, particularly for tasks like Argumentative Fallacy Classification (AFC) and Argumentative Fallacy Detection (AFD).

## Project Overview

The project utilizes the MAMKit framework for baseline implementations and focuses on two main tasks from the MM-USED-fallacy dataset:

- **Argumentative Fallacy Classification (AFC)**: Classify argumentative segments into 6 fallacy types
  - Appeal to emotion
  - Appeal to authority
  - Ad hominem
  - False cause
  - Slippery slope
  - Slogans

- **Argumentative Fallacy Detection (AFD)**: Binary classification to detect presence/absence of fallacies

### Key Innovation

Instead of processing audio as raw waveforms or traditional features (MFCCs, spectrograms), this approach:
- Discretizes audio into tokens similar to text tokens
- Selects discriminative audio tokens using supervised learning
- Integrates them into the LLM vocabulary as special tokens
- Trains embeddings that align with the model's semantic space

## Installation

### Prerequisites

- **Python 3.10** (tested version)
- **CUDA-compatible GPU** (recommended for training)
- **ffmpeg** (required for audio processing)
  ```bash
  # Debian/Ubuntu
  sudo apt install ffmpeg
  
  # macOS
  brew install ffmpeg
  
  # Windows (using Chocolatey)
  choco install ffmpeg
  ```

### Setup

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd memoria_online
   ```

2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Install MAMKit dependencies:
   ```bash
   # The mamkit package is included in the repository
   # Dependencies should be covered by requirements.txt
   ```

4. Download SpeechTokenizer checkpoint:
   ```bash
   # Place the SpeechTokenizer model files in:
   # ./speechtokenizer/config.json
   # ./speechtokenizer/ckpt.dev
   ```

5. Download LLaMA 3.2-3B model:
   ```bash
   # Ensure you have access to meta-llama/Llama-3.2-3B on Hugging Face
   # The scripts will download it automatically on first run
   ```

## Datasets

The project uses the **MM-USED-fallacy** dataset, which contains:
- Multimodal argumentative data (text + audio)
- Annotations for fallacy types
- Train/validation/test splits

The datasets are accessed and loaded using MAMKit's data processing utilities.

Dataset files are located in:
- `MMUSED-fallacy_dataset_csv/dataset_combined_afc.csv` (AFC task)
- `MMUSED-fallacy_dataset_csv/dataset_combined_afd.csv` (AFD task)

Pre-processed splits are stored in pickle format:
- `data/train_test_val_mmused_fallacy/train_data.pkl`
- `data/train_test_val_mmused_fallacy/val_data.pkl`
- `data/train_test_val_mmused_fallacy/test_data.pkl`

## Project Structure

```
memoria_online/
├── EACL26SpeechTokFallacy/                  # Main experimental code
│   ├── select_AT_and_generate_pretrain_txts/
│   │   ├── add_audio_tokens_afc.py          # Step 1: Audio token selection (AFC)
│   │   └── add_audio_tokens_afd.py          # Step 1: Audio token selection (AFD)
│   ├── train_audio_embeddings/
│   │   └── llama_training.py                # Step 2: Train audio embeddings
│   ├── lora_fine_tune/
│   │   ├── lora_fine_tune_afc.py            # Step 3: Fine-tune LLM (AFC)
│   │   ├── lora_fine_tune_afd.py            # Step 3: Fine-tune LLM (AFD)
│   │   ├── lora_fine_tune_only_text_afc.py  # Text-only baseline (AFC)
│   │   ├── lora_fine_tune_only_text_afd.py  # Text-only baseline (AFD)
│   │   ├── inference_with_lora_adapter_afc.py   # Inference with LoRA (AFC)
│   │   ├── inference_with_lora_adapter_afd.py   # Inference with LoRA (AFD)
│   │   ├── inference_text_only_lora_afc.py      # Inference text-only (AFC)
│   │   ├── inference_text_only_lora_afd.py      # Inference text-only (AFD)
│   │   ├── analyze_multi_seed_results_afc.py    # Multi-seed analysis (AFC)
│   │   ├── analyze_multi_seed_results_afd.py    # Multi-seed analysis (AFD)
│   │   ├── calculate_average_f1_afc.py          # F1 calculation (AFC)
│   │   ├── calculate_average_f1_class1_afd.py   # F1 calculation (AFD)
│   │   └── run_hyperparameter_sweep.sh          # Hyperparameter sweep script
│   ├── baselines/                           # Baseline models
│   │   ├── baseline_roberta_wavlm.py        # RoBERTa + WavLM baseline
│   │   ├── baseline_roberta.py              # RoBERTa baseline
│   │   ├── baseline_lstm_wavlm.py           # LSTM + WavLM baseline
│   │   ├── baseline_mfccs.py                # MFCC-based baseline
│   │   ├── baseline_bow_mfccs.py            # BoW + MFCC fusion
│   │   ├── baseline_bow_only.py             # BoW-only baseline
│   │   ├── baseline_llama.py                # LLaMA baseline
│   │   └── hyperparameter_sweep.sh          # Baseline hyperparameter sweep
│   ├── utilities/
│   │   ├── audio_token_manager.py           # Core audio token management
│   │   ├── generate_txts_afc.py             # Generate training texts (AFC)
│   │   ├── generate_txts_afd.py             # Generate training texts (AFD)
│   │   ├── generate_txts_afd_text_only.py   # Generate text-only data (AFD)
│   │   ├── generate_sets_afc.py             # Data split generation (AFC)
│   │   ├── generate_sets_afd.py             # Data split generation (AFD)
│   │   └── tokens_ratio.py                  # Token ratio analysis
│   └── MMUSED-fallacy_dataset/              # Local dataset folder
│       ├── dataset_combined_afc.csv
│       └── dataset_combined_afd.csv
├── mamkit/                                  # MAMKit framework
│   ├── configs/                             # Model configurations
│   ├── data/                                # Data loading and processing
│   ├── models/                              # Model architectures
│   ├── modules/                             # Neural network modules
│   └── utility/                             # Utility functions
├── MMUSED-fallacy_dataset_csv/              # Dataset files
│   ├── dataset_combined_afc.csv
│   └── dataset_combined_afd.csv
├── requirements.txt                         # Python dependencies
└── README.md                                # This file
```

## Workflow

The complete workflow consists of three main steps:

### Step 1: Audio Token Selection

**Purpose**: Extract and select discriminative audio tokens from the training data.

**Process**:
1. Load audio files from the MM-USED-fallacy dataset
2. Extract discrete codes using SpeechTokenizer's RVQ (8 layers, 1024 codebook size per layer)
3. Convert to bag-of-tokens representation
4. Select relevant tokens using one of two methods:
   - **Model-based selection**: Logistic regression with L1 regularization + TF-IDF
   - **Random selection**: Random sampling (for ablation studies)
5. Generate configuration files and pretraining texts

**Scripts**:
- `EACL26SpeechTokFallacy/select_AT_and_generate_pretrain_txts/add_audio_tokens_afc.py` (AFC task)
- `EACL26SpeechTokFallacy/select_AT_and_generate_pretrain_txts/add_audio_tokens_afd.py` (AFD task)

**Usage**:
```bash
cd EACL26SpeechTokFallacy/select_AT_and_generate_pretrain_txts

# Model-based selection (default)
python add_audio_tokens_afc.py --n_layers_rvq 8 --type model --use-bow --c_value 0.03

# Random selection (for ablation)
python add_audio_tokens_afc.py --n_layers_rvq 8 --type random --random-token-count 100
```

**Arguments**:
- `--n_layers_rvq`: Number of RVQ layers to use (1-8, default: 8)
- `--type`: Selection method - `model` or `random` (default: model)
- `--use-bow` / `--no-use-bow`: Enable/disable bag-of-words features (default: enabled)
- `--c_value`: Regularization parameter for logistic regression (default: 0.03)
- `--random-token-count`: Number of random tokens (required when `--type random`)

**Outputs**:
- `utilities/audio_token_configs/config_*.json`: Audio token configuration
- `training_data/config_*_pretrain_texts.txt`: Pretraining texts with audio tokens
- `ablation_results/*.json`: Ablation study results (if enabled)

### Step 2: Audio Embedding Training

**Purpose**: Train embeddings for the selected audio tokens to align with the LLM's semantic space.

**Process**:
1. Load base LLaMA model and extend vocabulary with audio tokens
2. Freeze all model parameters except new token embeddings
3. Train embeddings using causal language modeling on text+audio pretraining data
4. Save trained embeddings separately from the base model

**Script**:
- `EACL26SpeechTokFallacy/train_audio_embeddings/llama_training.py`

**Usage**:
```bash
cd EACL26SpeechTokFallacy/train_audio_embeddings

python llama_training.py \
  --config_name config_8layers_bow_False_c_0.03 \
  --learning_rate 3e-4 \
  --batch_size 8 \
  --gradient_accumulation_steps 4 \
  --epochs 5 \
  --seed 42
```

**Arguments**:
- `--config_name`: Audio token configuration name (from Step 1)
- `--learning_rate`: Learning rate (default: 3e-4)
- `--batch_size`: Per-device batch size (default: 8)
- `--gradient_accumulation_steps`: Gradient accumulation steps (default: 4)
- `--epochs`: Number of training epochs (default: 5)
- `--seed`: Random seed (default: 42)

**Outputs**:
- `utilities/audio_embeddings/trained_audio_embeddings_*.pt`: Trained embeddings
- `utilities/base_llama_model/`: Base model checkpoint (saved once)
- Training logs and checkpoints

### Step 3: LLM Fine-Tuning

**Purpose**: Fine-tune the LLM with LoRA adapters for downstream fallacy classification.

**Process**:
1. Load base LLaMA model with trained audio embeddings
2. Add LoRA adapters to specific layers (q_proj, v_proj)
3. Fine-tune on fallacy classification task using class-balanced loss
4. Evaluate on validation and test sets
5. Generate classification reports

**Scripts**:
- `EACL26SpeechTokFallacy/lora_fine_tune/lora_fine_tune_afc.py` (AFC task)
- `EACL26SpeechTokFallacy/lora_fine_tune/lora_fine_tune_afd.py` (AFD task)

**Usage**:
```bash
cd EACL26SpeechTokFallacy/lora_fine_tune

python lora_fine_tune_afc.py \
  --learning_rate 1e-3 \
  --r_lora 64 \
  --batch_size 32 \
  --epochs 10 \
  --seed 42
```

**Arguments**:
- `--learning_rate`: Learning rate (default: 1e-3)
- `--r_lora`: LoRA rank parameter (default: 64)
- `--batch_size`: Per-device batch size (default: 32)
- `--epochs`: Number of training epochs (default: 10)
- `--seed`: Random seed (default: 42)

**Outputs**:
- Classification reports (TXT, JSON, CSV formats)
- Fine-tuned LoRA adapters
- Model checkpoints

## Usage Examples

### Complete Pipeline Example (AFC Task)

```bash
# Step 1: Select audio tokens
cd EACL26SpeechTokFallacy/select_AT_and_generate_pretrain_txts
python add_audio_tokens_afc.py \
  --n_layers_rvq 8 \
  --type model \
  --use-bow \
  --c_value 0.03

# Step 2: Train audio embeddings
cd ../train_audio_embeddings
python llama_training.py \
  --config_name config_8layers_bow_True_c_0.03 \
  --learning_rate 3e-4 \
  --batch_size 8 \
  --epochs 5 \
  --seed 42

# Step 3a: Generate fine-tuning data
cd ../utilities
python generate_txts_afc.py config_8layers_bow_True_c_0.03

# Step 3b: Fine-tune with LoRA
cd ../lora_fine_tune
python lora_fine_tune_afc.py \
  --learning_rate 1e-3 \
  --r_lora 64 \
  --batch_size 32 \
  --epochs 10 \
  --seed 42
```

### Multi-Seed Evaluation

For robust evaluation across multiple random seeds:

```bash
cd EACL26SpeechTokFallacy/lora_fine_tune

# Run with different seeds
for seed in 42 123 456 789 1011; do
  python lora_fine_tune_afc.py --seed $seed
done

# Analyze results
python analyze_multi_seed_results_afc.py
```

### Hyperparameter Sweep

```bash
cd EACL26SpeechTokFallacy/lora_fine_tune

# Edit run_hyperparameter_sweep.sh to configure parameter ranges
bash run_hyperparameter_sweep.sh
```

## Baselines

The project includes several baseline models for comparison:

### Running Baselines

```bash
cd EACL26SpeechTokFallacy/baselines

# RoBERTa + WavLM baseline
python baseline_roberta_wavlm.py --seed 42 --learning_rate 1e-3

# MFCC baseline
python baseline_mfccs.py --seed 42
```

## Results Analysis

### Analyzing Multi-Seed Results

The project provides scripts to aggregate and analyze results across multiple random seeds:

**For AFC (6-class classification)**:
```bash
cd EACL26SpeechTokFallacy/lora_fine_tune
python analyze_multi_seed_results_afc.py

# Or calculate average F1 scores
python calculate_average_f1_afc.py
```

**For AFD (binary classification)**:
```bash
cd EACL26SpeechTokFallacy/lora_fine_tune
python analyze_multi_seed_results_afd.py

# Calculate class-1 F1 score
python calculate_average_f1_class1_afd.py
```

### Output Formats

Classification reports are saved in three formats:
- **TXT**: Human-readable format
- **JSON**: Structured format for programmatic access
- **CSV**: Tabular format for spreadsheet analysis

### Key Metrics

- **Macro-averaged F1**: Primary metric for AFC task (handles class imbalance)
- **Class-specific F1**: Per-class performance analysis
- **Precision/Recall**: Detailed performance breakdown

## Configuration Management

The `AudioTokenManager` class (`utilities/audio_token_manager.py`) manages:

- **Base model storage**: Saves LLaMA model once for reuse
- **Audio token configs**: Maps selected features to token strings
- **Embedding management**: Handles loading/saving of trained embeddings
- **Vocabulary extension**: Adds audio tokens to tokenizer

### Configuration File Format

Audio token configurations are JSON files with:
```json
{
  "config_name": "config_8layers_bow_False_c_0.03",
  "n_layers_rvq": 8,
  "selected_features": [1, 5, 12, ...],
  "feature_to_token_map": {
    "1": "<audio_token_0>",
    "5": "<audio_token_1>",
    ...
  },
  "special_tokens": ["<|text|>", "<|text_end|>", "<|audio|>", "<|audio_end|>"],
  "total_new_tokens": 104,
  "audio_tokens": ["<audio_token_0>", "<audio_token_1>", ...]
}
```


## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size
   - Increase gradient accumulation steps
   - Enable FP16 training (already default)

2. **SpeechTokenizer Not Found**
   - Ensure model files are in `./speechtokenizer/`
   - Check file paths in scripts

3. **Audio Token Mismatch**
   - Ensure same config used across all steps
   - Check config files in `utilities/audio_token_configs/`

4. **Determinism Issues**
   - Use same seed across all steps
   - Check CUDA determinism settings
   - Note: Some operations may still be non-deterministic on GPU


## Acknowledgments

This project uses the [MAMKit (Multimodal Argument Mining Toolkit)](https://nlp-unibo.github.io/mamkit/) framework. For detailed documentation about MAMKit's base functionality, see `MAMKIT_README.md`.

---

**Note**: This is a research project. While the code is functional, it may require adaptation for specific use cases or datasets.
