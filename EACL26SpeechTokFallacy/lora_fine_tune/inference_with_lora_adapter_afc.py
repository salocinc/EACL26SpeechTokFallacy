import os
import json
import torch
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import PeftModel, PeftConfig
from safetensors.torch import load_file, save_file
import argparse
from typing import List, Tuple, Optional, Dict, Any
from sklearn.metrics import classification_report
import csv
from pathlib import Path
import shutil
import tempfile

def load_tokenizer_and_audio_embeddings(
    base_model_path: str,
    audio_config_name: str,
    embeddings_path: str = None,
    ):
    """
    Load just the tokenizer (with added audio+special tokens).
    Returns:
      - tokenizer: AutoTokenizer
      - original_vocab_size: int
      - audio_embeddings: torch.Tensor, shape (n_new_tokens, embedding_dim)
    """
    # Base‐model metadata
    meta_file = os.path.join(base_model_path, "base_model_metadata.json")
    with open(meta_file, "r") as f:
        meta = json.load(f)
    orig_vocab = meta["original_vocab_size"]

    # Audio‐token config
    cfg_file = os.path.join("../utilities/audio_token_configs", f"{audio_config_name}.json")
    with open(cfg_file, "r") as f:
        audio_cfg = json.load(f)

    # Load & extend tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, use_fast=True)
    new_tokens = audio_cfg["audio_tokens"] + audio_cfg["special_tokens"]
    tokenizer.add_tokens(new_tokens)
    tokenizer.add_special_tokens({"additional_special_tokens": audio_cfg["special_tokens"]})

    # Load or randomly init audio embeddings
    if embeddings_path and os.path.exists(embeddings_path):
        data = torch.load(embeddings_path, map_location="cpu")
        audio_emb = data["audio_embeddings"]
    else:
        raise ValueError(
            f"No pre-trained audio embeddings found at {embeddings_path}."
        )

    return tokenizer, orig_vocab, audio_emb, audio_cfg["audio_tokens"]

def average_lora_adapters(adapter_prefix_path: str, output_path: str = None, seeds: List[str] = None) -> str:
    """
    Average LoRA adapter weights from multiple seed folders
    
    Args:
        adapter_prefix_path: Base path without seed (e.g., "/data/ncalbucu/config_8layers_bow_True_c_0.03_64_0.001")
        output_path: Path where to save the averaged adapter. If None, creates temp directory
        seeds: List of seed suffixes. If None, uses default seeds [10, 100, 42, 420, 6]
        
    Returns:
        Path to the averaged adapter directory
    """
    if seeds is None:
        seeds = ["10", "100", "42", "420", "6"]
    
    print(f"Averaging LoRA adapters from {len(seeds)} different seeds: {seeds}")
    
    # Collect all adapter paths
    adapter_paths = []
    for seed in seeds:
        path = f"{adapter_prefix_path}_{seed}"
        if os.path.exists(path):
            adapter_paths.append(path)
            print(f"Found adapter: {path}")
        else:
            print(f"Warning: Adapter not found at {path}")
    
    if len(adapter_paths) == 0:
        raise ValueError(f"No adapters found with prefix {adapter_prefix_path}")
    
    if len(adapter_paths) < len(seeds):
        print(f"Warning: Only found {len(adapter_paths)} out of {len(seeds)} expected adapters")
    
    # Create output directory
    if output_path is None:
        output_path = tempfile.mkdtemp(prefix="averaged_adapter_")
    else:
        os.makedirs(output_path, exist_ok=True)
    
    # Load and average adapter weights
    averaged_weights = None
    adapter_config = None
    
    for i, adapter_path in enumerate(adapter_paths):
        print(f"Loading adapter {i+1}/{len(adapter_paths)}: {adapter_path}")
        
        # Load adapter weights
        weights_path = os.path.join(adapter_path, "adapter_model.safetensors")
        if not os.path.exists(weights_path):
            print(f"Warning: No weights found at {weights_path}, skipping...")
            continue
            
        weights = load_file(weights_path)
        
        # Load adapter config (from the first adapter)
        if adapter_config is None:
            config_path = os.path.join(adapter_path, "adapter_config.json")
            if os.path.exists(config_path):
                with open(config_path, "r") as f:
                    adapter_config = json.load(f)
        
        # Initialize or accumulate weights
        if averaged_weights is None:
            averaged_weights = {key: tensor.clone().float() for key, tensor in weights.items()}
        else:
            for key, tensor in weights.items():
                if key in averaged_weights:
                    averaged_weights[key] += tensor.float()
                else:
                    print(f"Warning: Key {key} not found in previous adapters")
                    averaged_weights[key] = tensor.clone().float()
    
    # Average the weights
    num_adapters = len(adapter_paths)
    for key in averaged_weights:
        averaged_weights[key] /= num_adapters
    
    print(f"Averaged weights from {num_adapters} adapters")
    
    # Save averaged adapter
    # Save the weights
    weights_output_path = os.path.join(output_path, "adapter_model.safetensors")
    save_file(averaged_weights, weights_output_path)
    
    # Save config
    if adapter_config:
        config_output_path = os.path.join(output_path, "adapter_config.json")
        with open(config_output_path, "w") as f:
            json.dump(adapter_config, f, indent=2)
    
    # Copy other necessary files from the first adapter
    first_adapter_path = adapter_paths[0]
    files_to_copy = ["tokenizer_config.json", "tokenizer.json", "special_tokens_map.json"]
    
    for filename in files_to_copy:
        src_path = os.path.join(first_adapter_path, filename)
        if os.path.exists(src_path):
            dst_path = os.path.join(output_path, filename)
            shutil.copy2(src_path, dst_path)
            print(f"Copied {filename}")
    
    print(f"Averaged adapter saved to: {output_path}")
    return output_path

def load_model_for_inference(
    base_model_name: str,
    base_model_path: str,
    audio_config_name: str,
    embeddings_path: str,
    lora_adapter_path: str,
    device: str = "cpu",
    use_averaged_adapter: bool = False
) -> Tuple[torch.nn.Module, AutoTokenizer]:
    """
    Load the base model with LoRA adapter weights for inference
    
    Args:
        base_model_name: Huggingface model name (meta-llama/Llama-3.2-3B)
        base_model_path: Path to locally saved base model
        audio_config_name: Name of audio token configuration
        embeddings_path: Path to trained audio embeddings
        lora_adapter_path: Path to trained LoRA adapter (or prefix if using averaged)
        device: Computing device ('cpu' or 'cuda')
        use_averaged_adapter: If True, treats lora_adapter_path as prefix and averages adapters
        
    Returns:
        model: Model with LoRA adapter loaded
        tokenizer: Tokenizer with audio tokens
    """
    # Load tokenizer and audio embeddings
    tokenizer, orig_vocab, audio_emb, audio_tokens = load_tokenizer_and_audio_embeddings(
        base_model_path,
        audio_config_name,
        embeddings_path,
    )
    
    print(f"Original vocab size: {orig_vocab}")
    print(f"Current vocab size: {len(tokenizer)}")
    
    # Fallacy classification has 6 labels
    num_labels = 6
    
    # Load the base model for sequence classification
    base_model = AutoModelForSequenceClassification.from_pretrained(
        base_model_name,
        num_labels=num_labels
    )
    
    # Resize the model's embeddings to match the tokenizer
    base_model.resize_token_embeddings(len(tokenizer))
    
    # Initialize the audio embeddings
    with torch.no_grad():
        emb = base_model.get_input_embeddings().weight
        emb[orig_vocab : orig_vocab + audio_emb.size(0), :] = audio_emb.to(emb.device)
    
    # Set padding token ID
    base_model.config.pad_token_id = tokenizer.pad_token_id
    
    # Handle adapter loading
    if use_averaged_adapter:
        print("Using averaged adapter from multiple seeds...")
        averaged_adapter_path = average_lora_adapters(lora_adapter_path, output_path=f"{lora_adapter_path}_averaged")
        final_adapter_path = averaged_adapter_path
    else:
        final_adapter_path = lora_adapter_path
    
    # Load and apply the LoRA adapter
    model = PeftModel.from_pretrained(base_model, final_adapter_path)
    model.eval()  # Set to evaluation mode
    model.to(device)
    
    print(f"Model loaded successfully on {device}")
    
    return model, tokenizer

def batch_predict_fallacies(
    texts: List[str], 
    model: torch.nn.Module, 
    tokenizer: AutoTokenizer, 
    device: str = "cpu", 
    batch_size: int = 8
) -> Tuple[List[int], torch.Tensor]:
    """
    Process multiple text samples in batches for efficient inference
    
    Args:
        texts: List of text samples to classify
        model: The LoRA-adapted model
        tokenizer: Tokenizer with audio tokens
        device: Computing device ('cpu' or 'cuda')
        batch_size: Number of samples to process at once
        
    Returns:
        all_predictions: List of predicted class indices
        all_probabilities: Tensor of class probabilities for each sample
    """
    model.to(device)
    all_predictions = []
    all_probabilities = []
    
    # Process in batches to avoid out-of-memory errors
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        
        # Tokenize the batch with padding
        inputs = tokenizer(
            batch_texts, 
            padding=True, 
            truncation=True, 
            return_tensors="pt", 
            max_length=256
        ).to(device)
        
        # Run inference without gradient calculation
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Calculate probabilities and predictions
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
        predictions = torch.argmax(probabilities, dim=-1)
        
        # Store results
        all_predictions.extend(predictions.cpu().numpy())
        all_probabilities.append(probabilities.cpu())
    
    # Combine batched probability results
    if len(all_probabilities) > 1:
        all_probabilities = torch.cat(all_probabilities, dim=0)
    else:
        all_probabilities = all_probabilities[0]
    
    return all_predictions, all_probabilities

def run_inference_and_evaluate(
    texts: List[str],
    true_labels: List[int],
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    adapter_path: str,
    device: str = "cpu",
    batch_size: int = 8,
    texts_file_path: Optional[str] = None,
    labels_file_path: Optional[str] = None
):
    """
    Run inference on provided texts, compare with true labels, and save evaluation results.

    Args:
        texts: Input samples to classify.
        true_labels: Ground-truth class indices aligned with ``texts``.
        model: The LoRA-adapted model.
        tokenizer: Tokenizer with audio tokens.
        adapter_path: Path/name of the adapter (used for naming output folder).
        device: Computing device ('cpu' or 'cuda').
        batch_size: Number of samples to process at once.
        texts_file_path: Optional path to the source texts file (stored for metadata in the report).
        labels_file_path: Optional path to the labels file (stored for metadata in the report).
    """
    if len(texts) != len(true_labels):
        raise ValueError(f"Number of texts ({len(texts)}) does not match number of labels ({len(true_labels)})")

    # Analyze available labels in the test set
    unique_test_labels = sorted(list(set(true_labels)))
    print(f"Running inference on {len(texts)} samples")
    print(f"Unique labels present in test set: {unique_test_labels}")

    # Process in batches
    predictions, probabilities = batch_predict_fallacies(
        texts=texts,
        model=model,
        tokenizer=tokenizer,
        device=device,
        batch_size=batch_size
    )

    # Complete fallacy classes mapping (for all 6 original classes)
    all_fallacy_classes = ['Appeal to emotion', 'Appeal to authority',
                          'Ad hominem', 'False cause', 'Slippery slope', 'Slogans']
    
    # Create subset of classes that are actually present in the test set
    present_fallacy_classes = [all_fallacy_classes[i] for i in unique_test_labels]
    
    print(f"Fallacy classes present in test set: {present_fallacy_classes}")

    # Generate classification report only for labels present in test set
    report_str = classification_report(
        true_labels, predictions, digits=4, zero_division=0, 
        target_names=present_fallacy_classes, labels=unique_test_labels
    )
    report_dict = classification_report(
        true_labels, predictions, output_dict=True, digits=4, zero_division=0, 
        target_names=present_fallacy_classes, labels=unique_test_labels
    )    
    
    # Create results directory
    results_dir = Path(f"inference_results/{Path(adapter_path).name}")
    results_dir.mkdir(parents=True, exist_ok=True)
    base_name = results_dir / "classification_report"
    
    # Save text report
    txt_path = base_name.with_suffix('.txt')
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write("# Inference Classification Report\n\n")
        f.write(f"## Dataset Information\n")
        f.write(f"- Texts file: {texts_file_path}\n")
        f.write(f"- Labels file: {labels_file_path}\n")
        f.write(f"- Total samples: {len(texts)}\n\n")
        f.write(f"## Label Analysis\n")
        f.write(f"- Model trained with 6 classes: {all_fallacy_classes}\n")
        f.write(f"- Test set contains {len(unique_test_labels)} classes: {present_fallacy_classes}\n")
        f.write(f"- Labels present in test set: {unique_test_labels}\n")
        if len(unique_test_labels) < 6:
            missing_labels = [i for i in range(6) if i not in unique_test_labels]
            missing_classes = [all_fallacy_classes[i] for i in missing_labels]
            f.write(f"- Missing classes in test set: {missing_classes} (labels: {missing_labels})\n")
        f.write(f"\n## Classification Results\n")
        f.write(report_str)
    
    # Save JSON (structured) report
    json_path = base_name.with_suffix('.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump({
            'adapter_path': adapter_path,
            'texts_file': texts_file_path,
            'labels_file': labels_file_path,
            'unique_test_labels': unique_test_labels,
            'present_fallacy_classes': present_fallacy_classes,
            'report': report_dict
        }, f, indent=2)
    
    # Save CSV for easy tabular inspection
    csv_path = base_name.with_suffix('.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['label', 'precision', 'recall', 'f1-score', 'support'])
        for label, metrics in report_dict.items():
            if isinstance(metrics, dict):
                writer.writerow([
                    label,
                    metrics.get('precision', ''),
                    metrics.get('recall', ''),
                    metrics.get('f1-score', ''),
                    metrics.get('support', '')
                ])
            else:
                # For entries like 'accuracy' that are not dicts
                writer.writerow([label, '', '', metrics, ''])
    
    print(f"Classification report saved to:\n  {txt_path}\n  {json_path}\n  {csv_path}")
    
    # Print additional information about label handling
    print(f"\n=== Label Analysis ===")
    print(f"Model was trained with 6 classes: {all_fallacy_classes}")
    print(f"Test set contains {len(unique_test_labels)} classes: {present_fallacy_classes}")
    if len(unique_test_labels) < 6:
        missing_labels = [i for i in range(6) if i not in unique_test_labels]
        missing_classes = [all_fallacy_classes[i] for i in missing_labels]
        print(f"Missing classes in test set: {missing_classes} (labels: {missing_labels})")
    
    print(f"\nClassification Report:\n{report_str}")
    
    return probabilities, predictions, true_labels, report_dict

def main():
    parser = argparse.ArgumentParser(description="Run inference with LoRA-adapted model")
    parser.add_argument("--base_model_name", type=str, default="meta-llama/Llama-3.2-3B",
                        help="Base model name")
    parser.add_argument("--base_model_path", type=str, default="../utilities/base_llama_model",
                        help="Path to locally saved base model")
    parser.add_argument("--audio_config_name", type=str, default="config_8layers_bow_True_c_0.03",
                        help="Audio token configuration name")
    parser.add_argument("--lora_adapter_path", type=str, default="/data/ncalbucu/config_8layers_bow_True_c_0.03_64_0.001",
                        help="Path to trained LoRA adapter (or prefix path when using --use_averaged_adapter)")
    parser.add_argument("--use_averaged_adapter", action="store_true",
                        help="Average adapters from 5 different seeds (10, 100, 42, 420, 6)")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for inference")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device for inference (cuda or cpu)")
    
    args = parser.parse_args()
    
    # Build audio config name from parameters
    audio_config_name = args.audio_config_name
    embeddings_path = f"../utilities/audio_embeddings/trained_audio_embeddings_{audio_config_name}.pt"
    
    # Define paths to test data files (modify these paths as needed)
    texts_file_path = f"../train_val_test_data_llama/{audio_config_name}_val_texts.txt"
    labels_file_path = "../train_val_test_data_llama/val_labels.txt"

    print(f"Using audio config: {audio_config_name}")
    print(f"Embeddings path: {embeddings_path}")
    print(f"Texts file: {texts_file_path}")
    print(f"Labels file: {labels_file_path}")
    
    if args.use_averaged_adapter:
        print(f"Using averaged adapter with prefix: {args.lora_adapter_path}")
    else:
        print(f"Using single adapter: {args.lora_adapter_path}")
    
    # Load evaluation data
    with open(texts_file_path, 'r', encoding='utf-8') as f:
        texts = f.read().splitlines()
    with open(labels_file_path, 'r', encoding='utf-8') as f:
        true_labels_list = [int(x) for x in f.read().splitlines()]

    # Load model and tokenizer
    model, tokenizer = load_model_for_inference(
        base_model_name=args.base_model_name,
        base_model_path=args.base_model_path,
        audio_config_name=audio_config_name,
        embeddings_path=embeddings_path,
        lora_adapter_path=args.lora_adapter_path,
        device=args.device,
        use_averaged_adapter=args.use_averaged_adapter
    )
    
    # Run inference and evaluation
    probabilities, predictions, true_labels, report_dict = run_inference_and_evaluate(
        texts=texts,
        true_labels=true_labels_list,
        model=model,
        tokenizer=tokenizer,
        adapter_path=args.lora_adapter_path,
        device=args.device,
        batch_size=args.batch_size,
        texts_file_path=texts_file_path,
        labels_file_path=labels_file_path
    )

    # save probabilities and predictions if needed
    np.save(f"inference_results/{Path(args.lora_adapter_path).name}/probabilities.npy", probabilities.numpy())
    np.save(f"inference_results/{Path(args.lora_adapter_path).name}/predictions.npy", np.array(predictions))
    np.save(f"inference_results/{Path(args.lora_adapter_path).name}/true_labels.npy", np.array(true_labels))

if __name__ == "__main__":
    main()