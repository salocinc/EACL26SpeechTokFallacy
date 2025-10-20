import os
import json
import argparse
import csv
import shutil
import tempfile
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
import torch
from peft import PeftModel
from safetensors.torch import load_file, save_file
from sklearn.metrics import classification_report
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def average_lora_adapters(
    adapter_prefix_path: str,
    output_path: Optional[str] = None,
    seeds: Optional[List[str]] = None,
) -> str:
    """
    Average LoRA adapter weights from multiple seed directories.

    Args:
        adapter_prefix_path: Base path without the seed suffix.
        output_path: Target directory to store the averaged adapter weights.
        seeds: Optional list of seed suffixes to search for.

    Returns:
        Path to the directory containing the averaged adapter.
    """
    if seeds is None:
        seeds = ["10", "100", "42", "420", "6"]

    print(f"Averaging LoRA adapters from seeds: {seeds}")

    adapter_paths: List[str] = []
    for seed in seeds:
        candidate = f"{adapter_prefix_path}_{seed}"
        if os.path.exists(candidate):
            adapter_paths.append(candidate)
            print(f"  [OK] Found adapter: {candidate}")
        else:
            print(f"  [WARNING] Missing adapter: {candidate}")

    if not adapter_paths:
        raise ValueError(f"No adapter directories found with prefix {adapter_prefix_path}")

    if output_path is None:
        output_path = tempfile.mkdtemp(prefix="averaged_adapter_text_only_afd_")
    else:
        os.makedirs(output_path, exist_ok=True)

    averaged_weights: Optional[Dict[str, torch.Tensor]] = None
    adapter_config: Optional[Dict[str, Any]] = None

    for idx, adapter_dir in enumerate(adapter_paths, start=1):
        print(f"Loading adapter {idx}/{len(adapter_paths)}: {adapter_dir}")
        weights_path = os.path.join(adapter_dir, "adapter_model.safetensors")
        if not os.path.exists(weights_path):
            print(f"  [WARNING] Skipping {adapter_dir}; missing adapter_model.safetensors")
            continue

        weights = load_file(weights_path)

        if adapter_config is None:
            config_path = os.path.join(adapter_dir, "adapter_config.json")
            if os.path.exists(config_path):
                with open(config_path, "r", encoding="utf-8") as f:
                    adapter_config = json.load(f)

        if averaged_weights is None:
            averaged_weights = {key: tensor.clone().float() for key, tensor in weights.items()}
        else:
            for key, tensor in weights.items():
                if key in averaged_weights:
                    averaged_weights[key] += tensor.float()
                else:
                    print(f"  [WARNING] Encountered new tensor key {key}; adding to accumulator")
                    averaged_weights[key] = tensor.clone().float()

    if averaged_weights is None:
        raise RuntimeError("No adapter weights loaded; verify adapter directories contain weights.")

    count = len(adapter_paths)
    for key in averaged_weights:
        averaged_weights[key] /= count

    weights_output_path = os.path.join(output_path, "adapter_model.safetensors")
    save_file(averaged_weights, weights_output_path)

    if adapter_config:
        with open(os.path.join(output_path, "adapter_config.json"), "w", encoding="utf-8") as f:
            json.dump(adapter_config, f, indent=2)

    first_adapter_dir = adapter_paths[0]
    for filename in ["tokenizer_config.json", "tokenizer.json", "special_tokens_map.json", "vocab.json"]:
        src = os.path.join(first_adapter_dir, filename)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(output_path, filename))

    print(f"Averaged adapter saved to: {output_path}")
    return output_path


def load_model_for_inference(
    base_model_name: str,
    lora_adapter_path: str,
    device: str = "cpu",
    use_averaged_adapter: bool = False,
) -> Tuple[torch.nn.Module, AutoTokenizer, str]:
    """
    Load the base text-only AFD model with LoRA adapter weights for inference.

    Returns:
        model: Sequence classification model with the adapter applied.
        tokenizer: Tokenizer aligned with the adapter vocabulary.
        adapter_dir: Directory where the adapter weights were sourced (averaged or single seed).
    """
    num_labels = 2  # Binary classification: No Fallacy vs Fallacy

    base_model = AutoModelForSequenceClassification.from_pretrained(
        base_model_name,
        num_labels=num_labels,
    )

    if use_averaged_adapter:
        print("Using averaged adapter across multiple seeds.")
        averaged_path = average_lora_adapters(
            adapter_prefix_path=lora_adapter_path,
            output_path=f"{lora_adapter_path}_averaged",
        )
        adapter_dir = averaged_path
    else:
        adapter_dir = lora_adapter_path

    # Load tokenizer from adapter directory (contains tokenizer with correct vocab size)
    tokenizer = AutoTokenizer.from_pretrained(adapter_dir, use_fast=True)
    
    # Resize model embeddings to match tokenizer size BEFORE loading adapter
    base_model.resize_token_embeddings(len(tokenizer))

    base_model.config.pad_token_id = tokenizer.pad_token_id

    model = PeftModel.from_pretrained(base_model, adapter_dir)
    model.eval()
    model.to(device)
    print(f"Model loaded on device: {device}")

    return model, tokenizer, adapter_dir


def batch_predict_fallacies(
    texts: List[str],
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    device: str = "cpu",
    batch_size: int = 8,
) -> Tuple[List[int], torch.Tensor]:
    """Process samples in batches to obtain predictions and probabilities."""
    model.to(device)
    predictions: List[int] = []
    probability_chunks: List[torch.Tensor] = []

    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start : start + batch_size]
        inputs = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=256,
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)

        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        preds = torch.argmax(probs, dim=-1)

        predictions.extend(preds.cpu().tolist())
        probability_chunks.append(probs.cpu())

    probabilities = (
        torch.cat(probability_chunks, dim=0) if len(probability_chunks) > 1 else probability_chunks[0]
    )
    return predictions, probabilities


def run_inference_and_evaluate(
    texts: List[str],
    true_labels: List[int],
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    adapter_path: str,
    device: str = "cpu",
    batch_size: int = 8,
    texts_file_path: Optional[str] = None,
    labels_file_path: Optional[str] = None,
    results_dir_root: str = "inference_text_only_afd_results",
) -> Tuple[torch.Tensor, List[int], List[int], Dict[str, Any]]:
    """
    Generate predictions, compute metrics, and persist reports for AFD.
    """

    if len(texts) != len(true_labels):
        raise ValueError(
            f"Number of texts ({len(texts)}) does not match number of labels ({len(true_labels)})"
        )

    unique_labels = sorted(set(true_labels))
    print(f"Running inference on {len(texts)} samples")
    print(f"Labels present in evaluation set: {unique_labels}")

    predictions, probabilities = batch_predict_fallacies(
        texts=texts,
        model=model,
        tokenizer=tokenizer,
        device=device,
        batch_size=batch_size,
    )

    fallacy_classes = ["No Fallacy", "Fallacy"]
    present_classes = [fallacy_classes[i] for i in unique_labels]

    report_str = classification_report(
        true_labels,
        predictions,
        digits=4,
        zero_division=0,
        target_names=present_classes,
        labels=unique_labels,
    )
    report_dict = classification_report(
        true_labels,
        predictions,
        output_dict=True,
        digits=4,
        zero_division=0,
        target_names=present_classes,
        labels=unique_labels,
    )

    results_dir = Path(results_dir_root) / Path(adapter_path).name
    results_dir.mkdir(parents=True, exist_ok=True)
    report_base = results_dir / "classification_report"

    txt_path = report_base.with_suffix(".txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("# Text-Only AFD Inference Classification Report\n\n")
        f.write("## Dataset Information\n")
        f.write(f"- Texts file: {texts_file_path}\n")
        f.write(f"- Labels file: {labels_file_path}\n")
        f.write(f"- Total samples: {len(texts)}\n\n")
        f.write("## Label Analysis\n")
        f.write(f"- Model trained with 2 classes: {fallacy_classes}\n")
        f.write(
            f"- Evaluation set contains {len(unique_labels)} classes: {present_classes}\n"
        )
        f.write(f"- Labels present in evaluation set: {unique_labels}\n")
        if len(unique_labels) < len(fallacy_classes):
            missing = [i for i in range(len(fallacy_classes)) if i not in unique_labels]
            missing_classes = [fallacy_classes[i] for i in missing]
            f.write(
                f"- Missing classes in evaluation set: {missing_classes} (labels: {missing})\n"
            )
        f.write("\n## Classification Results\n")
        f.write(report_str)

    json_path = report_base.with_suffix(".json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "adapter_path": adapter_path,
                "texts_file": texts_file_path,
                "labels_file": labels_file_path,
                "unique_test_labels": unique_labels,
                "present_fallacy_classes": present_classes,
                "report": report_dict,
            },
            f,
            indent=2,
        )

    csv_path = report_base.with_suffix(".csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["label", "precision", "recall", "f1-score", "support"])
        for label, metrics in report_dict.items():
            if isinstance(metrics, dict):
                writer.writerow(
                    [
                        label,
                        metrics.get("precision", ""),
                        metrics.get("recall", ""),
                        metrics.get("f1-score", ""),
                        metrics.get("support", ""),
                    ]
                )
            else:
                writer.writerow([label, "", "", metrics, ""])

    print("Classification report saved to:")
    print(f"  {txt_path}")
    print(f"  {json_path}")
    print(f"  {csv_path}")

    print("\n=== Label Analysis ===")
    print("Model trained with classes: No Fallacy (0), Fallacy (1)")
    if len(unique_labels) < len(fallacy_classes):
        missing = [i for i in range(len(fallacy_classes)) if i not in unique_labels]
        missing_classes = [fallacy_classes[i] for i in missing]
        print(f"Missing classes in evaluation set: {missing_classes} (labels: {missing})")

    print(f"\nClassification Report:\n{report_str}")

    return probabilities, predictions, true_labels, report_dict


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run text-only inference with a LoRA-adapted LLaMA model for AFD",
    )
    parser.add_argument(
        "--base_model_name",
        type=str,
        default="meta-llama/Llama-3.2-3B",
        help="Base Hugging Face model name",
    )
    parser.add_argument(
        "--lora_adapter_path",
        type=str,
        default="/data/ncalbucu/afd_text_only_lora_64_0.001",
        help=(
            "Path to the trained LoRA adapter directory. "
            "When combined with --use_averaged_adapter, treated as prefix without seed suffix."
        ),
    )
    parser.add_argument(
        "--use_averaged_adapter",
        action="store_true",
        help="Average adapters across multiple seeds sharing the provided prefix",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for inference",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for inference (cuda or cpu)",
    )
    parser.add_argument(
        "--texts_file",
        type=str,
        default="../train_val_test_data_afd_text_only/test_texts.txt",
        help="Path to the evaluation texts file",
    )
    parser.add_argument(
        "--labels_file",
        type=str,
        default="../train_val_test_data_afd_text_only/test_labels.txt",
        help="Path to the evaluation labels file",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="inference_text_only_afd_results",
        help="Directory root where inference artifacts will be written",
    )

    args = parser.parse_args()

    print("Text-only AFD inference configuration:")
    print(f"  Base model: {args.base_model_name}")
    print(f"  Texts file: {args.texts_file}")
    print(f"  Labels file: {args.labels_file}")
    if args.use_averaged_adapter:
        print(f"  Adapter prefix (averaged): {args.lora_adapter_path}")
    else:
        print(f"  Adapter path: {args.lora_adapter_path}")

    # Load texts and labels
    with open(args.texts_file, "r", encoding="utf-8") as f:
        texts = f.read().splitlines()
    with open(args.labels_file, "r", encoding="utf-8") as f:
        true_labels = [int(x) for x in f.read().splitlines()]

    model, tokenizer, adapter_dir = load_model_for_inference(
        base_model_name=args.base_model_name,
        lora_adapter_path=args.lora_adapter_path,
        device=args.device,
        use_averaged_adapter=args.use_averaged_adapter,
    )

    probabilities, predictions, labels, report_dict = run_inference_and_evaluate(
        texts=texts,
        true_labels=true_labels,
        model=model,
        tokenizer=tokenizer,
        adapter_path=args.lora_adapter_path,
        device=args.device,
        batch_size=args.batch_size,
        texts_file_path=args.texts_file,
        labels_file_path=args.labels_file,
        results_dir_root=args.results_dir,
    )

    results_dir = Path(args.results_dir) / Path(args.lora_adapter_path).name
    np.save(results_dir / "probabilities.npy", probabilities.numpy())
    np.save(results_dir / "predictions.npy", np.array(predictions, dtype=np.int64))
    np.save(results_dir / "true_labels.npy", np.array(labels, dtype=np.int64))


if __name__ == "__main__":
    main()
