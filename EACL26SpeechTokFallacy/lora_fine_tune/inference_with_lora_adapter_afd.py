import os
import json
import torch
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import PeftModel
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
    embeddings_path: str,
) -> Tuple[AutoTokenizer, int, torch.Tensor, List[str]]:
    """
    Load tokenizer and audio embeddings for Argument Fallacy Detection (AFD).

    Args:
        base_model_path: Path to the locally cached base model.
        audio_config_name: Name of the audio-token configuration file (without extension).
        embeddings_path: Path to the trained audio embeddings .pt file.

    Returns:
        tokenizer: Tokenizer with audio tokens injected.
        original_vocab_size: Vocab size before audio tokens were added.
        audio_embeddings: Tensor of learned audio embeddings.
        audio_tokens: List of audio token strings for downstream analysis.
    """
    meta_file = os.path.join(base_model_path, "base_model_metadata.json")
    with open(meta_file, "r", encoding="utf-8") as f:
        meta = json.load(f)
    orig_vocab = meta["original_vocab_size"]

    cfg_file = os.path.join("../utilities/audio_token_configs", f"{audio_config_name}.json")
    with open(cfg_file, "r", encoding="utf-8") as f:
        audio_cfg = json.load(f)

    tokenizer = AutoTokenizer.from_pretrained(base_model_path, use_fast=True)
    new_tokens = audio_cfg["audio_tokens"] + audio_cfg["special_tokens"]
    tokenizer.add_tokens(new_tokens)
    tokenizer.add_special_tokens({"additional_special_tokens": audio_cfg["special_tokens"]})

    if not os.path.exists(embeddings_path):
        raise FileNotFoundError(
            f"Audio embeddings not found at {embeddings_path}. Ensure you ran fine-tuning first."
        )

    data = torch.load(embeddings_path, map_location="cpu")
    audio_emb = data["audio_embeddings"]

    return tokenizer, orig_vocab, audio_emb, audio_cfg["audio_tokens"]


def average_lora_adapters(
    adapter_prefix_path: str,
    output_path: Optional[str] = None,
    seeds: Optional[List[str]] = None,
) -> str:
    """
    Average multiple LoRA adapter checkpoints that share a prefix.

    Args:
        adapter_prefix_path: Base path without the seed suffix.
        output_path: Directory to store the averaged adapter weights.
        seeds: Optional list of seed suffixes. Defaults to ["10", "100", "42", "420", "6"].

    Returns:
        Path to the directory containing the averaged adapter.
    """
    if seeds is None:
        seeds = ["10", "100", "42", "420", "6"]

    print(f"Averaging LoRA adapters using seeds: {seeds}")

    adapter_paths: List[str] = []
    for seed in seeds:
        candidate = f"{adapter_prefix_path}_{seed}"
        if os.path.exists(candidate):
            adapter_paths.append(candidate)
            print(f"  [OK] found {candidate}")
        else:
            print(f"  [WARNING] missing {candidate}")

    if not adapter_paths:
        raise ValueError(f"No adapter directories found with prefix {adapter_prefix_path}")

    if output_path is None:
        output_path = tempfile.mkdtemp(prefix="averaged_adapter_")
    else:
        os.makedirs(output_path, exist_ok=True)

    averaged_weights: Optional[Dict[str, torch.Tensor]] = None
    adapter_config: Optional[Dict[str, Any]] = None

    for idx, adapter_dir in enumerate(adapter_paths, start=1):
        print(f"Loading adapter {idx}/{len(adapter_paths)}: {adapter_dir}")
        weights_path = os.path.join(adapter_dir, "adapter_model.safetensors")
        if not os.path.exists(weights_path):
            print(f"  [WARNING] skipping {adapter_dir}, missing safetensors file")
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
                    print(f"  [WARNING] new tensor key {key} encountered; adding to accumulator")
                    averaged_weights[key] = tensor.clone().float()

    if averaged_weights is None:
        raise RuntimeError("No adapter weights loaded; verify the adapter directories contain weights.")

    count = len(adapter_paths)
    for key in averaged_weights:
        averaged_weights[key] /= count

    weights_output_path = os.path.join(output_path, "adapter_model.safetensors")
    save_file(averaged_weights, weights_output_path)

    if adapter_config:
        with open(os.path.join(output_path, "adapter_config.json"), "w", encoding="utf-8") as f:
            json.dump(adapter_config, f, indent=2)

    first_adapter_dir = adapter_paths[0]
    for filename in ["tokenizer_config.json", "tokenizer.json", "special_tokens_map.json"]:
        src = os.path.join(first_adapter_dir, filename)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(output_path, filename))

    print(f"Averaged adapter saved to: {output_path}")
    return output_path


def load_model_for_inference(
    base_model_name: str,
    base_model_path: str,
    audio_config_name: str,
    embeddings_path: str,
    lora_adapter_path: str,
    device: str = "cpu",
    use_averaged_adapter: bool = False,
) -> Tuple[torch.nn.Module, AutoTokenizer]:
    """
    Load the PEFT-enhanced AFD classifier with audio embeddings applied.

    Returns:
        model: Sequence-classification model with LoRA adapter loaded.
        tokenizer: Tokenizer with audio tokens.
    """
    tokenizer, orig_vocab, audio_emb, audio_tokens = load_tokenizer_and_audio_embeddings(
        base_model_path=base_model_path,
        audio_config_name=audio_config_name,
        embeddings_path=embeddings_path,
    )

    print(f"Original vocab size: {orig_vocab}")
    print(f"Tokenizer vocab size after audio merge: {len(tokenizer)}")

    num_labels = 2  # Binary classification: No Fallacy vs Fallacy

    base_model = AutoModelForSequenceClassification.from_pretrained(
        base_model_name,
        num_labels=num_labels,
    )
    base_model.resize_token_embeddings(len(tokenizer))

    with torch.no_grad():
        emb = base_model.get_input_embeddings().weight
        emb[orig_vocab : orig_vocab + audio_emb.size(0), :] = audio_emb.to(emb.device)

    base_model.config.pad_token_id = tokenizer.pad_token_id

    if use_averaged_adapter:
        print("Using averaged adapter across multiple seeds.")
        averaged_path = average_lora_adapters(
            adapter_prefix_path=lora_adapter_path,
            output_path=f"{lora_adapter_path}_averaged",
        )
        adapter_dir = averaged_path
    else:
        adapter_dir = lora_adapter_path

    model = PeftModel.from_pretrained(base_model, adapter_dir)
    model.eval()
    model.to(device)
    print(f"Model loaded on device: {device}")

    return model, tokenizer


def batch_predict_fallacies(
    texts: List[str],
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    device: str = "cpu",
    batch_size: int = 8,
) -> Tuple[List[int], torch.Tensor]:
    """
    Run batched inference for Argument Fallacy Detection samples.
    """
    model.to(device)
    predictions: List[int] = []
    prob_chunks: List[torch.Tensor] = []

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
        prob_chunks.append(probs.cpu())

    probabilities = torch.cat(prob_chunks, dim=0) if len(prob_chunks) > 1 else prob_chunks[0]
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
    results_dir_root: str = "inference_results_afd",
) -> Tuple[torch.Tensor, List[int], List[int], Dict[str, Any]]:
    """
    Generate predictions, compute metrics, and persist reports for AFD.
    """
    if len(texts) != len(true_labels):
        raise ValueError(
            f"Number of texts ({len(texts)}) does not match number of labels ({len(true_labels)})"
        )

    unique_test_labels = sorted(set(true_labels))
    print(f"Running inference on {len(texts)} samples.")
    print(f"Unique labels present in evaluation set: {unique_test_labels}")

    predictions, probabilities = batch_predict_fallacies(
        texts=texts,
        model=model,
        tokenizer=tokenizer,
        device=device,
        batch_size=batch_size,
    )

    fallacy_classes = ["No Fallacy", "Fallacy"]
    present_classes = [fallacy_classes[i] for i in unique_test_labels]

    report_str = classification_report(
        true_labels,
        predictions,
        digits=4,
        zero_division=0,
        target_names=present_classes,
        labels=unique_test_labels,
    )
    report_dict = classification_report(
        true_labels,
        predictions,
        output_dict=True,
        digits=4,
        zero_division=0,
        target_names=present_classes,
        labels=unique_test_labels,
    )

    results_dir = Path(results_dir_root) / Path(adapter_path).name
    results_dir.mkdir(parents=True, exist_ok=True)
    base_name = results_dir / "classification_report"

    txt_path = base_name.with_suffix(".txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("# Inference Classification Report (AFD)\n\n")
        f.write("## Dataset Information\n")
        f.write(f"- Texts file: {texts_file_path}\n")
        f.write(f"- Labels file: {labels_file_path}\n")
        f.write(f"- Total samples: {len(texts)}\n\n")
        f.write("## Label Analysis\n")
        f.write(f"- Model trained with 2 classes: {fallacy_classes}\n")
        f.write(f"- Evaluation set contains {len(unique_test_labels)} classes: {present_classes}\n")
        f.write(f"- Labels present in evaluation set: {unique_test_labels}\n")
        if len(unique_test_labels) < len(fallacy_classes):
            missing = [i for i in range(len(fallacy_classes)) if i not in unique_test_labels]
            missing_classes = [fallacy_classes[i] for i in missing]
            f.write(f"- Missing classes in evaluation set: {missing_classes} (labels: {missing})\n")
        f.write("\n## Classification Results\n")
        f.write(report_str)

    json_path = base_name.with_suffix(".json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "adapter_path": adapter_path,
                "texts_file": texts_file_path,
                "labels_file": labels_file_path,
                "unique_test_labels": unique_test_labels,
                "present_fallacy_classes": present_classes,
                "report": report_dict,
            },
            f,
            indent=2,
        )

    csv_path = base_name.with_suffix(".csv")
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
    if len(unique_test_labels) < len(fallacy_classes):
        missing = [i for i in range(len(fallacy_classes)) if i not in unique_test_labels]
        missing_classes = [fallacy_classes[i] for i in missing]
        print(f"Missing classes in evaluation set: {missing_classes} (labels: {missing})")

    print(f"\nClassification Report:\n{report_str}")

    return probabilities, predictions, true_labels, report_dict


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run inference with LoRA-adapted LLaMA model for Argument Fallacy Detection",
    )
    parser.add_argument(
        "--base_model_name",
        type=str,
        default="meta-llama/Llama-3.2-3B",
        help="Base Hugging Face model name",
    )
    parser.add_argument(
        "--base_model_path",
        type=str,
        default="../utilities/base_llama_model",
        help="Path to locally cached base model",
    )
    parser.add_argument(
        "--audio_config_name",
        type=str,
        default="afd_config_8layers_bow_True_c_0.0025",
        help="Audio token configuration name (exclude .json)",
    )
    parser.add_argument(
        "--lora_adapter_path",
        type=str,
        default="/data/ncalbucu/afd/afd_config_8layers_bow_True_c_0.0025_64_0.001_42",
        help=(
            "Path to the trained LoRA adapter directory. "
            "When combined with --use_averaged_adapter, treated as prefix without seed suffix."
        ),
    )
    parser.add_argument(
        "--use_averaged_adapter",
        action="store_true",
        help="Average adapters across seeds (expects directories sharing the provided prefix)",
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
        "--texts_file_path",
        type=str,
        default=None,
        help="Optional override for the texts file used during evaluation",
    )
    parser.add_argument(
        "--labels_file_path",
        type=str,
        default=None,
        help="Optional override for the labels file used during evaluation",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="inference_results_afd",
        help="Directory root where inference artifacts will be written",
    )

    args = parser.parse_args()

    audio_config_name = args.audio_config_name
    embeddings_path = f"../utilities/audio_embeddings/trained_audio_embeddings_{audio_config_name}.pt"

    default_texts_path = f"../train_val_test_data_afd/{audio_config_name}_test_texts.txt"
    default_labels_path = "../train_val_test_data_afd/test_labels.txt"

    texts_file_path = args.texts_file_path or default_texts_path
    labels_file_path = args.labels_file_path or default_labels_path

    print(f"Using audio config: {audio_config_name}")
    print(f"Embeddings path: {embeddings_path}")
    print(f"Texts file: {texts_file_path}")
    print(f"Labels file: {labels_file_path}")
    if args.use_averaged_adapter:
        print(f"Adapter prefix (averaged): {args.lora_adapter_path}")
    else:
        print(f"Adapter path: {args.lora_adapter_path}")

    with open(texts_file_path, "r", encoding="utf-8") as f:
        texts = f.read().splitlines()
    with open(labels_file_path, "r", encoding="utf-8") as f:
        true_labels = [int(x) for x in f.read().splitlines()]

    model, tokenizer = load_model_for_inference(
        base_model_name=args.base_model_name,
        base_model_path=args.base_model_path,
        audio_config_name=audio_config_name,
        embeddings_path=embeddings_path,
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
        texts_file_path=texts_file_path,
        labels_file_path=labels_file_path,
        results_dir_root=args.results_dir,
    )

    results_dir = Path(args.results_dir) / Path(args.lora_adapter_path).name
    np.save(results_dir / "probabilities.npy", probabilities.numpy())
    np.save(results_dir / "predictions.npy", np.array(predictions, dtype=np.int64))
    np.save(results_dir / "true_labels.npy", np.array(labels, dtype=np.int64))

if __name__ == "__main__":
    main()
