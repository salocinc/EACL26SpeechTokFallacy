import os
# Enable synchronous CUDA errors for debugging
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
# Disable tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import math
import torch
from torch.utils.data import Dataset
from sklearn.metrics import f1_score, classification_report, accuracy_score
import json
import numpy as np
import torch.nn as nn
import csv
from datetime import datetime
from sklearn.utils.class_weight import compute_class_weight
from transformers import (
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    AutoTokenizer,
    EarlyStoppingCallback
)
from pathlib import Path
import sys
import random
import argparse
sys.path.append(str(Path(__file__).parent.parent))
from peft import LoraConfig, get_peft_model, TaskType

# Import the audio‐token manager
from utilities.audio_token_manager import AudioTokenManager

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
            f"No pre-trained audio embeddings found at {embeddings_path}. "
        )

    return tokenizer, orig_vocab, audio_emb

class FallacyDetectionDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=256):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding="longest",
            max_length=max_length,
        )
        # print encoding lengths
        count = 0
        for _, length in enumerate(self.encodings['input_ids']):
            if len(length) > max_length:
                count += 1
        if count > 0:
            print(f"[WARNING] Found {count} samples longer than max_length={max_length}. ")
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

# Metrics function for evaluation
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=-1)
    # compute macro-avg F1 and accuracy
    f1_macro = f1_score(labels, preds, average="macro")
    # compute F1 score for class 1 (Fallacy class)
    f1_per_class = f1_score(labels, preds, average=None)  # returns F1 for each class
    f1_class_1 = f1_per_class[1] if len(f1_per_class) > 1 else 0.0  # F1 for class 1 (Fallacy)
    accuracy = accuracy_score(labels, preds)
    return {"f1_macro": f1_macro, "f1_class_1": f1_class_1, "accuracy": accuracy}

def main(learning_rate=1e-3, r_lora=64, seed=42, batch_size=32, epochs=5):
    # Clear GPU cache first
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)  # raises if nondeterministic op used

    # For CUDA library determinism
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

    AUDIO_CONFIG_NAME = "afd_config_8layers_bow_True_c_0.0025"
    
    # Paths / config names can be set via environment variables
    base_model_name    = "meta-llama/Llama-3.2-3B"
    base_model_path    = "../utilities/base_llama_model"
    embeddings_path    = f"../utilities/audio_embeddings/trained_audio_embeddings_{AUDIO_CONFIG_NAME}.pt"
    training_data_path = "../train_val_test_data_afd"

    array_of_texts_train = open(f"{training_data_path}/{AUDIO_CONFIG_NAME}_train_texts.txt").read().splitlines()
    array_of_texts_val   = open(f"{training_data_path}/{AUDIO_CONFIG_NAME}_val_texts.txt").read().splitlines()
    array_of_texts_test  = open(f"{training_data_path}/{AUDIO_CONFIG_NAME}_test_texts.txt").read().splitlines()
    array_of_fallacy_labels_train = [int(x) for x in open(f"{training_data_path}/train_labels.txt").read().splitlines()]
    array_of_fallacy_labels_val   = [int(x) for x in open(f"{training_data_path}/val_labels.txt").read().splitlines()]
    array_of_fallacy_labels_test  = [int(x) for x in open(f"{training_data_path}/test_labels.txt").read().splitlines()]

    manager = AudioTokenManager()
    # Ensure base model + metadata are saved locally
    if not os.path.exists(base_model_path):
        manager.save_base_model_once(base_model_path)

    # Load just tokenizer + audio embeddings (no LM instantiation)
    tokenizer, orig_vocab, audio_emb = load_tokenizer_and_audio_embeddings(
        base_model_path,
        AUDIO_CONFIG_NAME,
        embeddings_path,
    )
    print(f"Original vocab size: {orig_vocab}")
    print(f"Current vocab size:  {len(tokenizer)}")

    # For binary classification (AFD), we have 2 labels: 0 (no fallacy) and 1 (fallacy)
    num_labels = 2
    print(f"Number of labels for AFD: {num_labels}")

    # Base sequence-classification model (no audio tokens yet)
    cls_model = AutoModelForSequenceClassification.from_pretrained(
        base_model_name,
        num_labels=num_labels
    )

    # Resize & patch in the audio embeddings slice
    cls_model.resize_token_embeddings(len(tokenizer))
    with torch.no_grad():
        emb = cls_model.get_input_embeddings().weight
        emb[orig_vocab : orig_vocab + audio_emb.size(0), :] = audio_emb.to(emb.device)

    cls_model.config.pad_token_id = tokenizer.pad_token_id
    print("Classification model embeddings resized and initialized with audio vectors.")

    # PEFT configuration for LoRA
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        inference_mode=False,
        r=r_lora,
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj"],
    )
    model = get_peft_model(cls_model, peft_config)
    model.config.label_names = ["labels"]

    train_ds = FallacyDetectionDataset(array_of_texts_train, array_of_fallacy_labels_train, tokenizer)
    val_ds   = FallacyDetectionDataset(array_of_texts_val, array_of_fallacy_labels_val, tokenizer)
    test_ds  = FallacyDetectionDataset(array_of_texts_test, array_of_fallacy_labels_test, tokenizer)

    labels = np.array(array_of_fallacy_labels_train)
    classes = np.unique(labels)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=labels)
    class_weights = torch.tensor(weights, dtype=torch.float)

    print(f"Class distribution in training data:")
    unique, counts = np.unique(labels, return_counts=True)
    for cls, count in zip(unique, counts):
        label_name = "No Fallacy" if cls == 0 else "Fallacy"
        print(f"  Class {cls} ({label_name}): {count} samples")
    print(f"Class weights: {class_weights}")

    data_collator = DataCollatorWithPadding(tokenizer)

    # Calculate training parameters optimized for fast hyperparameter search
    per_device_batch  = batch_size  # Configurable batch size
    grad_accum_steps  = 1   # No accumulation for faster training
    num_epochs        = epochs  # Configurable number of epochs (fewer for speed)

    steps_per_epoch   = math.ceil(len(array_of_texts_train) / per_device_batch)
    total_updates     = (steps_per_epoch * num_epochs) // grad_accum_steps
    warmup_steps      = max(1, int(total_updates * 0.05))  # Reduced warmup for speed
    logging_steps     = max(1, steps_per_epoch // 4)   # Less frequent logging
    eval_steps        = steps_per_epoch   # Evaluate once per epoch for speed

    eval_steps        = max(1, steps_per_epoch // 2)   # Evaluate twice per epoch for better monitoring

    effective_batch_size = per_device_batch * grad_accum_steps
    print(f"Training parameters optimized for fast hyperparameter search:")
    print(f"  Samples: {len(array_of_texts_train):,}")
    print(f"  Per-device batch: {per_device_batch}, Grad accum: {grad_accum_steps}")
    print(f"  Effective batch size: {effective_batch_size}")
    print(f"  Epochs: {num_epochs} (reduced for speed)")
    print(f"  Steps/epoch: {steps_per_epoch}, Total updates: {total_updates:,}")
    print(f"  Warmup: {warmup_steps}, Log every: {logging_steps}, Eval every: {eval_steps} steps")

    # Set up training arguments
    training_args = TrainingArguments(
        output_dir=f"llama_lora_afd_{AUDIO_CONFIG_NAME}",
        per_device_train_batch_size=per_device_batch,
        per_device_eval_batch_size=per_device_batch,
        gradient_accumulation_steps=grad_accum_steps,
        learning_rate=learning_rate,
        num_train_epochs=num_epochs,
        warmup_steps=warmup_steps,
        eval_strategy="epoch",  # Simplified: evaluate once per epoch
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=logging_steps,
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1_class_1",
        greater_is_better=True,
        save_total_limit=2,  # Keep fewer checkpoints for speed
        fp16=True,
        seed=seed,
        dataloader_num_workers=4,  # Speed up data loading
        remove_unused_columns=False,
        report_to=None,  # Disable wandb/tensorboard for speed
    )

    class WeightedTrainer(Trainer):
        def __init__(self, class_weights: torch.Tensor, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # send to the same device as model
            self.class_weights = class_weights.to(self.args.device)
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.get("labels")
            outputs = model(**inputs)
            logits = outputs.logits
            loss_fct = nn.CrossEntropyLoss(weight=self.class_weights)
            loss = loss_fct(logits.view(-1, model.config.num_labels), labels.view(-1))
            return (loss, outputs) if return_outputs else loss

    trainer = WeightedTrainer(
        class_weights=class_weights,
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]  # Less patience for faster hyperparameter search
    )

    trainer.train()

    print("\n*** Validation results ***")
    val_results = trainer.evaluate()
    print(val_results)

    # Get validation predictions for classification report
    print("\n*** Validation set classification report ***")
    val_predictions = trainer.predict(val_ds)
    val_preds = val_predictions.predictions.argmax(axis=1)
    val_labels = val_predictions.label_ids
    target_names = ['No Fallacy', 'Fallacy']

    # Get validation classification report
    val_report_str = classification_report(
        val_labels, val_preds, digits=4, zero_division=0, target_names=target_names
    )
    val_report_dict = classification_report(
        val_labels, val_preds, output_dict=True, digits=4, zero_division=0, target_names=target_names
    )

    print(val_report_str)

    print("\n*** Test set evaluation ***")
    predictions = trainer.predict(test_dataset=test_ds)
    preds = predictions.predictions.argmax(axis=1)
    labels = predictions.label_ids
    target_names = ['No Fallacy', 'Fallacy']

    # Get both string and structured dict forms
    report_str = classification_report(
        labels, preds, digits=4, zero_division=0, target_names=target_names
    )
    report_dict = classification_report(
        labels, preds, output_dict=True, digits=4, zero_division=0, target_names=target_names
    )

    print(report_str)

    # Prepare results directory and filename with timestamp and seed
    final_dir = f"/data/ncalbucu/afd/{AUDIO_CONFIG_NAME}_{r_lora}_{learning_rate}_{seed}"
    final_dir = Path(final_dir)
    final_dir.mkdir(parents=True, exist_ok=True)

    # Save validation classification report
    val_base_name = final_dir / f"validation_classification_report"
    
    # Save validation human-friendly text report
    val_txt_path = val_base_name.with_suffix('.txt')
    with open(val_txt_path, 'w', encoding='utf-8') as f:
        f.write(f"# Validation classification report (AFD - Argument Fallacy Detection)\n# seed: {seed}\n\n")
        f.write(val_report_str)

    # Save validation JSON (structured) report
    val_json_path = val_base_name.with_suffix('.json')
    with open(val_json_path, 'w', encoding='utf-8') as f:
        json.dump({
            'seed': seed,
            'task': 'AFD',
            'report': val_report_dict
        }, f, indent=2)

    # Save validation CSV for easy tabular inspection
    val_csv_path = val_base_name.with_suffix('.csv')
    with open(val_csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['label', 'precision', 'recall', 'f1-score', 'support'])
        for label, metrics in val_report_dict.items():
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

    print(f"Saved validation classification report to:\n  {val_txt_path}\n  {val_json_path}\n  {val_csv_path}")

    # Save test classification report
    test_base_name = final_dir / f"test_classification_report"

    # Save human-friendly text report
    txt_path = test_base_name.with_suffix('.txt')
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(f"# Test classification report (AFD - Argument Fallacy Detection)\n# seed: {seed}\n\n")
        f.write(report_str)

    # Save JSON (structured) report
    json_path = test_base_name.with_suffix('.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump({
            'seed': seed,
            'task': 'AFD',
            'report': report_dict
        }, f, indent=2)

    # Save CSV for easy tabular inspection
    csv_path = test_base_name.with_suffix('.csv')
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

    print(f"Saved test classification report to:\n  {txt_path}\n  {json_path}\n  {csv_path}")

    # Save final model and tokenizer
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"Saved LoRA adapter + tokenizer to: {final_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune Llama model with LoRA for fallacy detection (AFD)")
    parser.add_argument(
        "--learning_rate", 
        type=float, 
        default=1e-3, 
        help="Learning rate for training (default: 1e-3)"
    )
    parser.add_argument(
        "--r_lora", 
        type=int, 
        default=64, 
        help="LoRA rank parameter (default: 64)"
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=42, 
        help="Random seed for initialization (default: 42)"
    )
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=32, 
        help="Per-device batch size (default: 32, optimized for fast training)"
    )
    parser.add_argument(
        "--epochs", 
        type=int, 
        default=5, 
        help="Number of training epochs (default: 5, optimized for hyperparameter search)"
    )
    
    args = parser.parse_args()
    
    print(f"Starting AFD fine-tuning with:")
    print(f"  Task: Argument Fallacy Detection (binary classification)")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  LoRA rank (r): {args.r_lora}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Random seed: {args.seed}")
    print("-" * 50)

    main(learning_rate=args.learning_rate, r_lora=args.r_lora, seed=args.seed, 
         batch_size=args.batch_size, epochs=args.epochs)