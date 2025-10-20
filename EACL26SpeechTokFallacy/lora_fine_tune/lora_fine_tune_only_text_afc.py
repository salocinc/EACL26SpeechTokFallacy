import os
# Enable synchronous CUDA errors for debugging
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import math
import torch
from torch.utils.data import Dataset
from sklearn.metrics import f1_score, classification_report
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

# Dataset for text-only classification
def load_texts_and_labels(text_file: str, label_file: str):
    texts = open(text_file, "r", encoding="utf-8").read().splitlines()
    labels = [int(x) for x in open(label_file, "r").read().splitlines()]
    return texts, labels

class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=256):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding="longest",
            max_length=max_length,
        )
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
    f1_macro = f1_score(labels, preds, average="macro")
    return {"f1_macro": f1_macro}


def main(learning_rate=1e-3, r_lora=64, seed=42):
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

    # Configuration
    base_model_name    = "meta-llama/Llama-3.2-3B"
    base_model_dir    = "../utilities/base_llama_model"
    train_text_file   = "../train_val_test_data_llama/train_texts.txt"
    val_text_file     = "../train_val_test_data_llama/val_texts.txt"
    test_text_file    = "../train_val_test_data_llama/test_texts.txt"
    train_label_file  = "../train_val_test_data_llama/train_labels.txt"
    val_label_file    = "../train_val_test_data_llama/val_labels.txt"
    test_label_file   = "../train_val_test_data_llama/test_labels.txt"
    num_epochs        = 10
    per_device_batch  = 32
    grad_accum_steps  = 1

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast=True)

    # Determine number of labels and sanity-check
    labels_arr = np.array([int(x) for x in open(train_label_file).read().splitlines()])
    num_labels = len(set(labels_arr.tolist()))

    # Load model weights from local base model directory
    model = AutoModelForSequenceClassification.from_pretrained(
        base_model_name,
        num_labels=num_labels
    )
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer))  # Resize embeddings to match tokenizer
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id

    # Load data
    train_texts, train_labels = load_texts_and_labels(train_text_file, train_label_file)
    val_texts, val_labels     = load_texts_and_labels(val_text_file, val_label_file)
    test_texts, test_labels   = load_texts_and_labels(test_text_file, test_label_file)

    # Prepare datasets
    train_ds = TextDataset(train_texts, train_labels, tokenizer)
    val_ds   = TextDataset(val_texts,   val_labels,   tokenizer)
    test_ds  = TextDataset(test_texts,  test_labels,  tokenizer)

    # Compute class weights
    weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(labels_arr),
        y=labels_arr
    )
    class_weights = torch.tensor(weights, dtype=torch.float)

    # Data collator
    data_collator = DataCollatorWithPadding(tokenizer)

    # Compute scheduling
    steps_per_epoch = math.ceil(len(train_texts) / per_device_batch)
    total_updates   = (steps_per_epoch * num_epochs) // grad_accum_steps
    warmup_steps    = max(1, int(total_updates * 0.1))
    logging_steps   = max(1, steps_per_epoch // 5)

    # Training arguments (disable fp16 for stability)
    training_args = TrainingArguments(
        output_dir="llama_lora_text",
        per_device_train_batch_size=per_device_batch,
        per_device_eval_batch_size=per_device_batch,
        gradient_accumulation_steps=grad_accum_steps,
        learning_rate=learning_rate,
        num_train_epochs=num_epochs,
        warmup_steps=warmup_steps,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=logging_steps,
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1_macro", 
        greater_is_better=True,
        save_total_limit=2,
        fp16=True,
        seed=42,
    )

    # LoRA configuration
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        inference_mode=False,
        r=r_lora,
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj"],
    )
    peft_model = get_peft_model(model, peft_config)
    peft_model.config.label_names = ["labels"]

    # Custom Trainer with class weights
    class WeightedTrainer(Trainer):
        def __init__(self, class_weights: torch.Tensor, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.class_weights = class_weights.to(self.args.device)

        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.logits
            loss = nn.CrossEntropyLoss(weight=self.class_weights)(
                logits.view(-1, model.config.num_labels),
                labels.view(-1)
            )
            return (loss, outputs) if return_outputs else loss

    # Initialize Trainer
    trainer = WeightedTrainer(
        class_weights=class_weights,
        model=peft_model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)]
    )

    # Train and evaluate
    trainer.train()
    print("\n*** Validation results ***")
    val_results = trainer.evaluate()
    print(val_results)

    # Get validation predictions for classification report
    print("\n*** Validation set classification report ***")
    val_predictions = trainer.predict(val_ds)
    val_preds = val_predictions.predictions.argmax(axis=1)
    val_labels = val_predictions.label_ids
    target_names = [
        'Appeal to emotion', 'Appeal to authority',
        'Ad hominem', 'False cause', 'Slippery slope', 'Slogans'
    ]

    # Get validation classification report
    val_report_str = classification_report(
        val_labels, val_preds, digits=4, zero_division=0, target_names=target_names
    )
    val_report_dict = classification_report(
        val_labels, val_preds, output_dict=True, digits=4, zero_division=0, target_names=target_names
    )

    print("\n*** Test set evaluation ***")
    predictions = trainer.predict(test_ds)
    preds = predictions.predictions.argmax(axis=1)
    labels = predictions.label_ids
    target_names = [
        'Appeal to emotion', 'Appeal to authority',
        'Ad hominem', 'False cause', 'Slippery slope', 'Slogans'
    ]

    # Get both string and structured dict forms
    report_str = classification_report(
        labels, preds, digits=4, zero_division=0, target_names=target_names
    )
    report_dict = classification_report(
        labels, preds, output_dict=True, digits=4, zero_division=0, target_names=target_names
    )

    # Prepare results directory and filename with timestamp and seed
    final_dir = f"/data/ncalbucu/llama_lora_text_final_{r_lora}_{learning_rate}_{seed}"
    final_dir = Path(final_dir)
    final_dir.mkdir(parents=True, exist_ok=True)

    # Save validation classification report
    val_base_name = final_dir / f"validation_classification_report"
    
    # Save validation human-friendly text report
    val_txt_path = val_base_name.with_suffix('.txt')
    with open(val_txt_path, 'w', encoding='utf-8') as f:
        f.write(f"# Validation classification report\n# seed: {seed}\n\n")
        f.write(val_report_str)

    # Save validation JSON (structured) report
    val_json_path = val_base_name.with_suffix('.json')
    with open(val_json_path, 'w', encoding='utf-8') as f:
        json.dump({
            'seed': seed,
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
        f.write(f"# Test classification report\n# seed: {seed}\n\n")
        f.write(report_str)

    # Save JSON (structured) report
    json_path = test_base_name.with_suffix('.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump({
            'seed': seed,
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
    peft_model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"Saved LoRA adapter + tokenizer to: {final_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune Llama model with LoRA for text classification")
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
    
    args = parser.parse_args()
    
    print(f"Starting fine-tuning with:")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  LoRA rank (r): {args.r_lora}")
    print(f"  Random seed: {args.seed}")
    print("-" * 50)

    main(learning_rate=args.learning_rate, r_lora=args.r_lora, seed=args.seed)
