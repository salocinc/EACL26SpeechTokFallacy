import os
import torch
import random
import numpy as np
import argparse
from torch.utils.data import Dataset
from transformers import (
    Trainer,
    TrainerCallback,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback,
)
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))
from sklearn.model_selection import train_test_split
from utilities.audio_token_manager import AudioTokenManager


class MultimodalTextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=256):
        self.tokenizer = tokenizer
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors='pt'
        )

        # Sanity check: log samples that do not contain new tokens
        new_token_ids = [tokenizer.convert_tokens_to_ids(t) for t in tokenizer.additional_special_tokens]
        for i in range(min(3, len(self.encodings['input_ids']))):
            input_ids = self.encodings['input_ids'][i]
            if not any(id.item() in new_token_ids for id in input_ids):
                print(f"[SANITY CHECK] No new tokens found in sample {i}: {tokenizer.convert_ids_to_tokens(input_ids.tolist())}")
            else:
                decoded = tokenizer.decode(input_ids, skip_special_tokens=False)
                #print(f"[SANITY CHECK] Sample {i} decoded: {decoded}")

    def __len__(self):
        return len(self.encodings['input_ids'])

    def __getitem__(self, idx):
        return {
            'input_ids': self.encodings['input_ids'][idx],
            'attention_mask': self.encodings['attention_mask'][idx],
            'labels': self.encodings['input_ids'][idx].clone()
        }


def freeze_model_except_embeddings_with_hook(model, original_vocab_size):
    for param in model.parameters():
        param.requires_grad = False

    embedding_layer = model.get_input_embeddings()
    total_vocab_size = embedding_layer.weight.size(0)

    if total_vocab_size <= original_vocab_size:
        raise ValueError("No new tokens detected.")

    # Allow gradient updates to the embedding layer
    embedding_layer.weight.requires_grad = True

    # Define hook to mask gradients
    new_token_ids = list(range(original_vocab_size, total_vocab_size))

    def mask_gradient(grad):
        if grad is None:
            return None
        mask = torch.zeros_like(grad)
        mask[new_token_ids] = 1
        masked_grad = grad * mask
        return masked_grad

    embedding_layer.weight.register_hook(mask_gradient)

    print(f"Trainable new token IDs: {new_token_ids}")
    print(f"Embedding layer shape: {embedding_layer.weight.shape}")


class EmbeddingTrainingCallback(TrainerCallback):
    def __init__(self, original_vocab_size):
        self.original_vocab_size = original_vocab_size
        self.embedding_norms = []

    def on_log(self, args, state, control, logs=None, model=None, **kwargs):
        if model is not None:
            embedding_layer = model.get_input_embeddings()
            with torch.no_grad():
                mask = torch.isnan(embedding_layer.weight)
                print("How many NaNs?", mask.sum())
                new_weights = embedding_layer.weight[self.original_vocab_size:]
                norm = new_weights.norm(dim=1).mean().item()
                self.embedding_norms.append(norm)
                print(f"Average new embedding norm: {norm:.6f}")


def prepare_training_data(texts, test_size=0.1, seed=42):
    return train_test_split(texts, test_size=test_size, random_state=seed)


def train_audio_embeddings(
    base_model_path='../utilities/base_llama_model',
    audio_config_name=None,
    output_dir=f"../utilities/audio_embeddings_output",
    learning_rate=3e-4,
    batch_size=8,  # Per-device batch size
    epochs=5,
    gradient_accumulation_steps=4,  # Adjusted to achieve effective batch size of 32
    seed=42
):
    if audio_config_name is None:
        raise ValueError("audio_config_name is required")

    # Enable synchronous CUDA errors for debugging
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    
    # Disable tokenizers parallelism warning
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    # Set deterministic behavior
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Training with seed: {seed}")

    # Clear GPU cache first
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    manager = AudioTokenManager()
    if not os.path.exists(base_model_path):
        manager.save_base_model_once(base_model_path)

    model, tokenizer, metadata = manager.load_model_with_audio_tokens(
        base_model_path, audio_config_name, embeddings_path=None
    )
    model.to(device)

    original_vocab_size = metadata['original_vocab_size']
    freeze_model_except_embeddings_with_hook(model, original_vocab_size)

    training_data_path = f"../training_data/{audio_config_name}_pretrain_texts.txt"
    if not os.path.exists(training_data_path):
        raise FileNotFoundError(f"No training data at: {training_data_path}")

    with open(training_data_path, 'r', encoding='utf-8') as f:
        texts = [line.strip() for line in f if line.strip()]

    print(f"Loaded {len(texts)} training samples")
    train_texts, val_texts = prepare_training_data(texts, seed=seed)

    train_dataset = MultimodalTextDataset(train_texts, tokenizer)
    val_dataset = MultimodalTextDataset(val_texts, tokenizer)

    # Calculate effective batch size and log training parameters
    effective_batch_size = batch_size * gradient_accumulation_steps
    steps_per_epoch = len(train_texts) // effective_batch_size
    total_updates = (steps_per_epoch * epochs)
    warmup_steps = max(1, int(total_updates * 0.05))  # 5% warmup
    logging_steps = max(1, steps_per_epoch // 4)  # Log 4 times per epoch
    eval_steps = max(1, steps_per_epoch // 2)  # Evaluate twice per epoch

    print(f"Training parameters for audio embeddings:")
    print(f"  Samples: {len(train_texts):,}")
    print(f"  Per-device batch: {batch_size}, Grad accum: {gradient_accumulation_steps}")
    print(f"  Effective batch size: {effective_batch_size}")
    print(f"  Epochs: {epochs}")
    print(f"  Steps/epoch: {steps_per_epoch}, Total updates: {total_updates:,}")
    print(f"  Warmup: {warmup_steps}, Log every: {logging_steps}, Eval every: {eval_steps} steps")

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_steps=warmup_steps,
        weight_decay=0.0,
        lr_scheduler_type='linear',
        logging_dir='./logs',
        logging_steps=logging_steps,
        eval_strategy='steps',
        eval_steps=eval_steps,
        save_strategy='steps',
        save_steps=eval_steps * 2,  # Save less frequently
        save_total_limit=2,
        load_best_model_at_end=True,
        max_grad_norm=1.0,
        remove_unused_columns=False,
        fp16=True,  # Enable fp16 for speed and memory efficiency
        optim='adamw_torch',
        dataloader_num_workers=4,  # Speed up data loading
        seed=seed,
        report_to=None,  # Disable wandb/tensorboard for speed
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        callbacks=[
            EmbeddingTrainingCallback(original_vocab_size),
            EarlyStoppingCallback(early_stopping_patience=3)]  # Less patience for faster training
    )

    print("Starting training...")
    trainer.train()

    # Use AudioTokenManager to save trained embeddings
    # Extract final losses properly from log history
    final_train_loss = None
    final_eval_loss = None
    
    if trainer.state.log_history:
        # Find the last entry with train_loss
        for log_entry in reversed(trainer.state.log_history):
            if final_train_loss is None and 'train_loss' in log_entry:
                final_train_loss = log_entry['train_loss']
            if final_eval_loss is None and 'eval_loss' in log_entry:
                final_eval_loss = log_entry['eval_loss']
            if final_train_loss is not None and final_eval_loss is not None:
                break
    
    training_metadata = {
        'learning_rate': learning_rate,
        'batch_size': batch_size,
        'epochs': epochs,
        'gradient_accumulation_steps': gradient_accumulation_steps,
        'final_train_loss': final_train_loss,
        'final_eval_loss': final_eval_loss,
    }
    
    output_path = manager.save_audio_embeddings_only(
        model=model,
        original_vocab_size=original_vocab_size,
        config_name=audio_config_name,
        training_metadata=training_metadata
    )
    
    print(f"Saved trained embeddings to {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Train audio embeddings for Llama model")
    parser.add_argument(
        "--learning_rate", 
        type=float, 
        default=3e-4, 
        help="Learning rate for training (default: 3e-4)"
    )
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=8, 
        help="Per-device batch size (default: 8)"
    )
    parser.add_argument(
        "--gradient_accumulation_steps", 
        type=int, 
        default=4, 
        help="Gradient accumulation steps (default: 4, gives effective batch size of 32)"
    )
    parser.add_argument(
        "--epochs", 
        type=int, 
        default=5, 
        help="Number of training epochs (default: 5)"
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=42, 
        help="Random seed for initialization (default: 42)"
    )
    parser.add_argument(
        "--config_name", 
        type=str, 
        default="config_7layers_bow_False_c_0.02", 
        help="Audio config name (default: config_7layers_bow_False_c_0.02)"
    )
    
    args = parser.parse_args()
    
    effective_batch_size = args.batch_size * args.gradient_accumulation_steps
    print(f"Starting audio embeddings training with:")
    print(f"  Task: Training new audio token embeddings for Llama")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Gradient accumulation steps: {args.gradient_accumulation_steps}")
    print(f"  Effective batch size: {effective_batch_size}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Random seed: {args.seed}")
    print(f"  Audio config: {args.config_name}")
    print("-" * 50)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    trained_path = train_audio_embeddings(
        base_model_path='../utilities/base_llama_model',
        audio_config_name=args.config_name,
        output_dir=f"./audio_embeddings_output/{args.config_name}",
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        epochs=args.epochs,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        seed=args.seed
    )
    print(f"Embeddings saved to: {trained_path}")


if __name__ == '__main__':
    main()