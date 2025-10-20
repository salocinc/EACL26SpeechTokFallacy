import logging
from pathlib import Path
import sys
import argparse
import json
import csv
from datetime import datetime
import os

# Add the parent directory of 'mamkit' to sys.path
sys.path.append(str(Path(__file__).parent.parent.parent))

import lightning as L
import numpy as np
import torch as th
import pandas as pd
from lightning.pytorch import seed_everything
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from torch.utils.data import DataLoader
from torchmetrics.classification.f_beta import F1Score
from torchmetrics import MetricCollection
from mamkit.configs.base import ConfigKey
from mamkit.configs.text_audio import BowMFCCsConfig
from mamkit.data.collators import MultimodalCollator, AudioCollatorOutput, BoWTextCollator
from mamkit.data.datasets import MMUSEDFallacy, InputMode
from mamkit.data.processing import MultimodalProcessor, MFCCExtractor, BagOfWordsProcessor
from mamkit.models.text_audio import BoWMFCCs
from mamkit.utility.callbacks import PycharmProgressBar
from mamkit.utility.model import MAMKitLightingModel
from sklearn.metrics import classification_report


def save_classification_report(labels, preds, target_names, output_dir, prefix, seed):
    """Save classification report in multiple formats"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate reports
    report_str = classification_report(
        labels, preds, digits=4, zero_division=0, target_names=target_names
    )
    report_dict = classification_report(
        labels, preds, output_dict=True, digits=4, zero_division=0, target_names=target_names
    )
    
    base_name = output_dir / f"{prefix}_classification_report"
    
    # Save text report
    txt_path = base_name.with_suffix('.txt')
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(f"# {prefix.capitalize()} classification report\n# seed: {seed}\n\n")
        f.write(report_str)
    
    # Save JSON report
    json_path = base_name.with_suffix('.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump({
            'seed': seed,
            'report': report_dict
        }, f, indent=2)
    
    # Save CSV report
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
    
    print(f"Saved {prefix} classification report to:\n  {txt_path}\n  {json_path}\n  {csv_path}")
    return report_str


def main(seed=42, learning_rate=1e-3):
    logging.basicConfig(level=logging.WARNING)
    
    # Set up deterministic behavior
    seed_everything(seed=seed)

    save_path = Path(__file__).parent.parent.parent.resolve().joinpath('results', 'mmused-fallacy', 'afc', 'baseline_bow_mfccs',
                                                                f'lr_{learning_rate}_seed_{seed}')
    if not save_path.exists():
        save_path.mkdir(parents=True)

    base_data_path = Path(__file__).parent.parent.parent.resolve().joinpath('data')

    config = BowMFCCsConfig.from_config(key=ConfigKey(dataset='mmused-fallacy',
                                                         input_mode=InputMode.TEXT_AUDIO,
                                                         task_name='afc',
                                                         tags='anonymous'))
    
    mmused_fallacy_splits_dir = base_data_path.joinpath('train_test_val_mmused_fallacy')
    train_data = pd.read_pickle(mmused_fallacy_splits_dir.joinpath('train_data.pkl'))
    val_data = pd.read_pickle(mmused_fallacy_splits_dir.joinpath('val_data.pkl'))
    test_data = pd.read_pickle(mmused_fallacy_splits_dir.joinpath('test_data.pkl'))
    loader = MMUSEDFallacy(task_name='afc',
                        input_mode=InputMode.TEXT_AUDIO,
                        base_data_path=base_data_path)
    data = loader.build_info_from_splits(
        train_df=train_data,
        val_df=val_data,
        test_df=test_data
    )

    trainer_args = {
        'accelerator': 'auto',
        'devices': 1,
        'accumulate_grad_batches': 3,
        'max_epochs': 30,
        'enable_progress_bar': False,
    }

    # Disable tqdm progress bars for MFCC extraction
    os.environ['TQDM_DISABLE'] = '1'

    metrics = {}
    processor = MultimodalProcessor(text_processor=BagOfWordsProcessor(max_features=10000),
                                    audio_processor=MFCCExtractor(
                                        sampling_rate=config.sampling_rate,
                                        normalize=config.normalize,
                                        remove_energy=config.remove_energy,
                                        pooling_sizes=config.pooling_sizes,
                                        mfccs=config.mfccs
                                    )
                                )
    processor.fit(train_data=data.train)
    train_data1 = processor(data.train)
    val_data1 = processor(data.val)
    test_data1 = processor(data.test)

    # Re-enable tqdm
    os.environ['TQDM_DISABLE'] = '0'

    collator = MultimodalCollator(
        text_collator=BoWTextCollator(),
        audio_collator=AudioCollatorOutput(),
        label_collator=lambda labels: th.tensor(labels)
    )

    train_dataloader = DataLoader(train_data1,
                                    batch_size=config.batch_size,
                                    shuffle=True,
                                    collate_fn=collator)
    val_dataloader = DataLoader(val_data1,
                                batch_size=config.batch_size,
                                shuffle=False,
                                collate_fn=collator)
    test_dataloader = DataLoader(test_data1,
                                    batch_size=config.batch_size,
                                    shuffle=False,
                                    collate_fn=collator)

    model = BoWMFCCs(head=lambda: th.nn.Sequential(
                                th.nn.Linear(len(processor.text_processor.vocab)+config.audio_embedding_dim, 6)
                                ))
    model = MAMKitLightingModel(model=model,
                                loss_function=config.loss_function,
                                num_classes=config.num_classes,
                                optimizer_class=config.optimizer,
                                val_metrics=MetricCollection({
                                    'f1': F1Score(task='multiclass', num_classes=6),
                                    'f1_macro': F1Score(task='multiclass', num_classes=6, average='macro')
                                }),
                                test_metrics=MetricCollection({
                                    'f1': F1Score(task='multiclass', num_classes=6),
                                    'f1_macro': F1Score(task='multiclass', num_classes=6, average='macro')
                                }),
                                lr=learning_rate,
                                **config.optimizer_args)

    trainer = L.Trainer(**trainer_args,
                        callbacks=[EarlyStopping(monitor='val_f1_macro', mode='max', patience=5),
                                    ModelCheckpoint(monitor='val_f1_macro', mode='max')])
    trainer.fit(model,
                train_dataloaders=train_dataloader,
                val_dataloaders=val_dataloader)

    val_metrics = trainer.validate(ckpt_path='best', dataloaders=val_dataloader)[0]
    test_metrics = trainer.test(ckpt_path='best', dataloaders=test_dataloader)[0]

    target_names = ['Appeal to emotion','Appeal to authority',
                   'Ad hominem','False cause','Slippery slope','Slogans']

    # Validation classification report
    print("\n*** Validation set classification report ***")
    pred_dicts = trainer.predict(ckpt_path='best', dataloaders=val_dataloader)
    all_logits = th.cat([d["logits"] for d in pred_dicts], dim=0)
    all_labels = th.cat([d["labels"] for d in pred_dicts], dim=0)
    preds = th.argmax(all_logits, dim=1)
    
    val_report = save_classification_report(
        all_labels.cpu().numpy(),
        preds.cpu().numpy(),
        target_names,
        save_path,
        "validation",
        seed
    )
    print(val_report)

    # Test classification report
    print("\n*** Test set classification report ***")
    pred_dicts = trainer.predict(ckpt_path='best', dataloaders=test_dataloader)
    all_logits = th.cat([d["logits"] for d in pred_dicts], dim=0)
    all_labels = th.cat([d["labels"] for d in pred_dicts], dim=0)
    preds = th.argmax(all_logits, dim=1)

    test_report = save_classification_report(
        all_labels.cpu().numpy(),
        preds.cpu().numpy(),
        target_names,
        save_path,
        "test",
        seed
    )
    print(test_report)

    # Save training metrics
    metrics_file = save_path / "training_metrics.json"
    with open(metrics_file, 'w') as f:
        json.dump({
            'seed': seed,
            'learning_rate': learning_rate,
            'val_metrics': {k: float(v) if isinstance(v, th.Tensor) else v for k, v in val_metrics.items()},
            'test_metrics': {k: float(v) if isinstance(v, th.Tensor) else v for k, v in test_metrics.items()},
            'config': {
                'batch_size': config.batch_size,
                'max_epochs': trainer_args['max_epochs'],
                'sampling_rate': config.sampling_rate,
                'mfccs': config.mfccs,
                'text_vocab_size': len(processor.text_processor.vocab)
            }
        }, f, indent=2)
    
    print(f"Saved training metrics to: {metrics_file}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Baseline BoW+MFCCs model for fallacy classification")
    parser.add_argument(
        "--seed", 
        type=int, 
        default=42, 
        help="Random seed for initialization (default: 42)"
    )
    parser.add_argument(
        "--learning_rate", 
        type=float, 
        default=1e-3, 
        help="Learning rate for optimizer (default: 1e-3)"
    )
    
    args = parser.parse_args()
    
    print(f"Starting baseline BoW+MFCCs training with:")
    print(f"  Random seed: {args.seed}")
    print(f"  Learning rate: {args.learning_rate}")
    print("-" * 50)
    
    main(seed=args.seed, learning_rate=args.learning_rate)
