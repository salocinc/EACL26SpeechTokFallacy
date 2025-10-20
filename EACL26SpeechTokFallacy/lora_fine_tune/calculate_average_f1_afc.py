#!/usr/bin/env python3
"""
Script to calculate the average macro F1-score from validation classification reports
across multiple seeds for LoRA fine-tuning experiments.

This script assumes that results are stored in directories with the format:
/data/ncalbucu/{AUDIO_CONFIG_NAME}_{r_lora}_{learning_rate}_{seed}/

Each directory should contain a validation_classification_report.json file.
"""

import os
import json
import glob
import argparse
from pathlib import Path
import statistics
from typing import List, Dict, Tuple
import itertools


def find_result_directories_for_combination(base_dir: str, audio_config: str, r_lora: int, learning_rate: float, seeds: List[int]) -> List[str]:
    """
    Find all directories for a specific combination of parameters with the given seeds.
    
    Args:
        base_dir: Base directory path
        audio_config: Audio configuration name
        r_lora: LoRA rank parameter
        learning_rate: Learning rate value
        seeds: List of seed values to look for
    
    Returns:
        List of directory paths that exist for this combination
    """
    directories = []
    
    for seed in seeds:
        dir_path = f"{base_dir}/{audio_config}_{r_lora}_{learning_rate}_{seed}"
        if os.path.isdir(dir_path):
            directories.append(dir_path)
    
    return sorted(directories)


def extract_macro_f1_from_json(json_path: str) -> float:
    """
    Extract the macro average F1-score from a classification report JSON file.
    
    Args:
        json_path: Path to the validation_classification_report.json file
    
    Returns:
        Macro average F1-score as float
    """
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # The macro avg f1-score should be in data['report']['macro avg']['f1-score']
        macro_f1 = data['report']['macro avg']['f1-score']
        return float(macro_f1)
    
    except (FileNotFoundError, KeyError, ValueError, TypeError) as e:
        print(f"Error reading {json_path}: {e}")
        return None


def calculate_statistics(values: List[float]) -> Dict[str, float]:
    """
    Calculate mean, standard deviation, min, and max of a list of values.
    
    Args:
        values: List of numeric values
    
    Returns:
        Dictionary with statistical measures
    """
    if not values:
        return {}
    
    return {
        'mean': statistics.mean(values),
        'std_dev': statistics.stdev(values) if len(values) > 1 else 0.0,
        'min': min(values),
        'max': max(values),
        'count': len(values)
    }


def main():
    parser = argparse.ArgumentParser(
        description="Calculate average macro F1-score across multiple seeds for all parameter combinations"
    )
    parser.add_argument(
        "--audio_config", 
        type=str, 
        default="config_8layers_bow_True_c_0.03",
        help="Audio configuration name (default: config_8layers_bow_True_c_0.03)"
    )
    parser.add_argument(
        "--base_dir", 
        type=str, 
        default="/data/ncalbucu",
        help="Base directory where results are stored (default: /data/ncalbucu)"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="validation",
        help="Data split to use (default: validation)"
    )
    
    args = parser.parse_args()
    
    # Define parameter combinations
    LEARNING_RATES = [1e-4, 5e-4, 1e-3, 2e-3]
    R_LORA_VALUES = [32, 64]
    SEEDS = [42, 100, 6, 10, 420]
    
    print("Parameter combinations to analyze:")
    print(f"  Learning rates: {LEARNING_RATES}")
    print(f"  LoRA ranks: {R_LORA_VALUES}")
    print(f"  Seeds: {SEEDS}")
    print(f"  Audio config: {args.audio_config}")
    print(f"  Split: {args.split}")
    print("="*80)
    
    # Store all results for summary
    all_results = []
    
    # Iterate through all combinations of learning_rate and r_lora
    for learning_rate, r_lora in itertools.product(LEARNING_RATES, R_LORA_VALUES):
        print(f"\nAnalyzing: lr={learning_rate}, r_lora={r_lora}")
        print("-" * 50)
        
        # Find directories for this combination
        result_dirs = find_result_directories_for_combination(
            args.base_dir, args.audio_config, r_lora, learning_rate, SEEDS
        )
        
        if not result_dirs:
            print(f"No directories found for lr={learning_rate}, r_lora={r_lora}")
            continue
        
        print(f"Found {len(result_dirs)} directories:")
        for dir_path in result_dirs:
            print(f"  {dir_path}")
        
        # Extract macro F1-scores from each directory
        macro_f1_scores = []
        valid_dirs = []
        
        for dir_path in result_dirs:
            json_path = os.path.join(dir_path, f"{args.split}_classification_report.json")

            if not os.path.exists(json_path):
                print(f"Warning: {json_path} not found")
                continue
            
            macro_f1 = extract_macro_f1_from_json(json_path)
            
            if macro_f1 is not None:
                macro_f1_scores.append(macro_f1)
                valid_dirs.append(dir_path)
                seed = os.path.basename(dir_path).split('_')[-1]
                print(f"  Seed {seed}: macro F1 = {macro_f1:.4f}")
            else:
                print(f"Warning: Could not extract macro F1 from {json_path}")
        
        if not macro_f1_scores:
            print(f"No valid macro F1-scores found for lr={learning_rate}, r_lora={r_lora}")
            continue
        
        # Calculate statistics
        stats = calculate_statistics(macro_f1_scores)
        
        print(f"\nResults for lr={learning_rate}, r_lora={r_lora}:")
        print(f"  Valid results: {stats['count']}/{len(SEEDS)}")
        print(f"  Mean macro F1: {stats['mean']:.4f}")
        print(f"  Std Dev:       {stats['std_dev']:.4f}")
        print(f"  Min:           {stats['min']:.4f}")
        print(f"  Max:           {stats['max']:.4f}")
        
        # Store results for this combination
        combination_result = {
            'learning_rate': learning_rate,
            'r_lora': r_lora,
            'macro_f1_scores': macro_f1_scores,
            'statistics': stats,
            'valid_directories': valid_dirs
        }
        all_results.append(combination_result)
    
    # Print summary of all combinations
    print("\n" + "="*80)
    print("SUMMARY OF ALL COMBINATIONS")
    print("="*80)
    
    if not all_results:
        print("No valid results found for any combination!")
        return
    
    # Sort results by mean F1-score (descending)
    all_results.sort(key=lambda x: x['statistics']['mean'], reverse=True)
    
    print(f"{'Rank':<4} {'LR':<8} {'LoRA_r':<8} {'Mean_F1':<10} {'Std_Dev':<10} {'Count':<6}")
    print("-" * 60)
    
    for i, result in enumerate(all_results, 1):
        lr = result['learning_rate']
        r_lora = result['r_lora']
        mean_f1 = result['statistics']['mean']
        std_dev = result['statistics']['std_dev']
        count = result['statistics']['count']
        
        print(f"{i:<4} {lr:<8} {r_lora:<8} {mean_f1:<10.4f} {std_dev:<10.4f} {count:<6}")
    
    # Save comprehensive results
    summary_path = f"{args.base_dir}/{args.audio_config}_{args.split}_all_combinations_summary.json"
    summary_data = {
        'configuration': {
            'audio_config': args.audio_config,
            'learning_rates': LEARNING_RATES,
            'r_lora_values': R_LORA_VALUES,
            'seeds': SEEDS
        },
        'results': all_results
    }
    
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary_data, f, indent=2)
    
    print(f"\nComprehensive summary saved to: {summary_path}")
    
    # Find and highlight the best combination
    if all_results:
        best = all_results[0]
        print(f"\nBest combination:")
        print(f"  Learning rate: {best['learning_rate']}")
        print(f"  LoRA rank: {best['r_lora']}")
        print(f"  Mean macro F1: {best['statistics']['mean']:.4f} ± {best['statistics']['std_dev']:.4f}")
        print(f"  Based on {best['statistics']['count']} seeds")


if __name__ == "__main__":
    main()
