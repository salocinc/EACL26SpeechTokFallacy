#!/usr/bin/env python3
"""
Script to calculate the average F1-score for class 1 (Fallacy class) from validation classification reports
across multiple seeds for LoRA fine-tuning experiments on the AFD (Argument Fallacy Detection) task.

This script assumes that results are stored in directories with the format:
/data/ncalbucu/{prefix}_{r_lora}_{learning_rate}_{seed}/

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


def find_result_directories_for_combination(base_dir: str, prefix: str, r_lora: int, learning_rate: float, seeds: List[int]) -> List[str]:
    """
    Find all directories for a specific combination of parameters with the given seeds.
    
    Args:
        base_dir: Base directory path
        prefix: Directory prefix (e.g., 'afd_text_only' or audio config name)
        r_lora: LoRA rank parameter
        learning_rate: Learning rate value
        seeds: List of seed values to look for
    
    Returns:
        List of directory paths that exist for this combination
    """
    directories = []
    
    for seed in seeds:
        dir_path = f"{base_dir}/{prefix}_{r_lora}_{learning_rate}_{seed}"
        if os.path.isdir(dir_path):
            directories.append(dir_path)
    
    return sorted(directories)


def extract_f1_class1_from_json(json_path: str) -> float:
    """
    Extract the F1-score for class 1 (Fallacy class) from a classification report JSON file.
    
    Args:
        json_path: Path to the validation_classification_report.json file
    
    Returns:
        F1-score for class 1 (Fallacy class) as float
    """
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # The F1-score for class 1 (Fallacy) should be in data['report']['Fallacy']['f1-score']
        # or data['report']['1']['f1-score'] depending on how it's stored
        report = data['report']
        
        # Try different possible keys for class 1 (Fallacy)
        if 'Fallacy' in report:
            f1_class1 = report['Fallacy']['f1-score']
        elif '1' in report:
            f1_class1 = report['1']['f1-score']
        elif 1 in report:
            f1_class1 = report[1]['f1-score']
        else:
            # If neither key exists, look through all keys to find the fallacy class
            print(f"Available keys in report: {list(report.keys())}")
            raise KeyError("Could not find Fallacy class (class 1) in classification report")
        
        return float(f1_class1)
    
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
        description="Calculate average F1-score for class 1 (Fallacy) across multiple seeds for all parameter combinations on AFD task"
    )
    parser.add_argument(
        "--prefix", 
        type=str, 
        default="afd_text_only",
        help="Directory prefix (default: afd_text_only). Use audio config name for multimodal experiments"
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
    LEARNING_RATES = [5e-4, 1e-3]
    R_LORA_VALUES = [32, 64]
    SEEDS = [42, 100, 6, 10, 420]
    
    print("Parameter combinations to analyze:")
    print(f"  Learning rates: {LEARNING_RATES}")
    print(f"  LoRA ranks: {R_LORA_VALUES}")
    print(f"  Seeds: {SEEDS}")
    print(f"  Prefix: {args.prefix}")
    print(f"  Split: {args.split}")
    print(f"  Metric: F1-score for class 1 (Fallacy)")
    print("="*80)
    
    # Store all results for summary
    all_results = []
    
    # Iterate through all combinations of learning_rate and r_lora
    for learning_rate, r_lora in itertools.product(LEARNING_RATES, R_LORA_VALUES):
        print(f"\nAnalyzing: lr={learning_rate}, r_lora={r_lora}")
        print("-" * 50)
        
        # Find directories for this combination
        result_dirs = find_result_directories_for_combination(
            args.base_dir, args.prefix, r_lora, learning_rate, SEEDS
        )
        
        if not result_dirs:
            print(f"No directories found for lr={learning_rate}, r_lora={r_lora}")
            continue
        
        print(f"Found {len(result_dirs)} directories:")
        for dir_path in result_dirs:
            print(f"  {dir_path}")
        
        # Extract F1-scores for class 1 from each directory
        f1_class1_scores = []
        valid_dirs = []
        
        for dir_path in result_dirs:
            json_path = os.path.join(dir_path, f"{args.split}_classification_report.json")

            if not os.path.exists(json_path):
                print(f"Warning: {json_path} not found")
                continue
            
            f1_class1 = extract_f1_class1_from_json(json_path)
            
            if f1_class1 is not None:
                f1_class1_scores.append(f1_class1)
                valid_dirs.append(dir_path)
                seed = os.path.basename(dir_path).split('_')[-1]
                print(f"  Seed {seed}: F1 class 1 (Fallacy) = {f1_class1:.4f}")
            else:
                print(f"Warning: Could not extract F1 class 1 from {json_path}")
        
        if not f1_class1_scores:
            print(f"No valid F1 class 1 scores found for lr={learning_rate}, r_lora={r_lora}")
            continue
        
        # Calculate statistics
        stats = calculate_statistics(f1_class1_scores)
        
        print(f"\nResults for lr={learning_rate}, r_lora={r_lora}:")
        print(f"  Valid results: {stats['count']}/{len(SEEDS)}")
        print(f"  Mean F1 class 1: {stats['mean']:.4f}")
        print(f"  Std Dev:         {stats['std_dev']:.4f}")
        print(f"  Min:             {stats['min']:.4f}")
        print(f"  Max:             {stats['max']:.4f}")
        
        # Store results for this combination
        combination_result = {
            'learning_rate': learning_rate,
            'r_lora': r_lora,
            'f1_class1_scores': f1_class1_scores,
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
    
    # Sort results by mean F1-score for class 1 (descending)
    all_results.sort(key=lambda x: x['statistics']['mean'], reverse=True)
    
    print(f"{'Rank':<4} {'LR':<8} {'LoRA_r':<8} {'Mean_F1_C1':<12} {'Std_Dev':<10} {'Count':<6}")
    print("-" * 65)
    
    for i, result in enumerate(all_results, 1):
        lr = result['learning_rate']
        r_lora = result['r_lora']
        mean_f1 = result['statistics']['mean']
        std_dev = result['statistics']['std_dev']
        count = result['statistics']['count']
        
        print(f"{i:<4} {lr:<8} {r_lora:<8} {mean_f1:<12.4f} {std_dev:<10.4f} {count:<6}")
    
    # Save comprehensive results
    summary_path = f"{args.base_dir}/{args.prefix}_{args.split}_f1_class1_all_combinations_summary.json"
    summary_data = {
        'configuration': {
            'prefix': args.prefix,
            'learning_rates': LEARNING_RATES,
            'r_lora_values': R_LORA_VALUES,
            'seeds': SEEDS,
            'metric': 'f1_score_class_1_fallacy'
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
        print(f"  Mean F1 class 1: {best['statistics']['mean']:.4f} ± {best['statistics']['std_dev']:.4f}")
        print(f"  Based on {best['statistics']['count']} seeds")


if __name__ == "__main__":
    main()