#!/usr/bin/env python3
"""
Script to analyze inference results across multiple seeds for Argument Fallacy Detection (AFD).
Performs:
1. Calculate average and standard deviation of class 1 (Fallacy) f1-score across seeds
2. Probability averaging bagging (ensemble by averaging probabilities)
3. Majority voting bagging (ensemble by voting with tie-breaking)

Usage examples:
  # For AFD results:
  python analyze_multi_seed_results_afd.py --base_path afd_config_8layers_bow_True_c_0.0025_64_0.001 --results_dir inference_results_afd
  
  # With custom seeds:
  python analyze_multi_seed_results_afd.py --base_path your_prefix --results_dir your_results_dir --seeds 1 2 3 4 5
"""

import os
import json
import numpy as np
import argparse
from pathlib import Path
from typing import List, Tuple, Dict, Any
from sklearn.metrics import classification_report
import pandas as pd

def load_seed_results(base_path: str, seeds: List[str] = None, results_dir: str = "inference_results_afd") -> Dict[str, Dict]:
    """
    Load classification results from multiple seed folders.
    
    Args:
        base_path: Base path without seed suffix (e.g., "afd_config_8layers_bow_True_c_0.0025_64_0.001")
        seeds: List of seed suffixes. If None, uses default [10, 100, 42, 420, 6]
        results_dir: Directory containing the results (e.g., "inference_results_afd")
    
    Returns:
        Dictionary mapping seed to results data
    """
    if seeds is None:
        seeds = ["10", "100", "42", "420", "6"]
    
    results = {}
    inference_results_dir = Path(results_dir)
    
    for seed in seeds:
        seed_dir = inference_results_dir / f"{base_path}_{seed}"
        if not seed_dir.exists():
            print(f"Warning: Results directory not found: {seed_dir}")
            continue
            
        # Load classification report JSON
        json_path = seed_dir / "classification_report.json"
        if not json_path.exists():
            print(f"Warning: Classification report not found: {json_path}")
            continue
            
        with open(json_path, 'r') as f:
            report_data = json.load(f)
        
        # Load probabilities and predictions
        prob_path = seed_dir / "probabilities.npy"
        pred_path = seed_dir / "predictions.npy"
        labels_path = seed_dir / "true_labels.npy"
        
        if prob_path.exists() and pred_path.exists() and labels_path.exists():
            probabilities = np.load(prob_path)
            predictions = np.load(pred_path)
            true_labels = np.load(labels_path)
        else:
            print(f"Warning: Missing .npy files in {seed_dir}")
            probabilities = None
            predictions = None
            true_labels = None
        
        results[seed] = {
            'report_data': report_data,
            'probabilities': probabilities,
            'predictions': predictions,
            'true_labels': true_labels
        }
        
        print(f"Loaded results for seed {seed}")
    
    return results

def calculate_f1_statistics(results: Dict[str, Dict]) -> Tuple[float, float, List[float]]:
    """
    Calculate average and standard deviation of class 1 (Fallacy) f1-scores across seeds.
    
    Args:
        results: Dictionary mapping seed to results data
    
    Returns:
        Tuple of (mean_f1, std_f1, all_f1_scores)
    """
    f1_scores = []
    
    for seed, data in results.items():
        report = data['report_data']['report']
        
        # Try to get the F1-score for class 1 (Fallacy)
        # The key might be "Fallacy" or "1" depending on how it was saved
        class_1_f1 = None
        if 'Fallacy' in report:
            class_1_f1 = report['Fallacy'].get('f1-score', None)
        elif '1' in report:
            class_1_f1 = report['1'].get('f1-score', None)
        
        if class_1_f1 is not None:
            f1_scores.append(class_1_f1)
            print(f"Seed {seed}: class 1 (Fallacy) f1-score = {class_1_f1:.4f}")
        else:
            print(f"Warning: No class 1 (Fallacy) f1-score found for seed {seed}")
            print(f"  Available keys in report: {list(report.keys())}")
    
    if not f1_scores:
        raise ValueError("No f1-scores found for class 1 (Fallacy) in any seed results")
    
    mean_f1 = np.mean(f1_scores)
    std_f1 = np.std(f1_scores, ddof=1) if len(f1_scores) > 1 else 0.0
    
    return mean_f1, std_f1, f1_scores

def probability_averaging_ensemble(results: Dict[str, Dict]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Perform ensemble prediction by averaging probabilities across seeds.
    
    Args:
        results: Dictionary mapping seed to results data
    
    Returns:
        Tuple of (averaged_probabilities, ensemble_predictions, true_labels)
    """
    # Collect probabilities from all seeds
    all_probabilities = []
    true_labels = None
    
    for seed, data in results.items():
        if data['probabilities'] is not None:
            all_probabilities.append(data['probabilities'])
            if true_labels is None:
                true_labels = data['true_labels']
    
    if not all_probabilities:
        raise ValueError("No probability data found in seed results")
    
    # Verify all have same shape
    shapes = [probs.shape for probs in all_probabilities]
    if not all(shape == shapes[0] for shape in shapes):
        raise ValueError(f"Probability shapes don't match: {shapes}")
    
    # Average probabilities across seeds
    averaged_probabilities = np.mean(all_probabilities, axis=0)
    
    # Make predictions from averaged probabilities
    ensemble_predictions = np.argmax(averaged_probabilities, axis=1)
    
    print(f"Probability averaging ensemble: {len(all_probabilities)} seeds combined")
    print(f"Shape: {averaged_probabilities.shape}")
    
    return averaged_probabilities, ensemble_predictions, true_labels

def majority_voting_ensemble(results: Dict[str, Dict]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform ensemble prediction by majority voting with tie-breaking by probability sum.
    
    Args:
        results: Dictionary mapping seed to results data
    
    Returns:
        Tuple of (ensemble_predictions, true_labels)
    """
    # Collect predictions and probabilities from all seeds
    all_predictions = []
    all_probabilities = []
    true_labels = None
    
    for seed, data in results.items():
        if data['predictions'] is not None and data['probabilities'] is not None:
            all_predictions.append(data['predictions'])
            all_probabilities.append(data['probabilities'])
            if true_labels is None:
                true_labels = data['true_labels']
    
    if not all_predictions:
        raise ValueError("No prediction data found in seed results")
    
    all_predictions = np.array(all_predictions)  # Shape: (n_seeds, n_samples)
    all_probabilities = np.array(all_probabilities)  # Shape: (n_seeds, n_samples, n_classes)
    n_seeds, n_samples = all_predictions.shape
    n_classes = all_probabilities.shape[2]

    ensemble_predictions = np.zeros(n_samples, dtype=int)

    for i in range(n_samples):
        sample_predictions = all_predictions[:, i]

        # Count votes for each class
        vote_counts = np.bincount(sample_predictions, minlength=n_classes)

        max_votes = np.max(vote_counts)
        tied_classes = np.where(vote_counts == max_votes)[0]

        if len(tied_classes) == 1:
            ensemble_predictions[i] = tied_classes[0]
            continue

        # Tie: break by sum of probabilities for tied classes
        tied_prob_sums = []
        for class_idx in tied_classes:
            prob_sum = np.sum(all_probabilities[:, i, class_idx])
            tied_prob_sums.append(prob_sum)

        best_tied_idx = np.argmax(tied_prob_sums)
        ensemble_predictions[i] = tied_classes[best_tied_idx]

    print(f"Majority voting ensemble: {n_seeds} seeds combined")
    print(f"Samples processed: {n_samples}")

    return ensemble_predictions, true_labels

def save_classification_report(predictions: np.ndarray, true_labels: np.ndarray, 
                             output_dir: Path, method_name: str, 
                             fallacy_classes: List[str] = None):
    """
    Generate and save classification report for ensemble predictions.
    
    Args:
        predictions: Predicted labels
        true_labels: True labels
        output_dir: Directory to save results
        method_name: Name of the ensemble method (for file naming)
        fallacy_classes: List of class names
    """
    if fallacy_classes is None:
        fallacy_classes = ['No Fallacy', 'Fallacy']

    predictions = np.asarray(predictions)
    true_labels = np.asarray(true_labels)
    if predictions.shape[0] != true_labels.shape[0]:
        raise ValueError("Predictions and true_labels must have the same length")

    unique_test_labels = sorted(int(x) for x in np.unique(true_labels))
    present_fallacy_classes = [fallacy_classes[i] for i in unique_test_labels]

    report_str = classification_report(
        true_labels, predictions, digits=4, zero_division=0,
        target_names=present_fallacy_classes, labels=unique_test_labels
    )
    report_dict = classification_report(
        true_labels, predictions, output_dict=True, digits=4, zero_division=0,
        target_names=present_fallacy_classes, labels=unique_test_labels
    )

    txt_path = output_dir / f"{method_name}_classification_report.txt"
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(f"# {method_name.replace('_', ' ').title()} Classification Report (AFD)\n\n")
        f.write(f"## Dataset Information\n")
        f.write(f"- Total samples: {len(true_labels)}\n")
        f.write(f"- Evaluated samples: {len(true_labels)}\n")
        f.write(f"- Test set contains {len(unique_test_labels)} classes: {present_fallacy_classes}\n")
        f.write(f"- Labels present in test set: {unique_test_labels}\n\n")
        f.write(f"## Classification Results\n")
        f.write(report_str)

    json_path = output_dir / f"{method_name}_classification_report.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump({
            'method': method_name,
            'unique_test_labels': [int(x) for x in unique_test_labels],
            'present_fallacy_classes': present_fallacy_classes,
            'report': report_dict,
        }, f, indent=2)
    
    print(f"Saved {method_name} classification report to {txt_path} and {json_path}")
    
    return report_dict

def main():
    parser = argparse.ArgumentParser(description="Analyze AFD inference results across multiple seeds")
    parser.add_argument("--base_path", type=str, 
                        default="afd_config_8layers_bow_True_c_0.0025_64_0.001",
                        help="Base path without seed suffix")
    parser.add_argument("--seeds", nargs='+', type=str, 
                        default=["10", "100", "42", "420", "6"],
                        help="List of seed suffixes")
    parser.add_argument("--results_dir", type=str, default="inference_results_afd",
                        help="Directory containing the results (e.g., 'inference_results_afd')")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (if None, uses {results_dir}/{base_path}_analysis)")
    
    args = parser.parse_args()
    
    # Set up output directory
    if args.output_dir is None:
        output_dir = Path(args.results_dir) / f"{args.base_path}_analysis"
    else:
        output_dir = Path(args.output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Results directory: {args.results_dir}")
    print(f"Analyzing results for base path: {args.base_path}")
    print(f"Using seeds: {args.seeds}")
    print(f"Output directory: {output_dir}")
    
    # Load results from all seeds
    results = load_seed_results(args.base_path, args.seeds, args.results_dir)
    
    if not results:
        raise ValueError("No valid seed results found")
    
    print(f"\nLoaded results from {len(results)} seeds: {list(results.keys())}")
    
    # 1. Calculate F1 statistics for class 1 (Fallacy)
    print("\n" + "="*50)
    print("1. CALCULATING CLASS 1 (FALLACY) F1 STATISTICS")
    print("="*50)
    
    mean_f1, std_f1, f1_scores = calculate_f1_statistics(results)
    
    # Save F1 statistics
    stats_file = output_dir / "f1_statistics_class_1.txt"
    with open(stats_file, 'w') as f:
        f.write("Class 1 (Fallacy) F1-Score Statistics Across Seeds\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Base path: {args.base_path}\n")
        f.write(f"Seeds analyzed: {args.seeds}\n")
        f.write(f"Number of seeds: {len(f1_scores)}\n\n")
        
        f.write("Individual seed F1-scores (class 1 - Fallacy):\n")
        for i, (seed, f1) in enumerate(zip(results.keys(), f1_scores)):
            f.write(f"  Seed {seed}: {f1:.4f}\n")
        
        f.write(f"\nStatistics:\n")
        f.write(f"  Mean F1-score (class 1): {mean_f1:.4f}\n")
        f.write(f"  Standard deviation: {std_f1:.4f}\n")
        f.write(f"  Min F1-score: {min(f1_scores):.4f}\n")
        f.write(f"  Max F1-score: {max(f1_scores):.4f}\n")
    
    print(f"Class 1 (Fallacy) F1 Statistics:")
    print(f"  Mean: {mean_f1:.4f}")
    print(f"  Std:  {std_f1:.4f}")
    print(f"  Saved to: {stats_file}")
    
    # 2. Probability averaging ensemble
    print("\n" + "="*50)
    print("2. PROBABILITY AVERAGING ENSEMBLE")
    print("="*50)
    
    avg_probs, avg_predictions, true_labels = probability_averaging_ensemble(results)
    
    # Save probability averaging results
    np.save(output_dir / "probability_averaging_predictions.npy", avg_predictions)
    np.save(output_dir / "probability_averaging_probabilities.npy", avg_probs)
    
    avg_report_dict = save_classification_report(
        avg_predictions, true_labels, output_dir, "probability_averaging"
    )
    
    # 3. Majority voting ensemble
    print("\n" + "="*50)
    print("3. MAJORITY VOTING ENSEMBLE")
    print("="*50)
    
    vote_predictions, true_labels = majority_voting_ensemble(results)
    
    # Save majority voting results
    np.save(output_dir / "majority_voting_predictions.npy", vote_predictions)
    
    vote_report_dict = save_classification_report(
        vote_predictions, true_labels, output_dir, "majority_voting"
    )
    
    # Summary
    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    print(f"Results saved to: {output_dir}")
    print(f"Class 1 (Fallacy) F1 Statistics: Mean = {mean_f1:.4f}, Std = {std_f1:.4f}")
    
    # Extract class 1 F1-scores from ensemble reports
    avg_class_1_f1 = None
    vote_class_1_f1 = None
    
    if 'Fallacy' in avg_report_dict:
        avg_class_1_f1 = avg_report_dict['Fallacy'].get('f1-score', 0)
    elif '1' in avg_report_dict:
        avg_class_1_f1 = avg_report_dict['1'].get('f1-score', 0)
    
    if 'Fallacy' in vote_report_dict:
        vote_class_1_f1 = vote_report_dict['Fallacy'].get('f1-score', 0)
    elif '1' in vote_report_dict:
        vote_class_1_f1 = vote_report_dict['1'].get('f1-score', 0)
    
    if avg_class_1_f1 is not None:
        print(f"Probability Averaging Class 1 F1: {avg_class_1_f1:.4f}")
    if vote_class_1_f1 is not None:
        print(f"Majority Voting Class 1 F1: {vote_class_1_f1:.4f}")
    
    # Save summary
    summary_file = output_dir / "summary.txt"
    with open(summary_file, 'w') as f:
        f.write("Multi-Seed Analysis Summary (AFD)\n")
        f.write("=" * 35 + "\n\n")
        f.write(f"Base path: {args.base_path}\n")
        f.write(f"Seeds: {args.seeds}\n")
        f.write(f"Number of seeds: {len(results)}\n\n")
        f.write(f"Individual Seed Class 1 (Fallacy) F1 Statistics:\n")
        f.write(f"  Mean F1-score: {mean_f1:.4f}\n")
        f.write(f"  Standard deviation: {std_f1:.4f}\n\n")
        f.write(f"Ensemble Results (Class 1 - Fallacy):\n")
        if avg_class_1_f1 is not None:
            f.write(f"  Probability Averaging F1: {avg_class_1_f1:.4f}\n")
        if vote_class_1_f1 is not None:
            f.write(f"  Majority Voting F1: {vote_class_1_f1:.4f}\n")
    
    print(f"Summary saved to: {summary_file}")

if __name__ == "__main__":
    main()
