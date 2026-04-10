"""Visualization module for GDM prediction pipeline."""

import matplotlib.pyplot as plt
import numpy as np
import os


def plot_roc_curves(results_dict, cohort_name, output_dir="output"):
    """
    Plot ROC curves for all algorithms in a cohort.
    
    Args:
        results_dict: Dictionary with algorithm results
        cohort_name: Name of the cohort (e.g., "First-Trimester")
        output_dir: Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    for algorithm, data in results_dict.items():
        if "predictions" not in data:
            continue
        
        metrics = data["metrics"]
        predictions = data["predictions"]
        
        fpr = predictions["fpr"]
        tpr = predictions["tpr"]
        auroc = metrics["auroc"]
        
        ax.plot(fpr, tpr, label=f"{algorithm} (AUROC={auroc:.3f})", linewidth=2)
    
    # Diagonal line for random classifier
    ax.plot([0, 1], [0, 1], "k--", label="Random Classifier", linewidth=1)
    
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title(f"ROC Curves - {cohort_name}", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
    output_path = os.path.join(output_dir, f"roc_curves_{cohort_name.lower().replace(' ', '_')}.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved ROC curves to {output_path}")
    plt.close()


def plot_precision_recall_curves(results_dict, cohort_name, output_dir="output"):
    """
    Plot Precision-Recall curves for all algorithms in a cohort.
    
    Args:
        results_dict: Dictionary with algorithm results
        cohort_name: Name of the cohort
        output_dir: Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    for algorithm, data in results_dict.items():
        if "predictions" not in data:
            continue
        
        metrics = data["metrics"]
        predictions = data["predictions"]
        
        precision = predictions["precision"]
        recall = predictions["recall"]
        ap = metrics["ap"]
        
        ax.plot(recall, precision, label=f"{algorithm} (AP={ap:.3f})", linewidth=2)
    
    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title(f"Precision-Recall Curves - {cohort_name}", fontsize=14, fontweight="bold")
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
    output_path = os.path.join(output_dir, f"pr_curves_{cohort_name.lower().replace(' ', '_')}.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved PR curves to {output_path}")
    plt.close()


def plot_metrics_comparison(results_dict, cohort_name, output_dir="output"):
    """
    Plot comparison of metrics across algorithms.
    
    Args:
        results_dict: Dictionary with algorithm results
        cohort_name: Name of the cohort
        output_dir: Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    algorithms = []
    aurocs = []
    sensitivities = []
    specificities = []
    f1_scores = []
    
    for algorithm, data in results_dict.items():
        if "metrics" not in data:
            continue
        
        metrics = data["metrics"]
        algorithms.append(algorithm)
        aurocs.append(metrics["auroc"])
        sensitivities.append(metrics["sensitivity"])
        specificities.append(metrics["specificity"])
        f1_scores.append(metrics["f1"])
    
    x = np.arange(len(algorithms))
    width = 0.2
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.bar(x - 1.5 * width, aurocs, width, label="AUROC", alpha=0.8)
    ax.bar(x - 0.5 * width, sensitivities, width, label="Sensitivity", alpha=0.8)
    ax.bar(x + 0.5 * width, specificities, width, label="Specificity", alpha=0.8)
    ax.bar(x + 1.5 * width, f1_scores, width, label="F1 Score", alpha=0.8)
    
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title(f"Metrics Comparison - {cohort_name}", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(algorithms, rotation=45, ha="right")
    ax.legend(fontsize=10)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3, axis="y")
    
    output_path = os.path.join(output_dir, f"metrics_comparison_{cohort_name.lower().replace(' ', '_')}.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved metrics comparison to {output_path}")
    plt.close()


def print_metrics_summary(cohort_results, cohort_name):
    """Print a summary of metrics for a cohort."""
    print(f"\n{'='*80}")
    print(f"METRICS SUMMARY - {cohort_name}")
    print(f"{'='*80}")
    
    for algorithm, data in cohort_results.items():
        if "metrics" not in data:
            continue
        
        metrics = data["metrics"]
        
        print(f"\n{algorithm}:")
        print(f"  AUROC: {metrics['auroc']:.4f} (95% CI: {metrics['auroc_ci_lower']:.4f}-{metrics['auroc_ci_upper']:.4f})")
        print(f"  AP: {metrics['ap']:.4f}")
        print(f"  Sensitivity: {metrics['sensitivity']:.4f}")
        print(f"  Specificity: {metrics['specificity']:.4f}")
        print(f"  F1 Score: {metrics['f1']:.4f}")
        print(f"  Brier Score: {metrics['brier']:.4f}")
        print(f"  Calibration Slope: {metrics['calibration_slope']:.4f}")
        print(f"  Calibration Intercept: {metrics['calibration_intercept']:.4f}")
        print(f"  O:E Ratio: {metrics['o_e_ratio']:.4f}")
