"""Multiparous and Past-Pregnancy cohort pipeline."""

from src.preprocessing import prepare_multiparous_data
from src.models import train_model_with_hyperparameter_tuning, evaluate_model
from src.visualization import (
    plot_roc_curves, plot_precision_recall_curves, plot_metrics_comparison,
    print_metrics_summary
)
from src.utils import MODEL_ALGORITHMS


def run_multiparous_pipeline(df, output_dir="output"):
    """
    Run end-to-end pipeline for multiparous and past-pregnancy cohorts.
    
    Returns:
        results_dict: {
            "multiparous": {"models": {...}, "metrics": {...}, "predictions": {...}},
            "past_pregnancy": {"models": {...}, "metrics": {...}, "predictions": {...}}
        }
    """
    print("\n" + "="*80)
    print("MULTIPAROUS PIPELINE")
    print("="*80)
    
    results = {}
    
    # Process both cohorts
    for cohort_name in ["multiparous", "past_pregnancy"]:
        print(f"\n{'#'*80}")
        print(f"Processing {cohort_name.replace('_', ' ').title()}")
        print(f"{'#'*80}")
        
        # Prepare data
        X_train, X_val, X_test, y_train, y_val, y_test, preprocessor = \
            prepare_multiparous_data(df, cohort=cohort_name)
        
        cohort_results = {}
        
        # Train each algorithm
        for algorithm in MODEL_ALGORITHMS:
            print(f"\n{algorithm}:")
            
            # Train with hyperparameter tuning
            best_model, cv_results = train_model_with_hyperparameter_tuning(
                X_train, y_train, algorithm
            )
            
            # Evaluate on test set
            metrics, predictions = evaluate_model(best_model, X_test, y_test, algorithm)
            
            # Store results
            cohort_results[algorithm] = {
                "model": best_model,
                "preprocessor": preprocessor,
                "metrics": metrics,
                "predictions": predictions,
                "cv_results": cv_results,
            }
        
        results[cohort_name] = cohort_results
        
        # Generate visualizations
        print(f"\n--- Generating visualizations for {cohort_name} ---")
        plot_roc_curves(cohort_results, cohort_name.replace('_', ' ').title(), output_dir)
        plot_precision_recall_curves(cohort_results, cohort_name.replace('_', ' ').title(), output_dir)
        plot_metrics_comparison(cohort_results, cohort_name.replace('_', ' ').title(), output_dir)
        print_metrics_summary(cohort_results, cohort_name.replace('_', ' ').title())
    
    return results
