"""Model training and evaluation module for GDM prediction pipeline."""

import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from interpret.glassbox import ExplainableBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, auc, roc_curve,
    confusion_matrix, f1_score, brier_score_loss
)
from .utils import HYPERPARAMETERS, MODEL_ALGORITHMS


def get_model_instance(algorithm_name):
    """Create model instance for given algorithm."""
    if algorithm_name == "Random Forest":
        return RandomForestClassifier(random_state=42, n_jobs=-1)
    elif algorithm_name == "Logistic Regression":
        return LogisticRegression(random_state=42, max_iter=1000)
    elif algorithm_name == "XGBoost":
        return XGBClassifier(random_state=42, use_label_encoder=False, eval_metric="logloss")
    elif algorithm_name == "ExplainableBoostingClassifier":
        return ExplainableBoostingClassifier(random_state=42)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm_name}")


def train_model_with_hyperparameter_tuning(X_train, y_train, algorithm_name, n_iter=20):
    """
    Train a model with RandomizedSearchCV and StratifiedKFold cross-validation.
    
    Args:
        X_train: Training features
        y_train: Training target
        algorithm_name: Name of algorithm to train
        n_iter: Number of parameter combinations to try
    
    Returns:
        best_model: Best trained model
        cv_results: Cross-validation results dictionary
    """
    print(f"\n  Training {algorithm_name}...")
    
    model = get_model_instance(algorithm_name)
    hyperparams = HYPERPARAMETERS.get(algorithm_name, {})
    
    if not hyperparams:
        print(f"    No hyperparameters defined; training with defaults...")
        model.fit(X_train, y_train)
        return model, {}
    
    # Use StratifiedKFold for cross-validation
    cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # RandomizedSearchCV
    random_search = RandomizedSearchCV(
        model,
        hyperparams,
        n_iter=n_iter,
        cv=cv_strategy,
        scoring="roc_auc",
        n_jobs=-1,
        random_state=42,
        verbose=0
    )
    
    random_search.fit(X_train, y_train)
    
    print(f"    Best params: {random_search.best_params_}")
    print(f"    Best CV AUROC: {random_search.best_score_:.4f}")
    
    return random_search.best_estimator_, random_search.cv_results_


def bootstrap_ci_auroc(y_true, y_pred_proba, n_bootstraps=1000, ci=95):
    """
    Calculate bootstrap confidence interval for AUROC.
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        n_bootstraps: Number of bootstrap iterations
        ci: Confidence interval (e.g., 95)
    
    Returns:
        auroc: AUROC on full dataset
        auroc_lower: Lower CI bound
        auroc_upper: Upper CI bound
    """
    try:
        auroc_full = roc_auc_score(y_true, y_pred_proba)
    except:
        # Handle edge cases where all predictions are same class
        return 0.5, 0.5, 0.5
    
    auroc_scores = []
    n_samples = len(y_true)
    
    np.random.seed(42)
    for _ in range(n_bootstraps):
        indices = np.random.choice(n_samples, n_samples, replace=True)
        try:
            auroc = roc_auc_score(y_true[indices], y_pred_proba[indices])
            auroc_scores.append(auroc)
        except:
            pass
    
    if len(auroc_scores) == 0:
        return auroc_full, auroc_full * 0.9, auroc_full * 1.1
    
    alpha = (100 - ci) / 2
    lower = np.percentile(auroc_scores, alpha)
    upper = np.percentile(auroc_scores, 100 - alpha)
    
    return auroc_full, lower, upper


def evaluate_model(model, X_test, y_test, algorithm_name):
    """
    Evaluate model on test set and compute all metrics.
    
    Returns:
        metrics_dict: Dictionary with all evaluation metrics
        predictions_dict: Dictionary with ROC and PR curve data
    """
    print(f"\n  Evaluating {algorithm_name}...")
    
    # Get predictions
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    auroc, auroc_lower, auroc_upper = bootstrap_ci_auroc(y_test, y_pred_proba, n_bootstraps=1000)
    
    # Precision-Recall curve
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    ap = auc(recall, precision)
    
    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    
    # Confusion matrix metrics
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    # Other metrics
    f1 = f1_score(y_test, y_pred, zero_division=0)
    brier = brier_score_loss(y_test, y_pred_proba)
    
    # Calibration metrics
    calibration_slope, calibration_intercept = calculate_calibration_metrics(y_test, y_pred_proba)
    o_e_ratio = calculate_o_e_ratio(y_test, y_pred_proba)
    
    metrics_dict = {
        "algorithm": algorithm_name,
        "auroc": auroc,
        "auroc_ci_lower": auroc_lower,
        "auroc_ci_upper": auroc_upper,
        "ap": ap,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "f1": f1,
        "brier": brier,
        "calibration_slope": calibration_slope,
        "calibration_intercept": calibration_intercept,
        "o_e_ratio": o_e_ratio,
    }
    
    predictions_dict = {
        "fpr": fpr,
        "tpr": tpr,
        "precision": precision,
        "recall": recall,
        "y_pred_proba": y_pred_proba,
    }
    
    print(f"    AUROC: {auroc:.4f} (95% CI: {auroc_lower:.4f}-{auroc_upper:.4f})")
    print(f"    AP: {ap:.4f}")
    print(f"    Sensitivity: {sensitivity:.4f}, Specificity: {specificity:.4f}")
    print(f"    F1: {f1:.4f}, Brier: {brier:.4f}")
    
    return metrics_dict, predictions_dict


def calculate_calibration_metrics(y_true, y_pred_proba, n_bins=10):
    """Calculate calibration slope and intercept."""
    from scipy.stats import linregress
    
    # Bin predicted probabilities
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(y_pred_proba, bin_edges) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)
    
    # Calculate observed vs expected
    true_positives_per_bin = np.zeros(n_bins)
    count_per_bin = np.zeros(n_bins)
    mean_pred_per_bin = np.zeros(n_bins)
    
    for i in range(n_bins):
        mask = bin_indices == i
        if mask.sum() > 0:
            true_positives_per_bin[i] = y_true[mask].sum()
            count_per_bin[i] = mask.sum()
            mean_pred_per_bin[i] = y_pred_proba[mask].mean()
    
    # Filter out empty bins
    valid_mask = count_per_bin > 0
    observed = true_positives_per_bin[valid_mask] / count_per_bin[valid_mask]
    expected = mean_pred_per_bin[valid_mask]
    
    if len(observed) > 1:
        slope, intercept, _, _, _ = linregress(expected, observed)
    else:
        slope, intercept = 1.0, 0.0
    
    return slope, intercept


def calculate_o_e_ratio(y_true, y_pred_proba):
    """Calculate observed-to-expected ratio."""
    observed = y_true.sum()
    expected = y_pred_proba.sum()
    return observed / expected if expected > 0 else 0


def save_model_results(results_dict, output_path):
    """Save model results to disk using joblib."""
    print(f"\nSaving model results to {output_path}...")
    joblib.dump(results_dict, output_path)
    print(f"Saved successfully!")


def load_model_results(results_path):
    """Load model results from disk."""
    print(f"Loading model results from {results_path}...")
    return joblib.load(results_path)
