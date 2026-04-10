"""Tests for model training and evaluation module."""

import pytest
import numpy as np
from src.preprocessing import prepare_first_trimester_data
from src.models import (
    get_model_instance, train_model_with_hyperparameter_tuning,
    evaluate_model, bootstrap_ci_auroc
)
from src.utils import VALID_METRIC_RANGES


def test_get_model_instance():
    """Test that model instances are created correctly."""
    algorithms = ["Random Forest", "Logistic Regression", "XGBoost", "ExplainableBoostingClassifier"]
    
    for algo in algorithms:
        model = get_model_instance(algo)
        assert model is not None, f"Failed to create {algo}"
        assert hasattr(model, "fit"), f"{algo} missing fit method"
        assert hasattr(model, "predict_proba"), f"{algo} missing predict_proba method"


def test_unknown_algorithm_raises_error():
    """Test that unknown algorithm raises error."""
    with pytest.raises(ValueError):
        get_model_instance("UnknownAlgorithm")


def test_train_model_random_forest(dummy_data_small):
    """Test training Random Forest model."""
    X_train, X_val, X_test, y_train, y_val, y_test, _ = \
        prepare_first_trimester_data(dummy_data_small, cohort="first_trimester")
    
    model, cv_results = train_model_with_hyperparameter_tuning(
        X_train, y_train, "Random Forest", n_iter=5
    )
    
    assert model is not None
    assert hasattr(model, "predict_proba")
    
    # Test predictions
    y_pred_proba = model.predict_proba(X_test)
    assert y_pred_proba.shape == (len(X_test), 2)
    assert np.all((y_pred_proba >= 0) & (y_pred_proba <= 1))


def test_train_model_logistic_regression(dummy_data_small):
    """Test training Logistic Regression model."""
    X_train, X_val, X_test, y_train, y_val, y_test, _ = \
        prepare_first_trimester_data(dummy_data_small, cohort="first_trimester")
    
    model, cv_results = train_model_with_hyperparameter_tuning(
        X_train, y_train, "Logistic Regression", n_iter=5
    )
    
    assert model is not None
    y_pred_proba = model.predict_proba(X_test)
    assert y_pred_proba.shape == (len(X_test), 2)


def test_train_model_xgboost(dummy_data_small):
    """Test training XGBoost model."""
    X_train, X_val, X_test, y_train, y_val, y_test, _ = \
        prepare_first_trimester_data(dummy_data_small, cohort="first_trimester")
    
    model, cv_results = train_model_with_hyperparameter_tuning(
        X_train, y_train, "XGBoost", n_iter=5
    )
    
    assert model is not None
    y_pred_proba = model.predict_proba(X_test)
    assert y_pred_proba.shape == (len(X_test), 2)


def test_bootstrap_ci_auroc(dummy_data_small):
    """Test bootstrap confidence interval calculation."""
    y_true = np.array([0, 0, 0, 1, 1, 1, 0, 1, 0, 1])
    y_pred_proba = np.array([0.1, 0.2, 0.3, 0.6, 0.7, 0.8, 0.15, 0.65, 0.25, 0.75])
    
    auroc, ci_lower, ci_upper = bootstrap_ci_auroc(y_true, y_pred_proba, n_bootstraps=100)
    
    # Check valid ranges
    assert 0.5 <= auroc <= 1.0
    assert 0.5 <= ci_lower <= 1.0
    assert 0.5 <= ci_upper <= 1.0
    assert ci_lower <= auroc <= ci_upper


def test_evaluate_model_metrics_in_range(dummy_data_medium):
    """Test that evaluation metrics are in valid ranges."""
    X_train, X_val, X_test, y_train, y_val, y_test, _ = \
        prepare_first_trimester_data(dummy_data_medium, cohort="first_trimester")
    
    model, _ = train_model_with_hyperparameter_tuning(
        X_train, y_train, "Random Forest", n_iter=5
    )
    
    metrics, predictions = evaluate_model(model, X_test, y_test, "Random Forest")
    
    # Check all metrics are in valid ranges
    assert VALID_METRIC_RANGES["AUROC"][0] <= metrics["auroc"] <= VALID_METRIC_RANGES["AUROC"][1]
    assert VALID_METRIC_RANGES["sensitivity"][0] <= metrics["sensitivity"] <= VALID_METRIC_RANGES["sensitivity"][1]
    assert VALID_METRIC_RANGES["specificity"][0] <= metrics["specificity"] <= VALID_METRIC_RANGES["specificity"][1]
    assert VALID_METRIC_RANGES["F1"][0] <= metrics["f1"] <= VALID_METRIC_RANGES["F1"][1]
    assert VALID_METRIC_RANGES["Brier"][0] <= metrics["brier"] <= VALID_METRIC_RANGES["Brier"][1]


def test_evaluate_model_returns_predictions(dummy_data_small):
    """Test that evaluation returns prediction curves."""
    X_train, X_val, X_test, y_train, y_val, y_test, _ = \
        prepare_first_trimester_data(dummy_data_small, cohort="first_trimester")
    
    model, _ = train_model_with_hyperparameter_tuning(
        X_train, y_train, "Logistic Regression", n_iter=3
    )
    
    metrics, predictions = evaluate_model(model, X_test, y_test, "Logistic Regression")
    
    # Check prediction dictionary
    assert "fpr" in predictions
    assert "tpr" in predictions
    assert "precision" in predictions
    assert "recall" in predictions
    assert "y_pred_proba" in predictions
    
    assert len(predictions["fpr"]) > 0
    assert len(predictions["tpr"]) > 0
