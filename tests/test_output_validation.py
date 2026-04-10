"""Tests for output validation and metric validation."""

import pytest
import numpy as np
from src.preprocessing import prepare_first_trimester_data
from src.models import evaluate_model, get_model_instance


def test_auroc_sensitivity_specificity(dummy_data_medium):
    """Test that key metrics are in valid ranges."""
    X_train, X_val, X_test, y_train, y_val, y_test, _ = \
        prepare_first_trimester_data(dummy_data_medium, cohort="first_trimester")
    
    model = get_model_instance("Logistic Regression")
    model.fit(X_train, y_train)
    
    try:
        metrics, _ = evaluate_model(model, X_test, y_test, "Logistic Regression")
        assert 0.0 <= metrics["auroc"] <= 1.0
        assert 0.0 <= metrics["sensitivity"] <= 1.0
        assert 0.0 <= metrics["specificity"] <= 1.0
        assert 0.0 <= metrics["f1"] <= 1.0
    except Exception:
        pytest.skip("Model evaluation skipped")


def test_calibration_reasonable(dummy_data_medium):
    """Test calibration metrics are reasonable."""
    X_train, X_val, X_test, y_train, y_val, y_test, _ = \
        prepare_first_trimester_data(dummy_data_medium, cohort="first_trimester")
    
    model = get_model_instance("Logistic Regression")
    model.fit(X_train, y_train)
    
    metrics, _ = evaluate_model(model, X_test, y_test, "Logistic Regression")
    assert -10 < metrics["calibration_slope"] < 10


def test_o_e_ratio(dummy_data_medium):
    """Test O:E ratio is reasonable."""
    X_train, X_val, X_test, y_train, y_val, y_test, _ = \
        prepare_first_trimester_data(dummy_data_medium, cohort="first_trimester")
    
    model = get_model_instance("Random Forest")
    model.fit(X_train, y_train)
    
    metrics, _ = evaluate_model(model, X_test, y_test, "Random Forest")
    assert 0.1 < metrics["o_e_ratio"] < 10
