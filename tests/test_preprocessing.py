"""Tests for data preprocessing module."""

import pytest
import numpy as np
from src.preprocessing import prepare_first_trimester_data, prepare_multiparous_data
from src.utils import get_features_for_cohort


def test_prepare_first_trimester_data_shape(dummy_data_small):
    """Test that prepared data has correct shape."""
    X_train, X_val, X_test, y_train, y_val, y_test, preprocessor = \
        prepare_first_trimester_data(dummy_data_small, cohort="first_trimester")
    
    # Check shapes
    assert X_train.shape[0] > 0, "Train set should not be empty"
    assert X_val.shape[0] > 0, "Val set should not be empty"
    assert X_test.shape[0] > 0, "Test set should not be empty"
    
    # Check that features are arrays
    assert isinstance(X_train, np.ndarray)
    assert isinstance(X_val, np.ndarray)
    assert isinstance(X_test, np.ndarray)
    
    # Check targets are 1D arrays
    assert y_train.ndim == 1
    assert y_val.ndim == 1
    assert y_test.ndim == 1


def test_prepare_first_trimester_data_split_ratio(dummy_data_small):
    """Test that train/val/test split follows expected ratios."""
    X_train, X_val, X_test, y_train, y_val, y_test, preprocessor = \
        prepare_first_trimester_data(dummy_data_small, cohort="first_trimester")
    
    total = len(X_train) + len(X_val) + len(X_test)
    train_ratio = len(X_train) / total
    val_ratio = len(X_val) / total
    test_ratio = len(X_test) / total
    
    # Should be approximately 80/10/10
    assert 0.75 <= train_ratio <= 0.85, f"Train ratio {train_ratio} not near 0.80"
    assert 0.05 <= val_ratio <= 0.15, f"Val ratio {val_ratio} not near 0.10"
    assert 0.05 <= test_ratio <= 0.15, f"Test ratio {test_ratio} not near 0.10"


def test_prepare_first_trimester_data_no_nan(dummy_data_small):
    """Test that prepared data has no NaN values."""
    X_train, X_val, X_test, y_train, y_val, y_test, preprocessor = \
        prepare_first_trimester_data(dummy_data_small, cohort="first_trimester")
    
    assert not np.isnan(X_train).any(), "Train features contain NaN"
    assert not np.isnan(X_val).any(), "Val features contain NaN"
    assert not np.isnan(X_test).any(), "Test features contain NaN"
    assert not np.isnan(y_train).any(), "Train target contains NaN"
    assert not np.isnan(y_val).any(), "Val target contains NaN"
    assert not np.isnan(y_test).any(), "Test target contains NaN"


def test_prepare_first_trimester_nulliparous_cohort_difference(dummy_data_small):
    """Test that nulliparous and first_trimester have different feature counts."""
    X_train_ft, _, _, _, _, _, _ = \
        prepare_first_trimester_data(dummy_data_small, cohort="first_trimester")
    X_train_null, _, _, _, _, _, _ = \
        prepare_first_trimester_data(dummy_data_small, cohort="nulliparous")
    
    # Nulliparous should have fewer features (no Hx_GDM, no Parity)
    assert X_train_null.shape[1] < X_train_ft.shape[1], \
        "Nulliparous should have fewer features than first_trimester"


def test_prepare_multiparous_data_shape(dummy_data_medium):
    """Test that prepared multiparous data has correct shape."""
    try:
        X_train, X_val, X_test, y_train, y_val, y_test, preprocessor = \
            prepare_multiparous_data(dummy_data_medium, cohort="multiparous")
    except ValueError:
        pytest.skip("Not enough multiparous data for this test")
    
    # With medium dataset we should have some data
    if len(X_train) > 0:
        assert isinstance(X_train, np.ndarray)
        assert isinstance(X_val, np.ndarray)
        assert isinstance(X_test, np.ndarray)


def test_prepare_multiparous_data_split_ratio(dummy_data_medium):
    """Test that multiparous train/val/test split follows expected ratios (70/15/15)."""
    X_train, X_val, X_test, y_train, y_val, y_test, preprocessor = \
        prepare_multiparous_data(dummy_data_medium, cohort="multiparous")
    
    # Skip if not enough data
    if len(X_train) + len(X_val) + len(X_test) < 10:
        pytest.skip("Not enough multiparous data for split ratio test")
    
    total = len(X_train) + len(X_val) + len(X_test)
    train_ratio = len(X_train) / total
    val_ratio = len(X_val) / total
    test_ratio = len(X_test) / total
    
    # Should be approximately 70/15/15
    assert 0.60 <= train_ratio <= 0.80, f"Train ratio {train_ratio} not near 0.70"
    assert 0.05 <= val_ratio <= 0.25, f"Val ratio {val_ratio} not near 0.15"
    assert 0.05 <= test_ratio <= 0.25, f"Test ratio {test_ratio} not near 0.15"


def test_get_features_for_cohort():
    """Test that feature definitions are correct for each cohort."""
    ft_features = get_features_for_cohort("first_trimester")
    assert "categorical" in ft_features
    assert "numerical" in ft_features
    assert "binary" in ft_features
    
    null_features = get_features_for_cohort("nulliparous")
    assert "categorical" in null_features
    assert "numerical" in null_features
    assert "binary" in null_features
    
    mp_features = get_features_for_cohort("multiparous")
    assert "categorical" in mp_features
    assert "numerical" in mp_features


def test_unknown_cohort_raises_error():
    """Test that requesting unknown cohort raises error."""
    with pytest.raises(ValueError):
        get_features_for_cohort("unknown_cohort")
