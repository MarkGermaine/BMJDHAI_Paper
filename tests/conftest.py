"""Pytest configuration and shared fixtures."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


@pytest.fixture
def dummy_data_small():
    """Fixture: Small dummy dataset for quick tests."""
    np.random.seed(42)
    
    n_samples = 50
    today = datetime.now()
    
    data = {
        "ID": np.arange(1, n_samples + 1),
        "Date of Birth": [
            (today - timedelta(days=int(365 * age))).strftime("%Y-%m-%d")
            for age in np.random.uniform(25, 35, n_samples)
        ],
        "Ethnic Origin of Patient": np.random.choice(
            ["White", "Black", "Asian"], size=n_samples, p=[0.7, 0.15, 0.15]
        ),
        "Age at booking": np.random.uniform(20, 40, n_samples).astype(int),
        "BMI": np.random.uniform(20, 35, n_samples),
        "Systolic BP": np.random.uniform(110, 140, n_samples).astype(int),
        "Diastolic BP": np.random.uniform(70, 90, n_samples).astype(int),
        "Hx_GDM": np.random.binomial(1, 0.15, n_samples),
        "FH Diabetes": np.random.binomial(1, 0.25, n_samples),
        "Other Endocrine problems": np.random.binomial(1, 0.05, n_samples),
        "Parity": np.random.binomial(3, 0.3, n_samples),
        "Inter-pregnancy weight change": np.random.normal(1, 3, n_samples),
        "Inter-pregnancy interval": np.random.uniform(0.5, 5, n_samples),
        "Previous birth weight percentile": np.random.uniform(10, 90, n_samples),
        "GDM": np.random.binomial(1, 0.12, n_samples),
    }
    
    return pd.DataFrame(data)


@pytest.fixture
def dummy_data_medium():
    """Fixture: Medium dummy dataset for integration tests."""
    np.random.seed(43)
    
    n_samples = 200
    today = datetime.now()
    
    data = {
        "ID": np.arange(1, n_samples + 1),
        "Date of Birth": [
            (today - timedelta(days=int(365 * age))).strftime("%Y-%m-%d")
            for age in np.random.uniform(25, 35, n_samples)
        ],
        "Ethnic Origin of Patient": np.random.choice(
            ["White", "Black", "Asian"], size=n_samples, p=[0.7, 0.15, 0.15]
        ),
        "Age at booking": np.random.uniform(20, 40, n_samples).astype(int),
        "BMI": np.random.uniform(20, 35, n_samples),
        "Systolic BP": np.random.uniform(110, 140, n_samples).astype(int),
        "Diastolic BP": np.random.uniform(70, 90, n_samples).astype(int),
        "Hx_GDM": np.random.binomial(1, 0.15, n_samples),
        "FH Diabetes": np.random.binomial(1, 0.25, n_samples),
        "Other Endocrine problems": np.random.binomial(1, 0.05, n_samples),
        "Parity": np.random.binomial(3, 0.3, n_samples),
        "Inter-pregnancy weight change": np.random.normal(1, 3, n_samples),
        "Inter-pregnancy interval": np.random.uniform(0.5, 5, n_samples),
        "Previous birth weight percentile": np.random.uniform(10, 90, n_samples),
        "GDM": np.random.binomial(1, 0.12, n_samples),
    }
    
    return pd.DataFrame(data)
