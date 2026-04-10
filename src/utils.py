"""Utility functions and feature definitions for GDM prediction pipeline."""

# Feature definitions per cohort
COHORT_FEATURES = {
    "first_trimester": {
        "categorical": ["Ethnic Origin of Patient"],
        "numerical": ["Age at booking", "BMI", "Systolic BP", "Diastolic BP"],
        "binary": ["Hx_GDM", "FH Diabetes", "Other Endocrine problems", "Parity"],
        "description": "First-Trimester (FTP-9) model - 9 predictors from first antenatal visit",
    },
    "nulliparous": {
        "categorical": ["Ethnic Origin of Patient"],
        "numerical": ["Age at booking", "BMI", "Systolic BP", "Diastolic BP"],
        "binary": ["FH Diabetes", "Other Endocrine problems"],
        "description": "Nulliparous model - excludes parity and GDM history (first-time mothers only)",
    },
    "multiparous": {
        "categorical": ["Ethnic Origin of Patient"],
        "numerical": ["Age at booking", "BMI", "Inter-pregnancy weight change", "Inter-pregnancy interval"],
        "binary": ["Hx_GDM", "FH Diabetes"],
        "other": ["Previous birth weight percentile"],
        "description": "Multiparous model - 8 variables for women with previous pregnancies",
    },
    "past_pregnancy": {
        "categorical": ["Ethnic Origin of Patient"],
        "numerical": ["Age at booking", "BMI"],
        "binary": ["Hx_GDM", "FH Diabetes"],
        "other": ["Previous birth weight percentile"],
        "description": "Past-Pregnancy model - 6 variables using only previous pregnancy data",
    },
}

# Train/validation/test split ratios per workflow
SPLIT_RATIOS = {
    "first_trimester": {"train": 0.8, "val": 0.1, "test": 0.1},
    "multiparous": {"train": 0.7, "val": 0.15, "test": 0.15},
}

# Model algorithms
MODEL_ALGORITHMS = ["Random Forest", "Logistic Regression", "XGBoost", "ExplainableBoostingClassifier"]

# Hyperparameter search space for RandomizedSearchCV
HYPERPARAMETERS = {
    "Random Forest": {
        "n_estimators": [100, 200, 300],
        "max_depth": [10, 20, 30, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
    },
    "Logistic Regression": {
        "C": [0.001, 0.01, 0.1, 1, 10, 100],
        "max_iter": [100, 200, 500],
    },
    "XGBoost": {
        "n_estimators": [100, 200, 300],
        "max_depth": [5, 10, 15],
        "learning_rate": [0.01, 0.05, 0.1],
        "subsample": [0.8, 0.9, 1.0],
        "colsample_bytree": [0.8, 0.9, 1.0],
    },
    "ExplainableBoostingClassifier": {
        "max_rounds": [100, 200, 300],
        "max_bins": [32, 64, 128],
        "learning_rate": [0.001, 0.01, 0.1],
    },
}

# Metric thresholds for validation
VALID_METRIC_RANGES = {
    "AUROC": (0.5, 1.0),
    "sensitivity": (0.0, 1.0),
    "specificity": (0.0, 1.0),
    "F1": (0.0, 1.0),
    "Brier": (0.0, 1.0),
}


def get_features_for_cohort(cohort):
    """Get feature list for a specific cohort."""
    if cohort not in COHORT_FEATURES:
        raise ValueError(f"Unknown cohort: {cohort}. Choose from {list(COHORT_FEATURES.keys())}")
    return COHORT_FEATURES[cohort]


def get_all_feature_columns(cohort):
    """Get all feature column names for a cohort (excluding target)."""
    cohort_info = COHORT_FEATURES[cohort]
    features = []
    for key in ["categorical", "numerical", "binary", "other"]:
        if key in cohort_info:
            features.extend(cohort_info[key])
    return features
