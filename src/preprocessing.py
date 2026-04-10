"""Data preprocessing module for GDM prediction pipeline."""

import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from .utils import get_features_for_cohort, SPLIT_RATIOS


def load_data(csv_path):
    """Load data from CSV file."""
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows with {len(df.columns)} columns")
    return df


def prepare_first_trimester_data(df, cohort="first_trimester"):
    """
    Prepare data for first-trimester or nulliparous cohort.
    
    Steps:
    1. Drop post-first-trimester columns (keep only first visit data)
    2. Remove duplicates based on ID
    3. Extract and encode features
    4. Split into train/val/test (80/10/10) grouped by patient ID
    
    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test (preprocessed numpy arrays)
        preprocessor (ColumnTransformer for future use)
    """
    print(f"\n--- Preparing {cohort} data ---")
    
    # Remove duplicates
    df_clean = df.drop_duplicates(subset=("ID",), keep="first")
    print(f"After removing duplicates: {len(df_clean)} samples")
    
    # Get feature names for this cohort
    features_spec = get_features_for_cohort(cohort)
    categorical_features = features_spec.get("categorical", [])
    numerical_features = features_spec.get("numerical", [])
    binary_features = features_spec.get("binary", [])
    
    # Combine all feature columns
    feature_columns = categorical_features + numerical_features + binary_features
    
    # Create preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features),
            ("num", StandardScaler(), numerical_features + binary_features),
        ]
    )
    
    # Extract features and target
    X = df_clean[feature_columns].copy()
    y = df_clean["GDM"].copy()
    groups = df_clean["ID"].copy()
    
    print(f"Features: {feature_columns}")
    print(f"Target distribution: {y.value_counts().to_dict()}")
    
    # Fit preprocessor
    X_preprocessed = preprocessor.fit_transform(X)
    
    # Group-based split to keep same patient ID together
    split_ratio = SPLIT_RATIOS.get(cohort, SPLIT_RATIOS["first_trimester"])
    train_ratio = split_ratio["train"]
    val_ratio = split_ratio["val"]
    test_ratio = split_ratio["test"]
    
    # First split: train vs (val+test)
    first_split = GroupShuffleSplit(
        n_splits=1, 
        test_size=val_ratio + test_ratio, 
        random_state=42
    )
    train_idx, temp_idx = next(first_split.split(X_preprocessed, y, groups))
    
    # Second split: val vs test from the temp set
    val_test_ratio = test_ratio / (val_ratio + test_ratio)
    second_split = GroupShuffleSplit(
        n_splits=1,
        test_size=val_test_ratio,
        random_state=42
    )
    val_idx, test_idx = next(second_split.split(
        X_preprocessed[temp_idx],
        y.iloc[temp_idx],
        groups.iloc[temp_idx]
    ))
    
    # Adjust indices for second split
    test_idx = temp_idx[test_idx]
    val_idx = temp_idx[val_idx]
    
    X_train = X_preprocessed[train_idx]
    X_val = X_preprocessed[val_idx]
    X_test = X_preprocessed[test_idx]
    
    y_train = y.iloc[train_idx].values
    y_val = y.iloc[val_idx].values
    y_test = y.iloc[test_idx].values
    
    print(f"Train size: {len(X_train)}, Val size: {len(X_val)}, Test size: {len(X_test)}")
    print(f"Train GDM distribution: {np.bincount(y_train)}")
    print(f"Val GDM distribution: {np.bincount(y_val)}")
    print(f"Test GDM distribution: {np.bincount(y_test)}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test, preprocessor


def prepare_multiparous_data(df, cohort="multiparous"):
    """
    Prepare data for multiparous or past-pregnancy cohort.
    
    Steps:
    1. Sort by patient ID and date of birth
    2. Create future_GDM target by shifting GDM status forward one pregnancy
    3. Drop rows without future pregnancy data
    4. Extract and encode features
    5. Split into train/val/test (70/15/15) grouped by patient ID
    
    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test (preprocessed numpy arrays)
        preprocessor (ColumnTransformer for future use)
    """
    print(f"\n--- Preparing {cohort} data ---")
    
    df_clean = df.sort_values(["ID", "Date of Birth"]).copy()
    
    # Create future_GDM by shifting GDM forward one pregnancy per patient
    df_clean["future_GDM"] = df_clean.groupby("ID")["GDM"].shift(-1)
    
    # Drop rows without future pregnancy data
    df_clean = df_clean.dropna(subset=["future_GDM"])
    print(f"After creating future GDM target: {len(df_clean)} samples")
    
    # Get feature names for this cohort
    features_spec = get_features_for_cohort(cohort)
    categorical_features = features_spec.get("categorical", [])
    numerical_features = features_spec.get("numerical", [])
    binary_features = features_spec.get("binary", [])
    other_features = features_spec.get("other", [])
    
    # Combine all feature columns
    feature_columns = categorical_features + numerical_features + binary_features + other_features
    
    # Create preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features),
            ("num", StandardScaler(), numerical_features + binary_features + other_features),
        ]
    )
    
    # Extract features and target
    X = df_clean[feature_columns].copy()
    y = df_clean["future_GDM"].copy().astype(int)
    groups = df_clean["ID"].copy()
    
    print(f"Features: {feature_columns}")
    print(f"Target distribution: {y.value_counts().to_dict()}")
    
    # Fit preprocessor
    X_preprocessed = preprocessor.fit_transform(X)
    
    # Group-based split
    split_ratio = SPLIT_RATIOS.get(cohort, SPLIT_RATIOS["multiparous"])
    train_ratio = split_ratio["train"]
    val_ratio = split_ratio["val"]
    test_ratio = split_ratio["test"]
    
    # First split: train vs (val+test)
    first_split = GroupShuffleSplit(
        n_splits=1,
        test_size=val_ratio + test_ratio,
        random_state=42
    )
    train_idx, temp_idx = next(first_split.split(X_preprocessed, y, groups))
    
    # Second split: val vs test from the temp set
    val_test_ratio = test_ratio / (val_ratio + test_ratio)
    second_split = GroupShuffleSplit(
        n_splits=1,
        test_size=val_test_ratio,
        random_state=42
    )
    val_idx, test_idx = next(second_split.split(
        X_preprocessed[temp_idx],
        y.iloc[temp_idx],
        groups.iloc[temp_idx]
    ))
    
    # Adjust indices
    test_idx = temp_idx[test_idx]
    val_idx = temp_idx[val_idx]
    
    X_train = X_preprocessed[train_idx]
    X_val = X_preprocessed[val_idx]
    X_test = X_preprocessed[test_idx]
    
    y_train = y.iloc[train_idx].values
    y_val = y.iloc[val_idx].values
    y_test = y.iloc[test_idx].values
    
    print(f"Train size: {len(X_train)}, Val size: {len(X_val)}, Test size: {len(X_test)}")
    print(f"Train GDM distribution: {np.bincount(y_train)}")
    print(f"Val GDM distribution: {np.bincount(y_val)}")
    print(f"Test GDM distribution: {np.bincount(y_test)}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test, preprocessor
