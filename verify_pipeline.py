#!/usr/bin/env python
"""Quick verification that all pipeline components work."""

import sys
import pandas as pd
import numpy as np
from src.preprocessing import prepare_first_trimester_data, prepare_multiparous_data
from src.models import get_model_instance, train_model_with_hyperparameter_tuning, evaluate_model
from src.visualization import print_metrics_summary

print("=" * 80)
print("PIPELINE COMPONENT VERIFICATION")
print("=" * 80)

# Load data
print("\n1. Loading synthetic data...")
df = pd.read_csv("data/dummy_data_large.csv")
print(f"   ✓ Loaded {len(df)} rows, {len(df.columns)} columns")
print(f"   ✓ Unique patients: {df['ID'].nunique()}")
print(f"   ✓ GDM prevalence: {df['GDM'].mean():.2%}")

# Test first_trimester preprocessing
print("\n2. Testing first_trimester preprocessing...")
try:
    X_train, X_val, X_test, y_train, y_val, y_test, preprocessor = prepare_first_trimester_data(df, "first_trimester")
    print(f"   ✓ Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    print(f"   ✓ Train GDM rate: {y_train.mean():.2%}")
    print(f"   ✓ Preprocessor with {len(preprocessor.named_transformers_)} transformers")
except Exception as e:
    print(f"   ✗ Error: {e}")
    sys.exit(1)

# Test multiparous preprocessing
print("\n3. Testing multiparous preprocessing...")
try:
    X_train, X_val, X_test, y_train, y_val, y_test, preprocessor = prepare_multiparous_data(df,  "multiparous")
    print(f"   ✓ Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    print(f"   ✓ Train GDM rate: {y_train.mean():.2%}")
    print(f"   ✓ Preprocessor with {len(preprocessor.named_transformers_)} transformers")
except Exception as e:
    print(f"   ✗ Error: {e}")
    sys.exit(1)

# Test model instantiation
print("\n4. Testing model instantiation...")
models = ["Random Forest", "Logistic Regression", "XGBoost", "ExplainableBoostingClassifier"]
for model_name in models:
    try:
        model = get_model_instance(model_name)
        print(f"   ✓ {model_name}: {model.__class__.__name__}")
    except Exception as e:
        print(f"   ✗ {model_name}: {e}")

# Test model training (just Random Forest to keep it quick)
print("\n5. Testing model training (Random Forest)...")
try:
    import warnings
    warnings.filterwarnings('ignore')
    X_train, X_val, X_test, y_train, y_val, y_test, preprocessor = prepare_first_trimester_data(df, "first_trimester")
    trained_model, cv_results = train_model_with_hyperparameter_tuning(
        X_train, y_train, "Random Forest"
    )  
    cv_scores = cv_results['mean_test_score']
    print(f"   ✓ Model type: {type(trained_model).__name__}")
    print(f"   ✓ Mean CV AUROC: {np.mean(cv_scores):.4f}")
except Exception as e:
    print(f"   ✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test evaluation
print("\n6. Testing model evaluation...")
try:
    metrics, predictions = evaluate_model(trained_model, X_test, y_test, "Random Forest")
    print(f"   ✓ Metrics calculated: {len(metrics)} metrics")
    print(f"   ✓ AUROC: {metrics['auroc']:.4f}")
    print(f"   ✓ Sensitivity: {metrics['sensitivity']:.4f}")
    print(f"   ✓ Specificity: {metrics['specificity']:.4f}")
except Exception as e:
    print(f"   ✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 80)
print("✓ ALL COMPONENTS VERIFIED SUCCESSFULLY")
print("=" * 80)
