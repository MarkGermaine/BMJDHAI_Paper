"""
Generate synthetic dummy data for testing GDM prediction pipeline.

Creates realistic patient data with distributions mimicking clinical datasets.
Generates two sizes: small (100-200 rows for unit tests) and large (1000+ rows).
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import argparse


def generate_dummy_data(n_samples=200, random_seed=42, multiparous=False):
    """
    Generate realistic synthetic patient data for GDM prediction.
    
    Args:
        n_samples: Number of base patients (will be multiplied by pregnancies if multiparous)
        random_seed: Random seed for reproducibility
        multiparous: If True, generate multiple pregnancies per patient for multiparous cohort testing
    
    Returns:
        pd.DataFrame with columns for all cohorts
    """
    np.random.seed(random_seed)
    
    data_list = []
    patient_id = 1
    today = datetime.now()
    
    # Generate patient base data
    n_patients = n_samples // (3 if multiparous else 1)  # Reduce base patients if adding multiple pregnancies
    
    for _ in range(n_patients):
        # Determine number of pregnancies for this patient
        n_pregnancies = np.random.poisson(lam=2.5) + 1 if multiparous else 1
        n_pregnancies = max(1, min(n_pregnancies, 5))  # Limit to 1-5 pregnancies
        
        # Basic patient info (constant across pregnancies)
        dob = today - timedelta(days=int(365 * np.random.uniform(18, 45)))
        ethnicity = np.random.choice(
            ["White", "Black", "Asian", "Mixed", "Other"],
            p=[0.70, 0.10, 0.10, 0.05, 0.05]
        )
        family_diabetes = np.random.binomial(1, 0.25)
        other_endocrine = np.random.binomial(1, 0.05)
        
        # Generate pregnancies for this patient
        for preg_idx in range(n_pregnancies):
            data = {}
            data["ID"] = patient_id
            data["Date of Birth"] = dob.strftime("%Y-%m-%d")
            
            # Booking date for this pregnancy (pregnancies spaced 1-5 years apart)
            booking_date = today - timedelta(days=int(365 * (n_pregnancies - preg_idx - 1) * np.random.uniform(1, 5)))
            data["booking_date"] = booking_date
            
            # Age at booking
            age_at_booking = (booking_date - dob).days // 365
            age_at_booking = max(18, min(age_at_booking, 45))
            data["Age at booking"] = age_at_booking
            
            # Ethnic Origin
            data["Ethnic Origin of Patient"] = ethnicity
            
            # BMI (slightly varies per pregnancy)
            data["BMI"] = np.clip(
                np.random.lognormal(mean=3.1, sigma=0.25) + np.random.normal(0, 0.5),
                15, 50
            )
            
            # Blood pressure
            data["Systolic BP"] = int(np.clip(
                np.random.normal(loc=120, scale=12), 90, 180
            ))
            data["Diastolic BP"] = int(np.clip(
                np.random.normal(loc=75, scale=8), 50, 110
            ))
            
            # History of GDM increases over pregnancies
            if preg_idx > 0 and data_list[len(data_list) - 1]["GDM"] == 1:
                data["Hx_GDM"] = 1  # Had GDM in previous pregnancy
            else:
                data["Hx_GDM"] = np.random.binomial(1, max(0.05, 0.15 - preg_idx * 0.03))
            
            data["FH Diabetes"] = family_diabetes
            data["Other Endocrine problems"] = other_endocrine
            data["Parity"] = preg_idx  # Number of previous pregnancies (0, 1, 2, ...)
            
            # Inter-pregnancy variables (only for pregnancies after first)
            if preg_idx > 0:
                # Get weight from previous pregnancy
                prev_bmi = data_list[-1]["BMI"]
                data["Inter-pregnancy weight change"] = np.random.normal(loc=0.5 - preg_idx * 0.1, scale=2)
                # Inter-pregnancy interval increases with each pregnancy
                data["Inter-pregnancy interval"] = np.clip(
                    np.random.gamma(shape=2, scale=2 + preg_idx),
                    0.5, 10
                )
            else:
                data["Inter-pregnancy weight change"] = 0  # First pregnancy has no change
                data["Inter-pregnancy interval"] = 0
            
            # Previous birth weight percentile (from last pregnancy if exists)
            if preg_idx > 0:
                data["Previous birth weight percentile"] = np.clip(
                    np.random.normal(loc=50, scale=15) + np.random.normal(0, 10),
                    5, 95
                )
            else:
                data["Previous birth weight percentile"] = np.nan  # Unknown for first pregnancy
            
            # GDM outcome - increase risk with parity and age
            gdm_base_prob = 0.12
            gdm_prob = np.clip(
                gdm_base_prob
                + 0.15 * data["Hx_GDM"]
                + 0.05 * family_diabetes
                + 0.02 * (data["BMI"] > 30)
                + 0.05 * other_endocrine
                + 0.02 * (data["Age at booking"] > 35),
                0, 1
            )
            data["GDM"] = np.random.binomial(1, gdm_prob)
            data_list.append(data)
        
        patient_id += 1
    
    df = pd.DataFrame(data_list)
    
    # Sort by ID and booking date
    if "booking_date" in df.columns:
        df = df.sort_values(["ID", "booking_date"])
        df = df.drop(columns=["booking_date"])
    
    return df


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic GDM prediction data")
    parser.add_argument("--small", action="store_true", help="Generate small dataset (100-200 samples)")
    parser.add_argument("--large", action="store_true", help="Generate large dataset (1000+ samples)")
    parser.add_argument("--both", action="store_true", help="Generate both datasets")
    parser.add_argument("--output", type=str, default="data", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    # Default to both
    if not (args.small or args.large):
        args.both = True
    
    import os
    os.makedirs(args.output, exist_ok=True)
    
    if args.small or args.both:
        print("Generating small dataset (200 samples)...")
        df_small = generate_dummy_data(n_samples=200, random_seed=args.seed, multiparous=False)
        output_path = os.path.join(args.output, "dummy_data_small.csv")
        df_small.to_csv(output_path, index=False)
        print(f"✓ Saved to {output_path}")
        print(f"  Shape: {df_small.shape}")
        print(f"  GDM prevalence: {df_small['GDM'].mean():.2%}")
    
    if args.large or args.both:
        print("\nGenerating large dataset (1500 total pregnancies, multiple per patient)...")
        df_large = generate_dummy_data(n_samples=1500, random_seed=args.seed + 1, multiparous=True)
        output_path = os.path.join(args.output, "dummy_data_large.csv")
        df_large.to_csv(output_path, index=False)
        print(f"✓ Saved to {output_path}")
        print(f"  Shape: {df_large.shape}")
        print(f"  Unique patients: {df_large['ID'].nunique()}")
        print(f"  GDM prevalence: {df_large['GDM'].mean():.2%}")


if __name__ == "__main__":
    main()
