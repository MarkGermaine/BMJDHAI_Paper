"""
Main entry point for GDM Prediction Pipeline.

Usage:
    python run_pipeline.py                                    # Run with defaults (small dummy data)
    python run_pipeline.py --data data/dummy_data_large.csv   # Run with large dataset
    python run_pipeline.py --cohorts first_trimester          # Run specific cohorts only
"""

import argparse
import os
import sys
from datetime import datetime
import joblib

from src.preprocessing import load_data
from src.pipelines.first_trimester import run_first_trimester_pipeline
from src.pipelines.multiparous import run_multiparous_pipeline
from src.models import save_model_results


def main():
    parser = argparse.ArgumentParser(
        description="GDM Prediction Pipeline - Train and evaluate ML models for gestational diabetes prediction"
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/dummy_data_small.csv",
        help="Path to input CSV file (default: data/dummy_data_small.csv)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output",
        help="Output directory for results and visualizations (default: output)",
    )
    parser.add_argument(
        "--cohorts",
        type=str,
        nargs="+",
        choices=["first_trimester", "nulliparous", "multiparous", "past_pregnancy", "all"],
        default=["all"],
        help="Cohorts to process (default: all)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.data):
        print(f"❌ Error: Data file not found: {args.data}")
        sys.exit(1)
    
    os.makedirs(args.output, exist_ok=True)
    
    # Parse cohort selection
    run_all = "all" in args.cohorts
    run_ft = run_all or "first_trimester" in args.cohorts or "nulliparous" in args.cohorts
    run_mp = run_all or "multiparous" in args.cohorts or "past_pregnancy" in args.cohorts
    
    # Load data
    print("\n" + "="*80)
    print("GDM PREDICTION PIPELINE")
    print("="*80)
    df = load_data(args.data)
    
    all_results = {}
    
    # Run first-trimester pipeline
    if run_ft:
        print("\n[1/2] Running First-Trimester Pipeline...")
        ft_results = run_first_trimester_pipeline(df, output_dir=args.output)
        all_results.update(ft_results)
    
    # Run multiparous pipeline
    if run_mp:
        print("\n[2/2] Running Multiparous Pipeline...")
        mp_results = run_multiparous_pipeline(df, output_dir=args.output)
        all_results.update(mp_results)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = os.path.join(args.output, f"model_results_{timestamp}.joblib")
    save_model_results(all_results, results_path)
    
    # Print final summary
    print("\n" + "="*80)
    print("PIPELINE COMPLETE")
    print("="*80)
    print(f"✓ Results saved to: {results_path}")
    print(f"✓ Visualizations saved to: {args.output}")
    print(f"\nCohorts processed: {', '.join(all_results.keys())}")
    print(f"Algorithms trained per cohort: {len(list(all_results.values())[0])}")
    print("\nTo load and inspect results:")
    print(f"  import joblib")
    print(f"  results = joblib.load('{results_path}')")


if __name__ == "__main__":
    main()
