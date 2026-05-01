"""
scripts/recalculate_metrics.py
-------------------------------
Recomputes summary metrics from an existing raw_results.csv without re-running
the full benchmark.  Useful for applying updated metric logic to prior results.
The new summary_metrics.json is written next to the input CSV.

Usage:
    python scripts/recalculate_metrics.py --input-csv results/experiment/raw_results.csv
"""
import os
import sys
import argparse
import json
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.metrics import MetricsCalculator

def main(args):
    """Loads a raw_results.csv file and recalculates the summary metrics."""
    if not os.path.exists(args.input_csv):
        print(f"ERROR: Input file not found at {args.input_csv}")
        return

    print(f"Loading raw results from: {args.input_csv}")
    df = pd.read_csv(args.input_csv)

    print("Recalculating summary metrics...")
    calculator = MetricsCalculator()
    summary_metrics = calculator.calculate_summary_metrics(df)

    output_path = os.path.join(os.path.dirname(args.input_csv), "summary_metrics.json")
    with open(output_path, 'w') as f:
        json.dump(summary_metrics, f, indent=4)

    print(f"New summary metrics saved to: {output_path}")
    print("\n--- New Summary Metrics ---")
    print(json.dumps(summary_metrics, indent=4))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Recalculate metrics from a raw_results.csv file.")
    parser.add_argument(
        "--input-csv",
        type=str,
        required=True,
        help="Path to the raw_results.csv file to process."
    )
    args = parser.parse_args()
    main(args)