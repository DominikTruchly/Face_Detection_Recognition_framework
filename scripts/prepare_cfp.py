"""
scripts/prepare_cfp.py
-----------------------
Builds balanced verification-pair CSVs for the CFP (Celebrities in Frontal-Profile) dataset.
Samples a fixed number of same/different pairs from each of 10 official protocol folds
for both the Frontal-Frontal (FF) and Frontal-Profile (FP) evaluation splits.

Usage:
    python scripts/prepare_cfp.py --dataset_dir path/to/cfp
    python scripts/prepare_cfp.py --dataset_dir path/to/cfp --pairs_ff 500 --pairs_fp 1000
"""
import argparse
import os
import random

import pandas as pd
from tqdm import tqdm

CSV_HEADER = ["file1", "file2", "is_same"]

def load_pair_map(filepath):
    """Loads the 1-based index map file."""
    print(f"  Loading map file: {os.path.basename(filepath)}")
    mapping = {}
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()
        for i, line in enumerate(lines):
            index = i + 1
            image_path_raw = line.strip().split()[-1]
            
            if "Data/Images/" in image_path_raw:
                image_path_clean = image_path_raw.split("Data/Images/")[-1]
            else:
                image_path_clean = image_path_raw

            mapping[index] = image_path_clean
            
    except FileNotFoundError:
        print(f"  CRITICAL ERROR: Map file not found at {filepath}")
        return None
    print(f"  Loaded {len(mapping)} mappings.")
    return mapping

def process_split_files(split_dir, map1, map2, output_csv_path, target_pairs_per_class):
    """Collects all pairs and samples a balanced subset."""
    all_same_pairs = []
    all_diff_pairs = []
    
    print(f"Processing 10 folds from: {os.path.basename(split_dir)}")

    try:
        for i in tqdm(range(1, 11), desc="  Reading all folds"):
            fold_name = f"{i:02d}"
            same_file = os.path.join(split_dir, fold_name, "same.txt")
            diff_file = os.path.join(split_dir, fold_name, "diff.txt")

            # Process "same" pairs (label = 1)
            with open(same_file, 'r') as f:
                for line in f:
                    try:
                        num1, num2 = line.strip().split(',') # Using comma
                        file1 = map1[int(num1)]
                        file2 = map2[int(num2)]
                        all_same_pairs.append({"file1": file1, "file2": file2, "is_same": 1})
                    except (ValueError, KeyError): pass

            # Process "different" pairs (label = 0)
            with open(diff_file, 'r') as f:
                for line in f:
                    try:
                        num1, num2 = line.strip().split(',') # Using comma
                        file1 = map1[int(num1)]
                        file2 = map2[int(num2)]
                        all_diff_pairs.append({"file1": file1, "file2": file2, "is_same": 0})
                    except (ValueError, KeyError): pass

    except FileNotFoundError:
        print(f"CRITICAL ERROR: Could not find fold directories in {split_dir}")
        return
        
    print(f"  Found {len(all_same_pairs)} total 'same' pairs and {len(all_diff_pairs)} total 'diff' pairs.")

    print(f"  Sampling {target_pairs_per_class} 'same' and {target_pairs_per_class} 'diff' pairs...")
    
    if len(all_same_pairs) < target_pairs_per_class:
        sampled_same = all_same_pairs
    else:
        random.shuffle(all_same_pairs)
        sampled_same = all_same_pairs[:target_pairs_per_class]

    if len(all_diff_pairs) < target_pairs_per_class:
        sampled_diff = all_diff_pairs
    else:
        random.shuffle(all_diff_pairs)
        sampled_diff = all_diff_pairs[:target_pairs_per_class]

    final_pairs_list = sampled_same + sampled_diff
    random.shuffle(final_pairs_list)

    df = pd.DataFrame(final_pairs_list, columns=CSV_HEADER)
    df.to_csv(output_csv_path, index=False)
    print(f"Successfully created: {os.path.basename(output_csv_path)} with {len(df)} total pairs.")


def main(args):
    dataset_dir = args.dataset_dir
    protocol_dir = os.path.join(dataset_dir, "Protocol")
    map_f_path = os.path.join(protocol_dir, "Pair_list_F.txt")
    map_p_path = os.path.join(protocol_dir, "Pair_list_P.txt")
    split_dir_ff = os.path.join(protocol_dir, "Split/FF")
    split_dir_fp = os.path.join(protocol_dir, "Split/FP")
    output_csv_ff = os.path.join(dataset_dir, f"cfp_ff_pairs_{args.pairs_ff * 2}.csv")
    output_csv_fp = os.path.join(dataset_dir, f"cfp_fp_pairs_{args.pairs_fp * 2}.csv")

    print("--- Starting CFP Dataset Preparation (Sampled Version) ---")
    
    print("Step 1: Loading pair maps...")
    map_f = load_pair_map(map_f_path)
    map_p = load_pair_map(map_p_path)
    if not map_f or not map_p:
        print("Aborting due to missing map files.")
        return

    print(f"\nStep 2: Processing Frontal-Frontal (FF) pairs (Target: {args.pairs_ff * 2} pairs)...")
    process_split_files(split_dir_ff, map_f, map_f, output_csv_ff, args.pairs_ff)

    print(f"\nStep 3: Processing Frontal-Profile (FP) pairs (Target: {args.pairs_fp * 2} pairs)...")
    process_split_files(split_dir_fp, map_f, map_p, output_csv_fp, args.pairs_fp)

    print("\n--- All Done. ---")
    print(f"Your pair files are ready:\n1. {output_csv_ff}\n2. {output_csv_fp}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare the CFP dataset by sampling balanced FF/FP verification pairs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset_dir", required=True, help="Path to the CFP dataset root (must contain a Protocol/ subfolder).")
    parser.add_argument("--pairs_ff", type=int, default=500, help="Same/diff pairs to sample per class for the FF split.")
    parser.add_argument("--pairs_fp", type=int, default=1000, help="Same/diff pairs to sample per class for the FP split.")
    args = parser.parse_args()
    main(args)