"""
scripts/generate_agedb_protocol.py
-----------------------------------
Generates a balanced AgeDB verification protocol CSV.
Pairs are sampled uniformly across age-gap bins so that no single age gap
dominates the evaluation.  Both positive (same identity) and negative
(different identity) pairs carry an age_gap column for per-bin analysis.

Usage:
    python scripts/generate_agedb_protocol.py \\
        -i path/to/agedb_images/ \\
        -o path/to/agedb_pairs.csv
"""
import argparse
import random
from pathlib import Path
from tqdm import tqdm
import os
import collections
import numpy as np
import pandas as pd

def get_image_info(all_files):
    """Parses all filenames and returns a list of (name, age, file_stem) tuples."""
    image_info = []
    for f in all_files:
        try:
            parts = f.stem.split("_")
            name = parts[1]
            age = int(parts[2])
            image_info.append((name, age, f.stem)) # Use stem (no .jpg)
        except (IndexError, ValueError):
            print(f"Skipping malformed file: {f.name}")
    return image_info

def create_pairs(image_info, num_pairs_per_bin=300, bin_size=5):
    """Creates balanced bins of positive and negative pairs. Returns a list of tuples: (file1, file2, is_same, age_gap)"""
    people_to_images = collections.defaultdict(list)
    for name, age, file_stem in image_info:
        people_to_images[name].append((age, file_stem))

    pos_bins = collections.defaultdict(list)
    neg_bins = collections.defaultdict(list)
    all_names = list(people_to_images.keys())
    
    print("Generating positive pairs...")
    for name in tqdm(people_to_images):
        images = people_to_images[name]
        if len(images) < 2: continue
        
        for i in range(len(images)):
            for j in range(i + 1, len(images)):
                age1, file1 = images[i]
                age2, file2 = images[j]
                age_gap = abs(age1 - age2)
                bin_key = (age_gap // bin_size) * bin_size
                # Save the age gap
                pos_bins[bin_key].append((file1, file2, age_gap))

    print("Generating negative pairs...")
    target_neg_pairs = len(pos_bins) * num_pairs_per_bin * 2 # Oversample
    
    while sum(len(v) for v in neg_bins.values()) < target_neg_pairs:
        name1, name2 = random.sample(all_names, 2)
        img1, img2 = random.choice(people_to_images[name1]), random.choice(people_to_images[name2])
        age1, file1 = img1
        age2, file2 = img2
        age_gap = abs(age1 - age2)
        bin_key = (age_gap // bin_size) * bin_size
        # Save the age gap
        neg_bins[bin_key].append((file1, file2, age_gap))

    final_pairs = []
    for bin_key in sorted(pos_bins.keys()):
        pairs_in_bin = pos_bins[bin_key]
        random.shuffle(pairs_in_bin)
        sampled_pairs = pairs_in_bin[:num_pairs_per_bin]
        for p1, p2, gap in sampled_pairs:
            final_pairs.append((f"{p1}.jpg", f"{p2}.jpg", 1, gap))
        
    for bin_key in sorted(neg_bins.keys()):
        pairs_in_bin = neg_bins[bin_key]
        random.shuffle(pairs_in_bin)
        sampled_pairs = pairs_in_bin[:num_pairs_per_bin]
        for p1, p2, gap in sampled_pairs:
            final_pairs.append((f"{p1}.jpg", f"{p2}.jpg", 0, gap))

    return final_pairs

def main(input_folder: Path, out_file: Path):
    random.seed(42)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    all_files = list(input_folder.glob("*.jpg"))
    
    image_info = get_image_info(all_files)
    final_pairs = create_pairs(image_info)
    random.shuffle(final_pairs)

    print(f"Saving {len(final_pairs)} total pairs to {out_file}...")
    with open(out_file, "w") as fid:
        fid.write("file1,file2,is_same,age_gap\n")
        for file1, file2, is_same, age_gap in final_pairs:
            fid.write(f"{file1},{file2},{is_same},{age_gap}\n")
            
    print("✅ Protocol file generated successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a balanced AgeDB protocol file.")
    parser.add_argument("-i", "--input_folder", type=Path, help="Path to folder containing AgeDB images")
    parser.add_argument("-o", "--output_file", type=Path, help="Path to the output pairs file (e.g., agedb_uniform_pairs.csv)")
    args = parser.parse_args()
    main(args.input_folder, args.output_file)