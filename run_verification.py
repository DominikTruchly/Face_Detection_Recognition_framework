"""
run_verification.py
-------------------
Evaluates face recognition models on image datasets.

Two operating modes:
  - Annotated mode  : requires a pairs protocol CSV with ground-truth labels.
                      Computes accuracy, precision, recall, F1, TAR@FAR, and
                      (if age_gap column is present) per-age-gap accuracy breakdown.
                      Saves summary_metrics.json alongside raw_results.csv.
  - Exploration mode: no protocol CSV needed. Runs verification on every possible
                      pair inside --img_dir and saves raw distances only.

The pairs CSV must have at minimum these columns (tab or comma separated):
    file1      - path to first image, relative to --img_dir
    file2      - path to second image, relative to --img_dir
    is_same    - 1 if same person, 0 if different
  Optional column:
    age_gap    - integer age difference (used for agedb breakdowns; ignored otherwise)

Built-in dataset configs (lfw / agedb / cfp_ff / cfp_fp) can still be used by
passing --dataset; their paths can be overridden with --img_dir / --pairs_file.

Usage examples:
  # Single pair (no CSV needed)
  python run_verification.py --recognizer ArcFace --detector retinaface \\
      --img1 photo1.jpg --img2 photo2.jpg

  # Custom dataset with ground truth
  python run_verification.py --recognizer ArcFace --detector retinaface \\
      --img_dir my_dataset/images/ --pairs_file my_dataset/pairs.csv

  # Built-in dataset
  python run_verification.py --recognizer ArcFace --detector retinaface --dataset agedb

  # Exploration mode (no ground truth) - returns raw distances
  python run_verification.py --recognizer ArcFace --detector retinaface \\
      --img_dir my_dataset/images/
"""
import os
import argparse
import json
import sys
import glob
import csv
import time
import itertools

import pandas as pd
from tqdm import tqdm

sys.path.append(os.getcwd())

try:
    from src.recognizer_deepface import DeepFaceRecognizer
    from src.recognizer_vit import ViTRecognizer, HFViTRecognizer, TimmFTRecognizer
    from src.recognizer_insightface import InsightFaceRecognizer, InsightFaceCustomRecognizer
    from src.evaluation import Evaluator
    from src.recognizer_swin import SwinFaceRecognizer
except ImportError as e:
    print(f"Import Error: {e}. Make sure you are running this from the root project directory.")

IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')

# --- Built-in dataset configs (Colab paths) ---
_COLAB_BASE = "/content/drive/MyDrive/masters_thesis/face_detection_recognition"

BUILTIN_DATASETS = {
    "lfw": {
        "path": os.path.join(_COLAB_BASE, "datasets/lfw_kaggle/lfw-deepfunneled"),
        "protocol_type": "lfw_csv",
        "match_pairs_file": os.path.join(_COLAB_BASE, "datasets/lfw_kaggle/matchpairsDevTest.csv"),
        "mismatch_pairs_file": os.path.join(_COLAB_BASE, "datasets/lfw_kaggle/mismatchpairsDevTest.csv"),
    },
    "agedb": {
        "path": os.path.join(_COLAB_BASE, "datasets/AgeDB/images"),
        "protocol_type": "generic_csv",
        "pairs_file": os.path.join(_COLAB_BASE, "datasets/AgeDB/agedb_uniform_pairs.csv"),
    },
    "cfp_ff": {
        "path": os.path.join(_COLAB_BASE, "datasets/cfp/Data/Images"),
        "protocol_type": "generic_csv",
        "pairs_file": os.path.join(_COLAB_BASE, "datasets/cfp/cfp_ff_pairs_1000.csv"),
    },
    "cfp_fp": {
        "path": os.path.join(_COLAB_BASE, "datasets/cfp/Data/Images"),
        "protocol_type": "generic_csv",
        "pairs_file": os.path.join(_COLAB_BASE, "datasets/cfp/cfp_fp_pairs_2000.csv"),
    },
}


def run_single_pair(recognizer_instance, img1, img2, recognizer_name, detector):
    """Verifies a single image pair and prints the result."""
    result = recognizer_instance.verify_pair(img1, img2, recognizer_name, detector)
    if result is None:
        print("Could not process the pair (face detection may have failed).")
        return
    distance = result.get('distance', -1.0)
    threshold = result.get('threshold', None)
    verified = result.get('verified', distance <= threshold if threshold is not None else None)
    print(f"\n--- Single Pair Result ---")
    print(f"  Image 1  : {img1}")
    print(f"  Image 2  : {img2}")
    if threshold is not None:
        print(f"  Distance : {distance:.4f}  (threshold: {threshold})")
        print(f"  Same person: {'YES' if verified else 'NO'}")
    else:
        print(f"  Distance : {distance:.4f}  (no threshold — lower = more similar)")


def discover_images(img_dir):
    """Recursively finds all image files under img_dir, returning relative paths."""
    found = []
    for ext in IMAGE_EXTENSIONS:
        found.extend(glob.glob(os.path.join(img_dir, '**', f'*{ext}'), recursive=True))
        found.extend(glob.glob(os.path.join(img_dir, '**', f'*{ext.upper()}'), recursive=True))
    return sorted(set(os.path.relpath(p, img_dir) for p in found))


def run_exploration(recognizer_instance, img_dir, recognizer_name, detector, output_dir):
    """
    Exploration mode: computes pairwise distances for all image combinations
    in img_dir. No ground truth — results are saved to raw_results.csv only.
    """
    image_files = discover_images(img_dir)
    pairs = list(itertools.combinations(image_files, 2))
    print(f"Discovered {len(image_files)} images → {len(pairs)} pairs (exploration mode).")

    raw_results_path = os.path.join(output_dir, "raw_results.csv")
    header = ['image1', 'image2', 'distance', 'processing_time_s']

    processed = set()
    file_exists = os.path.exists(raw_results_path) and os.path.getsize(raw_results_path) > 0
    if file_exists:
        existing_df = pd.read_csv(raw_results_path, on_bad_lines='skip')
        for _, row in existing_df.iterrows():
            processed.add((row['image1'], row['image2']))
        print(f"Resuming: {len(processed)} pairs already done.")

    with open(raw_results_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=header)
        if not file_exists:
            writer.writeheader()

        for img1, img2 in tqdm(pairs):
            if (img1, img2) in processed or (img2, img1) in processed:
                continue
            path1 = os.path.join(img_dir, img1)
            path2 = os.path.join(img_dir, img2)
            start = time.time()
            result = recognizer_instance.verify_pair(path1, path2, recognizer_name, detector)
            elapsed = time.time() - start
            distance = result.get('distance', -1.0) if result else -1.0
            writer.writerow({'image1': img1, 'image2': img2, 'distance': distance, 'processing_time_s': elapsed})
            f.flush()

    print(f"Exploration finished. Raw results saved to {raw_results_path}")


def build_recognizer(name):
    """Instantiates the correct recognizer class from a model name string."""
    if name == 'ViT_timm':
        return ViTRecognizer()
    elif name == 'ViT_hf':
        return HFViTRecognizer()
    elif name == 'ViT_timm_ft':
        return TimmFTRecognizer()
    elif name == 'InsightFace':
        return InsightFaceRecognizer()
    elif name == 'InsightFace_Custom':
        return InsightFaceCustomRecognizer()
    elif name == 'SwinFace':
        return SwinFaceRecognizer()
    else:
        return DeepFaceRecognizer()


def main(args):
    # --- Single-pair shortcut ---
    if args.img1 and args.img2:
        recognizer_instance = build_recognizer(args.recognizer)
        run_single_pair(recognizer_instance, args.img1, args.img2, args.recognizer, args.detector)
        return

    # --- Resolve dataset config ---
    if args.dataset and args.dataset in BUILTIN_DATASETS:
        dataset_config = dict(BUILTIN_DATASETS[args.dataset])
        # Allow CLI overrides of built-in paths
        if args.img_dir:
            dataset_config['path'] = args.img_dir
        if args.pairs_file:
            dataset_config['pairs_file'] = args.pairs_file
        dataset_name = args.dataset
        annotated_mode = True
    elif args.img_dir and args.pairs_file:
        dataset_config = {
            'path': args.img_dir,
            'protocol_type': 'generic_csv',
            'pairs_file': args.pairs_file,
        }
        dataset_name = os.path.basename(args.img_dir.rstrip('/\\'))
        annotated_mode = True
    elif args.img_dir:
        # Exploration mode — no ground truth
        dataset_config = {'path': args.img_dir, 'protocol_type': None}
        dataset_name = os.path.basename(args.img_dir.rstrip('/\\'))
        annotated_mode = False
    else:
        raise ValueError("Provide either --dataset, or --img_dir (optionally with --pairs_file).")

    experiment_name = f"{dataset_name}_{args.detector}_{args.recognizer}"
    output_dir = os.path.join(args.output_dir, experiment_name)
    os.makedirs(output_dir, exist_ok=True)

    print(f"Starting recognition benchmark")
    print(f"  Recognizer     : {args.recognizer}")
    print(f"  Detector       : {args.detector}")
    print(f"  Image directory: {dataset_config['path']}")
    print(f"  Pairs file     : {dataset_config.get('pairs_file', 'None (exploration mode)')}")
    print(f"  Results dir    : {output_dir}")

    recognizer_instance = build_recognizer(args.recognizer)

    # --- Run benchmark ---
    if not annotated_mode:
        run_exploration(recognizer_instance, dataset_config['path'], args.recognizer, args.detector, output_dir)
        return

    evaluator = Evaluator(
        dataset_name=dataset_name,
        dataset_config=dataset_config,
        detector_backend=args.detector,
        recognizer_model=args.recognizer,
        recognizer_instance=recognizer_instance,
    )

    print("\n=== Running Evaluation ===")
    _, summary_metrics = evaluator.run_verification_benchmark(output_dir=output_dir)

    print(f"\nBenchmark finished. Full results: {os.path.join(output_dir, 'raw_results.csv')}")
    if summary_metrics is not None:
        summary_metrics_path = os.path.join(output_dir, "summary_metrics.json")
        with open(summary_metrics_path, 'w') as f:
            json.dump(summary_metrics, f, indent=4)
        print(f"Summary metrics saved to {summary_metrics_path}")
        print("\n--- Summary Metrics ---")
        print(json.dumps(summary_metrics, indent=4))
    else:
        print("No ground truth provided — raw distances saved, no metrics computed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run Face Recognition Benchmark.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--recognizer", type=str, required=True,
        choices=['VGG-Face', 'Facenet', 'Facenet512', 'ArcFace',
                 'ViT_timm', 'ViT_hf', 'ViT_timm_ft',
                 'InsightFace', 'InsightFace_Custom', 'SwinFace'],
        help="Face recognition model to evaluate.",
    )
    parser.add_argument(
        "--detector", type=str, required=True,
        choices=['opencv', 'retinaface', 'mtcnn', 'skip'],
        help="Face detector backend. Use 'skip' for models with built-in detection (InsightFace, SwinFace).",
    )
    parser.add_argument(
        "--img1", type=str, default=None,
        help="Path to the first image for a direct single-pair comparison.",
    )
    parser.add_argument(
        "--img2", type=str, default=None,
        help="Path to the second image for a direct single-pair comparison.",
    )
    parser.add_argument(
        "--dataset", type=str, default=None,
        choices=list(BUILTIN_DATASETS.keys()),
        help="Use a built-in dataset config. Paths can still be overridden with --img_dir / --pairs_file.",
    )
    parser.add_argument(
        "--img_dir", type=str, default=None,
        help="Path to the root image directory. Image paths in the pairs file are resolved relative to this.",
    )
    parser.add_argument(
        "--pairs_file", type=str, default=None,
        help="Path to the pairs protocol CSV (columns: file1, file2, is_same [, age_gap]). "
             "If omitted together with --dataset, runs in exploration mode (all pairs, no metrics).",
    )
    parser.add_argument(
        "--output_dir", type=str, default="results",
        help="Directory where result sub-folders will be created.",
    )
    parser.add_argument(
        "--models_dir", type=str, default=None,
        help="Optional path for DeepFace model weights cache (sets DEEPFACE_HOME).",
    )
    args = parser.parse_args()

    if args.models_dir:
        os.environ['DEEPFACE_HOME'] = args.models_dir
        os.makedirs(args.models_dir, exist_ok=True)

    main(args)