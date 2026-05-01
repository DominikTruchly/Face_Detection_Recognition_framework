"""
run_detection.py
----------------
Evaluates face detection models on image datasets.

Two operating modes:
  - Annotated mode  : requires a ground-truth annotation file in WIDER FACE format.
                      Computes mAP and per-size recall, saves summary_metrics.json.
  - Exploration mode: no annotations needed. Runs detection on all images found
                                            under --img_dir, or on a single image file, and saves raw
                                            predictions only.

In both modes the raw per-image predictions are written to raw_predictions.csv.
With --save_faces, images with bounding boxes drawn on them are saved alongside.

Usage examples:
  # Annotated dataset (full benchmark)
  python run_detection.py --detector retinaface \\
      --img_dir datasets/WIDER_val/images \\
      --annotation_file wider_face_val_bbx_gt.txt

  # Unannotated dataset (exploration)
  python run_detection.py --detector retinaface \\
      --img_dir my_images/ --save_faces

  # Single image
  python run_detection.py --detector retinaface \
      --img_dir my_images/example.jpg
"""
import os
import sys

# DEEPFACE_HOME must be set before DeepFace is imported (it reads the env var at import time).
# Do a quick pre-parse of --models_dir so we can set it early.
def _set_deepface_home_early():
    for i, arg in enumerate(sys.argv):
        if arg == '--models_dir' and i + 1 < len(sys.argv):
            path = sys.argv[i + 1]
            os.environ['DEEPFACE_HOME'] = path
            os.makedirs(path, exist_ok=True)
            return
_set_deepface_home_early()

import glob
import argparse
import json
import csv
import torch
import cv2
from tqdm import tqdm
from src.detection import Detector
from src.metrics import DetectionMetricsCalculator

IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')


def load_widerface_annotations(txt_path):
    """
    Parses the WIDER FACE ground truth annotation file.

    Expected file format (wider_face_val_bbx_gt.txt):
    -----------------------------------------------
    The file lists entries for each image sequentially. Each entry has:

      1. A line with the relative image path (e.g. "0--Parade/image.jpg")
      2. A line with the integer number of faces in that image (e.g. "5")
      3. One line per face with 10 space-separated integer fields:
            x  y  w  h  blur  expression  illumination  invalid  occlusion  pose

         - x, y : top-left corner of the bounding box in pixels
         - w, h : width and height of the bounding box in pixels
         - blur        : blur level (0=clear, 1=normal, 2=heavy)
         - expression  : 0=typical, 1=exaggerate
         - illumination: 0=normal, 1=extreme
         - invalid     : 0=valid face, 1=invalid (heavily occluded / not a face)
         - occlusion   : 0=no, 1=partial, 2=heavy
         - pose        : 0=typical, 1=atypical

      If num_faces is 0, a single dummy line "0 0 0 0 0 0 0 0 0 0" follows.

    Faces marked as invalid (invalid==1) or with a negative dimension are
    treated as "crowd" regions and excluded from mAP evaluation.

    Returns:
        dict mapping relative image path -> {'boxes': [[x,y,w,h], ...], 'iscrowd': [0/1, ...]}
    """
    print(f"Parsing WIDER FACE annotations from: {txt_path}")
    ground_truth_map = {}

    with open(txt_path, 'r') as f:
        lines = f.readlines()

    IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff', '.tif'}

    i = 0
    while i < len(lines):
        image_name = lines[i].strip()
        if not any(image_name.lower().endswith(ext) for ext in IMAGE_SUFFIXES):
            i += 1
            continue

        i += 1
        num_faces = int(lines[i].strip())
        i += 1

        boxes = []
        iscrowd_list = []

        if num_faces == 0:
            i += 1  # skip the single dummy annotation line
            ground_truth_map[image_name] = {'boxes': [], 'iscrowd': []}
            continue

        for j in range(num_faces):
            parts = lines[i + j].strip().split()
            x, y, w, h = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
            invalid = int(parts[7])

            is_ignored = 1 if (invalid == 1 or w < 0 or h < 0) else 0

            if w < 0: w = 0
            if h < 0: h = 0

            boxes.append([x, y, w, h])
            iscrowd_list.append(is_ignored)

        ground_truth_map[image_name] = {'boxes': boxes, 'iscrowd': iscrowd_list}
        i += num_faces

    print(f"Successfully parsed {len(ground_truth_map)} images.")
    return ground_truth_map


def discover_images(img_dir):
    """Recursively finds all image files under img_dir."""
    found = []
    for ext in IMAGE_EXTENSIONS:
        found.extend(glob.glob(os.path.join(img_dir, '**', f'*{ext}'), recursive=True))
        found.extend(glob.glob(os.path.join(img_dir, '**', f'*{ext.upper()}'), recursive=True))
    # Return paths relative to img_dir so they can be used as keys
    return sorted(set(os.path.relpath(p, img_dir) for p in found))


def resolve_image_source(img_path):
    """Normalizes --img_dir to an image root plus an optional single image name."""
    if os.path.isfile(img_path):
        image_root = os.path.dirname(img_path) or "."
        image_name = os.path.basename(img_path)
        dataset_name = os.path.splitext(image_name)[0]
        return image_root, image_name, dataset_name

    if os.path.isdir(img_path):
        dataset_name = os.path.basename(img_path.rstrip("/\\"))
        return img_path, None, dataset_name

    raise FileNotFoundError(f"Input path does not exist: {img_path}")


def draw_and_save(img_path, pred_results, save_path):
    """Draws bounding boxes on a copy of the image and saves it."""
    img = cv2.imread(img_path)
    if img is None:
        return

    for res in pred_results:
        if not isinstance(res, dict) or 'facial_area' not in res:
            continue
        box = res['facial_area']
        x, y, w, h = box['x'], box['y'], box['w'], box['h']
        conf = res.get('confidence', 0.0)

        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        label = f"{conf:.2f}" if conf else ""
        if label:
            cv2.putText(img, label, (x, max(y - 6, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, img)


def main(args):
    image_root, single_image_name, dataset_name = resolve_image_source(args.img_dir)
    experiment_name = f"{dataset_name}_{args.detector}"
    output_dir = os.path.join(args.output_dir, experiment_name)
    os.makedirs(output_dir, exist_ok=True)

    annotated_mode = args.annotation_file is not None

    print(f"Starting detection benchmark")
    print(f"  Detector       : {args.detector}")
    print(f"  Image source   : {args.img_dir}")
    print(f"  Annotations    : {args.annotation_file if annotated_mode else 'None (exploration mode)'}")
    print(f"  Save faces     : {args.save_faces}")
    print(f"  Results dir    : {output_dir}")

    detector = Detector()

    # --- Build image list and ground truth ---
    if annotated_mode:
        if single_image_name is not None:
            raise ValueError(
                "Single-image input is supported only in exploration mode. "
                "For annotated evaluation, pass the dataset root directory via --img_dir "
                "so annotation paths can be resolved correctly."
            )

        ground_truth_map = load_widerface_annotations(args.annotation_file)
        image_files = list(ground_truth_map.keys())
        metrics = DetectionMetricsCalculator()
    else:
        if single_image_name is not None:
            image_files = [single_image_name]
        else:
            image_files = discover_images(image_root)
        print(f"Discovered {len(image_files)} images (no annotations).")
        ground_truth_map = {}
        metrics = None

    # --- CSV header differs slightly between modes ---
    raw_results_csv = os.path.join(output_dir, "raw_predictions.csv")
    print(f"Logging raw predictions to: {raw_results_csv}")

    csv_header = ["image_name", "pred_x", "pred_y", "pred_w", "pred_h", "confidence"]
    if annotated_mode:
        csv_header.append("gt_faces_count")

    with open(raw_results_csv, 'w', newline='') as f:
        csv.writer(f).writerow(csv_header)

    if args.save_faces:
        annotated_img_dir = os.path.join(output_dir, "visualized_images")
        os.makedirs(annotated_img_dir, exist_ok=True)
        print(f"Saving visualized images to: {annotated_img_dir}")

    print(f"\n=== Running on {len(image_files)} images ===")
    for img_name in tqdm(image_files):
        img_path = os.path.join(image_root, img_name)
        if not os.path.exists(img_path):
            continue

        pred_results = detector.detect_faces(img_path, args.detector)

        if args.save_faces:
            save_path = os.path.join(annotated_img_dir, img_name)
            draw_and_save(img_path, pred_results, save_path)

        with open(raw_results_csv, 'a', newline='') as f:
            writer = csv.writer(f)
            gt_count = len(ground_truth_map.get(img_name, {}).get('boxes', []))

            if not pred_results:
                row = [img_name, "", "", "", "", ""]
                if annotated_mode:
                    row.append(gt_count)
                writer.writerow(row)
            else:
                for res in pred_results:
                    if 'facial_area' not in res:
                        continue
                    box = res['facial_area']
                    conf = res.get('confidence', 0.0)
                    row = [img_name, box['x'], box['y'], box['w'], box['h'], conf]
                    if annotated_mode:
                        row.append(gt_count)
                    writer.writerow(row)

        if annotated_mode:
            gt_data = ground_truth_map[img_name]
            metrics.update(pred_results, gt_data['boxes'], gt_data['iscrowd'])

    # --- Summary metrics (only when ground truth is available) ---
    if annotated_mode:
        print("Benchmark loop finished. Calculating summary metrics...")
        final_metrics = metrics.compute()

        final_metrics_clean = {}
        for key, val in final_metrics.items():
            if isinstance(val, torch.Tensor):
                final_metrics_clean[key] = round(val.item(), 4)
            elif isinstance(val, dict):
                final_metrics_clean[key] = val
            else:
                final_metrics_clean[key] = val

        summary_metrics_path = os.path.join(output_dir, "summary_metrics.json")
        with open(summary_metrics_path, 'w') as f:
            json.dump(final_metrics_clean, f, indent=4)

        print(f"Benchmark finished. Summary metrics saved to {summary_metrics_path}")
        print("\n--- Summary Metrics (mAP) ---")
        print(json.dumps(final_metrics_clean, indent=4))
    else:
        print(f"Exploration finished. Raw predictions saved to {raw_results_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run Face Detection Benchmark.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--detector",
        type=str,
        required=True,
        choices=['opencv', 'retinaface', 'mtcnn', 'yolov8n', 'ssd', 'dlib', 'mediapipe', 'centerface'],
        help="Face detector model to evaluate."
    )
    parser.add_argument(
        "--img_dir",
        type=str,
        required=True,
        help="Path to the root image directory, or to a single image file when running without annotations. "
             "Annotation file paths are resolved relative to the dataset root directory."
    )
    parser.add_argument(
        "--annotation_file",
        type=str,
        default=None,
        help="Path to the ground-truth annotation .txt file in WIDER FACE format. "
             "Omit this argument to run in exploration mode (no metrics, no ground truth required)."
    )
    parser.add_argument(
        "--save_faces",
        action="store_true",
        help="Save a copy of each image with detected bounding boxes drawn on it "
             "into <output_dir>/<experiment>/visualized_images/. Works in both annotated and exploration modes."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results",
        help="Directory where result sub-folders will be created."
    )
    parser.add_argument(
        "--models_dir",
        type=str,
        default=None,
        help="Optional path to a directory where DeepFace will store/look for downloaded models. Sets DEEPFACE_HOME."
    )
    args = parser.parse_args()

    if args.models_dir:
        # Already set before imports; just ensure the dir exists.
        os.makedirs(args.models_dir, exist_ok=True)

    main(args)