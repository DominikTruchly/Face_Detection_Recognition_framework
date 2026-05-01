"""
scripts/recalculate_detection_metrics.py
-----------------------------------------
Recalculates WIDER FACE detection metrics from a saved raw_predictions.csv without
re-running the full benchmark.  Outputs a detailed_summary_metrics.json with overall
recall, size-bin breakdown, and mAP/mAR figures alongside the input file.

Usage:
    python scripts/recalculate_detection_metrics.py \\
        --experiment_dir results/widerface_retinaface \\
        --annotation_file wider_face_val_bbx_gt.txt
"""
import os
import sys
import argparse
import json
import pandas as pd
import torch
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.metrics import DetectionMetricsCalculator

SIZE_BINS = {
    'Small (Hard)':  (0, 20),
    'Medium':        (20, 50),
    'Large (Easy)':  (50, 9999)
}

def load_widerface_annotations(txt_path):
    """Parses the WIDER FACE ground truth .txt file."""
    print(f"Parsing WIDER FACE annotations from {txt_path}...")
    ground_truth_map = {}
    
    with open(txt_path, 'r') as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        image_name = lines[i].strip()
        if not image_name.endswith('.jpg'):
            i += 1
            continue
            
        i += 1
        num_faces = int(lines[i].strip())
        i += 1
        
        boxes = []
        iscrowd_list = []
        
        if num_faces == 0:
            i += 1 
            ground_truth_map[image_name] = {'boxes': [], 'iscrowd': []}
            continue

        for j in range(num_faces):
            parts = lines[i + j].strip().split()
            x, y, w, h = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
            invalid = int(parts[7])
            
            # 1 = ignore/invalid, 0 = valid
            is_ignored = 1 if (invalid == 1 or w < 0 or h < 0) else 0
            if w < 0: w = 0
            if h < 0: h = 0

            boxes.append([x, y, w, h])
            iscrowd_list.append(is_ignored)
        
        ground_truth_map[image_name] = {'boxes': boxes, 'iscrowd': iscrowd_list}
        i += num_faces
        
    print(f"Successfully parsed {len(ground_truth_map)} images.")
    return ground_truth_map

def calculate_iou_hits(pred_boxes, gt_boxes, iou_threshold=0.5):
    """Returns a boolean list of length len(gt_boxes), where True means that specific GT box was found."""
    if len(gt_boxes) == 0:
        return []
    if len(pred_boxes) == 0:
        return [False] * len(gt_boxes)

    p_tensor = torch.tensor(pred_boxes, dtype=torch.float32)
    g_tensor = torch.tensor(gt_boxes, dtype=torch.float32)

    # Convert xywh -> xyxy
    p_x1, p_y1, p_w, p_h = p_tensor.unbind(1)
    p_boxes_xyxy = torch.stack([p_x1, p_y1, p_x1 + p_w, p_y1 + p_h], dim=1)
    
    g_x1, g_y1, g_w, g_h = g_tensor.unbind(1)
    g_boxes_xyxy = torch.stack([g_x1, g_y1, g_x1 + g_w, g_y1 + g_h], dim=1)

    lt = torch.max(p_boxes_xyxy[:, None, :2], g_boxes_xyxy[:, :2])
    rb = torch.min(p_boxes_xyxy[:, None, 2:], g_boxes_xyxy[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]

    area_p = p_tensor[:, 2] * p_tensor[:, 3]
    area_g = g_tensor[:, 2] * g_tensor[:, 3]
    union = area_p[:, None] + area_g - inter
    
    iou_matrix = inter / (union + 1e-6)

    max_iou_per_gt, _ = iou_matrix.max(dim=0)
    
    found_flags = (max_iou_per_gt >= iou_threshold).tolist()
    return found_flags

def main(args):
    ANNOTATION_TXT = args.annotation_file
    EXPERIMENT_DIR = args.experiment_dir
    CSV_PATH = os.path.join(EXPERIMENT_DIR, "raw_predictions.csv")
    
    if not os.path.exists(CSV_PATH):
        print(f"Error: Could not find raw_predictions.csv at {CSV_PATH}")
        return

    gt_map = load_widerface_annotations(ANNOTATION_TXT)
    print(f"Loading predictions from {CSV_PATH}...")
    df = pd.read_csv(CSV_PATH)
    preds_grouped = df.groupby('image_name')

    breakdown_stats = {
        k: {'total': 0, 'found': 0} for k in SIZE_BINS.keys()
    }
    
    total_valid_faces = 0
    total_faces_found = 0

    print("Recalculating Recall by Size Group...")
    image_files = list(gt_map.keys())
    
    metrics = DetectionMetricsCalculator()

    for img_name in tqdm(image_files):
        gt_data = gt_map[img_name]
        gt_boxes = gt_data['boxes']
        gt_iscrowd = gt_data['iscrowd']

        pred_boxes_list = []
        pred_results_for_map = []
        
        if img_name in preds_grouped.groups:
            group = preds_grouped.get_group(img_name)
            for _, row in group.iterrows():
                if pd.isna(row['pred_x']): continue
                
                pred_boxes_list.append([row['pred_x'], row['pred_y'], row['pred_w'], row['pred_h']])
                
                pred_results_for_map.append({
                    'facial_area': {'x': int(row['pred_x']), 'y': int(row['pred_y']), 'w': int(row['pred_w']), 'h': int(row['pred_h'])},
                    'confidence': float(row['confidence'])
                })

        valid_indices = [i for i, ignore in enumerate(gt_iscrowd) if ignore == 0]
        valid_gt_boxes = [gt_boxes[i] for i in valid_indices]
        
        hits_flags = calculate_iou_hits(pred_boxes_list, valid_gt_boxes)
        
        for box, is_found in zip(valid_gt_boxes, hits_flags):
            _, _, _, h = box
            
            for bin_name, (min_h, max_h) in SIZE_BINS.items():
                if min_h <= h < max_h:
                    breakdown_stats[bin_name]['total'] += 1
                    if is_found:
                        breakdown_stats[bin_name]['found'] += 1
                    break
        
        total_valid_faces += len(valid_gt_boxes)
        total_faces_found += sum(hits_flags)

        metrics.update(pred_results_for_map, gt_boxes, gt_iscrowd)

    final_metrics = metrics.compute()
    final_metrics_clean = {k: round(val.item(), 4) if isinstance(val, torch.Tensor) else val for k, val in final_metrics.items()}
    
    breakdown_results = {}
    print("\n=== Recall Breakdown by Face Height ===")
    for bin_name, stats in breakdown_stats.items():
        total = stats['total']
        found = stats['found']
        recall = (found / total) if total > 0 else 0.0
        
        breakdown_results[bin_name] = {
            'total_faces': total,
            'faces_found': found,
            'recall': round(recall, 4)
        }
        print(f"{bin_name:<15}: {recall:.2%} ({found}/{total})")

    final_metrics_clean['recall_breakdown'] = breakdown_results
    final_metrics_clean['total_gt_faces'] = total_valid_faces
    final_metrics_clean['total_found'] = total_faces_found
    final_metrics_clean['recall_percentage'] = round(total_faces_found / total_valid_faces, 4) if total_valid_faces > 0 else 0

    output_path = os.path.join(EXPERIMENT_DIR, "detailed_summary_metrics.json")
    with open(output_path, 'w') as f:
        json.dump(final_metrics_clean, f, indent=4)
        
    print(f"\nSaved detailed metrics to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Recalculate WIDER FACE detection metrics from an existing raw_predictions.csv.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--experiment_dir", type=str, required=True, help="Path to the experiment folder containing raw_predictions.csv.")
    parser.add_argument("--annotation_file", type=str, required=True, help="Path to the WIDER FACE ground-truth annotation file (wider_face_val_bbx_gt.txt).")
    args = parser.parse_args()
    main(args)