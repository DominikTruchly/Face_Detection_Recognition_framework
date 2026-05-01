"""
src/metrics.py
--------------
Metric calculators for the two benchmark types:

  MetricsCalculator        — Face verification / identification tasks.
                             Operates on a DataFrame of pair comparisons and computes
                             accuracy (at the best threshold), precision, recall, F1,
                             TAR@FAR, and per-age-gap accuracy breakdowns.

  DetectionMetricsCalculator — Face detection tasks.
                             Wraps torchmetrics MeanAveragePrecision (COCO-style mAP)
                             and additionally tracks per-size recall:
                               Small  (h < 20 px)  — hard faces
                               Medium (20 <= h < 50 px)
                               Large  (h >= 50 px) — easy faces
"""
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, confusion_matrix
import torch
import torchmetrics
from torchmetrics.detection import MeanAveragePrecision

class MetricsCalculator:
    """Calculates performance metrics from a raw results DataFrame (Verification Tasks)."""

    def calculate_summary_metrics(self, df):
        """Calculates a full suite of summary metrics."""
        successful_df = df[df['successful_comparison'] == True].copy()
        if successful_df.empty:
            return {"error": "No successful comparisons found in the data."}

        labels = successful_df['ground_truth'].tolist()
        scores = successful_df['distance_score'].tolist()

        thresholds = np.arange(0, 2.0, 0.001)
        accuracies = [accuracy_score(labels, [1 if s <= t else 0 for s in scores]) for t in thresholds]
        best_threshold_idx = np.argmax(accuracies)
        best_threshold = thresholds[best_threshold_idx]
        max_accuracy = accuracies[best_threshold_idx]

        predictions = [1 if s <= best_threshold else 0 for s in scores]

        # confusion_matrix is only 2x2 when both classes are present in labels.
        # With few pairs or a single class, pad to avoid unpack errors.
        cm = confusion_matrix(labels, predictions, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

        unique_classes = len(set(labels))
        if unique_classes < 2:
            fpr, tpr = np.array([0.0, 1.0]), np.array([0.0, 1.0])
        else:
            fpr, tpr, _ = roc_curve(labels, -np.array(scores))
        tar_at_far_0_1 = tpr[np.argmin(np.abs(fpr - 0.001))] if len(fpr) > 1 else 0.0
        tar_at_far_1 = tpr[np.argmin(np.abs(fpr - 0.01))] if len(fpr) > 1 else 0.0

        summary = {
            'total_pairs': len(df),
            'successful_pairs': len(successful_df),
            'failed_pairs': len(df) - len(successful_df),
            'best_threshold': best_threshold,
            'accuracy': max_accuracy,
            'precision': precision_score(labels, predictions, zero_division=0),
            'recall': recall_score(labels, predictions, zero_division=0),
            'f1_score': f1_score(labels, predictions, zero_division=0),
            'true_positives': int(tp),
            'false_positives': int(fp),
            'true_negatives': int(tn),
            'false_negatives': int(fn),
            'avg_processing_time_s': df['processing_time_s'].mean(),
            'TAR@FAR=0.1%': tar_at_far_0_1,
            'TAR@FAR=1%': tar_at_far_1,
        }

        accuracy_by_bin = {}
        if 'age_gap' in successful_df.columns and successful_df['age_gap'].max() > 0:
            max_gap = int(successful_df['age_gap'].max())
            bins = list(range(0, max_gap + 6, 5))
            labels_str = [f"{b}-{b+4} yrs" for b in bins[:-1]]
            
            successful_df['age_bin'] = pd.cut(successful_df['age_gap'], bins=bins, labels=labels_str, right=False)
            
            for bin_name, group_df in successful_df.groupby('age_bin'):
                if group_df.empty: continue
                bin_labels = group_df['ground_truth'].tolist()
                bin_scores = group_df['distance_score'].tolist()
                
                bin_predictions = [1 if s <= best_threshold else 0 for s in bin_scores]
                bin_accuracy = accuracy_score(bin_labels, bin_predictions)
                accuracy_by_bin[str(bin_name)] = round(bin_accuracy, 5)
        
        summary['accuracy_by_age_bin'] = accuracy_by_bin

        return {k: (v if isinstance(v, (int, bool, dict)) else round(v, 5)) for k, v in summary.items()}

class DetectionMetricsCalculator:
    """Calculates mAP and Detailed Recall Breakdown for detection tasks."""
    
    def __init__(self):
        self.mAP = MeanAveragePrecision(box_format='xywh', class_metrics=False, iou_thresholds=[0.5])
        
        self.stats = {
            'all': {'total': 0, 'found': 0},
            'Small (Hard)': {'total': 0, 'found': 0},
            'Medium': {'total': 0, 'found': 0},
            'Large (Easy)': {'total': 0, 'found': 0}
        }

    def _calculate_matches(self, pred_boxes, gt_boxes, iou_threshold=0.5):
        """Returns a boolean list indicating which GT boxes were found."""
        if len(gt_boxes) == 0: return []
        if len(pred_boxes) == 0: return [False] * len(gt_boxes)

        p_tensor = torch.tensor(pred_boxes, dtype=torch.float32)
        g_tensor = torch.tensor(gt_boxes, dtype=torch.float32)

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
        
        return (max_iou_per_gt >= iou_threshold).tolist()

    def _format_boxes_for_torchmetrics(self, pred_results, gt_boxes, gt_iscrowd):
        pred_boxes_list = []
        pred_scores_list = []
        
        for res in pred_results:
            if not isinstance(res, dict) or 'facial_area' not in res: continue
            area = res['facial_area']
            if area['w'] > 0 and area['h'] > 0:
                pred_boxes_list.append([area['x'], area['y'], area['w'], area['h']])
                pred_scores_list.append(res['confidence'])
        
        preds = [dict(
            boxes=torch.tensor(pred_boxes_list, dtype=torch.float32), 
            scores=torch.tensor(pred_scores_list, dtype=torch.float32), 
            labels=torch.zeros(len(pred_boxes_list), dtype=torch.int32)
        )]

        if not gt_boxes:
            target = [dict(
                boxes=torch.tensor([], dtype=torch.float32),
                labels=torch.tensor([], dtype=torch.int32),
                iscrowd=torch.tensor([], dtype=torch.int32)
            )]
        else:
            target = [dict(
                boxes=torch.tensor(gt_boxes, dtype=torch.float32), 
                labels=torch.zeros(len(gt_boxes), dtype=torch.int32),
                iscrowd=torch.tensor(gt_iscrowd, dtype=torch.int32) 
            )]
            
        return preds, target, pred_boxes_list

    def update(self, pred_results, gt_boxes, gt_iscrowd):
        preds_formatted, target_formatted, pred_boxes_list = self._format_boxes_for_torchmetrics(pred_results, gt_boxes, gt_iscrowd)
        try:
            self.mAP.update(preds_formatted, target_formatted)
        except Exception as e:
            pass

        valid_indices = [i for i, ignore in enumerate(gt_iscrowd) if ignore == 0]
        valid_gt_boxes = [gt_boxes[i] for i in valid_indices]
        
        if not valid_gt_boxes:
            return

        hits_flags = self._calculate_matches(pred_boxes_list, valid_gt_boxes)
        
        for box, is_found in zip(valid_gt_boxes, hits_flags):
            h = box[3]
            
            self.stats['all']['total'] += 1
            if is_found: self.stats['all']['found'] += 1
            
            if h < 20:
                self.stats['Small (Hard)']['total'] += 1
                if is_found: self.stats['Small (Hard)']['found'] += 1
            elif 20 <= h < 50:
                self.stats['Medium']['total'] += 1
                if is_found: self.stats['Medium']['found'] += 1
            else:
                self.stats['Large (Easy)']['total'] += 1
                if is_found: self.stats['Large (Easy)']['found'] += 1

    def compute(self):
        results = self.mAP.compute()
        
        breakdown = {}
        for category, data in self.stats.items():
            total = data['total']
            found = data['found']
            recall = (found / total) if total > 0 else 0.0
            
            if category == 'all':
                results['total_gt_faces'] = torch.tensor(total)
                results['total_found'] = torch.tensor(found)
                results['recall_percentage'] = torch.tensor(recall)
            else:
                breakdown[category] = {
                    'total_faces': total,
                    'faces_found': found,
                    'recall': round(recall, 4)
                }
        
        results['recall_breakdown'] = breakdown
        return results