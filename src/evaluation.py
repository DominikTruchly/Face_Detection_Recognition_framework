import os
import time
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, confusion_matrix
from src.metrics import MetricsCalculator
import cv2 
import csv
class Evaluator:
    def __init__(self, dataset_name, dataset_config, detector_backend, recognizer_model, recognizer_instance):
        self.dataset_name = dataset_name
        self.dataset_config = dataset_config
        self.dataset_path = dataset_config['path']
        self.protocol_type = dataset_config['protocol_type']
        
        self.detector_backend = detector_backend
        self.recognizer_model = recognizer_model
        self.recognizer = recognizer_instance
        self.path_cache = {}

    def _read_pairs_from_lfw_csv(self):
        """Parses LFW Kaggle CSVs. Returns list of tuples: (path1, path2, label, basename1, basename2, age_gap)"""
        pairs_data = []
        match_df = pd.read_csv(self.dataset_config['match_pairs_file'])
        mismatch_df = pd.read_csv(self.dataset_config['mismatch_pairs_file'])
        
        for _, row in match_df.iterrows():
            path1 = self._get_image_path_lfw(row['name'], row['imagenum1'])
            path2 = self._get_image_path_lfw(row['name'], row['imagenum2'])
            basename1 = os.path.basename(path1) if path1 else None
            basename2 = os.path.basename(path2) if path2 else None
            pairs_data.append((path1, path2, 1, basename1, basename2, -1))
            
        p1_col, n1_col, p2_col, n2_col = mismatch_df.columns
        for _, row in mismatch_df.iterrows():
            path1 = self._get_image_path_lfw(row[p1_col], row[n1_col])
            path2 = self._get_image_path_lfw(row[p2_col], row[n2_col])
            basename1 = os.path.basename(path1) if path1 else None
            basename2 = os.path.basename(path2) if path2 else None
            pairs_data.append((path1, path2, 0, basename1, basename2, -1))
        return pairs_data

    def _read_pairs_from_generic_csv(self):
        """Parses a pairs CSV (file1, file2 [, is_same] [, age_gap]).
        is_same is optional — if absent, labels default to -1 (no ground truth).
        """
        pairs_data = []
        # sep=None with engine='python' auto-detects tab vs comma
        pairs_df = pd.read_csv(self.dataset_config['pairs_file'], sep=None, engine='python')
        has_labels = 'is_same' in pairs_df.columns
        for _, row in pairs_df.iterrows():
            file1, file2 = row['file1'], row['file2']
            is_same = int(row['is_same']) if has_labels else -1
            age_gap = int(row['age_gap']) if 'age_gap' in pairs_df.columns else -1
            path1 = self._get_image_path_generic(file1)
            path2 = self._get_image_path_generic(file2)
            pairs_data.append((path1, path2, is_same, file1, file2, age_gap))
        return pairs_data

    def _get_image_path_lfw(self, person, image_num):
        key = (person, image_num)
        if key in self.path_cache: return self.path_cache[key]
        
        name_with_underscore = person.replace(' ', '_')
        path = os.path.join(self.dataset_path, name_with_underscore, f"{name_with_underscore}_{int(image_num):04d}.jpg")
        
        if not os.path.exists(path):
            print(f"Warning: LFW path not found: {path}"); path = None
            
        self.path_cache[key] = path
        return path
        
    def _get_image_path_generic(self, filename):
        key = filename
        if key in self.path_cache: return self.path_cache[key]
        
        path = os.path.join(self.dataset_path, filename)
        
        if not os.path.exists(path):
            print(f"Warning: Generic path not found: {path}"); path = None
            
        self.path_cache[key] = path
        return path

    def run_verification_benchmark(self, output_dir):
        if self.protocol_type == "lfw_csv":
            pairs = self._read_pairs_from_lfw_csv()
        elif self.protocol_type == "generic_csv":
            pairs = self._read_pairs_from_generic_csv()
        else:
            raise ValueError(f"Unknown protocol type: {self.protocol_type}")

        raw_results_path = os.path.join(output_dir, "raw_results.csv")
        processed_pairs = set()
        file_has_content = os.path.exists(raw_results_path) and os.path.getsize(raw_results_path) > 0
        if file_has_content:
            print(f"INFO: Resuming from {raw_results_path}.")
            try:
                # on_bad_lines='skip' and filtering out duplicate header rows
                # handles CSVs that were corrupted by a double-written header
                existing_df = pd.read_csv(raw_results_path, on_bad_lines='skip')
                existing_df = existing_df[existing_df['image1'] != 'image1']
                for _, row in existing_df.iterrows():
                    processed_pairs.add((row['image1'], row['image2']))
                print(f"INFO: {len(processed_pairs)} pairs already processed.")
            except pd.errors.EmptyDataError:
                print("INFO: Results file is empty. Starting from scratch.")
                processed_pairs = set()

        header = ['image1', 'image2', 'ground_truth', 'age_gap', 'distance_score', 'verified_prediction', 'threshold', 'processing_time_s', 'successful_comparison']
        
        with open(raw_results_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=header)
            
            if not file_has_content:
                writer.writeheader()
            
            print(f"Evaluating {len(pairs)} total pairs...")
            
            pbar = tqdm(pairs, initial=len(processed_pairs), total=len(pairs))
            for pair_data in pbar:
                
                path1, path2, label, basename1, basename2, age_gap = pair_data
                
                if (basename1, basename2) in processed_pairs or (basename2, basename1) in processed_pairs:
                    continue
                if not path1 or not path2:
                    print(f"CRITICAL ERROR: Could not find files for pair: {basename1}, {basename2}. Skipping.")
                    continue

                start_time = time.time()
                result = self.recognizer.verify_pair(path1, path2, self.recognizer_model, self.detector_backend)
                end_time = time.time()
                
                is_successful = (result is not None) and (result.get('distance') is not None) and (result.get('distance') > 0)
                
                result_row = {
                    'image1': basename1, 'image2': basename2, 'ground_truth': label, 'age_gap': age_gap,
                    'distance_score': result.get('distance', -1.0) if result else -1.0, 
                    'verified_prediction': result.get('verified', False) if result else False, 
                    'threshold': result.get('threshold', -1.0) if result else -1.0, 
                    'processing_time_s': end_time - start_time, 
                    'successful_comparison': is_successful
                }
                
                writer.writerow(result_row)
                f.flush()
                os.fsync(f.fileno())

        print("INFO: Benchmark loop finished.")
        final_df = pd.read_csv(raw_results_path)

        # Only compute metrics when ground truth labels are available
        has_ground_truth = 'ground_truth' in final_df.columns and (final_df['ground_truth'] != -1).any()
        if has_ground_truth:
            calculator = MetricsCalculator()
            summary_metrics = calculator.calculate_summary_metrics(final_df)
        else:
            print("INFO: No ground truth labels — skipping metrics calculation.")
            summary_metrics = None

        return final_df, summary_metrics