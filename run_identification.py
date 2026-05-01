"""
run_identification.py
---------------------
Open-set face identification benchmark: ranks gallery identities by embedding
distance for each probe and reports Rank-1/5/10 accuracy and ref-recall@10.

Two dataset modes:
  - Scan mode  : walks --dataset_path, auto-splits per identity into gallery
                 (num_shots images) and probes (remaining images).
  - CSV mode   : uses --gallery_csv / --probes_csv for an exact pre-defined split.
                 Gallery always uses cropped faces (ident_rel_path);
                 probes can optionally use full source images (--use_source_paths_for_probes).

Usage examples:
  # Scan mode
  python run_identification.py --dataset_path datasets/custom_dataset_cropped_ident \\
      --recognizer ArcFace --detector skip --num_shots 1

  # CSV split mode
  python run_identification.py --dataset_path drones/custom_dataset_cropped_ident \\
      --recognizer InsightFace_Custom --detector skip \\
      --gallery_csv drones/ident_gallery.csv --probes_csv drones/ident_probes.csv
"""
import os
import argparse
import numpy as np
import pandas as pd
import json
import pickle
import sys
import cv2
import random
from tqdm import tqdm
from scipy.spatial.distance import cdist
from deepface import DeepFace
from pathlib import Path

sys.path.append(os.getcwd())

try:
    from src.recognizer_insightface import InsightFaceCustomRecognizer
except ImportError:
    InsightFaceCustomRecognizer = None

try:
    from src.recognizer_vit import ViTRecognizer
except ImportError:
    ViTRecognizer = None

try:
    from src.recognizer_swin import SwinFaceRecognizer
except ImportError:
    SwinFaceRecognizer = None

class IdentificationBenchmark:
    def __init__(
        self,
        dataset_path,
        model_name,
        detector_backend,
        num_shots=1,
        results_base_dir="results",
        gallery_csv=None,
        probes_csv=None,
        use_source_paths_for_probes=False,
        source_images_root=None,
    ):
        self.dataset_path = dataset_path
        self.model_name = model_name
        self.detector_backend = detector_backend
        self.num_shots = num_shots
        self.gallery_csv = gallery_csv
        self.probes_csv = probes_csv
        self.use_source_paths_for_probes = use_source_paths_for_probes
        self.source_images_root = source_images_root
        
        # Include num_shots in folder name to keep experiments separate
        self.exp_name = f"ident_{os.path.basename(dataset_path)}_{detector_backend}_{model_name}_{num_shots}shot"
        self.results_dir = os.path.join(results_base_dir, self.exp_name)
        os.makedirs(self.results_dir, exist_ok=True)
        
        self.results_csv_path = os.path.join(self.results_dir, "raw_identification_results.csv")
        self.gallery_cache_path = os.path.join(self.results_dir, "gallery_embeddings.pkl")

        self.custom_model_obj = None

        if self.model_name == "InsightFace_Custom":
            if InsightFaceCustomRecognizer is None:
                raise ImportError("Error: src.recognizer_insightface_custom not found.")
            print(f"Initializing Custom InsightFace...")
            try:
                self.custom_model_obj = InsightFaceCustomRecognizer(detector_backend=self.detector_backend)
            except TypeError:
                self.custom_model_obj = InsightFaceCustomRecognizer()

        elif self.model_name == "ViT_timm":
            if ViTRecognizer is None:
                raise ImportError("Error: src.recognizer_vit not found.")
            print(f"Initializing Vision Transformer (ViT)...")
            self.custom_model_obj = ViTRecognizer(model_name='vit_base_patch16_224')

        elif self.model_name == "SwinFace":
            if SwinFaceRecognizer is None:
                raise ImportError("Error: src.recognizer_swin not found.")
            print(f"Initializing SwinFace Recognizer...")
            self.custom_model_obj = SwinFaceRecognizer()

    def _resolve_split_path(self, row, use_source_paths=False):
        """Resolves a split CSV row to an absolute image path.
        
        Args:
            row: CSV row with image metadata
            use_source_paths: If True, prefer source_file/source_rel_path (full images).
                             If False, prefer ident_rel_path (cropped faces).
        """
        if use_source_paths:
            source_rel_path = str(row.get("source_rel_path", "")).replace("\\", "/")
            source_file = str(row.get("source_file", "")).replace("\\", "/")

            root = Path(self.source_images_root) if self.source_images_root else None
            if root is not None:
                if source_rel_path:
                    p = root / source_rel_path
                    if p.exists():
                        return str(p)
                if source_file:
                    p = root / source_file
                    if p.exists():
                        return str(p)

            # If source path is already absolute in CSV, allow direct use.
            if source_rel_path:
                p = Path(source_rel_path)
                if p.is_absolute() and p.exists():
                    return str(p)
            if source_file:
                p = Path(source_file)
                if p.is_absolute() and p.exists():
                    return str(p)

        ident_rel_path = str(row.get("ident_rel_path", "")).replace("\\", "/")
        if ident_rel_path:
            p = Path(ident_rel_path)
            if p.is_absolute() and p.exists():
                return str(p)

            # Try relative to dataset_path (drones folder) first
            candidate = Path(self.dataset_path) / ident_rel_path
            if candidate.exists():
                return str(candidate)
            
            # Also try parent level (legacy support)
            candidate = Path(self.dataset_path).parent / ident_rel_path
            if candidate.exists():
                return str(candidate)

        # Fallback: build from dataset_path/person_id/filename
        person_id = str(row.get("person_id", ""))
        filename = str(row.get("filename", ""))
        if person_id and filename:
            candidate = Path(self.dataset_path) / person_id / filename
            if candidate.exists():
                return str(candidate)

        return None

    def load_and_split_dataset(self):
        # Optional exact split from pre-generated CSVs.
        if self.gallery_csv and self.probes_csv:
            print(f"Loading split from CSVs:\n  gallery={self.gallery_csv}\n  probes={self.probes_csv}")
            print(f"  [Gallery uses cropped faces: {not self.use_source_paths_for_probes}]")
            print(f"  [Probes use full images: {self.use_source_paths_for_probes}]")
            
            gallery_df = pd.read_csv(self.gallery_csv)
            probes_df = pd.read_csv(self.probes_csv)

            self.gallery = []
            self.probes = []

            for _, row in gallery_df.iterrows():
                # Gallery always uses cropped faces (ident_rel_path)
                img_path = self._resolve_split_path(row, use_source_paths=False)
                if img_path is None:
                    continue
                self.gallery.append({"path": img_path, "identity": str(row["person_id"])})

            for _, row in probes_df.iterrows():
                # Probes use source_file (full images) if flag is set
                img_path = self._resolve_split_path(row, use_source_paths=self.use_source_paths_for_probes)
                if img_path is None:
                    continue
                self.probes.append({"path": img_path, "identity": str(row["person_id"])})

            print(f"  -> Gallery Size: {len(self.gallery)} (from CSV)")
            print(f"  -> Probe Size:   {len(self.probes)} (from CSV)")
            return

        print(f"Scanning dataset: {self.dataset_path}")
        identity_map = {}
        for root, _, files in os.walk(self.dataset_path):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    parent = os.path.basename(root)
                    
                    if parent.lower() in ['frontal', 'profile']:
                        identity = os.path.basename(os.path.dirname(root))
                    else:
                        identity = parent

                    if identity not in identity_map:
                        identity_map[identity] = []
                    identity_map[identity].append(os.path.join(root, file))
        
        self.gallery = []
        self.probes = []
        
        # Sort for consistency before shuffling
        for identity in identity_map:
            identity_map[identity].sort()

        for identity, images in identity_map.items():
            if identity.lower() == "others":
                print(f"  -> Found 'Others' folder ({len(images)} images). Adding all as distractors.")
                for i, img_path in enumerate(images):
                    filename = os.path.basename(img_path)
                    distractor_id = filename 
                    self.gallery.append({"path": img_path, "identity": distractor_id})
                continue

            explicit_refs = [img for img in images if "_REF" in os.path.basename(img)]
            candidates = [img for img in images if "_REF" not in os.path.basename(img)]
            
            random.shuffle(candidates)
            
            all_available = explicit_refs + candidates
            total_images = len(all_available)
            
            if total_images == 1:
                k_gallery = 1
            else:
                k_gallery = min(self.num_shots, total_images - 1)
            
            gallery_imgs = all_available[:k_gallery]
            probe_imgs = all_available[k_gallery:]

            for ref_img in gallery_imgs:
                self.gallery.append({"path": ref_img, "identity": identity})
            
            for probe_img in probe_imgs:
                self.probes.append({"path": probe_img, "identity": identity})
                    
        print(f"  -> Gallery Size: {len(self.gallery)} (Identities + Distractors)")
        print(f"  -> Probe Size:   {len(self.probes)} (Test Queries)")

    def _get_embedding(self, img_path):
        try:
            if self.model_name == "InsightFace_Custom":
                img = cv2.imread(img_path)
                if img is None: return None
                res = self.custom_model_obj.represent(img)
                if isinstance(res, list):
                    if len(res) > 0 and isinstance(res[0], list): return res[0]
                    return res
                return res

            elif self.model_name in ("SwinFace", "ViT_timm"):
                return self.custom_model_obj.represent(img_path)

            else:
                embedding_objs = DeepFace.represent(
                    img_path=img_path,
                    model_name=self.model_name,
                    detector_backend=self.detector_backend,
                    enforce_detection=False, 
                    align=True
                )
                return embedding_objs[0]["embedding"]
                
        except Exception as e:
            return None

    def build_gallery(self):
        if os.path.exists(self.gallery_cache_path):
            print(f"Loading cached gallery from {self.gallery_cache_path}...")
            with open(self.gallery_cache_path, 'rb') as f:
                data = pickle.load(f)
                self.gallery_embeddings = data['embeddings']
                self.gallery_identities = data['identities']
                self.gallery_paths = data['paths']
            print("Gallery loaded from cache!")
            return

        print(f"\n--- Building Gallery for {self.model_name} ({self.num_shots}-shot) ---")
        self.gallery_embeddings = []
        self.gallery_identities = []
        self.gallery_paths = []
        
        for item in tqdm(self.gallery, desc="Encoding Gallery"):
            emb = self._get_embedding(item["path"])
            if emb is not None:
                self.gallery_embeddings.append(emb)
                self.gallery_identities.append(item["identity"])
                self.gallery_paths.append(item["path"])
                
        self.gallery_embeddings = np.array(self.gallery_embeddings)
        
        with open(self.gallery_cache_path, 'wb') as f:
            pickle.dump({
                'embeddings': self.gallery_embeddings,
                'identities': self.gallery_identities,
                'paths': self.gallery_paths
            }, f)
        print("Gallery saved to cache.")

    def run_identification(self):
        print("\n--- Running Identification (Resumable) ---")
        
        processed_paths = set()
        if os.path.exists(self.results_csv_path):
            try:
                existing_df = pd.read_csv(self.results_csv_path)
                processed_paths = set(existing_df['probe_path'].values)
                print(f"Resuming... {len(processed_paths)} probes already done.")
            except:
                pass
        else:
            pd.DataFrame(columns=[
                "probe_path", "probe_identity", "rank1_identity", "rank1_distance",
                "rank1_correct", "rank5_correct", "rank10_correct",
                "same_identity_in_top10_count", "total_gallery_refs_for_identity", "ref_recall_at10",
                "top_identities", "top_distances", "top_paths"
            ]).to_csv(self.results_csv_path, index=False)

        probes_to_do = [p for p in self.probes if p['path'] not in processed_paths]
        print(f"Probes remaining: {len(probes_to_do)}")
        
        for item in tqdm(probes_to_do, desc="Processing Probes"):
            probe_emb = self._get_embedding(item["path"])
            
            if probe_emb is not None:
                probe_emb = np.array(probe_emb).reshape(1, -1)
                
                distances = cdist(probe_emb, self.gallery_embeddings, metric='cosine')[0]
                top_k_indices = np.argsort(distances)[:10]
                
                top_identities = [self.gallery_identities[i] for i in top_k_indices]
                top_distances = [distances[i] for i in top_k_indices]
                top_paths = [self.gallery_paths[i] for i in top_k_indices]
                true_identity = item["identity"]
                total_refs_for_identity = sum(1 for gid in self.gallery_identities if gid == true_identity)
                same_identity_in_top10 = sum(1 for gid in top_identities if gid == true_identity)
                ref_recall_at10 = (same_identity_in_top10 / total_refs_for_identity) if total_refs_for_identity > 0 else 0.0
                
                record = {
                    "probe_path": item["path"],
                    "probe_identity": true_identity,
                    "rank1_identity": top_identities[0],
                    "rank1_distance": top_distances[0],
                    "rank1_correct": (top_identities[0] == true_identity),
                    "rank5_correct": (true_identity in top_identities[:5]),
                    "rank10_correct": (true_identity in top_identities[:10]),
                    "same_identity_in_top10_count": same_identity_in_top10,
                    "total_gallery_refs_for_identity": total_refs_for_identity,
                    "ref_recall_at10": round(float(ref_recall_at10), 5),
                    "top_identities": str(top_identities),
                    "top_distances": str([round(float(x), 4) for x in top_distances]),
                    "top_paths": str(top_paths)
                }
                
                pd.DataFrame([record]).to_csv(self.results_csv_path, mode='a', header=False, index=False)
        
        print("\nAll probes processed. Calculating metrics...")
        full_df = pd.read_csv(self.results_csv_path)
        if len(full_df) > 0:
            metrics = {
                "dataset": self.dataset_path,
                "model": self.model_name,
                "num_shots": self.num_shots,
                "total_probes": len(full_df),
                "rank1_accuracy": round(full_df['rank1_correct'].mean(), 5),
                "rank5_accuracy": round(full_df['rank5_correct'].mean(), 5),
                "rank10_accuracy": round(full_df['rank10_correct'].mean(), 5),
                "mean_ref_recall_at10": round(full_df['ref_recall_at10'].mean(), 5)
            }
            print(json.dumps(metrics, indent=4))
            with open(os.path.join(self.results_dir, "identification_metrics.json"), 'w') as f:
                json.dump(metrics, f, indent=4)

            per_id = full_df.groupby('probe_identity', as_index=False).agg(
                probes=('probe_identity', 'size'),
                rank1_accuracy=('rank1_correct', 'mean'),
                rank5_accuracy=('rank5_correct', 'mean'),
                rank10_accuracy=('rank10_correct', 'mean'),
                mean_ref_recall_at10=('ref_recall_at10', 'mean')
            )
            per_id[['rank1_accuracy', 'rank5_accuracy', 'rank10_accuracy', 'mean_ref_recall_at10']] = (
                per_id[['rank1_accuracy', 'rank5_accuracy', 'rank10_accuracy', 'mean_ref_recall_at10']].round(5)
            )
            per_id_path = os.path.join(self.results_dir, "identification_metrics_per_identity.csv")
            per_id.to_csv(per_id_path, index=False)
            print(f"Saved per-identity metrics: {per_id_path}")

def main():
    parser = argparse.ArgumentParser(
        description="Run Face Identification Benchmark.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset_path", type=str, required=True,
        help="Path to the root dataset folder (one sub-folder per identity).",
    )
    parser.add_argument(
        "--recognizer", type=str, default="ArcFace",
        choices=['VGG-Face', 'Facenet', 'Facenet512', 'ArcFace',
                 'ViT_timm', 'InsightFace_Custom', 'SwinFace'],
        help="Face recognition model.",
    )
    parser.add_argument(
        "--detector", type=str, default="retinaface",
        choices=['opencv', 'retinaface', 'mtcnn', 'skip'],
        help="Face detector backend. Use 'skip' for pre-cropped images.",
    )
    parser.add_argument(
        "--num_shots", type=int, default=1,
        help="Number of gallery images per identity.",
    )
    parser.add_argument(
        "--gallery_csv", type=str, default=None,
        help="Pre-defined gallery split CSV (from rescale_annotations.py).",
    )
    parser.add_argument(
        "--probes_csv", type=str, default=None,
        help="Pre-defined probe split CSV (from rescale_annotations.py).",
    )
    parser.add_argument(
        "--use_source_paths_for_probes", action="store_true",
        help="Load probes from full source images instead of cropped faces.",
    )
    parser.add_argument(
        "--source_images_root", type=str, default=None,
        help="Root folder for source images (used with --use_source_paths_for_probes).",
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

    benchmark = IdentificationBenchmark(
        args.dataset_path,
        args.recognizer,
        args.detector,
        num_shots=args.num_shots,
        results_base_dir=args.output_dir,
        gallery_csv=args.gallery_csv,
        probes_csv=args.probes_csv,
        use_source_paths_for_probes=args.use_source_paths_for_probes,
        source_images_root=args.source_images_root,
    )
    benchmark.load_and_split_dataset()
    benchmark.build_gallery()
    benchmark.run_identification()

if __name__ == "__main__":
    main()