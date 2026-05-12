# Face Detection & Recognition Framework

A modular benchmarking framework for evaluating face detection and recognition models. Supports multiple detector backends and recognition models, with three evaluation modes: **detection**, **verification**, and **identification**. Published under the MIT license.

## Colab Demo

An interactive demo notebook (`colab_demo.ipynb`) is included. Open it in Google Colab to run face detection, verification, and identification on sample images — no local setup required.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/DominikTruchly/Face_Detection_Recognition_framework/blob/main/colab_demo.ipynb)

## Installation

```bash
pip install -r requirements.txt
```

## Model Weights

| Model | Source |
|-------|--------|
| ArcFace, Facenet512, VGG-Face | Downloaded automatically by DeepFace on first use |
| InsightFace (buffalo\_l) | Downloaded automatically by InsightFace on first use |
| SwinFace (`SwinFace_MS1MV2.pth`, ~300 MB) | [Google Drive](https://drive.google.com/file/d/1fi4IuuFV8NjnWm-CufdrhMKrkjxhSmjx) — place in `weights/` |

## Repository Structure

```
run_detection.py        # Face detection benchmark
run_verification.py     # Face verification benchmark
run_identification.py   # Face identification benchmark
src/
  detection.py              # Detector wrapper (DeepFace backends)
  evaluation.py             # Verification benchmark runner
  metrics.py                # Metric calculators (mAP, accuracy, TAR@FAR, ...)
  recognizer_deepface.py    # DeepFace recognizer (ArcFace, Facenet512, VGG-Face, ...)
  recognizer_insightface.py # InsightFace buffalo_l recognizer
  recognizer_swin.py        # SwinFace recognizer
  recognizer_vit.py         # ViT-based recognizers
scripts/
  generate_agedb_protocol.py   # Build balanced AgeDB verification pairs CSV
  prepare_cfp.py               # Build CFP-FF / CFP-FP pairs CSVs
  finetune_classifier_head.py  # Fine-tune a timm ViT classifier head
  recalculate_metrics.py       # Recompute metrics from existing raw_results.csv
  recalculate_detection_metrics.py  # Recompute detection metrics from raw_predictions.csv
```

## Usage

### Face Detection

Evaluate a detector on an annotated dataset (WIDER FACE format):
```bash
python run_detection.py --detector retinaface \
    --img_dir datasets/WIDER_val/images \
    --annotation_file wider_face_val_bbx_gt.txt
```

Explore detections on your own images (no annotations needed):
```bash
python run_detection.py --detector retinaface \
    --img_dir my_images/ --save_faces
```

Supported detectors: `opencv`, `retinaface`, `mtcnn`, `yolov8n`, `ssd`, `dlib`, `mediapipe`, `centerface`

### Face Verification

Compare a single pair of images:
```bash
python run_verification.py --recognizer ArcFace --detector retinaface \
    --img1 photo1.jpg --img2 photo2.jpg
```

Benchmark on a dataset with a ground-truth pairs CSV:
```bash
python run_verification.py --recognizer ArcFace --detector retinaface \
    --img_dir my_dataset/images/ --pairs_file my_dataset/pairs.csv
```

Supported recognizers: `VGG-Face`, `Facenet`, `Facenet512`, `ArcFace`, `InsightFace`, `InsightFace_Custom`, `SwinFace`, `ViT_timm`

### Face Identification

Scan-mode (auto-splits each identity into gallery and probes):
```bash
python run_identification.py --dataset_path datasets/my_identity_dataset \
    --recognizer ArcFace --detector skip --num_shots 1
```

CSV-mode (use pre-defined gallery/probe splits):
```bash
python run_identification.py --dataset_path datasets/my_identity_dataset \
    --recognizer InsightFace_Custom --detector skip \
    --gallery_csv splits/gallery.csv --probes_csv splits/probes.csv
```

Results (Rank-1/5/10 accuracy, ref-recall@10) are saved to `results/`.