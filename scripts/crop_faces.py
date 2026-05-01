"""
scripts/crop_faces.py
---------------------
Detects and crops individual faces from a folder of images using DeepFace + RetinaFace.
Each detected face is saved as a separate file in the output directory.
Detections where the face region covers the entire image (no face found) are skipped.

Usage:
    python scripts/crop_faces.py --input_dir path/to/images --output_dir path/to/output
"""
import argparse
import os

import cv2
import numpy as np
from deepface import DeepFace
from tqdm import tqdm


def crop_faces(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    image_files = []
    for root, _, files in os.walk(INPUT_DIR):
        for file in files:
            if file.lower().endswith(valid_extensions):
                image_files.append(os.path.join(root, file))

    print(f"Found {len(image_files)} images. Starting detection...")

    success_count = 0
    faces_count = 0

    for img_path in tqdm(image_files, desc="Processing Images"):
        filename = os.path.basename(img_path)
        base_name, _ = os.path.splitext(filename)

        try:
            face_objs = DeepFace.extract_faces(
                img_path=img_path,
                detector_backend='retinaface',
                align=True,
                enforce_detection=False
            )

            img_h, img_w, _ = cv2.imread(img_path).shape
            
            for i, face_obj in enumerate(face_objs):
                face_img = face_obj['face']
                region = face_obj['facial_area']
                
                if region['w'] == img_w and region['h'] == img_h:
                    continue

                face_uint8 = (face_img * 255).astype(np.uint8)
                face_bgr = cv2.cvtColor(face_uint8, cv2.COLOR_RGB2BGR)

                save_name = f"{base_name}_face{i}.jpg"
                save_path = os.path.join(output_dir, save_name)
                cv2.imwrite(save_path, face_bgr)
                faces_count += 1
            
            success_count += 1

        except Exception:
            pass

    print("\n--- Processing Complete ---")
    print(f"Processed {success_count} images.")
    print(f"Extracted {faces_count} individual faces.")
    print(f"Saved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Detect and crop faces from images using DeepFace + RetinaFace.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input_dir", required=True, help="Path to the folder containing input images.")
    parser.add_argument("--output_dir", default="cropped_faces", help="Path to save the cropped face images.")
    args = parser.parse_args()
    crop_faces(args.input_dir, args.output_dir)