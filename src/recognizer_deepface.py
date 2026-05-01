"""
src/recognizer_deepface.py
--------------------------
Generic DeepFace-based face recognition wrapper. Handles Unicode/non-ASCII
image paths by reading files as bytes before passing numpy arrays to DeepFace.

Used by run_verification.py for all standard DeepFace models:
  VGG-Face, Facenet, Facenet512, ArcFace, etc.
"""
import os
import cv2
import numpy as np
from deepface import DeepFace


class DeepFaceRecognizer:
    """DeepFace-based face recognizer supporting multiple model backends."""

    def _read_image_safe(self, img_path):
        """Reads an image from disk, handling non-ASCII (Unicode) path characters.

        Args:
            img_path (str): Path to the image file.

        Returns:
            numpy.ndarray: The decoded BGR image array.

        Raises:
            IOError: If the file cannot be read or decoded.
        """
        try:
            with open(img_path, "rb") as f:
                img_bytes = f.read()
            nparr = np.frombuffer(img_bytes, np.uint8)
            img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img_np is None:
                raise IOError(f"cv2.imdecode failed to decode image: {img_path}")
            return img_np
        except Exception as e:
            raise IOError(f"Failed to read/decode image at {img_path}: {e}")

    def verify_pair(self, img1_path, img2_path, model_name, detector_backend):
        """Verifies whether two images depict the same person using DeepFace.

        Args:
            img1_path (str): Path to the first image.
            img2_path (str): Path to the second image.
            model_name (str): DeepFace recognition model (e.g. 'ArcFace', 'Facenet512').
            detector_backend (str): Face detector backend (e.g. 'retinaface', 'mtcnn').

        Returns:
            dict: DeepFace result with 'verified', 'distance', 'threshold', etc.
                  Returns None if processing fails.
        """
        try:
            img1 = self._read_image_safe(img1_path)
            img2 = self._read_image_safe(img2_path)
            return DeepFace.verify(
                img1_path=img1,
                img2_path=img2,
                model_name=model_name,
                detector_backend=detector_backend,
                enforce_detection=False,
                align=True,
            )
        except Exception as e:
            print(f"Warning: Could not process pair ({os.path.basename(img1_path)}, "
                  f"{os.path.basename(img2_path)}). Error: {e}")
            return None