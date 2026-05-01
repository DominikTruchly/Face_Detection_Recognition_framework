"""
src/detection.py
----------------
Thin wrapper around DeepFace.extract_faces that provides a unified interface
for all supported detector backends. Centralising detection here means the
benchmark scripts are decoupled from the DeepFace API.

Supported backends: opencv, retinaface, mtcnn, yolov8n, ssd, dlib, mediapipe, centerface.
"""
from deepface import DeepFace

class Detector:
    """A wrapper for SOTA face detection models using DeepFace."""

    def detect_faces(self, image_path, backend):
        """
        Detects faces in an image using a specified backend. This is a simple
        wrapper around DeepFace.extract_faces for modularity.

        Args:
            image_path (str): Path to the image file.
            backend (str): The detector backend to use (e.g., 'retinaface').

        Returns:
            list: A list of face objects from DeepFace. Each object is a
                  dictionary containing the cropped face, bounding box, and confidence.
                  Returns an empty list if an error occurs.
        """
        try:
            return DeepFace.extract_faces(
                img_path=image_path,
                detector_backend=backend,
                enforce_detection=False,
                align=False
            )
        except Exception as e:
            print(f"An error occurred with detector {backend}: {e}")
            return []