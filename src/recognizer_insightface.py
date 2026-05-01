"""
src/recognizer_insightface.py
------------------------------
Two InsightFace buffalo_l (w600k_r50) recognizer variants:
  InsightFaceRecognizer       — uses InsightFace's own built-in RetinaFace detector.
  InsightFaceCustomRecognizer — uses DeepFace for detection/alignment instead;
                                handles non-ASCII paths via np.fromfile + imdecode.
Both use cosine distance with threshold 0.6. Use --detector skip in run_verification.py.
"""
import os
import cv2
from insightface.app import FaceAnalysis
import numpy as np
from insightface.utils import face_align
from scipy.spatial.distance import cosine
from deepface import DeepFace
import warnings

warnings.filterwarnings("ignore")


class InsightFaceRecognizer:

    def __init__(self):
        print("Initializing InsightFaceRecognizer (buffalo_l)...")
        MODELS_DIR = os.path.join(os.environ.get('DEEPFACE_HOME', 'models'), 'insightface_stable')

        self.det_thresh = 0.3

        try:
            self.app = FaceAnalysis(name="buffalo_l", root=MODELS_DIR, providers=['CUDAExecutionProvider'])
            self.app.prepare(ctx_id=0, det_size=(640, 640), det_thresh=self.det_thresh)
            print(f"InsightFace loaded on CUDA (det_thresh={self.det_thresh}).")
        except Exception:
            print("CUDA failed. Falling back to CPU.")
            self.app = FaceAnalysis(name="buffalo_l", root=MODELS_DIR, providers=['CPUExecutionProvider'])
            self.app.prepare(ctx_id=0, det_size=(640, 640), det_thresh=self.det_thresh)
            print(f"InsightFace loaded on CPU (det_thresh={self.det_thresh}).")

        self.detector = self.app.det_model
        self.recognizer = self.app.models['recognition']

    def _get_embedding(self, image_path):
        try:
            image = cv2.imread(image_path)
            if image is None:
                return None
            bboxes, kpss = self.detector.detect(image, max_num=1)
            if bboxes.shape[0] == 0:
                return None
            aligned_face = face_align.norm_crop(image, landmark=kpss[0])
            return self.recognizer.get_feat(aligned_face).flatten()
        except Exception as e:
            print(f"ERROR in _get_embedding for {image_path}: {e}")
            return None

    def verify_pair(self, img1_path, img2_path, model_name, detector_backend):
        emb1 = self._get_embedding(img1_path)
        emb2 = self._get_embedding(img2_path)
        if emb1 is None or emb2 is None:
            return None
        distance = cosine(emb1, emb2)
        threshold = 0.6
        return {'verified': distance <= threshold, 'distance': distance, 'threshold': threshold}


class InsightFaceCustomRecognizer:
    """InsightFace buffalo_l using DeepFace for detection/alignment."""

    def __init__(self, detector_backend='retinaface'):
        self.detector_backend = detector_backend
        print("Initializing InsightFaceCustomRecognizer (buffalo_l recognizer only)...")
        MODELS_DIR = os.path.join(os.environ.get('DEEPFACE_HOME', 'models'), 'insightface_stable')

        try:
            app = FaceAnalysis(name="buffalo_l", root=MODELS_DIR, providers=['CUDAExecutionProvider'])
            app.prepare(ctx_id=0, det_size=(640, 640))
            self.recognizer = app.models['recognition']
            print("InsightFace recognizer (w600k_r50) loaded on CUDA.")
        except Exception:
            print("CUDA failed. Falling back to CPU.")
            app = FaceAnalysis(name="buffalo_l", root=MODELS_DIR, providers=['CPUExecutionProvider'])
            app.prepare(ctx_id=0, det_size=(640, 640))
            self.recognizer = app.models['recognition']
            print("InsightFace recognizer (w600k_r50) loaded on CPU.")
        del app

    def _get_embedding(self, face_crop_np):
        # DeepFace returns float32 RGB in [0,1]; InsightFace expects uint8 BGR
        try:
            face_bgr = cv2.cvtColor((face_crop_np * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
            return self.recognizer.get_feat(face_bgr).flatten()
        except Exception:
            return None

    def represent(self, img_array):
        try:
            face_objs = DeepFace.extract_faces(
                img_path=img_array,
                detector_backend=self.detector_backend,
                align=True,
                enforce_detection=False
            )
            if not face_objs:
                return None
            return self._get_embedding(face_objs[0]['face'])
        except Exception:
            return None

    def _load_image_robustly(self, image_path):
        try:
            img_np = np.fromfile(image_path, dtype=np.uint8)
            image = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
            if image is None:
                raise IOError(f"cv2.imdecode failed for {image_path}")
            return image
        except Exception as e:
            print(f"ERROR: Could not load image {image_path}: {e}")
            return None

    def verify_pair(self, img1_path, img2_path, model_name, detector_backend):
        try:
            img1 = self._load_image_robustly(img1_path)
            img2 = self._load_image_robustly(img2_path)
            if img1 is None or img2 is None:
                return None
            face1_objs = DeepFace.extract_faces(img_path=img1, detector_backend=detector_backend, align=True, enforce_detection=False)
            face2_objs = DeepFace.extract_faces(img_path=img2, detector_backend=detector_backend, align=True, enforce_detection=False)
            if not face1_objs or not face2_objs:
                return None
            emb1 = self._get_embedding(face1_objs[0]['face'])
            emb2 = self._get_embedding(face2_objs[0]['face'])
            if emb1 is None or emb2 is None:
                return None
            distance = cosine(emb1, emb2)
            threshold = 0.6
            return {'verified': distance <= threshold, 'distance': distance, 'threshold': threshold}
        except Exception as e:
            print(f"Error in verify_pair: {e}")
            return None