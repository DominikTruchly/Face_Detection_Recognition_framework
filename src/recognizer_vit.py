"""
src/recognizer_vit.py
----------------------
Three ViT-based face recognizer variants:
  ViTRecognizer    — timm vit_base_patch16_224, RetinaFace 5-point alignment to 224x224.
  HFViTRecognizer  — HuggingFace vit-base-patch16-224 (jayanta), MTCNN detection,
                     mean-pooled last hidden state embedding.
  TimmFTRecognizer — same timm backbone fine-tuned on drone dataset, MTCNN detection,
                     forward_features embedding. Requires results/vit_timm_finetuned_2.pth.
"""
import torch
import timm
import numpy as np
from PIL import Image
from torchvision import transforms
from transformers import AutoImageProcessor, AutoModelForImageClassification
from timm.data import resolve_data_config, create_transform
from scipy.spatial.distance import cosine
import os
import cv2
import json
import mtcnn
import warnings

warnings.filterwarnings("ignore")

try:
    from retinaface import RetinaFace
    RETINAFACE_AVAILABLE = True
    print("RetinaFace available — 5-point face alignment enabled for ViT.")
except ImportError:
    print("WARNING: RetinaFace not found. ViT accuracy will be degraded.")
    RETINAFACE_AVAILABLE = False

REFERENCE_PTS_112 = np.array([
    [30.2946, 51.6963],
    [65.5318, 51.6963],
    [48.0252, 71.7366],
    [33.5493, 92.3655],
    [62.7299, 92.3655]
], dtype=np.float32)

REFERENCE_PTS_224 = REFERENCE_PTS_112 * 2.0

class ViTRecognizer:
    def __init__(self, model_name='vit_base_patch16_224'):
        print(f"Initializing ViT Recognizer ({model_name})...")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        try:
            self.model = timm.create_model(model_name, pretrained=True, num_classes=0)
        except Exception as e:
            print(f"Error loading ViT: {e}")
            raise
            
        self.model.to(self.device)
        self.model.eval()
        
        self.preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        print(f"ViT loaded on {self.device}.")

    def _align_face(self, img_path):
        """Detects face and warps it to the standard 224x224 template."""
        if not RETINAFACE_AVAILABLE:
            return Image.open(img_path).convert('RGB').resize((224, 224))

        resp = RetinaFace.detect_faces(img_path)
        
        if not resp or not isinstance(resp, dict):
            return Image.open(img_path).convert('RGB').resize((224, 224))

        best_face = None
        max_score = 0
        for key in resp:
            face = resp[key]
            if face['score'] > max_score:
                max_score = face['score']
                best_face = face

        if best_face is None:
             return Image.open(img_path).convert('RGB').resize((224, 224))

        lm = best_face['landmarks']
        src_pts = np.array([
            lm['right_eye'],
            lm['left_eye'],
            lm['nose'],
            lm['mouth_right'],
            lm['mouth_left']
        ], dtype=np.float32)

        M, _ = cv2.estimateAffinePartial2D(src_pts, REFERENCE_PTS_224)
        
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
             return Image.open(img_path).convert('RGB').resize((224, 224))
             
        warped = cv2.warpAffine(img_bgr, M, (224, 224))
        
        img_rgb = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)
        return Image.fromarray(img_rgb)

    def represent(self, img_input):
        """Generates embedding for a generic image input."""
        try:
            if isinstance(img_input, str):
                pil_img = self._align_face(img_input)
            elif isinstance(img_input, np.ndarray):
                pil_img = Image.fromarray(img_input).resize((224, 224))
            elif isinstance(img_input, Image.Image):
                pil_img = img_input
            else:
                return None

            img_tensor = self.preprocess(pil_img).unsqueeze(0).to(self.device)

            with torch.no_grad():
                embedding = self.model(img_tensor)
            
            return embedding.cpu().numpy().flatten()

        except Exception as e:
            return None

    def verify_pair(self, img1_path, img2_path, model_name="ViT", detector_backend="skip"):
        emb1 = self.represent(img1_path)
        emb2 = self.represent(img2_path)

        if emb1 is None or emb2 is None:
            return {"distance": 1.0}

        distance = cosine(emb1, emb2)
        
        threshold = 0.4

        return {
            'verified': distance <= threshold,
            'distance': float(distance),
            'threshold': threshold
        }


class HFViTRecognizer:
    """HuggingFace ViT face recognizer with MTCNN detection."""

    def __init__(self):
        print("Initializing HFViTRecognizer...")
        self.detector = mtcnn.MTCNN()
        model_id = "jayanta/vit-base-patch16-224-in21k-face-recognition"
        self.processor = AutoImageProcessor.from_pretrained(model_id)
        self.recognition_model = AutoModelForImageClassification.from_pretrained(model_id)
        self.recognition_model.eval()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.recognition_model.to(self.device)
        print(f"HF ViT loaded on {self.device.upper()}.")

    def _detect_and_crop_face(self, image_path):
        try:
            image = cv2.imread(image_path)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.detector.detect_faces(image_rgb)
            if results:
                x1, y1, w, h = results[0]['box']
                x1, y1 = abs(x1), abs(y1)
                return Image.fromarray(image_rgb[y1:y1+h, x1:x1+w])
            return None
        except Exception:
            return None

    def _get_embedding(self, face_pil):
        try:
            inputs = self.processor(images=face_pil, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.recognition_model(**inputs, output_hidden_states=True)
                embedding = outputs.hidden_states[-1].mean(dim=1).squeeze().cpu().numpy()
            return embedding
        except Exception:
            return None

    def verify_pair(self, img1_path, img2_path, model_name, detector_backend):
        face1 = self._detect_and_crop_face(img1_path)
        face2 = self._detect_and_crop_face(img2_path)
        if face1 is None or face2 is None:
            return None
        emb1 = self._get_embedding(face1)
        emb2 = self._get_embedding(face2)
        if emb1 is None or emb2 is None:
            return None
        distance = cosine(emb1, emb2)
        threshold = 0.6
        return {'verified': distance <= threshold, 'distance': distance, 'threshold': threshold}


class TimmFTRecognizer:
    """Fine-tuned timm ViT recognizer with MTCNN detection."""

    def __init__(self):
        print("Initializing TimmFTRecognizer...")
        RESULTS_DIR = os.environ.get('BENCHMARK_RESULTS_DIR', 'results')
        MODEL_PATH = os.path.join(RESULTS_DIR, "vit_timm_finetuned_2.pth")
        CLASS_MAP_PATH = os.path.join(RESULTS_DIR, "vit_timm_class_to_idx.json")
        MODEL_ID = "vit_base_patch16_224.augreg_in21k_ft_in1k"

        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model file not found at {MODEL_PATH}.")

        with open(CLASS_MAP_PATH, 'r') as f:
            num_classes = len(json.load(f))

        self.detector = mtcnn.MTCNN()
        self.recognition_model = timm.create_model(MODEL_ID, pretrained=False, num_classes=num_classes)
        self.recognition_model.load_state_dict(torch.load(MODEL_PATH))
        self.recognition_model.eval()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.recognition_model.to(self.device)
        data_config = resolve_data_config({}, model=self.recognition_model)
        self.transform = create_transform(**data_config)
        print(f"TimmFT ViT loaded from {MODEL_PATH} on {self.device.upper()}.")

    def _detect_and_crop_face(self, image_path):
        try:
            image = cv2.imread(image_path)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.detector.detect_faces(image_rgb)
            if results:
                x1, y1, w, h = results[0]['box']
                x1, y1 = abs(x1), abs(y1)
                return Image.fromarray(image_rgb[y1:y1+h, x1:x1+w])
            return None
        except Exception:
            return None

    def _get_embedding(self, face_pil):
        try:
            tensor_img = self.transform(face_pil).unsqueeze(0).to(self.device)
            with torch.no_grad():
                embedding = self.recognition_model.forward_features(tensor_img)
            return embedding.cpu().numpy().flatten()
        except Exception:
            return None

    def verify_pair(self, img1_path, img2_path, model_name, detector_backend):
        face1 = self._detect_and_crop_face(img1_path)
        face2 = self._detect_and_crop_face(img2_path)
        if face1 is None or face2 is None:
            return None
        emb1 = self._get_embedding(face1)
        emb2 = self._get_embedding(face2)
        if emb1 is None or emb2 is None:
            return None
        distance = cosine(emb1, emb2)
        threshold = 0.5
        return {'verified': distance <= threshold, 'distance': distance, 'threshold': threshold}