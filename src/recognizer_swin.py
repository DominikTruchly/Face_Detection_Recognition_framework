"""
src/recognizer_swin.py
-----------------------
SwinFace (swin_t backbone + FAM) recognizer trained on MS1MV2.
Uses RetinaFace for 5-point alignment to a padded 112x112 template (padding
helps detect near-edge faces in tightly-cropped chips). Requires
weights/SwinFace_MS1MV2.pth. Use --detector skip in run_verification.py.
"""
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import os
import sys
import types
import cv2

current_dir = os.path.dirname(os.path.abspath(__file__))
swin_code_path = os.path.join(current_dir, "swinface_code")
if swin_code_path not in sys.path:
    sys.path.append(swin_code_path)

try:
    from retinaface import RetinaFace
    RETINAFACE_AVAILABLE = True
except ImportError:
    print("WARNING: RetinaFace not found.")
    RETINAFACE_AVAILABLE = False

try:
    from model import build_model
except ImportError:
    try:
        from src.swinface_code.model import build_model
    except ImportError:
        print(f"CRITICAL ERROR: Could not import 'build_model'.")
        raise

# Standard 5-point alignment template (112x112)
REFERENCE_PTS = np.array([
    [30.2946, 51.6963], [65.5318, 51.6963],
    [48.0252, 71.7366],
    [33.5493, 92.3655], [62.7299, 92.3655]
], dtype=np.float32)

def _align_face(img_input):
    """Detects and aligns a face to the 112x112 template using RetinaFace.
    Pads the image by its own size first so near-edge faces in tightly-cropped
    chips are still detected reliably."""
    if isinstance(img_input, str):
        img_bgr = cv2.imread(img_input)
    elif isinstance(img_input, np.ndarray):
        img_bgr = img_input
    else:
        return None

    if img_bgr is None: return None

    h, w = img_bgr.shape[:2]
    # Pad by the image's own size so near-edge faces are still detected
    img_padded = cv2.copyMakeBorder(img_bgr, h, h, w, w, cv2.BORDER_CONSTANT, value=[0,0,0])

    resp = RetinaFace.detect_faces(img_padded)

    if not resp or not isinstance(resp, dict):
        resp = RetinaFace.detect_faces(img_bgr)
        used_padded = False
    else:
        used_padded = True

    if not resp or not isinstance(resp, dict):
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        return Image.fromarray(img_rgb).resize((112, 112))

    best_face = None
    max_score = 0
    for key in resp:
        face = resp[key]
        if face['score'] > max_score:
            max_score = face['score']
            best_face = face

    if best_face is None:
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        return Image.fromarray(img_rgb).resize((112, 112))

    lm = best_face['landmarks']
    
    offset_x = w if used_padded else 0
    offset_y = h if used_padded else 0

    src_pts = np.array([
        lm['right_eye'], lm['left_eye'], lm['nose'],
        lm['mouth_right'], lm['mouth_left']
    ], dtype=np.float32)

    src_pts[:, 0] -= offset_x
    src_pts[:, 1] -= offset_y

    M, _ = cv2.estimateAffinePartial2D(src_pts, REFERENCE_PTS)
    warped = cv2.warpAffine(img_bgr, M, (112, 112))
    img_rgb = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)
    
    return Image.fromarray(img_rgb)

class SwinFaceConfig:
    def __init__(self):
        self.network = "swin_t"
        self.embedding_size = 512
        self.fam_kernel_size = 3
        self.fam_in_chans = 2112
        self.fam_conv_shared = False
        self.fam_conv_mode = "split"
        self.fam_channel_attention = "CBAM"
        self.fam_spatial_attention = None
        self.fam_pooling = "max"
        self.fam_la_num_list = [2 for j in range(11)]
        self.fam_feature = "all"

def patched_forward(self, x):
    local_features, global_features, embedding = self.forward_features(x)
    part1_192 = local_features[:, 96:288, :, :]
    part2_384 = local_features[:, 288:, :, :]
    part3_768 = global_features
    new_local_features = torch.cat([part1_192, part2_384, part3_768, part3_768], dim=1)
    return new_local_features, global_features, embedding

class SwinFaceRecognizer:
    def __init__(self, model_path='weights/SwinFace_MS1MV2.pth'):
        print(f"Initializing SwinFace Recognizer (Crop-Safe Alignment)...")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        cfg = SwinFaceConfig()
        self.model = build_model(cfg)
        self.model.backbone.forward = types.MethodType(patched_forward, self.model.backbone)
        
        if not os.path.exists(model_path):
            project_weights = "/content/drive/MyDrive/masters_thesis/face_detection_recognition/weights/SwinFace_MS1MV2.pth"
            if os.path.exists(project_weights):
                model_path = project_weights
            else:
                raise FileNotFoundError(f"SwinFace weights not found at {model_path}")

        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            if "state_dict_backbone" in checkpoint:
                self.model.backbone.load_state_dict(checkpoint["state_dict_backbone"])
                self.model.fam.load_state_dict(checkpoint["state_dict_fam"])
                self.model.tss.load_state_dict(checkpoint["state_dict_tss"])
                self.model.om.load_state_dict(checkpoint["state_dict_om"])
            else:
                state_dict = checkpoint.get('state_dict', checkpoint.get('model', checkpoint))
                self.model.load_state_dict(state_dict, strict=False)
            print(f"SwinFace weights loaded successfully.")
        except Exception as e:
            print(f"Error loading weights: {e}")
            raise
        
        self.model.to(self.device)
        self.model.eval()
        
        self.preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    def represent(self, img_input):
        try:
            pil_img = _align_face(img_input)
            if pil_img is None: return None
            
            img_tensor = self.preprocess(pil_img).unsqueeze(0).to(self.device)

            with torch.no_grad():
                output = self.model(img_tensor)
                if isinstance(output, dict):
                    if 'Recognition' in output: embedding = output['Recognition']
                    elif 'id' in output: embedding = output['id']
                    else: embedding = list(output.values())[-1]
                elif isinstance(output, (list, tuple)):
                    embedding = output[-1]
                else:
                    embedding = output

            emb_np = embedding.cpu().numpy().flatten()
            norm = np.linalg.norm(emb_np)
            if norm == 0: return emb_np
            return emb_np / norm
        except Exception as e:
            return None

    def verify_pair(self, img1_path, img2_path, model_name="SwinFace", detector_backend="skip"):
        emb1 = self.represent(img1_path)
        emb2 = self.represent(img2_path)
        if emb1 is None or emb2 is None: return {"distance": 1.0}
        similarity = np.dot(emb1, emb2)
        distance = float(1.0 - similarity)
        threshold = 0.35
        return {"distance": distance, "threshold": threshold, "verified": distance <= threshold}