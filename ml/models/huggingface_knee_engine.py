"""
Hugging Face KneeGrade Inference & Grad-CAM Engine
Reconstructs the CNN ensemble from the Hugging Face space: afzal2003/kneegrade
Uses the exact binary weights:
  - exp1_augmentation.bin (1,175,845 parameters)
  - exp3_aug_wider.bin    (2,641,589 parameters)
Implements:
  1. Exact CNN forward pass (4 blocks of Conv2D + BatchNorm folded + ReLU + MaxPool + GAP + Dense)
  2. Prior-adjusted probabilities to account for dataset class imbalance
  3. Closed-form analytical Grad-CAM from the last conv feature maps
  4. Joint band localization ([70, 154, 20, 204])
"""

import os
import json
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List

SIZE = 224

class KneeMemberCNN(nn.Module):
    def __init__(self, spec: Dict[str, Any], weights_data: np.ndarray):
        super().__init__()
        self.name = spec["name"]
        self.channels = spec["dense"]["kernel"]["shape"][0] # 256 or 384
        
        # Build 8 conv layers (4 blocks of 2)
        conv_list = []
        for c in spec["convs"]:
            k_off = c["kernel"]["offset"]
            k_shape = c["kernel"]["shape"] # [3, 3, in_ch, out_ch]
            k_size = int(np.prod(k_shape))
            k_raw = weights_data[k_off : k_off + k_size].reshape(k_shape)
            # Transpose TF [kH, kW, in_ch, out_ch] -> PyTorch [out_ch, in_ch, kH, kW]
            k_pt = np.transpose(k_raw, (3, 2, 0, 1))
            
            b_off = c["bias"]["offset"]
            b_shape = c["bias"]["shape"]
            b_size = int(np.prod(b_shape))
            b_raw = weights_data[b_off : b_off + b_size]
            
            out_ch, in_ch, kh, kw = k_pt.shape
            conv = nn.Conv2d(in_ch, out_ch, kernel_size=(kh, kw), padding=1)
            conv.weight = nn.Parameter(torch.from_numpy(k_pt.copy()).float(), requires_grad=False)
            conv.bias = nn.Parameter(torch.from_numpy(b_raw.copy()).float(), requires_grad=False)
            conv_list.append(conv)
            
        self.convs = nn.ModuleList(conv_list)
        
        # Dense layer
        d_k_off = spec["dense"]["kernel"]["offset"]
        d_k_shape = spec["dense"]["kernel"]["shape"] # [channels, 5]
        d_k_size = int(np.prod(d_k_shape))
        d_k_raw = weights_data[d_k_off : d_k_off + d_k_size].reshape(d_k_shape)
        
        d_b_off = spec["dense"]["bias"]["offset"]
        d_b_shape = spec["dense"]["bias"]["shape"] # [5]
        d_b_size = int(np.prod(d_b_shape))
        d_b_raw = weights_data[d_b_off : d_b_off + d_b_size]
        
        self.fc = nn.Linear(self.channels, 5)
        # Transpose [channels, 5] -> PyTorch [5, channels]
        self.fc.weight = nn.Parameter(torch.from_numpy(d_k_raw.T.copy()).float(), requires_grad=False)
        self.fc.bias = nn.Parameter(torch.from_numpy(d_b_raw.copy()).float(), requires_grad=False)
        
        # Keep W for closed-form Grad-CAM: [channels, 5]
        self.W = d_k_raw.astype(np.float32)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: [1, 1, 224, 224]
        h = x
        feat = None
        for i, conv in enumerate(self.convs):
            h = F.relu(conv(h))
            if i == len(self.convs) - 1:
                feat = h # Last conv feature map: [1, channels, 28, 28]
            if i % 2 == 1:
                h = F.max_pool2d(h, kernel_size=2, stride=2)
                
        # Global Average Pooling: [1, channels]
        pooled = torch.mean(h, dim=(2, 3))
        logits = self.fc(pooled)
        probs = F.softmax(logits, dim=-1)
        return probs, feat


class HuggingFaceKneeEngine:
    def __init__(self, model_dir: Optional[str] = None):
        if model_dir is None:
            model_dir = str(Path(__file__).parent / "huggingface_knee")
            
        self.model_dir = Path(model_dir)
        self.cfg_path = self.model_dir / "config.json"
        self.is_ready = False
        self.members: List[KneeMemberCNN] = []
        self.cfg: Dict[str, Any] = {}
        
        self.load_models()

    def load_models(self):
        if not self.cfg_path.exists():
            print(f"[-] Config not found at {self.cfg_path}")
            return
            
        with open(self.cfg_path, "r", encoding="utf-8") as f:
            self.cfg = json.load(f)
            
        self.members = []
        for m_spec in self.cfg["members"]:
            bin_path = self.model_dir / m_spec["file"]
            if not bin_path.exists():
                print(f"[-] Binary weights missing: {bin_path}")
                continue
            weights_data = np.fromfile(str(bin_path), dtype=np.float32)
            member = KneeMemberCNN(m_spec, weights_data)
            member.eval()
            self.members.append(member)
            print(f"[+] Loaded Hugging Face ensemble member: {m_spec['name']} ({m_spec['params']} parameters)")
            
        if len(self.members) > 0:
            self.is_ready = True
            print(f"[+] HuggingFaceKneeEngine initialized successfully with {len(self.members)} models.")

    def polarity_score(self, img_224: np.ndarray) -> float:
        """
        Center minus edge brightness. Negative means the X-ray is inverted.
        Matches js/preprocess.js polarityScore().
        """
        center = img_224[40:184, 62:162]
        left_edge = img_224[:, :20]
        right_edge = img_224[:, -20:]
        edges = np.concatenate([left_edge.ravel(), right_edge.ravel()])
        return float(np.mean(center) - np.mean(edges))

    def fix_polarity(self, img_224: np.ndarray) -> np.ndarray:
        if self.polarity_score(img_224) < 0:
            return 255 - img_224
        return img_224

    def stretch(self, img_224: np.ndarray) -> np.ndarray:
        """
        1st-99th percentile stretch measured strictly on the central 144x144 joint box.
        Matches js/preprocess.js stretch().
        """
        center_144 = img_224[40:184, 40:184]
        p_lo, p_hi = np.percentile(center_144, [1, 99])
        if p_hi <= p_lo:
            return img_224
        stretched = np.clip((img_224.astype(np.float32) - p_lo) * (255.0 / (p_hi - p_lo)), 0, 255)
        return stretched.astype(np.uint8)

    def preprocess_radiograph(self, image_input: np.ndarray) -> np.ndarray:
        """
        Matches js/preprocess.js KneePreprocess.preprocess():
        polarity fix -> 1st-99th percentile stretch on central joint region -> CLAHE (2.0, 8x8).
        """
        if len(image_input.shape) == 3 and image_input.shape[2] == 3:
            gray = cv2.cvtColor(image_input, cv2.COLOR_RGB2GRAY)
        else:
            gray = image_input

        # Resize to 224x224 (cv2.INTER_AREA for downscale, INTER_LINEAR for upscale)
        h, w = gray.shape[:2]
        interp = cv2.INTER_AREA if (w >= SIZE and h >= SIZE) else cv2.INTER_LINEAR
        resized = cv2.resize(gray, (SIZE, SIZE), interpolation=interp)

        # 1. Polarity fix
        fixed = self.fix_polarity(resized)

        # 2. Central joint percentile stretch
        stretched = self.stretch(fixed)

        # 3. OpenCV CLAHE (clipLimit 2.0, 8x8 tiles)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(stretched)
        return enhanced

    def predict(self, image_array: np.ndarray) -> Dict[str, Any]:
        """
        Runs ensemble forward pass, prior-adjusted probabilities, and analytical Grad-CAM.
        """
        if not self.is_ready:
            raise RuntimeError("Hugging Face models not loaded.")

        preprocessed = self.preprocess_radiograph(image_array)
        tensor = torch.from_numpy(preprocessed.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0) # [1, 1, 224, 224]

        all_probs = []
        all_feats = []

        with torch.no_grad():
            for member in self.members:
                probs, feat = member(tensor)
                all_probs.append(probs[0].numpy())
                all_feats.append(feat[0].numpy())

        # Ensemble average probabilities
        mean_probs = np.mean(all_probs, axis=0) # [5]

        # Prior probability adjustment to correct for dataset class imbalance:
        counts = np.array(self.cfg.get("train_counts", [2286, 1046, 1516, 757, 173]), dtype=np.float32)
        total_counts = np.sum(counts)
        raw_adjusted = mean_probs / (counts / total_counts)
        adjusted_scores = raw_adjusted / np.sum(raw_adjusted)

        predicted_grade = int(np.argmax(adjusted_scores))
        confidence = float(adjusted_scores[predicted_grade])

        # Closed-form Grad-CAM for the predicted grade
        cams = []
        for i, member in enumerate(self.members):
            probs_m = all_probs[i]
            feat_m = all_feats[i] # [C, 28, 28]
            C = member.channels
            mix = member.W @ probs_m # [C]
            w = probs_m[predicted_grade] * (member.W[:, predicted_grade] - mix) # [C]

            # Weighted combination of feature maps
            cam_28 = np.tensordot(w, feat_m, axes=(0, 0)) # [28, 28]
            cam_28 = np.maximum(cam_28, 0.0)

            # Resize to 224x224
            cam_224 = cv2.resize(cam_28, (SIZE, SIZE), interpolation=cv2.INTER_LINEAR)
            max_v = cam_224.max()
            if max_v > 0:
                cam_224 = cam_224 / max_v
            cams.append(cam_224)

        # Average ensemble Grad-CAM
        mean_cam = np.mean(cams, axis=0)
        mean_cam = (mean_cam - mean_cam.min()) / (mean_cam.max() - mean_cam.min() + 1e-8)

        # Joint share
        r0, r1, c0, c1 = self.cfg.get("joint_band", [70, 154, 20, 204])
        total_act = np.sum(mean_cam)
        joint_act = np.sum(mean_cam[r0:r1, c0:c1])
        joint_share = float(joint_act / total_act) if total_act > 0 else 0.0

        return {
            "predicted_grade": predicted_grade,
            "confidence": confidence,
            "probabilities": [float(p) for p in mean_probs],
            "adjusted_scores": [float(p) for p in adjusted_scores],
            "cam": mean_cam,
            "joint_share": joint_share,
            "preprocessed_image": preprocessed,
            "grade_names": self.cfg.get("grade_names", ["Healthy", "Doubtful", "Minimal", "Moderate", "Severe"]),
            "model_repo": self.cfg.get("model_repo", "afzal2003/knee-oa-kl-grading")
        }

# Global singleton
hf_knee_engine = HuggingFaceKneeEngine()
