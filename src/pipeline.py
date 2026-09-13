import os
import argparse
import glob
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.v2 as T
import torchvision.models as models
import segmentation_models_pytorch as smp

import matplotlib
matplotlib.use("Agg")  # Prevent headless memory leaks and GUI backend overhead
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import binary_erosion, label
from tqdm import tqdm

from src.explainability import explain_wound_classification, DualBranchGradCAM

# Base Directory of Project
REPO_ROOT = Path(__file__).resolve().parent.parent

# --- 1. CONFIGURATION CONSTANTS ---
DEFAULT_WOUND_SEG = str(REPO_ROOT / "checkpoints" / "segmentation" / "retrained_best_wound_model.pth")
DEFAULT_TISSUE_SEG = str(REPO_ROOT / "checkpoints" / "tissue_segmentation" / "retrained_best_tissue_model.pth")
DEFAULT_CLASSIFIER = str(REPO_ROOT / "checkpoints" / "classification" / "retrained_best_dual_branch_classifier.pth")

# Corrected Semantic Mapping & Visual Palette:
# 0: Background
# 1: Fibrin / Slough (Yellow: [240, 220, 50])
# 2: Granulation (Red: [230, 40, 40])
# 3: Callus (Light Gray: [210, 210, 210])
TISSUE_COLORS = {
    1: np.array([240, 220, 50], dtype=np.uint8),   # Yellow - Fibrin / Slough
    2: np.array([230, 40, 40], dtype=np.uint8),    # Red - Granulation
    3: np.array([210, 210, 210], dtype=np.uint8)   # Light Gray - Callus
}

# --- 2. UNIVERSAL CHECKPOINT LOADER ---
def load_checkpoint_payload(checkpoint_path, device):
    """
    Universally loads:
      1. Legacy raw state_dict (pure parameter OrderedDict).
      2. Legacy classifier dict (keys: 'model_state', 'classes').
      3. New rich metadata dict (keys: 'model_state_dict', 'class_names', etc.).
    """
    data = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(data, dict):
        if "model_state_dict" in data:
            state_dict = data["model_state_dict"]
            class_names = data.get("class_names")
            return state_dict, class_names, data
        elif "model_state" in data:
            state_dict = data["model_state"]
            class_names = data.get("classes")
            return state_dict, class_names, data
        elif any("weight" in k or "bias" in k for k in data.keys()):
            return data, None, {}
    return data, None, {}

# --- 3. DUAL-BRANCH CLASSIFIER ARCHITECTURE ---
class DualBranchResNet34(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.global_backbone = models.resnet34(weights=None)
        in_feat = self.global_backbone.fc.in_features
        self.global_backbone.fc = nn.Identity()

        self.roi_backbone = models.resnet34(weights=None)
        self.roi_backbone.fc = nn.Identity()

        self.fusion_head = nn.Sequential(
            nn.Linear(in_feat * 2, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )

    def forward(self, x_global, x_roi):
        f_global = self.global_backbone(x_global)
        f_roi = self.roi_backbone(x_roi)
        f_combined = torch.cat([f_global, f_roi], dim=1)
        return self.fusion_head(f_combined)

# --- 4. MASTER CLINICAL SYSTEM ---
class MasterWoundSystem:
    def __init__(self, wound_seg_path, tissue_seg_path, classifier_path, fallback_scale=0.15, device=None):
        """
        fallback_scale: Assumed 0.15 mm/pixel spatial resolution (standard clinical photography heuristic).
        Note: Exact pixel counts are image-derived; physical cm²/mm dimensions are estimates based on fallback_scale.
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device_type = "cuda" if "cuda" in self.device else "cpu"
        self.fallback_scale = fallback_scale

        print(f"[INIT] Initializing Master Clinical AI Pipeline on {self.device.upper()}...")

        # 1. Binary Wound Bed Segmenter
        wound_state, _, _ = load_checkpoint_payload(wound_seg_path, self.device)
        self.wound_model = smp.DeepLabV3Plus(encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=1)
        self.wound_model.load_state_dict(wound_state)
        self.wound_model.to(self.device).eval()

        # 2. Multi-Class Tissue Segmenter (4 classes: 0=BG, 1=Fibrin, 2=Granulation, 3=Callus)
        tissue_state, _, _ = load_checkpoint_payload(tissue_seg_path, self.device)
        self.tissue_model = smp.DeepLabV3Plus(encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=4)
        self.tissue_model.load_state_dict(tissue_state)
        self.tissue_model.to(self.device).eval()

        # 3. Etiology Classifier (Supports both Dual-Branch and Single-Branch ResNet34)
        cls_state, cls_classes, _ = load_checkpoint_payload(classifier_path, self.device)
        self.class_names = cls_classes or {
            0: "Diabetic Foot Ulcer", 1: "Pressure Ulcer", 2: "Surgical Wound", 3: "Venous Ulcer"
        }
        # Normalize class_names keys to int if necessary
        self.class_names = {int(k): v for k, v in self.class_names.items()}

        if any("global_backbone" in k for k in cls_state.keys()):
            self.is_dual_branch = True
            self.cls_model = DualBranchResNet34(num_classes=len(self.class_names))
            self.cls_model.load_state_dict(cls_state)
        else:
            self.is_dual_branch = False
            self.cls_model = models.resnet34(weights=None)
            in_feat = self.cls_model.fc.in_features
            self.cls_model.fc = nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(in_feat, len(self.class_names))
            )
            self.cls_model.load_state_dict(cls_state)
        self.cls_model.to(self.device).eval()

        # Preprocessing Pipelines
        self.seg_transform = T.Compose([
            T.Resize((256, 256), interpolation=T.InterpolationMode.BILINEAR),
            T.ToDtype(torch.float32, scale=True),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.cls_transform = T.Compose([
            T.Resize((224, 224), interpolation=T.InterpolationMode.BILINEAR),
            T.ToDtype(torch.float32, scale=True),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def enable_mc_dropout(self):
        """Forces dropout layers into train mode while keeping batch norm in eval mode."""
        for m in self.wound_model.modules():
            if isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
                m.train()

    def extract_roi_crop(self, pil_img):
        w, h = pil_img.size
        crop_w = int(w * 0.65)
        crop_h = int(h * 0.65)
        x0 = (w - crop_w) // 2
        y0 = (h - crop_h) // 2
        return pil_img.crop((x0, y0, x0 + crop_w, y0 + crop_h))

    def predict_diagnosis(self, pil_image):
        t_global = self.cls_transform(T.functional.to_image(pil_image)).unsqueeze(0).to(self.device)

        with torch.inference_mode():
            with torch.amp.autocast(device_type=self.device_type, enabled=(self.device_type == "cuda")):
                if self.is_dual_branch:
                    roi_img = self.extract_roi_crop(pil_image)
                    t_roi = self.cls_transform(T.functional.to_image(roi_img)).unsqueeze(0).to(self.device)
                    logits = self.cls_model(t_global, t_roi)
                else:
                    logits = self.cls_model(t_global)
                probs = F.softmax(logits, dim=1).squeeze().cpu().numpy()

        pred_idx = int(np.argmax(probs))
        pred_label = self.class_names[pred_idx]
        pred_conf = float(probs[pred_idx])

        return {
            "predicted_type": pred_label,
            "confidence": round(pred_conf * 100.0, 1),
            "class_probabilities": {
                self.class_names[i]: round(float(probs[i]) * 100.0, 1)
                for i in range(len(self.class_names))
            }
        }

    def predict_explainability(self, pil_image, target_class=None):
        """
        Computes Grad-CAM explainability for the etiology classifier.
        Isolated in try...except so explainability failures never crash clinical inference.
        """
        try:
            return explain_wound_classification(
                model=self.cls_model,
                pil_image=pil_image,
                cls_transform=self.cls_transform,
                device=self.device,
                target_class=target_class
            )
        except Exception as e:
            # Graceful degradation - explainability is non-blocking
            return None

    def predict_wound_bed_and_uncertainty(self, pil_image, mc_samples=8):
        orig_w, orig_h = pil_image.size
        tensor = self.seg_transform(T.functional.to_image(pil_image)).unsqueeze(0).to(self.device)
        batch_tensor = tensor.repeat(mc_samples, 1, 1, 1)

        self.wound_model.eval()
        self.enable_mc_dropout()

        # Batched MC-Dropout with Test-Time Augmentation (Horizontal Flip)
        with torch.inference_mode():
            with torch.amp.autocast(device_type=self.device_type, enabled=(self.device_type == "cuda")):
                p1 = torch.sigmoid(self.wound_model(batch_tensor))
                p2 = torch.flip(torch.sigmoid(self.wound_model(torch.flip(batch_tensor, dims=[3]))), dims=[3])
                mean_p = (p1 + p2) / 2.0

                out = F.interpolate(mean_p, size=(orig_h, orig_w), mode="bilinear", align_corners=False)

            mean_map = out.mean(dim=0).squeeze().cpu().numpy()
            uncertainty_map = out.std(dim=0).squeeze().cpu().numpy()

        binary_mask = (mean_map > 0.5).astype(np.uint8)

        # Retain all significant wound components (>= 50 px and >= 5% of largest component)
        labeled, num_features = label(binary_mask)
        if num_features > 1:
            component_sizes = np.bincount(labeled.ravel())[1:]
            max_size = component_sizes.max()
            min_threshold = max(50, int(0.05 * max_size))
            valid_components = np.where(component_sizes >= min_threshold)[0] + 1
            binary_mask = np.isin(labeled, valid_components).astype(np.uint8)

        boundary_zone = (mean_map > 0.1) & (mean_map < 0.9)
        avg_unc = float(np.mean(uncertainty_map[boundary_zone])) if np.sum(boundary_zone) > 0 else float(np.mean(uncertainty_map))
        ai_confidence = round(float(max(0.0, 1.0 - (avg_unc * 2.0))) * 100.0, 1)

        return binary_mask, uncertainty_map, ai_confidence

    def predict_supervised_tissues(self, pil_image, binary_mask):
        """
        4-Class Tissue Prediction:
          0: Background, 1: Fibrin / Slough, 2: Granulation, 3: Callus
        """
        orig_w, orig_h = pil_image.size
        tensor = self.seg_transform(T.functional.to_image(pil_image)).unsqueeze(0).to(self.device)

        wound_pixels = binary_mask > 0
        total_wound_pixels = int(np.sum(wound_pixels))

        if total_wound_pixels == 0:
            return {
                "fibrin_pct": 0.0, "granulation_pct": 0.0, "callus_pct": 0.0,
                "tissue_map": np.zeros_like(binary_mask, dtype=np.uint8)
            }

        with torch.inference_mode():
            with torch.amp.autocast(device_type=self.device_type, enabled=(self.device_type == "cuda")):
                logits = self.tissue_model(tensor)
                logits_full = F.interpolate(logits, size=(orig_h, orig_w), mode="bilinear", align_corners=False)
                
                # Perform argmax across all 4 classes on full image
                full_preds = logits_full.argmax(dim=1).squeeze().cpu().numpy()

        # Mask multi-class prediction strictly within the segmented wound bed
        tissue_map = np.zeros_like(full_preds, dtype=np.uint8)
        tissue_map[wound_pixels] = full_preds[wound_pixels].astype(np.uint8)

        fibrin_count = np.sum(tissue_map == 1)
        gran_count = np.sum(tissue_map == 2)
        callus_count = np.sum(tissue_map == 3)

        return {
            "fibrin_pct": round(float((fibrin_count / total_wound_pixels) * 100.0), 1),
            "granulation_pct": round(float((gran_count / total_wound_pixels) * 100.0), 1),
            "callus_pct": round(float((callus_count / total_wound_pixels) * 100.0), 1),
            "tissue_map": tissue_map
        }

    def compute_morphometrics(self, binary_mask):
        """
        Calculates geometric and physical morphometrics.
        Note: Physical measurements assume fallback_scale (0.15 mm/px).
        """
        scale = self.fallback_scale
        wound_pixels = binary_mask > 0
        area_px = int(np.sum(wound_pixels))

        if area_px == 0:
            return {"area_pixels": 0, "area_cm2": 0.0, "perimeter_mm": 0.0, "circularity": 0.0, "is_irregular": False}

        area_cm2 = (area_px * (scale ** 2)) / 100.0
        eroded = binary_erosion(wound_pixels)
        perim_px = int(np.sum(wound_pixels ^ eroded))
        perim_mm = perim_px * scale
        circ = min(float((4.0 * np.pi * area_px) / (perim_px ** 2)), 1.0) if perim_px > 0 else 0.0

        return {
            "area_pixels": area_px,
            "area_cm2": round(float(area_cm2), 2),
            "perimeter_mm": round(float(perim_mm), 1),
            "circularity": round(float(circ), 3),
            "is_irregular": bool(circ < 0.55)
        }

    def calculate_severity(self, morph, tissue):
        if morph.get("area_pixels", 0) == 0:
            return {
                "severity_score": 0.0,
                "severity_grade": "No Active Lesion Detected",
                "recommended_action": "No active wound bed delineated. Clinical re-assessment advised if an unsegmented lesion is suspected."
            }

        area = morph["area_cm2"]
        score_size = 5.0 if area < 1.0 else (15.0 if area < 4.0 else (28.0 if area < 12.0 else 40.0))
        # Tissue severity: Higher fibrin/slough increases risk, granulation reduces risk
        score_tissue = (tissue["fibrin_pct"] * 0.35) + (tissue["callus_pct"] * 0.15) - (tissue["granulation_pct"] * 0.10)
        score_tissue = max(0.0, min(45.0, score_tissue))
        score_morph = 15.0 if morph["is_irregular"] else 5.0

        total = round(float(np.clip(score_size + score_tissue + score_morph, 0.0, 100.0)), 1)

        if total <= 25.0:
            grade, action = "Low (Mild)", "Routine clinical assessment and skin protection protocol recommended."
        elif total <= 55.0:
            grade, action = "Moderate (Chronic Risk)", "Clinical review advised; evaluate for pressure offloading, moisture management, and devitalized tissue."
        elif total <= 80.0:
            grade, action = "High (Severe / Stagnant)", "Specialist multidisciplinary wound assessment indicated; high proportion of devitalized tissue or critical size."
        else:
            grade, action = "Critical (Urgent)", "Immediate clinical evaluation indicated; severe composite risk across lesion dimensions and necrotic tissue."

        return {"severity_score": total, "severity_grade": grade, "recommended_action": action}

    def generate_report_figure(self, orig_rgb, binary_mask, unc_map, tissue_data, diag_data, morph_data, sev_data, save_path, explainability_data=None):
        if explainability_data is not None and "fused_overlay" in explainability_data:
            # 6-panel clinical diagnostic report (2 rows x 3 columns)
            fig, axes = plt.subplots(2, 3, figsize=(18, 11))

            # Row 1, Col 0: Input Photo & Predicted Etiology
            axes[0, 0].imshow(orig_rgb)
            axes[0, 0].set_title(f"Input Photo\nDiagnosis: {diag_data['predicted_type']} ({diag_data['confidence']}%)", fontsize=11, fontweight="bold")
            axes[0, 0].axis("off")

            # Row 1, Col 1: Wound Bed Delineation
            overlay = orig_rgb.copy()
            overlay[binary_mask > 0] = [0, 220, 255]
            blended_seg = (0.65 * orig_rgb + 0.35 * overlay).astype(np.uint8)
            axes[0, 1].imshow(blended_seg)
            axes[0, 1].set_title(f"Wound Delineation\nArea: {morph_data['area_cm2']} cm2 | Perim: {morph_data['perimeter_mm']} mm", fontsize=11)
            axes[0, 1].axis("off")

            # Row 1, Col 2: Supervised Tissue Bed Breakdown
            t_map = tissue_data["tissue_map"]
            t_vis = np.zeros_like(orig_rgb)
            for class_id, rgb_col in TISSUE_COLORS.items():
                t_vis[t_map == class_id] = rgb_col
            t_blend = orig_rgb.copy()
            wound_pixels = binary_mask > 0
            t_blend[wound_pixels] = (0.35 * orig_rgb[wound_pixels] + 0.65 * t_vis[wound_pixels]).astype(np.uint8)
            axes[0, 2].imshow(t_blend)
            axes[0, 2].set_title(
                f"Supervised Tissue (DFUTissue)\n"
                f"[Red] Gran: {tissue_data['granulation_pct']}% | "
                f"[Yellow] Fibrin: {tissue_data['fibrin_pct']}% | "
                f"[White] Callus: {tissue_data['callus_pct']}%",
                fontsize=10
            )
            axes[0, 2].axis("off")

            # Row 2, Col 0: Epistemic Uncertainty Heatmap & Risk Tier
            norm_unc = np.clip(unc_map / 0.3, 0, 1)
            heat_rgb = (plt.cm.inferno(norm_unc)[:, :, :3] * 255).astype(np.uint8)
            blended_unc = (0.55 * orig_rgb + 0.45 * heat_rgb).astype(np.uint8)
            axes[1, 0].imshow(blended_unc)
            axes[1, 0].set_title(f"Epistemic Uncertainty & Risk Tier\nTier: {sev_data['severity_grade']} ({sev_data['severity_score']}/100)", fontsize=10, fontweight="bold")
            axes[1, 0].axis("off")

            # Row 2, Col 1: Global Attention (Context) with Central 65% ROI Bounding Box
            g_overlay = explainability_data["global_overlay"].copy()
            roi_box = explainability_data.get("roi_box")
            if roi_box is not None:
                x0, y0, x1, y1 = roi_box
                H, W = g_overlay.shape[:2]
                x0_c, x1_c = max(0, x0), min(W - 1, x1)
                y0_c, y1_c = max(0, y0), min(H - 1, y1)
                t_border = max(2, int(min(H, W) * 0.006))
                for t in range(t_border):
                    g_overlay[max(0, y0_c - t):y0_c + t + 1, x0_c:x1_c + 1] = [0, 255, 255]
                    g_overlay[max(0, y1_c - t):y1_c + t + 1, x0_c:x1_c + 1] = [0, 255, 255]
                    g_overlay[y0_c:y1_c + 1, max(0, x0_c - t):x0_c + t + 1] = [0, 255, 255]
                    g_overlay[y0_c:y1_c + 1, max(0, x1_c - t):x1_c + t + 1] = [0, 255, 255]
            axes[1, 1].imshow(g_overlay)
            axes[1, 1].set_title("Classifier Attention (Global CAM)\n[Cyan Box: Central 65% ROI Stream]", fontsize=10)
            axes[1, 1].axis("off")

            # Row 2, Col 2: Fused Classifier Attribution Overlay
            axes[1, 2].imshow(explainability_data["fused_overlay"])
            axes[1, 2].set_title(
                f"Fused Classifier Attribution (Grad-CAM)\n"
                f"Target: {diag_data['predicted_type']} (Attribution - Not Causal)",
                fontsize=10, fontweight="bold"
            )
            axes[1, 2].axis("off")

            plt.tight_layout()
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            fig.clf()
            plt.close(fig)
        else:
            # 4-panel legacy layout
            fig, axes = plt.subplots(1, 4, figsize=(20, 5))

            axes[0].imshow(orig_rgb)
            axes[0].set_title(f"Input Photo\nDiagnosis: {diag_data['predicted_type']} ({diag_data['confidence']}%)", fontsize=11, fontweight="bold")
            axes[0].axis("off")

            overlay = orig_rgb.copy()
            overlay[binary_mask > 0] = [0, 220, 255]
            blended_seg = (0.65 * orig_rgb + 0.35 * overlay).astype(np.uint8)
            axes[1].imshow(blended_seg)
            axes[1].set_title(f"Wound Delineation\nArea: {morph_data['area_cm2']} cm2 | Perim: {morph_data['perimeter_mm']} mm", fontsize=11)
            axes[1].axis("off")

            t_map = tissue_data["tissue_map"]
            t_vis = np.zeros_like(orig_rgb)
            for class_id, rgb_col in TISSUE_COLORS.items():
                t_vis[t_map == class_id] = rgb_col

            t_blend = orig_rgb.copy()
            wound_pixels = binary_mask > 0
            t_blend[wound_pixels] = (0.35 * orig_rgb[wound_pixels] + 0.65 * t_vis[wound_pixels]).astype(np.uint8)
            axes[2].imshow(t_blend)
            axes[2].set_title(
                f"Supervised Tissue (DFUTissue)\n"
                f"[Red] Gran: {tissue_data['granulation_pct']}% | "
                f"[Yellow] Fibrin: {tissue_data['fibrin_pct']}% | "
                f"[White] Callus: {tissue_data['callus_pct']}%",
                fontsize=10
            )
            axes[2].axis("off")

            norm_unc = np.clip(unc_map / 0.3, 0, 1)
            heat_rgb = (plt.cm.inferno(norm_unc)[:, :, :3] * 255).astype(np.uint8)
            blended_unc = (0.55 * orig_rgb + 0.45 * heat_rgb).astype(np.uint8)
            axes[3].imshow(blended_unc)
            axes[3].set_title(f"Risk Tier: {sev_data['severity_grade']} ({sev_data['severity_score']}/100)\nAction: {sev_data['recommended_action'][:28]}...", fontsize=10, fontweight="bold")
            axes[3].axis("off")

            plt.tight_layout()
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            fig.clf()
            plt.close(fig)

    def process_image(self, img_input, run_explainability=True):
        if isinstance(img_input, Image.Image):
            pil_img = img_input.convert("RGB")
        else:
            pil_img = Image.open(img_input).convert("RGB")
        orig_rgb = np.array(pil_img)

        diag = self.predict_diagnosis(pil_img)
        binary_mask, unc_map, ai_conf = self.predict_wound_bed_and_uncertainty(pil_img)
        tissue = self.predict_supervised_tissues(pil_img, binary_mask)
        morph = self.compute_morphometrics(binary_mask)
        severity = self.calculate_severity(morph, tissue)

        explainability_data = None
        if run_explainability:
            explainability_data = self.predict_explainability(pil_img)

        clean_explainability = None
        if explainability_data is not None:
            clean_explainability = {
                "available": True,
                "method": "Grad-CAM (Dual-Branch ResNet34)",
                "target_class": diag["predicted_type"],
                "global_target_layer": explainability_data.get("global_target_layer"),
                "roi_target_layer": explainability_data.get("roi_target_layer"),
                "fusion_method": explainability_data.get("fusion_method"),
                "academic_notice": explainability_data.get("academic_notice", "Attribution visualization indicates model feature activation patterns, not clinical causality.")
            }

        return {
            "diagnostics": diag,
            "morphometrics": morph,
            "tissue_breakdown_supervised": {
                "fibrin_slough_percent": tissue["fibrin_pct"],
                "granulation_percent": tissue["granulation_pct"],
                "callus_percent": tissue["callus_pct"]
            },
            "severity_assessment": severity,
            "safety_qa": {
                "ai_confidence_score": ai_conf,
                "requires_clinician_review": bool(ai_conf < 75.0 or diag["confidence"] < 60.0)
            },
            "explainability": clean_explainability,
            "_raw": {
                "orig_rgb": orig_rgb,
                "binary_mask": binary_mask,
                "uncertainty_map": unc_map,
                "tissue_data": tissue,
                "explainability_data": explainability_data
            }
        }

# --- 4. EXECUTION CLI ---
def parse_args():
    parser = argparse.ArgumentParser(description="End-to-End Clinical Wound AI Pipeline")
    parser.add_argument("--input-dir", type=str, required=True, help="Directory containing target wound images")
    parser.add_argument("--output-dir", type=str, default=str(REPO_ROOT / "outputs" / "batch_reports"), help="Output directory")
    parser.add_argument("--wound-model", type=str, default=DEFAULT_WOUND_SEG, help="Path to binary segmenter checkpoint")
    parser.add_argument("--tissue-model", type=str, default=DEFAULT_TISSUE_SEG, help="Path to tissue model checkpoint")
    parser.add_argument("--classifier-model", type=str, default=DEFAULT_CLASSIFIER, help="Path to classifier checkpoint")
    parser.add_argument("--scale", type=float, default=0.15, help="Millimeters per pixel calibration scale")
    return parser.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    system = MasterWoundSystem(
        wound_seg_path=args.wound_model,
        tissue_seg_path=args.tissue_model,
        classifier_path=args.classifier_model,
        fallback_scale=args.scale
    )

    image_extensions = ("*.png", "*.jpg", "*.jpeg", "*.bmp")
    test_files = []
    for ext in image_extensions:
        test_files.extend(glob.glob(os.path.join(args.input_dir, ext)))
    test_files = sorted(test_files)

    if not test_files:
        print(f"⚠️ No images found matching {image_extensions} in {args.input_dir}")
        return

    print(f"Running Diagnostic Inference on {len(test_files)} images...\n")

    for img_p in tqdm(test_files):
        fname = Path(img_p).stem
        res = system.process_image(img_p)

        report_png = os.path.join(args.output_dir, f"report_{fname}.png")
        system.generate_report_figure(
            res["_raw"]["orig_rgb"],
            res["_raw"]["binary_mask"],
            res["_raw"]["uncertainty_map"],
            res["_raw"]["tissue_data"],
            res["diagnostics"],
            res["morphometrics"],
            res["severity_assessment"],
            report_png,
            explainability_data=res["_raw"].get("explainability_data")
        )

        clean_record = {k: v for k, v in res.items() if k != "_raw"}
        json_path = os.path.join(args.output_dir, f"data_{fname}.json")
        with open(json_path, "w") as f:
            json.dump(clean_record, f, indent=2)

    print(f"\n✅ Pipeline execution complete. Diagnostic outputs saved to: {args.output_dir}")

if __name__ == "__main__":
    main()