"""
src/explainability.py
======================
Grad-CAM Visual Explainability Module for DualBranchResNet34 Classifier.

Provides attribution heatmaps for the dual-branch wound etiology classifier:
  1. Global Branch CAM: Attends to surrounding skin, anatomy, and global framing.
  2. ROI Branch CAM: Attends to the central 65% lesion bed micro-texture and wound bed morphology.
  3. Fused CAM: Mathematically mapped and normalized combination in original image coordinates.

Academic Notice:
  Attribution visualizations highlight model feature activation patterns.
  They do NOT establish clinical causality or standalone diagnostic certainty.
"""

from typing import Dict, Any, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.v2 as T
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

class DualBranchGradCAM:
    """
    Grad-CAM implementation specifically designed for DualBranchResNet34.
    
    Target Convolutional Layers:
      - global_backbone.layer4 (Shape: (B, 512, 7, 7))
      - roi_backbone.layer4    (Shape: (B, 512, 7, 7))
    """
    def __init__(self, model: nn.Module, device: Optional[str] = None):
        self.model = model
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        if hasattr(model, "global_backbone") and hasattr(model, "roi_backbone"):
            self.is_dual_branch = True
            self.global_target_layer = model.global_backbone.layer4
            self.roi_target_layer = model.roi_backbone.layer4
            self.global_layer_name = "global_backbone.layer4"
            self.roi_layer_name = "roi_backbone.layer4"
        elif hasattr(model, "layer4"):
            self.is_dual_branch = False
            self.global_target_layer = model.layer4
            self.roi_target_layer = None
            self.global_layer_name = "layer4"
            self.roi_layer_name = None
        else:
            raise ValueError(f"Model {type(model)} does not have recognized ResNet layer4 stages.")

    @staticmethod
    def _compute_branch_cam(activation: torch.Tensor, gradient: torch.Tensor) -> np.ndarray:
        """
        Computes standard Grad-CAM from activation tensor A and gradient tensor G.
          alpha_k = (1 / HW) * sum_{i, j} G_{k, i, j}
          cam = ReLU(sum_k alpha_k * A_k)
        """
        weights = gradient.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        cam = (weights * activation).sum(dim=1, keepdim=True)  # (1, 1, H, W)
        cam = F.relu(cam)
        cam_np = cam.squeeze().detach().cpu().float().numpy()

        c_min, c_max = float(cam_np.min()), float(cam_np.max())
        if c_max > c_min + 1e-8:
            cam_np = (cam_np - c_min) / (c_max - c_min + 1e-8)
        else:
            cam_np = np.zeros_like(cam_np)
        return cam_np

    def generate_cam(
        self,
        t_global: torch.Tensor,
        t_roi: Optional[torch.Tensor],
        orig_size: Tuple[int, int],  # (width, height)
        roi_box: Optional[Tuple[int, int, int, int]] = None,  # (x0, y0, x1, y1)
        target_class: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Computes Grad-CAM for global, ROI, and fused branches.
        Guarantees zero lingering hooks and zero parameter gradients on exit.
        """
        orig_w, orig_h = orig_size
        activations = {}
        gradients = {}
        hook_handles = []

        def get_hook(name: str):
            def forward_hook(module, inp, out):
                activations[name] = out
                out.register_hook(lambda grad: gradients.__setitem__(name, grad))
            return forward_hook

        # Register forward hooks to capture activations and register backward gradient hooks
        hook_handles.append(self.global_target_layer.register_forward_hook(get_hook("global")))
        if self.is_dual_branch and self.roi_target_layer is not None:
            hook_handles.append(self.roi_target_layer.register_forward_hook(get_hook("roi")))

        was_training = self.model.training
        self.model.eval()

        try:
            with torch.enable_grad():
                x_g = t_global.to(self.device)
                if self.is_dual_branch:
                    if t_roi is None:
                        raise ValueError("t_roi is required for DualBranchResNet34 explainability.")
                    x_r = t_roi.to(self.device)
                    logits = self.model(x_g, x_r)
                else:
                    logits = self.model(x_g)

                if target_class is None:
                    target_class = int(logits.argmax(dim=1).item())

                score = logits[0, target_class]

                self.model.zero_grad(set_to_none=True)
                score.backward(retain_graph=False)

            # 1. Global CAM
            act_g = activations.get("global")
            grad_g = gradients.get("global")
            if act_g is None or grad_g is None:
                raise RuntimeError("Failed to capture global activations or gradients.")

            global_cam_raw = self._compute_branch_cam(act_g, grad_g)

            # Resize to original image dimensions (orig_h, orig_w)
            global_cam_tensor = torch.from_numpy(global_cam_raw).unsqueeze(0).unsqueeze(0)
            global_cam_resized = F.interpolate(
                global_cam_tensor, size=(orig_h, orig_w), mode="bilinear", align_corners=False
            ).squeeze().numpy()

            g_min, g_max = float(global_cam_resized.min()), float(global_cam_resized.max())
            if g_max > g_min + 1e-8:
                global_cam = (global_cam_resized - g_min) / (g_max - g_min + 1e-8)
            else:
                global_cam = np.zeros_like(global_cam_resized)

            roi_cam = None
            roi_cam_full = None

            # 2. ROI CAM (Mapped back to original coordinates)
            if self.is_dual_branch and "roi" in activations and "roi" in gradients:
                act_r = activations["roi"]
                grad_r = gradients["roi"]
                roi_cam_raw = self._compute_branch_cam(act_r, grad_r)

                if roi_box is not None:
                    x0, y0, x1, y1 = roi_box
                else:
                    crop_w = int(orig_w * 0.65)
                    crop_h = int(orig_h * 0.65)
                    x0 = (orig_w - crop_w) // 2
                    y0 = (orig_h - crop_h) // 2
                    x1 = x0 + crop_w
                    y1 = y0 + crop_h

                crop_w = max(1, x1 - x0)
                crop_h = max(1, y1 - y0)

                roi_cam_tensor = torch.from_numpy(roi_cam_raw).unsqueeze(0).unsqueeze(0)
                roi_crop_resized = F.interpolate(
                    roi_cam_tensor, size=(crop_h, crop_w), mode="bilinear", align_corners=False
                ).squeeze().numpy()

                r_min, r_max = float(roi_crop_resized.min()), float(roi_crop_resized.max())
                if r_max > r_min + 1e-8:
                    roi_cam = (roi_crop_resized - r_min) / (r_max - r_min + 1e-8)
                else:
                    roi_cam = np.zeros_like(roi_crop_resized)

                # Project strictly into the central crop window on the full-image canvas
                roi_cam_full = np.zeros((orig_h, orig_w), dtype=np.float32)
                roi_cam_full[y0:y1, x0:x1] = roi_cam

                # 3. Fused CAM: Normalized average of Global CAM + Full-Canvas ROI CAM
                fused = (global_cam + roi_cam_full) / 2.0
                f_min, f_max = float(fused.min()), float(fused.max())
                if f_max > f_min + 1e-8:
                    fused_cam = (fused - f_min) / (f_max - f_min + 1e-8)
                else:
                    fused_cam = np.zeros_like(fused)
            else:
                fused_cam = global_cam.copy()

            return {
                "global_cam": global_cam,
                "roi_cam": roi_cam,
                "roi_cam_full": roi_cam_full,
                "fused_cam": fused_cam,
                "target_class": target_class,
                "global_target_layer": self.global_layer_name,
                "roi_target_layer": self.roi_layer_name,
                "fusion_method": "normalize((global_cam + roi_cam_full) / 2)" if self.is_dual_branch else "single_branch_passthrough",
                "roi_box": (x0, y0, x1, y1) if (self.is_dual_branch and roi_box is not None) else None
            }

        finally:
            # Guaranteed cleanup of all attached hooks
            for handle in hook_handles:
                try:
                    handle.remove()
                except Exception:
                    pass
            # Guaranteed zeroing of model parameter gradients to avoid leaks
            self.model.zero_grad(set_to_none=True)
            if was_training:
                self.model.train()

def overlay_cam_on_image(
    orig_rgb: np.ndarray,
    cam: np.ndarray,
    alpha: float = 0.45,
    colormap_name: str = "jet"
) -> np.ndarray:
    """
    Overlays a 2D heatmap bounded in [0, 1] onto an RGB uint8 image.
    """
    cmap = plt.get_cmap(colormap_name)
    heat_rgba = cmap(np.clip(cam, 0.0, 1.0))
    heat_rgb = (heat_rgba[:, :, :3] * 255).astype(np.uint8)
    blended = (alpha * heat_rgb + (1.0 - alpha) * orig_rgb).astype(np.uint8)
    return blended

def explain_wound_classification(
    model: nn.Module,
    pil_image: Image.Image,
    cls_transform,
    device: Optional[str] = None,
    target_class: Optional[int] = None
) -> Dict[str, Any]:
    """
    Convenience function to run complete explainability on a PIL image.
    Extracts the central 65% crop, runs DualBranchGradCAM, and generates overlays.
    """
    orig_w, orig_h = pil_image.size
    orig_rgb = np.array(pil_image.convert("RGB"))

    # Extract heuristic central 65% ROI crop (identical to production pipeline)
    crop_w = int(orig_w * 0.65)
    crop_h = int(orig_h * 0.65)
    x0 = (orig_w - crop_w) // 2
    y0 = (orig_h - crop_h) // 2
    x1 = x0 + crop_w
    y1 = y0 + crop_h
    roi_box = (x0, y0, x1, y1)

    roi_img = pil_image.crop((x0, y0, x1, y1))

    # Construct input tensors
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    t_global = cls_transform(T.functional.to_image(pil_image)).unsqueeze(0).to(dev)
    t_roi = cls_transform(T.functional.to_image(roi_img)).unsqueeze(0).to(dev)

    explainer = DualBranchGradCAM(model, device=dev)
    cam_results = explainer.generate_cam(
        t_global=t_global,
        t_roi=t_roi,
        orig_size=(orig_w, orig_h),
        roi_box=roi_box,
        target_class=target_class
    )

    # Generate visual overlays
    fused_cam = cam_results["fused_cam"]
    global_cam = cam_results["global_cam"]
    roi_cam_full = cam_results["roi_cam_full"]

    fused_overlay = overlay_cam_on_image(orig_rgb, fused_cam, alpha=0.45)
    global_overlay = overlay_cam_on_image(orig_rgb, global_cam, alpha=0.45)
    roi_overlay = overlay_cam_on_image(orig_rgb, roi_cam_full, alpha=0.45) if roi_cam_full is not None else None

    return {
        "global_cam": global_cam,
        "roi_cam": cam_results["roi_cam"],
        "roi_cam_full": roi_cam_full,
        "fused_cam": fused_cam,
        "fused_overlay": fused_overlay,
        "global_overlay": global_overlay,
        "roi_overlay": roi_overlay,
        "target_class": cam_results["target_class"],
        "global_target_layer": cam_results["global_target_layer"],
        "roi_target_layer": cam_results["roi_target_layer"],
        "fusion_method": cam_results["fusion_method"],
        "roi_box": roi_box,
        "academic_notice": "Attribution visualization indicates model feature activation patterns, not clinical causality."
    }
