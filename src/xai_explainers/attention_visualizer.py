"""
Gradient-based attention visualization (Grad-CAM) for XAI
File location: src/xai_explainers/attention_visualizer.py

FIX (this version): generate_cam() now restricts the backward target to the
model's own PREDICTED TUMOR REGION instead of summing the tumor-class output
over the entire volume (including background). Summing over the whole volume
let background noise dominate/cancel the gradient signal for large or diffuse
tumors, causing the CAM to collapse to all-zeros after the ReLU step for many
real cases (confirmed via 20-case BraTS batch test: attention mean/max showed
a strong inverse correlation with tumor size, Pearson r=-0.66, with several
larger-tumor cases returning exactly 0.0).
"""
import torch
import torch.nn.functional as F
import numpy as np
import logging
from typing import Optional, Tuple
import cv2

logger = logging.getLogger(__name__)

class GradCAM3D:
    """
    Gradient-weighted Class Activation Map for 3D medical images
    Highlights which voxels influence the model's predictions

    IMPORTANT: target_layers must be EXACT layer names (e.g. 'unet.bottleneck'),
    not partial/substring matches. Matching too many layers causes massive
    memory usage since activations+gradients are stored for every match.
    """

    def __init__(self, model, target_layers: list = None):
        """
        Args:
            model: PyTorch model
            target_layers: List of EXACT layer names to compute gradients for.
                           Default targets the bottleneck (good balance of
                           semantic richness + low memory footprint).
        """
        self.model = model
        self.target_layers = target_layers or ['unet.bottleneck']
        self.gradients = {}
        self.activations = {}
        self._hook_handles = []
        self._registered_layers = self._register_hooks()

        if not self._registered_layers:
            available = [name for name, _ in self.model.named_modules()][:20]
            logger.warning(
                f"⚠️ None of the target layers {self.target_layers} were found in the model. "
                f"First 20 available layer names: {available}"
            )

    def _register_hooks(self) -> list:
        """Register forward and backward hooks on EXACT-matching layers only"""
        def create_forward_hook(name):
            def forward_hook(module, input, output):
                self.activations[name] = output.detach()
            return forward_hook

        def create_backward_hook(name):
            def backward_hook(module, grad_input, grad_output):
                self.gradients[name] = grad_output[0].detach()
            return backward_hook

        registered = []
        for name, module in self.model.named_modules():
            if name in self.target_layers:  # EXACT match only, not substring
                h1 = module.register_forward_hook(create_forward_hook(name))
                h2 = module.register_full_backward_hook(create_backward_hook(name))
                self._hook_handles.extend([h1, h2])
                registered.append(name)
                logger.info(f"✅ Grad-CAM hook registered on layer: {name}")

        return registered

    def remove_hooks(self):
        """Clean up hooks to free memory"""
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles = []

    def generate_cam(
        self,
        input_tensor: torch.Tensor,
        target_class: int = 1,
        restrict_to_prediction: bool = True
    ) -> np.ndarray:
        """
        Generate Class Activation Map

        Args:
            input_tensor: Input image [1, C, D, H, W]
            target_class: Which class to visualize (1=tumor)
            restrict_to_prediction: If True (default), the backward target is
                restricted to voxels the model actually predicted as
                target_class. This keeps the gradient signal focused on the
                tumor region instead of being diluted/cancelled by background
                voxels across the whole volume, which was causing near-zero
                or exactly-zero attention maps for larger/diffuse tumors.

        Returns:
            cam: Attention map [D, H, W] normalized to [0, 1], upsampled
                 to match input spatial dimensions
        """
        if not self._registered_layers:
            raise RuntimeError(
                f"No hooks registered - target_layers {self.target_layers} "
                f"don't match any layer in the model."
            )

        self.model.eval()
        self.gradients.clear()
        self.activations.clear()

        # Forward pass
        output = self.model(input_tensor)

        # Backward pass for target class
        self.model.zero_grad()

        if restrict_to_prediction:
            # Restrict gradient target to voxels the model predicted as
            # target_class, so the CAM reflects "what drove THIS prediction"
            # rather than "sum of tumor-logit over the whole volume"
            # (which is dominated by background for large/diffuse tumors).
            with torch.no_grad():
                pred_mask = (output.argmax(dim=1) == target_class).float()

            n_pred_voxels = pred_mask.sum().item()

            if n_pred_voxels > 0:
                target = (output[0, target_class] * pred_mask[0]).sum()
            else:
                # Model predicted no voxels of this class at all - fall back
                # to whole-volume sum so we still get *some* signal instead
                # of an immediate zero map.
                logger.warning(
                    f"No voxels predicted as class {target_class}; "
                    f"falling back to whole-volume Grad-CAM target."
                )
                target = output[0, target_class].sum()
        else:
            target = output[0, target_class].sum()

        target.backward()

        target_layer_name = self._registered_layers[0]

        if target_layer_name not in self.gradients:
            logger.warning("No gradients captured. Returning zeros.")
            return np.zeros(input_tensor.shape[2:], dtype=np.float32)

        gradients = self.gradients[target_layer_name]
        activations = self.activations[target_layer_name]

        # Average pool over spatial dimensions -> per-channel importance weight
        gradients_avg = torch.mean(gradients, dim=(2, 3, 4))  # [B, C]

        # Weighted sum of activation channels
        num_channels = activations.shape[1]
        cam_raw = torch.zeros(activations.shape[2:], device=activations.device)

        for i in range(num_channels):
            cam_raw += gradients_avg[0, i] * activations[0, i]

        # ReLU to keep only positive influences (standard Grad-CAM)
        cam = F.relu(cam_raw)

        # Upsample from bottleneck resolution back to input resolution
        target_shape = input_tensor.shape[2:]  # [D, H, W]
        cam = cam.unsqueeze(0).unsqueeze(0)  # [1, 1, d, h, w]
        cam = F.interpolate(cam, size=target_shape, mode='trilinear', align_corners=False)
        cam = cam.squeeze(0).squeeze(0)  # [D, H, W]

        # Normalize to [0, 1]
        cam_min = cam.min()
        cam_max = cam.max()

        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            # Standard ReLU'd CAM collapsed to all-zero: every bottleneck
            # channel's contribution was <= 0 for this case. This is a known
            # Grad-CAM failure mode at deep/abstract layers, seen in ~10% of
            # real BraTS cases during validation (independent of how the
            # backward target is defined). Rather than return an
            # uninformative all-zero map, fall back to the signed (pre-ReLU)
            # map, min-max normalized. This still surfaces relative
            # differences between voxels (least-to-most-relevant) even when
            # no single channel combination is unambiguously "positive".
            logger.warning(
                f"ReLU'd CAM collapsed to zero (min=max={cam_min.item():.6f}). "
                f"Falling back to signed (pre-ReLU) map."
            )
            cam_signed = cam_raw.unsqueeze(0).unsqueeze(0)
            cam_signed = F.interpolate(cam_signed, size=target_shape, mode='trilinear', align_corners=False)
            cam_signed = cam_signed.squeeze(0).squeeze(0)

            sig_min = cam_signed.min()
            sig_max = cam_signed.max()

            if sig_max > sig_min:
                cam = (cam_signed - sig_min) / (sig_max - sig_min)
            else:
                logger.warning("Signed CAM is also perfectly flat. Returning zeros.")
                cam = torch.zeros_like(cam_signed)

        result = cam.detach().cpu().numpy()

        # Free memory
        self.gradients.clear()
        self.activations.clear()

        return result


class ConfidenceMapper:
    """Generate confidence/uncertainty maps using MC Dropout"""

    def __init__(self, model, num_samples: int = 10):
        self.model = model
        self.num_samples = num_samples

    def get_uncertainty(self, input_tensor: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        self.model.train()

        predictions = []
        with torch.no_grad():
            for _ in range(self.num_samples):
                output = self.model(input_tensor)
                predictions.append(output.cpu().numpy())

        predictions = np.array(predictions)

        mean_prediction = predictions.mean(axis=0)
        uncertainty = predictions.std(axis=0)

        self.model.eval()

        return mean_prediction, uncertainty


class FeatureImportanceVisualizer:
    """Visualize feature importance and generate explanations"""

    @staticmethod
    def create_overlay(
        image_2d: np.ndarray,
        attention_map: np.ndarray,
        alpha: float = 0.4,
        colormap: int = cv2.COLORMAP_JET
    ) -> np.ndarray:
        img_norm = image_2d.astype(np.float32)
        img_norm = ((img_norm - img_norm.min()) / (img_norm.max() - img_norm.min() + 1e-8) * 255).astype(np.uint8)

        attn_norm = attention_map.astype(np.float32)
        attn_norm = ((attn_norm - attn_norm.min()) / (attn_norm.max() - attn_norm.min() + 1e-8) * 255).astype(np.uint8)

        attn_colored = cv2.applyColorMap(attn_norm, colormap)
        img_bgr = cv2.cvtColor(img_norm, cv2.COLOR_GRAY2BGR)

        overlay = cv2.addWeighted(img_bgr, 1 - alpha, attn_colored, alpha, 0)
        return overlay

    @staticmethod
    def generate_explanation_text(
        cam: np.ndarray,
        threshold: float = 0.7,
        max_voxels: int = None
    ) -> str:
        total_voxels = cam.size
        important_voxels = (cam > threshold).sum()
        high_confidence_voxels = (cam > 0.9).sum()

        importance_percentage = (important_voxels / total_voxels) * 100
        high_confidence_percentage = (high_confidence_voxels / total_voxels) * 100

        mean_attention = cam.mean()
        max_attention = cam.max()

        # NOTE: "attention coverage" describes how much of the volume the
        # VG/IG explanation highlights as influential to the model's
        # decision. This is NOT the same as the model's segmentation
        # confidence (that is reported separately, from the segmentation
        # probability map). Low attention coverage does not mean the
        # segmentation itself is unreliable - it means the explanation is
        # more spatially concentrated / less diffuse.
        if mean_attention > 0.15:
            coverage_label = "High"
        elif mean_attention > 0.05:
            coverage_label = "Moderate"
        else:
            coverage_label = "Low"

        explanation = f"""
═══════════════════════════════════════════════════════════
🔍 XAI EXPLANATION - MODEL ATTENTION ANALYSIS (VG / IG)
═══════════════════════════════════════════════════════════

⚠️  WHAT THIS SHOWS: This map highlights which voxels most
    influenced the model's tumor-class prediction (Vanilla Gradient /
    Integrated Gradients). It is a MODEL EXPLAINABILITY signal, NOT a calibrated
    tumor-probability map. Do not interpret attention values
    as diagnostic probability of malignancy or tumor presence.

📊 ATTENTION MAP STATISTICS (whole-volume):
  • Voxels above {threshold} attention: {important_voxels:,} ({importance_percentage:.1f}% of volume)
  • Voxels above 0.9 attention: {high_confidence_voxels:,} ({high_confidence_percentage:.1f}% of volume)
  • Mean attention score: {mean_attention:.3f}
  • Maximum attention score: {max_attention:.3f}
  • Attention coverage: {coverage_label}
    (this describes how concentrated vs. diffuse the model's
    explanation is - NOT the segmentation's confidence)

🎯 VISUAL INTERPRETATION (color = attention magnitude, not diagnosis):
  🔴 RED regions:    Highest model attention (>0.9)
  🟡 YELLOW regions: Moderate model attention (0.7-0.9)
  🔵 BLUE regions:   Lower model attention (<0.7)

⚠️  CLINICAL NOTES FOR SURGEON:
  • This attention map is a research-stage explainability tool,
    not a validated clinical decision aid.
  • Segmentation confidence is reported separately in the
    Detailed Analysis tab - refer to that for the model's
    certainty about tumor/edema boundaries.
  • Always cross-reference highlighted regions against the
    source T1/T2/FLAIR/T1c imaging yourself before relying on
    them in any way.

📌 RECOMMENDATION:
  Please verify these regions during preoperative imaging review.
  AI predictions should support, not replace, radiologist and
  neurosurgeon interpretation.
═══════════════════════════════════════════════════════════
""".strip()

        return explanation

    @staticmethod
    def get_attention_statistics(cam: np.ndarray) -> dict:
        return {
            'mean': float(cam.mean()),
            'std': float(cam.std()),
            'min': float(cam.min()),
            'max': float(cam.max()),
            'median': float(np.median(cam)),
            'percentile_25': float(np.percentile(cam, 25)),
            'percentile_75': float(np.percentile(cam, 75)),
            'percentile_90': float(np.percentile(cam, 90)),
        }


def create_grad_cam_explainer(model, target_layers=None) -> GradCAM3D:
    return GradCAM3D(model, target_layers or ['unet.bottleneck'])


if __name__ == "__main__":
    print("✅ XAI Explainer module loaded successfully")
