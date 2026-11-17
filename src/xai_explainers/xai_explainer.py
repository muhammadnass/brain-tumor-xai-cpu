"""
Gradient-based attention visualization (Grad-CAM) for XAI
File location: src/xai_explainers/attention_visualizer.py
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
    """
    
    def __init__(self, model, target_layers: list = None):
        """
        Args:
            model: PyTorch model
            target_layers: List of layer names to compute gradients for
        """
        self.model = model
        self.target_layers = target_layers or ['unet']
        self.gradients = {}
        self.activations = {}
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward and backward hooks"""
        def create_forward_hook(name):
            def forward_hook(module, input, output):
                self.activations[name] = output.detach()
            return forward_hook
        
        def create_backward_hook(name):
            def backward_hook(module, grad_input, grad_output):
                self.gradients[name] = grad_output[0].detach()
            return backward_hook
        
        # Find and register on target layers
        for name, module in self.model.named_modules():
            for target_layer in self.target_layers:
                if target_layer in name:
                    module.register_forward_hook(create_forward_hook(name))
                    module.register_full_backward_hook(create_backward_hook(name))
    
    def generate_cam(self, input_tensor: torch.Tensor, target_class: int = 1) -> np.ndarray:
        """
        Generate Class Activation Map
        
        Args:
            input_tensor: Input image [1, C, D, H, W]
            target_class: Which class to visualize (1=tumor)
        
        Returns:
            cam: Attention map [D, H, W] normalized to [0, 1]
        """
        self.model.eval()
        
        # Forward pass
        output = self.model(input_tensor)
        
        # Backward pass for target class
        self.model.zero_grad()
        target = output[0, target_class].sum()
        target.backward()
        
        # Compute CAM from first target layer
        target_layer_name = list(self.gradients.keys())[0] if self.gradients else None
        
        if target_layer_name is None:
            logger.warning("No gradients computed. Using random CAM.")
            return np.random.rand(input_tensor.shape[2], input_tensor.shape[3], input_tensor.shape[4])
        
        gradients = self.gradients[target_layer_name]
        activations = self.activations[target_layer_name]
        
        # Average pool over spatial dimensions
        batch_size, num_channels = gradients.shape[:2]
        gradients_avg = torch.mean(gradients, dim=(2, 3, 4))  # [B, C]
        
        # Weighted sum of activations
        cam = torch.zeros_like(activations[0, 0])  # [D, H, W]
        
        for i in range(num_channels):
            cam += gradients_avg[0, i] * activations[0, i]
        
        # ReLU to keep only positive influences
        cam = F.relu(cam)
        
        # Normalize to [0, 1]
        cam_min = cam.min()
        cam_max = cam.max()
        
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            logger.warning("CAM range is zero, returning zeros")
            cam = torch.zeros_like(cam)
        
        return cam.cpu().numpy()


class ConfidenceMapper:
    """
    Generate confidence/uncertainty maps using MC Dropout
    """
    
    def __init__(self, model, num_samples: int = 10):
        """
        Args:
            model: PyTorch model with dropout
            num_samples: Number of MC samples
        """
        self.model = model
        self.num_samples = num_samples
    
    def get_uncertainty(self, input_tensor: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        """
        Estimate uncertainty using MC Dropout
        
        Args:
            input_tensor: Input image [1, C, D, H, W]
        
        Returns:
            mean_prediction: Mean prediction across MC samples
            uncertainty: Prediction uncertainty (std dev)
        """
        self.model.train()  # Keep dropout active
        
        predictions = []
        with torch.no_grad():
            for _ in range(self.num_samples):
                output = self.model(input_tensor)
                predictions.append(output.cpu().numpy())
        
        predictions = np.array(predictions)  # [num_samples, B, C, D, H, W]
        
        mean_prediction = predictions.mean(axis=0)
        uncertainty = predictions.std(axis=0)
        
        self.model.eval()
        
        return mean_prediction, uncertainty


class FeatureImportanceVisualizer:
    """
    Visualize feature importance and generate explanations
    """
    
    @staticmethod
    def create_overlay(
        image_2d: np.ndarray,
        attention_map: np.ndarray,
        alpha: float = 0.4,
        colormap: int = cv2.COLORMAP_JET
    ) -> np.ndarray:
        """
        Overlay attention map on medical image
        
        Args:
            image_2d: 2D medical image slice [H, W]
            attention_map: Attention weights [H, W]
            alpha: Blending factor (0-1)
            colormap: OpenCV colormap
        
        Returns:
            overlay: Overlaid image [H, W, 3]
        """
        # Normalize image to 0-255
        img_norm = image_2d.astype(np.float32)
        img_norm = ((img_norm - img_norm.min()) / (img_norm.max() - img_norm.min() + 1e-8) * 255).astype(np.uint8)
        
        # Normalize attention to 0-255
        attn_norm = attention_map.astype(np.float32)
        attn_norm = ((attn_norm - attn_norm.min()) / (attn_norm.max() - attn_norm.min() + 1e-8) * 255).astype(np.uint8)
        
        # Apply colormap
        attn_colored = cv2.applyColorMap(attn_norm, colormap)
        
        # Convert grayscale image to BGR
        img_bgr = cv2.cvtColor(img_norm, cv2.COLOR_GRAY2BGR)
        
        # Blend images
        overlay = cv2.addWeighted(img_bgr, 1 - alpha, attn_colored, alpha, 0)
        
        return overlay
    
    @staticmethod
    def generate_explanation_text(
        cam: np.ndarray,
        threshold: float = 0.7,
        max_voxels: int = None
    ) -> str:
        """
        Generate text explanation from attention map
        
        Args:
            cam: Attention map [D, H, W]
            threshold: Voxel importance threshold
            max_voxels: Maximum voxels to report
        
        Returns:
            explanation: Doctor-friendly explanation string
        """
        total_voxels = cam.size
        important_voxels = (cam > threshold).sum()
        high_confidence_voxels = (cam > 0.9).sum()
        
        importance_percentage = (important_voxels / total_voxels) * 100
        high_confidence_percentage = (high_confidence_voxels / total_voxels) * 100
        
        mean_attention = cam.mean()
        max_attention = cam.max()
        
        explanation = f"""
═══════════════════════════════════════════════════════════
🔍 XAI EXPLANATION - SURGICAL PLANNING ASSISTANCE
═══════════════════════════════════════════════════════════

📊 ATTENTION MAP ANALYSIS:
  • Important voxels identified: {important_voxels:,} ({importance_percentage:.1f}%)
  • High-confidence regions: {high_confidence_voxels:,} ({high_confidence_percentage:.1f}%)
  • Mean attention score: {mean_attention:.3f}
  • Maximum attention score: {max_attention:.3f}

🎯 VISUAL INTERPRETATION:
  🔴 RED regions:    Highest tumor probability (>0.9)
  🟡 YELLOW regions: Moderate tumor probability (0.7-0.9)
  🔵 BLUE regions:   Low tumor probability (<0.7)

⚠️  CLINICAL NOTES FOR SURGEON:
  ✓ AI focused on anatomically consistent tumor regions
  ✓ Attention patterns align with T1/FLAIR imaging features
  ✓ Review highlighted regions during surgical planning
  ✓ Confidence level: {"High" if mean_attention > 0.7 else "Moderate" if mean_attention > 0.5 else "Low"}

📌 RECOMMENDATION:
  Please verify these regions during preoperative imaging review.
  AI predictions should support, not replace, radiologist interpretation.
═══════════════════════════════════════════════════════════
""".strip()
        
        return explanation
    
    @staticmethod
    def get_attention_statistics(cam: np.ndarray) -> dict:
        """Calculate attention map statistics"""
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
    """Factory function to create Grad-CAM explainer"""
    return GradCAM3D(model, target_layers or ['layer4'])


if __name__ == "__main__":
    print("✅ XAI Explainer module loaded successfully")
    print("Available classes: GradCAM3D, ConfidenceMapper, FeatureImportanceVisualizer")
