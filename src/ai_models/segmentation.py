"""
3D U-Net for brain tumor segmentation
File location: src/ai_models/segmentation.py
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class ConvBlock3D(nn.Module):
    """3D Convolution block with BatchNorm and ReLU"""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size, padding=padding),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=False),
            nn.Conv3d(out_channels, out_channels, kernel_size, padding=padding),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=False)
        )
    
    def forward(self, x):
        return self.conv(x)


class UNet3D(nn.Module):
    """
    3D U-Net architecture for medical image segmentation
    Reference: Çiçek et al., "3D U-Net: Learning Dense Volumetric Segmentation 
    from Sparse Annotation", MICCAI 2016
    """
    
    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 3,
        features: Tuple[int, ...] = (32, 64, 128, 256, 512)
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.features = features
        
        # Encoder
        self.encoder = nn.ModuleList()
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        
        for i in range(len(features)):
            in_ch = in_channels if i == 0 else features[i-1]
            out_ch = features[i]
            self.encoder.append(ConvBlock3D(in_ch, out_ch))
        
        # Bottleneck
        self.bottleneck = ConvBlock3D(features[-1], features[-1] * 2)
        
        # Decoder
        self.decoder = nn.ModuleList()
        self.upconv = nn.ModuleList()
        
        for i in range(len(features)-1, -1, -1):
            up_in = features[i] * 2 if i == len(features)-1 else features[i] * 2
            up_out = features[i]
            self.upconv.append(nn.ConvTranspose3d(up_in, up_out, kernel_size=2, stride=2))
            
            dec_in = features[i] * 2 if i < len(features)-1 else up_out + features[i]
            dec_out = features[i]
            self.decoder.append(ConvBlock3D(dec_in, dec_out))
        
        # Final output layer
        self.final_conv = nn.Conv3d(features[0], out_channels, kernel_size=1)
    
    def forward(self, x):
        # Encoder
        encoder_outputs = []
        for encoder in self.encoder:
            x = encoder(x)
            encoder_outputs.append(x)
            x = self.pool(x)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Decoder
        for i in range(len(self.decoder)):
            x = self.upconv[i](x)
            skip = encoder_outputs[-(i+1)]
            x = torch.cat([x, skip], dim=1)
            x = self.decoder[i](x)
        
        # Final output
        x = self.final_conv(x)
        return x


class BrainTumorSegmentationModel(nn.Module):
    """
    Complete brain tumor segmentation model with preprocessing and postprocessing
    """
    
    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 3,
        pretrained_path: Optional[str] = None,
        features=None
    ):
        super().__init__()
        if features is None:
            self.unet = UNet3D(in_channels, out_channels)
        else:
            self.unet = UNet3D(in_channels, out_channels, features=tuple(features))
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        if pretrained_path:
            self.load_pretrained(pretrained_path)
    
    def forward(self, x):
        """Forward pass — returns softmax probabilities."""
        logits = self.unet(x)
        # Apply softmax for multi-class segmentation
        output = torch.softmax(logits, dim=1)
        return output

    def forward_logits(self, x):
        """
        Forward pass returning raw pre-softmax logits.

        Use this for gradient-based XAI methods (Vanilla Gradient, Integrated
        Gradients) instead of forward().  Softmax saturates near 0 or 1, which
        drives ∂softmax/∂logit → 0 and causes vanishing gradients all the way
        back to the input of a deep U-Net.  Working in logit space avoids that:
        argmax(logits) == argmax(softmax(logits)), so predicted masks are
        unchanged, but the gradient of the logit target wrt input is large and
        spatially informative.
        """
        return self.unet(x)
    
    def predict(self, x: torch.Tensor, return_logits: bool = False):
        """
        Inference prediction
        
        Args:
            x: Input tensor [B, C, D, H, W]
            return_logits: If True, return logits; else return probabilities
        
        Returns:
            predictions: [B, out_channels, D, H, W]
        """
        self.eval()
        with torch.no_grad():
            if return_logits:
                return self.unet(x)
            else:
                return self.forward(x)
    
    def predict_segmentation_mask(self, x: torch.Tensor, threshold: float = 0.5):
        """
        Predict binary segmentation mask
        
        Args:
            x: Input tensor
            threshold: Probability threshold
        
        Returns:
            mask: Binary mask
        """
        probs = self.predict(x)
        # Get argmax for each voxel
        mask = torch.argmax(probs, dim=1)  # [B, D, H, W]
        return mask
    
    def load_pretrained(self, checkpoint_path: str):
        """
        Load pretrained weights.

        Handles two checkpoint layouts. Checkpoints saved from this wrapper
        carry keys prefixed 'unet.'; checkpoints saved from a bare UNet3D
        (as scripts/train_cv.py does) do not. The prefix is added when absent.

        Any mismatch after that is raised rather than warned about: a model
        silently left at random initialisation is worse than a crash.
        """
        checkpoint = torch.load(checkpoint_path, map_location='cpu',
                                weights_only=False)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state = checkpoint['model_state_dict']
        else:
            state = checkpoint

        if not any(k.startswith('unet.') for k in state):
            state = {f'unet.{k}': v for k, v in state.items()}
            logger.info("checkpoint has bare UNet3D keys; 'unet.' prefix added")

        missing, unexpected = self.load_state_dict(state, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                f"checkpoint does not match this model: "
                f"{len(missing)} missing, {len(unexpected)} unexpected keys. "
                f"First missing: {missing[:3]}. First unexpected: {unexpected[:3]}. "
                f"Check that the 'features' argument matches the checkpoint."
            )

        n = sum(p.numel() for p in self.parameters())
        logger.info(f"✅ Loaded {checkpoint_path} cleanly | {n:,} parameters")
        if isinstance(checkpoint, dict):
            meta = {k: checkpoint[k] for k in
                    ('fold', 'epoch', 'val_dice', 'features') if k in checkpoint}
            if meta:
                logger.info(f"   checkpoint metadata: {meta}")
    
    def save_checkpoint(self, save_path: str, metadata: dict = None):
        """Save model checkpoint"""
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'model_config': {
                'in_channels': self.in_channels,
                'out_channels': self.out_channels
            }
        }
        if metadata:
            checkpoint.update(metadata)
        
        torch.save(checkpoint, save_path)
        logger.info(f"✅ Saved checkpoint to {save_path}")
    
    def get_layer_outputs(self, x: torch.Tensor, layer_names: list):
        """
        Get intermediate layer outputs for visualization
        Useful for XAI methods like Grad-CAM
        """
        outputs = {}
        
        def create_hook(name):
            def hook(module, input, output):
                outputs[name] = output.detach()
            return hook
        
        handles = []
        for name, module in self.named_modules():
            if name in layer_names:
                handle = module.register_forward_hook(create_hook(name))
                handles.append(handle)
        
        # Forward pass
        self.eval()
        with torch.no_grad():
            _ = self.forward(x)
        
        # Remove hooks
        for handle in handles:
            handle.remove()
        
        return outputs


def create_segmentation_model(
    in_channels: int = 4,
    out_channels: int = 3,
    checkpoint_path: Optional[str] = None,
    device: str = 'cpu',
    features=None
) -> BrainTumorSegmentationModel:
    """
    Factory function to create segmentation model
    
    Args:
        in_channels: Number of input channels (4 for multi-modal MRI)
        out_channels: Number of output channels (3 for tumor classes)
        checkpoint_path: Path to pretrained weights
        device: Device to load model on ('cpu' or 'cuda')
    
    Returns:
        model: BrainTumorSegmentationModel instance
    """
    model = BrainTumorSegmentationModel(in_channels, out_channels, checkpoint_path, features=features)
    model = model.to(device)
    logger.info(f"✅ Created segmentation model on {device}")
    return model


if __name__ == "__main__":
    # Test model
    device = 'cpu'  # Change to 'cuda' if GPU available
    
    model = create_segmentation_model(device=device)
    
    # Create dummy input
    dummy_input = torch.randn(1, 4, 128, 128, 128).to(device)
    
    # Forward pass
    output = model(dummy_input)
    print(f"✅ Model output shape: {output.shape}")
    print(f"✅ Output min: {output.min():.4f}, max: {output.max():.4f}")
    print(f"✅ Output sum (should be ~1): {output.sum(dim=1).mean():.4f}")
    
    # Test logits path
    logits = model.forward_logits(dummy_input)
    print(f"✅ Logits shape: {logits.shape}")
    print(f"✅ Logits min: {logits.min():.4f}, max: {logits.max():.4f}")
