"""
Additional lightweight, gradient-based XAI methods to compare against
Grad-CAM, without the prohibitive computational cost of perturbation-based
methods (SHAP, LIME) at full 3D volumetric resolution.

Both methods below require only a small, fixed number of backward passes
(1 for Vanilla Gradient, ~20-50 for Integrated Gradients) - tractable on
CPU, unlike KernelSHAP/LIME which require hundreds-thousands of perturbed
forward passes per sample at this input size.

File location: src/xai_explainers/gradient_methods.py
"""
import torch
import numpy as np
import logging

logger = logging.getLogger(__name__)


def _forward_for_grad(model, x):
    """
    Run a forward pass suitable for gradient computation.

    If the model exposes `forward_logits()` we use that instead of the
    standard `forward()`.  This matters because `forward()` applies softmax,
    whose derivative saturates for high-confidence predictions:

        ∂softmax(z_k)/∂z_i  =  softmax(z_k) · (δ_ki − softmax(z_i))
        ≈  0.99 × 0.01  =  0.0099   when P(tumor)=0.99

    Five levels of pooling in the U-Net multiply that small derivative
    through the chain rule, driving input-level gradients to ≈0.  Using
    raw logits avoids this:  argmax(logits)==argmax(softmax(logits)), so
    pred-masks are identical, but logit gradients are large and informative.
    """
    if hasattr(model, 'forward_logits'):
        logger.debug("VG/IG: using forward_logits() to avoid softmax saturation")
        return model.forward_logits(x)
    # Fallback for any model that doesn't expose forward_logits
    logger.debug("VG/IG: model has no forward_logits(); falling back to forward()")
    return model(x)


class VanillaGradient:
    """
    Simplest possible gradient-based explanation: the raw gradient of the
    target class output with respect to the input, taken directly
    (Simonyan et al., 2014 - the original saliency map method).

    Cost: exactly 1 forward + 1 backward pass. Essentially free compared
    to Grad-CAM (which also needs only 1 pass) or perturbation methods.
    """

    def __init__(self, model):
        self.model = model

    def generate(self, input_tensor: torch.Tensor, target_class: int = 1) -> np.ndarray:
        self.model.eval()
        # detach().clone() then re-enable grad — avoids any leftover graph state
        x = input_tensor.detach().clone().requires_grad_(True)

        # Use logits when available — avoids softmax saturation (see _forward_for_grad)
        output = _forward_for_grad(self.model, x)

        # Restrict the backward target to voxels the model actually predicted
        # as target_class.  Summing over the entire volume (including the vast
        # background) lets background cancellation drive the gradient near zero
        # for any case where the tumour is small relative to total volume.
        # argmax(logits) == argmax(softmax(logits)), so this mask is identical
        # regardless of whether we used forward() or forward_logits().
        with torch.no_grad():
            pred_mask = (output.argmax(dim=1) == target_class).float()

        n_pred = pred_mask.sum().item()
        if n_pred > 0:
            target = (output[0, target_class] * pred_mask[0]).sum()
        else:
            # No voxels predicted as this class — fall back to whole-volume sum
            logger.warning(
                f"VG: no voxels predicted as class {target_class}; "
                f"falling back to whole-volume sum."
            )
            target = output[0, target_class].sum()

        # Use autograd.grad() instead of .backward()/.grad: it raises immediately
        # when the gradient path is broken (allow_unused=True returns None so we
        # can log a clear warning instead of an opaque AttributeError later).
        try:
            grads = torch.autograd.grad(
                target, x,
                create_graph=False,
                retain_graph=False,
                allow_unused=True,
            )[0]
        except RuntimeError as exc:
            logger.error(f"VG autograd.grad failed: {exc}; returning zero map.")
            return np.zeros(input_tensor.shape[2:], dtype=np.float32)

        if grads is None:
            logger.error(
                "VG: autograd.grad returned None — no differentiable path from "
                "model output back to input.  Returning zero map."
            )
            return np.zeros(input_tensor.shape[2:], dtype=np.float32)

        # Gradient magnitude, averaged across the 4 input channels,
        # gives a single [D, H, W] saliency map matching Grad-CAM's output shape
        gradient = grads[0].abs().mean(dim=0)  # [D, H, W]

        g_min, g_max = gradient.min(), gradient.max()
        logger.info(
            f"VG raw gradient: min={g_min.item():.6f} max={g_max.item():.6f} "
            f"mean={gradient.mean().item():.6f}  (pred_voxels={int(n_pred)})"
        )
        if g_max > g_min:
            gradient = (gradient - g_min) / (g_max - g_min)
        else:
            logger.warning(
                f"VG gradient is perfectly flat (min=max={g_min.item():.6f}). "
                f"Returning zero map — check model output for this case."
            )
            return np.zeros_like(gradient.detach().cpu().numpy())

        return gradient.detach().cpu().numpy()


class IntegratedGradients:
    """
    Integrated Gradients (Sundararajan et al., 2017): accumulates
    gradients along a linear path from a baseline (zero) input to the
    actual input, addressing a known limitation of vanilla gradients
    (saturation - gradients can be near-zero even in genuinely important
    regions once the network is confident).

    Cost: `steps` forward+backward passes (default 20) - still far cheaper
    than perturbation-based methods, which typically require hundreds of
    samples even for 2D inputs, and would be prohibitive at this 3D scale.
    """

    def __init__(self, model, steps: int = 20):
        self.model = model
        self.steps = steps

    def generate(self, input_tensor: torch.Tensor, target_class: int = 1) -> np.ndarray:
        self.model.eval()
        baseline   = torch.zeros_like(input_tensor)
        inp_detach = input_tensor.detach()

        total_gradients = torch.zeros_like(inp_detach)

        for step in range(1, self.steps + 1):
            alpha        = step / self.steps
            interpolated = (baseline + alpha * (inp_detach - baseline)).requires_grad_(True)

            # Use logits when available — avoids softmax saturation
            output = _forward_for_grad(self.model, interpolated)

            # Restrict target to predicted tumor voxels (same fix as VG / Grad-CAM)
            with torch.no_grad():
                pred_mask = (output.argmax(dim=1) == target_class).float()

            n_pred = pred_mask.sum().item()
            if n_pred > 0:
                target = (output[0, target_class] * pred_mask[0]).sum()
            else:
                target = output[0, target_class].sum()

            try:
                grads = torch.autograd.grad(
                    target, interpolated,
                    create_graph=False,
                    retain_graph=False,
                    allow_unused=True,
                )[0]
            except RuntimeError as exc:
                logger.error(f"IG step {step} autograd.grad failed: {exc}")
                grads = None

            if grads is not None:
                total_gradients += grads.detach()

        avg_gradients    = total_gradients / self.steps
        integrated_grads = inp_detach * avg_gradients   # baseline is zeros

        attribution = integrated_grads[0].abs().mean(dim=0)  # [D, H, W]

        a_min, a_max = attribution.min(), attribution.max()
        logger.info(
            f"IG attribution: min={a_min.item():.6f} max={a_max.item():.6f} "
            f"mean={attribution.mean().item():.6f}"
        )
        if a_max > a_min:
            attribution = (attribution - a_min) / (a_max - a_min)
        else:
            logger.warning(
                "IG attribution is perfectly flat. Returning zero map."
            )
            return np.zeros_like(attribution.cpu().numpy())

        return attribution.detach().cpu().numpy()


def compare_xai_methods(model, input_tensor: torch.Tensor, target_class: int = 1) -> dict:
    """
    Run all three XAI methods (Grad-CAM, Vanilla Gradient, Integrated
    Gradients) on the same input and return their attention maps plus
    pairwise agreement metrics, for genuine multi-method comparison.
    """
    from .attention_visualizer import GradCAM3D

    results = {}

    logger.info("Computing Grad-CAM...")
    grad_cam = GradCAM3D(model, target_layers=['unet.bottleneck'])
    results['grad_cam'] = grad_cam.generate_cam(input_tensor, target_class=target_class)

    logger.info("Computing Vanilla Gradient...")
    vanilla = VanillaGradient(model)
    results['vanilla_gradient'] = vanilla.generate(input_tensor, target_class=target_class)

    logger.info("Computing Integrated Gradients...")
    ig = IntegratedGradients(model, steps=20)
    results['integrated_gradients'] = ig.generate(input_tensor, target_class=target_class)

    # Pairwise agreement: Pearson correlation between flattened attention maps
    # (a standard XAI-comparison metric - measures whether methods agree on
    # WHERE the important regions are, independent of absolute scale)
    agreements = {}
    method_names = list(results.keys())
    for i in range(len(method_names)):
        for j in range(i + 1, len(method_names)):
            name_a, name_b = method_names[i], method_names[j]
            map_a = results[name_a].flatten()
            map_b = results[name_b].flatten()
            correlation = np.corrcoef(map_a, map_b)[0, 1]
            agreements[f'{name_a}_vs_{name_b}'] = float(correlation)

    results['pairwise_correlations'] = agreements
    return results
