"""
Side-by-side Vanilla Gradient and Integrated Gradients attribution maps for
two held-out cases, drawn from the same model used for the interface figures.

Produces one panel: rows are cases, columns are FLAIR slice, Vanilla Gradient,
Integrated Gradients.

Usage (from the project root):
    python scripts/plot_vg_ig_comparison.py --cases BRATS_295 BRATS_474
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

torch.set_num_threads(8)

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
from ai_models.segmentation import UNet3D
from xai_explainers.gradient_methods import VanillaGradient, IntegratedGradients

CACHE = ROOT / "data/brats_dataset/cache_128"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="models/segmentation/unet3d_cv_fold1.pth")
    ap.add_argument("--cases", nargs="+", required=True)
    ap.add_argument("--ig-steps", type=int, default=20)
    ap.add_argument("--target-class", type=int, default=1)
    ap.add_argument("--out", default="figures/figure_vg_ig_comparison.png")
    args = ap.parse_args()

    ckpt = ROOT / args.checkpoint
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    model = UNet3D(features=tuple(state["features"]))
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    n = sum(p.numel() for p in model.parameters())
    print(f"loaded {ckpt.name} | {n:,} params | fold={state.get('fold')} "
          f"epoch={state.get('epoch')} val_dice={state.get('val_dice'):.4f}")

    vg = VanillaGradient(model)
    ig = IntegratedGradients(model, steps=args.ig_steps)

    rows = len(args.cases)
    fig, axes = plt.subplots(rows, 3, figsize=(11, 3.7 * rows))
    if rows == 1:
        axes = axes.reshape(1, 3)

    for r, cid in enumerate(args.cases):
        img_path = CACHE / "images" / f"{cid}.npy"
        lbl_path = CACHE / "labels" / f"{cid}.npy"
        if not img_path.exists():
            sys.exit(f"ABORT: {img_path} not found")

        arr = np.load(img_path).astype(np.float32)
        x = torch.from_numpy(arr).unsqueeze(0)

        # centre the slice on the tumour where a label is available
        if lbl_path.exists():
            lbl = np.load(lbl_path)
            fg = np.argwhere((lbl == 2) | (lbl == 3))
            z = int(fg[:, 0].mean()) if len(fg) else arr.shape[1] // 2
        else:
            z = arr.shape[1] // 2

        print(f"  {cid}: slice {z} ... vanilla gradient", flush=True)
        m_vg = vg.generate(x, args.target_class)
        print(f"  {cid}: integrated gradients ({args.ig_steps} steps)", flush=True)
        m_ig = ig.generate(x, args.target_class)

        bg = arr[0, z]          # FLAIR is channel 0
        axes[r, 0].imshow(bg, cmap="gray")
        axes[r, 0].set_title(f"{cid}\nFLAIR, slice {z}", fontsize=10)

        for c, (m, name) in enumerate([(m_vg, "Vanilla Gradient"),
                                       (m_ig, "Integrated Gradients")], start=1):
            axes[r, c].imshow(bg, cmap="gray")
            axes[r, c].imshow(m[z], cmap="inferno", alpha=0.55)
            axes[r, c].set_title(name, fontsize=10)

        for c in range(3):
            axes[r, c].axis("off")

    plt.tight_layout()
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"\nwrote {out}")
    print(f"checkpoint: {args.checkpoint} | IG steps: {args.ig_steps} | "
          f"target class: {args.target_class}")


if __name__ == "__main__":
    main()
