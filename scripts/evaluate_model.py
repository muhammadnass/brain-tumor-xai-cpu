"""
Comprehensive model evaluation on the HELD-OUT VALIDATION SET, at full
volume resolution (not training patches), with multiple standard metrics.

Critically: this reuses the EXACT same train/val split as train_model.py,
ensuring we evaluate only on cases the model never trained on.

File location: scripts/evaluate_model.py
Usage: python scripts/evaluate_model.py --threshold 0.5 --threshold 0.995
"""
import sys
import argparse
import logging
import json
import csv
from pathlib import Path
from datetime import datetime

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).parent))  # to import train_model.py helpers

from ai_models.segmentation import create_segmentation_model
from config import MODELS_DIR, RESULTS_DIR

# Reuse the EXACT same file list + label remapping used during training,
# so our validation split is guaranteed identical (no data leakage risk).
from train_model import build_file_list, remap_labels

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

RESULTS_METRICS_DIR = RESULTS_DIR / "metrics"


def load_and_preprocess_full_case(image_path: str, label_path: str, target_size=128):
    """
    Load a full case (all 4 modalities + label) and preprocess to full
    128^3 resolution - NO patch cropping, giving realistic full-volume
    evaluation conditions.
    """
    from monai.transforms import (
        Compose, LoadImaged, EnsureChannelFirstd, Orientationd,
        NormalizeIntensityd, Resized, Lambdad, EnsureTyped,
    )

    transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image"]),
        Lambdad(keys=["label"], func=remap_labels),
        EnsureChannelFirstd(keys=["label"], channel_dim="no_channel"),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
        Resized(keys=["image", "label"], spatial_size=(target_size,) * 3,
                mode=("trilinear", "nearest")),
        EnsureTyped(keys=["image", "label"]),
    ])

    data = transforms({"image": image_path, "label": label_path})
    return data["image"].unsqueeze(0), data["label"].unsqueeze(0)  # add batch dim


def compute_case_metrics(pred_probs: torch.Tensor, label: torch.Tensor, threshold: float):
    """
    Compute Dice, IoU, Sensitivity, Specificity, Precision, and 95th-percentile
    Hausdorff Distance for the tumor class (class index 1), using MONAI's
    standard metric implementations.
    """
    from monai.metrics import DiceMetric, compute_hausdorff_distance
    from monai.metrics.confusion_matrix import get_confusion_matrix, compute_confusion_matrix_metric

    pred_binary = (pred_probs[:, 1:2] > threshold).float()  # tumor class only, keep channel dim
    label_binary = (label == 1).float()

    metrics = {}

    # Dice
    dice_metric = DiceMetric(include_background=True, reduction="mean")
    dice_metric(y_pred=pred_binary, y=label_binary)
    metrics['dice'] = float(dice_metric.aggregate().item())
    dice_metric.reset()

    # Confusion-matrix-derived metrics (sensitivity, specificity, precision)
    cm = get_confusion_matrix(y_pred=pred_binary, y=label_binary)
    metrics['sensitivity'] = float(compute_confusion_matrix_metric("sensitivity", cm).item())
    metrics['specificity'] = float(compute_confusion_matrix_metric("specificity", cm).item())
    metrics['precision'] = float(compute_confusion_matrix_metric("precision", cm).item())

    # IoU / Jaccard = Dice / (2 - Dice)
    d = metrics['dice']
    metrics['iou'] = d / (2 - d) if d < 2 else 0.0

    # 95th percentile Hausdorff Distance (robust to outlier voxels)
    pred_sum = pred_binary.sum().item()
    label_sum = label_binary.sum().item()
    if pred_sum > 0 and label_sum > 0:
        try:
            hd95 = compute_hausdorff_distance(
                pred_binary, label_binary, percentile=95, include_background=True
            )
            metrics['hausdorff95'] = float(hd95.item())
        except Exception:
            metrics['hausdorff95'] = float('nan')
    else:
        metrics['hausdorff95'] = float('nan')  # undefined when one mask is empty

    metrics['pred_volume_voxels'] = int(pred_sum)
    metrics['true_volume_voxels'] = int(label_sum)
    metrics['volume_error_pct'] = (
        100 * abs(pred_sum - label_sum) / label_sum if label_sum > 0 else float('nan')
    )

    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, action="append",
                        help="Decision threshold(s) to evaluate. Can repeat flag for multiple. "
                             "Default: both 0.5 and 0.995")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to model checkpoint (default: models/segmentation/unet_3d.pth)")
    args = parser.parse_args()

    thresholds = args.threshold if args.threshold else [0.5, 0.995]
    checkpoint_path = args.checkpoint or str(MODELS_DIR / "segmentation" / "unet_3d.pth")

    print("=" * 70)
    print("📊 FULL-VOLUME MODEL EVALUATION ON HELD-OUT VALIDATION SET")
    print("=" * 70)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Thresholds tested: {thresholds}")
    print()

    # Rebuild the EXACT same split used during training
    file_list = build_file_list()
    split_idx = max(1, int(len(file_list) * 0.8))
    val_files = file_list[split_idx:] if len(file_list) > split_idx else file_list[-1:]

    print(f"✅ Held-out validation cases (never seen during training): {len(val_files)}")
    for f in val_files:
        print(f"   - {Path(f['image']).name}")
    print()

    model = create_segmentation_model(device='cpu', checkpoint_path=checkpoint_path)
    model.eval()

    all_results = {t: [] for t in thresholds}

    for i, file_pair in enumerate(val_files):
        case_name = Path(file_pair['image']).stem.replace('.nii', '')
        print(f"[{i+1}/{len(val_files)}] Evaluating {case_name}...")

        image, label = load_and_preprocess_full_case(file_pair['image'], file_pair['label'])

        with torch.no_grad():
            pred_probs = model.predict(image)

        for threshold in thresholds:
            case_metrics = compute_case_metrics(pred_probs, label, threshold)
            case_metrics['case'] = case_name
            case_metrics['threshold'] = threshold
            all_results[threshold].append(case_metrics)

            print(f"   [threshold={threshold}] Dice={case_metrics['dice']:.4f} "
                  f"IoU={case_metrics['iou']:.4f} Sens={case_metrics['sensitivity']:.4f} "
                  f"Spec={case_metrics['specificity']:.4f} Prec={case_metrics['precision']:.4f}")

    # ---- Aggregate summary per threshold ----
    print()
    print("=" * 70)
    print("📈 SUMMARY (mean ± std across held-out validation cases)")
    print("=" * 70)

    summary_rows = []
    for threshold in thresholds:
        results = all_results[threshold]
        summary = {'threshold': threshold, 'n_cases': len(results)}

        print(f"\n--- Threshold = {threshold} ---")
        for metric_name in ['dice', 'iou', 'sensitivity', 'specificity', 'precision', 'hausdorff95']:
            values = [r[metric_name] for r in results if not np.isnan(r[metric_name])]
            if values:
                mean_v = np.mean(values)
                std_v = np.std(values)
                summary[f'{metric_name}_mean'] = mean_v
                summary[f'{metric_name}_std'] = std_v
                print(f"   {metric_name:15s}: {mean_v:.4f} ± {std_v:.4f}")
            else:
                summary[f'{metric_name}_mean'] = float('nan')
                summary[f'{metric_name}_std'] = float('nan')
                print(f"   {metric_name:15s}: N/A (undefined for all cases)")

        summary_rows.append(summary)

    # ---- Save results ----
    RESULTS_METRICS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Full per-case CSV
    csv_path = RESULTS_METRICS_DIR / f"evaluation_percase_{timestamp}.csv"
    all_rows = [r for t in thresholds for r in all_results[t]]
    if all_rows:
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=all_rows[0].keys())
            writer.writeheader()
            writer.writerows(all_rows)
    print(f"\n💾 Per-case results saved to: {csv_path}")

    # Summary JSON
    summary_path = RESULTS_METRICS_DIR / f"evaluation_summary_{timestamp}.json"
    with open(summary_path, 'w') as f:
        json.dump(summary_rows, f, indent=2)
    print(f"💾 Summary saved to: {summary_path}")

    # ---- Paste-ready markdown table for manuscript ----
    print()
    print("=" * 70)
    print("📋 PASTE-READY MARKDOWN TABLE FOR YOUR MANUSCRIPT (Results section)")
    print("=" * 70)
    print()
    print(f"*Table: Full-volume segmentation performance on {len(val_files)} held-out "
          f"validation cases, evaluated at multiple decision thresholds.*")
    print()
    header = "| Threshold | Dice | IoU | Sensitivity | Specificity | Precision | HD95 (voxels) |"
    sep = "|---|---|---|---|---|---|---|"
    print(header)
    print(sep)
    for summary in summary_rows:
        row = (
            f"| {summary['threshold']} "
            f"| {summary['dice_mean']:.3f} ± {summary['dice_std']:.3f} "
            f"| {summary['iou_mean']:.3f} ± {summary['iou_std']:.3f} "
            f"| {summary['sensitivity_mean']:.3f} ± {summary['sensitivity_std']:.3f} "
            f"| {summary['specificity_mean']:.3f} ± {summary['specificity_std']:.3f} "
            f"| {summary['precision_mean']:.3f} ± {summary['precision_std']:.3f} "
            f"| {summary['hausdorff95_mean']:.2f} ± {summary['hausdorff95_std']:.2f} |"
        )
        print(row)
    print()
    print("=" * 70)


if __name__ == "__main__":
    main()
