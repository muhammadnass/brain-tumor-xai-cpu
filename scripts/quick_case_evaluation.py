"""
Quick diagnostic: compare model prediction against REAL ground truth
for one specific case, to sanity-check whether "high confidence" actually
means "correct" or just "confidently wrong".

File location: scripts/quick_case_evaluation.py
Usage: python scripts/quick_case_evaluation.py --case-dir data/sample_cases/real_case_001_XXX
"""
import sys
import argparse
from pathlib import Path
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ai_models.segmentation import create_segmentation_model
from data_processing.preprocessor import MRIPreprocessor
from config import MODELS_DIR
import nibabel as nib
from scipy.ndimage import zoom


def remap_labels(label):
    """Same remapping used in training: 4-class BraTS -> our 3-class scheme"""
    remapped = np.zeros_like(label)
    remapped[label == 1] = 2   # edema -> class 2
    remapped[label == 2] = 1   # non-enhancing tumor -> class 1 (tumor)
    remapped[label == 3] = 1   # enhancing tumor -> class 1 (tumor)
    return remapped


def dice_score(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """Compute Dice coefficient between two binary masks"""
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    total = pred_mask.sum() + gt_mask.sum()
    if total == 0:
        return 1.0  # both empty = perfect agreement
    return (2.0 * intersection) / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-dir", type=str, required=True,
                        help="Path to case folder containing T1/T2/FLAIR/T1c/ground_truth_label.nii.gz")
    args = parser.parse_args()

    case_dir = Path(args.case_dir)

    print("=" * 60)
    print("🔬 QUICK CASE EVALUATION: PREDICTION vs GROUND TRUTH")
    print("=" * 60)
    print(f"Case: {case_dir.name}")
    print()

    # ---- Load ground truth ----
    gt_path = case_dir / "ground_truth_label.nii.gz"
    if not gt_path.exists():
        print(f"❌ Ground truth not found at {gt_path}")
        sys.exit(1)

    gt_img = nib.load(str(gt_path))
    gt_data = gt_img.get_fdata()
    print(f"Ground truth raw shape: {gt_data.shape}")

    gt_remapped = remap_labels(gt_data)
    gt_tumor_voxels_original = int((gt_remapped == 1).sum())
    total_voxels_original = gt_remapped.size

    print(f"📊 GROUND TRUTH (original resolution):")
    print(f"   Total brain volume voxels: {total_voxels_original:,}")
    print(f"   Actual tumor voxels: {gt_tumor_voxels_original:,}")
    print(f"   Actual tumor as % of volume: {100*gt_tumor_voxels_original/total_voxels_original:.2f}%")
    print()

    # Resize ground truth to 128^3 to match model's working resolution
    zoom_factors = [128/s for s in gt_remapped.shape]
    gt_resized = zoom(gt_remapped, zoom_factors, order=0)  # nearest-neighbor, preserves labels
    gt_tumor_mask_128 = (gt_resized == 1)

    print(f"   Resized to 128³: tumor voxels = {gt_tumor_mask_128.sum():,} "
          f"({100*gt_tumor_mask_128.sum()/gt_tumor_mask_128.size:.2f}%)")
    print()

    # ---- Load model + run prediction ----
    print("Loading trained model and running prediction...")
    checkpoint_path = MODELS_DIR / "segmentation" / "unet_3d.pth"
    model = create_segmentation_model(device='cpu', checkpoint_path=str(checkpoint_path))
    model.eval()

    file_paths = {
        'T1': str(case_dir / "T1.nii.gz"),
        'T2': str(case_dir / "T2.nii.gz"),
        'FLAIR': str(case_dir / "FLAIR.nii.gz"),
        'T1c': str(case_dir / "T1c.nii.gz"),
    }

    preprocessor = MRIPreprocessor(target_shape=(128, 128, 128))
    modality_volumes = []
    for modality in ['T1', 'T2', 'FLAIR', 'T1c']:
        data = preprocessor.load_nifti(file_paths[modality])
        processed = preprocessor.preprocess(data)
        modality_volumes.append(processed)

    mri_tensor = torch.from_numpy(np.stack(modality_volumes, axis=0)).float().unsqueeze(0)

    with torch.no_grad():
        output = model.predict(mri_tensor)

    pred_tumor_prob = output[0, 1].numpy()
    pred_tumor_mask = pred_tumor_prob > 0.5

    print()
    print(f"📊 MODEL PREDICTION (128³ resolution):")
    print(f"   Predicted tumor voxels: {pred_tumor_mask.sum():,} "
          f"({100*pred_tumor_mask.sum()/pred_tumor_mask.size:.2f}%)")
    print(f"   Mean predicted probability: {pred_tumor_prob.mean():.4f}")
    print(f"   Max predicted probability: {pred_tumor_prob.max():.4f}")
    print()

    # ---- Direct comparison ----
    case_dice = dice_score(pred_tumor_mask, gt_tumor_mask_128)

    intersection = np.logical_and(pred_tumor_mask, gt_tumor_mask_128).sum()
    false_positive = np.logical_and(pred_tumor_mask, ~gt_tumor_mask_128).sum()
    false_negative = np.logical_and(~pred_tumor_mask, gt_tumor_mask_128).sum()

    print("=" * 60)
    print("🎯 DIRECT COMPARISON (threshold = 0.5)")
    print("=" * 60)
    print(f"   Dice score (this case): {case_dice:.4f}")
    print(f"   True positive voxels (correctly identified tumor): {intersection:,}")
    print(f"   False positive voxels (predicted tumor, actually healthy): {false_positive:,}")
    print(f"   False negative voxels (missed actual tumor): {false_negative:,}")
    print()

    recall = intersection / max(gt_tumor_mask_128.sum(), 1)
    precision = intersection / max(pred_tumor_mask.sum(), 1)
    print(f"   Recall (sensitivity): {recall:.4f}")
    print(f"   Precision: {precision:.4f}")
    print()

    if precision < 0.3 and recall > 0.8:
        print("   ⚠️  DIAGNOSIS: High recall, very low precision.")
        print("      The model rarely MISSES real tumor, but massively OVER-FLAGS")
        print("      healthy tissue as tumor. Likely cause: training/validation used")
        print("      tumor-biased patch sampling (pos:neg=2:1), which does not match")
        print("      the true ~99.3% background / 0.7% tumor ratio in full volumes.")
    elif false_negative > intersection * 2:
        print("   ⚠️  DIAGNOSIS: Model is substantially UNDER-PREDICTING tumor regions.")
    else:
        print("   ✅ Reasonable balance between over- and under-prediction.")

    print()
    print("=" * 60)
    print("🔍 THRESHOLD SWEEP (no retraining needed - testing if a stricter")
    print("   decision threshold alone fixes the false-positive problem)")
    print("=" * 60)
    print(f"{'Threshold':>10} | {'Dice':>8} | {'Precision':>10} | {'Recall':>8} | {'Pred voxels':>12}")
    print("-" * 60)

    best_threshold = 0.5
    best_dice = case_dice

    for threshold in [0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.97, 0.99, 0.995, 0.999]:
        thresh_mask = pred_tumor_prob > threshold
        thresh_dice = dice_score(thresh_mask, gt_tumor_mask_128)
        thresh_intersection = np.logical_and(thresh_mask, gt_tumor_mask_128).sum()
        thresh_precision = thresh_intersection / max(thresh_mask.sum(), 1)
        thresh_recall = thresh_intersection / max(gt_tumor_mask_128.sum(), 1)

        marker = ""
        if thresh_dice > best_dice:
            best_dice = thresh_dice
            best_threshold = threshold
            marker = "  <- best so far"

        print(f"{threshold:>10.3f} | {thresh_dice:>8.4f} | {thresh_precision:>10.4f} | "
              f"{thresh_recall:>8.4f} | {thresh_mask.sum():>12,}{marker}")

    print("-" * 60)
    print(f"✅ Best threshold found: {best_threshold} (Dice: {best_dice:.4f}, "
          f"vs {case_dice:.4f} at default 0.5)")
    print("=" * 60)


if __name__ == "__main__":
    main()
