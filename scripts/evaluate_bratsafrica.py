"""
Cross-population generalizability evaluation on BraTS-Africa.

Uses the ALREADY-FIXED deployment preprocessing pipeline (preprocessor.py)
directly, since BraTS-Africa's native per-modality file format matches
real-world deployment input - this evaluation exercises the same code
path a real user's upload would, including the normalization and
channel-order fixes discovered earlier.

IMPORTANT label mapping (empirically verified, NOT assumed - see chat
history): BraTS-Africa uses label 1=Necrotic Core, 2=Edema, 3=Enhancing
Tumor - DIFFERENT from the Decathlon dataset's label convention
(1=Edema, 2=Non-enhancing, 3=Enhancing) used elsewhere in this project.
Do NOT reuse train_model.py's remap_labels() for this dataset.

File location: scripts/evaluate_bratsafrica.py
Usage: python scripts/evaluate_bratsafrica.py --num-cases 30
"""
import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime
import json
import csv

import numpy as np
import torch
import nibabel as nib

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ai_models.segmentation import create_segmentation_model
from data_processing.preprocessor import MRIPreprocessor
from config import MODELS_DIR, RESULTS_DIR

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

DATASET_ROOT = Path.home() / "Downloads" / "BraTS-Africa"
RESULTS_METRICS_DIR = RESULTS_DIR / "metrics"

MODALITY_SUFFIX_MAP = {
    'FLAIR': 't2f',
    'T1': 't1n',
    'T1c': 't1c',
    'T2': 't2w',
}
CHANNEL_ORDER = ['FLAIR', 'T1', 'T1c', 'T2']


def remap_bratsafrica_labels(label: np.ndarray) -> np.ndarray:
    """
    BraTS-Africa native label convention (empirically verified):
    1 = Necrotic Core, 2 = Edema, 3 = Enhancing Tumor.
    Map to our scheme: 0=background, 1=tumor (NCR+ET), 2=edema.
    """
    remapped = np.zeros_like(label)
    remapped[label == 2] = 2
    remapped[label == 1] = 1
    remapped[label == 3] = 1
    return remapped


def find_cases(max_cases: int = None) -> list:
    cases = []
    for category_dir in sorted(DATASET_ROOT.iterdir()):
        if not category_dir.is_dir():
            continue
        for case_dir in sorted(category_dir.iterdir()):
            if not case_dir.is_dir():
                continue
            case_id = case_dir.name
            files = {}
            missing = False
            for mod_key, suffix in MODALITY_SUFFIX_MAP.items():
                f = case_dir / f"{case_id}-{suffix}.nii.gz"
                if not f.exists():
                    missing = True
                    break
                files[mod_key] = f
            seg_file = case_dir / f"{case_id}-seg.nii.gz"
            if not seg_file.exists():
                missing = True

            if not missing:
                cases.append({
                    'case_id': case_id,
                    'category': category_dir.name,
                    'files': files,
                    'seg': seg_file,
                })

    if max_cases:
        cases = cases[:max_cases]

    return cases


def dice_score(pred_mask, gt_mask):
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    total = pred_mask.sum() + gt_mask.sum()
    return 1.0 if total == 0 else (2.0 * intersection) / total


def evaluate_case(case: dict, model, preprocessor: MRIPreprocessor, target_size=128):
    modality_volumes = []
    for mod_key in CHANNEL_ORDER:
        raw = preprocessor.load_nifti(str(case['files'][mod_key]))
        processed = preprocessor.preprocess(raw)
        modality_volumes.append(processed)

    mri_tensor = torch.from_numpy(np.stack(modality_volumes, axis=0)).float().unsqueeze(0)

    with torch.no_grad():
        output = model.predict(mri_tensor)

    pred_probs = output[0, 1].numpy()
    pred_mask = pred_probs > 0.5

    label_img = nib.load(str(case['seg']))
    label_data = label_img.get_fdata()
    label_remapped = remap_bratsafrica_labels(label_data)

    from scipy.ndimage import zoom
    zoom_factors = [target_size / s for s in label_remapped.shape]
    label_resized = zoom(label_remapped, zoom_factors, order=0)
    gt_tumor_mask = (label_resized == 1)

    dice = dice_score(pred_mask, gt_tumor_mask)
    intersection = np.logical_and(pred_mask, gt_tumor_mask).sum()
    sensitivity = intersection / max(gt_tumor_mask.sum(), 1)
    precision = intersection / max(pred_mask.sum(), 1)

    return {
        'case_id': case['case_id'],
        'category': case['category'],
        'dice': float(dice),
        'sensitivity': float(sensitivity),
        'precision': float(precision),
        'pred_volume_voxels': int(pred_mask.sum()),
        'true_volume_voxels': int(gt_tumor_mask.sum()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-cases", type=int, default=30,
                        help="Number of BraTS-Africa cases to evaluate (default: 30, use -1 for all 146)")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Model checkpoint path (default: v3 best model)")
    args = parser.parse_args()

    checkpoint_path = args.checkpoint or str(MODELS_DIR / "segmentation" / "unet_3d_v3_moredata.pth")
    max_cases = None if args.num_cases == -1 else args.num_cases

    print("=" * 70)
    print("🌍 CROSS-POPULATION EVALUATION: BraTS-Africa")
    print("=" * 70)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Dataset: {DATASET_ROOT}")
    print()

    if not DATASET_ROOT.exists():
        logger.error(f"❌ Dataset not found at {DATASET_ROOT}")
        sys.exit(1)

    cases = find_cases(max_cases=max_cases)
    print(f"✅ Found {len(cases)} complete cases to evaluate")
    print(f"   (Note: this model was NEVER trained on any BraTS-Africa data -")
    print(f"    this is a genuine cross-population generalization test)")
    print()

    model = create_segmentation_model(device='cpu', checkpoint_path=checkpoint_path)
    model.eval()
    preprocessor = MRIPreprocessor(target_shape=(128, 128, 128))

    results = []
    for i, case in enumerate(cases):
        print(f"[{i+1}/{len(cases)}] Evaluating {case['case_id']} ({case['category']})...")
        try:
            result = evaluate_case(case, model, preprocessor)
            results.append(result)
            print(f"   Dice={result['dice']:.4f} Sens={result['sensitivity']:.4f} "
                  f"Prec={result['precision']:.4f} "
                  f"(pred={result['pred_volume_voxels']:,} true={result['true_volume_voxels']:,})")
        except Exception as e:
            logger.error(f"   ❌ Failed: {e}")

    if not results:
        print("❌ No cases successfully evaluated")
        sys.exit(1)

    dice_scores = [r['dice'] for r in results]
    sens_scores = [r['sensitivity'] for r in results]
    prec_scores = [r['precision'] for r in results]

    print()
    print("=" * 70)
    print(f"📈 SUMMARY (n={len(results)} BraTS-Africa cases)")
    print("=" * 70)
    print(f"   Dice:        {np.mean(dice_scores):.4f} ± {np.std(dice_scores):.4f}")
    print(f"   Sensitivity: {np.mean(sens_scores):.4f} ± {np.std(sens_scores):.4f}")
    print(f"   Precision:   {np.mean(prec_scores):.4f} ± {np.std(prec_scores):.4f}")
    print()
    print(f"   For comparison, primary (Decathlon) held-out validation was:")
    print(f"   Dice: 0.478 ± 0.224 (16 cases)")
    print("=" * 70)

    RESULTS_METRICS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    csv_path = RESULTS_METRICS_DIR / f"bratsafrica_evaluation_{timestamp}.csv"
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"💾 Per-case results saved to: {csv_path}")

    summary = {
        'n_cases': len(results),
        'dice_mean': float(np.mean(dice_scores)),
        'dice_std': float(np.std(dice_scores)),
        'sensitivity_mean': float(np.mean(sens_scores)),
        'sensitivity_std': float(np.std(sens_scores)),
        'precision_mean': float(np.mean(prec_scores)),
        'precision_std': float(np.std(prec_scores)),
        'comparison_decathlon_dice_mean': 0.478,
        'comparison_decathlon_dice_std': 0.224,
    }
    summary_path = RESULTS_METRICS_DIR / f"bratsafrica_summary_{timestamp}.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"💾 Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
