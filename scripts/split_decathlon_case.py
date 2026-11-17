"""
Split Medical Segmentation Decathlon 4-channel NIfTI files into separate
T1, T2, FLAIR, T1c files matching our pipeline's expected input format.

The Decathlon Task01_BrainTumour stores all 4 modalities stacked as
channels in ONE file per case. Our pipeline expects 4 SEPARATE files.
This script converts a few sample cases for easy testing.

File location: scripts/split_decathlon_case.py
Usage: python scripts/split_decathlon_case.py [--num-cases 3]
"""
import sys
import argparse
from pathlib import Path
import nibabel as nib
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DECATHLON_DIR = PROJECT_ROOT / "data" / "brats_dataset" / "Task01_BrainTumour_subset"
OUTPUT_DIR = PROJECT_ROOT / "data" / "sample_cases"

# Decathlon Task01_BrainTumour channel order (fixed, from the dataset spec)
MODALITY_CHANNELS = {
    0: "FLAIR",
    1: "T1",
    2: "T1c",   # T1-Gd (contrast-enhanced)
    3: "T2",
}


def split_case(image_path: Path, label_path: Path, output_case_dir: Path):
    """Split one 4-channel case into separate modality files"""
    output_case_dir.mkdir(parents=True, exist_ok=True)

    img = nib.load(str(image_path))
    data = img.get_fdata()  # shape: (H, W, D, 4)
    affine = img.affine

    print(f"  Volume shape: {data.shape}")

    for channel_idx, modality_name in MODALITY_CHANNELS.items():
        modality_volume = data[..., channel_idx]
        out_path = output_case_dir / f"{modality_name}.nii.gz"
        nib.save(nib.Nifti1Image(modality_volume.astype(np.float32), affine), str(out_path))
        print(f"    ✅ {modality_name}.nii.gz saved")

    # Also copy the ground truth segmentation label for reference
    if label_path.exists():
        label_img = nib.load(str(label_path))
        out_label_path = output_case_dir / "ground_truth_label.nii.gz"
        nib.save(label_img, str(out_label_path))
        print(f"    ✅ ground_truth_label.nii.gz saved (for your reference)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-cases", type=int, default=3,
                        help="How many cases to split (default: 3)")
    args = parser.parse_args()

    print("=" * 60)
    print("🧠 SPLITTING DECATHLON CASES INTO T1/T2/FLAIR/T1c FILES")
    print("=" * 60)
    print()

    images_dir = DECATHLON_DIR / "imagesTr"
    labels_dir = DECATHLON_DIR / "labelsTr"

    if not images_dir.exists():
        print(f"❌ Could not find: {images_dir}")
        print()
        print("💡 Have you run the download script first?")
        print("   python scripts/download_real_data.py")
        sys.exit(1)

    # Get case files (skip hidden/system files starting with '.')
    image_files = sorted([f for f in images_dir.glob("*.nii.gz") if not f.name.startswith('.')])

    if not image_files:
        print(f"❌ No cases found in {images_dir}")
        sys.exit(1)

    num_to_process = min(args.num_cases, len(image_files))
    print(f"📊 Found {len(image_files)} total cases. Processing {num_to_process}...")
    print()

    for i, image_path in enumerate(image_files[:num_to_process]):
        case_id = image_path.stem.replace(".nii", "")
        label_path = labels_dir / image_path.name
        output_case_dir = OUTPUT_DIR / f"real_case_{i+1:03d}_{case_id}"

        print(f"[{i+1}/{num_to_process}] Processing {case_id}...")
        split_case(image_path, label_path, output_case_dir)
        print(f"  📁 Saved to: {output_case_dir}")
        print()

    print("=" * 60)
    print("✅ DONE!")
    print("=" * 60)
    print()
    print(f"📁 Real test cases available in: {OUTPUT_DIR}")
    print()
    print("Each folder contains 4 files ready to upload in the Streamlit UI:")
    print("  T1.nii.gz, T2.nii.gz, FLAIR.nii.gz, T1c.nii.gz")
    print()


if __name__ == "__main__":
    main()
