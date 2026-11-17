"""
Download REAL brain tumor MRI data (Medical Segmentation Decathlon Task01_BrainTumour).

This dataset contains real clinical multi-modal MRI scans (FLAIR, T1w, T1-Gd, T2w)
with expert-annotated tumor segmentations. It's the same type of data used in
BraTS challenges, but distributed via the Medical Segmentation Decathlon (MSD)
with no registration required.

File location: scripts/download_real_data.py
Usage: python scripts/download_real_data.py

NOTE: This is a ~2GB download. On a slow connection this may take a while.
MONAI's downloader supports resuming, so if it's interrupted, just re-run
this script and it will pick up where it left off.
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data" / "brats_dataset"


def main():
    print("=" * 60)
    print("🧠 DOWNLOADING REAL BRAIN TUMOR MRI DATA")
    print("=" * 60)
    print()
    print("Dataset: Medical Segmentation Decathlon - Task01_BrainTumour")
    print("Contains: Real clinical MRI scans + expert tumor annotations")
    print("Size: ~2 GB (compressed)")
    print("Modalities: FLAIR, T1w, T1-Gd (=T1c), T2w")
    print()
    print("⚠️  This may take 10-60+ minutes depending on your connection.")
    print("    If interrupted, just re-run this script — MONAI resumes")
    print("    partial downloads automatically.")
    print()

    try:
        from monai.apps import DecathlonDataset
    except ImportError:
        print("❌ MONAI not found. Run: pip install monai")
        sys.exit(1)

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("📥 Starting download + extraction...")
    print(f"   Destination: {DATA_DIR}")
    print()

    try:
        # This downloads, extracts, and indexes the dataset automatically.
        # download=True triggers the fetch; if files already exist, it skips.
        dataset = DecathlonDataset(
            root_dir=str(DATA_DIR),
            task="Task01_BrainTumour",
            section="training",
            download=True,
            cache_rate=0.0,  # Don't cache into RAM, just download to disk
            num_workers=2,
        )

        print()
        print("=" * 60)
        print("✅ DOWNLOAD COMPLETE!")
        print("=" * 60)
        print()
        print(f"📁 Location: {DATA_DIR}")
        print(f"📊 Training cases available: {len(dataset)}")
        print()
        print("Each case contains a 4-channel NIfTI file with modalities:")
        print("  Channel 0: FLAIR")
        print("  Channel 1: T1w")
        print("  Channel 2: T1-Gd (contrast-enhanced, ≈ T1c)")
        print("  Channel 3: T2w")
        print()
        print("🚀 NEXT STEP: Run the data splitting script to convert")
        print("   these into separate T1/T2/FLAIR/T1c files our pipeline expects:")
        print()
        print("   python scripts/split_decathlon_case.py")
        print()

    except Exception as e:
        print()
        print("❌ Download failed or was interrupted.")
        print(f"   Error: {e}")
        print()
        print("💡 TIP: Just re-run this script — MONAI will resume")
        print("   from where it left off (partial files are kept).")
        print()
        print("💡 ALTERNATIVE: If downloads keep failing due to slow/")
        print("   unstable internet, use the synthetic data generator")
        print("   instead to keep developing:")
        print()
        print("   python scripts/generate_synthetic_data.py")
        sys.exit(1)


if __name__ == "__main__":
    main()
