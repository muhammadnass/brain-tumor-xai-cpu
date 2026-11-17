"""
Extract a SMALL SUBSET of cases directly from the already-downloaded
Task01_BrainTumour.tar file, without extracting the entire ~484-case dataset.

This solves disk space constraints: instead of needing room for the full
dataset (~7GB+ extracted), we extract only N cases (a few hundred MB).

The .tar file was already downloaded and MD5-verified in a previous step,
so this does NOT re-download anything - it just does a smarter extraction.

File location: scripts/extract_subset_from_tar.py
Usage: python scripts/extract_subset_from_tar.py --num-cases 30
"""
import sys
import tarfile
import argparse
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

TAR_PATH = PROJECT_ROOT / "data" / "brats_dataset" / "Task01_BrainTumour.tar"
OUTPUT_DIR = PROJECT_ROOT / "data" / "brats_dataset" / "Task01_BrainTumour_subset"
BROKEN_EXTRACTION_DIR = PROJECT_ROOT / "data" / "brats_dataset" / "Task01_BrainTumour"


def get_disk_free_gb(path: Path) -> float:
    """Get free disk space in GB for the partition containing `path`"""
    usage = shutil.disk_usage(path)
    return usage.free / (1024 ** 3)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-cases", type=int, default=30,
                        help="Number of cases to extract (default: 30)")
    parser.add_argument("--delete-tar-after", action="store_true", default=True,
                        help="Delete the 7GB tar file after extraction to free space")
    args = parser.parse_args()

    print("=" * 60)
    print("🧠 EXTRACTING SUBSET FROM EXISTING TAR FILE")
    print("=" * 60)
    print()

    if not TAR_PATH.exists():
        print(f"❌ Tar file not found at {TAR_PATH}")
        print("   Run: python scripts/download_real_data.py first")
        sys.exit(1)

    free_gb = get_disk_free_gb(PROJECT_ROOT)
    print(f"💾 Current free disk space: {free_gb:.1f} GB")

    if free_gb < 1.0:
        print("❌ Less than 1GB free - not enough space even for a small subset.")
        print("   Free up space first (see cleanup suggestions from before).")
        sys.exit(1)

    # Clean up the previous broken/incomplete extraction attempt
    if BROKEN_EXTRACTION_DIR.exists():
        print(f"🧹 Removing incomplete previous extraction at {BROKEN_EXTRACTION_DIR}...")
        shutil.rmtree(BROKEN_EXTRACTION_DIR, ignore_errors=True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    images_out = OUTPUT_DIR / "imagesTr"
    labels_out = OUTPUT_DIR / "labelsTr"
    images_out.mkdir(exist_ok=True)
    labels_out.mkdir(exist_ok=True)

    print(f"📂 Opening tar archive (7.1GB, this may take a moment to index)...")
    with tarfile.open(TAR_PATH, 'r') as tar:
        members = tar.getmembers()
        print(f"   Total files in archive: {len(members)}")

        # Find image files (skip hidden/system files starting with '.')
        image_members = sorted(
            [m for m in members if 'imagesTr/' in m.name and m.name.endswith('.nii.gz')
             and '/._' not in m.name and not Path(m.name).name.startswith('.')],
            key=lambda m: m.name
        )

        print(f"   Available training images in archive: {len(image_members)}")

        num_to_extract = min(args.num_cases, len(image_members))
        selected_images = image_members[:num_to_extract]

        print(f"📥 Extracting {num_to_extract} cases (image + matching label each)...")
        print()

        extracted_count = 0
        for img_member in selected_images:
            case_filename = Path(img_member.name).name  # e.g. BRATS_001.nii.gz
            label_member_name = None

            # Find matching label file
            for m in members:
                if 'labelsTr/' in m.name and Path(m.name).name == case_filename:
                    label_member_name = m
                    break

            if label_member_name is None:
                print(f"  ⚠️ Skipping {case_filename} - no matching label found")
                continue

            # Extract image
            img_target = images_out / case_filename
            with tar.extractfile(img_member) as src, open(img_target, 'wb') as dst:
                shutil.copyfileobj(src, dst)

            # Extract label
            label_target = labels_out / case_filename
            with tar.extractfile(label_member_name) as src, open(label_target, 'wb') as dst:
                shutil.copyfileobj(src, dst)

            extracted_count += 1
            print(f"  ✅ [{extracted_count}/{num_to_extract}] {case_filename}")

    print()
    print(f"✅ Extracted {extracted_count} complete cases (image + label pairs)")

    # Report disk usage of the subset
    subset_size_mb = sum(f.stat().st_size for f in OUTPUT_DIR.rglob('*') if f.is_file()) / (1024 ** 2)
    print(f"📊 Subset size on disk: {subset_size_mb:.0f} MB")

    if args.delete_tar_after:
        print()
        print(f"🧹 Deleting {TAR_PATH.name} (7.1GB) to free disk space...")
        TAR_PATH.unlink()
        print("   ✅ Tar file deleted")
        print("   (If you need more cases later, re-run download_real_data.py to re-fetch it)")

    free_gb_after = get_disk_free_gb(PROJECT_ROOT)
    print()
    print("=" * 60)
    print("✅ DONE!")
    print("=" * 60)
    print(f"📁 Subset location: {OUTPUT_DIR}")
    print(f"📊 Cases extracted: {extracted_count}")
    print(f"💾 Free disk space now: {free_gb_after:.1f} GB")
    print()
    print("🚀 NEXT STEP: python scripts/train_model.py --quick-test")
    print()


if __name__ == "__main__":
    main()
