"""
Generate synthetic MRI test data (NIfTI files) for immediate pipeline testing.
No download required - creates realistic-shaped fake MRI volumes in seconds.

File location: scripts/generate_synthetic_data.py
Usage: python scripts/generate_synthetic_data.py
"""
import numpy as np
import nibabel as nib
from pathlib import Path
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

OUTPUT_DIR = PROJECT_ROOT / "data" / "sample_cases" / "synthetic_case_001"


def create_synthetic_brain_volume(shape=(155, 240, 240), seed=42):
    """
    Create a synthetic 3D brain-like volume with a fake tumor region.

    Args:
        shape: Volume dimensions (D, H, W) - matches typical BraTS shape
        seed: Random seed for reproducibility

    Returns:
        volume: Base brain-like volume
        tumor_mask: Ground truth tumor location (for reference only)
    """
    rng = np.random.default_rng(seed)

    # Create a smooth "brain-like" background using layered noise
    volume = rng.normal(loc=400, scale=50, size=shape).astype(np.float32)

    # Smooth it to look less like pure noise
    from scipy.ndimage import gaussian_filter
    volume = gaussian_filter(volume, sigma=3)

    # Add a spherical "skull/brain boundary" mask so edges are dark (background)
    center = np.array(shape) / 2
    radii = np.array(shape) * 0.42
    zz, yy, xx = np.meshgrid(
        np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]), indexing='ij'
    )
    dist = np.sqrt(
        ((zz - center[0]) / radii[0]) ** 2 +
        ((yy - center[1]) / radii[1]) ** 2 +
        ((xx - center[2]) / radii[2]) ** 2
    )
    brain_mask = dist <= 1.0
    volume = volume * brain_mask

    # Add a fake "tumor" - a bright spherical region off-center
    tumor_center = center + np.array([10, 30, -20])
    tumor_radius = 18
    tumor_dist = np.sqrt(
        (zz - tumor_center[0]) ** 2 +
        (yy - tumor_center[1]) ** 2 +
        (xx - tumor_center[2]) ** 2
    )
    tumor_mask = (tumor_dist <= tumor_radius) & brain_mask
    volume[tumor_mask] += 250  # Tumor appears brighter (hyperintense)

    # Add edema ring around tumor (slightly less bright, wider)
    edema_mask = (tumor_dist <= tumor_radius * 1.6) & (tumor_dist > tumor_radius) & brain_mask
    volume[edema_mask] += 100

    volume = np.clip(volume, 0, None)

    return volume.astype(np.float32), tumor_mask.astype(np.uint8)


def generate_modality(base_volume, modality_name, seed_offset=0):
    """
    Generate a modality-specific variant of the base volume.
    Different MRI modalities (T1, T2, FLAIR, T1c) have different
    intensity characteristics for the same anatomy.
    """
    rng = np.random.default_rng(42 + seed_offset)

    # Add modality-specific intensity scaling + noise
    scale_factors = {
        'T1': 1.0,
        'T2': 1.3,
        'FLAIR': 1.15,
        'T1c': 1.05,  # Contrast-enhanced, tumor appears brighter
    }
    scale = scale_factors.get(modality_name, 1.0)

    modality_volume = base_volume * scale
    noise = rng.normal(0, 15, base_volume.shape)
    modality_volume = modality_volume + noise
    modality_volume = np.clip(modality_volume, 0, None)

    return modality_volume.astype(np.float32)


def save_as_nifti(volume, filepath, affine=None):
    """Save a 3D numpy array as a NIfTI file"""
    if affine is None:
        affine = np.eye(4)
        affine[0, 0] = affine[1, 1] = affine[2, 2] = 1.0  # 1mm isotropic voxels

    img = nib.Nifti1Image(volume, affine)
    nib.save(img, str(filepath))
    print(f"  ✅ Saved: {filepath.name} ({volume.shape}, {volume.nbytes / 1e6:.1f} MB)")


def main():
    print("=" * 60)
    print("🧠 GENERATING SYNTHETIC MRI TEST DATA")
    print("=" * 60)
    print()
    print("This creates fake but realistically-shaped MRI volumes")
    print("so you can test the full pipeline WITHOUT downloading")
    print("real clinical data first.")
    print()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("📐 Creating base brain volume with synthetic tumor...")
    base_volume, tumor_ground_truth = create_synthetic_brain_volume()
    print(f"  ✅ Base volume shape: {base_volume.shape}")
    print(f"  ✅ Synthetic tumor voxels: {tumor_ground_truth.sum():,}")
    print()

    print("🔬 Generating 4 MRI modalities (T1, T2, FLAIR, T1c)...")
    modalities = ['T1', 'T2', 'FLAIR', 'T1c']

    for i, modality in enumerate(modalities):
        mod_volume = generate_modality(base_volume, modality, seed_offset=i)
        filepath = OUTPUT_DIR / f"{modality}.nii.gz"
        save_as_nifti(mod_volume, filepath)

    # Also save ground truth tumor mask for reference (not used by pipeline,
    # but useful if you want to compare predictions later)
    gt_path = OUTPUT_DIR / "ground_truth_tumor_mask.nii.gz"
    save_as_nifti(tumor_ground_truth.astype(np.float32), gt_path)

    print()
    print("=" * 60)
    print("✅ SYNTHETIC TEST DATA READY!")
    print("=" * 60)
    print()
    print(f"📁 Location: {OUTPUT_DIR}")
    print()
    print("Files created:")
    for f in sorted(OUTPUT_DIR.glob("*.nii.gz")):
        print(f"  • {f.name}")
    print()
    print("🚀 NEXT STEP: Upload these 4 files in the Streamlit UI:")
    print(f"   {OUTPUT_DIR}/T1.nii.gz")
    print(f"   {OUTPUT_DIR}/T2.nii.gz")
    print(f"   {OUTPUT_DIR}/FLAIR.nii.gz")
    print(f"   {OUTPUT_DIR}/T1c.nii.gz")
    print()
    print("   (Don't upload ground_truth_tumor_mask.nii.gz - that's just")
    print("    for your own reference, not an input file)")
    print()


if __name__ == "__main__":
    main()
