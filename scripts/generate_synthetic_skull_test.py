"""
Generate a synthetic non-skull-stripped test file to validate the
skull-inclusive display code path, since we don't yet have real
non-skull-stripped clinical data (BraTS-Africa's raw package is
currently paused by TCIA; standard BraTS/Decathlon is pre-stripped).

This is PURELY for testing the display/overlay mechanics - the "skull"
here is a crude synthetic bright ring, not real anatomy. Do NOT use
for any scientific claims - only to confirm the code path runs without
error before the professor provides real raw clinical scans.

File location: scripts/generate_synthetic_skull_test.py
Usage: python scripts/generate_synthetic_skull_test.py
"""
import numpy as np
import nibabel as nib
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

OUTPUT_PATH = PROJECT_ROOT / "data" / "sample_cases" / "synthetic_case_001" / "T1_with_skull_TESTONLY.nii.gz"


def main():
    print("=" * 60)
    print("🦴 GENERATING SYNTHETIC SKULL-INCLUDED TEST FILE")
    print("=" * 60)
    print()
    print("⚠️  This is a CRUDE synthetic test file for validating the")
    print("    display code path only - NOT real anatomy, NOT for any")
    print("    scientific use. Real skull-inclusive testing requires")
    print("    actual non-skull-stripped clinical scans.")
    print()

    shape = (155, 240, 240)
    rng = np.random.default_rng(123)

    volume = rng.normal(loc=400, scale=50, size=shape).astype(np.float32)
    from scipy.ndimage import gaussian_filter
    volume = gaussian_filter(volume, sigma=3)

    center = np.array(shape) / 2
    zz, yy, xx = np.meshgrid(
        np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]), indexing='ij'
    )

    brain_radii = np.array(shape) * 0.36
    brain_dist = np.sqrt(
        ((zz - center[0]) / brain_radii[0]) ** 2 +
        ((yy - center[1]) / brain_radii[1]) ** 2 +
        ((xx - center[2]) / brain_radii[2]) ** 2
    )
    brain_mask = brain_dist <= 1.0

    skull_radii = np.array(shape) * 0.42
    skull_dist = np.sqrt(
        ((zz - center[0]) / skull_radii[0]) ** 2 +
        ((yy - center[1]) / skull_radii[1]) ** 2 +
        ((xx - center[2]) / skull_radii[2]) ** 2
    )
    skull_shell = (skull_dist <= 1.0) & (~brain_mask)

    volume_with_skull = np.zeros(shape, dtype=np.float32)
    volume_with_skull[brain_mask] = volume[brain_mask]
    volume_with_skull[skull_shell] = 900.0

    affine = np.eye(4)
    nib.save(nib.Nifti1Image(volume_with_skull, affine), str(OUTPUT_PATH))

    print(f"✅ Saved test file: {OUTPUT_PATH}")
    print()
    print("🚀 To test: upload this file as the 'Non-skull-stripped scan'")
    print("   in the Streamlit UI, alongside the normal 4 skull-stripped")
    print("   modalities, and confirm the attention overlay renders on")
    print("   this (skull-ring-visible) background instead of the default.")
    print()


if __name__ == "__main__":
    main()
