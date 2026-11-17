"""
Pre-compute normalized, resampled 128^3 volumes once, so training reads
arrays instead of re-running NIfTI loading + cubic resampling every epoch.
Saved as float16 to fit available disk.

Usage: python scripts/precompute_cache.py
"""
import sys
from pathlib import Path
import numpy as np
import nibabel as nib
from scipy import ndimage

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from data_processing.preprocessor import MRIPreprocessor

SRC = PROJECT_ROOT / "data/brats_dataset/full_cohort/Task01_BrainTumour"
OUT = PROJECT_ROOT / "data/brats_dataset/cache_128"
(OUT / "images").mkdir(parents=True, exist_ok=True)
(OUT / "labels").mkdir(parents=True, exist_ok=True)

TARGET = (128, 128, 128)


def main():
    pre = MRIPreprocessor(target_shape=TARGET)
    files = sorted(SRC.glob("imagesTr/BRATS_*.nii.gz"))
    total = len(files)
    print(f"Found {total} cases. Caching to {OUT}", flush=True)

    skipped = 0
    for i, f in enumerate(files, 1):
        cid = f.name.replace(".nii.gz", "")
        img_out = OUT / "images" / f"{cid}.npy"
        lbl_out = OUT / "labels" / f"{cid}.npy"

        if img_out.exists() and lbl_out.exists():
            skipped += 1
            continue

        lbl_path = SRC / "labelsTr" / f.name
        if not lbl_path.exists():
            print(f"  !! missing label for {cid}, skipping", flush=True)
            continue

        try:
            data = nib.load(str(f)).get_fdata()
            img = np.stack(
                [pre.resize(pre.normalize(data[..., c])) for c in range(4)]
            ).astype(np.float16)

            lbl = nib.load(str(lbl_path)).get_fdata()
            zoom = [t / c for t, c in zip(TARGET, lbl.shape)]
            lbl = ndimage.zoom(lbl, zoom, order=0).astype(np.uint8)

            np.save(img_out, img)
            np.save(lbl_out, lbl)
        except Exception as e:
            print(f"  !! failed {cid}: {e}", flush=True)
            continue

        if i % 10 == 0 or i == total:
            print(f"{i}/{total}  ({cid})", flush=True)

    n = len(list((OUT / "images").glob("*.npy")))
    print(f"\nDone. Cached {n} cases (skipped {skipped} already present).", flush=True)


if __name__ == "__main__":
    main()
