#!/usr/bin/env python3
"""
restore_decathlon_cache.py -- rebuild data/brats_dataset/cache_128 from the
Medical Segmentation Decathlon Task01_BrainTumour archive.

    # 1. download (resumable; re-run the same command after any interruption)
    curl -L -C - -o data/Task01_BrainTumour.tar \
      https://msd-for-monai.s3-us-west-2.amazonaws.com/Task01_BrainTumour.tar

    # 2. rebuild the cache (resumable; re-run to continue)
    python scripts/restore_decathlon_cache.py \
        --tar data/Task01_BrainTumour.tar \
        --folds data/brats_dataset/folds_seed42_3fold.json

    # 3. verify before trusting it
    python scripts/verify_restored_cache.py \
        --checkpoint models/segmentation/unet3d_cv_fold0.pth --fold 0

WHY THIS EXISTS
---------------
The caches recorded in the provenance of every reported result --
data/brats_dataset/cache_128 and cache_africa -- were deleted from this
machine during a disk-space recovery. Neither was ever tracked by git
(.gitignore excludes data/brats_dataset/), so there is nothing to restore
from history. Of the 484 Decathlon cases the folds file names, 80 survive
as raw NIfTI; 404 do not.

Nothing already reported depends on this: the per-case metrics behind
every table are intact in results/metrics/. What the missing cache blocks
is NEW computation -- specifically the three-fold SegResNet benchmark that
Reviewer 1 called "required".

PRESERVING THE ORIGINAL PREPROCESSING
-------------------------------------
This script does not invent a preprocessing path. It calls the same
MRIPreprocessor, in the same order, with the same target shape and the
same dtypes as scripts/precompute_cache.py, which produced the original
cache: images as float16 after per-channel normalize-then-resize, labels
as uint8 after ndimage.zoom with order=0.

That matters more than it looks. If the restored arrays differ from the
originals even slightly -- a different scipy interpolation path, say --
then SegResNet folds 1 and 2 would train on data subtly unlike the data
fold 0 was trained on, and the three-fold comparison would be invalid in
a way no reviewer could detect but which would be real. Run
verify_restored_cache.py before training anything.

DISK DISCIPLINE
---------------
The archive stores every image under imagesTr before any label under
labelsTr, so reading it as a stream would mean holding hundreds of
extracted images on disk while waiting for their labels to arrive. This
script therefore opens the archive for RANDOM ACCESS and pulls each
case's image and label together by name. The tar is uncompressed -- only
its members are gzipped -- so seeking is cheap, and exactly one case's two
NIfTI files exist on disk at any moment.

The run aborts rather than filling the filesystem if free space falls
below --min-free-gb.

Each cached case costs 18 MiB: a 4 x 128^3 float16 image (16 MiB) and a
128^3 uint8 label (2 MiB). The 484-case cache is therefore about 8.5 GiB.
"""

import os

import argparse
import json
import shutil
import sys
import tarfile
import tempfile
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

TARGET = (128, 128, 128)
BYTES_PER_CASE = 18 * 1024 * 1024
GIB = 1024.0 ** 3
SUFFIX = ".nii.gz"


def free_gb(path):
    return shutil.disk_usage(path).free / GIB


def wanted_case_ids(folds_path):
    """Every case the folds file names, across all folds. Training on any
    one fold touches all of them, so the full set is required."""
    cfg = json.loads(Path(folds_path).read_text())
    ids = set()
    for entry in cfg["folds"]:
        ids |= set(entry.get("train", []))
        ids |= set(entry.get("val", entry.get("validation", [])))
    return ids, cfg


def index_archive(tf):
    """Map case id -> (image member, label member).

    Built from the archive's own member list rather than from an assumed
    path prefix, so a differently rooted archive still indexes correctly.
    imagesTs is ignored: the test split is never used by this project.
    """
    images, labels = {}, {}
    for m in tf.getmembers():
        if not m.isfile() or not m.name.endswith(SUFFIX):
            continue
        base = os.path.basename(m.name)
        if not base.startswith("BRATS_"):
            continue
        cid = base[: -len(SUFFIX)]
        parts = m.name.split("/")
        if "imagesTr" in parts:
            images[cid] = m
        elif "labelsTr" in parts:
            labels[cid] = m
    return images, labels


def extract_to(tf, member, dest):
    src = tf.extractfile(member)
    if src is None:
        raise IOError("could not read %s from the archive" % member.name)
    with open(dest, "wb") as w:
        shutil.copyfileobj(src, w)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tar", required=True,
                    help="Task01_BrainTumour.tar as downloaded")
    ap.add_argument("--folds",
                    default=str(PROJECT_ROOT
                               / "data/brats_dataset/folds_seed42_3fold.json"))
    ap.add_argument("--out",
                    default=str(PROJECT_ROOT / "data/brats_dataset/cache_128"))
    ap.add_argument("--min-free-gb", type=float, default=1.0,
                    help="abort before free space falls below this")
    ap.add_argument("--limit", type=int,
                    help="stop after this many newly cached cases")
    ap.add_argument("--dry-run", action="store_true",
                    help="report what would be done, write nothing")
    args = ap.parse_args()

    tar_path = Path(args.tar)
    if not tar_path.exists():
        sys.exit("archive not found: %s\nDownload it first:\n"
                 "  curl -L -C - -o %s \\\n"
                 "    https://msd-for-monai.s3-us-west-2.amazonaws.com/"
                 "Task01_BrainTumour.tar" % (tar_path, tar_path))

    out = Path(args.out)
    img_dir, lbl_dir = out / "images", out / "labels"
    if not args.dry_run:
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

    wanted, cfg = wanted_case_ids(args.folds)
    print("folds file names %d case(s) across %d fold(s), seed %s"
          % (len(wanted), cfg.get("n_folds"), cfg.get("seed")), file=sys.stderr)

    # Sweep anything left behind by an interrupted or failed earlier run.
    # Both spellings are cleared: ".npy.part" from a clean interruption and
    # ".npy.part.npy" from the np.save extension-rewriting bug fixed below.
    swept = 0
    for d in (img_dir, lbl_dir):
        if not d.exists():
            continue
        for junk in list(d.glob("*.npy.part")) + list(d.glob("*.part.npy")):
            try:
                junk.unlink()
                swept += 1
            except OSError:
                pass
    if swept:
        print("swept %d incomplete file(s) from a previous run" % swept,
              file=sys.stderr)

    already = {p.stem for p in img_dir.glob("*.npy")
               if not p.name.endswith(".part.npy")} if img_dir.exists() else set()
    already &= wanted
    todo = sorted(wanted - already)
    print("already cached: %d   still to cache: %d" % (len(already), len(todo)),
          file=sys.stderr)

    need_gb = len(todo) * BYTES_PER_CASE / GIB
    have_gb = free_gb(out.parent if out.parent.exists() else PROJECT_ROOT)
    print("free space %.1f GiB   required for remaining cases %.1f GiB"
          % (have_gb, need_gb), file=sys.stderr)

    print("\nindexing archive (seeks only, no extraction)...", file=sys.stderr)
    t_index = time.time()
    with tarfile.open(tar_path, "r") as tf:
        images, labels = index_archive(tf)
        print("  %d training image(s), %d label(s) indexed in %.1f s"
              % (len(images), len(labels), time.time() - t_index),
              file=sys.stderr)

        have_both = {c for c in wanted if c in images and c in labels}
        no_image = sorted(wanted - set(images))
        no_label = sorted(wanted - set(labels))
        if no_image:
            print("  !! %d wanted case(s) have no image in the archive: %s"
                  % (len(no_image), ", ".join(no_image[:6])), file=sys.stderr)
        if no_label:
            print("  !! %d wanted case(s) have no label in the archive: %s"
                  % (len(no_label), ", ".join(no_label[:6])), file=sys.stderr)
        if len(have_both) != len(wanted):
            print("  the archive covers %d of the %d cases the folds file "
                  "names" % (len(have_both), len(wanted)), file=sys.stderr)

        todo = [c for c in todo if c in have_both]
        if not todo:
            print("\nnothing to do: every available case is already cached.",
                  file=sys.stderr)
            return
        if args.limit:
            todo = todo[:args.limit]

        if need_gb + args.min_free_gb > have_gb:
            print("\nWARNING: %.1f GiB free is not enough for all %d remaining "
                  "cases (%.1f GiB needed plus %.1f GiB reserve). The run will "
                  "cache what fits and stop cleanly; re-run after freeing more "
                  "space and it will continue where it left off."
                  % (have_gb, len(todo), need_gb, args.min_free_gb),
                  file=sys.stderr)

        if args.dry_run:
            print("\ndry run: would cache %d case(s), about %.1f GiB."
                  % (len(todo), len(todo) * BYTES_PER_CASE / GIB),
                  file=sys.stderr)
            print("Nothing written.", file=sys.stderr)
            return

        # Imported here so --dry-run needs none of the heavy stack
        import nibabel as nib
        from scipy import ndimage
        from data_processing.preprocessor import MRIPreprocessor

        pre = MRIPreprocessor(target_shape=TARGET)

        done = failed = 0
        stopped_for_space = False
        t0 = time.time()
        tmp = Path(tempfile.mkdtemp(prefix="msd_restore_"))
        try:
            for cid in todo:
                if free_gb(out) < args.min_free_gb + 0.05:
                    print("\nstopping: free space %.2f GiB has reached the "
                          "--min-free-gb floor. %d case(s) cached this run; "
                          "re-run after freeing space to continue."
                          % (free_gb(out), done), file=sys.stderr)
                    stopped_for_space = True
                    break

                ip = tmp / (cid + "_img" + SUFFIX)
                lp = tmp / (cid + "_lbl" + SUFFIX)
                try:
                    extract_to(tf, images[cid], ip)
                    extract_to(tf, labels[cid], lp)

                    # --- identical to scripts/precompute_cache.py ---
                    data = nib.load(str(ip)).get_fdata()
                    img = np.stack(
                        [pre.resize(pre.normalize(data[..., c]))
                         for c in range(4)]
                    ).astype(np.float16)

                    lbl = nib.load(str(lp)).get_fdata()
                    zoom = [t / c for t, c in zip(TARGET, lbl.shape)]
                    lbl = ndimage.zoom(lbl, zoom, order=0).astype(np.uint8)
                    # ------------------------------------------------

                    if img.shape != (4,) + TARGET:
                        raise ValueError("image shape %s, expected %s"
                                         % (img.shape, (4,) + TARGET))
                    if lbl.shape != TARGET:
                        raise ValueError("label shape %s, expected %s"
                                         % (lbl.shape, TARGET))

                    # Write under a temporary name so an interrupted run
                    # never leaves a half-written .npy that a resume trusts.
                    #
                    # np.save() appends ".npy" to any path that does not
                    # already end in it, so np.save("x.npy.part") silently
                    # writes "x.npy.part.npy" and the rename below then
                    # fails. Passing an open handle suppresses that
                    # rewriting entirely, so the bytes land where asked.
                    it = img_dir / (cid + ".npy.part")
                    lt = lbl_dir / (cid + ".npy.part")
                    with open(it, "wb") as fh:
                        np.save(fh, img)
                    with open(lt, "wb") as fh:
                        np.save(fh, lbl)
                    it.replace(img_dir / (cid + ".npy"))
                    lt.replace(lbl_dir / (cid + ".npy"))
                    done += 1
                except Exception as exc:
                    failed += 1
                    print("  !! %s failed: %s: %s"
                          % (cid, type(exc).__name__, exc), file=sys.stderr)
                finally:
                    for p in (ip, lp):
                        try:
                            p.unlink()
                        except OSError:
                            pass

                if done and done % 20 == 0:
                    rate = done / max(time.time() - t0, 1e-9)
                    left = len(todo) - done
                    print("  %d/%d cached  %.2f case/s  eta %.0f min  "
                          "free %.1f GiB"
                          % (done, len(todo), rate,
                             (left / rate) / 60 if rate else 0,
                             free_gb(out)), file=sys.stderr)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    n_img = len(list(img_dir.glob("*.npy")))
    n_lbl = len(list(lbl_dir.glob("*.npy")))
    print("\n--- restore summary ---", file=sys.stderr)
    print("  newly cached      %d" % done, file=sys.stderr)
    print("  failed            %d" % failed, file=sys.stderr)
    print("  cache now holds   %d image(s), %d label(s)" % (n_img, n_lbl),
          file=sys.stderr)
    print("  folds file needs  %d" % len(wanted), file=sys.stderr)
    print("  elapsed           %.1f min" % ((time.time() - t0) / 60.0),
          file=sys.stderr)
    print("  free space        %.1f GiB" % free_gb(out), file=sys.stderr)

    missing = sorted(wanted - {p.stem for p in img_dir.glob("*.npy")})
    if missing:
        print("\n%d case(s) still missing." % len(missing), file=sys.stderr)
        print("  first few: %s" % ", ".join(missing[:8]), file=sys.stderr)
        if stopped_for_space:
            print("  Free more space and re-run to continue.", file=sys.stderr)
        sys.exit(1)

    print("\nAll %d cases cached. Do NOT train on this yet -- verify it "
          "reproduces the original preprocessing first:" % len(wanted),
          file=sys.stderr)
    print("  python scripts/verify_restored_cache.py \\\n"
          "      --checkpoint models/segmentation/unet3d_cv_fold0.pth \\\n"
          "      --fold 0", file=sys.stderr)


if __name__ == "__main__":
    main()
