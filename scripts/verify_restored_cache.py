#!/usr/bin/env python3
"""
verify_restored_cache.py -- prove the rebuilt cache reproduces the original
preprocessing before anything is trained on it.

    python scripts/verify_restored_cache.py \
        --checkpoint models/segmentation/unet3d_cv_fold0.pth \
        --fold 0

    # quick smoke test on 20 cases before committing to the full fold
    python scripts/verify_restored_cache.py \
        --checkpoint models/segmentation/unet3d_cv_fold0.pth \
        --fold 0 --limit 20

WHAT THIS GUARDS AGAINST
------------------------
restore_decathlon_cache.py rebuilds data/brats_dataset/cache_128 by calling
the same preprocessing code that produced the original. "Same code" is not
the same as "same output": a different scipy or nibabel build can change an
interpolation path, and float16 storage will then preserve the difference.

If the restored arrays differ from the originals, SegResNet folds 1 and 2
would be trained on data subtly unlike the data fold 0 was trained on. The
resulting three-fold comparison -- the one Reviewer 1 called "required" --
would be invalid, and no reviewer could detect it from the manuscript. The
error would be entirely ours and entirely invisible.

So this script does not ask whether the cache looks plausible. It takes a
checkpoint whose per-case scores are already on record, re-evaluates it
against the rebuilt cache, and compares case by case with the values in
results/metrics/cross_population.json. Those 484 records were written
before the cache was lost, so they are an independent reference: if the
preprocessing is faithful, every Dice must come back the same.

HOW TO READ THE RESULT
----------------------
Model inference is not bit-deterministic across thread counts, so
deviations around 1e-7 are expected and harmless. A genuine preprocessing
difference moves Dice by far more -- typically 1e-3 or worse, because a
changed interpolation shifts whole voxels in or out of the mask.

The script exits non-zero if any case deviates by more than --tolerance
(default 1e-4). Do not train on a cache that fails this.
"""

import os

NUM_THREADS = int(os.environ.get("XAI_NUM_THREADS", "8"))
os.environ.setdefault("OMP_NUM_THREADS", str(NUM_THREADS))
os.environ.setdefault("MKL_NUM_THREADS", str(NUM_THREADS))

import argparse
import json
import statistics
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model_loader import load_trained_model, load_fold_split
from evaluate_cross_population import (load_case, collect, remap_decathlon,
                                       binary_metrics, THRESHOLD)

TUMOUR_CLASS = 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--cache",
                    default=str(PROJECT_ROOT / "data/brats_dataset/cache_128"))
    ap.add_argument("--folds",
                    default=str(PROJECT_ROOT
                               / "data/brats_dataset/folds_seed42_3fold.json"))
    ap.add_argument("--reference",
                    default=str(PROJECT_ROOT
                               / "results/metrics/cross_population.json"),
                    help="file holding the per-case scores recorded before "
                         "the cache was lost")
    ap.add_argument("--tolerance", type=float, default=1e-4,
                    help="maximum per-case Dice deviation to accept")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--threads", type=int, default=NUM_THREADS)
    ap.add_argument("--project-root")
    ap.add_argument("--out",
                    default=str(PROJECT_ROOT
                               / "results/metrics/cache_verification.json"))
    args = ap.parse_args()

    torch.set_num_threads(args.threads)

    # ---- the reference scores -------------------------------------
    ref_path = Path(args.reference)
    if not ref_path.exists():
        sys.exit("reference file not found: %s\nWithout the recorded per-case "
                 "scores there is nothing to verify against." % ref_path)
    ref_doc = json.loads(ref_path.read_text())
    ref = {r["case"]: r for r in ref_doc["per_case"]["decathlon"]
           if r.get("fold") == args.fold}
    if not ref:
        sys.exit("no fold %d records in %s" % (args.fold, ref_path))
    print("reference holds %d case(s) for fold %d" % (len(ref), args.fold),
          file=sys.stderr)

    # ---- recompute -----------------------------------------------
    provenance = {}
    model, meta = load_trained_model(args.checkpoint,
                                     project_root=args.project_root,
                                     provenance=provenance)
    model.eval()

    _, val_ids, _ = load_fold_split(args.folds, args.fold)
    paths = collect(args.cache, val_ids)
    if not paths:
        sys.exit("no cached cases found under %s for fold %d. Run "
                 "restore_decathlon_cache.py first." % (args.cache, args.fold))
    if args.limit:
        paths = paths[:args.limit]
    print("evaluating %d cached case(s) from %s" % (len(paths), args.cache),
          file=sys.stderr)

    rows, missing_ref = [], []
    for i, p in enumerate(paths, 1):
        name = os.path.basename(p)
        if name not in ref:
            missing_ref.append(name)
            continue
        img, lab = load_case(p, remap=remap_decathlon)
        if lab is None:
            missing_ref.append(name + " (no label)")
            continue
        with torch.no_grad():
            prob = torch.softmax(model(img), dim=1)[0, TUMOUR_CLASS].numpy()
        m = binary_metrics(prob >= THRESHOLD, lab.numpy() == TUMOUR_CLASS)
        d_now, d_ref = float(m["dice"]), float(ref[name]["dice"])
        rows.append({
            "case": name,
            "dice_recomputed": d_now,
            "dice_recorded": d_ref,
            "abs_deviation": abs(d_now - d_ref),
            "gt_voxels_recomputed": int(m.get("gt_voxels", -1)),
            "gt_voxels_recorded": int(ref[name].get("gt_voxels", -1)),
        })
        if i % 20 == 0 or i == len(paths):
            worst = max(r["abs_deviation"] for r in rows)
            print("  %d/%d  worst deviation so far %.3e"
                  % (i, len(paths), worst), file=sys.stderr)

    if not rows:
        sys.exit("no cases could be compared against the reference.")

    devs = [r["abs_deviation"] for r in rows]
    worst = max(devs)
    worst_case = max(rows, key=lambda r: r["abs_deviation"])
    bands = {
        "exceeding_1e-9": sum(1 for d in devs if d > 1e-9),
        "exceeding_1e-7": sum(1 for d in devs if d > 1e-7),
        "exceeding_1e-6": sum(1 for d in devs if d > 1e-6),
        "exceeding_1e-4": sum(1 for d in devs if d > 1e-4),
        "exceeding_1e-3": sum(1 for d in devs if d > 1e-3),
    }
    # A changed interpolation would move the ground-truth voxel count too,
    # which is a cleaner signal than Dice because it does not involve the
    # model at all.
    gt_mismatch = [r for r in rows
                   if r["gt_voxels_recorded"] >= 0
                   and r["gt_voxels_recomputed"] != r["gt_voxels_recorded"]]

    passed = worst <= args.tolerance and not gt_mismatch

    out = {
        "provenance": provenance,
        "checkpoint_meta": meta,
        "fold": args.fold,
        "cache": args.cache,
        "reference": str(ref_path),
        "n_compared": len(rows),
        "n_reference_for_fold": len(ref),
        "cases_without_reference": missing_ref,
        "tolerance": args.tolerance,
        "max_abs_deviation": worst,
        "mean_abs_deviation": statistics.mean(devs),
        "median_abs_deviation": statistics.median(devs),
        "worst_case": worst_case,
        "deviation_bands": bands,
        "ground_truth_voxel_mismatches": len(gt_mismatch),
        "ground_truth_voxel_mismatch_examples": gt_mismatch[:10],
        "mean_dice_recomputed": statistics.mean(
            r["dice_recomputed"] for r in rows),
        "mean_dice_recorded": statistics.mean(
            r["dice_recorded"] for r in rows),
        "passed": passed,
        "per_case": rows,
        "interpretation": (
            "Deviations near 1e-7 reflect non-deterministic reduction order "
            "in CPU inference and are harmless. Any ground-truth voxel-count "
            "mismatch, or Dice deviation above the tolerance, means the "
            "rebuilt cache is not the cache the reported results were "
            "computed on. In that case do not train on it: the three-fold "
            "SegResNet comparison would silently compare models trained on "
            "different data."),
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))

    print("\n--- cache verification, fold %d, n=%d ---"
          % (args.fold, len(rows)), file=sys.stderr)
    print("  mean Dice recomputed %.6f   recorded %.6f"
          % (out["mean_dice_recomputed"], out["mean_dice_recorded"]),
          file=sys.stderr)
    print("  max  abs deviation   %.3e  (%s)"
          % (worst, worst_case["case"]), file=sys.stderr)
    print("  mean abs deviation   %.3e" % out["mean_abs_deviation"],
          file=sys.stderr)
    for k, v in bands.items():
        print("  cases %-16s %d of %d" % (k, v, len(rows)), file=sys.stderr)
    print("  ground-truth voxel mismatches: %d" % len(gt_mismatch),
          file=sys.stderr)
    if missing_ref:
        print("  cases with no reference record: %d (%s)"
              % (len(missing_ref), ", ".join(missing_ref[:5])),
              file=sys.stderr)
    print("wrote %s" % args.out, file=sys.stderr)

    if passed:
        print("\nPASSED: the rebuilt cache reproduces the original "
              "preprocessing. Training on it is sound.", file=sys.stderr)
    else:
        print("\nFAILED: the rebuilt cache does NOT reproduce the original "
              "preprocessing.", file=sys.stderr)
        if gt_mismatch:
            print("  %d case(s) differ in ground-truth voxel count, which "
                  "points at the label resampling path rather than the model."
                  % len(gt_mismatch), file=sys.stderr)
        print("  Do not train on this cache. Check the scipy and nibabel "
              "versions against those used originally, and report the "
              "deviation bands above.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
