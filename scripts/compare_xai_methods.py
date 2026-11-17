#!/usr/bin/env python3
"""
compare_xai_methods.py -- inter-method agreement, Vanilla Gradient vs
Integrated Gradients.

Replaces the previous script, which had two defects: a hardcoded
three-case list that silently ignored --num-cases (slicing three items by
thirty yields three), and a dependency on GradCAM3D, whose output is
withdrawn. Cases are now drawn from the fold validation split, and
Grad-CAM is not constructed anywhere in this file.

    python scripts/compare_xai_methods.py \
        --checkpoint models/segmentation/unet3d_cv_fold0.pth \
        --cache data/brats_dataset/cache_128 \
        --folds data/brats_dataset/folds_seed42_3fold.json \
        --fold 0 --cases 15 \
        --out results/metrics/xai_agreement.json

Resumable: each case is appended to a JSONL sidecar and fsynced before
the next begins, so an interruption costs at most the case in flight.
Rerun the identical command to continue.

Cost is dominated by Integrated Gradients at roughly 178 s per case, so
15 cases is about 50 minutes.
"""

import os

NUM_THREADS = int(os.environ.get("XAI_NUM_THREADS", "8"))
os.environ.setdefault("OMP_NUM_THREADS", str(NUM_THREADS))
os.environ.setdefault("MKL_NUM_THREADS", str(NUM_THREADS))

import argparse
import json
import statistics
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_loader import (load_trained_model, load_fold_split,
                          load_gradient_methods)
from evaluate_cross_population import (load_case, collect, remap_decathlon,
                                       Progress, bootstrap_ci, sha256_of,
                                       THRESHOLD)

METHODS = ("vanilla_gradient", "integrated_gradients")


def as_array(out):
    if isinstance(out, (tuple, list)):
        out = out[0]
    if torch.is_tensor(out):
        return out.detach().cpu().numpy()
    return np.asarray(out)


def degenerate(a):
    return bool(float(a.max()) == float(a.min()))


def pearson(a, b):
    """Undefined if either map is constant; report None rather than 0."""
    u = np.asarray(a, dtype=np.float64).ravel()
    v = np.asarray(b, dtype=np.float64).ravel()
    if u.shape != v.shape or u.std() == 0 or v.std() == 0:
        return None
    return float(np.corrcoef(u, v)[0, 1])


def summarise(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    return {"n": len(vals), "mean": round(statistics.mean(vals), 4),
            "median": round(statistics.median(vals), 4),
            "sd": round(statistics.stdev(vals), 4) if len(vals) > 1 else 0.0,
            "min": round(min(vals), 4), "max": round(max(vals), 4),
            "mean_95ci": bootstrap_ci(vals) if len(vals) > 1 else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint",
                    default="models/segmentation/unet3d_cv_fold0.pth")
    ap.add_argument("--cache", default="data/brats_dataset/cache_128")
    ap.add_argument("--folds",
                    default="data/brats_dataset/folds_seed42_3fold.json")
    ap.add_argument("--fold", type=int, default=0,
                    help="default 0, matching Sections 4.4 and 4.6")
    ap.add_argument("--cases", type=int, default=15,
                    help="number of validation cases to use")
    ap.add_argument("--target-class", type=int, default=1)
    ap.add_argument("--ig-steps", type=int, default=20)
    ap.add_argument("--threads", type=int, default=NUM_THREADS)
    ap.add_argument("--project-root")
    ap.add_argument("--progress")
    ap.add_argument("--restart", action="store_true")
    ap.add_argument("--out", default="results/metrics/xai_agreement.json")
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    provenance = {"fold": args.fold, "target_class": args.target_class,
                  "threshold": THRESHOLD}
    model, meta = load_trained_model(args.checkpoint,
                                     project_root=args.project_root,
                                     provenance=provenance)

    # Cases come from the fold's validation split, never a fixed list.
    _, val_ids, split = load_fold_split(args.folds, args.fold, provenance)
    paths = collect(args.cache, val_ids)
    if not paths:
        sys.exit("no cases found for fold %d under %s" % (args.fold, args.cache))
    if args.cases > len(paths):
        print("WARNING: %d cases requested but the fold %d validation split "
              "holds %d; using all of them."
              % (args.cases, args.fold, len(paths)), file=sys.stderr)
    paths = paths[:args.cases]
    print("fold %d validation split: %d cases available, using %d"
          % (args.fold, split["n_val"], len(paths)), file=sys.stderr)

    fns, impl = load_gradient_methods(model, ig_steps=args.ig_steps,
                                      project_root=args.project_root,
                                      provenance=provenance)

    fingerprint = {
        "checkpoint": {"path": os.path.abspath(args.checkpoint),
                       "sha256": sha256_of(args.checkpoint)},
        "folds": os.path.abspath(args.folds), "fold": args.fold,
        "cache": os.path.abspath(args.cache),
        "target_class": args.target_class, "ig_steps": args.ig_steps,
        "methods": list(METHODS),
    }
    progress_path = args.progress or (os.path.splitext(args.out)[0]
                                      + ".progress.jsonl")
    progress = Progress(progress_path, fingerprint, restart=args.restart)
    provenance["progress_file"] = progress_path

    todo = [p for p in paths
            if not progress.has("agreement", args.fold, os.path.basename(p))]
    if len(todo) < len(paths):
        print("resuming: %d of %d already done" % (len(paths) - len(todo),
                                                   len(paths)), file=sys.stderr)

    for n, path in enumerate(todo, 1):
        case = os.path.basename(path)
        img, _ = load_case(path, remap=remap_decathlon)
        rec = {}
        maps = {}
        for name in METHODS:
            t0 = time.perf_counter()
            arr = as_array(fns[name](img, args.target_class))
            rec[name + "_s"] = round(time.perf_counter() - t0, 2)
            rec[name + "_degenerate"] = degenerate(arr)
            maps[name] = arr
        rec["shape"] = list(maps[METHODS[0]].shape)
        rec["pearson_vg_ig"] = pearson(maps[METHODS[0]], maps[METHODS[1]])
        progress.append("agreement", args.fold, case, rec)
        print("  [%2d/%2d] %-22s r=%s  vg=%.1fs ig=%.1fs"
              % (n, len(todo), case,
                 "n/a" if rec["pearson_vg_ig"] is None
                 else "%.4f" % rec["pearson_vg_ig"],
                 rec["vanilla_gradient_s"], rec["integrated_gradients_s"]),
              file=sys.stderr)

    rows = progress.select("agreement", args.fold)
    progress.close()
    if len(rows) < len(paths):
        sys.exit("INCOMPLETE: %d of %d cases recorded. No aggregate written. "
                 "Rerun the same command to continue from %s."
                 % (len(rows), len(paths), progress_path))

    out = {
        "provenance": provenance,
        "implementations": impl,
        "fingerprint": fingerprint,
        "n_cases": len(rows),
        "agreement_pearson": summarise([r["pearson_vg_ig"] for r in rows]),
        "degenerate": {m: sum(1 for r in rows if r.get(m + "_degenerate"))
                       for m in METHODS},
        "latency_s": {m: summarise([r[m + "_s"] for r in rows])
                      for m in METHODS},
        "undefined_correlations": sum(1 for r in rows
                                      if r["pearson_vg_ig"] is None),
        "note": ("Correlation is undefined where either map is constant and "
                 "is recorded as null rather than zero. Grad-CAM is excluded: "
                 "its results are withdrawn, see Section 4.4."),
        "per_case": rows,
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)

    ag = out["agreement_pearson"]
    print("\n--- Section 4.4 inter-method agreement, n = %d ---" % len(rows),
          file=sys.stderr)
    if ag:
        print("  Pearson r: mean %.4f, median %.4f, range %.4f to %.4f, "
              "95%% CI %s" % (ag["mean"], ag["median"], ag["min"], ag["max"],
                              ag["mean_95ci"]), file=sys.stderr)
    for m in METHODS:
        print("  %-22s degenerate %d of %d"
              % (m, out["degenerate"][m], len(rows)), file=sys.stderr)
    if out["undefined_correlations"]:
        print("  correlation undefined on %d case(s)"
              % out["undefined_correlations"], file=sys.stderr)
    print("wrote %s" % args.out, file=sys.stderr)


if __name__ == "__main__":
    main()
