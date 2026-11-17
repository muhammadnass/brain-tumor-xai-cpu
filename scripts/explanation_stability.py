#!/usr/bin/env python3
"""
explanation_stability.py -- perturbation stability of Vanilla Gradient and
Integrated Gradients.

Replaces the previous script, which drew from a hardcoded three-case list
and called GradCAM3D. Cases now come from the fold validation split, and
Grad-CAM is not constructed anywhere in this file.

For each case, an attribution map is computed on the unperturbed volume
and again on the volume with additive Gaussian noise at several
magnitudes. Stability is the Pearson correlation between the unperturbed
map and the perturbed one: a method whose explanation survives a small
input perturbation is more trustworthy than one whose does not.

    python scripts/explanation_stability.py \
        --checkpoint models/segmentation/unet3d_cv_fold0.pth \
        --cache data/brats_dataset/cache_128 \
        --folds data/brats_dataset/folds_seed42_3fold.json \
        --fold 0 --cases 15 \
        --out results/metrics/xai_stability.json

RESUMABLE PER (case, perturbation), not per case, because the full run is
long: 15 cases at 4 noise levels is 75 map pairs, and Integrated
Gradients alone is roughly 178 s each, so budget about four hours. An
interruption costs at most one perturbation of one case.

If time is short, --cases 8 --sigmas 0.02 0.10 halves it twice over and
still supports a stability claim; say n in the text either way.
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


def pearson(a, b):
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
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--cases", type=int, default=15)
    ap.add_argument("--sigmas", type=float, nargs="+",
                    default=[0.01, 0.05, 0.10, 0.20],
                    help="noise standard deviations, as a fraction of each "
                         "volume's own intensity standard deviation")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--target-class", type=int, default=1)
    ap.add_argument("--ig-steps", type=int, default=20)
    ap.add_argument("--threads", type=int, default=NUM_THREADS)
    ap.add_argument("--project-root")
    ap.add_argument("--progress")
    ap.add_argument("--restart", action="store_true")
    ap.add_argument("--out", default="results/metrics/xai_stability.json")
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    provenance = {"fold": args.fold, "target_class": args.target_class,
                  "threshold": THRESHOLD, "sigmas": args.sigmas,
                  "seed": args.seed}
    model, meta = load_trained_model(args.checkpoint,
                                     project_root=args.project_root,
                                     provenance=provenance)

    _, val_ids, split = load_fold_split(args.folds, args.fold, provenance)
    paths = collect(args.cache, val_ids)
    if not paths:
        sys.exit("no cases found for fold %d under %s" % (args.fold, args.cache))
    paths = paths[:args.cases]
    print("fold %d validation split: %d available, using %d"
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
        "sigmas": list(args.sigmas), "seed": args.seed,
        "methods": list(METHODS),
    }
    progress_path = args.progress or (os.path.splitext(args.out)[0]
                                      + ".progress.jsonl")
    progress = Progress(progress_path, fingerprint, restart=args.restart)
    provenance["progress_file"] = progress_path

    units = [(p, s) for p in paths for s in args.sigmas]
    expected = len(units)
    todo = [(p, s) for p, s in units
            if not progress.has("stability", "%g" % s, os.path.basename(p))]
    if len(todo) < expected:
        print("resuming: %d of %d (case, sigma) pairs already done"
              % (expected - len(todo), expected), file=sys.stderr)

    baseline_cache = {}
    for n, (path, sigma) in enumerate(todo, 1):
        case = os.path.basename(path)
        img, _ = load_case(path, remap=remap_decathlon)

        if case not in baseline_cache:
            base = {}
            for m in METHODS:
                base[m] = as_array(fns[m](img, args.target_class))
            baseline_cache = {case: base}   # one case at a time, bounded memory
        base = baseline_cache[case]

        # Noise scaled to each volume's own intensity spread, so sigma means
        # the same thing across cases with different normalisation.
        g = torch.Generator().manual_seed(args.seed)
        scale = float(img.std())
        noisy = img + torch.randn(img.shape, generator=g) * (sigma * scale)

        rec = {"sigma": sigma}
        for m in METHODS:
            t0 = time.perf_counter()
            pert = as_array(fns[m](noisy, args.target_class))
            rec[m + "_s"] = round(time.perf_counter() - t0, 2)
            rec[m + "_r"] = pearson(base[m], pert)
        progress.append("stability", "%g" % sigma, case, rec)
        print("  [%2d/%2d] %-20s sigma=%-5g  vg r=%s  ig r=%s"
              % (n, len(todo), case, sigma,
                 "n/a" if rec["vanilla_gradient_r"] is None
                 else "%.4f" % rec["vanilla_gradient_r"],
                 "n/a" if rec["integrated_gradients_r"] is None
                 else "%.4f" % rec["integrated_gradients_r"]), file=sys.stderr)

    rows = progress.select("stability")
    progress.close()
    if len(rows) < expected:
        sys.exit("INCOMPLETE: %d of %d (case, sigma) pairs recorded. No "
                 "aggregate written. Rerun the same command to continue from "
                 "%s." % (len(rows), expected, progress_path))

    by_sigma = {}
    for s in args.sigmas:
        sub = [r for r in rows if r.get("sigma") == s]
        by_sigma["%g" % s] = {
            "n_cases": len(sub),
            **{m: summarise([r[m + "_r"] for r in sub]) for m in METHODS}}

    out = {
        "provenance": provenance,
        "implementations": impl,
        "fingerprint": fingerprint,
        "n_cases": len({r["case"] for r in rows}),
        "sigmas": args.sigmas,
        "by_sigma": by_sigma,
        "overall": {m: summarise([r[m + "_r"] for r in rows])
                    for m in METHODS},
        "note": ("Stability is the Pearson correlation between the "
                 "attribution map on the unperturbed volume and on the same "
                 "volume with additive Gaussian noise, sigma expressed as a "
                 "fraction of the volume's own intensity standard deviation. "
                 "Correlation is null where either map is constant. Grad-CAM "
                 "is excluded: its results are withdrawn, see Section 4.4."),
        "per_case": rows,
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)

    print("\n--- Section 4.4 perturbation stability, %d case(s) ---"
          % out["n_cases"], file=sys.stderr)
    print("  %-8s %-28s %-28s" % ("sigma", "vanilla gradient", "integrated "
                                  "gradients"), file=sys.stderr)
    for s in args.sigmas:
        e = by_sigma["%g" % s]
        def f(m):
            v = e[m]
            return "n/a" if not v else "r = %.4f (SD %.4f)" % (v["mean"],
                                                               v["sd"])
        print("  %-8g %-28s %-28s" % (s, f(METHODS[0]), f(METHODS[1])),
              file=sys.stderr)
    print("wrote %s" % args.out, file=sys.stderr)


if __name__ == "__main__":
    main()
