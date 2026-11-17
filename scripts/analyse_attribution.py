#!/usr/bin/env python3
"""
analyse_attribution.py -- Section 4.4.

Two inference-only analyses, both resumable:

  PART 1, multi-depth Grad-CAM (Reviewer 4: "consider multi-scale feature
  aggregation or higher-resolution layers"). Extracts attribution at
  several network depths on the same cases and reports, per layer, the
  native map resolution before upsampling, peak memory, latency, and
  quantitative measures of spatial concentration and of alignment with
  the reference annotation.

  PART 2, degenerate-map prevalence (Reviewer 4: systematic analysis of
  zero-variance attribution outputs). Runs Grad-CAM over a larger sample
  and reports prevalence together with whether degenerate cases differ
  from the rest in segmentation confidence or tumour volume.

Both parts call the project's own GradCAM3D. Nothing is reimplemented:
Algorithm 2 documents that implementation, so it is the one whose
behaviour belongs in the paper.

    python scripts/analyse_attribution.py \
        --checkpoint models/segmentation/unet3d_cv_fold0.pth \
        --cache data/brats_dataset/cache_128 \
        --folds data/brats_dataset/folds_seed42_3fold.json --fold 0 \
        --layers bottleneck decoder.0 decoder.1 decoder.2 decoder.3 \
        --depth-cases 10 \
        --out results/metrics/attribution_analysis.json

Rerun the identical command after an interruption.

NOTE ON CONCENTRATION: a map can be concentrated and still be in the
wrong place. Concentration measures (effective support, entropy) and
alignment measures (mass inside the reference mask, centre-of-mass
distance, pointing hit) are reported separately for that reason, and a
shallower layer should only be described as better localised if it
improves both.
"""

import os

NUM_THREADS = int(os.environ.get("XAI_NUM_THREADS", "8"))
os.environ.setdefault("OMP_NUM_THREADS", str(NUM_THREADS))
os.environ.setdefault("MKL_NUM_THREADS", str(NUM_THREADS))

import argparse
import json
import math
import statistics
import sys
import threading
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_loader import (load_trained_model, resolve_layer, load_fold_split,
                          load_project_gradcam, release_project_gradcam)
from evaluate_cross_population import (load_case, collect, remap_decathlon,
                                       Progress, sha256_of, bootstrap_ci,
                                       compare, fmt_p, THRESHOLD)

TUMOUR_CLASS = 1


# ------------------------------------------------------------------ #
# Peak memory during a single call
# ------------------------------------------------------------------ #
def _rss_kb():
    try:
        with open("/proc/self/status") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1])
    except OSError:
        pass
    return None


class RssSampler:
    """Sample this process's resident set while a call runs.

    ru_maxrss is a high-water mark that never falls, so it cannot give a
    per-call peak once an earlier layer has already allocated more."""

    def __init__(self, interval=0.05):
        self.interval = interval
        self.peak = 0
        self._stop = threading.Event()
        self._thread = None

    def __enter__(self):
        self.peak = _rss_kb() or 0

        def loop():
            while not self._stop.is_set():
                v = _rss_kb()
                if v and v > self.peak:
                    self.peak = v
                self._stop.wait(self.interval)

        self._thread = threading.Thread(target=loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1.0)
        return False

    @property
    def peak_mib(self):
        return round(self.peak / 1024.0, 1)


# ------------------------------------------------------------------ #
# Attribution-map descriptors
# ------------------------------------------------------------------ #
def map_metrics(cam, gt_mask, pred_mask):
    """Quantify a normalised attribution map.

    Concentration (lower is more concentrated):
      fraction_above_half   voxels above half the map maximum
      normalised_entropy    entropy of the map as a distribution, / log N
    Alignment with the reference (higher is better, except distance):
      mass_in_reference     share of total attribution inside the tumour
      enrichment            mean attribution inside / mean outside
      com_distance_voxels   map centre of mass to tumour centroid
      pointing_hit          map maximum falls inside the tumour
    """
    cam = np.asarray(cam, dtype=np.float64)
    lo, hi = float(cam.min()), float(cam.max())
    out = {"map_min": lo, "map_max": hi, "degenerate": bool(hi == lo)}
    if out["degenerate"]:
        return out

    total = cam.sum()
    out["fraction_above_half"] = float((cam > 0.5 * hi).mean())
    p = cam / total if total > 0 else None
    if p is not None:
        nz = p[p > 0]
        out["normalised_entropy"] = float(
            -(nz * np.log(nz)).sum() / math.log(cam.size))

    if gt_mask is not None and gt_mask.any():
        inside = cam[gt_mask]
        outside = cam[~gt_mask]
        out["mass_in_reference"] = float(inside.sum() / total) if total else None
        mo = float(outside.mean()) if outside.size else 0.0
        out["enrichment"] = (float(inside.mean()) / mo) if mo > 0 else None
        out["reference_volume_fraction"] = float(gt_mask.mean())

        idx = np.array(np.unravel_index(int(np.argmax(cam)), cam.shape))
        out["pointing_hit"] = bool(gt_mask[tuple(idx)])

        grid = np.indices(cam.shape, dtype=np.float64)
        com = np.array([float((g * cam).sum() / total) for g in grid]) \
            if total else None
        cen = np.array([float(c.mean()) for c in np.where(gt_mask)])
        if com is not None:
            out["com_distance_voxels"] = float(np.linalg.norm(com - cen))

    if pred_mask is not None and pred_mask.any():
        out["mass_in_prediction"] = float(cam[pred_mask].sum() / total) \
            if total else None
    return out


def native_resolutions(model, x, layer_names):
    """One forward pass recording the output shape of each named layer.

    Shapes only: storing the tensors is what exhausts memory when several
    layers are hooked at once (Section 3.16)."""
    shapes, handles = {}, []
    for name in layer_names:
        layer = resolve_layer(model, name)

        def hook(_m, _i, out, _n=name):
            t = out[0] if isinstance(out, (tuple, list)) else out
            shapes[_n] = tuple(int(v) for v in t.shape)

        handles.append(layer.register_forward_hook(hook))
    try:
        with torch.no_grad():
            model(x)
    finally:
        for h in handles:
            h.remove()
    return shapes


# ------------------------------------------------------------------ #
def summarise(rows, key):
    vals = [r[key] for r in rows if r.get(key) is not None]
    if not vals:
        return None
    if isinstance(vals[0], bool):
        return {"n": len(vals), "true": sum(vals),
                "rate": round(sum(vals) / float(len(vals)), 4)}
    return {"n": len(vals), "mean": round(statistics.mean(vals), 6),
            "sd": round(statistics.stdev(vals), 6) if len(vals) > 1 else 0.0,
            "median": round(statistics.median(vals), 6),
            "min": round(min(vals), 6), "max": round(max(vals), 6)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--folds", required=True)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--layers", nargs="+",
                    default=["bottleneck", "decoder.0", "decoder.1",
                             "decoder.2", "decoder.3"],
                    help="exact module names, deepest first")
    ap.add_argument("--reference-layer", default="bottleneck",
                    help="layer used for the prevalence analysis")
    ap.add_argument("--depth-cases", type=int, default=10,
                    help="cases for the multi-depth comparison")
    ap.add_argument("--prevalence-cases", type=int,
                    help="cases for the prevalence analysis (default: all)")
    ap.add_argument("--repeats", type=int, default=3,
                    help="timed repetitions per layer per case")
    ap.add_argument("--target-class", type=int, default=TUMOUR_CLASS)
    ap.add_argument("--threads", type=int, default=NUM_THREADS)
    ap.add_argument("--project-root")
    ap.add_argument("--list-layers", action="store_true")
    ap.add_argument("--out", default="results/metrics/attribution_analysis.json")
    ap.add_argument("--progress")
    ap.add_argument("--restart", action="store_true")
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    provenance = {}
    model, meta = load_trained_model(args.checkpoint,
                                     project_root=args.project_root,
                                     provenance=provenance)
    if args.list_layers:
        for n, _ in model.named_modules():
            print(n)
        return

    _, val_ids, _ = load_fold_split(args.folds, args.fold, provenance)
    paths = collect(args.cache, val_ids)
    if not paths:
        sys.exit("no cases found for fold %d under %s" % (args.fold, args.cache))
    depth_paths = paths[:args.depth_cases]
    prev_paths = paths[:args.prevalence_cases] if args.prevalence_cases \
        else paths

    fingerprint = {
        "checkpoint": {"path": os.path.abspath(args.checkpoint),
                       "sha256": sha256_of(args.checkpoint)},
        "folds_file": os.path.abspath(args.folds),
        "fold": args.fold,
        "cache": os.path.abspath(args.cache),
        "layers": list(args.layers),
        "reference_layer": args.reference_layer,
        "target_class": args.target_class,
        "threshold": THRESHOLD,
        "repeats": args.repeats,
    }
    progress_path = args.progress or (os.path.splitext(args.out)[0]
                                      + ".progress.jsonl")
    progress = Progress(progress_path, fingerprint, restart=args.restart)

    # Native resolutions, one forward pass.
    img0, _ = load_case(depth_paths[0], remap=remap_decathlon)
    shapes = native_resolutions(model, img0, args.layers)

    # A name can resolve exactly and still be unusable: nn.ModuleList and
    # other containers are named modules but are never called, so a forward
    # hook on them never fires and Grad-CAM would fail mid-run with an
    # opaque KeyError. Catch it here, before hours of work.
    silent = [n for n in args.layers if n not in shapes]
    if silent:
        usable = sorted(shapes)
        sys.exit(
            "ABORT: no activation was produced by %s during a forward pass.\n"
            "These names resolve to modules that are never called \u2014 "
            "typically containers such as nn.ModuleList, where the children "
            "are called individually. Name a child instead, for example "
            "'decoder.0' rather than 'decoder'.\n"
            "Layers that did produce activations in this run: %s"
            % (silent, usable if usable else "(none)"))

    print("\nnative attribution resolution before upsampling:", file=sys.stderr)
    for name in args.layers:
        print("  %-14s %s" % (name, shapes.get(name)), file=sys.stderr)

    # ---------------- PART 1: multi-depth ----------------
    print("\n=== Part 1: Grad-CAM at %d depths over %d case(s) ==="
          % (len(args.layers), len(depth_paths)), file=sys.stderr)
    expected = len(args.layers) * len(depth_paths) + len(prev_paths)

    for name in args.layers:
        cohort = "depth:%s" % name
        todo = [p for p in depth_paths
                if not progress.has(cohort, args.fold, os.path.basename(p))]
        if not todo:
            print("  %s: already complete" % name, file=sys.stderr)
            continue
        cam_obj = load_project_gradcam(model, name,
                                       project_root=args.project_root)
        for i, path in enumerate(todo, 1):
            case = os.path.basename(path)
            img, lab = load_case(path, remap=remap_decathlon)
            gt = (lab.numpy() == args.target_class) if lab is not None else None
            # Not under torch.no_grad(): the explainer's forward hooks are
            # live here, and a hook that registers a tensor hook on its
            # output fails on a tensor that does not require grad.
            # Detaching immediately frees the graph.
            prob = torch.softmax(model(img), dim=1)[0, args.target_class] \
                .detach()
            pred = prob.numpy() >= THRESHOLD

            cam_obj.generate_cam(img, args.target_class)   # warm-up
            times, cam = [], None
            with RssSampler() as sampler:
                for _ in range(args.repeats):
                    t0 = time.perf_counter()
                    cam = cam_obj.generate_cam(img, args.target_class)
                    times.append(time.perf_counter() - t0)
            cam = np.asarray(cam.detach().cpu().numpy()
                             if torch.is_tensor(cam) else cam)

            rec = map_metrics(cam, gt, pred)
            rec.update({"layer": name,
                        "native_shape": list(shapes.get(name) or []),
                        "latency_s": round(statistics.median(times), 4),
                        "latency_sd_s": round(statistics.stdev(times), 4)
                        if len(times) > 1 else 0.0,
                        "peak_rss_MiB": sampler.peak_mib})
            progress.append(cohort, args.fold, case, rec)
            if i % 5 == 0 or i == len(todo):
                print("  %-14s %d/%d" % (name, i, len(todo)), file=sys.stderr)
        release_project_gradcam(model, cam_obj)
        del cam_obj

    # ---------------- PART 2: prevalence ----------------
    print("\n=== Part 2: degenerate prevalence over %d case(s) ==="
          % len(prev_paths), file=sys.stderr)
    cohort = "prevalence"
    todo = [p for p in prev_paths
            if not progress.has(cohort, args.fold, os.path.basename(p))]
    if todo:
        cam_obj = load_project_gradcam(model, args.reference_layer,
                                       project_root=args.project_root)
        for i, path in enumerate(todo, 1):
            case = os.path.basename(path)
            img, lab = load_case(path, remap=remap_decathlon)
            gt = (lab.numpy() == args.target_class) if lab is not None else None
            # Deliberately NOT under torch.no_grad(): the explainer's
            # forward hooks are live at this point, and a hook that
            # registers a tensor hook on its output fails on a tensor that
            # does not require grad. Detaching immediately frees the graph.
            prob = torch.softmax(model(img), dim=1)[0, args.target_class] \
                .detach().numpy()
            pred = prob >= THRESHOLD

            cam = cam_obj.generate_cam(img, args.target_class)
            cam = np.asarray(cam.detach().cpu().numpy()
                             if torch.is_tensor(cam) else cam)
            rec = map_metrics(cam, gt, pred)

            tp = float(np.logical_and(pred, gt).sum()) if gt is not None else 0.0
            denom = float(pred.sum() + (gt.sum() if gt is not None else 0))
            rec.update({
                "layer": args.reference_layer,
                "dice": (2 * tp / denom) if denom > 0 else None,
                "reference_volume_voxels": int(gt.sum()) if gt is not None else None,
                "predicted_volume_voxels": int(pred.sum()),
                "mean_prob_in_prediction": float(prob[pred].mean())
                if pred.any() else 0.0,
                "max_prob": float(prob.max()),
            })
            progress.append(cohort, args.fold, case, rec)
            if i % 10 == 0 or i == len(todo):
                print("  %d/%d" % (i, len(todo)), file=sys.stderr)
        release_project_gradcam(model, cam_obj)
        del cam_obj

    recorded = len(progress.rows)
    progress.close()
    if recorded < expected:
        sys.exit("INCOMPLETE: %d of %d records. No summary written. Rerun the "
                 "same command to continue from %s."
                 % (recorded, expected, progress_path))

    # ---------------- summaries ----------------
    depth = {}
    for name in args.layers:
        rows = progress.select("depth:%s" % name)
        if not rows:
            continue
        depth[name] = {
            "native_shape": rows[0].get("native_shape"),
            "n_cases": len(rows),
            "degenerate": summarise(rows, "degenerate"),
            "latency_s": summarise(rows, "latency_s"),
            "peak_rss_MiB": summarise(rows, "peak_rss_MiB"),
            "fraction_above_half": summarise(rows, "fraction_above_half"),
            "normalised_entropy": summarise(rows, "normalised_entropy"),
            "mass_in_reference": summarise(rows, "mass_in_reference"),
            "enrichment": summarise(rows, "enrichment"),
            "com_distance_voxels": summarise(rows, "com_distance_voxels"),
            "pointing_hit": summarise(rows, "pointing_hit"),
        }

    prev_rows = progress.select("prevalence")
    degen = [r for r in prev_rows if r.get("degenerate")]
    intact = [r for r in prev_rows if not r.get("degenerate")]
    prevalence = {
        "layer": args.reference_layer,
        "n_cases": len(prev_rows),
        "n_degenerate": len(degen),
        "prevalence": round(len(degen) / float(len(prev_rows)), 4)
        if prev_rows else None,
        "degenerate_cases": [r["case"] for r in degen],
    }
    comparisons = {}
    if degen and intact and len(degen) > 1:
        for key in ("dice", "reference_volume_voxels",
                    "predicted_volume_voxels", "mean_prob_in_prediction",
                    "max_prob"):
            a = [r[key] for r in intact if r.get(key) is not None]
            b = [r[key] for r in degen if r.get(key) is not None]
            if len(a) > 1 and len(b) > 1:
                comparisons[key] = compare(key, a, b, "intact", "degenerate")
    elif degen:
        prevalence["note"] = (
            "Too few degenerate cases (%d) to compare against the rest. "
            "Report the prevalence and list the cases; do not compute a "
            "correlation from this many." % len(degen))
    else:
        prevalence["note"] = (
            "No degenerate maps in this sample. Report the prevalence as "
            "zero of %d at this layer for this model, and note that "
            "degeneracy observed under an earlier model is a property of "
            "the fitted model rather than of the case."
            % len(prev_rows))

    out = {"provenance": provenance, "fingerprint": fingerprint,
           "checkpoint_meta": meta, "progress_file": progress_path,
           "native_resolutions": {k: list(v) for k, v in shapes.items()},
           "multi_depth": depth,
           "degenerate_prevalence": prevalence,
           "degenerate_vs_intact": comparisons,
           "per_case": {"multi_depth": [r for r in progress.rows
                                        if r["cohort"].startswith("depth:")],
                        "prevalence": prev_rows}}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)

    print("\n--- Section 4.4, attribution by depth ---", file=sys.stderr)
    print("%-14s %-16s %9s %9s %9s %9s %9s"
          % ("layer", "native shape", "latency", "RSS MiB", "supp>0.5",
             "mass_in", "com dist"), file=sys.stderr)
    for name, d in depth.items():
        shape = "x".join(str(v) for v in (d["native_shape"] or [])[-3:])
        def g(k):
            return d[k]["median"] if d.get(k) else float("nan")
        print("%-14s %-16s %9.2f %9.1f %9.4f %9.4f %9.2f"
              % (name, shape, g("latency_s"), g("peak_rss_MiB"),
                 g("fraction_above_half"), g("mass_in_reference"),
                 g("com_distance_voxels")), file=sys.stderr)

    print("\n--- degenerate prevalence at %s ---" % args.reference_layer,
          file=sys.stderr)
    print("  %d of %d cases (%s)"
          % (prevalence["n_degenerate"], prevalence["n_cases"],
             prevalence["prevalence"]), file=sys.stderr)
    for key, c in comparisons.items():
        print("  %-26s intact %.4f vs degenerate %.4f  diff %.4f %s  p=%s"
              % (key, c["intact"]["mean"], c["degenerate"]["mean"],
                 c["mean_difference"], c["mean_difference_95ci"],
                 fmt_p(c["mann_whitney"]["p_two_sided"])), file=sys.stderr)
    if "note" in prevalence:
        print("  %s" % prevalence["note"], file=sys.stderr)
    print("\nwrote %s" % args.out, file=sys.stderr)


if __name__ == "__main__":
    main()
