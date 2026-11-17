#!/usr/bin/env python3
"""
measure_resources.py -- Section 4.6 resource measurements.

Measures, on an idle machine:
  * model size (parameter count, checkpoint bytes on disk)
  * single-case inference latency
  * per-method explanation latency: Grad-CAM, Vanilla Gradient,
    Integrated Gradients (m = 20)
  * peak RSS for inference and for each explanation method
  * CPU clock at the start and end of the run, so the reported figures
    can be stated as boost-clock or sustained-clock measurements

Run AFTER training finishes and with nothing else on the machine:

    cd ~/neurosurgery-xai-planning && source .venv/bin/activate
    python scripts/measure_resources.py \
        --checkpoint results/checkpoints/cv_fold0_best.pt \
        --cache data/brats_dataset/cache_128 \
        --folds data/brats_dataset/folds_seed42_3fold.json \
        --fold 0 \
        --gradcam-layer unet.bottleneck \
        --cases 10 --repeats 5 \
        --out results/metrics/resources.json

Nothing here writes to the model or the cache; it is read-only.

--------------------------------------------------------------------
Model loading lives in model_loader.py, which imports the project's own
UNet3D from ai_models.segmentation, reads the layer widths from the
checkpoint, and aborts if any key fails to load. There is no fallback
architecture: a measurement taken against the wrong network is worse
than no measurement. Keep model_loader.py in this directory.

load_case() below is the one remaining project-specific part; check it
against your cache format.
"""

import os

# Thread count must be pinned before torch is imported, and must match the
# 8 threads used for training, or the reported latency will not correspond
# to the configuration described in the paper.
NUM_THREADS = int(os.environ.get("XAI_NUM_THREADS", "8"))
os.environ.setdefault("OMP_NUM_THREADS", str(NUM_THREADS))
os.environ.setdefault("MKL_NUM_THREADS", str(NUM_THREADS))

import argparse
import glob
import json
import platform
import resource
import statistics
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_loader import (load_trained_model, resolve_layer,
                          load_fold_split, load_project_explainers)


# ------------------------------------------------------------------ #
# Case loading -- check against your cache format
# ------------------------------------------------------------------ #
def load_case(path, device, provenance):
    """Load one preprocessed case from the float16 cache as (1,C,D,H,W)."""
    if path.endswith(".npz"):
        with np.load(path) as z:
            key = "image" if "image" in z else list(z.keys())[0]
            arr = z[key]
            provenance.setdefault("cache_key", key)
    elif path.endswith(".npy"):
        arr = np.load(path)
    elif path.endswith((".pt", ".pth")):
        obj = torch.load(path, map_location="cpu")
        arr = obj["image"] if isinstance(obj, dict) else obj
        arr = arr.numpy() if torch.is_tensor(arr) else arr
    else:
        raise ValueError("unrecognised cache format: %s" % path)

    t = torch.as_tensor(np.ascontiguousarray(arr)).float()
    while t.ndim < 5:
        t = t.unsqueeze(0)
    provenance.setdefault("input_shape", list(t.shape))
    return t.to(device)


# ------------------------------------------------------------------ #
# Attribution methods
#
# All three come from the project's own code via load_project_explainers.
# Nothing is reimplemented here: the manuscript documents those
# implementations, so they are the ones whose latency belongs in the paper.
# ------------------------------------------------------------------ #
def as_array(out):
    """Normalise whatever an explainer returns to a numpy array."""
    if isinstance(out, (tuple, list)):
        out = out[0]
    if torch.is_tensor(out):
        return out.detach().cpu().numpy()
    return np.asarray(out)


def wrap(fn, cls):
    """Time one explainer call and flag a degenerate (constant) map.

    All three return [D,H,W] arrays min-max normalised to [0,1], so a
    constant map means the normalisation had nothing to scale."""
    def run(x):
        arr = as_array(fn(x, cls))
        return arr, bool(float(arr.max()) == float(arr.min()))
    return run


# ------------------------------------------------------------------ #
# Harness
# ------------------------------------------------------------------ #
def peak_rss_mib():
    # ru_maxrss is kilobytes on Linux.
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def cpu_mhz():
    try:
        vals = []
        with open("/proc/cpuinfo") as fh:
            for line in fh:
                if line.lower().startswith("cpu mhz"):
                    vals.append(float(line.split(":")[1]))
        return round(sum(vals) / len(vals), 1) if vals else None
    except Exception:
        return None


def load_avg():
    try:
        return os.getloadavg()[0]
    except Exception:
        return None


def preflight(force):
    la = load_avg()
    print("1-minute load average: %s" % la, file=sys.stderr)
    if la is not None and la > 0.7 and not force:
        sys.exit(
            "Load average is %.2f. These figures go in the paper as "
            "measurements on an idle machine, so run this with nothing else "
            "active, or pass --force if you accept the contention." % la)


def timed(fn, warmup, repeats):
    """Return (per-call seconds list, extra) with warm-up discarded."""
    extra = None
    for _ in range(warmup):
        extra = fn()
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        extra = fn()
        times.append(time.perf_counter() - t0)
    return times, extra


def summarise(times):
    return {
        "n": len(times),
        "mean_s": round(statistics.mean(times), 4),
        "sd_s": round(statistics.stdev(times), 4) if len(times) > 1 else 0.0,
        "median_s": round(statistics.median(times), 4),
        "min_s": round(min(times), 4),
        "max_s": round(max(times), 4),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--cache", required=True, help="preprocessed cache directory")
    ap.add_argument("--folds", help="folds JSON, to draw validation cases from")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--gradcam-layer", help="exact module name for Grad-CAM")
    ap.add_argument("--list-layers", action="store_true",
                    help="print module names and exit")
    ap.add_argument("--target-class", type=int, default=1,
                    help="class index to explain (1 = tumour)")
    ap.add_argument("--cases", type=int, default=10)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--ig-steps", type=int, default=20,
                    help="passed to IntegratedGradients at construction")
    ap.add_argument("--threads", type=int, default=NUM_THREADS)
    ap.add_argument("--methods", nargs="+",
                    choices=["grad_cam", "vanilla_gradient",
                             "integrated_gradients"],
                    help="time only these methods. Vanilla Gradient and "
                         "Integrated Gradients take no layer argument and "
                         "are unaffected by Grad-CAM targeting, so after a "
                         "Grad-CAM fix only grad_cam need be re-timed.")
    ap.add_argument("--project-root",
                    help="project root containing src/ai_models (default: cwd)")
    ap.add_argument("--features", type=int, nargs="+",
                    help="override layer widths; normally read from the "
                         "checkpoint, so leave unset")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--out", default="results/metrics/resources.json")
    args = ap.parse_args()

    preflight(args.force)
    mhz_start = cpu_mhz()
    torch.set_num_threads(args.threads)
    torch.set_grad_enabled(True)
    device = torch.device("cpu")

    provenance = {"cwd": os.getcwd(), "checkpoint": args.checkpoint}
    model, meta = load_trained_model(
        args.checkpoint, project_root=args.project_root,
        features=args.features, provenance=provenance)

    if args.list_layers:
        for n, _ in model.named_modules():
            print(n)
        return

    if not args.gradcam_layer:
        sys.exit("--gradcam-layer is required (use --list-layers to find it)")
    layer = resolve_layer(model, args.gradcam_layer)

    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    weights_bytes = sum(p.numel() * p.element_size() for p in model.parameters())

    # The checkpoint file may carry optimiser state, in which case its size on
    # disk is roughly three times the weights for Adam and is NOT the figure to
    # quote as model size.
    _ckpt_keys = []
    try:
        _blob = torch.load(args.checkpoint, map_location="cpu")
        if isinstance(_blob, dict):
            _ckpt_keys = sorted(_blob.keys())
        del _blob
    except Exception:
        pass
    has_opt = any("optim" in k for k in _ckpt_keys)

    # Case selection: validation cases of the named fold where available.
    paths = []
    if args.folds and os.path.exists(args.folds):
        _, ids, _ = load_fold_split(args.folds, args.fold, provenance)
        for cid in ids:
            hits = glob.glob(os.path.join(args.cache, "*%s*" % cid))
            if hits:
                paths.append(sorted(hits)[0])
        provenance["case_source"] = "fold %d validation split" % args.fold
    if not paths:
        paths = sorted(glob.glob(os.path.join(args.cache, "*.np[yz]")))
        provenance["case_source"] = "cache directory listing (FALLBACK)"
    if not paths:
        sys.exit("no cases found under %s" % args.cache)
    paths = paths[:args.cases]

    # Explainers are constructed once, before any timing begins: building
    # them and registering their hooks is one-time setup, and the figure
    # that belongs in the paper is the per-case cost of an explanation.
    project_fns, impl = load_project_explainers(
        model, args.gradcam_layer, ig_steps=args.ig_steps,
        project_root=args.project_root, provenance=provenance)
    if args.methods:
        missing = [m for m in args.methods if m not in project_fns]
        if missing:
            sys.exit("unknown method(s): %s" % missing)
        project_fns = {k: v for k, v in project_fns.items()
                       if k in args.methods}
        impl = {k: v for k, v in impl.items() if k in args.methods}
        provenance["methods_timed"] = list(project_fns)
        provenance["methods_note"] = (
            "Only %s were timed in this run. Figures for any other method "
            "come from a different run and must be labelled as such."
            % ", ".join(project_fns))
        print("timing only: %s" % ", ".join(project_fns), file=sys.stderr)
    methods = {name: wrap(fn, args.target_class)
               for name, fn in project_fns.items()}

    results = {"inference": {}, "explanation": {m: {} for m in methods}}
    per_case = []
    degenerate_counts = {m: 0 for m in methods}

    print("timing %d cases x %d repeats (+%d warm-up)"
          % (len(paths), args.repeats, args.warmup), file=sys.stderr)

    for idx, path in enumerate(paths):
        x = load_case(path, device, provenance)
        rec = {"case": os.path.basename(path)}

        # The explainers are already constructed, so their forward hooks are
        # live. A hook that registers a tensor hook on its output fails on a
        # tensor that does not require grad, so the inference timing cannot
        # run under torch.no_grad(). Detaching immediately frees the graph;
        # the timing still covers a forward pass only.
        t, _ = timed(lambda: model(x).detach(), args.warmup, args.repeats)
        rec["inference_s"] = summarise(t)

        for name, fn in methods.items():
            t, out = timed(lambda: fn(x), args.warmup, args.repeats)
            rec[name + "_s"] = summarise(t)
            if out is not None and out[1]:
                degenerate_counts[name] += 1
                rec.setdefault("degenerate", []).append(name)

        per_case.append(rec)
        print("  [%2d/%2d] %s  inference %.2fs  grad-cam %.2fs"
              % (idx + 1, len(paths), rec["case"],
                 rec["inference_s"]["median_s"], rec["grad_cam_s"]["median_s"]),
              file=sys.stderr)

    def pool(key):
        return summarise([c[key]["median_s"] for c in per_case])

    results["inference"] = pool("inference_s")
    for name in methods:
        results["explanation"][name] = pool(name + "_s")
        results["explanation"][name]["degenerate_cases"] = degenerate_counts[name]

    out = {
        "provenance": provenance,
        "hardware": {
            "processor": platform.processor() or platform.machine(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "threads": args.threads,
            "cpu_mhz_at_start": mhz_start,
            "cpu_mhz_at_end": cpu_mhz(),
            "load_average_1min": load_avg(),
        },
        "model": {
            "checkpoint_meta": meta,
            "parameters_total": n_params,
            "parameters_trainable": n_trainable,
            "weights_only_bytes": weights_bytes,
            "weights_only_MiB": round(weights_bytes / 1048576.0, 2),
            "checkpoint_file_bytes": os.path.getsize(args.checkpoint),
            "checkpoint_file_MiB": round(
                os.path.getsize(args.checkpoint) / 1048576.0, 2),
            "checkpoint_contains_optimiser_state": has_opt,
            "checkpoint_keys": _ckpt_keys,
            "size_note": ("Quote weights_only_MiB as the model size. The "
                          "checkpoint file also stores optimiser state and is "
                          "about three times larger for Adam."
                          if has_opt else
                          "Checkpoint file appears to hold weights only."),
        },
        "implementations": impl,
        "config": {
            "cases": len(paths),
            "repeats": args.repeats,
            "warmup": args.warmup,
            "ig_steps": args.ig_steps,
            "target_class": args.target_class,
            "gradcam_layer": args.gradcam_layer,
        },
        "results": results,
        "peak_rss_MiB_this_process": round(peak_rss_mib(), 1),
        "per_case": per_case,
    }

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)

    print("\n--- Section 4.6 figures ---", file=sys.stderr)
    print("parameters: %s   weights: %.2f MiB   checkpoint file: %.2f MiB%s"
          % (f"{n_params:,}", out["model"]["weights_only_MiB"],
             out["model"]["checkpoint_file_MiB"],
             "  (includes optimiser state)" if has_opt else ""),
          file=sys.stderr)
    print("inference:  %.2f s (median of per-case medians, n=%d)"
          % (results["inference"]["median_s"], len(paths)), file=sys.stderr)
    for name in methods:
        r = results["explanation"][name]
        print("%-22s %.2f s  (SD %.2f, degenerate %d/%d)"
              % (name, r["median_s"], r["sd_s"], r["degenerate_cases"], len(paths)),
              file=sys.stderr)
    print("peak RSS (this process): %.1f MiB" % out["peak_rss_MiB_this_process"],
          file=sys.stderr)
    print("wrote %s" % args.out, file=sys.stderr)


if __name__ == "__main__":
    main()
