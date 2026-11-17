#!/usr/bin/env python3
"""
ablate_explainer_target.py -- attribute the change in VG/IG agreement to a
specific code change, rather than reporting that a number moved.

    python scripts/ablate_explainer_target.py \
        --checkpoint models/segmentation/unet3d_cv_fold0.pth \
        --cases 15 \
        --out results/metrics/explainer_target_ablation.json

WHAT HAPPENED
-------------
results/metrics/xai_comparison_n15.json (2026-09-11) records a mean
Pearson agreement between Vanilla Gradient and Integrated Gradients of
0.3200. Re-running the same 15 cases today gives 0.6221 -- every case
changed, several by more than 0.3. The manuscript reports 0.32 and builds
a "stark divergence" narrative on it.

The cause is an uncommitted change to src/xai_explainers/gradient_methods.py
dated 2026-10-16, the day after the manuscript was finalised. Two things
were altered:

  (1) a forward pass through `forward_logits()` when the model exposes it,
      to avoid taking gradients through a saturating softmax; and
  (2) restriction of the backward target to voxels the model predicted as
      the target class, instead of summing the target-class output over the
      entire volume.

Change (1) is INERT for this pipeline. load_trained_model() returns a bare
UNet3D, whose forward() already returns raw logits -- the softmax lives in
the BrainTumorSegmentationModel wrapper, which is what forward_logits() was
added to and which this analysis never constructs. So `hasattr(model,
'forward_logits')` is False here and both the old and new code differentiate
logits. Change (1) affects the Streamlit application, not these numbers.

That leaves change (2) as the sole candidate. This script tests it directly
rather than reasoning about it: it runs both target definitions over the
same cases with everything else held fixed.

HOW TO READ THE RESULT
----------------------
The run is only informative if it reproduces both known values:

  whole_volume -> mean r ~ 0.3200   (matches xai_comparison_n15.json)
  predicted    -> mean r ~ 0.6221   (matches today's run on those cases)

Both are checked automatically and reported as `reproduces_recorded`. If
either fails, this harness is not faithful to the code paths it claims to
compare and its conclusion must not be used.

The backward mechanism (torch.autograd.grad) and the normalisation are held
identical across configurations, so the only varying factor is the target
definition. `--include-softmax-axis` additionally runs the softmax variants
to demonstrate that change (1) is inert here; that doubles the runtime and
is not needed for the main conclusion.

Read-only. No checkpoint or cached volume is modified.
"""

import os

NUM_THREADS = int(os.environ.get("XAI_NUM_THREADS", "8"))
os.environ.setdefault("OMP_NUM_THREADS", str(NUM_THREADS))
os.environ.setdefault("MKL_NUM_THREADS", str(NUM_THREADS))

import argparse
import json
import statistics as st
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from model_loader import load_trained_model, load_fold_split
from evaluate_cross_population import load_case, collect, remap_decathlon

TARGET_CLASS = 1
IG_STEPS = 20


# ------------------------------------------------------------------ #
# Parameterised explainers.
#
# These mirror src/xai_explainers/gradient_methods.py exactly, with the two
# changed behaviours exposed as switches. They are reimplemented here rather
# than imported so that both sides of the comparison run through identical
# surrounding code; importing the current module would only give one side.
# ------------------------------------------------------------------ #
def _forward(model, x, use_softmax):
    out = model(x)                      # bare UNet3D -> logits
    if use_softmax:
        out = torch.softmax(out, dim=1)
    return out


def _target(output, masked):
    """The scalar that is differentiated.

    masked=False reproduces the pre-2026-10-16 behaviour: the target-class
    output summed over every voxel, background included.
    masked=True restricts the sum to voxels predicted as the target class.
    """
    if not masked:
        return output[0, TARGET_CLASS].sum(), None
    with torch.no_grad():
        pred_mask = (output.argmax(dim=1) == TARGET_CLASS).float()
    n_pred = float(pred_mask.sum().item())
    if n_pred > 0:
        return (output[0, TARGET_CLASS] * pred_mask[0]).sum(), n_pred
    # same fallback as the current code
    return output[0, TARGET_CLASS].sum(), 0.0


def _normalise(vol):
    lo, hi = vol.min(), vol.max()
    if hi > lo:
        return ((vol - lo) / (hi - lo)).detach().cpu().numpy()
    return np.zeros_like(vol.detach().cpu().numpy())


def vanilla_gradient(model, img, masked, use_softmax):
    model.eval()
    x = img.detach().clone().requires_grad_(True)
    output = _forward(model, x, use_softmax)
    target, n_pred = _target(output, masked)
    grads = torch.autograd.grad(target, x, allow_unused=True)[0]
    if grads is None:
        return None, n_pred
    return _normalise(grads[0].abs().mean(dim=0)), n_pred


def integrated_gradients(model, img, masked, use_softmax, steps=IG_STEPS):
    model.eval()
    inp = img.detach()
    total = torch.zeros_like(inp)
    n_pred_last = None
    for step in range(1, steps + 1):
        alpha = step / steps
        interp = (alpha * inp).requires_grad_(True)     # baseline is zeros
        output = _forward(model, interp, use_softmax)
        target, n_pred_last = _target(output, masked)
        g = torch.autograd.grad(target, interp, allow_unused=True)[0]
        if g is not None:
            total += g.detach()
    attribution = (inp * (total / steps))[0].abs().mean(dim=0)
    return _normalise(attribution), n_pred_last


def pearson(a, b):
    """Pearson correlation of two flattened maps, or None if either is
    constant -- matching the convention in compare_xai_methods.py, which
    records an undefined correlation as null rather than zero."""
    if a is None or b is None:
        return None
    if a.std() == 0 or b.std() == 0:
        return None
    return float(np.corrcoef(a.ravel(), b.ravel())[0, 1])


# ------------------------------------------------------------------ #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint",
                    default="models/segmentation/unet3d_cv_fold0.pth")
    ap.add_argument("--cache", default="data/brats_dataset/cache_128")
    ap.add_argument("--folds",
                    default="data/brats_dataset/folds_seed42_3fold.json")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--cases", type=int, default=15,
                    help="use 15 to compare against the recorded values")
    ap.add_argument("--ig-steps", type=int, default=IG_STEPS)
    ap.add_argument("--include-softmax-axis", action="store_true",
                    help="also run the softmax variants (doubles runtime; "
                         "expected to be inert for a bare UNet3D)")
    ap.add_argument("--recorded-whole", type=float, default=0.3200,
                    help="mean r recorded in xai_comparison_n15.json")
    ap.add_argument("--recorded-masked", type=float, default=0.6221,
                    help="mean r on the same 15 cases with the current code")
    ap.add_argument("--tolerance", type=float, default=0.005)
    ap.add_argument("--threads", type=int, default=NUM_THREADS)
    ap.add_argument("--project-root")
    ap.add_argument("--out",
                    default="results/metrics/explainer_target_ablation.json")
    args = ap.parse_args()

    torch.set_num_threads(args.threads)

    provenance = {}
    model, meta = load_trained_model(args.checkpoint,
                                     project_root=args.project_root,
                                     provenance=provenance)
    model.eval()

    has_fl = hasattr(model, "forward_logits")
    print("model type: %s | exposes forward_logits(): %s"
          % (type(model).__name__, has_fl), file=sys.stderr)
    print("  -> change (1) is %s for this pipeline"
          % ("ACTIVE" if has_fl else "INERT: forward() already returns logits"),
          file=sys.stderr)

    _, val_ids, _ = load_fold_split(args.folds, args.fold)
    paths = collect(args.cache, val_ids)[:args.cases]
    if not paths:
        sys.exit("no cached cases found under %s" % args.cache)
    print("evaluating %d case(s) from fold %d\n" % (len(paths), args.fold),
          file=sys.stderr)

    configs = [("whole_volume", False, False), ("predicted", True, False)]
    if args.include_softmax_axis:
        configs += [("whole_volume_softmax", False, True),
                    ("predicted_softmax", True, True)]

    per_case = {name: {} for name, _, _ in configs}
    t0 = time.time()
    for i, p in enumerate(paths, 1):
        case = os.path.basename(p)
        img, _ = load_case(p, remap=remap_decathlon)
        for name, masked, use_softmax in configs:
            vg, n_pred = vanilla_gradient(model, img, masked, use_softmax)
            ig, _ = integrated_gradients(model, img, masked, use_softmax,
                                         args.ig_steps)
            r = pearson(vg, ig)
            per_case[name][case] = {
                "pearson_vg_ig": r,
                "vg_degenerate": vg is None or float(np.std(vg)) == 0.0,
                "ig_degenerate": ig is None or float(np.std(ig)) == 0.0,
                "predicted_voxels": n_pred,
            }
        rs = "  ".join("%s=%s" % (n, ("%.4f" % per_case[n][case]["pearson_vg_ig"])
                                  if per_case[n][case]["pearson_vg_ig"] is not None
                                  else "undef")
                       for n, _, _ in configs)
        print("  [%2d/%d] %-16s %s" % (i, len(paths), case, rs),
              file=sys.stderr)

    summary = {}
    for name, _, _ in configs:
        vals = [v["pearson_vg_ig"] for v in per_case[name].values()
                if v["pearson_vg_ig"] is not None]
        summary[name] = {
            "n": len(vals),
            "n_undefined": len(per_case[name]) - len(vals),
            "mean": round(st.mean(vals), 4) if vals else None,
            "median": round(st.median(vals), 4) if vals else None,
            "sd": round(st.stdev(vals), 4) if len(vals) > 1 else 0.0,
            "min": round(min(vals), 4) if vals else None,
            "max": round(max(vals), 4) if vals else None,
            "degenerate_vg": sum(1 for v in per_case[name].values()
                                 if v["vg_degenerate"]),
            "degenerate_ig": sum(1 for v in per_case[name].values()
                                 if v["ig_degenerate"]),
        }

    checks = {
        "whole_volume_reproduces_recorded": (
            summary["whole_volume"]["mean"] is not None and
            abs(summary["whole_volume"]["mean"] - args.recorded_whole)
            <= args.tolerance),
        "predicted_reproduces_recorded": (
            summary["predicted"]["mean"] is not None and
            abs(summary["predicted"]["mean"] - args.recorded_masked)
            <= args.tolerance),
        "recorded_whole": args.recorded_whole,
        "recorded_masked": args.recorded_masked,
        "tolerance": args.tolerance,
    }
    reproduces = (checks["whole_volume_reproduces_recorded"] and
                  checks["predicted_reproduces_recorded"])

    # Does the effect scale with tumour size, as the Grad-CAM note claimed?
    size_effect = None
    sizes, deltas = [], []
    for case in per_case["predicted"]:
        a = per_case["whole_volume"][case]["pearson_vg_ig"]
        b = per_case["predicted"][case]["pearson_vg_ig"]
        n = per_case["predicted"][case]["predicted_voxels"]
        if a is not None and b is not None and n:
            sizes.append(n); deltas.append(b - a)
    if len(sizes) > 2:
        size_effect = {
            "n": len(sizes),
            "pearson_predvoxels_vs_delta":
                float(np.corrcoef(sizes, deltas)[0, 1]),
            "note": ("Correlation between the number of predicted tumour "
                     "voxels and the change in agreement. The Grad-CAM note "
                     "in attention_visualizer.py reported r = -0.66 between "
                     "attention magnitude and tumour size under the "
                     "whole-volume target; a positive correlation here would "
                     "mean the fix helps larger tumours most, which is the "
                     "same mechanism seen from the other direction."),
        }

    out = {
        "provenance": provenance,
        "checkpoint_meta": meta,
        "fold": args.fold,
        "n_cases": len(paths),
        "ig_steps": args.ig_steps,
        "model_type": type(model).__name__,
        "model_exposes_forward_logits": has_fl,
        "change_1_softmax_status": (
            "ACTIVE" if has_fl else
            "INERT -- forward() already returns logits for this model class"),
        "configurations": {
            "whole_volume": "target = output[0,1].sum() over all voxels "
                            "(pre-2026-10-16 behaviour)",
            "predicted": "target = (output[0,1] * predicted_mask).sum() "
                         "(current behaviour)",
        },
        "summary": summary,
        "reproduction_checks": checks,
        "attribution_sound": reproduces,
        "tumour_size_effect": size_effect,
        "per_case": per_case,
        "elapsed_min": round((time.time() - t0) / 60.0, 1),
        "interpretation": (
            "If both reproduction checks pass, the entire change in reported "
            "VG/IG agreement is attributable to the backward-target "
            "definition, and the manuscript's 0.32 is an artefact of summing "
            "the target-class output over the whole volume. If they fail, "
            "this harness does not faithfully reproduce either code path and "
            "no conclusion should be drawn from it."),
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))

    print("\n--- summary, n=%d cases ---" % len(paths), file=sys.stderr)
    print("%-22s %6s %9s %9s %8s %8s"
          % ("configuration", "n", "mean r", "median r", "min", "max"),
          file=sys.stderr)
    for name, _, _ in configs:
        s = summary[name]
        print("%-22s %6d %9s %9s %8s %8s"
              % (name, s["n"], s["mean"], s["median"], s["min"], s["max"]),
              file=sys.stderr)
    print("\nreproduction checks", file=sys.stderr)
    print("  whole_volume  %.4f vs recorded %.4f   %s"
          % (summary["whole_volume"]["mean"], args.recorded_whole,
             "OK" if checks["whole_volume_reproduces_recorded"] else "MISMATCH"),
          file=sys.stderr)
    print("  predicted     %.4f vs recorded %.4f   %s"
          % (summary["predicted"]["mean"], args.recorded_masked,
             "OK" if checks["predicted_reproduces_recorded"] else "MISMATCH"),
          file=sys.stderr)
    if size_effect:
        print("\n  predicted-voxels vs change in agreement: r = %+.3f"
              % size_effect["pearson_predvoxels_vs_delta"], file=sys.stderr)
    print("\nelapsed %.1f min -> %s" % (out["elapsed_min"], args.out),
          file=sys.stderr)

    if reproduces:
        print("\nATTRIBUTION SOUND: the change in agreement is explained by "
              "the backward-target definition alone.", file=sys.stderr)
    else:
        print("\nATTRIBUTION NOT ESTABLISHED: this harness did not reproduce "
              "one or both recorded values. Do not use these figures; "
              "the reimplementation differs from the code it models.",
              file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
