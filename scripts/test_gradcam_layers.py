#!/usr/bin/env python3
"""
test_gradcam_layers.py -- is Grad-CAM recoverable if it is hooked to a live
layer instead of the inert bottleneck?

    python scripts/test_gradcam_layers.py \
        --checkpoint models/segmentation/unet3d_cv_fold0.pth \
        --cases 5 \
        --out results/metrics/gradcam_layer_scan.json

WHY ASK
-------
The manuscript withdraws Grad-CAM, attributing the failure to a layer-name
resolution bug (substring rather than exact matching). That bug was real and
is fixed. But src/xai_explainers/gradient_methods.py still constructs the
explainer like this:

    GradCAM3D(model, target_layers=['unet.bottleneck'])

Grad-CAM forms its map from activations at the target layer weighted by the
gradients flowing through it. The bottleneck is the stage this work has shown
emits at most ~7e-37 -- a denormal constant. Weighting a constant of that
magnitude cannot produce a meaningful map, whatever the resolution logic
does. So the withdrawal may rest on the wrong diagnosis: not a broken
implementation, but a correct implementation pointed at a dead layer.

That matters for the revision. Reviewer 2 notes the quantitative comparison
is "dependent on only two gradient-based methods". If Grad-CAM works at a
live layer, a third method is restored and the bottleneck collapse is shown
to explain the Grad-CAM failure as well -- one cause with two symptoms,
rather than two unrelated defects.

NOTE ON LAYER NAMES
-------------------
load_trained_model() returns a bare UNet3D, so module names carry no 'unet.'
prefix: the bottleneck is 'bottleneck', not 'unet.bottleneck'. Names are
matched exactly. The script lists what is available if a requested name does
not resolve.

HOW TO READ THE RESULT
----------------------
'bottleneck' is the control and is expected to be degenerate. A candidate
layer is viable when `degenerate` is false and `activation_max` is of normal
magnitude. Agreement with Vanilla Gradient and Integrated Gradients is
reported for context, not as a correctness criterion: Grad-CAM answers a
different question at a coarser resolution, so moderate agreement is
expected rather than high.

This script decides nothing on its own. It establishes whether the question
is worth pursuing before any manuscript text is rewritten.

Read-only.
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
from xai_explainers.attention_visualizer import GradCAM3D
from xai_explainers.gradient_methods import VanillaGradient, IntegratedGradients

TARGET_CLASS = 1

# Candidates span the decoder and the deepest encoder stage. 'bottleneck' is
# included deliberately as the control that should fail.
DEFAULT_LAYERS = [
    "bottleneck",
    "encoder.4",
    "decoder.0",
    "decoder.1",
    "decoder.2",
    "decoder.3",
]


def activation_stats(model, layer_name, img):
    """Magnitude of the layer's output on this input, independent of Grad-CAM.

    This is the measurement that decides whether a layer can support a CAM
    at all: if its activations are denormal, no weighting scheme recovers
    signal from them.
    """
    captured = {}
    target = dict(model.named_modules()).get(layer_name)
    if target is None:
        return None
    h = target.register_forward_hook(
        lambda m, i, o: captured.update(out=o.detach()))
    try:
        with torch.no_grad():
            model(img)
    finally:
        h.remove()
    a = captured.get("out")
    if a is None:
        return None
    return {
        "shape": list(a.shape),
        "max_abs": float(a.abs().max()),
        "mean_abs": float(a.abs().mean()),
        "exact_zero_fraction": float((a == 0).float().mean()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint",
                    default="models/segmentation/unet3d_cv_fold0.pth")
    ap.add_argument("--cache", default="data/brats_dataset/cache_128")
    ap.add_argument("--folds",
                    default="data/brats_dataset/folds_seed42_3fold.json")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--cases", type=int, default=5)
    ap.add_argument("--layers", nargs="+", default=DEFAULT_LAYERS)
    ap.add_argument("--ig-steps", type=int, default=20)
    ap.add_argument("--skip-ig", action="store_true",
                    help="skip Integrated Gradients to run faster; "
                         "Grad-CAM viability does not depend on it")
    ap.add_argument("--threads", type=int, default=NUM_THREADS)
    ap.add_argument("--project-root")
    ap.add_argument("--out",
                    default="results/metrics/gradcam_layer_scan.json")
    args = ap.parse_args()

    torch.set_num_threads(args.threads)

    provenance = {}
    model, meta = load_trained_model(args.checkpoint,
                                     project_root=args.project_root,
                                     provenance=provenance)
    model.eval()

    available = [n for n, _ in model.named_modules()]
    missing = [l for l in args.layers if l not in available]
    if missing:
        print("these requested layer(s) do not exist: %s" % missing,
              file=sys.stderr)
        print("candidate block-level names in this model:", file=sys.stderr)
        for n in available:
            if n and n.count(".") <= 1 and "conv" not in n:
                print("   %s" % n, file=sys.stderr)
        sys.exit(1)

    _, val_ids, _ = load_fold_split(args.folds, args.fold)
    paths = collect(args.cache, val_ids)[:args.cases]
    if not paths:
        sys.exit("no cached cases found under %s" % args.cache)
    print("scanning %d layer(s) over %d case(s)\n"
          % (len(args.layers), len(paths)), file=sys.stderr)

    results = {l: {"per_case": {}} for l in args.layers}
    t0 = time.time()

    for i, p in enumerate(paths, 1):
        case = os.path.basename(p)
        img, _ = load_case(p, remap=remap_decathlon)

        vg = VanillaGradient(model).generate(img, target_class=TARGET_CLASS)
        ig = None
        if not args.skip_ig:
            ig = IntegratedGradients(model, steps=args.ig_steps).generate(
                img, target_class=TARGET_CLASS)

        for layer in args.layers:
            act = activation_stats(model, layer, img)
            cam, err = None, None
            gc = GradCAM3D(model, target_layers=[layer])
            try:
                cam = gc.generate_cam(img, target_class=TARGET_CLASS)
            except Exception as exc:
                err = "%s: %s" % (type(exc).__name__, exc)
            finally:
                gc.remove_hooks()

            rec = {"activation": act, "error": err}
            if cam is not None:
                cam = np.asarray(cam, dtype=np.float64)
                flat_ok = cam.std() > 0
                rec.update({
                    "cam_shape": list(cam.shape),
                    "cam_max": float(cam.max()),
                    "cam_mean": float(cam.mean()),
                    "cam_std": float(cam.std()),
                    "degenerate": not flat_ok,
                    "nonzero_fraction": float((cam > 0).mean()),
                })
                if flat_ok and cam.shape == vg.shape and vg.std() > 0:
                    rec["pearson_vs_vg"] = float(
                        np.corrcoef(cam.ravel(), vg.ravel())[0, 1])
                if (flat_ok and ig is not None and cam.shape == ig.shape
                        and ig.std() > 0):
                    rec["pearson_vs_ig"] = float(
                        np.corrcoef(cam.ravel(), ig.ravel())[0, 1])
            else:
                rec["degenerate"] = True

            results[layer]["per_case"][case] = rec

        line = "  ".join(
            "%s=%s" % (l, "DEGEN" if results[l]["per_case"][case].get("degenerate")
                       else "ok")
            for l in args.layers)
        print("  [%d/%d] %-16s %s" % (i, len(paths), case, line),
              file=sys.stderr)

    for layer in args.layers:
        pc = results[layer]["per_case"]
        degen = sum(1 for v in pc.values() if v.get("degenerate"))
        acts = [v["activation"]["max_abs"] for v in pc.values()
                if v.get("activation")]
        rvg = [v["pearson_vs_vg"] for v in pc.values() if "pearson_vs_vg" in v]
        rig = [v["pearson_vs_ig"] for v in pc.values() if "pearson_vs_ig" in v]
        results[layer]["summary"] = {
            "n_cases": len(pc),
            "degenerate_cases": degen,
            "viable": degen == 0,
            "activation_max_abs_mean": (round(st.mean(acts), 6)
                                        if acts else None),
            "mean_pearson_vs_vg": round(st.mean(rvg), 4) if rvg else None,
            "mean_pearson_vs_ig": round(st.mean(rig), 4) if rig else None,
            "errors": sorted({v["error"] for v in pc.values() if v.get("error")}),
        }

    out = {
        "provenance": provenance,
        "checkpoint_meta": meta,
        "fold": args.fold,
        "n_cases": len(paths),
        "layers_scanned": args.layers,
        "ig_included": not args.skip_ig,
        "results": results,
        "elapsed_min": round((time.time() - t0) / 60.0, 1),
        "interpretation": (
            "'bottleneck' is the control and should be degenerate, because "
            "that stage emits a denormal constant. Any layer with "
            "viable=true and activations of normal magnitude can support a "
            "Grad-CAM map, which establishes that the method was never "
            "broken -- only pointed at a dead layer. That is the sole "
            "conclusion this scan supports. Viability is NOT validity: "
            "mean_pearson_vs_vg near zero, or differing in sign between "
            "layers, means the map is non-flat but unverified, and no "
            "particular layer is thereby justified as the configuration to "
            "adopt. Reinstating Grad-CAM as a quantitative method would "
            "require localisation checks against ground truth that this "
            "scan does not perform."),
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))

    print("\n--- layer scan, %d case(s) ---" % len(paths), file=sys.stderr)
    print("%-14s %8s %14s %12s %12s %s"
          % ("layer", "viable", "act max|a|", "r vs VG", "r vs IG", "notes"),
          file=sys.stderr)
    for layer in args.layers:
        s = results[layer]["summary"]
        note = "CONTROL (expected to fail)" if layer == "bottleneck" else ""
        if s["errors"]:
            note = s["errors"][0][:40]
        print("%-14s %8s %14s %12s %12s %s"
              % (layer,
                 "yes" if s["viable"] else "no (%d/%d degen)"
                 % (s["degenerate_cases"], s["n_cases"]),
                 s["activation_max_abs_mean"],
                 s["mean_pearson_vs_vg"], s["mean_pearson_vs_ig"], note),
              file=sys.stderr)

    viable = [l for l in args.layers
              if results[l]["summary"]["viable"] and l != "bottleneck"]
    rvgs = {l: results[l]["summary"]["mean_pearson_vs_vg"] for l in viable
            if results[l]["summary"]["mean_pearson_vs_vg"] is not None}
    print("", file=sys.stderr)
    if viable:
        print("DIAGNOSIS: Grad-CAM produced non-degenerate maps at %s and "
              "failed only at the bottleneck. The failure is therefore caused "
              "by hooking an inert layer, not by the layer-name resolution "
              "bug the manuscript cites." % ", ".join(viable),
              file=sys.stderr)
        if rvgs:
            lo, hi = min(rvgs.values()), max(rvgs.values())
            print("\nVIABILITY IS NOT VALIDITY. Agreement with Vanilla "
                  "Gradient spans %+.3f to %+.3f across these layers%s. A "
                  "non-degenerate map is a non-flat array, not a verified "
                  "explanation, and a result that changes sign with the "
                  "choice of target layer has no principled configuration to "
                  "reinstate. Establishing one needs localisation checks "
                  "against ground truth, which is a study rather than a "
                  "revision task."
                  % (lo, hi,
                     ", two of them negative"
                     if sum(1 for v in rvgs.values() if v < 0) >= 2 else ""),
                  file=sys.stderr)
            print("\nThat layer-sensitivity is itself reportable: it is a "
                  "known weakness of Grad-CAM in encoder-decoder networks and "
                  "supports the choice of input-space attribution methods, "
                  "rather than merely excusing the withdrawal.",
                  file=sys.stderr)
    else:
        print("No candidate layer produced a usable map. The withdrawal "
              "stands, and the bottleneck explanation can be stated more "
              "precisely than the current substring-matching account.",
              file=sys.stderr)
    print("wrote %s" % args.out, file=sys.stderr)


if __name__ == "__main__":
    main()
