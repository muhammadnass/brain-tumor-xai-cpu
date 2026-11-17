#!/usr/bin/env python3
"""
diagnose_gradcam.py -- find where Grad-CAM collapses to a degenerate map.

Run this on ONE case before rerunning the full measurement:

    python scripts/diagnose_gradcam.py \
        --checkpoint results/checkpoints/cv_fold1_best.pt \
        --cache data/brats_dataset/cache_128 \
        --case BRATS_065 \
        --layer bottleneck

It reports, stage by stage:

  1. whether the forward hook fires on the named layer, and the shape and
     statistics of the activation it captures
  2. whether a gradient reaches that activation, and its statistics
  3. the channel weights alpha, and in particular HOW MANY ARE NEGATIVE --
     if all of them are, the ReLU in Eq. (5) zeroes the entire map and the
     result is degenerate by construction
  4. the map before and after the ReLU, for four different formulations of
     the target-class scalar
  5. the same case through the project's own GradCAM3D, for comparison
  6. timing of three Grad-CAM variants, to explain why the measured cost
     matched Vanilla Gradient

Read-only. Nothing is written except the optional --out JSON.
"""

import os

NUM_THREADS = int(os.environ.get("XAI_NUM_THREADS", "8"))
os.environ.setdefault("OMP_NUM_THREADS", str(NUM_THREADS))
os.environ.setdefault("MKL_NUM_THREADS", str(NUM_THREADS))

import argparse
import glob
import inspect
import json
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_loader import load_trained_model, resolve_layer


def load_case(path):
    if path.endswith(".npz"):
        with np.load(path) as z:
            keys = list(z.keys())
            ikey = next((k for k in ("image", "img", "data", "x") if k in keys),
                        keys[0])
            arr = z[ikey]
    elif path.endswith(".npy"):
        arr = np.load(path)
    else:
        obj = torch.load(path, map_location="cpu")
        arr = obj["image"] if isinstance(obj, dict) else obj
        arr = arr.numpy() if torch.is_tensor(arr) else arr
    t = torch.as_tensor(np.ascontiguousarray(np.asarray(arr))).float()
    while t.ndim < 5:
        t = t.unsqueeze(0)
    return t


def stats(t, name):
    v = t.detach().flatten()
    return {
        "name": name,
        "shape": list(t.shape),
        "min": round(float(v.min()), 8),
        "max": round(float(v.max()), 8),
        "mean": round(float(v.mean()), 8),
        "abs_mean": round(float(v.abs().mean()), 8),
        "frac_zero": round(float((v == 0).float().mean()), 4),
        "frac_positive": round(float((v > 0).float().mean()), 4),
    }


def show(d, indent="    "):
    print("%s%-22s shape=%s min=%.6g max=%.6g mean=%.6g |mean|=%.6g "
          "zero=%.1f%% pos=%.1f%%"
          % (indent, d["name"], d["shape"], d["min"], d["max"], d["mean"],
             d["abs_mean"], 100 * d["frac_zero"], 100 * d["frac_positive"]),
          file=sys.stderr)


# ------------------------------------------------------------------ #
# Target-class scalar formulations
# ------------------------------------------------------------------ #
def score_sum_all(logits, cls, pred_mask):
    """Sum of the class channel over EVERY voxel. This is what
    measure_resources.py used. On a volume where the tumour occupies a
    small fraction of voxels, the sum is dominated by background, and the
    gradient can be negative for most channels."""
    return logits[:, cls].sum()


def score_sum_predicted(logits, cls, pred_mask):
    """Sum of the class channel restricted to voxels the model predicts as
    that class. Concentrates the signal on the structure of interest."""
    sel = logits[:, cls][pred_mask]
    return sel.sum() if sel.numel() else logits[:, cls].sum()


def score_max(logits, cls, pred_mask):
    """Single most strongly activated voxel."""
    return logits[:, cls].max()


def score_mean(logits, cls, pred_mask):
    """Mean over all voxels; same direction as sum_all, different scale."""
    return logits[:, cls].mean()


SCORES = [
    ("sum over all voxels", score_sum_all),
    ("sum over predicted-class voxels", score_sum_predicted),
    ("max voxel", score_max),
    ("mean over all voxels", score_mean),
]


def gradcam_once(model, x, layer, cls, score_fn, collect=None):
    """One Grad-CAM pass. Returns (cam, degenerate, diagnostics)."""
    acts, grads = {}, {}

    def fwd_hook(_m, _i, out):
        acts["a"] = out
        if out.requires_grad:
            out.register_hook(lambda g: grads.__setitem__("g", g))
        else:
            acts["no_grad"] = True

    handle = layer.register_forward_hook(fwd_hook)
    diag = {}
    try:
        model.zero_grad(set_to_none=True)
        logits = model(x)
        diag["logits"] = stats(logits, "logits")
        diag["hook_fired"] = "a" in acts
        if not diag["hook_fired"]:
            return None, None, diag
        if acts.get("no_grad"):
            diag["activation_requires_grad"] = False
            return None, None, diag
        diag["activation_requires_grad"] = True
        diag["activation"] = stats(acts["a"], "activation a")

        pred_mask = (logits.argmax(dim=1) == cls)
        diag["predicted_class_voxels"] = int(pred_mask.sum())
        diag["predicted_class_fraction"] = round(
            float(pred_mask.float().mean()), 6)

        score_fn(logits, cls, pred_mask).backward()
        diag["backward_hook_fired"] = "g" in grads
        if "g" not in grads:
            return None, None, diag

        a, g = acts["a"], grads["g"]
        diag["gradient"] = stats(g, "gradient dS/da")

        alpha = g.mean(dim=(2, 3, 4), keepdim=True)
        af = alpha.flatten()
        diag["alpha"] = stats(alpha, "alpha (channel wts)")
        diag["alpha_channels"] = int(af.numel())
        diag["alpha_negative_channels"] = int((af < 0).sum())
        diag["alpha_positive_channels"] = int((af > 0).sum())
        diag["alpha_zero_channels"] = int((af == 0).sum())

        raw = (alpha * a).sum(dim=1, keepdim=True)
        diag["cam_pre_relu"] = stats(raw, "cam before ReLU")
        cam = torch.relu(raw)
        diag["cam_post_relu"] = stats(cam, "cam after ReLU")

        cam = torch.nn.functional.interpolate(
            cam, size=x.shape[2:], mode="trilinear", align_corners=False)
        lo, hi = cam.min(), cam.max()
        degenerate = bool((hi - lo).item() == 0.0)
        diag["degenerate"] = degenerate
        if not degenerate:
            cam = (cam - lo) / (hi - lo)
        if collect is not None:
            collect["cam"] = cam.detach()
        return cam.detach(), degenerate, diag
    finally:
        handle.remove()


def gradcam_detached(model, x, layer, cls):
    """Grad-CAM with the graph cut at the hooked layer.

    The forward hook replaces the module output with a detached copy that
    requires grad, so the backward pass runs only from the output back to
    this layer, not all the way to the input. The channel gradients are
    identical -- detaching stops propagation further back, it does not
    change the gradient at this point -- but the cost is only the part of
    the network above the layer."""
    holder = {}

    def fwd_hook(_m, _i, out):
        detached = out.detach().requires_grad_(True)
        holder["a"] = detached
        return detached

    handle = layer.register_forward_hook(fwd_hook)
    try:
        logits = model(x)
        logits[:, cls].sum().backward()
        a = holder["a"]
        alpha = a.grad.mean(dim=(2, 3, 4), keepdim=True)
        cam = torch.relu((alpha * a).sum(dim=1, keepdim=True))
        cam = torch.nn.functional.interpolate(
            cam, size=x.shape[2:], mode="trilinear", align_corners=False)
        lo, hi = cam.min(), cam.max()
        return cam.detach(), bool((hi - lo).item() == 0.0)
    finally:
        handle.remove()


def vanilla_gradient(model, x, cls):
    xi = x.clone().requires_grad_(True)
    model.zero_grad(set_to_none=True)
    logits = model(xi)
    logits[:, cls].sum().backward()
    return xi.grad.detach().abs().amax(dim=1, keepdim=True)


def correlate(a, b):
    if a is None or b is None:
        return None
    u = a.flatten().numpy().astype(np.float64)
    v = np.asarray(b).flatten().astype(np.float64)
    if u.shape != v.shape or u.std() == 0 or v.std() == 0:
        return None
    return round(float(np.corrcoef(u, v)[0, 1]), 4)


def try_project_gradcam(model, x, layer_name, cls, project_root):
    """Attempt to run the project's own GradCAM3D on the same input."""
    report = {}
    for path in (os.path.join(project_root, "src"), project_root):
        if os.path.isdir(path) and path not in sys.path:
            sys.path.insert(0, path)
    try:
        from xai_explainers.attention_visualizer import GradCAM3D
    except Exception as exc:
        report["import"] = "FAILED: %r" % (exc,)
        return report, None
    report["import"] = "ok"
    try:
        report["signature"] = str(inspect.signature(GradCAM3D.__init__))
    except (TypeError, ValueError):
        pass

    attempts = [
        ("GradCAM3D(model, layer_name)", lambda: GradCAM3D(model, layer_name)),
        ("GradCAM3D(model, target_layer=layer_name)",
         lambda: GradCAM3D(model, target_layer=layer_name)),
        ("GradCAM3D(model=model, layer_name=layer_name)",
         lambda: GradCAM3D(model=model, layer_name=layer_name)),
        ("GradCAM3D(model)", lambda: GradCAM3D(model)),
    ]
    obj = None
    for label, build in attempts:
        try:
            obj = build()
            report["constructed_with"] = label
            break
        except Exception as exc:
            report.setdefault("construction_errors", []).append(
                "%s -> %r" % (label, exc))
    if obj is None:
        report["result"] = ("could not construct automatically; check the "
                            "signature above and call it by hand")
        return report, None

    for label, call in (
            ("obj(x, cls)", lambda: obj(x, cls)),
            ("obj.generate(x, cls)", lambda: obj.generate(x, cls)),
            ("obj.generate_cam(x, cls)", lambda: obj.generate_cam(x, cls)),
            ("obj.explain(x, cls)", lambda: obj.explain(x, cls)),
            ("obj(x)", lambda: obj(x))):
        try:
            out = call()
            report["called_with"] = label
            arr = out[0] if isinstance(out, (tuple, list)) else out
            arr = arr.detach().cpu().numpy() if torch.is_tensor(arr) else np.asarray(arr)
            report["output_shape"] = list(arr.shape)
            report["output_min"] = float(np.min(arr))
            report["output_max"] = float(np.max(arr))
            report["degenerate"] = bool(np.max(arr) == np.min(arr))
            return report, arr
        except Exception as exc:
            report.setdefault("call_errors", []).append("%s -> %r" % (label, exc))
    report["result"] = "constructed but no known call signature worked"
    return report, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--case", help="substring of the case filename")
    ap.add_argument("--layer", default="bottleneck")
    ap.add_argument("--target-class", type=int, default=1)
    ap.add_argument("--project-root")
    ap.add_argument("--threads", type=int, default=NUM_THREADS)
    ap.add_argument("--skip-timing", action="store_true")
    ap.add_argument("--out")
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    root = os.path.abspath(args.project_root or os.getcwd())
    prov = {}
    model, meta = load_trained_model(args.checkpoint, project_root=root,
                                     provenance=prov)

    pattern = "*%s*" % args.case if args.case else "*.np[yz]"
    paths = sorted(glob.glob(os.path.join(args.cache, pattern)))
    if not paths:
        sys.exit("no case matching %r under %s" % (pattern, args.cache))
    path = paths[0]
    x = load_case(path)
    print("\ncase: %s   input %s" % (os.path.basename(path), list(x.shape)),
          file=sys.stderr)

    names = [n for n, _ in model.named_modules()]
    matches = [n for n in names if args.layer in n]
    print("modules containing %r: %s" % (args.layer, matches[:10]),
          file=sys.stderr)
    layer = resolve_layer(model, args.layer)
    print("hooking exactly: %r (%s)\n"
          % (args.layer, type(layer).__name__), file=sys.stderr)

    report = {"checkpoint_meta": meta, "case": os.path.basename(path),
              "layer": args.layer, "target_class": args.target_class,
              "modules_matching_layer": matches[:20], "variants": {}}

    first_cam = None
    for label, fn in SCORES:
        print("--- target scalar: %s ---" % label, file=sys.stderr)
        collect = {}
        cam, degen, diag = gradcam_once(model, x, layer, args.target_class,
                                        fn, collect)
        report["variants"][label] = diag

        if not diag.get("hook_fired"):
            print("    FORWARD HOOK DID NOT FIRE -- the named module is not "
                  "called in forward()", file=sys.stderr)
            continue
        if not diag.get("activation_requires_grad", True):
            print("    activation does not require grad -- no graph was built",
                  file=sys.stderr)
            continue
        if not diag.get("backward_hook_fired"):
            print("    no gradient reached the activation", file=sys.stderr)
            continue

        for key in ("activation", "gradient", "alpha", "cam_pre_relu",
                    "cam_post_relu"):
            show(diag[key])
        print("    alpha channels: %d negative, %d positive, %d zero (of %d)"
              % (diag["alpha_negative_channels"],
                 diag["alpha_positive_channels"],
                 diag["alpha_zero_channels"], diag["alpha_channels"]),
              file=sys.stderr)
        print("    predicted class-%d voxels: %d (%.4f%% of volume)"
              % (args.target_class, diag["predicted_class_voxels"],
                 100 * diag["predicted_class_fraction"]), file=sys.stderr)
        print("    DEGENERATE: %s\n" % diag["degenerate"], file=sys.stderr)
        if first_cam is None and not degen:
            first_cam = cam

    # Project implementation
    print("--- project GradCAM3D (xai_explainers.attention_visualizer) ---",
          file=sys.stderr)
    proj_report, proj_cam = try_project_gradcam(
        model, x, args.layer, args.target_class, root)
    report["project_gradcam"] = proj_report
    for k, v in proj_report.items():
        print("    %s: %s" % (k, str(v)[:300]), file=sys.stderr)
    if proj_cam is not None and first_cam is not None:
        c = correlate(first_cam, proj_cam)
        report["correlation_with_project"] = c
        print("    correlation with first non-degenerate variant: %s" % c,
              file=sys.stderr)

    # Timing
    if not args.skip_timing:
        print("\n--- timing (1 warm-up, 3 timed) ---", file=sys.stderr)
        timings = {}
        for label, call in (
                ("grad_cam (full backward, as measured)",
                 lambda: gradcam_once(model, x, layer, args.target_class,
                                      score_sum_all)),
                ("grad_cam (graph cut at layer)",
                 lambda: gradcam_detached(model, x, layer, args.target_class)),
                ("vanilla_gradient (backward to input)",
                 lambda: vanilla_gradient(model, x, args.target_class))):
            call()
            ts = []
            for _ in range(3):
                t0 = time.perf_counter()
                call()
                ts.append(time.perf_counter() - t0)
            timings[label] = round(sum(ts) / len(ts), 3)
            print("    %-42s %.3f s" % (label, timings[label]), file=sys.stderr)
        report["timing_s"] = timings

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as fh:
            json.dump(report, fh, indent=2)
        print("\nwrote %s" % args.out, file=sys.stderr)


if __name__ == "__main__":
    main()
