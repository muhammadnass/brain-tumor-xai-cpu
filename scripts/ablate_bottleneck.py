#!/usr/bin/env python3
"""
ablate_bottleneck.py -- prove the inert stage claim by measurement.

    python scripts/ablate_bottleneck.py \
        --checkpoint models/segmentation/unet3d_cv_fold0.pth \
        --cache data/brats_dataset/cache_128 \
        --folds data/brats_dataset/folds_seed42_3fold.json --fold 0 \
        --layer bottleneck \
        --out results/metrics/bottleneck_ablation.json

Section 4.4 currently INFERS that the bottleneck contributes nothing from
the magnitude of its activations. This measures it instead: a forward
hook replaces the layer's output with zeros, and the fold's validation
split is re-evaluated. If Dice is unchanged, the claim is demonstrated
rather than argued, and the paired per-case comparison quantifies "unchanged".

Also reports the magnitude of the layer's INPUT alongside its output,
which distinguishes two different situations that the output magnitude
alone cannot:

  input healthy, output dead  -> the stage's own weights collapsed
  input dead, output dead     -> the collapse originates upstream, and
                                 more of the network than this stage is
                                 implicated

Read-only with respect to checkpoints and caches. Nothing is retrained.
"""

import os

NUM_THREADS = int(os.environ.get("XAI_NUM_THREADS", "8"))
os.environ.setdefault("OMP_NUM_THREADS", str(NUM_THREADS))
os.environ.setdefault("MKL_NUM_THREADS", str(NUM_THREADS))

import argparse
import json
import statistics
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_loader import load_trained_model, resolve_layer, load_fold_split
from evaluate_cross_population import (load_case, collect, remap_decathlon,
                                       binary_metrics, bootstrap_ci, THRESHOLD)

TUMOUR_CLASS = 1


def tensor_stats(t):
    v = t.detach().reshape(-1)
    nz = v[v != 0]
    return {
        "shape": list(t.shape),
        "numel": int(v.numel()),
        "exact_zeros": int((v == 0).sum()),
        "zero_fraction": round(float((v == 0).float().mean()), 6),
        "max": float(v.max()), "min": float(v.min()),
        "mean_abs": float(v.abs().mean()),
        "nonzero_min_abs": float(nz.abs().min()) if nz.numel() else None,
        "nonzero_max_abs": float(nz.abs().max()) if nz.numel() else None,
    }


def show(name, s):
    print("  %-12s shape=%s  zeros=%d/%d (%.1f%%)  max=%.6e  mean|a|=%.6e"
          % (name, s["shape"], s["exact_zeros"], s["numel"],
             100 * s["zero_fraction"], s["max"], s["mean_abs"]),
          file=sys.stderr)


def evaluate(model, paths, tag):
    rows = []
    for i, p in enumerate(paths, 1):
        img, lab = load_case(p, remap=remap_decathlon)
        if lab is None:
            continue
        prob = torch.softmax(model(img), dim=1)[0, TUMOUR_CLASS].detach().numpy()
        m = binary_metrics(prob >= THRESHOLD, lab.numpy() == TUMOUR_CLASS)
        m["case"] = os.path.basename(p)
        rows.append(m)
        if i % 40 == 0 or i == len(paths):
            print("  %s %d/%d  running mean Dice %.4f"
                  % (tag, i, len(paths),
                     statistics.mean(r["dice"] for r in rows)), file=sys.stderr)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--folds", required=True)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--layer", default="bottleneck")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--threads", type=int, default=NUM_THREADS)
    ap.add_argument("--project-root")
    ap.add_argument("--out", default="results/metrics/bottleneck_ablation.json")
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    provenance = {}
    model, meta = load_trained_model(args.checkpoint,
                                     project_root=args.project_root,
                                     provenance=provenance)
    layer = resolve_layer(model, args.layer)

    _, val_ids, _ = load_fold_split(args.folds, args.fold)
    paths = collect(args.cache, val_ids)
    if args.limit:
        paths = paths[:args.limit]
    if not paths:
        sys.exit("no cases found for fold %d" % args.fold)

    # --- magnitudes of the layer's input and output on one case ---
    img0, _ = load_case(paths[0], remap=remap_decathlon)
    captured = {}

    def probe(_m, inputs, out):
        if inputs and torch.is_tensor(inputs[0]):
            captured["in"] = tensor_stats(inputs[0])
        captured["out"] = tensor_stats(out)

    h = layer.register_forward_hook(probe)
    try:
        with torch.no_grad():
            model(img0)
    finally:
        h.remove()

    print("\n%s magnitudes on %s:" % (args.layer, os.path.basename(paths[0])),
          file=sys.stderr)
    for key in ("in", "out"):
        if key in captured:
            show("%s (%s)" % (args.layer, "input" if key == "in" else "output"),
                 captured[key])
    verdict = None
    if "in" in captured and "out" in captured:
        i, o = captured["in"]["mean_abs"], captured["out"]["mean_abs"]
        if o < 1e-20 and i < 1e-20:
            verdict = ("input and output both vanish: the collapse originates "
                       "UPSTREAM of this stage, so more of the network than "
                       "this layer is implicated")
        elif o < 1e-20:
            verdict = ("input is of normal magnitude but output vanishes: "
                       "this stage's own transformation collapses")
        else:
            verdict = "output is of normal magnitude; this layer is not inert"
        print("  -> %s" % verdict, file=sys.stderr)

    # --- Dice with and without the layer ---
    print("\nintact model:", file=sys.stderr)
    intact = evaluate(model, paths, "intact")

    def zero_out(_m, _i, out):
        return torch.zeros_like(out)

    h = layer.register_forward_hook(zero_out)
    try:
        print("\n%s output forced to zero:" % args.layer, file=sys.stderr)
        ablated = evaluate(model, paths, "ablated")
    finally:
        h.remove()

    by_case = {r["case"]: r["dice"] for r in intact}
    pairs = [(by_case[r["case"]], r["dice"]) for r in ablated
             if r["case"] in by_case]
    diffs = [b - a for a, b in pairs]
    va = [a for a, _ in pairs]
    vb = [b for _, b in pairs]

    out = {
        "checkpoint_meta": meta,
        "layer": args.layer,
        "n_cases": len(pairs),
        "layer_magnitudes": captured,
        "magnitude_verdict": verdict,
        "intact_mean_dice": round(statistics.mean(va), 6),
        "ablated_mean_dice": round(statistics.mean(vb), 6),
        "mean_change": round(statistics.mean(diffs), 6),
        "max_abs_change": round(max(abs(d) for d in diffs), 6) if diffs else None,
        "n_cases_changed": sum(1 for d in diffs if d != 0),
        "intact_dice_95ci": bootstrap_ci(va) if len(va) > 1 else None,
        "ablated_dice_95ci": bootstrap_ci(vb) if len(vb) > 1 else None,
        "per_case_change": {r["case"]: round(d, 8)
                            for r, d in zip(ablated, diffs)},
        "interpretation": (
            "If mean_change is zero and n_cases_changed is zero, the stage "
            "contributes nothing to the output and the claim in Section 4.4 "
            "is a measurement rather than an inference drawn from activation "
            "magnitudes."),
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)

    print("\n--- %s ablation, fold %d, n=%d ---"
          % (args.layer, args.fold, len(pairs)), file=sys.stderr)
    print("  intact  mean Dice %.6f" % out["intact_mean_dice"], file=sys.stderr)
    print("  ablated mean Dice %.6f" % out["ablated_mean_dice"], file=sys.stderr)
    print("  change  %+.6f   cases changed at all: %d of %d   "
          "largest single change %.2e"
          % (out["mean_change"], out["n_cases_changed"], len(pairs),
             out["max_abs_change"] or 0.0), file=sys.stderr)
    print("wrote %s" % args.out, file=sys.stderr)


if __name__ == "__main__":
    main()
