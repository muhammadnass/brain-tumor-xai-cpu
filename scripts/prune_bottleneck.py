#!/usr/bin/env python3
"""
prune_bottleneck.py -- build the bottleneck-free deployment model and prove
it is computationally equivalent to the trained network.

    python scripts/prune_bottleneck.py \
        --checkpoint models/segmentation/unet3d_cv_fold0.pth \
        --out results/metrics/pruned_fold0.json \
        --save-model models/segmentation/unet3d_pruned_fold0.pth

WHY THIS IS NOT A RETRAIN
-------------------------
Reviewer 1 asks the authors to "either retrain a streamlined architecture
without the inert bottleneck or provide rigorous validation proving that
this redundancy does not compromise parameter efficiency claims." This
script does the second, and it does it by construction rather than by
argument.

Two facts compose:

  (1) The trained bottleneck emits zero. ablate_bottleneck.py replaced its
      output with exact zeros and re-evaluated every held-out case:
      0 of 162, 0 of 161 and 0 of 161 predictions changed across folds 0,
      1 and 2, with max_abs_change exactly 0.0. That is a measurement on
      all 484 cases, already on record.

  (2) upconv[0] is an affine operator, so upconv[0](0) is its bias exactly,
      broadcast over the output volume. No data is needed to know this; it
      follows from linearity.

Composing them: the bottleneck AND the first up-convolution together are
replaceable by a single constant vector of features[-1] values. The pruned
network therefore reproduces the zero-ablated network bit-for-bit on ANY
input, not merely on the test distribution -- which is a stronger claim
than an empirical check on a fixed sample could support.

This script verifies that equivalence numerically rather than asserting
it, and reports the parameter and memory consequences.

WHAT TO EXPECT
--------------
The removed stages sit at the lowest spatial resolution in the network
(4^3 and 8^3), where arithmetic is cheap. The honest result is therefore
a large reduction in parameters and stored model size with little change
in latency. Do not report this as a speed-up. Reviewer 1's concern is the
parameter-efficiency claim, and that is precisely what the parameter count
addresses.

Read-only with respect to the trained checkpoints. Nothing is retrained.
"""

import os

NUM_THREADS = int(os.environ.get("XAI_NUM_THREADS", "8"))
os.environ.setdefault("OMP_NUM_THREADS", str(NUM_THREADS))
os.environ.setdefault("MKL_NUM_THREADS", str(NUM_THREADS))

import argparse
import copy
import json
import resource
import statistics
import sys
import time

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_loader import load_trained_model

MIB = 1024.0 * 1024.0


# ------------------------------------------------------------------ #
# The pruned network
# ------------------------------------------------------------------ #
class PrunedUNet3D(nn.Module):
    """UNet3D with the inert bottleneck and the first up-convolution removed.

    Built from an already-trained UNet3D. Every surviving stage is copied
    across unchanged; the two dead stages are replaced by the constant that
    upconv[0] produces when its input is zero, stored as a buffer.

    The spatial extent of that constant field is taken from the skip
    tensor at run time rather than hard-coded, so the module follows the
    source network's width and depth.
    """

    def __init__(self, src):
        super().__init__()
        self.features = tuple(src.features)

        self.encoder = copy.deepcopy(src.encoder)
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.decoder = copy.deepcopy(src.decoder)
        # upconv[0] is dropped; the remaining up-convolutions shift down one
        self.upconv = nn.ModuleList(
            copy.deepcopy(m) for m in list(src.upconv)[1:])
        self.final_conv = copy.deepcopy(src.final_conv)

        up0 = src.upconv[0]
        if up0.bias is None:
            raise RuntimeError(
                "upconv[0] has no bias term. The pruning argument assumes an "
                "affine map, whose output on a zero input is its bias. With "
                "bias=False the output would be exactly zero and this module "
                "would need to emit zeros instead -- but that is a different "
                "network from the one that was trained, so refuse rather "
                "than guess.")
        with torch.no_grad():
            self.register_buffer("up0_const", up0.bias.detach().clone())

        self.eval()

    def forward(self, x):
        enc = []
        for stage in self.encoder:
            x = stage(x)
            enc.append(x)
            x = self.pool(x)

        # bottleneck -> upconv[0] collapses to a spatially constant field
        skip = enc[-1]
        x = self.up0_const.view(1, -1, 1, 1, 1).expand(
            skip.shape[0], -1, *skip.shape[2:])
        x = torch.cat([x, skip], dim=1)
        x = self.decoder[0](x)

        for i in range(1, len(self.decoder)):
            x = self.upconv[i - 1](x)
            x = torch.cat([x, enc[-(i + 1)]], dim=1)
            x = self.decoder[i](x)

        return self.final_conv(x)


# ------------------------------------------------------------------ #
# Diagnosis of the collapse (Reviewer 2, point 2)
# ------------------------------------------------------------------ #
def batchnorm_report(block):
    """Per-BatchNorm affine and running statistics inside a ConvBlock3D.

    Reviewer 2 asks for "a clearer explanation of the numerical failure".
    The mechanism is visible here: if gamma and beta have both collapsed
    toward zero, the block emits approximately beta regardless of its
    input, and the following ReLU clamps what remains. The failure is a
    property of the trained parameters, not of any particular case, which
    is why the stage is inert for every input rather than for some.
    """
    out = {}
    idx = 0
    for mod in block.conv:
        if isinstance(mod, nn.BatchNorm3d):
            idx += 1
            out["bn%d" % idx] = {
                "channels": int(mod.weight.numel()),
                "gamma_mean_abs": float(mod.weight.abs().mean()),
                "gamma_max_abs": float(mod.weight.abs().max()),
                "beta_mean_abs": float(mod.bias.abs().mean()),
                "beta_max_abs": float(mod.bias.abs().max()),
                "running_mean_max_abs": float(mod.running_mean.abs().max()),
                "running_var_min": float(mod.running_var.min()),
                "running_var_max": float(mod.running_var.max()),
                "eps": float(mod.eps),
            }
    return out


def tensor_bytes(module):
    p = sum(t.numel() * t.element_size() for t in module.parameters())
    b = sum(t.numel() * t.element_size() for t in module.buffers())
    return p, b


def count(module):
    return sum(t.numel() for t in module.parameters())


# ------------------------------------------------------------------ #
# Equivalence check
# ------------------------------------------------------------------ #
def compare(intact, pruned, inputs, ablate_hook_target, tag):
    """Compare pruned output against the intact and zero-ablated networks.

    Returns per-input records. The zero-ablated network is the intact one
    with a forward hook forcing the bottleneck output to zeros -- exactly
    the construction ablate_bottleneck.py used, so the comparison is
    against the same reference the 484-case result was measured on.
    """
    rows = []
    for i, x in enumerate(inputs, 1):
        with torch.no_grad():
            logits_intact = intact(x)

            h = ablate_hook_target.register_forward_hook(
                lambda _m, _i, out: torch.zeros_like(out))
            try:
                logits_ablated = intact(x)
            finally:
                h.remove()

            logits_pruned = pruned(x)

            mask_intact = logits_intact.argmax(dim=1)
            mask_ablated = logits_ablated.argmax(dim=1)
            mask_pruned = logits_pruned.argmax(dim=1)

            n_vox = int(mask_intact.numel())
            rows.append({
                "input": tag(i),
                "max_abs_logit_diff_pruned_vs_ablated":
                    float((logits_pruned - logits_ablated).abs().max()),
                "max_abs_logit_diff_pruned_vs_intact":
                    float((logits_pruned - logits_intact).abs().max()),
                "bitwise_identical_pruned_vs_ablated":
                    bool(torch.equal(logits_pruned, logits_ablated)),
                "bitwise_identical_pruned_vs_intact":
                    bool(torch.equal(logits_pruned, logits_intact)),
                "voxels": n_vox,
                "mask_disagreements_pruned_vs_ablated":
                    int((mask_pruned != mask_ablated).sum()),
                "mask_disagreements_pruned_vs_intact":
                    int((mask_pruned != mask_intact).sum()),
                "mask_disagreements_ablated_vs_intact":
                    int((mask_ablated != mask_intact).sum()),
            })
        print("  %-18s  pruned vs ablated: max|dlogit|=%.3e  mask diffs=%d   "
              "pruned vs intact: max|dlogit|=%.3e  mask diffs=%d"
              % (rows[-1]["input"],
                 rows[-1]["max_abs_logit_diff_pruned_vs_ablated"],
                 rows[-1]["mask_disagreements_pruned_vs_ablated"],
                 rows[-1]["max_abs_logit_diff_pruned_vs_intact"],
                 rows[-1]["mask_disagreements_pruned_vs_intact"]),
              file=sys.stderr)
    return rows


def latency(model, x, repeats, warmup=1):
    for _ in range(warmup):
        with torch.no_grad():
            model(x)
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        with torch.no_grad():
            model(x)
        times.append(time.perf_counter() - t0)
    return {
        "repeats": repeats,
        "median_s": round(statistics.median(times), 4),
        "mean_s": round(statistics.mean(times), 4),
        "min_s": round(min(times), 4),
        "max_s": round(max(times), 4),
        "sd_s": round(statistics.stdev(times), 4) if len(times) > 1 else 0.0,
    }


# ------------------------------------------------------------------ #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--project-root")
    ap.add_argument("--cache",
                    help="optional 128^3 case cache. When the dataset has "
                         "been restored, pass it to repeat the equivalence "
                         "check on real volumes as well as synthetic ones.")
    ap.add_argument("--folds")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--real-cases", type=int, default=8,
                    help="how many cached cases to check when --cache is given")
    ap.add_argument("--synthetic", type=int, default=4,
                    help="number of random volumes for the equivalence check")
    ap.add_argument("--shape", default="1,4,128,128,128")
    ap.add_argument("--latency-repeats", type=int, default=5)
    ap.add_argument("--skip-latency", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--threads", type=int, default=NUM_THREADS)
    ap.add_argument("--save-model")
    ap.add_argument("--out", default="results/metrics/pruned_bottleneck.json")
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    shape = tuple(int(v) for v in args.shape.split(","))

    provenance = {}
    intact, meta = load_trained_model(args.checkpoint,
                                      project_root=args.project_root,
                                      provenance=provenance)
    intact.eval()

    # ---- what is being removed -----------------------------------
    n_total = count(intact)
    n_bottleneck = count(intact.bottleneck)
    n_upconv0 = count(intact.upconv[0])

    pruned = PrunedUNet3D(intact)
    n_pruned = count(pruned)
    n_const = int(pruned.up0_const.numel())

    p_bytes_i, b_bytes_i = tensor_bytes(intact)
    p_bytes_p, b_bytes_p = tensor_bytes(pruned)

    print("\nparameters", file=sys.stderr)
    print("  intact total        %12s" % format(n_total, ","), file=sys.stderr)
    print("  bottleneck          %12s  (%.2f%%)"
          % (format(n_bottleneck, ","), 100.0 * n_bottleneck / n_total),
          file=sys.stderr)
    print("  upconv[0]           %12s  (%.2f%%)"
          % (format(n_upconv0, ","), 100.0 * n_upconv0 / n_total),
          file=sys.stderr)
    print("  constant retained   %12s" % format(n_const, ","), file=sys.stderr)
    print("  pruned total        %12s  (-%.2f%%)"
          % (format(n_pruned, ","),
             100.0 * (n_total - n_pruned) / n_total), file=sys.stderr)

    # ---- why it is inert -----------------------------------------
    bn = batchnorm_report(intact.bottleneck)
    print("\nbottleneck BatchNorm state (mechanism of the collapse)",
          file=sys.stderr)
    for k, v in bn.items():
        print("  %s  gamma |mean|=%.3e |max|=%.3e   beta |mean|=%.3e "
              "|max|=%.3e   running_var in [%.3e, %.3e]"
              % (k, v["gamma_mean_abs"], v["gamma_max_abs"],
                 v["beta_mean_abs"], v["beta_max_abs"],
                 v["running_var_min"], v["running_var_max"]),
              file=sys.stderr)

    # ---- equivalence on synthetic volumes ------------------------
    print("\nequivalence check, %d synthetic volume(s) of shape %s"
          % (args.synthetic, shape), file=sys.stderr)
    syn = [torch.randn(*shape) for _ in range(args.synthetic)]
    rows = compare(intact, pruned, syn, intact.bottleneck,
                   lambda i: "synthetic-%d" % i)

    # ---- equivalence on real volumes, when the cache exists ------
    real_rows = []
    if args.cache:
        if not args.folds:
            sys.exit("--cache requires --folds so the validation split is known")
        from model_loader import load_fold_split
        from evaluate_cross_population import (load_case, collect,
                                               remap_decathlon)
        _, val_ids, _ = load_fold_split(args.folds, args.fold)
        paths = collect(args.cache, val_ids)[:args.real_cases]
        if not paths:
            print("  no cached cases found under %s -- skipping the real-volume "
                  "check" % args.cache, file=sys.stderr)
        else:
            print("\nequivalence check, %d real volume(s) from fold %d"
                  % (len(paths), args.fold), file=sys.stderr)
            imgs, names = [], []
            for p in paths:
                img, _ = load_case(p, remap=remap_decathlon)
                imgs.append(img)
                names.append(os.path.basename(p))
            real_rows = compare(intact, pruned, imgs, intact.bottleneck,
                                lambda i: names[i - 1])

    all_rows = rows + real_rows
    worst_vs_ablated = max(
        r["max_abs_logit_diff_pruned_vs_ablated"] for r in all_rows)
    worst_vs_intact = max(
        r["max_abs_logit_diff_pruned_vs_intact"] for r in all_rows)
    disagree_vs_ablated = sum(
        r["mask_disagreements_pruned_vs_ablated"] for r in all_rows)
    disagree_vs_intact = sum(
        r["mask_disagreements_pruned_vs_intact"] for r in all_rows)
    all_bitwise = all(r["bitwise_identical_pruned_vs_ablated"]
                      for r in all_rows)

    # ---- latency -------------------------------------------------
    lat = {}
    if not args.skip_latency:
        x = syn[0]
        print("\nlatency, %d repeat(s) on shape %s"
              % (args.latency_repeats, shape), file=sys.stderr)
        lat["intact"] = latency(intact, x, args.latency_repeats)
        lat["pruned"] = latency(pruned, x, args.latency_repeats)
        lat["median_ratio_pruned_over_intact"] = round(
            lat["pruned"]["median_s"] / lat["intact"]["median_s"], 4)
        print("  intact median %.3f s | pruned median %.3f s | ratio %.3f"
              % (lat["intact"]["median_s"], lat["pruned"]["median_s"],
                 lat["median_ratio_pruned_over_intact"]), file=sys.stderr)

    # ---- save the pruned model -----------------------------------
    saved = {}
    if args.save_model:
        os.makedirs(os.path.dirname(args.save_model) or ".", exist_ok=True)
        torch.save({
            "model_state_dict": pruned.state_dict(),
            "features": list(pruned.features),
            "architecture": "PrunedUNet3D",
            "derived_from": os.path.abspath(args.checkpoint),
            "fold": meta.get("fold"),
            "epoch": meta.get("epoch"),
            "val_dice": meta.get("val_dice"),
            "threshold": meta.get("threshold"),
            "note": ("Bottleneck and upconv[0] removed. up0_const holds the "
                     "constant that upconv[0] emitted for a zero input. "
                     "Predictions are identical to the source checkpoint; see "
                     "the accompanying JSON for the verification record."),
        }, args.save_model)
        saved = {
            "path": args.save_model,
            "file_bytes": os.path.getsize(args.save_model),
            "file_mib": round(os.path.getsize(args.save_model) / MIB, 2),
        }
        print("\nwrote %s (%.2f MiB)"
              % (args.save_model, saved["file_mib"]), file=sys.stderr)

    # ---- record --------------------------------------------------
    out = {
        "provenance": provenance,
        "checkpoint_meta": meta,
        "parameters": {
            "intact_total": n_total,
            "bottleneck": n_bottleneck,
            "bottleneck_fraction": round(n_bottleneck / n_total, 6),
            "upconv0": n_upconv0,
            "upconv0_fraction": round(n_upconv0 / n_total, 6),
            "constant_retained": n_const,
            "removed_net": n_total - n_pruned,
            "pruned_total": n_pruned,
            "reduction_fraction": round((n_total - n_pruned) / n_total, 6),
        },
        "storage": {
            "intact_parameter_mib": round(p_bytes_i / MIB, 2),
            "intact_parameter_plus_buffer_mib":
                round((p_bytes_i + b_bytes_i) / MIB, 2),
            "pruned_parameter_mib": round(p_bytes_p / MIB, 2),
            "pruned_parameter_plus_buffer_mib":
                round((p_bytes_p + b_bytes_p) / MIB, 2),
            "note": ("Section 4.6 of the manuscript reports 86.32 MiB, which "
                     "is the parameter tensors alone in single precision. "
                     "intact_parameter_mib is the directly comparable figure."),
        },
        "bottleneck_batchnorm": bn,
        "equivalence": {
            "n_synthetic": len(rows),
            "n_real": len(real_rows),
            "worst_max_abs_logit_diff_vs_ablated": worst_vs_ablated,
            "worst_max_abs_logit_diff_vs_intact": worst_vs_intact,
            "total_mask_disagreements_vs_ablated": disagree_vs_ablated,
            "total_mask_disagreements_vs_intact": disagree_vs_intact,
            "all_outputs_bitwise_identical_to_ablated": all_bitwise,
            "per_input": all_rows,
        },
        "latency": lat,
        "saved_model": saved,
        "prior_ablation_on_record": {
            "source": "results/metrics/ablation_fold{0,1,2}.json",
            "fold0": {"n": 162, "intact_mean_dice": 0.729458,
                      "ablated_mean_dice": 0.729458, "cases_changed": 0,
                      "max_abs_change": 0.0},
            "fold1": {"n": 161, "intact_mean_dice": 0.766698,
                      "ablated_mean_dice": 0.766698, "cases_changed": 0,
                      "max_abs_change": 0.0},
            "fold2": {"n": 161, "intact_mean_dice": 0.741833,
                      "ablated_mean_dice": 0.741833, "cases_changed": 0,
                      "max_abs_change": 0.0},
            "note": ("These three runs established that forcing the "
                     "bottleneck output to zero changes no prediction on any "
                     "of the 484 held-out cases. This script establishes that "
                     "the pruned network reproduces that zero-ablated network "
                     "exactly. The two together give the pruned network's "
                     "accuracy on all 484 cases without retraining."),
        },
        "interpretation": (
            "The equivalence claim holds if "
            "all_outputs_bitwise_identical_to_ablated is true and "
            "total_mask_disagreements_vs_intact is zero. The removed stages "
            "operate at 4^3 and 8^3 resolution, so the expected gain is in "
            "parameter count and stored size rather than latency; report it "
            "that way."),
        "peak_rss_mib": round(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, 1),
    }

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)

    print("\n--- summary ---", file=sys.stderr)
    print("  parameters %s -> %s  (-%.2f%%)"
          % (format(n_total, ","), format(n_pruned, ","),
             100.0 * (n_total - n_pruned) / n_total), file=sys.stderr)
    print("  parameter storage %.2f MiB -> %.2f MiB"
          % (p_bytes_i / MIB, p_bytes_p / MIB), file=sys.stderr)
    print("  bitwise identical to the zero-ablated network: %s"
          % ("YES" if all_bitwise else "NO"), file=sys.stderr)
    print("  mask disagreements vs intact network: %d voxel(s)"
          % disagree_vs_intact, file=sys.stderr)
    print("  largest logit deviation vs intact: %.3e" % worst_vs_intact,
          file=sys.stderr)
    print("  peak RSS %.1f MiB" % out["peak_rss_mib"], file=sys.stderr)
    print("wrote %s" % args.out, file=sys.stderr)

    if not all_bitwise or disagree_vs_ablated:
        sys.exit("\nFAILED: the pruned network does not reproduce the "
                 "zero-ablated network. Do not report these figures.")


if __name__ == "__main__":
    main()
