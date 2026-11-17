#!/usr/bin/env python3
"""
measure_xai_latency.py -- clean per-case latency and peak memory for the two
gradient explainers.

    # close the browser and the editor first
    python scripts/measure_xai_latency.py --cases 10 \
        --out results/metrics/xai_latency_controlled.json

WHY THIS IS NEEDED
------------------
The manuscript reports 8.95 s for Vanilla Gradient and 177.77 s for
Integrated Gradients. Those figures came from a run in September. Re-running
the same cases in November gave roughly 5 s and 103 s -- about half, on
identical inputs with the same checkpoint.

Two causes are plausible and cannot be separated after the fact:

  * Machine state. The September run followed hours of other work; the
    November one followed a fresh boot. The same effect was measured during
    the pruned-model work, where the intact network's median drifted from
    2.551 s to 2.813 s across three sequential folds from thermal load
    alone.

  * A cheaper backward pass. The old code called `.backward()`, which
    populates gradients for every leaf tensor requiring grad -- including
    all 22,627,875 model parameters. The current code calls
    `torch.autograd.grad(target, x)`, which computes the input gradient
    alone and never touches the parameter buffers. For Integrated Gradients
    that is twenty backward passes each skipping 22.6M parameter-gradient
    computations, which is easily enough to account for a 40% reduction.

So neither set of numbers is trustworthy as published, and the difference
between them must not be reported as an improvement -- not because the code
could not have caused it, but because machine state could have, and the two
are no longer distinguishable. This script measures the current
implementation's absolute cost under a controlled protocol, which is what
the manuscript should report.

PROTOCOL
--------
  * Both explainers run on the SAME case back to back, so the pair shares
    its thermal state.
  * The order within each pair ALTERNATES, so neither method is
    systematically measured on the warmer core.
  * Warmup passes are discarded.
  * The drift slope of each series is reported, so thermal stability is a
    measurement rather than an assumption.
  * Peak resident memory is recorded for the resource table, which the
    manuscript reports as 3.04 GB for explanation generation.

The reported figure should be the median with its interval. A mean over a
handful of CPU timings on a passively cooled laptop is dominated by
whichever pass happened to coincide with a background task.

RUN THIS ON A QUIET MACHINE. Close the browser and the editor. Unlike the
stability and agreement runs, where contention only costs time, here it
corrupts the measurement itself.

Read-only with respect to checkpoints and cached volumes.
"""

import os

NUM_THREADS = int(os.environ.get("XAI_NUM_THREADS", "8"))
os.environ.setdefault("OMP_NUM_THREADS", str(NUM_THREADS))
os.environ.setdefault("MKL_NUM_THREADS", str(NUM_THREADS))

import argparse
import json
import random
import resource
import statistics as st
import sys
import time
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from model_loader import load_trained_model, load_fold_split
from evaluate_cross_population import load_case, collect, remap_decathlon
from xai_explainers.gradient_methods import VanillaGradient, IntegratedGradients

TARGET_CLASS = 1
MIB = 1024.0 * 1024.0


def percentile_bootstrap(values, statistic=st.median, n_boot=20000,
                         alpha=0.05, seed=42):
    if len(values) < 2:
        return None
    rng = random.Random(seed)
    n = len(values)
    boots = [statistic([values[rng.randrange(n)] for _ in range(n)])
             for _ in range(n_boot)]
    boots.sort()
    return [round(boots[int((alpha / 2.0) * n_boot)], 4),
            round(boots[min(int((1.0 - alpha / 2.0) * n_boot),
                            n_boot - 1)], 4)]


def drift_slope(times):
    """Least-squares slope of time against index, seconds per case. A slope
    indistinguishable from zero is evidence the run was thermally stable."""
    n = len(times)
    if n < 3:
        return None
    xbar = (n - 1) / 2.0
    ybar = st.mean(times)
    num = sum((i - xbar) * (t - ybar) for i, t in enumerate(times))
    den = sum((i - xbar) ** 2 for i in range(n))
    return round(num / den, 6) if den else None


def peak_rss_mib():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def summarise(times, label):
    return {
        "method": label,
        "n": len(times),
        "median_s": round(st.median(times), 3),
        "median_95ci": percentile_bootstrap(times),
        "mean_s": round(st.mean(times), 3),
        "sd_s": round(st.stdev(times), 3) if len(times) > 1 else 0.0,
        "min_s": round(min(times), 3),
        "max_s": round(max(times), 3),
        "drift_s_per_case": drift_slope(times),
        "raw_s": [round(t, 4) for t in times],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint",
                    default="models/segmentation/unet3d_cv_fold0.pth")
    ap.add_argument("--cache", default="data/brats_dataset/cache_128")
    ap.add_argument("--folds",
                    default="data/brats_dataset/folds_seed42_3fold.json")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--cases", type=int, default=10)
    ap.add_argument("--ig-steps", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=1,
                    help="discarded warmup case (one pass of each method)")
    ap.add_argument("--threads", type=int, default=NUM_THREADS)
    ap.add_argument("--project-root")
    ap.add_argument("--out",
                    default="results/metrics/xai_latency_controlled.json")
    args = ap.parse_args()

    torch.set_num_threads(args.threads)

    provenance = {}
    model, meta = load_trained_model(args.checkpoint,
                                     project_root=args.project_root,
                                     provenance=provenance)
    model.eval()

    vg = VanillaGradient(model)
    ig = IntegratedGradients(model, steps=args.ig_steps)

    _, val_ids, _ = load_fold_split(args.folds, args.fold)
    paths = collect(args.cache, val_ids)[:args.cases + args.warmup]
    if len(paths) < args.cases + args.warmup:
        sys.exit("need %d cached case(s), found %d"
                 % (args.cases + args.warmup, len(paths)))

    if args.warmup:
        print("warmup on %d case(s), discarded" % args.warmup, file=sys.stderr)
        for p in paths[:args.warmup]:
            img, _ = load_case(p, remap=remap_decathlon)
            vg.generate(img, target_class=TARGET_CLASS)
            ig.generate(img, target_class=TARGET_CLASS)

    work = paths[args.warmup:]
    print("\n%d case(s), both methods per case, order alternating"
          % len(work), file=sys.stderr)
    print("threads=%d  ig_steps=%d\n" % (args.threads, args.ig_steps),
          file=sys.stderr)

    t_vg, t_ig, order = [], [], []
    for i, p in enumerate(work):
        case = os.path.basename(p)
        img, _ = load_case(p, remap=remap_decathlon)

        if i % 2 == 0:
            t0 = time.perf_counter()
            vg.generate(img, target_class=TARGET_CLASS)
            a = time.perf_counter() - t0
            t0 = time.perf_counter()
            ig.generate(img, target_class=TARGET_CLASS)
            b = time.perf_counter() - t0
            order.append("vg_first")
        else:
            t0 = time.perf_counter()
            ig.generate(img, target_class=TARGET_CLASS)
            b = time.perf_counter() - t0
            t0 = time.perf_counter()
            vg.generate(img, target_class=TARGET_CLASS)
            a = time.perf_counter() - t0
            order.append("ig_first")

        t_vg.append(a)
        t_ig.append(b)
        print("  [%2d/%d] %-16s %-9s VG %7.3f s   IG %8.3f s   ratio %5.1fx"
              % (i + 1, len(work), case, order[-1], a, b, b / a),
              file=sys.stderr)

    ratios = [b / a for a, b in zip(t_vg, t_ig)]

    # An order effect would mean the alternation did not cancel position.
    vg_first = [t for t, o in zip(t_vg, order) if o == "vg_first"]
    vg_second = [t for t, o in zip(t_vg, order) if o == "ig_first"]

    out = {
        "provenance": provenance,
        "checkpoint_meta": meta,
        "protocol": {
            "cases": len(work),
            "warmup_cases": args.warmup,
            "ig_steps": args.ig_steps,
            "threads": args.threads,
            "interleaved": True,
            "within_case_order_alternated": True,
            "rationale": ("Both methods run on the same case back to back so "
                          "they share thermal state; the order alternates so "
                          "neither is systematically measured on the warmer "
                          "core. Report the median with its interval, not the "
                          "mean: a handful of CPU timings on a passively "
                          "cooled laptop is dominated by whichever pass "
                          "coincided with background activity."),
        },
        "vanilla_gradient": summarise(t_vg, "VanillaGradient"),
        "integrated_gradients": summarise(t_ig, "IntegratedGradients"),
        "cost_ratio_ig_over_vg": {
            "median": round(st.median(ratios), 2),
            "median_95ci": percentile_bootstrap(ratios),
            "min": round(min(ratios), 2),
            "max": round(max(ratios), 2),
        },
        "order_effect_vanilla_gradient": {
            "median_when_measured_first":
                round(st.median(vg_first), 3) if vg_first else None,
            "median_when_measured_second":
                round(st.median(vg_second), 3) if vg_second else None,
            "note": ("These should agree. A systematic gap means position "
                     "within the case still influences the timing and the "
                     "headline figures should not be trusted."),
        },
        "peak_rss_mib": round(peak_rss_mib(), 1),
        "peak_rss_gb": round(peak_rss_mib() / 1024.0, 2),
        "superseded_figures": {
            "manuscript_vanilla_gradient_s": 8.95,
            "manuscript_integrated_gradients_s": 177.77,
            "note": ("The manuscript figures came from a September run on a "
                     "machine in an unknown thermal state and should be "
                     "replaced by the medians here. The difference must NOT "
                     "be reported as a speed improvement, because it cannot "
                     "be attributed: either thermal state at the time of the "
                     "old run or the switch from .backward() to "
                     "autograd.grad() -- which skips 22.6M parameter "
                     "gradients per backward pass -- could account for it. "
                     "Report these absolute figures, not a comparison."),
        },
        "order": order,
        "interpretation": (
            "Report median_s with median_95ci for each method, and the "
            "median cost ratio for the practicality discussion. Check "
            "drift_s_per_case is near zero and that the two order-effect "
            "medians agree before quoting anything."),
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))

    v, g = out["vanilla_gradient"], out["integrated_gradients"]
    r = out["cost_ratio_ig_over_vg"]
    oe = out["order_effect_vanilla_gradient"]
    print("\n--- controlled latency, n=%d ---" % len(work), file=sys.stderr)
    print("  %-22s median %8.3f s  95%% CI %s  drift %+.4f s/case"
          % ("Vanilla Gradient", v["median_s"], v["median_95ci"],
             v["drift_s_per_case"] or 0.0), file=sys.stderr)
    print("  %-22s median %8.3f s  95%% CI %s  drift %+.4f s/case"
          % ("Integrated Gradients", g["median_s"], g["median_95ci"],
             g["drift_s_per_case"] or 0.0), file=sys.stderr)
    print("  cost ratio IG/VG       median %.1fx  95%% CI %s"
          % (r["median"], r["median_95ci"]), file=sys.stderr)
    print("  peak RSS               %.2f GB" % out["peak_rss_gb"],
          file=sys.stderr)
    print("  order check (VG)       first %.3f s vs second %.3f s "
          "(should agree)"
          % (oe["median_when_measured_first"] or 0.0,
             oe["median_when_measured_second"] or 0.0), file=sys.stderr)
    print("\n  manuscript currently says VG 8.95 s, IG 177.77 s",
          file=sys.stderr)
    print("wrote %s" % args.out, file=sys.stderr)


if __name__ == "__main__":
    main()
