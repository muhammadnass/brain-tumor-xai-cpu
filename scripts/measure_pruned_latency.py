#!/usr/bin/env python3
"""
measure_pruned_latency.py -- paired, drift-corrected latency comparison of
the intact and pruned networks.

    python scripts/measure_pruned_latency.py \
        --checkpoint models/segmentation/unet3d_cv_fold0.pth \
        --repeats 25 \
        --out results/metrics/pruned_latency_fold0.json

WHY A SEPARATE SCRIPT
---------------------
prune_bottleneck.py timed the two models in sequence: five repeats of the
intact network, then five of the pruned one. Across folds 0, 1 and 2 the
intact medians rose (2.551 s, 2.647 s, 2.813 s) while the pruned medians
stayed flat near 2.25 s, and the resulting ratios spanned 0.801 to 0.923.
A spread that wide over a six-minute run on a passively cooled laptop is
consistent with clock drift rather than with a stable 20% difference, and
"somewhere between 8% and 20%" is not a figure to put in front of a
reviewer.

This script removes the confound in two ways:

  * Measurements are INTERLEAVED. Each repeat times both networks back to
    back, so the pair shares its thermal state and the difference within a
    pair is insensitive to drift between pairs.

  * The order WITHIN each pair alternates: (intact, pruned) on even
    repeats, (pruned, intact) on odd ones. Strict A,B,A,B alternation
    would still always measure the second model on a slightly warmer core.

Because the pairing is the design, the analysis is paired: the reported
quantity is the median within-pair difference with a bootstrap interval
and a Wilcoxon signed-rank test, matching the paired-comparison
convention already used in Section 4.7. The raw time series is written
out so any residual drift is auditable rather than assumed absent.

The pruned network is imported from prune_bottleneck rather than
redefined, so there is exactly one definition of the pruned architecture
in the project.

Read-only with respect to checkpoints. Nothing is retrained.
"""

import os

NUM_THREADS = int(os.environ.get("XAI_NUM_THREADS", "8"))
os.environ.setdefault("OMP_NUM_THREADS", str(NUM_THREADS))
os.environ.setdefault("MKL_NUM_THREADS", str(NUM_THREADS))

import argparse
import json
import random
import statistics
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_loader import load_trained_model
from prune_bottleneck import PrunedUNet3D


def percentile_bootstrap(values, statistic=statistics.median,
                         n_boot=10000, alpha=0.05, seed=42):
    """Percentile bootstrap interval for a statistic of one sample."""
    if len(values) < 2:
        return None
    rng = random.Random(seed)
    n = len(values)
    boots = []
    for _ in range(n_boot):
        boots.append(statistic([values[rng.randrange(n)] for _ in range(n)]))
    boots.sort()
    lo = boots[int((alpha / 2.0) * n_boot)]
    hi = boots[min(int((1.0 - alpha / 2.0) * n_boot), n_boot - 1)]
    return [round(lo, 6), round(hi, 6)]


def drift_slope(times):
    """Least-squares slope of time against repeat index, in seconds per
    repeat. A slope indistinguishable from zero is evidence that the run
    was thermally stable; a positive slope documents drift."""
    n = len(times)
    if n < 3:
        return None
    xbar = (n - 1) / 2.0
    ybar = statistics.mean(times)
    num = sum((i - xbar) * (t - ybar) for i, t in enumerate(times))
    den = sum((i - xbar) ** 2 for i in range(n))
    return round(num / den, 8) if den else None


def time_once(model, x):
    t0 = time.perf_counter()
    with torch.no_grad():
        model(x)
    return time.perf_counter() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--project-root")
    ap.add_argument("--repeats", type=int, default=25,
                    help="number of interleaved pairs")
    ap.add_argument("--warmup", type=int, default=3,
                    help="discarded warmup forward passes per model")
    ap.add_argument("--shape", default="1,4,128,128,128")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--threads", type=int, default=NUM_THREADS)
    ap.add_argument("--out",
                    default="results/metrics/pruned_latency.json")
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    shape = tuple(int(v) for v in args.shape.split(","))

    provenance = {}
    intact, meta = load_trained_model(args.checkpoint,
                                      project_root=args.project_root,
                                      provenance=provenance)
    intact.eval()
    pruned = PrunedUNet3D(intact)
    pruned.eval()

    n_intact = sum(p.numel() for p in intact.parameters())
    n_pruned = sum(p.numel() for p in pruned.parameters())

    # One fixed input for every repeat. Inference cost in this network is
    # input-independent -- there is no data-dependent control flow -- so
    # holding the volume constant removes a nuisance source of variance
    # without biasing the comparison.
    x = torch.randn(*shape)

    print("\nwarmup: %d pass(es) per model" % args.warmup, file=sys.stderr)
    for _ in range(args.warmup):
        time_once(intact, x)
        time_once(pruned, x)

    print("interleaved timing: %d pair(s), order alternating within pair"
          % args.repeats, file=sys.stderr)
    t_intact, t_pruned, order_log = [], [], []
    for i in range(args.repeats):
        if i % 2 == 0:
            ti = time_once(intact, x)
            tp = time_once(pruned, x)
            order_log.append("intact_first")
        else:
            tp = time_once(pruned, x)
            ti = time_once(intact, x)
            order_log.append("pruned_first")
        t_intact.append(ti)
        t_pruned.append(tp)
        print("  pair %2d/%d  %-13s intact %.4f s  pruned %.4f s  "
              "diff %+.4f s  ratio %.4f"
              % (i + 1, args.repeats, order_log[-1], ti, tp, ti - tp, tp / ti),
              file=sys.stderr)

    diffs = [a - b for a, b in zip(t_intact, t_pruned)]
    ratios = [b / a for a, b in zip(t_intact, t_pruned)]

    # Paired test, matching the convention used for the SegResNet comparison
    try:
        from scipy.stats import wilcoxon as _w
        stat, p = _w(t_intact, t_pruned, alternative="two-sided")
        wilcoxon = {"statistic": float(stat), "p_two_sided": float(p),
                    "backend": "scipy.stats.wilcoxon"}
    except Exception as exc:                        # scipy absent or n too small
        wilcoxon = {"error": "%s: %s" % (type(exc).__name__, exc)}

    # Order effect: if alternating the within-pair order mattered, the
    # difference would depend on which model went first.
    d_if = [d for d, o in zip(diffs, order_log) if o == "intact_first"]
    d_pf = [d for d, o in zip(diffs, order_log) if o == "pruned_first"]

    out = {
        "provenance": provenance,
        "checkpoint_meta": meta,
        "design": {
            "interleaved": True,
            "within_pair_order_alternated": True,
            "repeats": args.repeats,
            "warmup_per_model": args.warmup,
            "input_shape": list(shape),
            "threads": args.threads,
            "fixed_input": True,
            "rationale": ("Paired, interleaved design so that thermal drift "
                          "between pairs cannot bias the within-pair "
                          "difference; within-pair order alternates so "
                          "neither model is systematically measured on the "
                          "warmer core."),
        },
        "parameters": {"intact": n_intact, "pruned": n_pruned,
                       "removed": n_intact - n_pruned},
        "intact_s": {
            "median": round(statistics.median(t_intact), 4),
            "mean": round(statistics.mean(t_intact), 4),
            "sd": round(statistics.stdev(t_intact), 4) if len(t_intact) > 1 else 0.0,
            "min": round(min(t_intact), 4), "max": round(max(t_intact), 4),
            "median_95ci": percentile_bootstrap(t_intact),
            "drift_s_per_repeat": drift_slope(t_intact),
        },
        "pruned_s": {
            "median": round(statistics.median(t_pruned), 4),
            "mean": round(statistics.mean(t_pruned), 4),
            "sd": round(statistics.stdev(t_pruned), 4) if len(t_pruned) > 1 else 0.0,
            "min": round(min(t_pruned), 4), "max": round(max(t_pruned), 4),
            "median_95ci": percentile_bootstrap(t_pruned),
            "drift_s_per_repeat": drift_slope(t_pruned),
        },
        "paired": {
            "median_difference_s": round(statistics.median(diffs), 4),
            "median_difference_95ci": percentile_bootstrap(diffs),
            "mean_difference_s": round(statistics.mean(diffs), 4),
            "median_ratio_pruned_over_intact": round(statistics.median(ratios), 4),
            "median_ratio_95ci": percentile_bootstrap(ratios),
            "median_reduction_percent": round(
                100.0 * (1.0 - statistics.median(ratios)), 2),
            "pairs_pruned_faster": sum(1 for d in diffs if d > 0),
            "pairs_intact_faster": sum(1 for d in diffs if d < 0),
            "n_pairs": len(diffs),
            "wilcoxon": wilcoxon,
        },
        "order_effect": {
            "median_diff_when_intact_first":
                round(statistics.median(d_if), 4) if d_if else None,
            "median_diff_when_pruned_first":
                round(statistics.median(d_pf), 4) if d_pf else None,
            "note": ("These two medians should agree. A systematic gap "
                     "between them would mean the within-pair order still "
                     "influences the measurement, and the headline figure "
                     "should not be trusted."),
        },
        "raw": {"intact_s": [round(t, 6) for t in t_intact],
                "pruned_s": [round(t, 6) for t in t_pruned],
                "order": order_log},
        "interpretation": (
            "Report median_ratio_pruned_over_intact with its interval, or "
            "equivalently median_reduction_percent. The reduction arises "
            "because the removed stages hold 11.67 million weights that no "
            "longer have to be streamed from memory on each forward pass; "
            "it is a memory-bandwidth effect, not a reduction in arithmetic, "
            "since the removed stages operate at 4^3 and 8^3 resolution."),
    }

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)

    p = out["paired"]
    print("\n--- paired latency, n=%d pairs ---" % p["n_pairs"], file=sys.stderr)
    print("  intact median %.4f s  (drift %+.2e s/repeat)"
          % (out["intact_s"]["median"], out["intact_s"]["drift_s_per_repeat"] or 0.0),
          file=sys.stderr)
    print("  pruned median %.4f s  (drift %+.2e s/repeat)"
          % (out["pruned_s"]["median"], out["pruned_s"]["drift_s_per_repeat"] or 0.0),
          file=sys.stderr)
    print("  median within-pair difference %+.4f s  95%% CI %s"
          % (p["median_difference_s"], p["median_difference_95ci"]),
          file=sys.stderr)
    print("  median ratio %.4f  95%% CI %s   -> %.2f%% reduction"
          % (p["median_ratio_pruned_over_intact"], p["median_ratio_95ci"],
             p["median_reduction_percent"]), file=sys.stderr)
    print("  pruned faster in %d of %d pairs"
          % (p["pairs_pruned_faster"], p["n_pairs"]), file=sys.stderr)
    if "p_two_sided" in p["wilcoxon"]:
        print("  Wilcoxon signed-rank p = %.3g" % p["wilcoxon"]["p_two_sided"],
              file=sys.stderr)
    oe = out["order_effect"]
    print("  order check: intact-first %.4f s vs pruned-first %.4f s "
          "(should agree)"
          % (oe["median_diff_when_intact_first"] or 0.0,
             oe["median_diff_when_pruned_first"] or 0.0), file=sys.stderr)
    print("wrote %s" % args.out, file=sys.stderr)


if __name__ == "__main__":
    main()
