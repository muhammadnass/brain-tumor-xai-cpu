#!/usr/bin/env python3
"""
compare_paired.py -- paired comparison of two models on the same split.

    python scripts/compare_paired.py \
        --a results/metrics/cv_fold0.json --a-name "Proposed UNet3D" \
        --b results/metrics/baseline_segresnet_fold0.json \
        --b-name "SegResNet" \
        --out results/metrics/baseline_comparison.json

Both models were evaluated on the identical 162 validation cases, so the
observations are PAIRED. A paired analysis is the correct one here and is
substantially more powerful than treating the two sets as independent
samples, because it removes between-case difficulty, which dominates the
variance in segmentation Dice.

Reports the mean paired difference with a bootstrap confidence interval,
the median paired difference, Cohen's dz, a Wilcoxon signed-rank test,
and a win/loss/tie count. Refuses to run if the two files do not cover
exactly the same cases, since an unmatched comparison would silently
stop being paired.
"""

import argparse
import json
import math
import os
import random
import statistics
import sys


def per_case(path):
    with open(path) as fh:
        blob = json.load(fh)
    for key in ("per_case_dice", "per_case", "cases"):
        found = blob.get(key)
        if isinstance(found, dict) and found:
            return {k: float(v) for k, v in found.items()
                    if isinstance(v, (int, float))}, blob
        if isinstance(found, list) and found:
            out = {}
            for row in found:
                if isinstance(row, dict) and "dice" in row and "case" in row:
                    out[str(row["case"])] = float(row["dice"])
            if out:
                return out, blob
    sys.exit("no per-case Dice values found in %s" % path)


def bootstrap_paired(diffs, n=10000, seed=42, stat=statistics.mean):
    rng = random.Random(seed)
    boots = []
    for _ in range(n):
        sample = [diffs[rng.randrange(len(diffs))] for _ in diffs]
        boots.append(stat(sample))
    boots.sort()
    return round(boots[int(0.025 * n)], 4), round(boots[int(0.975 * n) - 1], 4)


def wilcoxon(diffs):
    """Two-sided Wilcoxon signed-rank. scipy where available."""
    try:
        from scipy.stats import wilcoxon as w
        r = w(diffs, alternative="two-sided", zero_method="wilcox")
        return {"statistic": float(r.statistic), "p_two_sided": float(r.pvalue),
                "backend": "scipy"}
    except ImportError:
        pass
    nz = [d for d in diffs if d != 0]
    if len(nz) < 10:
        return {"p_two_sided": None,
                "note": "too few non-zero differences for the normal "
                        "approximation; install scipy for an exact test"}
    order = sorted(nz, key=abs)
    ranks, i = [0.0] * len(order), 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and abs(order[j + 1]) == abs(order[i]):
            j += 1
        avg = (i + j) / 2.0 + 1
        for k in range(i, j + 1):
            ranks[k] = avg
        i = j + 1
    wp = sum(r for r, d in zip(ranks, order) if d > 0)
    n = len(nz)
    mu = n * (n + 1) / 4.0
    sd = math.sqrt(n * (n + 1) * (2 * n + 1) / 24.0)
    z = (wp - mu) / sd if sd else 0.0
    return {"statistic": round(wp, 1), "z": round(z, 4),
            "p_two_sided": math.erfc(abs(z) / math.sqrt(2)),
            "backend": "internal (normal approximation)"}


def fmt_p(p):
    if p is None:
        return "n/a"
    return "<1e-308" if p == 0.0 else "%.3g" % p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--a-name", default="A")
    ap.add_argument("--b-name", default="B")
    ap.add_argument("--out", default="results/metrics/baseline_comparison.json")
    args = ap.parse_args()

    a, blob_a = per_case(args.a)
    b, blob_b = per_case(args.b)

    only_a, only_b = set(a) - set(b), set(b) - set(a)
    if only_a or only_b:
        sys.exit(
            "ABORT: the two files do not cover the same cases, so the "
            "comparison would not be paired.\n  only in %s: %d (e.g. %s)\n"
            "  only in %s: %d (e.g. %s)"
            % (args.a_name, len(only_a), sorted(only_a)[:3],
               args.b_name, len(only_b), sorted(only_b)[:3]))

    cases = sorted(a)
    va = [a[c] for c in cases]
    vb = [b[c] for c in cases]
    diffs = [y - x for x, y in zip(va, vb)]          # b minus a

    wins_b = sum(1 for d in diffs if d > 0)
    wins_a = sum(1 for d in diffs if d < 0)
    ties = sum(1 for d in diffs if d == 0)
    sd = statistics.stdev(diffs) if len(diffs) > 1 else 0.0

    out = {
        "a": {"name": args.a_name, "file": os.path.abspath(args.a),
              "n": len(va), "mean_dice": round(statistics.mean(va), 4),
              "median_dice": round(statistics.median(va), 4),
              "best_epoch": blob_a.get("best_epoch"),
              "architecture": blob_a.get("architecture"),
              "features": blob_a.get("features"),
              "init_filters": blob_a.get("init_filters")},
        "b": {"name": args.b_name, "file": os.path.abspath(args.b),
              "n": len(vb), "mean_dice": round(statistics.mean(vb), 4),
              "median_dice": round(statistics.median(vb), 4),
              "best_epoch": blob_b.get("best_epoch"),
              "architecture": blob_b.get("architecture"),
              "features": blob_b.get("features"),
              "init_filters": blob_b.get("init_filters")},
        "paired": {
            "n_cases": len(diffs),
            "mean_difference": round(statistics.mean(diffs), 4),
            "mean_difference_95ci": bootstrap_paired(diffs),
            "median_difference": round(statistics.median(diffs), 4),
            "median_difference_95ci": bootstrap_paired(
                diffs, stat=statistics.median),
            "sd_of_differences": round(sd, 4),
            "cohens_dz": round(statistics.mean(diffs) / sd, 4) if sd else None,
            "wilcoxon": wilcoxon(diffs),
            "wins_b": wins_b, "wins_a": wins_a, "ties": ties,
            "direction": "%s higher on %d of %d cases"
                         % (args.b_name, wins_b, len(diffs)),
        },
        "note": ("Paired on identical validation cases. Cohen's dz is the "
                 "mean paired difference over the SD of the differences and "
                 "is not comparable to an unpaired Hedges g."),
        "per_case_difference": {c: round(d, 6) for c, d in zip(cases, diffs)},
    }

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)

    p = out["paired"]
    print("paired over %d identical cases" % p["n_cases"], file=sys.stderr)
    print("  %-24s mean Dice %.4f" % (args.a_name, out["a"]["mean_dice"]),
          file=sys.stderr)
    print("  %-24s mean Dice %.4f" % (args.b_name, out["b"]["mean_dice"]),
          file=sys.stderr)
    print("  mean difference   %+.4f  95%% CI %s"
          % (p["mean_difference"], p["mean_difference_95ci"]), file=sys.stderr)
    print("  median difference %+.4f  95%% CI %s"
          % (p["median_difference"], p["median_difference_95ci"]),
          file=sys.stderr)
    print("  Cohen's dz %s   Wilcoxon p = %s"
          % (p["cohens_dz"], fmt_p(p["wilcoxon"]["p_two_sided"])),
          file=sys.stderr)
    print("  %s wins %d, %s wins %d, ties %d"
          % (args.b_name, p["wins_b"], args.a_name, p["wins_a"], p["ties"]),
          file=sys.stderr)
    print("wrote %s" % args.out, file=sys.stderr)


if __name__ == "__main__":
    main()
