#!/usr/bin/env python3
"""
compare_paired_3fold.py -- the three-fold paired comparison of the proposed
U-Net against the SegResNet baseline.

    python scripts/compare_paired_3fold.py \
        --out results/metrics/baseline_comparison_3fold.json

WHY
---
Reviewer 1: "The comparative evaluation against SegResNet is restricted to
a single fold (n=162), which is insufficient for establishing
generalizability or architectural superiority. A comprehensive three-fold
cross-validated benchmark across the entire dataset is required."

Reviewer 2 asks for the same thing more mildly and adds that the
conclusions on architectural efficiency "should therefore remain
appropriately qualified".

scripts/compare_paired.py answered the fold-0 question. This script
answers the three-fold one: it pools the per-case differences across all
folds so that every one of the 484 cases contributes exactly one paired
observation, scored by the fold that held it out -- the same pooling rule
Section 4.3 uses for the cross-validated Dice.

WHAT IT CHECKS BEFORE IT REPORTS
--------------------------------
A pooled paired test is only valid if the pairing is real, so the script
refuses to report until it has confirmed that:

  * each case appears in exactly one fold on each side;
  * the two sides cover identical case sets within each fold;
  * the per-fold validation counts match the folds file.

Any violation is fatal rather than warned about. A pooled statistic over a
broken pairing looks perfectly reasonable and means nothing.

Per-fold results are reported alongside the pooled figure, because a
pooled difference that masks disagreement between folds would itself be
worth knowing.
"""

import argparse
import json
import random
import statistics as st
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS = PROJECT_ROOT / "results/metrics"
FOLDS = PROJECT_ROOT / "data/brats_dataset/folds_seed42_3fold.json"

A_NAME = "Proposed UNet3D"
B_NAME = "SegResNet"


def percentile_bootstrap(values, statistic=st.median, n_boot=20000,
                         alpha=0.05, seed=42):
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


def wilcoxon(a, b):
    try:
        from scipy.stats import wilcoxon as _w
        stat, p = _w(a, b, alternative="two-sided")
        return {"statistic": float(stat), "p_two_sided": float(p),
                "backend": "scipy"}
    except Exception as exc:
        return {"error": "%s: %s" % (type(exc).__name__, exc)}


def cohens_dz(diffs):
    if len(diffs) < 2:
        return None
    sd = st.stdev(diffs)
    return round(st.mean(diffs) / sd, 6) if sd > 0 else None


def load_side(pattern, folds_present):
    """Read per-case Dice for one architecture, keyed case -> (fold, dice)."""
    out, per_fold = {}, {}
    for f in folds_present:
        path = RESULTS / (pattern % f)
        if not path.exists():
            return None, None, path
        doc = json.loads(path.read_text())
        pc = doc.get("per_case_dice")
        if not pc:
            sys.exit("%s has no per_case_dice block" % path)
        per_fold[f] = {
            "file": str(path),
            "n": len(pc),
            "mean_dice": round(doc.get("mean", float("nan")), 6),
            "median_dice": round(doc.get("median", float("nan")), 6),
            "best_epoch": doc.get("best_epoch"),
            "architecture": doc.get("architecture"),
            "features": doc.get("features"),
            "init_filters": doc.get("init_filters"),
            "threshold": doc.get("threshold"),
        }
        for cid, d in pc.items():
            if cid in out:
                sys.exit("case %s appears in more than one fold of %s -- the "
                         "pooling rule assumes each case is held out once"
                         % (cid, pattern))
            out[cid] = (f, float(d))
    return out, per_fold, None


def summarise(diffs, a_vals, b_vals, label):
    return {
        "label": label,
        "n_cases": len(diffs),
        "mean_dice_a": round(st.mean(a_vals), 6),
        "mean_dice_b": round(st.mean(b_vals), 6),
        "median_dice_a": round(st.median(a_vals), 6),
        "median_dice_b": round(st.median(b_vals), 6),
        "mean_difference": round(st.mean(diffs), 6),
        "mean_difference_95ci": percentile_bootstrap(diffs, st.mean),
        "median_difference": round(st.median(diffs), 6),
        "median_difference_95ci": percentile_bootstrap(diffs, st.median),
        "sd_of_differences": round(st.stdev(diffs), 6) if len(diffs) > 1 else 0.0,
        "cohens_dz": cohens_dz(diffs),
        "wilcoxon": wilcoxon(a_vals, b_vals),
        "wins_b": sum(1 for d in diffs if d > 0),
        "wins_a": sum(1 for d in diffs if d < 0),
        "ties": sum(1 for d in diffs if d == 0),
        "direction": "%s higher on %d of %d cases"
                     % (B_NAME, sum(1 for d in diffs if d > 0), len(diffs)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--out",
                    default=str(RESULTS / "baseline_comparison_3fold.json"))
    args = ap.parse_args()

    a, a_folds, miss = load_side("cv_fold%d.json", args.folds)
    if miss:
        sys.exit("missing %s -- the proposed model's fold is not on record"
                 % miss)
    b, b_folds, miss = load_side("baseline_segresnet_fold%d.json", args.folds)
    if miss:
        sys.exit("missing %s\nTrain it first:\n"
                 "  python scripts/train_baseline.py --fold %s --epochs 12"
                 % (miss, Path(str(miss)).stem[-1]))

    # ---- pairing integrity -------------------------------------
    only_a, only_b = set(a) - set(b), set(b) - set(a)
    if only_a or only_b:
        sys.exit("the two sides do not cover the same cases: %d only in %s, "
                 "%d only in %s. First few: %s | %s"
                 % (len(only_a), A_NAME, len(only_b), B_NAME,
                    sorted(only_a)[:5], sorted(only_b)[:5]))
    mismatched_fold = [c for c in a if a[c][0] != b[c][0]]
    if mismatched_fold:
        sys.exit("%d case(s) are assigned to different folds on the two "
                 "sides, so the comparison would not be controlled: %s"
                 % (len(mismatched_fold), mismatched_fold[:5]))

    if FOLDS.exists():
        cfg = json.loads(FOLDS.read_text())
        for entry in cfg["folds"]:
            f = entry["fold"]
            if f not in args.folds:
                continue
            expected = set(entry.get("val", entry.get("validation", [])))
            got = {c for c in a if a[c][0] == f}
            if expected != got:
                sys.exit("fold %d: scored cases do not match the folds file "
                         "(%d expected, %d scored, %d differing)"
                         % (f, len(expected), len(got),
                            len(expected ^ got)))
        print("pairing verified against %s" % FOLDS.name, file=sys.stderr)

    # ---- per fold and pooled -----------------------------------
    cases = sorted(a)
    per_fold_summary = {}
    for f in args.folds:
        ids = [c for c in cases if a[c][0] == f]
        av = [a[c][1] for c in ids]
        bv = [b[c][1] for c in ids]
        dv = [bb - aa for aa, bb in zip(av, bv)]
        per_fold_summary[str(f)] = summarise(dv, av, bv, "fold %d" % f)

    av = [a[c][1] for c in cases]
    bv = [b[c][1] for c in cases]
    dv = [bb - aa for aa, bb in zip(av, bv)]
    pooled = summarise(dv, av, bv, "pooled across folds %s"
                       % ",".join(str(f) for f in args.folds))

    out = {
        "a": {"name": A_NAME, "per_fold": a_folds},
        "b": {"name": B_NAME, "per_fold": b_folds},
        "folds_compared": args.folds,
        "per_fold": per_fold_summary,
        "pooled": pooled,
        "pairing_checks": {
            "identical_case_sets": True,
            "identical_fold_assignment": True,
            "validated_against_folds_file": FOLDS.exists(),
            "each_case_held_out_once": True,
        },
        "note": ("Paired on identical validation cases, each case scored once "
                 "by the fold that held it out. Cohen's dz is the mean paired "
                 "difference over the SD of the differences and is not "
                 "comparable to an unpaired Hedges g. A positive difference "
                 "favours %s." % B_NAME),
        "per_case_difference": {c: round(b[c][1] - a[c][1], 6) for c in cases},
        "per_case_fold": {c: a[c][0] for c in cases},
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))

    # ---- report --------------------------------------------------
    print("\n--- %s vs %s ---" % (A_NAME, B_NAME), file=sys.stderr)
    print("%-10s %7s %9s %9s %11s %-24s %9s %8s"
          % ("scope", "n", A_NAME[:9], B_NAME[:9], "mean diff",
             "median diff (95% CI)", "Wilcoxon", "B wins"),
          file=sys.stderr)
    for f in args.folds:
        s = per_fold_summary[str(f)]
        p = s["wilcoxon"].get("p_two_sided")
        print("%-10s %7d %9.4f %9.4f %+11.4f %-24s %9s %4d/%d"
              % ("fold %d" % f, s["n_cases"], s["mean_dice_a"],
                 s["mean_dice_b"], s["mean_difference"],
                 "%+.4f %s" % (s["median_difference"],
                               s["median_difference_95ci"]),
                 ("%.3g" % p) if p is not None else "n/a",
                 s["wins_b"], s["n_cases"]), file=sys.stderr)
    s = pooled
    p = s["wilcoxon"].get("p_two_sided")
    print("%-10s %7d %9.4f %9.4f %+11.4f %-24s %9s %4d/%d"
          % ("POOLED", s["n_cases"], s["mean_dice_a"], s["mean_dice_b"],
             s["mean_difference"],
             "%+.4f %s" % (s["median_difference"],
                           s["median_difference_95ci"]),
             ("%.3g" % p) if p is not None else "n/a",
             s["wins_b"], s["n_cases"]), file=sys.stderr)
    print("\n  pooled mean difference %+.4f  95%% CI %s"
          % (s["mean_difference"], s["mean_difference_95ci"]), file=sys.stderr)
    print("  Cohen's dz %s" % s["cohens_dz"], file=sys.stderr)
    print("  %s" % s["direction"], file=sys.stderr)
    print("wrote %s" % args.out, file=sys.stderr)

    if len(args.folds) < 3:
        print("\nNOTE: only fold(s) %s compared. The benchmark Reviewer 1 "
              "asked for needs all three."
              % ",".join(str(f) for f in args.folds), file=sys.stderr)


if __name__ == "__main__":
    main()
