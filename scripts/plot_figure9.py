#!/usr/bin/env python3
"""
plot_figure9.py -- regenerate Figure 9 from cross_population.json.

    python scripts/plot_figure9.py \
        --results results/metrics/cross_population.json \
        --out figures/figure9.png

Every number and every annotation is read from the results file. Nothing
is hard-coded. The previous version of this figure carried two "n.s."
annotations that outlived the analysis they described; deriving the
labels from the data is the point of this script.

Panel A: Decathlon (each case scored once by the fold that held it out)
         against BraTS-Africa (each patient's fold scores averaged, so one
         independent observation per patient).
Panel B: BraTS-Africa split by diagnostic category.

Both panels annotate the mean difference with its bootstrap 95% CI and
the two-sided Mann-Whitney p, and label significance from the p value
actually present in the file.
"""

import argparse
import json
import os
import re
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def pretty_category(name):
    """'95_Glioma' -> 'Glioma'; '51_OtherNeoplasms' -> 'Other neoplasms'."""
    label = re.sub(r"^\d+[_\- ]*", "", name).replace("_", " ")
    label = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", label)
    return label[:1].upper() + label[1:].lower() if label else name


def sig_label(p, alpha=0.05):
    if p is None:
        return "n/a"
    if p == 0.0:
        return "p < 1e-308"
    if p >= alpha:
        return "p = %.3g, n.s" % p
    if p < 1e-4:
        return "p = %.1e" % p
    return "p = %.4g" % p


def panel(ax, groups, title, ylabel, comparison, palette):
    """groups: list of (label, values)."""
    data = [np.asarray(v, dtype=float) for _, v in groups]
    labels = ["%s\n(n = %d)" % (lab, len(v)) for lab, v in groups]

    parts = ax.violinplot(data, showextrema=False, widths=0.75)
    for body, colour in zip(parts["bodies"], palette):
        body.set_facecolor(colour)
        body.set_alpha(0.30)
        body.set_edgecolor("none")

    bp = ax.boxplot(data, widths=0.22, showfliers=False, patch_artist=True,
                    medianprops=dict(color="black", linewidth=1.6))
    for box, colour in zip(bp["boxes"], palette):
        box.set_facecolor(colour)
        box.set_alpha(0.75)
        box.set_edgecolor("black")

    rng = np.random.default_rng(42)
    for i, values in enumerate(data, 1):
        jitter = rng.uniform(-0.06, 0.06, len(values))
        ax.scatter(np.full(len(values), i) + jitter, values, s=5,
                   color="black", alpha=0.20, zorder=3, linewidths=0)

    for i, values in enumerate(data, 1):
        ax.scatter([i], [values.mean()], marker="D", s=34, zorder=5,
                   color="white", edgecolor="black", linewidths=1.1)

    ax.set_xticks(range(1, len(groups) + 1))
    ax.set_xticklabels(labels)
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", fontsize=11, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.25, linewidth=0.6)

    if comparison:
        p = comparison["mann_whitney"]["p_two_sided"]
        lo, hi = comparison["mean_difference_95ci"]
        text = ("\u0394 mean = %.4f (95%% CI %.4f to %.4f)\n%s   Hedges g = %s"
                % (comparison["mean_difference"], lo, hi, sig_label(p),
                   comparison["hedges_g"]))
        top = max(v.max() for v in data)
        y = top + 0.10 * (1.0 if top <= 1.0 else top)
        ax.plot([1, 1, len(groups), len(groups)],
                [y * 0.985, y, y, y * 0.985], color="black", linewidth=1.0)
        ax.text((1 + len(groups)) / 2.0, y * 1.01, text, ha="center",
                va="bottom", fontsize=8.5)
        ax.set_ylim(top=y * 1.25)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/metrics/cross_population.json")
    ap.add_argument("--out", default="figures/figure9.png")
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--manifest",
                    help="BraTS-Africa cache manifest.json, if the results "
                         "file carries no category breakdown")
    args = ap.parse_args()

    if not os.path.exists(args.results):
        sys.exit("results file not found: %s" % args.results)
    with open(args.results) as fh:
        res = json.load(fh)

    dec = [r["dice"] for r in res["per_case"]["decathlon"]
           if r.get("dice") is not None]
    afr_rows = res["per_case"]["brats_africa_by_patient"]
    afr = [r["dice"] for r in afr_rows if r.get("dice") is not None]
    if not dec or not afr:
        sys.exit("no per-case Dice values in %s" % args.results)

    primary = next((c for c in res["comparisons"]["metrics"]
                    if c["metric"] == "dice"), None)
    if primary is None:
        sys.exit("no Dice comparison in %s" % args.results)

    # Category split
    manifest_cats = {}
    if args.manifest and os.path.exists(args.manifest):
        with open(args.manifest) as fh:
            man = json.load(fh)
        manifest_cats = {k: v.get("category")
                         for k, v in man.get("cases", {}).items()}

    by_cat = res.get("brats_africa_by_category") or {}
    cat_groups = []
    if by_cat and manifest_cats:
        buckets = {}
        for row in afr_rows:
            cat = manifest_cats.get(os.path.splitext(row["case"])[0])
            if cat and row.get("dice") is not None:
                buckets.setdefault(cat, []).append(row["dice"])
        # Larger subgroup first, so the panel reads consistently.
        cat_groups = sorted(buckets.items(),
                            key=lambda kv: -len(kv[1]))
    elif by_cat:
        print("results file has category means but not per-case category "
              "assignments; pass --manifest to plot panel B from the "
              "underlying values", file=sys.stderr)

    cat_comparison = res.get("brats_africa_category_comparison")

    ncols = 2 if cat_groups else 1
    fig, axes = plt.subplots(1, ncols, figsize=(6.2 * ncols, 5.0))
    axes = np.atleast_1d(axes)

    panel(axes[0],
          [("Decathlon", dec), ("BraTS-Africa", afr)],
          "(A) Development vs external cohort", "Dice coefficient",
          primary, ["#4C72B0", "#C44E52"])

    if cat_groups:
        panel(axes[1],
              [(pretty_category(c), v) for c, v in cat_groups],
              "(B) BraTS-Africa by diagnostic category", "Dice coefficient",
              cat_comparison, ["#55A868", "#8172B2"])

    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=args.dpi, bbox_inches="tight")
    print("wrote %s" % args.out, file=sys.stderr)

    # The caption should match the figure, so emit one built from the data.
    p = primary["mann_whitney"]["p_two_sided"]
    lo, hi = primary["mean_difference_95ci"]
    cap = ("Figure 9. Cross-population Dice comparison. (A) Decathlon "
           "cross-validated results (n = %d, each case evaluated once by the "
           "fold that held it out) against BraTS-Africa (n = %d patients, "
           "each patient's three fold scores averaged): mean Dice %.4f "
           "against %.4f, a difference of %.4f (95%% CI %.4f to %.4f), "
           "Hedges g = %s, %s."
           % (len(dec), len(afr), np.mean(dec), np.mean(afr),
              primary["mean_difference"], lo, hi, primary["hedges_g"],
              sig_label(p)))
    if cat_groups and cat_comparison:
        pc = cat_comparison["mann_whitney"]["p_two_sided"]
        clo, chi = cat_comparison["mean_difference_95ci"]
        cap += (" (B) BraTS-Africa by diagnostic category, %s: difference "
                "%.4f (95%% CI %.4f to %.4f), %s."
                % (" against ".join(pretty_category(c) for c, _ in cat_groups),
                   cat_comparison["mean_difference"], clo, chi, sig_label(pc)))
    cap += (" Boxes show the interquartile range and median, white diamonds "
            "the mean, points individual cases.")
    print("\n--- caption generated from the plotted data ---\n%s" % cap,
          file=sys.stderr)
    with open(os.path.splitext(args.out)[0] + "_caption.txt", "w") as fh:
        fh.write(cap + "\n")


if __name__ == "__main__":
    main()
