#!/usr/bin/env python3
"""
evaluate_cross_population.py -- Section 4.3, rerun against the three
cross-validated fold models.

The Section 4.3 figures currently in the manuscript come from the retired
v3 model and are invalid. This script recomputes them.

What it does:

  1. Verifies the label convention of the BraTS-Africa cohort empirically
     from per-label voxel counts, rather than assuming it from the
     documentation (Sections 3.2 and 3.11). It REFUSES TO PROCEED if the
     observed convention does not match what was declared, because a
     silent label mismatch produces plausible-looking but meaningless
     Dice values -- the single most dangerous failure mode here.
  2. Evaluates each of the three fold models on the full BraTS-Africa
     cohort at the globally fixed threshold of 0.5. No tuning of any
     kind is performed.
  3. Evaluates each fold model on its own held-out Decathlon validation
     split, using the identical metric code, so the two sides of the
     comparison are computed the same way.
  4. Compares the two cohorts with effect sizes and bootstrap 95%
     confidence intervals alongside the significance test, and reports
     the smallest difference the comparison could have detected, so the
     result can be stated as an absence of detected difference rather
     than as demonstrated equivalence (Reviewer 1).

Usage:

    python scripts/evaluate_cross_population.py \
        --checkpoints results/checkpoints/cv_fold0_best.pt \
                      results/checkpoints/cv_fold1_best.pt \
                      results/checkpoints/cv_fold2_best.pt \
        --decathlon-cache data/brats_dataset/cache_128 \
        --folds data/brats_dataset/folds_seed42_3fold.json \
        --africa-cache data/brats_africa/cache_128 \
        --gradcam-layer-check off \
        --out results/metrics/cross_population.json

Add --limit 4 for a pilot run before committing hours to the full cohort.
Add --reuse-decathlon results/metrics/cv_fold{fold}.json to read the
already-computed Decathlon per-case Dice instead of recomputing it (note
that only Dice is reused; the other metrics are then reported for the
BraTS-Africa side only).

Read-only with respect to checkpoints and caches.
"""

import os

NUM_THREADS = int(os.environ.get("XAI_NUM_THREADS", "8"))
os.environ.setdefault("OMP_NUM_THREADS", str(NUM_THREADS))
os.environ.setdefault("MKL_NUM_THREADS", str(NUM_THREADS))

import argparse
import glob
import json
import math
import random
import statistics
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_loader import (load_trained_model, load_fold_split,
                          verify_fold_partition, _add_paths)

THRESHOLD = 0.5          # fixed globally, a priori -- do not make this an argument
TUMOUR_CLASS = 1

# The two caches store labels at DIFFERENT stages, and the two cohorts use
# DIFFERENT conventions. Getting this wrong produces plausible, meaningless
# Dice, so both sides are remapped explicitly here and verified before any
# inference runs.
#
#   Decathlon cache: RAW {0,1,2,3}. precompute_cache.py stores the labels
#       unremapped; train_cv.py remaps them on load. That remap produced
#       every figure in cv_fold*.json and the 0.7460 pooled result, so it
#       is the one that must be used here.
#   BraTS-Africa cache: ALREADY {0,1,2}, remapped at cache time by
#       cache_bratsafrica.py.
#
# The two conventions transpose classes 1 and 2:
#   Decathlon  1=edema -> 2,          2=non-enhancing -> 1, 3=enhancing -> 1
#   Africa     1=necrotic core -> 1,  2=edema -> 2,         3=enhancing -> 1
DECATHLON_RAW_LABELS = {0, 1, 2, 3}
AFRICA_CACHE_LABELS = {0, 1, 2}
REMAPPED_LABELS = {0, 1, 2}


def remap_decathlon(label):
    """Decathlon Task01 -> the 3-class scheme. Transcribed from train_cv.py."""
    out = np.zeros_like(label)
    out[(label == 2) | (label == 3)] = 1     # non-enhancing, enhancing -> tumour
    out[label == 1] = 2                      # edema -> edema
    return out


def remap_identity(label):
    """BraTS-Africa cache is already in the 3-class scheme."""
    return label


def check_remap(fn, where, probe_in, probe_out):
    """Behaviour-test a remapping before trusting it.

    This is what catches the Decathlon and Africa mappings being swapped:
    they disagree on classes 1 and 2, so each fails the other's test."""
    src = np.asarray(probe_in, dtype=np.uint8)
    want = np.asarray(probe_out, dtype=np.uint8)
    try:
        got = np.asarray(fn(src)).astype(np.uint8).ravel()
    except Exception as exc:
        sys.exit("remap from %s raised on a probe volume: %r" % (where, exc))
    if got.shape != want.shape or not np.array_equal(got, want):
        sys.exit(
            "ABORT: remap from %s does not implement the expected mapping.\n"
            "  input    %s\n  expected %s\n  got      %s"
            % (where, " ".join(map(str, src)), " ".join(map(str, want)),
               " ".join(map(str, got))))
    print("remap verified (%s): %s -> %s"
          % (where, list(map(int, src)), list(map(int, want))), file=sys.stderr)


# ------------------------------------------------------------------ #
# Case loading -- check against your cache format
# ------------------------------------------------------------------ #
def _label_path_for(image_path):
    """Sibling label for an image in a {images,labels} cache layout."""
    parent = os.path.dirname(image_path)
    if os.path.basename(parent) != "images":
        return None
    cand = os.path.join(os.path.dirname(parent), "labels",
                        os.path.basename(image_path))
    return cand if os.path.exists(cand) else None


def load_case(path, remap=None):
    """Return (image (1,C,D,H,W) float32, label (D,H,W) int64 or None).

    Handles both cache layouts: a single file holding image and label
    together, and the {images,labels} split written by
    precompute_cache.py and cache_bratsafrica.py.
    """
    img = lab = None
    if path.endswith(".npz"):
        with np.load(path) as z:
            keys = list(z.keys())
            ikey = next((k for k in ("image", "img", "data", "x") if k in keys),
                        keys[0])
            lkey = next((k for k in ("label", "seg", "mask", "y") if k in keys),
                        None)
            img = z[ikey]
            if lkey:
                lab = z[lkey]
    elif path.endswith(".npy"):
        img = np.load(path)
    elif path.endswith((".pt", ".pth")):
        obj = torch.load(path, map_location="cpu")
        if isinstance(obj, dict):
            img = obj.get("image")
            lab = obj.get("label")
        else:
            img = obj
    else:
        raise ValueError("unrecognised cache format: %s" % path)

    if lab is None:
        sibling = _label_path_for(path)
        if sibling is not None:
            lab = np.load(sibling)

    t = torch.as_tensor(np.ascontiguousarray(np.asarray(img))).float()
    while t.ndim < 5:
        t = t.unsqueeze(0)
    l = None
    if lab is not None:
        arr = np.asarray(lab)
        if remap is not None:
            arr = np.asarray(remap(arr))
        l = torch.as_tensor(np.ascontiguousarray(arr)).long()
        while l.ndim > 3:
            l = l.squeeze(0)
    return t, l


# ------------------------------------------------------------------ #
# Label-convention verification (Sections 3.2 / 3.11)
# ------------------------------------------------------------------ #
def label_histogram(paths, limit, remap=None):
    counts, checked = {}, 0
    for p in paths[:limit]:
        _, lab = load_case(p, remap=remap)
        if lab is None:
            continue
        vals, cnts = np.unique(lab.numpy(), return_counts=True)
        for v, c in zip(vals.tolist(), cnts.tolist()):
            counts[int(v)] = counts.get(int(v), 0) + int(c)
        checked += 1
    return counts, checked


def _report(name, stage, counts, checked):
    total = sum(counts.values()) or 1
    print("%s labels %s remap (%d case(s)):" % (name, stage, checked),
          file=sys.stderr)
    for k in sorted(counts):
        print("   label %d: %12d voxels (%.4f%%)"
              % (k, counts[k], 100.0 * counts[k] / total), file=sys.stderr)
    return {str(k): {"voxels": v, "fraction": round(v / total, 6)}
            for k, v in sorted(counts.items())}


def verify_labels(name, paths, expected_raw, remap, args):
    """Check the cache's own label set, then the set after remapping.

    Both cohorts must resolve to {0,1,2} before any inference runs, but
    they start from different label sets: the Decathlon cache holds raw
    {0,1,2,3} and the BraTS-Africa cache is already remapped to {0,1,2}."""
    print("", file=sys.stderr)
    raw_counts, checked = label_histogram(paths, args.label_check_cases)
    if checked == 0:
        print("%s: no labels found in cache -- cannot verify" % name,
              file=sys.stderr)
        return {"checked": 0}
    raw_report = _report(name, "as stored, before", raw_counts, checked)

    observed = set(raw_counts)
    if not observed <= set(expected_raw):
        msg = ("%s label set %s is not a subset of the expected stored set "
               "%s. The Decathlon cache stores RAW labels {0,1,2,3} and the "
               "BraTS-Africa cache stores already-remapped labels {0,1,2}; "
               "the two conventions transpose classes 1 and 2."
               % (name, sorted(observed), sorted(expected_raw)))
        if args.force_labels:
            print("WARNING (forced): " + msg, file=sys.stderr)
        else:
            sys.exit("ABORT: " + msg + "\nPass --force-labels only if you have "
                     "confirmed the mapping by hand.")

    post_counts, _ = label_histogram(paths, args.label_check_cases, remap=remap)
    post_report = _report(name, "after", post_counts, checked)
    post = set(post_counts)
    if not post <= REMAPPED_LABELS:
        msg = ("%s resolves to %s after remapping, expected a subset of %s."
               % (name, sorted(post), sorted(REMAPPED_LABELS)))
        if args.force_labels:
            print("WARNING (forced): " + msg, file=sys.stderr)
        else:
            sys.exit("ABORT: " + msg)
    if TUMOUR_CLASS not in post:
        sys.exit("ABORT: %s has no class-%d (tumour) voxels after remapping. "
                 "Every Dice value would be computed against an empty "
                 "reference." % (name, TUMOUR_CLASS))

    return {"checked": checked, "before_remap": raw_report,
            "after_remap": post_report}


# ------------------------------------------------------------------ #
# Metrics
# ------------------------------------------------------------------ #
def binary_metrics(pred, gt):
    """pred, gt: bool numpy arrays of the same shape."""
    tp = float(np.logical_and(pred, gt).sum())
    fp = float(np.logical_and(pred, ~gt).sum())
    fn = float(np.logical_and(~pred, gt).sum())
    tn = float(np.logical_and(~pred, ~gt).sum())
    eps = 1e-8
    return {
        "dice": (2 * tp) / (2 * tp + fp + fn + eps),
        "iou": tp / (tp + fp + fn + eps),
        "sensitivity": tp / (tp + fn + eps),
        "specificity": tn / (tn + fp + eps),
        "precision": tp / (tp + fp + eps),
        "gt_voxels": int(gt.sum()),
        "pred_voxels": int(pred.sum()),
    }


_HD_FN = None
_HD_UNAVAILABLE = False


def hd95(pred, gt):
    """95th-percentile Hausdorff distance, or None where undefined.

    Returns None when either mask is empty (genuinely undefined) -- the
    caller counts and reports those exclusions. If MONAI is unavailable
    the script says so once and loudly, rather than letting every HD95
    silently become None and disappear from the results table."""
    global _HD_FN, _HD_UNAVAILABLE
    if _HD_UNAVAILABLE:
        return None
    if _HD_FN is None:
        try:
            from monai.metrics import compute_hausdorff_distance
            _HD_FN = compute_hausdorff_distance
        except ImportError:
            _HD_UNAVAILABLE = True
            print("WARNING: MONAI not importable, so HD95 will be omitted "
                  "entirely from these results. Every other metric is "
                  "unaffected. Install MONAI if HD95 is needed.",
                  file=sys.stderr)
            return None
    if not pred.any() or not gt.any():
        return None
    p = torch.as_tensor(pred[None, None].astype(np.uint8))
    g = torch.as_tensor(gt[None, None].astype(np.uint8))
    v = _HD_FN(p, g, percentile=95).item()
    return None if math.isnan(v) or math.isinf(v) else float(v)


def evaluate(model, paths, label_id, remap, cohort, fold, progress,
             limit=None, tag=""):
    """Evaluate a cohort for one fold, skipping cases already recorded.

    Each completed case is appended to the progress sidecar before the
    next begins, so an interruption costs at most the case in flight."""
    paths = paths[:limit] if limit else paths
    todo = [p for p in paths
            if not progress.has(cohort, fold, os.path.basename(p))]
    if len(todo) < len(paths):
        print("  %s: %d of %d already done, %d remaining"
              % (tag, len(paths) - len(todo), len(paths), len(todo)),
              file=sys.stderr)

    for i, p in enumerate(todo, 1):
        case = os.path.basename(p)
        img, lab = load_case(p, remap=remap)
        if lab is None:
            print("  skipping %s (no label in cache)" % case, file=sys.stderr)
            continue
        with torch.no_grad():
            logits = model(img)
            prob = torch.softmax(logits, dim=1)[0, label_id]
        pred = (prob.numpy() >= THRESHOLD)
        gt = (lab.numpy() == label_id)
        m = binary_metrics(pred, gt)
        m["hd95"] = hd95(pred, gt)
        m["source"] = "computed"
        progress.append(cohort, fold, case, m)
        if i % 10 == 0 or i == len(todo):
            rows = progress.select(cohort, fold)
            print("  %s %d/%d  running mean Dice %.4f"
                  % (tag, i, len(todo),
                     statistics.mean(r["dice"] for r in rows)), file=sys.stderr)
    return progress.select(cohort, fold)


# ------------------------------------------------------------------ #
# Handling repeated measures on the BraTS-Africa side
#
# Each BraTS-Africa patient is scored by all three fold models, so the
# pooled rows are repeated measures on 146 patients, not 438 independent
# observations. The Decathlon rows ARE independent: each case is scored
# only by the fold that held it out. Treating the arms alike understates
# the standard errors on one side only.
#
# Three views are reported:
#   primary   -- Africa averaged within patient (n = 146) against pooled
#                Decathlon (n = 484). Both arms independent.
#   per_fold  -- fold f's Decathlon validation cases against all Africa
#                patients scored by fold f's model. Symmetric: every
#                observation is one patient scored by one model. A
#                consistency check; the folds share patients, so their
#                effect sizes are not pooled into a single interval.
#   naive     -- the uncorrected n = 438 pooling, retained only so the
#                size of the inflation is visible. NOT valid for inference.
# ------------------------------------------------------------------ #
METRIC_KEYS = ("dice", "iou", "sensitivity", "specificity", "precision",
               "hd95")


def average_within_patient(rows):
    grouped = {}
    for r in rows:
        grouped.setdefault(r["case"], []).append(r)
    out = []
    for case, group in sorted(grouped.items()):
        rec = {"case": case, "n_folds": len(group)}
        for key in METRIC_KEYS:
            vals = [g[key] for g in group if g.get(key) is not None]
            rec[key] = statistics.mean(vals) if vals else None
            if key == "hd95":
                rec["hd95_n_defined"] = len(vals)
        out.append(rec)
    return out


def build_comparisons(dec_rows, afr_rows):
    comps = []
    for metric in METRIC_KEYS:
        a = [r[metric] for r in dec_rows if r.get(metric) is not None]
        b = [r[metric] for r in afr_rows if r.get(metric) is not None]
        if len(a) < 2 or len(b) < 2:
            continue
        c = compare(metric, a, b, "decathlon", "brats_africa")
        if metric == "hd95":
            c["note"] = ("undefined where either mask is empty; %d/%d "
                         "Decathlon and %d/%d BraTS-Africa observations "
                         "excluded"
                         % (len(dec_rows) - len(a), len(dec_rows),
                            len(afr_rows) - len(b), len(afr_rows)))
        comps.append(c)
    return comps


def print_comparisons(title, comps, basis):
    print("\n--- %s ---" % title, file=sys.stderr)
    print("    %s" % basis, file=sys.stderr)
    for c in comps:
        print("%-12s Dec %.4f (n=%d)  Afr %.4f (n=%d)  diff %.4f %s  "
              "med diff %.4f  g=%s  p=%s  ties=%.3f"
              % (c["metric"], c["decathlon"]["mean"], c["decathlon"]["n"],
                 c["brats_africa"]["mean"], c["brats_africa"]["n"],
                 c["mean_difference"], c["mean_difference_95ci"],
                 c["median_difference"], c["hedges_g"],
                 fmt_p(c["mann_whitney"]["p_two_sided"]), c["tie_fraction"]),
              file=sys.stderr)


# ------------------------------------------------------------------ #
# Resumable progress sidecar
# ------------------------------------------------------------------ #
def sha256_of(path, chunk=1 << 20):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def build_fingerprint(args, dec_remap_src, hash_files=True):
    """Everything that, if changed, invalidates existing partial results.

    Merging cases computed under different checkpoints, remaps or
    thresholds would produce an aggregate that corresponds to no single
    configuration, and nothing in the output would reveal it."""
    def describe(path):
        rec = {"path": os.path.abspath(path),
               "size": os.path.getsize(path) if os.path.exists(path) else None}
        if hash_files and os.path.exists(path):
            rec["sha256"] = sha256_of(path)
        return rec

    return {
        "checkpoints": [describe(c) for c in args.checkpoints],
        "folds_file": describe(args.folds),
        "decathlon_cache": os.path.abspath(args.decathlon_cache),
        "africa_cache": os.path.abspath(args.africa_cache),
        "decathlon_remap": dec_remap_src,
        "africa_remap": "applied at cache time by cache_bratsafrica.py",
        "threshold": THRESHOLD,
        "tumour_class": TUMOUR_CLASS,
        "reuse_decathlon": args.reuse_decathlon or None,
        "force_labels": bool(args.force_labels),
    }


def _diff(a, b, prefix=""):
    out = []
    for key in sorted(set(a) | set(b)):
        va, vb = a.get(key, "<absent>"), b.get(key, "<absent>")
        if isinstance(va, dict) and isinstance(vb, dict):
            out.extend(_diff(va, vb, prefix + key + "."))
        elif isinstance(va, list) and isinstance(vb, list):
            if len(va) != len(vb):
                out.append("%s%s: %d entries -> %d entries"
                           % (prefix, key, len(va), len(vb)))
            else:
                for i, (x, y) in enumerate(zip(va, vb)):
                    if isinstance(x, dict) and isinstance(y, dict):
                        out.extend(_diff(x, y, "%s%s[%d]." % (prefix, key, i)))
                    elif x != y:
                        out.append("%s%s[%d]: %r -> %r" % (prefix, key, i, x, y))
        elif va != vb:
            out.append("%s%s: %r -> %r" % (prefix, key, va, vb))
    return out


class Progress:
    """Append-only per-case results, flushed and fsynced after every case.

    The first line records the run fingerprint. On resume the current
    fingerprint is compared against it and any difference aborts, rather
    than silently merging results from two configurations."""

    def __init__(self, path, fingerprint, restart=False):
        self.path = path
        self.fingerprint = fingerprint
        self.rows = []
        self.done = set()

        if restart and os.path.exists(path):
            os.replace(path, path + ".discarded")
            print("previous progress moved to %s.discarded" % path,
                  file=sys.stderr)

        if os.path.exists(path):
            self._resume()
        else:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            with open(path, "w") as fh:
                fh.write(json.dumps({"_header": fingerprint}) + "\n")
            print("progress file created: %s" % path, file=sys.stderr)

        self.fh = open(path, "a")

    def _resume(self):
        header = None
        bad = 0
        with open(self.path) as fh:
            for lineno, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except ValueError:
                    # A power cut mid-write leaves a truncated final line.
                    bad += 1
                    continue
                if "_header" in rec:
                    header = rec["_header"]
                    continue
                key = (rec["cohort"], rec["fold"], rec["case"])
                if key in self.done:
                    continue
                self.done.add(key)
                self.rows.append(rec)
        if header is None:
            sys.exit("progress file %s has no header record; delete it or "
                     "pass --restart" % self.path)
        differences = _diff(header, self.fingerprint)
        if differences:
            sys.exit(
                "ABORT: %s was written under a different configuration, so "
                "its results cannot be merged with this run:\n  %s\n"
                "Pass --restart to discard it, or --progress to use a "
                "different file."
                % (self.path, "\n  ".join(differences)))
        if bad:
            print("progress file had %d unreadable line(s), most likely a "
                  "truncated final write; those cases will be recomputed"
                  % bad, file=sys.stderr)
        print("resuming: %d case-result(s) already recorded in %s"
              % (len(self.rows), self.path), file=sys.stderr)

    def has(self, cohort, fold, case):
        return (cohort, fold, case) in self.done

    def append(self, cohort, fold, case, metrics):
        rec = dict(metrics)
        rec.update({"cohort": cohort, "fold": fold, "case": case})
        self.fh.write(json.dumps(rec) + "\n")
        self.fh.flush()
        os.fsync(self.fh.fileno())
        self.done.add((cohort, fold, case))
        self.rows.append(rec)

    def select(self, cohort, fold=None):
        return [r for r in self.rows
                if r["cohort"] == cohort and (fold is None or r["fold"] == fold)]

    def close(self):
        try:
            self.fh.close()
        except Exception:
            pass


# ------------------------------------------------------------------ #
# Comparison statistics
# ------------------------------------------------------------------ #
def bootstrap_ci(vals, stat=statistics.mean, n=5000, alpha=0.05, seed=42):
    if len(vals) < 2:
        return (None, None)
    rng = random.Random(seed)
    boots = []
    for _ in range(n):
        boots.append(stat([vals[rng.randrange(len(vals))] for _ in vals]))
    boots.sort()
    lo = boots[int((alpha / 2) * n)]
    hi = boots[int((1 - alpha / 2) * n) - 1]
    return (round(lo, 4), round(hi, 4))


def hedges_g(a, b):
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return None
    va, vb = statistics.variance(a), statistics.variance(b)
    sp = math.sqrt(((na - 1) * va + (nb - 1) * vb) / (na + nb - 2))
    if sp == 0:
        return None
    d = (statistics.mean(a) - statistics.mean(b)) / sp
    j = 1 - (3 / (4 * (na + nb) - 9))       # small-sample correction
    return d * j


def cliffs_delta(a, b):
    gt = lt = 0
    for x in a:
        for y in b:
            if x > y:
                gt += 1
            elif x < y:
                lt += 1
    n = len(a) * len(b)
    return (gt - lt) / n if n else None


def diff_ci(a, b, n=5000, seed=42, stat=statistics.mean):
    rng = random.Random(seed)
    boots = []
    for _ in range(n):
        sa = [a[rng.randrange(len(a))] for _ in a]
        sb = [b[rng.randrange(len(b))] for _ in b]
        boots.append(stat(sa) - stat(sb))
    boots.sort()
    return round(boots[int(0.025 * n)], 4), round(boots[int(0.975 * n) - 1], 4)


def mann_whitney(a, b):
    """Two-sided Mann-Whitney U.

    Uses scipy where available: its tie handling and small-p accuracy are
    better than a normal approximation, which underflows to exactly 0.0
    for |z| beyond about 8. The internal fallback uses erfc rather than
    1 - erf for the same reason.

    Note that this tests a rank shift, not a difference of means. A large
    p alongside a mean-difference CI that excludes zero is not necessarily
    a contradiction: it is what two distributions differing in tail shape
    rather than location look like. Compare the median difference and the
    tie fraction reported alongside before concluding either way.
    """
    try:
        from scipy.stats import mannwhitneyu
        r = mannwhitneyu(a, b, alternative="two-sided")
        return {"U": round(float(r.statistic), 1),
                "p_two_sided": float(r.pvalue), "backend": "scipy"}
    except ImportError:
        pass

    combined = sorted([(v, 0) for v in a] + [(v, 1) for v in b])
    ranks, i = [0.0] * len(combined), 0
    while i < len(combined):
        j = i
        while j + 1 < len(combined) and combined[j + 1][0] == combined[i][0]:
            j += 1
        avg = (i + j) / 2.0 + 1
        for k in range(i, j + 1):
            ranks[k] = avg
        i = j + 1
    na, nb = len(a), len(b)
    ra = sum(r for r, (_, g) in zip(ranks, combined) if g == 0)
    u = ra - na * (na + 1) / 2.0
    mu = na * nb / 2.0
    tie_term = 0.0
    i = 0
    while i < len(combined):
        j = i
        while j + 1 < len(combined) and combined[j + 1][0] == combined[i][0]:
            j += 1
        t = j - i + 1
        tie_term += t ** 3 - t
        i = j + 1
    n = na + nb
    var = (na * nb / 12.0) * ((n + 1) - tie_term / (n * (n - 1)))
    if var <= 0:
        return {"U": u, "p_two_sided": None, "backend": "internal"}
    z = (u - mu) / math.sqrt(var)
    p = math.erfc(abs(z) / math.sqrt(2))      # not 2*(1-erf(...)): underflows
    return {"U": round(u, 1), "z": round(z, 4), "p_two_sided": p,
            "backend": "internal (normal approximation)"}


def fmt_p(p):
    if p is None:
        return "n/a"
    if p == 0.0:
        return "<1e-308"
    return "%.3g" % p


def tie_fraction(a, b):
    """Fraction of cross-pairs holding exactly equal values. High values
    mean the rank test is dominated by ties."""
    from collections import Counter
    cb = Counter(b)
    shared = sum(cb.get(v, 0) for v in a)
    return round(shared / float(len(a) * len(b)), 6)


def detectable_effect(na, nb, alpha=0.05, power=0.80):
    """Smallest standardised effect this comparison could detect, for the
    explicit statement of limited power that Reviewer 1 asked for."""
    z_a, z_b = 1.959964, 0.8416212
    if na == 0 or nb == 0:
        return None
    return (z_a + z_b) * math.sqrt(1.0 / na + 1.0 / nb)


def compare(name, a, b, label_a, label_b):
    out = {
        "metric": name,
        label_a: {"n": len(a), "mean": round(statistics.mean(a), 4),
                  "sd": round(statistics.stdev(a), 4) if len(a) > 1 else 0.0,
                  "median": round(statistics.median(a), 4),
                  "mean_95ci": bootstrap_ci(a)},
        label_b: {"n": len(b), "mean": round(statistics.mean(b), 4),
                  "sd": round(statistics.stdev(b), 4) if len(b) > 1 else 0.0,
                  "median": round(statistics.median(b), 4),
                  "mean_95ci": bootstrap_ci(b)},
        "mean_difference": round(statistics.mean(a) - statistics.mean(b), 4),
        "mean_difference_95ci": diff_ci(a, b),
        "median_difference": round(
            statistics.median(a) - statistics.median(b), 4),
        "median_difference_95ci": diff_ci(a, b, stat=statistics.median),
        "tie_fraction": tie_fraction(a, b),
        "hedges_g": round(hedges_g(a, b), 4) if hedges_g(a, b) is not None else None,
        "cliffs_delta": round(cliffs_delta(a, b), 4),
        "mann_whitney": mann_whitney(a, b),
        "min_detectable_g_at_80pct_power": round(
            detectable_effect(len(a), len(b)), 3),
    }
    return out


# ------------------------------------------------------------------ #
def image_dir(cache):
    """Where the image volumes live, for either cache layout."""
    nested = os.path.join(cache, "images")
    return nested if os.path.isdir(nested) else cache


def collect(cache, ids=None):
    base = image_dir(cache)
    if ids:
        out = []
        for cid in ids:
            hits = [h for h in glob.glob(os.path.join(base, "*%s*" % cid))
                    if not h.endswith(".partial")]
            if hits:
                out.append(sorted(hits)[0])
        return out
    return sorted(h for h in
                  glob.glob(os.path.join(base, "*.np[yz]"))
                  + glob.glob(os.path.join(base, "*.pt"))
                  if not h.endswith(".partial"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", nargs="+", required=True,
                    help="fold checkpoints, in fold order")
    ap.add_argument("--decathlon-cache", required=True)
    ap.add_argument("--folds", required=True)
    ap.add_argument("--africa-cache", required=True)
    ap.add_argument("--label-check-cases", type=int, default=25)
    ap.add_argument("--force-labels", action="store_true")
    ap.add_argument("--decathlon-remap-fn",
                    help="module.path:function for the Decathlon label remap. "
                         "Defaults to train_cv:remap if importable, else the "
                         "transcription in this file. Behaviour-tested either "
                         "way.")
    ap.add_argument("--reuse-decathlon",
                    help="template such as results/metrics/cv_fold{fold}.json")
    ap.add_argument("--limit", type=int, help="cases per cohort, for a pilot run")
    ap.add_argument("--threads", type=int, default=NUM_THREADS)
    ap.add_argument("--project-root",
                    help="project root containing src/ai_models (default: cwd)")
    ap.add_argument("--features", type=int, nargs="+",
                    help="override layer widths; normally read from the "
                         "checkpoint, so leave unset")
    ap.add_argument("--out", default="results/metrics/cross_population.json")
    ap.add_argument("--progress",
                    help="per-case results sidecar (default: --out with a "
                         ".progress.jsonl suffix)")
    ap.add_argument("--restart", action="store_true",
                    help="discard any existing progress file and start over")
    ap.add_argument("--no-hash", action="store_true",
                    help="fingerprint checkpoints by path and size only, "
                         "skipping the SHA-256 read")
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    # Project paths must be on sys.path before the remap is resolved, which
    # happens well before the first model is loaded.
    _add_paths(args.project_root)
    provenance = {"threshold": THRESHOLD, "tumour_class": TUMOUR_CLASS,
                  "checkpoints": args.checkpoints}

    africa_paths = collect(args.africa_cache)
    if not africa_paths:
        sys.exit("no cases found in %s" % args.africa_cache)
    # Splits must partition the cohort before anything is evaluated.
    verify_fold_partition(args.folds)

    # Resolve the Decathlon remap. Preference is the project's own function,
    # since that is what produced every figure in cv_fold*.json; the
    # transcription here is the fallback. Either way it is behaviour-tested.
    dec_remap, dec_remap_src = None, None
    for spec in ([args.decathlon_remap_fn] if args.decathlon_remap_fn
                 else ["scripts.train_cv:remap", "train_cv:remap"]):
        try:
            mod_name, attr = spec.split(":", 1)
            mod = __import__(mod_name, fromlist=[attr])
            candidate = getattr(mod, attr, None)
        except Exception:
            candidate = None
        if candidate is not None:
            dec_remap, dec_remap_src = candidate, spec
            break
    if dec_remap is None:
        if args.decathlon_remap_fn:
            sys.exit("could not import %s" % args.decathlon_remap_fn)
        dec_remap = remap_decathlon
        dec_remap_src = ("evaluate_cross_population.py, transcribed from "
                         "train_cv.py remap()")
    check_remap(dec_remap, dec_remap_src, [0, 1, 2, 3], [0, 2, 1, 1])
    check_remap(remap_identity, "BraTS-Africa cache (already remapped)",
                [0, 1, 2], [0, 1, 2])
    provenance["decathlon_remap"] = dec_remap_src
    provenance["africa_remap"] = ("none at load time; applied at cache time "
                                  "by cache_bratsafrica.py")

    # Label conventions verified before any inference is run. The two caches
    # store labels at different stages, so they are checked against
    # different expected sets and then both against {0,1,2} after remapping.
    label_report = {
        "brats_africa": verify_labels("BraTS-Africa", africa_paths,
                                      AFRICA_CACHE_LABELS, remap_identity,
                                      args),
    }
    _, dec_ids0, _ = load_fold_split(args.folds, 0, provenance)
    label_report["decathlon"] = verify_labels(
        "Decathlon", collect(args.decathlon_cache, dec_ids0),
        DECATHLON_RAW_LABELS, dec_remap, args)

    fingerprint = build_fingerprint(args, dec_remap_src,
                                    hash_files=not args.no_hash)
    progress_path = args.progress or (os.path.splitext(args.out)[0]
                                      + ".progress.jsonl")
    progress = Progress(progress_path, fingerprint, restart=args.restart)
    provenance["progress_file"] = progress_path
    provenance["fingerprint"] = fingerprint

    n_africa = len(africa_paths[:args.limit] if args.limit else africa_paths)
    expected = 0
    per_fold = {}

    for f, ckpt in enumerate(args.checkpoints):
        print("\n=== fold %d: %s ===" % (f, ckpt), file=sys.stderr)
        _, val_ids, _ = load_fold_split(args.folds, f, provenance)
        dec_paths = collect(args.decathlon_cache, val_ids)
        n_dec = len(dec_paths[:args.limit] if args.limit else dec_paths)

        africa_done = all(progress.has("brats_africa", f, os.path.basename(p))
                          for p in (africa_paths[:args.limit] if args.limit
                                    else africa_paths))
        dec_done = all(progress.has("decathlon", f, os.path.basename(p))
                       for p in (dec_paths[:args.limit] if args.limit
                                 else dec_paths))

        model = None
        if not (africa_done and dec_done):
            model, meta = load_trained_model(
                ckpt, project_root=args.project_root, features=args.features,
                provenance=provenance, expect_threshold=THRESHOLD)
            provenance.pop("checkpoint_meta", None)
            provenance.setdefault("per_fold_checkpoint_meta", {})[str(f)] = meta
        else:
            print("  fold %d complete in the progress file; model not loaded"
                  % f, file=sys.stderr)

        africa_rows = evaluate(model, africa_paths, TUMOUR_CLASS,
                               remap_identity, "brats_africa", f, progress,
                               args.limit, "africa")

        if args.reuse_decathlon:
            path = args.reuse_decathlon.format(fold=f)
            if not os.path.exists(path):
                sys.exit("--reuse-decathlon given but %s does not exist" % path)
            with open(path) as fh:
                blob = json.load(fh)
            found = blob.get("per_case") or blob.get("cases") or blob
            reused = []
            if isinstance(found, dict):
                reused = [(k, float(v)) for k, v in found.items()
                          if isinstance(v, (int, float))]
            elif isinstance(found, list):
                reused = [(r.get("case"), float(r["dice"])) for r in found
                          if "dice" in r]
            if not reused:
                sys.exit("no per-case Dice values found in %s" % path)
            for case, dice in reused:
                if not progress.has("decathlon", f, case):
                    progress.append("decathlon", f, case,
                                    {"dice": dice, "source": "reused:%s" % path})
            dec_rows = progress.select("decathlon", f)
            n_dec = len(reused)
            print("  reused %d Decathlon per-case Dice value(s) from %s"
                  % (len(reused), path), file=sys.stderr)
        else:
            dec_rows = evaluate(model, dec_paths, TUMOUR_CLASS, dec_remap,
                                "decathlon", f, progress, args.limit,
                                "decathlon")

        expected += n_africa + n_dec
        per_fold[str(f)] = {
            "checkpoint": ckpt,
            "africa_n": len(africa_rows),
            "africa_mean_dice": round(
                statistics.mean(r["dice"] for r in africa_rows), 4),
            "decathlon_n": len(dec_rows),
            "decathlon_mean_dice": round(
                statistics.mean(r["dice"] for r in dec_rows), 4),
        }
        print("  fold %d: Decathlon %.4f | BraTS-Africa %.4f"
              % (f, per_fold[str(f)]["decathlon_mean_dice"],
                 per_fold[str(f)]["africa_mean_dice"]), file=sys.stderr)
        del model

    # Aggregates are computed from the sidecar, never from in-memory state,
    # so a resumed run and an uninterrupted one produce the same output.
    africa_all = progress.select("brats_africa")
    dec_all = progress.select("decathlon")
    recorded = len(africa_all) + len(dec_all)
    progress.close()

    if recorded < expected:
        sys.exit(
            "INCOMPLETE: %d of %d case-results recorded. No aggregate written, "
            "because a partial aggregate is indistinguishable from a complete "
            "one once in the file. Rerun the same command to continue from "
            "%s." % (recorded, expected, progress_path))

    categories = {}
    man_path = os.path.join(args.africa_cache, "manifest.json")
    if os.path.exists(man_path):
        try:
            with open(man_path) as fh:
                man = json.load(fh)
            for cid, rec in man.get("cases", {}).items():
                categories[cid] = rec.get("category")
            provenance["africa_manifest"] = man.get("summary")
        except (OSError, ValueError):
            pass

    africa_by_patient = average_within_patient(africa_all)
    primary = build_comparisons(dec_all, africa_by_patient)
    primary_basis = ("Decathlon n=%d (each case scored once, by the fold that "
                     "held it out) vs BraTS-Africa n=%d patients (each "
                     "patient's fold scores averaged). Both arms independent."
                     % (len(dec_all), len(africa_by_patient)))
    print_comparisons("Section 4.3 primary", primary, primary_basis)

    per_fold_comparisons = {}
    for f in range(len(args.checkpoints)):
        d_f = [r for r in dec_all if r["fold"] == f]
        a_f = [r for r in africa_all if r["fold"] == f]
        if len(d_f) >= 2 and len(a_f) >= 2:
            per_fold_comparisons[str(f)] = build_comparisons(d_f, a_f)
    if per_fold_comparisons:
        print("\n--- per-fold consistency check (Dice) ---", file=sys.stderr)
        gs = []
        for f, comps in sorted(per_fold_comparisons.items()):
            d = next((c for c in comps if c["metric"] == "dice"), None)
            if not d:
                continue
            gs.append(d["hedges_g"])
            print("  fold %s: Dec %.4f (n=%d) vs Afr %.4f (n=%d)  diff %.4f  "
                  "g=%s  p=%s"
                  % (f, d["decathlon"]["mean"], d["decathlon"]["n"],
                     d["brats_africa"]["mean"], d["brats_africa"]["n"],
                     d["mean_difference"], d["hedges_g"],
                     fmt_p(d["mann_whitney"]["p_two_sided"])), file=sys.stderr)
        gs = [g for g in gs if g is not None]
        if gs:
            print("  Dice effect size across folds: %s (range %.4f to %.4f)"
                  % ([round(g, 4) for g in gs], min(gs), max(gs)),
                  file=sys.stderr)

    naive = build_comparisons(dec_all, africa_all)

    by_category = {}
    category_comparison = None
    if categories:
        buckets = {}
        for row in africa_by_patient:
            cat = categories.get(os.path.splitext(row["case"])[0])
            if cat:
                buckets.setdefault(cat, []).append(row["dice"])
        by_category = {
            cat: {"n": len(v), "mean_dice": round(statistics.mean(v), 4),
                  "sd": round(statistics.stdev(v), 4) if len(v) > 1 else 0.0,
                  "median": round(statistics.median(v), 4),
                  "mean_95ci": bootstrap_ci(v)}
            for cat, v in sorted(buckets.items())}
        print("\nBraTS-Africa by category (one observation per patient):",
              file=sys.stderr)
        for cat, st in by_category.items():
            print("  %-20s n=%-4d mean Dice %.4f (95%% CI %s)"
                  % (cat, st["n"], st["mean_dice"], st["mean_95ci"]),
                  file=sys.stderr)
        # Between-category comparison, so a figure or caption reporting one
        # has a computed value behind it rather than a carried-over number.
        if len(buckets) == 2:
            (ca, va), (cb, vb) = sorted(buckets.items())
            category_comparison = compare("dice", va, vb, ca, cb)
            category_comparison["categories"] = [ca, cb]
            print("  %s vs %s: diff %.4f %s  g=%s  p=%s"
                  % (ca, cb, category_comparison["mean_difference"],
                     category_comparison["mean_difference_95ci"],
                     category_comparison["hedges_g"],
                     fmt_p(category_comparison["mann_whitney"]["p_two_sided"])),
                  file=sys.stderr)

    out = {
        "provenance": provenance,
        "label_verification": label_report,
        "per_fold": per_fold,
        "pooled": {"decathlon_n": len(dec_all),
                   "brats_africa_observations": len(africa_all),
                   "brats_africa_patients": len(africa_by_patient)},
        "comparisons": {"basis": primary_basis, "metrics": primary},
        "per_fold_comparisons": per_fold_comparisons,
        "naive_pooled_comparisons": {
            "warning": ("Treats %d repeated measures on %d BraTS-Africa "
                        "patients as independent. Standard errors are "
                        "understated on the Africa arm only. Retained for "
                        "transparency; do not cite these p-values."
                        % (len(africa_all), len(africa_by_patient))),
            "metrics": naive},
        "brats_africa_by_category": by_category,
        "brats_africa_category_comparison": category_comparison,
        "per_case": {"decathlon": dec_all,
                     "brats_africa_by_patient": africa_by_patient,
                     "brats_africa_all_folds": africa_all},
        "interpretation_note": (
            "min_detectable_g_at_80pct_power gives the smallest standardised "
            "difference the comparison could have detected. Mann-Whitney "
            "tests a rank shift, not a difference of means; where the two "
            "disagree, compare median_difference and tie_fraction before "
            "drawing a conclusion."),
    }

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)

    print("wrote %s" % args.out, file=sys.stderr)


if __name__ == "__main__":
    main()
