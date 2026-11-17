#!/usr/bin/env python3
"""
cache_bratsafrica.py -- preprocess the raw BraTS-Africa NIfTI cohort into
the same cache format as the Decathlon cohort.

    python scripts/cache_bratsafrica.py \
        --data-root ~/Downloads/BraTS-Africa \
        --out data/brats_dataset/cache_africa

Written to be interrupted. Each case is written atomically and skipped on
a later run if already present, and the manifest is rewritten after every
case, so a power cut costs at most the case in flight.

--------------------------------------------------------------------
THE TWO THINGS THAT MUST MATCH THE DECATHLON CACHE
--------------------------------------------------------------------
Channel order is [FLAIR, T1, T1c, T2], which for these filenames is
[t2f, t1n, t1c, t2w]. Note that t2f is FLAIR and t2w is T2; transposing
those two is silent and produces a plausible but meaningless result.

Label convention differs between the cohorts. BraTS-Africa follows the
standard BraTS scheme (1 = necrotic core, 2 = edema, 3 = enhancing),
whereas Decathlon Task01 is (1 = edema, 2 = non-enhancing, 3 = enhancing).
The remapping below is the BraTS-Africa one, taken from
scripts/evaluate_bratsafrica.py. train_model.py's remap_labels() is for
Decathlon and MUST NOT be used here.

Whichever remapping function is used, it is behaviour-tested before any
case is processed, and the label histogram of the first few cases is
checked against the expected post-remap scheme. Both abort on mismatch.
"""

import argparse
import json
import os
import shutil
import sys
import traceback

import numpy as np

# Channel order for the stacked volume. Index 0 is FLAIR.
MODALITY_ORDER = [("FLAIR", "t2f"), ("T1", "t1n"), ("T1c", "t1c"), ("T2", "t2w")]

# Post-remap scheme shared by both cohorts.
EXPECTED_LABELS = {0, 1, 2}

# Raw BraTS-Africa labels. Label 4 would indicate the pre-2023 BraTS
# convention, in which 4 rather than 3 denotes enhancing tumour.
EXPECTED_RAW_LABELS = {0, 1, 2, 3}

TARGET_SHAPE = (128, 128, 128)


def remap_bratsafrica(label):
    """BraTS-Africa -> the 3-class scheme. From evaluate_bratsafrica.py.

    1 (necrotic core) -> 1 (tumour)
    2 (edema)         -> 2 (edema)
    3 (enhancing)     -> 1 (tumour)
    """
    remapped = np.zeros_like(label)
    remapped[label == 2] = 2
    remapped[label == 1] = 1
    remapped[label == 3] = 1
    return remapped


def check_remap(fn, where):
    """Behaviour-test a remapping function before trusting it.

    This is what catches train_model.py's Decathlon remap_labels() being
    used here by mistake: it maps a different set of source labels, so it
    fails this test.
    """
    probe = np.array([0, 1, 2, 3], dtype=np.uint8)
    expect = np.array([0, 1, 2, 1], dtype=np.uint8)
    try:
        got = np.asarray(fn(probe)).astype(np.uint8).ravel()
    except Exception as exc:
        sys.exit("remapping function from %s raised on a probe volume: %r"
                 % (where, exc))
    if got.shape != expect.shape or not np.array_equal(got, expect):
        sys.exit(
            "ABORT: remapping function from %s does not implement the "
            "BraTS-Africa mapping.\n  input   0 1 2 3\n  expected %s\n"
            "  got      %s\nBraTS-Africa is 1=necrotic, 2=edema, "
            "3=enhancing. The Decathlon remapping must not be used here."
            % (where, " ".join(map(str, expect)), " ".join(map(str, got))))
    print("remapping verified (%s): 0->0, 1->1, 2->2, 3->1" % where,
          file=sys.stderr)


# ------------------------------------------------------------------ #
# Project preprocessing
# ------------------------------------------------------------------ #
def add_paths(project_root):
    root = os.path.abspath(project_root or os.getcwd())
    for path in (os.path.join(root, "src"), root):
        if os.path.isdir(path) and path not in sys.path:
            sys.path.insert(0, path)
    return root


def find_preprocessor(project_root, spec=None):
    """Locate and instantiate MRIPreprocessor from the project."""
    root = add_paths(project_root)
    cls = None
    if spec:
        if ":" not in spec:
            sys.exit("--preprocessor expects module.path:ClassName")
        mod_name, attr = spec.split(":", 1)
        try:
            mod = __import__(mod_name, fromlist=[attr])
        except ImportError as exc:
            sys.exit("cannot import %s (%r)" % (mod_name, exc))
        cls = getattr(mod, attr, None)
        if cls is None:
            sys.exit("%s has no %r" % (mod_name, attr))
        source = spec
    else:
        src = os.path.join(root, "src")
        base = src if os.path.isdir(src) else root
        hits = []
        for dirpath, _dirs, files in os.walk(base):
            if any(p in dirpath for p in (".git", "__pycache__", ".venv")):
                continue
            for fname in files:
                if not fname.endswith(".py"):
                    continue
                path = os.path.join(dirpath, fname)
                try:
                    with open(path, errors="replace") as fh:
                        if "class MRIPreprocessor" in fh.read():
                            hits.append(path)
                except OSError:
                    continue
        if not hits:
            sys.exit(
                "Could not find 'class MRIPreprocessor' anywhere under %s.\n"
                "Pass --preprocessor module.path:ClassName. This script will "
                "not substitute its own preprocessing: the two cohorts must "
                "be preprocessed identically or the comparison is invalid."
                % base)
        path = sorted(hits)[0]
        rel = os.path.relpath(path, base)[:-3].replace(os.sep, ".")
        try:
            mod = __import__(rel, fromlist=["MRIPreprocessor"])
        except ImportError as exc:
            sys.exit("found MRIPreprocessor in %s but could not import %s "
                     "(%r). Pass --preprocessor explicitly." % (path, rel, exc))
        cls = getattr(mod, "MRIPreprocessor")
        source = "%s (discovered at %s)" % (rel, path)

    obj = None
    errors = []
    for label, build in (("MRIPreprocessor(target_shape=%s)" % (TARGET_SHAPE,),
                          lambda: cls(target_shape=TARGET_SHAPE)),
                         ("MRIPreprocessor(%s)" % (TARGET_SHAPE,),
                          lambda: cls(TARGET_SHAPE)),
                         ("MRIPreprocessor()", lambda: cls())):
        try:
            obj = build()
            ctor = label
            break
        except Exception as exc:
            errors.append("%s -> %r" % (label, exc))
    if obj is None:
        sys.exit("Could not construct MRIPreprocessor. Attempts:\n  %s"
                 % "\n  ".join(errors))
    for name in ("normalize", "resize"):
        if not hasattr(obj, name):
            sys.exit("MRIPreprocessor has no %s(); available: %s"
                     % (name, [a for a in dir(obj) if not a.startswith("_")]))
    print("preprocessor: %s via %s" % (source, ctor), file=sys.stderr)
    return obj, "%s via %s" % (source, ctor)


def make_resizer(pre):
    """Pick the resize() call signature that returns the target shape."""
    probe = np.zeros((16, 16, 16), dtype=np.float32)
    attempts = [
        ("resize(vol, target_shape)", lambda v: pre.resize(v, TARGET_SHAPE)),
        ("resize(vol, target_shape=...)",
         lambda v: pre.resize(v, target_shape=TARGET_SHAPE)),
        ("resize(vol, size=...)", lambda v: pre.resize(v, size=TARGET_SHAPE)),
        ("resize(vol, shape=...)", lambda v: pre.resize(v, shape=TARGET_SHAPE)),
        ("resize(vol)", lambda v: pre.resize(v)),
    ]
    errors = []
    for label, fn in attempts:
        try:
            out = np.asarray(fn(probe))
        except Exception as exc:
            errors.append("%s -> %r" % (label, exc))
            continue
        if tuple(out.shape) == TARGET_SHAPE:
            print("resize signature: %s" % label, file=sys.stderr)
            return fn, label
        errors.append("%s -> returned shape %s" % (label, tuple(out.shape)))
    import inspect
    try:
        sig = str(inspect.signature(pre.resize))
    except (TypeError, ValueError):
        sig = "unavailable"
    sys.exit("No resize() call produced %s. resize signature: %s\nAttempts:\n  %s"
             % (TARGET_SHAPE, sig, "\n  ".join(errors)))


# ------------------------------------------------------------------ #
def discover_cases(data_root):
    """Return [(case_id, category, case_dir)] sorted by id."""
    out = []
    if not os.path.isdir(data_root):
        sys.exit("data root not found: %s" % data_root)
    for category in sorted(os.listdir(data_root)):
        cat_dir = os.path.join(data_root, category)
        if not os.path.isdir(cat_dir):
            continue
        for case_id in sorted(os.listdir(cat_dir)):
            case_dir = os.path.join(cat_dir, case_id)
            if os.path.isdir(case_dir):
                out.append((case_id, category, case_dir))
    return out


def modality_path(case_dir, case_id, suffix):
    for name in ("%s-%s.nii.gz" % (case_id, suffix),
                 "%s_%s.nii.gz" % (case_id, suffix),
                 "%s-%s.nii" % (case_id, suffix)):
        path = os.path.join(case_dir, name)
        if os.path.exists(path):
            return path
    return None


def save_npy(path, array):
    tmp = path + ".partial.npy"
    np.save(tmp, array, allow_pickle=False)
    os.replace(tmp, path)


def write_manifest(path, manifest):
    tmp = path + ".partial"
    with open(tmp, "w") as fh:
        json.dump(manifest, fh, indent=2)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--out", default="data/brats_dataset/cache_africa")
    ap.add_argument("--project-root")
    ap.add_argument("--preprocessor",
                    help="module.path:ClassName for MRIPreprocessor")
    ap.add_argument("--remap-fn",
                    help="module.path:function for the BraTS-Africa label "
                         "remapping. Defaults to the mapping from "
                         "evaluate_bratsafrica.py encoded in this file. "
                         "Whichever is used is behaviour-tested first.")
    ap.add_argument("--verify-cases", type=int, default=5,
                    help="cases whose label histogram is checked (default 5)")
    ap.add_argument("--limit", type=int, help="process at most this many cases")
    ap.add_argument("--recheck", action="store_true",
                    help="re-verify shapes of already-cached cases")
    args = ap.parse_args()

    try:
        import nibabel as nib
    except ImportError:
        sys.exit("nibabel is required to read NIfTI files (pip install nibabel)")
    from scipy import ndimage

    data_root = os.path.abspath(os.path.expanduser(args.data_root))
    out_root = os.path.abspath(os.path.expanduser(args.out))
    img_dir = os.path.join(out_root, "images")
    lab_dir = os.path.join(out_root, "labels")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lab_dir, exist_ok=True)
    manifest_path = os.path.join(out_root, "manifest.json")

    # Remapping, behaviour-tested before anything is written.
    if args.remap_fn:
        add_paths(args.project_root)
        if ":" not in args.remap_fn:
            sys.exit("--remap-fn expects module.path:function")
        mod_name, attr = args.remap_fn.split(":", 1)
        try:
            mod = __import__(mod_name, fromlist=[attr])
        except ImportError as exc:
            sys.exit("cannot import %s (%r)" % (mod_name, exc))
        remap = getattr(mod, attr, None)
        if remap is None:
            sys.exit("%s has no %r" % (mod_name, attr))
        remap_source = args.remap_fn
    else:
        remap = remap_bratsafrica
        remap_source = ("cache_bratsafrica.py, transcribed from "
                        "scripts/evaluate_bratsafrica.py lines 53-63")
    check_remap(remap, remap_source)

    pre, pre_source = find_preprocessor(args.project_root, args.preprocessor)
    resize, resize_sig = make_resizer(pre)

    cases = discover_cases(data_root)
    if args.limit:
        cases = cases[:args.limit]
    if not cases:
        sys.exit("no case directories found under %s" % data_root)
    by_cat = {}
    for _cid, cat, _d in cases:
        by_cat[cat] = by_cat.get(cat, 0) + 1
    print("\nfound %d cases: %s" % (len(cases), by_cat), file=sys.stderr)

    # Disk preflight.
    per_case = int(np.prod(TARGET_SHAPE)) * (4 * 2 + 1)
    todo = [c for c in cases
            if not (os.path.exists(os.path.join(img_dir, c[0] + ".npy"))
                    and os.path.exists(os.path.join(lab_dir, c[0] + ".npy")))]
    need = per_case * len(todo)
    free = shutil.disk_usage(out_root).free
    print("cache needs ~%.2f GB for %d remaining case(s); %.2f GB free"
          % (need / 2 ** 30, len(todo), free / 2 ** 30), file=sys.stderr)
    if need > free * 0.95:
        sys.exit("not enough free space: need ~%.2f GB, have %.2f GB"
                 % (need / 2 ** 30, free / 2 ** 30))

    manifest = {"cases": {}}
    if os.path.exists(manifest_path):
        try:
            with open(manifest_path) as fh:
                manifest = json.load(fh)
            manifest.setdefault("cases", {})
        except (OSError, ValueError):
            print("manifest unreadable, starting a new one", file=sys.stderr)
            manifest = {"cases": {}}

    manifest["preprocessing"] = {
        "target_shape": list(TARGET_SHAPE),
        "channel_order": [m for m, _ in MODALITY_ORDER],
        "channel_files": [s for _, s in MODALITY_ORDER],
        "image_dtype": "float16",
        "label_dtype": "uint8",
        "label_resample": "scipy.ndimage.zoom order=0",
        "preprocessor": pre_source,
        "resize_signature": resize_sig,
        "remap_source": remap_source,
        "remap": {"0": 0, "1": 1, "2": 2, "3": 1},
        "label_scheme": "0=background, 1=tumour, 2=edema",
        "data_root": data_root,
    }

    verified = 0
    done = skipped = failed = 0
    for idx, (case_id, category, case_dir) in enumerate(cases, 1):
        img_out = os.path.join(img_dir, case_id + ".npy")
        lab_out = os.path.join(lab_dir, case_id + ".npy")

        if os.path.exists(img_out) and os.path.exists(lab_out):
            if args.recheck:
                try:
                    a = np.load(img_out, mmap_mode="r")
                    b = np.load(lab_out, mmap_mode="r")
                    ok = (tuple(a.shape) == (4,) + TARGET_SHAPE
                          and tuple(b.shape) == TARGET_SHAPE)
                except Exception:
                    ok = False
                if not ok:
                    print("  [%3d] %s cached but malformed, reprocessing"
                          % (idx, case_id), file=sys.stderr)
                    os.remove(img_out)
                    os.remove(lab_out)
                else:
                    skipped += 1
                    continue
            else:
                manifest["cases"].setdefault(case_id, {})["category"] = category
                skipped += 1
                continue

        try:
            channels = []
            for _name, suffix in MODALITY_ORDER:
                path = modality_path(case_dir, case_id, suffix)
                if path is None:
                    raise FileNotFoundError(
                        "missing %s volume for %s" % (suffix, case_id))
                vol = np.asarray(nib.load(path).dataobj).astype(np.float32)
                vol = np.asarray(pre.normalize(vol), dtype=np.float32)
                vol = np.asarray(resize(vol), dtype=np.float32)
                if tuple(vol.shape) != TARGET_SHAPE:
                    raise ValueError("resize gave %s for %s"
                                     % (tuple(vol.shape), suffix))
                channels.append(vol)
            image = np.stack(channels, axis=0).astype(np.float16)

            seg_path = modality_path(case_dir, case_id, "seg")
            if seg_path is None:
                raise FileNotFoundError("missing seg volume for %s" % case_id)
            raw = np.asarray(nib.load(seg_path).dataobj).astype(np.int16)

            raw_vals = set(np.unique(raw).tolist())
            if not raw_vals <= EXPECTED_RAW_LABELS:
                extra = sorted(raw_vals - EXPECTED_RAW_LABELS)
                sys.exit(
                    "ABORT: %s has raw label value(s) %s, outside the expected "
                    "BraTS-Africa scheme {0,1,2,3}.\nIf 4 is present the cohort "
                    "uses the pre-2023 convention in which 4 denotes enhancing "
                    "tumour; the remapping here handles 3 and would silently "
                    "drop enhancing tumour to background. Resolve before "
                    "caching." % (case_id, extra))

            remapped = np.asarray(remap(raw))
            factors = [t / float(s) for t, s in zip(TARGET_SHAPE, remapped.shape)]
            label = ndimage.zoom(remapped, factors, order=0).astype(np.uint8)
            if tuple(label.shape) != TARGET_SHAPE:
                raise ValueError("label resize gave %s" % (tuple(label.shape),))

            vals, counts = np.unique(label, return_counts=True)
            hist = {int(v): int(c) for v, c in zip(vals, counts)}
            if not set(hist) <= EXPECTED_LABELS:
                sys.exit("ABORT: %s has post-remap labels %s, expected a "
                         "subset of %s" % (case_id, sorted(hist),
                                           sorted(EXPECTED_LABELS)))

            if verified < args.verify_cases:
                total = sum(hist.values())
                print("  [%3d] %s label histogram: %s" % (
                    idx, case_id,
                    {k: "%d (%.3f%%)" % (v, 100.0 * v / total)
                     for k, v in sorted(hist.items())}), file=sys.stderr)
                if 1 not in hist:
                    sys.exit(
                        "ABORT: %s contains no tumour voxels after remapping. "
                        "Every case in this cohort should have a tumour, so "
                        "this indicates the remapping or the source labels are "
                        "wrong." % case_id)
                verified += 1

            save_npy(img_out, image)
            save_npy(lab_out, label)
            manifest["cases"][case_id] = {
                "category": category,
                "source_dir": os.path.relpath(case_dir, data_root),
                "original_shape": list(raw.shape),
                "raw_label_values": sorted(raw_vals),
                "label_voxels": {str(k): v for k, v in sorted(hist.items())},
            }
            write_manifest(manifest_path, manifest)
            done += 1
            if done % 10 == 0 or idx == len(cases):
                print("  [%3d/%3d] cached %d, skipped %d"
                      % (idx, len(cases), done, skipped), file=sys.stderr)

        except SystemExit:
            raise
        except Exception:
            failed += 1
            print("  [%3d] FAILED %s:\n%s" % (idx, case_id,
                                              traceback.format_exc()),
                  file=sys.stderr)
            for path in (img_out, lab_out):
                if os.path.exists(path):
                    os.remove(path)

    counts = {}
    for rec in manifest["cases"].values():
        counts[rec.get("category", "?")] = counts.get(rec.get("category", "?"), 0) + 1
    manifest["summary"] = {"total": len(manifest["cases"]),
                           "by_category": counts}
    write_manifest(manifest_path, manifest)

    print("\ncached %d, skipped %d, failed %d" % (done, skipped, failed),
          file=sys.stderr)
    print("manifest: %s" % manifest_path, file=sys.stderr)
    print("by category: %s" % counts, file=sys.stderr)
    if failed:
        print("Rerun to retry the failed cases; completed ones are skipped.",
              file=sys.stderr)


if __name__ == "__main__":
    main()
