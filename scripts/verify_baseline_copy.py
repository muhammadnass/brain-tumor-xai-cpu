#!/usr/bin/env python3
"""
verify_baseline_copy.py -- prove train_baseline.py differs from train_cv.py
only where it is supposed to.

    python scripts/verify_baseline_copy.py \
        --original scripts/train_cv.py \
        --copy scripts/train_baseline.py

Table 8 is only a controlled comparison if the two models were trained by
the same code. "I copied the file and changed the model" is an assertion;
this makes it checkable. Every differing line is classified against an
allowlist of intended changes, and anything outside it is reported as an
unexpected divergence.

Defaults cover the SegResNet substitution: the import, the model
construction, the checkpoint path, and the architecture label written into
the checkpoint. Add more with --allow (repeatable, case-insensitive
substring or regex with --regex).

Exit status is non-zero if any unexpected difference is found, so this can
gate the benchmark run.
"""

import argparse
import difflib
import re
import sys

DEFAULT_ALLOW = [
    "segresnet",            # the substituted architecture
    "init_filters",         # its width argument
    "unet3d",               # the architecture being replaced
    "baseline_segresnet",   # the distinct checkpoint path
    "architecture",         # the label recorded in the checkpoint
    "arch",
    "features",             # the constant no longer used
    "monai.networks.nets",  # the new import
    "cv_fold",              # the results-JSON path being replaced
    "train_cv",             # docstring references
    "train_baseline",
    "verify_baseline_copy",
    "single-fold",
    "section 4.7",
    "same code",
]

# Lines that differ only in these respects are noise, not divergence.
TRIVIAL = re.compile(r"^\s*(#.*)?$")


def load(path):
    try:
        with open(path) as fh:
            return fh.read().splitlines()
    except OSError as exc:
        sys.exit("cannot read %s: %s" % (path, exc))


def strip_docstring(lines):
    """Drop the leading module docstring.

    Documentation cannot change training behaviour, and diffing prose
    buries a real divergence in noise. Returns (body, n_skipped)."""
    triple_d = chr(34) * 3
    triple_s = chr(39) * 3
    i = 0
    while i < len(lines) and not lines[i].strip():
        i += 1
    if i >= len(lines):
        return lines, 0
    quote = lines[i].strip()[:3]
    if quote not in (triple_d, triple_s):
        return lines, 0
    stripped = lines[i].strip()
    if len(stripped) > 3 and stripped.endswith(quote):
        return lines[i + 1:], i + 1
    j = i + 1
    while j < len(lines) and quote not in lines[j]:
        j += 1
    return lines[j + 1:], j + 1


def allowed(line, patterns, use_regex):
    text = line.lower()
    for pat in patterns:
        if use_regex:
            if re.search(pat, line, re.I):
                return pat
        elif pat.lower() in text:
            return pat
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--original", required=True)
    ap.add_argument("--copy", required=True)
    ap.add_argument("--allow", action="append", default=[],
                    help="additional intended-change pattern (repeatable)")
    ap.add_argument("--regex", action="store_true",
                    help="treat --allow patterns as regular expressions")
    ap.add_argument("--no-defaults", action="store_true",
                    help="use only the patterns given with --allow")
    ap.add_argument("--include-docstring", action="store_true",
                    help="also diff the module docstring")
    ap.add_argument("--ignore-comments", action="store_true",
                    help="do not report differences in comments or blank lines")
    args = ap.parse_args()

    patterns = list(args.allow)
    if not args.no_defaults:
        patterns += DEFAULT_ALLOW

    a, b = load(args.original), load(args.copy)
    if not args.include_docstring:
        a, na = strip_docstring(a)
        b, nb = strip_docstring(b)
        if na or nb:
            print("skipping module docstrings (%d and %d lines): "
                  "documentation cannot change training behaviour"
                  % (na, nb), file=sys.stderr)
    print("original: %s (%d lines)" % (args.original, len(a)), file=sys.stderr)
    print("copy:     %s (%d lines)" % (args.copy, len(b)), file=sys.stderr)
    print("allowed patterns: %s\n" % ", ".join(patterns), file=sys.stderr)

    diff = list(difflib.unified_diff(a, b, lineterm="", n=0,
                                     fromfile=args.original, tofile=args.copy))
    changed, ok, unexpected = 0, 0, []
    for line in diff:
        if line.startswith(("---", "+++", "@@")):
            continue
        if not line.startswith(("+", "-")):
            continue
        body = line[1:]
        if args.ignore_comments and TRIVIAL.match(body):
            continue
        changed += 1
        why = allowed(body, patterns, args.regex)
        if why:
            ok += 1
            print("  ok  [%s] %s%s" % (why, line[0], body.strip()[:100]),
                  file=sys.stderr)
        else:
            unexpected.append(line)
            print("  ??  %s%s" % (line[0], body.strip()[:100]), file=sys.stderr)

    print("\n%d differing line(s): %d intended, %d unexplained"
          % (changed, ok, len(unexpected)), file=sys.stderr)

    if unexpected:
        print("\nUNEXPECTED DIVERGENCE. Every line above marked ?? is a "
              "difference between the two training scripts that is not "
              "accounted for by the architecture substitution. Either revert "
              "it, or add an --allow pattern and record in the manuscript "
              "that the two were not trained by identical code.", file=sys.stderr)
        return 1

    print("\nThe two scripts differ only where intended. Table 8 can state "
          "that both models were trained by the same code apart from the "
          "architecture.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
