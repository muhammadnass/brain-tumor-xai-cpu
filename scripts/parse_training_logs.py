#!/usr/bin/env python3
"""
parse_training_logs.py -- extract training-time figures for Section 4.6.

    python scripts/parse_training_logs.py results/metrics/train_fold*.log \
        --out results/metrics/training_time.json

Reports, per fold and pooled:
  * number of epochs found
  * per-epoch wall time
  * total training time
  * an early-phase vs sustained-phase split, so that the sustained figure
    can be reported rather than the boost-clock figure. The handoff notes
    epochs slow from roughly 15 min to 26-38 min once sustained clocks
    settle, so a single mean across the whole run understates real cost.

The log format of train_cv.py is not known to this script, so it tries a
set of common patterns and PRINTS WHICH ONE MATCHED. If none match it
shows sample lines and exits rather than guessing -- check the reported
pattern against your log before using the numbers.

Supply your own with --epoch-regex if needed. The regex must expose a
named group `epoch`, and either `secs` (epoch duration in seconds) or
`ts` (a timestamp, from which durations are differenced). An optional
`dice` group is picked up if present.
"""

import argparse
import json
import re
import statistics
import sys
from datetime import datetime

# Each entry: (label, regex). Named groups: epoch, secs, ts, dice.
PATTERNS = [
    ("epoch N ... took/elapsed <secs>s",
     r"epoch\s*[:=]?\s*(?P<epoch>\d+).*?(?:took|elapsed|time)\s*[:=]?\s*"
     r"(?P<secs>\d+(?:\.\d+)?)\s*s"),
    ("epoch N ... <mm:ss> duration",
     r"epoch\s*[:=]?\s*(?P<epoch>\d+).*?(?:took|elapsed|time)\s*[:=]?\s*"
     r"(?P<mm>\d+):(?P<ss>\d{2})"),
    ("ISO timestamp then epoch N",
     r"(?P<ts>\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}).*?epoch\s*[:=]?\s*"
     r"(?P<epoch>\d+)"),
    ("[HH:MM:SS] then epoch N",
     r"\[(?P<ts>\d{2}:\d{2}:\d{2})\].*?epoch\s*[:=]?\s*(?P<epoch>\d+)"),
]

DICE_RE = re.compile(r"dice\s*[:=]?\s*(?P<dice>\d?\.\d+)", re.I)

TS_FORMATS = ["%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%H:%M:%S"]


def parse_ts(s):
    for fmt in TS_FORMATS:
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    return None


def extract(lines, regex, label):
    rows = []
    prev_ts = None
    for line in lines:
        m = regex.search(line)
        if not m:
            continue
        g = m.groupdict()
        row = {"epoch": int(g["epoch"])}

        if g.get("secs"):
            row["secs"] = float(g["secs"])
        elif g.get("mm") is not None:
            row["secs"] = int(g["mm"]) * 60 + int(g["ss"])
        elif g.get("ts"):
            ts = parse_ts(g["ts"])
            if ts is None:
                continue
            if prev_ts is not None:
                delta = (ts - prev_ts).total_seconds()
                if delta < 0:          # crossed midnight on an HH:MM:SS log
                    delta += 86400
                row["secs"] = delta
            prev_ts = ts

        d = DICE_RE.search(line)
        if d:
            row["dice"] = float(d.group("dice"))
        if "secs" in row:
            rows.append(row)
    return rows


def choose_pattern(lines, override):
    if override:
        rx = re.compile(override, re.I)
        rows = extract(lines, rx, "user-supplied")
        return "user-supplied", rows
    best = (None, [])
    for label, pat in PATTERNS:
        rows = extract(lines, re.compile(pat, re.I), label)
        if len(rows) > len(best[1]):
            best = (label, rows)
    return best


def phase_split(rows, boundary_s=3600):
    """Split epochs into the first `boundary_s` of wall time and the rest.

    The handoff describes clocks falling below boost after about an hour
    of load, so epochs after that point are the sustained-state figures
    that belong in the paper."""
    early, sustained, elapsed = [], [], 0.0
    for r in rows:
        (early if elapsed < boundary_s else sustained).append(r)
        elapsed += r["secs"]
    return early, sustained


def stats(vals):
    if not vals:
        return None
    return {
        "n": len(vals),
        "mean_s": round(statistics.mean(vals), 1),
        "mean_min": round(statistics.mean(vals) / 60.0, 1),
        "median_min": round(statistics.median(vals) / 60.0, 1),
        "min_min": round(min(vals) / 60.0, 1),
        "max_min": round(max(vals) / 60.0, 1),
        "sd_s": round(statistics.stdev(vals), 1) if len(vals) > 1 else 0.0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="+")
    ap.add_argument("--epoch-regex", help="override the pattern")
    ap.add_argument("--boost-window-s", type=float, default=3600,
                    help="wall seconds before clocks are treated as sustained")
    ap.add_argument("--out", default="results/metrics/training_time.json")
    args = ap.parse_args()

    report = {"per_log": {}, "boost_window_s": args.boost_window_s}
    all_sustained = []
    total_all = 0.0

    for path in args.logs:
        try:
            lines = open(path, errors="replace").read().splitlines()
        except OSError as exc:
            print("skipping %s: %s" % (path, exc), file=sys.stderr)
            continue

        label, rows = choose_pattern(lines, args.epoch_regex)
        if not rows:
            print("\nNo epoch timings matched in %s. Sample lines:" % path,
                  file=sys.stderr)
            for line in lines[:8]:
                print("  " + line[:160], file=sys.stderr)
            print("Pass --epoch-regex with named groups "
                  "(epoch, and secs or ts).", file=sys.stderr)
            continue

        early, sustained = phase_split(rows, args.boost_window_s)
        total = sum(r["secs"] for r in rows)
        total_all += total
        all_sustained.extend(r["secs"] for r in sustained)

        dices = [r["dice"] for r in rows if "dice" in r]
        entry = {
            "pattern_matched": label,
            "epochs_found": len(rows),
            "total_s": round(total, 1),
            "total_h": round(total / 3600.0, 2),
            "all_epochs": stats([r["secs"] for r in rows]),
            "early_phase": stats([r["secs"] for r in early]),
            "sustained_phase": stats([r["secs"] for r in sustained]),
            "epochs": rows,
        }
        if dices:
            entry["best_dice"] = max(dices)
            entry["best_dice_epoch"] = max(
                (r for r in rows if "dice" in r), key=lambda r: r["dice"])["epoch"]
        report["per_log"][path] = entry

        print("\n%s" % path, file=sys.stderr)
        print("  pattern: %s" % label, file=sys.stderr)
        print("  epochs:  %d   total: %.2f h" % (len(rows), total / 3600.0),
              file=sys.stderr)
        if entry["early_phase"]:
            print("  early:     %.1f min/epoch (n=%d)"
                  % (entry["early_phase"]["mean_min"],
                     entry["early_phase"]["n"]), file=sys.stderr)
        if entry["sustained_phase"]:
            print("  sustained: %.1f min/epoch (n=%d, range %.1f-%.1f)"
                  % (entry["sustained_phase"]["mean_min"],
                     entry["sustained_phase"]["n"],
                     entry["sustained_phase"]["min_min"],
                     entry["sustained_phase"]["max_min"]), file=sys.stderr)
        if dices:
            print("  best Dice %.4f at epoch %d"
                  % (entry["best_dice"], entry["best_dice_epoch"]),
                  file=sys.stderr)

    report["pooled"] = {
        "total_training_h": round(total_all / 3600.0, 2),
        "sustained_epoch": stats(all_sustained),
    }

    if report["per_log"]:
        import os
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as fh:
            json.dump(report, fh, indent=2)
        print("\npooled total training time: %.2f h across %d fold log(s)"
              % (report["pooled"]["total_training_h"], len(report["per_log"])),
              file=sys.stderr)
        if report["pooled"]["sustained_epoch"]:
            print("pooled sustained epoch time: %.1f min"
                  % report["pooled"]["sustained_epoch"]["mean_min"],
                  file=sys.stderr)
        print("wrote %s" % args.out, file=sys.stderr)


if __name__ == "__main__":
    main()
