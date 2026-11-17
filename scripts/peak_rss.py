#!/usr/bin/env python3
"""
peak_rss.py -- measure peak resident set size (high-water mark) for a process.

Two modes:

  1. Wrap a command (measures the whole run from process start):
       python peak_rss.py --run -- python scripts/train_cv.py --fold 2 --epochs 12

  2. Attach to an already-running process (reads the kernel's high-water
     mark, which is retained from process start, so this is still exact
     even if you attach late):
       python peak_rss.py --pid 12345

Reports VmHWM from /proc/<pid>/status, which is the true peak RSS the
kernel recorded, including the peak reached before this script attached.
Also samples current RSS over time so you can see the shape of the curve
and confirm the peak is not an artefact of a transient spike.

Includes child processes when --run is used (DataLoader workers, etc.) by
summing over the process group at each sample; the reported group peak is
the maximum of those sums, which is an upper bound on true simultaneous
group usage.

Linux only (reads /proc). Output: JSON to stdout or --out.
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import time


def read_status(pid):
    """Return (VmHWM_kB, VmRSS_kB) for pid, or None if the process is gone."""
    try:
        with open("/proc/%d/status" % pid) as fh:
            hwm = rss = None
            for line in fh:
                if line.startswith("VmHWM:"):
                    hwm = int(line.split()[1])
                elif line.startswith("VmRSS:"):
                    rss = int(line.split()[1])
                if hwm is not None and rss is not None:
                    break
            return hwm, rss
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        return None


def descendants(pid):
    """All live descendants of pid, including pid itself."""
    out = [pid]
    stack = [pid]
    while stack:
        cur = stack.pop()
        try:
            with open("/proc/%d/task/%d/children" % (cur, cur)) as fh:
                kids = [int(t) for t in fh.read().split()]
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
        out.extend(kids)
        stack.extend(kids)
    return out


def sample_tree(pid):
    """Sum current RSS across the process tree; return (sum_rss_kB, root_hwm_kB)."""
    total = 0
    root_hwm = None
    for p in descendants(pid):
        st = read_status(p)
        if st is None:
            continue
        hwm, rss = st
        if rss:
            total += rss
        if p == pid and hwm:
            root_hwm = hwm
    return total, root_hwm


def monitor(pid, interval, stop_when_dead=True):
    samples = []
    group_peak = 0
    root_hwm = 0
    t0 = time.time()
    while True:
        total, hwm = sample_tree(pid)
        if hwm is None and total == 0:
            if stop_when_dead:
                break
        if total > group_peak:
            group_peak = total
        if hwm and hwm > root_hwm:
            root_hwm = hwm
        samples.append({"t_s": round(time.time() - t0, 2), "group_rss_kB": total})
        time.sleep(interval)
        if stop_when_dead and read_status(pid) is None:
            break
    return samples, group_peak, root_hwm


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--pid", type=int, help="attach to an already-running PID")
    g.add_argument("--run", action="store_true",
                   help="run the command after -- and measure it")
    ap.add_argument("--interval", type=float, default=2.0,
                    help="sampling interval in seconds (default 2.0)")
    ap.add_argument("--out", help="write JSON here instead of stdout")
    ap.add_argument("cmd", nargs=argparse.REMAINDER,
                    help="command to run, after --")
    args = ap.parse_args()

    if not sys.platform.startswith("linux"):
        sys.exit("peak_rss.py requires Linux (/proc)")

    result = {"mode": None, "interval_s": args.interval}

    if args.pid:
        if read_status(args.pid) is None:
            sys.exit("no such process: %d" % args.pid)
        result["mode"] = "attach"
        result["pid"] = args.pid
        print("attached to pid %d; Ctrl-C to stop and report" % args.pid,
              file=sys.stderr)
        try:
            samples, group_peak, root_hwm = monitor(args.pid, args.interval)
        except KeyboardInterrupt:
            samples, group_peak, root_hwm = [], 0, 0
            st = read_status(args.pid)
            if st:
                root_hwm = st[0]
        result["note"] = ("VmHWM is retained from process start, so the peak "
                          "is exact even though sampling began late. The "
                          "group_peak figure only covers the attached window.")
    else:
        cmd = [c for c in args.cmd if c != "--"]
        if not cmd:
            sys.exit("no command given after --")
        result["mode"] = "run"
        result["command"] = cmd
        t0 = time.perf_counter()
        proc = subprocess.Popen(cmd, start_new_session=True)
        result["pid"] = proc.pid
        try:
            samples, group_peak, root_hwm = monitor(proc.pid, args.interval)
        except KeyboardInterrupt:
            os.killpg(os.getpgid(proc.pid), signal.SIGINT)
            samples, group_peak, root_hwm = [], 0, 0
        rc = proc.wait()
        result["returncode"] = rc
        result["wall_time_s"] = round(time.perf_counter() - t0, 2)

    result["peak_rss_root_MiB"] = round(root_hwm / 1024.0, 1) if root_hwm else None
    result["peak_rss_group_MiB"] = round(group_peak / 1024.0, 1) if group_peak else None
    result["samples"] = samples

    text = json.dumps(result, indent=2)
    if args.out:
        with open(args.out, "w") as fh:
            fh.write(text + "\n")
        print("wrote %s" % args.out, file=sys.stderr)
    print("peak RSS (main process, kernel high-water mark): %s MiB"
          % result["peak_rss_root_MiB"], file=sys.stderr)
    print("peak RSS (process group, sampled):               %s MiB"
          % result["peak_rss_group_MiB"], file=sys.stderr)
    if not args.out:
        print(text)


if __name__ == "__main__":
    main()
