#!/usr/bin/env python3
"""Verify a cascade GPU pass (stdlib only; runs on the Polaris host).

  check_cascade.py --worklist <wl> --outdir <keypoint2 tree> --logdir <dir> --tag <tag>
                   [--throughput] [--no-count]

Per shard of the worklist ("shard start n" lines): done marker, log ends with
DONE, Traceback count, "SKIP unreadable" count, last processed event index,
events processed vs expected. Totals: keypoint2_event*_0.h5 and *_fm_0.h5
counts under --outdir (skip with --no-count on very large trees).

Unfinished shards are written to <worklist>.resume as "shard <last+1> <remaining>"
(the same shard id, so the log is appended and the marker lands in place) and a
qsub_cascade.sh line to relaunch them is printed. Exit 0 only when every shard
is done. --throughput prints s/event per shard from the timestamped log lines
(first event line -> DONE), i.e. excluding model build time.
"""
import argparse
import datetime as dt
import os
import re
import sys

EVENT_RE = re.compile(r"^(\S+) +\[(\d+)(?:\.[^\]]*)?\]")   # "<stamp> [123.0] ..." / "[123] no nu slice"
STAMP_RE = re.compile(r"^(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d) ")


def parse_stamp(s):
    return dt.datetime.strptime(s, "%Y-%m-%dT%H:%M:%S")


def read_worklist(path):
    out = []
    with open(path) as fh:
        for line in fh:
            p = line.split()
            if len(p) >= 3 and not line.lstrip().startswith("#"):
                out.append((int(p[0]), int(p[1]), int(p[2])))
    return out


def scan_log(path):
    """Stats of the LAST attempt in a (timestamped) shard log."""
    st = {"exists": os.path.exists(path), "done": False, "tracebacks": 0, "skips": 0,
          "last_idx": None, "n_events_seen": 0, "t_first": None, "t_done": None,
          "attempts": 0, "failed": False}
    if not st["exists"]:
        return st
    with open(path, errors="replace") as fh:
        lines = fh.readlines()
    # last attempt = after the last ">>> attempt" line
    start = 0
    for i, l in enumerate(lines):
        if ">>> attempt" in l:
            start = i; st["attempts"] += 1
    seen = set()
    for l in lines[start:]:
        m = EVENT_RE.match(l)
        if m:
            idx = int(m.group(2))
            if idx not in seen:
                seen.add(idx)
                st["n_events_seen"] += 1
            st["last_idx"] = idx if st["last_idx"] is None else max(st["last_idx"], idx)
            if st["t_first"] is None:
                st["t_first"] = m.group(1)
        if "Traceback" in l:
            st["tracebacks"] += 1
        if "SKIP unreadable" in l:
            st["skips"] += 1
        if re.search(r"\bDONE\b", l) and "DONE shard" not in l:
            st["done"] = True
            sm = STAMP_RE.match(l)
            st["t_done"] = sm.group(1) if sm else None
        if l.rstrip().endswith("FAILED") or "FAILED shard" in l:
            st["failed"] = True
    return st


def count_tree(outdir):
    n_nu = n_fm = 0
    for _root, _d, files in os.walk(outdir):
        for f in files:
            if not f.startswith("keypoint2_event"):
                continue
            if f.endswith("_fm_0.h5"):
                n_fm += 1
            elif f.endswith("_0.h5"):
                n_nu += 1
    return n_nu, n_fm


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--worklist", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--logdir", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--throughput", action="store_true")
    ap.add_argument("--no-count", action="store_true")
    args = ap.parse_args()

    wl = read_worklist(args.worklist)
    markdir = os.path.join(args.outdir, ".shards")
    unfinished = []
    tot_skips = tot_tb = 0
    rates = []
    print(f">>> check_cascade tag={args.tag}: {len(wl)} shards, out={args.outdir}")
    print(f"{'shard':>5} {'start':>7} {'n':>6} {'marker':>6} {'DONE':>5} {'seen':>6} {'last':>7} "
          f"{'skip':>4} {'tb':>3} {'s/ev':>6} status")
    for shard, start, n in wl:
        log = os.path.join(args.logdir, f"cascade_shard{shard:03d}.log")
        marker = os.path.exists(os.path.join(markdir, f"shard{shard:03d}.done"))
        st = scan_log(log)
        tot_skips += st["skips"]; tot_tb += st["tracebacks"]
        rate = ""
        if st["t_first"] and st["t_done"] and st["n_events_seen"] > 1:
            secs = (parse_stamp(st["t_done"]) - parse_stamp(st["t_first"])).total_seconds()
            r = secs / max(1, st["n_events_seen"] - 1)
            rates.append(r); rate = f"{r:6.2f}"
        if marker and st["done"]:
            status = "ok"
        elif not st["exists"]:
            status = "NOT STARTED"; unfinished.append((shard, start, n))
        else:
            last = st["last_idx"]
            if last is None or last < start:
                status = "no events yet -> resume whole"; unfinished.append((shard, start, n))
            elif last >= start + n - 1 and st["done"]:
                status = "done but no marker (rerun marker only)"; unfinished.append((shard, start + n, 0))
            else:
                rem = start + n - (last + 1)
                status = f"INCOMPLETE -> resume {last + 1} +{rem}"
                unfinished.append((shard, last + 1, rem))
            if st["failed"]:
                status += " [FAILED]"
        print(f"{shard:5d} {start:7d} {n:6d} {'yes' if marker else 'no':>6} {'yes' if st['done'] else 'no':>5} "
              f"{st['n_events_seen']:6d} {str(st['last_idx']):>7} {st['skips']:4d} {st['tracebacks']:3d} {rate:>6} {status}")
    exp_events = sum(n for _s, _st, n in wl)
    print(f">>> expected events: {exp_events}; SKIP unreadable: {tot_skips}; Tracebacks: {tot_tb}")
    if rates:
        rates.sort()
        print(f">>> throughput: median {rates[len(rates)//2]:.2f} s/event (min {rates[0]:.2f}, max {rates[-1]:.2f}); "
              f"a 2,755-event shard at the median = {rates[len(rates)//2] * 2755 / 3600:.2f} h")
    if not args.no_count:
        n_nu, n_fm = count_tree(args.outdir)
        print(f">>> files: {n_nu} nu-stream, {n_fm} fm-stream keypoint2 files "
              f"({100.0 * n_nu / max(1, exp_events):.1f}% of expected events have a nu file)")
    if unfinished:
        resume = args.worklist + ".resume"
        with open(resume, "w") as fo:
            for shard, s, n in unfinished:
                if n > 0:
                    fo.write(f"{shard} {s} {n}\n")
        nres = sum(1 for _s, _st, n in unfinished if n > 0)
        print(f">>> {len(unfinished)} shard(s) unfinished; {nres} resume ranges -> {resume}")
        print(f"    relaunch (after the job is gone from qstat):")
        print(f"    MODE=debug NODES=1 TAG={args.tag} OUTDIR={args.outdir} WORKLIST={resume} "
              f"bash lartpc/polaris/qsub_cascade.sh   # <= 8 ranges; MODE=preempt NODES=k for more")
        sys.exit(1)
    print(">>> check_cascade: ALL SHARDS DONE")


if __name__ == "__main__":
    main()
