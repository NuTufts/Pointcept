"""Dump the per-PMT vector<double> branches of a Wire-Cell T_BDTvars tree for a
given (run, subrun, event), and optionally lay them next to our own per-PMT
table for the same event.

TTree::Scan cannot expand a vector<double> inline (it prints a blank column),
which is why the branch looked empty. uproot reads them natively.

    # what PMT branches exist, and their sizes for this event?
    python3 dump_wc_pmt_info.py --file <official.root> \
        --run 15014 --subrun 234 --event 11701 --list

    # dump the vectors, side-by-side with our table
    python3 dump_wc_pmt_info.py --file <official.root> \
        --run 15014 --subrun 234 --event 11701 \
        --compare saturation_pmt_15014_234_11701.csv

IMPORTANT -- indexing. Our obs_pe is OPDET-indexed (larlite
larutil::Geometry::OpDetFromOpChannel, which is what the stage-A converter used
to fill merged_sp). Wire-Cell PMT vectors are conventionally OPCHANNEL-ordered.
The two differ on all 32 entries, so --compare reports the match under BOTH
interpretations and tells you which one lines up; do not assume.
"""
import argparse
import csv
import os
import sys

import numpy as np
import uproot

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "..", ".."))
from lartpc.larformer_analysis.flashmodel_calib.ophit_saturation_probe import (
    OPCH2OPDET)                                                   # noqa: E402

OD2CH = {v: k for k, v in OPCH2OPDET.items()}


def find_tree(f, want):
    keys = [k.split(";")[0] for k in f.keys(recursive=True)]
    if want in keys:
        return want
    for k in keys:
        if k.split("/")[-1] == want:
            return k
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True)
    ap.add_argument("--tree", default="T_BDTvars")
    ap.add_argument("--run", type=int, required=True)
    ap.add_argument("--subrun", type=int, required=True)
    ap.add_argument("--event", type=int, required=True)
    ap.add_argument("--pattern", default="pmt",
                    help="case-insensitive substring selecting PMT branches")
    ap.add_argument("--list", action="store_true",
                    help="just list matching branches + their per-event length")
    ap.add_argument("--compare", default=None,
                    help="our per-PMT CSV for the same event")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    f = uproot.open(args.file)
    tn = find_tree(f, args.tree)
    if tn is None:
        print("!!! tree %r not in %s" % (args.tree, args.file))
        print("    trees present:",
              sorted({k.split(";")[0] for k in f.keys(recursive=True)})[:40])
        return
    t = f[tn]
    print(">>> tree %s : %d entries" % (tn, t.num_entries))

    names = list(t.keys())
    idn = {n.lower(): n for n in names}
    rn, sn, en = (idn.get("run"), idn.get("subrun"), idn.get("event"))
    if not (rn and sn and en):
        print("!!! could not find run/subrun/event branches; have:", names[:30])
        return
    run = t[rn].array(library="np")
    sub = t[sn].array(library="np")
    evt = t[en].array(library="np")
    sel = np.nonzero((run == args.run) & (sub == args.subrun)
                     & (evt == args.event))[0]
    if not len(sel):
        print("!!! (%d,%d,%d) not found in this file"
              % (args.run, args.subrun, args.event))
        return
    i = int(sel[0])
    print(">>> (%d,%d,%d) -> entry %d%s"
          % (args.run, args.subrun, args.event, i,
             ("  [%d duplicate entries: %s]" % (len(sel), sel.tolist())
              if len(sel) > 1 else "")))

    pmt = [n for n in names if args.pattern.lower() in n.lower()]
    if not pmt:
        print("!!! no branch matching %r; branches:" % args.pattern)
        for n in names[:60]:
            print("   ", n)
        return

    vecs = {}
    print("\n== branches matching %r ==" % args.pattern)
    for n in pmt:
        try:
            v = t[n].array(entry_start=i, entry_stop=i + 1)[0]
            a = np.asarray(v)
            if a.ndim == 0:
                print("  %-34s scalar = %s" % (n, a))
                continue
            vecs[n] = a.astype(np.float64)
            print("  %-34s vector<%s> len=%-4d sum=%12.2f  max=%10.2f"
                  % (n, a.dtype, len(a), float(np.nansum(a)),
                     float(np.nanmax(a)) if len(a) else float("nan")))
        except Exception as ex:
            print("  %-34s [unreadable: %s]" % (n, ex))
    if args.list or not vecs:
        return

    v32 = {n: a for n, a in vecs.items() if len(a) == 32}
    print("\n== per-PMT dump (index = the vector's own ordering) ==")
    hdr = [n for n in v32]
    print("%5s | %s" % ("idx", " | ".join("%18s" % n[-18:] for n in hdr)))
    for k in range(32):
        print("%5d | %s" % (k, " | ".join("%18.3f" % v32[n][k] for n in hdr)))

    if not args.compare:
        return
    rows = [r for r in csv.DictReader(
        l for l in open(args.compare) if not l.startswith("#"))]
    if not rows:
        print("!!! --compare file has no rows")
        return
    ours_by_od = {int(r["opdet"]): float(r["obs_pe"]) for r in rows}
    obs_od = np.array([ours_by_od[o] for o in range(32)])
    obs_ch = np.array([ours_by_od[OPCH2OPDET[c]] for c in range(32)])

    print("\n== which indexing does the WC vector use? (corr vs our obs_pe) ==")
    print("%-34s %14s %14s   verdict" % ("branch", "as OPDET", "as OPCHANNEL"))
    for n, a in v32.items():
        if np.nanstd(a) == 0:
            continue
        c_od = float(np.corrcoef(a, obs_od)[0, 1])
        c_ch = float(np.corrcoef(a, obs_ch)[0, 1])
        verdict = ("OPDET-indexed" if c_od > c_ch else "OPCHANNEL-indexed")
        print("%-34s %14.3f %14.3f   %s" % (n, c_od, c_ch, verdict))

    best = max(v32, key=lambda n: abs(np.corrcoef(v32[n], obs_ch)[0, 1])
               if np.nanstd(v32[n]) else -1)
    a = v32[best]
    c_od = float(np.corrcoef(a, obs_od)[0, 1])
    c_ch = float(np.corrcoef(a, obs_ch)[0, 1])
    as_ch = c_ch >= c_od
    print("\n== side-by-side using %s, read as %s =="
          % (best, "OPCHANNEL" if as_ch else "OPDET"))
    print("%5s %5s | %12s %12s %12s | %s"
          % ("opdet", "opch", "ours_obs_PE", "WC_PE", "WC-ours", "flags"))
    fl = {int(r["opdet"]): r.get("flags", "") for r in rows}
    pr = {int(r["opdet"]): float(r["pred_pe"]) for r in rows}
    tot_o = tot_w = 0.0
    for od in range(32):
        k = OD2CH[od] if as_ch else od
        w = float(a[k]); o = ours_by_od[od]
        tot_o += o; tot_w += w
        print("%5d %5d | %12.2f %12.2f %12.2f | %s"
              % (od, OD2CH[od], o, w, w - o, fl.get(od, "")))
    print("  %-11s | %12.2f %12.2f %12.2f" % ("TOTAL", tot_o, tot_w,
                                              tot_w - tot_o))
    print("\n== the question: do the tubes our flash reports as 0 have light here? ==")
    for od in range(32):
        if ours_by_od[od] < 5.0 and pr.get(od, 0) > 300:
            k = OD2CH[od] if as_ch else od
            print("  opdet %-2d / opch %-2d : ours=%7.2f  pred=%8.1f  WC=%9.2f  %s"
                  % (od, OD2CH[od], ours_by_od[od], pr[od], float(a[k]),
                     "<== WC SEES THE LIGHT" if float(a[k]) > 50
                     else "<== WC also sees nothing"))

    if args.out:
        with open(args.out, "w") as fo:
            fo.write("opdet,opchannel,ours_obs_pe,ours_pred_pe,wc_pe,flags\n")
            for od in range(32):
                k = OD2CH[od] if as_ch else od
                fo.write("%d,%d,%.3f,%.3f,%.3f,%s\n"
                         % (od, OD2CH[od], ours_by_od[od], pr.get(od, float("nan")),
                            float(a[k]), fl.get(od, "")))
        print(">>> wrote", args.out)


if __name__ == "__main__":
    main()
