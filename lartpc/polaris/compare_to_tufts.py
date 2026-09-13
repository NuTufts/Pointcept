#!/usr/bin/env python3
"""Compare a Polaris keypoint2 tree with the Tufts reference JSON (needs h5py;
run inside the container via pol_exec).

  compare_to_tufts.py --ref lartpc/polaris/tufts_ref_bnb5e19_idx0-499.json --tree <dir> [--max-index N]

Checks, per reference index present on both sides:
  * src_file identical  -> the index<->event linkage is intact (hard requirement)
  * n_particles and the sorted per-particle point counts identical -> the
    partition is bit-identical (expected for most events; float32 differences
    between GPU types change a minority upstream of the flash table)
  * nu_vertex_cm distance
Exit 1 if any src_file differs or if presence differs for > 10% of events.
A low identical-partition fraction is reported as a WARNING (platform family).
"""
import argparse
import json
import math
import os
import sys

import h5py

from make_tufts_reference import event_dir, summarize  # noqa: E402  (same dir)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--ref", required=True)
    ap.add_argument("--tree", required=True)
    ap.add_argument("--max-index", type=int, default=None)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    ref = json.load(open(args.ref))
    mx = ref["max_index"] if args.max_index is None else min(args.max_index, ref["max_index"])
    n_both = n_same_part = n_same_np = n_src_bad = n_pres_diff = 0
    n_ref_only = n_pol_only = 0
    dv = []
    print(f"{'idx':>5} {'src_file':<42} {'nP tufts':>8} {'nP pol':>7} {'partition':>10} {'dvtx[cm]':>9}")
    for i in range(mx + 1):
        r = ref["events"][str(i)]
        p = os.path.join(event_dir(args.tree, i), f"keypoint2_event{i:05d}_0.h5")
        pol = summarize(p) if os.path.exists(p) else None
        if r.get("absent") and pol is None:
            continue
        if r.get("absent") or pol is None:
            n_pres_diff += 1
            if pol is None:
                n_ref_only += 1
            else:
                n_pol_only += 1
            print(f"{i:5d} {'(tufts only)' if pol is None else '(polaris only: ' + pol['src_file'] + ')':<42}")
            continue
        n_both += 1
        src_ok = (r["src_file"] == pol["src_file"])
        if not src_ok:
            n_src_bad += 1
        same_np = r["n_particles"] == pol["n_particles"]
        same_part = same_np and r["particle_npts"] == pol["particle_npts"]
        n_same_np += same_np; n_same_part += same_part
        d = float("nan")
        if r.get("nu_vertex_cm") and pol.get("nu_vertex_cm"):
            d = math.sqrt(sum((a - b) ** 2 for a, b in zip(r["nu_vertex_cm"], pol["nu_vertex_cm"])))
            dv.append(d)
        if args.verbose or not src_ok or not same_part:
            print(f"{i:5d} {pol['src_file']:<42} {r['n_particles']:8d} {pol['n_particles']:7d} "
                  f"{'same' if same_part else 'DIFF':>10} {d:9.2f}" + ("" if src_ok else "   SRC_FILE MISMATCH: tufts " + r["src_file"]))
    print(f">>> compared indices 0-{mx}: {n_both} on both sides; presence differs for {n_pres_diff} "
          f"(tufts-only {n_ref_only}, polaris-only {n_pol_only})")
    if n_both:
        print(f">>> src_file identical: {n_both - n_src_bad}/{n_both}; same n_particles: {n_same_np}/{n_both} "
              f"({100.0 * n_same_np / n_both:.0f}%); identical partition: {n_same_part}/{n_both} "
              f"({100.0 * n_same_part / n_both:.0f}%)")
    if dv:
        dv.sort()
        print(f">>> nu vertex distance: median {dv[len(dv)//2]:.2f} cm, 90% < {dv[int(0.9 * (len(dv) - 1))]:.2f} cm, max {dv[-1]:.2f} cm")
    ok = True
    if n_src_bad:
        print("!!! src_file mismatches: the event index linkage differs from Tufts (list order / tree layout)"); ok = False
    if n_both + n_pres_diff and n_pres_diff > 0.10 * (n_both + n_pres_diff):
        print("!!! presence differs for > 10% of events"); ok = False
    if n_both and n_same_part < 0.9 * n_both:
        # Not a failure: a different driver/GPU family reproduces the partition
        # only for a fraction of events (Polaris A100-SXM4: 50%, Tufts A100: 100%,
        # 2026-09-13). Quantify the analysis-level effect with compare_platforms.py.
        print("WARNING: fewer than 90% identical partitions -> different conformance family "
              "(see docs/reference/LArFormer_Reproducibility.md); the index linkage is intact")
    print(f">>> compare_to_tufts: {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    main()
