"""Diff two stage3pred output dirs produced from the SAME input list, to
measure run-to-run reproducibility of the cascade.

For every stage3pred_*.h5 present in BOTH dirs, compare per-SP:
  - post/pred_query, post/pred_class      (slicer / Stage-2 output)
  - stage3/pred_class, stage3/pred_query  (particle segmenter / Stage-3 output)
and the event-level reco-1g+0X label. Reports the fraction of events whose
output changed at all, the per-SP flip rate, and whether the 1g0X label flipped.

  python determinism_diff.py <dirA> <dirB> [--thr 20]
"""
import argparse
import glob
import os
import sys

import h5py
import numpy as np

# Reuse the 1g0X selection's reco_label as the test analysis (no duplication).
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "single_photon"))
import analyze_1g0X as A   # reco_label(f, thr)


def _get(f, key):
    grp, _, ds = key.partition("/")
    if grp in f and ds in f[grp]:
        return f[grp][ds][:]
    return None


def compare(pa, pb, thr):
    out = dict(name=os.path.basename(pa))
    with h5py.File(pa, "r") as fa, h5py.File(pb, "r") as fb:
        out["drop_a"] = int(fa.attrs.get("stage3_meta_event_dropped", 0))
        out["drop_b"] = int(fb.attrs.get("stage3_meta_event_dropped", 0))
        # per-SP coords must match (same input) — sanity that we compare like-for-like
        ca, cb = _get(fa, "post/coord"), _get(fb, "post/coord")
        out["n_post_a"] = 0 if ca is None else len(ca)
        out["n_post_b"] = 0 if cb is None else len(cb)
        out["coord_same"] = (ca is not None and cb is not None
                             and ca.shape == cb.shape and np.array_equal(ca, cb))
        fields = {}
        for key in ("post/pred_query", "post/pred_class",
                    "stage3/pred_class", "stage3/pred_query"):
            va, vb = _get(fa, key), _get(fb, key)
            if va is None or vb is None or va.shape != vb.shape:
                fields[key] = None   # shape mismatch / missing
            else:
                ndiff = int((va != vb).sum())
                fields[key] = (ndiff, len(va))
        out["fields"] = fields
        # 1g0X label each side
        _, _, out["reco_a"] = A.reco_label(fa, thr)
        _, _, out["reco_b"] = A.reco_label(fb, thr)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirA")
    ap.add_argument("dirB")
    ap.add_argument("--thr", type=float, default=20.0)
    args = ap.parse_args()

    a = {os.path.basename(p): p for p in
         glob.glob(os.path.join(args.dirA, "**", "stage3pred_*.h5"), recursive=True)}
    b = {os.path.basename(p): p for p in
         glob.glob(os.path.join(args.dirB, "**", "stage3pred_*.h5"), recursive=True)}
    common = sorted(set(a) & set(b))
    print(f"dirA={args.dirA}\ndirB={args.dirB}")
    print(f"files: A={len(a)} B={len(b)} common={len(common)}\n")
    if not common:
        sys.exit("no common stage3pred files")

    n_ev = len(common)
    n_changed = 0           # any per-SP field differs
    n_drop_diff = 0         # event dropped in one run, kept in other
    n_label_flip = 0        # reco-1g0X label differs
    n_coord_mismatch = 0
    agg = {k: [0, 0] for k in ("post/pred_query", "post/pred_class",
                               "stage3/pred_class", "stage3/pred_query")}
    examples = []
    for name in common:
        r = compare(a[name], b[name], args.thr)
        if not r["coord_same"]:
            n_coord_mismatch += 1
        if r["drop_a"] != r["drop_b"]:
            n_drop_diff += 1
        if r["reco_a"] != r["reco_b"]:
            n_label_flip += 1
        changed = False
        for k, v in r["fields"].items():
            if v is None:
                continue
            ndiff, n = v
            agg[k][0] += ndiff
            agg[k][1] += n
            if ndiff > 0:
                changed = True
        if changed or r["drop_a"] != r["drop_b"]:
            n_changed += 1
            if len(examples) < 8:
                s3 = r["fields"]["stage3/pred_class"]
                examples.append(
                    f"  {r['name']}: stage3/pred_class "
                    f"{('%d/%d' % s3) if s3 else 'shape-mismatch'} diff  "
                    f"drop {r['drop_a']}->{r['drop_b']}  reco {r['reco_a']}->{r['reco_b']}")

    print("=" * 64)
    print(f"events compared          : {n_ev}")
    print(f"events with ANY change   : {n_changed}  ({100*n_changed/n_ev:.1f}%)")
    print(f"events drop-flag differs : {n_drop_diff}")
    print(f"events 1g0X LABEL flips  : {n_label_flip}  ({100*n_label_flip/n_ev:.1f}%)")
    print(f"events coord mismatch    : {n_coord_mismatch}  (should be 0)")
    print("-" * 64)
    for k, (nd, nt) in agg.items():
        rate = (100 * nd / nt) if nt else float("nan")
        print(f"  per-SP {k:22s}: {nd}/{nt} SPs differ ({rate:.3f}%)")
    if examples:
        print("-" * 64)
        print("examples:")
        print("\n".join(examples))
    print("=" * 64)


if __name__ == "__main__":
    main()
