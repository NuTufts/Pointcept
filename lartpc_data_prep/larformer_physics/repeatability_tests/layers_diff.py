"""Diff two capture_deghost_layers.py output dirs per stage, to localize WHERE a
membership/cross-GPU divergence first enters the model (Tier-B). Each stage's
order-invariant ssq fingerprint is compared between the two runs; the first stage
whose ssq differs by >> the fp floor is the divergence onset.

  python layers_diff.py <layersDirA> <layersDirB>

Use after a membership or cross-GPU test FAILS: capture the layer fingerprints on
the two lists/GPUs (slurm/submit_capture_layers.sh, with TARGET=deghoster or
slicer), then run this to see which stage introduces the divergence.
"""
import argparse
import glob
import os
import re

import numpy as np


def _order(s):
    # rough forward-execution order for deghoster + slicer stage names
    base = {"embedding": 0, "bb.embedding": 0,
            "enc0.block0.cpe.spconv": 1, "enc0.block0.cpe": 2,
            "enc0.block0.attn": 3, "enc0.block0.mlp": 4,
            "tok.spacepoint": 30, "tok.voxel_16cm": 31, "tok.voxel_8cm": 32,
            "tok.ptv3_dec3": 33, "tok.ptv3_dec2": 34,
            "token_refiner": 40, "query_selector": 45, "dec.init_heads": 46}
    if s in base:
        return (base[s], 0)
    if s.startswith("bb.enc") or s.startswith("enc.enc"):
        return (1, int(re.findall(r"\d+", s)[0]))
    if s.startswith("bb.dec"):
        return (2, int(re.findall(r"\d+", s)[0]))
    if "dec.layer" in s:
        return (50, int(re.findall(r"\d+", s)[0]))
    if "head" in s:
        return (60, 0)
    return (99, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirA"); ap.add_argument("dirB")
    args = ap.parse_args()
    A = {os.path.basename(p): p for p in glob.glob(os.path.join(args.dirA, "layers_*.npz"))}
    B = {os.path.basename(p): p for p in glob.glob(os.path.join(args.dirB, "layers_*.npz"))}
    common = sorted(set(A) & set(B))
    print(f"common events: {len(common)}")
    if not common:
        raise SystemExit("no common capture files")
    stages = sorted({k[:-5] for k in np.load(A[common[0]]).files if k.endswith("__ssq")},
                    key=_order)
    print(f"{'stage (exec order)':>26} | {'#evt ssq-diff>1e-6':>18} | {'median rel':>11} | {'max rel':>10}")
    print("-" * 74)
    for s in stages:
        rels = []
        for n in common:
            a, b = np.load(A[n]), np.load(B[n])
            k = s + "__ssq"
            if k in a.files and k in b.files:
                sa, sb = float(a[k]), float(b[k])
                rels.append(abs(sa - sb) / max(abs(sa), 1e-30))
        rels = np.array(rels)
        print(f"{s:>26} | {int((rels > 1e-6).sum()):>8}/{len(rels):<9} | "
              f"{np.median(rels):>11.1e} | {rels.max():>10.1e}")
    print("-" * 74)
    print("First stage with a large rel-diff = divergence onset (everything upstream is clean).")


if __name__ == "__main__":
    main()
