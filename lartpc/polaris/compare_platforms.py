#!/usr/bin/env python3
"""Quantify the cross-platform (conformance-family) difference of the LArFormer
chain on the SAME events processed on two clusters/GPUs. Run inside the
container (h5py, uproot).

  # cascade level: two keypoint2 trees (2-level index layout), same event indices
  compare_platforms.py kp2 --a <treeA> --b <treeB> [--max-index N]

  # analysis level: two gen2ntuple files of the same events (matched by run/subrun/event)
  compare_platforms.py ntuple --a <A.root> --b <B.root> [--csv out.csv]

kp2 mode reports, per event and in summary: nu-file presence flips, n_particles
agreement, bit-identical partition (sorted per-particle point counts), slice
point-set Jaccard (coords rounded to 0.01 cm), nu-vertex distance.

ntuple mode reports event-level flip rates with binomial errors for the
quantities an analysis cuts on: foundVertex, primaryVtxStream, |dvtx| > 1/5 cm,
nTracks, nShowers, the LArFormer PID multiset of prongs, the LArPID PID multiset,
sum showerRecoE (> 10 % relative), max showerCosmicScore (> 0.1),
nuSliceFlashChi2 (> 10 % relative), recoNuE (> 10 % relative), plus the
distribution of the differences. Exit status is always 0: this is a
measurement, the verdict is the user's (see LArFormer_Reproducibility.md §4.3).
"""
import argparse
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_tufts_reference import event_dir, summarize  # noqa: E402


def binom(k, n):
    if n == 0:
        return "n/a"
    p = k / n
    return f"{k}/{n} = {100 * p:.2f}% +- {100 * math.sqrt(max(p * (1 - p), 1.0 / n) / n):.2f}%"


# ----------------------------------------------------------------------------- kp2
def slice_keys(path):
    import h5py
    with h5py.File(path, "r") as h:
        c = h["slice"]["coord_cm"][()]
    q = np.round(c / 0.01).astype(np.int64)
    return set(map(bytes, q))


def kp2_mode(args):
    idx_max = args.max_index
    if idx_max is None:
        # scan tree A for the highest index
        idx_max = -1
        for root, _d, files in os.walk(args.a):
            for f in files:
                if f.startswith("keypoint2_event") and f.endswith("_0.h5"):
                    idx_max = max(idx_max, int(f[len("keypoint2_event"):len("keypoint2_event") + 5]))
    n_a = n_b = n_both = n_same_np = n_same_part = 0
    jac, dv, dnp = [], [], []
    print(f"{'idx':>5} {'nP_A':>5} {'nP_B':>5} {'part':>5} {'jaccard':>8} {'dvtx':>8}")
    for i in range(idx_max + 1):
        pa = os.path.join(event_dir(args.a, i), f"keypoint2_event{i:05d}_0.h5")
        pb = os.path.join(event_dir(args.b, i), f"keypoint2_event{i:05d}_0.h5")
        ea, eb = os.path.exists(pa), os.path.exists(pb)
        n_a += ea; n_b += eb
        if not (ea and eb):
            if ea or eb:
                print(f"{i:5d} {'A only' if ea else 'B only':>11}")
            continue
        n_both += 1
        a, b = summarize(pa), summarize(pb)
        if a["src_file"] != b["src_file"]:
            print(f"!!! {i}: src_file differs ({a['src_file']} vs {b['src_file']}) -- trees not aligned"); sys.exit(2)
        same_np = a["n_particles"] == b["n_particles"]
        same_part = same_np and a["particle_npts"] == b["particle_npts"]
        n_same_np += same_np; n_same_part += same_part
        dnp.append(b["n_particles"] - a["n_particles"])
        ka, kb = slice_keys(pa), slice_keys(pb)
        j = len(ka & kb) / max(1, len(ka | kb)); jac.append(j)
        d = float("nan")
        if a["nu_vertex_cm"] and b["nu_vertex_cm"]:
            d = math.dist(a["nu_vertex_cm"], b["nu_vertex_cm"]); dv.append(d)
        if args.verbose or not same_part:
            print(f"{i:5d} {a['n_particles']:5d} {b['n_particles']:5d} {'same' if same_part else 'DIFF':>5} {j:8.4f} {d:8.2f}")
    n_pres_flip = (n_a - n_both) + (n_b - n_both)
    print(f"\n>>> indices 0-{idx_max}: nu file in A {n_a}, in B {n_b}, both {n_both}; presence flips {binom(n_pres_flip, max(n_a, n_b))}")
    if n_both:
        print(f">>> same n_particles: {binom(n_same_np, n_both)}; identical partition: {binom(n_same_part, n_both)}")
        dnp = np.array(dnp); print(f">>> n_particles B-A: mean {dnp.mean():+.3f}, |diff|>=1 in {int((dnp != 0).sum())}, >=2 in {int((abs(dnp) >= 2).sum())}")
        jac = np.array(jac)
        print(f">>> slice Jaccard: median {np.median(jac):.4f}, 10th pct {np.percentile(jac, 10):.4f}, "
              f"< 0.9 in {binom(int((jac < 0.9).sum()), n_both)}, < 0.5 in {binom(int((jac < 0.5).sum()), n_both)}")
    if dv:
        dv = np.array(dv)
        print(f">>> nu vertex distance: median {np.median(dv):.2f} cm, > 1 cm {binom(int((dv > 1).sum()), len(dv))}, "
              f"> 5 cm {binom(int((dv > 5).sum()), len(dv))}, > 20 cm {binom(int((dv > 20).sum()), len(dv))}")


# ------------------------------------------------------------------------- ntuple
NT_SCALARS = ["run", "subrun", "event", "foundVertex", "primaryVtxStream", "vtxX", "vtxY", "vtxZ",
              "vtxScore", "recoNuE", "nuSliceFlashChi2", "nuSliceNParticles", "nTracks", "nShowers"]
NT_JAGGED = ["trackLArFormerPID", "showerLArFormerPID", "trackPID", "showerPID",
             "showerRecoE", "showerCosmicScore", "trackRecoE"]


def load_ntuple(path):
    import uproot
    with uproot.open(path) as f:
        t = f["EventTree"]
        arr = t.arrays(NT_SCALARS + NT_JAGGED, library="np")
    ev = {}
    for k in range(len(arr["run"])):
        key = (int(arr["run"][k]), int(arr["subrun"][k]), int(arr["event"][k]))
        ev[key] = {n: arr[n][k] for n in NT_SCALARS + NT_JAGGED}
    return ev


def rel(a, b):
    a, b = float(a), float(b)
    if a == b:
        return 0.0
    return abs(a - b) / max(abs(a), abs(b), 1e-9)


def ntuple_mode(args):
    A, B = load_ntuple(args.a), load_ntuple(args.b)
    common = sorted(set(A) & set(B))
    print(f">>> events: A {len(A)}, B {len(B)}, common {len(common)} (matched by run/subrun/event)")
    n = len(common)
    flips = {}
    def flip(name, cond):
        flips[name] = flips.get(name, 0) + (1 if cond else 0)
    dv, dE, dchi, dscore = [], [], [], []
    rows = []
    for key in common:
        a, b = A[key], B[key]
        fa, fb = int(a["foundVertex"]), int(b["foundVertex"])
        flip("foundVertex", fa != fb)
        flip("primaryVtxStream", int(a["primaryVtxStream"]) != int(b["primaryVtxStream"]))
        flip("nTracks", int(a["nTracks"]) != int(b["nTracks"]))
        flip("nShowers", int(a["nShowers"]) != int(b["nShowers"]))
        flip("nProngs", int(a["nTracks"]) + int(a["nShowers"]) != int(b["nTracks"]) + int(b["nShowers"]))
        pa = sorted(list(a["trackLArFormerPID"]) + list(a["showerLArFormerPID"]))
        pb = sorted(list(b["trackLArFormerPID"]) + list(b["showerLArFormerPID"]))
        flip("LArFormerPID multiset", pa != pb)
        la = sorted(list(a["trackPID"]) + list(a["showerPID"]))
        lb = sorted(list(b["trackPID"]) + list(b["showerPID"]))
        flip("LArPID multiset", la != lb)
        ea, eb = float(np.sum(a["showerRecoE"])), float(np.sum(b["showerRecoE"]))
        flip("sum showerRecoE >10%", rel(ea, eb) > 0.10); dE.append(eb - ea)
        sa = float(np.max(a["showerCosmicScore"])) if len(a["showerCosmicScore"]) else -9.0
        sb = float(np.max(b["showerCosmicScore"])) if len(b["showerCosmicScore"]) else -9.0
        flip("max showerCosmicScore >0.1", abs(sa - sb) > 0.1); dscore.append(sb - sa)
        flip("nuSliceFlashChi2 >10%", rel(a["nuSliceFlashChi2"], b["nuSliceFlashChi2"]) > 0.10)
        dchi.append(rel(a["nuSliceFlashChi2"], b["nuSliceFlashChi2"]))
        flip("recoNuE >10%", rel(a["recoNuE"], b["recoNuE"]) > 0.10)
        if fa and fb:
            d = math.dist((a["vtxX"], a["vtxY"], a["vtxZ"]), (b["vtxX"], b["vtxY"], b["vtxZ"]))
            dv.append(d)
            flip("|dvtx| > 1 cm (both found)", d > 1)
            flip("|dvtx| > 5 cm (both found)", d > 5)
        rows.append((key, fa, fb, int(a["nTracks"]), int(b["nTracks"]), int(a["nShowers"]), int(b["nShowers"]), ea, eb))
    nboth = sum(1 for k in common if A[k]["foundVertex"] and B[k]["foundVertex"])
    print(f">>> foundVertex: A {sum(int(A[k]['foundVertex']) for k in common)}, B {sum(int(B[k]['foundVertex']) for k in common)}, both {nboth}")
    print("\n>>> event-level flip rates (B vs A):")
    for name, k in flips.items():
        denom = nboth if "both found" in name else n
        print(f"    {name:32s} {binom(k, denom)}")
    if dv:
        dv = np.array(dv); print(f"\n>>> vertex distance (both found): median {np.median(dv):.2f} cm, 90th pct {np.percentile(dv, 90):.2f} cm, max {dv.max():.1f} cm")
    dE = np.array(dE); print(f">>> sum showerRecoE B-A [MeV]: mean {dE.mean():+.2f}, median {np.median(dE):+.2f}, std {dE.std():.2f}")
    dscore = np.array(dscore); print(f">>> max showerCosmicScore B-A: mean {dscore.mean():+.4f}, std {dscore.std():.4f}")
    dchi = np.array(dchi); print(f">>> nuSliceFlashChi2 relative diff: median {np.median(dchi):.4f}, 90th pct {np.percentile(dchi, 90):.4f}")
    if args.csv:
        with open(args.csv, "w") as fo:
            fo.write("run,subrun,event,foundA,foundB,nTrkA,nTrkB,nShwA,nShwB,sumShwE_A,sumShwE_B\n")
            for r in rows:
                fo.write(",".join(str(x) for x in (r[0] + r[1:])) + "\n")
        print(f">>> per-event rows -> {args.csv}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("mode", choices=["kp2", "ntuple"])
    ap.add_argument("--a", required=True, help="reference (e.g. Tufts)")
    ap.add_argument("--b", required=True, help="candidate (e.g. Polaris)")
    ap.add_argument("--max-index", type=int, default=None)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()
    (kp2_mode if args.mode == "kp2" else ntuple_mode)(args)


if __name__ == "__main__":
    main()
