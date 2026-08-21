"""A2 label completion (SLICER_RETRAIN_PLAN): attach unlabeled triplets
in the thin shell around a particle's labeled points to that particle.

Adoption rule (v1): candidate point is adopted by its NEAREST labeled
donor (3D, <= --radius cm) IFF it also passes the image-shell guard vs
that donor: |d(row=tick//6)| <= --drow AND |d(wire)| <= --dwire on ALL
three planes (the section-23 measurement: 95% of true periphery lies
within drow<=2, dwire<=2; requiring all three planes + 3D proximity
suppresses ghost adoption). Adopted points inherit trackid/pid/origin
from the donor and get hasmatch=1; a new `label_completed` dataset marks
them (0 original label, 1 adopted, 255 untouched-unlabeled).

Output = copy of the input h5 with the label datasets replaced
(originals preserved as `<name>_precomplete`).

    python3 complete_labels.py --h5 in1.h5 [in2.h5 ...] --out-dir D \
        [--radius 0.75] [--drow 2] [--dwire 2]
"""
import argparse
import os
import shutil

import numpy as np
import h5py
from scipy.spatial import cKDTree

LABEL_KEYS = ("trackid", "pid", "origin", "hasmatch")


def complete_file(path, out_dir, radius, drow, dwire):
    if out_dir is None:
        out = path  # in-place: originals preserved as *_precomplete
    else:
        out = os.path.join(out_dir, os.path.basename(path))
        shutil.copy2(path, out)
    with h5py.File(out, "r+") as f:
        td = f["entry_0/triplet_data"]
        if "label_completed" in td:
            # idempotency guard: a second pass would use adopted points
            # as donors and cascade-grow the labels
            return -1, -1, -1
        pos = np.ascontiguousarray(td["pos"][()], np.float64)
        tid = td["trackid"][()].astype(np.int64)
        row = td["tick"][()].astype(np.int64) // 6
        uw = td["uwire"][()].astype(np.int64)
        vw = td["vwire"][()].astype(np.int64)
        yw = td["ywire"][()].astype(np.int64)

        donors = np.flatnonzero(tid > 0)
        cands = np.flatnonzero(tid <= 0)
        status = np.full(len(tid), 255, np.uint8)
        status[donors] = 0
        n_adopt = 0
        if len(donors) and len(cands):
            tree = cKDTree(pos[donors])
            d, ji = tree.query(pos[cands], distance_upper_bound=radius)
            near = np.isfinite(d)
            ci = cands[near]
            di = donors[ji[near]]
            ok = ((np.abs(row[ci] - row[di]) <= drow)
                  & (np.abs(uw[ci] - uw[di]) <= dwire)
                  & (np.abs(vw[ci] - vw[di]) <= dwire)
                  & (np.abs(yw[ci] - yw[di]) <= dwire))
            ci, di = ci[ok], di[ok]
            n_adopt = len(ci)
            new = {}
            for k in LABEL_KEYS:
                arr = td[k][()]
                if f"{k}_precomplete" not in td:
                    td.create_dataset(f"{k}_precomplete", data=arr,
                                      compression="gzip",
                                      compression_opts=4)
                if k == "hasmatch":
                    arr[ci] = 1
                else:
                    arr[ci] = td[k][()][di]
                new[k] = arr
            for k in LABEL_KEYS:
                del td[k]
                td.create_dataset(k, data=new[k], compression="gzip",
                                  compression_opts=4)
            status[ci] = 1
        if "label_completed" in td:
            del td["label_completed"]
        td.create_dataset("label_completed", data=status,
                          compression="gzip", compression_opts=4)
        f["entry_0"].attrs["label_completion"] = (
            f"r={radius}cm drow={drow} dwire={dwire} "
            f"adopted={n_adopt} donors={len(donors)}")
    return len(donors), len(cands), n_adopt


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--h5", nargs="+", required=True)
    ap.add_argument("--out-dir", default=None,
                    help="write completed copies here; OMIT for IN-PLACE "
                         "completion (originals kept as *_precomplete "
                         "datasets in the same file)")
    ap.add_argument("--radius", type=float, default=0.5)
    ap.add_argument("--drow", type=int, default=2)
    ap.add_argument("--dwire", type=int, default=2)
    args = ap.parse_args()
    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)
    n_err = 0
    for p in args.h5:
        try:
            nd, nc, na = complete_file(p, args.out_dir, args.radius,
                                       args.drow, args.dwire)
        except Exception as e:  # per-file fault tolerance: record, go on
            n_err += 1
            print(f"ERROR {os.path.basename(p)}: {e!r}")
            continue
        if nd < 0:
            print(f"{os.path.basename(p)}: ALREADY COMPLETED, skipped")
            continue
        print(f"{os.path.basename(p)}: donors={nd} candidates={nc} "
              f"adopted={na} ({na / max(nc, 1):.3f} of candidates, "
              f"+{na / max(nd, 1):.3f} rel to donors)")
    if n_err:
        print(f"[complete_labels] {n_err} files errored")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
