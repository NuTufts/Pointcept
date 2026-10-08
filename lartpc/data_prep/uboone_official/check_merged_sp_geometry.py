"""Smoke-test geometry check: are the converted spacepoints where truth says?

For each merged_sp event of one fileno, compares the centroid of the truth-
matched (hasmatch==1) points carrying the leading primary's PDG with that
primary's true energy-deposit start (photon: first conversion point; others:
start point) from the truth sidecar. A wrong tick-order flag mirrors the drift
coordinate (mcc9.10 converted with -tb: x_reco ~ -x_true - 80 cm), which this
catches immediately. Also prints the x range of all points (should mostly lie
in [-40, 290] cm: in-time + out-of-time cosmics around the 0-256 cm drift).

    python3 check_merged_sp_geometry.py --truth <truth_fileno00001.h5> <merged_sp h5 ...>

Pass: |dx| median of a few cm..~20 cm (centroid vs start of an extended
object), consistent sign. Fail: x offsets of order 100 cm, negative centroids.
Data mode has no truth: just inspect the all-points x range.
"""
import argparse
import re

import h5py
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--truth", help="truth sidecar of the same fileno (MC)")
    ap.add_argument("h5", nargs="+")
    args = ap.parse_args()
    tr = h5py.File(args.truth, "r") if args.truth else None
    dx = []
    for fn in sorted(args.h5):
        entry = int(re.search(r"entry(\d+)\.h5$", fn).group(1))
        with h5py.File(fn, "r") as h:
            t = h["entry_0/triplet_data"]
            pos, pid, hm = t["pos"][:], t["pid"][:], t["hasmatch"][:]
        xs = np.percentile(pos[:, 0], [1, 50, 99]).round(0) if len(pos) else "-"
        line = f"entry {entry:3d}: npts {len(pos):7d} real {int(hm.sum()):6d} all-x pct1/50/99 {xs}"
        if tr is not None:
            g = tr[f"entry_{entry}"]
            pp = g["truePrimPart"]
            if len(pp["PDG"]) == 0:
                print(line + " | no truth primaries")
                continue
            k = int(np.argmax(pp["E"][:]))
            pdg = int(pp["PDG"][k])
            sp = g["trueSimPart"]
            rows = np.where((sp["PDG"][:] == pdg) & (sp["Process"][:] == 0))[0]
            if len(rows):
                r = rows[0]
                key = ("EDepX", "EDepY", "EDepZ") if pdg == 22 else ("X", "Y", "Z")
                ref = np.array([sp[c][r] for c in key])
            else:
                ref = np.array([pp[c][k] for c in ("X", "Y", "Z")])
            m = (np.abs(pid) == abs(pdg)) & (hm == 1)
            if m.sum() and np.all(np.isfinite(ref)):
                c = pos[m].mean(0)
                dx.append(c[0] - ref[0])
                line += f" | pdg {pdg} true {ref.round(1)} reco-centroid {c.round(1)} (n={int(m.sum())})"
            else:
                line += f" | pdg {pdg} true {ref.round(1)} (no matched points)"
        print(line)
    if dx:
        dx = np.array(dx)
        print(f"centroid - true x: median {np.median(dx):.1f} cm, |dx|>50 cm in {np.mean(np.abs(dx) > 50):.0%} of {len(dx)} events "
              f"-> {'FAIL: check the tick-order flag (TICK)' if np.median(np.abs(dx)) > 50 else 'ok'}")


if __name__ == "__main__":
    main()
