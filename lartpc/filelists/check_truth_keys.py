#!/usr/bin/env python3
"""
WP1 smoke check: verify the MC files behind a file list carry the truth keys
Phase 0.5 needs when the dataloader switches to data_only=False +
true_points_only=True (see phase0_phase05_implementation_plan.md §1.2).

Run inside the pointcept container with the dataset squashfs bound at /data:

  apptainer exec --bind $SQSH:/data:image-src=/,ro /projects/u6jo/containers/pointcept-sandbox \
      python3 lartpc/filelists/check_truth_keys.py lartpc/filelists/h5list_v3_mc_diag1k.txt --n 20

For an EXTBNB list, pass --expect-no-truth (real data must NOT have truth keys).
"""
import argparse
import sys

import h5py
import numpy as np

TRUTH_KEYS = ["origin", "ssnet_label", "pid", "hasmatch", "trackid"]
ALWAYS_KEYS = ["pos", "pixval", "uwire", "vwire", "ywire"]
# larmatch_score is OPTIONAL: EXTBNB files have it, MC files do not
# (verified 2026-07-13 — so filter_larmatch silently no-ops on MC events).


def check_file(path, expect_truth):
    problems = []
    with h5py.File(path, "r") as f:
        td = f["/entry_0/triplet_data"]
        n = td["pos"].shape[0]
        has_larmatch = "larmatch_score" in td
        for k in ALWAYS_KEYS:
            if k not in td:
                problems.append(f"missing {k}")
        for k in TRUTH_KEYS:
            present = k in td
            if expect_truth and not present:
                problems.append(f"missing truth key {k}")
            if not expect_truth and present:
                problems.append(f"unexpected truth key {k} (data file?)")
        if expect_truth and not problems:
            hasmatch = np.asarray(td["hasmatch"])
            origin = np.asarray(td["origin"])
            n_real = int((hasmatch == 1).sum())
            n_ghost = int((hasmatch == 0).sum())
            n_nu = int((origin == 1).sum())
            if n_real == 0:
                problems.append("hasmatch has no real points")
            if len(hasmatch) != n:
                problems.append("hasmatch length != pos length")
            has_vtx = ("mckeypoints" in f["/entry_0"]
                       and (np.asarray(f["/entry_0/mckeypoints/kptype"]) == 0).any())
            return problems, dict(n=n, n_real=n_real, n_ghost=n_ghost,
                                  n_nu=n_nu, nu_vertex=has_vtx,
                                  larmatch=has_larmatch)
    return problems, dict(n=n, larmatch=has_larmatch)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("filelist")
    ap.add_argument("--n", type=int, default=20, help="files to check (spread over the list)")
    ap.add_argument("--expect-no-truth", action="store_true")
    args = ap.parse_args()

    with open(args.filelist) as f:
        paths = [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]
    idx = np.linspace(0, len(paths) - 1, min(args.n, len(paths)), dtype=int)

    n_bad, ghost_fracs, n_with_vtx = 0, [], 0
    for i in idx:
        try:
            problems, info = check_file(paths[i], expect_truth=not args.expect_no_truth)
        except Exception as e:
            problems, info = [f"exception: {e}"], {}
        if problems:
            n_bad += 1
            print(f"BAD  {paths[i]}: {'; '.join(problems)}")
        else:
            if "n_ghost" in info:
                ghost_fracs.append(info["n_ghost"] / max(info["n"], 1))
                n_with_vtx += int(info["nu_vertex"])
            print(f"OK   {paths[i]}: {info}")

    print(f"\nchecked {len(idx)} files, {n_bad} bad")
    if ghost_fracs:
        print(f"ghost fraction: mean={np.mean(ghost_fracs):.3f} "
              f"min={np.min(ghost_fracs):.3f} max={np.max(ghost_fracs):.3f}")
        print(f"files with nu vertex keypoint: {n_with_vtx}/{len(ghost_fracs)}")
    sys.exit(1 if n_bad else 0)


if __name__ == "__main__":
    main()
