"""Smoke test for compute_particle_labels (Stage 3 GT extraction).

Loads a merged H5, runs `compute_particle_labels` on the nu-origin tracks,
prints per-particle stats, and (when --inspect-merges is set) lists every
trackid that got merged into a surviving above-threshold particle.

Default test files: the local fileno=1 events.

Usage:
  ./run_in_container.sh python lartpc/data_prep/validation/test_particle_labels.py \\
      [--merged-h5 /path/...]
      [--inspect-merges]
      [--ke-thresh-other 60]
"""

import argparse
import os
import sys

import h5py
import numpy as np


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, REPO_ROOT)

from lartpc.data_prep.labels.slice_labels import (  # noqa: E402
    GHOST_SLICE_ID,
    compute_particle_labels,
    compute_slice_labels,
    summarize_particle_labels,
)


PDG_NAMES = {
    11: "e-",   -11: "e+",
    13: "mu-",  -13: "mu+",
    22: "gamma",
    111: "pi0", 211: "pi+", -211: "pi-",
    2212: "p",  2112: "n",
    321:  "K+", -321: "K-",
}


def _pdg(p):
    return PDG_NAMES.get(int(p), f"PDG:{int(p)}")


def smoke_one(merged_h5, inspect_merges=False, ke_thresh_other=60.0):
    print(f"\n=== {os.path.basename(merged_h5)} ===")
    with h5py.File(merged_h5, "r") as f:
        e0 = f["entry_0"]
        run    = int(e0.attrs.get("run", -1))
        subrun = int(e0.attrs.get("subrun", -1))
        event  = int(e0.attrs.get("event", -1))
        mpt = e0["mc_particle_tree"]
        td  = e0["triplet_data"]
        sp_trackid  = td["trackid"][:]
        sp_hasmatch = td["hasmatch"][:] if "hasmatch" in td else None
        # Reference: also call compute_slice_labels with merge_nu_slices=False
        # so we can count how many primaries each particle merges across.
        slice_info = compute_slice_labels(
            mpt, sp_trackid, sp_hasmatch, merge_nu_slices=False,
        )
        particle_info = compute_particle_labels(
            mpt, sp_trackid, sp_hasmatch,
            other_ke_threshold=ke_thresh_other,
        )

        # Per-track KE for the inspect-merges report
        mc_tids = mpt["trackid"][:].astype(np.int64)
        mc_pids = mpt["pid"][:].astype(np.int64)
        mc_ke   = mpt["energy_mev"][:].astype(np.float32)
        mc_origin = mpt["origin"][:].astype(np.int64)
        tid_to_row = {int(t): i for i, t in enumerate(mc_tids)}

    print(f"  (run, subrun, event) = ({run}, {subrun}, {event})")
    print(f"  n_sp_total = {sp_trackid.shape[0]}")
    if sp_hasmatch is not None:
        print(f"  n_sp_real  = {int((sp_hasmatch != 0).sum())}")
    # Counts from compute_slice_labels (treats each nu primary as its own slice).
    n_nu_prim = int((slice_info["primary_origin"] == 1).sum())
    n_cos_prim = int((slice_info["primary_origin"] == 2).sum())
    print(f"  legacy slice_labels (merge_nu_slices=False):  "
          f"{len(slice_info['primary_trackid'])} slices  "
          f"(nu={n_nu_prim}, cosmic={n_cos_prim})")
    print(f"  particle_labels:")
    for line in summarize_particle_labels(particle_info).split("\n"):
        print(f"    {line}")
    print()
    print(f"  surviving particles (KE-above-threshold, nu-origin):")
    print(f"    {'qid':>3s}  {'pdg':>5s}  {'name':>6s}  "
          f"{'KE[MeV]':>9s}  {'n_sp':>6s}  {'n_merged':>9s}")
    for k, tid in enumerate(particle_info["primary_trackid"]):
        members = particle_info["slice_member_trackids"][k]
        print(f"    {k:>3d}  "
              f"{int(particle_info['primary_pid'][k]):>5d}  "
              f"{_pdg(particle_info['primary_pid'][k]):>6s}  "
              f"{float(particle_info['primary_ke_MeV'][k]):>9.2f}  "
              f"{int(particle_info['primary_n_spacepoints'][k]):>6d}  "
              f"{len(members):>9d}")

    if inspect_merges:
        print()
        print("  per-surviving-particle: trackids merged into it (with KE):")
        for k, tid in enumerate(particle_info["primary_trackid"]):
            members = particle_info["slice_member_trackids"][k]
            if len(members) <= 1:
                continue  # nothing was merged into it
            host_pid = int(particle_info["primary_pid"][k])
            host_ke  = float(particle_info["primary_ke_MeV"][k])
            print(f"    [qid={k}]  HOST  tid={int(tid)}  "
                  f"pdg={host_pid}({_pdg(host_pid)})  "
                  f"KE={host_ke:.2f}  (n_merged={len(members) - 1})")
            for m in members:
                if int(m) == int(tid):
                    continue
                idx = tid_to_row.get(int(m), None)
                if idx is None:
                    print(f"        merged tid={int(m)}  (not in graph)")
                    continue
                print(f"        merged tid={int(m):>8d}  "
                      f"pdg={int(mc_pids[idx]):>5d}({_pdg(mc_pids[idx]):>5s})  "
                      f"KE={float(mc_ke[idx]):>8.2f}  "
                      f"origin={int(mc_origin[idx])}")

    return particle_info


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--merged-h5", action="append", default=None,
                    help="Path to a merged H5 (repeatable). Default: "
                         "the local fileno=1 entries 0/1/2.")
    ap.add_argument("--inspect-merges", action="store_true",
                    help="Print the per-host trackid merge structure.")
    ap.add_argument("--ke-thresh-other", type=float, default=60.0,
                    help="KE threshold for PDGs not in the default table "
                         "(default 60 MeV — protons/neutrons/K).")
    args = ap.parse_args()

    paths = args.merged_h5 or [
        f"/mnt/ddrive/data/ub_on_tufts/h5/bnb_nu_pi0filter_corsika/"
        f"merged_h5/000/000/merged_bnb_nu_pi0filter_corsika"
        f"_fileno00001_entry00000{i}.h5"
        for i in range(3)
    ]
    print(f"Inspecting {len(paths)} merged H5 file(s)…")
    for p in paths:
        if not os.path.exists(p):
            print(f"  SKIP missing: {p}")
            continue
        smoke_one(p, inspect_merges=args.inspect_merges,
                  ke_thresh_other=args.ke_thresh_other)
    print("\nSmoke test done.")


if __name__ == "__main__":
    main()
