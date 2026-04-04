"""Dump keypoint data from an HDF5 event file produced by SimChTripletLabelMaker."""
import sys
import argparse
import h5py
import numpy as np

parser = argparse.ArgumentParser(description="Dump keypoint info from HDF5 event file.")
parser.add_argument("h5file", type=str, help="Path to the HDF5 file")
parser.add_argument("--entry", type=str, default="entry_0", help="Entry group name (default: entry_0)")
args = parser.parse_args()

KPTYPE_NAMES = {
    0: "NuVertex",
    1: "TrackStart",
    2: "TrackEnd",
    3: "ShowerStart",
    4: "Michel",
    5: "Delta",
}

with h5py.File(args.h5file, "r") as f:
    entry_grp = f[args.entry]

    # List all groups in the entry
    print(f"Groups in /{args.entry}: {list(entry_grp.keys())}")

    if "mckeypoints" not in entry_grp:
        print("\nNo 'mckeypoints' group found in this entry.")
        sys.exit(0)

    kp_grp = entry_grp["mckeypoints"]
    print(f"\nDatasets in mckeypoints: {list(kp_grp.keys())}")

    # Load all available datasets
    pos = kp_grp["pos"][:] if "pos" in kp_grp else None
    startpos = kp_grp["startpos"][:] if "startpos" in kp_grp else None
    imgcoord = kp_grp["imgcoord"][:] if "imgcoord" in kp_grp else None
    kptype = kp_grp["kptype"][:] if "kptype" in kp_grp else None
    pid = kp_grp["pid"][:] if "pid" in kp_grp else None
    trackid = kp_grp["trackid"][:] if "trackid" in kp_grp else None

    nkp = len(pos) if pos is not None else 0
    print(f"\nNumber of keypoints: {nkp}")

    if nkp == 0:
        sys.exit(0)

    # Print shapes
    for name, arr in [("pos", pos), ("startpos", startpos), ("imgcoord", imgcoord),
                       ("kptype", kptype), ("pid", pid), ("trackid", trackid)]:
        if arr is not None:
            print(f"  {name:>10s}: shape={arr.shape}  dtype={arr.dtype}")

    # Summary by type
    print("\n--- Summary by keypoint type ---")
    if kptype is not None:
        for t in sorted(np.unique(kptype)):
            mask = kptype == t
            name = KPTYPE_NAMES.get(int(t), f"Unknown({t})")
            print(f"  {name:>12s} (type={t}): count={mask.sum()}")

    # Per-keypoint dump
    print("\n--- Per-keypoint details ---")
    header = f"{'idx':>4s}  {'type':>12s}  {'pid':>6s}  {'trkid':>6s}  {'pos_x':>10s} {'pos_y':>10s} {'pos_z':>10s}"
    if imgcoord is not None:
        header += f"  {'U':>5s} {'V':>5s} {'Y':>5s} {'row':>5s}"
    if startpos is not None:
        header += f"  {'spos_x':>10s} {'spos_y':>10s} {'spos_z':>10s}"
    print(header)
    print("-" * len(header))

    for i in range(nkp):
        t = int(kptype[i]) if kptype is not None else -1
        tname = KPTYPE_NAMES.get(t, f"Unk({t})")
        p = int(pid[i]) if pid is not None else -1
        tid = int(trackid[i]) if trackid is not None else -1
        px, py, pz = pos[i] if pos is not None else (0, 0, 0)

        line = f"{i:4d}  {tname:>12s}  {p:6d}  {tid:6d}  {px:10.2f} {py:10.2f} {pz:10.2f}"

        if imgcoord is not None:
            ic = imgcoord[i]
            line += f"  {ic[0]:5d} {ic[1]:5d} {ic[2]:5d} {ic[3]:5d}"

        if startpos is not None:
            sx, sy, sz = startpos[i]
            line += f"  {sx:10.2f} {sy:10.2f} {sz:10.2f}"

        print(line)
