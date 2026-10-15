#!/usr/bin/env python3
"""
Check the python detinfo utility against LArSoft's own answers stored in the
/validation table of a detinfo file (see SCHEMA.md).

    python lartpc/detinfo/tests/test_validation.py [--detector sbnd] [--path FILE.h5]
    pytest lartpc/detinfo/tests/test_validation.py        # uses DETINFO_TEST_DETECTOR (default sbnd)

Exit code 0 when every check passes.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
from lartpc.detinfo import load  # noqa: E402
from lartpc.detinfo.io import read_table  # noqa: E402


def run_checks(di, verbose=True):
    import h5py
    with h5py.File(di.path, "r") as f:
        V = read_table(f["validation"])
    pts = V["point"]
    results = []

    def check(name, ok, detail):
        results.append((name, bool(ok), detail))
        if verbose:
            print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")

    # containing TPC
    cryo, tpc = di.geometry.FindTPC(pts)
    agree = np.mean((cryo == V["cryo"]) & (tpc == V["tpc"]))
    check("FindTPC", agree == 1.0, f"agreement {agree:.4f} on {len(pts)} points")

    # per-plane projections (only for points inside a TPC, as stored)
    proj = di.project(pts)
    inside = V["tpc"] >= 0
    npl = V["wire_coordinate"].shape[1]
    wc_ref, wc = V["wire_coordinate"][inside], proj["wire_coordinate"][inside, :npl]
    d = np.nanmax(np.abs(wc - wc_ref))
    check("WireCoordinate", d < 1e-3, f"max |diff| = {d:.2e} wires")
    nw_ref, nw = V["nearest_wire"][inside], proj["nearest_wire"][inside, :npl]
    agree = np.mean(nw_ref == nw)
    check("NearestWire", agree == 1.0, f"agreement {agree:.4f}")
    tk_ref, tk = V["tick"][inside], proj["tick"][inside, :npl]
    d = np.nanmax(np.abs(tk - tk_ref))
    check("ConvertXToTicks", d < 1e-6, f"max |diff| = {d:.2e} ticks")

    # space charge
    sce_ported = type(di.sce).__name__ != "SpaceCharge"
    if "sce_pos_offsets" in V and np.isfinite(V["sce_pos_offsets"]).any() and not sce_ported:
        if verbose:
            print(f"  [SKIP] SCE: LArSoft reference values present, but no python port of the "
                  f"{di.detector} SCE provider yet (lartpc/detinfo/spacecharge.py)")
    elif "sce_pos_offsets" in V and np.isfinite(V["sce_pos_offsets"]).any():
        for name, ref, val in (
                ("SCE GetPosOffsets", V["sce_pos_offsets"], di.sce.GetPosOffsets(pts)),
                ("SCE GetEfieldOffsets", V["sce_efield_offsets"], di.sce.GetEfieldOffsets(pts)),
                ("SCE GetCalPosOffsets", V["sce_cal_pos_offsets"],
                 di.sce.GetCalPosOffsets(pts, V["sce_cal_tpcid"]))):
            d = np.nanmax(np.abs(val - ref))
            scale = np.nanmax(np.abs(ref))
            check(name, d < 1e-5 * max(scale, 1.0), f"max |diff| = {d:.2e} (max |ref| {scale:.3g})")
    elif sce_ported:
        check("SCE reference present", False, "no sce_* reference values in /validation")
    elif verbose:
        print(f"  [SKIP] SCE: no LArSoft reference values and no python port for {di.detector}")

    # photon library voxelization
    if "photonlib_voxel_id" in V and (V["photonlib_voxel_id"] > -2).any():
        try:
            pl = di.photonlib
            vid = pl.GetVoxelID(pts)
            m = V["photonlib_voxel_id"] > -2
            agree = np.mean(vid[m] == V["photonlib_voxel_id"][m])
            check("PhotonLibrary GetVoxelID", agree == 1.0, f"agreement {agree:.4f}")
            if pl.mapping is not None:
                # the mapping tool needs art, so no LArSoft reference values: check that
                # for in-TPC points the brightest channel is in the point's own cryostat
                # and among the 4 optical detectors nearest to the point
                inside = V["cryo"] >= 0
                vis = pl.GetVisibility(pts[inside])
                ch = vis.argmax(1)
                od = np.array([di.geometry.OpDetFromOpChannel(int(c)) for c in ch])
                centers = di.geometry.OpDetPositions()
                d = np.linalg.norm(pts[inside][:, None, :] - centers[None], axis=2)
                rank = (d < d[np.arange(len(od)), od][:, None]).sum(1)
                same = np.mean(np.sign(centers[od, 0]) == np.sign(pts[inside, 0]))
                ok = (vis.max(1) > 0).all() and same == 1.0 and np.mean(rank < 4) > 0.95
                check("PhotonLibrary mapping (sanity)", ok,
                      f"brightest channel in own cryostat {same:.3f}, among 4 nearest "
                      f"{np.mean(rank < 4):.3f}")
        except FileNotFoundError as e:
            check("PhotonLibrary present", False, str(e))
    return results


def test_validation():
    det = os.environ.get("DETINFO_TEST_DETECTOR", "sbnd")
    res = run_checks(load(det), verbose=True)
    failed = [r for r in res if not r[1]]
    assert not failed, failed


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--detector", default="sbnd")
    ap.add_argument("--version", default=None)
    ap.add_argument("--path", default=None)
    a = ap.parse_args()
    di = load(a.detector, a.version, path=a.path)
    print(f"validating {di.path}")
    res = run_checks(di)
    n_fail = sum(not ok for _, ok, _ in res)
    print(f"{len(res) - n_fail}/{len(res)} checks passed")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
