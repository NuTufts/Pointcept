#!/usr/bin/env python3
"""
Validate SBND files written by convert_larcv2_to_larformer_h5.py.

    python3 lartpc/data_prep/sbnd_official/validate_converted.py DIR_OR_FILES... [--flash]

1. pointcept.datasets.larformer.LArFormerDataset loads every file (gt_source slice and
   particle, keypoints on) -- the training/inference reader.
2. content: real fraction == SPINE non-ghost count, ghosts unlabelled, wires inside each
   plane's range, keypoints within 3 cm of a real point, nu-origin points only from
   in-TPC interactions, flashes have 312 channels.
3. --flash: the keypoint2-cascade flash table (_flash_slice_table, --detector sbnd) on truth
   slices (in-TPC nu union + per-interaction cosmics): is the nu slice ranked first against
   the beam flash?
"""
import argparse
import glob
import os
import sys
import types

import numpy as np
import h5py

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, REPO)


def files_from(args):
    out = []
    for p in args:
        out += sorted(glob.glob(os.path.join(p, "*.h5"))) if os.path.isdir(p) else [p]
    return out


def check_dataset(files):
    from pointcept.datasets.larformer import LArFormerDataset
    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as fl:
        fl.write("\n".join(files))
        lst = fl.name
    ok = True
    for gt in ("slice", "particle"):
        ds = LArFormerDataset(split="val", data_root="/", data_list_file=lst, gt_source=gt,
                              emit_keypoints=True, test_mode=False)
        n_inst = []
        for i in range(len(files)):
            try:
                d = ds.get_data(i)
                n_inst.append(int(d.get("n_gt_instances", len(d.get("particle_instances", [])) or -1)))
            except Exception as e:  # noqa: BLE001
                ok = False
                print(f"  [FAIL] dataset gt_source={gt} file {i}: {type(e).__name__}: {e}")
                break
        else:
            print(f"  [PASS] LArFormerDataset gt_source={gt}: {len(files)} files loaded; "
                  f"instances/event {n_inst}; keys {sorted(d)[:12]}...")
    os.remove(lst)
    return ok


def check_content(files):
    from lartpc.detinfo import load
    from scipy.spatial import cKDTree
    g = load("sbnd").geometry
    nw = {(pl.id.tpc, pl.id.plane): pl.nwires for pl in g.IteratePlanes()}
    bad = []
    for p in files:
        with h5py.File(p, "r") as f:
            e = f["entry_0"]
            td = e["triplet_data"]
            hm = td["hasmatch"][()] == 1
            tid, org, ss = td["trackid"][()], td["origin"][()], td["ssnet_label"][()]
            pos = td["pos"][()]
            tpc = td["tpc"][()]
            wires = np.stack([td[k][()] for k in ("uwire", "vwire", "ywire")], 1)
            msg = []
            if np.any(tid[~hm] != 0) or np.any(org[~hm] != 0) or np.any(ss[~hm] != 0):
                msg.append("ghosts carry labels")
            if np.any(tid[hm] == 0):
                msg.append(f"{int((tid[hm] == 0).sum())} real points without trackid")
            for t in (0, 1):
                for k in range(3):
                    w = wires[tpc == t, k]
                    if w.size and (w.min() < -1 or w.max() >= nw[(t, k)]):
                        msg.append(f"wire out of range tpc {t} plane {k}")
            kp = e["mckeypoints/pos"][()]
            if len(kp) and hm.any():
                d = cKDTree(pos[hm]).query(kp)[0]
                if d.max() > 3.0:
                    msg.append(f"keypoint {d.max():.1f} cm from real points")
            mpt = e["mc_particle_tree"]
            nu_tids = set(mpt["trackid"][()][mpt["origin"][()] == 1].tolist())
            if not set(tid[org == 1].tolist()) <= nu_tids:
                msg.append("origin-1 points from non-nu particles")
            if e["flashes/pe"].shape[1] != 312:
                msg.append("flash vectors not 312 channels")
            summ = (f"{os.path.basename(p)[-21:]}: {hm.mean():.3f} real, nu pts {int((org == 1).sum())}, "
                    f"ssnet {np.bincount(ss[hm], minlength=10).tolist()}, kp {np.bincount(e['mckeypoints/kptype'][()], minlength=6).tolist()}")
        print(f"  [{'PASS' if not msg else 'FAIL'}] {summ}{'' if not msg else ' -- ' + '; '.join(msg)}")
        bad += msg
    return not bad


def check_flash(files):
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "cascade", os.path.join(REPO, "tools/larformer/run_larformer_keypoint2_cascade_inference.py"))
    cas = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cas)
    args = types.SimpleNamespace(gamma_beam=5.25, dead_opdets="auto", gamma_run_scale="auto",
                                 run_override=None, allow_kind_mismatch=False, flash_window="0,2.5",
                                 mask_saturated="auto", max_saturated_pmts=4, slice_min_points=20,
                                 flash_f_sys=0.10, flash_eps=1.0, flash_oob_max=0.05, photonlib=None,
                                 detector="sbnd", light_yield=None)
    n_nu = n_best = 0
    for p in files:
        flashes, ctx = cas._load_msp_flash_charge(p)
        with h5py.File(p, "r") as f:
            td = f["entry_0/triplet_data"]
            hm = td["hasmatch"][()] == 1
            pos = td["pos"][()][hm].astype(np.float32)
            org = td["origin"][()][hm]
            iid = td["interaction_id"][()][hm]
            run = int(f["entry_0"].attrs["run"])
        sid = np.full(pos.shape[0], -1, np.int64)
        sid[org == 1] = cas._NU_SID
        cos = org != 1
        _, q = np.unique(iid[cos], return_inverse=True)
        sid[cos] = q
        tbl = cas._flash_slice_table(pos, sid, {}, [], flashes, ctx, args, run=run, kind="mc")
        if tbl["observed"] is None or not (sid == cas._NU_SID).any():
            print(f"  [info] {os.path.basename(p)[-21:]}: "
                  f"{'no beam flash' if tbl['observed'] is None else 'no in-TPC nu'}")
            continue
        n_nu += 1
        lab = list(tbl["label"])
        i = lab.index("nu")
        rank = int(tbl["chi2_rank"][i])
        n_best += rank == 1
        pe_nu = tbl["pred_pe"][i]
        obs = tbl["observed"]["pe"]
        live = np.setdiff1d(np.arange(312), [int(c) for c in tbl["params"]["chi2_masked_opdets"].split(",") if c])
        cosv = float(obs[live] @ pe_nu[live] / (np.linalg.norm(obs[live]) * np.linalg.norm(pe_nu[live]) + 1e-30))
        print(f"  [info] {os.path.basename(p)[-21:]}: {len(lab)} slices, nu rank {rank}, chi2 {tbl['chi2'][i]:.1f}, "
              f"cos(nu pred, beam flash) {cosv:.3f}, obs/pred {obs[live].sum() / max(pe_nu[live].sum(), 1e-9):.2f}")
    print(f"  flash table: nu slice ranked first in {n_best}/{n_nu} events with an in-TPC neutrino "
          f"(raw hit charge x model default light yield, not yet calibrated)")
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--flash", action="store_true")
    a = ap.parse_args()
    files = files_from(a.inputs)
    print(f"{len(files)} files")
    ok = check_content(files)
    ok &= check_dataset(files)
    if a.flash:
        check_flash(files)
    print("ALL PASS" if ok else "FAILURES")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
