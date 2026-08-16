"""Decompose the pi0 signal photons' charge by the cascade's per-SP slice_id
(from --slice-ids-only sidecars), old vs new chain: is missing photon charge
GHOSTED (deghoster), put in a COSMIC slice (slicer classification), or kept
but UNCLUSTERED?

slice_id: -2 ghost, -1 kept-unclustered, -5 nu slice, q>=0 cosmic slice q.
Same 0.25 cm cell-inheritance join as pi0_photon_charge_flow.py; charge =
dedup q_comb over the full triplet set.

    PYTHONPATH=./ python3 pi0_photon_sliceid_decomp.py \
        --flow-dir .../photon_charge_flow \
        --pilot-list .../merged_sp_mcc9_bnbnu_satfix_pilot10k.txt \
        --old-dir .../sliceids_old --new-dir .../sliceids_new
"""
import argparse
import glob
import os
import re
import sys

import numpy as np
import h5py

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..", "..", "..")))
from lartpc.larformer_reco.trajfit.calo import dedup_charge   # noqa: E402
from lartpc.larformer_analysis.physics.pilot_matrix.pi0_photon_charge_flow \
    import _cell_keys                                          # noqa: E402

CATS = ["nu slice", "cosmic slice", "unclustered", "ghosted", "no cell"]


def sliceid_index(d):
    out = {}
    for p in glob.glob(os.path.join(d, "**", "sliceid_event*.h5"),
                       recursive=True):
        m = re.search(r"sliceid_event0*(\d+)", os.path.basename(p))
        if m:
            out[int(m.group(1))] = p
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--flow-dir", required=True,
                    help="photon_charge_flow output dir (records + sig events)")
    ap.add_argument("--pilot-list", required=True)
    ap.add_argument("--old-dir", required=True)
    ap.add_argument("--new-dir", required=True)
    args = ap.parse_args()

    rec = np.load(os.path.join(args.flow_dir, "photon_records.npz"))
    sig_evs = np.load(os.path.join(args.flow_dir, "sig_event_indices.npy"))
    listpos = {int(e): i for i, e in enumerate(sig_evs)}   # pilot idx -> 174-idx
    msp = [l.strip() for l in open(args.pilot_list) if l.strip()]
    sid_idx = {"old": sliceid_index(args.old_dir),
               "new": sliceid_index(args.new_dir)}
    print(f"sliceid files: old {len(sid_idx['old'])} new {len(sid_idx['new'])}")

    fr = {c: {k: [] for k in CATS} for c in ("old", "new")}
    qs = []
    by_ev = {}
    for ev in sig_evs:
        with h5py.File(msp[int(ev)], "r") as f:
            td = f["entry_0/triplet_data"]
            tpos = np.ascontiguousarray(td["pos"][()], np.float32)
            trackid = td["trackid"][()].astype(np.int64)
            _, q_comb = dedup_charge(td["pixval"][()], td["tick"][()],
                                     td["uwire"][()], td["vwire"][()],
                                     td["ywire"][()])
        by_ev[int(ev)] = (tpos, trackid, q_comb)

    sel = np.ones(len(rec["event"]), bool)
    for j in np.nonzero(sel)[0]:
        ev, tid = int(rec["event"][j]), int(rec["tid"][j])
        tpos, trackid, q_comb = by_ev[ev]
        m = trackid == tid
        qt = float(q_comb[m].sum())
        if qt <= 0:
            continue
        qs.append(qt)
        tkeys = _cell_keys(tpos[m])
        for chain in ("old", "new"):
            path = sid_idx[chain].get(listpos[ev])
            cell_sid = {}
            if path is not None:
                with h5py.File(path, "r") as f:
                    sc = f["full_slice/coord_cm"][()].astype(np.float32)
                    sid = f["full_slice/slice_id"][()].astype(np.int64)
                for k, s in zip(_cell_keys(sc), sid):
                    cell_sid[k] = s
            q_by = dict.fromkeys(CATS, 0.0)
            for k, qv in zip(tkeys, q_comb[m]):
                s = cell_sid.get(k)
                if s is None:
                    q_by["no cell"] += qv
                elif s == -5:
                    q_by["nu slice"] += qv
                elif s == -2:
                    q_by["ghosted"] += qv
                elif s == -1:
                    q_by["unclustered"] += qv
                else:
                    q_by["cosmic slice"] += qv
            for k in CATS:
                fr[chain][k].append(q_by[k] / qt)

    q = np.asarray(qs)
    nph = len(q)
    lines = [f"pi0 photon charge by cascade slice_id — {nph} photons "
             f"(dedup q_comb, charge-weighted means)", "",
             f"{'':16s}{'OLD':>10s}{'NEW':>10s}"]
    for k in CATS:
        lines.append(f"{k:16s}{np.average(fr['old'][k], weights=q):10.3f}"
                     f"{np.average(fr['new'][k], weights=q):10.3f}")
    lines.append("")
    for c in ("old", "new"):
        cos = np.asarray(fr[c]["cosmic slice"])
        gh = np.asarray(fr[c]["ghosted"])
        lines.append(f"photons with >0.5 of charge in a cosmic slice ({c}): "
                     f"{int((cos > 0.5).sum()):3d}/{nph}   ghosted>0.5: "
                     f"{int((gh > 0.5).sum()):3d}/{nph}")
    txt = "\n".join(lines)
    print("\n" + txt)
    with open(os.path.join(args.flow_dir, "sliceid_decomp.txt"), "w") as f:
        f.write(txt + "\n")
    np.savez(os.path.join(args.flow_dir, "sliceid_decomp.npz"),
             q_total=q, **{f"{c}_{k.replace(' ', '_')}": np.asarray(fr[c][k])
                           for c in ("old", "new") for k in CATS})


if __name__ == "__main__":
    main()
