"""Purity counterpart to the recall-only battery gate: how BIG is the
predicted nu slice and what is in it? For each sliceid sidecar dir,
reports (charge-weighted, per event): fraction of ALL event charge put
in the nu slice, and the truth composition of that nu-slice charge
(nu-origin vs cosmic vs no-truth) via the 0.25 cm cell join.

  PYTHONPATH=./ python3 slice_purity_check.py --label X --sliceids-dir D \
      --msp-list L --sig-npy S
"""
import argparse, glob, os, re
import numpy as np, h5py
from lartpc.larformer_reco.trajfit.calo import dedup_charge
from lartpc.larformer_analysis.physics.pilot_matrix.pi0_photon_charge_flow import _cell_keys

ap = argparse.ArgumentParser()
ap.add_argument("--label", required=True)
ap.add_argument("--sliceids-dir", required=True)
ap.add_argument("--msp-list", required=True)
ap.add_argument("--sig-npy", required=True)
a = ap.parse_args()

sig = [int(x) for x in np.load(a.sig_npy)]
msp = [l.strip() for l in open(a.msp_list) if l.strip()]
idx = {}
for p in glob.glob(os.path.join(a.sliceids_dir, "sliceid_event*.h5")):
    m = re.search(r"sliceid_event0*(\d+)", os.path.basename(p))
    if m: idx[int(m.group(1))] = p

fr_nu, pur, cos_f, notruth_f, npts = [], [], [], [], []
for rank, ev in enumerate(sig):
    p = idx.get(rank)
    if p is None: continue
    with h5py.File(p, "r") as f:
        sc = f["full_slice/coord_cm"][()].astype(np.float32)
        sid = f["full_slice/slice_id"][()].astype(np.int64)
    with h5py.File(msp[ev], "r") as f:
        td = f["entry_0/triplet_data"]
        tpos = np.ascontiguousarray(td["pos"][()], np.float32)
        org = td["origin"][()].astype(np.int64) if "origin" in td else None
        _, q = dedup_charge(td["pixval"][()], td["tick"][()], td["uwire"][()],
                            td["vwire"][()], td["ywire"][()])
    cell_sid = dict(zip(_cell_keys(sc), sid))
    keys = _cell_keys(tpos)
    q_tot = q.sum(); q_nu = 0.0; q_nu_true = 0.0; q_nu_cos = 0.0; q_nu_nt = 0.0; n = 0
    for k, qv, o in zip(keys, q, (org if org is not None else np.full(len(q), -1))):
        s = cell_sid.get(k)
        if s == -5:
            q_nu += qv; n += 1
            if o == 1: q_nu_true += qv
            elif o == 2: q_nu_cos += qv
            else: q_nu_nt += qv
    if q_tot > 0 and q_nu > 0:
        fr_nu.append(q_nu / q_tot); pur.append(q_nu_true / q_nu)
        cos_f.append(q_nu_cos / q_nu); notruth_f.append(q_nu_nt / q_nu); npts.append(n)
print(f"[{a.label}] events={len(fr_nu)}")
print(f"  nu-slice charge / ALL event charge : {np.mean(fr_nu):.3f}  (median {np.median(fr_nu):.3f})")
print(f"  nu-slice PURITY (true nu-origin)   : {np.mean(pur):.3f}  (median {np.median(pur):.3f})")
print(f"  nu-slice cosmic-origin fraction    : {np.mean(cos_f):.3f}")
print(f"  nu-slice no-truth(ghost) fraction  : {np.mean(notruth_f):.3f}")
print(f"  nu-slice points/event (median)     : {np.median(npts):.0f}")
