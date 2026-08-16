"""Photon charge keep-vs-tau curves + context-proximity recall profiles for
ANY set of deghoster p_real dumps (schema: sliceid_event{i}.h5 with
full_slice/{coord_cm,deghost_p_real}, i = index into the 174-event pi0
signal list), on the run3b overlay pilot.

Consumes both the --slice-ids-only cascade sidecars (v7 baselines) and
run_deghost_eval --save-preal dumps (new checkpoints) interchangeably.

    PYTHONPATH=./ python3 photon_keep_from_preal.py \
        --flow-dir .../photon_charge_flow \
        --pilot-list .../merged_sp_mcc9_bnbnu_satfix_pilot10k.txt \
        --models v7-lora=.../sliceids_old v7-ftdec=.../sliceids_new \
                 p5b3-dec=.../preal_p5b3dec p5b3-lora=.../preal_p5b3lora
"""
import argparse
import glob
import os
import re
import sys

import numpy as np
import h5py
from scipy.spatial import cKDTree

sys.path.insert(0, ".")
from lartpc.larformer_analysis.physics.pilot_matrix.pi0_photon_charge_flow \
    import _cell_keys  # noqa: E402
from lartpc.larformer_reco.trajfit.calo import dedup_charge  # noqa: E402

TAUS = [0.05, 0.1, 0.2, 0.35, 0.5]
DBINS = np.array([0.0, 2.0, 5.0, 10.0, 20.0, 50.0, 1e9])
DLABELS = ["0-2", "2-5", "5-10", "10-20", "20-50", ">50"]


def preal_index(d):
    out = {}
    for p in glob.glob(os.path.join(d, "**", "sliceid_event*.h5"),
                       recursive=True):
        m = re.search(r"sliceid_event0*(\d+)", os.path.basename(p))
        if m:
            out[int(m.group(1))] = p
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--flow-dir", required=True)
    ap.add_argument("--pilot-list", required=True)
    ap.add_argument("--models", nargs="+", required=True,
                    help="label=preal_dir pairs")
    ap.add_argument("--prox-tau", type=float, default=0.2,
                    help="tau for the proximity recall profile (all models)")
    args = ap.parse_args()

    models = dict(m.split("=", 1) for m in args.models)
    idx = {lab: preal_index(d) for lab, d in models.items()}
    for lab, ix in idx.items():
        print(f"{lab}: {len(ix)} p_real files ({models[lab]})")

    rec = np.load(os.path.join(args.flow_dir, "photon_records.npz"))
    sig = np.load(os.path.join(args.flow_dir, "sig_event_indices.npy"))
    listpos = {int(e): i for i, e in enumerate(sig)}
    msp = [l.strip() for l in open(args.pilot_list) if l.strip()]

    labs = list(models)
    keep_num = {l: np.zeros(len(TAUS)) for l in labs}
    keep_den = 0.0
    nocell = {l: 0.0 for l in labs}
    prox = {l: dict(kept=np.zeros(len(DBINS) - 1),
                    tot=np.zeros(len(DBINS) - 1)) for l in labs}

    for ev in sig:
        with h5py.File(msp[int(ev)], "r") as f:
            td = f["entry_0/triplet_data"]
            tpos = np.ascontiguousarray(td["pos"][()], np.float32)
            trackid = td["trackid"][()].astype(np.int64)
            origin = td["origin"][()].astype(np.int64)
            hasm = td["hasmatch"][()].astype(np.int64)
            _, q = dedup_charge(td["pixval"][()], td["tick"][()],
                                td["uwire"][()], td["vwire"][()],
                                td["ywire"][()])
        tids = [int(t) for e2, t in zip(rec["event"], rec["tid"])
                if int(e2) == int(ev)]
        ph = np.isin(trackid, tids)
        nu_real = (origin == 1) & (hasm == 1)
        datacos = (origin != 1) & (hasm == 0)
        if not ph.any():
            continue
        keep_den += q[ph].sum()
        ph_keys = _cell_keys(tpos[ph])
        nu_keys = _cell_keys(tpos[nu_real]) if nu_real.any() else []
        d_nu = (cKDTree(tpos[datacos]).query(tpos[nu_real], k=1)[0]
                if (nu_real.any() and datacos.any()) else None)
        for lab in labs:
            path = idx[lab].get(listpos[int(ev)])
            if path is None:
                nocell[lab] += q[ph].sum()
                continue
            with h5py.File(path, "r") as f:
                sc = f["full_slice/coord_cm"][()].astype(np.float32)
                pr = f["full_slice/deghost_p_real"][()].astype(np.float64)
            cell_pr = dict(zip(_cell_keys(sc), pr))
            # photon keep-vs-tau (charge-weighted)
            prs = np.array([cell_pr.get(k, np.nan) for k in ph_keys])
            qm = q[ph]
            nocell[lab] += qm[np.isnan(prs)].sum()
            prs0 = np.nan_to_num(prs, nan=-1.0)
            for ti, tau in enumerate(TAUS):
                keep_num[lab][ti] += qm[prs0 > tau].sum()
            # proximity recall profile (all MC-nu points, count-weighted)
            if d_nu is not None and len(nu_keys):
                prn = np.array([cell_pr.get(k, np.nan) for k in nu_keys])
                ok = ~np.isnan(prn)
                bi = np.digitize(d_nu[ok], DBINS) - 1
                kp = prn[ok] > args.prox_tau
                for b in range(len(DBINS) - 1):
                    m = bi == b
                    prox[lab]["tot"][b] += int(m.sum())
                    prox[lab]["kept"][b] += int((kp & m).sum())

    print("\n== photon charge keep fraction vs tau (dedup q_comb, "
          "charge-weighted) ==")
    print(f"{'tau':>6s} " + " ".join(f"{l:>12s}" for l in labs))
    for ti, tau in enumerate(TAUS):
        print(f"{tau:6.2f} " + " ".join(
            f"{keep_num[l][ti] / keep_den:12.3f}" for l in labs))
    print("no-cell charge frac: " + "  ".join(
        f"{l}={nocell[l] / keep_den:.3f}" for l in labs))

    print(f"\n== MC-nu point recall @ tau={args.prox_tau} vs distance to "
          "nearest data-cosmic point ==")
    print(f"{'dist[cm]':>9s} " + " ".join(f"{l:>12s}" for l in labs)
          + f" {'n_pts':>10s}")
    for b, dl in enumerate(DLABELS):
        vals = " ".join(
            f"{prox[l]['kept'][b] / max(prox[l]['tot'][b], 1):12.3f}"
            for l in labs)
        print(f"{dl:>9s} {vals} {int(prox[labs[0]]['tot'][b]):10d}")


if __name__ == "__main__":
    main()
