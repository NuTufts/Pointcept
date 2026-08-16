"""Light CC1pi0 reco cutflow computed DIRECTLY from nu_reco shards (no
ntuple/truth-sidecar needed) — replicates the sbnd_cc1pi0 reco selection's
LArFormer-side cuts so val (flash-less) and overlay cells share one metric:

  vtx: primary interaction vertex (vertices_cm[0], nu stream by list
       construction) in the tight FV
  2g : >=2 showers with class gamma, E>20 MeV, att_confident
  mu : >=1 PRIMARY track (vtx_depth of its attach vertex == 0, the ntuple's
       IsSecondary definition) class mu, KE = E - m_mu > 143.425
  cpi: 0 primary charged pions KE = E - m_pi > 25
  mgg: two most-energetic confident photons, dirs = start - vtx,
       m_gg < 400 MeV

Signal events are given by --signal-list: either 'all' (every kp2-list entry
is signal, e.g. the val CC1pi0 twin) or an npy of PILOT event indices whose
kp2 filenames (keypoint2_event{i:05d}_0.h5) are located in the kp2 list.

    PYTHONPATH=./ python3 light_cc1pi0_cutflow.py --label X \
        --kp2-list L.txt --nu-reco-dir D [--signal-list sig.npy | all]
"""
import argparse
import glob
import os
import re

import numpy as np
import h5py

FV_LO = np.array([20.0, -96.5, 10.0])
FV_HI = np.array([236.35, 96.5, 986.8])
M_MU, M_CPI = 105.6584, 139.5704
MU_KE_MIN, CPI_KE_MIN = 143.425, 25.0
GAMMA_MIN, MGG_MAX = 20.0, 400.0


def in_fv(v):
    return bool(np.all(v >= FV_LO) and np.all(v <= FV_HI))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--label", required=True)
    ap.add_argument("--kp2-list", required=True)
    ap.add_argument("--nu-reco-dir", required=True)
    ap.add_argument("--signal-list", required=True,
                    help="'all' or .npy of pilot event indices")
    args = ap.parse_args()

    kp2 = [l.strip() for l in open(args.kp2_list) if l.strip()]
    if args.signal_list == "all":
        gidxs = list(range(len(kp2)))
    else:
        want = set(int(x) for x in np.load(args.signal_list))
        gidxs = []
        for i, p in enumerate(kp2):
            m = re.search(r"keypoint2_event0*(\d+)_", os.path.basename(p))
            if m and int(m.group(1)) in want:
                gidxs.append(i)
    n_sig = (len(gidxs) if args.signal_list == "all"
             else len(want))

    # load all shard event groups into an index
    ev_data = {}
    for sp in glob.glob(os.path.join(args.nu_reco_dir, "nu_reco_shard*.h5")):
        with h5py.File(sp, "r") as f:
            for k in f:
                if not k.startswith("event_"):
                    continue
                g = f[k]
                ev_data[int(k.split("_")[1])] = dict(
                    vtx=g["vertices_cm"][()],
                    kind=g["part_kind"][()], cls=g["part_pred_class"][()],
                    E=g["part_energy"][()],
                    conf=g["part_att_confident"][()],
                    start=g["part_start_cm"][()], pvtx=g["part_vtx"][()],
                    depth=g["vtx_depth"][()])

    c = dict(reco=0, vtx=0, g2=0, mu=0, cpi=0, mgg=0)
    for gi in gidxs:
        d = ev_data.get(gi)
        if d is None or len(d["vtx"]) == 0:
            continue
        c["reco"] += 1
        v0 = np.asarray(d["vtx"][0], np.float64)
        if not in_fv(v0):
            continue
        c["vtx"] += 1
        kind, cls, E, conf = d["kind"], d["cls"], d["E"], d["conf"]
        prim = np.array([(int(pv) >= 0 and d["depth"][int(pv)] == 0)
                         for pv in d["pvtx"]])
        is_g = (kind == 1) & (cls == 1) & (E > GAMMA_MIN) & (conf != 0)
        if is_g.sum() < 2:
            continue
        c["g2"] += 1
        if not np.any((kind == 0) & (cls == 2) & prim
                      & (E - M_MU > MU_KE_MIN)):
            continue
        c["mu"] += 1
        if np.any((kind == 0) & (cls == 3) & prim
                  & (E - M_CPI > CPI_KE_MIN)):
            continue
        c["cpi"] += 1
        gi_idx = np.nonzero(is_g)[0]
        o = gi_idx[np.argsort(E[gi_idx])[::-1][:2]]
        d1 = np.asarray(d["start"][o[0]], np.float64) - v0
        d2 = np.asarray(d["start"][o[1]], np.float64) - v0
        n1, n2 = np.linalg.norm(d1), np.linalg.norm(d2)
        if n1 < 1e-3 or n2 < 1e-3:
            continue
        mgg = np.sqrt(max(2.0 * E[o[0]] * E[o[1]]
                          * (1.0 - float(d1 @ d2) / (n1 * n2)), 0.0))
        if mgg < MGG_MAX:
            c["mgg"] += 1

    print(f"{args.label}: sig={n_sig}  reco'd={c['reco']}  vtxFV={c['vtx']}  "
          f">=2g={c['g2']}  +mu={c['mu']}  +0cpi={c['cpi']}  +mgg={c['mgg']}"
          f"   eff(pre-flash)={c['mgg'] / max(n_sig, 1):.3f}")


if __name__ == "__main__":
    main()
