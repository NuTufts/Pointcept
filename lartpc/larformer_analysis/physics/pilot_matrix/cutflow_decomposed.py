"""CC1pi0 cutflow with the photon and muon steps DECOMPOSED into
segmenter-found / correctly-classified / reco-attached sub-steps:

  gamma-deghost : BOTH true photons have >--gslice-min of dedup charge
                SURVIVING the deghoster (slice_id != -2 in the
                --sliceids-dir sidecars; step skipped if not given)
  gamma-slice : BOTH true photons have >--gslice-min (default 0.2, to
                match the instance bar) of dedup charge in the predicted
                nu slice (slicer delivery success)
  gamma-found : BOTH true photons have SOME kp2 instance holding >=20% of
                the photon's dedup charge (segmenter clustering success)
  gamma-ID    : both photons' best instance is gamma-classed (>=20% + cls)
  gamma-cut   : >=2 confident attached reco photons (the original cut,
                from nu_reco: kind shower, cls gamma, E>20, att_confident)
  mu-found / mu-ID / mu-cut : same for the highest-KE primary true muon
                (mu-cut = primary track, cls mu, KE>143.425)

Instance-charge matching uses the 0.25 cm cell-inheritance join (kp2
instances index the dedup-cell representatives). True photon tids come from
the flow-dir photon_records; the muon tid from mc_particle_tree (primary =
origin==1 & process_code==0, self-parenting; energy_mev = KE).

    PYTHONPATH=./ python3 cutflow_decomposed.py --label CELL \
        --kp2-list L.txt --nu-reco-dir D --flow-dir F --msp-list M.txt
"""
import argparse
import glob
import os
import re

import numpy as np
import h5py

from lartpc.larformer_reco.trajfit.calo import dedup_charge
from lartpc.larformer_analysis.physics.pilot_matrix.pi0_photon_charge_flow \
    import _cell_keys

FV_LO = np.array([20.0, -96.5, 10.0])
FV_HI = np.array([236.35, 96.5, 986.8])
M_MU, M_CPI = 105.6584, 139.5704
MU_KE_MIN, CPI_KE_MIN = 143.425, 25.0
GAMMA_MIN, MGG_MAX = 20.0, 400.0
FRAC_MIN = 0.20


def in_fv(v):
    return bool(np.all(v >= FV_LO) and np.all(v <= FV_HI))


def instance_fracs(kp_path, tpos, q, tid_mask_by_t):
    """Per true particle t: (best instance charge frac, best gamma-classed
    frac, best mu-classed frac, frac of charge in the nu slice)."""
    with h5py.File(kp_path, "r") as f:
        sc = np.ascontiguousarray(f["slice/coord_cm"][()], np.float32)
        insts = []
        if "particle" in f:
            for k in f["particle"]:
                g = f[f"particle/{k}"]
                insts.append((int(g.attrs["cls"]),
                              g["point_idx"][()].astype(np.int64)))
    skeys = _cell_keys(sc)
    slice_cells = set(skeys)
    out = {}
    for t, m in tid_mask_by_t.items():
        qt = float(q[m].sum())
        if qt <= 0:
            out[t] = (0.0, 0.0, 0.0, 0.0)
            continue
        pkeys = _cell_keys(tpos[m])
        qs = q[m]
        cell_q = {}
        for k, qv in zip(pkeys, qs):
            cell_q[k] = cell_q.get(k, 0.0) + float(qv)
        q_inslice = sum(qv for k, qv in cell_q.items()
                        if k in slice_cells)
        best_any = best_g = best_mu = 0.0
        for cls, pidx in insts:
            cells = set(skeys[j] for j in pidx if j < len(skeys))
            qin = sum(cell_q.get(k, 0.0) for k in cells)
            fr = qin / qt
            best_any = max(best_any, fr)
            if cls == 1:
                best_g = max(best_g, fr)
            if cls == 2:
                best_mu = max(best_mu, fr)
        out[t] = (best_any, best_g, best_mu, q_inslice / qt)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--label", required=True)
    ap.add_argument("--kp2-list", required=True)
    ap.add_argument("--nu-reco-dir", required=True)
    ap.add_argument("--flow-dir", required=True)
    ap.add_argument("--msp-list", required=True)
    ap.add_argument("--gslice-min", type=float, default=0.2)
    ap.add_argument("--sliceids-dir", default=None,
                    help="--slice-ids-only sidecar dir (files indexed by "
                         "rank of event in the signal list); enables the "
                         "gamma-deghost step")
    args = ap.parse_args()

    kp2 = [l.strip() for l in open(args.kp2_list) if l.strip()]
    kp2_by_ev = {}
    for gi, p in enumerate(kp2):
        m = re.search(r"keypoint2_event0*(\d+)_", os.path.basename(p))
        if m:
            kp2_by_ev[int(m.group(1))] = (gi, p)
    msp = [l.strip() for l in open(args.msp_list) if l.strip()]
    rec = np.load(os.path.join(args.flow_dir, "photon_records.npz"))
    sig = [int(x) for x in
           np.load(os.path.join(args.flow_dir, "sig_event_indices.npy"))]

    ev_data = {}
    for sp in glob.glob(os.path.join(args.nu_reco_dir, "nu_reco_shard*.h5")):
        with h5py.File(sp, "r") as f:
            for k in f:
                if k.startswith("event_"):
                    g = f[k]
                    ev_data[int(k.split("_")[1])] = dict(
                        vtx=g["vertices_cm"][()], kind=g["part_kind"][()],
                        cls=g["part_pred_class"][()], E=g["part_energy"][()],
                        conf=g["part_att_confident"][()],
                        start=g["part_start_cm"][()],
                        pvtx=g["part_vtx"][()], depth=g["vtx_depth"][()])

    steps = ["reco", "vtx", "gdeghost", "gslice", "gfound", "gID", "gcut",
             "mufound", "muID", "mucut", "cpi", "mgg"]
    c = dict.fromkeys(steps, 0)
    sid_by_rank = {}
    if args.sliceids_dir:
        for p in glob.glob(os.path.join(args.sliceids_dir,
                                        "sliceid_event*.h5")):
            m = re.search(r"sliceid_event0*(\d+)", os.path.basename(p))
            if m:
                sid_by_rank[int(m.group(1))] = p
    rank_of_ev = {ev: i for i, ev in enumerate(sig)}
    for ev in sig:
        hit = kp2_by_ev.get(ev)
        if hit is None:
            continue
        gi, kp_path = hit
        d = ev_data.get(gi)
        if d is None or len(d["vtx"]) == 0:
            continue
        c["reco"] += 1
        v0 = np.asarray(d["vtx"][0], np.float64)
        if not in_fv(v0):
            continue
        c["vtx"] += 1
        # ---- truth tids + charges ----------------------------------
        with h5py.File(msp[ev], "r") as f:
            td = f["entry_0/triplet_data"]
            tpos = np.ascontiguousarray(td["pos"][()], np.float32)
            trackid = td["trackid"][()].astype(np.int64)
            _, q = dedup_charge(td["pixval"][()], td["tick"][()],
                                td["uwire"][()], td["vwire"][()],
                                td["ywire"][()])
            t = f["entry_0/mc_particle_tree"]
            pid = t["pid"][()].astype(np.int64)
            org = t["origin"][()].astype(np.int64)
            proc = t["process_code"][()].astype(np.int64)
            E_t = t["energy_mev"][()].astype(np.float64)
            ttid = t["trackid"][()].astype(np.int64)
        ph_tids = [int(tt) for e2, tt in zip(rec["event"], rec["tid"])
                   if int(e2) == ev]
        prim = (org == 1) & (proc == 0)
        mus = prim & (np.abs(pid) == 13) & (E_t > MU_KE_MIN)
        mu_tid = (int(ttid[mus][np.argmax(E_t[mus])]) if mus.any() else None)
        masks = {tt: trackid == tt for tt in ph_tids}
        if mu_tid is not None:
            masks[mu_tid] = trackid == mu_tid
        fr = instance_fracs(kp_path, tpos, q, masks)
        # ---- gamma sub-steps ----------------------------------------
        if args.sliceids_dir:
            sp = sid_by_rank.get(rank_of_ev[ev])
            assert sp is not None, f"no sliceid sidecar for ev {ev}"
            with h5py.File(sp, "r") as f:
                fsc = f["full_slice/coord_cm"][()].astype(np.float32)
                fsid = f["full_slice/slice_id"][()].astype(np.int64)
            cell_sid = dict(zip(_cell_keys(np.ascontiguousarray(fsc)),
                                fsid))
            ok_deghost = True
            for tt in ph_tids:
                m = masks[tt]
                qt = float(q[m].sum())
                if qt <= 0:
                    ok_deghost = False
                    break
                qsurv = sum(
                    float(qv) for k, qv in
                    zip(_cell_keys(np.ascontiguousarray(tpos[m])), q[m])
                    if cell_sid.get(k, -2) != -2)
                if qsurv / qt <= args.gslice_min:
                    ok_deghost = False
                    break
            if not ok_deghost:
                continue
        c["gdeghost"] += 1
        if not all(fr[tt][3] > args.gslice_min for tt in ph_tids):
            continue
        c["gslice"] += 1
        if not all(fr[tt][0] >= FRAC_MIN for tt in ph_tids):
            continue
        c["gfound"] += 1
        if not all(fr[tt][1] >= FRAC_MIN for tt in ph_tids):
            continue
        c["gID"] += 1
        kind, cls, E, conf = d["kind"], d["cls"], d["E"], d["conf"]
        primk = np.array([(int(pv) >= 0 and d["depth"][int(pv)] == 0)
                          for pv in d["pvtx"]])
        is_g = (kind == 1) & (cls == 1) & (E > GAMMA_MIN) & (conf != 0)
        if is_g.sum() < 2:
            continue
        c["gcut"] += 1
        # ---- muon sub-steps -----------------------------------------
        if mu_tid is None or fr[mu_tid][0] < FRAC_MIN:
            continue
        c["mufound"] += 1
        if fr[mu_tid][2] < FRAC_MIN:
            continue
        c["muID"] += 1
        if not np.any((kind == 0) & (cls == 2) & primk
                      & (E - M_MU > MU_KE_MIN)):
            continue
        c["mucut"] += 1
        # ---- remaining cuts -----------------------------------------
        if np.any((kind == 0) & (cls == 3) & primk
                  & (E - M_CPI > CPI_KE_MIN)):
            continue
        c["cpi"] += 1
        gidx = np.nonzero(is_g)[0]
        o = gidx[np.argsort(E[gidx])[::-1][:2]]
        d1 = np.asarray(d["start"][o[0]], np.float64) - v0
        d2 = np.asarray(d["start"][o[1]], np.float64) - v0
        n1, n2 = np.linalg.norm(d1), np.linalg.norm(d2)
        if n1 < 1e-3 or n2 < 1e-3:
            continue
        mgg = np.sqrt(max(2.0 * E[o[0]] * E[o[1]]
                          * (1.0 - float(d1 @ d2) / (n1 * n2)), 0.0))
        if mgg < MGG_MAX:
            c["mgg"] += 1

    n_sig = len(sig)
    print(f"{args.label}: sig={n_sig}  " + "  ".join(
        f"{s}={c[s]}" for s in steps)
        + f"   eff={c['mgg'] / max(n_sig, 1):.3f}")


if __name__ == "__main__":
    main()
