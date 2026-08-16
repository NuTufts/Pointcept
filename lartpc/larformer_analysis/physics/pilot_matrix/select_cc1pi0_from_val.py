"""Isolate a CC-1pi0 signal sample from the LANTERN (recent-sim corsika) VAL
files — the in-domain twin of the 174-event overlay pilot, mirroring the
SBND-style truth definition of pi0mass_peak/sbnd_cc1pi0.py:

  - true nu vertex in the tight FV (>=20 cm X/Y walls, >=10 cm upstream,
    >=50 cm downstream);
  - >=1 PRIMARY muon (origin==1, process_code==0, |pid|==13) with
    KE > 143.425 MeV (energy_mev treated as KE, the repo convention);
  - exactly ONE pi0, identified as in the ntuple path via ORPHAN PHOTONS
    (Geant does not store the pi0): origin==1 photons whose parent_trackid
    is absent from the trackid table, grouped by parent; exactly one group
    of 2, and BOTH photons DETECTABLE: visible energy
    A_GAMMA * dedup-q_comb(triplet SPs with trackid==photon) > 20 MeV
    (the user's dedup-charge criterion — same physics as the ntuple's
    A_GAMMA*PixelSumQ, computed here directly from triplet_data);
  - 0 primary charged pions with KE > 25 MeV.

Outputs (--out-dir): val_cc1pi0_files.txt (selected merged-h5 paths),
photon_records.npz (event=index into that list, tid, evis, q_total) and
sig_event_indices.npy — the same artifact triplet the overlay pilot's
photon_charge_flow dir provides, so photon_keep_from_preal.py and the
proximity/keep machinery run on this sample unchanged.

    PYTHONPATH=./ python3 select_cc1pi0_from_val.py \
        --val-list .../h5list_mcall_lantern_val.txt \
        --out-dir .../pilot_ntuples/val_cc1pi0 --max-events 200
"""
import argparse
import os
import sys

import numpy as np
import h5py

sys.path.insert(0, ".")
from lartpc.larformer_reco.trajfit.calo import dedup_charge  # noqa: E402

A_GAMMA = 0.0253017
EVIS_MIN = 20.0
MU_KE_MIN = 143.425
CPI_KE_MIN = 25.0
FV_LO = np.array([20.0, -96.5, 10.0])
FV_HI = np.array([236.35, 96.5, 986.8])


def in_fv(v):
    return bool(np.all(v >= FV_LO) and np.all(v <= FV_HI))


def tree_stage(f):
    """Cheap tree-only cuts. Returns photon-pair trackids or None.

    Primary convention in these trees (verified on pi0filter val files):
    primaries have process_code==0 and SELF-PARENT (par == own tid); the pi0
    IS stored (pid=111, proc=0) and its decay photons carry par == pi0's
    trackid with proc==1; energy_mev is KINETIC energy. Older productions
    may omit the pi0 -> orphan-photon fallback (parent absent from table)."""
    t = f["entry_0/mc_particle_tree"]
    nu_v = t["nu_vertices"][()]
    if len(nu_v) == 0 or not in_fv(np.asarray(nu_v[0], np.float64)):
        return None
    pid = t["pid"][()].astype(np.int64)
    origin = t["origin"][()].astype(np.int64)
    proc = t["process_code"][()].astype(np.int64)
    par = t["parent_trackid"][()].astype(np.int64)
    tid = t["trackid"][()].astype(np.int64)
    E = t["energy_mev"][()].astype(np.float64)
    prim = (origin == 1) & (proc == 0)
    if not np.any(prim & (np.abs(pid) == 13) & (E > MU_KE_MIN)):
        return None
    if np.any(prim & (np.abs(pid) == 211) & (E > CPI_KE_MIN)):
        return None
    # --- exactly one PRIMARY pi0 with a 2-photon decay -----------------
    pi0 = prim & (pid == 111)
    if pi0.sum() == 1:
        d = (pid == 22) & (origin == 1) & (par == int(tid[pi0][0]))
        if d.sum() != 2:               # Dalitz / untracked -> not 2-gamma
            return None
        return tid[d].tolist()
    if pi0.sum() > 1:
        return None
    # --- fallback (older productions without the pi0 entry) ------------
    tset = set(tid.tolist())
    orphan_ph = ((pid == 22) & (origin == 1)
                 & np.asarray([int(m) not in tset for m in par], bool))
    if not orphan_ph.any():
        return None
    parents, counts = np.unique(par[orphan_ph], return_counts=True)
    pairs = parents[counts == 2]
    if len(parents) != 1 or len(pairs) != 1:
        return None
    return tid[orphan_ph & (par == pairs[0])].tolist()


def photon_evis(f, tids):
    td = f["entry_0/triplet_data"]
    trackid = td["trackid"][()].astype(np.int64)
    m_any = np.isin(trackid, np.asarray(tids, np.int64))
    if not m_any.any():
        return None
    _, q = dedup_charge(td["pixval"][()], td["tick"][()],
                        td["uwire"][()], td["vwire"][()], td["ywire"][()])
    out = []
    for t in tids:
        qt = float(q[trackid == int(t)].sum())
        out.append((int(t), A_GAMMA * qt, qt))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--val-list", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--max-events", type=int, default=200)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    files = [l.strip() for l in open(args.val_list) if l.strip()]
    sel_files, rec = [], dict(event=[], tid=[], evis=[], q_total=[])
    n_scan = n_tree = 0
    for p in files:
        if len(sel_files) >= args.max_events:
            break
        n_scan += 1
        try:
            with h5py.File(p, "r") as f:
                tids = tree_stage(f)
                if tids is None:
                    continue
                n_tree += 1
                ph = photon_evis(f, tids)
        except Exception as ex:
            print(f"  [skip] {os.path.basename(p)}: {ex}")
            continue
        if ph is None or len(ph) != 2 or any(e <= EVIS_MIN for _, e, _ in ph):
            continue
        idx = len(sel_files)
        sel_files.append(p)
        for t, e, qt in ph:
            rec["event"].append(idx)
            rec["tid"].append(t)
            rec["evis"].append(e)
            rec["q_total"].append(qt)
        if len(sel_files) % 25 == 0:
            print(f"  {len(sel_files)} selected after {n_scan} scanned "
                  f"({n_tree} passed tree cuts)")
    print(f">>> {len(sel_files)} CC-1pi0 signal events from {n_scan} scanned "
          f"({n_tree} passed tree-level cuts; detectable-pair pass rate "
          f"{len(sel_files) / max(n_tree, 1):.2f})")
    # LArFormerDataset SORTS its data list (get_data_list -> sorted); emit the
    # list sorted and remap record event indices so preal/event index == line.
    order = sorted(range(len(sel_files)), key=lambda i: sel_files[i])
    newpos = {old: new for new, old in enumerate(order)}
    sel_files = [sel_files[i] for i in order]
    rec["event"] = [newpos[e] for e in rec["event"]]
    with open(os.path.join(args.out_dir, "val_cc1pi0_files.txt"), "w") as fo:
        fo.write("\n".join(sel_files) + "\n")
    np.savez(os.path.join(args.out_dir, "photon_records.npz"),
             event=np.asarray(rec["event"]), tid=np.asarray(rec["tid"]),
             evis=np.asarray(rec["evis"]), q_total=np.asarray(rec["q_total"]))
    np.save(os.path.join(args.out_dir, "sig_event_indices.npy"),
            np.arange(len(sel_files)))
    ev = np.asarray(rec["evis"])
    print(f">>> photons: {len(ev)}  evis median {np.median(ev):.1f} MeV  "
          f"p90 {np.percentile(ev, 90):.1f}")


if __name__ == "__main__":
    main()
