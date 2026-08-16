"""Deghost recall of MC-nu points vs distance to the nearest DATA-cosmic
point (overlay pilot, 174 CC1pi0 signal events) — discriminates whether the
overlay-domain recall loss is driven by the points' INTRINSIC (old-sim)
features (recall low even far from data activity) or by data-cosmic CONTEXT
leaking in through attention (recall degrades with proximity)."""
import glob
import re
import sys

import numpy as np
import h5py
from scipy.spatial import cKDTree

sys.path.insert(0, ".")
from lartpc.larformer_analysis.physics.pilot_matrix.pi0_photon_charge_flow \
    import _cell_keys  # noqa: E402

FLOW = "lartpc/larformer_reco/output/pilot_ntuples"
BINS = np.array([0.0, 2.0, 5.0, 10.0, 20.0, 50.0, 1e9])
LABELS = ["0-2", "2-5", "5-10", "10-20", "20-50", ">50"]
TAUS = {"old": 0.5, "new": 0.2}

sig = np.load(f"{FLOW}/photon_charge_flow/sig_event_indices.npy")
listpos = {int(e): i for i, e in enumerate(sig)}
msp = [l.strip() for l in open(
    "lartpc/larformer_reco/inputlists/merged_sp_mcc9_bnbnu_satfix_pilot10k.txt")
    if l.strip()]
sidx = {c: {int(re.search(r"sliceid_event0*(\d+)", p).group(1)): p
            for p in glob.glob(f"{FLOW}/sliceids_{c}/**/sliceid_event*.h5",
                               recursive=True)}
        for c in ("old", "new")}

res = {c: dict(kept=np.zeros(len(BINS) - 1), tot=np.zeros(len(BINS) - 1))
       for c in ("old", "new")}
for ev in sig:
    with h5py.File(msp[int(ev)], "r") as f:
        td = f["entry_0/triplet_data"]
        tpos = np.ascontiguousarray(td["pos"][()], np.float32)
        origin = td["origin"][()].astype(np.int64)
        hasm = td["hasmatch"][()].astype(np.int64)
    nu_real = (origin == 1) & (hasm == 1)      # simulated nu deposits
    datacos = (origin != 1) & (hasm == 0)      # data cosmics + noise
    if not nu_real.any() or not datacos.any():
        continue
    d, _ = cKDTree(tpos[datacos]).query(tpos[nu_real], k=1)
    keys = _cell_keys(tpos[nu_real])
    for c in ("old", "new"):
        p = sidx[c].get(listpos[int(ev)])
        if p is None:
            continue
        with h5py.File(p, "r") as f:
            sc = f["full_slice/coord_cm"][()].astype(np.float32)
            pr = f["full_slice/deghost_p_real"][()].astype(np.float64)
        cell_pr = dict(zip(_cell_keys(sc), pr))
        prs = np.array([cell_pr.get(k, np.nan) for k in keys])
        ok = ~np.isnan(prs)
        bi = np.digitize(d[ok], BINS) - 1
        keep = prs[ok] > TAUS[c]
        for b in range(len(BINS) - 1):
            m = bi == b
            res[c]["tot"][b] += int(m.sum())
            res[c]["kept"][b] += int((keep & m).sum())

print("MC-nu point deghost recall vs distance to nearest DATA-cosmic point")
print(f"{'dist [cm]':>10s} {'OLD LoRA@0.5':>13s} {'NEW ft@0.2':>11s} "
      f"{'n_pts':>10s}")
for b, lab in enumerate(LABELS):
    ro = res["old"]["kept"][b] / max(res["old"]["tot"][b], 1)
    rn = res["new"]["kept"][b] / max(res["new"]["tot"][b], 1)
    print(f"{lab:>10s} {ro:13.3f} {rn:11.3f} {int(res['old']['tot'][b]):10d}")
