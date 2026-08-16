"""Paired event-level bootstrap on the photon keep difference + photon-recall
vs ghost-acceptance curves (matched-operating-point comparison) for deghoster
p_real dumps on the val CC1pi0 twin sample."""
import glob
import re
import sys

import numpy as np
import h5py

sys.path.insert(0, ".")
from lartpc.larformer_analysis.physics.pilot_matrix.pi0_photon_charge_flow \
    import _cell_keys  # noqa: E402
from lartpc.larformer_reco.trajfit.calo import dedup_charge  # noqa: E402

VCC = "lartpc/larformer_reco/output/pilot_ntuples/val_cc1pi0"
MODELS = ["v7-lora", "v7-ftdec", "p5b3-dec", "p5b3-lora"]
TAUS = np.linspace(0.02, 0.9, 45)
TAU0 = 0.2

rec = np.load(f"{VCC}/photon_records.npz")
files = [l.strip() for l in open(f"{VCC}/val_cc1pi0_files.txt") if l.strip()]
idx = {m: {int(re.search(r"sliceid_event0*(\d+)", p).group(1)): p
           for p in glob.glob(f"{VCC}/preal_{m}/sliceid_event*.h5")}
       for m in MODELS}

nev = len(files)
ph_kept = {m: np.zeros(nev) for m in MODELS}   # photon charge kept @TAU0
ph_tot = np.zeros(nev)
rec_num = {m: np.zeros(len(TAUS)) for m in MODELS}   # photon recall curve
gho_num = {m: np.zeros(len(TAUS)) for m in MODELS}   # ghost acceptance curve
rec_den = 0.0
gho_den = 0.0
for ev in range(nev):
    with h5py.File(files[ev], "r") as f:
        td = f["entry_0/triplet_data"]
        tpos = np.ascontiguousarray(td["pos"][()], np.float32)
        trackid = td["trackid"][()].astype(np.int64)
        hasm = td["hasmatch"][()].astype(np.int64)
        _, q = dedup_charge(td["pixval"][()], td["tick"][()],
                            td["uwire"][()], td["vwire"][()], td["ywire"][()])
    tids = [int(t) for e2, t in zip(rec["event"], rec["tid"])
            if int(e2) == ev]
    ph = np.isin(trackid, tids)
    gh = hasm == 0
    keys_ph = _cell_keys(tpos[ph])
    keys_gh = _cell_keys(tpos[gh])
    ph_tot[ev] = q[ph].sum()
    rec_den += q[ph].sum()
    gho_den += int(gh.sum())
    for m in MODELS:
        with h5py.File(idx[m][ev], "r") as f:
            sc = f["full_slice/coord_cm"][()].astype(np.float32)
            pr = f["full_slice/deghost_p_real"][()].astype(np.float64)
        cell = dict(zip(_cell_keys(sc), pr))
        prs_ph = np.nan_to_num(np.array([cell.get(k, np.nan)
                                         for k in keys_ph]), nan=-1.0)
        prs_gh = np.nan_to_num(np.array([cell.get(k, np.nan)
                                         for k in keys_gh]), nan=-1.0)
        ph_kept[m][ev] = q[ph][prs_ph > TAU0].sum()
        for ti, tau in enumerate(TAUS):
            rec_num[m][ti] += q[ph][prs_ph > tau].sum()
            gho_num[m][ti] += int((prs_gh > tau).sum())

# ---- paired event-level bootstrap on keep@0.2 differences ------------------
rng = np.random.default_rng(7)
B = 5000
print(f"keep@{TAU0} point estimates: " + "  ".join(
    f"{m}={ph_kept[m].sum() / ph_tot.sum():.3f}" for m in MODELS))
for a, b in (("p5b3-lora", "v7-lora"), ("p5b3-dec", "v7-ftdec"),
             ("p5b3-lora", "p5b3-dec")):
    diffs = np.empty(B)
    for i in range(B):
        s = rng.integers(0, nev, nev)
        diffs[i] = (ph_kept[a][s].sum() - ph_kept[b][s].sum()) / ph_tot[s].sum()
    d0 = (ph_kept[a].sum() - ph_kept[b].sum()) / ph_tot.sum()
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    print(f"paired Δkeep({a} - {b}) = {d0:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]"
          f"  ({'SIGNIFICANT' if lo > 0 or hi < 0 else 'consistent with 0'})")

# ---- photon recall at MATCHED ghost acceptance ------------------------------
print("\nphoton recall at matched ghost acceptance (interpolated):")
print(f"{'ghost-acc':>10s} " + " ".join(f"{m:>10s}" for m in MODELS))
for ga in (0.05, 0.10, 0.15, 0.20):
    row = []
    for m in MODELS:
        g = gho_num[m] / gho_den
        r = rec_num[m] / rec_den
        o = np.argsort(g)
        row.append(float(np.interp(ga, g[o], r[o])))
    print(f"{ga:10.2f} " + " ".join(f"{v:10.3f}" for v in row))
print("\nghost acceptance @ tau=0.2: " + "  ".join(
    f"{m}={np.interp(TAU0, TAUS, gho_num[m] / gho_den):.3f}" for m in MODELS))
