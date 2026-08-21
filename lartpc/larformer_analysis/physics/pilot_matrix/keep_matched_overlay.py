"""Matched-operating-point A/B verdict: per model, find tau* giving a target
ghost acceptance ON THE IN-DOMAIN (corsika CC1pi0) sample, then read the
OVERLAY photon charge keep at tau* — the calibration-fair domain-robustness
comparison."""
import glob
import re
import sys

import numpy as np
import h5py

sys.path.insert(0, ".")
from lartpc.larformer_analysis.physics.pilot_matrix.pi0_photon_charge_flow \
    import _cell_keys  # noqa: E402
from lartpc.larformer_reco.trajfit.calo import dedup_charge  # noqa: E402

PRE = "lartpc/larformer_reco/output/pilot_ntuples"
VCC = f"{PRE}/val_cc1pi0"
OVL = f"{PRE}/photon_charge_flow"
MODELS = {"v7-lora": ("preal_v7-lora", f"{PRE}/sliceids_old"),
          "v7-ftdec": ("preal_v7-ftdec", f"{PRE}/sliceids_new"),
          "v7-cropdec": ("preal_v7-cropdec", f"{PRE}/preal_v7cropdec"),
          "p5b3-dec": ("preal_p5b3-dec", f"{PRE}/preal_p5b3dec"),
          "p5b3-lora": ("preal_p5b3-lora", f"{PRE}/preal_p5b3lora"),
          "v6-lantern": ("preal_v6-lantern",
                         f"{PRE}/pilot174_lmscored/preal_v6-lantern")}
TAUS = np.linspace(0.02, 0.9, 89)
TARGETS = (0.10, 0.15, 0.20)


def pidx(d):
    return {int(re.search(r"sliceid_event0*(\d+)", p).group(1)): p
            for p in glob.glob(f"{d}/**/sliceid_event*.h5", recursive=True)}


def photon_and_ghost(files_list, flow_dir, preal_dir, want_ghost):
    rec = np.load(f"{flow_dir}/photon_records.npz")
    sig = np.load(f"{flow_dir}/sig_event_indices.npy")
    listpos = {int(e): i for i, e in enumerate(sig)}
    files = [l.strip() for l in open(files_list) if l.strip()]
    idx = pidx(preal_dir)
    keep = np.zeros(len(TAUS)); den = 0.0
    gnum = np.zeros(len(TAUS)); gden = 0.0
    for ev in sig:
        with h5py.File(files[int(ev)], "r") as f:
            td = f["entry_0/triplet_data"]
            tpos = np.ascontiguousarray(td["pos"][()], np.float32)
            trackid = td["trackid"][()].astype(np.int64)
            hasm = td["hasmatch"][()].astype(np.int64)
            _, q = dedup_charge(td["pixval"][()], td["tick"][()],
                                td["uwire"][()], td["vwire"][()],
                                td["ywire"][()])
        path = idx.get(listpos[int(ev)])
        if path is None:
            continue
        tids = [int(t) for e2, t in zip(rec["event"], rec["tid"])
                if int(e2) == int(ev)]
        ph = np.isin(trackid, tids)
        with h5py.File(path, "r") as f:
            sc = f["full_slice/coord_cm"][()].astype(np.float32)
            pr = f["full_slice/deghost_p_real"][()].astype(np.float64)
        cell = dict(zip(_cell_keys(sc), pr))
        prs = np.nan_to_num(np.array([cell.get(k, np.nan)
                                      for k in _cell_keys(tpos[ph])]),
                            nan=-1.0)
        qm = q[ph]; den += qm.sum()
        for ti, tau in enumerate(TAUS):
            keep[ti] += qm[prs > tau].sum()
        if want_ghost:
            gh = hasm == 0
            prg = np.nan_to_num(np.array([cell.get(k, np.nan)
                                          for k in _cell_keys(tpos[gh])]),
                                nan=-1.0)
            gden += int(gh.sum())
            for ti, tau in enumerate(TAUS):
                gnum[ti] += int((prg > tau).sum())
    return keep / max(den, 1), (gnum / max(gden, 1) if want_ghost else None)


rows = {}
for lab, (vd, od) in MODELS.items():
    k_in, g_in = photon_and_ghost(f"{VCC}/val_cc1pi0_files.txt", VCC,
                                  f"{VCC}/{vd}", True)
    k_ov, _ = photon_and_ghost(
        "lartpc/larformer_reco/inputlists/merged_sp_mcc9_bnbnu_satfix_pilot10k.txt",
        OVL, od, False)
    rows[lab] = (k_in, g_in, k_ov)
    print(f"{lab}: curves done")

print("\n== matched operating points: tau*(in-domain ghost acc) -> keeps ==")
print(f"{'model':>11s} {'ghostacc':>8s} {'tau*':>6s} {'keep_in':>8s} "
      f"{'keep_OVERLAY':>12s} {'transfer':>9s}")
for ga in TARGETS:
    for lab in MODELS:
        k_in, g_in, k_ov = rows[lab]
        o = np.argsort(g_in)
        tau_s = float(np.interp(ga, g_in[o], TAUS[o]))
        ki = float(np.interp(tau_s, TAUS, k_in))
        ko = float(np.interp(tau_s, TAUS, k_ov))
        print(f"{lab:>11s} {ga:8.2f} {tau_s:6.3f} {ki:8.3f} {ko:12.3f} "
              f"{ko / max(ki, 1e-9):9.3f}")
    print()
