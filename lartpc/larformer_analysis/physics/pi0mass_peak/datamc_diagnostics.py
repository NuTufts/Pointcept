"""Diagnostic data/MC/EXT overlays for the two-photon selection: where is the
EXT (cosmic) contribution coming from?

For every event passing the CURRENT working point — FV vertex, >=2 recal'd
photons > 20 MeV, per-stream flash-chi2 cut (reco-CC < --chi2-cut, reco-NC <
--chi2-cut-nc) — plots MC overlay (POT-weighted, from the table w) + EXT
(spill-scaled) stacked, with beam data overlaid as points, for:

  reco vertex x / y / z, vertex dwall, vtxScore, vtxFracHitsOnCosmic;
  leading / sub-leading photon KE (recalibrated), E1+E2;
  leading / sub-leading photon cos(theta_beam) and cos(theta_Y);
  leading / sub-leading photon conversion distance (showerDistToVtx);
  pair opening cos(theta_12) and m_gg (vertex->start, recal);
  in-time flash total PE (cascade flash/observed_pe via the cached RSE map);
  n non-shower primaries (tracks, trackIsSecondary==0), n reco photons;
  log10 flash chi2 (post-cut).

Shower-energy recalibration (--recal-gamma-a/-b) is applied exactly as in
pi0_mass_analysis.py (invert deployed calib, re-apply). flash_chi2 and w come
row-aligned from the pre-built --*-table npz (built with --cascade-dir).

    PYTHONPATH=./ python3 datamc_diagnostics.py \
        --mc-ntuple ... --mc-table ... --mc-cascade ... \
        --data-ntuple ... --data-table ... \
        --ext-ntuple ... --ext-table ... --ext-cascade ... \
        --ext-scale 0.5909 --plots plots_s1ep2p8_diag
"""
import argparse
import os
import sys

import numpy as np
import uproot
import awkward as ak

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

NTUP_G_A, NTUP_G_B = 0.020100999623537064, -15.489999771118164
TPC_LO = np.array([0.0, -116.5, 0.0])
TPC_HI = np.array([256.35, 116.5, 1036.8])

BR = ["run", "subrun", "event", "foundVertex", "primaryVtxStream",
      "vtxIsFiducial", "vtxX", "vtxY", "vtxZ", "vtxScore",
      "vtxFracHitsOnCosmic",
      "showerLArFormerPID", "showerRecoE", "showerCosTheta",
      "showerCosThetaY", "showerDistToVtx",
      "showerStartPosX", "showerStartPosY", "showerStartPosZ",
      "trackLArFormerPID", "trackIsSecondary", "trackRecoE",
      "showerCosmicScore"]


def load(ntuple, table, ga, gb, mu_ke, chi2_cc, chi2_nc,
         shower_bdt_min=None):
    t = uproot.open(ntuple)["EventTree"]
    a = t.arrays([b for b in BR if b in set(t.keys())])
    tab = np.load(table)
    n = len(a["run"])
    assert len(tab["w"]) == n, "table/ntuple row mismatch"
    w_all = np.asarray(tab["w"], np.float64)
    fchi2 = np.asarray(tab["flash_chi2"], np.float64)

    E = a["showerRecoE"]
    spid = a["showerLArFormerPID"]
    E = ak.where(spid == 22, (E - NTUP_G_B) / NTUP_G_A * ga + gb, E)

    vtx_ok = ((np.asarray(a["foundVertex"]) == 1)
              & (np.asarray(a["primaryVtxStream"]) == 0)
              & (np.asarray(a["vtxIsFiducial"]) == 1))
    is_g = (spid == 22) & (E > 20.0)
    if shower_bdt_min is not None:
        # per-shower cosmic-BDT gate re-selects the photon-candidate set
        # (branch absent on pre-score ntuples -> caller must not pass it)
        is_g = is_g & (a["showerCosmicScore"] >= shower_bdt_min)
    n_g = ak.to_numpy(ak.sum(is_g, axis=1))
    is_mu = ((a["trackLArFormerPID"] == 13) & (a["trackIsSecondary"] == 0)
             & (a["trackRecoE"] > mu_ke))
    reco_cc = ak.to_numpy(ak.any(is_mu, axis=1))
    chi_ok = np.where(reco_cc, fchi2 < chi2_cc, fchi2 < chi2_nc)
    sel = vtx_ok & (n_g >= 2) & np.isfinite(fchi2) & chi_ok

    out = {k: [] for k in
           ("vtxX", "vtxY", "vtxZ", "dwall", "vtxScore", "cosmicFrac",
            "E1", "E2", "Esum", "cosZ1", "cosZ2", "cosY1", "cosY2",
            "dist1", "dist2", "cos12", "mgg", "nPrimTrk", "nPhot",
            "logchi2", "w", "run", "subrun", "event", "recoCC", "row")}
    idx = np.nonzero(sel)[0]
    for i in idx:
        Ei = np.asarray(E[i])
        gi = np.nonzero(np.asarray(is_g[i]))[0]
        gi = gi[np.argsort(Ei[gi])[::-1][:2]]
        j1, j2 = int(gi[0]), int(gi[1])
        v = np.array([a["vtxX"][i], a["vtxY"][i], a["vtxZ"][i]], float)
        ds = []
        for j in (j1, j2):
            st = np.array([a["showerStartPosX"][i][j],
                           a["showerStartPosY"][i][j],
                           a["showerStartPosZ"][i][j]], float)
            d = st - v
            ds.append(d / max(np.linalg.norm(d), 1e-9))
        c12 = float(np.dot(ds[0], ds[1]))
        out["vtxX"].append(v[0]); out["vtxY"].append(v[1])
        out["vtxZ"].append(v[2])
        out["dwall"].append(float(min((v - TPC_LO).min(),
                                      (TPC_HI - v).min())))
        out["vtxScore"].append(float(a["vtxScore"][i]))
        out["cosmicFrac"].append(float(a["vtxFracHitsOnCosmic"][i]))
        out["E1"].append(float(Ei[j1])); out["E2"].append(float(Ei[j2]))
        out["Esum"].append(float(Ei[j1] + Ei[j2]))
        out["cosZ1"].append(float(a["showerCosTheta"][i][j1]))
        out["cosZ2"].append(float(a["showerCosTheta"][i][j2]))
        out["cosY1"].append(float(a["showerCosThetaY"][i][j1]))
        out["cosY2"].append(float(a["showerCosThetaY"][i][j2]))
        out["dist1"].append(float(a["showerDistToVtx"][i][j1]))
        out["dist2"].append(float(a["showerDistToVtx"][i][j2]))
        out["cos12"].append(c12)
        out["mgg"].append(float(np.sqrt(max(
            2 * Ei[j1] * Ei[j2] * (1 - c12), 0.0))))
        out["nPrimTrk"].append(int(np.sum(
            np.asarray(a["trackIsSecondary"][i]) == 0)))
        out["nPhot"].append(int(n_g[i]))
        out["logchi2"].append(float(np.log10(max(fchi2[i], 1e-3))))
        out["w"].append(float(w_all[i]))
        out["run"].append(int(a["run"][i]))
        out["subrun"].append(int(a["subrun"][i]))
        out["event"].append(int(a["event"][i]))
        out["recoCC"].append(bool(reco_cc[i]))
        out["row"].append(int(i))
    return {k: np.asarray(v) for k, v in out.items()}


def add_flash_pe(d, cascade_dir, cache):
    """In-time flash total PE for selected events via the cached RSE map."""
    from flash_correction import rse_map
    import h5py
    m = rse_map(cascade_dir, cache)
    pe = np.full(len(d["run"]), np.nan)
    for k in range(len(pe)):
        p = m.get((int(d["run"][k]), int(d["subrun"][k]),
                   int(d["event"][k])))
        if not p:
            continue
        try:
            with h5py.File(p if isinstance(p, str) else p[0], "r") as f:
                if "flash" in f and "observed_pe" in f["flash"]:
                    pe[k] = float(np.clip(
                        f["flash/observed_pe"][()], 0, None).sum())
        except Exception:
            pass
    d["flashPE"] = pe
    return d


SPECS = [
    ("vtxX", "reco vertex x [cm]", np.linspace(0, 256.35, 33)),
    ("vtxY", "reco vertex y [cm]", np.linspace(-116.5, 116.5, 33)),
    ("vtxZ", "reco vertex z [cm]", np.linspace(0, 1036.8, 40)),
    ("dwall", "vertex dwall [cm]", np.linspace(0, 120, 31)),
    ("vtxScore", "vertex score", np.linspace(0, 1, 31)),
    ("cosmicFrac", "vtx frac hits on cosmic", np.linspace(0, 1, 31)),
    ("E1", "leading photon KE [MeV]", np.linspace(0, 800, 41)),
    ("E2", "sub-leading photon KE [MeV]", np.linspace(0, 400, 41)),
    ("Esum", "E1 + E2 [MeV]", np.linspace(0, 1200, 41)),
    ("cosZ1", "leading photon cos(theta_beam)", np.linspace(-1, 1, 33)),
    ("cosZ2", "sub-leading photon cos(theta_beam)", np.linspace(-1, 1, 33)),
    ("cosY1", "leading photon cos(theta_Y)", np.linspace(-1, 1, 33)),
    ("cosY2", "sub-leading photon cos(theta_Y)", np.linspace(-1, 1, 33)),
    ("dist1", "leading photon conv. dist [cm]", np.linspace(0, 120, 41)),
    ("dist2", "sub-leading photon conv. dist [cm]", np.linspace(0, 120, 41)),
    ("cos12", "pair opening cos(theta_12)", np.linspace(-1, 1, 33)),
    ("mgg", "m_gg [MeV]", np.linspace(0, 500, 41)),
    ("flashPE", "in-time flash total PE", np.linspace(0, 12000, 41)),
    ("nPrimTrk", "n non-shower primaries", np.arange(-0.5, 10.5, 1)),
    ("nPhot", "n reco photons (>20 MeV)", np.arange(1.5, 10.5, 1)),
    ("logchi2", "log10 flash chi2 (post-cut)", np.linspace(-1, 4.2, 41)),
]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    for s in ("mc", "data", "ext"):
        ap.add_argument(f"--{s}-ntuple", required=True)
        ap.add_argument(f"--{s}-table", required=True)
    ap.add_argument("--mc-cascade"); ap.add_argument("--ext-cascade")
    ap.add_argument("--data-cascade")
    ap.add_argument("--ext-scale", type=float, required=True)
    ap.add_argument("--recal-gamma-a", type=float, default=0.01556)
    ap.add_argument("--recal-gamma-b", type=float, default=-11.47)
    ap.add_argument("--mu-ke-min", type=float, default=50.0)
    ap.add_argument("--chi2-cut", type=float, default=1e4)
    ap.add_argument("--chi2-cut-nc", type=float, default=1778.0)
    ap.add_argument("--plots", required=True)
    args = ap.parse_args()
    os.makedirs(args.plots, exist_ok=True)
    here = os.path.dirname(os.path.abspath(__file__))

    def cache_for(cdir):
        return os.path.join(here, "rse_" + os.path.basename(os.path.dirname(
            cdir.rstrip("/"))) + "_" + os.path.basename(
            cdir.rstrip("/")) + ".npz")

    smp = {}
    for s in ("mc", "data", "ext"):
        print(f">>> loading {s} ...", flush=True)
        smp[s] = load(getattr(args, f"{s}_ntuple"),
                      getattr(args, f"{s}_table"),
                      args.recal_gamma_a, args.recal_gamma_b,
                      args.mu_ke_min, args.chi2_cut, args.chi2_cut_nc)
        cdir = getattr(args, f"{s}_cascade")
        if cdir:
            smp[s] = add_flash_pe(smp[s], cdir, cache_for(cdir))
        else:
            smp[s]["flashPE"] = np.full(len(smp[s]["run"]), np.nan)
        wsum = (smp[s]["w"].sum() if s == "mc"
                else len(smp[s]["run"]) * (args.ext_scale if s == "ext"
                                           else 1.0))
        print(f"    {s}: {len(smp[s]['run'])} selected "
              f"(weighted {wsum:.1f})")
    smp["ext"]["w"] = np.full(len(smp["ext"]["run"]), args.ext_scale)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for var, lab, bins in SPECS:
        fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))
        for axi, (ax, mask_name) in enumerate(zip(
                axs, ("all", "reco-NC only"))):
            def sl(s):
                d = smp[s]
                m = np.ones(len(d["run"]), bool) if axi == 0 \
                    else ~d["recoCC"]
                x = d[var][m]
                w = d["w"][m]
                f = np.isfinite(x)
                return x[f], w[f]
            xm, wm = sl("mc")
            xe, we = sl("ext")
            xd, wd = sl("data")
            hm, _ = np.histogram(xm, bins, weights=wm)
            he, _ = np.histogram(xe, bins, weights=we)
            hd, _ = np.histogram(xd, bins)
            ctr = 0.5 * (bins[:-1] + bins[1:])
            ax.bar(ctr, hm, width=np.diff(bins), color="#7bafd4",
                   label="nu overlay MC")
            ax.bar(ctr, he, width=np.diff(bins), bottom=hm,
                   color="#d9d9d9", label="EXT cosmic")
            ax.errorbar(ctr, hd, yerr=np.sqrt(np.maximum(hd, 1)),
                        fmt="ko", ms=3, lw=1, label="beam data")
            ax.set(xlabel=lab, ylabel="events (4.4e19-scaled)",
                   title=mask_name)
            ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(os.path.join(args.plots, f"diag_{var}.png"), dpi=110)
        plt.close(fig)
    print(f">>> plots -> {args.plots}")


if __name__ == "__main__":
    main()
