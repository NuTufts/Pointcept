"""Truth characterization of the single-photon SIGNAL events (MC only):
stacked by signal category (in-FV 1g0X / in-FV 1g+X / entering no-mu /
entering CC-mu), POT-weighted,
with the subset that passes the full reco selection overlaid as a line
(same selection options as single_photon_selection.py).

Variables (per signal event; "the photon" = the one detectable photon):
  vtxX/Y/Z        true nu vertex
  gTrueE, gEvis   the photon's true energy and visible energy (A_GAMMA x Q)
  gCosBeam        cos(theta_beam) from the photon's true initial momentum
  gConvX/Y/Z      the photon's conversion point (trueSimPartEDep*)
  gConvDwall      distance of the conversion point to the TPC box wall
  ndTrueE, ndEvis  true / visible energy of every NON-detectable photon
                  (E_vis < 20 MeV) in the event -- one entry per photon
  nNonDetG        number of non-detectable photons per event
  keNonG          summed true KE of non-photon PRIMARY particles
                  (trueSimPartProcess==0; neutrons and nuclei excluded;
                  --ke-include-neutrons adds neutrons)
  nProtons, nChargedPions, nMuons  primary counts (Process==0, any KE;
                  --ke-min applies a KE threshold to all three)
  maxKE_p/mu/pi   highest true KE of any proton / muon / charged pion in the
                  event, PRIMARY OR SECONDARY (all trueSimPart); the *_dep
                  variants count only particles that deposit charge in the TPC
                  (PixelSumQ > 0). Vertical lines: MiniBooNE mineral-oil
                  Cherenkov thresholds (p 342, mu 39, pi 51 MeV KE) and the
                  100 MeV muon convention; fractions above are printed.

    python3 single_photon_truth.py --mc-ntuple <novtx MC> --novtx --mc-odd-only \
        --plots plots_v2_s1ep2p8_novtx/truth
"""
import argparse
import os
import sys

import numpy as np
import uproot
import awkward as ak

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import single_photon_selection as SPS   # noqa: E402
from single_photon_diagnostics import add_args   # noqa: E402

BR_T = ["trueVtxX", "trueVtxY", "trueVtxZ", "trueSimPartPx", "trueSimPartPy",
        "trueSimPartPz", "trueSimPartEDepX", "trueSimPartEDepY",
        "trueSimPartEDepZ"]
SIGCATS = [0, 1, 2, 3]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    add_args(ap)
    ap.add_argument("--ke-min", type=float, default=0.0,
                    help="KE threshold [MeV] for the primary p / pi+- / mu counts")
    ap.add_argument("--ke-include-neutrons", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.plots, exist_ok=True)

    a, pot = SPS.load(args.mc_ntuple, True, args.novtx)
    t = uproot.open(args.mc_ntuple)["EventTree"]
    x = t.arrays(BR_T)
    w0 = ak.to_numpy(a["xsecWeight"]).astype(np.float64)
    w = np.where(w0 > 0, w0, 0.0) * (args.pot / pot)
    if args.mc_odd_only:
        odd = ak.to_numpy(a["event"]) % 2 == 1
        w = np.where(odd, 2.0 * w, 0.0)
    T = SPS.truth_tables(a)
    R = SPS.reco_tables(a, args)
    R["novtx"] = args.novtx
    names, S, *_ = SPS.build_steps(R, args.shower_bdt_min, args.flashchi2_cut,
                                   args.bdt_after_multiplicity)
    cat = T["cat"]
    sig = T["sig_incl"]
    sel = S[-1] & sig                     # SIGNAL events passing the selection
    n = len(w)

    # ---- per-event truth variables --------------------------------------------
    pdg = a["trueSimPartPDG"]; apdg = np.abs(pdg)
    E = a["trueSimPartE"]; q = a["trueSimPartPixelSumQ"]
    evis = SPS.A_GAMMA * np.maximum(q, 0.0)
    proc = a["trueSimPartProcess"]
    ke = E - SPS._mass_of(pdg)
    vis_g = (apdg == 22) & (evis >= SPS.EVIS_MIN)
    nd_g = (apdg == 22) & (evis < SPS.EVIS_MIN)

    def first(arr, m):
        return np.asarray(ak.to_numpy(ak.fill_none(ak.firsts(arr[m]), np.nan)), np.float64)
    V = {}
    V["vtxX"], V["vtxY"], V["vtxZ"] = [ak.to_numpy(x[f"trueVtx{c}"]).astype(np.float64)
                                       for c in "XYZ"]
    V["gTrueE"], V["gEvis"] = T["g_trueE"], T["g_evis"]
    px, py, pz = [first(x[f"trueSimPartP{c}"], vis_g) for c in "xyz"]
    pn = np.sqrt(px ** 2 + py ** 2 + pz ** 2)
    V["gCosBeam"] = np.where(pn > 0, pz / np.maximum(pn, 1e-9), np.nan)
    V["gConvX"], V["gConvY"], V["gConvZ"] = [first(x[f"trueSimPartEDep{c}"], vis_g)
                                             for c in "XYZ"]
    conv = np.stack([V["gConvX"], V["gConvY"], V["gConvZ"]], 1)
    V["gConvDwall"] = np.minimum((conv - SPS.TPC_LO).min(1), (SPS.TPC_HI - conv).min(1))
    V["nNonDetG"] = ak.to_numpy(ak.sum(nd_g, axis=1)).astype(np.float64)
    prim = proc == 0
    keep = prim & (apdg != 22) & (apdg != 2112) & (apdg < 1000000000) & (apdg != 12) & (apdg != 14)
    if args.ke_include_neutrons:
        keep = keep | (prim & (apdg == 2112))
    V["keNonG"] = ak.to_numpy(ak.sum(ak.where(keep, ke, 0.0), axis=1)).astype(np.float64)
    dep = q > 0
    for nm, pid_ in (("p", 2212), ("mu", 13), ("pi", 211)):
        for suf, mm in (("", apdg == pid_), ("_dep", (apdg == pid_) & dep)):
            V[f"maxKE_{nm}{suf}"] = ak.to_numpy(ak.fill_none(
                ak.max(ak.where(mm, ke, -1.0), axis=1), -1.0)).astype(np.float64)
    thr = ke >= args.ke_min
    V["nProtons"] = ak.to_numpy(ak.sum(prim & (apdg == 2212) & thr, axis=1)).astype(np.float64)
    V["nChargedPions"] = ak.to_numpy(ak.sum(prim & (apdg == 211) & thr, axis=1)).astype(np.float64)
    V["nMuons"] = ak.to_numpy(ak.sum(prim & (apdg == 13) & thr, axis=1)).astype(np.float64)
    # per-photon (non-detectable) entries: expand event weights/cats
    ev_idx = ak.to_numpy(ak.flatten(ak.broadcast_arrays(np.arange(n), nd_g)[0][nd_g]))
    P = {"ndTrueE": ak.to_numpy(ak.flatten(E[nd_g])).astype(np.float64),
         "ndEvis": ak.to_numpy(ak.flatten(evis[nd_g])).astype(np.float64),
         "evt": ev_idx}

    # ---- plots ----------------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    SPECS = [
        ("vtxX", "true vertex x [cm]", np.linspace(-150, 400, 56)),
        ("vtxY", "true vertex y [cm]", np.linspace(-250, 250, 51)),
        ("vtxZ", "true vertex z [cm]", np.linspace(-200, 1250, 59)),
        ("gTrueE", "detectable photon true E [MeV]", np.linspace(0, 1000, 41)),
        ("gEvis", "detectable photon E_vis [MeV] (A_GAMMA x Q)", np.linspace(0, 1500, 51)),
        ("gCosBeam", r"detectable photon true $\cos\theta_{beam}$", np.linspace(-1, 1, 41)),
        ("gConvX", "photon conversion x [cm]", np.linspace(-20, 276, 38)),
        ("gConvY", "photon conversion y [cm]", np.linspace(-130, 130, 27)),
        ("gConvZ", "photon conversion z [cm]", np.linspace(-20, 1060, 37)),
        ("gConvDwall", "photon conversion point dwall [cm]", np.linspace(0, 130, 27)),
        ("nNonDetG", "n non-detectable photons (E_vis < 20 MeV)", np.arange(-0.5, 8.5, 1)),
        ("keNonG", "sum KE of non-photon primaries [MeV]", np.linspace(0, 1500, 51)),
        ("nProtons", f"n primary protons (KE >= {args.ke_min:.0f} MeV)", np.arange(-0.5, 8.5, 1)),
        ("nChargedPions", f"n primary charged pions (KE >= {args.ke_min:.0f} MeV)", np.arange(-0.5, 5.5, 1)),
        ("nMuons", f"n primary muons (KE >= {args.ke_min:.0f} MeV)", np.arange(-0.5, 3.5, 1)),
        ("maxKE_p", "max proton KE, any proton [MeV]", np.linspace(0, 1500, 51)),
        ("maxKE_p_dep", "max proton KE, protons depositing in TPC [MeV]", np.linspace(0, 1500, 51)),
        ("maxKE_mu", "max muon KE, any muon [MeV]", np.linspace(0, 600, 41)),
        ("maxKE_mu_dep", "max muon KE, muons depositing in TPC [MeV]", np.linspace(0, 600, 41)),
        ("maxKE_pi", "max charged-pion KE, any pion [MeV]", np.linspace(0, 600, 41)),
        ("maxKE_pi_dep", "max charged-pion KE, pions depositing in TPC [MeV]", np.linspace(0, 600, 41)),
    ]
    CHER = {"p": [(342.0, "MiniBooNE oil thr 342")],
            "mu": [(39.0, "MiniBooNE oil thr 39"), (100.0, "analysis mu thr 100")],
            "pi": [(51.0, "MiniBooNE oil thr 51")]}
    lines = [f"signal events (w): " + ", ".join(
        f"{SPS.CATS[c]} {w[cat == c].sum():.1f} (sel {w[(cat == c) & sel].sum():.1f})"
        for c in SIGCATS)]

    def draw(key, xlab, bins, vals, ww, cc, ss):
        lo, hi = bins[0], bins[-1] - 1e-6
        m = np.isfinite(vals)
        if key.startswith("maxKE_"):
            m = m & (vals >= 0)          # events WITH such a particle only
        fig, ax = plt.subplots(figsize=(7.2, 4.6))
        ax.hist([np.clip(vals[m & (cc == c)], lo, hi) for c in SIGCATS], bins=bins,
                weights=[ww[m & (cc == c)] for c in SIGCATS],
                stacked=True, color=[SPS.CAT_COLORS[c] for c in SIGCATS],
                label=[f"{SPS.CATS[c]} ({ww[m & (cc == c)].sum():.1f})" for c in SIGCATS])
        ax.hist(np.clip(vals[m & ss], lo, hi), bins=bins, weights=ww[m & ss],
                histtype="step", color="k", lw=1.8,
                label=f"signal passing full selection ({ww[m & ss].sum():.1f})")
        if key.startswith("maxKE_"):
            sp = key.split("_")[1]
            for thv, lab in CHER[sp]:
                ax.axvline(thv, color="k", ls="--", lw=1, label=lab)
                fr = " | ".join(
                    f"{SPS.CATS[c][:14]} {ww[m & (cc == c) & (vals >= thv)].sum() / max(ww[cc == c].sum(), 1e-9):.2f}"
                    for c in SIGCATS)
                lines.append(f"  {key:14s}: fraction of ALL events of the category with a "
                             f"{sp} above {thv:.0f} MeV: {fr} | selected "
                             f"{ww[m & ss & (vals >= thv)].sum() / max(ww[ss].sum(), 1e-9):.2f}")
            has = " | ".join(f"{SPS.CATS[c][:14]} {ww[m & (cc == c)].sum() / max(ww[cc == c].sum(), 1e-9):.2f}"
                             for c in SIGCATS)
            lines.append(f"  {key:14s}: fraction of events with any such particle: {has}")
        ax.set(xlabel=xlab, ylabel=f"events / {args.pot:.1e} POT",
               title=f"true signal: {xlab}")
        ax.legend(fontsize=7); ax.grid(alpha=0.3)
        fig.tight_layout(); fig.savefig(f"{args.plots}/{key}.png", dpi=110); plt.close(fig)
        med = []
        for c in SIGCATS:
            v = vals[m & (cc == c)]; wv = ww[m & (cc == c)]
            if wv.sum() > 0:
                o = np.argsort(v); cw = np.cumsum(wv[o]) / wv.sum()
                med.append(f"{SPS.CATS[c][:14]} {v[o][np.searchsorted(cw, 0.5)]:.1f}")
        v = vals[m & ss]; wv = ww[m & ss]
        if wv.sum() > 0:
            o = np.argsort(v); cw = np.cumsum(wv[o]) / wv.sum()
            med.append(f"selected {v[o][np.searchsorted(cw, 0.5)]:.1f}")
        lines.append(f"  {key:14s}: " + " | ".join(med))

    for key, xlab, bins in SPECS:
        draw(key, xlab, bins, V[key], w, cat, sel)
    # per-photon non-detectable entries
    pw, pc, ps = w[P["evt"]], cat[P["evt"]], sel[P["evt"]]
    draw("ndTrueE", "non-detectable photon true E [MeV]", np.linspace(0, 200, 41),
         P["ndTrueE"], pw, pc, ps)
    draw("ndEvis", "non-detectable photon E_vis [MeV]", np.linspace(0, 20, 21),
         P["ndEvis"], pw, pc, ps)
    txt = "\n".join(lines)
    print(txt)
    with open(os.path.join(args.plots, "truth_medians.txt"), "w") as fh:
        fh.write(txt + "\n")
    print(f">>> plots -> {args.plots}")


if __name__ == "__main__":
    main()
