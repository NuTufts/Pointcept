"""Bethe-Bloch CSDA range<->KE tables for mu/pi/p in liquid argon -> npz.

Computes dE/dx (Bethe-Bloch + Sternheimer density correction for LAr), integrates
to the CSDA range(KE), and saves KE(range) tables for the range-based track
momentum (spec particle_momentum_spec.md §3.2). ROOT-free at reco time.

    python make_range2ke_npz.py                 # write data/range2ke_lar.npz
    python make_range2ke_npz.py --validate ROOT # compare vs LANTERN TSplines

The optional --validate reads the ROOT muon/proton splines
(Proton_Muon_Range_dEdx_LAr_TSplines.root) and reports the max % difference.
"""
import os
import argparse

import numpy as np

# --- liquid argon + physics constants ---
ME = 0.510999          # electron mass, MeV
K = 0.307075           # 4*pi*N_A*r_e^2*m_e c^2, MeV cm^2/mol
Z_A = 18.0 / 39.948    # argon Z/A
RHO = 1.3954           # liquid argon density, g/cm^3
I_MEV = 188.0e-6       # mean excitation energy, MeV (188 eV)
# Sternheimer density-effect params for liquid argon (PDG)
_STERN = dict(C=-5.2146, X0=0.2000, X1=3.0000, a=0.19559, m=3.0000, d0=0.0)
MASS = {"muon": 105.6584, "pion": 139.5704, "proton": 938.2721}


def _delta(bg):
    """Sternheimer density correction as a function of beta*gamma."""
    X = np.log10(bg)
    s = _STERN
    d = np.where(X < s["X0"],
                 s["d0"] * 10.0 ** (2.0 * (X - s["X0"])),
                 np.where(X < s["X1"],
                          2 * np.log(10) * X + s["C"] + s["a"] * (s["X1"] - X) ** s["m"],
                          2 * np.log(10) * X + s["C"]))
    return d


def dedx_mev_cm(ke, mass, z=1):
    """Bethe-Bloch mean stopping power (MeV/cm) in LAr for KE (MeV)."""
    ke = np.asarray(ke, np.float64)
    E = ke + mass
    gamma = E / mass
    beta2 = 1.0 - 1.0 / gamma ** 2
    bg = np.sqrt(beta2) * gamma
    tmax = (2 * ME * beta2 * gamma ** 2
            / (1 + 2 * gamma * ME / mass + (ME / mass) ** 2))
    bracket = (0.5 * np.log(2 * ME * beta2 * gamma ** 2 * tmax / I_MEV ** 2)
               - beta2 - _delta(bg) / 2.0)
    return RHO * K * z * z * Z_A / beta2 * bracket


def csda_range(mass, ke_max=4000.0, n=200000, ke_min=1.0):
    """CSDA range(KE): cumulative integral of dE/(dE/dx). Returns (ke, range_cm),
    KE ascending. Integration starts at ke_min (sub-MeV range is negligible)."""
    ke = np.linspace(ke_min, ke_max, n)
    inv = 1.0 / dedx_mev_cm(ke, mass)
    rng = np.concatenate([[0.0], np.cumsum(0.5 * (inv[1:] + inv[:-1]) * np.diff(ke))])
    return ke, rng


def build_tables(**kw):
    out = {}
    for name, m in MASS.items():
        ke, rng = csda_range(m, **kw)
        out[f"{name}_ke"] = ke.astype(np.float32)
        out[f"{name}_range"] = rng.astype(np.float32)
    return out


def range_to_ke(length_cm, tables, particle):
    """Interpolate KE (MeV) at a track length (cm). Reco-time entry point."""
    return float(np.interp(length_cm, tables[f"{particle}_range"],
                           tables[f"{particle}_ke"]))


def _validate(tables, root_path):
    try:
        import ROOT
    except Exception as e:
        print(f"  ROOT unavailable ({type(e).__name__}); skipping validation")
        return
    f = ROOT.TFile(root_path)
    for name, spl in (("muon", "sMuonRange2T"), ("proton", "sProtonRange2T")):
        s = f.Get(spl)
        if not s:
            print(f"  {spl} not found")
            continue
        # spline is range[cm] -> KE[MeV]; compare on a range grid
        rg = np.linspace(1.0, 300.0, 60)
        ke_bb = np.interp(rg, tables[f"{name}_range"], tables[f"{name}_ke"])
        ke_root = np.array([s.Eval(float(r)) for r in rg])
        d = np.abs(ke_bb - ke_root) / np.clip(ke_root, 1e-6, None)
        print(f"  {name}: max diff {100*d.max():.1f}%  median {100*np.median(d):.1f}%"
              f"  (e.g. range=100cm: BB={ke_bb[np.argmin(abs(rg-100))]:.1f} "
              f"ROOT={ke_root[np.argmin(abs(rg-100))]:.1f} MeV)")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--out", default=os.path.join(here, "data", "range2ke_lar.npz"))
    ap.add_argument("--validate", default=None,
                    help="path to Proton_Muon_Range_dEdx_LAr_TSplines.root")
    args = ap.parse_args()
    tables = build_tables()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez(args.out, **tables)
    print(f">>> wrote {args.out}")
    for name in MASS:
        for L in (10, 50, 100, 200):
            print(f"    {name:6s} range={L:4d}cm -> KE="
                  f"{range_to_ke(L, tables, name):7.1f} MeV")
    if args.validate:
        print(">>> validation vs ROOT splines:")
        _validate(tables, args.validate)


if __name__ == "__main__":
    main()
