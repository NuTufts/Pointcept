"""Refactor equivalence: the shared C.base()/cut_mask() must reproduce the
ORIGINAL nue_cc_overlay.py base()+larpid_ok() bit-for-bit on the same tables."""
import os
import sys

import numpy as np
sys.path.insert(0, "/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/kpv2_pointcept/lartpc/larformer_analysis/physics/nue_cc")
import nue_cc_common as C

S = os.environ.get("NUECC_TABLES", ".")

class A:  # stand-in for the original argparse Namespace
    pass

def old_base(tab, args, cutv, use_flash=True):
    """VERBATIM logic of the original nue_cc_overlay.py:84-126."""
    m = tab["sel"].astype(bool) & np.isfinite(tab["reco_ele_E"])
    if use_flash and cutv is not None:
        fc = tab["flash_chi2"]
        m = m & np.isfinite(fc) & (fc > 0) & (np.log10(fc) < cutv)
    if not use_flash:
        return m
    # larpid_ok
    if args.elconf_cut is not None:
        ec = tab["el_score"] - 0.5 * (tab["pi_score"] + tab["ph_score"])
        m &= np.isfinite(ec) & (ec > args.elconf_cut)
    if args.egamma_cut is not None:
        eg = tab["el_score"] - tab["ph_score"]
        m &= np.isfinite(eg) & (eg > args.egamma_cut)
    if args.primariness_cut is not None:
        pr = tab["prim_score"] - np.maximum(tab["fromneut_score"], tab["fromchg_score"])
        m &= np.isfinite(pr) & (pr > args.primariness_cut)
    if args.mu_cut is not None:
        m &= np.isfinite(tab["mu_score"]) & (tab["mu_score"] < args.mu_cut)
    if args.vtxmu_cut is not None:
        vm = tab["vtx_mu_score"]
        m &= ~(np.isfinite(vm) & (vm >= args.vtxmu_cut))
    if args.elconf_lf_cut is not None:
        ec = tab["lf_el_score"] - 0.5 * (tab["lf_pi_score"] + tab["lf_ph_score"])
        m &= np.isfinite(ec) & (ec > args.elconf_lf_cut)
    if args.egamma_lf_cut is not None:
        eg = tab["lf_el_score"] - tab["lf_ph_score"]
        m &= np.isfinite(eg) & (eg > args.egamma_lf_cut)
    if args.mu_lf_cut is not None:
        m &= np.isfinite(tab["lf_mu_score"]) & (tab["lf_mu_score"] < args.mu_lf_cut)
    if args.vtxmu_lf_cut is not None:
        vm = tab["vtx_lf_mu_score"]
        m &= ~(np.isfinite(vm) & (vm >= args.vtxmu_lf_cut))
    if args.nphoton_max is not None:
        nph = np.where(tab["n_photons"] < 0, 0, tab["n_photons"])
        m &= nph <= args.nphoton_max
    return m

SCENARIOS = [
    dict(flashchi2_cut=3.0),
    dict(flashchi2_cut=3.0, elconf_cut=9.0),
    dict(flashchi2_cut=3.0, elconf_cut=7.0, primariness_cut=0.0, mu_cut=-3.7),
    dict(flashchi2_cut=3.0, elconf_cut=9.0, vtxmu_cut=-3.7),
    dict(flashchi2_cut=3.0, vtxmu_lf_cut=-2.0, mu_cut=-3.0, primariness_cut=1.0),
    dict(flashchi2_cut=3.0, elconf_lf_cut=5.0, egamma_lf_cut=1.0, mu_lf_cut=-6.0),
    dict(flashchi2_cut=3.0, nphoton_max=0),
    dict(flashchi2_cut=2.5, egamma_cut=2.0),
    dict(flashchi2_cut=-1.0),                      # disabled sentinel
]
FIELDS = ["flashchi2_cut","elconf_cut","egamma_cut","primariness_cut","mu_cut",
          "vtxmu_cut","elconf_lf_cut","egamma_lf_cut","mu_lf_cut","vtxmu_lf_cut",
          "nphoton_max","ngoodphoton_max","nnovtxphoton_max","objectness_cut",
          "vtxscore_cut","vtxfraccosmic_cut","flash_lo"]

tabs = {n: dict(np.load(f"{S}/test_{n}.npz"))
        for n in ("nue_july","bnb","ext","data")}
allok = True
for si, sc in enumerate(SCENARIOS):
    args = A()
    for f in FIELDS:
        setattr(args, f, None)
    for k, v in sc.items():
        setattr(args, k, v)
    cutv = None if (sc.get("flashchi2_cut") is not None
                    and sc["flashchi2_cut"] < 0) else sc.get("flashchi2_cut")
    cuts = C.cuts_from_args(args)
    for name, tab in tabs.items():
        old = old_base(tab, args, cutv, use_flash=True)
        new = C.base(tab, cuts)
        if not np.array_equal(old, new):
            allok = False
            print(f"  MISMATCH scenario {si} {sc} sample {name}: "
                  f"old {old.sum()} new {new.sum()} diff {(old^new).sum()}")
        # also the no-cut variant
        old0 = old_base(tab, args, cutv, use_flash=False)
        new0 = C.base(tab, cuts, apply_cuts=False)
        if not np.array_equal(old0, new0):
            allok = False
            print(f"  MISMATCH(nocut) scenario {si} sample {name}")
    print(f"  scenario {si}: {C.cut_label(cuts) or 'none':60s} OK")
print("\nEQUIVALENCE:", "PASS -- all scenarios bit-for-bit identical" if allok else "FAIL")
sys.exit(0 if allok else 1)
