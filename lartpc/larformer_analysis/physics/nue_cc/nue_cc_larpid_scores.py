"""LArPID score-separation + cut-scan plots for the nue CC selection.

For each candidate LArPID discriminant (built from the leading e-shower's PID
log-softmax [e,gamma,mu,pi,p] and process log-softmax [primary,fromN,fromC]),
in a chosen flash-chi2 band, makes a 2-panel figure:
  LEFT  : STACKED POT-normalized prediction -- nue CC signal + bnb-nu (numu CC,
          NC; true nue CC vetoed) + EXT cosmic -- with bnb5e19 data overlaid.
  RIGHT : signal efficiency (retained fraction) and purity vs the cut threshold.

Flash-chi2 band: --flash-lo/--flash-hi are log10(flash_chi2) limits. Default
band is (-inf, 3.0] = the standard cut. Use e.g. --flash-lo 2.3 --flash-hi 5 to
isolate the high-chi2 bnb5e19 data excess and see its LArPID character.

    PYTHONPATH=. python3 nue_cc_larpid_scores.py \
        --nue-npz .. --bnb-npz .. --ext-npz .. --data-npz .. --plots plots_larpid/
"""
import argparse
import os

import numpy as np

FULL_EXT_SCALE = 0.17682554549
COMPONENTS = ["nu_e CC (signal)", "nu_mu CC", "NC", "EXT cosmic"]
COLORS = ["#d62728", "#1f77b4", "#2ca02c", "#e5e5e5"]

# (key, label, cut-direction, README-suggested cut or None)
VARS = [
    ("el", r"electron score  $\log p_e$", "above", None),
    ("elconf", r"e-confidence  $\log p_e - 0.5(\log p_\pi+\log p_\gamma)$",
     "above", None),
    ("egamma", r"e/$\gamma$  $\log p_e-\log p_\gamma$", "above", None),
    ("mu", r"muon score  $\log p_\mu$", "below", -3.7),
    ("primariness",
     r"primariness  $\log p_{\rm prim}-\max(\log p_{fN},\log p_{fC})$",
     "above", 0.0),
]


def derive(tab):
    return {
        "el": tab["el_score"],
        "elconf": tab["el_score"] - 0.5 * (tab["pi_score"] + tab["ph_score"]),
        "egamma": tab["el_score"] - tab["ph_score"],
        "mu": tab["mu_score"],
        "primariness": tab["prim_score"]
        - np.maximum(tab["fromneut_score"], tab["fromchg_score"]),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--nue-npz", required=True)
    ap.add_argument("--bnb-npz", required=True)
    ap.add_argument("--ext-npz", required=True)
    ap.add_argument("--data-npz", required=True)
    ap.add_argument("--plots", required=True)
    ap.add_argument("--pot", type=float, default=4.4e19)
    ap.add_argument("--ext-scale", type=float, default=FULL_EXT_SCALE)
    ap.add_argument("--flash-lo", type=float, default=None,
                    help="lower log10(flash_chi2) band edge (default -inf)")
    ap.add_argument("--flash-hi", type=float, default=3.0,
                    help="upper log10(flash_chi2) band edge (default 3.0)")
    args = ap.parse_args()
    os.makedirs(args.plots, exist_ok=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    nue = dict(np.load(args.nue_npz)); bnb = dict(np.load(args.bnb_npz))
    ext = dict(np.load(args.ext_npz)); dat = dict(np.load(args.data_npz))
    lo = -np.inf if args.flash_lo is None else args.flash_lo
    hi = np.inf if args.flash_hi is None else args.flash_hi
    band = (f"log10(flashchi2) in "
            f"[{args.flash_lo if args.flash_lo is not None else '-inf'}, "
            f"{args.flash_hi if args.flash_hi is not None else 'inf'}]")

    def base(tab):
        m = tab["sel"].astype(bool) & np.isfinite(tab["reco_ele_E"])
        fc = tab["flash_chi2"]
        lg = np.where(np.isfinite(fc) & (fc > 0), np.log10(np.clip(fc, 1e-9, None)),
                      np.nan)
        return m & np.isfinite(lg) & (lg >= lo) & (lg < hi)

    mn = base(nue)
    mb = base(bnb) & ~bnb["is_nuecc"]
    m_numu = mb & (np.abs(bnb["nu_pdg"]) == 14) & (bnb["ccnc"] == 0)
    m_nc = mb & (bnb["ccnc"] == 1)
    me = base(ext); md = base(dat)

    dnue = derive(nue); dbnb = derive(bnb); dext = derive(ext); ddat = derive(dat)

    def comp(key):
        """per-component (values, weights) + data values, finite only."""
        def f(v, w=None):
            ok = np.isfinite(v)
            return (v[ok], None if w is None else w[ok])
        sv, sw = f(dnue[key][mn], nue["w"][mn])
        uv, uw = f(dbnb[key][m_numu], bnb["w"][m_numu])
        cv, cw = f(dbnb[key][m_nc], bnb["w"][m_nc])
        ev, _ = f(dext[key][me])
        ew = np.full(len(ev), args.ext_scale)
        dv, _ = f(ddat[key][md])
        return [sv, uv, cv, ev], [sw, uw, cw, ew], dv

    print(f"== LArPID stacked plots + cut-scan ({band}) ==")
    print(f"   raw sel in band: nue {int(mn.sum())} numuCC {int(m_numu.sum())} "
          f"NC {int(m_nc.sum())} EXT {int(me.sum())} data {int(md.sum())}")
    for key, label, direction, sugg in VARS:
        obs, wts, dv = comp(key)
        if sum(len(o) for o in obs) == 0:
            print(f"  {key}: empty, skip"); continue
        allv = np.concatenate(obs + [dv]) if len(dv) else np.concatenate(obs)
        blo, bhi = np.percentile(allv, [0.5, 99.5])
        bins = np.linspace(blo, bhi, 41)
        ctr = 0.5 * (bins[:-1] + bins[1:])
        obs = [np.clip(o, blo, bhi - 1e-9) for o in obs]
        dvc = np.clip(dv, blo, bhi - 1e-9)

        fig, (axL, axR) = plt.subplots(1, 2, figsize=(12.6, 4.7))
        # LEFT: stacked POT-normalized prediction + data
        labels = [f"{c} ({w.sum():.1f})" for c, w in zip(COMPONENTS, wts)]
        axL.hist(obs, bins=bins, weights=wts, stacked=True, color=COLORS,
                 label=labels)
        pred = sum(w.sum() for w in wts)
        dh, _ = np.histogram(dvc, bins=bins)
        axL.errorbar(ctr, dh, yerr=np.sqrt(np.clip(dh, 1, None)), fmt="ko",
                     ms=3.5, lw=1, capsize=0,
                     label=f"bnb5e19 data ({int(dh.sum())})")
        if sugg is not None:
            axL.axvline(sugg, color="k", ls="--", lw=1.0)
        axL.set(xlabel=label, ylabel=f"events / {args.pot:.1e} POT",
                title=f"{key}: prediction vs data\ndata {int(dh.sum())} "
                      f"vs pred {pred:.0f}")
        axL.legend(fontsize=7.5); axL.grid(alpha=0.3)

        # RIGHT: efficiency + purity vs threshold (MC signal vs MC+EXT bkg)
        sv, sw = obs[0], wts[0]
        bv = np.concatenate(obs[1:]); bw = np.concatenate(wts[1:])
        thr = np.linspace(blo, bhi, 120)
        sig_tot = sw.sum()
        eff, pur = [], []
        for tval in thr:
            sp = (sv >= tval) if direction == "above" else (sv <= tval)
            bp = (bv >= tval) if direction == "above" else (bv <= tval)
            s = sw[sp].sum(); b = bw[bp].sum()
            eff.append(s / sig_tot if sig_tot > 0 else np.nan)
            pur.append(s / (s + b) if (s + b) > 0 else np.nan)
        eff = np.array(eff); pur = np.array(pur)
        axR.plot(thr, eff, color="#d62728", lw=2, label="signal efficiency")
        axR.plot(thr, pur, color="#1f77b4", lw=2, label="purity")
        if sugg is not None:
            axR.axvline(sugg, color="k", ls="--", lw=1.0,
                        label=f"README cut {sugg:g}")
        axR.set(xlabel=label + f"  (keep {direction})", ylabel="fraction",
                ylim=(0, 1.02), title=f"{key}: efficiency & purity vs cut")
        axR.legend(fontsize=8, loc="center right"); axR.grid(alpha=0.3)
        fig.tight_layout()
        p = os.path.join(args.plots, f"larpid_{key}.png")
        fig.savefig(p, dpi=115); plt.close(fig)
        good = eff >= 0.8
        bestp = np.nanmax(pur[good]) if good.any() else np.nan
        print(f"  {key:12s}: sig {sig_tot:.1f} bkg {bw.sum():.1f} data "
              f"{int(len(dv))}; max purity @eff>=0.8 = {bestp:.3f} -> {p}")


if __name__ == "__main__":
    main()
