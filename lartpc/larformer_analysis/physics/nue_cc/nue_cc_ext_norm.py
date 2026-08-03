"""EXT (beam-off cosmic) normalization diagnostic for the nue CC selection.

The bnb5e19 data sits ~20% above the (MC + EXT) prediction. Is that a cosmic
(EXT) under-normalization, or a global offset (POT / overall data-MC)? The
flash-chi2 shape tells them apart:
  - EXT under-normalized  -> data/pred RISES with log10(flash_chi2) (EXT
    dominates the high-chi2, out-of-time region).
  - global offset (POT)   -> data/pred is FLAT in log10(flash_chi2).

Top panel: stacked POT-normalized prediction (nueCC + numuCC + NC + EXT) + data
over the FULL flash-chi2 range (base reco selection, no flash cut). Bottom panel:
data/pred ratio per bin. Also FITS the EXT scale in a cosmic-pure high-chi2
sideband:  ext_scale = (N_data - N_MC) / N_ext_raw  (beam nu is in-time -> ~0 at
high chi2), and prints it against the spill ratio 0.17682554549.

    PYTHONPATH=. python3 nue_cc_ext_norm.py --nue-npz .. --bnb-npz .. \
        --ext-npz .. --data-npz .. --plots plots_extnorm/ [--sideband 3.5]
"""
import argparse
import os

import numpy as np

FULL_EXT_SCALE = 0.17682554549
COLORS = ["#d62728", "#1f77b4", "#2ca02c", "#e5e5e5"]
LABELS = ["nu_e CC", "nu_mu CC", "NC", "EXT cosmic"]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--nue-npz", required=True)
    ap.add_argument("--bnb-npz", required=True)
    ap.add_argument("--ext-npz", required=True)
    ap.add_argument("--data-npz", required=True)
    ap.add_argument("--plots", required=True)
    ap.add_argument("--pot", type=float, default=4.4e19)
    ap.add_argument("--ext-scale", type=float, default=FULL_EXT_SCALE)
    ap.add_argument("--sideband", type=float, default=3.5,
                    help="fit EXT scale where log10(flash_chi2) > this (cosmic-"
                         "pure). Beam nu is in-time so it's ~absent here.")
    args = ap.parse_args()
    os.makedirs(args.plots, exist_ok=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    nue = dict(np.load(args.nue_npz)); bnb = dict(np.load(args.bnb_npz))
    ext = dict(np.load(args.ext_npz)); dat = dict(np.load(args.data_npz))

    def sel_logchi2(tab):
        m = tab["sel"].astype(bool) & np.isfinite(tab["reco_ele_E"])
        fc = tab["flash_chi2"]
        ok = m & np.isfinite(fc) & (fc > 0)
        return np.log10(fc[ok]), ok

    ln, on = sel_logchi2(nue); lb, ob = sel_logchi2(bnb)
    le, oe = sel_logchi2(ext); ld, od = sel_logchi2(dat)
    veto = ~bnb["is_nuecc"][ob]
    numu = veto & (np.abs(bnb["nu_pdg"][ob]) == 14) & (bnb["ccnc"][ob] == 0)
    nc = veto & (bnb["ccnc"][ob] == 1)

    bins = np.linspace(0, 6, 31)
    ctr = 0.5 * (bins[:-1] + bins[1:])
    comps = [(ln, nue["w"][on]), (lb[numu], bnb["w"][ob][numu]),
             (lb[nc], bnb["w"][ob][nc]),
             (le, np.full(len(le), args.ext_scale))]
    H = [np.histogram(v, bins=bins, weights=w)[0] for v, w in comps]
    pred = np.sum(H, axis=0)
    dh, _ = np.histogram(ld, bins=bins)

    # ---- sideband EXT-scale fit ------------------------------------------
    sb = args.sideband
    n_data_sb = int((ld > sb).sum())
    n_mc_sb = (nue["w"][on][ln > sb].sum()
               + bnb["w"][ob][numu][lb[numu] > sb].sum()
               + bnb["w"][ob][nc][lb[nc] > sb].sum())
    n_ext_raw_sb = int((le > sb).sum())
    ext_fit = (n_data_sb - n_mc_sb) / n_ext_raw_sb if n_ext_raw_sb else np.nan

    # ---- global factor over the full selection ---------------------------
    glob = dh.sum() / pred.sum()

    print(f"== EXT normalization diagnostic ==")
    print(f"  full selection: data {int(dh.sum())} vs pred {pred.sum():.0f} "
          f"-> global data/pred = {glob:.3f}")
    print(f"  sideband log10(chi2)>{sb}: data {n_data_sb}, MC {n_mc_sb:.1f}, "
          f"EXT_raw {n_ext_raw_sb}")
    print(f"  -> fitted EXT scale = {ext_fit:.4f} "
          f"(current {args.ext_scale:.4f}, spill ratio {FULL_EXT_SCALE:.4f}); "
          f"ratio {ext_fit / args.ext_scale:.2f}x")

    # ---- plot -------------------------------------------------------------
    fig, (ax, axr) = plt.subplots(2, 1, figsize=(7.6, 6.0), sharex=True,
                                  gridspec_kw={"height_ratios": [3, 1]})
    ax.hist([c[0] for c in comps], bins=bins, weights=[c[1] for c in comps],
            stacked=True, color=COLORS,
            label=[f"{l} ({np.sum(w):.0f})" for l, (_, w) in zip(LABELS, comps)])
    ax.errorbar(ctr, dh, yerr=np.sqrt(np.clip(dh, 1, None)), fmt="ko", ms=3.5,
                lw=1, capsize=0, label=f"bnb5e19 data ({int(dh.sum())})")
    ax.axvline(sb, color="0.4", ls=":", lw=1.2)
    ax.axvline(3.0, color="k", ls="--", lw=1.0, label="std cut log10<3")
    ax.set(ylabel=f"events / {args.pot:.1e} POT",
           title=f"nu-vtx flash-chi2 (base sel): global data/pred={glob:.2f}, "
                 f"sideband EXT fit={ext_fit:.4f} ({ext_fit/args.ext_scale:.2f}x)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    ratio = np.divide(dh, pred, out=np.full_like(pred, np.nan), where=pred > 0)
    rerr = np.divide(np.sqrt(np.clip(dh, 1, None)), pred,
                     out=np.full_like(pred, np.nan), where=pred > 0)
    axr.errorbar(ctr, ratio, yerr=rerr, fmt="ko", ms=3, lw=1, capsize=0)
    axr.axhline(1.0, color="0.4", lw=1)
    axr.axhline(glob, color="#d62728", ls="--", lw=1, label=f"global {glob:.2f}")
    axr.axvline(sb, color="0.4", ls=":", lw=1.2)
    axr.axvline(3.0, color="k", ls="--", lw=1.0)
    axr.set(xlabel=r"$\log_{10}(\mathrm{flash}\ \chi^2)$", ylabel="data/pred",
            ylim=(0, 2.2))
    axr.legend(fontsize=8); axr.grid(alpha=0.3)
    fig.tight_layout()
    p = os.path.join(args.plots, "flashchi2_extnorm.png")
    fig.savefig(p, dpi=120); plt.close(fig)
    print(">>> wrote", p)
    print("\n  INTERPRETATION: ratio ~flat across chi2 -> GLOBAL offset (POT / "
          "overall); ratio RISES with chi2 -> EXT under-normalized.")


if __name__ == "__main__":
    main()
