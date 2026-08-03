"""Observed in-time beam-flash PE: bnb5e19 (Run 1) data vs Run-3 prediction.

Tests the light-yield / run-period hypothesis for the flash-chi2 transition
excess: EXT + MC are Run 3, bnb5e19 data is Run 1, and Run 1 has higher PMT
light yield. Expect the Run-1 data observed-PE distribution shifted to HIGHER PE
than the Run-3-based prediction, AND -- because more faint in-time cosmics cross
the (fixed) light trigger in Run 1 -- an excess of data piled up at LOW PE.

Top: stacked POT-normalized prediction (nueCC + numuCC + NC + EXT) + data over
observed PE. Bottom: data/pred ratio (a rising trend toward high PE = the shift;
a spike at low PE = the newly-triggered faint cosmics). Requires the tables to
carry `observed_pe` (add_observed_pe.py). Optional flash-chi2 band via
--flash-lo/--flash-hi.

    PYTHONPATH=. python3 nue_cc_observed_pe.py --nue-npz .. --bnb-npz .. \
        --ext-npz .. --data-npz .. --plots plots_pe/ [--flash-lo 2.3 --flash-hi 3]
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
    ap.add_argument("--pe-max", type=float, default=2500.0)
    ap.add_argument("--nbins", type=int, default=50)
    ap.add_argument("--flash-lo", type=float, default=None)
    ap.add_argument("--flash-hi", type=float, default=None)
    ap.add_argument("--tag", default="full")
    ap.add_argument("--logy", action="store_true",
                    help="log-scale y (makes the small nueCC signal visible)")
    args = ap.parse_args()
    os.makedirs(args.plots, exist_ok=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    nue = dict(np.load(args.nue_npz)); bnb = dict(np.load(args.bnb_npz))
    ext = dict(np.load(args.ext_npz)); dat = dict(np.load(args.data_npz))
    for tag, tb in (("nue", nue), ("bnb", bnb), ("ext", ext), ("data", dat)):
        if "observed_pe" not in tb:
            raise SystemExit(f"{tag} table has no observed_pe -- run "
                             f"add_observed_pe.py first")
    lo = -np.inf if args.flash_lo is None else args.flash_lo
    hi = np.inf if args.flash_hi is None else args.flash_hi

    def base(tab):
        m = (tab["sel"].astype(bool) & np.isfinite(tab["reco_ele_E"])
             & np.isfinite(tab["observed_pe"]))
        fc = tab["flash_chi2"]
        lg = np.where(np.isfinite(fc) & (fc > 0), np.log10(np.clip(fc, 1e-9, None)),
                      np.nan)
        return m & np.isfinite(lg) & (lg >= lo) & (lg < hi)

    mn = base(nue); mb = base(bnb) & ~bnb["is_nuecc"]
    m_numu = mb & (np.abs(bnb["nu_pdg"]) == 14) & (bnb["ccnc"] == 0)
    m_nc = mb & (bnb["ccnc"] == 1); me = base(ext); md = base(dat)

    bins = np.linspace(0, args.pe_max, args.nbins + 1)
    ctr = 0.5 * (bins[:-1] + bins[1:])
    clip = lambda v: np.clip(v, 0, args.pe_max - 1e-6)
    comps = [(clip(nue["observed_pe"][mn]), nue["w"][mn]),
             (clip(bnb["observed_pe"][m_numu]), bnb["w"][m_numu]),
             (clip(bnb["observed_pe"][m_nc]), bnb["w"][m_nc]),
             (clip(ext["observed_pe"][me]), np.full(int(me.sum()), args.ext_scale))]
    H = [np.histogram(v, bins=bins, weights=w)[0] for v, w in comps]
    pred = np.sum(H, axis=0)
    dh, _ = np.histogram(clip(dat["observed_pe"][md]), bins=bins)

    # median PE (data vs prediction) -- the "shift" summary
    def wmedian(v, w):
        o = np.argsort(v); v, w = v[o], w[o]; c = np.cumsum(w)
        return v[np.searchsorted(c, 0.5 * c[-1])] if len(v) else np.nan
    predv = np.concatenate([c[0] for c in comps])
    predw = np.concatenate([c[1] for c in comps])
    med_pred = wmedian(predv, predw)
    med_data = np.median(clip(dat["observed_pe"][md])) if md.sum() else np.nan
    band = ("all flash-chi2" if args.flash_lo is None and args.flash_hi is None
            else f"log10(chi2) in [{args.flash_lo},{args.flash_hi}]")
    print(f"== observed PE ({band}) ==")
    print(f"  data {int(dh.sum())} vs pred {pred.sum():.0f}; "
          f"median PE  data={med_data:.0f}  pred={med_pred:.0f}")

    fig, (ax, axr) = plt.subplots(2, 1, figsize=(7.8, 6.0), sharex=True,
                                  gridspec_kw={"height_ratios": [3, 1]})
    ax.hist([c[0] for c in comps], bins=bins, weights=[c[1] for c in comps],
            stacked=True, color=COLORS,
            label=[f"{l} ({np.sum(w):.0f})" for l, (_, w) in zip(LABELS, comps)])
    ax.errorbar(ctr, dh, yerr=np.sqrt(np.clip(dh, 1, None)), fmt="ko", ms=3.5,
                lw=1, capsize=0, label=f"bnb5e19 data ({int(dh.sum())})")
    ax.axvline(med_data, color="k", ls=":", lw=1, label=f"data median {med_data:.0f}")
    ax.axvline(med_pred, color="r", ls=":", lw=1, label=f"pred median {med_pred:.0f}")
    ax.set(ylabel=f"events / {args.pot:.1e} POT",
           title=f"in-time flash observed PE ({band})\n"
                 f"data {int(dh.sum())} vs pred {pred.sum():.0f}; "
                 f"median data {med_data:.0f} vs pred {med_pred:.0f} PE")
    if args.logy:
        ax.set_yscale("log")
        ax.set_ylim(0.02, None)   # signal (nueCC) is O(0.1-1)/bin -> visible
    ax.legend(fontsize=7.5); ax.grid(alpha=0.3, which="both")
    ratio = np.divide(dh, pred, out=np.full_like(pred, np.nan), where=pred > 0)
    rerr = np.divide(np.sqrt(np.clip(dh, 1, None)), pred,
                     out=np.full_like(pred, np.nan), where=pred > 0)
    axr.errorbar(ctr, ratio, yerr=rerr, fmt="ko", ms=3, lw=1, capsize=0)
    axr.axhline(1.0, color="0.4", lw=1)
    axr.set(xlabel="in-time beam-flash observed PE", ylabel="data/pred",
            ylim=(0, 2.5))
    axr.grid(alpha=0.3)
    fig.tight_layout()
    suffix = args.tag + ("_logy" if args.logy else "")
    p = os.path.join(args.plots, f"observed_pe_{suffix}.png")
    fig.savefig(p, dpi=120); plt.close(fig)
    print(">>> wrote", p)


if __name__ == "__main__":
    main()
