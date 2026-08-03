"""Background-topology variables for the nue CC selection (flash-cut selection).

(1) vtx_mu_score = max LArPID muon score among the OTHER reco particles sharing
    the e-shower's vertex (tracks + other showers). Targets the hard background
    where the true e-shower comes from a decay mu/pi whose decay muon was merged
    into a track: that track keeps a high muon score even though the e-shower
    itself does not. A RECO variable -> usable as a cut (keep low, or no other
    particle). Made as a stacked prediction + data + efficiency/purity-vs-cut.

(2) vtx_dist_true = reco-vtx to SCE-corrected true-vtx distance [cm] (MC only,
    already in the ntuple). A long non-nueCC tail flags secondary-interaction
    topologies (n travels, makes a secondary interaction relabeled "true", with
    a pi->mu->e chain far from the nu vertex). DIAGNOSTIC (truth-based, not a
    data cut): signal vs non-nueCC-background shapes on a log axis.

    PYTHONPATH=. python3 nue_cc_bg_vars.py --nue-npz .. --bnb-npz .. \
        --ext-npz .. --data-npz .. --plots plots_bgvars/ [--flashchi2-cut 3]
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
    ap.add_argument("--flashchi2-cut", type=float, default=3.0)
    ap.add_argument("--logy", action="store_true",
                    help="log-scale y on the vtx_mu_score stack (small nueCC "
                         "signal visible under the numuCC pile-up)")
    args = ap.parse_args()
    os.makedirs(args.plots, exist_ok=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    nue = dict(np.load(args.nue_npz)); bnb = dict(np.load(args.bnb_npz))
    ext = dict(np.load(args.ext_npz)); dat = dict(np.load(args.data_npz))
    cutv = None if args.flashchi2_cut < 0 else args.flashchi2_cut

    def base(tab):
        m = tab["sel"].astype(bool) & np.isfinite(tab["reco_ele_E"])
        if cutv is not None:
            fc = tab["flash_chi2"]
            m = m & np.isfinite(fc) & (fc > 0) & (np.log10(fc) < cutv)
        return m

    mn = base(nue); mb = base(bnb) & ~bnb["is_nuecc"]
    m_numu = mb & (np.abs(bnb["nu_pdg"]) == 14) & (bnb["ccnc"] == 0)
    m_nc = mb & (bnb["ccnc"] == 1); me = base(ext); md = base(dat)

    # ================= (1) vertex muon score ==============================
    key = "vtx_mu_score"
    def col(tab, m): return tab[key][m]
    sv = col(nue, mn); sw = nue["w"][mn]
    comp_v = [col(nue, mn), col(bnb, m_numu), col(bnb, m_nc), col(ext, me)]
    comp_w = [nue["w"][mn], bnb["w"][m_numu], bnb["w"][m_nc],
              np.full(int(me.sum()), args.ext_scale)]
    dv = col(dat, md)
    # NaN = e-shower vertex has NO other particle -> no muon -> signal-like.
    for lab, v, w in zip(LABELS, comp_v, comp_w):
        print(f"  {lab}: {np.isfinite(v).mean()*100:.0f}% have another particle "
              f"at the e-vertex (weighted {w[np.isfinite(v)].sum():.1f}/{w.sum():.1f})")
    fin = np.concatenate([v[np.isfinite(v)] for v in comp_v])
    blo, bhi = np.percentile(fin, [1, 99]) if len(fin) else (-15, 0)
    bins = np.linspace(blo, bhi, 41); ctr = 0.5 * (bins[:-1] + bins[1:])
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12.6, 4.7))
    obs = [np.clip(v[np.isfinite(v)], blo, bhi - 1e-9) for v in comp_v]
    wts = [w[np.isfinite(v)] for v, w in zip(comp_v, comp_w)]
    axL.hist(obs, bins=bins, weights=wts, stacked=True, color=COLORS,
             label=[f"{l} ({w.sum():.1f})" for l, w in zip(LABELS, wts)])
    dfin = dv[np.isfinite(dv)]
    dh, _ = np.histogram(np.clip(dfin, blo, bhi - 1e-9), bins=bins)
    axL.errorbar(ctr, dh, yerr=np.sqrt(np.clip(dh, 1, None)), fmt="ko", ms=3.5,
                 lw=1, capsize=0, label=f"data ({int(dh.sum())})")
    axL.set(xlabel="max muon score of OTHER particles at e-vertex  "
                   r"($\log p_\mu$)",
            ylabel=f"events / {args.pot:.1e} POT",
            title="vertex muon score (events with another particle at the vertex)")
    if args.logy:
        axL.set_yscale("log"); axL.set_ylim(0.05, None)
    axL.legend(fontsize=8); axL.grid(alpha=0.3, which="both")
    # eff/purity: keep vtx_mu_score < t  OR  NaN (no other particle)
    bv = np.concatenate(comp_v[1:]); bw = np.concatenate(comp_w[1:])
    thr = np.linspace(blo, bhi, 120); sig_tot = sw.sum()
    eff, pur = [], []
    for t in thr:
        sp = ~(np.isfinite(sv) & (sv >= t))     # kept: below t or NaN
        bp = ~(np.isfinite(bv) & (bv >= t))
        s = sw[sp].sum(); b = bw[bp].sum()
        eff.append(s / sig_tot if sig_tot > 0 else np.nan)
        pur.append(s / (s + b) if (s + b) > 0 else np.nan)
    axR.plot(thr, eff, "#d62728", lw=2, label="signal efficiency")
    axR.plot(thr, pur, "#1f77b4", lw=2, label="purity")
    axR.set(xlabel=r"cut: keep $\log p_\mu <$ x (or no other particle)",
            ylabel="fraction", ylim=(0, 1.02),
            title="vertex-muon-veto efficiency & purity")
    axR.legend(fontsize=8, loc="center left"); axR.grid(alpha=0.3)
    fig.tight_layout()
    p1 = os.path.join(args.plots,
                      "vtx_mu_score_logy.png" if args.logy else "vtx_mu_score.png")
    fig.savefig(p1, dpi=115); plt.close(fig); print(">>> wrote", p1)

    # ================= (2) reco-true vertex distance (MC only) =============
    def dcol(tab, m):
        v = tab["vtx_dist_true"][m]
        return v[np.isfinite(v) & (v >= 0)]
    sig_d = dcol(nue, mn)
    bkg_d = np.concatenate([dcol(bnb, m_numu), dcol(bnb, m_nc)])
    # log-spaced bins (0.1 cm to ~1000 cm)
    dbins = np.logspace(-1, 3, 41)
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    ax.hist(np.clip(sig_d, 1e-1, 1e3), bins=dbins, density=True,
            histtype="step", color="#d62728", lw=2,
            label=f"nu_e CC signal ({len(sig_d)})")
    ax.hist(np.clip(bkg_d, 1e-1, 1e3), bins=dbins, density=True,
            histtype="step", color="#1f77b4", lw=2,
            label=f"non-nueCC bkg (numuCC+NC) ({len(bkg_d)})")
    ax.set_xscale("log")
    ax.set_yscale("log")            # tail is small vs the bulk -> log-y to see it
    ax.set_ylim(1e-4, None)
    ax.axvline(5.0, color="0.5", ls=":", lw=1)
    med_s = np.median(sig_d) if len(sig_d) else np.nan
    med_b = np.median(bkg_d) if len(bkg_d) else np.nan
    tail_s = np.mean(sig_d > 5) if len(sig_d) else np.nan
    tail_b = np.mean(bkg_d > 5) if len(bkg_d) else np.nan
    ax.set(xlabel="reco-vtx to SCE-corrected true-vtx distance [cm]",
           ylabel="unit-area density",
           title=f"reco-true vertex distance (MC)\n"
                 f"median sig {med_s:.1f} / bkg {med_b:.1f} cm; "
                 f">5cm tail: sig {100*tail_s:.0f}% / bkg {100*tail_b:.0f}%")
    ax.legend(fontsize=9); ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    p2 = os.path.join(args.plots, "vtx_dist_true.png")
    fig.savefig(p2, dpi=115); plt.close(fig); print(">>> wrote", p2)
    print(f"  reco-true dist >5cm: signal {100*tail_s:.1f}%, "
          f"non-nueCC bkg {100*tail_b:.1f}%")


if __name__ == "__main__":
    main()
