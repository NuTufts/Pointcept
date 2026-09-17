"""Flash-chi2 data-vs-prediction, BEFORE (cew6: run-3 MC/EXT + legacy gamma)
and AFTER (run-1 MC/EXT/data, calibrated gamma table) side by side.

Reads the *_nochi2bdt tables of two prefixes (event-BDT applied, chi2 gate
open). EXT rows weighted by scale x table w in BOTH columns, so the old
column is normalised correctly even where its hygiene-doubled rows exist.

Figure 1 (flashchi2_before_after.png): rows = reco-CC eq2 / reco-NC eq2 /
CC+NC ge2; columns = old / new; stacked true-pi0 signal + other MC + EXT,
data points, the cut line, and data/pred below the cut in the title.
Figure 2 (flashchi2_shapes_before_after.png): the same streams, data and
total prediction of both versions as step histograms (absolute, per 4.4e19
POT) -- the shift of the data peak and of the prediction are visible
directly.

    python3 flashchi2_before_after.py --old-prefix cew6_ts0164 \
        --old-ext-scale 1.1818 --new-prefix run1tg_ts0164 \
        --new-ext-scale 16.355 --plots plots_run1tg_flashchi2_ts0164
"""
import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
STREAMS = [("reco-CC, exactly 2", True, True), ("reco-NC, exactly 2", False, True),
           ("CC+NC combined, >=2", None, False)]


def load(prefix, ext_scale, tdir=HERE):
    d = {}
    for s in ("mc", "data", "ext"):
        t = np.load(os.path.join(tdir, f"{s}_{prefix}_nochi2bdt_table.npz"))
        d[s] = {k: np.asarray(t[k]) for k in
                ("sel_ge2", "sel_eq2", "reco_cc", "flash_chi2", "w")}
        if s == "mc":
            d[s]["cat"] = np.asarray(t["cat"])
    d["ext"]["w"] = d["ext"]["w"].astype(float) * ext_scale
    d["data"]["w"] = np.ones(len(d["data"]["w"]))
    return d


def mask(t, want_cc, eq2):
    s = (t["sel_eq2"] if eq2 else t["sel_ge2"]).astype(bool)
    if want_cc is not None:
        s = s & (t["reco_cc"].astype(bool) == want_cc)
    return s & np.isfinite(t["flash_chi2"])


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--old-prefix", default="cew6_ts0164")
    ap.add_argument("--old-ext-scale", type=float, default=1.1818)
    ap.add_argument("--old-label", default="cew6: run-3 MC/EXT vs run-1 data, legacy gamma")
    ap.add_argument("--new-prefix", required=True)
    ap.add_argument("--new-ext-scale", type=float, required=True)
    ap.add_argument("--new-label", default="run-1 MC/EXT/data, calibrated gamma table")
    ap.add_argument("--chi2-cc", type=float, default=1e4)
    ap.add_argument("--chi2-nc", type=float, default=1778.0)
    ap.add_argument("--chi2-combined", type=float, default=3162.3)
    ap.add_argument("--old-dir", default=HERE, help="dir of the old tables")
    ap.add_argument("--new-dir", default=HERE, help="dir of the new tables")
    ap.add_argument("--plots", required=True)
    args = ap.parse_args()
    os.makedirs(args.plots, exist_ok=True)
    versions = [(args.old_label, load(args.old_prefix, args.old_ext_scale, args.old_dir)),
                (args.new_label, load(args.new_prefix, args.new_ext_scale, args.new_dir))]
    bins = np.linspace(0, 8, 33); ctr = 0.5 * (bins[:-1] + bins[1:])
    lchi = lambda t, m: np.clip(np.log10(np.clip(t["flash_chi2"][m], 1, None)), 0, 7.999)

    fig, axes = plt.subplots(3, 2, figsize=(12.5, 12))
    summary = []
    for r, (slab, want_cc, eq2) in enumerate(STREAMS):
        cut = (args.chi2_combined if want_cc is None
               else args.chi2_cc if want_cc else args.chi2_nc)
        for c, (vlab, d) in enumerate(versions):
            ax = axes[r, c]
            mm, dm, em = (mask(d["mc"], want_cc, eq2), mask(d["data"], want_cc, eq2),
                          mask(d["ext"], want_cc, eq2))
            sig = mm & (d["mc"]["cat"] >= 0) & (d["mc"]["cat"] < 2)
            oth = mm & (d["mc"]["cat"] >= 2)
            stack = [lchi(d["mc"], sig), lchi(d["mc"], oth), lchi(d["ext"], em)]
            ws = [d["mc"]["w"][sig], d["mc"]["w"][oth], d["ext"]["w"][em]]
            ax.hist(stack, bins=bins, weights=ws, stacked=True,
                    color=["#d62728", "#ff7f0e", "#e5e5e5"],
                    label=[f"true pi0 signal ({ws[0].sum():.0f})",
                           f"other nu MC ({ws[1].sum():.0f})",
                           f"EXT cosmic ({ws[2].sum():.0f})"])
            dh, _ = np.histogram(lchi(d["data"], dm), bins=bins)
            ax.errorbar(ctr, dh, yerr=np.sqrt(np.clip(dh, 1, None)), fmt="ko",
                        ms=3.5, lw=1, label=f"beam data ({int(dh.sum())})")
            lc = np.log10(cut)
            ax.axvline(lc, color="crimson", ls="--", lw=1.3)
            below = lambda t, m: m & (t["flash_chi2"] < cut)
            pb = (d["mc"]["w"][below(d["mc"], mm)].sum()
                  + d["ext"]["w"][below(d["ext"], em)].sum())
            db = int(below(d["data"], dm).sum())
            pa = sum(w.sum() for w in ws) - pb; da_ = int(dh.sum()) - db
            ax.set(xlabel=r"$\log_{10}$ flash $\chi^2$ (nu-stream vertex slice)",
                   ylabel="events / 4.4e19 POT")
            ax.set_title(f"{slab} -- {vlab}\nchi2<{cut:g}: data {db} / pred "
                         f"{pb:.0f} = {db/max(pb,1e-9):.2f} | above: {da_} / "
                         f"{pa:.0f} = {da_/max(pa,1e-9):.2f}", fontsize=9)
            ax.legend(fontsize=7); ax.grid(alpha=0.3)
            summary.append((slab, "old" if c == 0 else "new", db, pb, da_, pa))
    fig.tight_layout()
    fig.savefig(f"{args.plots}/flashchi2_before_after.png", dpi=110)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    for ax, (slab, want_cc, eq2) in zip(axes, STREAMS):
        for (vlab, d), col in zip(versions, ("tab:gray", "tab:blue")):
            mm, dm, em = (mask(d["mc"], want_cc, eq2), mask(d["data"], want_cc, eq2),
                          mask(d["ext"], want_cc, eq2))
            ph, _ = np.histogram(np.r_[lchi(d["mc"], mm), lchi(d["ext"], em)],
                                 bins=bins, weights=np.r_[d["mc"]["w"][mm], d["ext"]["w"][em]])
            dh, _ = np.histogram(lchi(d["data"], dm), bins=bins)
            ax.step(bins, np.r_[ph, ph[-1]], where="post", color=col, lw=1.6,
                    label=f"pred: {vlab.split(':')[0].split(',')[0]} ({ph.sum():.0f})")
            ax.errorbar(ctr, dh, yerr=np.sqrt(np.clip(dh, 1, None)), fmt="o",
                        color=col, ms=3.2, lw=1, alpha=0.9,
                        label=f"data: {vlab.split(':')[0].split(',')[0]} ({int(dh.sum())})")
        ax.set(title=slab, xlabel=r"$\log_{10}$ flash $\chi^2$", ylabel="events / 4.4e19 POT")
        ax.legend(fontsize=7); ax.grid(alpha=0.3)
    fig.suptitle("flash-chi2 spectra: prediction (line) and data (points), old (grey) vs new (blue)", fontsize=11)
    fig.tight_layout()
    fig.savefig(f"{args.plots}/flashchi2_shapes_before_after.png", dpi=110)
    plt.close(fig)
    print(f"{'stream':<22}{'ver':>5}{'data<cut':>10}{'pred<cut':>10}{'d/p':>6}"
          f"{'data>cut':>10}{'pred>cut':>10}{'d/p':>6}")
    for slab, ver, db, pb, da_, pa in summary:
        print(f"{slab:<22}{ver:>5}{db:>10d}{pb:>10.1f}{db/max(pb,1e-9):>6.2f}"
              f"{da_:>10d}{pa:>10.1f}{da_/max(pa,1e-9):>6.2f}")
    print(f">>> plots -> {args.plots}")


if __name__ == "__main__":
    main()
