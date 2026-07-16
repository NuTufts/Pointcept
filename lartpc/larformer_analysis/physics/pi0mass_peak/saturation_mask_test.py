"""Analysis-level test of the saturated-PMT mask: recompute the nu-slice flash
chi2 from the per-PMT arrays already stored in the cascade files, with no GPU
re-run of the chain.

For each selected event this recomputes the nu-slice Neyman chi2 four ways --
no mask / dead-only / dead+saturation at several caps -- and plots the log10
distributions, plus the distribution of saturation candidates per event (which
is what motivates the cap: without one, a pathological event could mask away
most of the array and earn an artificially low chi2).

CAVEAT (same as flash_correction.py): this corrects the chi2 OF the nu slice the
cascade already picked. The full fix also changes WHICH slice the flashmatch
stream selects, which needs a cascade re-run. On the traced event the nu slice
was already the correct one, so the recomputation is the honest preview.

    PYTHONPATH=./ python3 .../saturation_mask_test.py \
        --ntuple <mc.root> --cascade-dir <keypoint2_streams> \
        --plots plots_saturation [--reco-cc] [--eq2]
"""
import argparse
import os
import re
import sys

import numpy as np
import awkward as ak
import uproot
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "..", ".."))
from lartpc.flashmatch.saturation import find_saturated          # noqa: E402
from flash_correction import neyman_masked, rse_map              # noqa: E402

RECO_G_MIN = 20.0     # photon shower energy threshold [MeV]
MU_KE_MIN = 100.0     # primary muon KE threshold [MeV]
CAPS = (2, 4, 6, None)


def select(ntuple, reco_cc, eq2):
    """Same reco selection as flashchi2_ncpi0 / make_ncpi0_browse_list."""
    a = uproot.open(ntuple)["EventTree"].arrays(
        ["run", "subrun", "event", "foundVertex", "primaryVtxStream",
         "vtxIsFiducial", "showerLArFormerPID", "showerRecoE",
         "trackLArFormerPID", "trackIsSecondary", "trackRecoE"])
    vok = ((np.asarray(a["foundVertex"]) == 1)
           & (np.asarray(a["primaryVtxStream"]) == 0)
           & (np.asarray(a["vtxIsFiducial"]) == 1))
    is_g = (a["showerLArFormerPID"] == 22) & (a["showerRecoE"] > RECO_G_MIN)
    n_g = ak.to_numpy(ak.sum(is_g, axis=1))
    is_mu = ((a["trackLArFormerPID"] == 13) & (a["trackIsSecondary"] == 0)
             & (a["trackRecoE"] > MU_KE_MIN))
    reco_is_cc = ak.to_numpy(ak.any(is_mu, axis=1))
    sel = vok & (n_g >= 2) & (reco_is_cc == bool(reco_cc))
    if eq2:
        sel &= (n_g == 2)
    return (np.asarray(a["run"]), np.asarray(a["subrun"]),
            np.asarray(a["event"]), np.nonzero(sel)[0])


def recompute(cascade_dir, cache, run, sub, evt, idx, dead):
    """Per-event nu-slice chi2 under each mask + the saturation-candidate count."""
    m = rse_map(cascade_dir, cache)
    rows = []
    nofile = noslice = 0
    for i in idx:
        p = m.get((int(run[i]), int(sub[i]), int(evt[i])))
        if p is None:
            nofile += 1
            continue
        try:
            with h5py.File(p, "r") as f:
                if "slices" not in f or "observed_pe" not in f.get("flash", {}):
                    noslice += 1
                    continue
                labs = [l.decode() if isinstance(l, bytes) else str(l)
                        for l in f["slices/label"][()]]
                if "nu" not in labs:
                    noslice += 1
                    continue
                pred = f["slices/pred_pe"][()][labs.index("nu")]
                obs = f["flash/observed_pe"][()]
        except Exception:
            noslice += 1
            continue
        if not np.isfinite(obs).any() or obs.sum() <= 0:
            noslice += 1
            continue
        r = {"i": int(i), "obs_pe": float(obs.sum()),
             "none": neyman_masked(pred, obs, dead=()),
             "dead": neyman_masked(pred, obs, dead=dead)}
        _, allc = find_saturated(obs, dead=dead, max_masked=None,
                                 return_all=True)
        r["ncand"] = len(allc)
        for cap in CAPS:
            sat = find_saturated(obs, dead=dead, max_masked=cap)
            key = "cap%s" % ("inf" if cap is None else cap)
            r[key] = neyman_masked(pred, obs,
                                   dead=tuple(sorted(set(dead) | set(sat))))
            r["n" + key] = len(sat)
        rows.append(r)
    print(">>> %d events with a nu-slice chi2 (%d no cascade file, %d no nu "
          "slice/flash)" % (len(rows), nofile, noslice))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ntuple", required=True)
    ap.add_argument("--cascade-dir", required=True)
    ap.add_argument("--plots", default="plots_saturation")
    ap.add_argument("--reco-cc", action="store_true")
    ap.add_argument("--eq2", action="store_true")
    ap.add_argument("--dead", default="15")
    ap.add_argument("--label", default="MC")
    ap.add_argument("--tag", default=None,
                    help="filename tag (defaults to a slug of --label)")
    ap.add_argument("--chi2-cut", type=float, default=1e4)
    args = ap.parse_args()
    os.makedirs(args.plots, exist_ok=True)
    dead = tuple(int(x) for x in args.dead.split(",") if x != "")

    tag = args.tag or re.sub(r"[^a-z0-9]+", "", args.label.lower())

    run, sub, evt, idx = select(args.ntuple, args.reco_cc, args.eq2)
    sabb = "cc" if args.reco_cc else "nc"
    vabb = "eq2" if args.eq2 else "ge2"
    print(">>> %d reco-%s %s events selected of %d"
          % (len(idx), sabb.upper(), vabb, len(run)))

    cache = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "rse_" + os.path.basename(os.path.dirname(
                             args.cascade_dir.rstrip("/")))
                         + "_" + os.path.basename(args.cascade_dir.rstrip("/"))
                         + ".npz")
    rows = recompute(args.cascade_dir, cache, run, sub, evt, idx, dead)
    if not rows:
        print("!!! nothing to plot")
        return

    def col(k):
        return np.array([r[k] for r in rows], np.float64)

    variants = [("no mask (as reconstructed)", col("none"), "0.4"),
                ("dead-mask {15} only", col("dead"), "tab:blue"),
                ("dead + sat, cap 2", col("cap2"), "tab:green"),
                ("dead + sat, cap 4", col("cap4"), "tab:red"),
                ("dead + sat, cap 6", col("cap6"), "tab:purple"),
                ("dead + sat, NO cap", col("capinf"), "tab:orange")]

    # ---- chi2 distributions -------------------------------------------------
    bins = np.linspace(0, 8, 41)
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    for lab, v, c in variants:
        ls = ":" if "NO cap" in lab else "-"
        ax.hist(np.log10(np.clip(v, 1, None)), bins=bins, histtype="step",
                lw=2.0, color=c, ls=ls,
                label="%s  (median %.0f)" % (lab, np.median(v)))
    ax.axvline(np.log10(args.chi2_cut), color="k", ls="--", lw=1,
               label="chi2 = %.0f" % args.chi2_cut)
    ax.set_xlabel(r"$\log_{10}$(flash $\chi^2$), nu slice")
    ax.set_ylabel("events")
    ax.set_title("%s reco-%s %s photons: saturated-PMT mask (n=%d)"
                 % (args.label, sabb.upper(), "exactly 2" if args.eq2
                    else ">=2", len(rows)))
    ax.legend(fontsize=8)
    fig.tight_layout()
    p1 = os.path.join(args.plots, "satmask_chi2_%s_%s_%s.png" % (tag, sabb, vabb))
    fig.savefig(p1, dpi=110)
    print(">>> wrote", p1)

    # ---- how many tubes does the finder want to mask? -----------------------
    nc = col("ncand")
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    mx = int(max(6, nc.max()))
    ax.hist(nc, bins=np.arange(-0.5, mx + 1.5), color="tab:red", alpha=0.75)
    ax.set_yscale("log")
    ax.set_xlabel("saturation candidates found per event (before cap)")
    ax.set_ylabel("events")
    ax.set_title("%s reco-%s %s: how many tubes the finder wants to mask"
                 % (args.label, sabb.upper(), vabb))
    fig.tight_layout()
    p2 = os.path.join(args.plots, "satmask_ncand_%s_%s_%s.png" % (tag, sabb, vabb))
    fig.savefig(p2, dpi=110)
    print(">>> wrote", p2)

    # ---- numbers ------------------------------------------------------------
    hi = col("none") > args.chi2_cut
    print("\n== nu-slice chi2 (n=%d), high-chi2 = chi2 > %.0f ==" % (len(rows),
                                                                     args.chi2_cut))
    print("%-28s %10s %10s %10s %12s" % ("variant", "median", "mean",
                                         "frac>cut", "median(hi)"))
    for lab, v, _ in variants:
        print("%-28s %10.1f %10.1f %9.1f%% %12.1f"
              % (lab, np.median(v), v.mean(), 100 * (v > args.chi2_cut).mean(),
                 np.median(v[hi]) if hi.any() else np.nan))
    print("\nsaturation candidates/event: mean %.2f, median %.0f, max %d"
          % (nc.mean(), np.median(nc), int(nc.max())))
    for k in range(0, int(nc.max()) + 1):
        f = (nc == k).mean()
        if f > 0:
            print("   %d candidate(s): %5.1f%% of events" % (k, 100 * f))
    print("\nof the %d events above chi2=%.0f before masking, %d (%.0f%%) fall "
          "below it with dead+sat cap 4"
          % (hi.sum(), args.chi2_cut, int((col("cap4")[hi] <= args.chi2_cut).sum()),
             100 * (col("cap4")[hi] <= args.chi2_cut).mean() if hi.any() else 0))


if __name__ == "__main__":
    main()
