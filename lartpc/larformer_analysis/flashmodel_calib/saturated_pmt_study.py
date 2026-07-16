"""Diagnose the residual high-flash-chi2 CC events: PMT saturation.

Visual scan of the high-chi2 (log10 chi2 > 5) reco-CC 2-photon MC events shows
the reco and the flash PREDICTION both look right, but one or two PMTs carry a
huge predicted PE (1000-3900) with observed PE == 0 -- while their NEIGHBOURS
match well. A PMT between two lit PMTs cannot physically see zero light, so this
is a readout/reco artifact: the brightest (nearest) PMT saturates and the flash
reco drops it to 0 instead of falling back to the low-gain channel.

That is the same failure mode as the dead PMT: obs=0 with pred large gives a
Neyman term pred^2/eps (e.g. 3900^2 ~ 1.5e7), which alone explains the 1e6-1e7
chi2 tail. This script quantifies it over a selection and tests whether masking
the suspect PMTs collapses the chi2.

"suspect saturated" := live PMT with observed < --obs-max PE but predicted >
--pred-min PE.

    python3 saturated_pmt_study.py --ntuple ... --cascade-dir ... --plots ...
"""
import argparse
import os
import sys

import numpy as np
import h5py
import uproot
import awkward as ak

_PI0 = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    "..", "physics", "pi0mass_peak")
sys.path.insert(0, _PI0)
from flash_correction import rse_map, corrected_chi2_by_rse   # noqa: E402


def neyman(pred, obs, mask, f_sys=0.10, eps=1.0):
    var = obs + (f_sys * obs) ** 2 + eps
    return float(((obs - pred) ** 2 / var)[mask].sum())


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--ntuple", required=True)
    ap.add_argument("--cascade-dir", required=True)
    ap.add_argument("--rse-cache", default=None)
    ap.add_argument("--dead-channels", default="15")
    ap.add_argument("--chi2-min", type=float, default=1e5,
                    help="study events with corrected nu-slice chi2 above this")
    ap.add_argument("--obs-max", type=float, default=1.0)
    ap.add_argument("--pred-min", type=float, default=500.0)
    ap.add_argument("--reco-cc", action="store_true", default=True)
    ap.add_argument("--plots", default=None)
    args = ap.parse_args()

    dead = tuple(int(x) for x in args.dead_channels.split(",") if x != "")
    cache = args.rse_cache or os.path.join(
        _PI0, "rse_" + os.path.basename(os.path.dirname(
            args.cascade_dir.rstrip("/"))) + "_"
        + os.path.basename(args.cascade_dir.rstrip("/")) + ".npz")

    t = uproot.open(args.ntuple)["EventTree"]
    a = t.arrays(["run", "subrun", "event", "foundVertex", "primaryVtxStream",
                  "vtxIsFiducial", "showerLArFormerPID", "showerRecoE",
                  "trackLArFormerPID", "trackIsSecondary", "trackRecoE"])
    run = np.asarray(a["run"]); sub = np.asarray(a["subrun"])
    evt = np.asarray(a["event"])
    vok = ((np.asarray(a["foundVertex"]) == 1)
           & (np.asarray(a["primaryVtxStream"]) == 0)
           & (np.asarray(a["vtxIsFiducial"]) == 1))
    n_g = ak.to_numpy(ak.sum((a["showerLArFormerPID"] == 22)
                             & (a["showerRecoE"] > 20), axis=1))
    reco_cc = ak.to_numpy(ak.any((a["trackLArFormerPID"] == 13)
                                 & (a["trackIsSecondary"] == 0)
                                 & (a["trackRecoE"] > 100), axis=1))
    sel = vok & (n_g == 2) & (reco_cc == bool(args.reco_cc))
    ent = np.nonzero(sel)[0]
    cmap = corrected_chi2_by_rse(args.cascade_dir, run, sub, evt, ent, dead,
                                 cache)
    hi = [int(i) for i in ent
          if np.isfinite(cmap.get(int(i), np.nan))
          and cmap[int(i)] > args.chi2_min]
    print(f">>> reco-{'CC' if args.reco_cc else 'NC'} eq2 with corrected "
          f"chi2 > {args.chi2_min:.0e}: {len(hi)} events")

    rmap = rse_map(args.cascade_dir, cache)
    live = np.ones(32, bool)
    for d in dead:
        live[d] = False
    nsus, tot, mask_chi2, share, p_sus, o_sus, ratio_all, ratio_mask = (
        [] for _ in range(8))
    for i in hi:
        p_ = rmap.get((int(run[i]), int(sub[i]), int(evt[i])))
        if p_ is None:
            continue
        with h5py.File(p_, "r") as f:
            labs = [l.decode() if isinstance(l, bytes) else str(l)
                    for l in f["slices/label"][()]]
            if "nu" not in labs:
                continue
            pred = np.nan_to_num(np.asarray(
                f["slices/pred_pe"][()][labs.index("nu")], float))
            obs = np.nan_to_num(np.asarray(f["flash/observed_pe"][()], float))
        sus = live & (obs < args.obs_max) & (pred > args.pred_min)
        T = neyman(pred, obs, live)
        S = neyman(pred, obs, live & sus)
        M = neyman(pred, obs, live & ~sus)
        nsus.append(int(sus.sum())); tot.append(T); mask_chi2.append(M)
        share.append(S / max(T, 1e-9))
        keep = live & ~sus
        if obs[live].sum() > 0:
            ratio_all.append(pred[live].sum() / obs[live].sum())
        if obs[keep].sum() > 0:
            ratio_mask.append(pred[keep].sum() / obs[keep].sum())
        if sus.any():
            p_sus += list(pred[sus]); o_sus += list(obs[sus])
    nsus = np.array(nsus); tot = np.array(tot); mask_chi2 = np.array(mask_chi2)
    share = np.array(share)
    print(f">>> N={len(tot)} events read")
    print(f"  with >=1 suspect PMT (obs<{args.obs_max:g} PE, pred>"
          f"{args.pred_min:g} PE): {int((nsus>=1).sum())} "
          f"({(nsus>=1).mean():.0%})")
    print(f"  suspect PMTs/event: median={int(np.median(nsus))} max={nsus.max()}")
    print(f"  suspect share of chi2: median={np.median(share):.2f}")
    print(f"  chi2 median: {np.median(tot):.0f}  ->  masking suspects: "
          f"{np.median(mask_chi2):.0f}")
    if p_sus:
        print(f"  suspect PMT predicted PE: median={np.median(p_sus):.0f} | "
              f"their observed PE: max={np.max(o_sus):.2f}")
    print(f"  pred/obs total: all-live median={np.median(ratio_all):.2f} -> "
          f"suspects masked {np.median(ratio_mask):.2f}")

    # ---- what drives the RESIDUAL chi2 after masking obs~0 suspects? --------
    # Neyman uses OBSERVED in the denominator, so a partially-saturated PMT
    # (small-but-nonzero obs, large pred) still blows up: obs=10,pred=1000 ->
    # (990)^2/12 ~ 8e4. Split by has-suspect and report the top-contributing
    # PMT after masking.
    has, hasnt = [], []
    top_obs, top_pred, top_term = [], [], []
    for i in hi:
        p_ = rmap.get((int(run[i]), int(sub[i]), int(evt[i])))
        if p_ is None:
            continue
        with h5py.File(p_, "r") as f:
            labs = [l.decode() if isinstance(l, bytes) else str(l)
                    for l in f["slices/label"][()]]
            if "nu" not in labs:
                continue
            pred = np.nan_to_num(np.asarray(
                f["slices/pred_pe"][()][labs.index("nu")], float))
            obs = np.nan_to_num(np.asarray(f["flash/observed_pe"][()], float))
        sus = live & (obs < args.obs_max) & (pred > args.pred_min)
        keep = live & ~sus
        var = obs + (0.10 * obs) ** 2 + 1.0
        per = (obs - pred) ** 2 / var
        per_keep = np.where(keep, per, 0.0)
        j = int(np.argmax(per_keep))
        (has if sus.any() else hasnt).append(neyman(pred, obs, keep))
        top_obs.append(obs[j]); top_pred.append(pred[j])
        top_term.append(per_keep[j] / max(per_keep.sum(), 1e-9))
    has = np.array(has); hasnt = np.array(hasnt)
    print(f"\n  --- residual after masking obs~0 suspects ---")
    print(f"  events WITH a saturated PMT  (N={len(has)}): chi2 median="
          f"{np.median(has):.0f}")
    print(f"  events WITHOUT one           (N={len(hasnt)}): chi2 median="
          f"{np.median(hasnt):.0f}")
    print(f"  top residual PMT: obs median={np.median(top_obs):.1f} PE, "
          f"pred median={np.median(top_pred):.0f} PE, "
          f"pred/obs median={np.median(np.array(top_pred)/np.maximum(top_obs,0.5)):.1f}")
    print(f"  that single PMT is median {np.median(top_term):.0%} of the "
          "remaining chi2")

    if args.plots:
        os.makedirs(args.plots, exist_ok=True)
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.5))
        lb = np.linspace(0, 8, 41)
        lg = lambda x: np.clip(np.log10(np.clip(x, 1, None)), 0, 8)
        ax[0].hist(lg(tot), bins=lb, histtype="step", lw=2, color="0.45",
                   ls="--", label=f"as-is  med {np.median(tot):.0f}")
        ax[0].hist(lg(mask_chi2), bins=lb, histtype="step", lw=2,
                   color="#d62728",
                   label=f"suspects masked  med {np.median(mask_chi2):.0f}")
        ax[0].set(xlabel=r"$\log_{10}$ nu-slice flash $\chi^2$", ylabel="events",
                  title=f"high-chi2 reco-CC eq2 (N={len(tot)}):\n"
                        "masking obs=0 / pred-large PMTs")
        ax[0].legend(fontsize=9); ax[0].grid(alpha=0.3)
        ax[1].hist(np.clip(p_sus, 0, 5000), bins=30, color="#d62728",
                   alpha=0.85)
        ax[1].set(xlabel="predicted PE on the suspect (obs=0) PMT",
                  ylabel="PMTs", title="suspect PMTs: predicted PE\n"
                                       "(observed is ~0 on all of them)")
        ax[1].grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(f"{args.plots}/saturated_pmt_study.png", dpi=110)
        plt.close(fig)
        print(f">>> plots -> {args.plots}/saturated_pmt_study.png")


if __name__ == "__main__":
    main()
