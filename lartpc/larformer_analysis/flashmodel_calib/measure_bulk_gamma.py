"""Bulk in-time light-yield ratio r = median(sum_live obs_pe / sum_live pred_pe).

The clean stopping-muon estimator (fit_gamma_run.py) is ~0.1% efficient -- a
12.8k-event pilot yields ~13 muons, far too few to set a light-yield constant.
This measures the SAME quantity over every event with a nu-candidate slice and
an in-time flash (~10^3x more statistics), which is what the flash-chi2 actually
integrates over.

USE IT DIFFERENTIALLY. The bulk-slice population is not where gamma_beam=5.25
was tuned: run-3 MC at gamma_scale=1.0 gives r ~ 0.809, not 1.0
(SLICER_RETRAIN_PLAN 2026-08-31). So the run-1 light-yield scale is

    GAMMA_run1 = GAMMA_ref * r(run1 sample) / r(run3 reference sample)

with both arms produced at the SAME --gamma-run-scale. The absolute offset
cancels; only the run-to-run difference survives.

Dead PMTs are resolved PER RUN (dead_channels.dead_opdets_for_run): opdet 15 is
dead in run 3 but LIVE in run 1 -- hardcoding (15,) would silently drop a live
tube from both sums.

    PYTHONPATH=<repo> python3 measure_bulk_gamma.py \
        --ntuple <ntuple.root> --cascade-dir <keypoint2_streams> \
        --sample-tag run1_ovl_g100 [--out r_run1_ovl.npz]
"""
import argparse
import os
import sys

import numpy as np
import uproot
import awkward as ak
import h5py

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "physics", "pi0mass_peak"))
from flash_correction import rse_map                                # noqa: E402
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..", "..")))
from lartpc.flashmatch.dead_channels import dead_opdets_for_run     # noqa: E402
from lartpc.flashmatch.saturation import find_saturated             # noqa: E402


def bulk_ratio(ntuple, cascade_dir, mu_ke_min=50.0, min_pred_pe=50.0):
    """Per-event (obs_sum/pred_sum) over live PMTs of the nu-candidate slice."""
    t = uproot.open(ntuple)["EventTree"]
    br = ["run", "subrun", "event", "foundVertex", "primaryVtxStream",
          "vtxIsFiducial", "trackLArFormerPID", "trackIsSecondary",
          "trackRecoE"]
    a = t.arrays([b for b in br if b in set(t.keys())])
    vok = ((np.asarray(a["foundVertex"]) == 1)
           & (np.asarray(a["primaryVtxStream"]) == 0)
           & (np.asarray(a["vtxIsFiducial"]) == 1))
    is_mu = ((a["trackLArFormerPID"] == 13) & (a["trackIsSecondary"] == 0)
             & (a["trackRecoE"] > mu_ke_min))
    reco_cc = ak.to_numpy(ak.any(is_mu, axis=1))
    run = np.asarray(a["run"]); sub = np.asarray(a["subrun"])
    evt = np.asarray(a["event"])

    cache = os.path.join(_HERE, "rse_" + os.path.basename(os.path.dirname(
        cascade_dir.rstrip("/"))) + "_" + os.path.basename(
        cascade_dir.rstrip("/")) + ".npz")
    m = rse_map(cascade_dir, cache)

    ratio, is_cc, runs = [], [], []
    n_nofile = n_noslice = n_lowpred = 0
    for i in np.nonzero(vok)[0]:
        p = m.get((int(run[i]), int(sub[i]), int(evt[i])))
        if not p:
            n_nofile += 1
            continue
        try:
            with h5py.File(p if isinstance(p, str) else p[0], "r") as f:
                if "slices" not in f or "flash" not in f:
                    n_noslice += 1
                    continue
                labs = [l.decode() if isinstance(l, bytes) else str(l)
                        for l in f["slices/label"][()]]
                if "nu" not in labs or "observed_pe" not in f["flash"]:
                    n_noslice += 1
                    continue
                j = labs.index("nu")
                obs = np.clip(f["flash/observed_pe"][()], 0, None)
                pred = np.clip(f["slices/pred_pe"][()][j], 0, None)
        except Exception:
            n_nofile += 1
            continue
        dead = dead_opdets_for_run(int(run[i]))          # run-aware, NOT (15,)
        live = np.ones(32, bool)
        if dead:
            live[list(dead)] = False
        try:
            sat = find_saturated(obs, dead=dead)
            if len(sat):
                live[list(sat)] = False
        except Exception:
            pass
        ps, os_ = float(pred[live].sum()), float(obs[live].sum())
        if ps < min_pred_pe or os_ <= 0:
            n_lowpred += 1
            continue
        ratio.append(os_ / ps); is_cc.append(bool(reco_cc[i]))
        runs.append(int(run[i]))
    return (np.array(ratio), np.array(is_cc, bool), np.array(runs),
            dict(vtx=int(vok.sum()), nofile=n_nofile, noslice=n_noslice,
                 lowpred=n_lowpred))


def summarize(r, tag, nboot=2000, seed=7):
    if not len(r):
        raise SystemExit(f"{tag}: no events survived")
    med = float(np.median(r))
    lo, hi = np.percentile(r, [25, 75])
    rng = np.random.default_rng(seed)
    boot = np.array([np.median(rng.choice(r, len(r), replace=True))
                     for _ in range(nboot)])
    err = float(boot.std())
    print(f"  {tag:>28}: N={len(r):6d} | median {med:.4f} +- {err:.4f} "
          f"(boot) | IQR {lo:.3f}-{hi:.3f}")
    return med, err


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--ntuple", required=True)
    ap.add_argument("--cascade-dir", required=True)
    ap.add_argument("--sample-tag", required=True)
    ap.add_argument("--mu-ke-min", type=float, default=50.0)
    ap.add_argument("--min-pred-pe", type=float, default=50.0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    r, cc, runs, stats = bulk_ratio(args.ntuple, args.cascade_dir,
                                    args.mu_ke_min, args.min_pred_pe)
    print(f"\n>>> {args.sample_tag}: nu-stream FV vertices {stats['vtx']} | "
          f"no cascade file {stats['nofile']} | no nu slice/flash "
          f"{stats['noslice']} | pred<{args.min_pred_pe:g} {stats['lowpred']}")
    print(f">>> runs {runs.min()}-{runs.max()} "
          f"(period-resolved dead PMTs applied per event)")
    print("== bulk in-time obs/pred ==")
    med_all, err_all = summarize(r, "all nu-slice events")
    if cc.any():
        summarize(r[cc], "reco-CC subset")
    if (~cc).any():
        summarize(r[~cc], "non-reco-CC subset")
    if args.out:
        np.savez(args.out, ratio=r, reco_cc=cc, runs=runs,
                 median=med_all, boot_err=err_all,
                 sample_tag=args.sample_tag, **{k: v for k, v in stats.items()})
        print(f">>> {args.out}")


if __name__ == "__main__":
    main()
