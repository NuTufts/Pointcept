"""Discriminate SATURATION from additional BAD CHANNELS in the high-chi2 events.

The high-chi2 reco-CC events are each dominated (~93% of chi2) by a single PMT
reading ~0-8 PE where ~1300-1500 PE is predicted. Two hypotheses:

  (a) SATURATION: the brightest (nearest) PMT rails and the flash reco drops it
      -> the affected PMT ID should VARY event-to-event and track argmax(pred).
  (b) BAD CHANNEL: a few extra dead/miscalibrated tubes -> the affected PMT ID
      should CONCENTRATE on a handful of IDs, independent of pred.

Prints the offending-PMT ID histogram, how often it is the argmax(pred) PMT,
and its rank in predicted PE.
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


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--ntuple", required=True)
    ap.add_argument("--cascade-dir", required=True)
    ap.add_argument("--dead-channels", default="15")
    ap.add_argument("--chi2-min", type=float, default=1e5)
    args = ap.parse_args()

    dead = tuple(int(x) for x in args.dead_channels.split(",") if x != "")
    cache = os.path.join(_PI0, "rse_" + os.path.basename(os.path.dirname(
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
    ent = np.nonzero(vok & (n_g == 2) & reco_cc)[0]
    cmap = corrected_chi2_by_rse(args.cascade_dir, run, sub, evt, ent, dead,
                                 cache)
    hi = [int(i) for i in ent if np.isfinite(cmap.get(int(i), np.nan))
          and cmap[int(i)] > args.chi2_min]
    rmap = rse_map(args.cascade_dir, cache)
    live = np.ones(32, bool)
    for d in dead:
        live[d] = False

    ids, is_argmax, pred_rank, obs_at, pred_at = [], [], [], [], []
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
        var = obs + (0.10 * obs) ** 2 + 1.0
        per = np.where(live, (obs - pred) ** 2 / var, 0.0)
        j = int(np.argmax(per))                      # the offending PMT
        ids.append(j)
        pl = np.where(live, pred, -1)
        is_argmax.append(j == int(np.argmax(pl)))
        pred_rank.append(int((pl > pl[j]).sum()) + 1)   # 1 = brightest predicted
        obs_at.append(obs[j]); pred_at.append(pred[j])

    ids = np.array(ids); is_argmax = np.array(is_argmax)
    pred_rank = np.array(pred_rank)
    print(f">>> N={len(ids)} high-chi2 events")
    u, c = np.unique(ids, return_counts=True)
    order = np.argsort(-c)
    print(f"  distinct offending PMT IDs: {len(u)} of 31 live")
    print("  top offenders (id:count): "
          + "  ".join(f"{u[k]}:{c[k]}" for k in order[:8]))
    print(f"  top-1 ID share: {c[order[0]]/len(ids):.0%}  |  "
          f"top-3 share: {c[order[:3]].sum()/len(ids):.0%}")
    print(f"  offending PMT IS the brightest-predicted: {is_argmax.mean():.0%}")
    print(f"  its rank in predicted PE: median={int(np.median(pred_rank))} "
          f"(1 = brightest)  |  rank<=3: {(pred_rank<=3).mean():.0%}")
    print(f"  at that PMT: obs median={np.median(obs_at):.1f} PE, "
          f"pred median={np.median(pred_at):.0f} PE")
    print("\n  => IDs spread + tracks brightest-predicted  == SATURATION")
    print("  => IDs concentrated on a few              == BAD CHANNELS")


if __name__ == "__main__":
    main()
