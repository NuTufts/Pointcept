"""Meta-parameter sweep for ElPiGraph on the dev track instances.

Grids NumNodes-density / Lambda / Mu / TrimmingRadius over every track instance
in the dev events and tabulates, per config: median residual RMS (cloud and
truth-cloud), median endpoint error, mean #segments, and median runtime. Use it
to pick sensible defaults before wiring ElPiGraph into the §8 package.

Run INSIDE the container:
    ./run_in_local_pointcept_container.sh python \
        lartpc/larformer_reco/studies/sweep_elpigraph.py
"""
import os
import sys
sys.path.insert(0, __import__("os").path.abspath(__import__("os").path.join(
    __import__("os").path.dirname(__import__("os").path.abspath(__file__)), "..", "..", "..")))
import glob
import time
import argparse
import statistics as st
from itertools import product

import numpy as np

from lartpc.larformer_reco.trajfit import trajfit_io as tio
from lartpc.larformer_reco.trajfit.run_elpigraph import fit_elpigraph, evaluate


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    here = os.path.dirname(os.path.abspath(__file__))
    dev = os.path.join(here, "..", "reco_dev_data")
    ap.add_argument("--keypoint2-dir", default=os.path.join(dev, "keypoint2_out"))
    ap.add_argument("--merged-sp-dir", default=os.path.join(dev, "merged_sp"))
    ap.add_argument("--min-points", type=int, default=20)
    ap.add_argument("--gpu", action="store_true")
    # grids (comma lists)
    ap.add_argument("--node-cm", default="1,2,4",
                    help="cm-per-node densities (track extent / this -> NumNodes)")
    ap.add_argument("--Mu", default="0.05,0.1,0.3")
    ap.add_argument("--Lambda", default="0.001,0.01,0.1")
    ap.add_argument("--trim", default="inf",
                    help="TrimmingRadius cm values, e.g. 'inf,2,1'")
    args = ap.parse_args()

    node_cm = [float(x) for x in args.node_cm.split(",")]
    mus = [float(x) for x in args.Mu.split(",")]
    lams = [float(x) for x in args.Lambda.split(",")]
    trims = [float(x) for x in args.trim.split(",")]

    # load all track instances once
    recs = []
    for fp in sorted(glob.glob(os.path.join(args.keypoint2_dir, "*.h5"))):
        recs += tio.load_instances(fp, args.merged_sp_dir, tracks_only=True,
                                   min_points=args.min_points)
    print(f">>> {len(recs)} track instances | gpu={args.gpu}")

    # warmup
    try:
        fit_elpigraph(np.random.RandomState(0).randn(40, 3), num_nodes=6,
                      use_gpu=args.gpu)
    except Exception:
        pass

    print(f"\n{'node_cm':>7} {'Mu':>6} {'Lambda':>7} {'trim':>5} | "
          f"{'RMS':>5} {'RMSt':>5} {'startE':>6} {'endE':>6} {'seg':>4} "
          f"{'t_ms':>6} {'fail':>4}")
    best = None
    for ncm, mu, lam, tr in product(node_cm, mus, lams, trims):
        rms, rmst, se, ee, segs, ts, fail = [], [], [], [], [], [], 0
        for rec in recs:
            nn = int(np.clip(np.ptp(rec.points, 0).max() / ncm, 5, 80))
            w = rec.weights
            try:
                poly, info = fit_elpigraph(
                    rec.points, weights=w, num_nodes=nn, Lambda=lam, Mu=mu,
                    trimming_radius=tr, use_gpu=args.gpu)
            except Exception:
                fail += 1
                continue
            m = evaluate(rec, poly)
            rms.append(m["residual_rms_cm"])
            if "truthcloud_rms_cm" in m:
                rmst.append(m["truthcloud_rms_cm"])
            if "gt_start_err_cm" in m:
                se.append(m["gt_start_err_cm"])
            if "gt_end_err_cm" in m:
                ee.append(m["gt_end_err_cm"])
            segs.append(info["n_seg"])
            ts.append(info["runtime_s"])
        if not rms:
            continue
        med = lambda x: st.median(x) if x else float("nan")
        row = (ncm, mu, lam, tr, med(rms), med(rmst), med(se), med(ee),
               st.mean(segs), med(ts) * 1e3, fail)
        print(f"{ncm:7.1f} {mu:6.2f} {lam:7.3f} {tr:5.1f} | "
              f"{row[4]:5.2f} {row[5]:5.2f} {row[6]:6.2f} {row[7]:6.2f} "
              f"{row[8]:4.1f} {row[9]:6.1f} {fail:4d}")
        score = row[4] + 0.1 * med(se if se else [0]) + 0.1 * med(ee if ee else [0])
        if best is None or score < best[0]:
            best = (score, row)

    if best:
        _, r = best
        print(f"\n>>> lowest combined RMS+endpoint: node_cm={r[0]} Mu={r[1]} "
              f"Lambda={r[2]} trim={r[3]} (RMS={r[4]:.2f} startE={r[6]:.2f} "
              f"endE={r[7]:.2f})")


if __name__ == "__main__":
    main()
