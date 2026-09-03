"""In-situ electron-lifetime fit from reco-CC muon tracks (plan step 2).

For reco-CC events (FV vertex, primary LArFormerPID==13 track with
KE > --mu-ke-min), takes the largest muon-classed kp2 instance (>=
--min-points points), reads each point's comb pixel charge from the parent
merged_sp (Y plane, else mean(U,V); exact-coordinate row match), normalizes
per track by the track's own median charge (removes angle/energy
dependence), and accumulates (x_drift, q/q_track_median) pairs. The median
ratio per drift bin is fit with ln(r) = c - x/(v*tau):

    slope -> tau (v = 0.1098 cm/us) and the full-drift (256.35 cm)
    attenuation, per leg (MC overlay and beam data separately, so the MC
    fit captures whatever lifetime the simulation actually used).

GATE (plan): if the fitted full-drift attenuation is < ~2-3% in a leg, the
lifetime correction cannot improve shower resolution there — stop.

    PYTHONPATH=./ python3 elifetime_fit.py \
        --ntuple <ntuple.root> --cascade-dir <keypoint2_streams> \
        --merged-sp-list <msp list> --tag mc --plots plots_s1ep2p8_diag
"""
import argparse
import os
import sys

import numpy as np
import h5py
import uproot

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from flash_correction import rse_map  # noqa: E402

V_DRIFT = 0.1098          # cm/us @ 273 V/cm
X_FULL = 256.35
MU_CLASS = 2              # kp2 class_scores argmax: e,gamma,mu,...


def comb_charge(msp_path):
    with h5py.File(msp_path, "r") as f:
        td = f["entry_0/triplet_data"]
        pos = td["pos"][()].astype(np.float32)
        pix = td["pixval"][()].astype(np.float64)
    q = pix[:, 2].copy()
    bad = q <= 0
    q[bad] = 0.5 * (pix[bad, 0] + pix[bad, 1])
    return {pos[i].tobytes(): float(q[i]) for i in range(len(pos))}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--ntuple", required=True)
    ap.add_argument("--cascade-dir", required=True)
    ap.add_argument("--merged-sp-list", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--mu-ke-min", type=float, default=100.0)
    ap.add_argument("--min-points", type=int, default=150)
    ap.add_argument("--max-events", type=int, default=4000)
    ap.add_argument("--plots", required=True)
    args = ap.parse_args()
    os.makedirs(args.plots, exist_ok=True)
    here = os.path.dirname(os.path.abspath(__file__))
    cache = os.path.join(here, "rse_" + os.path.basename(os.path.dirname(
        args.cascade_dir.rstrip("/"))) + "_" + os.path.basename(
        args.cascade_dir.rstrip("/")) + ".npz")

    t = uproot.open(args.ntuple)["EventTree"]
    a = t.arrays(["run", "subrun", "event", "foundVertex",
                  "primaryVtxStream", "vtxIsFiducial",
                  "trackLArFormerPID", "trackIsSecondary", "trackRecoE"])
    import awkward as ak
    vtx_ok = ((np.asarray(a["foundVertex"]) == 1)
              & (np.asarray(a["primaryVtxStream"]) == 0)
              & (np.asarray(a["vtxIsFiducial"]) == 1))
    is_mu = ((a["trackLArFormerPID"] == 13) & (a["trackIsSecondary"] == 0)
             & (a["trackRecoE"] > args.mu_ke_min))
    cc = vtx_ok & ak.to_numpy(ak.any(is_mu, axis=1))
    idx = np.nonzero(cc)[0][:args.max_events]
    print(f">>> [{args.tag}] {int(cc.sum())} reco-CC; using {len(idx)}")

    m = rse_map(args.cascade_dir, cache)
    msp_by_base = {os.path.basename(l.strip()): l.strip()
                   for l in open(args.merged_sp_list) if l.strip()}
    xs, rs = [], []
    n_used = n_nokp = n_nomu = n_nomsp = 0
    for i in idx:
        p = m.get((int(a["run"][i]), int(a["subrun"][i]),
                   int(a["event"][i])))
        if not p:
            n_nokp += 1
            continue
        p = p if isinstance(p, str) else p[0]
        try:
            with h5py.File(p, "r") as fk:
                if "particle" not in fk or "slice" not in fk:
                    n_nomu += 1
                    continue
                coords = fk["slice/coord_cm"][()].astype(np.float32)
                best = None
                for inst in fk["particle"]:
                    g = fk[f"particle/{inst}"]
                    if "class_scores" not in g or "point_idx" not in g:
                        continue
                    if int(np.argmax(g["class_scores"][()][:6])) != MU_CLASS:
                        continue
                    pidx = g["point_idx"][()]
                    if best is None or len(pidx) > len(best):
                        best = pidx
                if best is None or len(best) < args.min_points:
                    n_nomu += 1
                    continue
                src = fk.attrs.get("src_file", "")
                src = src.decode() if isinstance(src, bytes) else str(src)
            msp = msp_by_base.get(os.path.basename(src))
            if not msp:
                n_nomsp += 1
                continue
            qmap = comb_charge(msp)
            pts = coords[best]
            q = np.array([qmap.get(pts[k].tobytes(), np.nan)
                          for k in range(len(pts))])
            fin = np.isfinite(q) & (q > 0)
            if fin.sum() < args.min_points:
                n_nomu += 1
                continue
            q, x = q[fin], pts[fin, 0].astype(np.float64)
            med = np.median(q)
            keep = (q > 0.2 * med) & (q < 5 * med)   # trim deltas/junk
            xs.append(x[keep])
            rs.append(q[keep] / med)
            n_used += 1
        except Exception:
            n_nokp += 1
    print(f">>> [{args.tag}] tracks used {n_used} | no-kp2 {n_nokp} | "
          f"no-mu-inst {n_nomu} | no-msp {n_nomsp}")
    x = np.concatenate(xs)
    r = np.concatenate(rs)
    print(f">>> [{args.tag}] {len(x)} points")

    edges = np.linspace(5, 250, 17)
    ctr, medr, err = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mm = (x >= lo) & (x < hi)
        if mm.sum() < 500:
            continue
        v = r[mm]
        ctr.append(0.5 * (lo + hi))
        medr.append(np.median(v))
        err.append(1.253 * v.std() / np.sqrt(len(v)))   # se of median
    ctr, medr, err = map(np.asarray, (ctr, medr, err))
    w = medr / err
    s, c = np.polyfit(ctr, np.log(medr), 1, w=w)
    ds = np.sqrt(1.0 / np.sum((w * (ctr - ctr.mean())) ** 2)) \
        if len(ctr) > 2 else np.nan
    att = 1.0 - np.exp(s * X_FULL)
    tau_us = -1.0 / (V_DRIFT * s) if s < 0 else np.inf
    print(f"\n== [{args.tag}] lifetime fit ==")
    for cc_, mr, er in zip(ctr, medr, err):
        print(f"  x={cc_:6.1f} cm  median r={mr:.4f} +- {er:.4f}")
    print(f"  slope {s:+.3e} /cm (stat ~{ds:.1e})")
    print(f"  full-drift (256 cm) attenuation: {att*100:+.2f}%")
    print(f"  implied tau: "
          f"{'inf/negative (no attenuation)' if s >= 0 else f'{tau_us/1000.0:.1f} ms'}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    ax.errorbar(ctr, medr, yerr=err, fmt="ko", ms=4)
    xx = np.linspace(0, 256, 50)
    ax.plot(xx, np.exp(c + s * xx), "r-", lw=1.5,
            label=f"fit: att(256cm)={att*100:+.1f}%")
    ax.set(xlabel="drift coordinate x [cm]",
           ylabel="median q / track-median q",
           title=f"muon dQ point charge vs drift [{args.tag}]")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(args.plots, f"elifetime_{args.tag}.png"),
                dpi=120)
    np.savez(os.path.join(args.plots, f"elifetime_{args.tag}.npz"),
             ctr=ctr, medr=medr, err=err, slope=s, att=att)
    print(f">>> elifetime_{args.tag}.png / .npz")


if __name__ == "__main__":
    main()
