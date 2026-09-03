"""YZ wire-response uniformity map from reco-CC muon tracks (detector
performance), on SPACE-CHARGE-CORRECTED positions.

Same per-point extraction as elifetime_fit.py (largest muon-classed kp2
instance, comb pixel charge from merged_sp, per-track median
normalization, delta-ray trim). Each point's reco position is corrected
with the MCC9 backward SCE map (lartpc/flashmatch/sce_microboone.py); the
median normalized charge is mapped in (z, y) bins. Outputs per leg: the map,
its RMS non-uniformity (the correction gate), and npz; run for MC and data,
then --ratio to draw data/MC.

    python3 yz_uniformity_map.py --ntuple ... --cascade-dir ... \
        --merged-sp-list ... --tag data --plots plots_s1ep2p8_diag
    python3 yz_uniformity_map.py --ratio plots_s1ep2p8_diag
"""
import argparse, os, sys
import numpy as np, h5py, uproot

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")))
from flash_correction import rse_map  # noqa: E402
from elifetime_fit import comb_charge, MU_CLASS  # noqa: E402
from lartpc.flashmatch.sce_microboone import SCEBackward  # noqa: E402

ZB = np.linspace(0, 1036.8, 27)      # ~40 cm
YB = np.linspace(-116.5, 116.5, 13)  # ~19 cm


def collect(args):
    here = os.path.dirname(os.path.abspath(__file__))
    cache = os.path.join(here, "rse_" + os.path.basename(os.path.dirname(
        args.cascade_dir.rstrip("/"))) + "_" + os.path.basename(args.cascade_dir.rstrip("/")) + ".npz")
    t = uproot.open(args.ntuple)["EventTree"]
    a = t.arrays(["run", "subrun", "event", "foundVertex", "primaryVtxStream", "vtxIsFiducial",
                  "trackLArFormerPID", "trackIsSecondary", "trackRecoE"])
    import awkward as ak
    vtx_ok = ((np.asarray(a["foundVertex"]) == 1) & (np.asarray(a["primaryVtxStream"]) == 0)
              & (np.asarray(a["vtxIsFiducial"]) == 1))
    is_mu = ((a["trackLArFormerPID"] == 13) & (a["trackIsSecondary"] == 0) & (a["trackRecoE"] > args.mu_ke_min))
    idx = np.nonzero(vtx_ok & ak.to_numpy(ak.any(is_mu, axis=1)))[0][:args.max_events]
    m = rse_map(args.cascade_dir, cache)
    msp_by_base = {os.path.basename(l.strip()): l.strip() for l in open(args.merged_sp_list) if l.strip()}
    sce = SCEBackward()
    pts_all, r_all = [], []
    n_used = 0
    for i in idx:
        p = m.get((int(a["run"][i]), int(a["subrun"][i]), int(a["event"][i])))
        if not p:
            continue
        p = p if isinstance(p, str) else p[0]
        try:
            with h5py.File(p, "r") as fk:
                if "particle" not in fk or "slice" not in fk:
                    continue
                coords = fk["slice/coord_cm"][()].astype(np.float32)
                best = None
                for inst in fk["particle"]:
                    g = fk[f"particle/{inst}"]
                    if "class_scores" not in g or int(np.argmax(g["class_scores"][()][:6])) != MU_CLASS:
                        continue
                    pidx = g["point_idx"][()]
                    if best is None or len(pidx) > len(best):
                        best = pidx
                if best is None or len(best) < args.min_points:
                    continue
                src = fk.attrs.get("src_file", "")
                src = src.decode() if isinstance(src, bytes) else str(src)
            msp = msp_by_base.get(os.path.basename(src))
            if not msp:
                continue
            qmap = comb_charge(msp)
            pts = coords[best]
            q = np.array([qmap.get(pts[k].tobytes(), np.nan) for k in range(len(pts))])
            fin = np.isfinite(q) & (q > 0)
            if fin.sum() < args.min_points:
                continue
            q, P = q[fin], pts[fin].astype(np.float64)
            med = np.median(q)
            keep = (q > 0.2 * med) & (q < 5 * med)
            pts_all.append(sce.correct(P[keep]))
            r_all.append(q[keep] / med)
            n_used += 1
        except Exception:
            pass
    P = np.concatenate(pts_all); r = np.concatenate(r_all)
    print(f">>> [{args.tag}] tracks {n_used}, points {len(r)} (SCE-corrected)")
    H = np.full((len(ZB) - 1, len(YB) - 1), np.nan); N = np.zeros_like(H)
    zi = np.clip(np.digitize(P[:, 2], ZB) - 1, 0, len(ZB) - 2)
    yi = np.clip(np.digitize(P[:, 1], YB) - 1, 0, len(YB) - 2)
    for a_ in range(len(ZB) - 1):
        for b_ in range(len(YB) - 1):
            mm = (zi == a_) & (yi == b_)
            N[a_, b_] = mm.sum()
            if mm.sum() >= 200:
                H[a_, b_] = np.median(r[mm])
    H = H / np.nanmedian(H)
    fin = np.isfinite(H)
    print(f">>> [{args.tag}] map: {int(fin.sum())}/{H.size} bins filled | RMS non-uniformity "
          f"{np.nanstd(H)*100:.2f}% | min {np.nanmin(H):.3f} max {np.nanmax(H):.3f}")
    np.savez(os.path.join(args.plots, f"yz_map_{args.tag}.npz"), H=H, N=N, ZB=ZB, YB=YB)
    draw(H, f"YZ uniformity map [{args.tag}] (median norm. charge, SCE-corrected)",
         os.path.join(args.plots, f"yz_map_{args.tag}.png"))


def draw(H, title, out, vlim=(0.85, 1.15), cmap="RdBu_r"):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(11, 3.6))
    im = ax.pcolormesh(ZB, YB, H.T, vmin=vlim[0], vmax=vlim[1], cmap=cmap)
    ax.set(xlabel="z [cm] (SCE-corrected)", ylabel="y [cm]", title=title)
    fig.colorbar(im, ax=ax); fig.tight_layout(); fig.savefig(out, dpi=120); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--ntuple"); ap.add_argument("--cascade-dir"); ap.add_argument("--merged-sp-list")
    ap.add_argument("--tag", default="x")
    ap.add_argument("--mu-ke-min", type=float, default=100.0)
    ap.add_argument("--min-points", type=int, default=150)
    ap.add_argument("--max-events", type=int, default=4000)
    ap.add_argument("--plots", required=True)
    ap.add_argument("--ratio", action="store_true", help="draw data/MC from saved maps")
    args = ap.parse_args()
    os.makedirs(args.plots, exist_ok=True)
    if args.ratio:
        d = np.load(os.path.join(args.plots, "yz_map_data.npz"))["H"]
        m = np.load(os.path.join(args.plots, "yz_map_mc.npz"))["H"]
        R = d / m
        print(f">>> data/MC map RMS {np.nanstd(R)*100:.2f}% | min {np.nanmin(R):.3f} max {np.nanmax(R):.3f}")
        np.savez(os.path.join(args.plots, "yz_map_ratio.npz"), H=R, ZB=ZB, YB=YB)
        draw(R, "YZ uniformity: data / MC", os.path.join(args.plots, "yz_map_ratio.png"))
        return
    collect(args)


if __name__ == "__main__":
    main()
