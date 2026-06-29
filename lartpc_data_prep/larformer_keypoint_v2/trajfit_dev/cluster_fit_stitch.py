"""Cluster -> per-fragment sliding-PCA fit -> direction-scored stitch.

An alternative to letting ElPiGraph bridge detector gaps with its elastic energy:
explicitly split the instance cloud into continuous fragments (DBSCAN), fit each
gap-free fragment with a sliding-window PCA centerline (brief Method A), then
stitch fragments into one ordered track by linking endpoints scored on
gap-length + end-tangent collinearity + (stub) dead-region consistency.

Why: a single geometric radius can't separate "sparse-but-continuous" track from
"real dead-region gap" (a clean muon shatters into ~20 pieces at eps=0.6 cm). So
cluster at a forgiving eps, accept a few fragments, and make the bridge decision
explicit in the stitch step -- which is exactly where the known dead-channel map
belongs (see `dead_region_fraction`, currently a stub).

Run INSIDE the container (head-to-head vs ElPiGraph + HTML overlays):
    ./run_in_local_pointcept_container.sh python \
        lartpc_data_prep/larformer_keypoint_v2/trajfit_dev/cluster_fit_stitch.py
"""
import os
import glob
import time
import argparse

import numpy as np
from sklearn.cluster import DBSCAN

import trajfit_io as tio
from run_elpigraph import point_to_polyline_distance, fit_elpigraph
from mcs_rdp import rdp_variable, make_mcs_tolerance, beta_from_p, path_length


# ---------------------------------------------------------------------------
# dead-region hook (STUB) -- the one place detector knowledge enters the stitch
# ---------------------------------------------------------------------------
def dead_region_fraction(p0, p1, dead_map=None, n_samp=16):
    """Fraction of the straight segment p0->p1 lying in known-dead detector regions.

    STUB: returns 0.0 until a real dead-channel / wire-status map is supplied. A
    real version would sample points along the segment, project each to the three
    MicroBooNE wireplanes (u,v,y), and test against per-plane dead-wire lists (or
    a precomputed 3D dead-region mask). A return > 0.5 authorises a long bridge
    across the gap; ~0 means the gap is in live detector and a long jump should be
    penalised (more likely a true endpoint or a different particle).
    """
    if dead_map is None:
        return 0.0
    ts = np.linspace(0.0, 1.0, n_samp)[:, None]
    pts = (1 - ts) * np.asarray(p0) + ts * np.asarray(p1)
    return float(np.mean([dead_map(p) for p in pts]))


# ---------------------------------------------------------------------------
# Method A: sliding-window PCA centerline for ONE gap-free fragment
# ---------------------------------------------------------------------------
def sliding_pca_centerline(P, W=None, window=2.0, step=1.0, min_in_window=2):
    """Ordered, charge-weighted centerline of a continuous point set (brief §4.2).

    Global-PCA arc-length ordering + windowed charge-weighted centroids. Returns
    (M,3) ordered points. Short fragments collapse to their two extreme points.
    (Known limitation: PCA ordering scrambles on hairpins -- a kNN/MST geodesic
    ordering is the documented fallback; out of scope for this prototype.)
    """
    P = np.asarray(P, np.float64)
    n = len(P)
    if n <= 2:
        return P.copy()
    c = P.mean(0)
    _, _, Vt = np.linalg.svd(P - c, full_matrices=False)
    s = (P - c) @ Vt[0]
    order = np.argsort(s)
    s_s, P_s = s[order], P[order]
    W_s = (np.asarray(W, np.float64)[order] if W is not None else None)
    extent = float(s_s[-1] - s_s[0])
    if extent < window or n < 5:
        return np.array([P_s[0], P_s[-1]])
    npos = max(int(np.ceil(extent / step)) + 1, 2)
    out = []
    for sp in np.linspace(s_s[0], s_s[-1], npos):
        m = (s_s >= sp - window / 2) & (s_s <= sp + window / 2)
        if m.sum() < min_in_window:
            idx = np.argsort(np.abs(s_s - sp))[:max(min_in_window, 3)]
            m = np.zeros(n, bool)
            m[idx] = True
        pts = P_s[m]
        w = W_s[m] if W_s is not None else None
        ctr = ((pts * w[:, None]).sum(0) / w.sum()
               if (w is not None and w.sum() > 0) else pts.mean(0))
        out.append(ctr)
    out = np.asarray(out)
    keep = [out[0]]
    for ct in out[1:]:
        if np.linalg.norm(ct - keep[-1]) > 1e-6:
            keep.append(ct)
    return np.asarray(keep)


def _tangent(cl, at_start, k=3):
    """Outward unit tangent at an end of a centerline (points away from the body)."""
    if len(cl) < 2:
        return np.zeros(3)
    j = min(k, len(cl) - 1)
    v = (cl[0] - cl[j]) if at_start else (cl[-1] - cl[-1 - j])
    nrm = np.linalg.norm(v)
    return v / nrm if nrm > 1e-9 else np.zeros(3)


def fit_fragments(points, weights=None, eps=1.2, min_samples=3, min_frag_pts=3,
                  window=2.0, step=1.0):
    """DBSCAN the cloud, fit each cluster's centerline. Returns (frags, labels)."""
    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(points)
    frags = []
    for c in sorted(set(labels[labels >= 0])):
        m = labels == c
        if m.sum() < min_frag_pts:
            continue
        P = points[m]
        Wf = weights[m] if weights is not None else None
        cl = sliding_pca_centerline(P, Wf, window=window, step=step)
        frags.append(dict(label=int(c), points=P, centerline=cl, n=int(m.sum()),
                          A=cl[0], B=cl[-1],
                          tA=_tangent(cl, True), tB=_tangent(cl, False)))
    return frags, labels


# ---------------------------------------------------------------------------
# stitch: greedy agglomerative linking of fragment endpoints
# ---------------------------------------------------------------------------
def _reverse(ch):
    ch["seq"] = [(fi, not fl) for (fi, fl) in reversed(ch["seq"])]
    ch["term"] = [ch["term"][1], ch["term"][0]]


def stitch_fragments(frags, dead_map=None, max_gap_live=3.0, max_gap_dead=40.0,
                     min_dir=0.3, short_n=8):
    """Link fragments into chains. Returns list of chains (dict with `seq` =
    ordered [(frag_idx, flipped)]). The longest chain is the main track."""
    # each chain starts as one fragment; term[0] = centerline start, term[1] = end
    chains = []
    for i, f in enumerate(frags):
        chains.append(dict(
            seq=[(i, False)],
            term=[dict(pos=f["A"], tan=f["tA"], n=f["n"]),
                  dict(pos=f["B"], tan=f["tB"], n=f["n"])]))

    def candidate(a, b):
        """Best admissible link between chains a,b: (cost, ti, tj) or None."""
        best = None
        for ti in (0, 1):
            for tj in (0, 1):
                ta, tb = chains[a]["term"][ti], chains[b]["term"][tj]
                gap = float(np.linalg.norm(tb["pos"] - ta["pos"]))
                if gap < 1e-6:
                    continue
                d = (tb["pos"] - ta["pos"]) / gap
                dscore = 0.5 * (float(ta["tan"] @ d) - float(tb["tan"] @ d))
                fdead = dead_region_fraction(ta["pos"], tb["pos"], dead_map)
                ok_gap = (gap <= max_gap_dead) if fdead > 0.5 else (gap <= max_gap_live)
                is_short = (ta["n"] <= short_n) or (tb["n"] <= short_n)
                ok_dir = is_short or (dscore >= min_dir)
                if ok_gap and ok_dir:
                    cost = gap * (1.5 - max(dscore, 0.0))
                    if best is None or cost < best[0]:
                        best = (cost, ti, tj)
        return best

    while len(chains) > 1:
        best = None
        for a in range(len(chains)):
            for b in range(a + 1, len(chains)):
                cand = candidate(a, b)
                if cand and (best is None or cand[0] < best[0]):
                    best = (cand[0], a, b, cand[1], cand[2])
        if best is None:
            break
        _, a, b, ti, tj = best
        ca, cb = chains[a], chains[b]
        if ti == 0:
            _reverse(ca)            # junction at ca.term[1]
        if tj == 1:
            _reverse(cb)            # junction at cb.term[0]
        merged = dict(seq=ca["seq"] + cb["seq"],
                      term=[ca["term"][0], cb["term"][1]])
        chains = [c for k, c in enumerate(chains) if k not in (a, b)] + [merged]
    return chains


def assemble(frags, chain):
    parts = []
    for fi, flip in chain["seq"]:
        cl = frags[fi]["centerline"]
        parts.append(cl[::-1] if flip else cl)
    return np.vstack(parts).astype(np.float32)


def extend_to_extremes(poly, points):
    """Push each global end outward along its terminal direction to the farthest
    projected cloud point -- recovers the range lost to centerline end-inset."""
    poly = poly.astype(np.float64).copy()
    if len(poly) < 2:
        return poly.astype(np.float32)
    for end, nbr in ((0, 1), (-1, -2)):
        d = poly[end] - poly[nbr]
        L = np.linalg.norm(d)
        if L < 1e-9:
            continue
        d /= L
        mx = float(((points - poly[end]) @ d).max())
        if mx > 0:
            poly[end] = poly[end] + d * mx
    return poly.astype(np.float32)


def cluster_fit_stitch(points, weights=None, eps=1.2, min_samples=3,
                       window=2.0, step=1.0, dead_map=None, extend=True,
                       rdp_tol=None, **stitch_kw):
    """Full pipeline. Returns dict with the main-track polyline + diagnostics.

    Stages: DBSCAN -> per-fragment sliding-PCA centerline -> stitch -> (optional
    Method-B RDP smoothing if `rdp_tol` callable given) -> (optional endpoint
    extension). `centerline_raw` is the pre-RDP, pre-extension stitched centerline
    (its length over-counts range via MCS wiggle); `polyline` is the final."""
    t0 = time.perf_counter()
    frags, labels = fit_fragments(points, weights, eps=eps,
                                  min_samples=min_samples, window=window, step=step)
    if not frags:
        return dict(polyline=points[:1], centerline_raw=points[:1], frags=[],
                    kink_vertices=np.zeros((0, 3), np.float32),
                    labels=labels, n_frag=0, n_chain=0, main_frac=0.0,
                    main_mask=np.zeros(len(points), bool),
                    runtime_s=time.perf_counter() - t0, n_seg=0)
    chains = stitch_fragments(frags, dead_map=dead_map, **stitch_kw)
    polys = [assemble(frags, ch) for ch in chains]
    npts = [sum(frags[fi]["n"] for fi, _ in ch["seq"]) for ch in chains]
    main = int(np.argmax(npts))
    center_raw = polys[main]
    main_labels = [frags[fi]["label"] for fi, _ in chains[main]["seq"]]
    main_mask = np.isin(labels, main_labels)

    # The TRAJECTORY is the DENSE sliding-PCA centerline -- it hugs the points
    # (range, local direction, residual). We do NOT RDP it down: RDP is a
    # corner-cutter and would chord across continuous (cumulative-MCS) curvature.
    poly = center_raw
    if extend:                                  # extend only to the MAIN chain's
        poly = extend_to_extremes(poly, points[main_mask])   # own points
    # MCS-RDP runs ONLY as a discrete-kink finder -> candidate hard-scatter
    # vertices, reported separately (NOT the trajectory). Interior vertices are
    # the kinks; endpoints are dropped from the kink list.
    kink_vertices = (rdp_variable(center_raw, rdp_tol)[1:-1]
                     if rdp_tol is not None else np.zeros((0, 3), np.float32))
    dt = time.perf_counter() - t0
    return dict(polyline=poly, centerline_raw=center_raw,
                kink_vertices=np.asarray(kink_vertices, np.float32),
                frags=frags, labels=labels, chains=chains, chain_polys=polys,
                chain_npts=npts, n_frag=len(frags), n_chain=len(chains),
                main_labels=main_labels, main_mask=main_mask,
                main_frac=npts[main] / len(points),
                runtime_s=dt, n_seg=max(len(poly) - 1, 0))


# ---------------------------------------------------------------------------
# viz + comparison driver
# ---------------------------------------------------------------------------
_PALETTE = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b",
            "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]


def write_html(rec, cfs, elpi_poly, metrics, out_path):
    import plotly.graph_objects as go
    P, lab = rec.points, cfs["labels"]
    fig = go.Figure()
    for c in sorted(set(lab)):
        m = lab == c
        col = "lightgray" if c < 0 else _PALETTE[c % len(_PALETTE)]
        nm = "noise" if c < 0 else f"frag{c} (n={m.sum()})"
        fig.add_trace(go.Scatter3d(
            x=P[m, 0], y=P[m, 1], z=P[m, 2], mode="markers",
            marker=dict(size=2, color=col, opacity=0.55), name=nm))
    sp = cfs["polyline"]
    fig.add_trace(go.Scatter3d(
        x=sp[:, 0], y=sp[:, 1], z=sp[:, 2], mode="lines",
        line=dict(color="green", width=5), name="centerline (trajectory)"))
    kv = cfs.get("kink_vertices")
    if kv is not None and len(kv):
        fig.add_trace(go.Scatter3d(
            x=kv[:, 0], y=kv[:, 1], z=kv[:, 2], mode="markers",
            marker=dict(size=8, color="red", symbol="diamond",
                        line=dict(width=1, color="black")),
            name="MCS-RDP kink (candidate hard scatter)"))
    if elpi_poly is not None:
        fig.add_trace(go.Scatter3d(
            x=elpi_poly[:, 0], y=elpi_poly[:, 1], z=elpi_poly[:, 2],
            mode="lines", line=dict(color="red", width=3, dash="dot"),
            name="elpigraph"))
    for nm, pt, col in (("GT start", rec.gt_start, "darkgreen"),
                        ("GT end", rec.gt_end, "orange")):
        if np.all(np.isfinite(pt)):
            fig.add_trace(go.Scatter3d(
                x=[pt[0]], y=[pt[1]], z=[pt[2]], mode="markers",
                marker=dict(size=6, color=col, symbol="diamond"), name=nm))
    if len(rec.true_kinks):
        K = rec.true_kinks
        fig.add_trace(go.Scatter3d(
            x=K[:, 0], y=K[:, 1], z=K[:, 2], mode="markers",
            marker=dict(size=7, color="black", symbol="x"), name="true kinks"))
    nkv = 0 if cfs.get("kink_vertices") is None else len(cfs["kink_vertices"])
    title = (f"{rec.event_file} inst{rec.inst_idx} {rec.pred_cls_name} "
             f"n={rec.n_points} | {cfs['n_frag']}frag {cfs['n_chain']}chain "
             f"main={cfs['main_frac']*100:.0f}% | centerline RMS={metrics['s_rms']:.2f} "
             f"range={metrics['s_len']:.1f}cm | {nkv} MCS-RDP kink(s)")
    fig.update_layout(title=title, scene_aspectmode="data",
                      margin=dict(l=0, r=0, t=30, b=0))
    fig.write_html(out_path)


def _endpoint_errs(poly, rec):
    ends = np.array([poly[0], poly[-1]])
    out = {}
    for nm, gt in (("s", rec.gt_start), ("e", rec.gt_end)):
        out[nm] = (float(np.linalg.norm(ends - gt, axis=1).min())
                   if np.all(np.isfinite(gt)) else float("nan"))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    here = os.path.dirname(os.path.abspath(__file__))
    dev = os.path.join(here, "..", "reco_dev_data")
    ap.add_argument("--keypoint2-dir", default=os.path.join(dev, "keypoint2_out"))
    ap.add_argument("--merged-sp-dir", default=os.path.join(dev, "merged_sp"))
    ap.add_argument("--out-dir", default=os.path.join(here, "cfs_out"))
    ap.add_argument("--eps", type=float, default=1.2)
    ap.add_argument("--window", type=float, default=2.0)
    ap.add_argument("--step", type=float, default=1.0)
    # --- stitch gating (how aggressively to bridge fragment gaps) ---
    ap.add_argument("--max-gap", type=float, default=20.0,
                    help="max endpoint gap [cm] to link two fragments (the "
                         "max_gap_live gate). Wide by default: within one "
                         "predicted instance the segmenter already claims all "
                         "points are one particle, so bridge aggressively. The "
                         "dead-channel map (once wired) will further raise this "
                         "only across known-dead bands.")
    ap.add_argument("--min-dir", type=float, default=0.3,
                    help="min end-tangent collinearity to link (relax toward -1 "
                         "to link regardless of direction; short fragments bypass "
                         "this gate already).")
    ap.add_argument("--no-extend", action="store_true")
    ap.add_argument("--no-charge", action="store_true")
    ap.add_argument("--min-points", type=int, default=20)
    ap.add_argument("--no-html", action="store_true")
    ap.add_argument("--with-elpi", action="store_true",
                    help="also run ElPiGraph and add its RMS to the table")
    # --- Method-B RDP smoothing ---
    ap.add_argument("--no-rdp", action="store_true",
                    help="disable the MCS-tied RDP smoothing stage")
    ap.add_argument("--kappa", type=float, default=1.0,
                    help="RDP significance multiplier (higher -> fewer kinks). "
                         "~1 keeps real curvature here; 3 over-collapses to a "
                         "single segment because sigma_MCS=L*theta0/sqrt3 (brief "
                         "spec) over-estimates the chord sagitta.")
    ap.add_argument("--momentum-source", choices=["truth", "fixed"],
                    default="truth", help="p for the MCS tolerance")
    ap.add_argument("--fixed-p", type=float, default=500.0,
                    help="MeV, used when momentum-source=fixed or truth missing")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.keypoint2_dir, "*.h5")))
    msp = (args.merged_sp_dir if args.merged_sp_dir
           and os.path.isdir(args.merged_sp_dir) else None)
    os.makedirs(args.out_dir, exist_ok=True)

    if args.with_elpi:                    # warm up numba JIT off the clock
        try:
            fit_elpigraph(np.random.RandomState(0).randn(40, 3), num_nodes=6)
        except Exception:
            pass

    # The polyline is the DENSE centerline (hugs points). range = its length;
    # RMS over the main chain's own points. nkink = MCS-RDP discrete kinks
    # (candidate hard scatters); tkink = true kink vertices (mc daughters).
    print(f"{'event/inst':>12} {'cls':>4} {'n':>5} {'pMeV':>5} | {'nfr':>3} "
          f"{'nch':>3} {'cov%':>4} | {'range':>6} {'RMS':>5} {'nvtx':>4} "
          f"{'nkink':>5} {'tkink':>5} {'sE':>5} {'eE':>5} {'ms':>5}"
          + ("  e_RMS" if args.with_elpi else ""))
    rows = []
    for fp in files:
        for rec in tio.load_instances(fp, msp, tracks_only=True,
                                      min_points=args.min_points):
            w = None if (args.no_charge or rec.weights is None) else rec.weights
            mass = tio._PDG_MASS.get(int(rec.pid), 105.658)   # mu mass fallback
            p_use = (rec.true_p_mev if (args.momentum_source == "truth"
                     and np.isfinite(rec.true_p_mev) and rec.true_p_mev > 0)
                     else args.fixed_p)
            beta = beta_from_p(p_use, mass)
            tol = (None if args.no_rdp
                   else make_mcs_tolerance(p_use, beta, kappa=args.kappa))

            cfs = cluster_fit_stitch(rec.points, weights=w, eps=args.eps,
                                     window=args.window, step=args.step,
                                     extend=not args.no_extend, rdp_tol=tol,
                                     max_gap_live=args.max_gap,
                                     min_dir=args.min_dir)
            sp = cfs["polyline"]
            mpts = rec.points[cfs["main_mask"]] if cfs["n_frag"] else rec.points
            rms = float(np.sqrt((point_to_polyline_distance(mpts, sp) ** 2).mean()))
            ee = _endpoint_errs(sp, rec)
            rng = path_length(sp)
            nkink = len(cfs["kink_vertices"])
            tkink = len(rec.true_kinks)

            e_rms = float("nan")
            ep = None
            if args.with_elpi:
                nn = int(np.clip(np.ptp(rec.points, 0).max() / 2.0, 5, 60))
                try:
                    ep, _ = fit_elpigraph(rec.points, weights=w, num_nodes=nn)
                    e_rms = float(np.sqrt((point_to_polyline_distance(
                        rec.points, ep) ** 2).mean()))
                except Exception:
                    pass

            tag = f"{fp.split('_event')[1][:5]}_i{rec.inst_idx}"
            print(f"{tag:>12} {rec.pred_cls_name:>4} {rec.n_points:5d} "
                  f"{p_use:5.0f} | {cfs['n_frag']:3d} {cfs['n_chain']:3d} "
                  f"{cfs['main_frac']*100:4.0f} | {rng:6.1f} {rms:5.2f} "
                  f"{len(sp):4d} {nkink:5d} {tkink:5d} {ee['s']:5.1f} "
                  f"{ee['e']:5.1f} {cfs['runtime_s']*1e3:5.1f}"
                  + (f"  {e_rms:5.2f}" if args.with_elpi else ""))
            rows.append(dict(nfr=cfs["n_frag"], nch=cfs["n_chain"],
                             main=cfs["main_frac"], rms=rms, e_rms=e_rms,
                             rng=rng, nvtx=len(sp), nkink=nkink, tkink=tkink,
                             ms=cfs["runtime_s"] * 1e3))
            if not args.no_html:
                op = os.path.join(args.out_dir,
                                  f"{os.path.splitext(rec.event_file)[0]}_inst{rec.inst_idx}.html")
                write_html(rec, cfs, ep, dict(s_rms=rms, s_len=rng,
                                              e_len=float("nan")), op)

    if rows:
        import statistics as st
        med = lambda k: st.median([r[k] for r in rows if np.isfinite(r[k])])
        n1 = sum(1 for r in rows if r["nch"] == 1)
        print(f"\n=== {len(rows)} track instances "
              f"(RDP {'OFF' if args.no_rdp else f'kappa={args.kappa}'}, "
              f"momentum={args.momentum_source}) ===")
        print(f"fragments/instance:  median={med('nfr'):.0f}  "
              f"fully-stitched={n1}/{len(rows)}  main-chain frac={med('main')*100:.0f}%")
        print(f"residual RMS cm (dense centerline): median={med('rms'):.2f}"
              + (f"   (elpi {med('e_rms'):.2f})" if args.with_elpi else ""))
        print(f"MCS-RDP kinks found vs true: median {med('nkink'):.0f} vs "
              f"{med('tkink'):.0f} (precision/recall = step c)")
        print(f"runtime ms/instance: median={med('ms'):.1f}")
        if not args.no_html:
            print(f"HTML overlays -> {args.out_dir}")


if __name__ == "__main__":
    main()
