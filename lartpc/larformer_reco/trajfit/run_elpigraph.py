"""ElPiGraph spike: fit an elastic principal curve to track-class instances.

For each predicted track instance in a keypoint2_out event, this:
  1. fits elpigraph.computeElasticPrincipalCurve on the (optionally charge-
     weighted) point cloud,
  2. traces the returned node graph into an ORDERED polyline,
  3. reports runtime, residual RMS (cloud->polyline), endpoint error vs GT,
  4. writes a Plotly HTML overlay (cloud + polyline + GT endpoints + true kinks).

This is a throwaway dev harness for understanding ElPiGraph behaviour, tuning
meta-parameters, and sanity-checking output quality on real data. See
`../trajectory_fitting_brief.md` §6 and §10.

Run INSIDE the container:
    ./run_in_local_pointcept_container.sh python \
        lartpc/larformer_reco/trajfit/run_elpigraph.py \
        --merged-sp-dir lartpc/larformer_reco/reco_dev_data/merged_sp
"""
import os
import time
import glob
import argparse

import numpy as np

from . import trajfit_io as tio


# ----------------------------------------------------------------------------
# geometry helpers
# ----------------------------------------------------------------------------
def point_to_polyline_distance(points, vertices):
    """Min perpendicular distance from each point to a polyline (segments).
    points (N,3), vertices (M,3) -> (N,) distances. Brief §3 helper."""
    P = np.asarray(points, np.float64)
    V = np.asarray(vertices, np.float64)
    if V.shape[0] < 2:
        return np.linalg.norm(P - V[0], axis=1)
    a = V[:-1]                       # (S,3)
    b = V[1:]                        # (S,3)
    ab = b - a                       # (S,3)
    ab2 = (ab * ab).sum(1) + 1e-12   # (S,)
    # (N,S) projection param t
    ap = P[:, None, :] - a[None, :, :]            # (N,S,3)
    t = (ap * ab[None]).sum(2) / ab2[None]        # (N,S)
    t = np.clip(t, 0.0, 1.0)
    proj = a[None] + t[..., None] * ab[None]      # (N,S,3)
    d = np.linalg.norm(P[:, None, :] - proj, axis=2)   # (N,S)
    return d.min(1)


def trace_path(node_pos, edges):
    """Order ElPiGraph nodes into a single polyline.

    For a curve topology edges form (mostly) a path. Walk from a degree-1
    endpoint following unused edges; if the graph branches, take the longest
    arm at each junction so we return the principal thread. Falls back to a
    PCA-projection sort if the graph is degenerate.
    """
    n = node_pos.shape[0]
    if n == 0:
        return node_pos
    adj = [[] for _ in range(n)]
    for u, v in edges:
        u, v = int(u), int(v)
        adj[u].append(v)
        adj[v].append(u)
    deg = np.array([len(a) for a in adj])
    endpoints = np.where(deg == 1)[0]

    def bfs_farthest(src):
        seen = {src}
        order = [src]
        frontier = [src]
        dist = {src: 0}
        while frontier:
            nxt = []
            for u in frontier:
                for w in adj[u]:
                    if w not in seen:
                        seen.add(w)
                        dist[w] = dist[u] + 1
                        order.append(w)
                        nxt.append(w)
            frontier = nxt
        far = max(dist, key=dist.get)
        return far, dist

    if endpoints.size >= 1 and edges.shape[0] >= 1:
        # longest path between two graph-far endpoints (handles small branches)
        a, _ = bfs_farthest(int(endpoints[0]))
        b, _ = bfs_farthest(a)
        # reconstruct a->b via parent pointers
        parent = {a: -1}
        stack = [a]
        seen = {a}
        while stack:
            u = stack.pop()
            for w in adj[u]:
                if w not in seen:
                    seen.add(w)
                    parent[w] = u
                    stack.append(w)
        path = []
        cur = b
        while cur != -1:
            path.append(cur)
            cur = parent.get(cur, -1)
        path = path[::-1]
        if len(path) >= 2:
            return node_pos[path]

    # fallback: order by projection on first principal axis
    c = node_pos.mean(0)
    u, s, vt = np.linalg.svd(node_pos - c, full_matrices=False)
    s_order = np.argsort((node_pos - c) @ vt[0])
    return node_pos[s_order]


# ----------------------------------------------------------------------------
# fit
# ----------------------------------------------------------------------------
def fit_elpigraph(points, weights=None, num_nodes=30, Lambda=0.01, Mu=0.1,
                  trimming_radius=float("inf"), use_gpu=False, n_cores=1,
                  normalize=True):
    """Fit one elastic principal curve. Returns (polyline, info)."""
    import elpigraph

    X = np.asarray(points, np.float64)
    # scale-normalize so Lambda/Mu mean the same thing on 5 cm vs 200 cm tracks
    center = X.mean(0)
    scale = float(np.linalg.norm(X - center, axis=1).std()) if normalize else 1.0
    scale = scale if scale > 1e-6 else 1.0
    Xn = (X - center) / scale
    tr = (trimming_radius / scale if np.isfinite(trimming_radius)
          else float("inf"))

    kw = dict(NumNodes=int(num_nodes), Lambda=Lambda, Mu=Mu,
              TrimmingRadius=tr, GPU=use_gpu, n_cores=n_cores, verbose=False)
    if weights is not None:
        w = np.asarray(weights, np.float64).reshape(-1, 1)
        if w.sum() > 1e-6:                # skip all-zero charge (e.g. ghost match)
            kw["PointWeights"] = w / w.mean()

    t0 = time.perf_counter()
    res = elpigraph.computeElasticPrincipalCurve(Xn, **kw)[0]
    dt = time.perf_counter() - t0

    nodes_n = np.asarray(res["NodePositions"])
    edges = np.asarray(res["Edges"][0]) if isinstance(res["Edges"], (list, tuple)) \
        else np.asarray(res["Edges"])
    nodes = nodes_n * scale + center
    poly = trace_path(nodes, edges)
    info = dict(runtime_s=dt, n_nodes=nodes.shape[0], n_edges=len(edges),
                n_seg=max(len(poly) - 1, 0))
    return poly.astype(np.float32), info


def evaluate(rec, poly):
    """Residual + endpoint metrics for one fit."""
    res_cloud = point_to_polyline_distance(rec.points, poly)
    out = dict(residual_rms_cm=float(np.sqrt((res_cloud ** 2).mean())),
               residual_p95_cm=float(np.percentile(res_cloud, 95)),
               length_cm=float(np.linalg.norm(np.diff(poly, axis=0), axis=1).sum()))
    # truth-cloud residual (polyline vs GT cloud) if available
    if rec.truth_cloud is not None and len(rec.truth_cloud) >= 2:
        rt = point_to_polyline_distance(rec.truth_cloud, poly)
        out["truthcloud_rms_cm"] = float(np.sqrt((rt ** 2).mean()))
    # endpoint error: match polyline ends to GT start/end (both orderings)
    ends = np.array([poly[0], poly[-1]])
    for name, gt in (("gt_start", rec.gt_start), ("gt_end", rec.gt_end)):
        if np.all(np.isfinite(gt)):
            out[f"{name}_err_cm"] = float(np.linalg.norm(ends - gt, axis=1).min())
    return out


# ----------------------------------------------------------------------------
# viz
# ----------------------------------------------------------------------------
def write_html(rec, poly, metrics, out_path):
    import plotly.graph_objects as go
    P = rec.points
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(
        x=P[:, 0], y=P[:, 1], z=P[:, 2], mode="markers",
        marker=dict(size=2, color="lightblue", opacity=0.5), name="cloud"))
    if rec.truth_cloud is not None and len(rec.truth_cloud):
        T = rec.truth_cloud
        fig.add_trace(go.Scatter3d(
            x=T[:, 0], y=T[:, 1], z=T[:, 2], mode="markers",
            marker=dict(size=1.5, color="rgba(150,150,150,0.25)"),
            name="GT cloud"))
    fig.add_trace(go.Scatter3d(
        x=poly[:, 0], y=poly[:, 1], z=poly[:, 2], mode="lines+markers",
        line=dict(color="red", width=4), marker=dict(size=3, color="red"),
        name="elpigraph polyline"))
    for nm, pt, col in (("GT start", rec.gt_start, "green"),
                        ("GT end", rec.gt_end, "orange"),
                        ("pred start", rec.pred_start, "cyan"),
                        ("pred end", rec.pred_end, "magenta")):
        if np.all(np.isfinite(pt)):
            fig.add_trace(go.Scatter3d(
                x=[pt[0]], y=[pt[1]], z=[pt[2]], mode="markers",
                marker=dict(size=6, color=col, symbol="diamond"), name=nm))
    if len(rec.true_kinks):
        K = rec.true_kinks
        fig.add_trace(go.Scatter3d(
            x=K[:, 0], y=K[:, 1], z=K[:, 2], mode="markers",
            marker=dict(size=7, color="black", symbol="x"), name="true kinks"))
    title = (f"{rec.event_file} inst{rec.inst_idx} {rec.pred_cls_name} "
             f"n={rec.n_points} p={rec.true_p_mev:.0f}MeV | "
             f"RMS={metrics['residual_rms_cm']:.2f}cm seg={metrics.get('n_seg','?')}")
    fig.update_layout(title=title, scene_aspectmode="data",
                      margin=dict(l=0, r=0, t=30, b=0))
    fig.write_html(out_path)


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    here = os.path.dirname(os.path.abspath(__file__))
    dev = os.path.join(here, "..", "reco_dev_data")
    ap.add_argument("--keypoint2-dir",
                    default=os.path.join(dev, "keypoint2_out"))
    ap.add_argument("--merged-sp-dir",
                    default=os.path.join(dev, "merged_sp"),
                    help="parent merged_sp dir for charge+truth (optional)")
    ap.add_argument("--out-dir", default=os.path.join(here, "elpigraph_out"))
    ap.add_argument("--num-nodes", type=int, default=0,
                    help="0 = length-adaptive (~1 node / 2 cm, clamped 5..60)")
    ap.add_argument("--Lambda", type=float, default=0.01)
    ap.add_argument("--Mu", type=float, default=0.1)
    ap.add_argument("--trimming-radius", type=float, default=float("inf"),
                    help="cm; reject points beyond this from the graph (delta-rays)")
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--no-charge", action="store_true",
                    help="ignore charge weights even if merged_sp present")
    ap.add_argument("--min-points", type=int, default=10)
    ap.add_argument("--max-instances", type=int, default=-1)
    ap.add_argument("--no-html", action="store_true")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.keypoint2_dir, "*.h5")))
    if not files:
        raise SystemExit(f"no keypoint2 files in {args.keypoint2_dir}")
    os.makedirs(args.out_dir, exist_ok=True)
    msp = None if not args.merged_sp_dir or not os.path.isdir(args.merged_sp_dir) \
        else args.merged_sp_dir
    print(f">>> {len(files)} events | merged_sp={'yes' if msp else 'no'} | "
          f"gpu={args.gpu}")

    # Warm up numba/CuPy JIT so the first real instance isn't charged ~7 s of
    # compile time (timings below are then steady-state).
    try:
        _t = time.perf_counter()
        fit_elpigraph(np.random.RandomState(0).randn(40, 3), num_nodes=6,
                      use_gpu=args.gpu)
        print(f">>> JIT warmup {time.perf_counter()-_t:.1f}s (excluded from timings)")
    except Exception as e:
        print(f">>> warmup skipped: {type(e).__name__}: {e}")

    rows = []
    done = 0
    for fp in files:
        recs = tio.load_instances(fp, msp, tracks_only=True,
                                  min_points=args.min_points)
        for rec in recs:
            if args.max_instances >= 0 and done >= args.max_instances:
                break
            nn = args.num_nodes or int(np.clip(
                np.ptp(rec.points, 0).max() / 2.0, 5, 60))
            w = None if (args.no_charge or rec.weights is None) else rec.weights
            try:
                poly, info = fit_elpigraph(
                    rec.points, weights=w, num_nodes=nn, Lambda=args.Lambda,
                    Mu=args.Mu, trimming_radius=args.trimming_radius,
                    use_gpu=args.gpu)
            except Exception as e:
                print(f"  [{rec.event_file} inst{rec.inst_idx}] FIT FAILED: "
                      f"{type(e).__name__}: {e}")
                continue
            m = evaluate(rec, poly)
            m.update(info)
            row = dict(event=rec.event_file, inst=rec.inst_idx,
                       cls=rec.pred_cls_name, n=rec.n_points,
                       p_MeV=round(rec.true_p_mev, 1), nodes=nn, **m)
            rows.append(row)
            print(f"  {rec.event_file} inst{rec.inst_idx:2d} "
                  f"{rec.pred_cls_name:5s} n={rec.n_points:4d} nodes={nn:2d} "
                  f"t={info['runtime_s']*1e3:6.1f}ms RMS={m['residual_rms_cm']:.2f}cm "
                  f"len={m['length_cm']:5.1f}cm seg={info['n_seg']:2d} "
                  f"start_err={m.get('gt_start_err_cm', float('nan')):.2f} "
                  f"end_err={m.get('gt_end_err_cm', float('nan')):.2f}")
            if not args.no_html:
                op = os.path.join(
                    args.out_dir,
                    f"{os.path.splitext(rec.event_file)[0]}_inst{rec.inst_idx}.html")
                write_html(rec, poly, m, op)
            done += 1

    if rows:
        import statistics as st
        rt = [r["runtime_s"] for r in rows]
        rms = [r["residual_rms_cm"] for r in rows]
        print(f"\n=== {len(rows)} track instances fit ===")
        print(f"runtime/instance: median={st.median(rt)*1e3:.1f}ms "
              f"max={max(rt)*1e3:.1f}ms")
        print(f"residual RMS cm:  median={st.median(rms):.2f} max={max(rms):.2f}")
        if not args.no_html:
            print(f"HTML overlays -> {args.out_dir}")


if __name__ == "__main__":
    main()
