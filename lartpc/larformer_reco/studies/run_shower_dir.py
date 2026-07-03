"""Phase-1 shower-direction study: run the 3 trunk methods on every predicted
shower instance, evaluate direction accuracy + nu-vertex connection vs truth,
and report a comparison table (accuracy + compute). See `../shower_reco_spec.md`.

Run INSIDE the container:
    ./run_in_local_pointcept_container.sh python \
        lartpc/larformer_reco/studies/run_shower_dir.py \
        --keypoint2-dir ../reco_dev_data/bnb_pi0_valdata/keypoint2_out
"""
import os
import sys
sys.path.insert(0, __import__("os").path.abspath(__import__("os").path.join(
    __import__("os").path.dirname(__import__("os").path.abspath(__file__)), "..", "..", "..")))
import glob
import argparse
import statistics as st

import numpy as np

from lartpc.larformer_reco.trajfit import trajfit_io as tio
from lartpc.larformer_reco.trajfit.shower_trunk import trunk_pca, trunk_elpigraph, trunk_vertex_biased
from lartpc.larformer_reco.trajfit.shower_connect import connection_geometry, connects
from lartpc.larformer_reco.trajfit.nu_interaction import vertex_candidates, read_nu_vertices

METHOD_COLORS = {"pca": "#1f77b4", "elpigraph": "#2ca02c",
                 "vertex_biased": "#ff7f0e", "true": "#000000"}
IMPACT_TRUTH_CM = 5.0       # true trunk line within this of GT vtx -> "connects"


def true_trunk_dir(gt_cloud, gt_start, R=5.0):
    """True trunk direction = from the GT shower-start toward the centroid of the
    GT points within R cm of it (the trunk region the start keypoint tags)."""
    if gt_cloud is None or len(gt_cloud) == 0 or not np.all(np.isfinite(gt_start)):
        return None
    near = gt_cloud[np.linalg.norm(gt_cloud - gt_start, axis=1) <= R]
    if len(near) < 3:
        near = gt_cloud
    d = near.mean(0) - gt_start
    n = np.linalg.norm(d)
    return d / n if n > 1e-6 else None


def angle_deg(a, b):
    return float(np.degrees(np.arccos(np.clip(np.dot(a, b), -1.0, 1.0))))


def add_arrow(fig, start, d, length, color, name, width=5):
    import plotly.graph_objects as go
    e = np.asarray(start) + np.asarray(d) * length
    fig.add_trace(go.Scatter3d(
        x=[start[0], e[0]], y=[start[1], e[1]], z=[start[2], e[2]],
        mode="lines", line=dict(color=color, width=width), name=name))


def visualize(event_tag, showers, vertex, gt_nu, out_path, arrow_len=15.0):
    import plotly.graph_objects as go
    fig = go.Figure()
    for sh in showers:
        P = sh["points"]
        fig.add_trace(go.Scatter3d(
            x=P[:, 0], y=P[:, 1], z=P[:, 2], mode="markers",
            marker=dict(size=1.6, color="lightgray", opacity=0.4),
            showlegend=False, name=f"sh{sh['inst']} {sh['cls']}"))
        for m, tk in sh["trunks"].items():
            add_arrow(fig, tk.start, tk.direction, arrow_len, METHOD_COLORS[m],
                      f"sh{sh['inst']} {m}")
        if sh["true_dir"] is not None:
            add_arrow(fig, sh["gt_start"], sh["true_dir"], arrow_len,
                      METHOD_COLORS["true"], f"sh{sh['inst']} TRUE", width=7)
    fig.add_trace(go.Scatter3d(
        x=[vertex[0]], y=[vertex[1]], z=[vertex[2]], mode="markers",
        marker=dict(size=10, color="gold", symbol="diamond"), name="nu vertex"))
    if np.all(np.isfinite(gt_nu)):
        fig.add_trace(go.Scatter3d(
            x=[gt_nu[0]], y=[gt_nu[1]], z=[gt_nu[2]], mode="markers",
            marker=dict(size=8, color="red", symbol="x"), name="GT nu vertex"))
    fig.update_layout(title=event_tag, scene_aspectmode="data",
                      margin=dict(l=0, r=0, t=30, b=0))
    fig.write_html(out_path)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    here = os.path.dirname(os.path.abspath(__file__))
    dev = os.path.join(here, "..", "reco_dev_data", "bnb_pi0_valdata")
    ap.add_argument("--keypoint2-dir", default=os.path.join(dev, "keypoint2_out"))
    ap.add_argument("--merged-sp-dir", default=os.path.join(dev, "merged_sp"))
    ap.add_argument("--out-dir", default=os.path.join(here, "shower_dir_out"))
    ap.add_argument("--vertex-source", choices=["reco", "gt"], default="reco")
    ap.add_argument("--min-points", type=int, default=20)
    ap.add_argument("--d-impact", type=float, default=5.0)
    ap.add_argument("--cos-min", type=float, default=0.9)
    ap.add_argument("--max-events", type=int, default=-1)
    ap.add_argument("--no-html", action="store_true")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.keypoint2_dir, "*.h5")))
    if args.max_events >= 0:
        files = files[:args.max_events]
    msp = (args.merged_sp_dir if args.merged_sp_dir
           and os.path.isdir(args.merged_sp_dir) else None)
    os.makedirs(args.out_dir, exist_ok=True)
    # warm up ElPiGraph JIT off the clock
    try:
        trunk_elpigraph(np.random.RandomState(0).randn(40, 3) * 3, np.zeros(3))
    except Exception:
        pass

    methods = ["pca", "elpigraph", "vertex_biased"]
    agg = {m: dict(ang=[], start=[], ms=[], tp=0, fp=0, fn=0, tn=0)
           for m in methods}
    n_show = 0
    print(f">>> shower-direction study | vertex={args.vertex_source}")

    for fp in files:
        recs = tio.load_instances(fp, msp, tracks_only=False,
                                  min_points=args.min_points)
        showers_rec = [r for r in recs if r.pred_cls in tio.SHOWER_CLASSES
                       and np.all(np.isfinite(r.pred_start))]
        if not showers_rec:
            continue
        if args.vertex_source == "reco":
            cands = vertex_candidates(fp)
            vertex = cands[0][0] if cands else read_nu_vertices(fp)[0]
        else:
            vertex = read_nu_vertices(fp)[1]
        if not np.all(np.isfinite(vertex)):
            continue
        gt_nu = read_nu_vertices(fp)[1]

        showers = []
        for r in showers_rec:
            n_show += 1
            P = r.points
            trunks = {
                "pca": trunk_pca(P, r.pred_start, weights=r.weights),
                "elpigraph": trunk_elpigraph(P, r.pred_start),
                "vertex_biased": trunk_vertex_biased(P, vertex)}
            tdir = true_trunk_dir(r.truth_cloud, r.gt_start)
            # truth connection label: does the TRUE trunk line pass near GT vtx?
            truth_conn = False
            if tdir is not None and np.all(np.isfinite(gt_nu)):
                gi = connection_geometry(r.gt_start, tdir, gt_nu)["impact"]
                truth_conn = gi <= IMPACT_TRUTH_CM
            for m, tk in trunks.items():
                agg[m]["ms"].append(tk.runtime_s * 1e3)
                if tdir is not None:
                    agg[m]["ang"].append(angle_deg(tk.direction, tdir))
                    agg[m]["start"].append(
                        float(np.linalg.norm(tk.start - r.gt_start)))
                ok, _ = connects(tk.start, tk.direction, vertex,
                                 d_impact=args.d_impact, cos_min=args.cos_min)
                key = ("tp" if (ok and truth_conn) else "fp" if ok else
                       "fn" if truth_conn else "tn")
                agg[m][key] += 1
            showers.append(dict(inst=r.inst_idx, cls=r.pred_cls_name, points=P,
                                trunks=trunks, gt_start=r.gt_start, true_dir=tdir))

        if not args.no_html:
            op = os.path.join(args.out_dir,
                              f"{os.path.splitext(os.path.basename(fp))[0]}_shower.html")
            visualize(os.path.basename(fp), showers, vertex, gt_nu, op)

    # ---- comparison table ----
    def pctl(x, q):
        return float(np.percentile(x, q)) if x else float("nan")
    print(f"\n=== {n_show} showers in {len(files)} events "
          f"(d_impact={args.d_impact}, cos_min={args.cos_min}) ===")
    print(f"{'method':>14} | {'ang_med':>7} {'ang_p95':>7} {'start_med':>9} "
          f"| {'conn_P':>6} {'conn_R':>6} | {'ms/shower':>9}")
    for m in methods:
        a = agg[m]
        prec = a["tp"] / max(a["tp"] + a["fp"], 1)
        rec = a["tp"] / max(a["tp"] + a["fn"], 1)
        print(f"{m:>14} | {pctl(a['ang'],50):7.1f} {pctl(a['ang'],95):7.1f} "
              f"{pctl(a['start'],50):9.2f} | {prec:6.2f} {rec:6.2f} | "
              f"{st.median(a['ms']) if a['ms'] else float('nan'):9.1f}")
    if not args.no_html:
        print(f"\nHTML -> {args.out_dir}")


if __name__ == "__main__":
    main()
