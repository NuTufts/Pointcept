"""Prototype neutrino-interaction reco: nu-vertex + track trajectories -> tree.

Roots a particle tree at the (highest-score) nu vertex, attaches tracks whose
endpoint is near the vertex AND whose initial direction extrapolates back to it,
then repeats at the far ends of attached tracks (secondary vertices) to chain the
remaining tracks. See `nu_interaction_spec.md`.

Run INSIDE the container:
    ./run_in_local_pointcept_container.sh python \
        lartpc_data_prep/larformer_keypoint_v2/trajfit_dev/nu_interaction.py \
        --vertex-source gt
"""
import os
import glob
import argparse
from collections import deque

import numpy as np
import h5py

import trajfit_io as tio
from cluster_fit_stitch import cluster_fit_stitch


# ---------------------------------------------------------------------------
# track building
# ---------------------------------------------------------------------------
def _initial_dir(poly, at_start, seg_cm=3.0):
    """Endpoint position + unit direction into the body over the first seg_cm
    (or the whole track if shorter)."""
    pts = (poly if at_start else poly[::-1]).astype(np.float64)
    p0 = pts[0]
    acc, prev, tgt = 0.0, p0, pts[-1]
    for q in pts[1:]:
        acc += np.linalg.norm(q - prev)
        prev = q
        if acc >= seg_cm:
            tgt = q
            break
    u = tgt - p0
    n = np.linalg.norm(u)
    return p0, (u / n if n > 1e-9 else np.zeros(3))


def build_tracks(recs, seg_cm=3.0, **cfg):
    """Reconstruct each track-class instance and extract its two ends + dirs."""
    tracks = []
    for tid, rec in enumerate(recs):
        cfs = cluster_fit_stitch(rec.points, weights=rec.weights, **cfg)
        poly = np.asarray(cfs["polyline"], np.float64)
        if len(poly) < 2:
            continue
        ends = []
        for at_start in (True, False):
            pos, u = _initial_dir(poly, at_start, seg_cm)
            ends.append(dict(pos=pos, u_in=u))
        tracks.append(dict(
            id=tid, inst_idx=rec.inst_idx, cls=rec.pred_cls,
            cls_name=rec.pred_cls_name, gt_trackid=rec.gt_trackid,
            poly=poly, points=rec.points, ends=ends,
            truth_cloud=np.asarray(rec.truth_cloud, np.float32),
            length=float(np.linalg.norm(np.diff(poly, axis=0), axis=1).sum())))
    return tracks


# ---------------------------------------------------------------------------
# attachment test
# ---------------------------------------------------------------------------
def attach_cost(end, V, d_vertex, d_perp, front_tol):
    """(ok, cost, info) for attaching `end` to vertex position V. See spec."""
    pos, u = end["pos"], end["u_in"]
    r = V - pos
    gap = float(np.linalg.norm(r))
    if gap > d_vertex:
        return False, np.inf, None
    if np.linalg.norm(u) < 1e-6:                 # no usable direction -> gap only
        return True, gap, dict(gap=gap, perp=0.0, along=0.0)
    a = -u                                        # backward-extrapolation axis
    along = float(r @ a)                          # >0: vertex behind the start
    perp = float(np.linalg.norm(r - along * a))   # dist of V to the back-ray
    if perp > d_perp or along < -front_tol:
        return False, np.inf, None
    return True, gap + 2.0 * perp, dict(gap=gap, perp=perp, along=along)


def snap_vertex(V0, tracks, radius, conv_radius=5.0):
    """Refine the seed vertex to the densest cluster of track ENDPOINTS within
    `radius` of V0 -- i.e. where track starts converge (a real interaction point),
    not merely the nearest endpoint (which can be a wrong far end when the seed is
    badly placed). Returns (snapped, shift, n_in_cluster). Large offsets (no
    endpoint within radius) are left alone -- they need a better vertex candidate
    upstream (score-map clustering), not snapping."""
    V0 = np.asarray(V0, np.float64)
    pos, tid = [], []
    for T in tracks:
        for e in T["ends"]:
            if np.linalg.norm(e["pos"] - V0) <= radius:
                pos.append(e["pos"])
                tid.append(T["id"])
    if not pos:
        return V0, 0.0, 0
    pos, tid = np.asarray(pos), np.asarray(tid)
    # convergence score: # endpoints from OTHER tracks within conv_radius
    counts = [int(((np.linalg.norm(pos - p, axis=1) <= conv_radius)
                   & (tid != tid[i])).sum()) for i, p in enumerate(pos)]
    seed = pos[int(np.argmax(counts))]                   # densest cross-track point
    cluster = pos[np.linalg.norm(pos - seed, axis=1) <= conv_radius]
    snapped = cluster.mean(0)
    return snapped, float(np.linalg.norm(snapped - V0)), len(cluster)


# ---------------------------------------------------------------------------
# BFS tree growth
# ---------------------------------------------------------------------------
def reco_interaction(primary, tracks, d_vertex=8.0, d_perp=3.0, front_tol=1.5,
                     merge_radius=3.0):
    vertices = []

    def add_vertex(pos, depth, parent_track):
        pos = np.asarray(pos, np.float64)
        for v in vertices:
            if np.linalg.norm(pos - v["pos"]) < merge_radius:
                return v["id"], False
        vid = len(vertices)
        vertices.append(dict(id=vid, pos=pos, depth=depth,
                             parent_track=parent_track, attached=[]))
        return vid, True

    for T in tracks:
        T["attached"] = False
        T["attach"] = None
        T["far_vertex"] = None

    pv, _ = add_vertex(np.asarray(primary, np.float64), 0, None)
    queue = deque([pv])
    edges = []
    while queue:
        V = vertices[queue.popleft()]
        cands = []
        for T in tracks:
            if T["attached"]:
                continue
            best = None
            for ei, e in enumerate(T["ends"]):
                ok, cost, info = attach_cost(e, V["pos"], d_vertex, d_perp, front_tol)
                if ok and (best is None or cost < best[1]):
                    best = (ei, cost, info)
            if best is not None:
                cands.append((T, best))
        for T, (ei, cost, info) in sorted(cands, key=lambda c: c[1][1]):
            if T["attached"]:
                continue
            T["attached"] = True
            T["attach"] = dict(vertex=V["id"], end=ei, depth=V["depth"], **info)
            V["attached"].append(T["id"])
            far_pos = T["ends"][1 - ei]["pos"]
            fvid, is_new = add_vertex(far_pos, V["depth"] + 1, T["id"])
            T["far_vertex"] = fvid
            edges.append(dict(vertex=V["id"], track=T["id"], end=ei, far=fvid))
            if is_new:
                queue.append(fvid)
    unattached = [T["id"] for T in tracks if not T["attached"]]
    return dict(vertices=vertices, tracks=tracks, edges=edges,
                unattached=unattached, primary_vertex=np.asarray(primary, float))


# ---------------------------------------------------------------------------
# event IO
# ---------------------------------------------------------------------------
def read_nu_vertices(kp_path):
    with h5py.File(kp_path, "r") as f:
        pred = f["nu_vertex_cm"][()].astype(np.float64)
        gt = f["gt_nu_vertex_cm"][()].astype(np.float64)
    return pred, gt


def vertex_candidates(kp_path, max_cand=8, score_thresh=0.5):
    """Ranked nu-vertex candidates from the reco-package score-field fitter
    (`reco.KeypointRecoTorch`: greedy Gaussian peak-finder with NMS peeling,
    fitting each peak's mean rather than a score-weighted centroid). Returns
    [(pos_cm, peak_score), ...] score-desc. Falls back to the `nu_vertex_cm`
    decode if the score map / reco package is unavailable.

    NB the fitter only emits peaks above its score threshold, so an event whose
    nu-vertex score never peaks at the true primary (e.g. the model's strongest
    response sits on a hadronic secondary) yields candidates elsewhere -- the
    convergence snap is what then recovers the primary."""
    import sys
    rp = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # ..._v2/
    if rp not in sys.path:
        sys.path.insert(0, rp)
    try:
        from reco import KeypointRecoTorch, KeypointRecoParams
        from reco.io import load_score_maps
    except Exception:                                # pragma: no cover
        pred, _ = read_nu_vertices(kp_path)
        return [(pred, 1.0)] if np.all(np.isfinite(pred)) else []
    try:
        loaded = load_score_maps(kp_path)
    except KeyError:                                 # no score maps in this H5
        pred, _ = read_nu_vertices(kp_path)
        return [(pred, 1.0)] if np.all(np.isfinite(pred)) else []
    nu = next((d for name, d in loaded["score_maps"].items()
               if name == "nu_vertex" or 0 in (d.get("kp_types") or [])), None)
    if nu is None:
        pred, _ = read_nu_vertices(kp_path)
        return [(pred, 1.0)] if np.all(np.isfinite(pred)) else []
    reco = KeypointRecoTorch(KeypointRecoParams(score_thresh=score_thresh))
    kps = reco.reconstruct(nu["coords_cm"], nu["score"], max_keypoints=max_cand)
    if not kps:
        pred, _ = read_nu_vertices(kp_path)
        return [(pred, 1.0)] if np.all(np.isfinite(pred)) else []
    return [(np.asarray(k.pos_cm, np.float64), float(k.peak_score)) for k in kps]


# ---------------------------------------------------------------------------
# visualization
# ---------------------------------------------------------------------------
_DEPTH_COLORS = ["#1f77b4", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b", "#e377c2"]


# JS injected into the HTML so dragging either 3D panel rotates BOTH (shared
# camera) -- makes the reco-vs-truth comparison direct.
_SYNC_CAMERAS = """
var gd = document.getElementById('{plot_id}');
var lock = false;
gd.on('plotly_relayout', function(ed){
    if (lock) return;
    var cam = ed['scene.camera'] || ed['scene2.camera'];
    if (!cam) return;
    lock = true;
    var upd = {};
    upd[ed['scene.camera'] ? 'scene2.camera' : 'scene.camera'] = cam;
    Plotly.relayout(gd, upd).then(function(){ lock = false; });
});
"""


def visualize(res, gt_nu, src_label, title, out_path):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    tracks = {T["id"]: T for T in res["tracks"]}
    fig = make_subplots(
        rows=1, cols=2, specs=[[{"type": "scene"}, {"type": "scene"}]],
        horizontal_spacing=0.01,
        subplot_titles=("reco interaction",
                        "true ionization (GT particle spacepoints)"))

    def trk_color(T):
        return ("lightgray" if not T["attached"]
                else _DEPTH_COLORS[T["attach"]["depth"] % len(_DEPTH_COLORS)])

    bbox = []   # collect all plotted points for a shared cube range

    # ---- LEFT (col 1): reco interaction ----
    for T in res["tracks"]:
        col = trk_color(T)
        P = T["points"]
        bbox.append(P)
        fig.add_trace(go.Scatter3d(
            x=P[:, 0], y=P[:, 1], z=P[:, 2], mode="markers",
            marker=dict(size=1.6, color=col, opacity=0.35), showlegend=False,
            name=f"t{T['id']} {T['cls_name']}"), row=1, col=1)
        poly = T["poly"]
        fig.add_trace(go.Scatter3d(
            x=poly[:, 0], y=poly[:, 1], z=poly[:, 2], mode="lines",
            line=dict(color=col, width=5),
            name=f"t{T['id']} {T['cls_name']} L={T['length']:.0f}cm"
                 + ("" if T["attached"] else " UNATT")), row=1, col=1)

    for e in res["edges"]:                          # attachment bridges
        V = res["vertices"][e["vertex"]]["pos"]
        ep = tracks[e["track"]]["ends"][e["end"]]["pos"]
        fig.add_trace(go.Scatter3d(
            x=[V[0], ep[0]], y=[V[1], ep[1]], z=[V[2], ep[2]], mode="lines",
            line=dict(color="black", width=2, dash="dot"),
            showlegend=False), row=1, col=1)

    for v in res["vertices"]:                       # vertices
        nprong = len(v["attached"])
        is_primary = v["depth"] == 0
        fig.add_trace(go.Scatter3d(
            x=[v["pos"][0]], y=[v["pos"][1]], z=[v["pos"][2]], mode="markers",
            marker=dict(size=12 if is_primary else 6 + 2 * nprong,
                        color="gold" if is_primary
                        else _DEPTH_COLORS[v["depth"] % len(_DEPTH_COLORS)],
                        symbol="diamond", line=dict(width=1, color="black")),
            name=(f"PRIMARY V ({nprong}p)" if is_primary
                  else f"V{v['id']} d{v['depth']} {nprong}p")), row=1, col=1)

    seed = res.get("seed_input")
    if seed is not None and np.linalg.norm(seed - res["vertices"][0]["pos"]) > 0.5:
        fig.add_trace(go.Scatter3d(
            x=[seed[0]], y=[seed[1]], z=[seed[2]], mode="markers",
            marker=dict(size=7, color="gray", symbol="circle", opacity=0.6),
            name=f"seed {src_label} (pre-snap)"), row=1, col=1)

    # ---- RIGHT (col 2): GT particle spacepoints, colored to match left ----
    for T in res["tracks"]:
        G = T.get("truth_cloud")
        if G is None or len(G) == 0:
            continue
        bbox.append(G)
        fig.add_trace(go.Scatter3d(
            x=G[:, 0], y=G[:, 1], z=G[:, 2], mode="markers",
            marker=dict(size=1.8, color=trk_color(T), opacity=0.55),
            showlegend=False,
            name=f"t{T['id']} {T['cls_name']} GT (trk {T['gt_trackid']})"),
            row=1, col=2)

    # GT nu vertex on BOTH panels for reference
    if gt_nu is not None and np.all(np.isfinite(gt_nu)):
        for c in (1, 2):
            fig.add_trace(go.Scatter3d(
                x=[gt_nu[0]], y=[gt_nu[1]], z=[gt_nu[2]], mode="markers",
                marker=dict(size=9, color="red", symbol="x"),
                name="GT nu vertex", showlegend=(c == 1)), row=1, col=c)

    # shared cube range so both panels frame the same volume identically
    pts = np.concatenate(bbox, axis=0) if bbox else np.zeros((1, 3))
    ctr = 0.5 * (pts.min(0) + pts.max(0))
    half = max((pts.max(0) - pts.min(0)).max() * 0.5, 1.0) * 1.05
    rng = [[ctr[i] - half, ctr[i] + half] for i in range(3)]
    scene = dict(aspectmode="cube",
                 xaxis=dict(range=rng[0]), yaxis=dict(range=rng[1]),
                 zaxis=dict(range=rng[2]))
    fig.update_layout(title=title, margin=dict(l=0, r=0, t=40, b=0),
                      scene=scene, scene2=scene)
    fig.write_html(out_path, post_script=_SYNC_CAMERAS)


def best_over_candidates(cands, tracks, args):
    """Try each ranked vertex candidate (snap + grow); keep the one that builds
    the best interaction. This is the iterative seeding: a candidate that attaches
    nothing is passed over for the next. Objective = (#attached, primary prongs,
    peak score). Returns the winning dict (or None)."""
    best = None
    for rank, (cpos, csc) in enumerate(cands):
        seed, shift, n = (snap_vertex(cpos, tracks, args.snap_radius)
                          if not args.no_snap else (np.asarray(cpos, float), 0.0, 0))
        res = reco_interaction(seed, tracks, d_vertex=args.d_vertex,
                               d_perp=args.d_perp, front_tol=args.front_tol,
                               merge_radius=args.merge_radius)
        n_att = sum(T["attached"] for T in tracks)
        prong = len(res["vertices"][0]["attached"])
        key = (n_att, prong, csc)
        if best is None or key > best["key"]:
            best = dict(key=key, seed=seed, shift=shift, n_snap=n,
                        rank=rank, score=csc, cand=np.asarray(cpos, float),
                        n_cand=len(cands))
    if best is not None:                             # re-grow the winner so the
        best["res"] = reco_interaction(              # shared `tracks` reflect it
            best["seed"], tracks, d_vertex=args.d_vertex, d_perp=args.d_perp,
            front_tol=args.front_tol, merge_radius=args.merge_radius)
        best["res"]["seed_input"] = best["cand"]
    return best


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    here = os.path.dirname(os.path.abspath(__file__))
    dev = os.path.join(here, "..", "reco_dev_data")
    ap.add_argument("--keypoint2-dir", default=os.path.join(dev, "keypoint2_out"))
    ap.add_argument("--merged-sp-dir", default=os.path.join(dev, "merged_sp"))
    ap.add_argument("--out-dir", default=os.path.join(here, "nu_int_out"))
    ap.add_argument("--vertex-source", choices=["reco", "pred", "gt"],
                    default="reco",
                    help="reco = ranked candidates from the score-field fitter "
                         "(reco.KeypointRecoTorch), iterated; pred = single "
                         "nu_vertex_cm centroid decode; gt = MC truth vertex")
    # attachment params
    ap.add_argument("--d-vertex", type=float, default=12.0,
                    help="endpoint->vertex max [cm]; loose to absorb the ~5-10cm "
                         "reconstructed track-start scatter")
    ap.add_argument("--d-perp", type=float, default=4.0)
    ap.add_argument("--seg-cm", type=float, default=3.0)
    ap.add_argument("--front-tol", type=float, default=1.5)
    ap.add_argument("--merge-radius", type=float, default=3.0)
    ap.add_argument("--no-snap", action="store_true",
                    help="disable seed-vertex refinement to nearby track ends")
    ap.add_argument("--snap-radius", type=float, default=30.0,
                    help="max dist [cm] for snap to pull track endpoints in")
    # track-reco params (passed to cluster_fit_stitch)
    ap.add_argument("--eps", type=float, default=1.2)
    ap.add_argument("--max-gap", type=float, default=20.0)
    ap.add_argument("--min-points", type=int, default=20)
    ap.add_argument("--no-html", action="store_true")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.keypoint2_dir, "*.h5")))
    msp = (args.merged_sp_dir if args.merged_sp_dir
           and os.path.isdir(args.merged_sp_dir) else None)
    os.makedirs(args.out_dir, exist_ok=True)
    print(f">>> {len(files)} events | vertex-source={args.vertex_source}")

    for fp in files:
        recs = tio.load_instances(fp, msp, tracks_only=True,
                                  min_points=args.min_points)
        if not recs:
            continue
        pred_nu, gt_nu = read_nu_vertices(fp)
        if args.vertex_source == "reco":
            cands = vertex_candidates(fp)
        elif args.vertex_source == "pred":
            cands = [(pred_nu, 1.0)] if np.all(np.isfinite(pred_nu)) else []
        else:
            cands = [(gt_nu, 1.0)] if np.all(np.isfinite(gt_nu)) else []
        if not cands:
            print(f"  {os.path.basename(fp)}: no {args.vertex_source} vertex, skip")
            continue
        tracks = build_tracks(recs, seg_cm=args.seg_cm, eps=args.eps,
                              max_gap_live=args.max_gap)
        best = best_over_candidates(cands, tracks, args)
        res = best["res"]
        n_att = sum(T["attached"] for T in tracks)
        primary = res["vertices"][0]
        n_sec = sum(1 for v in res["vertices"]
                    if v["depth"] > 0 and len(v["attached"]) > 0)
        # accuracy of the chosen (snapped) primary vs GT nu vertex
        verr = (np.linalg.norm(primary["pos"] - gt_nu)
                if np.all(np.isfinite(gt_nu)) else float("nan"))
        tag = os.path.basename(fp)
        print(f"  {tag}: {len(tracks)} tracks | cand {best['rank']+1}/"
              f"{best['n_cand']} (score {best['score']:.2f}) snap "
              f"{best['shift']:.1f}cm | primary {len(primary['attached'])} prong, "
              f"{n_att}/{len(tracks)} attached, {n_sec} sec-vtx, "
              f"{len(res['unattached'])} unatt | vtx-err vs GT={verr:.1f}cm")
        if not args.no_html:
            title = (f"{tag} | V={args.vertex_source} cand{best['rank']+1}/"
                     f"{best['n_cand']} snap{best['shift']:.0f}cm | "
                     f"{len(primary['attached'])}-prong, {n_att}/{len(tracks)} att, "
                     f"{n_sec} sec-vtx, {len(res['unattached'])} unatt | "
                     f"vtx-err={verr:.1f}cm")
            op = os.path.join(args.out_dir,
                              f"{os.path.splitext(tag)[0]}_nuint.html")
            visualize(res, gt_nu, args.vertex_source, title, op)
    if not args.no_html:
        print(f">>> HTML -> {args.out_dir}")


if __name__ == "__main__":
    main()
