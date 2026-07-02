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
from shower_trunk import trunk_vertex_biased
from shower_connect import connects
from shower_truth import load_shower_fragments, shower_is_primary
from range_momentum import RangeMomentum
from particle_momentum import assign_momenta, load_shower_calib, _entry


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


def build_tracks(recs, seg_cm=3.0, kink_tol=3.0, **cfg):
    """Reconstruct each track-class instance: two ends + dirs, plus interior
    KINKS (constant-tolerance RDP break points on the centerline) — these become
    extra connection points where mid-track scatters can seed a shower (item b)."""
    tracks = []
    for tid, rec in enumerate(recs):
        cfs = cluster_fit_stitch(rec.points, weights=rec.weights,
                                 rdp_tol=(lambda L: kink_tol), **cfg)
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
            kinks=np.asarray(cfs.get("kink_vertices", np.zeros((0, 3))), np.float64),
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


def snap_vertex(V0, tracks, radius, conv_radius=5.0, support_radius=5.0):
    """Refine the seed vertex ONLY when it is UNSUPPORTED.

    The score-field fitter seed is usually accurate (~1-2 cm), while reconstructed
    track STARTS scatter ~5-10 cm, so snapping a good seed to the track-start
    convergence centroid degrades it. So: if any track endpoint already lies
    within `support_radius` of V0, the seed sits on a real track start -> leave it
    untouched. Only a seed with NO nearby track endpoint (a badly placed seed,
    tens of cm off) is snapped to the densest cross-track endpoint convergence
    within `radius` (a real interaction point, not merely the nearest endpoint).
    Returns (snapped, shift, n_in_cluster); shift=0 when the seed is kept."""
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
    # supported: the seed already sits on a track start -> trust the fitter, keep
    if float(np.linalg.norm(pos - V0, axis=1).min()) <= support_radius:
        return V0, 0.0, 0
    # unsupported: snap to the densest cross-track endpoint convergence
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
# Phase 2: attach showers to interaction connection points
# ---------------------------------------------------------------------------
def reco_showers(shower_recs, conn_points, mode="greedy", d_impact=10.0,
                 cos_min=0.9, d_gap=60.0, gap_touch=3.0):
    """Attach predicted shower instances to interaction connection points.

    `conn_points`: list of {vid, pos, dist} (the nu vertex + track endpoints from
    the interaction tree), sorted closest-to-nu-vertex first. For each (shower,
    connection point) the trunk is re-fit biased toward that point
    (`trunk_vertex_biased`, the Phase-1 winner, faithful to LANTERN which makes
    the trunk relative to the vertex under test), then tested with `connects`
    (impact + back-pointing cosine; showers attach across a gap).

    mode='greedy': walk connection points closest-first, attach each shower to the
    FIRST point it points back to. mode='exhaustive': test every point, attach to
    the best (smallest impact). Returns a list of shower records."""
    showers = [dict(inst=r.inst_idx, cls=r.pred_cls, cls_name=r.pred_cls_name,
                    gt_trackid=r.gt_trackid, points=r.points,
                    truth_cloud=getattr(r, "truth_cloud", None),
                    pred_start=r.pred_start, gt_start=r.gt_start,
                    attached=False, cp_id=None, cp_pos=None, cp_kind=None,
                    trunk=None, geom=None) for r in shower_recs]
    if not conn_points:
        return showers

    def _attach(sh, cp, tk, g):
        sh.update(attached=True, cp_id=cp["id"], cp_pos=cp["pos"],
                  cp_kind=cp["kind"], trunk=tk, geom=g)

    if mode == "greedy":
        for cp in conn_points:                       # closest-to-nu-vtx first
            for sh in showers:
                if sh["attached"]:
                    continue
                tk = trunk_vertex_biased(sh["points"], cp["pos"])
                ok, g = connects(tk.start, tk.direction, cp["pos"],
                                 d_impact=d_impact, cos_min=cos_min, d_gap=d_gap,
                                 gap_touch=gap_touch)
                if ok:
                    _attach(sh, cp, tk, g)
    else:                                            # exhaustive
        for sh in showers:
            best = None
            for cp in conn_points:
                tk = trunk_vertex_biased(sh["points"], cp["pos"])
                ok, g = connects(tk.start, tk.direction, cp["pos"],
                                 d_impact=d_impact, cos_min=cos_min, d_gap=d_gap,
                                 gap_touch=gap_touch)
                if ok and (best is None or g["impact"] < best[0]):
                    best = (g["impact"], cp, tk, g)
            if best is not None:
                _attach(sh, best[1], best[2], best[3])

    for sh in showers:                               # default trunk for the viz
        if sh["trunk"] is None:
            sh["trunk"] = trunk_vertex_biased(sh["points"], conn_points[0]["pos"])
    return showers


def connection_points(res, tracks):
    """Shower connection points for one interaction = its tree vertices (nu vtx +
    track endpoints/junctions) + interior kinks of its attached tracks, sorted
    closest-to-the-interaction-vertex first."""
    nuv0 = res["vertices"][0]["pos"]
    cps = [dict(id=f"v{v['id']}", pos=v["pos"], kind="vertex")
           for v in res["vertices"]]
    for T in tracks:
        if T["attached"]:
            for ki, kp in enumerate(T.get("kinks", [])):
                cps.append(dict(id=f"k{T['id']}_{ki}", pos=kp, kind="kink"))
    for c in cps:
        c["dist"] = float(np.linalg.norm(c["pos"] - nuv0))
    cps.sort(key=lambda c: c["dist"])
    return cps


def reco_interactions(cands, tracks, shower_recs, args):
    """Iterative multi-vertex reco. Build the best interaction, then RE-RUN on the
    leftover (unassociated) tracks+showers with candidates away from the vertices
    already used — forming additional nu-vertex candidates. This recovers the true
    interaction when the score-field fitter latched onto a displaced (e.g.
    hadronic-secondary) vertex: the true vertex's particles are left over and a
    later iteration reconstructs them. Continues while >=1 unassociated particle
    has >= reco_min_unassoc_points (and up to max_interactions). The analyzer
    chooses which interaction is THE nu vertex; the rest are displaced secondaries.

    Returns (interactions, leftover_tracks, leftover_shrec)."""
    rem_tracks = list(tracks)
    rem_shrec = list(shower_recs)
    used = []
    out = []
    sk = dict(d_impact=args.shower_d_impact, cos_min=args.shower_cos_min,
              d_gap=args.shower_d_gap,
              gap_touch=getattr(args, "shower_gap_touch", 3.0))
    for it in range(args.max_interactions):
        pool = [c for c in cands if all(
            np.linalg.norm(np.asarray(c[0], float) - u) > args.reco_exclude_radius
            for u in used)]
        if not pool or not rem_tracks:
            break
        best = best_over_candidates(pool, rem_tracks, args)
        if best is None or sum(T["attached"] for T in rem_tracks) == 0:
            break
        res = best["res"]
        cps = connection_points(res, rem_tracks)
        showers = reco_showers(rem_shrec, cps, args.shower_mode, **sk)
        res["showers"] = showers
        out.append(dict(res=res, best=best, iter=it, conn_points=cps,
                        vertex=np.asarray(res["vertices"][0]["pos"], float),
                        tracks=[T for T in rem_tracks if T["attached"]],
                        showers=showers))
        used.append(np.asarray(res["vertices"][0]["pos"], float))
        rem_tracks = [T for T in rem_tracks if not T["attached"]]
        rem_shrec = [r for r, s in zip(rem_shrec, showers) if not s["attached"]]
        n_unassoc = (sum(len(T["points"]) >= args.reco_min_unassoc_points
                         for T in rem_tracks)
                     + sum(r.n_points >= args.reco_min_unassoc_points
                           for r in rem_shrec))
        if n_unassoc == 0:
            break
    return out, rem_tracks, rem_shrec


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
# GT particle gathering (for the "what reco misses" panel)
# ---------------------------------------------------------------------------
_PDG_NAME = {11: "e", -11: "e", 22: "gamma", 13: "mu", -13: "mu",
             211: "pi", -211: "pi", 111: "pi0", 2212: "p", 2112: "n"}
_TYPE_COLORS = {"e": "#d62728", "gamma": "#ff7f0e", "mu": "#1f77b4",
                "pi": "#2ca02c", "p": "#9467bd", "n": "#8c564b",
                "pi0": "#e377c2", "other": "#7f7f7f"}


def _ptype(pid, cls_name):
    """True particle type from PID when available (merged_sp present), else the
    predicted-class name as a proxy."""
    return _PDG_NAME.get(int(pid), cls_name) if pid else cls_name


def gather_gt_particles(all_recs, reco_trackids):
    """One entry per TRUE particle (dedup by gt_trackid) from ALL instances —
    tracks AND showers — so the GT panel shows the full true ionization, not just
    what the track reco used. `status`: 'reco' if the particle was reconstructed
    and attached to the interaction, else 'missed'. Skips the unassigned/neutrino
    catch-all (gt_trackid <= 0)."""
    seen = {}
    for r in all_recs:
        t = int(r.gt_trackid)
        G = getattr(r, "truth_cloud", None)
        if t <= 0 or G is None or len(G) == 0 or t in seen:
            continue
        seen[t] = dict(trackid=t, cloud=np.asarray(G, np.float32),
                       typ=_ptype(int(getattr(r, "pid", 0) or 0), r.pred_cls_name),
                       is_track=r.is_track,
                       status="reco" if t in reco_trackids else "missed")
    return list(seen.values())


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


# per-interaction colors: tracks/secondary vertices, and the primary vertex marker
_INT_COLORS = ["#1f77b4", "#2ca02c", "#9467bd", "#8c564b", "#17becf"]
_INT_VTX_COLORS = ["gold", "darkorange", "cyan", "magenta", "lime"]


def visualize(interactions, leftover_tracks, leftover_shrec, gt_nu, title,
              out_path, gt_particles=None):
    """LEFT: one or more reconstructed interactions (nu-vertex candidates), each a
    distinct color; leftover/unassociated particles in gray. RIGHT: all true
    particles by PID. The analyzer picks which interaction is THE nu vertex."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    fig = make_subplots(
        rows=1, cols=2, specs=[[{"type": "scene"}, {"type": "scene"}]],
        horizontal_spacing=0.01,
        subplot_titles=("reco interactions (nu-cand 0=gold,1=orange,2=cyan)",
                        "true ionization (all GT particles; ✗=missed by reco)"))
    bbox = []

    # ---- LEFT (col 1): each interaction in its own color ----
    for I in interactions:
        res = I["res"]
        ci = I["iter"]
        col = _INT_COLORS[ci % len(_INT_COLORS)]
        tracks_by_id = {T["id"]: T for T in res["tracks"]}
        for T in I["tracks"]:                        # attached tracks only
            P, poly = T["points"], T["poly"]
            bbox.append(P)
            fig.add_trace(go.Scatter3d(
                x=P[:, 0], y=P[:, 1], z=P[:, 2], mode="markers",
                marker=dict(size=1.6, color=col, opacity=0.35),
                showlegend=False, name=f"i{ci} t{T['id']}"), row=1, col=1)
            mom = T.get("mom")
            pstr = (f" p={float(np.linalg.norm(mom['momentum'])):.0f}MeV"
                    if mom and "momentum" in mom else "")
            fig.add_trace(go.Scatter3d(
                x=poly[:, 0], y=poly[:, 1], z=poly[:, 2], mode="lines",
                line=dict(color=col, width=5),
                name=f"i{ci} t{T['id']} {T['cls_name']} L={T['length']:.0f}cm{pstr}"),
                row=1, col=1)
        for e in res["edges"]:                       # bridges
            V = res["vertices"][e["vertex"]]["pos"]
            ep = tracks_by_id[e["track"]]["ends"][e["end"]]["pos"]
            fig.add_trace(go.Scatter3d(
                x=[V[0], ep[0]], y=[V[1], ep[1]], z=[V[2], ep[2]], mode="lines",
                line=dict(color="black", width=2, dash="dot"),
                showlegend=False), row=1, col=1)
        for v in res["vertices"]:                    # vertices
            nprong = len(v["attached"])
            if not nprong and v["depth"] > 0:
                continue
            is_primary = v["depth"] == 0
            fig.add_trace(go.Scatter3d(
                x=[v["pos"][0]], y=[v["pos"][1]], z=[v["pos"][2]], mode="markers",
                marker=dict(size=13 if is_primary else 6 + 2 * nprong,
                            color=(_INT_VTX_COLORS[ci % len(_INT_VTX_COLORS)]
                                   if is_primary else col),
                            symbol="diamond", line=dict(width=1, color="black")),
                name=(f"nu-cand{ci} ({nprong}p)" if is_primary
                      else f"i{ci} V{v['id']} {nprong}p")), row=1, col=1)
        for sh in I["showers"]:                      # attached showers only
            if not sh["attached"]:
                continue
            tk = sh["trunk"]
            P = sh["points"]
            bbox.append(P)
            fig.add_trace(go.Scatter3d(
                x=P[:, 0], y=P[:, 1], z=P[:, 2], mode="markers",
                marker=dict(size=1.4, color="#d62728", opacity=0.3),
                showlegend=False, name=f"i{ci} sh{sh['inst']}"), row=1, col=1)
            tip = tk.start + tk.direction * 15.0
            g = sh.get("geom") or {}
            mom = sh.get("mom")
            estr = (f" E={mom['ke_calo']:.0f}MeV"
                    if mom and np.isfinite(mom.get("ke_calo", np.nan)) else "")
            fig.add_trace(go.Scatter3d(
                x=[tk.start[0], tip[0]], y=[tk.start[1], tip[1]],
                z=[tk.start[2], tip[2]], mode="lines",
                line=dict(color="#d62728", width=6),
                name=f"i{ci} sh{sh['inst']} {sh['cls_name']}{estr} @{sh['cp_kind']} "
                     f"cos={g.get('cosine',0):.2f}"),
                row=1, col=1)
            cp = sh["cp_pos"]
            fig.add_trace(go.Scatter3d(
                x=[cp[0], tk.start[0]], y=[cp[1], tk.start[1]],
                z=[cp[2], tk.start[2]], mode="lines",
                line=dict(color="#d62728", width=2, dash="dash"),
                showlegend=False), row=1, col=1)

    # leftover / unassociated particles (gray)
    for T in leftover_tracks:
        P = T["points"]
        bbox.append(P)
        fig.add_trace(go.Scatter3d(
            x=P[:, 0], y=P[:, 1], z=P[:, 2], mode="markers",
            marker=dict(size=1.4, color="lightgray", opacity=0.4),
            name=f"unassoc t{T['id']} {T['cls_name']}"), row=1, col=1)
    for r in leftover_shrec:
        P = r.points
        bbox.append(P)
        fig.add_trace(go.Scatter3d(
            x=P[:, 0], y=P[:, 1], z=P[:, 2], mode="markers",
            marker=dict(size=1.4, color="lightgray", opacity=0.4),
            name=f"unassoc sh{r.inst_idx} {r.pred_cls_name}"), row=1, col=1)

    # ---- RIGHT (col 2): ALL true particles by PID; MISSED ones as ✗ ----
    for gp in (gt_particles or []):
        G = gp["cloud"]
        if G is None or len(G) == 0:
            continue
        bbox.append(G)
        missed = gp["status"] == "missed"
        col = _TYPE_COLORS.get(gp["typ"], _TYPE_COLORS["other"])
        fig.add_trace(go.Scatter3d(
            x=G[:, 0], y=G[:, 1], z=G[:, 2], mode="markers",
            marker=dict(size=2.6 if missed else 1.8,
                        color=col, opacity=0.8 if missed else 0.45,
                        symbol="x" if missed else "circle"),
            name=f"{gp['typ']} trk{gp['trackid']} "
                 + ("MISSED" if missed else "reco")), row=1, col=2)

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
        seed, shift, n = (
            snap_vertex(cpos, tracks, args.snap_radius,
                        support_radius=args.snap_support_radius)
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
    ap.add_argument("--snap-support-radius", type=float, default=5.0,
                    help="snap ONLY if no track endpoint is within this [cm] of "
                         "the seed (i.e. only refine an unsupported seed; leave a "
                         "good fitter seed sitting on a track start untouched)")
    # track-reco params (passed to cluster_fit_stitch)
    ap.add_argument("--eps", type=float, default=1.2)
    ap.add_argument("--max-gap", type=float, default=20.0)
    ap.add_argument("--min-points", type=int, default=20)
    # shower attachment (Phase 2)
    ap.add_argument("--shower-mode", choices=["greedy", "exhaustive", "off"],
                    default="greedy", help="how showers attach to connection "
                    "points (greedy=closest-first; exhaustive=best over all)")
    # balanced point from scan_shower_attach.py (P=0.98, R=0.80); tighten to
    # (10, 0.80) for P=1.00/R=0.72, or loosen to (30, 0.50) for R=0.85/P=0.96.
    ap.add_argument("--shower-d-impact", type=float, default=15.0)
    ap.add_argument("--shower-cos-min", type=float, default=0.80)
    ap.add_argument("--shower-d-gap", type=float, default=60.0)
    ap.add_argument("--shower-min-points", type=int, default=20)
    ap.add_argument("--kink-tol", type=float, default=3.0,
                    help="constant RDP tolerance [cm] for track kinks used as "
                         "extra shower connection points (item b)")
    # iterative multi-vertex reco (handle displaced/secondary vertices)
    ap.add_argument("--max-interactions", type=int, default=3,
                    help="max nu-vertex candidates per event (re-run on leftover "
                         "slice after excluding used particles/vertices)")
    ap.add_argument("--reco-exclude-radius", type=float, default=10.0,
                    help="exclude vertex candidates within this [cm] of an "
                         "already-used interaction vertex")
    ap.add_argument("--reco-min-unassoc-points", type=int, default=30,
                    help="re-iterate only if an unassociated particle has at "
                         "least this many points")
    ap.add_argument("--no-html", action="store_true")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.keypoint2_dir, "*.h5")))
    msp = (args.merged_sp_dir if args.merged_sp_dir
           and os.path.isdir(args.merged_sp_dir) else None)
    os.makedirs(args.out_dir, exist_ok=True)
    print(f">>> {len(files)} events | vertex-source={args.vertex_source} "
          f"| shower-mode={args.shower_mode}")
    tot = dict(show=0, g=0, tp=0, fp=0, fn=0, tn=0, nprim=0, att_vtx=0,
               att_kink=0, nkink=0, nint=0, nev=0, recov=0, recov1=0)
    rmom = RangeMomentum()                    # range->KE tables (tracks)
    shower_calib = load_shower_calib()        # KE = a*Q_comb (showers); {} if none

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
        tag = os.path.basename(fp)
        tracks = build_tracks(recs, seg_cm=args.seg_cm, kink_tol=args.kink_tol,
                              eps=args.eps, max_gap_live=args.max_gap)
        all_recs = tio.load_instances(fp, msp, tracks_only=False, min_points=1)
        shower_recs = [r for r in all_recs if r.pred_cls in tio.SHOWER_CLASSES
                       and np.all(np.isfinite(r.pred_start))
                       and r.n_points >= args.shower_min_points]
        # iterative multi-vertex reco: build interactions on the slice, re-running
        # on the leftover after excluding each used vertex's particles.
        interactions, lo_tracks, lo_shrec = reco_interactions(
            cands, tracks, shower_recs, args)
        if not interactions:
            print(f"  {tag}: no interaction formed, skip")
            continue
        # 4-momentum per particle (tracks: range; showers: calo). Attaches
        # obj["mom"] onto each track/shower; needs merged_sp image charge for calo.
        mom_entry, mom_fh = _entry(fp, msp)
        try:
            assign_momenta(interactions, mom_entry, rmom, shower_calib)
        finally:
            if mom_fh is not None:
                mom_fh.close()
        n_trk_att = sum(len(I["tracks"]) for I in interactions)
        # ONE record per shower instance: the interaction where it attached (first
        # wins), so a shower passing through several interactions isn't counted
        # repeatedly.
        att_by_inst = {}
        for I in interactions:
            for s in I["showers"]:
                if s["attached"] and s["inst"] not in att_by_inst:
                    att_by_inst[s["inst"]] = s
        n_shw_att = len(att_by_inst)

        # --- (a) provenance truth: shower attached AT the true nu vertex? ---
        frag = load_shower_fragments(fp, msp)
        for r in shower_recs:                        # one per instance
            prim, _, _ = shower_is_primary(r.gt_start, frag, gt_nu)
            if prim is None:
                continue
            tot["nprim"] += int(prim)
            s = att_by_inst.get(r.inst_idx)
            at_nuvtx = (s is not None and s["cp_pos"] is not None
                        and np.all(np.isfinite(gt_nu))
                        and np.linalg.norm(s["cp_pos"] - gt_nu) <= 5.0)
            tot["tp" if (at_nuvtx and prim) else "fp" if at_nuvtx else
                "fn" if prim else "tn"] += 1
        tot["show"] += len(shower_recs)
        tot["g"] += n_shw_att
        tot["nkink"] += sum(1 for I in interactions for c in I["conn_points"]
                            if c["kind"] == "kink")
        tot["att_vtx"] += sum(1 for s in att_by_inst.values()
                              if s["cp_kind"] == "vertex")
        tot["att_kink"] += sum(1 for s in att_by_inst.values()
                               if s["cp_kind"] == "kink")

        reco_tids = {T["gt_trackid"] for I in interactions for T in I["tracks"]
                     if T["gt_trackid"] > 0}
        reco_tids |= {s["gt_trackid"] for s in att_by_inst.values()
                      if s["gt_trackid"] > 0}
        gt_particles = gather_gt_particles(all_recs, reco_tids)
        n_miss = sum(1 for gp in gt_particles if gp["status"] == "missed")

        # per-interaction vtx-err; "recovery" = any interaction within 5cm of GT
        verrs = [float(np.linalg.norm(I["vertex"] - gt_nu))
                 if np.all(np.isfinite(gt_nu)) else float("nan")
                 for I in interactions]
        best_verr = min(verrs) if verrs and np.all(np.isfinite(gt_nu)) else float("nan")
        tot["nint"] += len(interactions)
        if np.all(np.isfinite(gt_nu)):
            tot["nev"] += 1
            tot["recov"] += int(best_verr <= 5.0)
            tot["recov1"] += int(verrs[0] <= 5.0)        # 1st interaction only
        def _evis(I):
            objs = list(I["tracks"]) + [s for s in I["showers"] if s["attached"]]
            return sum(o["mom"]["energy"] for o in objs
                       if o.get("mom") and np.isfinite(o["mom"].get("energy", np.nan)))
        ipart = " ".join(f"int{I['iter']}:{len(I['tracks'])}p/v{ve:.0f}/Evis{_evis(I):.0f}"
                         for I, ve in zip(interactions, verrs))
        print(f"  {tag}: {len(tracks)}trk {len(shower_recs)}shw | "
              f"{len(interactions)} interaction(s) [{ipart}] | "
              f"{n_trk_att} trk-att, showers {n_shw_att}/{len(shower_recs)} att | "
              f"{n_miss} GT missed | best vtx-err={best_verr:.1f}cm")
        if not args.no_html:
            title = (f"{tag} | {len(interactions)} nu-cand(s) | "
                     f"{n_trk_att} trk, {n_shw_att}/{len(shower_recs)} shw att | "
                     f"{n_miss} GT missed | best vtx-err={best_verr:.1f}cm")
            op = os.path.join(args.out_dir,
                              f"{os.path.splitext(tag)[0]}_nuint.html")
            visualize(interactions, lo_tracks, lo_shrec, gt_nu, title, op,
                      gt_particles=gt_particles)
    print(f"\n=== interactions: {tot['nint']} over {tot['nev']} events | "
          f"true-vertex recovered (<=5cm) by 1st interaction {tot['recov1']}/"
          f"{tot['nev']}, by ANY interaction {tot['recov']}/{tot['nev']} "
          f"(iteration gain +{tot['recov']-tot['recov1']}) ===")
    print(f"=== showers: {tot['show']} total | attached {tot['g']} "
          f"({100*tot['g']/max(tot['show'],1):.0f}%) | kinks added {tot['nkink']} | "
          f"attach by kind: vertex {tot['att_vtx']}, kink {tot['att_kink']} ===")
    ntruth = tot["tp"] + tot["fp"] + tot["fn"] + tot["tn"]
    if ntruth:
        prec = tot["tp"] / max(tot["tp"] + tot["fp"], 1)
        rec = tot["tp"] / max(tot["tp"] + tot["fn"], 1)
        print(f"=== nu-vertex attachment vs provenance truth (origin<=5cm of true "
              f"vtx): {ntruth} showers w/ truth, {tot['nprim']} primary | "
              f"precision={prec:.2f} recall={rec:.2f} "
              f"(TP={tot['tp']} FP={tot['fp']} FN={tot['fn']} TN={tot['tn']}) ===")
    if not args.no_html:
        print(f">>> HTML -> {args.out_dir}")


if __name__ == "__main__":
    main()
