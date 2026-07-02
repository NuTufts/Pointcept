"""Cascade / nu-reco RECO-PERFORMANCE visualizer (reads output files, no models).

Purpose: understand *which true spacepoints of which particles were missed, and
at what stage* of the reco. Unlike the GT-through-levels visualizer this was
forked from, it loads nothing through the training dataloader and never runs the
deghoster/slicer/segmenter — it reads what those stages already wrote to disk:

  stage            product (on disk)                         what we read
  ---------------  ----------------------------------------  ----------------------
  truth / input    merged_sp  entry_0/triplet_data           full spacepoint cloud
                                                              + per-point truth
                                                              (trackid, origin, pid)
                   merged_sp  entry_0/mc_particle_tree        per-particle pid / KE
  deghost+slice    keypoint2_event*.h5  slice/coord_cm        the surviving nu slice
  segmenter        keypoint2_event*.h5  particle/{i}          per-instance masks/cls
  nu reco          nu_reco_shard*.h5  event_{gidx}            attached particles + p

Pick an event by its **merged_sp** file (or by index into the merged_sp list).
The downstream files are found automatically:
  * cascade: the dataset SORTS its file list, so merged_sp at sorted-index i ->
    output/.../keypoint2_event{i:05d}_*.h5 (verified against the file's src_file
    attr; falls back to a directory scan on mismatch). Events with no nu slice
    have no cascade file (fully missed by the slicer).
  * nu reco: gidx = line index of the cascade file in the keypoint2 list; the
    shard holding event_{gidx:07d} is found from each shard's start/n attrs.

Per triplet_data point we derive a STAGE:
  missed_preslice  — a true nu point that never made it into the slice
  in_slice_unseg   — in the slice but no predicted instance claimed it
  segmented        — in the slice and inside a predicted instance
and per particle whether nu-reco attached it. Color modes expose each stage.

Run inside the container, e.g.
    ./run_in_tufts_pointcept_container.sh python \
        lartpc_data_prep/larformer_keypoint_v2/visualize_cascade_output.py \
        --merged-sp /cluster/.../merged_..._fileno00005_entry000001.h5
then open http://<host>:8050 .  Pure h5py + numpy + scipy + dash/plotly.
"""

import argparse
import colorsys
import glob
import os
import sys

import numpy as np
import h5py
from scipy.spatial import cKDTree

from dash import Dash, Input, Output, State, dcc, html
import plotly.graph_objects as go

# detectoroutline lives one dir up (lartpc_data_prep/).
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
from detectoroutline import DetectorOutline  # noqa: E402

# reuse the reco's de-double-counted calorimetric charge (trajfit_dev/calo.py):
# each wire pixel's ADC is split among the spacepoints sharing it, so summing the
# per-point charge over a set == that set's unique-pixel charge. `comb` = Y plane
# where present else mean(U,V) -- the same quantity the shower energy reco uses.
sys.path.insert(0, os.path.join(_HERE, "trajfit_dev"))
from calo import dedup_charge  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CLASS_NAMES = ["e", "gamma", "mu", "pi", "p", "other", "(unused)", "no_object"]
PID2NAME = {11: "e", -11: "e", 22: "gamma", 13: "mu", -13: "mu",
            211: "pi", -211: "pi", 2212: "p"}
MASS = {"mu": 105.6584, "pi": 139.5704, "p": 938.2721}   # for reco KE of tracks
ORIGIN_NU, ORIGIN_COSMIC = 1, 2

# default file locations (relative to this script)
DEF_CASCADE_DIR = os.path.join(_HERE, "output", "valdata_all_with_score_maps")
DEF_MSP_LIST = os.path.join(_HERE, "inputlists", "merged_sp_valdata_all.txt")
DEF_KP_LIST = os.path.join(_HERE, "outputlists", "keypoint2_out_valdata_all.txt")
DEF_NURECO_DIR = os.path.join(_HERE, "output", "nu_reco_valdata_all")

# stage -> (label, rgba)
STAGE_STYLE = {
    "segmented":       ("segmented (kept)",     "rgba(60,200,90,1)"),
    "in_slice_unseg":  ("in slice, unsegmented", "rgba(255,175,40,1)"),
    "missed_preslice": ("missed pre-slice",     "rgba(240,50,50,1)"),
}


# ---------------------------------------------------------------------------
# Color helpers (kept from the fork's style)
# ---------------------------------------------------------------------------
def track_color(tid: int) -> str:
    """Golden-ratio hash on |trackid| -> a stable, well-spread color."""
    h = (abs(int(tid)) * 0.61803398875) % 1.0
    r, g, b = colorsys.hsv_to_rgb(h, 0.80, 0.95)
    return f"rgba({int(r*255)},{int(g*255)},{int(b*255)},1)"


def cls_color(c: int) -> str:
    """Stable per predicted-class color. <0 -> light gray."""
    if c is None or c < 0:
        return "rgba(180,180,180,0.6)"
    palette = [
        "rgba(255,60,60,1)",     # 0 e
        "rgba(60,140,255,1)",    # 1 gamma
        "rgba(80,200,80,1)",     # 2 mu
        "rgba(255,160,40,1)",    # 3 pi
        "rgba(180,60,220,1)",    # 4 p
        "rgba(40,200,200,1)",    # 5 other
    ]
    return palette[c % len(palette)]


def reco_ke(energy, cls_idx):
    name = CLASS_NAMES[cls_idx] if 0 <= cls_idx < len(CLASS_NAMES) else "other"
    return float(energy) - MASS.get(name, 0.0) if name in MASS else float(energy)


def _xyz(coord):
    """Detector (x,y,z) cm -> plotly (x=z, y=x, z=y) to match the detector box."""
    coord = np.asarray(coord)
    if coord.ndim == 1:
        return [coord[2]], [coord[0]], [coord[1]]
    return coord[:, 2], coord[:, 0], coord[:, 1]


# ---------------------------------------------------------------------------
# File resolution (merged_sp -> cascade -> nu_reco)
# ---------------------------------------------------------------------------
def read_list(path):
    with open(path) as f:
        return [ln.strip() for ln in f if ln.strip()]


def _attr_str(attrs, key):
    v = attrs.get(key, "")
    return v.decode() if isinstance(v, bytes) else v


class Context:
    """Holds the lists / indices shared across events (built once)."""

    def __init__(self, cascade_dir, msp_list_path, kp_list_path, nureco_dir):
        self.cascade_dir = cascade_dir
        self.nureco_dir = nureco_dir
        self.msp_sorted = sorted(read_list(msp_list_path)) if \
            os.path.exists(msp_list_path) else []
        self.msp_index = {os.path.basename(p): i
                          for i, p in enumerate(self.msp_sorted)}
        # gidx = line index of a cascade file in the keypoint2 list
        self.kp_gidx = {}
        if os.path.exists(kp_list_path):
            for gidx, p in enumerate(read_list(kp_list_path)):
                self.kp_gidx[os.path.basename(p)] = gidx
        # nu_reco shard ranges: [(path, start, n_requested), ...]
        self.shards = []
        for sp in sorted(glob.glob(os.path.join(nureco_dir, "*.h5"))):
            try:
                with h5py.File(sp, "r") as f:
                    self.shards.append(
                        (sp, int(f.attrs.get("shard_start", -1)),
                         int(f.attrs.get("n_requested", 0))))
            except Exception:
                continue
        self._cascade_scan = None   # lazy {src_basename: path}

    # -- cascade -----------------------------------------------------------
    def _scan_cascade(self):
        if self._cascade_scan is not None:
            return self._cascade_scan
        print(">>> scanning cascade dir for src_file attrs (one-time) ...",
              flush=True)
        m = {}
        for p in glob.glob(os.path.join(self.cascade_dir, "keypoint2_event*.h5")):
            try:
                with h5py.File(p, "r") as f:
                    src = _attr_str(f.attrs, "src_file")
                if src:
                    m.setdefault(os.path.basename(src), p)
            except Exception:
                continue
        self._cascade_scan = m
        return m

    def cascade_file_for(self, msp_basename):
        """Return the keypoint2 file for a merged_sp basename (or None)."""
        i = self.msp_index.get(msp_basename)
        if i is not None:
            cands = sorted(glob.glob(os.path.join(
                self.cascade_dir, f"keypoint2_event{i:05d}_*.h5")))
            for c in cands:
                try:
                    with h5py.File(c, "r") as f:
                        if _attr_str(f.attrs, "src_file") == msp_basename:
                            return c
                except Exception:
                    continue
        return self._scan_cascade().get(msp_basename)

    # -- nu_reco -----------------------------------------------------------
    def nureco_event_for(self, cascade_basename):
        """Return (part_dict, vertices, attrs) for the cascade file, or None."""
        gidx = self.kp_gidx.get(cascade_basename)
        if gidx is None:
            return None
        grp_name = f"event_{gidx:07d}"
        for sp, start, n in self.shards:
            if start < 0 or not (start <= gidx < start + n):
                continue
            with h5py.File(sp, "r") as f:
                if grp_name not in f:
                    return None
                g = f[grp_name]
                out = {"attrs": {k: g.attrs[k] for k in g.attrs},
                       "gidx": gidx, "shard": os.path.basename(sp)}
                for k in g.keys():
                    out[k] = g[k][()]
                return out
        return None


# ---------------------------------------------------------------------------
# Event loading + per-spacepoint stage derivation
# ---------------------------------------------------------------------------
def load_event(ctx, msp_path):
    """Load truth + cascade + nu_reco for one event; derive per-point stage."""
    msp_base = os.path.basename(msp_path)
    ev = {"msp_path": msp_path, "msp_base": msp_base, "warnings": []}

    # ---- truth (merged_sp) ----
    with h5py.File(msp_path, "r") as f:
        e = f["entry_0"]
        td = e["triplet_data"]
        pos = np.asarray(td["pos"][()], np.float32)
        tid = np.asarray(td["trackid"][()], np.int64)
        origin = np.asarray(td["origin"][()], np.int64)
        pid = np.asarray(td["pid"][()], np.int64)
        # per-point wire charge (for the charge-based, over-count-corrected coverage)
        pixval = np.asarray(td["pixval"][()], np.float64)
        tick = np.asarray(td["tick"][()], np.int64)
        uw = np.asarray(td["uwire"][()], np.int64)
        vw = np.asarray(td["vwire"][()], np.int64)
        yw = np.asarray(td["ywire"][()], np.int64)
        mt = e["mc_particle_tree"]
        mc = {int(t): dict(pid=int(p), ke=float(k), origin=int(o))
              for t, p, k, o in zip(mt["trackid"][()], mt["pid"][()],
                                    mt["energy_mev"][()], mt["origin"][()])}
    ev.update(td_pos=pos, td_tid=tid, td_origin=origin, td_pid=pid, mc=mc,
              td_pixval=pixval, td_tick=tick, td_uw=uw, td_vw=vw, td_yw=yw)
    n = len(pos)

    # ---- cascade (deghost+slice + segmenter) ----
    kp_path = ctx.cascade_file_for(msp_base)
    ev["kp_path"] = kp_path
    in_slice = np.zeros(n, bool)
    seg = np.zeros(n, bool)
    pred_cls = np.full(n, -1, np.int64)
    particles = []
    ev["nu_vertex_cm"] = ev["gt_nu_vertex_cm"] = None
    if kp_path is not None:
        with h5py.File(kp_path, "r") as f:
            slice_coord = np.asarray(f["slice/coord_cm"][()], np.float32)
            np_ = int(f.attrs["n_particles"])
            for i in range(np_):
                g = f[f"particle/{i}"]
                particles.append(dict(
                    cls=int(g.attrs["cls"]),
                    gt_trackid=int(g.attrs["gt_trackid"]),
                    has_match=bool(g.attrs["has_match"]),
                    iou=float(g.attrs["iou"]),
                    point_idx=np.asarray(g["point_idx"][()], np.int64),
                    start_cm=np.asarray(g["start_cm"][()], np.float32),
                    end_cm=np.asarray(g["end_cm"][()], np.float32)))
            if "nu_vertex_cm" in f:
                ev["nu_vertex_cm"] = np.asarray(f["nu_vertex_cm"][()], np.float32)
            if "gt_nu_vertex_cm" in f:
                ev["gt_nu_vertex_cm"] = np.asarray(
                    f["gt_nu_vertex_cm"][()], np.float32)
        ev["slice_coord"] = slice_coord
        # per-slice-point segmentation + predicted class
        seg_slice = np.zeros(len(slice_coord), bool)
        cls_slice = np.full(len(slice_coord), -1, np.int64)
        for p in particles:
            pi = p["point_idx"]
            pi = pi[(pi >= 0) & (pi < len(slice_coord))]
            seg_slice[pi] = True
            cls_slice[pi] = p["cls"]
        # map every triplet_data point to its nearest slice point
        if len(slice_coord):
            d, nn = cKDTree(slice_coord).query(pos, k=1)
            in_slice = d < 0.05
            if in_slice.any() and np.median(d[in_slice]) > 0.05:
                ev["warnings"].append("slice<->triplet match >0.05 cm")
            seg = in_slice & seg_slice[nn]
            pc = cls_slice[nn]
            pred_cls = np.where(in_slice, pc, -1)
    else:
        ev["slice_coord"] = np.zeros((0, 3), np.float32)
        ev["warnings"].append("no cascade output (no nu slice found for event)")
    ev.update(in_slice=in_slice, seg=seg, pred_cls=pred_cls, particles=particles)

    # ---- nu reco ----
    ev["nureco"] = None
    ev["attached"] = {}
    if kp_path is not None:
        nr = ctx.nureco_event_for(os.path.basename(kp_path))
        if nr is not None:
            ev["nureco"] = nr
            gt = np.asarray(nr.get("part_gt_trackid", []), np.int64)
            en = np.asarray(nr.get("part_energy", []), np.float64)
            cl = np.asarray(nr.get("part_pred_class", []), np.int64)
            for t in np.unique(gt):
                m = np.where(gt == t)[0]
                j = m[np.argmax(en[m])]
                ev["attached"][int(t)] = dict(
                    reco_ke=reco_ke(en[j], int(cl[j])), pred_cls=int(cl[j]))

    # ---- per-particle stage summary (nu-origin, has spacepoints) ----
    ev["particle_rows"] = _particle_rows(ev)
    return ev


def _charge_sum(ev, sel):
    """Unique-pixel (Y else mean-U,V) charge over the selected triplet_data points.

    == the reco's de-double-counted shower-charge sum (calo.dedup_charge): each
    wire pixel's ADC is split among the spacepoints sharing it, so summing the
    per-point charge over `sel` counts every pixel the set touches exactly once.
    """
    if not np.any(sel):
        return 0.0
    _, q_comb = dedup_charge(ev["td_pixval"][sel], ev["td_tick"][sel],
                             ev["td_uw"][sel], ev["td_vw"][sel], ev["td_yw"][sel])
    return float(q_comb.sum())


def _particle_rows(ev):
    """One row per true particle with >0 triplet_data points, nu first.

    Coverage is CHARGE-based (fraction of the particle's visible ionization
    captured), not spacepoint-count-based: the denominator is the unique-pixel
    charge of the particle's GT spacepoints and the numerator the unique-pixel
    charge of the ones the slice/segment kept. This is robust to the GT labeller
    being over-liberal on the low-charge edges/tails of the ionization (many
    edge spacepoints, little charge), which count-based coverage over-penalises.
    """
    tid, origin = ev["td_tid"], ev["td_origin"]
    rows = []
    for t in np.unique(tid):
        if t <= 0:
            continue
        sel = tid == t
        ntrue = int(sel.sum())
        info = ev["mc"].get(int(t), {})
        o = info.get("origin", int(origin[sel][0]) if ntrue else 0)
        name = PID2NAME.get(info.get("pid"), f"pid{info.get('pid','?')}")
        n_slice = int(ev["in_slice"][sel].sum())
        n_seg = int(ev["seg"][sel].sum())
        pcs = ev["pred_cls"][sel & ev["seg"]]
        pcmaj = int(np.bincount(pcs).argmax()) if pcs.size else -1
        att = ev["attached"].get(int(t))
        q_true = _charge_sum(ev, sel)
        q_slice = _charge_sum(ev, sel & ev["in_slice"])
        q_seg = _charge_sum(ev, sel & ev["seg"])
        rows.append(dict(
            trackid=int(t), origin=o, name=name, ke=info.get("ke", float("nan")),
            n_true=ntrue, n_slice=n_slice, n_seg=n_seg,
            q_true=q_true,
            cov=q_slice / q_true if q_true > 0 else 0.0,        # charge coverage
            segfrac=q_seg / q_true if q_true > 0 else 0.0,
            pred_cls=pcmaj, attached=att is not None,
            reco_ke=att["reco_ke"] if att else float("nan")))
    # nu first, then by charge desc
    rows.sort(key=lambda r: (r["origin"] != ORIGIN_NU, -r["q_true"]))
    return rows


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
def make_detector_outline_trace():
    do = DetectorOutline()
    traces = []
    for line in (do.top_pts, do.bot_pts):
        xs = [p[0] for p in line]; ys = [p[1] for p in line]
        zs = [p[2] for p in line]
        traces.append(go.Scatter3d(
            x=zs, y=xs, z=ys, mode="lines",
            line=dict(color="rgba(120,120,120,0.4)", width=2),
            showlegend=False, hoverinfo="skip"))
    for i in range(4):
        a, b = do.top_pts[i], do.bot_pts[i]
        traces.append(go.Scatter3d(
            x=[a[2], b[2]], y=[a[0], b[0]], z=[a[1], b[1]], mode="lines",
            line=dict(color="rgba(120,120,120,0.4)", width=2),
            showlegend=False, hoverinfo="skip"))
    return traces


def _scatter(coord, color, name, size=1.6, opacity=1.0, hover=None):
    x, y, z = _xyz(coord)
    kw = {}
    if hover is not None:
        kw = dict(text=hover, hoverinfo="text")
    return go.Scatter3d(
        x=x, y=y, z=z, mode="markers", name=name,
        marker=dict(size=size, color=color, opacity=opacity), **kw)


def _subsample(mask, cap, seed=0):
    """Return an index array of points in mask, capped to `cap` (random)."""
    idx = np.where(mask)[0]
    if cap and len(idx) > cap:
        rng = np.random.default_rng(seed)
        idx = rng.choice(idx, cap, replace=False)
    return idx


def figure_for_event(ev, color_by, focus_tid, show_cosmic, show_ghost,
                     show_vertices, size, other_cap):
    pos = ev["td_pos"]; tid = ev["td_tid"]; origin = ev["td_origin"]
    traces = make_detector_outline_trace()

    nu = origin == ORIGIN_NU
    cosmic = origin == ORIGIN_COSMIC
    ghost = ~(nu | cosmic)

    # ---- background (cosmic / ghost), subsampled + dim ----
    if show_cosmic and cosmic.any():
        ci = _subsample(cosmic, other_cap, 1)
        traces.append(_scatter(pos[ci], "rgba(70,110,160,0.35)",
                               "cosmic", size * 0.8))
    if show_ghost and ghost.any():
        gi = _subsample(ghost, other_cap, 2)
        traces.append(_scatter(pos[gi], "rgba(140,140,140,0.25)",
                               "ghost/none", size * 0.7))

    # ---- main content by color mode ----
    if color_by == "truth_origin":
        if nu.any():
            traces.append(_scatter(pos[nu], "rgba(240,50,50,1)", "nu", size))

    elif color_by == "truth_particle":
        for r in ev["particle_rows"]:
            if r["origin"] != ORIGIN_NU:
                continue
            sel = (tid == r["trackid"])
            traces.append(_scatter(
                pos[sel], track_color(r["trackid"]),
                f"{r['name']} {r['ke']:.0f}MeV (trk {r['trackid']})", size))

    elif color_by == "slice_stage":
        # every nu point colored by which stage it reached
        for key, (lab, clr) in STAGE_STYLE.items():
            if key == "segmented":
                m = nu & ev["seg"]
            elif key == "in_slice_unseg":
                m = nu & ev["in_slice"] & ~ev["seg"]
            else:
                m = nu & ~ev["in_slice"]
            if m.any():
                traces.append(_scatter(pos[m], clr, f"nu: {lab}", size))

    elif color_by == "pred_class":
        # slice points colored by predicted class; unsegmented slice = gray
        for c in range(6):
            m = ev["in_slice"] & (ev["pred_cls"] == c)
            if m.any():
                traces.append(_scatter(pos[m], cls_color(c),
                                       f"pred {CLASS_NAMES[c]}", size))
        m = ev["in_slice"] & ~ev["seg"]
        if m.any():
            traces.append(_scatter(pos[m], "rgba(150,150,150,0.6)",
                                   "slice (no instance)", size))

    elif color_by == "particle_focus" and focus_tid not in (None, "__all__"):
        t = int(focus_tid)
        sel = tid == t
        # faint everything else that is nu
        rest = nu & ~sel
        if rest.any():
            traces.append(_scatter(pos[_subsample(rest, other_cap, 3)],
                                   "rgba(120,120,120,0.2)", "other nu", size*0.7))
        for key, (lab, clr) in STAGE_STYLE.items():
            if key == "segmented":
                m = sel & ev["seg"]
            elif key == "in_slice_unseg":
                m = sel & ev["in_slice"] & ~ev["seg"]
            else:
                m = sel & ~ev["in_slice"]
            if m.any():
                traces.append(_scatter(pos[m], clr, lab, size + 0.6))
    else:
        if nu.any():
            traces.append(_scatter(pos[nu], "rgba(240,50,50,1)", "nu", size))

    # ---- vertices / keypoints ----
    if show_vertices:
        if ev.get("gt_nu_vertex_cm") is not None and \
                np.all(np.isfinite(ev["gt_nu_vertex_cm"])):
            x, y, z = _xyz(ev["gt_nu_vertex_cm"])
            traces.append(go.Scatter3d(
                x=x, y=y, z=z, mode="markers", name="gt nu vtx",
                marker=dict(size=7, color="black", symbol="cross")))
        if ev.get("nu_vertex_cm") is not None and \
                np.all(np.isfinite(ev["nu_vertex_cm"])):
            x, y, z = _xyz(ev["nu_vertex_cm"])
            traces.append(go.Scatter3d(
                x=x, y=y, z=z, mode="markers", name="pred nu vtx",
                marker=dict(size=6, color="magenta", symbol="diamond")))
        nr = ev.get("nureco")
        if nr is not None and "vertices_cm" in nr and len(nr["vertices_cm"]):
            x, y, z = _xyz(np.atleast_2d(nr["vertices_cm"]))
            traces.append(go.Scatter3d(
                x=x, y=y, z=z, mode="markers", name="nu_reco vtx",
                marker=dict(size=6, color="cyan", symbol="circle")))

    title = (f"{ev['msp_base']}  |  "
             f"cascade={'yes' if ev['kp_path'] else 'MISSING'}  "
             f"nu_reco={'yes' if ev.get('nureco') else 'none'}")
    fig = go.Figure(data=traces, layout=go.Layout(
        title=title,
        scene=dict(xaxis=dict(title="z (cm)"), yaxis=dict(title="x (cm)"),
                   zaxis=dict(title="y (cm)"), aspectmode="data"),
        margin=dict(l=0, r=0, b=0, t=40),
        legend=dict(itemsizing="constant"),
        uirevision="keep"))
    return fig


# ---------------------------------------------------------------------------
# Stats table
# ---------------------------------------------------------------------------
def stats_table(ev):
    hdr = ["trackid", "origin", "species", "KE[MeV]", "n_true", "Qtrue",
           "in_slice", "covQ%", "segmented", "segQ%", "pred_cls", "attached",
           "recoKE"]
    head = html.Tr([html.Th(h) for h in hdr])
    rows = [head]
    for r in ev["particle_rows"]:
        og = {1: "nu", 2: "cosmic"}.get(r["origin"], str(r["origin"]))
        pcn = CLASS_NAMES[r["pred_cls"]] if 0 <= r["pred_cls"] < len(CLASS_NAMES)\
            else "-"
        cells = [r["trackid"], og, r["name"], f"{r['ke']:.0f}", r["n_true"],
                 f"{r['q_true']:.0f}", r["n_slice"], f"{100*r['cov']:.0f}",
                 r["n_seg"], f"{100*r['segfrac']:.0f}", pcn,
                 "Y" if r["attached"] else "-",
                 f"{r['reco_ke']:.0f}" if r["attached"] else "-"]
        bg = "#fff0f0" if r["origin"] == ORIGIN_NU else "#f6f6f6"
        rows.append(html.Tr([html.Td(str(c)) for c in cells],
                            style={"background": bg}))
    return html.Table(rows, style={"fontSize": "12px", "borderCollapse":
                                    "collapse", "width": "100%"},
                      className="stats")


def info_block(ev, ctx):
    nr = ev.get("nureco")
    lines = [
        f"merged_sp: {ev['msp_path']}",
        f"cascade  : {ev['kp_path'] or '(none — no nu slice)'}",
        f"nu_reco  : shard {nr['shard']} event_{nr['gidx']:07d}" if nr
        else "nu_reco  : (none)",
        f"triplet_data points: {len(ev['td_pos'])}  "
        f"(nu={int((ev['td_origin']==1).sum())}, "
        f"cosmic={int((ev['td_origin']==2).sum())}, "
        f"ghost={int((~np.isin(ev['td_origin'],[1,2])).sum())})",
        f"slice points: {len(ev['slice_coord'])}   "
        f"predicted instances: {len(ev['particles'])}",
    ]
    if ev["warnings"]:
        lines.append("WARN: " + "; ".join(ev["warnings"]))
    return html.Pre("\n".join(lines), style={"fontSize": "11px"})


# ---------------------------------------------------------------------------
# Dash app
# ---------------------------------------------------------------------------
_CTX = None
_CACHE = {}


def _get_event(index):
    if index in _CACHE:
        return _CACHE[index]
    path = _CTX.msp_sorted[index]
    ev = load_event(_CTX, path)
    _CACHE.clear()          # keep memory small: only the current event
    _CACHE[index] = ev
    return ev


def build_app(initial_index):
    app = Dash(__name__)
    app.title = "cascade reco visualizer"
    n_events = len(_CTX.msp_sorted)

    controls = html.Div([
        html.Div([
            html.Button("◀ prev", id="prev", n_clicks=0),
            dcc.Input(id="index", type="number", min=0, max=n_events - 1,
                      step=1, value=initial_index, style={"width": "90px"}),
            html.Button("next ▶", id="next", n_clicks=0),
            html.Span(f" / {n_events - 1}", style={"marginRight": "16px"}),
            html.Label("color by "),
            dcc.Dropdown(id="color_by", clearable=False, style={"width":"200px",
                         "display": "inline-block", "verticalAlign": "middle"},
                         options=[
                             {"label": "slice stage (missed?)",
                              "value": "slice_stage"},
                             {"label": "truth: particle", "value":
                              "truth_particle"},
                             {"label": "truth: origin (nu)", "value":
                              "truth_origin"},
                             {"label": "predicted class", "value": "pred_class"},
                             {"label": "focus one particle", "value":
                              "particle_focus"}],
                         value="slice_stage"),
            html.Label(" focus trk "),
            dcc.Dropdown(id="focus", clearable=False, value="__all__",
                         style={"width": "220px", "display": "inline-block",
                                "verticalAlign": "middle"},
                         options=[{"label": "(all)", "value": "__all__"}]),
        ], style={"display": "flex", "gap": "6px", "alignItems": "center",
                  "flexWrap": "wrap"}),
        html.Div([
            dcc.Checklist(id="toggles",
                          options=[{"label": "cosmic", "value": "cosmic"},
                                   {"label": "ghost", "value": "ghost"},
                                   {"label": "vertices", "value": "vtx"}],
                          value=["vtx"], inline=True),
            html.Div([
                html.Label("pt size", style={"fontSize": "12px"}),
                dcc.Slider(id="size", min=0.6, max=4, step=0.2, value=1.8,
                           marks=None, tooltip={"placement": "bottom",
                                                "always_visible": True}),
            ], style={"width": "220px"}),
            html.Div([
                html.Label("bg cap", style={"fontSize": "12px"}),
                dcc.Slider(id="cap", min=2000, max=80000, step=2000, value=20000,
                           marks=None, tooltip={"placement": "bottom",
                                                "always_visible": True}),
            ], style={"width": "260px"}),
        ], style={"display": "flex", "gap": "20px", "alignItems": "center",
                  "marginTop": "4px"}),
    ], style={"padding": "6px"})

    app.layout = html.Div([
        controls,
        html.Div([
            dcc.Graph(id="graph", style={"height": "78vh"}),
        ]),
        html.Div([
            html.Div(id="info", style={"flex": "1"}),
            html.Div(id="stats", style={"flex": "2"}),
        ], style={"display": "flex", "gap": "16px", "padding": "6px"}),
    ])

    # prev/next -> index
    @app.callback(Output("index", "value"),
                  Input("prev", "n_clicks"), Input("next", "n_clicks"),
                  State("index", "value"))
    def _step(p, nx, idx):
        idx = int(idx or 0)
        return max(0, min(n_events - 1, idx + int(nx) - int(p)))

    # index -> focus dropdown options
    @app.callback(Output("focus", "options"), Output("focus", "value"),
                  Input("index", "value"))
    def _focus_opts(index):
        ev = _get_event(int(index or 0))
        opts = [{"label": "(all)", "value": "__all__"}]
        for r in ev["particle_rows"]:
            if r["origin"] != ORIGIN_NU:
                continue
            opts.append({"label": f"{r['name']} {r['ke']:.0f}MeV "
                         f"(trk {r['trackid']}, cov {100*r['cov']:.0f}%)",
                         "value": str(r["trackid"])})
        return opts, "__all__"

    # main render
    @app.callback(Output("graph", "figure"), Output("stats", "children"),
                  Output("info", "children"),
                  Input("index", "value"), Input("color_by", "value"),
                  Input("focus", "value"), Input("toggles", "value"),
                  Input("size", "value"), Input("cap", "value"))
    def _render(index, color_by, focus, toggles, size, cap):
        ev = _get_event(int(index or 0))
        toggles = toggles or []
        fig = figure_for_event(
            ev, color_by, focus, "cosmic" in toggles, "ghost" in toggles,
            "vtx" in toggles, float(size), int(cap))
        return fig, stats_table(ev), info_block(ev, _CTX)

    return app


def main():
    global _CTX
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--merged-sp", default=None,
                    help="initial event's merged_sp file (path or basename)")
    ap.add_argument("--cascade-dir", default=DEF_CASCADE_DIR)
    ap.add_argument("--merged-sp-list", default=DEF_MSP_LIST)
    ap.add_argument("--keypoint2-list", default=DEF_KP_LIST)
    ap.add_argument("--nu-reco-dir", default=DEF_NURECO_DIR)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8050)
    args = ap.parse_args()

    _CTX = Context(args.cascade_dir, args.merged_sp_list,
                   args.keypoint2_list, args.nu_reco_dir)
    if not _CTX.msp_sorted:
        raise SystemExit(f"empty/missing merged_sp list: {args.merged_sp_list}")
    print(f">>> {len(_CTX.msp_sorted)} merged_sp events, "
          f"{len(_CTX.kp_gidx)} cascade files, {len(_CTX.shards)} nu_reco shards")

    initial = 0
    if args.merged_sp:
        base = os.path.basename(args.merged_sp)
        initial = _CTX.msp_index.get(base, 0)
        if base not in _CTX.msp_index:
            print(f"[warn] {base} not in list; starting at index 0")
    app = build_app(initial)
    print(f">>> serving on http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
