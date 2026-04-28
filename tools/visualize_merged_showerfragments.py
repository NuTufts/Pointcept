#!/usr/bin/env python3
"""
Visualization tool for shower reconstruction results.

Reads a ROOT file (from step 5-7 pipeline) and corresponding merged H5 files
to display reconstructed and missed showers in 3D.

Points are colored by merged shower ID. Each shower's predicted origin is
shown as a star marker with a visible text label matching the shower ID.
Missed true showers are also displayed.

Usage:
    python tools/visualize_merged_showerfragments.py \
        --root-file showerreco_*.root \
        --h5-dir /path/to/merged_h5_files/ \
        [--port 8050]

Example:
    python tools/visualize_merged_showerfragments.py \
        --root-file pointcept/showerreco_showerorigin_reco_mcc9_v29e_dl_run3b_bnb_nu_overlay_test_fileno00001.root \
        --h5-dir /cluster/tufts/wongjiradlab/larbys/data/ub_on_tufts/hdf5/shower_origin_reco/mcc9_v29e_dl_run3b_bnb_nu_overlay_test/000/000/
"""

import argparse
import os
import glob

import numpy as np
import h5py
import uproot
import plotly.graph_objects as go
from dash import Dash, html, dcc, callback, Output, Input

from detectoroutline import DetectorOutline

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CLASS_NAMES = {0: "inside (nu)", 1: "outside (cosmic)", 2: "on_track"}

# Qualitative palette for up to ~20 distinct showers
SHOWER_COLORS = [
    "#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231",
    "#911eb4", "#42d4f4", "#f032e6", "#bfef45", "#fabed4",
    "#469990", "#dcbeff", "#9A6324", "#fffac8", "#800000",
    "#aaffc3", "#808000", "#ffd8b1", "#000075", "#a9a9a9",
]


def shower_color(idx):
    return SHOWER_COLORS[idx % len(SHOWER_COLORS)]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_root_data(root_path):
    """Load ROOT TTree into dict of numpy/awkward arrays.  Return per-event list."""
    f = uproot.open(root_path)
    tree = f["shower_reco"]
    branches = tree.arrays(library="np")

    n_events = len(branches["run"])
    events = []
    for i in range(n_events):
        ev = {key: val[i] for key, val in branches.items()}
        events.append(ev)
    return events


def find_h5_for_event(h5_dir, run, subrun, event):
    """Find the merged H5 file that contains a given (run, subrun, event).

    Strategy: scan H5 files in h5_dir for matching RSE in mc_particle_tree
    or triplet_data metadata.  Cache on first call.
    """
    if not hasattr(find_h5_for_event, "_cache"):
        find_h5_for_event._cache = {}

    key = (h5_dir, run, subrun, event)
    if key in find_h5_for_event._cache:
        return find_h5_for_event._cache[key]

    for h5_path in sorted(glob.glob(os.path.join(h5_dir, "merged_*.h5"))):
        try:
            with h5py.File(h5_path, "r") as hf:
                entry = hf.get("entry_0")
                if entry is None:
                    continue
                r = int(entry.attrs.get("run", -1))
                s = int(entry.attrs.get("subrun", -1))
                e = int(entry.attrs.get("event", -1))
                if r == run and s == subrun and e == event:
                    find_h5_for_event._cache[key] = h5_path
                    return h5_path
        except Exception:
            continue

    return None


def build_h5_rse_index(h5_dir):
    """Pre-scan all H5 files and return {(run,subrun,event): h5_path}."""
    index = {}
    for h5_path in sorted(glob.glob(os.path.join(h5_dir, "merged_*.h5"))):
        try:
            with h5py.File(h5_path, "r") as hf:
                entry = hf.get("entry_0")
                if entry is None:
                    continue
                r = int(entry.attrs.get("run", -1))
                s = int(entry.attrs.get("subrun", -1))
                e = int(entry.attrs.get("event", -1))
                index[(r, s, e)] = h5_path
        except Exception:
            continue
    return index


def load_h5_fragments(h5_path):
    """Load fragment spacepoint data from a merged H5 file.

    Returns:
        pos: (N, 3) float32 array of all spacepoints in cm
        origin: (N,) int array (0=unknown, 1=neutrino, 2=cosmic)
        hasmatch: (N,) int array (1=true point, 0=ghost)
        fragment_point_indices: list of arrays, one per fragment
        num_fragments: int
    """
    with h5py.File(h5_path, "r") as hf:
        entry = hf["entry_0"]
        td = entry["triplet_data"]
        pos = td["pos"][:]
        origin = td["origin"][:] if "origin" in td else np.zeros(len(pos), dtype=np.int64)
        hasmatch = td["hasmatch"][:] if "hasmatch" in td else np.ones(len(pos), dtype=np.int64)

        sf = entry.get("shower_fragments")
        if sf is None:
            return pos, origin, hasmatch, [], 0

        num_fragments = int(sf.attrs.get("num_fragments", 0))
        flat = sf["pointindices_flat"][:]
        counts = sf["pointindices_counts"][:]

        fragment_point_indices = []
        offset = 0
        for i in range(num_fragments):
            c = int(counts[i])
            fragment_point_indices.append(flat[offset:offset + c])
            offset += c

    return pos, origin, hasmatch, fragment_point_indices, num_fragments


# ---------------------------------------------------------------------------
# Figure construction
# ---------------------------------------------------------------------------

def build_figure(root_event, h5_path, marker_size=1.5, opacity=0.6):
    """Build a Plotly 3D figure for one event."""
    fig = go.Figure()

    run = int(root_event["run"])
    subrun = int(root_event["subrun"])
    event = int(root_event["event"])
    title = f"Run {run}  Subrun {subrun}  Event {event}"

    if h5_path is None:
        fig.update_layout(title=f"{title} — H5 file not found")
        return fig

    pos, origin, hasmatch, frag_indices, num_frags = load_h5_fragments(h5_path)

    # Merged-shower metadata from ROOT
    shower_ids = root_event.get("shower_id", np.array([], dtype=np.int32))
    shower_pred_class = root_event.get("shower_pred_class", np.array([], dtype=np.int32))
    shower_origin_x = root_event.get("shower_pred_origin_x", np.array([], dtype=np.float32))
    shower_origin_y = root_event.get("shower_pred_origin_y", np.array([], dtype=np.float32))
    shower_origin_z = root_event.get("shower_pred_origin_z", np.array([], dtype=np.float32))
    shower_nfrags = root_event.get("shower_n_fragments", np.array([], dtype=np.int32))
    shower_npts = root_event.get("shower_n_total_points", np.array([], dtype=np.int32))
    shower_start_x = root_event.get("shower_start_x", np.array([], dtype=np.float32))
    shower_start_y = root_event.get("shower_start_y", np.array([], dtype=np.float32))
    shower_start_z = root_event.get("shower_start_z", np.array([], dtype=np.float32))
    shower_axis_x = root_event.get("shower_axis_x", np.array([], dtype=np.float32))
    shower_axis_y = root_event.get("shower_axis_y", np.array([], dtype=np.float32))
    shower_axis_z = root_event.get("shower_axis_z", np.array([], dtype=np.float32))

    # Per-fragment metadata from ROOT
    frag_merged_id = root_event.get("frag_merged_shower_id", np.array([], dtype=np.int32))
    frag_gt_class = root_event.get("frag_gt_class", np.array([], dtype=np.int32))
    frag_pred_class = root_event.get("frag_pred_class", np.array([], dtype=np.int32))
    frag_trackid = root_event.get("frag_geant_trackid", np.array([], dtype=np.int32))
    frag_is_trunk = root_event.get("frag_is_trunk", np.array([], dtype=np.int32))
    frag_pred_ox = root_event.get("frag_pred_origin_x", np.array([], dtype=np.float32))
    frag_pred_oy = root_event.get("frag_pred_origin_y", np.array([], dtype=np.float32))
    frag_pred_oz = root_event.get("frag_pred_origin_z", np.array([], dtype=np.float32))
    frag_npts = root_event.get("frag_n_points", np.array([], dtype=np.int32))

    # Build merged shower ID -> color mapping
    unique_shower_ids = np.unique(shower_ids) if len(shower_ids) > 0 else []
    sid_to_color = {int(sid): shower_color(i) for i, sid in enumerate(unique_shower_ids)}

    # ---- Plot points colored by merged shower / unmerged fragment ----
    # Collect points per merged shower and per unmerged fragment
    shower_points = {}  # shower_id -> list of (x,y,z)
    unmerged_frags = []  # list of (frag_idx, pts)

    for fi in range(min(num_frags, len(frag_merged_id))):
        if fi >= len(frag_indices):
            break
        indices = frag_indices[fi]
        if len(indices) == 0:
            continue
        pts = pos[indices]
        sid = int(frag_merged_id[fi])
        if sid >= 0:
            shower_points.setdefault(sid, []).append(pts)
        else:
            unmerged_frags.append((fi, pts))

    # Assign colors: merged showers first, then unmerged fragments continue the palette
    n_merged = len(unique_shower_ids)
    unmerged_color_offset = n_merged

    # Add merged shower point clouds
    for sid in sorted(shower_points.keys()):
        pts = np.concatenate(shower_points[sid], axis=0)
        color = sid_to_color.get(sid, "gray")

        # Find shower-level info
        smask = shower_ids == sid
        s_idx = np.where(smask)[0]
        if len(s_idx) > 0:
            si = s_idx[0]
            cls_name = CLASS_NAMES.get(int(shower_pred_class[si]), "?")
            nf = int(shower_nfrags[si])
            npt = int(shower_npts[si])
        else:
            cls_name = "?"
            nf = 0
            npt = len(pts)

        hover = (
            f"Shower {sid}<br>"
            f"Class: {cls_name}<br>"
            f"Fragments: {nf}<br>"
            f"Points: {npt}"
        )

        fig.add_trace(go.Scatter3d(
            x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
            mode="markers",
            marker=dict(size=marker_size, color=color, opacity=opacity),
            name=f"Shower {sid} ({cls_name}, {len(pts)} pts)",
            hovertext=hover,
            hoverinfo="text",
            legendgroup=f"shower_{sid}",
        ))

    # Add unmerged fragments, each with its own color and origin marker
    for u_idx, (fi, pts) in enumerate(unmerged_frags):
        color = shower_color(unmerged_color_offset + u_idx)
        label = f"U{fi}"
        legend_group = f"unmerged_{fi}"

        cls_name = CLASS_NAMES.get(int(frag_pred_class[fi]), "?") if fi < len(frag_pred_class) else "?"
        tid = int(frag_trackid[fi]) if fi < len(frag_trackid) else -1
        npt = int(frag_npts[fi]) if fi < len(frag_npts) else len(pts)

        hover = (
            f"Unmerged frag {fi}<br>"
            f"Pred class: {cls_name}<br>"
            f"TrackID: {tid}<br>"
            f"Points: {npt}"
        )

        fig.add_trace(go.Scatter3d(
            x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
            mode="markers",
            marker=dict(size=marker_size, color=color, opacity=opacity),
            name=f"Unmerged {label} ({cls_name}, {len(pts)} pts)",
            hovertext=hover,
            hoverinfo="text",
            legendgroup=legend_group,
        ))

        # Origin marker for unmerged fragment
        if fi < len(frag_pred_ox):
            ox = float(frag_pred_ox[fi])
            oy = float(frag_pred_oy[fi])
            oz = float(frag_pred_oz[fi])

            origin_hover = (
                f"Unmerged {label} Origin<br>"
                f"Pred class: {cls_name}<br>"
                f"TrackID: {tid}<br>"
                f"({ox:.1f}, {oy:.1f}, {oz:.1f})"
            )

            fig.add_trace(go.Scatter3d(
                x=[ox], y=[oy], z=[oz],
                mode="markers+text",
                marker=dict(size=8, color=color, symbol="diamond",
                            line=dict(width=2, color="black")),
                text=[label],
                textposition="top center",
                textfont=dict(size=11, color=color),
                name=f"Origin {label}",
                hovertext=origin_hover,
                hoverinfo="text",
                legendgroup=legend_group,
                showlegend=False,
            ))

    # ---- Neutrino true (non-ghost) spacepoints ----
    nu_mask = (origin == 1) & (hasmatch == 1)
    nu_pts = pos[nu_mask]
    if len(nu_pts) > 0:
        fig.add_trace(go.Scatter3d(
            x=nu_pts[:, 0], y=nu_pts[:, 1], z=nu_pts[:, 2],
            mode="markers",
            marker=dict(size=marker_size, color="rgba(255,255,0,0.5)",
                        opacity=0.4),
            name=f"Nu true pts ({len(nu_pts)})",
            hovertext="Neutrino true point",
            hoverinfo="text",
            visible="legendonly",  # hidden by default to avoid clutter
        ))

    # ---- Merged shower origin markers + labels ----
    for i_s, sid in enumerate(shower_ids):
        sid = int(sid)
        ox = float(shower_origin_x[i_s])
        oy = float(shower_origin_y[i_s])
        oz = float(shower_origin_z[i_s])
        cls = int(shower_pred_class[i_s])
        cls_name = CLASS_NAMES.get(cls, "?")
        color = sid_to_color.get(sid, "white")

        hover = (
            f"Shower {sid} Origin<br>"
            f"Class: {cls_name}<br>"
            f"({ox:.1f}, {oy:.1f}, {oz:.1f})"
        )

        # Origin marker
        fig.add_trace(go.Scatter3d(
            x=[ox], y=[oy], z=[oz],
            mode="markers+text",
            marker=dict(size=10, color=color, symbol="diamond",
                        line=dict(width=2, color="black")),
            text=[f"S{sid}"],
            textposition="top center",
            textfont=dict(size=12, color=color),
            name=f"Origin S{sid}",
            hovertext=hover,
            hoverinfo="text",
            legendgroup=f"shower_{sid}",
            showlegend=False,
        ))

    # ---- Missed true showers ----
    missed_trackid = root_event.get("missed_trackid", np.array([], dtype=np.int32))
    missed_npts = root_event.get("missed_n_points", np.array([], dtype=np.int32))
    missed_ox = root_event.get("missed_gt_origin_x", np.array([], dtype=np.float32))
    missed_oy = root_event.get("missed_gt_origin_y", np.array([], dtype=np.float32))
    missed_oz = root_event.get("missed_gt_origin_z", np.array([], dtype=np.float32))
    missed_cls = root_event.get("missed_gt_class", np.array([], dtype=np.int32))
    missed_trunk = root_event.get("missed_is_trunk", np.array([], dtype=np.int32))

    if len(missed_trackid) > 0:
        for i_m in range(len(missed_trackid)):
            tid = int(missed_trackid[i_m])
            mx = float(missed_ox[i_m])
            my = float(missed_oy[i_m])
            mz = float(missed_oz[i_m])
            mc = int(missed_cls[i_m])
            mnp = int(missed_npts[i_m])
            mtrunk = int(missed_trunk[i_m])
            cls_name = CLASS_NAMES.get(mc, "?")

            hover = (
                f"MISSED true shower<br>"
                f"TrackID: {tid}<br>"
                f"Class: {cls_name}<br>"
                f"Points: {mnp}<br>"
                f"Trunk: {mtrunk}<br>"
                f"({mx:.1f}, {my:.1f}, {mz:.1f})"
            )

            fig.add_trace(go.Scatter3d(
                x=[mx], y=[my], z=[mz],
                mode="markers+text",
                marker=dict(size=12, color="red", symbol="x",
                            line=dict(width=2, color="white")),
                text=[f"M{i_m}"],
                textposition="top center",
                textfont=dict(size=12, color="red"),
                name=f"Missed M{i_m} (tid={tid}, {cls_name})",
                hovertext=hover,
                hoverinfo="text",
                showlegend=True,
            ))

    # ---- Detector outline ----
    detdata = DetectorOutline()
    det_traces = detdata.getlines(color=(255, 255, 255))
    for dt in det_traces:
        fig.add_trace(go.Scatter3d(
            x=dt["x"], y=dt["y"], z=dt["z"],
            mode="lines",
            line=dict(color=dt["line"]["color"], width=dt["line"]["width"]),
            showlegend=False,
        ))

    # ---- Layout ----
    n_showers = len(unique_shower_ids)
    n_missed = len(missed_trackid)
    axis_template = dict(
        showbackground=True,
        backgroundcolor="#141414",
        gridcolor="rgb(80, 80, 80)",
        zerolinecolor="rgb(128, 128, 128)",
        title_font=dict(color="white"),
        tickfont=dict(color="white"),
    )

    fig.update_layout(
        scene=dict(
            xaxis=dict(**axis_template, title="X (cm)"),
            yaxis=dict(**axis_template, title="Y (cm)"),
            zaxis=dict(**axis_template, title="Z (cm)"),
            aspectratio=dict(x=1, y=1, z=4),
            camera=dict(eye=dict(x=2, y=2, z=2), up=dict(x=0, y=1, z=0)),
        ),
        title=dict(
            text=f"{title}  |  {n_showers} merged showers, {n_missed} missed",
            font=dict(color="white"),
        ),
        paper_bgcolor="#1e1e1e",
        plot_bgcolor="#1e1e1e",
        legend=dict(font=dict(color="white")),
        margin=dict(l=0, r=0, t=40, b=0),
    )

    return fig


# ---------------------------------------------------------------------------
# Dash app
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Visualize merged shower fragments.")
    p.add_argument("--root-file", required=True, help="ROOT file from step 5-7 pipeline")
    p.add_argument("--h5-dir", required=True,
                   help="Directory containing merged_*.h5 files from steps 1-4")
    p.add_argument("--port", type=int, default=8050, help="Dash server port (default: 8050)")
    return p.parse_args()


def main():
    args = parse_args()

    # Load ROOT data
    print(f"Loading ROOT file: {args.root_file}")
    events = load_root_data(args.root_file)
    print(f"  {len(events)} events loaded")

    # Pre-index H5 files
    print(f"Indexing H5 files in: {args.h5_dir}")
    rse_index = build_h5_rse_index(args.h5_dir)
    print(f"  {len(rse_index)} H5 files indexed")

    # Build dropdown options
    dropdown_options = []
    for i, ev in enumerate(events):
        r, s, e = int(ev["run"]), int(ev["subrun"]), int(ev["event"])
        ns = int(ev["n_merged_showers"])
        nm = int(ev["n_missed_true"])
        nf = int(ev["n_reco_fragments"])
        has_h5 = "OK" if (r, s, e) in rse_index else "NO H5"
        label = (f"[{i}] R{r} S{s} E{e}  |  "
                 f"{nf} frags, {ns} showers, {nm} missed  [{has_h5}]")
        dropdown_options.append({"label": label, "value": i})

    app = Dash(__name__)
    app.layout = html.Div(
        style={"backgroundColor": "#1e1e1e", "minHeight": "100vh", "padding": "10px"},
        children=[
            html.H3("Shower Reconstruction Viewer",
                     style={"color": "white", "textAlign": "center"}),
            html.Div(
                style={"maxWidth": "900px", "margin": "0 auto 10px auto"},
                children=[
                    dcc.Dropdown(
                        id="event-dropdown",
                        options=dropdown_options,
                        value=0,
                        style={"color": "black"},
                    ),
                ],
            ),
            dcc.Graph(
                id="main-graph",
                style={"height": "85vh"},
            ),
        ],
    )

    @callback(Output("main-graph", "figure"), Input("event-dropdown", "value"))
    def update_figure(event_idx):
        if event_idx is None:
            event_idx = 0
        ev = events[event_idx]
        r, s, e = int(ev["run"]), int(ev["subrun"]), int(ev["event"])
        h5_path = rse_index.get((r, s, e))
        if h5_path is None:
            print(f"  WARNING: no H5 for R{r} S{s} E{e}")
        return build_figure(ev, h5_path)

    print(f"\nStarting Dash server on port {args.port}")
    app.run(debug=False, port=args.port, host="0.0.0.0")


if __name__ == "__main__":
    main()
