"""
Visualize shower fragment ground truth labels from HDF5 files.

Supports both formats:
  - New flat format (from C++ ShowerFragmentOriginMaker): pointindices_flat
  - Legacy per-fragment group format (from Python labeler): fragment_0/, etc.

Usage:
    python vis_shower_fragments.py -i <hdf5_file> -e <entry_num>
    python vis_shower_fragments.py -i <hdf5_file> -e 0 --show-non-shower
    python vis_shower_fragments.py -i <hdf5_file> -e 0 --colorby trackid

Color modes (--colorby):
    fragment   : color shower points by DBSCAN cluster index (default)
    origin     : color shower points by origin type (INSIDE=green, OUTSIDE=red, ON_TRACK=gold)
    trackid    : all fragments from same particle share one color

Displays:
    - Shower fragment spacepoints (colored by mode)
    - Origin point markers (one per unique trackid)
    - Start point markers (per fragment, where the fragment begins)
    - Non-shower true-match points in gray (with --show-non-shower)
    - MC keypoints (with --show-keypoints)
    - TPC detector outline
"""
import os
import sys
import argparse
import h5py
import numpy as np

import dash
from dash import dcc, html

from detectoroutline import DetectorOutline

# ──────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser("Visualize shower fragment labels")
parser.add_argument("-i", "--input-h5", required=True, type=str,
                    help="Input HDF5 file")
parser.add_argument("-e", "--entry", required=True, type=int,
                    help="Entry number")
parser.add_argument("-c", "--colorby", default="fragment",
                    choices=["fragment", "origin", "trackid"],
                    help="Color mode for shower points")
parser.add_argument("-p", "--pos-mode", default="reco",
                    choices=["true", "reco"],
                    help="Position mode: true or reco (SCE-corrected)")
parser.add_argument("--show-non-shower", action="store_true", default=False,
                    help="Show non-shower true-match points in gray")
parser.add_argument("--include-ghosts", action="store_true", default=False,
                    help="Include ghost points (default: true-match only)")
parser.add_argument("--max-bg-points", type=int, default=50000,
                    help="Max non-shower background points to display (0=all)")
parser.add_argument("--show-keypoints", action="store_true", default=False,
                    help="Overlay MC keypoints")
parser.add_argument("--marker-size", type=float, default=2.0,
                    help="Marker size for spacepoints")
parser.add_argument("--filter-pid", type=int, default=None,
                    help="Only show fragments with this particle PID (e.g. 22, 11)")
parser.add_argument("--min-frag-pts", type=int, default=0,
                    help="Minimum number of points per fragment to display (0=show all)")
parser.add_argument("--port", type=int, default=8050,
                    help="Dash server port")
args = parser.parse_args()

# ──────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────
ORIGIN_COLORS = {
    0: "rgba(0,255,0,1)",     # INSIDE = green
    1: "rgba(255,0,0,1)",     # OUTSIDE = red
    2: "rgba(255,215,0,1)",   # ON_TRACK = gold
}
ORIGIN_NAMES = {0: "INSIDE", 1: "OUTSIDE", 2: "ON_TRACK"}

# ──────────────────────────────────────────────────────────────────
# Load data
# ──────────────────────────────────────────────────────────────────
fh5 = h5py.File(args.input_h5, "r")
entry_key = f"entry_{args.entry}"

if entry_key not in fh5:
    print(f"Entry '{entry_key}' not found. Available: {list(fh5.keys())}")
    sys.exit(1)

entry = fh5[entry_key]

# Masks are built against triplet_data (includes ghosts),
# so we must use triplet_data as the base.
triplet_key = "triplet_data" if "triplet_data" in entry else "triplet_truth"
triplet = entry[triplet_key]

pos_key = "pos_reco" if args.pos_mode == "reco" else "pos"
if pos_key not in triplet:
    pos_key = "pos"
pos = np.array(triplet[pos_key], dtype=np.float32)
npts = len(pos)

trackid = np.array(triplet["trackid"], dtype=np.int64).ravel()
pid = np.array(triplet["pid"], dtype=np.int64).ravel()
origin_arr = np.array(triplet["origin"], dtype=np.int64).ravel() if "origin" in triplet else np.zeros(npts, dtype=np.int64)

# hasmatch for filtering ghosts
hasmatch = None
if "hasmatch" in triplet:
    hasmatch = np.array(triplet["hasmatch"], dtype=np.int64).ravel()
    n_true = int(np.sum(hasmatch == 1))
    n_ghost = npts - n_true
    print(f"Loaded {npts} spacepoints ({n_true} true, {n_ghost} ghost)")
else:
    print(f"Loaded {npts} spacepoints (no hasmatch info)")

# Load shower fragments
if "shower_fragments" not in entry:
    print("No shower_fragments group found in this entry.")
    sys.exit(1)

sf = entry["shower_fragments"]

fragments = []

# Detect format: new flat format vs legacy per-fragment groups
if "pointindices_flat" in sf:
    # === New flat format from C++ ShowerFragmentOriginMaker ===
    num_fragments_raw = int(sf.attrs.get("num_fragments", 0))
    print(f"Found {num_fragments_raw} shower fragments (new flat format)")

    trackids_arr = np.array(sf["trackid"])
    pids_arr = np.array(sf["pid"])
    istrunk_arr = np.array(sf["istrunk"])
    frag_types_arr = np.array(sf["type"])
    startpts_arr = np.array(sf["startpt"], dtype=np.float32)
    originpts_arr = np.array(sf["originpt"], dtype=np.float32)
    pret0pts_arr = np.array(sf["pret0shiftedoriginpt"], dtype=np.float32) if "pret0shiftedoriginpt" in sf else None
    flat_indices = np.array(sf["pointindices_flat"])
    index_counts = np.array(sf["pointindices_counts"])

    offset = 0
    for i in range(num_fragments_raw):
        count = int(index_counts[i])
        indices = flat_indices[offset:offset+count]
        offset += count

        # Build boolean mask
        fmask = np.zeros(npts, dtype=bool)
        valid_idx = indices[indices < npts]
        fmask[valid_idx] = True

        frag = {
            "mask": fmask,
            "origin_coord": originpts_arr[i],
            "start_coord": startpts_arr[i],
            "pret0_origin": pret0pts_arr[i] if pret0pts_arr is not None else None,
            "origin_type": int(frag_types_arr[i]),
            "istrunk": int(istrunk_arr[i]),
            "trackid": int(trackids_arr[i]),
            "particle_pid": int(pids_arr[i]),
        }

        # Apply PID filter
        if args.filter_pid is not None and frag["particle_pid"] != args.filter_pid:
            continue

        nvis = int(np.sum(frag["mask"]))

        # Apply minimum fragment size filter
        if args.min_frag_pts > 0 and nvis < args.min_frag_pts:
            continue

        otype = ORIGIN_NAMES.get(frag["origin_type"], f"UNK({frag['origin_type']})")
        trunk_str = "trunk" if frag["istrunk"] == 1 else "secondary"
        pid_str = {22: "photon", 11: "e-", -11: "e+"}.get(frag["particle_pid"], str(frag["particle_pid"]))
        print(f"  Fragment {i}: {nvis} pts, {otype}, {trunk_str}, pid={pid_str}, "
              f"trackid={frag['trackid']}")
        fragments.append(frag)
else:
    # === Legacy per-fragment group format ===
    num_fragments_raw = int(sf.attrs.get("num_fragments", 0))
    print(f"Found {num_fragments_raw} shower fragments (legacy format)")

    for i in range(num_fragments_raw):
        fg = sf[f"fragment_{i}"]
        frag = {
            "mask": np.array(fg["mask"], dtype=bool),
            "origin_coord": np.array(fg["origin_coord"], dtype=np.float32),
            "start_coord": np.array(fg["origin_coord"], dtype=np.float32),  # no startpt in legacy
            "pret0_origin": None,
            "origin_type": int(np.array(fg["origin_type"])),
            "istrunk": 0,
            "trackid": int(np.array(fg.get("primary_trackid", fg.get("photon_trackid", -1)))),
            "particle_pid": int(np.array(fg["particle_pid"])) if "particle_pid" in fg else 22,
        }

        if args.filter_pid is not None and frag["particle_pid"] != args.filter_pid:
            continue

        nvis = int(np.sum(frag["mask"]))

        if args.min_frag_pts > 0 and nvis < args.min_frag_pts:
            continue

        otype = ORIGIN_NAMES.get(frag["origin_type"], f"UNK({frag['origin_type']})")
        pid_str = {22: "photon", 11: "e-", -11: "e+"}.get(frag["particle_pid"], str(frag["particle_pid"]))
        print(f"  Fragment {i}: {nvis} pts, {otype}, pid={pid_str}, trackid={frag['trackid']}")
        fragments.append(frag)

num_fragments = len(fragments)  # update after filtering

# Load keypoints if requested
kpdata = None
if args.show_keypoints and "mckeypoints" in entry:
    mckp = entry["mckeypoints"]
    kpdata = {
        "pos": np.array(mckp["pos"], dtype=np.float32),
        "kptype": np.array(mckp["kptype"], dtype=np.int64).ravel(),
        "trackid": np.array(mckp["trackid"], dtype=np.int64).ravel(),
        "pid": np.array(mckp["pid"], dtype=np.int64).ravel(),
    }

fh5.close()

# ──────────────────────────────────────────────────────────────────
# Build point filter (true-match only by default)
# ──────────────────────────────────────────────────────────────────
# For shower fragment points, we apply hasmatch filter to reduce ghost clutter
if hasmatch is not None and not args.include_ghosts:
    true_mask = hasmatch == 1
else:
    true_mask = np.ones(npts, dtype=bool)

# ──────────────────────────────────────────────────────────────────
# Build traces
# ──────────────────────────────────────────────────────────────────
traces = []
opacity = 0.8
msize = args.marker_size

# Color palette for fragments (distinct colors)
FRAG_COLORS = [
    "rgba(31,119,180,1)",   # blue
    "rgba(255,127,14,1)",   # orange
    "rgba(44,160,44,1)",    # green
    "rgba(214,39,40,1)",    # red
    "rgba(148,103,189,1)",  # purple
    "rgba(227,119,194,1)",  # pink
    "rgba(23,190,207,1)",   # cyan
    "rgba(255,215,0,1)",    # gold
    "rgba(0,255,127,1)",    # spring green
    "rgba(255,69,0,1)",     # orange-red
    "rgba(138,43,226,1)",   # blue-violet
    "rgba(0,191,255,1)",    # deep sky blue
]

# Build combined shower mask (union of all fragment masks)
any_shower_mask = np.zeros(npts, dtype=bool)
for frag in fragments:
    any_shower_mask |= frag["mask"]

# ── Non-shower points (gray background) ──
if args.show_non_shower:
    non_shower = ~any_shower_mask & true_mask
    non_indices = np.where(non_shower)[0]
    n_non = len(non_indices)

    # Downsample if needed
    if args.max_bg_points > 0 and n_non > args.max_bg_points:
        rng = np.random.default_rng(42)
        chosen = rng.choice(non_indices, args.max_bg_points, replace=False)
        chosen.sort()
        label = f"non-shower ({args.max_bg_points}/{n_non} sampled)"
    else:
        chosen = non_indices
        label = f"non-shower ({n_non} pts)"

    if len(chosen) > 0:
        customdata_bg = np.column_stack([
            trackid[chosen].astype(np.float64),
            pid[chosen].astype(np.float64),
            origin_arr[chosen].astype(np.float64),
        ])
        traces.append({
            "type": "scatter3d",
            "x": pos[chosen, 0],
            "y": pos[chosen, 1],
            "z": pos[chosen, 2],
            "mode": "markers",
            "name": label,
            "hovertemplate": (
                "<b>x</b>: %{x:.1f}<br>"
                "<b>y</b>: %{y:.1f}<br>"
                "<b>z</b>: %{z:.1f}<br>"
                "<b>TID</b>: %{customdata[0]:d}<br>"
                "<b>PID</b>: %{customdata[1]:d}<br>"
                "<b>Origin</b>: %{customdata[2]:d}<br>"
            ),
            "customdata": customdata_bg,
            "marker": {"color": "rgba(80,80,80,0.3)", "size": msize * 0.8},
        })
        print(f"  Background: {len(chosen)} non-shower points plotted")

# ── Shower fragment points ──
# Build trackid-to-color mapping for trackid color mode
unique_trackids = sorted(set(f["trackid"] for f in fragments))
trackid_color_map = {tid: FRAG_COLORS[j % len(FRAG_COLORS)]
                     for j, tid in enumerate(unique_trackids)}

for i, frag in enumerate(fragments):
    # Apply both fragment mask and true_mask filter
    frag_display = frag["mask"] & true_mask
    n_display = int(np.sum(frag_display))

    if n_display == 0:
        continue

    otype = ORIGIN_NAMES.get(frag["origin_type"], "UNK")
    trunk_str = "T" if frag["istrunk"] == 1 else "S"
    pid_str = {22: "g", 11: "e-", -11: "e+"}.get(frag["particle_pid"], str(frag["particle_pid"]))
    label = f"frag{i} ({pid_str},{otype},{trunk_str}, tid={frag['trackid']}, {n_display}pts)"

    # Choose color based on mode
    if args.colorby == "fragment":
        color = FRAG_COLORS[i % len(FRAG_COLORS)]
        marker_cfg = {"color": color, "opacity": opacity, "size": msize}
    elif args.colorby == "origin":
        color = ORIGIN_COLORS[frag["origin_type"]]
        marker_cfg = {"color": color, "opacity": opacity, "size": msize}
    elif args.colorby == "trackid":
        color = trackid_color_map.get(frag["trackid"], FRAG_COLORS[i % len(FRAG_COLORS)])
        marker_cfg = {"color": color, "opacity": opacity, "size": msize}

    customdata_frag = np.column_stack([
        trackid[frag_display].astype(np.float64),
        pid[frag_display].astype(np.float64),
        origin_arr[frag_display].astype(np.float64),
        np.full(n_display, i, dtype=np.float64),
        np.full(n_display, frag["trackid"], dtype=np.float64),
    ])

    traces.append({
        "type": "scatter3d",
        "x": pos[frag_display, 0],
        "y": pos[frag_display, 1],
        "z": pos[frag_display, 2],
        "mode": "markers",
        "name": label,
        "hovertemplate": (
            "<b>x</b>: %{x:.1f}<br>"
            "<b>y</b>: %{y:.1f}<br>"
            "<b>z</b>: %{z:.1f}<br>"
            "<b>TID</b>: %{customdata[0]:d}<br>"
            "<b>PID</b>: %{customdata[1]:d}<br>"
            "<b>Origin</b>: %{customdata[2]:d}<br>"
            "<b>Frag</b>: %{customdata[3]:d}<br>"
            "<b>TrackID</b>: %{customdata[4]:d}<br>"
        ),
        "customdata": customdata_frag,
        "marker": marker_cfg,
    })

# ── Origin point markers (one per unique trackid to avoid clutter) ──
seen_origin_trackids = set()
for i, frag in enumerate(fragments):
    oc = frag["origin_coord"]
    otype = frag["origin_type"]
    ftid = frag["trackid"]
    ppid = frag["particle_pid"]
    p0 = frag["pret0_origin"]

    otype_str = ORIGIN_NAMES.get(otype, f"UNK({otype})")
    origin_color = ORIGIN_COLORS.get(otype, "rgba(255,255,255,1)")
    pid_str = {22: "photon", 11: "e-", -11: "e+"}.get(ppid, str(ppid))

    # Build pret0 info string for hover
    if p0 is not None:
        pret0_str = f"({p0[0]:.1f}, {p0[1]:.1f}, {p0[2]:.1f}, t={p0[3]:.1f})"
    else:
        pret0_str = "N/A"

    # Only show one origin marker per unique trackid
    if ftid not in seen_origin_trackids:
        seen_origin_trackids.add(ftid)
        traces.append({
            "type": "scatter3d",
            "x": [oc[0]],
            "y": [oc[1]],
            "z": [oc[2]],
            "mode": "markers+text",
            "name": f"origin tid={ftid} ({pid_str},{otype_str})",
            "text": [f"O:{ftid}"],
            "textposition": "top center",
            "textfont": {"color": "white", "size": 10},
            "hovertemplate": (
                f"<b>Origin (tid={ftid})</b><br>"
                f"<b>Type</b>: {otype_str}<br>"
                f"<b>Particle</b>: {pid_str} (pid={ppid})<br>"
                "<b>x</b>: %{x:.2f}<br>"
                "<b>y</b>: %{y:.2f}<br>"
                "<b>z</b>: %{z:.2f}<br>"
                f"<b>PreT0 Origin</b>: {pret0_str}<br>"
            ),
            "marker": {
                "color": origin_color,
                "size": 12,
                "opacity": 1.0,
                "line": {"color": "white", "width": 2},
            },
        })

# ── Start point markers (one per fragment) ──
for i, frag in enumerate(fragments):
    sc = frag["start_coord"]
    trunk_str = "trunk" if frag["istrunk"] == 1 else "secondary"
    pid_str = {22: "g", 11: "e-", -11: "e+"}.get(frag["particle_pid"], str(frag["particle_pid"]))

    # Marker at true position
    traces.append({
        "type": "scatter3d",
        "x": [sc[0]],
        "y": [sc[1]],
        "z": [sc[2]],
        "mode": "markers",
        "name": f"start{i} ({trunk_str})",
        "hovertemplate": (
            f"<b>Start Pt {i}</b><br>"
            f"<b>Trunk</b>: {trunk_str}<br>"
            f"<b>Particle</b>: {pid_str}<br>"
            f"<b>TrackID</b>: {frag['trackid']}<br>"
            "<b>x</b>: %{x:.2f}<br>"
            "<b>y</b>: %{y:.2f}<br>"
            "<b>z</b>: %{z:.2f}<br>"
        ),
        "marker": {
            "color": "rgba(0,255,255,1)",
            "size": 7,
            "opacity": 1.0,
            "symbol": "cross",
            "line": {"color": "white", "width": 1},
        },
    })
    # Text label offset above marker
    traces.append({
        "type": "scatter3d",
        "x": [sc[0]],
        "y": [sc[1] + 3.0],
        "z": [sc[2]],
        "mode": "text",
        "text": [f"S{i}"],
        "textposition": "top center",
        "textfont": {"color": "cyan", "size": 14},
        "showlegend": False,
        "hoverinfo": "skip",
    })

# ── MC Keypoints overlay ──
if kpdata is not None:
    kptype_color = {
        0: "rgba(255,153,51,1.0)",   # nu
        1: "rgba(255,0,0,1.0)",      # track start
        2: "rgba(0,0,255,1.0)",      # track end
        3: "rgba(255,0,125,1.0)",    # shower start
        4: "rgba(125,0,255,1.0)",    # michel
        5: "rgba(0,125,255,1.0)",    # delta
    }
    kptype_name = {
        0: "Nu", 1: "TrackStart", 2: "TrackEnd",
        3: "Shower", 4: "Michel", 5: "Delta",
    }
    for kpt in np.unique(kpdata["kptype"]):
        kmask = kpdata["kptype"] == kpt
        kpos = kpdata["pos"][kmask]
        ktid = kpdata["trackid"][kmask]
        kpid = kpdata["pid"][kmask]
        ksize = 10.0 if kpt == 0 else (8.0 if kpt >= 3 else 5.0)
        kcustom = np.column_stack([
            kpid.astype(np.float64),
            ktid.astype(np.float64),
        ])
        traces.append({
            "type": "scatter3d",
            "x": kpos[:, 0], "y": kpos[:, 1], "z": kpos[:, 2],
            "mode": "markers",
            "name": f"KP:{kptype_name.get(kpt, str(kpt))}",
            "hovertemplate": (
                "<b>x</b>: %{x:.1f}<br>"
                "<b>y</b>: %{y:.1f}<br>"
                "<b>z</b>: %{z:.1f}<br>"
                "<b>PID</b>: %{customdata[0]:d}<br>"
                "<b>TID</b>: %{customdata[1]:d}<br>"
            ),
            "customdata": kcustom,
            "marker": {
                "color": kptype_color.get(kpt, "rgba(255,255,255,1)"),
                "size": ksize,
                "opacity": 1.0,
            },
        })

# ── Detector outline ──
detdata = DetectorOutline()
traces = detdata.getlines() + traces

# ──────────────────────────────────────────────────────────────────
# Dash app
# ──────────────────────────────────────────────────────────────────
app = dash.Dash(
    __name__,
    meta_tags=[{"name": "viewport",
                "content": "width=device-width, initial-scale=1"}],
)

axis_template = {
    "showbackground": True,
    "backgroundcolor": "#141414",
    "gridcolor": "rgb(255, 255, 255)",
    "zerolinecolor": "rgb(255, 255, 255)",
}

fname = os.path.basename(args.input_h5)
title_text = (f"{fname} | {entry_key} | {num_fragments} fragments | "
              f"colorby={args.colorby}")

plot_layout = {
    "title": {"text": title_text, "font": {"color": "white", "size": 14}},
    "height": 850,
    "margin": {"t": 40, "b": 0, "l": 0, "r": 0},
    "font": {"size": 12, "color": "white"},
    "showlegend": True,
    "legend": {"font": {"size": 10, "color": "white"},
               "bgcolor": "rgba(30,30,30,0.8)"},
    "plot_bgcolor": "#141414",
    "paper_bgcolor": "#141414",
    "scene": {
        "xaxis": {**axis_template, "title": "X (drift) [cm]"},
        "yaxis": {**axis_template, "title": "Y (vertical) [cm]"},
        "zaxis": {**axis_template, "title": "Z (beam) [cm]"},
        "aspectratio": {"x": 1, "y": 1, "z": 4},
        "camera": {"eye": {"x": 2, "y": 2, "z": 2},
                   "up": {"x": 0, "y": 1, "z": 0}},
    },
}

# Build summary table HTML
_ORIGIN_TYPE_COLORS = {0: "#00ff00", 1: "#ff0000", 2: "#ffd700"}
summary_rows = []
for i, frag in enumerate(fragments):
    otype = ORIGIN_NAMES.get(frag["origin_type"], "UNK")
    trunk_str = "trunk" if frag["istrunk"] == 1 else ("sec" if frag["istrunk"] == 2 else "?")
    oc = frag["origin_coord"]
    sc = frag["start_coord"]
    nvis = int(np.sum(frag["mask"] & true_mask))
    pid_str = {22: "photon", 11: "e-", -11: "e+"}.get(frag["particle_pid"], str(frag["particle_pid"]))
    p0 = frag["pret0_origin"]
    pret0_str = f"({p0[0]:.1f}, {p0[1]:.1f}, {p0[2]:.1f})" if p0 is not None else "N/A"
    summary_rows.append(
        html.Tr([
            html.Td(str(i), style={"padding": "4px 8px"}),
            html.Td(pid_str, style={"padding": "4px 8px"}),
            html.Td(str(nvis), style={"padding": "4px 8px"}),
            html.Td(otype, style={
                "padding": "4px 8px",
                "color": _ORIGIN_TYPE_COLORS.get(frag["origin_type"], "#ffffff"),
                "fontWeight": "bold",
            }),
            html.Td(trunk_str, style={"padding": "4px 8px"}),
            html.Td(str(frag["trackid"]), style={"padding": "4px 8px"}),
            html.Td(f"({sc[0]:.1f}, {sc[1]:.1f}, {sc[2]:.1f})",
                     style={"padding": "4px 8px"}),
            html.Td(f"({oc[0]:.1f}, {oc[1]:.1f}, {oc[2]:.1f})",
                     style={"padding": "4px 8px"}),
            html.Td(pret0_str, style={"padding": "4px 8px"}),
        ])
    )

summary_table = html.Table(
    [html.Thead(html.Tr([
        html.Th("Frag", style={"padding": "4px 8px"}),
        html.Th("PID", style={"padding": "4px 8px"}),
        html.Th("Pts", style={"padding": "4px 8px"}),
        html.Th("Type", style={"padding": "4px 8px"}),
        html.Th("Trunk", style={"padding": "4px 8px"}),
        html.Th("TrackID", style={"padding": "4px 8px"}),
        html.Th("Start Pt", style={"padding": "4px 8px"}),
        html.Th("Origin Pt", style={"padding": "4px 8px"}),
        html.Th("PreT0 Origin", style={"padding": "4px 8px"}),
    ]))] + [html.Tbody(summary_rows)],
    style={
        "color": "white",
        "borderCollapse": "collapse",
        "margin": "10px",
        "fontSize": "13px",
    },
)

app.layout = html.Div(
    style={"backgroundColor": "#141414"},
    children=[
        summary_table,
        html.Div([
            dcc.Graph(
                id="det3d",
                figure={"data": traces, "layout": plot_layout},
                config={"editable": True, "scrollZoom": False},
            ),
        ], className="graph__container"),
    ],
)

if __name__ == "__main__":
    print(f"\nStarting Dash server on port {args.port}...")
    print(f"Open http://127.0.0.1:{args.port} in your browser\n")
    app.run(debug=True, port=args.port)
