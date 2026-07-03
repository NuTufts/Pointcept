#!/usr/bin/env python3
"""
Visualization tool for LArTPC HDF5 data files.

Displays:
- 3D point cloud visualization with multiple color options
- 2D wire plane images (U, V, Y planes) as sparse representations
- Navigation between entries in the HDF5 file

Usage:
    python visualize_lartpc_h5data.py <hdf5_file> [--start-entry N] [--port PORT]

Example:
    python visualize_lartpc_h5data.py /path/to/file.h5 --start-entry 0 --port 8050
"""

import argparse
import os
import sys
import colorsys
import h5py
import numpy as np
from dash import Dash, html, dcc, callback, Output, Input, State
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from detectoroutline import DetectorOutline

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from lartpc.data_prep.labels.slice_labels import compute_slice_labels, GHOST_SLICE_ID


# PDG code to particle name mapping
PDG_NAMES = {
    11: 'e-',
    -11: 'e+',
    13: 'mu-',
    -13: 'mu+',
    22: 'gamma',
    111: 'pi0',
    211: 'pi+',
    -211: 'pi-',
    2212: 'proton',
    2112: 'neutron',
    321: 'K+',
    -321: 'K-',
    0: 'ghost/unknown',
}

# SSNET class definitions (from SimChTripletLabelMaker)
SSNET_CLASS_NAMES = {
    0: 'background',
    1: 'electron',
    2: 'photon',
    3: 'muon',
    4: 'proton',
    5: 'pion',
    6: 'michel',
    7: 'delta',
    8: 'led',
    9: 'other'
}

SSNET_CLASS_COLORS = {
    0: 'rgba(128,128,128,0.5)',   # background - gray
    1: 'rgba(255,0,0,1)',         # electron - red
    2: 'rgba(255,165,0,1)',       # photon - orange
    3: 'rgba(0,0,255,1)',         # muon - blue
    4: 'rgba(0,191,255,1)',       # proton - deep sky blue
    5: 'rgba(148,0,211,1)',       # pion - violet
    6: 'rgba(255,20,147,1)',      # michel - pink
    7: 'rgba(138,43,226,1)',      # delta - blue violet
    8: 'rgba(173,216,230,1)',     # led - light blue
    9: 'rgba(128,128,0,1)',       # other - olive
}

# Origin type definitions
ORIGIN_NAMES = {
    0: 'unknown',
    1: 'neutrino',
    2: 'cosmic',
}

ORIGIN_COLORS = {
    0: 'rgba(128,128,128,0.5)',  # unknown - gray
    1: 'rgba(255,0,0,1)',        # neutrino - red
    2: 'rgba(0,0,255,1)',        # cosmic - blue
}

def _hsv_palette(n, sat=0.85, val=0.95):
    """Return n distinct RGBA strings spaced evenly in hue."""
    if n <= 0:
        return []
    out = []
    for i in range(n):
        h = (i + 0.1) / max(n, 1)
        r, g, b = colorsys.hsv_to_rgb(h, sat, val)
        out.append(f"rgba({int(r*255)},{int(g*255)},{int(b*255)},1)")
    return out


NU_SLICE_COLOR = 'rgba(255, 60, 60, 1)'      # bright red, reserved for nu slice
NOSLICE_COLOR = 'rgba(110, 110, 110, 0.55)'  # gray, ghosts / orphans


# Keypoint type definitions
KPTYPE_NAMES = {
    0: 'Nu Vertex',
    1: 'Track Start',
    2: 'Track End',
    3: 'Shower',
    4: 'Michel',
    5: 'Delta',
}

KPTYPE_COLORS = {
    0: 'rgba(255,215,0,1)',      # Nu - gold
    1: 'rgba(0,255,0,1)',        # Track Start - green
    2: 'rgba(255,0,0,1)',        # Track End - red
    3: 'rgba(255,105,180,1)',    # Shower - hot pink
    4: 'rgba(138,43,226,1)',     # Michel - blue violet
    5: 'rgba(0,255,255,1)',      # Delta - cyan
}


def get_pdg_name(pdg_code):
    """Convert PDG code to human-readable particle name."""
    return PDG_NAMES.get(pdg_code, f'PDG:{pdg_code}')


def load_event_data(h5file, entry_key):
    """Load data for a single entry from the HDF5 file.

    Handles both simulation (MC) and data-only files. In data-only mode,
    truth fields (pid, origin, hasmatch, trackid, ssnet_label) are not
    present in triplet_data and will be filled with defaults.
    """
    entry_grp = h5file[entry_key]

    data = {}

    # Load triplet data (3D points)
    triplet_data = entry_grp['triplet_data']
    data['pos'] = triplet_data['pos'][:]
    data['pixval'] = triplet_data['pixval'][:]
    data['uwire'] = triplet_data['uwire'][:]
    data['vwire'] = triplet_data['vwire'][:]
    data['ywire'] = triplet_data['ywire'][:]
    data['tick'] = triplet_data['tick'][:]

    n_points = len(data['pos'])

    # Detect data-only mode: truth fields absent from triplet_data
    data['is_data'] = 'pid' not in triplet_data

    # Load truth fields if available, otherwise fill with defaults
    if 'pid' in triplet_data:
        data['pid'] = triplet_data['pid'][:]
    else:
        data['pid'] = np.zeros(n_points, dtype=np.int32)

    if 'origin' in triplet_data:
        data['origin'] = triplet_data['origin'][:]
    else:
        data['origin'] = np.zeros(n_points, dtype=np.int32)

    if 'hasmatch' in triplet_data:
        data['hasmatch'] = triplet_data['hasmatch'][:]
    else:
        # In data-only mode, all points are "real" (no ghost labeling)
        data['hasmatch'] = np.ones(n_points, dtype=np.int32)

    if 'trackid' in triplet_data:
        data['trackid'] = triplet_data['trackid'][:]
    else:
        data['trackid'] = np.zeros(n_points, dtype=np.int32)

    # Load ssnet labels if available
    if 'ssnet_label' in triplet_data:
        data['ssnet_label'] = triplet_data['ssnet_label'][:]
    else:
        data['ssnet_label'] = np.zeros(n_points, dtype=np.int32)

    # Load energy deposition
    if 'edep' in triplet_data:
        data['edep'] = triplet_data['edep'][:]
    else:
        data['edep'] = np.zeros((n_points, 3), dtype=np.float32)

    # Load wire plane images (sparse format). Some merged files (e.g. the
    # reco+truth merge product) intentionally omit image_data — handle that
    # gracefully by leaving wire_images empty so the 2D-plot helpers fall
    # through to their "plane not available" path.
    data['wire_images'] = {}
    if 'image_data' in entry_grp:
        image_data = entry_grp['image_data']
        for plane_idx in range(3):
            plane_key = f'plane{plane_idx}'
            if plane_key in image_data:
                plane_grp = image_data[plane_key]
                data['wire_images'][plane_idx] = {
                    'coord': plane_grp['coord'][:],
                    'feat': plane_grp['feat'][:],
                    'dims': plane_grp['dims'][:],
                    'origin': plane_grp['origin'][:],
                    'pixsize': plane_grp['pixsize'][:]
                }

        # Load triplet to image pixel index mapping
        if 'triplet_imgpix_index' in image_data:
            data['triplet_imgpix_index'] = image_data['triplet_imgpix_index'][:]

    # Compute slice labels from mc_particle_tree, when available. Slices are
    # the instance-segmentation target for the event-slicer model: one slice
    # per cosmic primary, one merged slice for all charge from a nu vertex.
    data['slice_info'] = None
    if not data['is_data'] and 'mc_particle_tree' in entry_grp:
        try:
            data['slice_info'] = compute_slice_labels(
                entry_grp['mc_particle_tree'],
                data['trackid'],
                data['hasmatch'],
            )
        except Exception as exc:
            print(f"[warn] compute_slice_labels failed: {exc}")
            data['slice_info'] = None

    # Load keypoints if available
    if 'mckeypoints' in entry_grp:
        kp_grp = entry_grp['mckeypoints']
        data['keypoints'] = {
            'pos': kp_grp['pos'][:],
            'kptype': kp_grp['kptype'][:],
            'pid': kp_grp['pid'][:],
            'trackid': kp_grp['trackid'][:],
        }
        if 'startpos' in kp_grp:
            data['keypoints']['startpos'] = kp_grp['startpos'][:]

    return data


def create_3d_scatter(event_data, color_mode='ssnet', show_keypoints=True, show_ghosts=False, opacity=0.8, marker_size=2):
    """Create a 3D scatter plot of the point cloud with various coloring options.

    Args:
        event_data: Event data dictionary
        color_mode: Coloring mode ('ssnet', 'origin', 'trackid', 'hasmatch', 'pixval', 'edep')
        show_keypoints: Whether to show keypoints
        show_ghosts: Whether to show ghost points (hasmatch == 0). Default False.
        opacity: Marker opacity
        marker_size: Marker size
    """
    pos = event_data['pos']
    hasmatch = event_data['hasmatch']

    slice_info = event_data.get('slice_info')

    # Filter out ghosts if show_ghosts is False
    if not show_ghosts:
        true_mask = hasmatch == 1
        # Create filtered event data for display
        filtered_event_data = {
            'pos': pos[true_mask],
            'pid': event_data['pid'][true_mask],
            'ssnet_label': event_data['ssnet_label'][true_mask],
            'origin': event_data['origin'][true_mask],
            'trackid': event_data['trackid'][true_mask],
            'hasmatch': hasmatch[true_mask],
            'uwire': event_data['uwire'][true_mask],
            'vwire': event_data['vwire'][true_mask],
            'ywire': event_data['ywire'][true_mask],
            'tick': event_data['tick'][true_mask],
            'pixval': event_data['pixval'][true_mask],
            'edep': event_data['edep'][true_mask],
            'is_data': event_data.get('is_data', False),
        }
        if slice_info is not None:
            filtered_event_data['slice_id'] = slice_info['slice_id'][true_mask]
        # Keep keypoints unchanged
        if 'keypoints' in event_data:
            filtered_event_data['keypoints'] = event_data['keypoints']
        display_data = filtered_event_data
        n_total = len(pos)
        n_displayed = len(display_data['pos'])
        ghost_info = f" (showing {n_displayed}/{n_total}, ghosts hidden)"
    else:
        display_data = dict(event_data)
        if slice_info is not None:
            display_data['slice_id'] = slice_info['slice_id']
        ghost_info = ""

    pos = display_data['pos']

    if len(pos) == 0:
        fig = go.Figure()
        fig.update_layout(title="No points available")
        return fig

    fig = go.Figure()

    is_data = event_data.get('is_data', False)

    # Create hover text
    def make_hover_text(indices):
        hover_texts = []
        for i in indices:
            text = (
                f"x: {pos[i, 0]:.1f}<br>"
                f"y: {pos[i, 1]:.1f}<br>"
                f"z: {pos[i, 2]:.1f}<br>"
                f"Wires: ({display_data['uwire'][i]}, {display_data['vwire'][i]}, {display_data['ywire'][i]})<br>"
                f"Tick: {display_data['tick'][i]}<br>"
                f"PixVal: ({display_data['pixval'][i, 0]:.1f}, {display_data['pixval'][i, 1]:.1f}, {display_data['pixval'][i, 2]:.1f})"
            )
            if not is_data:
                text += (
                    f"<br>PID: {get_pdg_name(display_data['pid'][i])} ({display_data['pid'][i]})<br>"
                    f"SSNET: {SSNET_CLASS_NAMES.get(display_data['ssnet_label'][i], 'unknown')}<br>"
                    f"Origin: {ORIGIN_NAMES.get(display_data['origin'][i], 'unknown')}<br>"
                    f"TrackID: {display_data['trackid'][i]}"
                )
                if 'slice_id' in display_data:
                    sid_i = int(display_data['slice_id'][i])
                    slabel = 'no-slice' if sid_i == GHOST_SLICE_ID else str(sid_i)
                    text += f"<br>Slice: {slabel}"
            hover_texts.append(text)
        return hover_texts

    if color_mode == 'ssnet':
        # Color by SSNET class
        for ssnet_class in np.unique(display_data['ssnet_label']):
            mask = display_data['ssnet_label'] == ssnet_class
            indices = np.where(mask)[0]
            if len(indices) == 0:
                continue

            fig.add_trace(go.Scatter3d(
                x=pos[mask, 0],
                y=pos[mask, 1],
                z=pos[mask, 2],
                mode='markers',
                marker=dict(
                    size=marker_size,
                    color=SSNET_CLASS_COLORS.get(ssnet_class, 'gray'),
                    opacity=opacity
                ),
                name=f"{SSNET_CLASS_NAMES.get(ssnet_class, f'Class {ssnet_class}')} ({len(indices)})",
                hovertext=make_hover_text(indices),
                hoverinfo='text'
            ))

    elif color_mode == 'origin':
        # Color by origin (neutrino vs cosmic)
        for origin_type in np.unique(display_data['origin']):
            mask = display_data['origin'] == origin_type
            indices = np.where(mask)[0]
            if len(indices) == 0:
                continue

            fig.add_trace(go.Scatter3d(
                x=pos[mask, 0],
                y=pos[mask, 1],
                z=pos[mask, 2],
                mode='markers',
                marker=dict(
                    size=marker_size,
                    color=ORIGIN_COLORS.get(origin_type, 'gray'),
                    opacity=opacity
                ),
                name=f"{ORIGIN_NAMES.get(origin_type, f'Origin {origin_type}')} ({len(indices)})",
                hovertext=make_hover_text(indices),
                hoverinfo='text'
            ))

    elif color_mode == 'trackid':
        # Color by track ID
        unique_trackids = np.unique(display_data['trackid'])
        np.random.seed(42)  # For reproducible colors
        for tid in unique_trackids:
            if tid <= 0:
                continue
            mask = display_data['trackid'] == tid
            indices = np.where(mask)[0]
            if len(indices) == 0:
                continue

            r, g, b = np.random.randint(50, 255, 3)
            color = f'rgba({r},{g},{b},1)'

            fig.add_trace(go.Scatter3d(
                x=pos[mask, 0],
                y=pos[mask, 1],
                z=pos[mask, 2],
                mode='markers',
                marker=dict(size=marker_size, color=color, opacity=opacity),
                name=f"TrackID {tid} ({len(indices)})",
                hovertext=make_hover_text(indices),
                hoverinfo='text'
            ))

    elif color_mode in ('slice', 'slice_pts'):
        # Color by truth slice id (instance-segmentation target).
        # 'slice'      = points only
        # 'slice_pts'  = also overlay nu_vertex + per-cosmic-primary start_pos
        # The origin points sit well above the detector volume in z and clutter
        # the default view, so they're opt-in via the second dropdown entry.
        show_origin_points = (color_mode == 'slice_pts')
        if 'slice_id' not in display_data:
            fig.update_layout(title="No slice info (mc_particle_tree missing)")
            return fig

        sid = np.asarray(display_data['slice_id'])
        info = event_data.get('slice_info')
        keys = info['primary_trackid'] if info is not None else np.unique(
            sid[sid != GHOST_SLICE_ID])
        origins = info['primary_origin'] if info is not None else None
        pids = info['primary_pid'] if info is not None else None
        n_pts = info['primary_n_spacepoints'] if info is not None else None

        # Build color map: nu slices red; cosmic slices spread over HSV.
        if origins is not None:
            cosmic_idx = [k for k, o in enumerate(origins) if int(o) == 2]
        else:
            cosmic_idx = list(range(len(keys)))
        cosmic_palette = _hsv_palette(len(cosmic_idx))
        color_for_key = {}
        cosmic_color_iter = iter(cosmic_palette)
        for k, key in enumerate(keys):
            if origins is not None and int(origins[k]) == 1:
                color_for_key[int(key)] = NU_SLICE_COLOR
            else:
                color_for_key[int(key)] = next(cosmic_color_iter, 'rgba(200,200,200,1)')

        # Ghosts / orphans first (drawn underneath in the legend ordering).
        ghost_mask = (sid == GHOST_SLICE_ID)
        if ghost_mask.any():
            indices = np.where(ghost_mask)[0]
            fig.add_trace(go.Scatter3d(
                x=pos[ghost_mask, 0], y=pos[ghost_mask, 1], z=pos[ghost_mask, 2],
                mode='markers',
                marker=dict(size=max(1, marker_size - 1),
                            color=NOSLICE_COLOR, opacity=opacity * 0.6),
                name=f"no-slice / ghost ({int(ghost_mask.sum())})",
                hovertext=make_hover_text(indices),
                hoverinfo='text',
            ))

        for k, key in enumerate(keys):
            mask = sid == int(key)
            indices = np.where(mask)[0]
            if len(indices) == 0:
                continue
            color = color_for_key[int(key)]
            if origins is not None and int(origins[k]) == 1:
                origin_lbl = 'nu'
            elif origins is not None and int(origins[k]) == 2:
                origin_lbl = 'cosmic'
            else:
                origin_lbl = '?'
            pid_str = (get_pdg_name(int(pids[k]))
                       if pids is not None and int(pids[k]) != 0 else origin_lbl)
            name = f"slice {int(key)} [{origin_lbl}/{pid_str}] ({int(mask.sum())})"
            fig.add_trace(go.Scatter3d(
                x=pos[mask, 0], y=pos[mask, 1], z=pos[mask, 2],
                mode='markers',
                marker=dict(size=marker_size, color=color, opacity=opacity),
                name=name,
                hovertext=make_hover_text(indices),
                hoverinfo='text',
            ))

        # Overlay markers for slice "origin" points: nu vertex (gold star) and
        # each cosmic primary's start_pos (small marker matching its slice color).
        if show_origin_points and info is not None and len(info.get('nu_vertices', [])) > 0:
            nuv = info['nu_vertices']
            fig.add_trace(go.Scatter3d(
                x=nuv[:, 0], y=nuv[:, 1], z=nuv[:, 2],
                mode='markers',
                marker=dict(size=14, color='rgba(255,215,0,1)',
                            symbol='diamond', line=dict(width=2, color='black')),
                name='nu vertex',
                hovertext=[f"nu vertex<br>({p[0]:.1f},{p[1]:.1f},{p[2]:.1f})"
                           for p in nuv],
                hoverinfo='text',
            ))
        if show_origin_points and info is not None:
            cosmic_starts = []
            cosmic_colors = []
            cosmic_texts = []
            for k, key in enumerate(info['primary_trackid']):
                if int(info['primary_origin'][k]) != 2:
                    continue
                sp = info['primary_start_pos'][k]
                cosmic_starts.append(sp)
                cosmic_colors.append(color_for_key[int(key)])
                cosmic_texts.append(
                    f"cosmic primary tid={int(key)}<br>"
                    f"pid={get_pdg_name(int(info['primary_pid'][k]))}<br>"
                    f"start=({sp[0]:.1f},{sp[1]:.1f},{sp[2]:.1f})"
                )
            if cosmic_starts:
                cosmic_starts = np.asarray(cosmic_starts)
                fig.add_trace(go.Scatter3d(
                    x=cosmic_starts[:, 0], y=cosmic_starts[:, 1], z=cosmic_starts[:, 2],
                    mode='markers',
                    marker=dict(size=6, color=cosmic_colors,
                                symbol='circle', line=dict(width=1, color='black')),
                    name=f"cosmic primaries ({len(cosmic_starts)})",
                    hovertext=cosmic_texts,
                    hoverinfo='text',
                ))

    elif color_mode == 'hasmatch':
        # Color by has match (true vs ghost)
        for match_val, name, color in [(1, 'True', 'green'), (0, 'Ghost', 'red')]:
            mask = display_data['hasmatch'] == match_val
            indices = np.where(mask)[0]
            if len(indices) == 0:
                continue

            fig.add_trace(go.Scatter3d(
                x=pos[mask, 0],
                y=pos[mask, 1],
                z=pos[mask, 2],
                mode='markers',
                marker=dict(size=marker_size, color=color, opacity=opacity),
                name=f"{name} ({len(indices)})",
                hovertext=make_hover_text(indices),
                hoverinfo='text'
            ))

    elif color_mode == 'pixval':
        # Color by pixel value (sum of 3 planes)
        pixval_sum = np.sum(display_data['pixval'], axis=1)
        indices = np.arange(len(pos))

        fig.add_trace(go.Scatter3d(
            x=pos[:, 0],
            y=pos[:, 1],
            z=pos[:, 2],
            mode='markers',
            marker=dict(
                size=marker_size,
                color=pixval_sum,
                colorscale='Viridis',
                opacity=opacity,
                colorbar=dict(title='Pixel Value'),
                cmin=0,
                cmax=np.percentile(pixval_sum, 95)
            ),
            name='Pixel Values',
            hovertext=make_hover_text(indices),
            hoverinfo='text'
        ))

    elif color_mode == 'edep':
        # Color by energy deposition (sum of 3 planes)
        edep_sum = np.sum(display_data['edep'], axis=1)
        indices = np.arange(len(pos))

        fig.add_trace(go.Scatter3d(
            x=pos[:, 0],
            y=pos[:, 1],
            z=pos[:, 2],
            mode='markers',
            marker=dict(
                size=marker_size,
                color=edep_sum,
                colorscale='Hot',
                opacity=opacity,
                colorbar=dict(title='Energy Dep (MeV)'),
                cmin=0,
                cmax=np.percentile(edep_sum[edep_sum > 0], 95) if np.any(edep_sum > 0) else 1
            ),
            name='Energy Deposition',
            hovertext=make_hover_text(indices),
            hoverinfo='text'
        ))

    # Add keypoints if available and requested
    if show_keypoints and 'keypoints' in event_data:
        kp_data = event_data['keypoints']
        kp_pos = kp_data['pos']
        kp_types = kp_data['kptype']

        kp_trackids = kp_data['trackid']
        kp_pids = kp_data['pid']

        for kptype in np.unique(kp_types):
            mask = kp_types == kptype
            kp_pts = kp_pos[mask]
            kp_tids = kp_trackids[mask]
            kp_pids_masked = kp_pids[mask]

            if len(kp_pts) == 0:
                continue

            hover_texts = [
                f"{KPTYPE_NAMES.get(kptype, 'KP')}<br>"
                f"({p[0]:.1f}, {p[1]:.1f}, {p[2]:.1f})<br>"
                f"TrackID: {tid}<br>"
                f"PID: {get_pdg_name(pid)} ({pid})"
                for p, tid, pid in zip(kp_pts, kp_tids, kp_pids_masked)
            ]

            fig.add_trace(go.Scatter3d(
                x=kp_pts[:, 0],
                y=kp_pts[:, 1],
                z=kp_pts[:, 2],
                mode='markers',
                marker=dict(
                    size=8,
                    color=KPTYPE_COLORS.get(kptype, 'white'),
                    symbol='diamond',
                    line=dict(width=2, color='black')
                ),
                name=f"KP: {KPTYPE_NAMES.get(kptype, f'Type {kptype}')}",
                hoverinfo='text',
                hovertext=hover_texts,
            ))

        # Add photon start positions (pid == 22)
        if 'startpos' in kp_data:
            photon_mask = kp_pids == 22
            print("number of photon keypoints: ",len(photon_mask))
            if np.any(photon_mask):
                photon_startpos = kp_data['startpos'][photon_mask]
                photon_tids = kp_trackids[photon_mask]
                photon_pids = kp_pids[photon_mask]
                photon_kptypes = kp_types[photon_mask]

                hover_texts = [
                    f"Photon Start<br>"
                    f"({p[0]:.1f}, {p[1]:.1f}, {p[2]:.1f})<br>"
                    f"TrackID: {tid}<br>"
                    f"PID: {get_pdg_name(pid)} ({pid})<br>"
                    f"KP Type: {KPTYPE_NAMES.get(kpt, f'Type {kpt}')}"
                    for p, tid, pid, kpt in zip(photon_startpos, photon_tids, photon_pids, photon_kptypes)
                ]

                fig.add_trace(go.Scatter3d(
                    x=photon_startpos[:, 0],
                    y=photon_startpos[:, 1],
                    z=photon_startpos[:, 2],
                    mode='markers',
                    marker=dict(
                        size=8,
                        color='rgba(255,165,0,1)',  # orange
                        symbol='circle',
                        line=dict(width=2, color='black')
                    ),
                    name=f"Photon Start ({len(photon_startpos)})",
                    hoverinfo='text',
                    hovertext=hover_texts,
                ))
        else:
            print("startpos not in mckeypoints group")

    # Add detector outline
    detdata = DetectorOutline()
    det_traces = detdata.getlines(color=(255, 255, 255))
    for det_trace in det_traces:
        fig.add_trace(go.Scatter3d(
            x=det_trace['x'],
            y=det_trace['y'],
            z=det_trace['z'],
            mode='lines',
            name='Detector',
            line=dict(color=det_trace['line']['color'], width=det_trace['line']['width']),
            showlegend=False
        ))

    # Dark theme axis template
    axis_template = dict(
        showbackground=True,
        backgroundcolor="#141414",
        gridcolor="rgb(80, 80, 80)",
        zerolinecolor="rgb(128, 128, 128)",
        title_font=dict(color="white"),
        tickfont=dict(color="white")
    )

    fig.update_layout(
        scene=dict(
            xaxis=dict(**axis_template, title='X (cm)'),
            yaxis=dict(**axis_template, title='Y (cm)'),
            zaxis=dict(**axis_template, title='Z (cm)'),
            aspectratio=dict(x=1, y=1, z=4),
            camera=dict(
                eye=dict(x=2, y=2, z=2),
                up=dict(x=0, y=1, z=0)
            ),
        ),
        title=dict(
            text=f"3D Point Cloud - {color_mode.upper()} coloring ({len(pos)} points){ghost_info}",
            font=dict(color="white")
        ),
        height=800,
        margin=dict(l=0, r=0, t=40, b=0),
        legend=dict(
            yanchor="top", y=0.99, xanchor="left", x=0.01,
            font=dict(color="white"),
            bgcolor="rgba(20, 20, 20, 0.8)"
        ),
        plot_bgcolor="#141414",
        paper_bgcolor="#141414",
    )

    return fig


def create_wire_plane_images(event_data, plane_idx, vmax_percentile=99):
    """Create a 2D image visualization for a single wire plane."""
    if plane_idx not in event_data['wire_images']:
        fig = go.Figure()
        fig.update_layout(title=f"Plane {plane_idx} not available")
        return fig

    plane_data = event_data['wire_images'][plane_idx]
    coords = plane_data['coord']  # (N, 2) - [col, row]
    feats = plane_data['feat']    # (N,)
    dims = plane_data['dims']     # [ncols, nrows]

    if len(coords) == 0:
        fig = go.Figure()
        fig.update_layout(title=f"Plane {plane_idx} - No pixels")
        return fig

    # Create dense image for visualization
    # Note: dims is [ncols, nrows], coords is [col, row]
    ncols, nrows = int(dims[0]), int(dims[1])

    # Clip coordinates to valid range
    cols = np.clip(coords[:, 0], 0, ncols - 1).astype(int)
    rows = np.clip(coords[:, 1], 0, nrows - 1).astype(int)

    # Calculate value limits for better visualization
    vmax = np.percentile(feats, vmax_percentile) if len(feats) > 0 else 1
    vmin = 0

    # Use scatter plot for sparse data (more efficient than dense image)
    fig = go.Figure()

    fig.add_trace(go.Scattergl(
        x=cols,
        y=rows,
        mode='markers',
        marker=dict(
            size=2,
            color=feats,
            colorscale='Viridis',
            cmin=vmin,
            cmax=vmax,
            colorbar=dict(title='ADC')
        ),
        hovertemplate='Wire: %{x}<br>Tick/Row: %{y}<br>ADC: %{marker.color:.1f}<extra></extra>'
    ))

    plane_names = ['U Plane', 'V Plane', 'Y Plane']
    fig.update_layout(
        title=f"{plane_names[plane_idx]} ({len(coords)} pixels)",
        xaxis_title='Wire',
        yaxis_title='Tick/Row',
        height=500,
        margin=dict(l=50, r=20, t=40, b=40),
        xaxis=dict(range=[0, 3456]),
        yaxis=dict(range=[0, 1050], scaleanchor='x', scaleratio=1)
    )

    return fig


def create_combined_wire_images(event_data, vmax_percentile=99):
    """Create a combined figure with all 3 wire plane images."""
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=['U Plane', 'V Plane', 'Y Plane'],
        horizontal_spacing=0.05
    )

    plane_names = ['U', 'V', 'Y']

    for plane_idx in range(3):
        if plane_idx not in event_data['wire_images']:
            continue

        plane_data = event_data['wire_images'][plane_idx]
        coords = plane_data['coord']
        feats = plane_data['feat']
        dims = plane_data['dims']

        if len(coords) == 0:
            continue

        ncols, nrows = int(dims[0]), int(dims[1])
        cols = np.clip(coords[:, 0], 0, ncols - 1).astype(int)
        rows = np.clip(coords[:, 1], 0, nrows - 1).astype(int)

        vmax = np.percentile(feats, vmax_percentile) if len(feats) > 0 else 1

        fig.add_trace(
            go.Scattergl(
                x=cols,
                y=rows,
                mode='markers',
                marker=dict(
                    size=1,
                    color=feats,
                    colorscale='Viridis',
                    cmin=0,
                    cmax=vmax,
                    showscale=(plane_idx == 2)  # Only show colorbar on last plane
                ),
                name=f'{plane_names[plane_idx]} ({len(coords)} px)',
                hovertemplate=f'{plane_names[plane_idx]}<br>Wire: %{{x}}<br>Tick: %{{y}}<br>ADC: %{{marker.color:.1f}}<extra></extra>'
            ),
            row=1, col=plane_idx + 1
        )

        fig.update_xaxes(title_text='Wire', range=[0, 3456], row=1, col=plane_idx + 1)
        fig.update_yaxes(title_text='Tick', range=[0, 1050], row=1, col=plane_idx + 1)

    fig.update_layout(
        height=400,
        title='Wire Plane Images (Full View)',
        showlegend=True,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
    )

    return fig


def create_zoomed_wire_images(event_data, uwire, vwire, ywire, tick, half_width=25, vmax_percentile=99):
    """Create zoomed wire plane images centered on the clicked 3D point using heatmaps.

    Args:
        event_data: Event data dictionary
        uwire: U plane wire coordinate of clicked point
        vwire: V plane wire coordinate of clicked point
        ywire: Y plane wire coordinate of clicked point
        tick: Time tick coordinate of clicked point
        half_width: Half width of the zoom window (default 25 for 50x50 view)
        vmax_percentile: Percentile for color scale max
    """
    # Convert tick to row coordinate: row = (tick - 2400) / 6.0
    row_center = (tick - 2400) / 6.0

    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=[
            f'U Plane (wire={uwire})',
            f'V Plane (wire={vwire})',
            f'Y Plane (wire={ywire})'
        ],
        horizontal_spacing=0.08
    )

    plane_names = ['U', 'V', 'Y']
    wire_centers = [uwire, vwire, ywire]

    # Size of the dense image
    img_size = 2 * half_width + 1

    for plane_idx in range(3):
        if plane_idx not in event_data['wire_images']:
            continue

        plane_data = event_data['wire_images'][plane_idx]
        coords = plane_data['coord']
        feats = plane_data['feat']

        if len(coords) == 0:
            continue

        wire_center = wire_centers[plane_idx]

        # Define zoom window using converted row coordinate
        wire_min = wire_center - half_width
        wire_max = wire_center + half_width
        row_min = row_center - half_width
        row_max = row_center + half_width

        # Filter pixels within the zoom window
        cols = coords[:, 0]
        rows = coords[:, 1]
        mask = (cols >= wire_min) & (cols <= wire_max) & (rows >= row_min) & (rows <= row_max)

        filtered_cols = cols[mask]
        filtered_rows = rows[mask]
        filtered_feats = feats[mask]

        # Create dense image array (initialize with zeros/NaN for empty pixels)
        dense_img = np.zeros((img_size, img_size), dtype=np.float32)

        # Fill in the sparse values
        for c, r, f in zip(filtered_cols, filtered_rows, filtered_feats):
            # Convert to array indices
            col_idx = int(c - wire_min)
            row_idx = int(r - row_min)
            if 0 <= col_idx < img_size and 0 <= row_idx < img_size:
                dense_img[row_idx, col_idx] = f

        vmax = np.percentile(feats, vmax_percentile) if len(feats) > 0 else 1

        # Create x and y axis values for proper labeling
        x_vals = np.arange(wire_min, wire_max + 1)
        y_vals = np.arange(row_min, row_max + 1)

        # Add heatmap for the zoomed region
        fig.add_trace(
            go.Heatmap(
                z=dense_img,
                x=x_vals,
                y=y_vals,
                colorscale='Viridis',
                zmin=0,
                zmax=vmax,
                showscale=(plane_idx == 2),
                colorbar=dict(title='ADC') if plane_idx == 2 else None,
                hovertemplate=f'{plane_names[plane_idx]}<br>Wire: %{{x}}<br>Row: %{{y}}<br>ADC: %{{z:.1f}}<extra></extra>'
            ),
            row=1, col=plane_idx + 1
        )

        # Add crosshair at the center point
        fig.add_trace(
            go.Scatter(
                x=[wire_center],
                y=[row_center],
                mode='markers',
                marker=dict(
                    size=15,
                    color='red',
                    symbol='cross',
                    line=dict(width=2, color='white')
                ),
                name='Selected Point' if plane_idx == 0 else None,
                showlegend=(plane_idx == 0),
                hovertemplate=f'Selected<br>Wire: {wire_center}<br>Row: {row_center:.1f}<extra></extra>'
            ),
            row=1, col=plane_idx + 1
        )

        # Set axis ranges and titles
        fig.update_xaxes(
            title_text='Wire',
            row=1, col=plane_idx + 1
        )
        fig.update_yaxes(
            title_text='Row',
            row=1, col=plane_idx + 1
        )

    fig.update_layout(
        height=400,
        title=f'Zoomed Wire Plane Images (50x50 around tick={tick}, row={row_center:.1f})',
        showlegend=True,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
    )

    return fig


def create_empty_zoomed_view():
    """Create an empty placeholder for the zoomed view before any point is clicked."""
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=['U Plane', 'V Plane', 'Y Plane'],
        horizontal_spacing=0.08
    )

    fig.update_layout(
        height=400,
        title='Click on a 3D point to see zoomed wire plane images (50x50 view)',
        showlegend=False,
        annotations=[
            dict(
                text="Click on a point in the 3D view above",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=16, color="gray")
            )
        ]
    )

    return fig


def main():
    parser = argparse.ArgumentParser(description='Visualize LArTPC HDF5 data files')
    parser.add_argument('h5file', type=str, help='Path to HDF5 file')
    parser.add_argument('--start-entry', type=int, default=0, help='Starting entry index')
    parser.add_argument('--port', type=int, default=8050, help='Port for Dash server')
    args = parser.parse_args()

    # Open HDF5 file and get entry keys
    h5file = h5py.File(args.h5file, 'r')
    entry_keys = [k for k in h5file.keys() if k.startswith('entry_')]
    entry_keys = sorted(entry_keys, key=lambda x: int(x.split('_')[1]))
    num_entries = len(entry_keys)

    if num_entries == 0:
        print("No entries found in HDF5 file")
        h5file.close()
        return

    start_entry = min(args.start_entry, num_entries - 1)

    # Detect data-only mode from the first entry
    first_entry_grp = h5file[entry_keys[0]]
    is_data_file = 'pid' not in first_entry_grp['triplet_data']

    # Color mode options - filter out MC-only options for data files
    if is_data_file:
        color_options = [
            {'label': 'Pixel Value', 'value': 'pixval'},
            {'label': 'Energy Deposition', 'value': 'edep'},
        ]
        default_color_mode = 'pixval'
    else:
        color_options = [
            {'label': 'SSNET Class', 'value': 'ssnet'},
            {'label': 'Origin (Nu/Cosmic)', 'value': 'origin'},
            {'label': 'Slice (Truth Instance)', 'value': 'slice'},
            {'label': 'Slice + Origin Points', 'value': 'slice_pts'},
            {'label': 'Track ID', 'value': 'trackid'},
            {'label': 'True/Ghost', 'value': 'hasmatch'},
            {'label': 'Pixel Value', 'value': 'pixval'},
            {'label': 'Energy Deposition', 'value': 'edep'},
        ]
        default_color_mode = 'ssnet'

    # Create Dash app
    app = Dash(__name__, suppress_callback_exceptions=True)

    app.layout = html.Div([
        html.H1("LArTPC Data Viewer" + (" (Data-Only Mode)" if is_data_file else ""),
                style={'textAlign': 'center'}),

        # File info
        html.Div([
            html.P(f"File: {args.h5file}", style={'fontWeight': 'bold'}),
            html.P(f"Total entries: {num_entries}")
        ], style={'textAlign': 'center', 'marginBottom': '10px'}),

        # Navigation controls
        html.Div([
            html.Button('Previous', id='prev-btn', n_clicks=0,
                        style={'marginRight': '10px', 'fontSize': '16px', 'padding': '10px 20px'}),
            html.Span(id='entry-info', style={'fontSize': '18px', 'marginRight': '10px'}),
            html.Button('Next', id='next-btn', n_clicks=0,
                        style={'marginLeft': '10px', 'fontSize': '16px', 'padding': '10px 20px'}),
            html.Span(" | Go to entry: ", style={'marginLeft': '20px'}),
            dcc.Input(id='entry-input', type='number', value=start_entry, min=0, max=num_entries - 1,
                      style={'width': '80px', 'marginRight': '10px'}),
            html.Button('Go', id='go-btn', n_clicks=0,
                        style={'fontSize': '16px', 'padding': '5px 15px'}),
        ], style={'textAlign': 'center', 'marginBottom': '20px', 'padding': '10px'}),

        # Store current entry index
        dcc.Store(id='current-entry', data=start_entry),

        # Store for clicked point data
        dcc.Store(id='clicked-point-data', data=None),

        # Store for current event data (wire coordinates)
        dcc.Store(id='event-wire-data', data=None),

        # Visualization options
        html.Div([
            html.Div([
                html.Label('3D Color Mode:', style={'fontWeight': 'bold', 'marginRight': '10px'}),
                dcc.Dropdown(
                    id='color-mode-dropdown',
                    options=color_options,
                    value=default_color_mode,
                    style={'width': '200px', 'display': 'inline-block'}
                ),
            ], style={'display': 'inline-block', 'marginRight': '30px'}),

            html.Div([
                dcc.Checklist(
                    id='show-keypoints',
                    options=[{'label': ' Show Keypoints', 'value': 'show'}],
                    value=['show'] if not is_data_file else [],
                    style={'display': 'inline-block'}
                ),
            ], style={'display': 'inline-block' if not is_data_file else 'none', 'marginRight': '30px'}),

            html.Div([
                dcc.Checklist(
                    id='show-ghosts',
                    options=[{'label': ' Show Ghosts', 'value': 'show'}],
                    value=[],  # Default off
                    style={'display': 'inline-block'}
                ),
            ], style={'display': 'inline-block' if not is_data_file else 'none', 'marginRight': '30px'}),

            html.Div([
                html.Label('Marker Size:', style={'fontWeight': 'bold', 'marginRight': '10px'}),
                dcc.Slider(
                    id='marker-size-slider',
                    min=1,
                    max=10,
                    step=0.5,
                    value=2,
                    marks={1: '1', 5: '5', 10: '10'},
                    tooltip={'placement': 'bottom', 'always_visible': True}
                ),
            ], style={'display': 'inline-block', 'width': '200px'}),
        ], style={'textAlign': 'center', 'marginBottom': '20px', 'padding': '10px', 'backgroundColor': '#f0f0f0'}),

        # Statistics display
        html.Div(id='stats-display', style={
            'textAlign': 'center',
            'marginBottom': '10px',
            'padding': '10px',
            'backgroundColor': '#e8e8e8',
            'borderRadius': '5px'
        }),

        # 3D scatter plot
        html.Div([
            dcc.Graph(id='scatter-3d', style={'width': '100%'})
        ], style={'marginBottom': '20px'}),

        # Zoomed wire plane images (click-responsive)
        html.H3("Wire Plane Images (Click 3D Point to Zoom)", style={'textAlign': 'center'}),
        html.Div(id='clicked-point-info', style={
            'textAlign': 'center',
            'marginBottom': '10px',
            'padding': '5px',
            'backgroundColor': '#d4edda',
            'borderRadius': '5px',
            'display': 'none'  # Hidden until a point is clicked
        }),
        html.Div([
            dcc.Graph(id='wire-images-zoomed', style={'width': '100%'})
        ], style={'marginBottom': '20px'}),

        # Full wire plane views and individual planes (collapsible)
        html.Details([
            html.Summary("Full Wire Plane Views (click to expand)", style={'cursor': 'pointer', 'fontWeight': 'bold'}),
            html.Div([
                # Combined full view
                html.H4("Combined View", style={'textAlign': 'center', 'marginTop': '10px'}),
                html.Div([
                    dcc.Graph(id='wire-images-combined', style={'width': '100%'})
                ], style={'marginBottom': '20px'}),

                # Individual plane views
                html.H4("Individual Plane Views", style={'textAlign': 'center'}),
                html.Div([
                    dcc.Graph(id='wire-image-0', style={'width': '100%'})
                ], style={'marginBottom': '10px'}),
                html.Div([
                    dcc.Graph(id='wire-image-1', style={'width': '100%'})
                ], style={'marginBottom': '10px'}),
                html.Div([
                    dcc.Graph(id='wire-image-2', style={'width': '100%'})
                ], style={'marginBottom': '10px'}),
            ])
        ], style={'marginBottom': '20px'}),
    ], style={'padding': '20px'})

    @callback(
        Output('current-entry', 'data'),
        Output('entry-input', 'value'),
        Input('prev-btn', 'n_clicks'),
        Input('next-btn', 'n_clicks'),
        Input('go-btn', 'n_clicks'),
        State('current-entry', 'data'),
        State('entry-input', 'value')
    )
    def update_entry(prev_clicks, next_clicks, go_clicks, current_entry, input_value):
        from dash import ctx
        triggered_id = ctx.triggered_id

        if triggered_id == 'prev-btn':
            new_entry = max(0, current_entry - 1)
        elif triggered_id == 'next-btn':
            new_entry = min(num_entries - 1, current_entry + 1)
        elif triggered_id == 'go-btn':
            new_entry = max(0, min(num_entries - 1, input_value if input_value is not None else 0))
        else:
            new_entry = current_entry

        return new_entry, new_entry

    @callback(
        Output('entry-info', 'children'),
        Output('stats-display', 'children'),
        Output('scatter-3d', 'figure'),
        Output('wire-images-combined', 'figure'),
        Output('wire-image-0', 'figure'),
        Output('wire-image-1', 'figure'),
        Output('wire-image-2', 'figure'),
        Output('wire-images-zoomed', 'figure'),
        Output('event-wire-data', 'data'),
        Output('clicked-point-data', 'data'),
        Output('clicked-point-info', 'children'),
        Output('clicked-point-info', 'style'),
        Input('current-entry', 'data'),
        Input('color-mode-dropdown', 'value'),
        Input('show-keypoints', 'value'),
        Input('show-ghosts', 'value'),
        Input('marker-size-slider', 'value')
    )
    def update_display(entry_idx, color_mode, show_keypoints, show_ghosts, marker_size):
        entry_key = entry_keys[entry_idx]
        event_data = load_event_data(h5file, entry_key)

        # Entry info
        entry_info = f"Entry {entry_idx}/{num_entries - 1} ({entry_key})"

        # Statistics
        n_points = len(event_data['pos'])
        is_data = event_data.get('is_data', False)

        if is_data:
            stats_text = f"Total Points: {n_points} | Data-only mode (no MC truth)"
        else:
            n_true = np.sum(event_data['hasmatch'] == 1)
            n_ghost = np.sum(event_data['hasmatch'] == 0)
            n_nu = np.sum(event_data['origin'] == 1)
            n_cosmic = np.sum(event_data['origin'] == 2)

            stats_text = (
                f"Total Points: {n_points} | "
                f"True: {n_true} ({100*n_true/n_points:.1f}%) | "
                f"Ghost: {n_ghost} ({100*n_ghost/n_points:.1f}%) | "
                f"Neutrino: {n_nu} ({100*n_nu/n_points:.1f}%) | "
                f"Cosmic: {n_cosmic} ({100*n_cosmic/n_points:.1f}%)"
            )
            sinfo = event_data.get('slice_info')
            if sinfo is not None:
                n_slices = len(sinfo['primary_trackid'])
                n_nu_slices = int((sinfo['primary_origin'] == 1).sum())
                n_cos_slices = int((sinfo['primary_origin'] == 2).sum())
                stats_text += (
                    f" | Slices: {n_slices} (nu={n_nu_slices}, cos={n_cos_slices})"
                )

        # 3D scatter plot
        scatter_fig = create_3d_scatter(
            event_data,
            color_mode=color_mode,
            show_keypoints='show' in (show_keypoints or []),
            show_ghosts='show' in (show_ghosts or []),
            marker_size=marker_size
        )

        # Wire plane images (full view)
        combined_fig = create_combined_wire_images(event_data)
        plane0_fig = create_wire_plane_images(event_data, 0)
        plane1_fig = create_wire_plane_images(event_data, 1)
        plane2_fig = create_wire_plane_images(event_data, 2)

        # Empty zoomed view initially
        zoomed_fig = create_empty_zoomed_view()

        # Store wire coordinate data for click handling
        wire_data = {
            'uwire': event_data['uwire'].tolist(),
            'vwire': event_data['vwire'].tolist(),
            'ywire': event_data['ywire'].tolist(),
            'tick': event_data['tick'].tolist(),
            'pos': event_data['pos'].tolist(),
            'pid': event_data['pid'].tolist(),
            'ssnet_label': event_data['ssnet_label'].tolist(),
            'origin': event_data['origin'].tolist(),
            'hasmatch': event_data['hasmatch'].tolist(),
            'is_data': is_data,
        }

        # Reset clicked point data and hide info box
        clicked_point_info = ""
        info_style = {
            'textAlign': 'center',
            'marginBottom': '10px',
            'padding': '5px',
            'backgroundColor': '#d4edda',
            'borderRadius': '5px',
            'display': 'none'
        }

        return (entry_info, stats_text, scatter_fig, combined_fig,
                plane0_fig, plane1_fig, plane2_fig, zoomed_fig,
                wire_data, None, clicked_point_info, info_style)

    @callback(
        Output('wire-images-zoomed', 'figure', allow_duplicate=True),
        Output('clicked-point-info', 'children', allow_duplicate=True),
        Output('clicked-point-info', 'style', allow_duplicate=True),
        Input('scatter-3d', 'clickData'),
        State('event-wire-data', 'data'),
        State('current-entry', 'data'),
        prevent_initial_call=True
    )
    def handle_3d_click(click_data, wire_data, entry_idx):
        if click_data is None or wire_data is None:
            return create_empty_zoomed_view(), "", {'display': 'none'}

        # Get the clicked point info
        point = click_data['points'][0]

        # Get the point index from customdata or find nearest point
        # The click_data contains x, y, z coordinates of clicked point
        clicked_x = point['x']
        clicked_y = point['y']
        clicked_z = point['z']

        # Find the index of the clicked point by matching coordinates
        pos_array = np.array(wire_data['pos'])
        distances = np.sqrt(
            (pos_array[:, 0] - clicked_x)**2 +
            (pos_array[:, 1] - clicked_y)**2 +
            (pos_array[:, 2] - clicked_z)**2
        )
        point_idx = np.argmin(distances)

        # Get wire coordinates for this point
        uwire = wire_data['uwire'][point_idx]
        vwire = wire_data['vwire'][point_idx]
        ywire = wire_data['ywire'][point_idx]
        tick = wire_data['tick'][point_idx]

        # Load event data again for the zoomed image
        entry_key = entry_keys[entry_idx]
        event_data = load_event_data(h5file, entry_key)

        # Create zoomed view
        zoomed_fig = create_zoomed_wire_images(
            event_data, uwire, vwire, ywire, tick, half_width=25
        )

        # Create info text
        is_data = wire_data.get('is_data', False)
        info_text = (
            f"Selected Point: 3D=({clicked_x:.1f}, {clicked_y:.1f}, {clicked_z:.1f}) | "
            f"Wires: U={uwire}, V={vwire}, Y={ywire} | Tick={tick}"
        )
        if not is_data:
            pid = wire_data['pid'][point_idx]
            ssnet = wire_data['ssnet_label'][point_idx]
            origin = wire_data['origin'][point_idx]
            hasmatch = wire_data['hasmatch'][point_idx]
            origin_name = ORIGIN_NAMES.get(origin, 'unknown')
            ssnet_name = SSNET_CLASS_NAMES.get(ssnet, 'unknown')
            pdg_name = get_pdg_name(pid)
            match_str = "True" if hasmatch == 1 else "Ghost"
            info_text += f" | PID={pdg_name} | SSNET={ssnet_name} | Origin={origin_name} | {match_str}"

        info_style = {
            'textAlign': 'center',
            'marginBottom': '10px',
            'padding': '10px',
            'backgroundColor': '#d4edda',
            'borderRadius': '5px',
            'display': 'block',
            'fontWeight': 'bold'
        }

        return zoomed_fig, info_text, info_style

    print(f"Starting Dash server on port {args.port}")
    print(f"Loaded {num_entries} entries from {args.h5file}")
    print(f"Starting at entry {start_entry}")
    print(f"Open http://localhost:{args.port} in your browser")

    app.run(debug=False, port=args.port, host='0.0.0.0')

    h5file.close()


if __name__ == '__main__':
    main()
