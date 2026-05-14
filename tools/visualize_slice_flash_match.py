"""Slice + matched-flash visualizer.

Given a paired (merged H5, flashinfo H5) for the same entry, lets you pick a
slice from a dropdown and shows:
  - Left: 3D scatter of the spacepoints. The selected slice is colored
    (red for nu, hue-spread for cosmic); all other real spacepoints (and
    ghosts) are gray for context.
  - Right: 2D y-z map of PMTs, each as a circle at its (z, y) position,
    color-scaled by the PE of the matched flash. The matched flash's
    metadata (producer, time, total PE, Δtick) is in the title.

PMT positions and flash PE in the flashinfo H5 are both indexed by **OpDet**
— the channel<->opdet remap is done at prep time so this viewer can plot
pmt_positions[i] with pe[i] directly.

Usage:
    python visualize_slice_flash_match.py \\
        --merged-h5    /path/to/merged_<basename>_entry<NNNN>.h5 \\
        --flashinfo-h5 /path/to/flashinfo_<basename>_entry<NNNN>.h5 \\
        [--port 8051]
"""

import argparse
import colorsys
import os
import sys

import h5py
import numpy as np
from dash import Dash, Input, Output, callback, dcc, html
import plotly.graph_objects as go
from plotly.subplots import make_subplots

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from detectoroutline import DetectorOutline  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from lartpc_data_prep.slice_labels import compute_slice_labels, GHOST_SLICE_ID  # noqa: E402


PDG_NAMES = {
    0: '(nu group)', 11: 'e-', -11: 'e+', 13: 'mu-', -13: 'mu+',
    22: 'gamma', 111: 'pi0', 211: 'pi+', -211: 'pi-',
    2212: 'proton', 2112: 'neutron', 321: 'K+', -321: 'K-',
}


def pdg_name(pdg):
    return PDG_NAMES.get(int(pdg), f'PDG:{int(pdg)}')


def slice_color(origin, idx, n_cosmic):
    """Return rgba string. nu (origin=1) = red. cosmic spread across HSV."""
    if int(origin) == 1:
        return 'rgba(255,60,60,1)'
    if n_cosmic <= 0:
        return 'rgba(120,200,255,1)'
    h = (idx + 0.1) / n_cosmic
    r, g, b = colorsys.hsv_to_rgb(h, 0.85, 0.95)
    return f'rgba({int(r*255)},{int(g*255)},{int(b*255)},1)'


def load_data(merged_h5, flashinfo_h5):
    """Read both H5 files and assemble a flat dict of arrays we'll need."""
    out = {}
    with h5py.File(merged_h5, 'r') as mh:
        e = mh['entry_0']
        out['run'] = int(e.attrs.get('run', -1))
        out['subrun'] = int(e.attrs.get('subrun', -1))
        out['event'] = int(e.attrs.get('event', -1))
        td = e['triplet_data']
        out['pos'] = td['pos'][:].astype(np.float32)
        out['hasmatch'] = td['hasmatch'][:].astype(np.int32)
        out['trackid'] = td['trackid'][:].astype(np.int64)
        out['pixval'] = td['pixval'][:].astype(np.float32)
        out['origin_per_pt'] = td['origin'][:].astype(np.int32)
        sinfo = compute_slice_labels(
            e['mc_particle_tree'], out['trackid'], out['hasmatch'],
        )
        out['slice_id_per_pt'] = sinfo['slice_id']
        out['primary_trackid'] = sinfo['primary_trackid']
        out['primary_origin'] = sinfo['primary_origin']
        out['primary_pid'] = sinfo['primary_pid']
        out['primary_start_pos'] = sinfo['primary_start_pos']
        out['nu_vertices'] = sinfo['nu_vertices']

    with h5py.File(flashinfo_h5, 'r') as fh:
        e = fh['entry_0']
        out['fi_attrs'] = dict(e.attrs)
        out['pmt_positions'] = e['pmt_positions'][:].astype(np.float32)
        fl = e['flashes']
        out['flash_pe'] = fl['pe'][:].astype(np.float32)
        out['flash_total_pe'] = fl['total_pe'][:].astype(np.float32)
        out['flash_time_us'] = fl['time_us'][:].astype(np.float32)
        out['flash_tpc_tick'] = fl['tpc_tick'][:].astype(np.float32)
        out['flash_producer_id'] = fl['producer_id'][:].astype(np.int32)
        out['flash_matched_slice_id'] = fl['matched_slice_id'][:].astype(np.int64)
        sl = e['slice_flash_matches']
        out['sfm_slice_id'] = sl['slice_id'][:].astype(np.int64)
        out['sfm_matched_flash_idx'] = sl['matched_flash_idx'][:].astype(np.int32)
        out['sfm_match_dtick'] = sl['match_dtick'][:].astype(np.float32)
        out['sfm_is_null'] = sl['is_null_flash'][:].astype(np.int8)
        out['sfm_crosses_boundary'] = sl['crosses_image_boundary'][:].astype(np.int8)
        out['sfm_primary_tick'] = sl['primary_tpc_tick'][:].astype(np.float32)
        out['sfm_total_pe_matched'] = sl['total_pe_matched'][:].astype(np.float32)
    return out


def build_slice_options(d):
    """Build dropdown options labelled with the per-slice metadata."""
    n_cos = int((d['primary_origin'] == 2).sum())
    cos_seen = 0
    options = []
    for k, key in enumerate(d['primary_trackid']):
        # find matching row in sfm_*
        sfm_row = int(np.where(d['sfm_slice_id'] == int(key))[0][0])
        origin = int(d['primary_origin'][k])
        pid = int(d['primary_pid'][k])
        n_pts = int((d['slice_id_per_pt'] == int(key)).sum())
        matched_idx = int(d['sfm_matched_flash_idx'][sfm_row])
        total_pe = float(d['sfm_total_pe_matched'][sfm_row])
        crosses = int(d['sfm_crosses_boundary'][sfm_row])
        dtick = float(d['sfm_match_dtick'][sfm_row])

        if origin == 1:
            origin_lbl = 'nu'
            color_idx = -1
        else:
            origin_lbl = 'cos'
            color_idx = cos_seen
            cos_seen += 1

        match_str = (f"flash{matched_idx} dt={dtick:.2f}t {total_pe:.0f}PE"
                     if matched_idx >= 0 else "no match")
        boundary_str = " [boundary]" if crosses else ""
        pid_str = pdg_name(pid)

        label = (f"slice {int(key)} [{origin_lbl}/{pid_str}] {n_pts}pts | "
                 f"{match_str}{boundary_str}")
        options.append({
            'label': label,
            'value': int(key),
        })
        # stash color via separate dict in d so the figure builder can recover
        d.setdefault('_slice_color', {})[int(key)] = slice_color(origin, color_idx, n_cos)
    return options


def make_3d_figure(d, slice_id, marker_size=2, show_ghosts=False):
    pos = d['pos']
    hm = d['hasmatch']
    sid_per_pt = d['slice_id_per_pt']

    fig = go.Figure()

    # context: all OTHER real spacepoints in gray
    other_mask = (hm == 1) & (sid_per_pt != slice_id)
    if other_mask.any():
        fig.add_trace(go.Scatter3d(
            x=pos[other_mask, 0], y=pos[other_mask, 1], z=pos[other_mask, 2],
            mode='markers',
            marker=dict(size=max(1, marker_size - 1),
                        color='rgba(120,120,120,0.35)'),
            name=f'other slices ({int(other_mask.sum())})',
            hoverinfo='skip',
        ))

    # optional: ghosts as faint dots
    if show_ghosts:
        ghost_mask = (hm == 0)
        if ghost_mask.any():
            fig.add_trace(go.Scatter3d(
                x=pos[ghost_mask, 0], y=pos[ghost_mask, 1], z=pos[ghost_mask, 2],
                mode='markers',
                marker=dict(size=1, color='rgba(60,60,60,0.18)'),
                name=f'ghosts ({int(ghost_mask.sum())})',
                hoverinfo='skip',
            ))

    # selected slice
    sel_mask = sid_per_pt == slice_id
    color = d.get('_slice_color', {}).get(int(slice_id), 'rgba(255,60,60,1)')
    if sel_mask.any():
        fig.add_trace(go.Scatter3d(
            x=pos[sel_mask, 0], y=pos[sel_mask, 1], z=pos[sel_mask, 2],
            mode='markers',
            marker=dict(size=marker_size + 1, color=color, opacity=0.95),
            name=f'slice {slice_id} ({int(sel_mask.sum())})',
            hoverinfo='skip',
        ))

    # nu_vertex marker if nu slice
    sfm_row = int(np.where(d['sfm_slice_id'] == int(slice_id))[0][0])
    if int(d['sfm_slice_id'][sfm_row]) == slice_id:
        origin = None
        for k, key in enumerate(d['primary_trackid']):
            if int(key) == int(slice_id):
                origin = int(d['primary_origin'][k])
                break
        if origin == 1 and len(d['nu_vertices']) > 0:
            nuv = d['nu_vertices']
            fig.add_trace(go.Scatter3d(
                x=nuv[:, 0], y=nuv[:, 1], z=nuv[:, 2],
                mode='markers',
                marker=dict(size=12, color='rgba(255,215,0,1)',
                            symbol='diamond', line=dict(width=2, color='black')),
                name='nu vertex', hoverinfo='text',
                hovertext=[f"nu vertex ({p[0]:.1f},{p[1]:.1f},{p[2]:.1f})"
                           for p in nuv],
            ))

    # detector outline
    det = DetectorOutline()
    for tr in det.getlines(color=(255, 255, 255)):
        fig.add_trace(go.Scatter3d(
            x=tr['x'], y=tr['y'], z=tr['z'], mode='lines',
            line=dict(color=tr['line']['color'], width=tr['line']['width']),
            showlegend=False, hoverinfo='skip',
        ))

    axis_tmpl = dict(showbackground=True, backgroundcolor="#141414",
                     gridcolor="rgb(70,70,70)", zerolinecolor="rgb(120,120,120)",
                     title_font=dict(color="white"), tickfont=dict(color="white"))
    fig.update_layout(
        scene=dict(
            xaxis=dict(**axis_tmpl, title='X (cm)'),
            yaxis=dict(**axis_tmpl, title='Y (cm)'),
            zaxis=dict(**axis_tmpl, title='Z (cm)'),
            aspectratio=dict(x=1, y=1, z=4),
            camera=dict(eye=dict(x=2, y=2, z=2), up=dict(x=0, y=1, z=0)),
        ),
        title=dict(text=f"Slice {slice_id}", font=dict(color="white")),
        paper_bgcolor="#141414", plot_bgcolor="#141414",
        height=720, margin=dict(l=0, r=0, t=40, b=0),
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01,
                    font=dict(color="white"), bgcolor="rgba(20,20,20,0.8)"),
    )
    return fig


def make_pmt_figure(d, slice_id):
    """2D y-z plot of PMTs colored by PE of the matched flash. Returns a fig."""
    pmts = d['pmt_positions']  # (32, 3)
    sfm_row = int(np.where(d['sfm_slice_id'] == int(slice_id))[0][0])
    matched_idx = int(d['sfm_matched_flash_idx'][sfm_row])

    title_lines = [f"PMT view (32 OpDets), slice {slice_id}"]
    if matched_idx < 0:
        pe = np.zeros(pmts.shape[0], dtype=np.float32)
        title_lines.append("no matched flash (null)")
    else:
        pe = d['flash_pe'][matched_idx]
        prod = int(d['flash_producer_id'][matched_idx])
        prod_name = 'simpleFlashBeam' if prod == 0 else 'simpleFlashCosmic'
        title_lines.append(
            f"flash[{matched_idx}] {prod_name}  t={float(d['flash_time_us'][matched_idx]):.2f} us  "
            f"tick={float(d['flash_tpc_tick'][matched_idx]):.1f}  "
            f"ΣPE={float(d['flash_total_pe'][matched_idx]):.1f}  "
            f"Δtick={float(d['sfm_match_dtick'][sfm_row]):.2f}"
        )

    fig = go.Figure()

    # Detector y-z outline (TPC volume y ∈ [-116.5, 116.5], z ∈ [0, 1036.8])
    tpc_y = (-116.5, 116.5)
    tpc_z = (0.0, 1036.8)
    fig.add_shape(type='rect', x0=tpc_z[0], x1=tpc_z[1],
                  y0=tpc_y[0], y1=tpc_y[1],
                  line=dict(color='rgb(180,180,180)', width=1),
                  fillcolor='rgba(0,0,0,0)')

    # PMTs
    if pe.max() > 0:
        log_pe = np.log10(pe + 1.0)
        cmax = float(log_pe.max())
    else:
        log_pe = pe
        cmax = 1.0

    text_lbls = [
        f"OpDet {i}<br>pos=(x={pmts[i,0]:.1f}, y={pmts[i,1]:.1f}, z={pmts[i,2]:.1f})<br>"
        f"PE={pe[i]:.2f}"
        for i in range(pmts.shape[0])
    ]
    fig.add_trace(go.Scatter(
        x=pmts[:, 2], y=pmts[:, 1],
        mode='markers+text',
        marker=dict(
            size=22, color=log_pe, colorscale='Hot',
            cmin=0.0, cmax=max(cmax, 0.1),
            line=dict(color='rgba(80,80,80,1)', width=1),
            colorbar=dict(title='log10(PE+1)', tickfont=dict(color='white'),
                          title_font=dict(color='white')),
        ),
        text=[str(i) for i in range(pmts.shape[0])],
        textfont=dict(size=8, color='black'),
        hovertext=text_lbls, hoverinfo='text', showlegend=False,
    ))

    # If this is a nu slice, overlay the nu vertex y-z position
    sfm_row = int(np.where(d['sfm_slice_id'] == int(slice_id))[0][0])
    origin = None
    for k, key in enumerate(d['primary_trackid']):
        if int(key) == int(slice_id):
            origin = int(d['primary_origin'][k])
            break
    if origin == 1 and len(d['nu_vertices']) > 0:
        nuv = d['nu_vertices'][0]
        fig.add_trace(go.Scatter(
            x=[nuv[2]], y=[nuv[1]], mode='markers',
            marker=dict(size=14, color='rgba(255,215,0,1)',
                        symbol='diamond', line=dict(width=2, color='black')),
            name='nu vertex (y, z proj.)', hoverinfo='text',
            hovertext=[f"nu vertex y={nuv[1]:.1f}, z={nuv[2]:.1f}, x={nuv[0]:.1f}"],
        ))

    fig.update_layout(
        title=dict(text='<br>'.join(title_lines), font=dict(color='white', size=12)),
        xaxis=dict(title='Z (cm)', range=[-50, 1086], color='white',
                   gridcolor='rgb(60,60,60)', zerolinecolor='rgb(120,120,120)'),
        yaxis=dict(title='Y (cm)', range=[-140, 140], scaleanchor='x',
                   scaleratio=1, color='white', gridcolor='rgb(60,60,60)',
                   zerolinecolor='rgb(120,120,120)'),
        plot_bgcolor='#141414', paper_bgcolor='#141414',
        height=720, margin=dict(l=40, r=20, t=70, b=40),
    )
    return fig


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--merged-h5', required=True)
    ap.add_argument('--flashinfo-h5', required=True)
    ap.add_argument('--port', type=int, default=8051)
    args = ap.parse_args()

    d = load_data(args.merged_h5, args.flashinfo_h5)
    slice_options = build_slice_options(d)
    if not slice_options:
        print('No slices found.')
        return
    initial_slice = int(slice_options[0]['value'])

    header_lines = [
        f"merged H5: {os.path.basename(args.merged_h5)}",
        f"flash H5:  {os.path.basename(args.flashinfo_h5)}",
        f"run={d['run']}  subrun={d['subrun']}  event={d['event']}",
        f"n_spacepoints={len(d['pos'])}  n_slices={len(d['primary_trackid'])}  "
        f"n_flashes={len(d['flash_pe'])}",
    ]

    app = Dash(__name__)
    app.layout = html.Div([
        html.H2("Slice / Flash Match Viewer",
                style={'color': 'white', 'textAlign': 'center'}),
        html.Div([html.Div(line, style={'color': '#bbb'}) for line in header_lines],
                 style={'textAlign': 'center', 'marginBottom': '12px'}),
        html.Div([
            html.Label('Slice:', style={'color': 'white', 'marginRight': '8px'}),
            dcc.Dropdown(
                id='slice-dd', options=slice_options, value=initial_slice,
                clearable=False,
                style={'width': '900px', 'display': 'inline-block',
                       'color': 'black'},
            ),
            dcc.Checklist(
                id='show-ghosts',
                options=[{'label': ' show ghosts in 3D', 'value': 'show'}],
                value=[],
                style={'display': 'inline-block', 'color': 'white',
                       'marginLeft': '20px'},
            ),
        ], style={'textAlign': 'center', 'marginBottom': '10px'}),
        html.Div([
            html.Div(dcc.Graph(id='fig-3d'),
                     style={'width': '60%', 'display': 'inline-block',
                            'verticalAlign': 'top'}),
            html.Div(dcc.Graph(id='fig-pmt'),
                     style={'width': '40%', 'display': 'inline-block',
                            'verticalAlign': 'top'}),
        ]),
    ], style={'backgroundColor': '#0a0a0a', 'padding': '14px',
              'minHeight': '100vh'})

    @callback(
        Output('fig-3d', 'figure'),
        Output('fig-pmt', 'figure'),
        Input('slice-dd', 'value'),
        Input('show-ghosts', 'value'),
    )
    def update(slice_id, show_ghosts):
        return (
            make_3d_figure(d, int(slice_id),
                           show_ghosts='show' in (show_ghosts or [])),
            make_pmt_figure(d, int(slice_id)),
        )

    print(f"Open http://localhost:{args.port} in your browser")
    app.run(debug=False, port=args.port, host='0.0.0.0')


if __name__ == '__main__':
    main()
