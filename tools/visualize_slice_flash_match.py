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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from detectoroutline import DetectorOutline  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from lartpc_data_prep.slice_labels import compute_slice_labels  # noqa: E402


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
        # Alias used by predict_slice_flash; same data, separate key so a
        # future change (e.g. switching to edep) is localised.
        out['pixval_for_predict'] = out['pixval']
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


def make_pmt_panel(d, slice_id, pe, title_lines, cmax_log=None):
    """Generic 2D y-z PMT panel. ``pe`` is the (32,) vector to color by;
    ``title_lines`` is a list of lines. ``cmax_log`` lets the caller force
    a shared log10 color-scale max across two panels (observed vs predicted).
    """
    pmts = d['pmt_positions']  # (32, 3)

    fig = go.Figure()
    tpc_y = (-116.5, 116.5)
    tpc_z = (0.0, 1036.8)
    fig.add_shape(type='rect', x0=tpc_z[0], x1=tpc_z[1],
                  y0=tpc_y[0], y1=tpc_y[1],
                  line=dict(color='rgb(180,180,180)', width=1),
                  fillcolor='rgba(0,0,0,0)')

    if pe.max() > 0:
        log_pe = np.log10(pe + 1.0)
        local_max = float(log_pe.max())
    else:
        log_pe = pe
        local_max = 1.0
    cmax = float(cmax_log) if cmax_log is not None else max(local_max, 0.1)

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
            cmin=0.0, cmax=cmax,
            line=dict(color='rgba(80,80,80,1)', width=1),
            colorbar=dict(title='log10(PE+1)', tickfont=dict(color='white'),
                          title_font=dict(color='white')),
        ),
        text=[str(i) for i in range(pmts.shape[0])],
        textfont=dict(size=8, color='black'),
        hovertext=text_lbls, hoverinfo='text', showlegend=False,
    ))

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


def get_observed_pe_and_title(d, slice_id):
    """Returns (pe (32,), title_lines, has_match)."""
    sfm_row = int(np.where(d['sfm_slice_id'] == int(slice_id))[0][0])
    matched_idx = int(d['sfm_matched_flash_idx'][sfm_row])
    if matched_idx < 0:
        return (np.zeros(d['pmt_positions'].shape[0], dtype=np.float32),
                [f"Observed PMTs, slice {slice_id}", "no matched flash (null)"],
                False)
    prod = int(d['flash_producer_id'][matched_idx])
    prod_name = 'simpleFlashBeam' if prod == 0 else 'simpleFlashCosmic'
    pe = d['flash_pe'][matched_idx]
    title_lines = [
        f"Observed: flash[{matched_idx}] {prod_name}  slice {slice_id}",
        f"t={float(d['flash_time_us'][matched_idx]):.2f} us  "
        f"tick={float(d['flash_tpc_tick'][matched_idx]):.1f}  "
        f"ΣPE={float(d['flash_total_pe'][matched_idx]):.1f}  "
        f"Δtick={float(d['sfm_match_dtick'][sfm_row]):.2f}",
    ]
    return pe, title_lines, True


PRODUCER_NAMES = ('simpleFlashBeam', 'simpleFlashCosmic')  # index = producer_id


def predict_slice_flash(d, slice_id, pl, gamma_by_producer, v_drift):
    """Run PhotonLibLookup for the selected slice with drift correction to
    the matched flash's t0 (when present), then scale by the producer-
    specific γ.

    Args:
        gamma_by_producer: tuple/list (γ_beam, γ_cosmic). The matched flash's
            producer_id selects which scalar is applied. For unmatched slices
            (no flash within tolerance) we fall back to γ_beam (arbitrary —
            the panel just shows what the predictor would produce with no
            t0 correction).
    Returns:
        (pe (32,) float32, t0_us, producer_id_used, gamma_used)
    """
    import torch
    from pointcept.models.event_slicer.photonlib import (
        select_charge_y_with_uv_fallback,
    )
    sfm_row = int(np.where(d['sfm_slice_id'] == int(slice_id))[0][0])
    matched_idx = int(d['sfm_matched_flash_idx'][sfm_row])
    if matched_idx >= 0:
        flash_t0_us = float(d['flash_time_us'][matched_idx])
        producer_id = int(d['flash_producer_id'][matched_idx])
    else:
        flash_t0_us = 0.0
        producer_id = 0   # fall back to beam γ; no time shift
    gamma = float(gamma_by_producer[producer_id])

    mask = d['slice_id_per_pt'] == int(slice_id)
    if not mask.any():
        return (np.zeros(d['pmt_positions'].shape[0], dtype=np.float32),
                flash_t0_us, producer_id, gamma)
    device = pl.vis_table.device
    pos = torch.from_numpy(d['pos'][mask]).to(device)
    q = select_charge_y_with_uv_fallback(
        torch.from_numpy(d['pixval_for_predict'][mask])
    ).to(device)
    pos[:, 0] = pos[:, 0] - v_drift * flash_t0_us
    cid = torch.zeros(int(mask.sum()), dtype=torch.int64, device=device)
    pe = pl.predict_flash(pos, q, cid, n_clusters=1)[0]
    pe = (pe * gamma).cpu().numpy().astype(np.float32)
    return pe, flash_t0_us, producer_id, gamma


def cosine_sim(a, b):
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    return float((a * b).sum() / (na * nb + 1e-12))


def make_observed_pmt_figure(d, slice_id, shared_cmax_log=None):
    pe, title_lines, _ = get_observed_pe_and_title(d, slice_id)
    return make_pmt_panel(d, slice_id, pe, title_lines,
                          cmax_log=shared_cmax_log)


def make_predicted_pmt_figure(d, slice_id, pl, gamma_by_producer, v_drift,
                              shared_cmax_log=None):
    pe, flash_t0_us, producer_id, gamma_used = predict_slice_flash(
        d, slice_id, pl, gamma_by_producer, v_drift,
    )
    obs, _, has_match = get_observed_pe_and_title(d, slice_id)
    cos = cosine_sim(pe, obs) if has_match else float('nan')
    prod_name = (PRODUCER_NAMES[producer_id]
                 if 0 <= producer_id < len(PRODUCER_NAMES) else f'prod{producer_id}')
    title_lines = [
        f"Predicted: photonlib + drift (t0={flash_t0_us:.2f} us)",
        f"γ_{prod_name.replace('simpleFlash', '').lower()}={gamma_used:.3g}  "
        f"ΣPE_pred={float(pe.sum()):.1f}  cosine(obs)={cos:.3f}",
    ]
    return make_pmt_panel(d, slice_id, pe, title_lines,
                          cmax_log=shared_cmax_log)


def shared_log_cmax(pe_a, pe_b):
    """Color-scale max for two PE vectors so both panels share an axis."""
    m = max(float(pe_a.max()) if pe_a.size else 0.0,
            float(pe_b.max()) if pe_b.size else 0.0)
    return max(np.log10(m + 1.0), 0.1)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--merged-h5', required=True)
    ap.add_argument('--flashinfo-h5', required=True)
    ap.add_argument('--port', type=int, default=8051)
    ap.add_argument('--photonlib-cache', default=None,
                    help="Path to photonlib_v6_70kV.npz. When given, enables "
                         "a 'predicted flash' panel next to the observed one.")
    ap.add_argument('--gamma', type=float, default=1.0,
                    help="Default photons-per-charge scalar; used for any "
                         "producer whose specific γ flag is not set. "
                         "Empirically ~1.6 on the canonical example's nu "
                         "slice (beam readout).")
    ap.add_argument('--gamma-beam', type=float, default=None,
                    help="γ applied when the matched flash is simpleFlashBeam "
                         "(producer_id=0). Defaults to --gamma if unset.")
    ap.add_argument('--gamma-cosmic', type=float, default=None,
                    help="γ applied when the matched flash is "
                         "simpleFlashCosmic (producer_id=1). Defaults to "
                         "--gamma if unset. Typically much smaller than "
                         "γ_beam because the cosmic stream's triggered, "
                         "fixed-window readout truncates the long-component "
                         "scintillation tail.")
    ap.add_argument('--v-drift-cm-per-us', type=float, default=0.1098,
                    help="MicroBooNE drift velocity used for the predicted "
                         "side's t0 correction.")
    ap.add_argument('--device', default=None,
                    help="torch device for the predictor (default: cuda if "
                         "available else cpu). Ignored if --photonlib-cache "
                         "is not given.")
    args = ap.parse_args()

    d = load_data(args.merged_h5, args.flashinfo_h5)

    gamma_by_producer = (
        args.gamma_beam if args.gamma_beam is not None else args.gamma,
        args.gamma_cosmic if args.gamma_cosmic is not None else args.gamma,
    )

    pl = None
    if args.photonlib_cache:
        import torch
        from pointcept.models.event_slicer.photonlib import PhotonLibLookup
        device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading PhotonLibLookup from {args.photonlib_cache} (device={device})")
        pl = PhotonLibLookup(args.photonlib_cache,
                             fp16=False, use_trilinear=True).to(device)
        print(f"  vis_table {tuple(pl.vis_table.shape)} on {pl.vis_table.device}")
        print(f"  γ_beam={gamma_by_producer[0]:.4g}  "
              f"γ_cosmic={gamma_by_producer[1]:.4g}")
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

    panel_row = [
        html.Div(dcc.Graph(id='fig-3d'),
                 style={'width': '50%' if pl is not None else '60%',
                        'display': 'inline-block', 'verticalAlign': 'top'}),
        html.Div(dcc.Graph(id='fig-pmt-obs'),
                 style={'width': '25%' if pl is not None else '40%',
                        'display': 'inline-block', 'verticalAlign': 'top'}),
    ]
    if pl is not None:
        panel_row.append(
            html.Div(dcc.Graph(id='fig-pmt-pred'),
                     style={'width': '25%', 'display': 'inline-block',
                            'verticalAlign': 'top'})
        )

    app.layout = html.Div([
        html.H2("Slice / Flash Match Viewer"
                + ("  (+ photonlib prediction)" if pl is not None else ""),
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
        html.Div(panel_row),
    ], style={'backgroundColor': '#0a0a0a', 'padding': '14px',
              'minHeight': '100vh'})

    if pl is None:
        @callback(
            Output('fig-3d', 'figure'),
            Output('fig-pmt-obs', 'figure'),
            Input('slice-dd', 'value'),
            Input('show-ghosts', 'value'),
        )
        def update(slice_id, show_ghosts):
            return (
                make_3d_figure(d, int(slice_id),
                               show_ghosts='show' in (show_ghosts or [])),
                make_observed_pmt_figure(d, int(slice_id)),
            )
    else:
        @callback(
            Output('fig-3d', 'figure'),
            Output('fig-pmt-obs', 'figure'),
            Output('fig-pmt-pred', 'figure'),
            Input('slice-dd', 'value'),
            Input('show-ghosts', 'value'),
        )
        def update(slice_id, show_ghosts):
            sid = int(slice_id)
            obs_pe, _, _ = get_observed_pe_and_title(d, sid)
            pred_pe, _, _, _ = predict_slice_flash(
                d, sid, pl, gamma_by_producer, args.v_drift_cm_per_us,
            )
            shared = shared_log_cmax(obs_pe, pred_pe)
            return (
                make_3d_figure(d, sid,
                               show_ghosts='show' in (show_ghosts or [])),
                make_observed_pmt_figure(d, sid, shared_cmax_log=shared),
                make_predicted_pmt_figure(d, sid, pl, gamma_by_producer,
                                          args.v_drift_cm_per_us,
                                          shared_cmax_log=shared),
            )

    print(f"Open http://localhost:{args.port} in your browser")
    app.run(debug=False, port=args.port, host='0.0.0.0')


if __name__ == '__main__':
    main()
