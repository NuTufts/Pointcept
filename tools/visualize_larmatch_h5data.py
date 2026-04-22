#!/usr/bin/env python3
"""
Visualization tool for LArTPC HDF5 data files with LArMatch ghost scores.

Displays:
- 3D point cloud with color options: larmatch score, pixval, min ADC,
  and larmatch-thresholded views
- Interactive threshold slider to preview ghost removal
- Score distribution histogram
- Per-point hover info with coordinates, wire indices, ADC, and score

Designed for HDF5 files produced by the extbnb_larmatch pipeline
(which include a `larmatch_score` field), but also works on standard
extbnb files (score defaults to 1.0) and simulation files with hasmatch.

Usage:
    python visualize_larmatch_h5data.py <hdf5_file> [--port PORT]

    # Can also point at a directory to load all .h5 files:
    python visualize_larmatch_h5data.py /path/to/dir/ --port 8050

Example:
    python visualize_larmatch_h5data.py pointceptdata_merged_dlreco_00deb37b_entry000005.h5
"""

import argparse
import os
import glob
import h5py
import numpy as np
from dash import Dash, html, dcc, callback, Output, Input, State
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from detectoroutline import DetectorOutline


def load_event_data(h5file, entry_key):
    """Load data for a single entry from the HDF5 file."""
    entry_grp = h5file[entry_key]
    triplet_data = entry_grp['triplet_data']

    data = {}
    data['pos'] = triplet_data['pos'][:]
    data['pixval'] = triplet_data['pixval'][:]

    n_points = len(data['pos'])

    # Wire coordinates
    if 'uwire' in triplet_data:
        data['uwire'] = triplet_data['uwire'][:]
        data['vwire'] = triplet_data['vwire'][:]
        data['ywire'] = triplet_data['ywire'][:]
    else:
        data['uwire'] = np.zeros(n_points, dtype=np.float32)
        data['vwire'] = np.zeros(n_points, dtype=np.float32)
        data['ywire'] = np.zeros(n_points, dtype=np.float32)

    if 'tick' in triplet_data:
        data['tick'] = triplet_data['tick'][:]
    else:
        data['tick'] = np.zeros(n_points, dtype=np.int32)

    # LArMatch score
    if 'larmatch_score' in triplet_data:
        data['larmatch_score'] = triplet_data['larmatch_score'][:]
        data['has_larmatch'] = True
    else:
        data['larmatch_score'] = np.ones(n_points, dtype=np.float32)
        data['has_larmatch'] = False

    # Truth labels (simulation only)
    if 'hasmatch' in triplet_data:
        data['hasmatch'] = triplet_data['hasmatch'][:]
        data['has_truth'] = True
    else:
        data['hasmatch'] = np.ones(n_points, dtype=np.int32)
        data['has_truth'] = False

    if 'origin' in triplet_data:
        data['origin'] = triplet_data['origin'][:]
    else:
        data['origin'] = np.zeros(n_points, dtype=np.int32)

    if 'ssnet_label' in triplet_data:
        data['ssnet_label'] = triplet_data['ssnet_label'][:]
    else:
        data['ssnet_label'] = np.zeros(n_points, dtype=np.int32)

    return data


def make_hover_text(pos, data, indices):
    """Build hover text for a set of point indices."""
    texts = []
    for i in indices:
        text = (
            f"x: {pos[i, 0]:.1f}, y: {pos[i, 1]:.1f}, z: {pos[i, 2]:.1f}<br>"
            f"Wires: ({data['uwire'][i]:.0f}, {data['vwire'][i]:.0f}, {data['ywire'][i]:.0f})<br>"
            f"Tick: {data['tick'][i]:.0f}<br>"
            f"PixVal: ({data['pixval'][i, 0]:.1f}, {data['pixval'][i, 1]:.1f}, {data['pixval'][i, 2]:.1f})<br>"
            f"Min ADC: {data['pixval'][i].min():.1f}<br>"
            f"LM Score: {data['larmatch_score'][i]:.3f}"
        )
        if data['has_truth']:
            match_str = "True" if data['hasmatch'][i] == 1 else "Ghost"
            text += f"<br>Truth: {match_str}"
        texts.append(text)
    return texts


def create_3d_scatter(event_data, color_mode='larmatch', lm_threshold=0.0,
                      opacity=0.8, marker_size=2):
    """Create a 3D scatter plot with various coloring options."""
    pos = event_data['pos']
    lm_score = event_data['larmatch_score']

    # Apply larmatch threshold filter
    if lm_threshold > 0:
        keep = lm_score >= lm_threshold
        n_before = len(pos)
        n_after = keep.sum()
        threshold_info = f" (LM>={lm_threshold:.2f}: {n_after}/{n_before} kept, {n_before - n_after} removed)"
    else:
        keep = np.ones(len(pos), dtype=bool)
        threshold_info = ""

    # Apply filter to all arrays
    filt = {k: v[keep] if isinstance(v, np.ndarray) and len(v) == len(pos) else v
            for k, v in event_data.items()}
    filt_pos = filt['pos']

    if len(filt_pos) == 0:
        fig = go.Figure()
        fig.update_layout(title="No points after threshold filter")
        return fig

    fig = go.Figure()

    if color_mode == 'larmatch':
        indices = np.arange(len(filt_pos))
        fig.add_trace(go.Scatter3d(
            x=filt_pos[:, 0], y=filt_pos[:, 1], z=filt_pos[:, 2],
            mode='markers',
            marker=dict(
                size=marker_size,
                color=filt['larmatch_score'],
                colorscale='RdYlGn',
                cmin=0, cmax=1,
                opacity=opacity,
                colorbar=dict(title='LM Score')
            ),
            name='LArMatch Score',
            hovertext=make_hover_text(filt_pos, filt, indices),
            hoverinfo='text'
        ))

    elif color_mode == 'larmatch_binary':
        # Binary view: high vs low score
        mid = max(lm_threshold, 0.5)
        for label, mask_fn, color in [
            ('High Score', lambda s: s >= mid, 'green'),
            ('Low Score', lambda s: s < mid, 'red'),
        ]:
            mask = mask_fn(filt['larmatch_score'])
            indices = np.where(mask)[0]
            if len(indices) == 0:
                continue
            fig.add_trace(go.Scatter3d(
                x=filt_pos[mask, 0], y=filt_pos[mask, 1], z=filt_pos[mask, 2],
                mode='markers',
                marker=dict(size=marker_size, color=color, opacity=opacity),
                name=f"{label} ({len(indices)})",
                hovertext=make_hover_text(filt_pos, filt, indices),
                hoverinfo='text'
            ))

    elif color_mode == 'hasmatch' and event_data['has_truth']:
        for match_val, name, color in [(1, 'True', 'green'), (0, 'Ghost', 'red')]:
            mask = filt['hasmatch'] == match_val
            indices = np.where(mask)[0]
            if len(indices) == 0:
                continue
            fig.add_trace(go.Scatter3d(
                x=filt_pos[mask, 0], y=filt_pos[mask, 1], z=filt_pos[mask, 2],
                mode='markers',
                marker=dict(size=marker_size, color=color, opacity=opacity),
                name=f"{name} ({len(indices)})",
                hovertext=make_hover_text(filt_pos, filt, indices),
                hoverinfo='text'
            ))

    elif color_mode == 'pixval':
        pixval_sum = np.sum(filt['pixval'], axis=1)
        indices = np.arange(len(filt_pos))
        fig.add_trace(go.Scatter3d(
            x=filt_pos[:, 0], y=filt_pos[:, 1], z=filt_pos[:, 2],
            mode='markers',
            marker=dict(
                size=marker_size,
                color=pixval_sum,
                colorscale='Viridis',
                opacity=opacity,
                colorbar=dict(title='Sum PixVal'),
                cmin=0,
                cmax=np.percentile(pixval_sum, 95)
            ),
            name='Pixel Values',
            hovertext=make_hover_text(filt_pos, filt, indices),
            hoverinfo='text'
        ))

    elif color_mode == 'min_adc':
        min_adc = filt['pixval'].min(axis=1)
        indices = np.arange(len(filt_pos))
        fig.add_trace(go.Scatter3d(
            x=filt_pos[:, 0], y=filt_pos[:, 1], z=filt_pos[:, 2],
            mode='markers',
            marker=dict(
                size=marker_size,
                color=min_adc,
                colorscale='Viridis',
                opacity=opacity,
                colorbar=dict(title='Min ADC'),
                cmin=0,
                cmax=np.percentile(min_adc, 95) if len(min_adc) > 0 else 100
            ),
            name='Min ADC',
            hovertext=make_hover_text(filt_pos, filt, indices),
            hoverinfo='text'
        ))

    # Detector outline
    detdata = DetectorOutline()
    det_traces = detdata.getlines(color=(255, 255, 255))
    for det_trace in det_traces:
        fig.add_trace(go.Scatter3d(
            x=det_trace['x'], y=det_trace['y'], z=det_trace['z'],
            mode='lines',
            line=dict(color=det_trace['line']['color'], width=det_trace['line']['width']),
            showlegend=False
        ))

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
            camera=dict(eye=dict(x=2, y=2, z=2), up=dict(x=0, y=1, z=0)),
        ),
        title=dict(
            text=f"3D Point Cloud - {color_mode} ({len(filt_pos)} points){threshold_info}",
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


def create_score_histogram(event_data):
    """Create a histogram of larmatch scores, optionally split by truth label."""
    lm_score = event_data['larmatch_score']

    fig = go.Figure()

    if event_data['has_truth']:
        ghost = event_data['hasmatch'] == 0
        true = event_data['hasmatch'] == 1

        bins = dict(start=0, end=1.02, size=0.02)
        fig.add_trace(go.Histogram(
            x=lm_score[ghost], name=f"Ghost ({ghost.sum():,})",
            marker_color='red', opacity=0.6,
            xbins=bins,
        ))
        fig.add_trace(go.Histogram(
            x=lm_score[true], name=f"True ({true.sum():,})",
            marker_color='blue', opacity=0.6,
            xbins=bins,
        ))
        fig.update_layout(barmode='overlay')
    else:
        fig.add_trace(go.Histogram(
            x=lm_score, name=f"All ({len(lm_score):,})",
            marker_color='steelblue', opacity=0.8,
            xbins=dict(start=0, end=1.02, size=0.02),
        ))

    fig.update_layout(
        title='LArMatch Score Distribution',
        xaxis_title='LArMatch Score',
        yaxis_title='Count',
        yaxis_type='log',
        height=350,
        margin=dict(l=60, r=20, t=40, b=40),
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
    )

    return fig


def create_threshold_stats(event_data, thresholds=None):
    """Create a plot showing ghost rejection / true efficiency vs threshold."""
    if thresholds is None:
        thresholds = np.linspace(0, 1, 101)

    lm_score = event_data['larmatch_score']

    if event_data['has_truth']:
        ghost = event_data['hasmatch'] == 0
        true = event_data['hasmatch'] == 1
        n_ghost = ghost.sum()
        n_true = true.sum()

        ghost_rej = []
        true_eff = []
        for t in thresholds:
            keep = lm_score >= t
            ghost_rej.append(1.0 - (keep & ghost).sum() / max(n_ghost, 1))
            true_eff.append((keep & true).sum() / max(n_true, 1))

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=thresholds, y=ghost_rej,
            name='Ghost Rejection', line=dict(color='red', width=2)
        ))
        fig.add_trace(go.Scatter(
            x=thresholds, y=true_eff,
            name='True Efficiency', line=dict(color='blue', width=2)
        ))
        fig.update_layout(
            title='LArMatch Threshold Performance (this event)',
            xaxis_title='Score Threshold',
            yaxis_title='Fraction',
            height=350,
            margin=dict(l=60, r=20, t=40, b=40),
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
            yaxis=dict(range=[0, 1.05]),
        )
    else:
        # No truth - just show fraction of points surviving vs threshold
        frac_kept = []
        for t in thresholds:
            frac_kept.append((lm_score >= t).mean())

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=thresholds, y=frac_kept,
            name='Fraction Kept', line=dict(color='steelblue', width=2)
        ))
        fig.update_layout(
            title='Points Surviving vs Threshold (this event)',
            xaxis_title='Score Threshold',
            yaxis_title='Fraction of Points Kept',
            height=350,
            margin=dict(l=60, r=20, t=40, b=40),
            yaxis=dict(range=[0, 1.05]),
        )

    return fig


def main():
    parser = argparse.ArgumentParser(description='Visualize LArTPC HDF5 data with LArMatch scores')
    parser.add_argument('h5path', type=str,
                        help='Path to HDF5 file or directory containing .h5 files')
    parser.add_argument('--port', type=int, default=8050, help='Port for Dash server')
    args = parser.parse_args()

    # Build file list
    if os.path.isdir(args.h5path):
        h5_files = sorted(glob.glob(os.path.join(args.h5path, '*.h5')))
        if not h5_files:
            print(f"No .h5 files found in {args.h5path}")
            return
    else:
        h5_files = [args.h5path]

    print(f"Found {len(h5_files)} file(s)")

    # Build a flat list of (file_path, entry_key) tuples
    file_entries = []
    for fpath in h5_files:
        with h5py.File(fpath, 'r') as f:
            entry_keys = sorted(
                [k for k in f.keys() if k.startswith('entry_')],
                key=lambda x: int(x.split('_')[1])
            )
            for ek in entry_keys:
                file_entries.append((fpath, ek))

    num_entries = len(file_entries)
    if num_entries == 0:
        print("No entries found")
        return

    print(f"Total entries across all files: {num_entries}")

    # Detect capabilities from first entry
    fpath0, ek0 = file_entries[0]
    with h5py.File(fpath0, 'r') as f:
        td = f[ek0]['triplet_data']
        has_larmatch = 'larmatch_score' in td
        has_truth = 'hasmatch' in td

    # Color mode options
    color_options = [
        {'label': 'LArMatch Score', 'value': 'larmatch'},
        {'label': 'LM Binary (High/Low)', 'value': 'larmatch_binary'},
        {'label': 'Pixel Value (sum)', 'value': 'pixval'},
        {'label': 'Min ADC (3 planes)', 'value': 'min_adc'},
    ]
    if has_truth:
        color_options.append({'label': 'True/Ghost (MC truth)', 'value': 'hasmatch'})

    default_color = 'larmatch' if has_larmatch else 'pixval'

    # Create Dash app
    app = Dash(__name__, suppress_callback_exceptions=True)

    app.layout = html.Div([
        html.H1("LArTPC LArMatch Viewer", style={'textAlign': 'center'}),

        # File info
        html.Div(id='file-info', style={'textAlign': 'center', 'marginBottom': '10px'}),

        # Navigation
        html.Div([
            html.Button('Prev', id='prev-btn', n_clicks=0,
                        style={'marginRight': '10px', 'fontSize': '16px', 'padding': '8px 16px'}),
            html.Span(id='entry-info', style={'fontSize': '18px', 'marginRight': '10px'}),
            html.Button('Next', id='next-btn', n_clicks=0,
                        style={'marginLeft': '10px', 'fontSize': '16px', 'padding': '8px 16px'}),
            html.Span(" | Go to: ", style={'marginLeft': '20px'}),
            dcc.Input(id='entry-input', type='number', value=0, min=0, max=num_entries - 1,
                      style={'width': '80px', 'marginRight': '10px'}),
            html.Button('Go', id='go-btn', n_clicks=0,
                        style={'fontSize': '16px', 'padding': '5px 15px'}),
        ], style={'textAlign': 'center', 'marginBottom': '20px', 'padding': '10px'}),

        dcc.Store(id='current-entry', data=0),

        # Controls
        html.Div([
            html.Div([
                html.Label('Color Mode:', style={'fontWeight': 'bold', 'marginRight': '10px'}),
                dcc.Dropdown(
                    id='color-mode',
                    options=color_options,
                    value=default_color,
                    style={'width': '220px', 'display': 'inline-block'}
                ),
            ], style={'display': 'inline-block', 'marginRight': '30px'}),

            html.Div([
                html.Label('Marker Size:', style={'fontWeight': 'bold', 'marginRight': '10px'}),
                dcc.Slider(
                    id='marker-size',
                    min=1, max=10, step=0.5, value=2,
                    marks={1: '1', 5: '5', 10: '10'},
                    tooltip={'placement': 'bottom', 'always_visible': True}
                ),
            ], style={'display': 'inline-block', 'width': '200px', 'marginRight': '30px'}),

            html.Div([
                html.Label('LM Threshold:', style={'fontWeight': 'bold', 'marginRight': '10px'}),
                dcc.Slider(
                    id='lm-threshold',
                    min=0, max=1, step=0.01, value=0,
                    marks={0: '0', 0.25: '0.25', 0.5: '0.5', 0.75: '0.75', 1.0: '1.0'},
                    tooltip={'placement': 'bottom', 'always_visible': True}
                ),
            ], style={'display': 'inline-block', 'width': '300px'}),
        ], style={
            'textAlign': 'center', 'marginBottom': '20px', 'padding': '10px',
            'backgroundColor': '#f0f0f0'
        }),

        # Stats
        html.Div(id='stats-display', style={
            'textAlign': 'center', 'marginBottom': '10px', 'padding': '10px',
            'backgroundColor': '#e8e8e8', 'borderRadius': '5px'
        }),

        # 3D scatter
        html.Div([dcc.Graph(id='scatter-3d', style={'width': '100%'})],
                 style={'marginBottom': '20px'}),

        # Score distribution and threshold performance side by side
        html.Div([
            html.Div([dcc.Graph(id='score-histogram')],
                     style={'display': 'inline-block', 'width': '49%'}),
            html.Div([dcc.Graph(id='threshold-plot')],
                     style={'display': 'inline-block', 'width': '49%'}),
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
    def update_entry(prev_clicks, next_clicks, go_clicks, current, input_val):
        from dash import ctx
        tid = ctx.triggered_id
        if tid == 'prev-btn':
            new = max(0, current - 1)
        elif tid == 'next-btn':
            new = min(num_entries - 1, current + 1)
        elif tid == 'go-btn':
            new = max(0, min(num_entries - 1, input_val if input_val is not None else 0))
        else:
            new = current
        return new, new

    @callback(
        Output('file-info', 'children'),
        Output('entry-info', 'children'),
        Output('stats-display', 'children'),
        Output('scatter-3d', 'figure'),
        Output('score-histogram', 'figure'),
        Output('threshold-plot', 'figure'),
        Input('current-entry', 'data'),
        Input('color-mode', 'value'),
        Input('marker-size', 'value'),
        Input('lm-threshold', 'value'),
    )
    def update_display(entry_idx, color_mode, marker_size, lm_threshold):
        fpath, ek = file_entries[entry_idx]

        with h5py.File(fpath, 'r') as f:
            event_data = load_event_data(f, ek)

        fname = os.path.basename(fpath)
        file_info = f"File: {fname} | Entry: {ek}"
        entry_info = f"Event {entry_idx}/{num_entries - 1}"

        # Stats
        n = len(event_data['pos'])
        lm = event_data['larmatch_score']
        stats_parts = [f"Points: {n:,}"]

        if event_data['has_larmatch']:
            n_above = (lm >= lm_threshold).sum()
            stats_parts.append(
                f"LM score: mean={lm.mean():.3f}, median={np.median(lm):.3f}"
            )
            stats_parts.append(
                f"Above threshold ({lm_threshold:.2f}): {n_above:,} ({100*n_above/n:.1f}%)"
            )

        if event_data['has_truth']:
            n_ghost = (event_data['hasmatch'] == 0).sum()
            n_true = (event_data['hasmatch'] == 1).sum()
            stats_parts.append(
                f"Truth: {n_true:,} true ({100*n_true/n:.1f}%), "
                f"{n_ghost:,} ghost ({100*n_ghost/n:.1f}%)"
            )

        stats_text = " | ".join(stats_parts)

        # Plots
        scatter_fig = create_3d_scatter(
            event_data, color_mode=color_mode,
            lm_threshold=lm_threshold,
            marker_size=marker_size
        )
        hist_fig = create_score_histogram(event_data)
        thresh_fig = create_threshold_stats(event_data)

        return file_info, entry_info, stats_text, scatter_fig, hist_fig, thresh_fig

    print(f"Starting Dash server on port {args.port}")
    print(f"Open http://localhost:{args.port} in your browser")

    app.run(debug=False, port=args.port, host='0.0.0.0')


if __name__ == '__main__':
    main()
