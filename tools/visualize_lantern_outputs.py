#!/usr/bin/env python3
"""
Plotly/Dash visualization for lantern outputs stored in HDF5 files.

Usage:
    python visualize_lantern_outputs.py <hdf5_file> [--start-entry N]

Features:
- 3D scatter plot of spacepoints colored by prong
- 2D image crops arranged in (2,3) grid for each prong
- Navigation buttons to move through events
"""

import argparse
import h5py
import numpy as np
from dash import Dash, html, dcc, callback, Output, Input, State
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import random

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
    0: 'unknown',
}


def get_pdg_name(pdg_code):
    """Convert PDG code to human-readable particle name."""
    return PDG_NAMES.get(pdg_code, f'PDG:{pdg_code}')


def load_event_data(h5file, event_key):
    """Load data for a single event from the HDF5 file."""
    event_grp = h5file[event_key]

    # Load spacepoints: shape (N, 5) where columns are [x, y, z, prong_idx, prong_type]
    spacepoints = event_grp['spacepoints'][:]

    # Load track images
    track_images = {}
    if 'track_images' in event_grp:
        track_grp = event_grp['track_images']
        for key in track_grp.keys():
            idx = int(key.split('_')[1])
            track_images[idx] = track_grp[key][:]

    # Load shower images
    shower_images = {}
    if 'shower_images' in event_grp:
        shower_grp = event_grp['shower_images']
        for key in shower_grp.keys():
            idx = int(key.split('_')[1])
            shower_images[idx] = shower_grp[key][:]

    # Load track info: [prong_type, index, is_secondary, pdg_code]
    track_info = {}
    if 'track_info' in event_grp:
        track_info_grp = event_grp['track_info']
        for key in track_info_grp.keys():
            idx = int(key.split('_')[1])
            track_info[idx] = track_info_grp[key][:]

    # Load shower info: [prong_type, index, is_secondary, pdg_code]
    shower_info = {}
    if 'shower_info' in event_grp:
        shower_info_grp = event_grp['shower_info']
        for key in shower_info_grp.keys():
            idx = int(key.split('_')[1])
            shower_info[idx] = shower_info_grp[key][:]

    return {
        'spacepoints': spacepoints,
        'track_images': track_images,
        'shower_images': shower_images,
        'track_info': track_info,
        'shower_info': shower_info,
        'event_key': event_key
    }


def generate_prong_colors(num_prongs):
    """Generate random colors for each prong."""
    colors = []
    for _ in range(num_prongs):
        r = random.randint(50, 255)
        g = random.randint(50, 255)
        b = random.randint(50, 255)
        colors.append(f'rgb({r},{g},{b})')
    return colors


def create_3d_scatter(spacepoints, track_info=None, shower_info=None):
    """Create a 3D scatter plot of spacepoints colored by prong."""
    if len(spacepoints) == 0:
        fig = go.Figure()
        fig.update_layout(title="No spacepoints available")
        return fig

    # Get unique prong indices
    prong_indices = np.unique(spacepoints[:, 3]).astype(int)
    num_prongs = len(prong_indices)
    colors = generate_prong_colors(num_prongs)

    fig = go.Figure()

    for i, prong_idx in enumerate(prong_indices):
        mask = spacepoints[:, 3] == prong_idx
        pts = spacepoints[mask]

        # Determine prong type (0=track, 1=shower)
        prong_type = int(pts[0, 4]) if len(pts) > 0 else 0
        prong_type_str = "shower" if prong_type == 1 else "track"

        # Get PDG code and primary/secondary info
        pdg_code = 0
        is_secondary = False
        if prong_type == 0 and track_info is not None:
            # Find track index within tracks (prong_idx may include showers before it)
            # We need to find the local track index
            for tidx, tinfo in track_info.items():
                if len(tinfo) > 0 and tinfo[0, 1] == prong_idx:
                    pdg_code = int(tinfo[0, 3])
                    is_secondary = bool(tinfo[0, 2])
                    break
                elif len(tinfo.shape) == 1 and tinfo[1] == tidx:
                    pdg_code = int(tinfo[3])
                    is_secondary = bool(tinfo[2])
                    break
            # Fallback: use the track index directly
            if pdg_code == 0 and prong_idx in track_info:
                tinfo = track_info[prong_idx]
                if len(tinfo.shape) == 1:
                    pdg_code = int(tinfo[3])
                    is_secondary = bool(tinfo[2])
                elif len(tinfo) > 0:
                    pdg_code = int(tinfo[0, 3])
                    is_secondary = bool(tinfo[0, 2])
        elif prong_type == 1 and shower_info is not None:
            # Find shower info by local shower index
            # Shower indices in shower_info are local (0, 1, 2, ...)
            # We need to map prong_idx to shower local index
            for sidx, sinfo in shower_info.items():
                if len(sinfo.shape) == 1:
                    pdg_code = int(sinfo[3])
                    is_secondary = bool(sinfo[2])
                    break
                elif len(sinfo) > 0:
                    pdg_code = int(sinfo[0, 3])
                    is_secondary = bool(sinfo[0, 2])
                    break

        pdg_name = get_pdg_name(pdg_code)
        primary_str = "secondary" if is_secondary else "primary"

        # Create hover text
        hover_text = [f"Prong {int(prong_idx)} ({prong_type_str})<br>"
                      f"PDG: {pdg_name} ({pdg_code})<br>"
                      f"Type: {primary_str}<br>"
                      f"x: {p[0]:.2f}<br>y: {p[1]:.2f}<br>z: {p[2]:.2f}"
                      for p in pts]

        fig.add_trace(go.Scatter3d(
            x=pts[:, 0],
            y=pts[:, 1],
            z=pts[:, 2],
            mode='markers',
            marker=dict(size=2, color=colors[i], opacity=0.8),
            name=f"Prong {int(prong_idx)} ({prong_type_str}, {pdg_name}, {primary_str})",
            hovertext=hover_text,
            hoverinfo='text'
        ))

    fig.update_layout(
        scene=dict(
            xaxis_title='X (cm)',
            yaxis_title='Y (cm)',
            zaxis_title='Z (cm)',
            aspectmode='data',
            aspectratio=dict(x=1,y=1,z=1),
        ),
        title="3D Spacepoints by Prong",
        height=600,
        margin=dict(l=0, r=0, t=40, b=0)
    )

    return fig


def create_prong_image_figure(images, prong_type, prong_idx, pdg_code=0, is_secondary=False):
    """Create a 2x3 subplot figure for the 6 image crops of a prong.

    Layout: odd indices (1,3,5) in top row, even indices (0,2,4) in bottom row.
    """
    if images is None or len(images) != 6:
        fig = go.Figure()
        fig.update_layout(title=f"No images for {prong_type} {prong_idx}")
        return fig

    pdg_name = get_pdg_name(pdg_code)
    primary_str = "secondary" if is_secondary else "primary"

    # Create 2x3 subplot
    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=[
            f"U Plane Context", f"V Plane Context", f"Y Plane Context",  # Top row (odd indices)
            f"U Plane Prong", f"V Plane Prong", f"Y Plane Prong"   # Bottom row (even indices)
        ],
        vertical_spacing=0.1,
        horizontal_spacing=0.05
    )

    # Top row: odd indices (1, 3, 5)
    odd_indices = [1, 3, 5]
    for col, img_idx in enumerate(odd_indices, start=1):
        fig.add_trace(
            go.Heatmap(
                z=images[img_idx],
                colorscale='Viridis',
                showscale=(col == 3),  # Only show colorbar on last column
                hovertemplate='x: %{x}<br>y: %{y}<br>value: %{z:.2f}<extra></extra>'
            ),
            row=1, col=col
        )

    # Bottom row: even indices (0, 2, 4)
    even_indices = [0, 2, 4]
    for col, img_idx in enumerate(even_indices, start=1):
        fig.add_trace(
            go.Heatmap(
                z=images[img_idx],
                colorscale='Viridis',
                showscale=False,
                hovertemplate='x: %{x}<br>y: %{y}<br>value: %{z:.2f}<extra></extra>'
            ),
            row=2, col=col
        )

    fig.update_layout(
        title=f"{prong_type.capitalize()} {prong_idx} - {pdg_name} (PDG: {pdg_code}) - {primary_str}",
        height=1000,
        margin=dict(l=20, r=20, t=60, b=20)
    )

    # Update axes to have equal aspect and hide ticks for cleaner look
    for i in range(1, 7):
        fig.update_xaxes(showticklabels=False, row=(i-1)//3 + 1, col=(i-1)%3 + 1)
        fig.update_yaxes(showticklabels=False, row=(i-1)//3 + 1, col=(i-1)%3 + 1)

    return fig


def main():
    parser = argparse.ArgumentParser(description='Visualize lantern outputs from HDF5 file')
    parser.add_argument('h5file', type=str, help='Path to HDF5 file')
    parser.add_argument('--start-entry', type=int, default=0, help='Starting entry index')
    parser.add_argument('--port', type=int, default=8050, help='Port for Dash server')
    args = parser.parse_args()

    # Open HDF5 file and get event keys
    h5file = h5py.File(args.h5file, 'r')
    event_keys = list(h5file.keys())
    num_events = len(event_keys)

    if num_events == 0:
        print("No events found in HDF5 file")
        h5file.close()
        return

    start_entry = min(args.start_entry, num_events - 1)

    # Create Dash app
    app = Dash(__name__)

    app.layout = html.Div([
        html.H1("Lantern Output Viewer", style={'textAlign': 'center'}),

        # Navigation controls
        html.Div([
            html.Button('Previous', id='prev-btn', n_clicks=0,
                       style={'marginRight': '10px', 'fontSize': '16px', 'padding': '10px 20px'}),
            html.Span(id='event-info', style={'fontSize': '18px', 'marginRight': '10px'}),
            html.Button('Next', id='next-btn', n_clicks=0,
                       style={'marginLeft': '10px', 'fontSize': '16px', 'padding': '10px 20px'}),
            html.Span(" | Go to entry: ", style={'marginLeft': '20px'}),
            dcc.Input(id='entry-input', type='number', value=start_entry, min=0, max=num_events-1,
                     style={'width': '80px', 'marginRight': '10px'}),
            html.Button('Go', id='go-btn', n_clicks=0,
                       style={'fontSize': '16px', 'padding': '5px 15px'}),
        ], style={'textAlign': 'center', 'marginBottom': '20px', 'padding': '10px'}),

        # Store current entry index
        dcc.Store(id='current-entry', data=start_entry),

        # 3D scatter plot
        html.Div([
            dcc.Graph(id='scatter-3d', style={'width': '100%'})
        ], style={'marginBottom': '20px'}),

        # Prong images container
        html.Div(id='prong-images-container')
    ])

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
            new_entry = min(num_events - 1, current_entry + 1)
        elif triggered_id == 'go-btn':
            new_entry = max(0, min(num_events - 1, input_value if input_value is not None else 0))
        else:
            new_entry = current_entry

        return new_entry, new_entry

    @callback(
        Output('event-info', 'children'),
        Output('scatter-3d', 'figure'),
        Output('prong-images-container', 'children'),
        Input('current-entry', 'data')
    )
    def update_display(entry_idx):
        event_key = event_keys[entry_idx]
        event_data = load_event_data(h5file, event_key)

        # Parse event key for display
        parts = event_key.split('_')
        if len(parts) >= 6:
            fileid, run, subrun, event, vtxidx = parts[1], parts[2], parts[3], parts[4], parts[5]
            event_info = f"Entry {entry_idx}/{num_events-1} | Run: {run}, Subrun: {subrun}, Event: {event}, VtxIdx: {vtxidx}"
        else:
            event_info = f"Entry {entry_idx}/{num_events-1} | {event_key}"

        # Create 3D scatter
        scatter_fig = create_3d_scatter(
            event_data['spacepoints'],
            track_info=event_data.get('track_info'),
            shower_info=event_data.get('shower_info')
        )

        # Create prong image figures
        prong_figures = []

        # Track images
        for track_idx in sorted(event_data['track_images'].keys()):
            images = event_data['track_images'][track_idx]
            # Get track info for this track
            pdg_code = 0
            is_secondary = False
            if track_idx in event_data.get('track_info', {}):
                tinfo = event_data['track_info'][track_idx]
                if len(tinfo.shape) == 1:
                    pdg_code = int(tinfo[3])
                    is_secondary = bool(tinfo[2])
                elif len(tinfo) > 0:
                    pdg_code = int(tinfo[0, 3])
                    is_secondary = bool(tinfo[0, 2])
            fig = create_prong_image_figure(images, 'track', track_idx, pdg_code, is_secondary)
            prong_figures.append(
                html.Div([
                    dcc.Graph(figure=fig)
                ], style={'marginBottom': '20px', 'border': '1px solid #ddd', 'padding': '10px'})
            )

        # Shower images
        for shower_idx in sorted(event_data['shower_images'].keys()):
            images = event_data['shower_images'][shower_idx]
            # Get shower info for this shower
            pdg_code = 0
            is_secondary = False
            if shower_idx in event_data.get('shower_info', {}):
                sinfo = event_data['shower_info'][shower_idx]
                if len(sinfo.shape) == 1:
                    pdg_code = int(sinfo[3])
                    is_secondary = bool(sinfo[2])
                elif len(sinfo) > 0:
                    pdg_code = int(sinfo[0, 3])
                    is_secondary = bool(sinfo[0, 2])
            fig = create_prong_image_figure(images, 'shower', shower_idx, pdg_code, is_secondary)
            prong_figures.append(
                html.Div([
                    dcc.Graph(figure=fig)
                ], style={'marginBottom': '20px', 'border': '1px solid #ddd', 'padding': '10px'})
            )

        return event_info, scatter_fig, prong_figures

    print(f"Starting Dash server on port {args.port}")
    print(f"Loaded {num_events} events from {args.h5file}")
    print(f"Starting at entry {start_entry}")

    app.run(debug=False, port=args.port, host='0.0.0.0')

    h5file.close()


if __name__ == '__main__':
    main()
