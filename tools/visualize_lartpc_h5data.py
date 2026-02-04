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
import h5py
import numpy as np
from dash import Dash, html, dcc, callback, Output, Input, State
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from detectoroutline import DetectorOutline


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
    """Load data for a single entry from the HDF5 file."""
    entry_grp = h5file[entry_key]

    data = {}

    # Load triplet data (3D points)
    triplet_data = entry_grp['triplet_data']
    data['pos'] = triplet_data['pos'][:]
    data['pixval'] = triplet_data['pixval'][:]
    data['pid'] = triplet_data['pid'][:]
    data['origin'] = triplet_data['origin'][:]
    data['hasmatch'] = triplet_data['hasmatch'][:]
    data['trackid'] = triplet_data['trackid'][:]
    data['uwire'] = triplet_data['uwire'][:]
    data['vwire'] = triplet_data['vwire'][:]
    data['ywire'] = triplet_data['ywire'][:]
    data['tick'] = triplet_data['tick'][:]

    # Load ssnet labels if available
    if 'ssnet_label' in triplet_data:
        data['ssnet_label'] = triplet_data['ssnet_label'][:]
    else:
        data['ssnet_label'] = np.zeros(len(data['pos']), dtype=np.int32)

    # Load energy deposition
    if 'edep' in triplet_data:
        data['edep'] = triplet_data['edep'][:]
    else:
        data['edep'] = np.zeros((len(data['pos']), 3), dtype=np.float32)

    # Load wire plane images (sparse format)
    image_data = entry_grp['image_data']
    data['wire_images'] = {}
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

    # Load keypoints if available
    if 'mckeypoints' in entry_grp:
        kp_grp = entry_grp['mckeypoints']
        data['keypoints'] = {
            'pos': kp_grp['pos'][:],
            'kptype': kp_grp['kptype'][:],
            'pid': kp_grp['pid'][:],
            'trackid': kp_grp['trackid'][:]
        }

    return data


def create_3d_scatter(event_data, color_mode='ssnet', show_keypoints=True, opacity=0.8, marker_size=2):
    """Create a 3D scatter plot of the point cloud with various coloring options."""
    pos = event_data['pos']

    if len(pos) == 0:
        fig = go.Figure()
        fig.update_layout(title="No points available")
        return fig

    fig = go.Figure()

    # Create hover text
    def make_hover_text(indices):
        hover_texts = []
        for i in indices:
            text = (
                f"x: {pos[i, 0]:.1f}<br>"
                f"y: {pos[i, 1]:.1f}<br>"
                f"z: {pos[i, 2]:.1f}<br>"
                f"PID: {get_pdg_name(event_data['pid'][i])} ({event_data['pid'][i]})<br>"
                f"SSNET: {SSNET_CLASS_NAMES.get(event_data['ssnet_label'][i], 'unknown')}<br>"
                f"Origin: {ORIGIN_NAMES.get(event_data['origin'][i], 'unknown')}<br>"
                f"TrackID: {event_data['trackid'][i]}<br>"
                f"Wires: ({event_data['uwire'][i]}, {event_data['vwire'][i]}, {event_data['ywire'][i]})<br>"
                f"Tick: {event_data['tick'][i]}<br>"
                f"PixVal: ({event_data['pixval'][i, 0]:.1f}, {event_data['pixval'][i, 1]:.1f}, {event_data['pixval'][i, 2]:.1f})"
            )
            hover_texts.append(text)
        return hover_texts

    if color_mode == 'ssnet':
        # Color by SSNET class
        for ssnet_class in np.unique(event_data['ssnet_label']):
            mask = event_data['ssnet_label'] == ssnet_class
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
        for origin_type in np.unique(event_data['origin']):
            mask = event_data['origin'] == origin_type
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
        unique_trackids = np.unique(event_data['trackid'])
        np.random.seed(42)  # For reproducible colors
        for tid in unique_trackids:
            if tid <= 0:
                continue
            mask = event_data['trackid'] == tid
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

    elif color_mode == 'hasmatch':
        # Color by has match (true vs ghost)
        for match_val, name, color in [(1, 'True', 'green'), (0, 'Ghost', 'red')]:
            mask = event_data['hasmatch'] == match_val
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
        pixval_sum = np.sum(event_data['pixval'], axis=1)
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
        edep_sum = np.sum(event_data['edep'], axis=1)
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

        for kptype in np.unique(kp_types):
            mask = kp_types == kptype
            kp_pts = kp_pos[mask]

            if len(kp_pts) == 0:
                continue

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
                hovertext=[f"{KPTYPE_NAMES.get(kptype, 'KP')}<br>({p[0]:.1f}, {p[1]:.1f}, {p[2]:.1f})" for p in kp_pts]
            ))

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
            text=f"3D Point Cloud - {color_mode.upper()} coloring ({len(pos)} points)",
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
        title='Wire Plane Images (Sparse)',
        showlegend=True,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
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

    # Color mode options
    color_options = [
        {'label': 'SSNET Class', 'value': 'ssnet'},
        {'label': 'Origin (Nu/Cosmic)', 'value': 'origin'},
        {'label': 'Track ID', 'value': 'trackid'},
        {'label': 'True/Ghost', 'value': 'hasmatch'},
        {'label': 'Pixel Value', 'value': 'pixval'},
        {'label': 'Energy Deposition', 'value': 'edep'},
    ]

    # Create Dash app
    app = Dash(__name__)

    app.layout = html.Div([
        html.H1("LArTPC Data Viewer", style={'textAlign': 'center'}),

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

        # Visualization options
        html.Div([
            html.Div([
                html.Label('3D Color Mode:', style={'fontWeight': 'bold', 'marginRight': '10px'}),
                dcc.Dropdown(
                    id='color-mode-dropdown',
                    options=color_options,
                    value='ssnet',
                    style={'width': '200px', 'display': 'inline-block'}
                ),
            ], style={'display': 'inline-block', 'marginRight': '30px'}),

            html.Div([
                dcc.Checklist(
                    id='show-keypoints',
                    options=[{'label': ' Show Keypoints', 'value': 'show'}],
                    value=['show'],
                    style={'display': 'inline-block'}
                ),
            ], style={'display': 'inline-block', 'marginRight': '30px'}),

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

        # Wire plane images
        html.H3("Wire Plane Images", style={'textAlign': 'center'}),
        html.Div([
            dcc.Graph(id='wire-images-combined', style={'width': '100%'})
        ], style={'marginBottom': '20px'}),

        # Individual plane views (collapsible)
        html.Details([
            html.Summary("Individual Plane Views (click to expand)", style={'cursor': 'pointer', 'fontWeight': 'bold'}),
            html.Div([
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
        Input('current-entry', 'data'),
        Input('color-mode-dropdown', 'value'),
        Input('show-keypoints', 'value'),
        Input('marker-size-slider', 'value')
    )
    def update_display(entry_idx, color_mode, show_keypoints, marker_size):
        entry_key = entry_keys[entry_idx]
        event_data = load_event_data(h5file, entry_key)

        # Entry info
        entry_info = f"Entry {entry_idx}/{num_entries - 1} ({entry_key})"

        # Statistics
        n_points = len(event_data['pos'])
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

        # 3D scatter plot
        scatter_fig = create_3d_scatter(
            event_data,
            color_mode=color_mode,
            show_keypoints='show' in (show_keypoints or []),
            marker_size=marker_size
        )

        # Wire plane images
        combined_fig = create_combined_wire_images(event_data)
        plane0_fig = create_wire_plane_images(event_data, 0)
        plane1_fig = create_wire_plane_images(event_data, 1)
        plane2_fig = create_wire_plane_images(event_data, 2)

        return entry_info, stats_text, scatter_fig, combined_fig, plane0_fig, plane1_fig, plane2_fig

    print(f"Starting Dash server on port {args.port}")
    print(f"Loaded {num_entries} entries from {args.h5file}")
    print(f"Starting at entry {start_entry}")
    print(f"Open http://localhost:{args.port} in your browser")

    app.run(debug=False, port=args.port, host='0.0.0.0')

    h5file.close()


if __name__ == '__main__':
    main()
