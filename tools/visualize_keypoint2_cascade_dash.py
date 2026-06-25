"""Interactive Dash viewer for attempt-2 keypoint-cascade output.

A live web app over a DIRECTORY of per-event H5 files written by
run_larformer_keypoint2_cascade_inference.py. Two dropdowns drive it:
  - Event   : which event file.
  - Particle: "All particles" or an individual predicted particle. When one is
              selected it is colour-highlighted and the other particles' points
              are drawn small + grey for context.
The figure is the same two-panel 3D layout as the static HTML tool
(visualize_keypoint2_cascade.py, whose trace logic this reuses): LEFT = predicted
particles + predicted keypoints + nu vertex; RIGHT (if GT present) = the
trackid-matched GT particle + GT keypoints (empty when the selection has no
match). Camera is preserved across selections.

  python tools/visualize_keypoint2_cascade_dash.py <dir-of-event-h5> [--port 8050]
then open http://127.0.0.1:8050 in a browser.
"""
import argparse
import glob
import os
from functools import lru_cache

from dash import Dash, dcc, html, Input, Output, State, Patch, no_update

from visualize_keypoint2_cascade import (
    load_event, figure_for_view, figure_score_map, _cls_name)


@lru_cache(maxsize=64)
def _cached_load(path):
    return load_event(path)


def _particle_options(parts):
    opts = [{"label": "All particles", "value": -1}]
    for i, p in enumerate(parts):
        tag = "" if p["has_match"] else "  (no GT match)"
        opts.append({"label": f"P{i}: {_cls_name(p['cls'])}{tag}", "value": i})
    return opts


def _scoremap_heads(ev):
    """Sorted score-map head names present in an event (empty if none saved)."""
    return sorted((ev[5].get("score_maps") or {}).keys())


def build_app(files):
    app = Dash(__name__)
    ev0 = _cached_load(files[0])
    app.layout = html.Div([
        html.H3("Keypoint-2 cascade viewer"),
        html.Div([
            html.Div([html.Label("Event"),
                      dcc.Dropdown(
                          id="event",
                          options=[{"label": os.path.basename(f), "value": f}
                                   for f in files],
                          value=files[0], clearable=False)],
                     style={"width": "60%", "display": "inline-block",
                            "paddingRight": "1em"}),
            html.Div([html.Label("Particle"),
                      dcc.Dropdown(id="particle",
                                   options=_particle_options(ev0[3]),
                                   value=-1, clearable=False)],
                     style={"width": "38%", "display": "inline-block"}),
        ]),
        html.Div(id="info", style={"margin": "0.4em 0", "fontFamily":
                                   "monospace", "fontSize": "0.9em"}),
        dcc.Graph(id="graph", style={"height": "82vh"}),
        dcc.Store(id="cam"),   # last synced camera (links the two scenes)

        # ---- dense head score-map diagnostic (only if the file carries them) --
        html.Hr(),
        html.Div([
            html.Div([html.Label("Score-map head"),
                      dcc.Dropdown(id="head",
                                   options=[{"label": h, "value": h}
                                            for h in _scoremap_heads(ev0)],
                                   value=(_scoremap_heads(ev0) or [None])[0],
                                   clearable=False)],
                     style={"width": "40%", "display": "inline-block",
                            "paddingRight": "1em"}),
            html.Div([html.Label("Score threshold"),
                      dcc.Slider(id="score-thresh", min=0.0, max=1.0, step=0.05,
                                 value=0.0,
                                 marks={0: "0", 0.5: "0.5", 1: "1"})],
                     style={"width": "55%", "display": "inline-block",
                            "verticalAlign": "top"}),
        ]),
        dcc.Graph(id="scoremap", style={"height": "82vh"}),
    ], style={"margin": "1em"})

    @app.callback(Output("particle", "options"), Output("particle", "value"),
                  Input("event", "value"))
    def _opts(path):
        parts = _cached_load(path)[3]
        return _particle_options(parts), -1

    @app.callback(Output("head", "options"), Output("head", "value"),
                  Input("event", "value"), State("head", "value"))
    def _head_opts(path, cur):
        heads = _scoremap_heads(_cached_load(path))
        opts = [{"label": h, "value": h} for h in heads]
        val = cur if cur in heads else (heads[0] if heads else None)
        return opts, val

    @app.callback(Output("scoremap", "figure"),
                  Input("event", "value"), Input("head", "value"),
                  Input("score-thresh", "value"))
    def _scoremap(path, head, thresh):
        if not head:
            return figure_score_map(_cached_load(path), "", thresh or 0.0)
        return figure_score_map(_cached_load(path), head, thresh or 0.0)

    @app.callback(Output("graph", "figure"), Output("info", "children"),
                  Input("event", "value"), Input("particle", "value"))
    def _fig(path, view):
        ev = _cached_load(path)
        _sc, _nu, _gtnu, parts, has_gt, meta = ev
        view = -1 if view is None else int(view)
        fig = figure_for_view(ev, view)
        n_match = sum(p["has_match"] for p in parts)
        sel = ("all" if view < 0 else
               f"P{view} ({_cls_name(parts[view]['cls'])})")
        info = (f"{meta.get('src_file')} | run {meta.get('run')} "
                f"subrun {meta.get('subrun')} event {meta.get('event')} | "
                f"{len(parts)} predicted particles, {n_match} GT-matched | "
                f"has_gt={has_gt} | showing: {sel}")
        return fig, info

    # ---- linked rotation: rotating either 3D scene drives both ----
    # Loop-safe via the `cam` Store: _capture only updates the Store when the
    # incoming camera differs from it, so the programmatic camera set by
    # _apply (which re-emits relayoutData with the SAME camera) is a no-op and
    # the cycle stops.
    @app.callback(Output("cam", "data"), Input("graph", "relayoutData"),
                  State("cam", "data"), prevent_initial_call=True)
    def _capture(relayout, cur):
        if not relayout:
            return no_update
        cam = relayout.get("scene.camera") or relayout.get("scene2.camera")
        if cam is None or cam == cur:
            return no_update
        return cam

    @app.callback(Output("graph", "figure", allow_duplicate=True),
                  Input("cam", "data"), prevent_initial_call=True)
    def _apply(cam):
        if cam is None:
            return no_update
        patch = Patch()
        patch["layout"]["scene"]["camera"] = cam
        patch["layout"]["scene2"]["camera"] = cam
        return patch

    return app


def main():
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("dir", help="directory of keypoint2_event*.h5 files")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8050)
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.dir, "*.h5")))
    if not files:
        raise SystemExit(f"no .h5 in {args.dir}")
    print(f"serving {len(files)} events from {args.dir}")
    build_app(files).run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
