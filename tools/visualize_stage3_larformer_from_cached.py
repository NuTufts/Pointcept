"""LArFormer Stage-3 GT visualizer — reads a Stage-1+2 cache and shows
per-level truth labels (and, optionally, inference predictions) as the
trainer / model would see them.

This is the Stage-3 sibling of `tools/visualize_larformer_gt.py`. The
GT panel reuses that tool's level-building + figure-construction code
path (`build_event_gt`, `figure_for_event`, `metadata_panel`) directly,
so the visualizer can't drift from the Stage-3 training pipeline.

What changes from the Stage-2 GT viz:

  - The dataset is a `LArFormerStage12CacheDataset` reading per-event
    HDF5 caches produced by `tools/build_stage12_cache_event.py` /
    `_shard.py`. The cascade's deghoster + slicer filter has already
    been applied; this tool does NOT re-run them.
  - The Stage-3 config declares Stage 3's level pyramid and the 7-class
    particle taxonomy. Both are read from the config and threaded into
    the same `CompositeTokenizer` + `build_per_level_gt` the trainer
    uses.
  - The deghoster slider / cascade-config flag are gone (cache is
    already filtered).
  - A `source_set_filter` dropdown lets you flip between the cached
    SP subsets (`stage2_pass`, `gt_nu`, `union`, ...) live.
  - Optional `--stage3pred-dir`: directory of `stage3pred_*.h5` files
    produced by `tools/run_larformer_stage3_inference.py`. When set,
    a second 3D scene appears below the GT scene showing the model's
    Stage-3 predictions for the same event — colored by predicted
    particle / class / mask probability / source_mask / etc., with
    overlayed predicted-origin diamonds and optional pred ↔ GT origin
    lines. The visualizer + inference share their figure-building code
    via `pointcept/models/LArFormer/viz_inference.py` so the panel
    can't drift from the on-disk schema.

Usage (GT only):

    python tools/visualize_stage3_larformer_from_cached.py \\
        --config configs/lartpc/larformer-particle-v1-cached.py \\
        --cache /tmp/stage12_cache_v2/val

Usage (with inference overlay):

    python tools/visualize_stage3_larformer_from_cached.py \\
        --config configs/lartpc/larformer-particle-v1-cached-ptv3crosslevel.py \\
        --cache exp/cache_stage12_ptv3crosslevelslicer_iter_75750/val \\
        --stage3pred-dir exp/.../inference \\
        --min-mask-prob 0.5

Once running, open http://<host>:<port> in a browser.
"""
import argparse
import atexit
import os
import sys
import tempfile

# Repo root + tools/ on sys.path so imports work from anywhere.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dash import Dash, Input, Output, dcc, html  # noqa: E402

# Reuse the Stage-2 GT visualizer's per-level GT pipeline + figure
# constructors directly. THE WHOLE POINT of this tool is to share that
# code path so the Stage-3 viz can't disagree with the Stage-2 viz or
# with the model's loss-time GT lifting.
from visualize_larformer_gt import (  # noqa: E402
    build_event_gt,
    figure_for_event,
    metadata_panel,
)

# Shared prediction-rendering code lives in pointcept/ so this tool
# stays thin. See pointcept/models/LArFormer/viz_inference.py.
from pointcept.models.LArFormer.inference import load_event_h5  # noqa: E402
from pointcept.models.LArFormer.viz_inference import (  # noqa: E402
    figure_for_stage3_prediction,
    stage3_color_by_options,
    stage3_metadata_summary,
)


VALID_FILTERS = (
    "stage2_pass", "gt_nu", "stage2_delta", "union", "all",
    "stage2_random_tau", "stage2_plus_gt_dropout",
)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--config", required=True,
        help="Stage-3 LArFormer config — typically "
             "configs/lartpc/larformer-particle-v1-cached.py. The viz "
             "reads cfg.model.levels + cfg.model.token_dim for the "
             "tokenizer and cfg.data.<split> for the cache dataset.",
    )
    ap.add_argument(
        "--cache", required=True,
        help="Path to a single Stage-1+2 cache .h5 file OR a directory "
             "of cached files (the shard driver's 3-level hash layout). "
             "A single file restricts the viz to that one event.",
    )
    ap.add_argument(
        "--split", default="val", choices=("train", "val", "test"),
        help="Which cfg.data.<split> stanza to read (for the cache "
             "reader's source_set_filter, recenter_to_centroid, etc. "
             "default knobs).",
    )
    ap.add_argument(
        "--entry", type=int, default=0,
        help="Initial event index.",
    )
    ap.add_argument(
        "--source-set-filter", default=None,
        choices=(None,) + VALID_FILTERS,
        help="Override the cache reader's source_set_filter. Default: "
             "use the cfg's setting (typically stage2_pass).",
    )
    ap.add_argument(
        "--no-recenter", action="store_true",
        help="Disable centroid recentering (default: use cfg's setting).",
    )
    ap.add_argument(
        "--stage3pred-dir", default=None,
        help="Optional directory of stage3pred_*.h5 files produced by "
             "tools/run_larformer_stage3_inference.py. When set, a "
             "second 3D scene appears below the GT scene showing the "
             "model's Stage-3 predictions for the same event. Matched "
             "by filename: viz looks for "
             "stage3pred_<cache_basename_without_ext>.h5.",
    )
    ap.add_argument(
        "--min-mask-prob", type=float, default=0.0,
        help="(Prediction panel only) Demote per-SP assignments whose "
             "sigmoid mask probability is below this floor to "
             "no_object/unassigned. Useful for hiding low-confidence "
             "panoptic assignments. 0 = strict argmax.",
    )
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8050)
    args = ap.parse_args()

    # Late imports — keep startup fast for `--help`.
    from pointcept.utils.config import Config
    import pointcept.models   # noqa: F401  side-effect: register MODELS
    import pointcept.datasets  # noqa: F401  side-effect: register DATASETS
    from pointcept.datasets import build_dataset

    cfg = Config.fromfile(args.config)

    # ---- Stage-3 model levels (the source of truth for the viz) ------
    # The cached config has `model = dict(type="LArFormer", ...)`. If a
    # user accidentally points this tool at a wrapped (cascade) config,
    # surface the issue with a useful message rather than a cryptic
    # AttributeError on `cfg.model.levels`.
    model_type = cfg.model.get("type", "?")
    if "levels" not in cfg.model:
        raise ValueError(
            f"--config has model.type={model_type!r} which doesn't "
            f"expose `model.levels` at the top level. The Stage-3 viz "
            f"expects a flat LArFormer config (e.g. "
            f"`larformer-particle-v1-cached.py`). If you have a "
            f"CascadedParticleSegmenter cfg, point at its inner "
            f"`particle_segmenter` block."
        )
    levels_cfg = list(cfg.model.levels)
    token_dim = int(cfg.model.token_dim)
    level_names = [L["name"] for L in levels_cfg]
    print(f"Stage-3 config: model.type={model_type}  "
          f"token_dim={token_dim}")
    print(f"Stage-3 levels declared in config: {level_names}")

    # ---- Build the cache dataset, with the user's cache path ---------
    ds_cfg = dict(cfg.data[args.split])
    if args.source_set_filter is not None:
        ds_cfg["source_set_filter"] = args.source_set_filter
    if args.no_recenter:
        ds_cfg["recenter_to_centroid"] = False

    cache_abs = os.path.abspath(args.cache)
    if not os.path.exists(cache_abs):
        raise FileNotFoundError(f"--cache not found: {cache_abs}")

    if os.path.isfile(cache_abs):
        # Single-file mode: 1-line tempfile filelist. data_list_file is
        # preferred over data_root by LArFormerStage12CacheDataset, so we
        # set both defensively.
        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", prefix="viz_s3_cache_", delete=False,
        )
        tmp.write(cache_abs + "\n")
        tmp.close()
        atexit.register(lambda p=tmp.name:
                        os.path.exists(p) and os.unlink(p))
        ds_cfg["data_list_file"] = tmp.name
        ds_cfg["data_root"] = "/"
        print(f"[viz] single-file mode: {cache_abs}")
        print(f"[viz]   tempfile filelist: {tmp.name}")
    else:
        # Directory mode: cache reader walks recursively for .h5 files.
        ds_cfg["data_root"] = cache_abs
        ds_cfg.pop("data_list_file", None)
        print(f"[viz] directory mode (recursive crawl): {cache_abs}")

    dataset = build_dataset(ds_cfg)
    n_events = len(dataset)
    if n_events == 0:
        raise RuntimeError(
            f"No cached events found under {cache_abs}. Check that the "
            f"path actually contains .h5 cache files (or, for "
            f"single-file mode, that the path resolves)."
        )
    src_filter = ds_cfg.get("source_set_filter", "stage2_pass")
    print(f"Loaded {args.split} dataset: {n_events} events  "
          f"(source_set_filter={src_filter}, "
          f"recenter_to_centroid={ds_cfg.get('recenter_to_centroid', False)})")

    # ---- Prediction overlay setup ------------------------------------
    pred_dir = (os.path.abspath(args.stage3pred_dir)
                if args.stage3pred_dir else None)
    has_pred = pred_dir is not None
    if has_pred:
        if not os.path.isdir(pred_dir):
            raise FileNotFoundError(
                f"--stage3pred-dir does not exist: {pred_dir}")
        print(f"[viz] prediction overlay: {pred_dir}")
        print(f"[viz]   default min_mask_prob = {args.min_mask_prob}")

    def _pred_path_for_event(idx: int) -> "str | None":
        """Stage-3 pred files are named `stage3pred_<cache_basename>.h5`
        where cache_basename is the cache file's name minus its `.h5`."""
        if not has_pred:
            return None
        try:
            cache_path = dataset.data_list[idx % len(dataset.data_list)]
        except Exception:
            return None
        stem = os.path.splitext(os.path.basename(cache_path))[0]
        return os.path.join(pred_dir, f"stage3pred_{stem}.h5")

    pred_color_options = stage3_color_by_options()

    # ---- Dash app ----------------------------------------------------
    app = Dash(__name__, suppress_callback_exceptions=True)
    app.title = "LArFormer Stage-3 GT visualizer (from cache)"

    # Default initial scene heights — overridden by toggle_layout when
    # the side-by-side / stacked checkbox flips.
    gt_scene_height = "78vh" if has_pred else "78vh"  # side-by-side default
    pred_scene_height = "78vh" if has_pred else "0vh"

    pred_controls = []
    if has_pred:
        pred_controls = [
            html.Div([
                html.Span("Pred color by: ",
                          style={"marginRight": "6px"}),
                dcc.Dropdown(
                    id="pred_color",
                    options=pred_color_options,
                    value=pred_color_options[0]["value"],
                    clearable=False,
                    style={"width": "360px",
                           "display": "inline-block",
                           "marginRight": "16px"},
                ),
                html.Span("min_mask_prob: ",
                          style={"marginRight": "6px"}),
                dcc.Input(
                    id="min_mask_prob",
                    type="number", min=0.0, max=1.0, step=0.05,
                    value=args.min_mask_prob,
                    style={"width": "80px", "marginRight": "16px"},
                ),
                dcc.Checklist(
                    id="show_origin_diamonds",
                    options=[{"label": " show pred-origin diamonds",
                              "value": "on"}],
                    value=["on"],
                    style={"display": "inline-block",
                           "marginRight": "16px"},
                ),
                dcc.Checklist(
                    id="show_origin_lines",
                    options=[{"label": " show pred↔GT origin lines",
                              "value": "on"}],
                    value=["on"],
                    style={"display": "inline-block",
                           "marginRight": "16px"},
                ),
                dcc.Checklist(
                    id="sync_cameras",
                    options=[{"label": " sync rotation/zoom",
                              "value": "on"}],
                    value=["on"],
                    style={"display": "inline-block",
                           "marginRight": "16px"},
                ),
                dcc.Checklist(
                    id="side_by_side",
                    options=[{"label": " side-by-side panels",
                              "value": "on"}],
                    value=["on"],
                    style={"display": "inline-block",
                           "marginRight": "16px"},
                ),
            ], style={"marginTop": "6px"}),
        ]

    layout_children = [
        html.Div([
            html.H3("LArFormer Stage-3 GT visualizer (from cache)",
                    style={"marginBottom": "4px"}),
            html.Div([
                html.Span("Event: ", style={"marginRight": "6px"}),
                dcc.Input(id="entry", type="number", min=0,
                          max=n_events - 1, step=1, value=args.entry,
                          style={"width": "80px", "marginRight": "16px"}),
                html.Span("Level: ", style={"marginRight": "6px"}),
                dcc.Dropdown(
                    id="level",
                    options=[{"label": n, "value": n} for n in level_names],
                    value=level_names[-1],
                    clearable=False,
                    style={"width": "200px", "display": "inline-block",
                           "marginRight": "16px"},
                ),
                html.Span("GT color by: ", style={"marginRight": "6px"}),
                dcc.Dropdown(
                    id="color",
                    options=[
                        {"label": "GT instance id (= particle)",
                         "value": "instance_id"},
                        {"label": "per-token cls target",
                         "value": "cls_target"},
                        {"label": "sp_to_level_id (token partition)",
                         "value": "sp_to_level_id"},
                        {"label": "SPs by level-cluster GT (back-projected)",
                         "value": "sp_by_level_inst"},
                    ],
                    value="instance_id",
                    clearable=False,
                    style={"width": "360px",
                           "display": "inline-block",
                           "marginRight": "16px"},
                ),
                html.Span("source_set: ", style={"marginRight": "6px"}),
                dcc.Dropdown(
                    id="src",
                    options=[{"label": f, "value": f}
                             for f in VALID_FILTERS],
                    value=src_filter,
                    clearable=False,
                    style={"width": "200px",
                           "display": "inline-block",
                           "marginRight": "16px"},
                ),
                html.Button("Reload event", id="reload", n_clicks=0,
                            style={"marginLeft": "12px"}),
            ], style={"marginBottom": "4px"}),
        ] + pred_controls),
    ]

    # GT scene wrapper (always present).
    gt_wrapper = html.Div(
        id="gt_wrapper",
        children=[
            html.Div("GT (per-level)",
                     style={"fontWeight": "bold",
                            "marginTop": "6px",
                            "marginLeft": "8px"}),
            dcc.Graph(id="scene", style={"height": gt_scene_height}),
        ],
        style={"flex": "1 1 0", "minWidth": "0"} if has_pred
              else {"width": "100%"},
    )

    # Pred scene wrapper — only built when prediction overlay is on.
    pred_wrapper_children = []
    if has_pred:
        pred_wrapper_children = [
            html.Div("Prediction (Stage-3 inference)",
                     style={"fontWeight": "bold",
                            "marginTop": "6px",
                            "marginLeft": "8px"}),
            dcc.Graph(id="pred_scene",
                      style={"height": pred_scene_height}),
        ]
    pred_wrapper = html.Div(
        id="pred_wrapper",
        children=pred_wrapper_children,
        style={"flex": "1 1 0", "minWidth": "0"} if has_pred
              else {"display": "none"},
    )

    # Container that holds both wrappers. Toggled between
    # `display: flex` (side-by-side) and `display: block` (stacked) by
    # the side_by_side checkbox callback.
    scene_container_style = (
        {"display": "flex", "flexDirection": "row", "gap": "8px"}
        if has_pred else {"display": "block"}
    )
    layout_children.append(
        html.Div(
            id="scene_container",
            children=[gt_wrapper, pred_wrapper],
            style=scene_container_style,
        )
    )

    layout_children += [
        html.Div(id="metadata",
                 style={"fontFamily": "monospace",
                        "fontSize": "12px",
                        "padding": "10px",
                        "borderTop": "1px solid #ccc"}),
    ]
    app.layout = html.Div(layout_children)

    # ---- Callbacks ----------------------------------------------------
    # We register two callbacks (GT panel + pred panel) when pred is on,
    # one (GT only) when off. Sharing inputs via the entry / src controls.

    @app.callback(
        Output("scene", "figure"),
        Output("metadata", "children"),
        Input("entry", "value"),
        Input("level", "value"),
        Input("color", "value"),
        Input("src", "value"),
        Input("reload", "n_clicks"),
    )
    def update_gt(entry_val, level_val, color_val, src_val, _n):
        idx = int(entry_val or 0)
        idx = max(0, min(idx, n_events - 1))
        if src_val and src_val != dataset.source_set_filter:
            dataset.source_set_filter = src_val
        event_data = build_event_gt(
            model_levels_cfg=levels_cfg, token_dim=token_dim,
            dataset=dataset, idx=idx,
            deghoster_filter=None, tau=0.5,
        )
        fig = figure_for_event(event_data, level_val, color_val)
        gt_meta = metadata_panel(event_data, prediction=None)

        # If predictions are configured, append a prediction-text panel
        # below the GT metadata so the user can see both at a glance.
        if has_pred:
            pred_path = _pred_path_for_event(idx)
            pred = load_event_h5(pred_path) if pred_path else None
            pred_summary = stage3_metadata_summary(pred)
            pred_meta = html.Div([
                html.Hr(),
                html.Div("Stage-3 prediction summary",
                         style={"fontWeight": "bold"}),
                *[html.Div(ln) for ln in pred_summary],
                html.Div(
                    f"  ← {pred_path}" if pred is not None
                    else f"  ← (no file at {pred_path})",
                    style={"color": "#888", "fontSize": "10px"},
                ),
            ])
            meta_children = [gt_meta, pred_meta]
        else:
            meta_children = gt_meta
        return fig, meta_children

    if has_pred:
        @app.callback(
            Output("pred_scene", "figure"),
            Input("entry", "value"),
            Input("pred_color", "value"),
            Input("min_mask_prob", "value"),
            Input("show_origin_lines", "value"),
            Input("show_origin_diamonds", "value"),
            Input("src", "value"),
            Input("reload", "n_clicks"),
        )
        def update_pred(entry_val, color_val, min_prob, lines_val,
                        diamonds_val, src_val, _n):
            idx = int(entry_val or 0)
            idx = max(0, min(idx, n_events - 1))
            # Mirror update_gt's source-set mutation so the indices align
            # if the user changes the filter.
            if src_val and src_val != dataset.source_set_filter:
                dataset.source_set_filter = src_val
            pred_path = _pred_path_for_event(idx)
            pred = load_event_h5(pred_path) if pred_path else None
            try:
                mp = float(min_prob) if min_prob is not None else 0.0
            except (TypeError, ValueError):
                mp = 0.0
            show_lines = bool(lines_val and "on" in lines_val)
            show_diamonds = bool(diamonds_val and "on" in diamonds_val)
            return figure_for_stage3_prediction(
                pred,
                color_by=color_val or "pred_particle_idx",
                min_mask_prob=mp,
                show_origin_lines=show_lines,
                show_origin_diamonds=show_diamonds,
                title_suffix=(f"  —  {os.path.basename(pred_path)}"
                              if pred_path else ""),
            )

        # ---- Side-by-side / stacked layout toggle ---------------------
        from dash import Patch, no_update
        from dash import State as _State

        @app.callback(
            Output("scene_container", "style"),
            Output("gt_wrapper", "style"),
            Output("pred_wrapper", "style"),
            Output("scene", "style"),
            Output("pred_scene", "style"),
            Input("side_by_side", "value"),
        )
        def toggle_layout(side_by_side_val):
            sbs = bool(side_by_side_val and "on" in side_by_side_val)
            if sbs:
                container = {"display": "flex",
                             "flexDirection": "row", "gap": "8px"}
                wrap = {"flex": "1 1 0", "minWidth": "0"}
                # Side-by-side: each panel is half-width, so a taller
                # vertical extent gives a nicer aspect ratio for talks.
                h = "78vh"
            else:
                container = {"display": "block"}
                wrap = {"width": "100%"}
                # Stacked: two panels of ~half-screen height each.
                h = "42vh"
            return container, wrap, wrap, {"height": h}, {"height": h}

        # ---- Camera sync (server-side Patch in both directions) -------
        # When the user rotates/zooms one scene, mirror the camera state
        # onto the other. Programmatic camera updates do NOT trigger a
        # follow-up `relayoutData` event in Plotly, so there's no
        # feedback loop. Reads `relayoutData["scene.camera"]` (full dict,
        # which Plotly emits at the end of a drag) and ignores fine-
        # grained mid-drag deltas, which keeps the patch payload small.

        def _camera_from_relayout(rl: "dict | None") -> "dict | None":
            if not rl:
                return None
            if "scene.camera" in rl:
                return rl["scene.camera"]
            # Some Plotly versions emit individual axis updates instead.
            cam_keys = [k for k in rl if k.startswith("scene.camera.")]
            if not cam_keys:
                return None
            cam: dict = {}
            for k in cam_keys:
                parts = k[len("scene.camera."):].split(".")
                d = cam
                for p in parts[:-1]:
                    d = d.setdefault(p, {})
                d[parts[-1]] = rl[k]
            return cam

        @app.callback(
            Output("pred_scene", "figure", allow_duplicate=True),
            Input("scene", "relayoutData"),
            _State("sync_cameras", "value"),
            prevent_initial_call=True,
        )
        def sync_pred_to_gt(gt_rl, sync_val):
            if not sync_val or "on" not in sync_val:
                return no_update
            cam = _camera_from_relayout(gt_rl)
            if cam is None:
                return no_update
            patch = Patch()
            patch["layout"]["scene"]["camera"] = cam
            return patch

        @app.callback(
            Output("scene", "figure", allow_duplicate=True),
            Input("pred_scene", "relayoutData"),
            _State("sync_cameras", "value"),
            prevent_initial_call=True,
        )
        def sync_gt_to_pred(pred_rl, sync_val):
            if not sync_val or "on" not in sync_val:
                return no_update
            cam = _camera_from_relayout(pred_rl)
            if cam is None:
                return no_update
            patch = Patch()
            patch["layout"]["scene"]["camera"] = cam
            return patch

    print(f"\nRunning on http://{args.host}:{args.port}\n")
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
