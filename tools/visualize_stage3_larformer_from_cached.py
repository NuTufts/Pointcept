"""LArFormer Stage-3 GT visualizer — reads a Stage-1+2 cache and shows
per-level truth labels as the trainer / model would see them.

This is the Stage-3 sibling of `tools/visualize_larformer_gt.py`. It
reuses that tool's level-building + figure-construction code path
(`build_event_gt`, `figure_for_event`, `metadata_panel`) directly, so
the visualizer can't drift from the Stage-3 training pipeline. No
parallel re-implementation of tokenization or per-level GT lifting.

What changes from the Stage-2 GT viz:

  - The dataset is a `LArFormerStage12CacheDataset` reading per-event
    HDF5 caches produced by `tools/build_stage12_cache_event.py` /
    `_shard.py`. The cascade's deghoster + slicer filter has already
    been applied; this tool does NOT re-run them.
  - The Stage-3 config (`larformer-particle-v1-cached.py`) declares
    Stage 3's level pyramid (16 cm / 8 cm / spacepoint by default)
    and the 7-class particle taxonomy. Both are read from the config
    and threaded into the same `CompositeTokenizer` + `build_per_level_gt`
    the trainer uses.
  - The deghoster slider / cascade-config flag are gone (cache is
    already filtered).
  - A `source_set_filter` dropdown lets you flip between the cached
    SP subsets (`stage2_pass`, `gt_nu`, `union`, ...) live, so you can
    see how the GT renders against each training-time SP set.

Usage:

    # Single event from a local file
    python tools/visualize_stage3_larformer_from_cached.py \\
        --config configs/lartpc/larformer-particle-v1-cached.py \\
        --cache /tmp/stage12_cache_v2/val/000/0000/event_*.h5

    # A directory of cached events (the shard driver's 3-level hash)
    python tools/visualize_stage3_larformer_from_cached.py \\
        --config configs/lartpc/larformer-particle-v1-cached.py \\
        --cache /tmp/stage12_cache_v2/val

Once running, open http://<host>:8050 in a browser.
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

    # ---- Dash app ----------------------------------------------------
    app = Dash(__name__)
    app.title = "LArFormer Stage-3 GT visualizer (from cache)"
    app.layout = html.Div([
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
                html.Span("Color by: ", style={"marginRight": "6px"}),
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
                # source_set_filter override (lets the user flip between
                # cached subsets without restarting the tool).
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
            ], style={"marginBottom": "8px"}),
        ]),
        dcc.Graph(id="scene", style={"height": "78vh"}),
        html.Div(id="metadata",
                 style={"fontFamily": "monospace",
                        "fontSize": "12px",
                        "padding": "10px",
                        "borderTop": "1px solid #ccc"}),
    ])

    @app.callback(
        Output("scene", "figure"),
        Output("metadata", "children"),
        Input("entry", "value"),
        Input("level", "value"),
        Input("color", "value"),
        Input("src", "value"),
        Input("reload", "n_clicks"),
    )
    def update(entry_val, level_val, color_val, src_val, _n):
        idx = int(entry_val or 0)
        idx = max(0, min(idx, n_events - 1))
        # Mutate the dataset's source_set_filter live so the next
        # __getitem__ uses it. Cheap — the dataset's get_data just
        # rereads the filter attr.
        if src_val and src_val != dataset.source_set_filter:
            dataset.source_set_filter = src_val
        # Same code path the Stage-2 GT viz uses, with deghoster_filter
        # disabled because the cache is already deghoster-filtered.
        event_data = build_event_gt(
            model_levels_cfg=levels_cfg, token_dim=token_dim,
            dataset=dataset, idx=idx,
            deghoster_filter=None, tau=0.5,
        )
        fig = figure_for_event(event_data, level_val, color_val)
        meta = metadata_panel(event_data, prediction=None)
        return fig, meta

    print(f"\nRunning on http://{args.host}:{args.port}\n")
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
