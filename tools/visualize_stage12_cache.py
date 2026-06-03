"""Visualize a single Stage-1+2 cache file produced by
`tools/build_stage12_cache_event.py` (or the shard driver).

Three side-by-side 3D Plotly panels sharing the TPC outline + initial
camera:

  Panel 1 — Cached SPs colored by `source_mask`
    - Distinct colors for each bitmask combination (stage2-only,
      GT-only, both, delta-only, ...). Legend tells the trainer
      exactly what the cache is offering.

  Panel 2 — Cached SPs colored by particle GT instance
    - One color per surviving particle (origin_type=0 always red for
      the "leading" particle if any). GT origins drawn as diamonds.
    - Legend lists pid, KE, n_truth_in_cache for each instance.

  Panel 3 — Cached SPs colored by `stage2_nu_mask_prob`
    - Continuous "Plasma" colormap. Helps eyeball where Stage 2 is
      confident vs uncertain.
    - Optionally rings around SPs that belong to a GT particle so
      false negatives jump out.

Usage:

    python tools/visualize_stage12_cache.py \\
        --cache /tmp/cache_v2_event0.h5 \\
        --output /tmp/cache_v2_event0.html
"""
from __future__ import annotations

import argparse
import colorsys
import os
import sys
import webbrowser

import h5py
import numpy as np

import plotly.graph_objects as go
from plotly.subplots import make_subplots

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from detectoroutline import DetectorOutline  # noqa: E402


# ---------------------------------------------------------------------------
# Color helpers (match tools/visualize_larformer_gt.py palette)
# ---------------------------------------------------------------------------

def instance_color(k: int, origin_type: int = -1, alpha: float = 1.0) -> str:
    if origin_type == 0 and k == 0:
        # only mark the first nu-origin instance as the "leading" red one
        return f"rgba(255,60,60,{alpha})"
    h = (k * 0.61803398875) % 1.0
    r, g, b = colorsys.hsv_to_rgb(h, 0.80, 0.95)
    return f"rgba({int(r*255)},{int(g*255)},{int(b*255)},{alpha})"


# 8-color palette for source_mask bitmask values (0..7).
SOURCE_MASK_COLORS = {
    0: "rgba(140,140,140,0.5)",   # neither — floor-only noise
    1: "rgba(80,160,255,0.95)",   # stage2 only (false positive)
    2: "rgba(255,210,40,0.95)",   # GT only (Stage-2 false negative)
    3: "rgba(60,200,80,0.95)",    # stage2 + GT (true positive at nominal)
    4: "rgba(120,80,200,0.5)",    # delta only — borderline
    5: "rgba(80,200,200,0.95)",   # delta + stage2  (= 1 implies 4)
    6: "rgba(255,140,40,0.95)",   # delta + GT (Stage-2 borderline truth)
    7: "rgba(220,40,200,0.95)",   # all three (delta+nominal+GT)
}

SOURCE_MASK_LABELS = {
    0: "floor-only (mask>τfloor, not GT)",
    1: "[impossible]",
    2: "GT-only (mask ≤ τ-δ)",
    3: "[impossible]",
    4: "δ-only (mask in (τ-δ, τ], not GT)",
    5: "stage2-pass NOT GT (false pos)",
    6: "δ-pass + GT (borderline truth)",
    7: "stage2-pass + GT (true pos)",
}


PDG_TO_NAME = {
    11: "e-", -11: "e+", 22: "γ",
    13: "μ-", -13: "μ+",
    211: "π+", -211: "π-",
    111: "π0",
    2212: "p", 2112: "n",
    321: "K+", -321: "K-",
    130: "K0L", 310: "K0S",
}


def pdg_name(pdg: int) -> str:
    if pdg in PDG_TO_NAME:
        return PDG_TO_NAME[pdg]
    if pdg > 1_000_000_000:
        return "nuclear"
    return f"pdg={pdg}"


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def _read_instances(grp: h5py.Group) -> list[dict]:
    out: list[dict] = []
    keys = sorted(grp.keys(),
                  key=lambda s: int(s.split("_")[-1])
                  if s.startswith("instance_") else -1)
    for k in keys:
        if not k.startswith("instance_"):
            continue
        g = grp[k]
        d: dict = {}
        for ak, av in g.attrs.items():
            d[ak] = av.item() if hasattr(av, "item") else av
        for dk in g.keys():
            d[dk] = g[dk][...]
        out.append(d)
    return out


def load_cache(path: str) -> dict:
    with h5py.File(path, "r") as f:
        def _attr(v):
            if hasattr(v, "item") and getattr(v, "size", 1) == 1:
                return v.item()
            return v
        top = {k: _attr(v) for k, v in f.attrs.items()}
        if "entry_0" not in f:
            top_groups = sorted(f.keys())
            # v1 schema marker — the deprecated `cache_stage12_for_event.py`
            # wrote `raw/`, `slicer/`, `stage3/`, `gt/`. That builder has
            # been removed; rebuild with build_stage12_cache_event.py.
            hint = ""
            if "raw" in f and "stage3" in f:
                hint = (
                    " — this looks like the deprecated v1 cache schema "
                    "(raw/+slicer/+stage3/+gt/). Rebuild with "
                    "`tools/build_stage12_cache_event.py` and rerun the "
                    "visualizer."
                )
            raise KeyError(
                f"{path}: no 'entry_0' group (got top-level groups: "
                f"{top_groups}){hint}"
            )
        e0 = f["entry_0"]
        per_sp = {k: e0[k][...]
                  for k in e0.keys()
                  if not isinstance(e0[k], h5py.Group)}
        particles = (_read_instances(e0["particle_instances"])
                     if "particle_instances" in e0 else [])
        slicer = {}
        if "slicer" in e0:
            sg = e0["slicer"]
            slicer = {k: sg[k][...] for k in sg.keys()}
            slicer["_attrs"] = {k: _attr(v) for k, v in sg.attrs.items()}
    return dict(top=top, per_sp=per_sp, particles=particles, slicer=slicer)


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def _detector_lines_trace() -> go.Scatter3d:
    raw = DetectorOutline().getlines(color=(60, 60, 60))[0]
    return go.Scatter3d(
        x=raw["x"], y=raw["y"], z=raw["z"],
        mode="lines",
        line=dict(color="rgb(60,60,60)", width=3),
        name="TPC outline", showlegend=False, hoverinfo="skip",
    )


def _denorm_coord(norm_xyz: np.ndarray, top_attrs: dict) -> np.ndarray:
    center = np.asarray(top_attrs.get("coord_center", (125.0, 0.0, 518.0)),
                        dtype=np.float32)
    scale = float(top_attrs.get("coord_scale", 179.55))
    return norm_xyz * scale + center


def _scatter(coord, *, color, size=2.0, name=None,
             hovertemplate=None, showlegend=True, line=None,
             legend_group=None) -> go.Scatter3d:
    marker = dict(size=size, color=color, opacity=1.0)
    if line is not None:
        marker["line"] = line
    return go.Scatter3d(
        x=coord[:, 0], y=coord[:, 1], z=coord[:, 2],
        mode="markers", marker=marker,
        name=name or "", showlegend=showlegend,
        hovertemplate=hovertemplate, legendgroup=legend_group,
    )


def _origin_diamond(xyz, *, color, name,
                    legend_group=None, showlegend=False) -> go.Scatter3d:
    return go.Scatter3d(
        x=[float(xyz[0])], y=[float(xyz[1])], z=[float(xyz[2])],
        mode="markers",
        marker=dict(size=10, color=color, symbol="diamond",
                    line=dict(color="black", width=2)),
        name=name, legendgroup=legend_group, showlegend=showlegend,
        hovertemplate=f"{name}<extra></extra>",
    )


# ---------------------------------------------------------------------------
# Build figure
# ---------------------------------------------------------------------------

def build_figure(cache: dict) -> go.Figure:
    top = cache["top"]
    per_sp = cache["per_sp"]
    particles = cache["particles"]
    coord = per_sp["coord"]
    n_cache = coord.shape[0]
    source_mask = per_sp["source_mask"].astype(np.uint8)
    stage2_prob = per_sp["stage2_nu_mask_prob"]
    is_gt_nu = (source_mask & 2).astype(bool)

    # ---- Subplots / titles ----
    titles = [
        f"<b>Panel 1 — by source_mask</b><br>"
        f"<sub>{n_cache:,} cached SPs &nbsp; "
        f"τ_nominal={top.get('tau_loose_nominal'):.2f} &nbsp; "
        f"τ_floor={top.get('tau_loose_floor'):.2f}</sub>",
        f"<b>Panel 2 — by particle GT</b><br>"
        f"<sub>{len(particles)} particle instances &nbsp; "
        f"{int(is_gt_nu.sum()):,} GT-nu SPs in cache</sub>",
        f"<b>Panel 3 — by stage2 nu mask prob</b><br>"
        f"<sub>τ_nominal pass: {top.get('n_passes_tau_loose')}/"
        f"{n_cache:,} &nbsp; GT-rings on Stage-2 misses</sub>",
    ]
    fig = make_subplots(
        rows=1, cols=3,
        specs=[[{"type": "scatter3d"}] * 3],
        subplot_titles=titles,
        horizontal_spacing=0.015,
    )

    # ---- Panel 1: source_mask ----
    fig.add_trace(_detector_lines_trace(), row=1, col=1)
    # One trace per source_mask value so the legend lets the viewer
    # toggle each provenance set on/off.
    for val in sorted(np.unique(source_mask).tolist()):
        idx = np.where(source_mask == val)[0]
        if idx.size == 0:
            continue
        color = SOURCE_MASK_COLORS.get(int(val), "rgba(0,0,0,1)")
        name = f"sm={val} {SOURCE_MASK_LABELS.get(int(val), '?')} (n={idx.size})"
        size = 1.6 if val in (0, 4) else 2.6   # de-emphasize "noise" buckets
        fig.add_trace(_scatter(
            coord[idx], color=color, size=size, name=name,
            hovertemplate=name + "<extra></extra>",
        ), row=1, col=1)

    # ---- Panel 2: particle GT ----
    fig.add_trace(_detector_lines_trace(), row=1, col=2)
    # Background: SPs not in any particle instance (false positives / noise).
    in_any_instance = np.zeros(n_cache, dtype=bool)
    for inst in particles:
        ti = inst["truth_indices"].astype(np.int64)
        in_any_instance[ti] = True
    bg_idx = np.where(~in_any_instance)[0]
    if bg_idx.size > 0:
        fig.add_trace(_scatter(
            coord[bg_idx], color="rgba(180,180,180,0.5)", size=1.4,
            name=f"no GT (n={bg_idx.size})",
            hovertemplate=f"no GT<extra></extra>",
        ), row=1, col=2)
    # Per-instance.
    for k, inst in enumerate(particles):
        ti = inst["truth_indices"].astype(np.int64)
        if ti.size == 0:
            continue
        pid = int(inst.get("pid", 0))
        cls = int(inst.get("class_id", -1))
        ke = float(inst.get("ke_mev", 0.0))
        n_o = int(inst.get("n_truth_points_orig", ti.size))
        # origin_type=0 → nu; first such instance gets red.
        color = instance_color(k, int(inst.get("origin_type", -1)))
        label = (f"GT[{k}] {pdg_name(pid)} (cls={cls}, KE={ke:.0f}MeV, "
                 f"n={ti.size}/{n_o})")
        fig.add_trace(_scatter(
            coord[ti], color=color, size=2.8, name=label,
            hovertemplate=label + "<extra></extra>",
            legend_group=f"gt_{k}",
        ), row=1, col=2)
        if "origin_cm" in inst:
            xyz = np.asarray(inst["origin_cm"], dtype=np.float32)
        elif "origin_coord_norm" in inst:
            xyz = _denorm_coord(
                np.asarray(inst["origin_coord_norm"], dtype=np.float32), top,
            )
        else:
            continue
        fig.add_trace(_origin_diamond(
            xyz, color=color, name=f"origin GT[{k}]",
            legend_group=f"gt_{k}",
        ), row=1, col=2)

    # ---- Panel 3: stage2_nu_mask_prob ----
    fig.add_trace(_detector_lines_trace(), row=1, col=3)
    fig.add_trace(go.Scatter3d(
        x=coord[:, 0], y=coord[:, 1], z=coord[:, 2],
        mode="markers",
        marker=dict(
            size=2.0, color=stage2_prob, cmin=0.0, cmax=1.0,
            colorscale="Plasma",
            colorbar=dict(title="stage2_nu<br>mask_prob",
                          x=0.99, len=0.55, y=0.55),
            opacity=1.0,
        ),
        name="cached SPs",
        hovertemplate="stage2_nu_mask_prob=%{marker.color:.3f}<extra></extra>",
        showlegend=False,
    ), row=1, col=3)
    # Ring around GT-nu SPs that Stage 2 missed (source_mask = GT but
    # NOT delta — i.e. bit 1 set, bit 2 not set, bit 0 not set). Those
    # are the false negatives. Highlighting them surfaces "events where
    # Stage 2 needs help".
    fn_mask = (source_mask == 2)
    if fn_mask.any():
        fig.add_trace(_scatter(
            coord[fn_mask], color="rgba(255,255,255,0.0)", size=4.5,
            name=f"Stage-2 false negatives (n={int(fn_mask.sum())})",
            line=dict(color="cyan", width=1),
            hovertemplate="false negative<extra></extra>",
        ), row=1, col=3)

    # ---- Camera + layout ----
    camera = dict(eye=dict(x=1.7, y=1.3, z=1.0),
                  up=dict(x=0, y=1, z=0),
                  center=dict(x=0, y=0, z=0))
    scene = dict(
        xaxis=dict(title="X (cm)", range=[-10, 270]),
        yaxis=dict(title="Y (cm)", range=[-125, 125]),
        zaxis=dict(title="Z (cm)", range=[-10, 1050]),
        aspectmode="data",
        camera=camera,
    )
    title = (
        f"<b>Stage-1+2 cache:</b> {top.get('source_h5', '?')} &nbsp;|&nbsp; "
        f"run {top.get('run', -1)} subrun {top.get('subrun', -1)} "
        f"event {top.get('event', -1)} &nbsp;|&nbsp; "
        f"raw {top.get('n_after_dataset_filter', 0):,} → "
        f"postdh {top.get('n_after_deghost', 0):,} → "
        f"cache {top.get('n_in_cache', 0):,} &nbsp;|&nbsp; "
        f"GT-nu in cache {top.get('n_gt_nu_in_cache', 0):,} "
        f"({top.get('n_particle_instances', 0)} instances)"
    )
    fig.update_layout(
        scene=scene, scene2=scene, scene3=scene,
        title=title,
        margin=dict(l=0, r=0, t=120, b=0),
        legend=dict(orientation="v", x=1.02, y=1.0, font=dict(size=10)),
        height=780,
    )
    return fig


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cache", required=True,
                   help="Cache .h5 from build_stage12_cache_event.py.")
    p.add_argument("--output", default=None,
                   help="HTML output (default: alongside the cache).")
    p.add_argument("--browser", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    out_path = (args.output if args.output
                else os.path.splitext(args.cache)[0] + ".html")
    print(f"Loading cache: {args.cache}")
    cache = load_cache(args.cache)
    top = cache["top"]
    print(f"  source_h5     : {top.get('source_h5')}")
    print(f"  cache SPs     : {top.get('n_in_cache')}")
    print(f"  GT particles  : {top.get('n_particle_instances')}")
    print(f"  GT-nu SPs     : {top.get('n_gt_nu_in_cache')}")
    print(f"  pass-nominal  : {top.get('n_passes_tau_loose')}")
    print(f"Building figure ...")
    fig = build_figure(cache)
    print(f"Writing HTML: {out_path}")
    fig.write_html(out_path, include_plotlyjs="cdn",
                   full_html=True, auto_open=False)
    print(f"Wrote {os.path.getsize(out_path)/1024**2:.2f} MiB")
    if args.browser:
        webbrowser.open("file://" + os.path.abspath(out_path))


if __name__ == "__main__":
    main()
