"""Side-by-side 3D scatter visualizer for the Stage 3 particle GT.

Renders two views of the same event's nu-origin spacepoints:

  Left  panel — colored by RAW Geant4 trackid (one color per trackid).
                Shows the unmerged structure: every sub-threshold
                δ-ray / brem / conversion electron has its own color.

  Right panel — colored by the constructed particle GT slice id
                (= compute_particle_labels output, with KE-threshold
                merging applied).

Color matching: the right-panel color of a surviving above-threshold
host is propagated to every trackid that got MERGED INTO it. So if you
see "this orange dot in the left view is part of the green cluster on
the right" — that's the merge in action.

Cosmic SPs (origin != nu_origin) are drawn in light gray for context;
their color is unchanged between panels (they're never assigned to any
particle GT instance).

Output: a single self-contained HTML file with two synchronized 3D
scatters (rotate/zoom either panel; legend toggles individual instances).

Usage:
  ./run_in_container.sh python tools/viz/view_particle_gt.py \\
      --merged-h5 /path/to/merged_*.h5 \\
      [--output /tmp/particle_gt.html]
      [--no-cosmic]                 # hide cosmic SPs
      [--max-sp 20000]              # downsample for fast load
"""

import argparse
import os
import sys

import h5py
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO_ROOT)

from lartpc.data_prep.labels.slice_labels import (  # noqa: E402
    GHOST_SLICE_ID,
    compute_particle_labels,
)


PDG_NAMES = {
    11: "e-",   -11: "e+",
    13: "mu-",  -13: "mu+",
    22: "gamma",
    111: "pi0", 211: "pi+", -211: "pi-",
    2212: "p",  2112: "n",
    321:  "K+", -321: "K-",
}


def _pdg(p):
    return PDG_NAMES.get(int(p), f"PDG:{int(p)}")


# Plotly's `Alphabet` qualitative palette has 26 distinct colors; we
# cycle through several palettes when there are more instances.
_PALETTES = [
    "Alphabet", "Light24", "Bold", "Vivid", "Dark24",
]


def _make_color_map(unique_ids, palettes=_PALETTES):
    """Return {id -> rgb hex string} cycling through Plotly palettes."""
    import plotly.express as px
    pool = []
    for name in palettes:
        pool.extend(getattr(px.colors.qualitative, name))
    return {int(i): pool[k % len(pool)] for k, i in enumerate(unique_ids)}


def load_event(merged_h5, nu_origin=1):
    """Pull everything needed by both panels into a flat dict."""
    with h5py.File(merged_h5, "r") as f:
        e0 = f["entry_0"]
        run    = int(e0.attrs.get("run", -1))
        subrun = int(e0.attrs.get("subrun", -1))
        event  = int(e0.attrs.get("event", -1))
        td = e0["triplet_data"]
        sp_pos      = td["pos"][:].astype(np.float32)
        sp_trackid  = td["trackid"][:].astype(np.int64)
        sp_origin   = td["origin"][:].astype(np.int64)
        sp_hasmatch = (td["hasmatch"][:].astype(np.int64)
                       if "hasmatch" in td else None)
        mpt = e0["mc_particle_tree"]
        particle_info = compute_particle_labels(
            mpt, sp_trackid, sp_hasmatch, nu_origin=nu_origin,
        )
        # For the LEFT panel "raw trackid" view we want each track's PDG
        # so we can label the legend entries. Build a trackid → pdg map.
        mc_tids = mpt["trackid"][:].astype(np.int64)
        mc_pids = mpt["pid"][:].astype(np.int64)
        mc_ke   = mpt["energy_mev"][:].astype(np.float32)
        tid_to_pid = {int(t): int(p) for t, p in zip(mc_tids, mc_pids)}
        tid_to_ke  = {int(t): float(k) for t, k in zip(mc_tids, mc_ke)}
    return dict(
        run=run, subrun=subrun, event=event,
        sp_pos=sp_pos, sp_trackid=sp_trackid,
        sp_origin=sp_origin, sp_hasmatch=sp_hasmatch,
        particle_info=particle_info,
        tid_to_pid=tid_to_pid, tid_to_ke=tid_to_ke,
        nu_origin=int(nu_origin),
    )


def build_figure(d, hide_cosmic=False, max_sp=None):
    """Build the side-by-side 3D scatter figure."""
    sp_pos      = d["sp_pos"]
    sp_trackid  = d["sp_trackid"]
    sp_origin   = d["sp_origin"]
    sp_hasmatch = d["sp_hasmatch"]
    nu_origin   = d["nu_origin"]
    pinfo       = d["particle_info"]

    # Real-SP mask (drop ghosts).
    real_mask = (sp_hasmatch != 0) if sp_hasmatch is not None else np.ones(
        sp_trackid.shape, dtype=bool,
    )
    # Origin-based partition (nu vs cosmic vs other).
    nu_mask  = real_mask & (sp_origin == nu_origin)
    cos_mask = real_mask & (sp_origin == 2)

    # Optional downsample (per-mask, preserving structure within each).
    if max_sp is not None and max_sp > 0:
        rng = np.random.default_rng(0)
        def _down(m):
            if m.sum() <= max_sp:
                return m
            idx = np.flatnonzero(m)
            keep = rng.choice(idx, size=max_sp, replace=False)
            out = np.zeros_like(m); out[keep] = True
            return out
        nu_mask  = _down(nu_mask)
        cos_mask = _down(cos_mask)

    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "scatter3d"}, {"type": "scatter3d"}]],
        subplot_titles=(
            "Left — raw Geant4 trackid (one color per track)",
            "Right — constructed particle GT slice id (KE-threshold merged)",
        ),
        horizontal_spacing=0.02,
    )

    # ----- LEFT: per-trackid -----
    nu_tids_in_use = np.unique(sp_trackid[nu_mask])
    tid_palette = _make_color_map(nu_tids_in_use)
    for tid in nu_tids_in_use:
        m = nu_mask & (sp_trackid == tid)
        if not m.any():
            continue
        pdg = d["tid_to_pid"].get(int(tid), 0)
        ke  = d["tid_to_ke"].get(int(tid), 0.0)
        n   = int(m.sum())
        fig.add_trace(
            go.Scatter3d(
                x=sp_pos[m, 0], y=sp_pos[m, 1], z=sp_pos[m, 2],
                mode="markers",
                marker=dict(size=1.5, color=tid_palette[int(tid)]),
                name=f"tid={int(tid)} {_pdg(pdg)} KE={ke:.1f} n={n}",
                legendgroup=f"left_{int(tid)}",
                showlegend=True,
                hovertemplate=(
                    f"tid=%{{customdata[0]}}<br>"
                    f"pdg={_pdg(pdg)}  KE={ke:.1f} MeV<br>"
                    "(x,y,z)=(%{x:.1f},%{y:.1f},%{z:.1f})<extra></extra>"
                ),
                customdata=np.full((n, 1), int(tid)),
            ),
            row=1, col=1,
        )

    # ----- RIGHT: per-particle-GT-slice -----
    # Build per-SP slice_id from particle_info.
    pgt = pinfo["slice_id"]
    surviving_ids = pinfo["primary_trackid"]
    slice_palette = _make_color_map(surviving_ids)
    # Each surviving particle → one trace, with all its SPs
    # (which include the merged-in sub-threshold trackids).
    for k, sid in enumerate(surviving_ids):
        m = nu_mask & (pgt == int(sid))
        if not m.any():
            continue
        members = pinfo["slice_member_trackids"][k]
        pdg = int(pinfo["primary_pid"][k])
        ke  = float(pinfo["primary_ke_MeV"][k])
        n   = int(m.sum())
        n_merged_extra = max(0, len(members) - 1)
        label = (f"qid={k} tid={int(sid)} {_pdg(pdg)} KE={ke:.1f} "
                 f"n={n}" + (f"  (+{n_merged_extra} merged)"
                              if n_merged_extra else ""))
        fig.add_trace(
            go.Scatter3d(
                x=sp_pos[m, 0], y=sp_pos[m, 1], z=sp_pos[m, 2],
                mode="markers",
                marker=dict(size=1.5, color=slice_palette[int(sid)]),
                name=label,
                legendgroup=f"right_{int(sid)}",
                showlegend=True,
                hovertemplate=(
                    f"slice_id=%{{customdata[0]}}<br>"
                    f"pdg={_pdg(pdg)}  KE={ke:.1f} MeV<br>"
                    f"n_merged_trackids={len(members)}<br>"
                    "(x,y,z)=(%{x:.1f},%{y:.1f},%{z:.1f})<extra></extra>"
                ),
                customdata=np.full((n, 1), int(sid)),
            ),
            row=1, col=2,
        )

    # nu SPs that ended up GHOST_SLICE_ID in the particle GT — they're
    # nu-origin tracks below threshold with no visible ancestor (e.g.
    # the π0 itself, low-KE neutrons, sub-threshold orphan secondaries).
    # Worth showing as gray-ish dots on the RIGHT so the user can see
    # what "got lost" by the merging policy.
    ghost_nu_mask = nu_mask & (pgt == GHOST_SLICE_ID)
    if ghost_nu_mask.any():
        n = int(ghost_nu_mask.sum())
        fig.add_trace(
            go.Scatter3d(
                x=sp_pos[ghost_nu_mask, 0],
                y=sp_pos[ghost_nu_mask, 1],
                z=sp_pos[ghost_nu_mask, 2],
                mode="markers",
                marker=dict(size=1.2, color="#888888", opacity=0.5),
                name=f"nu SPs with no visible host (n={n})",
                legendgroup="right_ghost",
                showlegend=True,
                hoverinfo="skip",
            ),
            row=1, col=2,
        )

    # ----- Cosmic SPs (light gray, on BOTH panels for context) -----
    if not hide_cosmic and cos_mask.any():
        for col_i in (1, 2):
            fig.add_trace(
                go.Scatter3d(
                    x=sp_pos[cos_mask, 0],
                    y=sp_pos[cos_mask, 1],
                    z=sp_pos[cos_mask, 2],
                    mode="markers",
                    marker=dict(size=1.0, color="#aaaaaa", opacity=0.25),
                    name=f"cosmic SPs (n={int(cos_mask.sum())})",
                    legendgroup="cosmic",
                    showlegend=(col_i == 1),
                    hoverinfo="skip",
                ),
                row=1, col=col_i,
            )

    # Shared axis ranges + dark theme.
    pos_for_range = sp_pos[real_mask]
    if len(pos_for_range) > 0:
        x_rng = [float(pos_for_range[:, 0].min()),
                 float(pos_for_range[:, 0].max())]
        y_rng = [float(pos_for_range[:, 1].min()),
                 float(pos_for_range[:, 1].max())]
        z_rng = [float(pos_for_range[:, 2].min()),
                 float(pos_for_range[:, 2].max())]
    else:
        x_rng = y_rng = z_rng = [0, 1]

    scene_kwargs = dict(
        xaxis=dict(title="x [cm]", range=x_rng, backgroundcolor="#222"),
        yaxis=dict(title="y [cm]", range=y_rng, backgroundcolor="#222"),
        zaxis=dict(title="z [cm]", range=z_rng, backgroundcolor="#222"),
        bgcolor="#222",
        aspectmode="data",
    )
    fig.update_layout(
        scene=scene_kwargs, scene2=scene_kwargs,
        template="plotly_dark",
        height=820,
        title=(
            f"Particle GT — event "
            f"({d['run']}, {d['subrun']}, {d['event']})  "
            f"particles={len(pinfo['primary_trackid'])}  "
            f"nu_SPs={int(nu_mask.sum())}  "
            f"cosmic_SPs={int(cos_mask.sum())}"
        ),
        legend=dict(
            x=1.02, y=1.0, xanchor="left", yanchor="top",
            font=dict(size=10),
            bgcolor="rgba(34,34,34,0.85)",
        ),
        margin=dict(l=10, r=200, t=60, b=10),
    )
    return fig


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--merged-h5", required=True,
                    help="Path to a merged_<...>_entry<NNNNNN>.h5")
    ap.add_argument("--output", default=None,
                    help="Output HTML path (default: same dir as merged-h5)")
    ap.add_argument("--no-cosmic", action="store_true",
                    help="Hide cosmic SPs (faster render)")
    ap.add_argument("--max-sp", type=int, default=None,
                    help="Per-panel SP cap (random downsample). Default: all.")
    ap.add_argument("--nu-origin", type=int, default=1,
                    help="Which origin value counts as 'nu-origin' (default 1)")
    args = ap.parse_args()

    if not os.path.exists(args.merged_h5):
        sys.exit(f"missing merged H5: {args.merged_h5}")
    d = load_event(args.merged_h5, nu_origin=args.nu_origin)
    fig = build_figure(d, hide_cosmic=args.no_cosmic, max_sp=args.max_sp)

    if args.output is None:
        base = os.path.splitext(os.path.basename(args.merged_h5))[0]
        out_path = os.path.join(os.path.dirname(args.merged_h5) or ".",
                                f"particle_gt_{base}.html")
    else:
        out_path = args.output
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".",
                exist_ok=True)
    fig.write_html(out_path, include_plotlyjs="cdn")
    print(f"wrote {out_path}")
    print(f"  N particles = {len(d['particle_info']['primary_trackid'])}")
    real = (d["sp_hasmatch"] != 0) if d["sp_hasmatch"] is not None else np.ones(
        d["sp_origin"].shape, dtype=bool,
    )
    n_nu_sp = int(((d["sp_origin"] == args.nu_origin) & real).sum())
    print(f"  N nu-origin SPs (real) = {n_nu_sp}")


if __name__ == "__main__":
    main()
