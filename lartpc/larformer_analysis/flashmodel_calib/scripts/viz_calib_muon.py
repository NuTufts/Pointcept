"""Inspect the single-muon calibration events behind a gamma fit.

For each selected muon the page shows:
  * 3D: the full event cloud (grey), the nu-union slice (blue), the MUON POINTS
    that made the prediction (coloured by comb charge), start/end markers, and
    the 32 PMT discs coloured by the OBSERVED in-time flash;
  * PMT layout (z vs y): observed PE (filled) vs predicted PE at the sample's
    fitted scale (open circles); dead / saturated tubes marked;
  * per-opdet bars: observed, predicted at gamma_ref, predicted x s;
  * the three wire-plane images (wire vs tick, sparse pixels, grey) with the
    nu-union pixels in blue and the muon's pixels in red, so completeness of
    the muon reconstruction can be judged by eye.

Selection = exactly the protocol cuts of fit_gamma.py (same flags), so what
you browse is what entered the fit. `--all-candidates` shows failing ones too
(the hover/info block says which cut they failed).

    PYTHONPATH=$K python3 scripts/viz_calib_muon.py --sample bnb5e19_cew6 \
        --n 10 --out-dir results/s1ep2p8cew6/viz            # HTML files
    PYTHONPATH=$K python3 scripts/viz_calib_muon.py --sample bnb5e19_cew6 --serve --port 8050
"""
import glob
import json
import os
import sys

import numpy as np
import h5py
import plotly.graph_objects as go
from plotly.subplots import make_subplots

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..", "..", "..")))
sys.path.insert(0, _HERE)

import fit_gamma as FG  # noqa: E402
from lartpc.larformer_analysis.flashmodel_calib.gammacal import (  # noqa: E402
    CALIB_DIR, GAMMA_BEAM_REF, samples)
from lartpc.larformer_analysis.flashmodel_calib.gammacal.records import load_records  # noqa: E402
from lartpc.larformer_analysis.flashmodel_calib.gammacal.flash import in_window_ok  # noqa: E402
from lartpc.flashmatch.saturation import _OPDET_POS  # noqa: E402
from lartpc.viz.detector import DetectorOutline  # noqa: E402

PMT = np.asarray(_OPDET_POS, np.float64)
PLANE_NAMES = ("U (plane 0)", "V (plane 1)", "Y (plane 2)")


# ---------------------------------------------------------------------------
# selection
# ---------------------------------------------------------------------------
def select(ev, mu, args, window):
    """Passing muons (index array) + per-candidate reason of first failed cut."""
    m, drops, aux = FG.select_muons(ev, mu, args, window)
    return np.nonzero(m)[0], aux, drops


def failure_reason(ev, mu, j, args, window):
    """First protocol cut this candidate fails ('' if it passes)."""
    i = int(mu["ev"][j])
    if not (ev["has_flash"][i] and in_window_ok(ev["flash_time_us"][i], ev["flash_times"][i],
                                                ev["flash_pes"][i], window,
                                                margin=args.window_margin,
                                                min_pe=args.flash_min_pe)):
        return "in-window single flash"
    if args.require_mu_class and mu["pdg"][j] != 13:
        return f"class {mu['cls'][j]} (not muon)"
    if args.rms_perp_max < 1e8 and "rms_perp" in mu and not (
            mu["rms_perp"][j] < args.rms_perp_max and mu["lin"][j] > args.lin_min):
        return f"not track-like (rms_perp {mu['rms_perp'][j]:.1f}, lin {mu['lin'][j]:.2f})"
    if mu["length"][j] <= args.min_len:
        return f"length <= {args.min_len:g}"
    if args.calib_object == "cluster" and mu["orphan"][j] != 2:
        return "not the cluster object"
    if args.calib_object == "instance" and mu["orphan"][j] == 2:
        return "cluster object (instances requested)"
    if args.n_boundary >= 0 and mu["n_boundary"][j] != args.n_boundary:
        return f"n_boundary = {mu['n_boundary'][j]}"
    if args.n_boundary < 0 and mu["n_boundary"][j] < 1:
        return "no boundary end"
    bxm = mu["boundary_x_min"][j] if "boundary_x_min" in mu else mu["boundary_end_x"][j]
    if args.n_boundary != 0 and not (bxm > args.x_boundary_min):
        return f"boundary end x = {bxm:.0f}"
    if not args.allow_secondary and mu["is_secondary"][j] > 0:
        return "secondary"
    if mu["iso_track_ke_max"][j] >= args.iso_track_ke:
        return f"other track KE {mu['iso_track_ke_max'][j]:.0f}"
    if mu["iso_shower_e_max"][j] >= args.iso_shower_e:
        return f"shower E {mu['iso_shower_e_max'][j]:.0f}"
    if mu["proton_ke_max"][j] >= args.proton_veto_ke:
        return f"proton KE {mu['proton_ke_max'][j]:.0f}"
    qf = mu["q_mu"][j] / max(ev["q_union"][i], 1e-9)
    if qf < args.q_frac_min:
        return f"q_frac {qf:.2f}"
    live = ev["live"][i]
    if mu["pred_mu_ref"][j][live].sum() < args.min_pred:
        return "pred < min"
    if mu["cos_sim"][j] < args.cos_min:
        return f"cos {mu['cos_sim'][j]:.2f}"
    if not (mu["dcy"][j] < args.match_dy and mu["dcz"][j] < args.match_dz):
        return f"centroid dy {mu['dcy'][j]:.0f} dz {mu['dcz'][j]:.0f}"
    return ""


# ---------------------------------------------------------------------------
# event loading
# ---------------------------------------------------------------------------
def load_muon_event(ev, mu, j):
    i = int(mu["ev"][j])
    out = dict(i=i, j=j)
    with h5py.File(ev["kp2_path"][i], "r") as kp:
        out["slice"] = kp["slice/coord_cm"][()].astype(np.float32)
        inst = int(mu["inst"][j])
        out["pidx"] = (kp[f"particle/{inst}/point_idx"][()].astype(np.int64) if inst >= 0
                       else np.arange(out["slice"].shape[0], dtype=np.int64))
        out["stored_pred_nu"] = ev["pred_nu_file"][i]
    with h5py.File(ev["msp_path"][i], "r") as f:
        e = f["entry_0"]
        td = e["triplet_data"]
        pos = td["pos"][()].astype(np.float32)
        out["all_pos"] = pos
        out["all_tick"] = td["tick"][()].astype(np.int64)
        out["all_wires"] = [td[k][()].astype(np.int64) for k in ("uwire", "vwire", "ywire")]
        out["all_pixval"] = td["pixval"][()].astype(np.float32)
        row = {pos[k].tobytes(): k for k in range(len(pos))}
        out["slice_rows"] = np.asarray([row.get(out["slice"][k].tobytes(), -1)
                                        for k in range(len(out["slice"]))], np.int64)
        out["images"] = []
        for p in range(3):
            g = e[f"image_data/plane{p}"]
            c = g["coord"][()]; org = g["origin"][()]; ps = g["pixsize"][()]
            out["images"].append(dict(wire=c[:, 0] * ps[0] + org[0],
                                      tick=c[:, 1] * ps[1] + org[1],
                                      feat=g["feat"][()].astype(np.float32)))
    return out


def _disc_traces(pe, label, radius=15.0):
    """Filled PMT discs in 3D coloured by PE (plotly axes: x=z, y=x, z=y)."""
    pe = np.nan_to_num(np.asarray(pe, float))
    pmax = pe.max() if pe.max() > 0 else 1.0
    th = np.linspace(0, 2 * np.pi, 25)
    traces = []
    for k in range(32):
        v = float(pe[k] / pmax)
        ys = PMT[k, 1] + radius * np.cos(th); zs = PMT[k, 2] + radius * np.sin(th)
        xs = np.full(th.size, PMT[k, 0])
        n = th.size
        r = int(40 + 215 * v); g = int(30 + 140 * v ** 1.5); b = 40
        traces.append(go.Mesh3d(
            x=np.append(zs, PMT[k, 2]), y=np.append(xs, PMT[k, 0]),
            z=np.append(ys, PMT[k, 1]), i=[n] * n, j=list(range(n)),
            k=[(m + 1) % n for m in range(n)], color=f"rgb({r},{g},{b})",
            opacity=0.45 + 0.5 * v, hoverinfo="text",
            text=f"opdet {k}: {label} {pe[k]:.0f} PE", showlegend=False))
    return traces


def _outline():
    do = DetectorOutline()
    tr = []
    for line in (do.top_pts, do.bot_pts):
        tr.append(go.Scatter3d(x=[p[2] for p in line], y=[p[0] for p in line],
                               z=[p[1] for p in line], mode="lines",
                               line=dict(color="rgba(120,120,120,0.5)", width=2),
                               showlegend=False, hoverinfo="skip"))
    for a, b in zip(do.top_pts[:4], do.bot_pts[:4]):
        tr.append(go.Scatter3d(x=[a[2], b[2]], y=[a[0], b[0]], z=[a[1], b[1]], mode="lines",
                               line=dict(color="rgba(120,120,120,0.5)", width=2),
                               showlegend=False, hoverinfo="skip"))
    return tr


# ---------------------------------------------------------------------------
# figure
# ---------------------------------------------------------------------------
def figure(sample, ev, mu, j, data, s_fit, reason="", cap=40000):
    i = data["i"]
    obs = np.clip(ev["obs_pe"][i], 0, None); live = ev["live"][i]
    pred = np.clip(mu["pred_mu_ref"][j], 0, None)
    o_l, p_l = obs[live].sum(), pred[live].sum()
    r = o_l / p_l if p_l > 0 else np.nan
    q_union = ev["q_union"][i]
    fig = make_subplots(
        rows=3, cols=3,
        specs=[[{"type": "scene", "colspan": 2}, None, {"type": "xy"}],
               [{"type": "xy", "colspan": 3}, None, None],
               [{"type": "xy"}, {"type": "xy"}, {"type": "xy"}]],
        row_heights=[0.46, 0.18, 0.36], column_widths=[0.4, 0.3, 0.3],
        subplot_titles=("3D: muon points (colour = comb charge), stage-3 slice (blue), event (grey), "
                        "PMTs coloured by observed PE",
                        "PMT layout: observed (filled) vs predicted x s (open)",
                        "per-opdet PE: observed vs predicted",
                        *PLANE_NAMES),
        horizontal_spacing=0.05, vertical_spacing=0.08)
    # ---- 3D
    allp = data["all_pos"]
    rng = np.random.default_rng(0)
    sel = rng.choice(len(allp), min(cap, len(allp)), replace=False)
    fig.add_trace(go.Scatter3d(x=allp[sel, 2], y=allp[sel, 0], z=allp[sel, 1], mode="markers",
                               marker=dict(size=1.6, color="rgba(70,70,70,0.6)"),
                               name="event", hoverinfo="skip"), row=1, col=1)
    sl = data["slice"]
    slice_name = ("calib cluster" if str(ev.get("stream", [""] * (i + 1))[i]) == "calib"
                  else "nu union")
    fig.add_trace(go.Scatter3d(x=sl[:, 2], y=sl[:, 0], z=sl[:, 1], mode="markers",
                               marker=dict(size=1.6, color="rgba(60,110,220,0.45)"),
                               name=f"{slice_name} ({len(sl)} pts)", hoverinfo="skip"), row=1, col=1)
    mp = sl[data["pidx"]]
    rows = data["slice_rows"][data["pidx"]]
    q = np.where(rows >= 0, data["all_pixval"][np.maximum(rows, 0), 2], 0.0)
    fig.add_trace(go.Scatter3d(x=mp[:, 2], y=mp[:, 0], z=mp[:, 1], mode="markers",
                               marker=dict(size=2.6, color=q, colorscale="Hot", cmin=0,
                                           cmax=max(np.percentile(q, 95), 1),
                                           colorbar=dict(title="Y pixval", len=0.35, y=0.82, x=0.62)),
                               name=f"muon points ({len(mp)})",
                               hovertext=[f"q={v:.0f}" for v in q], hoverinfo="text"), row=1, col=1)
    st, en = mu["start"][j], mu["end"][j]
    fig.add_trace(go.Scatter3d(x=[st[2]], y=[st[0]], z=[st[1]], mode="markers",
                               marker=dict(size=6, color="lime", symbol="diamond"),
                               name="track start"), row=1, col=1)
    if np.isfinite(en).all():
        fig.add_trace(go.Scatter3d(x=[en[2]], y=[en[0]], z=[en[1]], mode="markers",
                                   marker=dict(size=6, color="magenta", symbol="x"),
                                   name="track end"), row=1, col=1)
    for t in _outline() + _disc_traces(obs, "observed"):
        fig.add_trace(t, row=1, col=1)
    fig.update_scenes(xaxis_title="z [cm]", yaxis_title="x [cm]", zaxis_title="y [cm]",
                      aspectmode="data", row=1, col=1)
    # ---- PMT layout (z vs y)
    ps = pred * s_fit
    size_o = 6 + 40 * np.sqrt(obs / max(obs.max(), 1))
    size_p = 6 + 40 * np.sqrt(ps / max(obs.max(), 1))
    fig.add_trace(go.Scatter(x=PMT[:, 2], y=PMT[:, 1], mode="markers",
                             marker=dict(size=size_o, color="rgba(31,119,180,0.55)",
                                         line=dict(width=0)),
                             name="observed", hovertext=[f"opdet {k}: obs {obs[k]:.0f}" for k in range(32)],
                             hoverinfo="text"), row=1, col=3)
    fig.add_trace(go.Scatter(x=PMT[:, 2], y=PMT[:, 1], mode="markers",
                             marker=dict(size=size_p, color="rgba(0,0,0,0)",
                                         line=dict(width=2, color="#d62728")),
                             name=f"predicted x s={s_fit:.3f}",
                             hovertext=[f"opdet {k}: pred x s {ps[k]:.0f}" for k in range(32)],
                             hoverinfo="text"), row=1, col=3)
    dead = np.nonzero(~live)[0]
    if len(dead):
        fig.add_trace(go.Scatter(x=PMT[dead, 2], y=PMT[dead, 1], mode="markers",
                                 marker=dict(size=14, symbol="x", color="black"),
                                 name="masked (dead/saturated)"), row=1, col=3)
    fig.add_trace(go.Scatter(x=PMT[:, 2], y=PMT[:, 1], mode="text", text=[str(k) for k in range(32)],
                             textfont=dict(size=8), showlegend=False, hoverinfo="skip"), row=1, col=3)
    fig.update_xaxes(title_text="z [cm]", row=1, col=3)
    fig.update_yaxes(title_text="y [cm]", row=1, col=3, scaleanchor="x3", scaleratio=1)
    # ---- bars
    k = np.arange(32)
    fig.add_trace(go.Bar(x=k, y=obs, name=f"observed (live sum {o_l:.0f})", marker_color="#1f77b4",
                         opacity=0.85), row=2, col=1)
    fig.add_trace(go.Bar(x=k, y=pred, name=f"pred @ gamma_ref (live sum {p_l:.0f})",
                         marker_color="#ff7f0e", opacity=0.5), row=2, col=1)
    fig.add_trace(go.Bar(x=k, y=ps, name=f"pred x s (live sum {p_l * s_fit:.0f})",
                         marker_color="#d62728", opacity=0.5), row=2, col=1)
    fig.update_layout(barmode="overlay")
    fig.update_xaxes(title_text="opdet", row=2, col=1)
    fig.update_yaxes(title_text="PE", row=2, col=1)
    # ---- wire planes
    union_rows = data["slice_rows"][data["slice_rows"] >= 0]
    for p in range(3):
        im = data["images"][p]
        fig.add_trace(go.Scattergl(x=im["wire"], y=im["tick"], mode="markers",
                                   marker=dict(size=2, color=np.log10(1 + im["feat"]),
                                               colorscale="Greys", cmin=0, cmax=2.5, showscale=False),
                                   name="image", showlegend=(p == 0), hoverinfo="skip"), row=3, col=p + 1)
        w = data["all_wires"][p]; t = data["all_tick"]
        fig.add_trace(go.Scattergl(x=w[union_rows], y=t[union_rows], mode="markers",
                                   marker=dict(size=2.5, color="rgba(60,110,220,0.5)"),
                                   name=f"{slice_name} pixels", showlegend=(p == 0), hoverinfo="skip"),
                      row=3, col=p + 1)
        mr = rows[rows >= 0]
        fig.add_trace(go.Scattergl(x=w[mr], y=t[mr], mode="markers",
                                   marker=dict(size=3, color="red"),
                                   name="muon pixels", showlegend=(p == 0), hoverinfo="skip"),
                      row=3, col=p + 1)
        fig.update_xaxes(title_text="wire", row=3, col=p + 1)
        fig.update_yaxes(title_text="tick", row=3, col=p + 1)
    # ---- info
    win = sample["flash_window_us"]
    info = (f"<b>{sample['tag']}</b> ({sample['kind']}, period {sample['period']})  gidx {ev['gidx'][i]}  "
            f"RSE {ev['run'][i]}/{ev['subrun'][i]}/{ev['event'][i]}  inst {mu['inst'][j]}"
            f"{' (whole cluster)' if mu['orphan'][j] == 2 else (' (vertex-less)' if mu['orphan'][j] else '')}<br>"
            f"muon: L={mu['length'][j]:.0f} cm, KE={mu['ke'][j]:.0f} MeV, n_boundary={mu['n_boundary'][j]}, "
            f"boundary x={mu['boundary_end_x'][j]:.0f}, npts={mu['npts'][j]}, cls={mu['cls'][j]}, "
            f"rms_perp={mu['rms_perp'][j]:.1f} cm, lin={mu['lin'][j]:.3f}, q_mu={mu['q_mu'][j]:.0f} "
            f"({mu['q_mu'][j] / max(q_union, 1e-9):.2f} of union {q_union:.0f}); "
            f"other track KE {mu['iso_track_ke_max'][j]:.0f}, shower E {mu['iso_shower_e_max'][j]:.0f}<br>"
            f"flash: t={ev['flash_time_us'][i]:.2f} us (window {win[0]}-{win[1]}), total {ev['flash_total_pe'][i]:.0f} PE, "
            f"masked {ev['dead_file'][i] or '-'}/{ev['sat_file'][i] or '-'}; "
            f"<b>obs/pred_ref = {r:.3f}</b> (sample s = {s_fit:.3f}), cos = {mu['cos_sim'][j]:.3f}, "
            f"centroid dy={mu['dcy'][j]:.0f} dz={mu['dcz'][j]:.0f}"
            + (f"<br><span style='color:red'>FAILS: {reason}</span>" if reason else ""))
    fig.update_layout(height=1350, width=1500, template="plotly_white",
                      title=dict(text=info, font=dict(size=12), x=0.01, xanchor="left"),
                      margin=dict(t=130, l=40, r=20, b=40),
                      legend=dict(orientation="h", y=-0.03, font=dict(size=10)))
    return fig


def sample_scale(sample):
    p = os.path.join(CALIB_DIR, "results", sample["chain"], f"{sample['tag']}__muon.json")
    if os.path.exists(p):
        j = json.load(open(p))
        if j.get("scale_abs") is not None:
            return float(j["scale_abs"])
    return 1.0


def main():
    ap = FG.build_parser()
    ap.add_argument("--out-dir", default=None, help="write one HTML per muon here")
    ap.add_argument("--n", type=int, default=10, help="how many muons to write")
    ap.add_argument("--start-index", type=int, default=0, help="offset into the selected list")
    ap.add_argument("--gidx", type=int, nargs="*", default=None, help="only these event indices")
    ap.add_argument("--all-candidates", action="store_true",
                    help="include candidates that fail the cuts (reason shown)")
    ap.add_argument("--serve", action="store_true", help="Dash browser instead of HTML files")
    ap.add_argument("--port", type=int, default=8050)
    ap.add_argument("--s-fit", type=float, default=None,
                    help="scale to draw 'pred x s' with (default: the sample's "
                         "<tag>__muon.json result, else 1.0)")
    args = ap.parse_args()
    s = samples.get(args.sample)
    paths = sorted(glob.glob(args.records or os.path.join(
        CALIB_DIR, "results", s["chain"], "records", f"{s['tag']}*.npz")))
    ev, mu, meta = load_records(paths)
    window = (tuple(float(x) for x in args.window.split(",")) if args.window
              else tuple(s["flash_window_us"]))
    s_fit = args.s_fit if args.s_fit is not None else sample_scale(s)
    passing, aux, drops = select(ev, mu, args, window)
    if args.all_candidates:
        cand = np.arange(len(mu["ev"]))
    else:
        cand = passing
    if args.gidx:
        want = set(args.gidx)
        cand = np.array([j for j in cand if int(ev["gidx"][mu["ev"][j]]) in want], int)
    print(f">>> {s['tag']}: {len(passing)} muons pass, {len(cand)} selected for display, s_fit={s_fit:.3f}")
    if not len(cand):
        return
    passing_set = set(passing.tolist())

    def build(j):
        reason = "" if j in passing_set else failure_reason(ev, mu, j, args, window)
        data = load_muon_event(ev, mu, j)
        return figure(s, ev, mu, j, data, s_fit, reason=reason)

    if args.serve:
        from dash import Dash, Input, Output, dcc, html
        app = Dash(__name__)
        opts = [{"label": f"[{n}] gidx {ev['gidx'][mu['ev'][j]]} RSE {ev['run'][mu['ev'][j]]}/"
                          f"{ev['subrun'][mu['ev'][j]]}/{ev['event'][mu['ev'][j]]} L={mu['length'][j]:.0f} "
                          f"cos={mu['cos_sim'][j]:.2f}", "value": int(j)} for n, j in enumerate(cand)]
        app.layout = html.Div([
            html.Div([dcc.Dropdown(id="pick", options=opts, value=int(cand[0]), style={"width": "700px"}),
                      html.Button("prev", id="prev"), html.Button("next", id="next")],
                     style={"display": "flex", "gap": "8px"}),
            dcc.Graph(id="fig")])
        order = [int(j) for j in cand]

        @app.callback(Output("pick", "value"), Input("prev", "n_clicks"), Input("next", "n_clicks"),
                      Input("pick", "value"))
        def step(p, n, cur):
            from dash import callback_context as cc
            trig = cc.triggered[0]["prop_id"].split(".")[0] if cc.triggered else ""
            k = order.index(cur) if cur in order else 0
            if trig == "prev":
                k = max(k - 1, 0)
            elif trig == "next":
                k = min(k + 1, len(order) - 1)
            return order[k]

        @app.callback(Output("fig", "figure"), Input("pick", "value"))
        def show(j):
            return build(int(j))
        app.run(host="0.0.0.0", port=args.port, debug=False)
        return

    out_dir = args.out_dir or os.path.join(CALIB_DIR, "results", s["chain"], "viz", s["tag"])
    os.makedirs(out_dir, exist_ok=True)
    for j in cand[args.start_index:args.start_index + args.n]:
        i = int(mu["ev"][j])
        fig = build(int(j))
        name = f"muon_{s['tag']}_gidx{ev['gidx'][i]:06d}_rse{ev['run'][i]}_{ev['subrun'][i]}_{ev['event'][i]}.html"
        fig.write_html(os.path.join(out_dir, name), include_plotlyjs="cdn")
        print(f"   -> {os.path.join(out_dir, name)}")


if __name__ == "__main__":
    main()
