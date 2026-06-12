"""
Multi-event flash/slice viewer for the single-photon study, reading stage3pred +
merged_sp DIRECTLY (no perevent preprocessing). Predicts each slice's PMT pattern on
the fly via PhotonLib and overlays it on the observed in-time beam flash, alongside a
3D view of the predicted slices with the true photon highlighted.

Built for inspecting 1g+0X events: scan an event list (e.g. inspect_1g0X.csv), see for
each whether the photon's slice is mislabeled cosmic and whether its flash prediction
best matches the in-time beam flash (the flash-recovery idea).

  python view_1g0X_flash.py \
     --event-list workdir_scale/inspect_1g0X.csv \
     --merged-dir /cluster/.../merged_sp [--gamma 5.25] [--port 8053]

Runs in the pointcept container (GPU recommended for fast on-the-fly prediction).
"""
import argparse
import csv
import glob
import os
import sys

import dash
from dash import Input, Output, State, dcc, html
import h5py
import numpy as np
import plotly.graph_objects as go

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, REPO)
from lartpc_data_prep.larformer_analysis.lib.flash_predict import (   # noqa: E402
    predict_many_slices_pe, select_charge_y_with_uv_fallback_np)
from lartpc_data_prep.larformer_analysis.lib.flash_chi2 import neyman_chi2  # noqa: E402

PHOTON_PDG = 22
NU_CLASS = 0
VOX = 1.0
MIN_PTS = 20
TPC = ((0.0, 256.0), (-116.5, 116.5), (0.0, 1036.0))
F_SYS, EPS = 0.10, 1.0
CLSNAME = {0: "nu", 1: "cosmic", 2: "no_obj"}

_CACHE = {}   # stage3pred path -> loaded dict


def _vox(pts):
    return [tuple(v) for v in np.floor(pts / VOX).astype(np.int64)]


def _oob_frac(pts):
    if len(pts) == 0:
        return 1.0
    inb = ((pts[:, 0] >= TPC[0][0]) & (pts[:, 0] <= TPC[0][1]) &
           (pts[:, 1] >= TPC[1][0]) & (pts[:, 1] <= TPC[1][1]) &
           (pts[:, 2] >= TPC[2][0]) & (pts[:, 2] <= TPC[2][1]))
    return float(1.0 - inb.mean())


def load_event(s3path, merged_path):
    if s3path in _CACHE:
        return _CACHE[s3path]
    with h5py.File(s3path, "r") as ph5:
        a = ph5.attrs
        run, sub, evt = (int(a.get("meta_run", -1)), int(a.get("meta_subrun", -1)),
                         int(a.get("meta_event", -1)))
        post = ph5["post"]["coord"][:].astype(np.float32)
        pq = ph5["post"]["pred_query"][:].astype(np.int64)
        qcls = ph5["queries"]["class_argmax"][:].astype(np.int64)
    with h5py.File(merged_path, "r") as mh5:
        e = mh5["entry_0"]
        pos = e["triplet_data"]["pos"][:].astype(np.float32)
        charge = select_charge_y_with_uv_fallback_np(e["triplet_data"]["pixval"][:])
        sf = e["shower_fragments"]; spid = sf["pid"][:]; stid = sf["trackid"][:]
        scnt = sf["pointindices_counts"][:]; sflat = sf["pointindices_flat"][:]
        soff = np.concatenate([[0], np.cumsum(scnt)])
        mt = e["mc_particle_tree"]
        tid2E = {int(t): float(en) for t, en in zip(mt["trackid"][:], mt["energy_mev"][:])}
        fl = e["flashes"]
        f_pe = fl["pe"][:]; f_pid = fl["producer_id"][:]
        f_tpe = fl["total_pe"][:]; f_t = fl["time_us"][:]
    # photon truth points (largest nu photon fragment)
    photon_idx = None; photon_E = -1.0; photon_tid = -1
    for t in np.unique(stid[spid == PHOTON_PDG]):
        fr = np.where((stid == t) & (spid == PHOTON_PDG))[0]
        idx = np.unique(np.concatenate([sflat[soff[i]:soff[i + 1]] for i in fr]))
        if len(idx) > (0 if photon_idx is None else len(photon_idx)):
            photon_idx = idx; photon_E = tid2E.get(int(t), -1.0); photon_tid = int(t)
    photon_pts = pos[photon_idx] if photon_idx is not None else np.zeros((0, 3), np.float32)

    # in-time beam flash
    beam = np.where(f_pid == 0)[0]
    if len(beam):
        bi = beam[int(np.argmax(f_tpe[beam]))]
        pe_obs = f_pe[bi].astype(np.float64); t0 = float(f_t[bi])
    else:
        pe_obs = np.zeros(f_pe.shape[1] if f_pe.ndim == 2 else 32); t0 = 0.0

    # vox -> charge (mean)
    vc = {}
    for k, q in zip(_vox(pos), charge):
        vc.setdefault(k, []).append(q)
    vc = {k: float(np.mean(v)) for k, v in vc.items()}

    # photon dominant slice
    postvox = {}
    for k, q in zip(_vox(post), pq):
        postvox.setdefault(k, []).append(int(q))
    from collections import Counter
    postvox = {k: Counter(v).most_common(1)[0][0] for k, v in postvox.items()}
    ph_sl = [postvox[k] for k in _vox(photon_pts) if k in postvox]
    photon_slice = Counter(ph_sl).most_common(1)[0][0] if ph_sl else -1

    # per-slice predictions (gamma=1; scale later)
    counts = Counter(int(q) for q in pq)
    slices = sorted([s for s, c in counts.items() if c >= MIN_PTS])
    if photon_slice >= 0 and photon_slice not in slices:
        slices.append(photon_slice)
    sid = {s: i for i, s in enumerate(slices)}
    sel = np.array([int(q) in sid for q in pq])
    spos = post[sel]
    scharge = np.array([vc.get(k, 0.0) for k in _vox(spos)], dtype=np.float32)
    scid = np.array([sid[int(q)] for q in pq[sel]], dtype=np.int64)
    if len(spos):
        pe_pred = predict_many_slices_pe(spos, scharge, scid, len(slices), t0,
                                         producer_id=0, gamma_by_producer=(1.0, 1.0))
    else:
        pe_pred = np.zeros((len(slices), len(pe_obs)), np.float32)

    sl_info = []
    for s in slices:
        m = pq == s
        sl_info.append(dict(query=s, cls=int(qcls[s]) if s < len(qcls) else -1,
                            n_sp=int(m.sum()), oob=_oob_frac(post[m]),
                            pe_pred=pe_pred[sid[s]].astype(np.float64),
                            is_photon=(s == photon_slice),
                            photon_frac=(np.isin([postvox.get(k, -1) for k in _vox(photon_pts)], s).mean()
                                         if len(photon_pts) else 0.0)))
    d = dict(run=run, sub=sub, evt=evt, pe_obs=pe_obs, t0=t0,
             post=post, pq=pq, slices=sl_info, photon_slice=photon_slice,
             photon_pts=photon_pts, photon_E=photon_E, photon_tid=photon_tid)
    _CACHE[s3path] = d
    return d


def chi2(pe_obs, pe_pred):
    var = pe_obs + (F_SYS * pe_obs) ** 2 + EPS
    return float(((pe_obs - pe_pred) ** 2 / var).sum())


def flash_fig(d, sel_q, gamma, log_y):
    pe_obs = d["pe_obs"]; pmt = np.arange(len(pe_obs))
    info = {s["query"]: s for s in d["slices"]}
    chis = {s["query"]: chi2(pe_obs, gamma * s["pe_pred"]) for s in d["slices"]
            if s["oob"] <= 0.2}
    min_q = min(chis, key=chis.get) if chis else None
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=pmt, y=pe_obs, mode="lines+markers", name=f"observed ΣPE={pe_obs.sum():.0f}",
                             line=dict(color="white", width=2)))
    if sel_q in info:
        sp = gamma * info[sel_q]["pe_pred"]
        fig.add_trace(go.Scatter(x=pmt, y=sp, mode="lines+markers",
                                 name=f"sel slice q={sel_q} ({CLSNAME.get(info[sel_q]['cls'],'?')}) "
                                      f"ΣPE={sp.sum():.0f} χ²={chi2(pe_obs,sp):.0f}",
                                 line=dict(color="royalblue", width=2)))
    if d["photon_slice"] in info:
        pp = gamma * info[d["photon_slice"]]["pe_pred"]
        fig.add_trace(go.Scatter(x=pmt, y=pp, mode="lines+markers",
                                 name=f"PHOTON slice q={d['photon_slice']} ΣPE={pp.sum():.0f} χ²={chi2(pe_obs,pp):.0f}",
                                 line=dict(color="gold", width=2, dash="dot")))
    if min_q is not None and min_q != sel_q:
        mp = gamma * info[min_q]["pe_pred"]
        fig.add_trace(go.Scatter(x=pmt, y=mp, mode="lines+markers",
                                 name=f"min-χ² slice q={min_q} χ²={chis[min_q]:.0f}",
                                 line=dict(color="darkorange", width=2)))
    fig.update_layout(template="plotly_dark", height=420,
                      xaxis=dict(title="PMT id", dtick=2),
                      yaxis=dict(title="PE", type="log" if log_y else "linear"),
                      legend=dict(orientation="h", y=-0.25, font=dict(size=10)),
                      margin=dict(l=50, r=10, t=40, b=40),
                      title=f"({d['run']},{d['sub']},{d['evt']})  photon E={d['photon_E']:.0f} MeV  "
                            f"photon_slice={d['photon_slice']} "
                            f"({'BEST flash match' if min_q==d['photon_slice'] else 'rank>1'})  γ={gamma:.3g}")
    return fig, min_q


def view3d(d, sel_q):
    fig = go.Figure()
    # subsample post points for speed, color by slice membership category
    post = d["post"]; pq = d["pq"]
    step = max(1, len(post) // 30000)
    p = post[::step]; q = pq[::step]
    photon_q = d["photon_slice"]
    col = np.where(q == photon_q, "gold", np.where(q == sel_q, "royalblue", "#555"))
    fig.add_trace(go.Scatter3d(x=p[:, 2], y=p[:, 0], z=p[:, 1], mode="markers",
                               marker=dict(size=1.3, color=col), name="slices", hoverinfo="skip"))
    if len(d["photon_pts"]):
        pt = d["photon_pts"]
        fig.add_trace(go.Scatter3d(x=pt[:, 2], y=pt[:, 0], z=pt[:, 1], mode="markers",
                                   marker=dict(size=2.2, color="red"), name="true photon"))
    fig.update_layout(template="plotly_dark", height=520,
                      scene=dict(xaxis_title="z", yaxis_title="x", zaxis_title="y",
                                 aspectmode="data"),
                      margin=dict(l=0, r=0, t=24, b=0),
                      title="gold=photon slice  blue=selected slice  red=true photon pts")
    return fig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--event-list", required=True)
    ap.add_argument("--merged-dir", required=True)
    ap.add_argument("--gamma", type=float, default=5.25)
    ap.add_argument("--port", type=int, default=8053)
    args = ap.parse_args()

    events = list(csv.DictReader(open(args.event_list)))
    merged = {os.path.basename(p): p for p in
              glob.glob(os.path.join(args.merged_dir, "**", "merged_*entry*.h5"), recursive=True)}

    def evlabel(i, r):
        return (f"[{i}] ({r['run']},{r['subrun']},{r['event']}) E={float(r['lead_photon_E']):.0f} "
                f"{r['slice_category']} recov={r['flash_recovered']} reco={r['reco_1g0X']}")
    ev_opts = [{"label": evlabel(i, r), "value": i} for i, r in enumerate(events)]

    app = dash.Dash(__name__)
    app.title = "1g0X flash viewer"
    app.layout = html.Div([
        html.H3("1γ+0X flash / slice viewer", style={"color": "white"}),
        html.Div([
            html.Button("◀ prev", id="prev", n_clicks=0),
            html.Button("next ▶", id="next", n_clicks=0, style={"marginLeft": "6px"}),
            dcc.Dropdown(id="ev", options=ev_opts, value=0, clearable=False,
                         style={"width": "720px", "color": "black", "display": "inline-block",
                                "marginLeft": "10px", "verticalAlign": "middle"}),
            html.Label("γ:", style={"color": "white", "margin": "0 6px 0 16px"}),
            dcc.Input(id="gamma", type="number", value=args.gamma, step=0.25, style={"width": "80px"}),
            dcc.Checklist(id="logy", options=[{"label": " log-y", "value": "log"}], value=[],
                          style={"display": "inline-block", "color": "white", "marginLeft": "12px"}),
        ], style={"marginBottom": "10px"}),
        dcc.Dropdown(id="slice", options=[], value=None, clearable=False,
                     style={"width": "420px", "color": "black", "marginBottom": "8px"}),
        html.Div([
            html.Div(dcc.Graph(id="flash"), style={"width": "49%", "display": "inline-block"}),
            html.Div(dcc.Graph(id="g3d"), style={"width": "50%", "display": "inline-block"}),
        ]),
        html.Pre(id="slicetab", style={"color": "white", "fontSize": "12px"}),
    ], style={"backgroundColor": "#222", "padding": "14px", "minHeight": "100vh"})

    @app.callback(Output("ev", "value"),
                  Input("prev", "n_clicks"), Input("next", "n_clicks"), State("ev", "value"))
    def _nav(p, n, cur):
        return max(0, min(len(events) - 1, (cur or 0) + n - p))

    @app.callback(Output("slice", "options"), Output("slice", "value"),
                  Input("ev", "value"))
    def _slices(i):
        r = events[i]
        mp = merged.get(os.path.basename(r["stage3pred_path"])[len("stage3pred_"):])
        d = load_event(r["stage3pred_path"], mp)
        opts = [{"label": f"q={s['query']} {CLSNAME.get(s['cls'],'?')} n={s['n_sp']} "
                          f"OOB={s['oob']:.2f}{' [PHOTON]' if s['is_photon'] else ''}",
                 "value": s["query"]} for s in sorted(d["slices"], key=lambda s: -s["n_sp"])]
        return opts, d["photon_slice"]

    @app.callback(Output("flash", "figure"), Output("g3d", "figure"), Output("slicetab", "children"),
                  Input("ev", "value"), Input("slice", "value"), Input("gamma", "value"), Input("logy", "value"))
    def _upd(i, sel_q, gamma, logy):
        r = events[i]; mp = merged.get(os.path.basename(r["stage3pred_path"])[len("stage3pred_"):])
        d = load_event(r["stage3pred_path"], mp)
        g = float(gamma) if gamma else 5.25
        if sel_q is None:
            sel_q = d["photon_slice"]
        ff, min_q = flash_fig(d, int(sel_q), g, "log" in (logy or []))
        f3 = view3d(d, int(sel_q))
        lines = ["q   class    n_sp  OOB   ΣPE_pred(γ)   χ²(γ)  photon?"]
        for s in sorted(d["slices"], key=lambda s: chi2(d["pe_obs"], g * s["pe_pred"])):
            lines.append("%-4d %-7s %5d %5.2f %12.1f %8.1f   %s%s" % (
                s["query"], CLSNAME.get(s["cls"], "?"), s["n_sp"], s["oob"],
                (g * s["pe_pred"]).sum(), chi2(d["pe_obs"], g * s["pe_pred"]),
                "Y" if s["is_photon"] else " ", " <min-χ²" if s["query"] == min_q else ""))
        return ff, f3, "\n".join(lines[:18])

    print(f"Listening on http://0.0.0.0:{args.port}  ({len(events)} events)")
    app.run(debug=False, port=args.port, host="0.0.0.0")


if __name__ == "__main__":
    main()
