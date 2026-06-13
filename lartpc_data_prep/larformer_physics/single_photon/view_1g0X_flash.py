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

# Cascade spacepoint displays (same as tools/visualize_full_cascade.py).
import atexit                                                          # noqa: E402
import tempfile                                                       # noqa: E402
from pointcept.models.LArFormer.inference import load_event_h5        # noqa: E402
from pointcept.models.LArFormer.viz_inference import (                # noqa: E402
    figure_for_cascade_slices, figure_for_stage3_prediction,
    figure_for_cascade_gt, cascade_nu_color_options)

_VIZ_CACHE = {}   # stage3pred path -> (pred_dict, gt_dict)


def _compute_gt(merged_path):
    """Build the GT dict for figure_for_cascade_gt from a merged_h5 (reads it
    twice through LArFormerDataset for slice + particle GT). None if no MC truth.
    Replicated from tools/visualize_full_cascade.py."""
    from pointcept.datasets.larformer import LArFormerDataset
    with h5py.File(merged_path, "r") as f:
        if "entry_0" not in f or "mc_particle_tree" not in f["entry_0"]:
            return None
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".txt",
                                      prefix="viz1g_gt_", delete=False)
    tmp.write(os.path.abspath(merged_path) + "\n"); tmp.close()
    atexit.register(lambda p=tmp.name: os.path.exists(p) and os.unlink(p))

    def _ds(src):
        # lm_score_val_threshold=0 keeps EVERY true spacepoint — including the
        # low-larmatch-score nu points the deghoster drops — so the GT panel
        # shows the WHOLE neutrino-induced point cloud, not the deghosted subset.
        return LArFormerDataset(split="test", data_root="/",
                                data_list_file=tmp.name, gt_source=src,
                                lm_score_val_threshold=0.0)[0]
    try:
        s_slice = _ds("slice"); s_part = _ds("particle")
    except Exception as e:
        print(f"[viz] GT compute failed for {merged_path}: {e}")
        return None
    coord = np.asarray(s_slice["coord"], np.float32)
    slice_id = np.asarray(s_slice["slice_id"], np.int64)
    slice_origin = {int(g["primary_trackid"]): int(g.get("origin_type", 1))
                    for g in s_slice.get("gt_instances", [])}
    is_nu = np.array([slice_origin.get(int(t), 1) == 0 for t in slice_id], bool)
    part_coord = np.asarray(s_part["coord"], np.float32)
    part_cls = np.asarray(s_part["particle_class_id"], np.int64)
    part_inst = np.full(len(part_coord), -1, np.int64)
    for k, g in enumerate(s_part.get("gt_instances", [])):
        ti = np.asarray(g["truth_indices"], np.int64)
        if ti.size:
            part_inst[ti] = k
    nu_mask = part_cls >= 0
    return dict(coord=coord, slice_id=slice_id, slice_is_nu=is_nu,
                slice_origin=slice_origin,
                hasmatch=np.asarray(s_slice["hasmatch"], np.int64),
                nu_coord=part_coord[nu_mask], nu_class=part_cls[nu_mask],
                nu_instance=part_inst[nu_mask])


def load_viz(s3path, merged_path):
    """(pred_dict, gt_dict) for the cascade displays, cached per stage3pred."""
    if s3path in _VIZ_CACHE:
        return _VIZ_CACHE[s3path]
    pred = load_event_h5(s3path)
    gt = _compute_gt(merged_path)
    _VIZ_CACHE[s3path] = (pred, gt)
    return pred, gt

PHOTON_PDG = 22
NU_CLASS = 0
VOX = 1.0
MIN_PTS = 20
TPC = ((0.0, 256.0), (-116.5, 116.5), (0.0, 1036.0))
F_SYS, EPS = 0.10, 1.0
CLSNAME = {0: "nu", 1: "cosmic", 2: "no_obj"}
OOB_MAX = 0.05        # slices with > this OOB fraction can't match the in-time flash
PHOTON_FRAC = 0.5     # a slice is a "photon slice" if >= this frac of its pts are true photon

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

    # mark which POST points are true-photon (voxel matches a photon merged point),
    # so photon-fraction of a slice is consistent post-vs-post (always <= 1).
    from collections import Counter
    photon_voxset = set(_vox(photon_pts))
    post_is_photon = np.array([k in photon_voxset for k in _vox(post)], dtype=bool)
    photon_total_post = int(post_is_photon.sum())
    ph_by_slice = Counter(int(q) for q, ip in zip(pq, post_is_photon) if ip)
    photon_slice = ph_by_slice.most_common(1)[0][0] if ph_by_slice else -1  # dominant (ref)

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
        nsp = int(m.sum()); nph = int(post_is_photon[m].sum())
        frac = nph / max(nsp, 1)            # fraction of THIS slice that is true photon (<=1)
        sl_info.append(dict(query=s, cls=int(qcls[s]) if s < len(qcls) else -1,
                            n_sp=nsp, oob=_oob_frac(post[m]),
                            pe_pred=pe_pred[sid[s]].astype(np.float64),
                            photon_frac=frac,
                            photon_cov=nph / max(photon_total_post, 1),  # frac of photon in this slice
                            is_photon=(frac >= PHOTON_FRAC)))            # multiple allowed
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
    photon_qs = {s["query"] for s in d["slices"] if s["is_photon"]}   # >=1 allowed
    chis = {s["query"]: chi2(pe_obs, gamma * s["pe_pred"]) for s in d["slices"]
            if s["oob"] <= OOB_MAX}                                   # OOB<=0.05
    min_q = min(chis, key=chis.get) if chis else None
    best_is_photon = (min_q in photon_qs)
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
                      title=f"({d['run']},{d['sub']},{d['evt']})  γ={gamma:.3g}  "
                            f"photon E={d['photon_E']:.0f} MeV  "
                            f"photon slices(OOB≤{OOB_MAX:g})={sorted(photon_qs)}  "
                            f"best-match q={min_q} -> "
                            f"{'RECOVERED (photon slice)' if best_is_photon else 'NOT a photon slice'}")
    return fig, min_q, photon_qs



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--event-list", required=True)
    ap.add_argument("--merged-dir", default=None,
                    help="dir with merged_sp H5; auto-derived from the event list's "
                         "stage3pred paths (sibling merged_sp/) when omitted.")
    ap.add_argument("--gamma", type=float, default=5.25)
    ap.add_argument("--port", type=int, default=8053)
    args = ap.parse_args()

    events = list(csv.DictReader(open(args.event_list)))
    if not events:
        sys.exit("ERROR: empty event list %s" % args.event_list)

    # Auto-derive merged-dir from the stage3pred path (DATADIR/stage3pred/... ->
    # DATADIR/merged_sp) when not given or when the given one has no files.
    merged_dir = args.merged_dir
    if not merged_dir or not glob.glob(os.path.join(merged_dir, "**", "merged_*entry*.h5"),
                                       recursive=True):
        datadir = os.path.dirname(os.path.dirname(events[0]["stage3pred_path"]))
        cand = os.path.join(datadir, "merged_sp")
        if glob.glob(os.path.join(cand, "**", "merged_*entry*.h5"), recursive=True):
            print("auto-derived --merged-dir = %s" % cand)
            merged_dir = cand
    merged = {os.path.basename(p): p for p in
              glob.glob(os.path.join(merged_dir or "", "**", "merged_*entry*.h5"),
                        recursive=True)}
    if not merged:
        sys.exit("ERROR: no merged_sp H5 found (merged-dir=%r). Pass --merged-dir "
                 "explicitly (export the path; a 'VAR=x python' prefix does NOT expand "
                 "$VAR in the same line)." % merged_dir)
    print("indexed %d merged_sp H5" % len(merged))

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
            html.Label("nu color:", style={"color": "white", "margin": "0 6px 0 16px"}),
            dcc.Dropdown(id="nu_color", options=cascade_nu_color_options(), value="instance",
                         clearable=False, style={"width": "150px", "color": "black",
                                                 "display": "inline-block", "verticalAlign": "middle"}),
        ], style={"marginBottom": "10px"}),
        dcc.Dropdown(id="slice", options=[], value=None, clearable=False,
                     style={"width": "420px", "color": "black", "marginBottom": "8px"}),
        html.Div([
            html.Div(dcc.Graph(id="flash"), style={"width": "56%", "display": "inline-block",
                                                   "verticalAlign": "top"}),
            html.Div(html.Pre(id="slicetab", style={"color": "white", "fontSize": "11px"}),
                     style={"width": "43%", "display": "inline-block", "verticalAlign": "top"}),
        ]),
        html.Div("STAGE 2 — slicer output only (post-deghost SPs by predicted slice; "
                 "nu-candidate slice in red)",
                 style={"fontWeight": "bold", "color": "#9cf", "marginTop": "6px"}),
        dcc.Graph(id="slice_scene", style={"height": "46vh"}),
        html.Div("STAGE 3 — particle-segmenter output only (on the processed nu-candidate slice)",
                 style={"fontWeight": "bold", "color": "#fc9"}),
        dcc.Graph(id="part_scene", style={"height": "46vh"}),
        html.Div("GROUND TRUTH  (full nu point cloud — deghoster-dropped true SPs included)",
                 style={"fontWeight": "bold", "color": "#9fc"}),
        dcc.Graph(id="gt_scene", style={"height": "46vh"}),
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

    @app.callback(Output("flash", "figure"), Output("slice_scene", "figure"),
                  Output("part_scene", "figure"), Output("gt_scene", "figure"),
                  Output("slicetab", "children"),
                  Input("ev", "value"), Input("slice", "value"), Input("gamma", "value"),
                  Input("logy", "value"), Input("nu_color", "value"))
    def _upd(i, sel_q, gamma, logy, nu_color):
        r = events[i]; s3 = r["stage3pred_path"]
        mp = merged.get(os.path.basename(s3)[len("stage3pred_"):])
        d = load_event(s3, mp)
        g = float(gamma) if gamma else 5.25
        if sel_q is None:
            sel_q = d["photon_slice"]
        ff, min_q, photon_qs = flash_fig(d, int(sel_q), g, "log" in (logy or []))
        # three stacked spacepoint displays: stage-2 slicer, stage-3 particles, GT.
        pred, gt = load_viz(s3, mp)
        suffix = f"  ({d['run']},{d['sub']},{d['evt']})  photon E={d['photon_E']:.0f} MeV"
        slice_fig = figure_for_cascade_slices(pred, title_suffix=suffix)
        part_fig = figure_for_stage3_prediction(
            pred, color_by=("pred_query" if nu_color == "instance" else "pred_class"),
            title_suffix=suffix)
        gt_fig = figure_for_cascade_gt(gt, nu_color_by=nu_color, title_suffix=suffix)
        lines = ["q   class    n_sp  OOB   ph_frac ph_cov  ΣPE_pred(γ)   χ²(γ)  photon?"]
        for s in sorted(d["slices"], key=lambda s: chi2(d["pe_obs"], g * s["pe_pred"])):
            oob_rej = s["oob"] > OOB_MAX
            lines.append("%-4d %-7s %5d %5.2f %6.2f %6.2f %12.1f %8s   %s%s" % (
                s["query"], CLSNAME.get(s["cls"], "?"), s["n_sp"], s["oob"],
                s["photon_frac"], s["photon_cov"], (g * s["pe_pred"]).sum(),
                "OOB-rej" if oob_rej else "%.0f" % chi2(d["pe_obs"], g * s["pe_pred"]),
                "Y" if s["is_photon"] else " ", " <min-χ²" if s["query"] == min_q else ""))
        return ff, slice_fig, part_fig, gt_fig, "\n".join(lines[:20])

    print(f"Listening on http://0.0.0.0:{args.port}  ({len(events)} events)")
    app.run(debug=False, port=args.port, host="0.0.0.0")


if __name__ == "__main__":
    main()
