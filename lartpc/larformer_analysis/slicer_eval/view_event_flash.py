"""Per-event predicted-vs-observed flash viewer (Dash + Plotly).

Takes one `perevent_<run>_<sub>_<evt>.h5` from `analyze_event.py` and
serves an interactive viewer that overlays — per PMT — four PE histograms
on the same axis:

  1. observed in-time flash PE                              (pe_obs)
  2. predicted PE from the user-selected slice              (× γ)
  3. predicted PE from the GT-nu mask (truth-side baseline) (× γ)
  4. predicted PE from the slice with the min χ² under the current γ
     (= the slice flash-matching would pick if you ignored class)

Controls:
  - slice selector dropdown (sorted by IoU vs GT-nu, with annotations)
  - γ slider / input (applies to all predicted curves AND the χ² recompute)
  - OOB-threshold dropdown (which slices are considered "surviving")
  - log/linear y-axis toggle

A side panel lists per-slice metrics at the current γ: query_id, class,
n_sp, IoU vs GT-nu, OOB fraction, χ² (recomputed from stored pe_pred ×
γ), Σ PE_pred.

Usage:
  python view_event_flash.py \\
      --perevent-h5 /path/to/perevent_<run>_<sub>_<evt>.h5 \\
      [--gamma 225]      # default starting γ
      [--port 8052]

Requires: h5py, numpy, dash, plotly — already in the pointcept_cuml
container.
"""

import argparse
import os
import sys

import dash
from dash import Input, Output, dcc, html
import h5py
import numpy as np
import plotly.graph_objects as go


REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)
sys.path.insert(0, REPO_ROOT)
from lartpc.larformer_analysis.slicer_eval.lib import categorize  # noqa: E402


# Same Neyman-floor defaults as analyze_event.py.
DEFAULT_F_SYS = 0.10
DEFAULT_EPS   = 1.0

_MODEL_CLASS_NAMES = ["nu", "cosmic", "no_object"]


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def load_perevent(perevent_h5):
    """Pull everything we need into a flat dict; do NOT keep the file open."""
    with h5py.File(perevent_h5, "r") as f:
        out = dict(
            run    = int(f.attrs["run"]),
            subrun = int(f.attrs["subrun"]),
            event  = int(f.attrs["event"]),
            model_tag = str(f.attrs.get("model_tag", "?")),
            oob_thresholds = np.asarray(
                f.attrs["oob_thresholds"], dtype=np.float32),
            default_oob_idx = int(f.attrs["default_oob_idx"]),
            no_object_class_id = int(f.attrs.get("no_object_class_id", 2)),
            nu_class_id        = int(f.attrs.get("nu_class_id", 0)),
            has_nu_prediction  = bool(f.attrs.get("has_nu_prediction", False)),
            category_mask      = int(f["truth"].attrs.get("category_mask", 0)),
        )
        # In-time flash
        it = f["in_time_flash"]
        out["pe_obs"]       = it["pe_obs"][:].astype(np.float64)
        out["flash_t0_us"]  = float(it.attrs["t0_us"])
        out["producer_id"]  = int(it.attrs["producer_id"])
        out["paired_slice_id"] = int(it.attrs["paired_slice_id"])
        # GT baseline
        gb = f["gt_baseline"]
        out["gt_pe_pred"]   = gb["pe_pred"][:].astype(np.float64)
        out["gt_oob_frac"]  = float(gb.attrs["oob_frac"])
        out["gt_n_sp"]      = int(gb.attrs["n_sp"])
        # Per-slice
        ps = f["pred_slices"]
        out["slice_query_id"]   = ps["query_id"][:].astype(np.int64)
        out["slice_class"]      = ps["class_argmax"][:].astype(np.int64)
        out["slice_n_sp"]       = ps["n_sp"][:].astype(np.int64)
        out["slice_iou"]        = ps["iou_vs_gt_nu"][:].astype(np.float32)
        out["slice_oob_frac"]   = ps["oob_frac"][:].astype(np.float32)
        out["slice_pe_pred"]    = ps["pe_pred"][:].astype(np.float64)
        # Metrics (for annotations only)
        m = f["metrics"]
        out["m1_slice_id"]      = int(m.attrs.get("m1_slice_id", -1))
        out["m1_slice_id_intent"] = int(m.attrs.get("m1_slice_id_intent", -1))
        # Truth attrs
        tr = f["truth"].attrs
        out["truth"] = dict(
            has_neutrino=bool(tr.get("has_neutrino", False)),
            nu_pdg=int(tr.get("nu_pdg", 0)),
            ccnc=int(tr.get("ccnc", -1)),
            mode=int(tr.get("mode", -1)),
            nu_energy_MeV=float(tr.get("nu_energy_MeV", 0.0)),
        )
    return out


# ---------------------------------------------------------------------------
# Chi-2 helpers (Neyman with sys floor — same as flash_chi2.py)
# ---------------------------------------------------------------------------

def chi2_neyman(pe_obs, pe_pred, f_sys=DEFAULT_F_SYS, eps=DEFAULT_EPS):
    var = pe_obs + (f_sys * pe_obs) ** 2 + eps
    return float(((pe_obs - pe_pred) ** 2 / var).sum())


def chi2_per_pmt(pe_obs, pe_pred, f_sys=DEFAULT_F_SYS, eps=DEFAULT_EPS):
    var = pe_obs + (f_sys * pe_obs) ** 2 + eps
    return ((pe_obs - pe_pred) ** 2 / var).astype(np.float32)


def slice_chi2s_at_gamma(d, gamma, oob_threshold, f_sys, eps):
    """(Q,) chi-2 per slice at the given γ. NaN where OOB-rejected."""
    pe_obs = d["pe_obs"]
    chi2s = np.full(d["slice_query_id"].shape[0], np.nan, dtype=np.float64)
    for qi in range(d["slice_query_id"].shape[0]):
        if d["slice_oob_frac"][qi] > oob_threshold:
            continue
        chi2s[qi] = chi2_neyman(
            pe_obs, gamma * d["slice_pe_pred"][qi], f_sys, eps,
        )
    return chi2s


def gt_chi2_at_gamma(d, gamma, oob_threshold, f_sys, eps):
    if d["gt_oob_frac"] > oob_threshold:
        return float("nan")
    return chi2_neyman(d["pe_obs"], gamma * d["gt_pe_pred"], f_sys, eps)


# ---------------------------------------------------------------------------
# Plot builders
# ---------------------------------------------------------------------------

def _slice_name(cls_id, nu_class_id, no_object_class_id):
    if cls_id == nu_class_id:        return "nu"
    if cls_id == no_object_class_id: return "no_obj"
    return f"c{cls_id}"


def build_dropdown_options(d):
    """Sorted by IoU descending so the most-nu-looking slices float up."""
    order = np.argsort(-d["slice_iou"])
    opts = []
    nu_cls = d["nu_class_id"]
    no_obj = d["no_object_class_id"]
    for q in order:
        qid   = int(d["slice_query_id"][q])
        cls   = int(d["slice_class"][q])
        n_sp  = int(d["slice_n_sp"][q])
        iou   = float(d["slice_iou"][q])
        oob   = float(d["slice_oob_frac"][q])
        tag = ""
        if qid == d["m1_slice_id"]:
            tag += " [M1]"
        if qid == d["paired_slice_id"]:
            tag += " [in-time-paired]"
        label = (f"qid={qid:>3d}  cls={_slice_name(cls, nu_cls, no_obj):>6s}  "
                 f"n_sp={n_sp:>6d}  IoU={iou:.3f}  OOB={oob:.2f}{tag}")
        opts.append({"label": label, "value": int(q)})
    return opts


def build_main_figure(d, selected_idx, gamma, oob_threshold, log_y,
                      f_sys, eps):
    """Bar chart: 4 series of (32,) PE on a PMT-id axis."""
    pe_obs = d["pe_obs"]
    pmt_ids = np.arange(pe_obs.shape[0])

    # Slice picks
    sel_pred  = gamma * d["slice_pe_pred"][selected_idx]
    gt_pred   = gamma * d["gt_pe_pred"]
    # min-chi2 slice at the current γ
    chi2s = slice_chi2s_at_gamma(d, gamma, oob_threshold, f_sys, eps)
    min_chi2_idx = (int(np.nanargmin(chi2s)) if np.isfinite(chi2s).any()
                    else None)
    min_chi2_pred = (gamma * d["slice_pe_pred"][min_chi2_idx]
                     if min_chi2_idx is not None
                     else np.zeros_like(pe_obs))

    # Labels with totals + chi-2
    sel_qid   = int(d["slice_query_id"][selected_idx])
    sel_cls   = _slice_name(int(d["slice_class"][selected_idx]),
                            d["nu_class_id"], d["no_object_class_id"])
    chi2_sel  = chi2_neyman(pe_obs, sel_pred,    f_sys, eps)
    chi2_gt   = chi2_neyman(pe_obs, gt_pred,     f_sys, eps)
    chi2_min  = (chi2_neyman(pe_obs, min_chi2_pred, f_sys, eps)
                 if min_chi2_idx is not None else float("nan"))
    sum_obs   = float(pe_obs.sum())
    sum_sel   = float(sel_pred.sum())
    sum_gt    = float(gt_pred.sum())
    sum_min   = float(min_chi2_pred.sum())

    series = [
        # (name, x, y, color, line_dash)
        # Observed: white so it stands out on the plotly_dark background.
        ("Observed (in-time flash)",
            pmt_ids, pe_obs,        "white",       None,
            f"ΣPE={sum_obs:.1f}"),
        (f"Predicted: selected slice qid={sel_qid} ({sel_cls})",
            pmt_ids, sel_pred,      "royalblue",   None,
            f"ΣPE={sum_sel:.1f}  χ²={chi2_sel:.1f}"),
        ("Predicted: GT-nu mask (truth)",
            pmt_ids, gt_pred,       "seagreen",    "dash",
            f"ΣPE={sum_gt:.1f}  χ²={chi2_gt:.1f}"),
    ]
    if min_chi2_idx is not None:
        mqid = int(d["slice_query_id"][min_chi2_idx])
        mcls = _slice_name(int(d["slice_class"][min_chi2_idx]),
                           d["nu_class_id"], d["no_object_class_id"])
        series.append((
            f"Predicted: min-χ² slice qid={mqid} ({mcls})",
            pmt_ids, min_chi2_pred, "darkorange",  None,
            f"ΣPE={sum_min:.1f}  χ²={chi2_min:.1f}",
        ))
    else:
        series.append((
            "Predicted: min-χ² slice (none survive OOB)",
            pmt_ids, min_chi2_pred, "darkorange", None, "(NaN)",
        ))

    fig = go.Figure()
    # Lines + markers (4 series overlaid) — clearer than grouped bars at 32 PMTs.
    for name, x, y, color, dash, totstr in series:
        fig.add_trace(go.Scatter(
            x=x, y=y, mode="lines+markers",
            name=f"{name}  [{totstr}]",
            line=dict(color=color, width=2,
                      dash=("dash" if dash == "dash" else None)),
            marker=dict(size=6, color=color),
        ))
    fig.update_layout(
        template="plotly_dark",
        xaxis=dict(title="OpDet (PMT) id  [0..31]", dtick=1,
                   showgrid=True, gridcolor="#444"),
        yaxis=dict(title="PE (per PMT)" + (" — log" if log_y else ""),
                   type=("log" if log_y else "linear"),
                   showgrid=True, gridcolor="#444"),
        margin=dict(l=60, r=20, t=40, b=50),
        legend=dict(orientation="h", y=-0.18, x=0,
                    font=dict(size=10)),
        title=(f"PE per PMT — event ({d['run']}, {d['subrun']}, {d['event']})  "
               f"category={categorize.category_str(d['category_mask'])}  "
               f"γ={gamma:.4g}  OOB≤{oob_threshold:.2f}"),
        height=520,
        hovermode="x unified",
    )
    return fig, min_chi2_idx, chi2_sel, chi2_gt, chi2_min


def build_per_pmt_residual_figure(d, selected_idx, gamma, f_sys, eps):
    """(pe_obs - γ·pe_pred)² / var per PMT for the 3 prediction series."""
    pe_obs = d["pe_obs"]
    pmt_ids = np.arange(pe_obs.shape[0])
    sel_pred = gamma * d["slice_pe_pred"][selected_idx]
    gt_pred  = gamma * d["gt_pe_pred"]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=pmt_ids, y=chi2_per_pmt(pe_obs, sel_pred, f_sys, eps),
        name="selected", marker_color="royalblue", opacity=0.7,
    ))
    fig.add_trace(go.Bar(
        x=pmt_ids, y=chi2_per_pmt(pe_obs, gt_pred, f_sys, eps),
        name="GT", marker_color="seagreen", opacity=0.7,
    ))
    fig.update_layout(
        template="plotly_dark", barmode="group",
        xaxis=dict(title="PMT id", dtick=1),
        yaxis=dict(title="χ² contribution per PMT"),
        margin=dict(l=60, r=20, t=30, b=40),
        legend=dict(orientation="h", y=-0.22, x=0),
        title="Per-PMT χ² contribution  (selected vs GT)",
        height=300,
    )
    return fig


# ---------------------------------------------------------------------------
# Side panel
# ---------------------------------------------------------------------------

def build_slice_table_rows(d, selected_idx, gamma, oob_threshold, f_sys, eps,
                           min_chi2_idx, max_rows=20):
    """Per-slice metrics, sorted by IoU desc, top max_rows. Highlight the
    selected and min-chi2 rows.
    """
    chi2s = slice_chi2s_at_gamma(d, gamma, oob_threshold, f_sys, eps)
    pe_pred_totals = (gamma * d["slice_pe_pred"]).sum(axis=-1)
    order = np.argsort(-d["slice_iou"])[:max_rows]
    rows = [html.Tr([
        html.Th("qid"), html.Th("class"), html.Th("n_sp"), html.Th("IoU"),
        html.Th("OOB"), html.Th("ΣPE_pred(γ)"), html.Th("χ²(γ)"),
    ], style={"color": "white"})]
    for q in order:
        cls = _slice_name(int(d["slice_class"][q]),
                          d["nu_class_id"], d["no_object_class_id"])
        oob = float(d["slice_oob_frac"][q])
        ch  = float(chi2s[q]) if np.isfinite(chi2s[q]) else float("nan")
        row_style = {"color": "white"}
        if q == selected_idx:
            row_style["backgroundColor"] = "#1f5fa6"
        elif min_chi2_idx is not None and q == min_chi2_idx:
            row_style["backgroundColor"] = "#a35400"
        rows.append(html.Tr([
            html.Td(str(int(d["slice_query_id"][q]))),
            html.Td(cls),
            html.Td(str(int(d["slice_n_sp"][q]))),
            html.Td(f"{float(d['slice_iou'][q]):.3f}"),
            html.Td(f"{oob:.3f}"),
            html.Td(f"{float(pe_pred_totals[q]):.1f}"),
            html.Td("rej" if np.isnan(ch) else f"{ch:.1f}"),
        ], style=row_style))
    return rows


# ---------------------------------------------------------------------------
# Dash app
# ---------------------------------------------------------------------------

def build_app(d, initial_gamma):
    app = dash.Dash(__name__)
    app.title = (f"Flash viewer — event "
                 f"({d['run']}, {d['subrun']}, {d['event']})")

    slice_opts = build_dropdown_options(d)
    # Initial pick: M1 slice if it's in pred slices, else first
    initial_idx = slice_opts[0]["value"]
    for opt in slice_opts:
        if int(d["slice_query_id"][opt["value"]]) == d["m1_slice_id"]:
            initial_idx = opt["value"]
            break

    oob_opts = [
        {"label": f"OOB ≤ {float(th):.2f}", "value": float(th)}
        for th in d["oob_thresholds"]
    ]
    initial_oob = float(d["oob_thresholds"][d["default_oob_idx"]])

    header_lines = [
        (f"({d['run']}, {d['subrun']}, {d['event']})  "
         f"model_tag={d['model_tag']}"),
        (f"category={categorize.category_str(d['category_mask'])}  "
         f"in-time flash idx=?  t0={d['flash_t0_us']:.3f}us  "
         f"producer={d['producer_id']}  ΣPE_obs={float(d['pe_obs'].sum()):.1f}"),
        (f"GT-nu mask: n_sp={d['gt_n_sp']}  oob_frac={d['gt_oob_frac']:.3f}"),
        (f"truth: has_nu={d['truth']['has_neutrino']}  "
         f"nu_pdg={d['truth']['nu_pdg']}  ccnc={d['truth']['ccnc']}  "
         f"Enu={d['truth']['nu_energy_MeV']:.1f} MeV"),
    ]

    app.layout = html.Div([
        html.H3("LArFormer per-event flash viewer",
                style={"color": "white"}),
        html.Div(
            [html.Div(line, style={"color": "#bbb"}) for line in header_lines],
            style={"fontFamily": "monospace", "marginBottom": "12px"},
        ),
        html.Div([
            html.Div([
                html.Label("Slice:", style={"color": "white",
                                             "marginRight": "8px"}),
                dcc.Dropdown(
                    id="slice-dropdown",
                    options=slice_opts,
                    value=initial_idx,
                    clearable=False,
                    style={"width": "640px", "color": "black"},
                ),
            ], style={"display": "inline-block", "marginRight": "20px"}),
            html.Div([
                html.Label("γ:", style={"color": "white",
                                         "marginRight": "8px"}),
                dcc.Input(
                    id="gamma-input", type="number", value=float(initial_gamma),
                    min=0.001, step=1.0, debounce=True,
                    style={"width": "100px"},
                ),
            ], style={"display": "inline-block", "marginRight": "20px"}),
            html.Div([
                html.Label("OOB threshold:",
                           style={"color": "white", "marginRight": "8px"}),
                dcc.Dropdown(
                    id="oob-dropdown",
                    options=oob_opts,
                    value=initial_oob,
                    clearable=False,
                    style={"width": "150px", "color": "black"},
                ),
            ], style={"display": "inline-block", "marginRight": "20px"}),
            dcc.Checklist(
                id="log-y",
                options=[{"label": " log y-axis", "value": "log"}],
                value=[],
                style={"display": "inline-block", "color": "white"},
            ),
        ], style={"marginBottom": "12px"}),
        html.Div(dcc.Graph(id="fig-main"), style={"marginBottom": "8px"}),
        html.Div(dcc.Graph(id="fig-residual"), style={"marginBottom": "8px"}),
        html.Div([
            html.Div("Top slices by IoU vs GT-nu  "
                     "(blue=selected, orange=min-χ²)",
                     style={"color": "#bbb", "marginBottom": "4px"}),
            html.Table(id="slice-table",
                       style={"color": "white",
                              "borderCollapse": "collapse",
                              "width": "auto",
                              "fontFamily": "monospace",
                              "fontSize": "12px"}),
        ]),
    ], style={"backgroundColor": "#222", "padding": "16px",
              "minHeight": "100vh"})

    @app.callback(
        Output("fig-main", "figure"),
        Output("fig-residual", "figure"),
        Output("slice-table", "children"),
        Input("slice-dropdown", "value"),
        Input("gamma-input", "value"),
        Input("oob-dropdown", "value"),
        Input("log-y", "value"),
    )
    def _update(selected_idx, gamma_val, oob_val, log_y_val):
        try:
            gamma = float(gamma_val) if gamma_val else 1.0
        except (TypeError, ValueError):
            gamma = 1.0
        log_y = bool(log_y_val and "log" in log_y_val)
        oob_threshold = float(oob_val) if oob_val is not None else initial_oob
        fig_main, min_chi2_idx, _c_sel, _c_gt, _c_min = build_main_figure(
            d, int(selected_idx), gamma, oob_threshold, log_y,
            DEFAULT_F_SYS, DEFAULT_EPS,
        )
        fig_res = build_per_pmt_residual_figure(
            d, int(selected_idx), gamma, DEFAULT_F_SYS, DEFAULT_EPS,
        )
        rows = build_slice_table_rows(
            d, int(selected_idx), gamma, oob_threshold,
            DEFAULT_F_SYS, DEFAULT_EPS, min_chi2_idx,
        )
        return fig_main, fig_res, rows

    return app


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--perevent-h5", required=True,
                    help="Path to one perevent_<run>_<sub>_<evt>.h5")
    ap.add_argument("--gamma", type=float, default=1.0,
                    help="Initial γ. Default 1.0 (same as analyzer default).")
    ap.add_argument("--port", type=int, default=8052)
    args = ap.parse_args()

    d = load_perevent(args.perevent_h5)
    print(f"Loaded {args.perevent_h5}")
    print(f"  event=({d['run']}, {d['subrun']}, {d['event']})  "
          f"n_pred_slices={d['slice_query_id'].shape[0]}  "
          f"category={categorize.category_str(d['category_mask'])}")
    app = build_app(d, initial_gamma=args.gamma)
    print(f"Listening on http://0.0.0.0:{args.port}  (Ctrl-C to stop)")
    app.run(debug=False, port=args.port, host="0.0.0.0")


if __name__ == "__main__":
    main()
