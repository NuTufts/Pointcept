#!/usr/bin/env python3
"""
Static HTML page to review a detinfo cache: 3D geometry (plotly), automatic
sanity checks, LArSoft validation results, and every detector / electronics /
LAr / space-charge / photon-library property with its provenance.

    python lartpc/detinfo/viz/make_detinfo_html.py --detector sbnd -o sbnd_detinfo.html
    python lartpc/detinfo/viz/make_detinfo_html.py --detector sbnd --wires 0,1,0 --wire-stride 4 -o w.html
    options: --version, --path FILE.h5, --crt-strips, --no-crt, --offline (embed plotly.js),
             --artifact (page fragment for publishing as a claude.ai artifact)

--wires CRYO,TPC,PLANE draws that plane's wires (one plane at a time keeps the
page responsive; --wire-stride N draws every N-th wire). Wire 0 is red, the last
wire blue, and a red cone shows the increasing-wire direction.
"""
import argparse
import html
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
from lartpc.detinfo import load  # noqa: E402
from lartpc.detinfo.viz.geometry3d import geometry_figure, sce_slice_figure  # noqa: E402

PLOTLYJS_CDN = "https://cdn.jsdelivr.net/npm/plotly.js-dist-min@3.0.1/plotly.min.js"
FONTS = ("https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500"
         "&family=IBM+Plex+Sans:wght@400;600&display=swap")

CSS = """
/* layout: one reading column, figure panels kept light (instrument display) in both themes */
:root { --bg:#f6f7f9; --fg:#1b1f24; --muted:#5b6470; --line:#d5dae1; --head:#eaedf1;
        --accent:#1f5fa8; --plot-bg:#ffffff;
        --pass:#1a7f37; --fail:#c62828; --warn:#9a5b00;
        --sans:"IBM Plex Sans",system-ui,-apple-system,"Segoe UI",sans-serif;
        --mono:"IBM Plex Mono",ui-monospace,SFMono-Regular,Menlo,monospace; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
        --bg:#14171c; --fg:#e4e7ec; --muted:#9aa3ae; --line:#2e343d; --head:#1d2128;
        --accent:#7fb0ea; --plot-bg:#ffffff; --pass:#4ac26b; --fail:#ff7b72; --warn:#e3a33b;
        color-scheme: dark; } }
:root[data-theme="dark"] { --bg:#14171c; --fg:#e4e7ec; --muted:#9aa3ae; --line:#2e343d;
        --head:#1d2128; --accent:#7fb0ea; --plot-bg:#ffffff; --pass:#4ac26b; --fail:#ff7b72;
        --warn:#e3a33b; color-scheme: dark; }
body { background:var(--bg); color:var(--fg); font:14px/1.5 var(--sans); margin:0; }
main { max-width:1400px; margin:0 auto; padding-inline:16px; padding-block:16px 40px; }
h1 { font-size:22px; margin:4px 0 2px; text-wrap:balance; }
h2 { font-size:17px; margin:28px 0 8px; border-bottom:1px solid var(--line); padding-bottom:4px;
     text-wrap:balance; }
h3 { font-size:14px; margin:10px 0 4px; color:var(--muted); text-transform:uppercase;
     letter-spacing:.04em; }
.sub { color:var(--muted); margin-bottom:12px; overflow-wrap:anywhere; }
code, pre { font-family:var(--mono); }
.plot { background:var(--plot-bg); border:1px solid var(--line); border-radius:6px; overflow:hidden;
        max-width:100%; }
table { border-collapse:collapse; margin:6px 0 10px; font-size:13px; }
th, td { border:1px solid var(--line); padding:3px 8px; text-align:left; vertical-align:top; }
th { background:var(--head); font-weight:600; }
td { overflow-wrap:anywhere; }
td.num { text-align:right; font-variant-numeric:tabular-nums; font-family:var(--mono); font-size:12px; }
.pass { color:var(--pass); font-weight:600; } .fail { color:var(--fail); font-weight:600; }
.warn { color:var(--warn); font-weight:600; }
details { margin:6px 0; } summary { cursor:pointer; font-weight:600; color:var(--accent); }
pre { background:var(--head); padding:8px; overflow:auto; max-height:420px; font-size:12px; }
.grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(min(380px,100%),1fr)); gap:14px; }
.grid > div { min-width:0; }
.scroll { overflow-x:auto; max-width:100%; }
"""



DISPLAY_NAMES = {"sbnd": "SBND", "uboone": "MicroBooNE", "icarus": "ICARUS"}

def esc(v):
    if isinstance(v, float):
        return f"{v:.6g}"
    if isinstance(v, np.ndarray):
        return esc(v.tolist())
    if isinstance(v, list):
        return "[" + ", ".join(esc(x) for x in v) + "]"
    return html.escape(str(v))


def kv_table(d, title=None, skip=()):
    rows = "".join(f"<tr><th>{esc(k)}</th><td>{esc(v)}</td></tr>"
                   for k, v in d.items() if k not in skip)
    cap = f"<h3>{esc(title)}</h3>" if title else ""
    return f"{cap}<div class='scroll'><table>{rows}</table></div>"


def rows_table(header, rows):
    th = "".join(f"<th>{esc(h)}</th>" for h in header)
    body = "".join("<tr>" + "".join(
        f"<td class='num'>{esc(c)}</td>" if isinstance(c, (int, float, np.integer, np.floating))
        else f"<td>{esc(c)}</td>" for c in r) + "</tr>" for r in rows)
    return f"<div class='scroll'><table><tr>{th}</tr>{body}</table></div>"


# ---------------------------------------------------------------- sanity checks
def sanity_checks(di):
    g = di.geometry
    out = []

    def add(name, status, detail):
        out.append((name, status, detail))

    # channels
    wt = g._wire_tab
    ch = wt["channel"]
    uniq = np.unique(ch)
    contiguous = uniq[0] == 0 and uniq[-1] == len(uniq) - 1
    add("channel numbering", "pass" if contiguous and len(uniq) == g.Nchannels() else "fail",
        f"{len(uniq)} unique channels over {len(ch)} wires, range [{uniq[0]}, {uniq[-1]}], "
        f"Nchannels={g.Nchannels()}; channels shared by >1 wire: {len(ch) - len(uniq)}")
    for pl in g.IteratePlanes():
        s, e = pl.WireEndPoints()
        # geometric consistency of the per-plane frame
        perp = abs(float(np.dot(pl.wire_dir, pl.increasing_wire_dir)))
        a = g.GetTPC(pl.id.tpc, pl.id.cryo).active_bbox
        # wire endpoints should lie on the plane's own frame (centre +- depth/2 in y,
        # +- width/2 in z); angled wires meet the frame corners within ~1 pitch/sin
        lo = pl.center[1:] - 0.5 * np.array([pl.depth, pl.width])
        hi = pl.center[1:] + 0.5 * np.array([pl.depth, pl.width])
        ends = np.concatenate([s, e])
        dframe = np.min(np.abs(np.stack([ends[:, 1] - lo[0], ends[:, 1] - hi[0],
                                         ends[:, 2] - lo[1], ends[:, 2] - hi[1]], 1)), axis=1)
        # wires 0 and n-1 match the projected wire coordinate
        wc_first = pl.WireCoordinate(0.5 * (s[0] + e[0]))
        wc_last = pl.WireCoordinate(0.5 * (s[-1] + e[-1]))
        ok = (perp < 1e-4 and abs(wc_first) < 1e-3 and abs(wc_last - (pl.nwires - 1)) < 1e-3
              and dframe.max() < 1.5)
        add(f"plane {tuple(pl.id)} {pl.ViewName}", "pass" if ok else "warn",
            f"wire_dir.incr_dir={perp:.1e}; WireCoordinate(wire0)={wc_first:.2e}, "
            f"(wire N-1)={wc_last:.3f} (N-1={pl.nwires-1}); max endpoint distance to plane frame "
            f"{dframe.max():.2f} cm; frame y[{lo[0]:.2f},{hi[0]:.2f}] z[{lo[1]:.2f},{hi[1]:.2f}] vs "
            f"active y[{a.min[1]:.2f},{a.max[1]:.2f}] z[{a.min[2]:.2f},{a.max[2]:.2f}]")
    # drift direction: planes are on the drift side of the cathode
    for t in g.IterateTPCs():
        d = [float(np.dot(pl.center - t.cathode_center, t.drift_dir)) for pl in t.planes]
        add(f"drift TPC {tuple(t.id)}", "pass" if min(d) > 0 else "fail",
            f"plane distance from cathode along drift dir: {np.round(d, 2).tolist()} "
            f"(drift_distance {t.drift_distance:.2f})")
    # optical detectors outside the TPC active volumes
    pos = g.OpDetPositions()
    inside = np.zeros(len(pos), bool)
    for t in g.IterateTPCs():
        inside |= t.active_bbox.contains(pos)
    add("opdets outside active volume", "pass" if not inside.any() else "warn",
        f"{int(inside.sum())} of {len(pos)} photon detectors inside a TPC active volume")
    types = di.conventions.opdet_types()
    tc = {k: int(np.sum(types == k)) for k in sorted(set(types))}
    add("opdet types", "pass" if "unknown" not in tc else "warn", json.dumps(tc))
    if g.NAuxDets():
        walls = di.conventions.crt_walls()
        add("CRT walls", "pass", json.dumps({k: int(np.sum(walls == k)) for k in sorted(set(walls))}))
    fq = di.failed_queries
    add("extraction failed queries", "pass" if not fq else "warn", json.dumps(fq) if fq else "none")
    return out


def validation_checks(di):
    from lartpc.detinfo.tests.test_validation import run_checks
    try:
        return [(n, "pass" if ok else "fail", d) for n, ok, d in run_checks(di, verbose=False)]
    except Exception as e:  # noqa: BLE001
        return [("validation", "fail", f"{type(e).__name__}: {e}")]


def checks_table(checks):
    rows = "".join(f"<tr><td class='{s}'>{s.upper()}</td><th>{esc(n)}</th><td>{esc(d)}</td></tr>"
                   for n, s, d in checks)
    return f"<div class='scroll'><table>{rows}</table></div>"


# ---------------------------------------------------------------- page
def build_page(di, args):
    import plotly.io as pio
    wires = tuple(int(x) for x in args.wires.split(",")) if args.wires else None
    fig = geometry_figure(di, wires=wires, wire_stride=args.wire_stride, crt=not args.no_crt,
                          crt_strips=args.crt_strips)
    include = False if args.artifact else (True if args.offline else "cdn")
    parts = []
    g = di.geometry
    pv = di.provenance
    parts.append(f"<h1>detinfo: {esc(di.detector)} &mdash; {esc(g.detector_name)}</h1>")
    parts.append(f"<div class='sub'>{esc(pv.get('experiment_product'))} {esc(di.version)} "
                 f"({esc(pv.get('qualifiers'))}) &middot; schema {esc(di.schema_version)} &middot; "
                 f"{esc(di.path)}</div>")
    parts.append("<div class='plot'>" + pio.to_html(fig, include_plotlyjs=include, full_html=False,
                                                     config=dict(responsive=True)) + "</div>")
    if wires is None:
        parts.append("<div class='sub'>Wires not drawn: rerun with <code>--wires CRYO,TPC,PLANE "
                     "[--wire-stride N]</code>.</div>")

    checks = sanity_checks(di)
    vchecks = validation_checks(di)
    nfail = sum(s == "fail" for _, s, _ in checks + vchecks)
    nwarn = sum(s == "warn" for _, s, _ in checks + vchecks)
    parts.append(f"<h2>Checks &mdash; <span class='{'fail' if nfail else 'pass'}'>{nfail} fail</span>, "
                 f"<span class='{'warn' if nwarn else 'pass'}'>{nwarn} warn</span></h2>")
    parts.append("<div class='grid'><div><h3>Geometry sanity</h3>" + checks_table(checks) +
                 "</div><div><h3>Python utility vs LArSoft reference (/validation)</h3>" +
                 checks_table(vchecks) + "</div></div>")

    # geometry tables
    parts.append("<h2>Geometry</h2>")
    parts.append(kv_table(g.attrs))
    rows = []
    for t in g.IterateTPCs():
        rows.append([str(tuple(t.id)), np.round(t.bbox.min, 3).tolist() + np.round(t.bbox.max, 3).tolist(),
                     np.round(t.active_bbox.min, 3).tolist() + np.round(t.active_bbox.max, 3).tolist(),
                     np.round(t.drift_dir, 3).tolist(), round(t.drift_distance, 3),
                     np.round(t.cathode_center, 3).tolist(), t.Nplanes()])
    parts.append(rows_table(["TPC (c,t)", "bbox [min,max]", "active bbox", "drift dir",
                             "drift dist", "cathode center", "planes"], rows))
    rows = []
    for pl in g.IteratePlanes():
        rows.append([str(tuple(pl.id)), pl.ViewName, pl.signal_type.name, pl.nwires,
                     round(pl.wire_pitch, 4), round(float(np.degrees(pl.theta_z)), 2),
                     pl.first_channel, pl.first_channel + pl.nwires - 1,
                     np.round(pl.center, 3).tolist(), np.round(pl.wire_dir, 4).tolist(),
                     np.round(pl.increasing_wire_dir, 4).tolist(),
                     round(di.detprop.GetXTicksCoefficient(pl.id.plane, pl.id.tpc, pl.id.cryo), 4),
                     round(di.detprop.GetXTicksOffset(pl.id.plane, pl.id.tpc, pl.id.cryo), 3)])
    parts.append(rows_table(["plane (c,t,p)", "view", "signal", "wires", "pitch", "theta_z deg",
                             "ch first", "ch last", "center", "wire dir", "incr. wire dir",
                             "ticks/cm", "tick @ x=0"], rows))

    # properties
    parts.append("<h2>Electronics clocks, detector and LAr properties</h2><div class='grid'>")
    parts.append("<div>" + kv_table(di.clocks.attrs, "DetectorClocks") + "</div>")
    parts.append("<div>" + kv_table(di.detprop.attrs, "DetectorProperties") + "</div>")
    parts.append("<div>" + kv_table(di.larprop.attrs, "LArProperties") + "</div></div>")
    if di.larprop.spectra:
        sp = {k: f"{len(e)} points, E [{e.min():.3g}, {e.max():.3g}] eV"
              for k, (e, v) in di.larprop.spectra.items()}
        parts.append(kv_table(sp, "LArProperties spectra"))

    # space charge
    parts.append("<h2>Space charge</h2>")
    parts.append(kv_table(di.sce.config, skip=("fhicl",)))
    if di.sce.config.get("fhicl"):
        parts.append(f"<details><summary>SCE FHiCL block</summary><pre>{esc(di.sce.config['fhicl'])}</pre></details>")
    if type(di.sce).__name__ != "SpaceCharge":
        sfigs = [sce_slice_figure(di, "fwd", 0), sce_slice_figure(di, "bkwd", 0),
                 sce_slice_figure(di, "efield", 0)]
        parts.append("<div class='grid'>" + "".join(
            "<div class='plot'>" + pio.to_html(f, include_plotlyjs=False, full_html=False,
                                               config=dict(responsive=True)) + "</div>" for f in sfigs)
            + "</div>")

    # photon visibility
    parts.append("<h2>Photon visibility</h2>")
    parts.append(kv_table(di.photon_visibility_config))

    # provenance
    parts.append("<h2>Provenance</h2>")
    parts.append(kv_table(pv, skip=("product_versions_json", "failed_queries_json", "aux_files_json")))
    parts.append(kv_table(di.product_versions, "product versions"))
    aux = json.loads(pv.get("aux_files_json", "[]"))
    if aux:
        parts.append(rows_table(["aux file source", "copied to", "md5"],
                                [[a["source"], a["copied_to"], a["md5"]] for a in aux]))
    import h5py
    with h5py.File(di.path, "r") as f:
        fcl = f["provenance/services_fhicl"][()]
    fcl = fcl.decode() if isinstance(fcl, bytes) else fcl
    parts.append(f"<details><summary>services FHiCL ({len(fcl.splitlines())} lines)</summary>"
                 f"<pre>{esc(fcl)}</pre></details>")

    if wires is None:
        title = f"{DISPLAY_NAMES.get(di.detector, di.detector.upper())} Detector Geometry"
    else:
        wpl = g.GetPlane(wires[2], wires[1], wires[0])
        title = f"{DISPLAY_NAMES.get(di.detector, di.detector.upper())} TPC {wires[1]} {wpl.ViewName}-Plane Wires"
    head = (f"<title>{esc(title)}</title>"
            f'<link rel="stylesheet" href="{FONTS}">'
            f"<style>{CSS}</style>")
    if args.artifact:
        # claude.ai artifact: the host adds doctype/head/body; plotly.js from an allowed CDN
        return head + f'<script src="{PLOTLYJS_CDN}"></script><main>{"".join(parts)}</main>'
    return (f"<!doctype html><html><head><meta charset='utf-8'>"
            f"<meta name='viewport' content='width=device-width, initial-scale=1'>{head}</head>"
            f"<body><main>{''.join(parts)}</main></body></html>")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--detector", default="sbnd")
    ap.add_argument("--version", default=None)
    ap.add_argument("--path", default=None)
    ap.add_argument("-o", "--out", default=None)
    ap.add_argument("--wires", default=None, help="CRYO,TPC,PLANE to draw wires for")
    ap.add_argument("--wire-stride", type=int, default=1)
    ap.add_argument("--crt-strips", action="store_true")
    ap.add_argument("--no-crt", action="store_true")
    ap.add_argument("--offline", action="store_true", help="embed plotly.js (no CDN)")
    ap.add_argument("--artifact", action="store_true",
                    help="write a claude.ai artifact page (no html/head/body; plotly.js from jsDelivr)")
    args = ap.parse_args()
    di = load(args.detector, args.version, path=args.path)
    out = args.out or f"detinfo_{di.detector}_{di.version}.html"
    with open(out, "w") as f:
        f.write(build_page(di, args))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
