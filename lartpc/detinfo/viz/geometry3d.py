"""
plotly traces for a detinfo geometry (in the style of icdl/lardly detector outlines).

All functions return lists of plotly trace dicts / graph_objects usable in
lardly-style dash apps or a static figure (see make_detinfo_html.py).
"""
import numpy as np
import plotly.graph_objects as go

# fixed edge list of an axis-aligned/oriented box given its 8 corners in
# the order produced by _box_corners
_EDGES = [(0, 1), (1, 3), (3, 2), (2, 0), (4, 5), (5, 7), (7, 6), (6, 4),
          (0, 4), (1, 5), (2, 6), (3, 7)]

# arrows: line shaft + a small cone mesh drawn in detector coordinates (plotly's Cone
# trace sizes heads relative to the scene, which made them metres long)
ARROW_SHAFT_CM = 60.0
ARROW_HEAD_CM = 12.0
ARROW_HEAD_RADIUS_CM = 4.0


def arrow_head_mesh(tips, dirs, name, color, length=ARROW_HEAD_CM, radius=ARROW_HEAD_RADIUS_CM,
                    nseg=16, legendgroup=None, hovertext=None):
    """One Mesh3d holding a cone (apex at each tip, pointing along dir) per arrow."""
    X, Y, Z, I, J, K = [], [], [], [], [], []
    for tip, d in zip(tips, dirs):
        d = np.asarray(d, float) / np.linalg.norm(d)
        a = np.array([1.0, 0, 0]) if abs(d[0]) < 0.9 else np.array([0, 1.0, 0])
        e1 = np.cross(d, a); e1 /= np.linalg.norm(e1)
        e2 = np.cross(d, e1)
        base = np.asarray(tip) - d * length
        i0 = len(X)
        X.append(tip[0]); Y.append(tip[1]); Z.append(tip[2])                 # apex
        X.append(base[0]); Y.append(base[1]); Z.append(base[2])              # base centre
        for k in range(nseg):
            ph = 2 * np.pi * k / nseg
            q = base + radius * (np.cos(ph) * e1 + np.sin(ph) * e2)
            X.append(q[0]); Y.append(q[1]); Z.append(q[2])
        for k in range(nseg):
            r0, r1 = i0 + 2 + k, i0 + 2 + (k + 1) % nseg
            I += [i0, i0 + 1]; J += [r0, r1]; K += [r1, r0]                   # side, base
    return go.Mesh3d(x=X, y=Y, z=Z, i=I, j=J, k=K, color=color, opacity=1.0, flatshading=True,
                     name=name, showlegend=False, legendgroup=legendgroup or name,
                     hoverinfo="text" if hovertext else "name",
                     hovertext=hovertext)

OPDET_COLORS = {
    "pmt_coated": "rgb(220,120,0)", "pmt_uncoated": "rgb(255,200,60)",
    "xarapuca_vuv": "rgb(40,110,200)", "xarapuca_vis": "rgb(110,190,240)",
    "pmt": "rgb(220,120,0)", "box": "rgb(40,110,200)", "unknown": "rgb(150,150,150)",
}


def _box_corners(center, axes, half):
    """8 corners of an oriented box: center + sum_i s_i*half_i*axes_i."""
    c = []
    for sx in (-1, 1):
        for sy in (-1, 1):
            for sz in (-1, 1):
                c.append(center + sx * half[0] * axes[0] + sy * half[1] * axes[1] + sz * half[2] * axes[2])
    return np.array(c)


def _segments_xyz(segs):
    """list of (p0, p1) -> x, y, z lists with None separators."""
    x, y, z = [], [], []
    for p0, p1 in segs:
        x += [p0[0], p1[0], None]
        y += [p0[1], p1[1], None]
        z += [p0[2], p1[2], None]
    return x, y, z


def _box_segments(corners):
    return [(corners[i], corners[j]) for i, j in _EDGES]


def aabb_segments(bmin, bmax):
    center = 0.5 * (np.asarray(bmin) + np.asarray(bmax))
    half = 0.5 * (np.asarray(bmax) - np.asarray(bmin))
    return _box_segments(_box_corners(center, np.eye(3), half))


def lines_trace(segs, name, color, width=2, dash=None, showlegend=True, legendgroup=None,
                hoverinfo="name", visible=True):
    x, y, z = _segments_xyz(segs)
    line = dict(color=color, width=width)
    if dash:
        line["dash"] = dash
    return go.Scatter3d(x=x, y=y, z=z, mode="lines", name=name, line=line,
                        showlegend=showlegend, legendgroup=legendgroup or name,
                        hoverinfo=hoverinfo, visible=visible)


def detector_traces(di, show_full_tpc=True):
    """Cryostat, TPC (full + active) boxes, cathodes, plane outlines, drift arrows."""
    g = di.geometry
    tr = []
    segs = []
    for c in range(g.Ncryostats()):
        b = g.GetCryostat(c).bbox
        segs += aabb_segments(b.min, b.max)
    tr.append(lines_trace(segs, "cryostat", "rgb(120,120,120)", 1, visible="legendonly"))

    act, full, cath, planes = [], [], [], []
    cones = dict(x=[], y=[], z=[], u=[], v=[], w=[], text=[])
    shafts = []
    for t in g.IterateTPCs():
        a = t.active_bbox
        act += aabb_segments(a.min, a.max)
        full += aabb_segments(t.bbox.min, t.bbox.max)
        # cathode: the active-box x face opposite to the drift direction
        ax = int(np.argmax(np.abs(t.drift_dir)))
        xc = t.cathode_center[ax]
        lo, hi = a.min.copy(), a.max.copy()
        lo[ax] = hi[ax] = xc
        cath += aabb_segments(lo, hi)
        for pl in t.planes:
            lo, hi = a.min.copy(), a.max.copy()
            lo[ax] = hi[ax] = pl.center[ax]
            planes += aabb_segments(lo, hi)
        # drift arrow: 60 cm shaft from the active-volume centre, small cone at the tip
        ctr = a.center
        tip = ctr + t.drift_dir * ARROW_SHAFT_CM
        shafts.append((ctr, tip))
        cones["x"].append(tip[0]); cones["y"].append(tip[1]); cones["z"].append(tip[2])
        cones["u"].append(t.drift_dir[0]); cones["v"].append(t.drift_dir[1]); cones["w"].append(t.drift_dir[2])
        cones["text"].append(f"TPC {tuple(t.id)} drift {np.round(t.drift_dir, 3).tolist()}")
    tr.append(lines_trace(act, "TPC active volume", "rgb(0,0,0)", 4))
    if show_full_tpc:
        tr.append(lines_trace(full, "TPC full volume", "rgb(90,90,90)", 1, dash="dot",
                              visible="legendonly"))
    tr.append(lines_trace(cath, "cathode", "rgb(200,30,30)", 3))
    tr.append(lines_trace(planes, "wire planes", "rgb(30,150,60)", 2))
    tr.append(lines_trace(shafts, "drift direction", "rgb(200,30,30)", 5, legendgroup="drift"))
    tips = np.stack([cones["x"], cones["y"], cones["z"]], 1)
    dirs = np.stack([cones["u"], cones["v"], cones["w"]], 1)
    tr.append(arrow_head_mesh(tips, dirs, "drift direction", "rgb(200,30,30)",
                              legendgroup="drift", hovertext=cones["text"][0] if len(cones["text"]) == 1 else None))
    return tr


def opdet_traces(di):
    g = di.geometry
    t = g.OpDetTable()
    types = di.conventions.opdet_types()
    ch_of = {g.OpDetFromOpChannel(ch): ch for ch in range(g.NOpChannels())}
    tr = []
    for typ in sorted(set(types)):
        m = np.array([x == typ for x in types])
        idx = np.nonzero(m)[0]
        c = t["center"][m]
        text = [f"opdet {i} (opch {ch_of.get(i, '?')})<br>{typ}<br>{t['volume_name'][i]}<br>"
                f"({c[k,0]:.1f}, {c[k,1]:.1f}, {c[k,2]:.1f})" for k, i in enumerate(idx)]
        tr.append(go.Scatter3d(x=c[:, 0], y=c[:, 1], z=c[:, 2], mode="markers",
                               name=f"opdet: {typ} ({m.sum()})", text=text, hoverinfo="text",
                               marker=dict(size=4, color=OPDET_COLORS.get(typ, "gray"),
                                           symbol="circle" if "pmt" in typ else "square")))
    return tr


def crt_traces(di, strips=False):
    g = di.geometry
    aux = g.AuxDetTable()
    if aux is None:
        return []
    walls = di.conventions.crt_walls()
    tr = []
    palette = ["rgb(150,60,180)", "rgb(60,160,160)", "rgb(160,120,60)", "rgb(90,90,200)",
               "rgb(200,90,140)", "rgb(80,150,80)"]
    for k, wall in enumerate(sorted(set(walls))):
        m = np.nonzero(walls == wall)[0]
        segs = []
        for i in m:
            axes = np.array([aux["axis_x"][i], aux["axis_y"][i], aux["axis_z"][i]])
            half = np.array([aux["half_width1"][i], aux["half_height"][i], 0.5 * aux["length"][i]])
            segs += _box_segments(_box_corners(aux["center"][i], axes, half))
        tr.append(lines_trace(segs, f"CRT {wall} ({len(m)} modules)", palette[k % len(palette)], 2,
                              legendgroup="crt"))
    if strips:
        s = g.AuxDetSensitiveTable()
        segs = []
        for i in range(len(s["auxdet"])):
            axes = np.array([s["axis_x"][i], s["axis_y"][i], s["axis_z"][i]])
            half = np.array([s["half_width1"][i], s["half_height"][i], 0.5 * s["length"][i]])
            segs += _box_segments(_box_corners(s["center"][i], axes, half))
        tr.append(lines_trace(segs, f"CRT strips ({len(s['auxdet'])})", "rgb(180,180,180)", 1,
                              visible="legendonly"))
    return tr


def wire_traces(di, plane, tpc=0, cryo=0, stride=1, color="rgb(30,150,60)"):
    """Wires of one plane (every `stride`-th), with wire 0 and the last wire highlighted."""
    g = di.geometry
    pl = g.GetPlane(plane, tpc, cryo)
    start, end = pl.WireEndPoints()
    idx = np.arange(0, pl.nwires, max(1, stride))
    segs = [(start[i], end[i]) for i in idx]
    name = f"wires C{cryo} T{tpc} P{plane} ({pl.ViewName}) every {stride}"
    tr = [lines_trace(segs, name, color, 1)]
    for i, col, lab in ((0, "rgb(230,0,0)", "wire 0"), (pl.nwires - 1, "rgb(0,0,230)", f"wire {pl.nwires-1}")):
        w = pl.Wire(i)
        tr.append(go.Scatter3d(x=[w.start[0], w.end[0]], y=[w.start[1], w.end[1]],
                               z=[w.start[2], w.end[2]], mode="lines+markers",
                               line=dict(color=col, width=6), marker=dict(size=3, color=col),
                               name=f"{lab} (ch {w.channel})", hoverinfo="name"))
    # increasing-wire direction arrow from wire 0 centre
    c0 = pl.first_wire_center
    d = pl.increasing_wire_dir
    tip = c0 + d * ARROW_SHAFT_CM
    tr.append(lines_trace([(c0, tip)], "increasing wire dir", "rgb(230,0,0)", 5,
                          legendgroup="incdir"))
    tr.append(arrow_head_mesh([tip], [d], "increasing wire dir", "rgb(230,0,0)",
                              legendgroup="incdir"))
    return tr


def geometry_figure(di, wires=None, wire_stride=1, crt=True, crt_strips=False, height=850):
    """wires: None or (cryo, tpc, plane)."""
    tr = detector_traces(di) + opdet_traces(di)
    if crt:
        tr += crt_traces(di, strips=crt_strips)
    if wires is not None:
        c, t, p = wires
        tr += wire_traces(di, p, t, c, stride=wire_stride)
    fig = go.Figure(data=tr)
    fig.update_layout(
        height=height, margin=dict(l=0, r=0, t=30, b=0),
        title=f"{di.detector} geometry ({di.geometry.detector_name}, "
              f"{di.provenance.get('experiment_product')} {di.version})",
        legend=dict(itemsizing="constant"),
        scene=dict(xaxis_title="x [cm]", yaxis_title="y [cm]", zaxis_title="z [cm]",
                   aspectmode="data", camera=dict(eye=dict(x=-1.6, y=0.9, z=-1.2),
                                                  up=dict(x=0, y=1, z=0))))
    return fig


def sce_slice_figure(di, kind="fwd", component=0, z=250.0, side_maps=True, height=420):
    """Heatmap of an SCE map component on an x-y slice at fixed z (both TPC sides)."""
    sce = di.sce
    x = np.linspace(-205, 205, 165)
    y = np.linspace(-205, 205, 165)
    X, Y = np.meshgrid(x, y)
    P = np.stack([X.ravel(), Y.ravel(), np.full(X.size, z)], axis=1)
    if kind == "fwd":
        V = sce.GetPosOffsets(P)
        title = "GetPosOffsets"
    elif kind == "bkwd":
        V = sce.GetCalPosOffsets(P, np.where(P[:, 0] < 0, 0, 1))
        title = "GetCalPosOffsets"
    else:
        V = sce.GetEfieldOffsets(P)
        title = "GetEfieldOffsets"
    comp = "xyz"[component]
    fig = go.Figure(go.Heatmap(x=x, y=y, z=V[:, component].reshape(X.shape), colorscale="RdBu",
                               zmid=0, colorbar=dict(title=f"d{comp}")))
    fig.update_layout(height=height, title=f"SCE {title} {comp}-component at z = {z:.0f} cm",
                      xaxis_title="x [cm]", yaxis_title="y [cm]", margin=dict(l=40, r=10, t=40, b=40))
    fig.update_yaxes(scaleanchor="x")
    return fig
