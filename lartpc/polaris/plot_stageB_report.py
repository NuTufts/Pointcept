#!/usr/bin/env python3
"""Figures for the stage-B GPU pass (allocation-proposal material).

  plot_stageB_report.py --logs <polaris_logs dir> --out <figure dir> [--prod-tag prod]
                        [--test-tags testB_p1m0,testB_p2m1,testB_p3m1]

Reads the files summarize_gpu_util.py reads (gpumon_<node>.csv, cascade_shard*.log,
report_<tag>*.md brief lines) and writes:
  stageB_gpu_util_timeseries.png   mean GPU utilization vs time for the production job
                                   (all GPUs), with the per-node spread and the phases
  stageB_config_matrix.png         per-GPU throughput and utilization for the 1-node
                                   PPG/MPS matrix next to the production run
  stageB_summary.md                the numbers behind both figures
Runs inside the pointcept container (matplotlib); stdlib + numpy + matplotlib only.
"""
import argparse
import datetime as dt
import glob
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# reference palette (dataviz skill): categorical slot 1 blue, slot 2 orange; chrome inks
C_BLUE, C_ORANGE, C_AQUA = "#2a78d6", "#eb6834", "#1baf7a"
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BAND = "#cde2fb"  # sequential blue step 100 for the per-node spread band

BRIEF_RE = re.compile(r"tag=(\S+) nodes=(\d+) ppg=(\d+) mps=(\d+) shards=(\d+) done=(\d+) events=(\d+) "
                      r"elapsed=(\d+)s agg=([\d.]+) ev/s per_proc_median=([\d.]+) s/ev gpu_util_mean=([\d.]+)% "
                      r"ev_per_gpu_h=(\d+) ev_per_node_h=(\d+) node_hours=([\d.]+)")
KV_RE = re.compile(r"(SHARD_T0|SHARD_INF_END|SHARD_T1)=(\d+)")
STAMP_RE = re.compile(r"^(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d) ")
EVENT_RE = re.compile(r"^(\S+) +\[(\d+)(?:\.[^\]]*)?\]")


def read_brief(path):
    with open(path, errors="replace") as fh:
        for l in fh:
            m = BRIEF_RE.search(l)
            if m:
                g = m.groups()
                return dict(tag=g[0], nodes=int(g[1]), ppg=int(g[2]), mps=int(g[3]), shards=int(g[4]),
                            events=int(g[6]), elapsed=int(g[7]), agg=float(g[8]), s_per_ev=float(g[9]),
                            util=float(g[10]), ev_gpu_h=int(g[11]), ev_node_h=int(g[12]), node_h=float(g[13]))
    return None


def read_gpumon(path, off=0.0):
    host = os.path.basename(path)[len("gpumon_"):-4]
    t, g, u = [], [], []
    with open(path, errors="replace") as fh:
        for l in fh:
            if l.startswith("#") or not l.strip():
                continue
            p = [x.strip() for x in l.split(",")]
            if len(p) < 7:
                continue
            try:
                ts = dt.datetime.strptime(p[0].split(".")[0], "%Y/%m/%d %H:%M:%S").timestamp()
                t.append(ts + off); g.append(int(p[1])); u.append(float(p[2]))
            except ValueError:
                continue
    return host, np.array(t), np.array(g), np.array(u)


def stamp_offset(logdir):
    """String stamps (shard logs, gpumon) are the Polaris node's local clock; the SHARD_* values are
    epoch seconds. Lines carrying both calibrate the offset, so this runs correctly in any timezone."""
    offs = []
    for path in sorted(glob.glob(os.path.join(logdir, "cascade_shard*.log")))[:5]:
        with open(path, errors="replace") as fh:
            for l in fh:
                m = STAMP_RE.match(l); kv = KV_RE.findall(l)
                if m and kv:
                    offs.append(int(kv[0][1]) - dt.datetime.strptime(m.group(1), "%Y-%m-%dT%H:%M:%S").timestamp())
    return float(np.median(offs)) if offs else 0.0


def shard_phases(logdir, off):
    """per shard: t0 (process start), t_first (first event), t_inf_end, t1 (image on Lustre)"""
    out = []
    for path in sorted(glob.glob(os.path.join(logdir, "cascade_shard*.log"))):
        st = {"t0": None, "t_first": None, "t_inf_end": None, "t1": None}
        with open(path, errors="replace") as fh:
            for l in fh:
                for k, v in KV_RE.findall(l):
                    st[{"SHARD_T0": "t0", "SHARD_INF_END": "t_inf_end", "SHARD_T1": "t1"}[k]] = int(v)
                m = EVENT_RE.match(l)
                if m and st["t_first"] is None:
                    st["t_first"] = dt.datetime.strptime(m.group(1), "%Y-%m-%dT%H:%M:%S").timestamp() + off
        out.append(st)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], epilog=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--logs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--prod-tag", default="prod")
    ap.add_argument("--test-tags", default="testB_p1m0,testB_p2m1,testB_p3m1")
    ap.add_argument("--bin", type=float, default=30.0, help="time-series bin (s)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": MUTED,
                         "ytick.color": MUTED, "axes.titlecolor": INK, "figure.facecolor": SURFACE,
                         "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE, "axes.spines.top": False,
                         "axes.spines.right": False, "grid.color": GRID, "grid.linewidth": 0.8})

    # ---- production job: time series ---------------------------------------------
    pdir = os.path.join(args.logs, args.prod_tag)
    prod = None
    for r in sorted(glob.glob(os.path.join(pdir, "report_*.md"))):
        prod = read_brief(r) or prod
    off = stamp_offset(pdir)
    mons = [read_gpumon(p, off) for p in sorted(glob.glob(os.path.join(pdir, "gpumon_*.csv")))]
    phases = shard_phases(pdir, off)
    t_job0 = min(s["t0"] for s in phases if s["t0"])
    t_first = np.median([s["t_first"] - t_job0 for s in phases if s["t_first"]])
    t_inf_end = np.array([s["t_inf_end"] - t_job0 for s in phases if s["t_inf_end"]])
    t_end = max(s["t1"] - t_job0 for s in phases if s["t1"])
    tmax = t_end + 60
    edges = np.arange(0, tmax + args.bin, args.bin)
    centers = (edges[:-1] + edges[1:]) / 2 / 60.0
    node_curves = []
    all_t, all_u = [], []
    for host, t, g, u in mons:
        rel = t - t_job0
        keep = (rel >= 0) & (rel <= tmax)
        rel, uu = rel[keep], u[keep]
        idx = np.digitize(rel, edges) - 1
        curve = np.full(len(centers), np.nan)
        for b in range(len(centers)):
            sel = idx == b
            if sel.any():
                curve[b] = uu[sel].mean()
        node_curves.append(curve)
        all_t.append(rel); all_u.append(uu)
    node_curves = np.array(node_curves)
    mean_curve = np.nanmean(node_curves, axis=0)
    lo, hi = np.nanmin(node_curves, axis=0), np.nanmax(node_curves, axis=0)
    inf_lo, inf_hi = t_first, np.median(t_inf_end)
    sel = (centers * 60 >= inf_lo) & (centers * 60 <= inf_hi)
    plateau = float(np.nanmean(mean_curve[sel]))

    fig, ax = plt.subplots(figsize=(8.0, 3.6), dpi=160)
    ax.fill_between(centers, lo, hi, color=BAND, linewidth=0, label="per-node range (10 nodes)")
    ax.plot(centers, mean_curve, color=C_BLUE, linewidth=2, label="mean of 40 GPUs")
    ax.axvspan(0, t_first / 60, color=GRID, alpha=0.6, linewidth=0)
    ax.text(t_first / 120, 97, "model\nbuild", ha="center", va="top", fontsize=8, color=INK2)
    ax.axvline(np.median(t_inf_end) / 60, color=AXIS, linewidth=1, linestyle=(0, (3, 3)))
    ax.text(np.median(t_inf_end) / 60 - 0.3, 97, "inference done\n(median shard)", ha="right", va="top", fontsize=8, color=INK2)
    ax.hlines(plateau, inf_lo / 60, inf_hi / 60, color=C_ORANGE, linewidth=1.2, linestyle=(0, (1, 2)))
    ax.text(inf_hi / 60 - 0.5, plateau + 3, f"{plateau:.0f} % during inference", ha="right", va="bottom", fontsize=8, color=INK2)
    ax.set_xlim(0, tmax / 60); ax.set_ylim(0, 100)
    ax.set_xlabel("minutes since job start"); ax.set_ylabel("GPU utilization (nvidia-smi, %)")
    ax.set_title(f"LArFormer stage B on Polaris: {prod['nodes']} nodes x 4 A100, {prod['ppg']} processes/GPU"
                 f"{', MPS' if prod['mps'] else ''}, {prod['events']:,} events in {prod['elapsed'] / 60:.0f} min",
                 fontsize=10, loc="left")
    ax.grid(axis="y"); ax.legend(loc="lower center", frameon=False, fontsize=8, ncol=2)
    fig.tight_layout(); f1 = os.path.join(args.out, "stageB_gpu_util_timeseries.png"); fig.savefig(f1); plt.close(fig)

    # ---- config matrix: per-GPU steady-state rate + measured utilization ------------
    rows = []
    for tag in args.test_tags.split(","):
        for r in sorted(glob.glob(os.path.join(args.logs, tag, "report_*.md"))):
            b = read_brief(r)
            if b:
                rows.append(b)
    rows.append(prod)
    labels = [f"{r['ppg']}/GPU{' + MPS' if r['mps'] else ''}" + ("\nproduction" if i == len(rows) - 1 else "")
              for i, r in enumerate(rows)]
    test0, prod0 = rows[0], rows[-1]
    steady = [3600 * r["ppg"] / r["s_per_ev"] for r in rows]         # events per GPU-hour, inference only
    measured = [r["ev_gpu_h"] for r in rows]                          # events per GPU-hour, charged wall
    util = [r["util"] for r in rows]
    colors = [C_BLUE] * (len(rows) - 1) + [C_ORANGE]

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.0, 3.9), dpi=160, gridspec_kw={"wspace": 0.3})
    x = np.arange(len(rows)); w = 0.36
    a1.bar(x - w / 2, steady, w, color=colors, alpha=0.45, linewidth=0, label="inference only (per-process rate)")
    a1.bar(x + w / 2, measured, w, color=colors, linewidth=0, label="end to end (charged wall)")
    for i in range(len(rows)):
        a1.text(x[i] - w / 2, steady[i] + 60, f"{steady[i]:,.0f}", ha="center", va="bottom", fontsize=7.5, color=INK2)
        a1.text(x[i] + w / 2, measured[i] + 60, f"{measured[i]:,.0f}", ha="center", va="bottom", fontsize=7.5, color=INK2)
    a1.set_xticks(x); a1.set_xticklabels(labels, fontsize=8); a1.set_ylabel("events per GPU-hour")
    a1.set_ylim(0, max(steady + measured) * 1.38); a1.grid(axis="y"); a1.set_title("Throughput", fontsize=10, loc="left")
    a1.legend(frameon=False, fontsize=7.5, loc="upper left")
    a2.bar(x, util, 0.55, color=colors, linewidth=0)
    for i in range(len(rows)):
        a2.text(x[i], util[i] + 1.5, f"{util[i]:.0f} %", ha="center", va="bottom", fontsize=8, color=INK2)
    a2.set_xticks(x); a2.set_xticklabels(labels, fontsize=8); a2.set_ylabel("mean GPU utilization (%)")
    a2.set_ylim(0, 100); a2.grid(axis="y"); a2.set_title("GPU utilization over the active window", fontsize=10, loc="left")
    fig.suptitle(f"Processes per GPU: 1-node test matrix, {test0['events'] // test0['shards']} events per process (blue) "
                 f"vs the {prod0['nodes']}-node production run, {prod0['events'] // prod0['shards']:,} per process (orange)",
                 fontsize=10, x=0.01, ha="left")
    fig.tight_layout(); f2 = os.path.join(args.out, "stageB_config_matrix.png"); fig.savefig(f2); plt.close(fig)

    # ---- numbers ----------------------------------------------------------------------
    build = np.median([s["t_first"] - s["t0"] for s in phases if s["t_first"] and s["t0"]])
    pack = np.median([s["t1"] - s["t_inf_end"] for s in phases if s["t1"] and s["t_inf_end"]])
    inf_span = np.median(t_inf_end) - t_first
    L = [f"# Stage-B figures: {args.prod_tag}", "",
         "| run | nodes | proc/GPU | MPS | events | ev/shard | wall s | ev/s | ev/GPU-h (wall) | ev/GPU-h (inference) | s/ev per proc | GPU util % |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r, s_ in zip(rows, steady):
        L.append(f"| {r['tag']} | {r['nodes']} | {r['ppg']} | {r['mps']} | {r['events']} | {r['events'] // r['shards']} | {r['elapsed']} | "
                 f"{r['agg']:.2f} | {r['ev_gpu_h']} | {s_:.0f} | {r['s_per_ev']:.2f} | {r['util']:.1f} |")
    L += ["", f"Production phases (median over {len(phases)} shards): model build {build:.0f} s, inference {inf_span / 60:.1f} min, "
          f"pack + copy {pack:.0f} s; mean utilization during the inference phase {plateau:.1f} % "
          f"(active-window mean incl. build/tail {prod['util']:.1f} %).",
          f"Per-node utilization range over the whole job: {np.nanmean(node_curves, axis=1).min():.0f}-{np.nanmean(node_curves, axis=1).max():.0f} %.",
          "", f"figures: {f1}, {f2}"]
    with open(os.path.join(args.out, "stageB_summary.md"), "w") as fo:
        fo.write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
