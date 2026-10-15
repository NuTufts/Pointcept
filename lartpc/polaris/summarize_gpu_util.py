#!/usr/bin/env python3
"""Stage-B scaling / utilization report (stdlib only; runs on the Polaris host).

  summarize_gpu_util.py --logdir <$POL_LOGDIR/$TAG> --tag <tag> [--worklist <wl>]
                        [--nodes N] [--ppg P] [--mps 0|1] [--elapsed <job seconds>]
                        [--out <report.md>] [--csv <timeseries.csv>] [--brief]

Inputs: gpumon_<host>.csv (gpu_monitor.sh samples) and cascade_shard*.log
(run_cascade_shard.sh: SHARD_T0 / first event line / SHARD_INF_END / SHARD_T1).
Per shard: events processed, inference s/event (first event -> inference end,
i.e. excluding model build), pack+copy time. Per node: active window = first
shard start .. last shard end on that node; mean nvidia-smi utilization of each
GPU inside that window. Job: mean GPU utilization (all GPUs, active windows),
aggregate events/s, events per GPU-hour and per node-hour (charged = nodes x
--elapsed, or the span of the logs when --elapsed is absent).
--csv writes a 1-minute time series of mean utilization across all GPUs.
--brief prints one summary line (used by the test job's PPG/MPS matrix).
"""
import argparse
import datetime as dt
import glob
import os
import re
import statistics

EVENT_RE = re.compile(r"^(\S+) +\[(\d+)(?:\.[^\]]*)?\]")
STAMP_RE = re.compile(r"^(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d) ")
KV_RE = re.compile(r"(SHARD_T0|SHARD_INF_END|SHARD_T1)=(\d+)")
HOST_RE = re.compile(r"host=(\S+)")
GPU_RE = re.compile(r"gpu=(\d+)")


def pstamp(s):
    return dt.datetime.strptime(s, "%Y-%m-%dT%H:%M:%S").timestamp()


def scan_shard(path):
    st = {"shard": os.path.basename(path), "t0": None, "t_inf_end": None, "t1": None,
          "t_first": None, "t_last": None, "events": set(), "host": None, "gpu": None,
          "done": False, "tracebacks": 0}
    with open(path, errors="replace") as fh:
        for l in fh:
            for k, v in KV_RE.findall(l):
                st[{"SHARD_T0": "t0", "SHARD_INF_END": "t_inf_end", "SHARD_T1": "t1"}[k]] = int(v)
            if st["host"] is None and "host=" in l and "slot=" in l:
                m = HOST_RE.search(l); st["host"] = m.group(1).split(".")[0] if m else None
                m = GPU_RE.search(l); st["gpu"] = int(m.group(1)) if m else None
            m = EVENT_RE.match(l)
            if m:
                st["events"].add(int(m.group(2)))
                t = pstamp(m.group(1))
                st["t_first"] = t if st["t_first"] is None else min(st["t_first"], t)
                st["t_last"] = t if st["t_last"] is None else max(st["t_last"], t)
            if "Traceback" in l:
                st["tracebacks"] += 1
            if re.search(r"\bDONE shard\b", l):
                st["done"] = True
    return st


def read_gpumon(path):
    host = os.path.basename(path)[len("gpumon_"):-4]
    rows = []
    with open(path, errors="replace") as fh:
        for l in fh:
            if l.startswith("#") or not l.strip():
                continue
            p = [x.strip() for x in l.split(",")]
            if len(p) < 7:
                continue
            try:
                t = dt.datetime.strptime(p[0].split(".")[0], "%Y/%m/%d %H:%M:%S").timestamp()
                rows.append((t, int(p[1]), float(p[2]), float(p[3]), float(p[4]), float(p[6]) if p[6] not in ("[N/A]", "") else 0.0))
            except ValueError:
                continue
    return host, rows


def mean(xs):
    return statistics.fmean(xs) if xs else float("nan")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--logdir", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--worklist")
    ap.add_argument("--nodes", type=int, default=0)
    ap.add_argument("--ppg", type=int, default=1)
    ap.add_argument("--mps", type=int, default=0)
    ap.add_argument("--elapsed", type=float, default=0.0, help="job wall seconds (charged); default: span of the logs")
    ap.add_argument("--out")
    ap.add_argument("--csv")
    ap.add_argument("--brief", action="store_true")
    args = ap.parse_args()

    shards = [scan_shard(p) for p in sorted(glob.glob(os.path.join(args.logdir, "cascade_shard*.log")))]
    mons = dict(read_gpumon(p) for p in sorted(glob.glob(os.path.join(args.logdir, "gpumon_*.csv"))))
    hosts = sorted(set(s["host"] for s in shards if s["host"]) | set(mons))
    nnodes = args.nodes or len(hosts) or 1
    ngpu_node = 4

    # ---- per shard -----------------------------------------------------------
    rates, packs, n_events = [], [], 0
    for s in shards:
        s["n"] = len(s["events"]); n_events += s["n"]
        if s["t_first"] and s["t_inf_end"] and s["n"] > 1:
            s["s_per_ev"] = (s["t_inf_end"] - s["t_first"]) / s["n"]; rates.append(s["s_per_ev"])
        else:
            s["s_per_ev"] = None
        if s["t1"] and s["t_inf_end"]:
            packs.append(s["t1"] - s["t_inf_end"])
        s["build_s"] = (s["t_first"] - s["t0"]) if (s["t_first"] and s["t0"]) else None
    starts = [s["t0"] for s in shards if s["t0"]]
    ends = [(s["t1"] or s["t_inf_end"]) for s in shards if (s["t1"] or s["t_inf_end"])]
    span = (max(ends) - min(starts)) if starts and ends else 0.0
    elapsed = args.elapsed or span

    # ---- per node / GPU utilization inside the node's active window -----------
    node_rows, all_gpu_means, series = [], [], {}
    for h in hosts:
        sh = [s for s in shards if s["host"] == h]
        w0 = min([s["t0"] for s in sh if s["t0"]] or [0]); w1 = max([(s["t1"] or s["t_inf_end"] or 0) for s in sh] or [0])
        rows = mons.get(h, [])
        per_gpu = {}
        for t, g, ug, um, mu, pw in rows:
            if w0 and w1 and not (w0 <= t <= w1):
                continue
            per_gpu.setdefault(g, []).append((ug, um, mu, pw))
            series.setdefault(int(t // 60) * 60, []).append(ug)
        gm = {g: mean([r[0] for r in v]) for g, v in per_gpu.items()}
        all_gpu_means.extend(gm.values())
        node_rows.append((h, len(sh), sum(s["n"] for s in sh), w1 - w0 if w0 and w1 else 0,
                          gm, {g: max((r[2] for r in v), default=0) for g, v in per_gpu.items()},
                          {g: mean([r[3] for r in v]) for g, v in per_gpu.items()}))
    util = mean(all_gpu_means)
    gpu_hours = nnodes * ngpu_node * elapsed / 3600 if elapsed else float("nan")
    node_hours = nnodes * elapsed / 3600 if elapsed else float("nan")
    agg = n_events / elapsed if elapsed else float("nan")
    med_rate = statistics.median(rates) if rates else float("nan")
    ndone = sum(1 for s in shards if s["done"])

    brief = (f"tag={args.tag} nodes={nnodes} ppg={args.ppg} mps={args.mps} shards={len(shards)} done={ndone} "
             f"events={n_events} elapsed={elapsed:.0f}s agg={agg:.3f} ev/s per_proc_median={med_rate:.2f} s/ev "
             f"gpu_util_mean={util:.1f}% ev_per_gpu_h={n_events / gpu_hours if gpu_hours else float('nan'):.0f} "
             f"ev_per_node_h={n_events / node_hours if node_hours else float('nan'):.0f} node_hours={node_hours:.2f}")
    print(brief)
    if args.brief and not args.out:
        return

    L = [f"# Stage-B GPU pass report: {args.tag}", "",
         f"nodes {nnodes} x {ngpu_node} GPU, processes per GPU {args.ppg}, MPS {'on' if args.mps else 'off'}, "
         f"shards {len(shards)} ({ndone} done), events {n_events}", "",
         "| quantity | value |", "|---|---|",
         f"| job wall (charged) | {elapsed:.0f} s = {elapsed / 3600:.2f} h |",
         f"| node-hours | {node_hours:.2f} |",
         f"| GPU-hours | {gpu_hours:.2f} |",
         f"| aggregate throughput | {agg:.3f} events/s |",
         f"| events per GPU-hour | {n_events / gpu_hours if gpu_hours else float('nan'):.0f} |",
         f"| events per node-hour | {n_events / node_hours if node_hours else float('nan'):.0f} |",
         f"| per-process inference rate (median over shards, excl. model build) | {med_rate:.2f} s/event |",
         f"| model build per shard (median) | {statistics.median([s['build_s'] for s in shards if s['build_s']]) if any(s['build_s'] for s in shards) else float('nan'):.0f} s |",
         f"| pack + copy per shard (median) | {statistics.median(packs) if packs else float('nan'):.0f} s |",
         f"| mean GPU utilization (nvidia-smi, active windows) | {util:.1f} % over {len(all_gpu_means)} GPUs |",
         f"| tracebacks | {sum(s['tracebacks'] for s in shards)} |", "",
         "## Per node", "", "| node | shards | events | active s | GPU util % (0..3) | max mem MiB | mean W |", "|---|---|---|---|---|---|---|"]
    for h, nsh, nev, act, gm, mm, pw in node_rows:
        L.append(f"| {h} | {nsh} | {nev} | {act:.0f} | " + " / ".join(f"{gm.get(g, float('nan')):.0f}" for g in range(ngpu_node)) +
                 " | " + " / ".join(f"{mm.get(g, 0):.0f}" for g in range(ngpu_node)) +
                 " | " + " / ".join(f"{pw.get(g, float('nan')):.0f}" for g in range(ngpu_node)) + " |")
    L += ["", "## Per shard", "", "| shard | host | gpu | events | build s | s/event | pack s | done |", "|---|---|---|---|---|---|---|---|"]
    for s in shards:
        name = s["shard"].replace("cascade_", "").replace(".log", "")
        rate = f"{s['s_per_ev']:.2f}" if s["s_per_ev"] is not None else "-"
        build = f"{s['build_s']:.0f}" if s["build_s"] is not None else "-"
        pack = f"{s['t1'] - s['t_inf_end']}" if (s["t1"] and s["t_inf_end"]) else "-"
        L.append(f"| {name} | {s['host']} | {s['gpu']} | {s['n']} | {build} | {rate} | {pack} | {'yes' if s['done'] else 'no'} |")
    L += ["", "```", brief, "```"]
    if args.out:
        with open(args.out, "w") as fo:
            fo.write("\n".join(L) + "\n")
        print(f">>> report -> {args.out}")
    else:
        print("\n".join(L))
    if args.csv and series:
        with open(args.csv, "w") as fo:
            fo.write("minute_epoch,iso,mean_gpu_util_pct,n_samples\n")
            for t in sorted(series):
                fo.write(f"{t},{dt.datetime.fromtimestamp(t).isoformat()},{mean(series[t]):.1f},{len(series[t])}\n")
        print(f">>> time series -> {args.csv}")


if __name__ == "__main__":
    main()
