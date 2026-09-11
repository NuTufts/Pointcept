"""Fit the light-yield scale s (multiplier on GAMMA_BEAM_REF) from a record file.

    PYTHONPATH=$K python3 scripts/fit_gamma.py --sample extbnb200k_cew6 \
        [--records 'results/<chain>/records/<tag>*.npz'] [--arm muon|union|truth_nu]
        [--out results/<chain>/<tag>__muon.json] [--plots results/<chain>/plots]

Arms:
  muon      PRIMARY. Isolated one-boundary MIP muon, own-points prediction,
            flash-source test (scale-free shape match). Same cuts every sample.
  union     nu-union prediction (recomputed at GAMMA_BEAM_REF) for events with
            a candidate muon (or every event if records were built with
            --union-every). Legacy-like population; diagnostic only.
  truth_nu  MC only: union arm restricted to nu_qfrac >= --qfrac-min (the nu
            union IS the true neutrino); needs --union-every records.

Every cut is a CLI knob so it can be scanned; the JSON carries the cuts used,
the sequential drop counts, the statistic with bootstrap error, validity
gates, and the robustness scans.
"""
import argparse
import glob
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..", "..", "..")))

from lartpc.larformer_analysis.flashmodel_calib.gammacal import (  # noqa: E402
    CALIB_DIR, GAMMA_BEAM_REF, samples)
from lartpc.larformer_analysis.flashmodel_calib.gammacal.records import load_records  # noqa: E402
from lartpc.larformer_analysis.flashmodel_calib.gammacal.flash import in_window_ok  # noqa: E402
from lartpc.larformer_analysis.flashmodel_calib.gammacal.estimators import (  # noqa: E402
    ratio_stats, neyman_gamma, pooled_gamma, binned_medians, run_groups)
from lartpc.larformer_analysis.flashmodel_calib.gammacal.results import write_result  # noqa: E402


def _sum_live(pe, live):
    return float(np.clip(np.asarray(pe, np.float64), 0, None)[live].sum())


def select_muons(ev, mu, args, window):
    """Sequential cuts on the muon table; returns (mask, drop table)."""
    n = len(mu["ev"])
    keep = np.ones(n, bool)
    drops = []

    def cut(name, m):
        nonlocal keep
        before = int(keep.sum())
        keep &= m
        drops.append((name, before, int(keep.sum())))

    evi = mu["ev"]
    ok_flash = np.array([bool(ev["has_flash"][i]) and in_window_ok(
        ev["flash_time_us"][i], ev["flash_times"][i], ev["flash_pes"][i],
        window, margin=args.window_margin, min_pe=args.flash_min_pe)
        for i in evi])
    cut("in-window single flash", ok_flash)
    cut(f"length > {args.min_len:g} cm", mu["length"] > args.min_len)
    cut(f"n_boundary == {args.n_boundary}", mu["n_boundary"] == args.n_boundary)
    if args.n_boundary >= 1:
        cut(f"boundary end x > {args.x_boundary_min:g} cm",
            mu["boundary_end_x"] > args.x_boundary_min)
    if not args.allow_secondary:
        cut("primary (not secondary)", mu["is_secondary"] <= 0)
    cut(f"other tracks KE < {args.iso_track_ke:g}",
        mu["iso_track_ke_max"] < args.iso_track_ke)
    cut(f"showers E < {args.iso_shower_e:g}", mu["iso_shower_e_max"] < args.iso_shower_e)
    cut(f"proton veto KE < {args.proton_veto_ke:g}",
        mu["proton_ke_max"] < args.proton_veto_ke)
    q_union = ev["q_union"][evi]
    qfrac = np.where(q_union > 0, mu["q_mu"] / np.maximum(q_union, 1e-9), np.nan)
    cut(f"muon charge >= {args.q_frac_min:g} of union", qfrac >= args.q_frac_min)
    live = ev["live"][evi]
    pred_l = np.array([_sum_live(p, l) for p, l in zip(mu["pred_mu_ref"], live)])
    obs_l = np.array([_sum_live(ev["obs_pe"][i], l) for i, l in zip(evi, live)])
    cut(f"sum pred_ref >= {args.min_pred:g} PE & obs > 0",
        (pred_l >= args.min_pred) & (obs_l > 0))
    cut(f"shape cos >= {args.cos_min:g}", mu["cos_sim"] >= args.cos_min)
    cut(f"centroid |dy| < {args.match_dy:g}, |dz| < {args.match_dz:g}",
        (mu["dcy"] < args.match_dy) & (mu["dcz"] < args.match_dz))
    # one entry per event: the longest surviving muon
    best = {}
    for j in np.nonzero(keep)[0]:
        i = int(evi[j])
        if i not in best or mu["length"][j] > mu["length"][best[i]]:
            best[i] = j
    m = np.zeros(n, bool)
    m[list(best.values())] = True
    drops.append(("one muon per event", int(keep.sum()), int(m.sum())))
    return m, drops, dict(pred_l=pred_l, obs_l=obs_l, qfrac=qfrac)


def select_union(ev, args, window, qfrac_min=None):
    n = len(ev["gidx"])
    keep = np.ones(n, bool)
    drops = []

    def cut(name, m):
        nonlocal keep
        before = int(keep.sum()); keep &= m
        drops.append((name, before, int(keep.sum())))

    cut("union prediction available", np.isfinite(ev["pred_union_ref"]).all(1))
    ok_flash = np.array([bool(ev["has_flash"][i]) and in_window_ok(
        ev["flash_time_us"][i], ev["flash_times"][i], ev["flash_pes"][i],
        window, margin=args.window_margin, min_pe=args.flash_min_pe) for i in range(n)])
    cut("in-window single flash", ok_flash)
    pred_l = np.array([_sum_live(p, l) for p, l in zip(ev["pred_union_ref"], ev["live"])])
    obs_l = np.array([_sum_live(o, l) for o, l in zip(ev["obs_pe"], ev["live"])])
    cut(f"sum pred_ref >= {args.min_pred:g} PE & obs > 0",
        (pred_l >= args.min_pred) & (obs_l > 0))
    if qfrac_min is not None:
        cut(f"nu_qfrac >= {qfrac_min:g}", ev["nu_qfrac"] >= qfrac_min)
    return keep, drops, dict(pred_l=pred_l, obs_l=obs_l)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--sample", required=True, choices=sorted(samples.SAMPLES))
    ap.add_argument("--records", default=None, help="glob of record npz shards")
    ap.add_argument("--arm", default="muon", choices=["muon", "union", "truth_nu"])
    ap.add_argument("--out", default=None)
    ap.add_argument("--plots", default=None)
    ap.add_argument("--tag", default="", help="suffix for out/plot names (scans)")
    # flash window
    ap.add_argument("--window", default=None, help="lo,hi [us]; default = registry")
    ap.add_argument("--window-margin", type=float, default=0.2)
    ap.add_argument("--flash-min-pe", type=float, default=20.0)
    # muon cuts (the protocol defaults)
    ap.add_argument("--min-len", type=float, default=50.0)
    ap.add_argument("--n-boundary", type=int, default=1)
    ap.add_argument("--x-boundary-min", type=float, default=125.0)
    ap.add_argument("--allow-secondary", action="store_true")
    ap.add_argument("--iso-track-ke", type=float, default=50.0)
    ap.add_argument("--iso-shower-e", type=float, default=30.0)
    ap.add_argument("--proton-veto-ke", type=float, default=50.0)
    ap.add_argument("--q-frac-min", type=float, default=0.8)
    ap.add_argument("--min-pred", type=float, default=50.0)
    ap.add_argument("--cos-min", type=float, default=0.9)
    ap.add_argument("--match-dy", type=float, default=60.0)
    ap.add_argument("--match-dz", type=float, default=100.0)
    ap.add_argument("--qfrac-min", type=float, default=0.9, help="truth_nu arm")
    ap.add_argument("--nboot", type=int, default=2000)
    args = ap.parse_args()

    s = samples.get(args.sample)
    rec_glob = args.records or os.path.join(
        CALIB_DIR, "results", s["chain"], "records", f"{s['tag']}*.npz")
    paths = sorted(glob.glob(rec_glob))
    if not paths:
        raise SystemExit(f"no records match {rec_glob}")
    ev, mu, meta = load_records(paths)
    window = (tuple(float(x) for x in args.window.split(",")) if args.window
              else tuple(s["flash_window_us"]))
    suffix = f"__{args.arm}" + (f"_{args.tag}" if args.tag else "")
    out = args.out or os.path.join(CALIB_DIR, "results", s["chain"], f"{s['tag']}{suffix}.json")
    print(f">>> {s['tag']} ({s['kind']}, period {s['period']}, {s['chain']}) | "
          f"{len(paths)} shard(s), {len(ev['gidx'])} events, {len(mu.get('ev', []))} "
          f"muon candidates | window {window} margin {args.window_margin}")

    if args.arm == "muon":
        if not mu:
            raise SystemExit("no muon candidates in records")
        m, drops, aux = select_muons(ev, mu, args, window)
        evi = mu["ev"][m]
        r = aux["obs_l"][m] / aux["pred_l"][m]
        runs = ev["run"][evi]
        obs_rows = [ev["obs_pe"][i] for i in evi]
        pred_rows = [p for p in mu["pred_mu_ref"][m]]
        live_rows = [ev["live"][i] for i in evi]
        aux_x = dict(boundary_x=mu["boundary_end_x"][m], length=mu["length"][m],
                     vertexed=mu["vertexed"][m], orphan=mu["orphan"][m],
                     cos=mu["cos_sim"][m], obs=aux["obs_l"][m],
                     t_in_window=(ev["flash_time_us"][evi] - window[0]) / (window[1] - window[0]),
                     q_per_cm=mu["q_mu"][m] / np.maximum(mu["length"][m], 1e-9),
                     larpid_mu=mu["larpid_mu"][m], gt_origin=mu["gt_origin"][m])
    else:
        m, drops, aux = select_union(ev, args, window,
                                     qfrac_min=(args.qfrac_min if args.arm == "truth_nu" else None))
        evi = np.nonzero(m)[0]
        r = aux["obs_l"][m] / aux["pred_l"][m]
        runs = ev["run"][evi]
        obs_rows = [ev["obs_pe"][i] for i in evi]
        pred_rows = [ev["pred_union_ref"][i] for i in evi]
        live_rows = [ev["live"][i] for i in evi]
        aux_x = dict(obs=aux["obs_l"][m],
                     t_in_window=(ev["flash_time_us"][evi] - window[0]) / (window[1] - window[0]),
                     npts=ev["n_union_pts"][evi], nu_qfrac=ev["nu_qfrac"][evi])

    print("== sequential cuts ==")
    for name, before, after in drops:
        print(f"   {name:45s} {before:7d} -> {after:7d}")
    st = ratio_stats(r, nboot=args.nboot)
    g_ev = np.array([neyman_gamma(o, p, l) for o, p, l in zip(obs_rows, pred_rows, live_rows)])
    g_pool = pooled_gamma(obs_rows, pred_rows, live_rows)
    print(f"== {args.arm} arm: N={st['N']} | s = median(obs/pred_ref) = {st['median']:.4f} "
          f"+- {st['err_boot']:.4f} | p16-84 {st['p16']:.3f}-{st['p84']:.3f} | peak {st['peak']:.3f} "
          f"core {st['core_frac']:.2f} | gates {'OK' if st['gate_ok'] else 'FAIL'}")
    print(f"   neyman g: median {np.nanmedian(g_ev) if len(g_ev) else np.nan:.4f} | pooled {g_pool:.4f} "
          f"| geometric mean {st['mean_log']:.4f} | gamma_eff_fit = {GAMMA_BEAM_REF * st['median']:.3f}")

    scans = {}
    if st["N"]:
        scans["run_groups"] = run_groups(runs, r, ngroups=8)
        scans["brightness"] = binned_medians(aux_x["obs"], r, np.array([0, 100, 300, 1000, 3000, 1e9]))
        scans["t_in_window"] = binned_medians(aux_x["t_in_window"], r, np.linspace(0, 1, 6))
        if args.arm == "muon":
            scans["boundary_x"] = binned_medians(aux_x["boundary_x"], r,
                                                 np.array([args.x_boundary_min, 150, 180, 210, 240, 300]))
            scans["length"] = binned_medians(aux_x["length"], r, np.array([50, 80, 120, 180, 300, 1e4]))
            scans["cos_sim"] = binned_medians(aux_x["cos"], r, np.array([args.cos_min, 0.95, 0.98, 1.01]))
            for nm, mask in (("vertexed", aux_x["vertexed"] == 1), ("vertexless", aux_x["vertexed"] == 0),
                             ("orphan", aux_x["orphan"] == 1)):
                scans["subset_" + nm] = ratio_stats(r[mask], nboot=200) if mask.any() else None
            if np.isfinite(aux_x["gt_origin"]).any() and (aux_x["gt_origin"] >= 0).any():
                for nm, mask in (("gt_nu", aux_x["gt_origin"] == 1), ("gt_cosmic", aux_x["gt_origin"] == 2)):
                    scans["subset_" + nm] = ratio_stats(r[mask], nboot=200) if mask.any() else None
            scans["mip_q_per_cm_median"] = float(np.median(aux_x["q_per_cm"]))
        for k in ("run_groups", "brightness", "t_in_window", "boundary_x", "length", "cos_sim"):
            if k in scans:
                print(f"   scan {k:12s}: " + " | ".join(
                    f"[{lo:g},{hi:g}) N={N} {med:.3f}" for lo, hi, N, med in scans[k]))

    payload = dict(sample=s["tag"], kind=s["kind"], period=s["period"], chain=s["chain"],
                   description=s["description"], calib_only=s["calib_only"], arm=args.arm,
                   window_us=list(window), window_source=s["window_source"],
                   gamma_beam_ref=GAMMA_BEAM_REF,
                   production=dict(gamma_eff=sorted(set(np.round(ev["gamma_eff_file"][np.isfinite(ev["gamma_eff_file"])], 4).tolist())),
                                   gamma_scale=sorted(set(np.round(ev["gamma_scale_file"][np.isfinite(ev["gamma_scale_file"])], 4).tolist()))),
                   cuts={k: v for k, v in vars(args).items()
                         if k not in ("sample", "records", "out", "plots", "tag")},
                   drops=drops, N=st["N"], stats=st, scale_abs=st["median"],
                   scale_err=st["err_boot"], gamma_eff_fit=GAMMA_BEAM_REF * st["median"],
                   g_neyman_median=float(np.nanmedian(g_ev)) if len(g_ev) else None,
                   g_pooled=g_pool, scans=scans, records=paths,
                   n_events=int(len(ev["gidx"])), n_candidates=int(len(mu.get("ev", []))))
    write_result(out, payload)
    print(f">>> {out}")

    if args.plots and st["N"]:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        os.makedirs(args.plots, exist_ok=True)
        fig, ax = plt.subplots(1, 3, figsize=(15, 4.3))
        lr = np.log10(r)
        ax[0].hist(np.clip(lr, -1.5, 1.5), bins=60, color="#1f77b4", alpha=0.85)
        ax[0].axvline(np.log10(st["median"]), color="r", ls="--", label=f"median {st['median']:.3f}")
        ax[0].axvline(0, color="0.4", ls=":")
        ax[0].set(xlabel="log10(obs / pred_ref) live PMTs", ylabel="events",
                  title=f"{s['tag']} {args.arm}: s={st['median']:.3f}+-{st['err_boot']:.3f}\n"
                        f"N={st['N']} core={st['core_frac']:.2f} peak={st['peak']:.3f}")
        ax[0].legend(fontsize=8)
        o = np.array([_sum_live(x, l) for x, l in zip(obs_rows, live_rows)])
        p = np.array([_sum_live(x, l) for x, l in zip(pred_rows, live_rows)])
        hi = np.percentile(np.r_[o, p], 99)
        ax[1].plot([0, hi], [0, hi], "0.4", ls=":"); ax[1].plot([0, hi], [0, hi / st["median"]], "r--", lw=1)
        ax[1].scatter(o, p, s=5, alpha=0.4)
        ax[1].set(xlabel="observed PE", ylabel="predicted PE (gamma_ref)", xlim=(0, hi), ylim=(0, hi))
        if args.arm == "muon":
            ax[2].scatter(aux_x["boundary_x"], np.clip(r, 0, 4), s=5, alpha=0.3)
            for lo, hi_, N, med in scans["boundary_x"]:
                if N:
                    ax[2].plot([lo, min(hi_, 256)], [med, med], "r-", lw=2)
            ax[2].axhline(st["median"], color="0.4", ls=":")
            ax[2].set(xlabel="boundary-end drift x [cm]", ylabel="obs/pred_ref", ylim=(0, 4),
                      title="attenuation residual")
        else:
            ax[2].scatter(aux_x["t_in_window"], np.clip(r, 0, 4), s=5, alpha=0.3)
            ax[2].set(xlabel="flash time position in window", ylabel="obs/pred_ref", ylim=(0, 4))
        fig.tight_layout()
        png = os.path.join(args.plots, f"gamma_{s['tag']}{suffix}.png")
        fig.savefig(png, dpi=110); plt.close(fig)
        print(f">>> {png}")


if __name__ == "__main__":
    main()
