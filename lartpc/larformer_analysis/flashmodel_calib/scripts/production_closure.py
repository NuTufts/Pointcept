"""Production-side closure: on a production's OWN stored predictions (at the
gamma it ran with), obs / pred on flash-matched nu slices should be ~1.

Reads a record file built from the production nu stream (build_records.py
--union-every not needed: the stored `slices/pred_pe` nu row is kept as
`pred_nu_file`). Selects events whose nu slice IS the in-time flash source by
the scale-free shape test (cosine of stored pred vs observed over live PMTs)
and reports median obs/pred_stored with the same validity gates.

    PYTHONPATH=$K python3 scripts/production_closure.py --sample extbnb_run1A_cew6 [--cos-min 0.95]
"""
import argparse
import glob
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..", "..", "..")))
from lartpc.larformer_analysis.flashmodel_calib.gammacal import CALIB_DIR, samples  # noqa: E402
from lartpc.larformer_analysis.flashmodel_calib.gammacal.records import load_records  # noqa: E402
from lartpc.larformer_analysis.flashmodel_calib.gammacal.flash import in_window_ok  # noqa: E402
from lartpc.larformer_analysis.flashmodel_calib.gammacal.estimators import ratio_stats, binned_medians  # noqa: E402
from lartpc.larformer_analysis.flashmodel_calib.gammacal.results import write_result  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--sample", required=True, choices=sorted(samples.SAMPLES))
    ap.add_argument("--records", default=None)
    ap.add_argument("--cos-min", type=float, default=0.95)
    ap.add_argument("--min-pred", type=float, default=100.0)
    ap.add_argument("--window-margin", type=float, default=0.2)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    s = samples.get(args.sample)
    paths = sorted(glob.glob(args.records or os.path.join(
        CALIB_DIR, "results", s["chain"], "records", f"{s['tag']}*.npz")))
    ev, mu, meta = load_records(paths)
    window = tuple(s["flash_window_us"])
    n = len(ev["gidx"])
    ge = ev["gamma_eff_file"]
    print(f">>> {s['tag']}: {n} events | production gamma_eff "
          f"{sorted(set(np.round(ge[np.isfinite(ge)], 4)))} scale "
          f"{sorted(set(np.round(ev['gamma_scale_file'][np.isfinite(ev['gamma_scale_file'])], 4)))}")
    r, cos, obs_tot, runs = [], [], [], []
    n_ok = dict(flash=0, nurow=0, cos=0)
    for i in range(n):
        if not (ev["has_flash"][i] and in_window_ok(ev["flash_time_us"][i], ev["flash_times"][i],
                                                    ev["flash_pes"][i], window,
                                                    margin=args.window_margin)):
            continue
        n_ok["flash"] += 1
        p = ev["pred_nu_file"][i]
        if not np.isfinite(p).all():
            continue
        n_ok["nurow"] += 1
        live = ev["live"][i]
        o = np.clip(ev["obs_pe"][i], 0, None)[live].astype(np.float64)
        q = np.clip(p, 0, None)[live].astype(np.float64)
        d = np.linalg.norm(o) * np.linalg.norm(q)
        c = float(o @ q / d) if d > 0 else np.nan
        if not (c >= args.cos_min and q.sum() >= args.min_pred and o.sum() > 0):
            continue
        n_ok["cos"] += 1
        r.append(o.sum() / q.sum()); cos.append(c); obs_tot.append(o.sum()); runs.append(ev["run"][i])
    r = np.asarray(r)
    st = ratio_stats(r)
    print(f"   in-window flash {n_ok['flash']} | nu row {n_ok['nurow']} | nu slice is the flash "
          f"source (cos >= {args.cos_min}) {n_ok['cos']}")
    print(f"== obs / stored pred (production gamma): N={st['N']} median {st['median']:.4f} "
          f"+- {st['err_boot']:.4f} | p16-84 {st['p16']:.3f}-{st['p84']:.3f} | core median "
          f"{st['core_median']:.3f} core frac {st['core_frac']:.2f} | gates {'OK' if st['gate_ok'] else 'FAIL'}")
    if st["N"]:
        for nm, x, e in (("brightness", np.asarray(obs_tot), np.array([0, 100, 300, 1000, 3000, 1e9])),):
            print(f"   scan {nm}: " + " | ".join(f"[{lo:g},{hi:g}) N={N} {m:.3f}"
                                                for lo, hi, N, m in binned_medians(x, r, e)))
    payload = dict(sample=s["tag"], kind=s["kind"], period=s["period"], chain=s["chain"],
                   check="production_closure", production_gamma_eff=sorted(set(np.round(ge[np.isfinite(ge)], 4)).tolist()),
                   cuts=vars(args), counts=n_ok, N=st["N"], stats=st, obs_over_pred_stored=st["median"])
    out = args.out or os.path.join(CALIB_DIR, "results", s["chain"], f"{s['tag']}__production_closure.json")
    write_result(out, payload); print(f">>> {out}")


if __name__ == "__main__":
    main()
