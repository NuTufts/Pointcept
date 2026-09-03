"""CC flash-chi2 peak SHAPE quantification (follow-up to cc_flash_breakdown):
the data chi2 peak is shifted high and broadened relative to MC in ALL
direction/containment splits — a GLOBAL effect. Prime suspect: the flash
light-yield scale (gamma) for the run-1 DATA leg was fit on the OLD chain's
slices; the new slicer's fuller slices predict more light, and a per-mille
pred/obs scale error inflates the Neyman chi2 quadratically.

For reco-CC, post-BDT (nu-dominated) events in MC and DATA:
  1. peak-region log10 chi2 median + IQR per split (contained/exiting/all);
  2. per-event flash amplitude ratio  r = sum_live(obs) / sum_live(pred_nu)
     (fit_gamma_run-style; dead {15} + saturation-hole masked) — data vs MC.
     r != 1 medians quantify the gamma retune each leg needs.
  3. peak-zoom figure: per split, area-normalized data vs MC chi2 shapes
     with medians marked; plus the r distributions.

    PYTHONPATH=./ python3 cc_chi2_shape.py ... (same inputs as breakdown)
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")))
from datamc_diagnostics import load  # noqa: E402
from flash_correction import rse_map  # noqa: E402
from cc_flash_breakdown import mu_vars  # noqa: E402
from lartpc.flashmatch.saturation import find_saturated  # noqa: E402


def flash_sums(d, cascade_dir, cache, dead=(15,)):
    """Per selected event: (obs_sum, pred_nu_sum) over live PMTs."""
    import h5py
    m = rse_map(cascade_dir, cache)
    obs_s = np.full(len(d["run"]), np.nan)
    pred_s = np.full(len(d["run"]), np.nan)
    for k in range(len(obs_s)):
        p = m.get((int(d["run"][k]), int(d["subrun"][k]),
                   int(d["event"][k])))
        if not p:
            continue
        try:
            with h5py.File(p if isinstance(p, str) else p[0], "r") as f:
                labs = [l.decode() if isinstance(l, bytes) else str(l)
                        for l in f["slices/label"][()]]
                if "nu" not in labs or "observed_pe" not in f["flash"]:
                    continue
                j = labs.index("nu")
                obs = np.clip(f["flash/observed_pe"][()], 0, None)
                pred = np.clip(f["slices/pred_pe"][()][j], 0, None)
                live = np.ones(32, bool)
                live[list(dead)] = False
                try:
                    live[list(find_saturated(obs, dead=dead))] = False
                except Exception:
                    pass
                obs_s[k] = obs[live].sum()
                pred_s[k] = pred[live].sum()
        except Exception:
            pass
    return obs_s, pred_s


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    for s in ("mc", "data"):
        ap.add_argument(f"--{s}-ntuple", required=True)
        ap.add_argument(f"--{s}-table", required=True)
    ap.add_argument("--recal-gamma-a", type=float, default=0.01556)
    ap.add_argument("--recal-gamma-b", type=float, default=-11.47)
    ap.add_argument("--mu-ke-min", type=float, default=50.0)
    ap.add_argument("--bdt-model", required=True)
    ap.add_argument("--bdt-thr", type=float, default=0.280)
    ap.add_argument("--plots", required=True)
    args = ap.parse_args()
    os.makedirs(args.plots, exist_ok=True)
    import joblib
    M = joblib.load(args.bdt_model)
    clf, FEATS = M["clf"], M["feats"]
    here = os.path.dirname(os.path.abspath(__file__))

    def cache_for(c):
        return os.path.join(here, "rse_" + os.path.basename(os.path.dirname(
            c.rstrip("/"))) + "_" + os.path.basename(c.rstrip("/")) + ".npz")

    smp = {}
    for s in ("mc", "data"):
        print(f">>> loading {s} ...", flush=True)
        nt = getattr(args, f"{s}_ntuple")
        casc = os.path.dirname(nt) + "/keypoint2_streams"
        d = load(nt, getattr(args, f"{s}_table"),
                 args.recal_gamma_a, args.recal_gamma_b,
                 args.mu_ke_min, 1e12, 1e12)
        from datamc_diagnostics import add_flash_pe
        d = add_flash_pe(d, casc, cache_for(casc))
        X = np.column_stack([d[f].astype(float) for f in FEATS])
        keep = d["recoCC"] & (clf.predict_proba(X)[:, 1] >= args.bdt_thr)
        if s == "mc":
            cat = np.load(getattr(args, f"{s}_table"))["cat"][d["row"]]
            keep &= ~((cat < 2) & (d["event"] % 2 == 0))
        sub = {k: v[keep] for k, v in d.items()}
        dirx, dwall = mu_vars(nt, sub["row"], args.mu_ke_min)
        obs_s, pred_s = flash_sums(sub, casc, cache_for(casc))
        smp[s] = dict(logchi2=sub["logchi2"], w=sub["w"], dwall=dwall,
                      r=obs_s / np.where(pred_s > 0, pred_s, np.nan))
        print(f"    {s}: {int(keep.sum())} events")

    splits = [("all", lambda d: np.ones(len(d["logchi2"]), bool)),
              ("contained (dwall>15)", lambda d: d["dwall"] > 15),
              ("exiting (dwall<5)", lambda d: d["dwall"] < 5)]
    print(f"\n== peak-region (log10 chi2 < 4) location/width ==")
    print(f"{'split':>22} {'MC med':>7} {'MC IQR':>7} {'DAT med':>8} "
          f"{'DAT IQR':>8} {'shift':>6}")
    for nm, fn in splits:
        row = []
        for s in ("mc", "data"):
            x = smp[s]["logchi2"][fn(smp[s])]
            x = x[x < 4]
            row += [np.median(x), np.subtract(*np.percentile(x, [75, 25]))]
        print(f"{nm:>22} {row[0]:7.2f} {row[1]:7.2f} {row[2]:8.2f} "
              f"{row[3]:8.2f} {row[2]-row[0]:+6.2f}")
    print(f"\n== flash amplitude ratio r = obs/pred (nu slice, live PMTs) ==")
    for nm, fn in splits:
        for s in ("mc", "data"):
            r = smp[s]["r"][fn(smp[s])]
            r = r[np.isfinite(r)]
            print(f"  {nm:>22} {s:>5}: median {np.median(r):.3f} "
                  f"IQR {np.subtract(*np.percentile(r,[75,25])):.3f} "
                  f"N {len(r)}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(2, 2, figsize=(11.5, 8.4))
    bins = np.linspace(0, 4, 33)
    for ax, (nm, fn) in zip(axs.ravel()[:3], splits):
        for s, sty in (("mc", dict(histtype="stepfilled", alpha=.45,
                                   color="#7bafd4")),
                       ("data", dict(histtype="step", lw=1.8, color="k"))):
            x = smp[s]["logchi2"][fn(smp[s])]
            w = smp[s]["w"][fn(smp[s])] if s == "mc" else None
            x = x[x < 4]
            ax.hist(x, bins, weights=(w[smp[s]["logchi2"][fn(smp[s])] < 4]
                                      if w is not None else None),
                    density=True, label=s.upper(), **sty)
            ax.axvline(np.median(x), color=("#33628f" if s == "mc" else "k"),
                       ls=":", lw=1.2)
        ax.set(xlabel="log10 flash chi2 (peak zoom)", title=nm)
        ax.legend(fontsize=8)
    ax = axs.ravel()[3]
    rb = np.linspace(0, 3, 41)
    for s, sty in (("mc", dict(histtype="stepfilled", alpha=.45,
                               color="#7bafd4")),
                   ("data", dict(histtype="step", lw=1.8, color="k"))):
        r = smp[s]["r"]
        r = r[np.isfinite(r)]
        ax.hist(r, rb, density=True, label=s.upper(), **sty)
        ax.axvline(np.median(r), color=("#33628f" if s == "mc" else "k"),
                   ls=":", lw=1.2)
    ax.axvline(1.0, color="r", lw=1)
    ax.set(xlabel="flash amplitude ratio obs/pred (nu slice)",
           title="light-yield scale check")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(args.plots, "cc_chi2_shape.png"), dpi=120)
    print(f">>> {args.plots}/cc_chi2_shape.png")


if __name__ == "__main__":
    main()
