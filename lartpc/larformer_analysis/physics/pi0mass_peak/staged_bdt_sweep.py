"""Staged-cut sweep: per-shower cosmic-BDT gate FIRST (weak thresholds),
event-level flash-blind BDT second. Can the combination beat the event BDT
alone?

For each shower threshold t_s (0 = event-BDT-only baseline) the photon
CANDIDATE set is re-gated (showerCosmicScore >= t_s inside the >=2-photon
working point, so the pair itself can change), the event BDT is scored on
the re-selected pair features, and its threshold t_e is swept continuously.

Populations & hygiene (both BDTs):
  signal = MC truth cat<2, ODD events only, w x2 (event BDT trained on even)
  MC bkg = cat>=2, all events (never trained on)
  EXT    = rows>=100k (shower-BDT analysis half) AND odd events
           (event-BDT holdout), w = 1.1818 x 2
  beam data: closure only, at the official event WP t_e=0.280.

Metrics vs FIXED denominator (t_s=0, no event cut, odd-signal weight):
  eff        = kept signal / denominator
  purity     = near-peak (100<=mgg<170) sig/(sig+mcbkg+EXT)
  EXT peak   = near-peak EXT weight

    PYTHONPATH=./ python3 staged_bdt_sweep.py --plots plots_s1ep2p8_diag
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from datamc_diagnostics import load, add_flash_pe  # noqa: E402

MC_D = "/cluster/tufts/wongjiradlab/larbys/data/ub_on_tufts/larformer_mcoverlay67k_s1ep2p8"
DA_D = "/cluster/tufts/wongjiradlab/larbys/data/ub_on_tufts/larformer_bnb5e19_s1ep2p8"
EX_D = "/cluster/tufts/wongjiradlab/larbys/data/ub_on_tufts/larformer_extbnb200k_s1ep2p8_flash"
NT = {"mc": f"{MC_D}/dlgen2_larformer_ntuple_mc_overlay_s1ep2p8_run3.root",
      "data": f"{DA_D}/dlgen2_larformer_ntuple_bnb5e19_s1ep2p8.root",
      "ext": f"{EX_D}/dlgen2_larformer_ntuple_extbnb200k_s1ep2p8f.root"}
TB = {"mc": "mc_s1ep2p8_recal3_table.npz",
      "data": "data_s1ep2p8_recal3_table.npz",
      "ext": "ext_s1ep2p8_recal3_table.npz"}
CASC = {k: os.path.dirname(v) + "/keypoint2_streams" for k, v in NT.items()}
EXT_SCALE = 1.1818
GA, GB = 0.01553, -12.80
TS_LIST = [0.0, 0.01, 0.02, 0.03, 0.05, 0.09, 0.192]
PEAK = (100.0, 170.0)
OFFICIAL_TE = 0.280


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--bdt-model", default="ext_bdt_model_flashblind.joblib")
    ap.add_argument("--plots", required=True)
    args = ap.parse_args()
    os.makedirs(args.plots, exist_ok=True)
    here = os.path.dirname(os.path.abspath(__file__))
    import joblib
    M = joblib.load(os.path.join(here, args.bdt_model))
    clf, FEATS = M["clf"], M["feats"]
    need_flash = any(f in FEATS for f in ("flashPE", "logchi2"))
    print(f">>> event BDT feats: {FEATS} | needs flashPE: {need_flash}")

    def cache_for(c):
        return os.path.join(here, "rse_" + os.path.basename(os.path.dirname(
            c.rstrip("/"))) + "_" + os.path.basename(c.rstrip("/")) + ".npz")

    def grab(leg, ts):
        d = load(NT[leg], os.path.join(here, TB[leg]), GA, GB,
                 50.0, 1e4, 1778.0,
                 shower_bdt_min=(ts if ts > 0 else None))
        if need_flash:
            d = add_flash_pe(d, CASC[leg], cache_for(CASC[leg]))
        d["escore"] = clf.predict_proba(
            np.column_stack([d[f].astype(float) for f in FEATS]))[:, 1] \
            if len(d["run"]) else np.zeros(0)
        return d

    te_grid = np.r_[np.linspace(0.0, 0.95, 191), 0.28]
    te_grid = np.unique(te_grid)
    curves, denom = {}, None
    for ts in TS_LIST:
        print(f"\n>>> t_s = {ts}", flush=True)
        mc = grab("mc", ts)
        ex = grab("ext", ts)
        da = grab("data", ts)
        cat = np.load(os.path.join(here, TB["mc"]))["cat"][mc["row"]]
        sig = cat < 2
        odd = mc["event"] % 2 == 1
        m_sig = sig & odd
        m_bkg = ~sig
        w_sig = mc["w"][m_sig] * 2.0
        w_bkg = mc["w"][m_bkg]
        e_keep = (ex["row"] >= 100000) & (ex["event"] % 2 == 1)
        w_ext = np.full(int(e_keep.sum()), EXT_SCALE * 2.0)
        pk = lambda d, m: (d["mgg"][m] >= PEAK[0]) & (d["mgg"][m] < PEAK[1])
        pk_sig, pk_bkg = pk(mc, m_sig), pk(mc, m_bkg)
        pk_ext, pk_da = pk(ex, e_keep), (da["mgg"] >= PEAK[0]) & (da["mgg"] < PEAK[1])
        s_sig, s_bkg = mc["escore"][m_sig], mc["escore"][m_bkg]
        s_ext, s_da = ex["escore"][e_keep], da["escore"]
        if denom is None:
            denom = w_sig.sum()          # t_s=0, no event cut
            print(f">>> eff denominator (post-WP signal, odd x2): {denom:.1f}")
        eff, pur, extpk, sigpk = [], [], [], []
        for te in te_grid:
            ks, kb, ke = s_sig >= te, s_bkg >= te, s_ext >= te
            eff.append(w_sig[ks].sum() / denom)
            sp = w_sig[ks & pk_sig].sum()
            bp = w_bkg[kb & pk_bkg].sum()
            ep = w_ext[ke & pk_ext].sum()
            tot = sp + bp + ep
            pur.append(sp / tot if tot > 0 else np.nan)
            extpk.append(ep)
            sigpk.append(sp)
        curves[ts] = dict(te=te_grid, eff=np.array(eff), pur=np.array(pur),
                          extpk=np.array(extpk), sigpk=np.array(sigpk))
        ko = OFFICIAL_TE
        dpk = int((pk_da & (s_da >= ko)).sum())
        ppk = (w_sig[(s_sig >= ko) & pk_sig].sum()
               + w_bkg[(s_bkg >= ko) & pk_bkg].sum()
               + w_ext[(s_ext >= ko) & pk_ext].sum())
        print(f"    N post-WP: sig-odd {int(m_sig.sum())} | mcbkg {int(m_bkg.sum())} "
              f"| ext-eval {int(e_keep.sum())} | data {len(s_da)}")
        print(f"    @official t_e=0.280: data near-peak {dpk} | pred {ppk:.1f} "
              f"| d/p {dpk/max(ppk,1e-9):.2f}")

    print(f"\n== best near-peak purity at matched efficiency "
          f"(eff vs t_s=0 no-event-cut signal) ==")
    hdr = "  ".join(f"ts={ts:<5}" for ts in TS_LIST)
    print(f"{'eff':>6}  " + hdr + "   (cell: purity @ t_e)")
    for target in (0.99, 0.97, 0.95, 0.92, 0.90, 0.87, 0.85, 0.80):
        row = []
        for ts in TS_LIST:
            c = curves[ts]
            ok = c["eff"] >= target
            if not ok.any():
                row.append("   --      ")
                continue
            j = int(np.nanargmax(np.where(ok, c["pur"], np.nan)))
            row.append(f"{c['pur'][j]:.3f}@{c['te'][j]:.2f}")
        print(f"{target:6.2f}  " + "  ".join(f"{r:<11}" for r in row))

    np.savez(os.path.join(args.plots, "staged_bdt_sweep.npz"),
             ts_list=np.array(TS_LIST), denom=denom,
             **{f"c{i}_{k}": v for i, ts in enumerate(TS_LIST)
                for k, v in curves[ts].items()})

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 2, figsize=(12.5, 5.0))
    cmap = plt.get_cmap("viridis")
    for i, ts in enumerate(TS_LIST):
        c = curves[ts]
        col = "k" if ts == 0 else cmap(0.15 + 0.8 * i / (len(TS_LIST) - 1))
        lw = 2.2 if ts == 0 else 1.4
        lab = "event BDT only" if ts == 0 else f"shower cut {ts} first"
        o = np.argsort(c["eff"])
        axs[0].plot(c["eff"][o], c["pur"][o], color=col, lw=lw, label=lab)
        axs[1].plot(c["eff"][o], c["extpk"][o], color=col, lw=lw, label=lab)
    j0 = int(np.argmin(np.abs(curves[0.0]["te"] - OFFICIAL_TE)))
    axs[0].plot(curves[0.0]["eff"][j0], curves[0.0]["pur"][j0], "r*", ms=13,
                label="official (t_e=0.280)")
    axs[1].plot(curves[0.0]["eff"][j0], curves[0.0]["extpk"][j0], "r*", ms=13)
    axs[0].set(xlabel="signal efficiency (vs no-event-cut, t_s=0)",
               ylabel=f"near-peak purity ({int(PEAK[0])}-{int(PEAK[1])} MeV)",
               xlim=(0.72, 1.0))
    axs[1].set(xlabel="signal efficiency (vs no-event-cut, t_s=0)",
               ylabel="near-peak EXT (weighted events)", xlim=(0.72, 1.0))
    axs[1].set_yscale("symlog", linthresh=5)
    for ax in axs:
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7.5)
    fig.suptitle("staged cuts: per-shower cosmic BDT then event-level BDT")
    fig.tight_layout()
    fig.savefig(os.path.join(args.plots, "staged_bdt_sweep.png"), dpi=120)
    print(f">>> {args.plots}/staged_bdt_sweep.png / .npz")


if __name__ == "__main__":
    main()
