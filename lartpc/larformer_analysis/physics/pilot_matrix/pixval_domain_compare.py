"""Pixel-value domain comparison: corsika val sample vs run3b data-overlay
pilot, split by TRUE-NU (origin==1, simulated in both samples — should agree
if the MC domains match) vs REST (corsika MC cosmics vs data cosmics+noise).

Two views per (category x plane):
  raw pixval on a log axis (frac<=0 reported in the legend, excluded from
  the log hist), and the LogTransform_v6 value the deghoster actually sees
  (clip(x,0,1000) -> log10 -> [-1,1]; the <=0 population lands at -1).

Outputs: summary txt + two PNGs to --out-dir.

    PYTHONPATH=./ python3 pixval_domain_compare.py \
        --val-list exp/deghost_ptv3decoder_v2_fullevent_ft/ft_val_2k.txt \
        --overlay-list lartpc/larformer_reco/inputlists/merged_sp_mcc9_bnbnu_satfix_pilot10k.txt \
        --out-dir lartpc/larformer_reco/output/pilot_ntuples/pixval_domain
"""
import argparse
import os

import numpy as np
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PLANES = ("U", "V", "Y")
CATS = ("true nu", "rest")
COLORS = {"val": "#5c5c5c", "overlay": "#1f77b4"}
LOG_MIN, LOG_MAX = 0.01, 1000.0


def v6(x):
    y0, y1 = np.log10(LOG_MIN), np.log10(LOG_MAX + LOG_MIN)
    return 2 * (np.log10(np.clip(x, 0, LOG_MAX) + LOG_MIN) - y0) / (y1 - y0) - 1


def collect(list_path, n_events, seed):
    files = [l.strip() for l in open(list_path) if l.strip()]
    rng = np.random.default_rng(seed)
    files = [files[i] for i in rng.permutation(len(files))[:n_events]]
    acc = {c: [] for c in CATS}
    n_read = 0
    for p in files:
        try:
            with h5py.File(p, "r") as f:
                td = f["entry_0/triplet_data"]
                pix = td["pixval"][()].astype(np.float64)
                origin = td["origin"][()].astype(np.int64)
        except Exception:
            continue
        n_read += 1
        nu = origin == 1
        acc["true nu"].append(pix[nu])
        acc["rest"].append(pix[~nu])
    return {c: (np.concatenate(v, 0) if v else np.zeros((0, 3)))
            for c, v in acc.items()}, n_read


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--val-list", required=True)
    ap.add_argument("--overlay-list", required=True)
    ap.add_argument("--n-events", type=int, default=200)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    data, nread = {}, {}
    for tag, lp in (("val", args.val_list), ("overlay", args.overlay_list)):
        data[tag], nread[tag] = collect(lp, args.n_events, args.seed)
        print(f">>> {tag}: {nread[tag]} events  "
              + "  ".join(f"{c}: {len(data[tag][c])} SPs" for c in CATS))

    # ---- summary table ------------------------------------------------------
    lines = [f"pixval domain comparison — val(corsika) {nread['val']} events "
             f"vs overlay {nread['overlay']} events (origin==1 = true nu)", ""]
    lines.append(f"{'cat':9s}{'pl':>3s}   {'frac<=0':>16s}   {'median(>0)':>18s}"
                 f"   {'p95(>0)':>18s}")
    lines.append(f"{'':12s}   {'val':>7s}{'ovl':>8s}   {'val':>8s}{'ovl':>9s}"
                 f"   {'val':>8s}{'ovl':>9s}")
    for c in CATS:
        for ip, pl in enumerate(PLANES):
            row = []
            for tag in ("val", "overlay"):
                x = data[tag][c][:, ip]
                pos = x[x > 0]
                row.append((float((x <= 0).mean()) if len(x) else np.nan,
                            float(np.median(pos)) if len(pos) else np.nan,
                            float(np.percentile(pos, 95)) if len(pos) else np.nan))
            lines.append(f"{c:9s}{pl:>3s}   {row[0][0]:7.3f}{row[1][0]:8.3f}"
                         f"   {row[0][1]:8.1f}{row[1][1]:9.1f}"
                         f"   {row[0][2]:8.1f}{row[1][2]:9.1f}")
    txt = "\n".join(lines)
    print("\n" + txt)
    with open(os.path.join(args.out_dir, "summary.txt"), "w") as f:
        f.write(txt + "\n")

    # ---- fig 1: raw pixval, log axis ---------------------------------------
    bins = np.logspace(np.log10(0.5), np.log10(2000.0), 60)
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.2), sharey="row")
    for ic, c in enumerate(CATS):
        for ip, pl in enumerate(PLANES):
            ax = axes[ic][ip]
            for tag in ("val", "overlay"):
                x = data[tag][c][:, ip]
                pos = x[x > 0]
                ax.hist(np.clip(pos, bins[0], bins[-1] * (1 - 1e-9)),
                        bins=bins, density=True, histtype="step", lw=1.8,
                        color=COLORS[tag],
                        label=f"{tag}  (≤0: {(x <= 0).mean():.2f})")
            ax.set_xscale("log")
            ax.set_title(f"{c} — plane {pl}", fontsize=10)
            if ic == 1:
                ax.set_xlabel("pixval [ADC]")
            if ip == 0:
                ax.set_ylabel("density (pixval>0)")
            ax.legend(frameon=False, fontsize=8)
            ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("pixval distributions: corsika val vs data-overlay pilot",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "pixval_raw.png"), dpi=140)

    # ---- fig 2: what the network sees (v6-transformed, incl. -1 spike) -----
    tb = np.linspace(-1.02, 1.0, 62)
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.2), sharey="row")
    for ic, c in enumerate(CATS):
        for ip, pl in enumerate(PLANES):
            ax = axes[ic][ip]
            for tag in ("val", "overlay"):
                x = data[tag][c][:, ip]
                ax.hist(v6(x), bins=tb, density=True, histtype="step",
                        lw=1.8, color=COLORS[tag], label=tag)
            ax.set_title(f"{c} — plane {pl}", fontsize=10)
            if ic == 1:
                ax.set_xlabel("LogTransform_v6(pixval)  (network input)")
            if ip == 0:
                ax.set_ylabel("density")
            ax.legend(frameon=False, fontsize=8)
            ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("deghoster strength inputs: corsika val vs data-overlay pilot",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "pixval_network_input.png"), dpi=140)
    print(f">>> wrote {args.out_dir}")


if __name__ == "__main__":
    main()
