"""Hole-mask rate vs flash brightness, MC vs EXT vs beam data.

The saturated-PMT ("hole") mask fires on ~12% of MC reco-CC 2-photon events and
0% of run1 beam data / EXT cosmics. Taken at face value that is an alarming
data/MC asymmetry -- if the sim saturates and real data does not, the mask would
be correcting an artifact. But the reco-CC-2gamma samples are tiny for EXT
(N~24), and more importantly saturation only HAPPENS under a bright flash: a tube
must be hit hard to rail. EXT cosmics have dim in-time flashes, so a low rate
there could be pure brightness selection rather than a detector difference.

So compare the rate AT MATCHED BRIGHTNESS: hole rate as a function of total
observed PE. This uses every nu-slice cascade event (via the cached RSE->path
maps), not the CC-2gamma subset, so the stats are ~250x larger.

    python3 hole_rate_vs_brightness.py --cache <rse_*.npz> --label MC --dead 15
"""
import argparse
import os

import numpy as np
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "..", ".."))
from lartpc.flashmatch.saturation import find_saturated        # noqa: E402

# total observed PE bins; saturation should only switch on at the bright end
PE_BINS = np.array([0, 250, 500, 1000, 2000, 3000, 4000, 6000, 8000, 1e9])
PE_TOP = 10000.0   # representative x for the open-ended top bin


def scan(cache, dead, limit=None):
    # scanning ~100k cascade files takes minutes; cache the (total PE, n_holes)
    # pair next to the RSE map so re-plotting is instant
    save = cache.replace(".npz", "") + "_holerate.npz"
    if os.path.exists(save):
        z = np.load(save)
        return z["tot"], z["nh"]
    z = np.load(cache, allow_pickle=True)
    paths = [str(p) for p in z["paths"]]
    if limit:
        paths = paths[:limit]
    tot, nh = [], []
    for p in paths:
        try:
            with h5py.File(p, "r") as f:
                if "flash" not in f or "observed_pe" not in f["flash"]:
                    continue
                obs = f["flash/observed_pe"][()]
        except Exception:
            continue
        if not np.isfinite(obs).any() or obs.sum() <= 0:
            continue
        tot.append(float(obs.sum()))
        nh.append(len(find_saturated(obs, dead=dead, max_masked=None)))
    tot, nh = np.array(tot), np.array(nh)
    try:
        np.savez(save, tot=tot, nh=nh)
    except Exception:
        pass
    return tot, nh


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", nargs="+", required=True,
                    help="rse_*.npz cache(s), one per sample")
    ap.add_argument("--label", nargs="+", required=True)
    ap.add_argument("--dead", nargs="+", required=True,
                    help="per-sample dead opdets, '' for none (run1)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default="plots_saturation/hole_rate_vs_pe.png")
    args = ap.parse_args()
    assert len(args.cache) == len(args.label) == len(args.dead)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.8))
    for cache, label, ds in zip(args.cache, args.label, args.dead):
        dead = tuple(int(x) for x in ds.split(",") if x.strip() != "")
        tot, nh = scan(cache, dead, args.limit)
        if not len(tot):
            print("!!! %s: nothing" % label)
            continue
        print("\n== %s : %d events, dead=%s ==" % (label, len(tot), dead or "none"))
        print("   total observed PE: median %.0f, p90 %.0f, max %.0f"
              % (np.median(tot), np.percentile(tot, 90), tot.max()))
        print("   overall hole rate: %.2f%%" % (100.0 * (nh > 0).mean()))
        print("   %14s %8s %10s" % ("obs PE bin", "n_evt", "hole rate"))
        cx, cy, ce = [], [], []
        for lo, hi in zip(PE_BINS[:-1], PE_BINS[1:]):
            m = (tot >= lo) & (tot < hi)
            if m.sum() == 0:
                print("   %6.0f-%-7.0f %8d %10s" % (lo, min(hi, 99999), 0, "--"))
                continue
            r = (nh[m] > 0).mean()
            print("   %6.0f-%-7.0f %8d %9.2f%%" % (lo, min(hi, 99999), m.sum(),
                                                   100 * r))
            cx.append(0.5 * (lo + min(hi, PE_TOP)))
            cy.append(100 * r)
            ce.append(100 * np.sqrt(max(r * (1 - r), 1e-9) / m.sum()))
        ax[0].hist(np.clip(tot, 0, 12000), bins=60, histtype="step", lw=2,
                   label="%s (n=%d)" % (label, len(tot)))
        ax[1].errorbar(cx, cy, yerr=ce, marker="o", lw=2, capsize=3,
                       label="%s (n=%d)" % (label, len(tot)))
    ax[0].set_xlabel("total observed in-time PE")
    ax[0].set_ylabel("events")
    ax[0].set_yscale("log")
    ax[0].set_title("flash brightness -- do the samples even overlap?")
    ax[0].legend(fontsize=8)
    ax[1].set_xlabel("total observed in-time PE")
    ax[1].set_ylabel("events with >=1 hole candidate [%]")
    ax[1].set_title("hole-mask rate AT MATCHED BRIGHTNESS")
    ax[1].set_xlim(0, PE_TOP + 1000)
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.out, dpi=110)
    print("\n>>> wrote", args.out)


if __name__ == "__main__":
    main()
