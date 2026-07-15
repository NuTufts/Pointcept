"""Flashmatch-path impact study: OLD (buggy-flash) vs NEW (fixed-flash) cascade.

The dead-PMT mask + per-run gamma change the per-slice flash chi2, which changes
chi2_rank, which changes WHICH slice becomes the flashmatch stream (rank 1) and
whether the nu union is also the best match (stream 'nu,flashmatch'). This
quantifies those changes on the same events (matched by run/subrun/event):

  - nu-slice chi2 old vs new (should drop: run3 dead-mask, run1 gamma*0.8);
  - nu-slice chi2_rank old vs new (does the reco nu slice rank better?);
  - is-nu-the-best-flash-match (rank1==nu) old vs new -> the nu==fm union rate;
  - # slices that PASS the oob/rank gate (rank>0) old vs new -> more fm
    candidates now that the dead-PMT no longer inflates every slice's chi2;
  - best-slice label churn (nu <-> cosmic).

    python3 flashmatch_impact.py --old-cascade <dir> --new-cascade <dir> \
        --sample-tag ... --plots plots/
"""
import argparse
import os
import sys

import numpy as np
import h5py

_PI0 = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    "..", "physics", "pi0mass_peak")
sys.path.insert(0, _PI0)
from flash_correction import rse_map                # noqa: E402


def _cache(cascade_dir):
    p = os.path.join(_PI0, "rse_" + os.path.basename(
        os.path.dirname(cascade_dir.rstrip("/"))) + "_"
        + os.path.basename(cascade_dir.rstrip("/")) + ".npz")
    return p                                          # may or may not exist


def nu_slice_info(path):
    """dict(nu_chi2, nu_rank, best_is_nu, n_ranked) for the cascade nu slice."""
    try:
        with h5py.File(path, "r") as f:
            if "slices" not in f:
                return None
            labs = [l.decode() if isinstance(l, bytes) else str(l)
                    for l in f["slices/label"][()]]
            if "nu" not in labs:
                return None
            j = labs.index("nu")
            chi2 = np.asarray(f["slices/chi2"][()], float)
            rank = np.asarray(f["slices/chi2_rank"][()], int)
            r1 = np.nonzero(rank == 1)[0]
            return dict(nu_chi2=float(chi2[j]), nu_rank=int(rank[j]),
                        best_is_nu=bool(len(r1) and labs[int(r1[0])] == "nu"),
                        n_ranked=int((rank > 0).sum()))
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--old-cascade", required=True)
    ap.add_argument("--new-cascade", required=True)
    ap.add_argument("--sample-tag", required=True)
    ap.add_argument("--plots", required=True)
    args = ap.parse_args()
    os.makedirs(args.plots, exist_ok=True)

    old_map = rse_map(args.old_cascade, _cache(args.old_cascade))
    new_map = rse_map(args.new_cascade, _cache(args.new_cascade))
    keys = sorted(set(old_map) & set(new_map))
    print(f">>> {args.sample_tag}: {len(old_map)} old / {len(new_map)} new "
          f"nu-slice cascade files; {len(keys)} matched by RSE")

    oc, nc, orank, nrank, obn, nbn, onr, nnr = ([] for _ in range(8))
    for k in keys:
        o = nu_slice_info(old_map[k]); n = nu_slice_info(new_map[k])
        if o is None or n is None:
            continue
        oc.append(o["nu_chi2"]); nc.append(n["nu_chi2"])
        orank.append(o["nu_rank"]); nrank.append(n["nu_rank"])
        obn.append(o["best_is_nu"]); nbn.append(n["best_is_nu"])
        onr.append(o["n_ranked"]); nnr.append(n["n_ranked"])
    oc = np.array(oc); nc = np.array(nc)
    orank = np.array(orank); nrank = np.array(nrank)
    obn = np.array(obn); nbn = np.array(nbn)
    onr = np.array(onr); nnr = np.array(nnr)
    N = len(oc)
    print(f">>> N={N} events with a nu slice in both")
    print(f"  nu-slice chi2 median:  old {np.median(oc):.0f} -> new "
          f"{np.median(nc):.0f}")
    print(f"  nu is best flash match: old {obn.mean():.1%} -> new {nbn.mean():.1%}"
          f"  (nu==fm union rate)")
    print(f"  nu-slice rank==1:       old {(orank==1).mean():.1%} -> new "
          f"{(nrank==1).mean():.1%}")
    print(f"  # ranked slices/event median: old {int(np.median(onr))} -> new "
          f"{int(np.median(nnr))}")
    print(f"  best-slice label changed (nu<->cosmic): {(obn!=nbn).mean():.1%} "
          "of events")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.6))
    lb = np.linspace(0, 8, 41)
    lg = lambda x: np.clip(np.log10(np.clip(x, 1, None)), 0, 8)
    ax[0].hist(lg(oc), bins=lb, histtype="step", lw=2, color="0.45", ls="--",
               label=f"old  med {np.median(oc):.0f}")
    ax[0].hist(lg(nc), bins=lb, histtype="step", lw=2, color="#d62728",
               label=f"new  med {np.median(nc):.0f}")
    ax[0].set(xlabel=r"$\log_{10}$ nu-slice flash $\chi^2$", ylabel="events",
              title=f"{args.sample_tag}: nu-slice chi2 old vs new (N={N})")
    ax[0].legend(fontsize=9); ax[0].grid(alpha=0.3)
    # is-nu-best + rank1 + churn summary bars
    cats = ["nu is\nbest match", "nu rank==1", "best label\nchanged"]
    ov = [obn.mean(), (orank == 1).mean(), 0]
    nv = [nbn.mean(), (nrank == 1).mean(), (obn != nbn).mean()]
    x = np.arange(3)
    ax[1].bar(x - 0.2, ov, 0.4, color="0.6", label="old")
    ax[1].bar(x + 0.2, nv, 0.4, color="#d62728", label="new")
    ax[1].set(xticks=x, ylim=(0, 1), ylabel="fraction of events",
              title="flashmatch-stream selection")
    ax[1].set_xticklabels(cats, fontsize=8)
    ax[1].legend(fontsize=9); ax[1].grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(f"{args.plots}/flashmatch_impact_{args.sample_tag}.png", dpi=110)
    plt.close(fig)
    print(f">>> plots -> {args.plots}/flashmatch_impact_{args.sample_tag}.png")


if __name__ == "__main__":
    main()
