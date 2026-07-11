"""Histogram-density log-likelihood-ratio for shower-vertex attachment.

Consumes the shower_attachment_study record table. Per size bin (n_pts),
estimates 1D histogram densities P(x|correct) and P(x|incorrect) for each
decision variable, and scores pairs with the summed log ratio (naive Bayes
WITHIN a size bin -- the size-conditioning absorbs the dominant correlation:
direction quality collapses for small showers). Honest evaluation: densities
fit on even-event pairs, ROC/working points reported on odd-event pairs.

Variables (transformed for histogram friendliness):
  cosine            trunk back-point cosine
  pca_cosine        full-shower 1st-PCA cosine
  log10(sin_tk)     trunk pointing-error sine = impact/gap (geometry-decoupled:
                    a fixed impact cut at large gap demands unattainable
                    pointing accuracy)
  log10(sin_pca)    pca_impact / gap
  log10(gap)        conversion gap [cm]
  trunk_q           trunk PCA elongation

Figure of merit (user-defined): fraction of correct pairs attached vs
fraction of incorrect pairs attached. Reports the LLR working points matched
to the current hard cuts' false-attach rate and correct-attach rate, plus
far-conversion (>55 cm) and small-shower recovery at the matched point.

    PYTHONPATH=./ python3 lartpc/larformer_reco/eval/fit_attachment_likelihood.py \
        --records lartpc/larformer_reco/results_shower_attachment_<TAG>.npz \
        --plots lartpc/larformer_reco/plots/shower_attachment_<TAG>/
"""
import argparse
import os

import numpy as np

EPS = 1e-3
SIZE_EDGES = [10, 60, 250, 1e9]          # n_pts bins: small / medium / large


def _vars(d):
    gap = np.maximum(d["gap"], 0.1)
    out = {
        "cosine": (d["cosine"], np.linspace(-1, 1, 41)),
        "pca_cosine": (d["pca_cosine"], np.linspace(-1, 1, 41)),
        "log_sin_tk": (np.log10(np.clip(d["impact"] / gap, EPS, 2.0)),
                       np.linspace(-3, 0.3, 41)),
        "log_sin_pca": (np.log10(np.clip(d["pca_impact"] / gap, EPS, 20.0)),
                        np.linspace(-3, 1.3, 41)),
        "log_gap": (np.log10(np.clip(gap, 0.1, 500)),
                    np.linspace(-1, 2.7, 41)),
        "trunk_q": (d["trunk_q"], np.linspace(0.3, 1.0, 36)),
    }
    if "cone_qfrac" in d:                    # cone-shape prior variables
        out["cone_qfrac"] = (d["cone_qfrac"], np.linspace(0, 1, 41))
        out["ang_rms"] = (np.clip(d["ang_rms"], 0, 120),
                          np.linspace(0, 120, 41))
    return out


def fit_llr(d, train, label="correct"):
    """Per size-bin, per-variable histogram density ratios -> LLR scorer."""
    V = _vars(d)
    ok = d[label] > 0
    size = d["n_pts"]
    tables = []                             # [size_bin][var] = (bins, logratio)
    for lo, hi in zip(SIZE_EDGES[:-1], SIZE_EDGES[1:]):
        sb = train & (size >= lo) & (size < hi)
        tabs = {}
        for name, (x, bins) in V.items():
            hs, _ = np.histogram(x[sb & ok], bins)
            hb, _ = np.histogram(x[sb & ~ok], bins)
            # additive smoothing so empty bins don't produce +-inf
            ps = (hs + 0.5) / (hs.sum() + 0.5 * len(hs))
            pb = (hb + 0.5) / (hb.sum() + 0.5 * len(hb))
            tabs[name] = (bins, np.log(ps / pb))
        tables.append(tabs)

    def score(idx):
        s = np.zeros(idx.sum(), np.float64)
        xs = {name: x[idx] for name, (x, _) in V.items()}
        sz = size[idx]
        for bi, (lo, hi) in enumerate(zip(SIZE_EDGES[:-1], SIZE_EDGES[1:])):
            m = (sz >= lo) & (sz < hi)
            for name, (bins, lr) in tables[bi].items():
                x = xs[name][m]
                fin = np.isfinite(x)
                j = np.clip(np.digitize(x[fin], bins) - 1, 0, len(lr) - 1)
                add = np.zeros(m.sum())
                add[fin] = lr[j]              # nan variable -> no contribution
                s[m] += add
        return s
    return score, tables


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--records", required=True)
    ap.add_argument("--plots", default=None)
    ap.add_argument("--label", default="correct_origin",
                    help="truth label column: correct_origin (cp within "
                         "vtx-cut of the TRUE shower origin; works for "
                         "track-end cps) or correct (GT nu vertex only)")
    ap.add_argument("--save-tables", default=None,
                    help="npz path for the fitted LLR tables + matched "
                         "threshold (consumed by trajfit.shower_attach_llr)")
    args = ap.parse_args()

    d = dict(np.load(args.records))
    if args.label not in d:
        args.label = "correct"
    ok = d[args.label] > 0
    train = (d["ev"].astype(np.int64) % 2) == 0
    test = ~train
    print(f">>> {len(ok)} pairs; train {int(train.sum())} / test "
          f"{int(test.sum())}; test correct[{args.label}] "
          f"{int((test & ok).sum())}")

    score, tables = fit_llr(d, train, label=args.label)
    llr = np.full(len(ok), np.nan)
    llr[test] = score(test)
    tok, tbad = test & ok, test & ~ok
    cur = d["cur_pass"] > 0

    # ROC over LLR threshold (figure of merit: frac correct vs frac incorrect)
    ths = np.percentile(llr[test], np.linspace(0, 100, 201))
    eff = np.array([(llr[tok] >= t).mean() for t in ths])
    fal = np.array([(llr[tbad] >= t).mean() for t in ths])
    cur_eff, cur_fal = cur[tok].mean(), cur[tbad].mean()

    # working points matched to the current cuts
    i_f = int(np.argmin(np.abs(fal - cur_fal)))
    i_e = int(np.argmin(np.abs(eff - cur_eff)))
    print(f"\n== LLR vs current hard cuts (test half) ==")
    print(f"  current cuts        : correct {cur_eff:.3f} | incorrect "
          f"{cur_fal:.3f}")
    print(f"  LLR @ same false    : correct {eff[i_f]:.3f} | incorrect "
          f"{fal[i_f]:.3f}  (thr {ths[i_f]:+.2f})")
    print(f"  LLR @ same correct  : correct {eff[i_e]:.3f} | incorrect "
          f"{fal[i_e]:.3f}  (thr {ths[i_e]:+.2f})")

    thr = ths[i_f]                          # matched-false-rate working point
    at = llr >= thr
    cd = d["conv_dist"]
    far = tok & np.isfinite(cd) & (cd > 55)
    small = tok & (d["n_pts"] < 60)
    print(f"\n== recovery at the matched-false-rate point ==")
    print(f"  far-converting (>55cm) correct pairs: N={int(far.sum())} | "
          f"current {cur[far].mean():.3f} -> LLR {at[far].mean():.3f}")
    print(f"  small showers (<60 pts) correct     : N={int(small.sum())} | "
          f"current {cur[small].mean():.3f} -> LLR {at[small].mean():.3f}")
    if "cp_kind" in d:
        for kk, kn in ((0, "vertex"), (1, "track-end")):
            km = test & (d["cp_kind"] == kk)
            if km.sum():
                kok = km & ok
                print(f"  cp kind {kn:9s}: pairs {int(km.sum()):6d} | "
                      f"correct {int(kok.sum()):5d} | LLR attaches "
                      f"{at[kok].mean() if kok.sum() else 0:.3f} of correct")
    if "origin_dist" in d:
        od = d["origin_dist"]
        m_at = test & at & np.isfinite(od)
        m_cu = test & cur & np.isfinite(od)
        print(f"\n== attached-point vs TRUE ORIGIN (secondary FOM) ==")
        for nm, mm in (("current", m_cu), ("LLR", m_at)):
            if mm.sum():
                print(f"  {nm:8s}: median |cp - true origin| "
                      f"{np.median(od[mm]):6.2f} cm | <3cm "
                      f"{(od[mm] < 3).mean():.3f} (N={int(mm.sum())})")
    if args.save_tables:
        payload = {"size_edges": np.asarray(SIZE_EDGES, np.float64),
                   "thr_matched_false": np.float64(thr),
                   "label": np.array(args.label),
                   "var_names": np.array(sorted(tables[0].keys()))}
        for bi, tabs in enumerate(tables):
            for name, (bins, lr) in tabs.items():
                payload[f"bins_{bi}_{name}"] = bins
                payload[f"lr_{bi}_{name}"] = lr
        np.savez(args.save_tables, **payload)
        print(f">>> LLR tables -> {args.save_tables}")

    if not args.plots:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(args.plots, exist_ok=True)

    fig, ax = plt.subplots(figsize=(5.4, 4.6))
    ax.plot(fal, eff, "-", lw=1.8, label="LLR (size-binned naive Bayes)")
    ax.plot([cur_fal], [cur_eff], "r*", ms=14, label="current hard cuts")
    ax.set(xlabel="fraction of INCORRECT pairs attached",
           ylabel="fraction of CORRECT pairs attached",
           title="attachment ROC (test half)", xlim=(0, 1), ylim=(0, 1))
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(f"{args.plots}/llr_roc.png", dpi=110)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4.2))
    bins = np.linspace(np.nanpercentile(llr, 0.5),
                       np.nanpercentile(llr, 99.5), 61)
    ax.hist(llr[tok], bins=bins, histtype="stepfilled", alpha=0.45,
            density=True, label=f"correct (N={int(tok.sum())})")
    ax.hist(llr[tbad], bins=bins, histtype="step", lw=1.8, density=True,
            label=f"incorrect (N={int(tbad.sum())})")
    ax.axvline(thr, color="r", ls="--", lw=1, label="matched-false thr")
    ax.set(xlabel="log-likelihood ratio", ylabel="density",
           title="LLR separation (test half)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{args.plots}/llr_dist.png", dpi=110)
    plt.close(fig)

    # efficiency vs conversion distance at the matched-false working point
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    edges = np.array([0, 5, 10, 20, 35, 55, 80, 120, 200])
    ctr = 0.5 * (edges[:-1] + edges[1:])
    for sel, lab, st in ((cur, "current cuts", "s--"),
                         (at, "LLR @ matched false rate", "o-")):
        e = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            b = tok & np.isfinite(cd) & (cd >= lo) & (cd < hi)
            e.append(sel[b].mean() if b.sum() > 10 else np.nan)
        ax.plot(ctr, e, st, ms=4, label=lab)
    ax.set(xlabel="true conversion distance [cm]",
           ylabel="fraction of correct pairs attached", ylim=(0, 1.05),
           title="correct-attachment efficiency vs conversion distance")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{args.plots}/llr_eff_vs_convdist.png", dpi=110)
    plt.close(fig)
    print(f">>> plots -> {args.plots}")


if __name__ == "__main__":
    main()
