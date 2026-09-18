"""Discriminant scan: for every candidate cut variable, at the CURRENT selection,
make the plots needed to CHOOSE a cut -- and the truth-resolved background
breakdown needed to choose WHICH background to attack.

Replaces the hand-edited `run_nuecc_cutflow.sh` loop (which had two live bugs:
`--mu-cut` passed twice, and `--vtxmu-lf-cut` hardcoded instead of using its
shell variable). One command now emits, per variable:

  var_<key>.png  2 panels -- LEFT: stacked prediction by sample + data, with the
                 current cut line; RIGHT: signal efficiency and purity vs the
                 threshold, with the best-purity-at-target-efficiency marked.
  cat_<key>.png  the same variable stacked by TRUTH CATEGORY of the reco'd
                 electron candidate (MC only) -- this is what says whether a
                 variable separates the background you actually have.

plus, once per run:

  bg_composition.png   weighted yield per truth category at the current
                       selection, signal and background side by side.
  scan_summary.md      cutflow + per-variable ranking, pasteable into the README.

    PYTHONPATH=./ python3 nue_cc_scan.py --nue-npz .. --bnb-npz .. --ext-npz .. \
        --data-npz .. --plots plots_scan/ [--flashchi2-cut 3.0] [--elconf-cut 9]
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nue_cc_common as C  # noqa: E402


def _quantile_bins(vals, nbins=40, lo_q=0.5, hi_q=99.5):
    v = np.concatenate([x for x in vals if len(x)]) if vals else np.array([])
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return None
    lo, hi = np.percentile(v, [lo_q, hi_q])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(v.min()), float(v.max())
    if hi <= lo:
        hi = lo + 1.0
    return np.linspace(lo, hi, nbins + 1)


def _int_bins(vals):
    v = np.concatenate([x for x in vals if len(x)]) if vals else np.array([])
    v = v[np.isfinite(v)]
    hi = int(np.nanmax(v)) if len(v) else 5
    return np.arange(-0.5, min(hi, 10) + 1.5, 1.0)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--nue-npz", required=True)
    ap.add_argument("--bnb-npz", required=True)
    ap.add_argument("--ext-npz", required=True)
    ap.add_argument("--data-npz", default=None)
    ap.add_argument("--plots", required=True)
    ap.add_argument("--pot", type=float, default=C.DEFAULT_POT)
    ap.add_argument("--vars", default=None,
                    help="comma-separated subset of variables to scan "
                         "(default: all in nue_cc_common.VARS)")
    ap.add_argument("--target-eff", type=float, default=0.5,
                    help="efficiency floor used when reporting the best "
                         "purity working point per variable")
    ap.add_argument("--nsteps", type=int, default=60,
                    help="threshold steps in the efficiency/purity scan")
    C.add_cut_args(ap)
    args = ap.parse_args()
    os.makedirs(args.plots, exist_ok=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cuts = C.cuts_from_args(args)
    tabs = C.load_tables(args.nue_npz, args.bnb_npz, args.ext_npz, args.data_npz,
                         ext_scale_override=args.ext_scale,
                         ext_rescale=args.ext_rescale)
    dat = tabs.get("data")
    d = {k: C.derive(v) for k, v in tabs.items()}
    label = C.cut_label(cuts)
    label = f"[{C.active_set()}] {label}"
    sbdt = C.uses_shower_bdt(cuts)
    print(f">>> selection: {label}")
    print(f">>> EXT per-event weight {C.ext_scale(shower_bdt_used=sbdt):.4f}"
          f"{' (rows>=100k hygiene half)' if sbdt and C.SAMPLE_SETS[C.active_set()]['hygiene'] else ''}")

    # ---- current-selection summary ---------------------------------------
    res = C.summarize(tabs, cuts, dat=dat, pot=args.pot)
    masks, wts = C.components(tabs, cuts)

    # ---- background composition by truth category ------------------------
    nue, bnb = tabs["nue"], tabs["bnb"]
    cat_nue, cat_bnb = C.truth_category(nue), C.truth_category(bnb)
    mn = masks[0]
    mb_bkg = (masks[1] | masks[2])          # numuCC + NC, nue-CC already vetoed
    wn, wb = np.asarray(nue["w"], float), np.asarray(bnb["w"], float)
    ext_w = C.ext_scale(shower_bdt_used=sbdt)

    rows = []
    for ci, name in enumerate(C.TRUTH_CATS):
        s = float(wn[mn & (cat_nue == ci)].sum())
        b = float(wb[mb_bkg & (cat_bnb == ci)].sum())
        if s > 0 or b > 0:
            rows.append((name, s, b))
    ext_tot = float(wts[3].sum())

    if rows:
        fig, ax = plt.subplots(figsize=(9, 4.8))
        x = np.arange(len(rows))
        ax.bar(x - 0.2, [r[1] for r in rows], 0.4, color="#d62728",
               label=f"nue sample ({sum(r[1] for r in rows):.1f})")
        ax.bar(x + 0.2, [r[2] for r in rows], 0.4, color="#1f77b4",
               label=f"bnb-nu background ({sum(r[2] for r in rows):.1f})")
        ax.set_xticks(x)
        ax.set_xticklabels([r[0] for r in rows], rotation=25, ha="right")
        ax.set_yscale("log")
        ax.set(ylabel=f"events / {args.pot:.1e} POT",
               title=f"What the reco'd electron candidate really is\n({label})"
                     f"   [EXT cosmic, no truth: {ext_tot:.1f}]")
        ax.grid(alpha=0.3, axis="y", which="both")
        ax.legend(fontsize=8)
        fig.tight_layout()
        p = os.path.join(args.plots, "bg_composition.png")
        fig.savefig(p, dpi=120); plt.close(fig); print(">>> wrote", p)

    # ---- per-variable scan ------------------------------------------------
    keys = ([k.strip() for k in args.vars.split(",")] if args.vars
            else [v[0] for v in C.VARS])
    specs = {v[0]: v for v in C.VARS}
    sig_all = float(wn[np.asarray(nue["is_nuecc_fv"]).astype(bool)].sum())
    ranking = []

    for key in keys:
        if key not in specs:
            print(f"  (skip unknown variable {key})")
            continue
        _, xlabel, direction, _ = specs[key]
        cutval = None
        for cname, (vkey, _) in C.CUT_SPECS.items():
            if vkey == key and cname in cuts:
                cutval = cuts[cname]

        sv = d["nue"][key][mn]
        uv = d["bnb"][key][masks[1]]
        cv = d["bnb"][key][masks[2]]
        ev = d["ext"][key][masks[3]]
        dv = d["data"][key][C.data_mask(dat, cuts)] if dat is not None else None
        obs = [sv, uv, cv, ev]
        ws = [wts[0], wts[1], wts[2], wts[3]]
        keep = [np.isfinite(o) for o in obs]
        obs = [o[m] for o, m in zip(obs, keep)]
        ws = [w[m] for w, m in zip(ws, keep)]
        if dv is not None:
            dv = dv[np.isfinite(dv)]
        if sum(len(o) for o in obs) == 0:
            print(f"  (skip {key}: no finite values)")
            continue

        is_count = direction == "max"
        # truth-only diagnostics (e.g. vtx_dist_true): EXT and data carry no
        # value, so no efficiency/purity scan -- the plot is for the eye only
        mc_only = key in C.MC_ONLY_VARS
        if mc_only:
            pos = np.concatenate([o[o > 0] for o in obs if len(o)])
            bins = (np.logspace(np.log10(max(pos.min(), 1e-2)),
                                np.log10(pos.max()), 41) if len(pos) else None)
        else:
            bins = (_int_bins(obs + ([dv] if dv is not None else []))
                    if is_count
                    else _quantile_bins(obs + ([dv] if dv is not None else [])))
        if bins is None:
            continue
        lo, hi = bins[0], bins[-1]
        clip = lambda z: np.clip(z, lo, hi - (hi - lo) * 1e-6)  # noqa: E731

        fig, (axL, axR) = plt.subplots(1, 2, figsize=(12.4, 4.6))
        axL.hist([clip(o) for o in obs], bins=bins, weights=ws, stacked=True,
                 color=C.COLORS,
                 label=[f"{c} ({w.sum():.1f})"
                        for c, w in zip(C.COMPONENTS, ws)])
        if dv is not None and len(dv):
            dh, _ = np.histogram(clip(dv), bins=bins)
            ctr = 0.5 * (bins[:-1] + bins[1:])
            axL.errorbar(ctr, dh, yerr=np.sqrt(np.clip(dh, 1, None)), fmt="ko",
                         ms=3.5, lw=1, capsize=0, label=f"data ({int(dh.sum())})")
        if cutval is not None:
            axL.axvline(cutval, color="k", ls="--", lw=1.2,
                        label=f"cut {cutval:g}")
        axL.set_yscale("log"); axL.set_ylim(0.03, None)
        if mc_only:
            axL.set_xscale("log")
        axL.set(xlabel=xlabel, ylabel=f"events / {args.pot:.1e} POT",
                title=f"{key} at current selection")
        axL.legend(fontsize=7); axL.grid(alpha=0.3, which="both")

        # --- efficiency / purity vs threshold ---
        allv = np.concatenate([o for o in obs if len(o)])
        if mc_only:
            steps = np.array([])
            axR.text(0.5, 0.5, "MC-only truth diagnostic:\nno efficiency/purity scan\n"
                     "(EXT and data carry no truth)", ha="center", va="center",
                     transform=axR.transAxes)
        elif is_count:
            steps = np.arange(0, min(int(np.nanmax(allv)), 6) + 1)
        else:
            steps = np.linspace(np.percentile(allv, 1),
                                np.percentile(allv, 99), args.nsteps)
        eff, pur = [], []
        sig_flag = np.asarray(nue["is_nuecc_fv"]).astype(bool)
        for th in steps:
            def pas(vals, direction=direction, th=th):
                if direction == "above":
                    return np.isfinite(vals) & (vals > th)
                if direction == "below":
                    return np.isfinite(vals) & (vals < th)
                return vals <= th
            ms = mn & pas(d["nue"][key])
            s_sel = float(wn[ms & sig_flag].sum())
            tot = (float(wn[ms].sum())
                   + float(wb[masks[1] & pas(d["bnb"][key])].sum())
                   + float(wb[masks[2] & pas(d["bnb"][key])].sum())
                   + ext_w * int((masks[3] & pas(d["ext"][key])).sum()))
            eff.append(s_sel / sig_all if sig_all > 0 else np.nan)
            pur.append(s_sel / tot if tot > 0 else np.nan)
        eff, pur = np.asarray(eff), np.asarray(pur)

        axR.plot(steps, eff, "o-", color="#d62728", ms=3, label="signal efficiency")
        axR.plot(steps, pur, "s-", color="#1f77b4", ms=3, label="purity")
        ok = np.isfinite(pur) & (eff >= args.target_eff)
        best = None
        if ok.any():
            j = int(np.nanargmax(np.where(ok, pur, -np.inf)))
            best = (float(steps[j]), float(pur[j]), float(eff[j]))
            axR.axvline(steps[j], color="k", ls=":", lw=1.2,
                        label=f"best @eff>={args.target_eff:g}: "
                              f"{steps[j]:.3g}\n(pur {pur[j]:.3f}, eff {eff[j]:.3f})")
        if cutval is not None:
            axR.axvline(cutval, color="k", ls="--", lw=1.0)
        axR.set(xlabel=f"{key} threshold ({direction})", ylabel="fraction",
                ylim=(0, 1.02), title="efficiency / purity vs threshold")
        axR.grid(alpha=0.3); axR.legend(fontsize=7.5)
        if mc_only:
            axR.set_axis_off()
            if axR.get_legend() is not None:
                axR.get_legend().remove()
        fig.tight_layout()
        p = os.path.join(args.plots, f"var_{key}.png")
        fig.savefig(p, dpi=115); plt.close(fig); print(">>> wrote", p)
        if not mc_only:
            ranking.append((key, best))

        # --- same variable, stacked by TRUTH CATEGORY (MC only) ---
        cobs, cws, clab, ccol = [], [], [], []
        for ci, name in enumerate(C.TRUTH_CATS):
            vv = np.concatenate([d["nue"][key][mn & (cat_nue == ci)],
                                 d["bnb"][key][mb_bkg & (cat_bnb == ci)]])
            ww = np.concatenate([wn[mn & (cat_nue == ci)],
                                 wb[mb_bkg & (cat_bnb == ci)]])
            good = np.isfinite(vv)
            if not good.any():
                continue
            cobs.append(vv[good]); cws.append(ww[good])
            clab.append(f"{name} ({ww[good].sum():.1f})")
            ccol.append(C.TRUTH_COLORS[ci])
        if cobs:
            fig, ax = plt.subplots(figsize=(7.6, 4.8))
            ax.hist([clip(o) for o in cobs], bins=bins, weights=cws,
                    stacked=True, color=ccol, label=clab)
            if cutval is not None:
                ax.axvline(cutval, color="k", ls="--", lw=1.2)
            ax.set_yscale("log"); ax.set_ylim(0.03, None)
            if mc_only:
                ax.set_xscale("log")
            ax.set(xlabel=xlabel, ylabel=f"events / {args.pot:.1e} POT",
                   title=f"{key} by TRUTH category of the e-candidate\n({label})")
            ax.legend(fontsize=6.5); ax.grid(alpha=0.3, which="both")
            fig.tight_layout()
            p = os.path.join(args.plots, f"cat_{key}.png")
            fig.savefig(p, dpi=115); plt.close(fig); print(">>> wrote", p)

    # ---- summary markdown -------------------------------------------------
    md = [f"# nue CC scan -- {label}", "",
          f"POT {args.pot:.2e}; EXT per-event weight "
          f"{C.ext_scale(shower_bdt_used=sbdt):.4f}", "",
          "## Yields at this selection", "",
          "| component | events |", "|---|---|"]
    for c, wv in zip(C.COMPONENTS, wts):
        md.append(f"| {c} | {wv.sum():.2f} |")
    md.append(f"| **total pred** | **{res['pred']:.2f}** |")
    if "data" in res:
        md.append(f"| bnb5e19 data | {res['data']} "
                  f"(data/pred {res['data_over_pred']:.2f}) |")
    md += ["", f"purity **{res['purity']:.3f}**, efficiency "
           f"**{res['eff']:.3f}**  (LANTERN 0.90 / 0.55)", ""]
    if rows:
        md += ["## Candidate truth composition", "",
               "| category | nue sample | bnb-nu bkg |", "|---|---|---|"]
        for name, s, b in rows:
            md.append(f"| {name} | {s:.2f} | {b:.2f} |")
        md.append(f"| EXT cosmic (no truth) | - | {ext_tot:.2f} |")
        md.append("")
    md += [f"## Best single-cut working points (efficiency >= {args.target_eff:g})",
           "", "| variable | threshold | purity | efficiency |", "|---|---|---|---|"]
    for key, best in ranking:
        if best is None:
            md.append(f"| {key} | - | - | (never reaches the efficiency floor) |")
        else:
            md.append(f"| {key} | {best[0]:.4g} | {best[1]:.3f} | {best[2]:.3f} |")
    p = os.path.join(args.plots, "scan_summary.md")
    open(p, "w").write("\n".join(md) + "\n")
    print(">>> wrote", p)

    print(f"\n== BEST SINGLE CUTS (eff >= {args.target_eff:g}) ==")
    for key, best in sorted([r for r in ranking if r[1]],
                            key=lambda r: -r[1][1]):
        print(f"  {key:20s} {best[0]:10.4g}  purity {best[1]:.3f}  "
              f"eff {best[2]:.3f}")


if __name__ == "__main__":
    main()
