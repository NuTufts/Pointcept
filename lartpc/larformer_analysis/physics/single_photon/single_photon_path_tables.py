"""Efficiency / purity tables for the single-photon selection with the
VERTEX sample (candidate attached to the primary in-FV nu vertex, cosmic BDT)
and the NO-VERTEX sample (vertex-less or non-FV-vertex candidate, vertex-free
BDT) kept SEPARATE. Same selection machinery / options as
single_photon_selection.py (--novtx required; the path is decided by the
leading candidate at the "1 photon" step).

Motivation (2026-09-06): rather than merging the two paths into one sample,
use the no-vertex sample as the entering-photon sideband / constraint (and a
second place to look for an excess) and the vertex sample as the in-FV
1g+X / 1g0X sample where MiniBooNE-invisible p / pi can be seen.

Outputs (--plots): path_tables.txt with, per sample and per step from S2 on:
signal by category (in-FV 1g0X, in-FV 1g+X, entering no-mu, entering CC-mu)
with per-category efficiency (denominator = ALL true events of that category),
MC background by category, EXT, prediction, data, purity (all signal),
in-FV purity, entering purity, and the in-FV : entering signal ratio; plus
stacked candidate-energy spectra for each sample at the final step.

    python3 single_photon_path_tables.py --mc-ntuple ... --ext-ntuple ... \
        --data-ntuple ... --novtx --mc-odd-only --plots plots_v2_s1ep2p8_novtx/paths
"""
import argparse
import os
import sys

import numpy as np
import awkward as ak

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import single_photon_selection as SPS            # noqa: E402
from single_photon_diagnostics import add_args   # noqa: E402


def run(ntuple, is_mc, args):
    a, pot = SPS.load(ntuple, is_mc, args.novtx)
    R = SPS.reco_tables(a, args)
    R["novtx"] = args.novtx
    names, S, E, _, _, path = SPS.build_steps(R, args.shower_bdt_min, args.flashchi2_cut,
                                              args.bdt_after_multiplicity)
    return a, pot, names, S, E, path


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    add_args(ap)
    args = ap.parse_args()
    if not args.novtx:
        sys.exit("--novtx is required (the path split only exists there)")
    os.makedirs(args.plots, exist_ok=True)

    a, pot, names, S, E, path = run(args.mc_ntuple, True, args)
    w0 = ak.to_numpy(a["xsecWeight"]).astype(np.float64)
    w = np.where(w0 > 0, w0, 0.0) * (args.pot / pot)
    if args.mc_odd_only:
        odd = ak.to_numpy(a["event"]) % 2 == 1
        w = np.where(odd, 2.0 * w, 0.0)
    T = SPS.truth_tables(a); cat = T["cat"]; sig = T["sig_incl"]
    del a
    SX = EX = PX = None; wx = None
    if args.ext_ntuple:
        _, _, _, SX, EX, PX = run(args.ext_ntuple, False, args)
        wx = np.zeros(len(PX)); wx[args.ext_row_min:] = args.ext_scale
    SD = ED = PD = None
    if args.data_ntuple:
        _, _, _, SD, ED, PD = run(args.data_ntuple, False, args)

    CATS, NSIG, NMC = SPS.CATS, SPS.N_SIG_CAT, SPS.N_MC_CAT
    den = {c: w[cat == c].sum() for c in range(NSIG)}
    infv = np.isin(cat, [0, 1]); enter = np.isin(cat, [2, 3])
    lines = [f"denominators (w @ {args.pot:.1e} POT): "
             + " | ".join(f"{CATS[c]} {den[c]:.1f}" for c in range(NSIG))]
    k2 = 2                                     # first step with a candidate
    for pv, plab in ((0, "VERTEX sample (candidate on the primary in-FV vertex, cosmic BDT)"),
                     (1, "NO-VERTEX sample (vertex-less / non-FV-vertex candidate, vertex-free BDT)")):
        lines.append(f"\n===== {plab} =====")
        hdr = (f"{'step':<14}" + "".join(f"{CATS[c][:12]:>13}{'eff':>6}" for c in range(NSIG))
               + f"{'sigTot':>8}{'MCbkg':>8}{'EXT':>8}{'pred':>8}{'data':>6}{'d/p':>6}"
               f"{'pur':>6}{'purFV':>7}{'purEnt':>7}{'FV:Ent':>7}")
        lines.append(hdr)
        for k in range(k2, len(names)):
            m = S[k] & (path == pv)
            cells = ""
            for c in range(NSIG):
                s = w[m & (cat == c)].sum()
                cells += f"{s:13.1f}{s / max(den[c], 1e-9):6.3f}"
            st = w[m & sig].sum(); b = w[m & ~sig].sum()
            e = wx[SX[k] & (PX == pv)].sum() if SX is not None else 0.0
            d = int((SD[k] & (PD == pv)).sum()) if SD is not None else 0
            tot = max(st + b + e, 1e-9)
            sfv, sen = w[m & infv].sum(), w[m & enter].sum()
            lines.append(f"{names[k]:<14}" + cells
                         + f"{st:8.1f}{b:8.1f}{e:8.1f}{tot:8.1f}{d:6d}{d / tot:6.2f}"
                         f"{st / tot:6.3f}{sfv / tot:7.3f}{sen / tot:7.3f}"
                         f"{sfv / max(sen, 1e-9):7.2f}")
        m = S[-1] & (path == pv)
        lines.append("  final MC background composition: " + " | ".join(
            f"{CATS[c][5:]} {w[m & (cat == c)].sum():.1f}" for c in range(NSIG, NMC)
            if w[m & (cat == c)].sum() > 0))
    txt = "\n".join(lines)
    print(txt)
    with open(os.path.join(args.plots, "path_tables.txt"), "w") as fh:
        fh.write(txt + "\n")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    bins = np.linspace(0, 1000, 41)
    for pv, plab, fn in ((0, "vertex sample", "vertex"), (1, "no-vertex sample", "novertex")):
        m = S[-1] & (path == pv) & np.isfinite(E)
        data = [np.clip(E[m & (cat == c)], 0, 999.9) for c in range(NMC)]
        ws = [w[m & (cat == c)] for c in range(NMC)]
        cols = list(SPS.CAT_COLORS[:NMC])
        if SX is not None:
            mx = SX[-1] & (PX == pv) & np.isfinite(EX)
            data.append(np.clip(EX[mx], 0, 999.9)); ws.append(wx[mx]); cols.append(SPS.CAT_COLORS[NMC])
        fig, ax = plt.subplots(figsize=(7.4, 4.8))
        ax.hist(data, bins=bins, weights=ws, stacked=True, color=cols,
                label=[f"{CATS[c]} ({ws[c].sum():.1f})" for c in range(len(ws))])
        if SD is not None:
            md = SD[-1] & (PD == pv) & np.isfinite(ED)
            hd, _ = np.histogram(np.clip(ED[md], 0, 999.9), bins=bins)
            ctr = 0.5 * (bins[:-1] + bins[1:])
            ax.errorbar(ctr[hd > 0], hd[hd > 0], yerr=np.sqrt(hd[hd > 0]), fmt="ko", ms=3.5,
                        label=f"beam data ({int(hd.sum())})")
        ax.set(xlabel="reco photon energy [MeV]", ylabel=f"events / {args.pot:.1e} POT",
               title=f"candidate photon energy, {plab} ({names[-1]})")
        ax.legend(fontsize=6.5); ax.grid(alpha=0.3)
        fig.tight_layout(); fig.savefig(f"{args.plots}/photon_energy_{fn}.png", dpi=110)
        plt.close(fig)
    print(f">>> -> {args.plots}")


if __name__ == "__main__":
    main()
