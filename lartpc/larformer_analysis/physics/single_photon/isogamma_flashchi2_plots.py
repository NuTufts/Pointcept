"""isogamma_run45: flash-chi2 plots by vertex category.

    python3 isogamma_flashchi2_plots.py <DATADIR> [--out plots_isogamma]

Three figures:
  1. stacked histogram of log10(nuSliceFlashChi2) over found-vertex events, split
     by where the reco vertex sits relative to the gun photon (photon vertex <5 cm
     from the true conversion point / ambiguous 5-20 cm / cosmic pick in an event
     whose photon converted in the TPC / cosmic pick, photon never converted), with
     a separate "no in-window flash" bar (chi2 sentinel -1).
  2. efficiency vs the cut log10(chi2) < c. Denominator = events whose photon
     converts in the TPC. Solid: numerator = found vertex passing the cut (event
     level, the user's definition). Dashed: additionally the vertex is on the
     photon (<5 cm).
  3. purity vs the same cut: fraction of selected events (found vertex, chi2 < c)
     whose photon converted in the TPC (solid) / whose vertex is on the photon
     (dashed).
Categories and truth matching as in isogamma_flashchi2_study.py.
"""
import argparse
import glob
import os

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import uproot

# dataviz reference palette, categorical slots 1-4 (validated light mode)
C_PHOTON, C_AMBIG, C_COSCONV, C_COSNOCONV = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"


def load(D, tag):
    ev = uproot.open(f"{D}/dlgen2_larformer_ntuple_{tag}.root")["EventTree"]
    a = ev.arrays(["run", "subrun", "event", "foundVertex", "vtxX", "vtxY", "vtxZ",
                   "nuSliceFlashChi2"], library="np")
    idx = {k: i for i, k in enumerate(zip(a["run"], a["subrun"], a["event"]))}
    n = len(idx)
    ed = np.full((n, 3), np.nan)
    for fn in glob.glob(f"{D}/truth_sidecar/truth_fileno*.h5"):
        with h5py.File(fn, "r") as h:
            for k in h:
                g = h[k]
                i = idx.get((g.attrs["run"], g.attrs["subrun"], g.attrs["event"]))
                if i is None:
                    continue
                sp = g["trueSimPart"]
                ph = np.where(sp["PDG"][:] == 22)[0]
                if len(ph):
                    ed[i] = (sp["EDepX"][ph[0]], sp["EDepY"][ph[0]], sp["EDepZ"][ph[0]])
    conv = (np.all(np.isfinite(ed), 1) & (ed[:, 0] > 0) & (ed[:, 0] < 256)
            & (np.abs(ed[:, 1]) < 116) & (ed[:, 2] > 0) & (ed[:, 2] < 1036))
    fv = a["foundVertex"] == 1
    d = np.linalg.norm(np.stack([a["vtxX"], a["vtxY"], a["vtxZ"]], 1) - ed, axis=1)
    chi = np.asarray(a["nuSliceFlashChi2"], float)
    lchi = np.where(chi > 0, np.log10(np.where(chi > 0, chi, 1.0)), np.nan)
    cats = [("vertex on photon (<5 cm from conversion)", fv & conv & (d < 5), C_PHOTON),
            ("ambiguous (5-20 cm)", fv & conv & (d >= 5) & (d < 20), C_AMBIG),
            ("cosmic pick, photon converted in TPC", fv & conv & (d >= 20), C_COSCONV),
            ("cosmic pick, photon never converted", fv & ~conv, C_COSNOCONV)]
    return conv, fv, d, lchi, cats


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.yaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def fig_stack(lchi, cats, out):
    edges = np.arange(0.0, 7.01, 0.2)
    fig, ax = plt.subplots(figsize=(8, 4.6), facecolor="white")
    bottom = np.zeros(len(edges) - 1)
    nf_bottom = 0.0
    xnf = -0.6
    for lbl, m, col in cats:
        h, _ = np.histogram(lchi[m & np.isfinite(lchi)], bins=edges)
        ax.bar(edges[:-1], h, width=0.2 - 0.02, bottom=bottom, align="edge", color=col,
               label=f"{lbl}  (N={m.sum():,})", linewidth=0)
        bottom += h
        nnf = int(np.sum(m & ~np.isfinite(lchi)))
        ax.bar(xnf, nnf, width=0.4, bottom=nf_bottom, color=col, linewidth=0)
        nf_bottom += nnf
    ax.axvline(-0.2, color=GRID, lw=1)
    top = max(nf_bottom, bottom.max()) * 1.22
    ax.set_ylim(0, top)
    ax.text(xnf, nf_bottom * 1.03, "no in-window\nflash", ha="center", va="bottom", fontsize=8, color=INK2)
    for c in (2.5, 3.0):
        ax.axvline(c, color=INK2, lw=1, ls=":")
        ax.text(c - 0.04, top * 0.50, f"cut {c}", color=INK2, fontsize=8, rotation=90, ha="right", va="center")
    ax.set_xlim(-0.9, 7)
    ax.set_xlabel("log10(nu-slice flash chi2)", color=INK)
    ax.set_ylabel("found-vertex events / 0.2", color=INK)
    ax.set_title("isogamma_run45: flash chi2 by reco-vertex category (events with a found vertex)",
                 color=INK, fontsize=11, loc="left", pad=10)
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    style(ax)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig_eff_pur(conv, fv, d, lchi, out_eff, out_pur):
    cuts = np.arange(0.5, 7.01, 0.1)
    sel = lambda c: fv & np.isfinite(lchi) & (lchi < c)
    n_conv = conv.sum()
    eff_evt, eff_vtx, pur_evt, pur_vtx = [], [], [], []
    for c in cuts:
        s = sel(c)
        eff_evt.append(np.sum(s & conv) / n_conv)
        eff_vtx.append(np.sum(s & conv & (d < 5)) / n_conv)
        ns = s.sum()
        pur_evt.append(np.sum(s & conv) / ns if ns else np.nan)
        pur_vtx.append(np.sum(s & conv & (d < 5)) / ns if ns else np.nan)
    for ys, labels, ylab, title, out in [
        ((eff_evt, eff_vtx),
         ("found vertex (any position)", "found vertex on the photon (<5 cm)"),
         "efficiency", "efficiency vs flash-chi2 cut (denominator: photon converts in TPC, N=%d)" % n_conv, out_eff),
        ((pur_evt, pur_vtx),
         ("photon converted in TPC", "vertex on the photon (<5 cm)"),
         "purity", "purity of selected events vs flash-chi2 cut (found vertex & log10 chi2 < c)", out_pur)]:
        fig, ax = plt.subplots(figsize=(8.2, 4.4), facecolor="white")
        ax.plot(cuts, ys[0], color=C_PHOTON, lw=2, label=labels[0])
        ax.plot(cuts, ys[1], color=C_PHOTON, lw=2, ls="--", label=labels[1])
        for c in (2.5, 3.0):
            ax.axvline(c, color=INK2, lw=1, ls=":")
            i = int(round((c - cuts[0]) / 0.1))
            ax.text(c + 0.05, ys[0][i] + 0.02, f"{ys[0][i]:.2f}", color=INK2, fontsize=8)
            ax.text(c + 0.05, ys[1][i] - 0.05, f"{ys[1][i]:.2f}", color=INK2, fontsize=8)
        no_cut = (np.sum(fv & conv) / n_conv, np.sum(fv & conv & (d < 5)) / n_conv) if "ffic" in ylab \
            else (np.sum(fv & conv) / fv.sum(), np.sum(fv & conv & (d < 5)) / fv.sum())
        ax.axhline(no_cut[0], color=C_PHOTON, lw=0.8, alpha=0.5)
        ax.axhline(no_cut[1], color=C_PHOTON, lw=0.8, ls="--", alpha=0.5)
        ax.text(0.55, no_cut[0] + 0.015, f"no chi2 cut (no-flash events kept): {no_cut[0]:.2f}", ha="left", fontsize=8, color=INK2)
        ax.text(0.55, no_cut[1] - 0.045, f"no chi2 cut (no-flash events kept): {no_cut[1]:.2f}", ha="left", fontsize=8, color=INK2)
        ax.set_ylim(0, 1)
        ax.set_xlim(0.5, 7)
        ax.set_xlabel("cut value c:  log10(nu-slice flash chi2) < c", color=INK)
        ax.set_ylabel(ylab, color=INK)
        ax.set_title("isogamma_run45: " + title, color=INK, fontsize=9.5, loc="left")
        ax.legend(frameon=False, fontsize=9, loc="lower right" if "ffic" in ylab else "center right")
        style(ax)
        fig.tight_layout()
        fig.savefig(out, dpi=150)
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("datadir")
    ap.add_argument("--tag", default="isogamma_run45")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots_isogamma"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    conv, fv, d, lchi, cats = load(args.datadir, args.tag)
    fig_stack(lchi, cats, os.path.join(args.out, "flashchi2_stacked_by_vertex_category.png"))
    fig_eff_pur(conv, fv, d, lchi, os.path.join(args.out, "efficiency_vs_flashchi2_cut.png"),
                os.path.join(args.out, "purity_vs_flashchi2_cut.png"))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
