"""Diagnostic distributions for the single-photon selection: MC truth categories
+ EXT stacked, beam data as points, for events passing the selection (final
step) and for the step before the flash-chi2 cut. Same selection machinery /
options as single_photon_selection.py (imported), so the populations are the
ones in the cutflow.

Variables (per selected event, leading photon candidate):
  flashPE      : total in-time observed PE (cascade flash/observed_pe, by opdet)
  startX/Y/Z   : candidate shower start point (NOTE: vertex-less starts are the
                 keypoint-model start, see README TODO)
  cosBeam      : shower direction cos(theta) w.r.t. +z
  nProtons     : nu-stream tracks with LArFormer PID 2212 and KE > --proton-ke-min
  flashDist    : distance in (y,z) between the PE-weighted PMT centroid and the
                 intersection of the shower LINE (start + t*dir, either sign of
                 t) with the x=0 plane; NaN if |dir_x| < 1e-3
  flashDistFwd : same but only counted when the intersection is FORWARD (t>0),
                 else NaN -- separates "light ahead" from "light behind"
  distToWall   : distance from start to the TPC wall along +dir
  distFromWall : distance from start to the TPC wall along -dir
                 (both NaN when the start is outside the TPC box)
  flashPE vs E : 2-D MC-signal / EXT scatter (light per unit shower energy)

Flash PE needs the cascade files: --{mc,ext,data}-cascade = keypoint2_streams
dirs; the (run,subrun,event)->file maps are the cached rse_*.npz of the pi0
study (built there, read-only here). PMT positions: lartpc/flashmatch/
saturation._OPDET_POS (opdet-indexed, matches observed_pe).

    python3 single_photon_diagnostics.py --mc-ntuple ... --ext-ntuple ... \
        --data-ntuple ... --mc-cascade ... --ext-cascade ... --data-cascade ... \
        --novtx --mc-odd-only --plots plots_v2_s1ep2p8_novtx/diagnostics
"""
import argparse
import os
import sys

import numpy as np
import uproot
import awkward as ak
import h5py

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "pi0mass_peak"))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "..", "..")))
import single_photon_selection as SPS                       # noqa: E402
from flash_correction import rse_map                        # noqa: E402
from lartpc.flashmatch.saturation import _OPDET_POS         # noqa: E402

TPC_LO, TPC_HI = SPS.TPC_LO, SPS.TPC_HI
BR_EXTRA = ["showerStartDirX", "showerStartDirY", "showerStartDirZ",
            "showerStartPosX", "showerStartPosY", "showerStartPosZ",
            "showerCosTheta", "trackLArFormerPID", "trackRecoE", "trackStream",
            "trackVtxIdx"]


def add_args(ap):
    """Selection options mirrored from single_photon_selection (same names)."""
    ap.add_argument("--mc-ntuple", required=True)
    ap.add_argument("--ext-ntuple", default=None)
    ap.add_argument("--data-ntuple", default=None)
    ap.add_argument("--mc-cascade", default=None)
    ap.add_argument("--ext-cascade", default=None)
    ap.add_argument("--data-cascade", default=None)
    ap.add_argument("--ext-scale", type=float, default=1.1818)
    ap.add_argument("--ext-row-min", type=int, default=100000)
    ap.add_argument("--pot", type=float, default=4.4e19)
    ap.add_argument("--recal-gamma-a", type=float, default=None)
    ap.add_argument("--recal-gamma-b", type=float, default=-12.80)
    ap.add_argument("--shower-bdt-min", type=float, default=0.192)
    ap.add_argument("--bdt-after-multiplicity", action="store_true")
    ap.add_argument("--flashchi2-cut", type=float, default=316.2)
    ap.add_argument("--mu-ke-min", type=float, default=100.0)
    ap.add_argument("--muon-finder", default="union",
                    choices=["segmenter", "larpid", "union"])
    ap.add_argument("--mu-primary-only", action="store_true")
    ap.add_argument("--require-confident", action="store_true")
    ap.add_argument("--primary-only", action="store_true")
    ap.add_argument("--novtx", action="store_true")
    ap.add_argument("--novtx-bdt-min", type=float, default=0.5)
    ap.add_argument("--novtx-model", default=None)
    ap.add_argument("--mc-odd-only", action="store_true")
    ap.add_argument("--proton-ke-min", type=float, default=35.0)
    ap.add_argument("--path", type=int, default=None, choices=[0, 1],
                    help="restrict to candidates of one path: 0 = vertex sample "
                         "(primary in-FV vertex), 1 = no-vertex sample (needs --novtx)")
    ap.add_argument("--plots", required=True)


def ray_to_wall(s, d):
    """Distance from s along +d to the TPC box boundary (NaN if s outside)."""
    if np.any(s < TPC_LO) or np.any(s > TPC_HI):
        return np.nan
    t = np.inf
    for k in range(3):
        if abs(d[k]) < 1e-9:
            continue
        tk = ((TPC_HI[k] if d[k] > 0 else TPC_LO[k]) - s[k]) / d[k]
        if tk >= 0:
            t = min(t, tk)
    return float(t) if np.isfinite(t) else np.nan


def per_sample(ntuple, is_mc, args, cascade, tag):
    a, pot = SPS.load(ntuple, is_mc, args.novtx)
    t = uproot.open(ntuple)["EventTree"]
    ex = t.arrays([b for b in BR_EXTRA if b in set(t.keys())])
    n = len(ex["run"]) if "run" in ex.fields else len(a["run"])
    R = SPS.reco_tables(a, args)
    R["novtx"] = args.novtx
    names, S, E_cand, score_cand, n_g, path_cand = SPS.build_steps(
        R, args.shower_bdt_min, args.flashchi2_cut, args.bdt_after_multiplicity)
    idx = R["_cand_idx"]
    if getattr(args, "path", None) is not None:
        S = S.copy()
        for k in range(2, len(names)):
            S[k] = S[k] & (path_cand == args.path)
    out = dict(steps=S, names=names, E=E_cand, path=path_cand,
               logchi2=np.log10(np.clip(np.nan_to_num(R["chi2"], nan=-1.0), 1e-3, None)))
    out["logchi2"][~np.isfinite(R["chi2"]) | (R["chi2"] <= 0)] = np.nan

    def pick(x, fill=np.nan):
        v = np.asarray(ak.to_numpy(ak.fill_none(ak.firsts(x[idx]), fill)), np.float64)
        v[n_g == 0] = np.nan
        return v
    st = np.stack([pick(ex[f"showerStartPos{c}"]) for c in "XYZ"], 1)
    dr = np.stack([pick(ex[f"showerStartDir{c}"]) for c in "XYZ"], 1)
    out["startX"], out["startY"], out["startZ"] = st.T
    out["cosBeam"] = dr[:, 2]
    # protons: nu-stream tracks (any vertex incl. orphans when --novtx)
    is_p = (ex["trackLArFormerPID"] == 2212) & (ex["trackRecoE"] > args.proton_ke_min)
    if args.novtx and "trackStream" in ex.fields:
        is_p = is_p & (ex["trackStream"] == 0)
    out["nProtons"] = ak.to_numpy(ak.sum(is_p, axis=1)).astype(np.float64)
    # walls
    d2w = np.full(n, np.nan); dfw = np.full(n, np.nan)
    ok = np.isfinite(st).all(1) & np.isfinite(dr).all(1)
    for i in np.nonzero(ok)[0]:
        d2w[i] = ray_to_wall(st[i], dr[i]); dfw[i] = ray_to_wall(st[i], -dr[i])
    out["distToWall"], out["distFromWall"] = d2w, dfw

    # flash PE + centroid vs shower line (only for events past S3, i.e. the
    # step before the chi2 cut; index -2)
    pe = np.full(n, np.nan); fd = np.full(n, np.nan); fdf = np.full(n, np.nan)
    if cascade:
        cache = os.path.join(HERE, "..", "pi0mass_peak",
                             "rse_" + os.path.basename(os.path.dirname(cascade.rstrip("/")))
                             + "_" + os.path.basename(cascade.rstrip("/")) + ".npz")
        m = rse_map(cascade, cache if os.path.exists(cache) else None)
        run, sub, evt = R["run"], R["subrun"], R["event"]
        need = np.nonzero(S[-2])[0]
        print(f">>> {tag}: reading flashes for {len(need)} events", flush=True)
        yz = _OPDET_POS[:32, 1:3]
        for i in need:
            p = m.get((int(run[i]), int(sub[i]), int(evt[i])))
            if not p:
                continue
            try:
                with h5py.File(p, "r") as f:
                    if "flash" not in f or "observed_pe" not in f["flash"]:
                        continue
                    o = np.clip(f["flash/observed_pe"][()][:32], 0, None)
            except Exception:
                continue
            tot = o.sum()
            pe[i] = tot
            if tot <= 0 or not ok[i] or abs(dr[i, 0]) < 1e-3:
                continue
            c = (o[:, None] * yz).sum(0) / tot
            tt = -st[i, 0] / dr[i, 0]
            hit = st[i, 1:3] + tt * dr[i, 1:3]
            fd[i] = float(np.linalg.norm(hit - c))
            if tt > 0:
                fdf[i] = fd[i]
    out["flashPE"], out["flashDist"], out["flashDistFwd"] = pe, fd, fdf
    return out, a, pot, R


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    add_args(ap)
    args = ap.parse_args()
    os.makedirs(args.plots, exist_ok=True)

    print(">>> MC ...", flush=True)
    M, a, pot, R = per_sample(args.mc_ntuple, True, args, args.mc_cascade, "MC")
    w0 = ak.to_numpy(a["xsecWeight"]).astype(np.float64)
    w = np.where(w0 > 0, w0, 0.0) * (args.pot / pot)
    if args.mc_odd_only:
        odd = ak.to_numpy(a["event"]) % 2 == 1
        w = np.where(odd, 2.0 * w, 0.0)
    T = SPS.truth_tables(a)
    cat = T["cat"]
    del a
    X = D = None
    wx = wd = None
    if args.ext_ntuple:
        print(">>> EXT ...", flush=True)
        X, _, _, RX = per_sample(args.ext_ntuple, False, args, args.ext_cascade, "EXT")
        wx = np.zeros(len(RX["run"])); wx[args.ext_row_min:] = args.ext_scale
    if args.data_ntuple:
        print(">>> data ...", flush=True)
        D, _, _, RD = per_sample(args.data_ntuple, False, args, args.data_cascade, "DATA")
        wd = np.ones(len(RD["run"]))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    CATS, COL, NMC = SPS.CATS, SPS.CAT_COLORS, SPS.N_MC_CAT
    SPECS = [
        ("logchi2", r"$\log_{10}$ flash $\chi^2$ (nu slice)", np.linspace(0, 7, 57)),
        ("flashPE", "in-time flash total PE", np.linspace(0, 3000, 31)),
        ("startX", "candidate shower start x [cm]", np.linspace(-20, 276, 38)),
        ("startY", "candidate shower start y [cm]", np.linspace(-130, 130, 27)),
        ("startZ", "candidate shower start z [cm]", np.linspace(-20, 1060, 37)),
        ("cosBeam", r"shower $\cos\theta_{beam}$", np.linspace(-1, 1, 41)),
        ("nProtons", f"n reco protons (KE > {args.proton_ke_min:.0f} MeV)",
         np.arange(-0.5, 5.5, 1)),
        ("flashDist", "|PE centroid - shower-line hit at x=0| [cm]", np.linspace(0, 600, 31)),
        ("flashDistFwd", "same, forward intersections only [cm]", np.linspace(0, 600, 31)),
        ("distToWall", "distance to wall along +dir [cm]", np.linspace(0, 1000, 41)),
        ("distFromWall", "distance to wall along -dir [cm]", np.linspace(0, 1000, 41)),
        ("E", "candidate photon energy [MeV]", np.linspace(0, 1000, 41)),
    ]
    steps = [(len(M["names"]) - 2, "preChi2"), (len(M["names"]) - 1, "final")]
    lines = []
    for k, sname in steps:
        d = os.path.join(args.plots, sname); os.makedirs(d, exist_ok=True)
        lines.append(f"== {M['names'][k]} : weighted medians per category")
        for key, xlab, bins in SPECS:
            lo, hi = bins[0], bins[-1] - 1e-6
            m = M["steps"][k] & np.isfinite(M[key])
            data = [np.clip(M[key][m & (cat == c)], lo, hi) for c in range(NMC)]
            ws = [w[m & (cat == c)] for c in range(NMC)]
            cols = list(COL[:NMC])
            if X is not None:
                mx = X["steps"][k] & np.isfinite(X[key])
                data.append(np.clip(X[key][mx], lo, hi)); ws.append(wx[mx]); cols.append(COL[NMC])
            labs = [f"{CATS[c]} ({ws[c].sum():.1f})" for c in range(len(ws))]
            fig, ax = plt.subplots(figsize=(7.4, 4.8))
            ax.hist(data, bins=bins, weights=ws, stacked=True, color=cols, label=labs)
            if D is not None:
                md = D["steps"][k] & np.isfinite(D[key])
                hd, _ = np.histogram(np.clip(D[key][md], lo, hi), bins=bins)
                ctr = 0.5 * (bins[:-1] + bins[1:])
                ax.errorbar(ctr[hd > 0], hd[hd > 0], yerr=np.sqrt(hd[hd > 0]), fmt="ko",
                            ms=3.5, label=f"beam data ({int(hd.sum())})")
            ax.set(xlabel=xlab, ylabel=f"events / {args.pot:.1e} POT",
                   title=f"{xlab}  ({M['names'][k]})")
            ax.legend(fontsize=6.5); ax.grid(alpha=0.3)
            fig.tight_layout(); fig.savefig(f"{d}/{key}.png", dpi=110); plt.close(fig)
            # medians
            med = []
            for c in range(NMC):
                v = M[key][m & (cat == c)]; ww = w[m & (cat == c)]
                if ww.sum() > 0:
                    o = np.argsort(v); cw = np.cumsum(ww[o]) / ww.sum()
                    med.append(f"{CATS[c][:16]} {v[o][np.searchsorted(cw, 0.5)]:.1f}")
            if X is not None and mx.any():
                med.append(f"EXT {np.median(X[key][mx]):.1f}")
            if D is not None and md.any():
                med.append(f"data {np.median(D[key][md]):.1f}")
            lines.append(f"  {key:13s}: " + " | ".join(med))
        # 2-D flash PE vs energy: signal vs EXT
        m = M["steps"][k] & np.isfinite(M["flashPE"]) & np.isfinite(M["E"])
        sig = m & (cat < SPS.N_SIG_CAT)
        fig, ax = plt.subplots(figsize=(6.4, 5))
        ax.scatter(M["E"][m & ~sig], M["flashPE"][m & ~sig], s=6, c="#9467bd", alpha=0.5,
                   label="MC background")
        if X is not None:
            mx = X["steps"][k] & np.isfinite(X["flashPE"])
            ax.scatter(X["E"][mx], X["flashPE"][mx], s=6, c="k", alpha=0.35, label="EXT")
        ax.scatter(M["E"][sig], M["flashPE"][sig], s=8, c="#d62728", label="MC signal")
        ax.set(xlabel="candidate photon energy [MeV]", ylabel="in-time flash PE",
               xlim=(0, 1000), ylim=(0, 4000), title=f"flash PE vs shower energy ({M['names'][k]})")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
        fig.tight_layout(); fig.savefig(f"{d}/flashPE_vs_E.png", dpi=110); plt.close(fig)
    txt = "\n".join(lines)
    print("\n" + txt)
    with open(os.path.join(args.plots, "medians.txt"), "w") as fh:
        fh.write(txt + "\n")
    np.savez(os.path.join(args.plots, "diag_mc.npz"), w=w, cat=cat,
             **{k: v for k, v in M.items() if isinstance(v, np.ndarray)})
    print(f">>> plots -> {args.plots}")


if __name__ == "__main__":
    main()
