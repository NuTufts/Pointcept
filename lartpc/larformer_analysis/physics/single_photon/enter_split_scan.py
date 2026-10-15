#!/usr/bin/env python3
"""Scan the in-FV / entering split variables on a diagnostics cell: for each
threshold on distFromWall (>= x) and on the chain BDT_enter scores (>= s), the
in-FV signal kept, entering signal and entering background removed, and the
in-FV purity of the kept sample. Needs diag_mc.npz (+ diag_ext.npz) from
single_photon_diagnostics.py.
    python3 enter_split_scan.py <cell_dir>"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import single_photon_selection as SPS

d = sys.argv[1]
M = np.load(os.path.join(d, "diag_mc.npz"), allow_pickle=True)
w, cat, fin = M["w"], M["cat"], M["steps"][-1]
X = np.load(os.path.join(d, "diag_ext.npz"), allow_pickle=True) if os.path.exists(os.path.join(d, "diag_ext.npz")) else None
wx, finx = (X["w"], X["steps"][-1]) if X is not None else (None, None)
infv = fin & np.isin(cat, [0, 1]); sig0 = fin & (cat == 0); sig1 = fin & (cat == 1)
ent_sig = fin & np.isin(cat, [2, 3]); ent_bkg = fin & np.isin(cat, [4, 5, 6]); infv_bkg = fin & np.isin(cat, [7, 8, 9, 10])
W = lambda m: w[m].sum()
VARS = [("distFromWall", "backward wall distance >= x [cm]", np.arange(0, 301, 10)),
        ("bdtChainEnter", "BDT_enter (primary model) >= s", np.round(np.arange(0, 0.96, 0.05), 2))]
for k in sorted(M.keys()):
    if k.startswith("bdtChainEnter_"):
        VARS.append((k, f"BDT_enter ({k.split('_', 1)[1]}) >= s", np.round(np.arange(0, 0.96, 0.05), 2)))
L = [f"IN-FV vs ENTERING split scan on {d} (final step); totals: in-FV sig {W(infv):.1f} (1g0X {W(sig0):.1f}, 1g+X {W(sig1):.1f}), "
     f"entering sig {W(ent_sig):.1f}, entering bkg {W(ent_bkg):.1f}, in-FV bkg {W(infv_bkg):.1f}"
     + (f", EXT {wx[finx].sum():.1f}" if X is not None else "")]
curves = {}
for key, lab, grid in VARS:
    v = np.nan_to_num(np.asarray(M[key], np.float64), nan=-1.0)
    vx = np.nan_to_num(np.asarray(X[key], np.float64), nan=-1.0) if X is not None else None
    L.append(f"\n== {lab}")
    L.append(f"{'cut':>7}{'keep1g0X':>9}{'keep1g+X':>9}{'keepFV':>8}{'rmEntSig':>9}{'rmEntBkg':>9}{'keepFVbkg':>10}{'EXTkept':>8}{'purFV':>7}{'FVsig':>7}{'ent':>7}")
    pts = []
    for t in grid:
        k = v >= t
        kx = (vx >= t) if vx is not None else None
        a0, a1, af = W(sig0 & k) / max(W(sig0), 1e-9), W(sig1 & k) / max(W(sig1), 1e-9), W(infv & k) / max(W(infv), 1e-9)
        res, reb = 1 - W(ent_sig & k) / max(W(ent_sig), 1e-9), 1 - W(ent_bkg & k) / max(W(ent_bkg), 1e-9)
        kfb = W(infv_bkg & k) / max(W(infv_bkg), 1e-9)
        ex = wx[finx & kx].sum() if kx is not None else 0.0
        tot = W(fin & k) + ex
        pur = W(infv & k) / max(tot, 1e-9)
        L.append(f"{t:7.2f}{a0:9.3f}{a1:9.3f}{af:8.3f}{res:9.3f}{reb:9.3f}{kfb:10.3f}{ex:8.1f}{pur:7.3f}{W(infv & k):7.1f}{W((ent_sig | ent_bkg) & k):7.1f}")
        pts.append((t, af, 1 - W((ent_sig | ent_bkg) & k) / max(W(ent_sig | ent_bkg), 1e-9), pur, a0))
    curves[key] = (lab, np.array(pts))
# 2-D grid: distFromWall >= x AND BDT_enter (primary) >= s
v1 = np.nan_to_num(np.asarray(M["distFromWall"], np.float64), nan=-1.0); v2 = np.nan_to_num(np.asarray(M["bdtChainEnter"], np.float64), nan=-1.0)
x1 = np.nan_to_num(np.asarray(X["distFromWall"], np.float64), nan=-1.0) if X is not None else None
x2 = np.nan_to_num(np.asarray(X["bdtChainEnter"], np.float64), nan=-1.0) if X is not None else None
L.append("\n== 2-D: distFromWall >= x AND BDT_enter >= s  (cells: keepFV / rmEntering / purFV)")
ss = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
hdr2 = "x\\s"
L.append(f"{hdr2:>6}" + "".join(f"{s_:>22.2f}" for s_ in ss))
for x_ in (0, 20, 40, 60, 80, 100, 150):
    row = f"{x_:6d}"
    for s_ in ss:
        k = (v1 >= x_) & (v2 >= s_)
        ex = wx[finx & (x1 >= x_) & (x2 >= s_)].sum() if X is not None else 0.0
        kf = W(infv & k) / max(W(infv), 1e-9); re = 1 - W((ent_sig | ent_bkg) & k) / max(W(ent_sig | ent_bkg), 1e-9)
        pur = W(infv & k) / max(W(fin & k) + ex, 1e-9)
        row += f"   {kf:.2f}/{re:.2f}/{pur:.2f}"
    L.append(row)
txt = "\n".join(L); print(txt)
open(os.path.join(d, "enter_split_scan.txt"), "w").write(txt + "\n")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, axs = plt.subplots(1, 3, figsize=(16, 5))
for key, (lab, P) in curves.items():
    axs[0].plot(P[:, 2], P[:, 1], "o-", ms=3, label=lab)
    axs[1].plot(P[:, 2], P[:, 4], "o-", ms=3, label=lab)
    axs[2].plot(P[:, 1], P[:, 3], "o-", ms=3, label=lab)
    for i in range(0, len(P), max(1, len(P) // 6)):
        axs[0].annotate(f"{P[i, 0]:g}", (P[i, 2], P[i, 1]), fontsize=6, xytext=(3, 3), textcoords="offset points")
axs[0].set(xlabel="fraction of ENTERING (signal + bkg) removed", ylabel="fraction of in-FV signal kept", title="in-FV signal vs entering removal", xlim=(0, 1), ylim=(0, 1))
axs[1].set(xlabel="fraction of ENTERING removed", ylabel="fraction of 1g0X kept", title="1g0X only", xlim=(0, 1), ylim=(0, 1))
axs[2].set(xlabel="fraction of in-FV signal kept", ylabel="in-FV purity of the kept sample", title="purity vs efficiency", xlim=(0, 1))
for ax in axs:
    ax.grid(alpha=0.3); ax.legend(fontsize=7)
fig.suptitle(f"in-FV / entering split scan: {os.path.basename(d.rstrip('/'))}")
fig.tight_layout(); fig.savefig(os.path.join(d, "enter_split_scan.png"), dpi=110)
print(">>> ->", os.path.join(d, "enter_split_scan.png"))
