"""Per-true-photon CHARGE contamination breakdown by contaminating particle
type, and E_reco vs E_true per (charge-based) purity bin.

The ntuple's showerTrueXxPurity are POINT-count fractions; this study goes
back to the kp2 instance points + merged_sp truth to build CHARGE fractions
(comb pixel charge: Y plane else mean(U,V)). For each truth-matched reco
photon (showerTrueTID -> PDG 22, E_true=|p_true|>20 MeV, FV event), the kp2
instance with gt_trackid == showerTrueTID is taken and its charge split by
the true owner of each point:
  this photon | other photon (pid 22, other tid) | electron (|11|) |
  muon (|13|) | pion (|211|) | proton (2212) | ghost/unlabeled (no truth
  owner — includes real-cosmic charge in overlay originals) | other
Outputs: 6 fraction histograms (zero bin KEPT and drawn separately), a bar
chart of the fraction of photons where a type exceeds 10% of the charge,
and E_reco vs E_true 2D for all / purity>0.9 / [0.7,0.9) / <0.7.
"""
import argparse, os, sys
import numpy as np, h5py, uproot
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from flash_correction import rse_map  # noqa: E402

OA, OB = 0.020101, -15.49
CATS = ["other photon", "electron", "muon", "pion", "proton", "ghost/unlabeled"]


def msp_truth(path):
    with h5py.File(path, "r") as f:
        e = f["entry_0"]; mt = e["mc_particle_tree"]
        pid_by = {int(t): int(p) for t, p in zip(mt["trackid"][()], mt["pid"][()])}
        td = e["triplet_data"]
        pos = td["pos"][()].astype(np.float32)
        tid = np.asarray(td["trackid"][()], np.int64)
        pix = td["pixval"][()].astype(np.float64)
    q = pix[:, 2].copy(); bad = q <= 0; q[bad] = 0.5 * (pix[bad, 0] + pix[bad, 1])
    q = np.clip(q, 0, None)
    return {pos[i].tobytes(): i for i in range(len(pos))}, tid, q, pid_by


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--ntuple", required=True)
    ap.add_argument("--cascade-dir", required=True)
    ap.add_argument("--merged-sp-list", required=True)
    ap.add_argument("--recal-gamma-a", type=float, default=0.01556)
    ap.add_argument("--recal-gamma-b", type=float, default=-11.47)
    ap.add_argument("--max-events", type=int, default=100000)
    ap.add_argument("--plots", required=True)
    args = ap.parse_args()
    os.makedirs(args.plots, exist_ok=True)
    here = os.path.dirname(os.path.abspath(__file__))
    cache = os.path.join(here, "rse_" + os.path.basename(os.path.dirname(
        args.cascade_dir.rstrip("/"))) + "_" + os.path.basename(args.cascade_dir.rstrip("/")) + ".npz")
    t = uproot.open(args.ntuple)["EventTree"]
    a = t.arrays(["run", "subrun", "event", "foundVertex", "primaryVtxStream", "vtxIsFiducial",
                  "trueVtxInWCFV", "showerLArFormerPID", "showerRecoE", "showerTrueTID",
                  "trueSimPartTID", "trueSimPartPDG", "trueSimPartPx", "trueSimPartPy", "trueSimPartPz"])
    m = rse_map(args.cascade_dir, cache)
    msp_by_base = {os.path.basename(l.strip()): l.strip() for l in open(args.merged_sp_list) if l.strip()}
    F = {c: [] for c in CATS}; F["this photon"] = []; F["other"] = []
    Er, Et = [], []
    n_ph = n_noinst = n_nomsp = 0
    n = min(len(a["run"]), args.max_events)
    for i in range(n):
        if not (a["foundVertex"][i] == 1 and a["primaryVtxStream"][i] == 0 and a["vtxIsFiducial"][i] == 1
                and a["trueVtxInWCFV"][i] == 1):
            continue
        pid = np.asarray(a["showerLArFormerPID"][i]); E0 = np.asarray(a["showerRecoE"][i])
        E = np.where(pid == 22, (E0 - OB) / OA * args.recal_gamma_a + args.recal_gamma_b, E0)
        tids = np.asarray(a["trueSimPartTID"][i]); pdg = np.asarray(a["trueSimPartPDG"][i])
        Pm = np.stack([np.asarray(a["trueSimPartPx"][i]), np.asarray(a["trueSimPartPy"][i]),
                       np.asarray(a["trueSimPartPz"][i])], 1)
        cand = []
        for j in np.nonzero((pid == 22) & (E > 20))[0]:
            ttid = int(a["showerTrueTID"][i][j]); mm = np.nonzero(tids == ttid)[0]
            if len(mm) == 0 or abs(pdg[mm[0]]) != 22:
                continue
            ea = float(np.linalg.norm(Pm[mm[0]]))
            if ea >= 20:
                cand.append((j, ttid, ea, float(E[j])))
        if not cand:
            continue
        p = m.get((int(a["run"][i]), int(a["subrun"][i]), int(a["event"][i])))
        if not p:
            continue
        p = p if isinstance(p, str) else p[0]
        try:
            with h5py.File(p, "r") as fk:
                if "particle" not in fk:
                    continue
                coords = fk["slice/coord_cm"][()].astype(np.float32)
                inst = {}
                for k in fk["particle"]:
                    g = fk[f"particle/{k}"]
                    gt = int(g.attrs["gt_trackid"]) if "gt_trackid" in g.attrs else (
                        int(g["gt_trackid"][()]) if "gt_trackid" in g else -1)
                    if gt < 0:
                        continue
                    pidx = g["point_idx"][()]
                    if gt not in inst or len(pidx) > len(inst[gt]):
                        inst[gt] = pidx
                src = fk.attrs.get("src_file", ""); src = src.decode() if isinstance(src, bytes) else str(src)
            msp = msp_by_base.get(os.path.basename(src))
            if not msp:
                n_nomsp += 1; continue
            row_by_pos, rtid, rq, pid_by = msp_truth(msp)
        except Exception:
            continue
        for j, ttid, ea, er in cand:
            if ttid not in inst:
                n_noinst += 1; continue
            pts = coords[inst[ttid]]
            rows = np.array([row_by_pos.get(pts[k].tobytes(), -1) for k in range(len(pts))])
            rows = rows[rows >= 0]
            if len(rows) == 0:
                n_noinst += 1; continue
            q = rq[rows]; tt = rtid[rows]; Q = q.sum()
            if Q <= 0:
                continue
            owner_pid = np.array([pid_by.get(int(x), 0) if x > 0 else 0 for x in tt])
            ghost = (tt <= 0) | (owner_pid == 0) | (owner_pid == -1)
            f_this = q[tt == ttid].sum() / Q
            f_oph = q[(np.abs(owner_pid) == 22) & (tt != ttid) & ~ghost].sum() / Q
            f_el = q[(np.abs(owner_pid) == 11) & ~ghost].sum() / Q
            f_mu = q[(np.abs(owner_pid) == 13) & ~ghost].sum() / Q
            f_pi = q[(np.abs(owner_pid) == 211) & ~ghost].sum() / Q
            f_pr = q[(owner_pid == 2212) & ~ghost].sum() / Q
            f_gh = q[ghost].sum() / Q
            f_ot = max(0.0, 1.0 - (f_this + f_oph + f_el + f_mu + f_pi + f_pr + f_gh))
            for c, v in zip(["this photon"] + CATS + ["other"],
                            [f_this, f_oph, f_el, f_mu, f_pi, f_pr, f_gh, f_ot]):
                F[c].append(v)
            Er.append(er); Et.append(ea); n_ph += 1
    F = {k: np.asarray(v) for k, v in F.items()}; Er, Et = np.asarray(Er), np.asarray(Et)
    print(f"photons analysed {n_ph} | no matching kp2 instance {n_noinst} | no msp {n_nomsp}")
    pur = F["this photon"]
    print(f"charge purity (this photon): median {np.median(pur):.3f} | >0.9 {np.mean(pur>0.9):.3f} | <0.7 {np.mean(pur<0.7):.3f}")
    print(f"\n{'type':>16} {'mean frac':>9} {'frac==0':>8} {'frac>10%':>9} {'frac>30%':>9}")
    for c in CATS + ["other"]:
        v = F[c]
        print(f"{c:>16} {v.mean():9.3f} {np.mean(v==0):8.3f} {np.mean(v>0.10):9.3f} {np.mean(v>0.30):9.3f}")
    np.savez(os.path.join(args.plots, "photon_charge_contamination.npz"), Er=Er, Et=Et, **{k.replace(" ","_").replace("/","_"): v for k, v in F.items()})

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    edges = np.concatenate([[-0.05, 1e-6], np.linspace(0.05, 1.0, 20)])
    fig, axs = plt.subplots(2, 3, figsize=(15, 8))
    for ax, c in zip(axs.ravel(), CATS):
        v = F[c]
        h, _ = np.histogram(v, edges)
        ctr = 0.5 * (edges[:-1] + edges[1:]); wd = np.diff(edges)
        ax.bar(ctr[1:], h[1:], width=wd[1:], color="#7bafd4", edgecolor="k", lw=.3, label=">0 fraction")
        ax.bar([-0.025], [h[0]], width=[0.05], color="#d9534f", edgecolor="k", lw=.3, label=f"exactly 0 ({h[0]/len(v):.0%})")
        ax.set(xlabel=f"charge fraction from {c}", ylabel="true photons", yscale="log",
               title=f"{c}: >10% in {np.mean(v>0.10):.1%} of photons")
        ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(args.plots, "photon_contam_fractions.png"), dpi=115); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.2))
    labs = CATS + ["other"]
    vals = [np.mean(F[c] > 0.10) for c in labs]
    ax.bar(range(len(labs)), vals, color="#7bafd4", edgecolor="k")
    for k, v in enumerate(vals):
        ax.text(k, v + 0.005, f"{v:.1%}", ha="center", fontsize=9)
    ax.set_xticks(range(len(labs))); ax.set_xticklabels(labs, rotation=20)
    ax.set(ylabel="fraction of true photons", title="contaminant type contributes >10% of the cluster charge")
    fig.tight_layout(); fig.savefig(os.path.join(args.plots, "photon_contam_over10pct.png"), dpi=115); plt.close(fig)

    fig, axs = plt.subplots(2, 2, figsize=(11, 9.5))
    sels = [("all photons", np.ones(len(pur), bool)), ("purity > 0.9", pur > 0.9),
            ("purity in [0.7, 0.9)", (pur >= 0.7) & (pur < 0.9)), ("purity < 0.7", pur < 0.7)]
    b = np.linspace(0, 600, 41)
    for ax, (nm, msk) in zip(axs.ravel(), sels):
        ax.hist2d(Et[msk], Er[msk], bins=[b, b], norm=LogNorm(), cmap="viridis")
        ax.plot(b, b, "r-", lw=1)
        r = Er[msk] / Et[msk]
        ax.set(xlabel="E_true [MeV]", ylabel="E_reco (recal) [MeV]",
               title=f"{nm}: N={int(msk.sum())}, median E/Etrue {np.median(r):.2f}, IQR {np.subtract(*np.percentile(r,[75,25])):.2f}")
    fig.tight_layout(); fig.savefig(os.path.join(args.plots, "ereco_vs_etrue_by_purity.png"), dpi=115); plt.close(fig)
    print(f">>> {args.plots}/photon_contam_fractions.png, photon_contam_over10pct.png, ereco_vs_etrue_by_purity.png")


if __name__ == "__main__":
    main()
