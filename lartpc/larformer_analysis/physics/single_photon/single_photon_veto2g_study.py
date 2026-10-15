#!/usr/bin/env python3
"""Second-photon veto study: pairwise geometry between the single-photon
candidate and every OTHER shower prong of the nu slice at the VERTEX stage
(base selection final step, vertex path, before any 2nd-photon veto).

Per (candidate, other shower) pair (directions = the exporter's start
directions = PCA/trunk axes):
  dstart   start-to-start distance [cm]
  cos12    direction cosine between the two axes
  ip2, t2  distance of the other start from the candidate axis, and its signed
           projection along the candidate direction (t2 > 0 = downstream)
  ip1      distance of the candidate start from the other axis
  skew     distance of closest approach of the two axes
  dc1,dc2  distance of the crossing point (mid-point of closest approach) to each
           start; dcmax = max; behind = both closest-approach parameters < 0
           (a pi0 pair points AWAY from a common decay point)
  dcv      crossing point to the reco vertex
  E2, objectness, photon score, nhits, vtxidx, dist-to-vertex of the other shower
  mgg      pi0-hypothesis invariant mass from the two energies and cos12 (reported)
MC truth class of the other shower (showerTrueTID / Purity, trueSimPart MID):
  0 fragment of the candidate (same true TID)   1 pi0 partner (same mother, photon)
  2 other nu photon (different mother)            3 nu non-photon (e, track-like)
  4 unmatched / cosmic / low purity
Outputs: class composition of the vetoing cluster, stacked distributions per
variable by class (MC) with data overlay, and an event-level table of veto
variants (signal kept vs >=2-photon background kept, data counts).
    python3 single_photon_veto2g_study.py --mc-ntuple ... --ext-ntuple ... --data-ntuple ... --novtx --plots <dir>
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

EXTRA = ("showerTrueTID", "showerTruePID", "showerTruePurity", "showerTrueComp", "showerLArFormerPhScore",
         "showerLArFormerElScore", "showerObjectness", "showerNHits", "showerCharge", "recoVtxX", "recoVtxY", "recoVtxZ")
CLS = ["fragment of candidate (same TID)", "pi0 partner (same mother)", "other nu photon", "nu non-photon (e/track)", "unmatched / cosmic / low purity"]
CLS_COL = ["#ff7f0e", "#9467bd", "#1f77b4", "#2ca02c", "#7f7f7f"]
M_PI0 = 134.977


def pair_vars(s1, d1, s2, d2, vtx, E1, E2):
    d1 = d1 / max(np.linalg.norm(d1), 1e-9); d2 = d2 / max(np.linalg.norm(d2), 1e-9)
    r = s2 - s1
    out = {"dstart": float(np.linalg.norm(r)), "cos12": float(np.dot(d1, d2))}
    out["t2"] = float(np.dot(r, d1)); out["ip2"] = float(np.linalg.norm(r - out["t2"] * d1))
    u1 = float(np.dot(-r, d2)); out["ip1"] = float(np.linalg.norm(-r - u1 * d2))
    # closest approach of the two lines
    b = float(np.dot(d1, d2)); den = 1.0 - b * b
    if den < 1e-6:
        t, u = 0.0, 0.0
    else:
        e, f = float(np.dot(d1, r)), float(np.dot(d2, r))
        t = (e - b * f) / den; u = (b * e - f) / den
    p1, p2 = s1 + t * d1, s2 + u * d2
    c = 0.5 * (p1 + p2)
    out["skew"] = float(np.linalg.norm(p1 - p2)); out["tstar"], out["ustar"] = t, u
    out["behind"] = float(t < 0 and u < 0)
    out["dc1"] = float(np.linalg.norm(c - s1)); out["dc2"] = float(np.linalg.norm(c - s2)); out["dcmax"] = max(out["dc1"], out["dc2"])
    out["dcv"] = float(np.linalg.norm(c - vtx)) if vtx is not None and np.all(np.isfinite(vtx)) else np.nan
    out["mgg"] = float(np.sqrt(max(2.0 * E1 * max(E2, 0.0) * (1.0 - out["cos12"]), 0.0)))
    return out


def run(ntuple, is_mc, args):
    a, pot = SPS.load(ntuple, is_mc, args.novtx, extra=EXTRA)
    R = SPS.reco_tables(a, args); R["novtx"] = args.novtx
    names, S, E, _, _, path = SPS.build_steps(R, args.shower_bdt_min, args.flashchi2_cut, args.bdt_after_multiplicity, args.flashpe_ratio_max)
    fin = S[-1] & (path == 0)
    idx = np.asarray(ak.to_numpy(ak.fill_none(ak.firsts(R["_cand_idx"]), -1)), np.int64)
    Es = R["_E"]
    spid = a["showerLArFormerPID"]; stream = a["showerStream"] if args.novtx else ak.zeros_like(spid)
    nu_v = a["recoVtxStream"] == 0
    vx = np.stack([np.asarray(ak.to_numpy(ak.fill_none(ak.firsts(a["recoVtx" + c][nu_v]), np.nan)), np.float64) for c in "XYZ"], 1)
    if is_mc:
        T = SPS.truth_tables(a); cat = T["cat"]
        tids_all, mids_all, pdg_all = a["trueSimPartTID"], a["trueSimPartMID"], a["trueSimPartPDG"]
    rows, ev_rows = [], []
    for i in np.where(fin)[0]:
        j = int(idx[i])
        if j < 0:
            continue
        s1 = np.array([a["showerStartPosX"][i][j], a["showerStartPosY"][i][j], a["showerStartPosZ"][i][j]], float)
        d1 = np.array([a["showerStartDirX"][i][j], a["showerStartDirY"][i][j], a["showerStartDirZ"][i][j]], float)
        E1 = float(Es[i][j]); ns = len(spid[i])
        if is_mc:
            tid1 = int(a["showerTrueTID"][i][j])
            look_mid = dict(zip(np.asarray(tids_all[i]).tolist(), np.asarray(mids_all[i]).tolist()))
            look_pdg = dict(zip(np.asarray(tids_all[i]).tolist(), np.asarray(pdg_all[i]).tolist()))
            mid1 = look_mid.get(tid1, -999)
        n_other = 0; first = len(rows)
        for o in range(ns):
            if o == j or int(stream[i][o]) != 0:
                continue
            s2 = np.array([a["showerStartPosX"][i][o], a["showerStartPosY"][i][o], a["showerStartPosZ"][i][o]], float)
            d2 = np.array([a["showerStartDirX"][i][o], a["showerStartDirY"][i][o], a["showerStartDirZ"][i][o]], float)
            if not (np.all(np.isfinite(s2)) and np.all(np.isfinite(d2)) and np.linalg.norm(d2) > 0):
                continue
            E2 = float(Es[i][o]); pv = pair_vars(s1, d1, s2, d2, vx[i], E1, E2)
            pv.update(ev=i, osh=o, E1=E1, E2=E2, pid2=int(spid[i][o]), obj2=float(a["showerObjectness"][i][o]),
                      ph2=float(a["showerLArFormerPhScore"][i][o]), nhits2=int(a["showerNHits"][i][o]),
                      vtxidx2=int(a["showerVtxIdx"][i][o]) if args.novtx else 0,
                      d2v=float(np.linalg.norm(s2 - vx[i])) if np.all(np.isfinite(vx[i])) else np.nan)
            if is_mc:
                tid2 = int(a["showerTrueTID"][i][o]); pur2 = float(a["showerTruePurity"][i][o])
                if tid2 <= 0 or pur2 < 0.3:
                    c = 4
                elif tid2 == tid1:
                    c = 0
                elif abs(look_pdg.get(tid2, 0)) == 22 and look_mid.get(tid2, -998) == mid1 and mid1 != -999:
                    c = 1
                elif abs(look_pdg.get(tid2, 0)) == 22:
                    c = 2
                else:
                    c = 3
                pv["cls"] = c
            rows.append(pv); n_other += 1
        ev_rows.append(dict(ev=i, n_other=n_other, first=first, E1=E1))
    w = None
    if is_mc:
        w0 = ak.to_numpy(a["xsecWeight"]).astype(np.float64); w = np.where(w0 > 0, w0, 0.0) * (args.pot / pot)
        return rows, ev_rows, w, cat, T["sig_incl"], T["in_fv"]
    return rows, ev_rows, None, None, None, None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_args(ap); ap.set_defaults(signal_def="lar")
    ap.add_argument("--veto2g-emin", type=float, default=10.0)
    args = ap.parse_args()
    SPS.configure(args); os.makedirs(args.plots, exist_ok=True)
    if not args.novtx:
        sys.exit("--novtx required")
    rows, evs, w, cat, sig, infv = run(args.mc_ntuple, True, args)
    print(f">>> MC: {len(evs)} vertex-stage events, {len(rows)} candidate-other pairs", flush=True)
    rx, evx = (run(args.ext_ntuple, False, args)[:2] if args.ext_ntuple else ([], []))
    rd, evd = (run(args.data_ntuple, False, args)[:2] if args.data_ntuple else ([], []))
    wx_scale = args.ext_scale

    def col(rs, k, fill=np.nan):
        return np.array([r.get(k, fill) for r in rs], float) if rs else np.zeros(0)
    ev_i = col(rows, "ev").astype(int); wpair = w[ev_i] if len(rows) else np.zeros(0)
    cls = col(rows, "cls").astype(int); E2 = col(rows, "E2"); pid2 = col(rows, "pid2").astype(int)
    catp = cat[ev_i]
    grp = {"in-FV signal": np.isin(catp, [0, 1]), "entering signal": np.isin(catp, [2, 3]), ">=2g bkg (in-FV + entering)": np.isin(catp, [4, 9]),
           "CC numu": catp == 7, "other bkg": np.isin(catp, [5, 6, 8, 10])}
    L = [f"vertex-stage sample (base final step, vertex path): MC events {len(evs)} (w {w[[e['ev'] for e in evs]].sum():.1f}), pairs {len(rows)}; EXT events {len(evx)} pairs {len(rx)}; data events {len(evd)} pairs {len(rd)}"]
    veto_now = (pid2 == 22) & (E2 > args.veto2g_emin)
    L.append(f"\nCLASS of the other-shower prongs (w): all pairs | prongs that FIRE the current veto (photon-PID, E > {args.veto2g_emin:.0f} MeV)")
    for g, gm in grp.items():
        a_ = " | ".join(f"{CLS[c][:18]} {wpair[gm & (cls == c)].sum():.1f}" for c in range(5))
        b_ = " | ".join(f"{CLS[c][:18]} {wpair[gm & veto_now & (cls == c)].sum():.1f}" for c in range(5))
        L.append(f"  {g:28s}: {a_}\n  {'':28s}  veto-firing: {b_}")
    # event-level: which class is the HIGHEST-E veto-firing prong (the one that decides the veto)?
    L.append("\nEVENT-LEVEL: class of the highest-energy veto-firing prong (events that fail the current veto), w")
    ev_first = {e["ev"]: e for e in evs}
    best = {}
    for k, r in enumerate(rows):
        if veto_now[k] and (r["ev"] not in best or r["E2"] > rows[best[r["ev"]]]["E2"]):
            best[r["ev"]] = k
    bk = np.array(list(best.values()), int) if best else np.zeros(0, int)
    for g, gm_ev in (("in-FV signal", lambda c: c in (0, 1)), ("entering signal", lambda c: c in (2, 3)), (">=2g bkg", lambda c: c in (4, 9)), ("CC numu", lambda c: c == 7), ("other bkg", lambda c: c in (5, 6, 8, 10))):
        sel = np.array([gm_ev(int(cat[rows[k]["ev"]])) for k in bk], bool) if len(bk) else np.zeros(0, bool)
        tot = w[[rows[k]["ev"] for k in bk[sel]]].sum() if sel.any() else 0.0
        L.append(f"  {g:18s} fail-veto events {tot:6.1f}: " + " | ".join(f"{CLS[c][:18]} {w[[rows[k]['ev'] for k in bk[sel] if rows[k]['cls'] == c]].sum():.1f}" for c in range(5)))
    # ---- veto variants (event level) ----------------------------------------
    def fires(rs, cond):
        f = {}
        for r in rs:
            if r["pid2"] == 22 and r["E2"] > args.veto2g_emin and cond(r):
                f[r["ev"]] = True
        return f
    fraglike = lambda r: (r["ip2"] < 5.0 and r["t2"] > 0 and r["cos12"] > 0.9)
    VARIANTS = [("current: any photon prong E > emin", lambda r: True),
                ("+ not fragment-like (ip2<5, downstream, cos>0.9)", lambda r: not fraglike(r)),
                ("+ crossing point within 100 cm of both starts", lambda r: r["dcmax"] < 100),
                ("+ crossing within 200 cm of both starts", lambda r: r["dcmax"] < 200),
                ("+ both axes point away from the crossing (behind) and skew < 30", lambda r: r["behind"] > 0 and r["skew"] < 30),
                ("+ behind, skew < 30, dcmax < 150", lambda r: r["behind"] > 0 and r["skew"] < 30 and r["dcmax"] < 150),
                ("+ crossing within 50 cm of the reco vertex", lambda r: np.isfinite(r["dcv"]) and r["dcv"] < 50),
                ("+ start-to-start < 100 cm", lambda r: r["dstart"] < 100),
                ("+ other E > 20 MeV", lambda r: r["E2"] > 20),
                ("+ not fragment-like AND dcmax < 150", lambda r: (not fraglike(r)) and r["dcmax"] < 150),
                ("+ not fragment-like AND behind AND skew < 30", lambda r: (not fraglike(r)) and r["behind"] > 0 and r["skew"] < 30)]
    evi = np.array([e["ev"] for e in evs], int)
    L.append(f"\nVETO VARIANTS on the vertex-stage sample (w): PASS cell contents | FAIL cell contents | data pass / fail")
    L.append(f"{'variant':62s}{'sigFV pass':>11}{'sigEnt pass':>12}{'>=2g pass':>10}{'CCnumu pass':>12}{'other pass':>11}{'EXT pass':>9}{'data pass':>10}{'d/p pass':>9} || {'sigFV fail':>11}{'>=2g fail':>10}{'EXT fail':>9}{'data fail':>10}{'d/p fail':>9}")
    for lab, cond in VARIANTS:
        f = fires(rows, cond); fx = fires(rx, cond); fd = fires(rd, cond)
        passm = np.array([e["ev"] not in f for e in evs], bool)
        wpass = w[evi[passm]]; cpass = cat[evi[passm]]; wfail = w[evi[~passm]]; cfail = cat[evi[~passm]]
        ext_pass = sum(1 for e in evx if e["ev"] not in fx) * wx_scale; ext_fail = len(evx) * wx_scale - ext_pass
        dpass = sum(1 for e in evd if e["ev"] not in fd); dfail = len(evd) - dpass
        sfv = wpass[np.isin(cpass, [0, 1])].sum(); sen = wpass[np.isin(cpass, [2, 3])].sum(); g2 = wpass[np.isin(cpass, [4, 9])].sum()
        cc = wpass[cpass == 7].sum(); oth = wpass[np.isin(cpass, [5, 6, 8, 10])].sum()
        predp = wpass.sum() + ext_pass; predf = wfail.sum() + ext_fail
        L.append(f"{lab[:62]:62s}{sfv:11.1f}{sen:12.1f}{g2:10.1f}{cc:12.1f}{oth:11.1f}{ext_pass:9.1f}{dpass:10d}{dpass / max(predp, 1e-9):9.2f} || "
                 f"{wfail[np.isin(cfail, [0, 1])].sum():11.1f}{wfail[np.isin(cfail, [4, 9])].sum():10.1f}{ext_fail:9.1f}{dfail:10d}{dfail / max(predf, 1e-9):9.2f}")
    txt = "\n".join(L); print(txt)
    open(os.path.join(args.plots, "veto2g_study.txt"), "w").write(txt + "\n")
    # ---- plots: pair variables by class (MC, veto-firing prongs) + data overlay ---
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    SPECS = [("dstart", "start-to-start distance [cm]", np.linspace(0, 600, 41)), ("cos12", "cos(axis1, axis2)", np.linspace(-1, 1, 41)),
             ("ip2", "other start to candidate axis [cm]", np.linspace(0, 300, 41)), ("t2", "other start projection along candidate [cm]", np.linspace(-300, 300, 41)),
             ("ip1", "candidate start to other axis [cm]", np.linspace(0, 300, 41)), ("skew", "axes closest approach [cm]", np.linspace(0, 200, 41)),
             ("dcmax", "crossing point to farther start [cm]", np.linspace(0, 600, 41)), ("dc1", "crossing point to candidate start [cm]", np.linspace(0, 400, 41)),
             ("dcv", "crossing point to reco vertex [cm]", np.linspace(0, 400, 41)), ("behind", "both axes point away from crossing", np.array([-0.5, 0.5, 1.5])),
             ("E2", "other shower E [MeV]", np.linspace(0, 200, 41)), ("obj2", "other shower objectness", np.linspace(0, 1, 26)),
             ("ph2", "other shower photon score", np.linspace(0, 1, 26)), ("nhits2", "other shower n hits", np.linspace(0, 400, 41)),
             ("d2v", "other start to reco vertex [cm]", np.linspace(0, 400, 41)), ("mgg", "pi0-hypothesis m_gg [MeV] (reported)", np.linspace(0, 600, 41))]
    for sel_name, selm, selx, seld in (("vetofiring", veto_now, lambda r: r["pid2"] == 22 and r["E2"] > args.veto2g_emin, None), ("allpairs", np.ones(len(rows), bool), lambda r: True, None)):
        d = os.path.join(args.plots, sel_name); os.makedirs(d, exist_ok=True)
        for key, xlab, bins in SPECS:
            v = col(rows, key); lo, hi = bins[0], bins[-1] - 1e-6
            fig, ax = plt.subplots(figsize=(7.4, 4.8))
            data = [np.clip(v[selm & (cls == c) & np.isfinite(v)], lo, hi) for c in range(5)]; ws = [wpair[selm & (cls == c) & np.isfinite(v)] for c in range(5)]
            vx_ = np.array([r[key] for r in rx if selx(r)], float) if rx else np.zeros(0); vx_ = vx_[np.isfinite(vx_)]
            data.append(np.clip(vx_, lo, hi)); ws.append(np.full(len(vx_), wx_scale))
            ax.hist(data, bins=bins, weights=ws, stacked=True, color=CLS_COL + ["#bbbbbb"], label=[f"{CLS[c]} ({ws[c].sum():.1f})" for c in range(5)] + [f"EXT ({ws[5].sum():.1f})"])
            vd = np.array([r[key] for r in rd if seld is None and selx(r)], float) if rd else np.zeros(0); vd = vd[np.isfinite(vd)]
            if len(vd):
                hd, _ = np.histogram(np.clip(vd, lo, hi), bins=bins); ctr = 0.5 * (bins[:-1] + bins[1:])
                ax.errorbar(ctr[hd > 0], hd[hd > 0], yerr=np.sqrt(hd[hd > 0]), fmt="ko", ms=3.5, label=f"beam data ({int(hd.sum())})")
            ax.set(xlabel=xlab, ylabel=f"prongs / {args.pot:.1e} POT", title=f"{xlab}  [{sel_name}]"); ax.legend(fontsize=7); ax.grid(alpha=0.3)
            fig.tight_layout(); fig.savefig(os.path.join(d, f"{key}.png"), dpi=110); plt.close(fig)
    # 2-D: dcmax vs E2 and ip2 vs t2 for the veto-firing prongs, by class
    for (kx, ky, xl, yl, xr, yr) in (("E2", "dcmax", "other E [MeV]", "crossing to farther start [cm]", (0, 200), (0, 600)), ("t2", "ip2", "projection along candidate [cm]", "distance to candidate axis [cm]", (-300, 300), (0, 300))):
        fig, ax = plt.subplots(figsize=(7, 5.5))
        for c in range(5):
            m = veto_now & (cls == c)
            ax.scatter(np.clip(col(rows, kx)[m], *xr), np.clip(col(rows, ky)[m], *yr), s=6, alpha=0.5, color=CLS_COL[c], label=f"{CLS[c]} ({wpair[m].sum():.1f})")
        if rd:
            xd = np.array([r[kx] for r in rd if r["pid2"] == 22 and r["E2"] > args.veto2g_emin]); yd = np.array([r[ky] for r in rd if r["pid2"] == 22 and r["E2"] > args.veto2g_emin])
            ax.scatter(np.clip(xd, *xr), np.clip(yd, *yr), s=14, marker="x", color="k", label=f"beam data ({len(xd)})")
        ax.set(xlabel=xl, ylabel=yl, xlim=xr, ylim=yr, title="veto-firing prongs"); ax.legend(fontsize=7); ax.grid(alpha=0.3)
        fig.tight_layout(); fig.savefig(os.path.join(args.plots, f"scatter_{ky}_vs_{kx}.png"), dpi=110); plt.close(fig)
    np.savez(os.path.join(args.plots, "pairs_mc.npz"), **{k: col(rows, k) for k in rows[0].keys()} if rows else {}, w=wpair, cat=catp)
    print(f">>> -> {args.plots}")


if __name__ == "__main__":
    main()
