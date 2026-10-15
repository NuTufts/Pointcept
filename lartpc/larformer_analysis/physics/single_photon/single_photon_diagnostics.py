"""Diagnostic distributions for the single-photon selection: MC truth categories
+ EXT stacked, beam data as points, for events passing the selection (final
step) and for the step before the flash-chi2 cut. Same selection machinery /
options as single_photon_selection.py (imported), so the populations are the
ones in the cutflow.

Variables (per selected event, leading photon candidate):
  flashPE      : total in-time observed PE -- the ntuple's `flashObsPE` when
                 present (table-gamma exporter), else summed from the cascade
  flashPredPE  : predicted PE of the nu slice (`nuSlicePredPE`)
  flashPERatio : flashObsPE / nuSlicePredPE -- gamma-independent flash
                 consistency; a neutrino slice should sit near 1
  flashTimeUs  : in-time flash time
  startX/Y/Z   : candidate shower start point (NOTE: vertex-less starts are the
                 keypoint-model start, see README TODO)
  cosBeam      : shower direction cos(theta) w.r.t. +z
  nProtons     : nu-stream tracks with LArFormer PID 2212 and KE > --proton-ke-min
  flashDist    : distance in (y,z) between the PE-weighted PMT centroid and the
                 intersection of the shower LINE (start + t*dir, either sign of
                 t) with the x=0 plane; NaN if |dir_x| < 1e-3
  flashDistFwd : same but only counted when the intersection is FORWARD (t>0),
                 else NaN -- separates "light ahead" from "light behind"
  vtxDwall     : wall distance of the reco primary vertex; 0 for events in the
                 NO-VERTEX path (candidate not on a primary in-FV vertex)
  startDwall   : wall distance of the candidate shower start point (all showers)
  distToWall   : distance from start to the TPC wall along +dir
  distFromWall : distance from start to the TPC wall along -dir
  bdtCosmic    : candidate cosmic-shower BDT score (vertex path; the --shower-bdt-min cut)
  bdtNoVtx     : candidate vertex-free shower BDT score (no-vertex path; --novtx-bdt-min cut)
  bdtScore     : the unified score build_steps compares to --shower-bdt-min
                 (both NaN when the start is outside the TPC box)
  flashPE vs E : 2-D MC-signal / EXT scatter (light per unit shower energy)
Sample selectors: --path 0|1 (vertex | no-vertex candidate) and --veto2g
pass|fail (second-photon veto outcome, --veto2g-emin) -> the 2x2 cells.

Flash PE needs the cascade files: --{mc,ext,data}-cascade = keypoint2_streams
dirs (comma-separated, one per ntuple half; the maps are merged); the (run,subrun,event)->file maps are the cached rse_*.npz of the pi0
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
            "trackVtxIdx", "vtxX", "vtxY", "vtxZ"]
# flash quantities carried by the table-gamma exporter (no cascade files
# needed); absent on older ntuples, in which case --*-cascade still works
BR_FLASH = ["flashObsPE", "nuSlicePredPE", "flashTimeUs", "flashGammaEff"]


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
    ap.add_argument("--truth-scale", default="deposited", choices=["deposited", "legacy"])
    ap.add_argument("--signal-def", default="lar",
                    choices=list(SPS.SIGNAL_DEFS) + list(SPS.SIGNAL_DEF_ALIASES),
                    help="truth signal definition: 'lar' (alias 'present'), 'wc' or 'mb'")
    ap.add_argument("--shower-bdt-min", type=float, default=0.192)
    ap.add_argument("--bdt-after-multiplicity", action="store_true")
    ap.add_argument("--flashchi2-cut", type=float, default=316.2)
    ap.add_argument("--flashpe-ratio-max", type=float, default=3.0)
    ap.add_argument("--vtx-cosmic-bdt-min", type=float, default=0.6,
                    help="S6 pre-selection: vertex-path candidate cosmic BDT >= this (0 disables)")
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


def rays_to_wall(st, dr, sel=None):
    """Vectorized distance from each start along +dir to the TPC box boundary.
    NaN where the start is outside the box, the direction is not finite, or
    `sel` is False. st, dr: (n,3)."""
    n = len(st)
    out = np.full(n, np.nan)
    ok = (np.isfinite(st).all(1) & np.isfinite(dr).all(1)
          & (st >= TPC_LO).all(1) & (st <= TPC_HI).all(1))
    if sel is not None:
        ok = ok & sel
    if not ok.any():
        return out
    S, D = st[ok], dr[ok]
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(D > 0, (TPC_HI - S) / D, (TPC_LO - S) / D)
    t[np.abs(D) < 1e-9] = np.inf
    t[t < 0] = np.inf
    tmin = t.min(1)
    tmin[~np.isfinite(tmin)] = np.nan
    out[ok] = tmin
    return out


def per_sample(ntuple, is_mc, args, cascade, tag):
    have_flash = True
    import single_photon_mb_selection as MBS          # lazy: avoids an import cycle
    xtra = tuple(BR_EXTRA) + tuple(MBS.BR_EXTRA)
    try:
        a, pot = SPS.load(ntuple, is_mc, args.novtx, extra=xtra + tuple(BR_FLASH))
    except RuntimeError:        # older ntuple without the flash branches
        a, pot = SPS.load(ntuple, is_mc, args.novtx, extra=xtra)
        have_flash = False
    ex = a                      # load() concatenates comma-separated files
    n = len(a["run"])
    R = SPS.reco_tables(a, args)
    R["novtx"] = args.novtx
    names, S, E_cand, score_cand, n_g, path_cand = SPS.build_steps(R, args.shower_bdt_min, args.flashchi2_cut, args.bdt_after_multiplicity, args.flashpe_ratio_max, veto2g_dcmax=getattr(args, 'veto2g_dcmax', 0.0))
    idx = R["_cand_idx"]
    # MB-chain features (hadronic side, topology) + the three chain BDT scores
    import single_photon_mb_bdt_chain as MBC
    mdl = getattr(args, "mb_chain_model", None)
    chain = MBC.chain_scores(a, R, names, S, E_cand, path_cand, args, mdl)
    if chain is None:
        from mb_variable_study import features_from
        Mf = features_from(a, R, names, S, E_cand, path_cand, args); chain = {}
    else:
        Mf = chain["_features"]
        print(f">>> {tag}: MB-chain scores from {os.path.basename(mdl)}", flush=True)
    pathsel = getattr(args, "path", None)
    v2 = getattr(args, "veto2g", None)
    ct = getattr(args, "contain", None)
    dfs = getattr(args, "dfw", None)
    ens = getattr(args, "enter", None)
    cbm = getattr(args, "cand_bdt_min", None)
    cxm, cnm = getattr(args, "chain_ext_min", None), getattr(args, "chain_numu_min", None)
    cnx = getattr(args, "chain_numu_max", None)
    prs = getattr(args, "protons", None)
    if (ens is not None or cxm is not None or cnm is not None or cnx is not None) and "enter" not in chain:
        sys.exit("--enter / --chain-*-min need the MB-chain model (--mb-chain-model)")
    if any(x is not None for x in (pathsel, v2, ct, dfs, ens, cbm, cxm, cnm, cnx, prs)):
        S = S.copy()
        keep = np.ones(len(path_cand), bool)
        if pathsel is not None:
            keep &= (path_cand == pathsel)
        if v2 is not None:            # 'fail' = another photon prong above e_min
            fails = R["other_gE_max"] > args.veto2g_emin
            keep &= (fails if v2 == "fail" else ~fails)
        if ct is not None:            # 'pc' = a nu-stream track ends/starts < dwall of a wall
            pcm = R["track_min_dwall"] < args.contain_dwall
            keep &= (pcm if ct == "pc" else ~pcm)
        if dfs is not None:           # 'in' = backward wall distance >= cut (in-TPC-like)
            din = R["cand_dist_from_wall"] >= args.dfw_cut
            keep &= (din if dfs == "in" else ~din)
        if ens is not None:           # 'in' = chain BDT_enter >= cut (in-FV-like), 'ent' = below
            thr = np.full(len(path_cand), args.enter_cut)
            if getattr(args, "enter_cut_0p", None) is not None:      # proton-dependent threshold
                thr = np.where(np.asarray(Mf["n_p"]) >= 1, args.enter_cut, args.enter_cut_0p)
            ein = np.nan_to_num(chain["enter"], nan=-1.0) >= thr
            keep &= (ein if ens == "in" else ~ein)
        if cxm is not None:
            keep &= np.nan_to_num(chain["ext"], nan=-1.0) > cxm
        if cnm is not None:
            keep &= np.nan_to_num(chain["numu"], nan=-1.0) > cnm
        if prs is not None:           # '0p' = no accepted proton (>= --proton-ke-min), 'np' = at least one
            npm = np.asarray(Mf["n_p"]) >= 1
            keep &= (npm if prs == "np" else ~npm)
        if cnx is not None:           # the CC-numu-like complement: BDT_numu (flattened) <= this
            keep &= np.nan_to_num(chain["numu"], nan=-1.0) <= cnx
        if cbm is not None:
            raw = np.where(path_cand == 1, score_cand + args.novtx_bdt_min - args.shower_bdt_min,
                           score_cand)
            keep &= np.nan_to_num(raw, nan=-9.0) >= cbm
        for k in range(2, len(names)):
            S[k] = S[k] & keep
    out = dict(steps=S, names=names, E=E_cand, path=path_cand,
               run=np.asarray(R["run"], np.int64), subrun=np.asarray(R["subrun"], np.int64), event=np.asarray(R["event"], np.int64),
               logchi2=np.log10(np.clip(np.nan_to_num(R["chi2"], nan=-1.0), 1e-3, None)))
    out["mupiKEmax"] = np.asarray(Mf["mupi_kemax"], np.float64)
    for nm in ("ext", "numu", "enter"):
        out["bdtChain" + nm.capitalize()] = chain.get(nm, np.full(n, np.nan))
    # EXTRA chain models (plot only, never cut): scores named bdtChain<Nm>_<tag>
    for mp in [x for x in (getattr(args, "mb_chain_model_extra", "") or "").split(",") if x]:
        if os.path.abspath(mp) == os.path.abspath(mdl or ""):
            continue
        ch2 = MBC.chain_scores(a, R, names, S, E_cand, path_cand, args, mp)
        if ch2 is None:
            print(f">>> {tag}: extra chain model {mp} absent, skipped", flush=True); continue
        t2 = os.path.basename(mp).replace("mb_bdt_chain_", "").replace(".joblib", "")
        for nm in ("ext", "numu", "enter"):
            out[f"bdtChain{nm.capitalize()}_{t2}"] = ch2[nm]
        print(f">>> {tag}: extra MB-chain scores from {os.path.basename(mp)} as *_{t2}", flush=True)
    out["logchi2"][~np.isfinite(R["chi2"]) | (R["chi2"] <= 0)] = np.nan

    def pick(x, fill=np.nan):
        v = np.asarray(ak.to_numpy(ak.fill_none(ak.firsts(x[idx]), fill)), np.float64)
        v[n_g == 0] = np.nan
        return v
    st = np.stack([pick(ex[f"showerStartPos{c}"]) for c in "XYZ"], 1)
    dr = np.stack([pick(ex[f"showerStartDir{c}"]) for c in "XYZ"], 1)
    out["startX"], out["startY"], out["startZ"] = st.T
    # candidate photon BDT score(s): build_steps compares ONE unified score to
    # --shower-bdt-min (vertex-less candidates carry NoVtx - novtx_bdt_min +
    # shower_bdt_min); undo the shift so each path shows its own raw score.
    sc = np.asarray(score_cand, np.float64).copy()
    sc[~np.isfinite(E_cand)] = np.nan
    raw_nv = sc + args.novtx_bdt_min - args.shower_bdt_min
    out["bdtScore"] = sc                                   # unified (selection) score
    out["bdtCosmic"] = np.where(path_cand == 0, sc, np.nan)    # vertex path: cosmic BDT
    out["bdtNoVtx"] = np.where(path_cand == 1, raw_nv, np.nan)  # vertex-less path: NoVtx BDT
    out["cosBeam"] = dr[:, 2]
    out["E_QE"] = MBS.e_qe(np.asarray(E_cand, np.float64), dr[:, 2])   # e-CCQE hypothesis (reported only)
    with np.errstate(invalid="ignore"):
        out["startDwall"] = np.minimum((st - TPC_LO).min(1), (TPC_HI - st).min(1))
    vx = np.stack([np.asarray(ak.to_numpy(ex["vtx" + c]), np.float64) for c in "XYZ"], 1)
    vdw = np.minimum((vx - TPC_LO).min(1), (TPC_HI - vx).min(1))
    vdw = np.where(R["vtx_ok"] & (path_cand == 0), vdw, 0.0)     # no-vertex path -> 0
    out["vtxDwall"] = vdw
    # protons: nu-stream tracks (any vertex incl. orphans when --novtx)
    is_p = (ex["trackLArFormerPID"] == 2212) & (ex["trackRecoE"] > args.proton_ke_min)
    if args.novtx and "trackStream" in ak.fields(ex):
        is_p = is_p & (ex["trackStream"] == 0)
    out["nProtons"] = ak.to_numpy(ak.sum(is_p, axis=1)).astype(np.float64)
    # walls
    kc = SPS.step_index(names, "flash chi2") - 1
    plotted = S[kc]                  # superset of every drawn step
    out["distToWall"] = rays_to_wall(st, dr, plotted)
    out["distFromWall"] = rays_to_wall(st, -dr, plotted)

    # flash PE + centroid vs shower line (only for events past S3, i.e. the
    # step before the chi2 cut; index -2)
    pe = np.full(n, np.nan); fd = np.full(n, np.nan); fdf = np.full(n, np.nan)
    ppe = np.full(n, np.nan); rat = np.full(n, np.nan); tus = np.full(n, np.nan)
    if have_flash:
        pe = np.asarray(ak.to_numpy(a["flashObsPE"]), np.float64)
        ppe = np.asarray(ak.to_numpy(a["nuSlicePredPE"]), np.float64)
        tus = np.asarray(ak.to_numpy(a["flashTimeUs"]), np.float64)
        pe[pe < 0] = np.nan; ppe[ppe <= 0] = np.nan; tus[tus < -100] = np.nan
        rat = pe / ppe
        print(f">>> {tag}: flash PE from the ntuple ({int(np.isfinite(pe).sum())} events)",
              flush=True)
    out["flashPredPE"], out["flashPERatio"], out["flashTimeUs"] = ppe, rat, tus
    if cascade:
        ok = (np.isfinite(st).all(1) & np.isfinite(dr).all(1)
              & (st >= TPC_LO).all(1) & (st <= TPC_HI).all(1))
        m = {}
        for cdir in [c for c in cascade.split(",") if c]:
            cache = os.path.join(HERE, "..", "pi0mass_peak",
                                 "rse_" + os.path.basename(os.path.dirname(cdir.rstrip("/")))
                                 + "_" + os.path.basename(cdir.rstrip("/")) + ".npz")
            m.update(rse_map(cdir, cache))        # builds + caches if absent
        run, sub, evt = R["run"], R["subrun"], R["event"]
        need = np.nonzero(S[kc])[0]
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
            if not have_flash:
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
    ap.add_argument("--veto2g", default=None, choices=["pass", "fail"],
                    help="restrict to events that pass / fail the second-photon veto "
                         "(another nu-stream photon prong with reco E > --veto2g-emin)")
    ap.add_argument("--veto2g-emin", type=float, default=10.0)
    ap.add_argument("--veto2g-dcmax", type=float, default=0.0,
                    help="geometric 2nd-photon veto: axes must cross within this [cm] of both starts (0 = old veto)")
    ap.add_argument("--contain", default=None, choices=["fc", "pc"],
                    help="restrict to FULLY (fc) or PARTIALLY (pc) contained events: pc = any "
                         "nu-stream track prong start/end < --contain-dwall from a TPC wall")
    ap.add_argument("--contain-dwall", type=float, default=10.0)
    ap.add_argument("--dfw", default=None, choices=["in", "ent"],
                    help="restrict by the candidate's backward wall distance: 'in' >= --dfw-cut "
                         "(in-TPC-like), 'ent' < cut or start outside (entering-like)")
    ap.add_argument("--dfw-cut", type=float, default=SPS.DFW_CUT)
    ap.add_argument("--enter", default=None, choices=["in", "ent"],
                    help="split the vertex-pass-FC cell by the MB-chain BDT_enter score instead of dwall: "
                         "'in' = score >= --enter-cut (in-FV-like), 'ent' = below (entering-like)")
    ap.add_argument("--enter-cut", type=float, default=0.05)
    ap.add_argument("--unit-weight", action="store_true", help="weight every MC event 1 (particle-gun samples; xsecWeight = -1)")
    ap.add_argument("--protons", default=None, choices=["0p", "np"],
                    help="restrict to events with no accepted proton (0p) or at least one (np)")
    ap.add_argument("--enter-cut-0p", type=float, default=None,
                    help="BDT_enter threshold for events with NO accepted proton (>= --proton-ke-min); --enter-cut then applies to Np")
    ap.add_argument("--chain-ext-min", type=float, default=None, help="extra cut: MB-chain BDT_ext (flattened) > this")
    ap.add_argument("--chain-numu-min", type=float, default=None, help="extra cut: MB-chain BDT_numu (flattened) > this")
    ap.add_argument("--chain-numu-max", type=float, default=None, help="complement: keep MB-chain BDT_numu (flattened) <= this (CC-numu-like cell)")
    for k, v in (("mb_mu_kemin", 40.0), ("mb_pi_kemin", 50.0), ("mb_p_kemax", 342.0), ("extra_emax", 10.0),
                 ("cone_e2_min", 5.0), ("cone_e2_max", 600.0)):
        ap.add_argument("--" + k.replace("_", "-"), type=float, default=v, help="MB-chain feature option")
    ap.add_argument("--mb-chain-model", default=os.path.join(HERE, "plots_run1_mbchain", "mb_bdt_chain_noflash_novtx.joblib"),
                    help="joblib from single_photon_mb_bdt_chain.py; its three scores are plotted (skipped if absent)")
    ap.add_argument("--mb-chain-model-extra",
                    default=os.path.join(HERE, "plots_run1_mbchain", "mb_bdt_chain_noflash_novtx_lar.joblib"),
                    help="comma-separated extra chain joblibs whose scores are PLOTTED as bdtChain*_<tag> (no cuts); "
                         "default = the lar-truth model; '' to disable")
    ap.add_argument("--cand-bdt-min", type=float, default=None,
                    help="extra cut: leading candidate's raw BDT score (cosmic BDT on the vertex path, "
                         "NoVtx BDT on the vertex-less path) >= this; candidate set unchanged")
    args = ap.parse_args()
    SPS.configure(args)
    os.makedirs(args.plots, exist_ok=True)

    print(">>> MC ...", flush=True)
    M, a, pot, R = per_sample(args.mc_ntuple, True, args, args.mc_cascade, "MC")
    w0 = ak.to_numpy(a["xsecWeight"]).astype(np.float64)
    w = np.where(w0 > 0, w0, 0.0) * (args.pot / pot)
    if getattr(args, "unit_weight", False):        # particle-gun samples: count events, no POT scaling
        w = np.ones(len(w0))
    if args.mc_odd_only:
        odd = ak.to_numpy(a["event"]) % 2 == 1
        w = np.where(odd, 2.0 * w, 0.0)
    T = SPS.truth_tables(a)
    cat = T["cat"]
    # MC-truth-only variables (no EXT / data entries)
    M["trueOpenAngle"] = T["g_open_min_true"]
    M["trueOpenAngleSameVtx"] = T["g_open_min_samevtx_true"]
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
        ("flashPredPE", "predicted PE, nu slice", np.linspace(0, 3000, 31)),
        ("flashPERatio", "flash PE observed / predicted", np.linspace(0, 4, 41)),
        ("flashTimeUs", "in-time flash time [us]", np.linspace(2.5, 6.0, 36)),
        ("startX", "candidate shower start x [cm]", np.linspace(-20, 276, 38)),
        ("startY", "candidate shower start y [cm]", np.linspace(-130, 130, 27)),
        ("startZ", "candidate shower start z [cm]", np.linspace(-20, 1060, 37)),
        ("cosBeam", r"shower $\cos\theta_{beam}$", np.linspace(-1, 1, 41)),
        ("nProtons", f"n reco protons (KE > {args.proton_ke_min:.0f} MeV)",
         np.arange(-0.5, 5.5, 1)),
        ("flashDist", "|PE centroid - shower-line hit at x=0| [cm]", np.linspace(0, 600, 31)),
        ("flashDistFwd", "same, forward intersections only [cm]", np.linspace(0, 600, 31)),
        ("vtxDwall", "reco vertex dwall [cm] (0 = no-vertex path)", np.linspace(0, 130, 27)),
        ("startDwall", "candidate shower start dwall [cm]", np.linspace(0, 130, 27)),
        ("distToWall", "distance to wall along +dir [cm]", np.linspace(0, 1000, 41)),
        ("distFromWall", "distance to wall along -dir [cm]", np.linspace(0, 1000, 41)),
        ("E", "candidate photon energy [MeV]", np.linspace(0, 1000, 41)),
        ("E_QE", "E_QE [MeV] (electron-CCQE hypothesis from E and cos theta; reported, never cut)", np.linspace(0, 1500, 31)),
        ("bdtCosmic", f"candidate cosmic-shower BDT score, vertex path (cut >= {args.shower_bdt_min:.3f})",
         np.linspace(0, 1, 41)),
        ("bdtNoVtx", f"candidate vertex-free shower BDT score, no-vertex path (cut >= {args.novtx_bdt_min:.2f})",
         np.linspace(0, 1, 41)),
        ("bdtScore", "candidate unified selection score (cosmic BDT | shifted NoVtx BDT)",
         np.linspace(0, 1, 41)),
        ("mupiKEmax", "max mu/pi-PID track KE [MeV] (-1 = none)", np.linspace(-20, 600, 32)),
        ("bdtChainExt", "MB-chain BDT_ext flattened score (signal-uniform)", np.linspace(0, 1, 26)),
        ("bdtChainNumu", "MB-chain BDT_numu flattened score (signal-uniform)", np.linspace(0, 1, 26)),
        ("bdtChainEnter", "MB-chain BDT_enter flattened score (1 = in-FV-like)", np.linspace(0, 1, 26)),
    ] + [(k, f"MB-chain {k[8:].split('_')[0]} flattened score, model {k.split('_', 1)[1]}", np.linspace(0, 1, 26))
         for k in sorted(M.keys()) if k.startswith("bdtChain") and "_" in k] + [
        ("trueOpenAngle", "TRUTH min opening angle between detectable photons [deg] (MC, >= 2 photons)",
         np.linspace(0, 180, 37)),
        ("trueOpenAngleSameVtx", "TRUTH min opening angle, same-start-point pairs only [deg] (MC)",
         np.linspace(0, 180, 37)),
    ]
    MC_ONLY = {"trueOpenAngle", "trueOpenAngleSameVtx"}
    kc = SPS.step_index(M["names"], "flash chi2")
    steps = [(kc - 1, "preChi2")]
    if any("PE ratio" in n for n in M["names"]):
        steps.append((SPS.step_index(M["names"], "PE ratio") - 1, "preRatio"))
    if any("cosmic BDT" in n for n in M["names"]):
        steps.append((SPS.step_index(M["names"], "cosmic BDT") - 1, "preBDT"))
    steps.append((len(M["names"]) - 1, "final"))
    lines = []
    for k, sname in steps:
        d = os.path.join(args.plots, sname); os.makedirs(d, exist_ok=True)
        lines.append(f"== {M['names'][k]} : weighted medians per category")
        for key, xlab, bins in SPECS:
            if not np.isfinite(M[key]).any():
                continue
            lo, hi = bins[0], bins[-1] - 1e-6
            m = M["steps"][k] & np.isfinite(M[key])
            data = [np.clip(M[key][m & (cat == c)], lo, hi) for c in range(NMC)]
            ws = [w[m & (cat == c)] for c in range(NMC)]
            cols = list(COL[:NMC])
            mx = md = None
            if X is not None and key not in MC_ONLY:
                mx = X["steps"][k] & np.isfinite(X[key])
                data.append(np.clip(X[key][mx], lo, hi)); ws.append(wx[mx]); cols.append(COL[NMC])
            labs = [f"{CATS[c]} ({ws[c].sum():.1f})" for c in range(len(ws))]
            fig, ax = plt.subplots(figsize=(7.4, 4.8))
            ax.hist(data, bins=bins, weights=ws, stacked=True, color=cols, label=labs)
            if D is not None and key not in MC_ONLY:
                md = D["steps"][k] & np.isfinite(D[key])
                hd, _ = np.histogram(np.clip(D[key][md], lo, hi), bins=bins)
                ctr = 0.5 * (bins[:-1] + bins[1:])
                ax.errorbar(ctr[hd > 0], hd[hd > 0], yerr=np.sqrt(hd[hd > 0]), fmt="ko",
                            ms=3.5, label=f"beam data ({int(hd.sum())})")
            ax.set(xlabel=xlab, ylabel="events" if getattr(args, "unit_weight", False) else f"events / {args.pot:.1e} POT",
                   title=f"{xlab}  ({M['names'][k]})")
            ax.legend(fontsize=6.5); ax.grid(alpha=0.3)
            fig.tight_layout(); fig.savefig(f"{d}/{key}.png", dpi=110); plt.close(fig)
            # medians
            med = []
            fm = ".3f" if key.startswith("bdt") else ".1f"
            for c in range(NMC):
                v = M[key][m & (cat == c)]; ww = w[m & (cat == c)]
                if ww.sum() > 0:
                    o = np.argsort(v); cw = np.cumsum(ww[o]) / ww.sum()
                    med.append(f"{CATS[c][:16]} {v[o][np.searchsorted(cw, 0.5)]:{fm}}")
            if mx is not None and mx.any():
                med.append(f"EXT {np.median(X[key][mx]):{fm}}")
            if md is not None and md.any():
                med.append(f"data {np.median(D[key][md]):{fm}}")
            lines.append(f"  {key:13s}: " + " | ".join(med))
            if key in MC_ONLY:
                # share of each category with a second detectable photon, and of
                # those, the share with the min opening angle below 20 deg
                parts = []
                for c in range(NMC):
                    mc_ = M["steps"][k] & (cat == c); W_ = w[mc_].sum()
                    if W_ <= 0:
                        continue
                    has2 = mc_ & np.isfinite(M[key]); W2 = w[has2].sum()
                    col_ = w[has2 & (M[key] < 20.0)].sum()
                    parts.append(f"{CATS[c][:16]} >=2g {W2 / W_:.2f}, <20deg {col_ / max(W2, 1e-9):.2f} "
                                 f"({col_:.1f}/{W_:.1f})")
                lines.append(f"  {key:13s}  frac with >=2 detectable photons; of those, frac < 20 deg (w <20 / w all): "
                             + " | ".join(parts))
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
    if X is not None:
        np.savez(os.path.join(args.plots, "diag_ext.npz"), w=wx, **{k: v for k, v in X.items() if isinstance(v, np.ndarray)})
    if D is not None:
        np.savez(os.path.join(args.plots, "diag_data.npz"), w=wd, **{k: v for k, v in D.items() if isinstance(v, np.ndarray)})
    print(f">>> plots -> {args.plots}")


if __name__ == "__main__":
    main()
