"""Flash-match quality study from keypoint2 files (slices/ + flash/ tables).

Per event, extracts each stream's CHOSEN slice and asks whether the choice was
the true nu interaction. Correctness = CHARGE FRACTION (the eval's slicer
metric C, event-level): the chosen slice must collect >= --qfrac-cut (default
0.50, matching eval SLICE_COVERAGE) of the nu interaction's de-double-counted
charge (calo.dedup_charge over the nu-origin truth-matched spacepoints; the
in-slice numerator deduped over its own subset, mirroring the eval). This
directly measures slice choice, unlike the earlier decoded-vertex proxy which
convolved in keypoint-decode resolution (slices containing the true vertex
were scored wrong when the decode missed by >5 cm):

  nu stream: the slicer's nu-union row (label == 'nu' in slices/).
  fm stream: the chi2_rank==1 row. When that row IS the nu union, no separate
             _fm_0.h5 exists (the nu file carries stream='nu,flashmatch'), so
             pairing BOTH per-stream lists per event is required -- iterating
             the fm list alone would drop exactly the fm stream's correct
             choices.

This cannot run off the eval npz (per-particle records, no flash info) or the
exported ntuple (recoVtxFlashChi2 only -- no PE vectors); the kp2 files hold
the per-slice chi2, per-slice predicted PE (PhotonLib, drift-corrected) and
the observed in-time beam flash.

Shard + merge exactly like eval_reco_performance:

    PYTHONPATH=./ python3 lartpc/larformer_reco/eval/flashmatch_quality.py \
        --kp2-nu-list ... --kp2-fm-list ... --out shard.npz [--start I --n N]
    PYTHONPATH=./ python3 ... --merge 'dir/fmq_shard*.npz' --out merged.npz \
        --plots plots/dir

Plots (correct = solid/filled, incorrect = outline):
  chi2_{nu,fm}.png            chosen-slice chi2, correct vs incorrect
  pe_{nu,fm}_{correct,incorrect}.png
                              total predicted vs observed in-time PE overlay
  pe_ratio_{nu,fm}.png        predicted/observed total PE (calibration scale)
  pe_pred_vs_obs.png          2D pred vs obs (correct nu-union rows)
  nu_chi2_rank.png            chi2 rank of the (correct) nu union among all
                              ranked slices -- rank 1 = flash match alone
                              would have picked the nu slice
"""
import argparse
import glob
import os
import re
import sys

import numpy as np
import h5py

sys.path.insert(0, os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")))

# MicroBooNE TPC active volume [cm] -- same canonical bounds as
# eval_reco_performance --true-vtx-in-tpc
TPC_LO = (0.0, -116.5, 0.0)
TPC_HI = (256.35, 116.5, 1036.8)

KEYS = ["has_gt", "obs_pe", "vtx_in_tpc", "vtx_dwall", "qtrue", "vtx_x",
        "nu_present", "nu_chi2", "nu_pred_pe", "nu_p_nu", "nu_rank",
        "nu_correct", "nu_qfrac",
        "fm_present", "fm_chi2", "fm_pred_pe", "fm_is_nu", "fm_correct",
        "fm_qfrac",
        "n_ranked"]


def _index_by_event(paths):
    out = {}
    for p in paths:
        m = re.search(r"event(\d+)(_fm)?_0\.h5$", os.path.basename(p))
        if m:
            out[int(m.group(1))] = p
    return out


def _dwall(v):
    """Signed distance to the nearest TPC wall: >0 inside, <0 outside."""
    return float(min(v[0] - TPC_LO[0], TPC_HI[0] - v[0],
                     v[1] - TPC_LO[1], TPC_HI[1] - v[1],
                     v[2] - TPC_LO[2], TPC_HI[2] - v[2]))


def _dedup_sum(ctx, m):
    """De-double-counted comb charge sum over truth-row subset mask `m`
    (each set deduped over itself, mirroring the eval's _charge_sum)."""
    if not np.any(m):
        return 0.0
    from lartpc.larformer_reco.trajfit.calo import dedup_charge
    _, q = dedup_charge(ctx["pixval"][m], ctx["tick"][m], ctx["uwire"][m],
                        ctx["vwire"][m], ctx["ywire"][m])
    return float(q.sum())


def _nu_truth_ctx(msp_path):
    """Nu-origin truth spacepoints from the merged_sp file: per-row pixel
    columns + exact-position keys, and the GENIE vertex list."""
    with h5py.File(msp_path, "r") as f:
        e = f["entry_0"]
        d = e["mc_particle_tree/nu_vertices"]
        nv = (np.asarray(d[()], np.float64).reshape(-1, 3)
              if d.size else np.zeros((0, 3)))
        mt = e["mc_particle_tree"]
        nu_tids = np.asarray(mt["trackid"][()], np.int64)[
            np.asarray(mt["origin"][()], np.int64) == 1]
        td = e["triplet_data"]
        sel = np.isin(np.asarray(td["trackid"][()], np.int64), nu_tids)
        if not sel.any():
            return {"nv": nv, "n": 0}
        pos = td["pos"][()].astype(np.float32)[sel]
        ctx = {"nv": nv, "n": int(sel.sum()),
               "pos_bytes": [pos[i].tobytes() for i in range(len(pos))],
               "pixval": td["pixval"][()][sel],
               "tick": td["tick"][()][sel],
               "uwire": td["uwire"][()][sel],
               "vwire": td["vwire"][()][sel],
               "ywire": td["ywire"][()][sel]}
        ctx["qtrue"] = _dedup_sum(ctx, np.ones(ctx["n"], bool))
        return ctx


def _slice_qfrac(fsel, ctx):
    """Fraction of the nu interaction's dedup charge inside fsel's slice."""
    if ctx.get("n", 0) == 0 or ctx.get("qtrue", 0.0) <= 0 or fsel is None \
            or "slice" not in fsel:
        return np.nan
    sc = fsel["slice/coord_cm"][()].astype(np.float32)
    sset = {sc[i].tobytes() for i in range(len(sc))}
    m = np.asarray([b in sset for b in ctx["pos_bytes"]], bool)
    return _dedup_sum(ctx, m) / ctx["qtrue"]


def process(args):
    nu_by_ev = _index_by_event([l.strip() for l in open(args.kp2_nu_list)])
    fm_by_ev = _index_by_event([l.strip() for l in open(args.kp2_fm_list)])
    msp_by_base = {}
    if args.merged_sp_list:
        msp_by_base = {os.path.basename(l.strip()): l.strip()
                       for l in open(args.merged_sp_list)}
    events = sorted(set(nu_by_ev) | set(fm_by_ev))
    lo = args.start
    hi = len(events) if args.n is None else min(len(events), lo + args.n)
    events = events[lo:hi]
    print(f">>> {len(nu_by_ev)} nu / {len(fm_by_ev)} fm kp2 files; "
          f"this shard: events [{lo}:{hi}] ({len(events)})", flush=True)

    rec = {k: [] for k in KEYS}
    n_noflash = 0
    for ev in events:
        row = {k: np.nan for k in KEYS}
        row.update(nu_present=0, fm_present=0, fm_is_nu=0,
                   nu_correct=0, fm_correct=0, has_gt=0, nu_rank=0,
                   n_ranked=0)
        tctx = None
        fnu = h5py.File(nu_by_ev[ev], "r") if ev in nu_by_ev else None
        ffm = h5py.File(fm_by_ev[ev], "r") if ev in fm_by_ev else None
        base = fnu if fnu is not None else ffm
        try:
            if "slices" not in base or "flash" not in base:
                n_noflash += 1
                continue
            gt = None
            if bool(base.attrs.get("has_gt", False)):
                g = np.asarray(base["gt_nu_vertex_cm"][()],
                               np.float64).reshape(-1)[:3]
                if np.isfinite(g).all():
                    gt = g
            row["has_gt"] = int(gt is not None)
            if fnu is not None and "nu_vertex_cm" in fnu:
                v_ = np.asarray(fnu["nu_vertex_cm"][()],
                                np.float64).reshape(-1)
                if v_.size >= 3 and np.isfinite(v_[0]):
                    row["vtx_x"] = float(v_[0])   # drift coord (PMTs at x~0)
            # GENIE nu vertex -> in-TPC flag + signed wall distance (the
            # flash prediction only models ionization INSIDE the TPC, so
            # out-of-TPC interactions and boundary events with escaping
            # particles produce observed light we cannot predict).
            msp = msp_by_base.get(str(base.attrs.get("src_file", "")))
            if msp:
                try:
                    tctx = _nu_truth_ctx(msp)
                    if len(tctx["nv"]):
                        row["vtx_dwall"] = _dwall(tctx["nv"][0])
                        row["vtx_in_tpc"] = int(row["vtx_dwall"] > 0)
                    row["qtrue"] = float(tctx.get("qtrue", np.nan)) \
                        if tctx.get("n", 0) else 0.0
                except Exception:
                    tctx = None
            if "observed_pe" not in base["flash"]:
                n_noflash += 1        # no in-time beam flash: no chi2 anywhere
                continue
            row["obs_pe"] = float(np.nansum(base["flash/observed_pe"][()]))
            sl = base["slices"]
            labels = [l.decode() if isinstance(l, bytes) else str(l)
                      for l in sl["label"][()]]
            chi2 = np.asarray(sl["chi2"][()], np.float64)
            rank = np.asarray(sl["chi2_rank"][()], np.int64)
            pred = np.asarray(sl["pred_pe"][()], np.float64)
            p_nu = np.asarray(sl["p_nu"][()], np.float64)
            row["n_ranked"] = int((rank > 0).sum())

            # ---- nu stream: the nu-union row + the nu pass's vertex ---------
            if fnu is not None and "nu" in labels:
                i = labels.index("nu")
                row["nu_present"] = 1
                row["nu_chi2"] = float(chi2[i])
                row["nu_pred_pe"] = float(np.nansum(pred[i]))
                row["nu_p_nu"] = float(p_nu[i])
                row["nu_rank"] = int(rank[i])
                if tctx is not None:
                    row["nu_qfrac"] = _slice_qfrac(fnu, tctx)
                    if np.isfinite(row["nu_qfrac"]):
                        row["nu_correct"] = int(
                            row["nu_qfrac"] >= args.qfrac_cut)

            # ---- fm stream: the chi2_rank==1 row + the choosing pass -------
            if (rank == 1).any():
                j = int(np.nonzero(rank == 1)[0][0])
                is_nu = labels[j] == "nu"
                fsel = fnu if is_nu else ffm
                if fsel is not None:      # decode of the chosen slice exists
                    row["fm_present"] = 1
                    row["fm_is_nu"] = int(is_nu)
                    row["fm_chi2"] = float(chi2[j])
                    row["fm_pred_pe"] = float(np.nansum(pred[j]))
                    if tctx is not None:
                        row["fm_qfrac"] = _slice_qfrac(fsel, tctx)
                        if np.isfinite(row["fm_qfrac"]):
                            row["fm_correct"] = int(
                                row["fm_qfrac"] >= args.qfrac_cut)
        finally:
            for f in (fnu, ffm):
                if f is not None:
                    f.close()
        for k in KEYS:
            rec[k].append(row[k])
    for k in rec:
        rec[k] = np.asarray(rec[k], np.float64)
    np.savez(args.out, qfrac_cut=np.float64(args.qfrac_cut), **rec)
    print(f">>> {len(rec['obs_pe'])} events ({n_noflash} without flash table) "
          f"-> {args.out}", flush=True)


# ---------------------------------------------------------------------------
def _hist_pair(ax, a, b, bins, la, lb):
    ax.hist(a, bins=bins, histtype="stepfilled", alpha=0.45, label=la)
    ax.hist(b, bins=bins, histtype="step", lw=1.8, label=lb)


def make_plots(rec, outdir, qfrac_cut, region="all", suffix=""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(outdir, exist_ok=True)
    lx = np.linspace(0, 5, 51)                 # log10(chi2) bins
    lpe = np.linspace(0, 5, 51)                # log10(PE) bins
    rmask = {"all": np.ones(len(rec["obs_pe"]), bool),
             "intpc": rec["vtx_in_tpc"] > 0,
             "outtpc": rec["vtx_in_tpc"] == 0}[region]
    rnote = {"all": "", "intpc": ", true vtx in TPC",
             "outtpc": ", true vtx OUT of TPC"}[region]

    for s, name in (("nu", "nu stream (nu-union slice)"),
                    ("fm", "flashmatch stream (best-chi2 slice)")):
        name += rnote
        pres = rmask & (rec[f"{s}_present"] > 0)
        judged = pres & np.isfinite(rec[f"{s}_qfrac"])
        ok = judged & (rec[f"{s}_correct"] > 0)
        bad = judged & (rec[f"{s}_correct"] == 0)
        # events whose nu deposited no truth-matched charge (fully out-of-TPC)
        # cannot be judged -- the 'all' variant is the usable diagnostic for
        # the unmodelable-light population.
        chi = rec[f"{s}_chi2"]
        fin = np.isfinite(chi) & (chi > 0)

        fig, ax = plt.subplots(figsize=(6, 4.2))
        _hist_pair(ax, np.log10(chi[ok & fin]), np.log10(chi[bad & fin]), lx,
                   f"correct nu choice (N={int((ok & fin).sum())})",
                   f"incorrect (N={int((bad & fin).sum())})")
        ax.set(xlabel="log10(flash-match chi2)", ylabel="events",
               title=f"{name}\nchosen-slice chi2 (correct = slice has >="
                     f"{qfrac_cut:.0%} of nu dedup charge)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(f"{outdir}/chi2_{s}{suffix}.png", dpi=110)
        plt.close(fig)

        for m, tag in ((ok, "correct"), (bad, "incorrect"), (pres, "all")):
            pe_p = rec[f"{s}_pred_pe"][m]
            pe_o = rec["obs_pe"][m]
            f2 = (pe_p > 0) & (pe_o > 0)
            fig, ax = plt.subplots(figsize=(6, 4.2))
            _hist_pair(ax, np.log10(pe_p[f2]), np.log10(pe_o[f2]), lpe,
                       "predicted total PE", "observed in-time total PE")
            ax.set(xlabel="log10(total PE)", ylabel="events",
                   title=f"{name}, {tag} choices (N={int(f2.sum())})\n"
                         "predicted vs observed flash scale")
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)
            fig.tight_layout()
            fig.savefig(f"{outdir}/pe_{s}_{tag}{suffix}.png", dpi=110)
            plt.close(fig)

        fig, ax = plt.subplots(figsize=(6, 4.2))
        for m, lab, st in ((ok, "correct", "stepfilled"),
                           (bad, "incorrect", "step"),
                           (pres, "all choices", "step")):
            r = rec[f"{s}_pred_pe"][m] / rec["obs_pe"][m]
            r = r[np.isfinite(r) & (r > 0)]
            ax.hist(np.log10(r), bins=np.linspace(-2, 2, 61), histtype=st,
                    alpha=0.45 if st == "stepfilled" else 1.0, lw=1.8,
                    label=f"{lab} (median {np.median(r):.2f})" if len(r)
                    else lab)
        ax.axvline(0, color="k", ls=":", lw=1)
        ax.set(xlabel="log10(predicted / observed total PE)", ylabel="events",
               title=f"{name}\nPE scale calibration (0 = perfect)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(f"{outdir}/pe_ratio_{s}{suffix}.png", dpi=110)
        plt.close(fig)

    # 2D pred vs obs on correct nu-union rows (the calibration population)
    m = rmask & (rec["nu_present"] > 0) & (rec["nu_correct"] > 0)
    pe_p, pe_o = rec["nu_pred_pe"][m], rec["obs_pe"][m]
    f2 = (pe_p > 0) & (pe_o > 0)
    if not f2.any():
        return
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    hb = ax.hexbin(np.log10(pe_o[f2]), np.log10(pe_p[f2]), gridsize=45,
                   bins="log", cmap="viridis")
    ax.plot([0, 5], [0, 5], "r--", lw=1, label="pred = obs")
    ax.set(xlabel="log10(observed total PE)",
           ylabel="log10(predicted total PE)",
           title=f"correct nu-union slices: PE prediction calibration{rnote}")
    fig.colorbar(hb, ax=ax, label="events")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{outdir}/pe_pred_vs_obs{suffix}.png", dpi=110)
    plt.close(fig)

    # chi2 rank of the correct nu union: rank 1 = flash match alone finds nu
    m = (rmask & (rec["nu_present"] > 0) & (rec["nu_correct"] > 0)
         & (rec["nu_rank"] > 0))
    rk = rec["nu_rank"][m].astype(int)
    fig, ax = plt.subplots(figsize=(6, 4.2))
    ax.hist(np.clip(rk, 1, 10), bins=np.arange(0.5, 11.5), rwidth=0.85)
    fr1 = float((rk == 1).mean()) if len(rk) else 0.0
    ax.set(xlabel="chi2 rank of the (correct) nu-union slice (10 = >=10)",
           ylabel="events",
           title=f"flash-match ranking accuracy{rnote}: nu slice is rank 1 "
                 f"in {fr1:.1%} of events (N={len(rk)})")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{outdir}/nu_chi2_rank{suffix}.png", dpi=110)
    plt.close(fig)


def dwall_plots(rec, outdir):
    """Boundary effect: prediction quality vs signed vertex wall distance.
    Escaping particles deposit ionization outside the TPC, so even in-TPC
    events near a wall should show under-predicted light (ratio dropping)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    edges = np.array([-60, -30, -15, -5, 0, 5, 10, 20, 35, 55, 80, 118])
    ctr = 0.5 * (edges[:-1] + edges[1:])
    m0 = ((rec["nu_present"] > 0) & (rec["nu_correct"] > 0)
          & np.isfinite(rec["vtx_dwall"]))
    ratio = np.log10(rec["nu_pred_pe"] / rec["obs_pe"])
    med, q16, q84, n = [], [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        b = m0 & (rec["vtx_dwall"] >= lo) & (rec["vtx_dwall"] < hi) \
            & np.isfinite(ratio)
        r = ratio[b]
        n.append(len(r))
        med.append(np.median(r) if len(r) else np.nan)
        q16.append(np.percentile(r, 16) if len(r) else np.nan)
        q84.append(np.percentile(r, 84) if len(r) else np.nan)
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.fill_between(ctr, q16, q84, alpha=0.25, label="16-84%")
    ax.plot(ctr, med, "o-", ms=4, label="median")
    ax.axhline(0, color="k", ls=":", lw=1)
    ax.axvline(0, color="r", ls="--", lw=1, label="TPC wall")
    ax.set(xlabel="true vtx signed distance to nearest TPC wall [cm] "
                  "(<0 = outside)",
           ylabel="log10(predicted / observed total PE)",
           title="PE prediction vs wall distance (correct nu-union slices)\n"
                 "escaping ionization -> under-prediction near/outside walls")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{outdir}/pe_ratio_vs_dwall.png", dpi=110)
    plt.close(fig)

    # PE prediction agreement vs drift coordinate X (light model weakest
    # near the PMTs at x~0): median log10(pred/obs) for the nu-union slice,
    # all choices (works truth-free on beam data)
    mx = ((rec["nu_present"] > 0) & np.isfinite(rec["vtx_x"])
          & np.isfinite(rec["nu_pred_pe"]) & (rec["obs_pe"] > 0))
    if mx.sum() > 50:
        ratio = np.log10(np.clip(rec["nu_pred_pe"] / rec["obs_pe"],
                                 1e-3, 1e3))
        xe = np.linspace(0, 260, 14)
        xc = 0.5 * (xe[:-1] + xe[1:])
        med, q16, q84 = [], [], []
        for lo, hi in zip(xe[:-1], xe[1:]):
            b = mx & (rec["vtx_x"] >= lo) & (rec["vtx_x"] < hi)
            r = ratio[b]
            med.append(np.median(r) if len(r) > 5 else np.nan)
            q16.append(np.percentile(r, 16) if len(r) > 5 else np.nan)
            q84.append(np.percentile(r, 84) if len(r) > 5 else np.nan)
        fig, ax = plt.subplots(figsize=(6.4, 4.4))
        ax.fill_between(xc, q16, q84, alpha=0.25, label="16-84%")
        ax.plot(xc, med, "o-", ms=4, label="median")
        ax.axhline(0, color="k", ls=":", lw=1)
        ax.set(xlabel="reco nu-vertex X [cm]  (PMTs at x~0)",
               ylabel="log10(predicted / observed total PE)",
               title="flash prediction vs drift coordinate (nu-union slices)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(f"{outdir}/pe_ratio_vs_x.png", dpi=110)
        plt.close(fig)

    m1 = ((rec["nu_present"] > 0) & (rec["nu_correct"] > 0)
          & (rec["nu_rank"] > 0) & np.isfinite(rec["vtx_dwall"]))
    fr, nn = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        b = m1 & (rec["vtx_dwall"] >= lo) & (rec["vtx_dwall"] < hi)
        nn.append(int(b.sum()))
        fr.append((rec["nu_rank"][b] == 1).mean() if b.sum() else np.nan)
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.errorbar(ctr, fr, yerr=[np.sqrt(f*(1-f)/max(k,1)) if np.isfinite(f)
                               else 0 for f, k in zip(fr, nn)], marker="o",
                ms=4)
    ax.axvline(0, color="r", ls="--", lw=1, label="TPC wall")
    ax.set(xlabel="true vtx signed distance to nearest TPC wall [cm]",
           ylabel="fraction where correct nu slice is chi2 rank 1",
           ylim=(0, 1),
           title="flash-match ranking accuracy vs wall distance")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{outdir}/nu_rank1_vs_dwall.png", dpi=110)
    plt.close(fig)
    print(f">>> plots -> {outdir}")


def summary(rec, qfrac_cut):
    print(f"\n== FLASH-MATCH QUALITY (correct = chosen slice collects >= "
          f"{qfrac_cut:.0%} of the nu interaction's dedup charge) ==")
    regions = [("all", np.ones(len(rec["obs_pe"]), bool)),
               ("in-TPC", rec["vtx_in_tpc"] > 0),
               ("out-TPC", rec["vtx_in_tpc"] == 0)]
    for rname, rmask in regions:
        for s in ("nu", "fm"):
            pres = rmask & (rec[f"{s}_present"] > 0)
            judged = pres & np.isfinite(rec[f"{s}_qfrac"])
            ok = judged & (rec[f"{s}_correct"] > 0)
            chi = rec[f"{s}_chi2"]
            cok = chi[ok & np.isfinite(chi)]
            cbad = chi[judged & (rec[f"{s}_correct"] == 0) & np.isfinite(chi)]
            ratio = rec[f"{s}_pred_pe"][ok] / rec["obs_pe"][ok]
            ratio = ratio[np.isfinite(ratio) & (ratio > 0)]
            print(f"  [{rname:7s}] {s}: present {int(pres.sum()):6d} | "
                  f"judged {int(judged.sum()):6d} | correct "
                  f"{ok.sum()/max(judged.sum(),1):.3f} | median chi2 "
                  f"{np.median(cok) if len(cok) else np.nan:8.1f} (corr) vs "
                  f"{np.median(cbad) if len(cbad) else np.nan:8.1f} (inc) | "
                  f"med pred/obs {np.median(ratio) if len(ratio) else np.nan:.2f}")
        m = (rmask & (rec["nu_present"] > 0) & (rec["nu_correct"] > 0)
             & (rec["nu_rank"] > 0))
        rk = rec["nu_rank"][m]
        if len(rk):
            print(f"  [{rname:7s}] correct nu slice wins chi2 ranking: "
                  f"{(rk == 1).mean():.3f} (N={len(rk)})")
    fm = rec["fm_present"] > 0
    print(f"  fm choice == nu union: {rec['fm_is_nu'][fm].mean():.3f} "
          f"of {int(fm.sum())} events")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--kp2-nu-list")
    ap.add_argument("--kp2-fm-list")
    ap.add_argument("--merged-sp-list",
                    help="merged_sp list for the GENIE vertex -> in-TPC flag "
                         "+ wall distance (matched by kp2 src_file attr); "
                         "omit to skip the TPC split")
    ap.add_argument("--out", default="fmq_records.npz")
    ap.add_argument("--plots", default=None)
    ap.add_argument("--qfrac-cut", type=float, default=0.50,
                    help="correct choice = chosen slice collects at least this "
                         "fraction of the nu interaction's de-double-counted "
                         "charge (eval SLICE_COVERAGE convention)")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--merge", metavar="GLOB",
                    help="merge shard npz files -> --out (+summary/plots)")
    args = ap.parse_args()

    if args.merge:
        paths = sorted(glob.glob(args.merge))
        if not paths:
            raise SystemExit(f"no shard npz matched {args.merge!r}")
        rec = {k: [] for k in KEYS}
        qfrac_cut = args.qfrac_cut
        for p in paths:
            with np.load(p) as z:
                qfrac_cut = float(z["qfrac_cut"])
                for k in KEYS:
                    rec[k].append(z[k])
        rec = {k: np.concatenate(v) for k, v in rec.items()}
        np.savez(args.out, qfrac_cut=np.float64(qfrac_cut), **rec)
        print(f">>> merged {len(paths)} shards -> {len(rec['obs_pe'])} "
              f"events -> {args.out}")
        summary(rec, qfrac_cut)
        if args.plots:
            for region, sfx in (("all", ""), ("intpc", "_intpc"),
                                ("outtpc", "_outtpc")):
                make_plots(rec, args.plots, qfrac_cut, region=region,
                           suffix=sfx)
            dwall_plots(rec, args.plots)
        return

    for req in ("kp2_nu_list", "kp2_fm_list"):
        if getattr(args, req) is None:
            raise SystemExit(f"--{req.replace('_','-')} required "
                             "(unless --merge)")
    process(args)


if __name__ == "__main__":
    main()
