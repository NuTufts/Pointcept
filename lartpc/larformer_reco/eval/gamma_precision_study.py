"""Gamma PRECISION study: what are the predicted-and-attached photons, really?

Complement to the attachment-recall work (LLR union rule pushed gamma
attachment 0.705 -> 0.794): for every ATTACHED shower with predicted class
gamma (electrons recorded too), truth-classify it and look for RECO-ONLY
quality variables that reject the false positives without touching the true
ones -- the precision/recall trade-off is then an analyzer choice on the
persisted variables.

Truth categories (majority trackid over exact-position-matched spacepoints):
  0 true-nu-gamma   majority tid is a nu-origin photon (pdg 22)
  1 mis-ID nu       majority tid is a nu-origin NON-photon (e, mu-delta, ...)
  2 cosmic          majority tid is cosmic-origin (any pdg)
  3 ghost/junk      no valid majority tid, or match purity < 0.3

Per-shower reco variables recorded (all analyzer-available):
  gamma_score/e_score (kp2 class_scores), att_score (LLR), att_confident,
  n_pts, charge, energy, interaction idx; truth: category, match purity,
  matched pdg/origin, conversion distance.

Shard + merge like the other eval tools.
"""
import argparse
import glob
import os
import sys

import numpy as np
import h5py

sys.path.insert(0, os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")))
from lartpc.larformer_reco.utils import read_list  # noqa: E402
from lartpc.larformer_reco.export.export_gen2ntuple import (  # noqa: E402
    build_reco_map)

KEYS = ["ev", "inst", "pred_cls", "n_pts", "charge", "energy",
        "gamma_score", "e_score", "att_score", "att_confident",
        "interaction", "purity", "match_pdg", "match_origin", "conv_dist",
        "category"]
GAMMA_CLS, E_CLS = 1, 0


def _index_by_event(paths):
    import re
    out = {}
    for p in paths:
        m = re.search(r"event(\d+)(_fm)?_0\.h5$", os.path.basename(p))
        if m:
            out[int(m.group(1))] = p
    return out


def _msp_truth(msp_path):
    with h5py.File(msp_path, "r") as f:
        e = f["entry_0"]
        mt = e["mc_particle_tree"]
        tid = np.asarray(mt["trackid"][()], np.int64)
        org = np.asarray(mt["origin"][()], np.int64)
        pid = np.asarray(mt["pid"][()], np.int64)
        td = e["triplet_data"]
        pos = td["pos"][()].astype(np.float32)
        ctx = {"row_tid": np.asarray(td["trackid"][()], np.int64),
               "row_by_pos": {pos[i].tobytes(): i for i in range(len(pos))},
               "org_by_tid": {int(t): int(o) for t, o in zip(tid, org)},
               "pdg_by_tid": {int(t): int(p) for t, p in zip(tid, pid)},
               "conv": {}}
        sf = e["shower_fragments"]
        ftid = np.atleast_1d(sf["trackid"][()])
        if ftid.size:
            so = np.atleast_2d(sf["startpt"][()]).astype(np.float64)
            oo = np.atleast_2d(sf["originpt"][()]).astype(np.float64)
            dd = np.linalg.norm(so - oo, axis=1)
            for i in range(ftid.size):
                t = int(ftid[i])
                if t not in ctx["conv"] or dd[i] < ctx["conv"][t]:
                    ctx["conv"][t] = float(dd[i])
    return ctx


def process(args):
    kp_by_ev = _index_by_event(read_list(args.kp2_list))
    kp_list = read_list(args.kp2_list)
    gidx_by_path = {p: i for i, p in enumerate(kp_list)}
    msp_by_base = {os.path.basename(p): p
                   for p in read_list(args.merged_sp_list)}
    reco = build_reco_map(args.nu_reco_dir)
    events = sorted(kp_by_ev)
    lo = args.start
    hi = len(events) if args.n is None else min(len(events), lo + args.n)
    events = events[lo:hi]
    print(f">>> {len(kp_by_ev)} kp2, {len(reco)} reco; shard [{lo}:{hi}]",
          flush=True)

    rec = {k: [] for k in KEYS}
    for ev in events:
        kp_path = kp_by_ev[ev]
        rr = reco.get(gidx_by_path.get(kp_path, -1))
        if rr is None:
            continue
        with h5py.File(kp_path, "r") as fk, h5py.File(rr[0], "r") as fr:
            gr = fr[rr[1]]
            kind = gr["part_kind"][()]
            sh = np.nonzero(kind == 1)[0]
            if not len(sh) or "slice" not in fk:
                continue
            coords = fk["slice/coord_cm"][()].astype(np.float32)
            pcls = gr["part_pred_class"][()]
            pinst = gr["part_inst_idx"][()]
            pchg = gr["part_charge"][()]
            pen = gr["part_energy"][()]
            pint = gr["part_interaction"][()]
            ats = (gr["part_att_score"][()] if "part_att_score" in gr
                   else np.full(len(kind), np.nan))
            atc = (gr["part_att_confident"][()]
                   if "part_att_confident" in gr
                   else np.ones(len(kind), np.int64))
            tctx = None
            msp = msp_by_base.get(str(fk.attrs.get("src_file", "")))
            if msp:
                try:
                    tctx = _msp_truth(msp)
                except Exception:
                    tctx = None

            for i in sh:
                if int(pcls[i]) not in (GAMMA_CLS, E_CLS):
                    continue
                inst = int(pinst[i])
                gi = (fk.get(f"particle/{inst}")
                      if inst >= 0 else None)
                if gi is None or "point_idx" not in gi:
                    continue
                pidx = gi["point_idx"][()]
                cs = (gi["class_scores"][()]
                      if "class_scores" in gi else np.full(8, np.nan))
                purity, mpdg, morg, cdist, cat = np.nan, 0, -1, np.nan, 3
                if tctx is not None and len(pidx):
                    rows = np.asarray(
                        [tctx["row_by_pos"].get(coords[j].tobytes(), -1)
                         for j in pidx], np.int64)
                    rows = rows[rows >= 0]
                    if len(rows):
                        tids, cnt = np.unique(tctx["row_tid"][rows],
                                              return_counts=True)
                        k = int(cnt.argmax())
                        maj = int(tids[k])
                        purity = float(cnt[k] / len(pidx))
                        mpdg = tctx["pdg_by_tid"].get(maj, 0)
                        morg = tctx["org_by_tid"].get(maj, -1)
                        cdist = tctx["conv"].get(maj, np.nan)
                        if maj <= 0 or purity < 0.3:
                            cat = 3                      # ghost/junk
                        elif morg != 1:
                            cat = 2                      # cosmic
                        elif abs(mpdg) == 22:
                            cat = 0                      # true nu gamma
                        else:
                            cat = 1                      # mis-ID nu particle
                row = dict(ev=ev, inst=inst, pred_cls=int(pcls[i]),
                           n_pts=len(pidx), charge=float(pchg[i]),
                           energy=float(pen[i]),
                           gamma_score=float(cs[GAMMA_CLS]),
                           e_score=float(cs[E_CLS]),
                           att_score=float(ats[i]),
                           att_confident=int(atc[i]),
                           interaction=int(pint[i]), purity=purity,
                           match_pdg=mpdg, match_origin=morg,
                           conv_dist=cdist, category=cat)
                for k2 in KEYS:
                    rec[k2].append(row[k2])
    for k in rec:
        rec[k] = np.asarray(rec[k], np.float64)
    np.savez(args.out, **rec)
    print(f">>> {len(rec['ev'])} attached shower rows -> {args.out}",
          flush=True)


CAT_NAMES = ["true-nu-gamma", "mis-ID nu", "cosmic", "ghost/junk"]


def summary(rec, sel, tag):
    n = sel.sum()
    if not n:
        return
    cat = rec["category"][sel]
    fr = [(cat == c).mean() for c in range(4)]
    print(f"  {tag:28s} N={int(n):6d} | precision(true-nu-gamma) {fr[0]:.3f}"
          f" | mis-ID {fr[1]:.3f} | cosmic {fr[2]:.3f} | junk {fr[3]:.3f}")


def cut_scan(rec, sel, var, cuts, direction=">="):
    """TP retention vs FP rejection for a reco-only cut."""
    tp = sel & (rec["category"] == 0)
    fp = sel & (rec["category"] != 0)
    print(f"    cut on {var} {direction}:  (TP kept | FP kept | precision)")
    x = rec[var]
    for c in cuts:
        keep = (x >= c) if direction == ">=" else (x <= c)
        ktp, kfp = keep[tp].mean(), keep[fp].mean()
        n_k = (sel & keep).sum()
        prec = (rec["category"][sel & keep] == 0).mean() if n_k else np.nan
        print(f"      {c:7.2f}: {ktp:.3f} | {kfp:.3f} | {prec:.3f}")


def merge_main(args):
    paths = sorted(glob.glob(args.merge))
    if not paths:
        raise SystemExit(f"no shard npz matched {args.merge!r}")
    rec = {k: [] for k in KEYS}
    for p in paths:
        with np.load(p) as z:
            for k in KEYS:
                rec[k].append(z[k])
    rec = {k: np.concatenate(v) for k, v in rec.items()}
    np.savez(args.out, **rec)
    g = rec["pred_cls"] == GAMMA_CLS
    print(f">>> merged {len(paths)} shards -> {len(g)} attached showers "
          f"({int(g.sum())} predicted gamma) -> {args.out}")
    print("\n== GAMMA PRECISION (attached, predicted gamma) ==")
    summary(rec, g, "all attached")
    summary(rec, g & (rec["att_confident"] > 0), "confident only")
    summary(rec, g & (rec["att_confident"] == 0), "forced (no-shower-left-b.)")
    print("\n== reco-only quality-cut scans (predicted gamma, attached) ==")
    cut_scan(rec, g, "gamma_score", [0.2, 0.4, 0.6, 0.8, 0.9])
    cut_scan(rec, g, "att_score", [-5, 0, 3, 5, 8])
    cut_scan(rec, g, "n_pts", [20, 40, 80, 150])
    if args.plots:
        make_plots(rec, args.plots)


def make_plots(rec, outdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(outdir, exist_ok=True)
    g = rec["pred_cls"] == GAMMA_CLS
    VARS = [("gamma_score", np.linspace(0, 1, 41), "kp2 gamma class score"),
            ("att_score", np.linspace(-15, 25, 41), "attachment LLR score"),
            ("n_pts", np.geomspace(10, 5000, 41), "shower n_points"),
            ("purity", np.linspace(0, 1, 41), "truth-match purity"),
            ("energy", np.linspace(0, 800, 41), "reco energy [MeV]")]
    for name, bins, xl in VARS:
        fig, ax = plt.subplots(figsize=(6.2, 4.2))
        for c, lab in enumerate(CAT_NAMES):
            m = g & (rec["category"] == c)
            if m.sum() < 5:
                continue
            ax.hist(np.clip(rec[name][m], bins[0], bins[-1] - 1e-9),
                    bins=bins, histtype="step", lw=1.8, density=True,
                    label=f"{lab} (N={int(m.sum())})")
        if name == "n_pts":
            ax.set_xscale("log")
        ax.set(xlabel=xl, ylabel="density",
               title=f"attached predicted-gamma: {name} by truth category")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(f"{outdir}/prec_{name}.png", dpi=110)
        plt.close(fig)

    # precision vs TP-retention trade-off curves
    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    tp = g & (rec["category"] == 0)
    for var, sty in (("gamma_score", "o-"), ("att_score", "s--"),
                     ("n_pts", "^:")):
        x = rec[var]
        ths = np.nanpercentile(x[g], np.linspace(0, 95, 40))
        ret, prc = [], []
        for t in ths:
            keep = g & (x >= t)
            if keep.sum() < 20:
                continue
            ret.append(keep[tp].mean() if tp.sum() else np.nan)
            prc.append((rec["category"][keep] == 0).mean())
        ax.plot(ret, prc, sty, ms=3.5, label=f"cut on {var}")
    base = (rec["category"][g] == 0).mean()
    ax.plot([1], [base], "r*", ms=14, label=f"no cut ({base:.2f})")
    ax.set(xlabel="true-nu-gamma retention", ylabel="precision",
           title="gamma precision vs retention (reco-only cuts)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{outdir}/prec_vs_retention.png", dpi=110)
    plt.close(fig)
    print(f">>> plots -> {outdir}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--kp2-list")
    ap.add_argument("--nu-reco-dir")
    ap.add_argument("--merged-sp-list")
    ap.add_argument("--out", default="gamma_precision.npz")
    ap.add_argument("--plots", default=None)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--merge", metavar="GLOB")
    args = ap.parse_args()
    if args.merge:
        merge_main(args)
        return
    for req in ("kp2_list", "nu_reco_dir", "merged_sp_list"):
        if getattr(args, req) is None:
            raise SystemExit(f"--{req.replace('_', '-')} required")
    process(args)


if __name__ == "__main__":
    main()
