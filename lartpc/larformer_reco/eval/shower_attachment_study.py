"""Shower-attachment tuning study: variable distributions for correct vs
incorrect (shower, vertex) combinations, on the nu-stream reco output.

Motivation: the conversion-distance eval showed photon efficiency is flat out
to ~55 cm then cliffs, with the loss dominated by seg!att -- the segmenter
makes the instance but reco_showers() does not attach it. The production
attachment (`trajfit.shower_connect.connects` on the vertex-biased trunk) cuts
hard at impact<=10 cm, cosine>=0.9, gap<=60 cm -- and the 60 cm gap bound sits
exactly at the observed cliff. This study collects, for EVERY segmented shower
instance x EVERY reco vertex candidate in the event, the current decision
variables plus proposed ones, truth-labels each pair, and reports which cut
kills recoverable (correct) pairs -- the input for likelihood- or
size/distance-dependent attachment tuning.

Per pair record:
  current trunk variables (vertex-biased trunk): gap, impact, along, cosine,
      trunk quality (PCA elongation), trunk length;
  proposed full-shower variables: pca_impact (impact parameter of the vertex
      to the whole-cluster 1st-PCA line), pca_cosine, pca_quality;
  shower descriptors: n_pts, pred class (e/gamma), spread (rms about PCA axis);
  vertex descriptors: vertex score, |V - gt nu vertex|;
  truth: is_nu (majority trackid is nu-origin), true pdg, conversion distance
      (min |startpt-originpt| over the matched trackid's shower fragments),
      CORRECT = is_nu AND |V - gt nu vertex| < --vtx-cut (3 cm);
  current decision: cur_pass (production knobs) + which cut failed.

Shard + merge like the other eval tools:
    PYTHONPATH=./ python3 .../shower_attachment_study.py \
        --kp2-list ... --nu-reco-dir ... --merged-sp-list ... \
        --out shard.npz [--start I --n N]
    ... --merge 'dir/att_shard*.npz' --out merged.npz --plots plots/dir
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
from lartpc.larformer_reco.trajfit.shower_trunk import (  # noqa: E402
    trunk_vertex_biased, trunk_pca, _pca)
from lartpc.larformer_reco.trajfit.shower_connect import (  # noqa: E402
    connection_geometry, connects)
from lartpc.larformer_reco.export.export_gen2ntuple import (  # noqa: E402
    build_reco_map)

# production knobs (run_nu_reco.py / reco_showers defaults)
CUR = dict(d_impact=10.0, cos_min=0.9, d_gap=60.0, gap_touch=3.0)
SHOWER_CLASSES = (0, 1)                    # LArFormer class ids: e, gamma

KEYS = ["ev", "inst", "n_pts", "pred_cls", "spread",
        "gap", "impact", "along", "cosine", "trunk_q", "trunk_len",
        "pca_impact", "pca_cosine", "pca_q",
        "v_score", "v_dist_gt", "is_nu", "true_pdg", "conv_dist",
        "correct", "cur_pass", "fail_gap", "fail_impact", "fail_cos"]


def _index_by_event(paths):
    import re
    out = {}
    for p in paths:
        m = re.search(r"event(\d+)(_fm)?_0\.h5$", os.path.basename(p))
        if m:
            out[int(m.group(1))] = p
    return out


def _msp_truth(msp_path):
    """Truth context: pos->trackid rows, nu-origin tid set, pdg-by-tid,
    conversion distance by tid (min |startpt - originpt| over fragments)."""
    with h5py.File(msp_path, "r") as f:
        e = f["entry_0"]
        mt = e["mc_particle_tree"]
        tid = np.asarray(mt["trackid"][()], np.int64)
        org = np.asarray(mt["origin"][()], np.int64)
        pid = np.asarray(mt["pid"][()], np.int64)
        td = e["triplet_data"]
        pos = td["pos"][()].astype(np.float32)
        ttid = np.asarray(td["trackid"][()], np.int64)
        ctx = {"row_tid": ttid,
               "row_by_pos": {pos[i].tobytes(): i for i in range(len(pos))},
               "nu_tids": set(tid[org == 1].tolist()),
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
    msp_by_base = {os.path.basename(p): p
                   for p in read_list(args.merged_sp_list)}
    reco = build_reco_map(args.nu_reco_dir)
    # gidx == line index in the kp2 list (the chain's linkage convention)
    kp_list = read_list(args.kp2_list)
    events = sorted(kp_by_ev)
    lo = args.start
    hi = len(events) if args.n is None else min(len(events), lo + args.n)
    events = events[lo:hi]
    gidx_by_path = {p: i for i, p in enumerate(kp_list)}
    print(f">>> {len(kp_by_ev)} kp2, {len(reco)} reco events; shard "
          f"[{lo}:{hi}] ({len(events)})", flush=True)

    rec = {k: [] for k in KEYS}
    n_no_reco = n_pairs = 0
    for ev in events:
        kp_path = kp_by_ev[ev]
        rr = reco.get(gidx_by_path.get(kp_path, -1))
        if rr is None:
            n_no_reco += 1
            continue
        with h5py.File(kp_path, "r") as fk, h5py.File(rr[0], "r") as fr:
            gr = fr[rr[1]]
            vcm = np.atleast_2d(gr["vertices_cm"][()]).astype(np.float64)
            vsc = np.atleast_1d(gr["vertices_score"][()]).astype(np.float64)
            if len(vcm) == 0 or "particle" not in fk or "slice" not in fk:
                continue
            coords = fk["slice/coord_cm"][()].astype(np.float32)
            gt_v = None
            if bool(fk.attrs.get("has_gt", False)):
                g = np.asarray(fk["gt_nu_vertex_cm"][()],
                               np.float64).reshape(-1)[:3]
                if np.isfinite(g).all():
                    gt_v = g
            tctx = None
            msp = msp_by_base.get(str(fk.attrs.get("src_file", "")))
            if msp:
                try:
                    tctx = _msp_truth(msp)
                except Exception:
                    tctx = None

            for inst in fk["particle"]:
                gi = fk[f"particle/{inst}"]
                if "class_scores" not in gi or "point_idx" not in gi:
                    continue
                cls = int(np.argmax(gi["class_scores"][()][:6]))
                if cls not in SHOWER_CLASSES:
                    continue
                pidx = gi["point_idx"][()]
                pts = coords[pidx].astype(np.float64)
                if len(pts) < 10:
                    continue
                # truth match: majority trackid of exact-position rows
                is_nu, tpdg, cdist = 0, 0, np.nan
                if tctx is not None:
                    rows = [tctx["row_by_pos"].get(coords[i].tobytes(), -1)
                            for i in pidx]
                    rows = np.asarray([r for r in rows if r >= 0], np.int64)
                    if len(rows):
                        tids, cnt = np.unique(tctx["row_tid"][rows],
                                              return_counts=True)
                        maj = int(tids[cnt.argmax()])
                        is_nu = int(maj in tctx["nu_tids"])
                        tpdg = tctx["pdg_by_tid"].get(maj, 0)
                        cdist = tctx["conv"].get(maj, np.nan)
                # full-shower PCA (vertex-independent): line through centroid
                c, evals, evecs = _pca(pts)
                e1 = evecs[:, 0]
                pca_q = float(evals[0] / (evals.sum() + 1e-12))
                spread = float(np.sqrt(max(evals[1:].sum(), 0.0)))

                for vi in range(len(vcm)):
                    V = vcm[vi]
                    tk = trunk_vertex_biased(pts, V)
                    ok, g = connects(tk.start, tk.direction, V, **CUR)
                    # full-PCA variables: orient axis away from V
                    d1 = e1 if (c - V) @ e1 >= 0 else -e1
                    rv = V - c
                    a1 = float(rv @ d1)
                    pca_imp = float(np.linalg.norm(rv - a1 * d1))
                    v2c = c - V
                    nn = np.linalg.norm(v2c)
                    pca_cos = float((d1 @ v2c) / nn) if nn > 1e-9 else 1.0
                    vdg = (float(np.linalg.norm(V - gt_v))
                           if gt_v is not None else np.nan)
                    correct = int(bool(is_nu) and np.isfinite(vdg)
                                  and vdg < args.vtx_cut)
                    row = dict(
                        ev=ev, inst=int(inst), n_pts=len(pts), pred_cls=cls,
                        spread=spread,
                        gap=g["gap"], impact=g["impact"], along=g["along"],
                        cosine=g["cosine"], trunk_q=tk.quality,
                        trunk_len=tk.length_cm,
                        pca_impact=pca_imp, pca_cosine=pca_cos, pca_q=pca_q,
                        v_score=float(vsc[vi]), v_dist_gt=vdg,
                        is_nu=is_nu, true_pdg=tpdg, conv_dist=cdist,
                        correct=correct, cur_pass=int(ok),
                        fail_gap=int(g["gap"] > CUR["d_gap"]),
                        fail_impact=int(g["impact"] > CUR["d_impact"]),
                        fail_cos=int(g["cosine"] < CUR["cos_min"]
                                     and g["gap"] > CUR["gap_touch"]))
                    for k in KEYS:
                        rec[k].append(row[k])
                    n_pairs += 1
    for k in rec:
        rec[k] = np.asarray(rec[k], np.float64)
    np.savez(args.out, vtx_cut=np.float64(args.vtx_cut), **rec)
    print(f">>> {n_pairs} (shower,vertex) pairs from {len(events)} events "
          f"({n_no_reco} without reco) -> {args.out}", flush=True)


# ---------------------------------------------------------------------------
def summary(rec):
    ok = rec["correct"] > 0
    bad = rec["correct"] == 0
    print(f"\n== SHOWER ATTACHMENT STUDY: {len(ok)} pairs, "
          f"{int(ok.sum())} correct / {int(bad.sum())} incorrect ==")
    cp = rec["cur_pass"] > 0
    print(f"  current cuts: attach {cp[ok].mean():.3f} of correct pairs | "
          f"{cp[bad].mean():.3f} of incorrect (false-attach rate)")
    lost = ok & ~cp
    if lost.any():
        print(f"  correct pairs LOST to current cuts: {int(lost.sum())} "
              f"({lost.mean():.1%} of all pairs); failing: "
              f"gap>60 {rec['fail_gap'][lost].mean():.2f}, "
              f"impact>10 {rec['fail_impact'][lost].mean():.2f}, "
              f"cos<0.9 {rec['fail_cos'][lost].mean():.2f} "
              f"(fractions of lost, cuts can overlap)")
    m = ok & np.isfinite(rec["conv_dist"])
    far = m & (rec["conv_dist"] > 55)
    if far.any():
        print(f"  far-converting (>55 cm) correct pairs: {int(far.sum())}, "
              f"current cuts keep {cp[far].mean():.3f}")


def make_plots(rec, outdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(outdir, exist_ok=True)
    ok = rec["correct"] > 0
    bad = rec["correct"] == 0
    VARS = [("gap", np.linspace(0, 150, 61), "gap |trunk start - V| [cm]",
             CUR["d_gap"]),
            ("impact", np.linspace(0, 30, 61), "trunk impact param [cm]",
             CUR["d_impact"]),
            ("cosine", np.linspace(-1, 1, 61), "trunk back-point cosine",
             CUR["cos_min"]),
            ("along", np.linspace(-100, 50, 61), "signed along-axis dist [cm]",
             None),
            ("pca_impact", np.linspace(0, 30, 61),
             "full-shower 1st-PCA impact param [cm]", None),
            ("pca_cosine", np.linspace(-1, 1, 61),
             "full-shower PCA cosine", None),
            ("trunk_q", np.linspace(0.3, 1, 36), "trunk PCA elongation", None),
            ("pca_q", np.linspace(0.3, 1, 36), "full-PCA elongation", None)]
    for name, bins, xlabel, cut in VARS:
        v = rec[name]
        fig, ax = plt.subplots(figsize=(6, 4.2))
        ax.hist(v[ok], bins=bins, histtype="stepfilled", alpha=0.45,
                density=True, label=f"correct (N={int(ok.sum())})")
        ax.hist(v[bad], bins=bins, histtype="step", lw=1.8, density=True,
                label=f"incorrect (N={int(bad.sum())})")
        if cut is not None:
            ax.axvline(cut, color="r", ls="--", lw=1, label="current cut")
        ax.set(xlabel=xlabel, ylabel="density",
               title=f"shower-vertex pairs: {name}")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(f"{outdir}/att_{name}.png", dpi=110)
        plt.close(fig)

    # the money plot: correct-pair GAP distribution vs the 60 cm bound,
    # split by conversion distance
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    m = ok & np.isfinite(rec["conv_dist"])
    for lo, hi, lab in ((0, 20, "conv<20cm"), (20, 55, "conv 20-55cm"),
                        (55, 1e9, "conv>55cm")):
        b = m & (rec["conv_dist"] >= lo) & (rec["conv_dist"] < hi)
        ax.hist(np.clip(rec["gap"][b], 0, 149), bins=np.linspace(0, 150, 51),
                histtype="step", lw=1.8, label=f"{lab} (N={int(b.sum())})")
    ax.axvline(CUR["d_gap"], color="r", ls="--", lw=1.2,
               label="current d_gap=60")
    ax.set(xlabel="gap |trunk start - vertex| [cm]", ylabel="correct pairs",
           title="correct attachments: gap vs the hard 60 cm bound")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{outdir}/att_gap_by_convdist.png", dpi=110)
    plt.close(fig)

    # size dependence of direction quality (tuning axis 2): cosine of correct
    # pairs profiled vs n_pts
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    edges = np.array([10, 20, 40, 80, 160, 320, 640, 1280, 5000])
    ctr = np.sqrt(edges[:-1] * edges[1:])
    for name, lab in (("cosine", "trunk cosine"),
                      ("pca_cosine", "full-PCA cosine")):
        med, q16 = [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            b = ok & (rec["n_pts"] >= lo) & (rec["n_pts"] < hi)
            med.append(np.median(rec[name][b]) if b.sum() else np.nan)
            q16.append(np.percentile(rec[name][b], 16) if b.sum() else np.nan)
        ax.plot(ctr, med, "o-", ms=4, label=f"{lab} median")
        ax.plot(ctr, q16, ":", label=f"{lab} 16%")
    ax.axhline(CUR["cos_min"], color="r", ls="--", lw=1, label="cos_min=0.9")
    ax.set(xscale="log", xlabel="shower n_points", ylabel="back-point cosine",
           ylim=(-0.2, 1.05),
           title="direction quality vs shower size (correct pairs)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{outdir}/att_cosine_vs_size.png", dpi=110)
    plt.close(fig)
    print(f">>> plots -> {outdir}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--kp2-list")
    ap.add_argument("--nu-reco-dir")
    ap.add_argument("--merged-sp-list")
    ap.add_argument("--out", default="att_records.npz")
    ap.add_argument("--plots", default=None)
    ap.add_argument("--vtx-cut", type=float, default=3.0,
                    help="correct vertex = within this of the GT nu vertex")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--merge", metavar="GLOB")
    args = ap.parse_args()

    if args.merge:
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
        print(f">>> merged {len(paths)} shards -> {len(rec['gap'])} pairs "
              f"-> {args.out}")
        summary(rec)
        if args.plots:
            make_plots(rec, args.plots)
        return
    for req in ("kp2_list", "nu_reco_dir", "merged_sp_list"):
        if getattr(args, req) is None:
            raise SystemExit(f"--{req.replace('_', '-')} required")
    process(args)


if __name__ == "__main__":
    main()
