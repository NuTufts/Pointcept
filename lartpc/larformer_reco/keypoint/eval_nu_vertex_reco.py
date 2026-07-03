"""Evaluate reconstructed nu-vertex keypoints against truth on real inference output.

This is the §7 "GT comparison on sim" evaluation (``../keypoint_reco_spec.md``),
specialised to the **nu vertex** (keypoint type 0). It consumes the cascade
inference H5s written by ``run_larformer_keypoint2_cascade_inference.py
--save-score-maps`` (the ``keypoint2_out/`` files in ``reco_dev_data/``), runs the
greedy peel-and-fit reco (``KeypointRecoTorch``) on the ``nu_vertex`` score field,
and answers three questions about *what nu-vertex score threshold to trust*:

  1. **ROC: efficiency & purity vs the max nu-vertex score in a slice.**
     Per slice we keep the highest-score reco nu candidate. Sweeping a score
     threshold ``t``:
         selected(t) = slices whose max nu candidate score >= t
         correct(t)  = selected slices whose kept vertex is within --match-dist
                       cm of the true nu vertex
         efficiency(t) = correct(t) / (# slices with a true nu vertex)
         purity(t)     = correct(t) / selected(t)
     Output: efficiency & purity vs threshold, and the purity-vs-efficiency ROC.

  2. **Distance(true, reco) distribution.** One vertex per slice — the max-score
     reco nu candidate — histogrammed against the true nu vertex.

  3. **Nu vertices per slice vs threshold.** Distribution of the number of reco nu
     candidates per slice with score >= t, for t in {0.1, 0.2, 0.5, 0.7, 0.9}.

The true nu vertex is taken from ``gt_keypoints`` (type == 0), falling back to
``gt_nu_vertex_cm``. The reco is run down to a low score floor (--min-score) with
NMS via the reco's Gaussian subtraction, so all candidates needed for (1)-(3)
above the floor are captured in one pass.

Run inside the pointcept container (numpy/torch/scipy/h5py/matplotlib):

    python -m lartpc.larformer_reco.keypoint.eval_nu_vertex_reco \
        lartpc/larformer_reco/reco_dev_data/keypoint2_out \
        --output-dir nu_vertex_eval_out

(or invoke the file directly:
 ``python lartpc/larformer_reco/keypoint/eval_nu_vertex_reco.py ...``)
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np

# allow both ``python -m ...reco.eval_nu_vertex_reco`` and direct-path invocation
if __package__ in (None, ""):                        # pragma: no cover
    # add the parent of the ``reco`` package dir so ``import reco`` resolves
    sys.path.insert(0, os.path.abspath(os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")))
    from lartpc.larformer_reco.keypoint import KeypointRecoTorch, KeypointRecoParams, io
else:
    from . import io
    from .keypoint_reco import KeypointRecoTorch, KeypointRecoParams

NU_TYPE = 0
# thresholds for the per-slice nu-vertex-count distribution (user spec)
COUNT_THRESHOLDS = (0.1, 0.2, 0.5, 0.7, 0.9)


# --------------------------------------------------------------------------- #
#  per-slice reconstruction + truth extraction                                #
# --------------------------------------------------------------------------- #
def _nu_head(score_maps):
    """Return the (name, head-dict) of the nu-vertex score map, or None.

    The nu head is named ``nu_vertex`` and/or targets keypoint type 0.
    """
    for name, sm in score_maps.items():
        if name == "nu_vertex" or NU_TYPE in (sm.get("kp_types") or []):
            return name, sm
    return None, None


def _true_nu_vertex(loaded):
    """The true nu vertex (3,) cm, or None. Prefer gt_keypoints type==0."""
    gt = loaded.get("gt_keypoints")
    if gt is not None and gt["pos_cm"].size:
        m = gt["type"] == NU_TYPE
        if np.any(m):
            return np.asarray(gt["pos_cm"][m][0], np.float64)
    g0 = loaded.get("gt_nu_vertex_cm")
    if g0 is not None and np.all(np.isfinite(g0)):
        return np.asarray(g0, np.float64)
    return None


def reco_slice(reco, loaded, max_candidates):
    """Reconstruct nu-vertex candidates for one slice (sorted high->low score).

    Returns dict: cand_pos (K,3), cand_score (K,), true_nu (3,)|None,
    plus the existing single-centroid decode for cross-check.
    """
    name, nu = _nu_head(loaded["score_maps"])
    if nu is None:
        cand_pos = np.zeros((0, 3), np.float64)
        cand_score = np.zeros(0, np.float64)
    else:
        kps = reco.reconstruct(nu["coords_cm"], nu["score"],
                               max_keypoints=max_candidates)
        order = np.argsort([-k.peak_score for k in kps])
        cand_pos = (np.stack([kps[i].pos_cm for i in order]).astype(np.float64)
                    if len(kps) else np.zeros((0, 3), np.float64))
        cand_score = np.asarray([kps[i].peak_score for i in order], np.float64)
    return dict(cand_pos=cand_pos, cand_score=cand_score,
                true_nu=_true_nu_vertex(loaded),
                decode_nu=loaded.get("nu_vertex_cm"))


# --------------------------------------------------------------------------- #
#  metrics                                                                     #
# --------------------------------------------------------------------------- #
def build_roc(slices, match_dist, thresholds):
    """Efficiency & purity vs max-nu-score threshold.

    ``slices`` = ALL per-slice dicts from ``reco_slice``. Each slice is either
    *signal* (has a true nu vertex) or *background* (none — e.g. a cosmic slice).
    Per threshold ``t``:
        selected  = any slice whose max candidate score >= t  (signal OR bkg)
        correct   = a SIGNAL slice that is selected AND whose max-score candidate
                    is within match_dist cm of the true nu vertex
        efficiency = correct / (# signal slices)
        purity     = correct / selected   (background selections are impurities,
                     as are signal selections whose vertex is too far)
    ``best_dist`` is finite only for signal slices (NaN for background).
    """
    n = len(slices)
    is_sig = np.asarray([s["true_nu"] is not None for s in slices], bool)
    n_sig = int(is_sig.sum())
    max_score = np.zeros(n)
    best_dist = np.full(n, np.nan)
    for i, s in enumerate(slices):
        if s["cand_score"].size:
            max_score[i] = s["cand_score"][0]            # sorted high->low
            if s["true_nu"] is not None:
                best_dist[i] = np.linalg.norm(s["cand_pos"][0] - s["true_nu"])

    close = np.isfinite(best_dist) & (best_dist <= match_dist)
    eff = np.zeros(len(thresholds))
    pur = np.full(len(thresholds), np.nan)
    n_sel = np.zeros(len(thresholds), np.int64)
    n_cor = np.zeros(len(thresholds), np.int64)
    for k, t in enumerate(thresholds):
        sel = max_score >= t
        cor = sel & is_sig & close
        n_sel[k] = int(sel.sum())
        n_cor[k] = int(cor.sum())
        eff[k] = n_cor[k] / n_sig if n_sig else np.nan
        pur[k] = n_cor[k] / n_sel[k] if n_sel[k] else np.nan
    return dict(thresholds=np.asarray(thresholds), efficiency=eff, purity=pur,
                n_selected=n_sel, n_correct=n_cor, n_truth=n_sig,
                n_background=n - n_sig, max_score=max_score, best_dist=best_dist)


def counts_per_slice(slices, thresholds):
    """For each threshold, array (n_slices,) of #candidates with score >= t."""
    out = {}
    for t in thresholds:
        out[t] = np.asarray([int((s["cand_score"] >= t).sum())
                             for s in slices], np.int64)
    return out


# --------------------------------------------------------------------------- #
#  plotting                                                                    #
# --------------------------------------------------------------------------- #
def make_plots(roc, dist_best, count_dist, outdir, match_dist):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # (1a) efficiency & purity vs threshold ------------------------------------
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(roc["thresholds"], roc["efficiency"], "-o", ms=3,
            color="C0", label="efficiency (recall)")
    ax.plot(roc["thresholds"], roc["purity"], "-s", ms=3,
            color="C3", label=f"purity (within {match_dist:g} cm)")
    for t in COUNT_THRESHOLDS:
        ax.axvline(t, color="0.8", lw=0.8, zorder=0)
    ax.set_xlabel("max nu-vertex score threshold")
    ax.set_ylabel("fraction")
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlim(roc["thresholds"][0], roc["thresholds"][-1])
    ax.set_title(f"Nu-vertex selection vs score threshold "
                 f"({roc['n_truth']} signal + {roc.get('n_background', 0)} bkg "
                 f"slices)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    p1 = os.path.join(outdir, "nu_vertex_eff_purity_vs_threshold.png")
    fig.savefig(p1, dpi=130)
    plt.close(fig)

    # (1b) purity vs efficiency (ROC) ------------------------------------------
    fig, ax = plt.subplots(figsize=(6, 5.5))
    ax.plot(roc["efficiency"], roc["purity"], "-", color="C2", lw=1)
    sc = ax.scatter(roc["efficiency"], roc["purity"], c=roc["thresholds"],
                    cmap="viridis", s=18, zorder=3)
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label("score threshold")
    ax.set_xlabel("efficiency (recall)")
    ax.set_ylabel(f"purity (within {match_dist:g} cm)")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("Nu-vertex ROC: purity vs efficiency")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    p2 = os.path.join(outdir, "nu_vertex_roc_purity_vs_efficiency.png")
    fig.savefig(p2, dpi=130)
    plt.close(fig)

    # (2) distance(true, reco) distribution ------------------------------------
    fig, ax = plt.subplots(figsize=(7, 5))
    d = dist_best[np.isfinite(dist_best)]
    p3 = os.path.join(outdir, "nu_vertex_distance_distribution.png")
    if d.size:
        hi = float(np.percentile(d, 98)) if d.size > 4 else float(d.max())
        hi = max(hi, 1.0)
        ax.hist(d, bins=np.linspace(0, hi, 31), color="C0",
                edgecolor="k", alpha=0.8)
        ax.axvline(np.median(d), color="C3", lw=1.5,
                   label=f"median={np.median(d):.2f} cm")
        ax.axvline(d.mean(), color="C1", lw=1.5, ls="--",
                   label=f"mean={d.mean():.2f} cm")
        ax.legend()
    else:
        ax.text(0.5, 0.5, "no reco nu vertices", ha="center",
                transform=ax.transAxes)
    ax.set_xlabel("distance true -> reco nu vertex (cm)")
    ax.set_ylabel("slices")
    ax.set_title(f"Nu-vertex resolution (max-score vertex per slice, "
                 f"N={d.size})")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(p3, dpi=130)
    plt.close(fig)

    # (3) nu vertices per slice vs threshold -----------------------------------
    fig, ax = plt.subplots(figsize=(7, 5))
    allcounts = np.concatenate([v for v in count_dist.values()]) \
        if count_dist else np.zeros(0)
    cmax = int(allcounts.max()) if allcounts.size else 0
    bins = np.arange(-0.5, cmax + 1.5, 1.0)
    centers = np.arange(0, cmax + 1)
    width = 0.8 / max(len(count_dist), 1)
    for j, (t, cnts) in enumerate(sorted(count_dist.items())):
        h, _ = np.histogram(cnts, bins=bins)
        ax.bar(centers + (j - (len(count_dist) - 1) / 2) * width, h,
               width=width, label=f"score>={t:g}  (mean={cnts.mean():.2f})")
    ax.set_xticks(centers)
    ax.set_xlabel("# reco nu vertices per slice")
    ax.set_ylabel("slices")
    ax.set_title("Nu-vertex multiplicity per slice vs score threshold")
    ax.grid(alpha=0.3, axis="y")
    ax.legend()
    fig.tight_layout()
    p4 = os.path.join(outdir, "nu_vertices_per_slice_vs_threshold.png")
    fig.savefig(p4, dpi=130)
    plt.close(fig)

    return [p1, p2, p3, p4]


# --------------------------------------------------------------------------- #
#  text / CSV reporting                                                        #
# --------------------------------------------------------------------------- #
def write_reports(roc, dist_best, count_dist, per_slice, outdir, match_dist,
                  args):
    # ROC table at the user-requested thresholds
    roc_path = os.path.join(outdir, "nu_vertex_roc_table.csv")
    with open(roc_path, "w") as f:
        f.write("threshold,n_selected,n_correct,n_truth,efficiency,purity\n")
        for k, t in enumerate(roc["thresholds"]):
            f.write(f"{t:.4f},{roc['n_selected'][k]},{roc['n_correct'][k]},"
                    f"{roc['n_truth']},{roc['efficiency'][k]:.4f},"
                    f"{roc['purity'][k]:.4f}\n")

    # per-slice dump
    sl_path = os.path.join(outdir, "nu_vertex_per_slice.csv")
    with open(sl_path, "w") as f:
        f.write("file,max_score,best_dist_cm," +
                ",".join(f"ncand_ge_{t:g}" for t in COUNT_THRESHOLDS) + "\n")
        for s in per_slice:
            ms = s["cand_score"][0] if s["cand_score"].size else 0.0
            bd = (np.linalg.norm(s["cand_pos"][0] - s["true_nu"])
                  if (s["cand_score"].size and s["true_nu"] is not None)
                  else float("nan"))
            ncs = ",".join(str(int((s["cand_score"] >= t).sum()))
                           for t in COUNT_THRESHOLDS)
            f.write(f"{os.path.basename(s['file'])},{ms:.4f},{bd:.3f},{ncs}\n")

    # human-readable summary
    sum_path = os.path.join(outdir, "nu_vertex_eval_summary.txt")
    d = dist_best[np.isfinite(dist_best)]
    lines = []
    lines.append("=== Nu-vertex reco evaluation ===")
    lines.append(f"input dir            : {args.input}")
    lines.append(f"signal slices (nu)   : {roc['n_truth']}")
    lines.append(f"background slices     : {roc.get('n_background', 0)} "
                 f"(no true nu vertex; selections here are impurities)")
    lines.append(f"reco score floor     : {args.min_score}")
    lines.append(f"match distance (cm)  : {match_dist}")
    lines.append(f"sigma / radius (cm)  : {args.sigma_cm} / {args.radius_cm}")
    lines.append("")
    lines.append("-- distance true->reco (max-score vertex per slice) --")
    if d.size:
        lines.append(f"  n={d.size}  median={np.median(d):.2f} cm  "
                     f"mean={d.mean():.2f} cm  "
                     f"p90={np.percentile(d, 90):.2f} cm")
        for thr in (1.0, 3.0, 5.0, 10.0):
            lines.append(f"  frac within {thr:5.1f} cm : "
                         f"{float((d < thr).mean()):.3f}")
    else:
        lines.append("  (no reco nu vertices)")
    lines.append("")
    lines.append("-- ROC at user thresholds --")
    lines.append(f"  {'thr':>5} {'eff':>7} {'purity':>7} {'n_sel':>6} "
                 f"{'n_cor':>6}")
    for t in COUNT_THRESHOLDS:
        k = int(np.argmin(np.abs(roc["thresholds"] - t)))
        lines.append(f"  {t:5.2f} {roc['efficiency'][k]:7.3f} "
                     f"{roc['purity'][k]:7.3f} {roc['n_selected'][k]:6d} "
                     f"{roc['n_correct'][k]:6d}")
    lines.append("")
    lines.append("-- nu vertices per slice (mean +/- std) --")
    for t in COUNT_THRESHOLDS:
        c = count_dist[t]
        lines.append(f"  score>={t:g} : mean={c.mean():.2f}  std={c.std():.2f}"
                     f"  max={int(c.max()) if c.size else 0}")
    text = "\n".join(lines)
    with open(sum_path, "w") as f:
        f.write(text + "\n")
    return text, [roc_path, sl_path, sum_path]


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("input",
                    help="dir of keypoint2 inference H5s (with score_maps), "
                         "or a glob/single file")
    ap.add_argument("--output-dir", default="nu_vertex_eval_out")
    ap.add_argument("--match-dist", type=float, default=5.0,
                    help="max true->reco distance (cm) counted as a correct "
                         "nu-vertex selection (default 5.0)")
    ap.add_argument("--min-score", type=float, default=0.05,
                    help="reco score floor; candidates below this are not "
                         "reconstructed (must be <= smallest count threshold; "
                         "default 0.05)")
    ap.add_argument("--max-candidates", type=int, default=32,
                    help="cap on reco nu candidates per slice (default 32)")
    ap.add_argument("--n-roc-points", type=int, default=99,
                    help="number of threshold points in the ROC sweep")
    ap.add_argument("--radius-cm", type=float, default=10.0)
    ap.add_argument("--sigma-cm", type=float, default=3.0)
    ap.add_argument("--fit-method", default="nls",
                    choices=["nls", "loglinear", "centroid"])
    ap.add_argument("--amplitude", default="peak", choices=["peak", "fit"])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--n-events", type=int, default=-1)
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()

    if os.path.isdir(args.input):
        files = sorted(glob.glob(os.path.join(args.input, "*.h5")))
    else:
        files = sorted(glob.glob(args.input))
    if args.n_events >= 0:
        files = files[:args.n_events]
    if not files:
        print(f"no H5 files matched {args.input!r}")
        return
    os.makedirs(args.output_dir, exist_ok=True)

    params = KeypointRecoParams(
        radius_cm=args.radius_cm, sigma_cm=args.sigma_cm,
        score_thresh=args.min_score, max_keypoints=args.max_candidates,
        fit_method=args.fit_method, amplitude=args.amplitude,
        device=args.device)
    reco = KeypointRecoTorch(params)

    per_slice = []
    n_no_score_maps = n_no_truth = 0
    for i, path in enumerate(files):
        try:
            loaded = io.load_score_maps(path)
        except KeyError:
            n_no_score_maps += 1
            print(f"  [skip] {os.path.basename(path)}: no score_maps")
            continue
        s = reco_slice(reco, loaded, args.max_candidates)
        s["file"] = path
        per_slice.append(s)
        ms = s["cand_score"][0] if s["cand_score"].size else 0.0
        bd = (np.linalg.norm(s["cand_pos"][0] - s["true_nu"])
              if (s["cand_score"].size and s["true_nu"] is not None)
              else float("nan"))
        if s["true_nu"] is None:
            n_no_truth += 1
        print(f"  [{i}] {os.path.basename(path)}: "
              f"{s['cand_score'].size} nu candidates  "
              f"max_score={ms:.3f}  best_dist={bd:.2f} cm")

    n_capped = sum(s["cand_score"].size >= args.max_candidates
                   for s in per_slice)
    if n_capped:
        print(f"  [warn] {n_capped} slice(s) hit the --max-candidates cap "
              f"({args.max_candidates}); the per-slice count distribution is "
              f"truncated at low thresholds. Raise --max-candidates to widen it.")

    n_signal = sum(s["true_nu"] is not None for s in per_slice)
    if not n_signal:
        print("\nNo slices with a true nu vertex — cannot build ROC/resolution.")
        return

    thresholds = np.linspace(args.min_score, 0.99, args.n_roc_points)
    roc = build_roc(per_slice, args.match_dist, thresholds)
    count_dist = counts_per_slice(per_slice, COUNT_THRESHOLDS)
    text, report_paths = write_reports(
        roc, roc["best_dist"], count_dist, per_slice, args.output_dir,
        args.match_dist, args)

    print(f"\n{text}\n")
    print(f"slices processed     : {len(per_slice)}  "
          f"(no score_maps: {n_no_score_maps}, no truth: {n_no_truth})")
    plot_paths = []
    if not args.no_plots:
        try:
            plot_paths = make_plots(roc, roc["best_dist"], count_dist,
                                    args.output_dir, args.match_dist)
        except Exception as exc:                       # pragma: no cover
            print(f"  [warn] plotting failed: {exc}")
    for p in report_paths + plot_paths:
        print(f"  wrote {p}")


if __name__ == "__main__":
    main()
