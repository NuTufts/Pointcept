"""Quantify keypoint-2 performance from the cascade INFERENCE output.

Reads the per-event H5 files written by
`tools/run_larformer_keypoint2_cascade_inference.py` and computes the SAME
per-particle keypoint metrics as the training-time evaluator
(`LArFormerLevelKeypointEvaluator`), so you get a number instead of eyeballing
the 3D viz — and you can compare runs (e.g. `--particle-source predicted` vs
`gt`) side by side.

Per matched particle (has_match=True): start error = ||pred start_cm − GT
start_cm|| (GT = the VISIBLE start the loss now uses), end error likewise for
track particles. Per event: nu-vertex resolution = ||pred − GT nu_vertex_cm||.
Reports, micro-averaged exactly like the evaluator:
    start/end : n, median, mean, R@{1,3,10}cm  (R = fraction within threshold)
    nu-vertex : n, median, mean, recall@3cm
plus a per-predicted-class breakdown of the start metric (showers vs tracks —
the key split for the origin-vs-visible-start question) and matched-IoU stats.

NOTE: object-head AP (val/object_ap) is NOT computed here — the inference H5
stores decoded keypoints, not the per-dec2-token object scores. Watch that one
in the training log; this script covers the per-particle + nu-vertex metrics.

  python eval_keypoint2_inference.py <dir> [<dir2> ...]
"""
import argparse
import glob
import os

import numpy as np
import h5py

CLS = ["e", "gamma", "mu", "pi", "p", "other", "(unused)", "no_object"]
THRS = (1.0, 3.0, 10.0)


def _cls(c):
    return CLS[c] if 0 <= c < len(CLS) else str(c)


def _err(pred, gt):
    pred, gt = np.asarray(pred), np.asarray(gt)
    if not (np.all(np.isfinite(pred)) and np.all(np.isfinite(gt))):
        return np.nan
    return float(np.linalg.norm(pred - gt))


def load_dir(d):
    recs, nu_res = [], []
    n_ev = n_part = n_match = 0
    for p in sorted(glob.glob(os.path.join(d, "*.h5"))):
        with h5py.File(p, "r") as f:
            n_ev += 1
            r = _err(f["nu_vertex_cm"][:], f["gt_nu_vertex_cm"][:])
            if np.isfinite(r):
                nu_res.append(r)
            if "particle" not in f:
                continue
            for i in f["particle"]:
                g = f["particle"][i]
                n_part += 1
                if not bool(g.attrs.get("has_match", False)):
                    continue
                n_match += 1
                recs.append(dict(
                    cls=int(g.attrs.get("cls", -1)),
                    iou=float(g.attrs.get("iou", 0.0)),
                    start=_err(g["start_cm"][:], g["gt_start_cm"][:]),
                    end=_err(g["end_cm"][:], g["gt_end_cm"][:])))
    return dict(n_ev=n_ev, n_part=n_part, n_match=n_match,
                recs=recs, nu_res=nu_res)


def _line(name, errs):
    e = np.asarray([x for x in errs if np.isfinite(x)], dtype=np.float64)
    if e.size == 0:
        return f"  {name:22s} (no GT)"
    s = (f"  {name:22s} n={e.size:4d}  median={np.median(e):6.2f}  "
         f"mean={e.mean():7.2f} cm")
    for t in THRS:
        s += f"  R{t:g}={float((e < t).mean()):.3f}"
    return s


def report(d):
    r = load_dir(d)
    print(f"\n=== {d} ===")
    print(f"  events={r['n_ev']}  predicted particles={r['n_part']}  "
          f"matched(GT)={r['n_match']}")
    if r["recs"]:
        ious = np.array([x["iou"] for x in r["recs"]])
        print(f"  match IoU: median={np.median(ious):.2f} mean={ious.mean():.2f}")
    print(_line("start (all)", [x["start"] for x in r["recs"]]))
    print(_line("end (all)", [x["end"] for x in r["recs"]]))
    # per-class start breakdown (showers vs tracks)
    classes = sorted({x["cls"] for x in r["recs"]})
    for c in classes:
        errs = [x["start"] for x in r["recs"] if x["cls"] == c]
        print(_line(f"start [{_cls(c)}]", errs))
    # nu vertex
    nu = np.asarray(r["nu_res"], dtype=np.float64)
    if nu.size:
        print(f"  {'nu_vertex':22s} n={nu.size:4d}  median={np.median(nu):6.2f}  "
              f"mean={nu.mean():7.2f} cm  recall@3cm="
              f"{float((nu < 3.0).mean()):.3f}")
    else:
        print(f"  {'nu_vertex':22s} (no GT)")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("dirs", nargs="+",
                    help="one or more dirs of keypoint2_event*.h5")
    args = ap.parse_args()
    for d in args.dirs:
        report(d)


if __name__ == "__main__":
    main()
