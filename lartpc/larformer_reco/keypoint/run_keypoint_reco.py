"""Reconstruct keypoints from a directory of cascade score-map H5 files.

Standalone post-processor (Option A in ``../keypoint_reco_spec.md``): reads the
``--save-score-maps`` output of
``tools/larformer/run_larformer_keypoint2_cascade_inference.py``, runs the greedy
peel-and-fit reco on each head's dense score field, writes per-event reco H5s,
and (with ``--eval``, on sim) reports start/object + nu-vertex resolution vs the
GT keypoints — micro-averaged, in the style of ``eval_keypoint2_inference.py``.

Run inside the pointcept container (needs numpy/torch/scipy/h5py):

    python -m lartpc.larformer_reco.keypoint.run_keypoint_reco \
        <score_map_dir> --output-dir kp_reco_out --eval

(or ``python lartpc/larformer_reco/keypoint/run_keypoint_reco.py``)
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np

# allow both ``python -m ...`` and direct-path invocation
if __package__ in (None, ""):                        # pragma: no cover
    sys.path.insert(0, os.path.abspath(os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")))
    from lartpc.larformer_reco.keypoint import (KeypointRecoTorch, KeypointRecoParams, best_nu_vertex,
                      io)
else:
    from . import io
    from .keypoint_reco import (KeypointRecoTorch, KeypointRecoParams,
                                best_nu_vertex)

THRS = (1.0, 3.0, 10.0)
# GT keypoint types (lartpc/data_prep/labels/keypoint_labels.py)
NU_TYPE = 0
OBJECT_TYPES = (1, 2, 3, 4, 5)   # track_start/end, shower, michel, delta


def _greedy_match(pred_cm, pred_score, gt_cm):
    """Greedy nearest match pred->gt (highest-score pred first). Returns the
    list of matched distances (cm), one per matched pred."""
    if len(pred_cm) == 0 or len(gt_cm) == 0:
        return []
    pred_cm = np.asarray(pred_cm, np.float64).reshape(-1, 3)
    gt_cm = np.asarray(gt_cm, np.float64).reshape(-1, 3)
    order = np.argsort(-np.asarray(pred_score, np.float64))
    used = np.zeros(len(gt_cm), dtype=bool)
    dists = []
    for i in order:
        d = np.linalg.norm(gt_cm - pred_cm[i], axis=1)
        d[used] = np.inf
        j = int(np.argmin(d))
        if np.isfinite(d[j]):
            used[j] = True
            dists.append(float(d[j]))
    return dists


def _line(name, errs, with_recall=True):
    e = np.asarray([x for x in errs if np.isfinite(x)], np.float64)
    if e.size == 0:
        return f"  {name:20s} (none)"
    s = (f"  {name:20s} n={e.size:4d}  median={np.median(e):6.2f}  "
         f"mean={e.mean():7.2f} cm")
    if with_recall:
        for t in THRS:
            s += f"  R{t:g}={float((e < t).mean()):.3f}"
    return s


def main():
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("input", help="dir of score-map H5s, or a glob/single file")
    ap.add_argument("--output-dir", default="kp_reco_out")
    ap.add_argument("--eval", action="store_true",
                    help="compare reco keypoints to GT (sim only)")
    ap.add_argument("--radius-cm", type=float, default=10.0)
    ap.add_argument("--sigma-cm", type=float, default=3.0)
    ap.add_argument("--score-thresh", type=float, default=0.67)
    ap.add_argument("--max-keypoints", type=int, default=64)
    ap.add_argument("--fit-method", default="nls",
                    choices=["nls", "loglinear", "centroid"])
    ap.add_argument("--amplitude", default="peak", choices=["peak", "fit"])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--n-events", type=int, default=-1)
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
        score_thresh=args.score_thresh, max_keypoints=args.max_keypoints,
        fit_method=args.fit_method, amplitude=args.amplitude,
        device=args.device)
    reco = KeypointRecoTorch(params)

    nu_errs, obj_errs = [], []
    n_ev = n_kp = 0
    for i, path in enumerate(files):
        loaded = io.load_score_maps(path)
        event_result = reco.reconstruct_event(loaded["score_maps"])
        best_nu = best_nu_vertex(event_result)
        n_ev += 1
        n_kp += sum(len(r["keypoints"]) for r in event_result.values())

        outp = os.path.join(args.output_dir,
                            "reco_" + os.path.basename(path))
        io.write_reco_h5(outp, event_result, best_nu, loaded, params)

        if args.eval and loaded.get("gt_keypoints") is not None:
            gt = loaded["gt_keypoints"]
            # nu-vertex resolution (best nu peak vs GT type-0)
            gt_nu = gt["pos_cm"][gt["type"] == NU_TYPE]
            if best_nu is not None and len(gt_nu):
                nu_errs.append(float(np.linalg.norm(best_nu.pos_cm - gt_nu[0])))
            elif loaded.get("gt_nu_vertex_cm") is not None and best_nu is not None:
                g0 = loaded["gt_nu_vertex_cm"]
                if np.all(np.isfinite(g0)):
                    nu_errs.append(float(np.linalg.norm(best_nu.pos_cm - g0)))
            # object keypoints vs GT non-nu types
            obj_pred = [kp for name, r in event_result.items()
                        if not r["is_nu"] for kp in r["keypoints"]]
            gt_obj = gt["pos_cm"][np.isin(gt["type"], OBJECT_TYPES)]
            obj_errs += _greedy_match(
                [k.pos_cm for k in obj_pred],
                [k.peak_score for k in obj_pred], gt_obj)

        nkp = sum(len(r['keypoints']) for r in event_result.values())
        nu_str = (best_nu.pos_cm.round(1).tolist() if best_nu is not None
                  else None)
        print(f"  [{i}] {os.path.basename(path)}: {nkp} keypoints  "
              f"nu_vertex_cm={nu_str} -> {outp}")

    print(f"\nDONE  events={n_ev}  total reco keypoints={n_kp}")
    if args.eval:
        print("\n=== reco vs GT (micro-averaged) ===")
        print(_line("nu_vertex", nu_errs))
        print(_line("object kp", obj_errs))


if __name__ == "__main__":
    main()
