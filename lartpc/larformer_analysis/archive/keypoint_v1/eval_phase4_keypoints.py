"""Phase 4 evaluation: run the dense keypoint model, decode, score vs GT.

Builds the Phase-1 dense model (LArFormerKeypoint, score + offset heads) from
`larformer-keypoint-v1.py`, runs it on dev-cache events, decodes discrete
keypoints (vote + cluster), and reports per-type precision/recall within
distance thresholds vs the `mckeypoints` GT (PPN-style), plus nu-vertex
resolution.

With no trained weights yet (cluster down), `--overfit-steps N` first overfits
the heads on the eval events so the decode pipeline is exercised on a model
that has actually learned the field — recall should then be high, validating
decode + metrics end-to-end. Pass `--weights PATH` to score a real checkpoint.

    python lartpc/larformer_analysis/archive/keypoint_v1/eval_phase4_keypoints.py \
        --overfit-steps 300 --device cuda
"""

import argparse
import copy
import sys

import numpy as np
import torch

from pointcept.utils.config import Config
from pointcept.models.builder import build_model
from pointcept.datasets.builder import build_dataset
from pointcept.datasets.larformer import larformer_collate
from pointcept.models.LArFormer.keypoint_eval import (
    decode_dense_votes, decode_nu_vertex, accumulate_metrics,
    format_metrics_table, denorm,
)
from lartpc.data_prep.labels.keypoint_labels import N_KEYPOINT_TYPES, KPTYPE_NU_VERTEX

CONFIG = "configs/lartpc/larformer/stage4_keypoint/archive/larformer-keypoint-v1.py"
# 1 cm is the precision target (GT proximity sigma = 3 cm); 3/10 cm are
# coarse/loose context. Watch recall@1cm.
THR = [1.0, 3.0, 10.0]


def gt_by_type(sample, center, scale, coord_cm=None, acceptance_cm=None,
               n_types=N_KEYPOINT_TYPES):
    """GT keypoints per type, in cm. If `acceptance_cm` is set, keep only GT
    keypoints within that distance of any SP in `coord_cm` — i.e. the
    "reconstructable" keypoints this (nu-slice) SP set could vote for. Cosmic
    keypoints with no SPs in the slice are otherwise unfair guaranteed misses.
    """
    pos = denorm(np.asarray(sample["mckeypoints_pos_norm"], dtype=np.float32),
                 center, scale)
    typ = np.asarray(sample["mckeypoints_type"], dtype=np.int64)
    if acceptance_cm is not None and coord_cm is not None and pos.shape[0] > 0:
        from scipy.spatial import cKDTree
        tree = cKDTree(np.asarray(coord_cm, dtype=np.float32))
        d, _ = tree.query(pos, k=1)
        keep = d <= acceptance_cm
        pos, typ = pos[keep], typ[keep]
    out = {}
    for t in range(n_types):
        sel = typ == t
        out[t] = pos[sel] if sel.any() else np.zeros((0, 3), np.float32)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=CONFIG)
    ap.add_argument("--weights", default=None)
    ap.add_argument("--split", default="train")
    ap.add_argument("--n-events", type=int, default=10)
    ap.add_argument("--overfit-steps", type=int, default=0)
    ap.add_argument("--score-thresh", type=float, default=0.3)
    ap.add_argument("--cluster-radius-cm", type=float, default=3.0)
    ap.add_argument("--gt-acceptance-cm", type=float, default=5.0,
                    help="Score only GT keypoints within this distance of any "
                         "SP in the (nu-slice) event. 0/negative = score all "
                         "GT (includes unreconstructable cosmic keypoints).")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    device = torch.device(args.device)

    cfg = Config.fromfile(args.config)
    center = np.asarray(cfg.get("coord_center", (125.0, 0.0, 518.0)),
                        dtype=np.float32)
    scale = float(cfg.get("coord_scale", 179.55))

    ds = build_dataset(cfg.data[args.split])
    n = min(args.n_events, len(ds))
    samples = [ds[i] for i in range(n)]
    batch = larformer_collate(samples)
    batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v)
             for k, v in batch.items()}
    print(f"[p4-eval] {n} events; "
          f"{sum(s['mckeypoints_type'].shape[0] for s in samples)} GT keypoints")

    model = build_model(copy.deepcopy(cfg.model)).to(device)
    if args.weights:
        ck = torch.load(args.weights, map_location="cpu", weights_only=False)
        sd = ck.get("state_dict", ck)
        sd = {(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()}
        miss, unexp = model.load_state_dict(sd, strict=False)
        print(f"[p4-eval] loaded {args.weights}: missing={len(miss)} "
              f"unexpected={len(unexp)}")

    if args.overfit_steps > 0:
        model.train()
        opt = torch.optim.Adam(
            [p for p in model.parameters() if p.requires_grad], lr=1e-3)
        for step in range(args.overfit_steps):
            opt.zero_grad()
            out = model(batch)
            out["loss"].backward()
            opt.step()
            if step % max(1, args.overfit_steps // 5) == 0:
                print(f"  overfit step {step:4d}  loss={float(out['loss']):.5f}")

    # ---- eval forward → decode → metrics ----
    model.eval()
    with torch.no_grad():
        out = model(batch)
    preds = out["predictions"]

    events_pred, events_gt = [], []
    vertex_res = []
    for i, (s, p) in enumerate(zip(samples, preds)):
        coord_norm = p["coord_norm"].cpu().numpy()
        scores = p["kpscores_pred"].cpu().numpy()
        offsets = (p["kpoffsets_pred"].cpu().numpy()
                   if "kpoffsets_pred" in p else None)
        decoded = decode_dense_votes(
            coord_norm, scores, offsets, center, scale,
            score_thresh=args.score_thresh,
            cluster_radius_cm=args.cluster_radius_cm)
        acc = args.gt_acceptance_cm if args.gt_acceptance_cm > 0 else None
        gt = gt_by_type(s, center, scale,
                        coord_cm=np.asarray(s["coord"], dtype=np.float32),
                        acceptance_cm=acc)
        events_pred.append(decoded)
        events_gt.append(gt)
        # nu-vertex resolution (best type-0 cluster vs GT vertex)
        nv = decode_nu_vertex(decoded)
        gtv = gt[KPTYPE_NU_VERTEX]
        if nv is not None and gtv.shape[0] > 0:
            vertex_res.append(float(
                np.linalg.norm(gtv - nv["pos_cm"], axis=1).min()))

    metrics = accumulate_metrics(events_pred, events_gt, THR)
    print("\n[p4-eval] per-type precision/recall (PPN-style):")
    print(format_metrics_table(metrics, THR))
    if vertex_res:
        vr = np.array(vertex_res)
        print(f"\n[p4-eval] nu-vertex resolution: median={np.median(vr):.2f}cm "
              f"mean={vr.mean():.2f}cm over {len(vr)} events")
    else:
        print("\n[p4-eval] no nu-vertex predicted/GT to score")

    # Sanity for the overfit validation: the nu vertex (the design-critical,
    # ~1-per-event, non-coincident keypoint the vertex decision rests on)
    # should be recovered with high recall + small residual. (track_start /
    # track_end recall are coincidence-limited — many primaries share the
    # vertex — so they're not the right pass/fail gate.)
    if args.overfit_steps > 0:
        r = metrics[KPTYPE_NU_VERTEX][10.0]["recall"]
        res = float(np.median(vertex_res)) if vertex_res else float("inf")
        ok = r >= 0.5 and res < 10.0
        print(f"\n[p4-eval] nu_vertex recall@10cm = {r:.3f}, "
              f"median resolution = {res:.2f}cm ({'OK' if ok else 'LOW'})")
        sys.exit(0 if ok else 3)
    sys.exit(0)


if __name__ == "__main__":
    main()
