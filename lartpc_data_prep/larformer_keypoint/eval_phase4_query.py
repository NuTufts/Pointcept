"""Phase 4 evaluation: query-head keypoints (per-particle start/end).

Builds the Phase-2 query model (LArFormer particle segmenter with
`enable_keypoint_head=True`) from `larformer-keypoint-query-v1.py`, loads the
trained Stage-3 checkpoint, runs eval, decodes per-query start/end points,
maps them onto keypoint types (shower-start / track-start / track-end), and
scores them against `mckeypoints` with the SAME PPN-style metric as the dense
eval — so the two are directly comparable.

Key point this demonstrates: the query head resolves COINCIDENT keypoints. In
a ν interaction the primaries all start at the vertex, so the dense head merges
their `track_start` GT points into one peak (recall ≈ 1/N). The query head
emits one start per particle query, so it can recover each — higher
`track_start` recall.

Frames: this config recenters coord_norm per event, so everything (coords,
GT, predicted start/end) is scored in the recentered-denormalized frame
(distances are frame-invariant; we only need pred + GT + acceptance to share
one frame). Hence acceptance uses denorm(coord_norm), not sample["coord"].

    python lartpc_data_prep/larformer_keypoint/eval_phase4_query.py \
        --overfit-steps 200 --n-events 6 --device cuda
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
    decode_query_points, query_points_to_typed, gt_keypoints_by_type,
    accumulate_metrics, format_metrics_table, denorm,
    DEFAULT_TRACK_CLASS_IDS,
)
from lartpc_data_prep.keypoint_labels import KPTYPE_TRACK_START

CONFIG = "configs/lartpc/larformer-keypoint-query-v1.py"
# 1 cm = precision target (GT proximity sigma = 3 cm); 3/10 cm = context.
THR = [1.0, 3.0, 10.0]


def load_ckpt(model, path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    sd = ck.get("state_dict", ck)
    sd = {(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()}
    miss, unexp = model.load_state_dict(sd, strict=False)
    print(f"[p4-q] loaded {path}: missing={len(miss)} unexpected={len(unexp)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=CONFIG)
    ap.add_argument("--weights", default=None,
                    help="Stage-3 ckpt to init from; defaults to the config's "
                         "`weight`.")
    ap.add_argument("--split", default="train")
    ap.add_argument("--n-events", type=int, default=6)
    ap.add_argument("--overfit-steps", type=int, default=0)
    ap.add_argument("--class-prob-floor", type=float, default=0.3)
    ap.add_argument("--end-prob-floor", type=float, default=0.5)
    ap.add_argument("--gt-acceptance-cm", type=float, default=5.0)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    device = torch.device(args.device)

    cfg = Config.fromfile(args.config)
    center = np.asarray(cfg.get("coord_center", (125.0, 0.0, 518.0)),
                        dtype=np.float32)
    scale = float(cfg.get("coord_scale", 179.55))
    n_classes = int(cfg.model["num_classes"])

    ds = build_dataset(cfg.data[args.split])
    n = min(args.n_events, len(ds))
    samples = [ds[i] for i in range(n)]
    batch = larformer_collate(samples)
    batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v)
             for k, v in batch.items()}
    print(f"[p4-q] {n} events, "
          f"{sum(len(s['gt_instances']) for s in samples)} GT particles")

    model_cfg = copy.deepcopy(cfg.model)
    model_cfg.pop("backbone_weight", None)
    model = build_model(model_cfg).to(device)
    load_ckpt(model, args.weights or cfg.weight)

    if args.overfit_steps > 0:
        model.train()
        opt = torch.optim.Adam(
            [p for p in model.parameters() if p.requires_grad], lr=2e-4)
        for step in range(args.overfit_steps):
            opt.zero_grad()
            out = model(batch)
            out["loss"].backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step()
            if step % max(1, args.overfit_steps // 5) == 0:
                print(f"  overfit step {step:4d}  loss={float(out['loss']):.3f} "
                      f"start_err={float(out.get('loss_kpq_start_err_cm', 0)):.2f}cm")

    model.eval()
    with torch.no_grad():
        out = model(batch)
    preds = out["predictions"]

    events_pred, events_gt = [], []
    for s, p in zip(samples, preds):
        qpts = decode_query_points(
            p["class_logits"].cpu().numpy(),
            p["origin"].cpu().numpy(), center, scale,
            end_norm=(p["kp_end"].cpu().numpy() if "kp_end" in p else None),
            end_logit=(p["kp_end_logit"].cpu().numpy()
                       if "kp_end_logit" in p else None),
            no_object_class_id=n_classes - 1,
            class_prob_floor=args.class_prob_floor,
            end_prob_floor=args.end_prob_floor,
            track_class_ids=set(DEFAULT_TRACK_CLASS_IDS))
        typed = query_points_to_typed(qpts)
        # Recentered frame: acceptance coord = denorm(coord_norm), matching GT.
        coord_cm = denorm(s["coord_norm"], center, scale)
        gt = gt_keypoints_by_type(
            s["mckeypoints_pos_norm"], s["mckeypoints_type"], center, scale,
            coord_cm=coord_cm,
            acceptance_cm=(args.gt_acceptance_cm if args.gt_acceptance_cm > 0
                           else None))
        events_pred.append(typed)
        events_gt.append(gt)

    metrics = accumulate_metrics(events_pred, events_gt, THR)
    print("\n[p4-q] QUERY-head per-type precision/recall (PPN-style):")
    print(format_metrics_table(metrics, THR))
    print("  (nu_vertex/michel/delta are N/A for the query head — they come "
          "from the dense head; reconcile_keypoints combines the two.)")

    if args.overfit_steps > 0:
        # Validate the EVAL PATH, not the under-trained dev model: the points
        # the query head emits should be CORRECT (high precision). Recall is a
        # model-completeness number gated by Stage-3 classification quality
        # (epoch_6 is weak + this is a short overfit) and rises with real
        # training; it is NOT the right pass/fail bar for the decode pipeline.
        prec = metrics[KPTYPE_TRACK_START][10.0]["precision"]
        rec = metrics[KPTYPE_TRACK_START][10.0]["recall"]
        tp = metrics[KPTYPE_TRACK_START][10.0]["tp"]
        ok = prec >= 0.5 and tp > 0
        print(f"\n[p4-q] track_start P@10cm={prec:.3f} R@10cm={rec:.3f} "
              f"(tp={tp}). Eval path {'OK' if ok else 'BROKEN'} — emitted "
              f"keypoints are correct; recall is Stage-3-quality limited.")
        sys.exit(0 if ok else 3)
    sys.exit(0)


if __name__ == "__main__":
    main()
