"""Full-cascade inference WITH keypoints (deghoster + slicer + particle
segmenter + keypoint heads).

Extension of `tools/larformer/run_larformer_stage3_inference.py` (full-cascade mode) that
additionally decodes and writes KEYPOINTS into each per-event H5:

  - Query head (per-particle start/end): the particle segmenter is built with
    `enable_keypoint_head=True`, so each matched particle query predicts a
    start (= origin head) and, for track classes, an end. Decoded into typed
    keypoints (track_start / track_end / shower) — particle-associated.
  - Nu vertex (slice-level) + michel/delta: the DENSE per-SP keypoint head.
    Source priority, per event: (1) the particle segmenter's OWN INTEGRATED
    dense head (`enable_keypoint_dense_head`, injected here by default — no
    extra forward; its score channel + offset votes give the vertex);
    (2) else a SEPARATE `LArFormerKeypoint` model via `--keypoint-dense-config`
    / `--keypoint-dense-weights`. Disable the integrated head (e.g. for old
    checkpoints) with `--no-integrated-dense`.

The two are combined by `reconcile_keypoints` (query for start/end/shower,
dense for vertex/michel/delta) and written under `keypoints/...` (see
`keypoint_arrays_for_h5`), alongside the existing slicer + stage3 keys. The
slicer/stage3 halves are produced exactly as in the stage3 script, so the
output is a superset of `stage3pred_*.h5`.

Frames: keypoint positions are detector cm. Query positions are denormalized
via the per-event affine recovered from the nu-slice's (coord, coord_norm)
pairs (correct under slice-recentering). The INTEGRATED dense head shares that
recentered frame (same affine); a SEPARATE dense model is run on a
NON-recentered copy of the nu-slice so its coord_norm denormalizes with the
fixed (coord_center, coord_scale).

Weights:
  --weights                     full CascadedParticleSegmenter ckpt (slicer +
                                particle). Loaded strict=False; the keypoint
                                heads are random unless they're in this ckpt.
  --particle-keypoint-weights   optional: overlay a keypoint-trained particle
                                segmenter (e.g. from larformer-keypoint-query-
                                v1) into the particle_segmenter slot. Use this
                                to mix a trained slicer ckpt + keypoint particle
                                ckpt. Its architecture must match the config's
                                particle_segmenter.
  --keypoint-dense-config/-weights  optional dense (vertex) model.

The --config should be a CascadedParticleSegmenter whose particle_segmenter
matches the keypoint-trained weights, e.g.
`configs/lartpc/larformer/stage3_particle/larformer-particle-fullcascade-ptv3crosslevel.py`.

Usage:
    python tools/larformer/run_larformer_fullcascade_inference.py \\
        --config configs/lartpc/larformer/stage3_particle/larformer-particle-fullcascade-ptv3crosslevel.py \\
        --weights exp/.../cascade_or_slicer.pth \\
        --particle-keypoint-weights exp/larformer_keypoint_query_v1/model/model_best.pth \\
        --keypoint-dense-config configs/lartpc/larformer/stage4_keypoint/archive/larformer-keypoint-v1.py \\
        --keypoint-dense-weights exp/larformer_keypoint_v1/model/model_best.pth \\
        --input-list inputlists/run3b_ncpi0.txt \\
        --output-dir exp/.../keypoint_inference
"""
import argparse
import copy
import os
import sys

import numpy as np
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

# Reuse the stage-3 script's helpers verbatim (loading, batch move, resolve,
# unpack) so the cascade boundary logic is identical.
from tools.larformer.run_larformer_stage3_inference import (   # noqa: E402
    _load_weights_into, _move_batch, _resolve_particle_segmenter,
    _resolve_cascaded_slicer, _unpack_ps_sample, set_deterministic,
    reseed_per_event,
)
from pointcept.models.LArFormer.inference import (   # noqa: E402
    slicer_predict_event_from_out, stage3_predict_event_from_out,
    write_event_h5,
)
from pointcept.models.LArFormer.cascade_particle_filter import (  # noqa: E402
    build_nu_keep_mask, filter_batch_for_particle_segmenter,
    slicer_predictions_empty,
)
from pointcept.models.LArFormer.cascade_filter import drop_empty_events  # noqa: E402
from pointcept.models.LArFormer.keypoint_eval import (  # noqa: E402
    decode_event_keypoints, keypoint_arrays_for_h5, dedup_query_effective_argmax,
)


def _np(x):
    return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)


def _ev0(ps_batch, key):
    """First-event slice of a per-SP field in a (1-event) ps_batch."""
    n_sp = (int(ps_batch["n_spacepoints"][0].item())
            if ps_batch["n_spacepoints"].numel() > 0 else 0)
    v = ps_batch.get(key)
    return _np(v)[:n_sp] if v is not None else None


def decode_keypoints_for_event(ev_pred, ps_batch, dense_pred, n_classes, args):
    """Return a keypoints/* dict for one event (or {} if no keypoint head).

    Dense (nu vertex + michel/delta) source priority:
      1. the particle segmenter's OWN integrated dense head, if its eval
         output carries `dense_kpscores` (same model — no extra forward). Its
         coords are the cascade's recentered frame, so they share the queries'
         affine (dense_in_query_frame=True).
      2. else a SEPARATE dense model's prediction (`dense_pred`), run on a
         NON-recentered batch (fixed coord_center/scale).
    """
    if "origin" not in ev_pred:
        return {}
    coord_cm = _ev0(ps_batch, "coord")
    coord_norm = _ev0(ps_batch, "coord_norm")
    if coord_cm is None or coord_cm.shape[0] == 0:
        return {}

    dense_cn = dense_sc = dense_off = None
    dense_in_query_frame = False
    if "dense_kpscores" in ev_pred:                       # integrated head
        dense_cn = _np(ev_pred["dense_coord_norm"])
        dense_sc = _np(ev_pred["dense_kpscores"])
        dense_off = (_np(ev_pred["dense_kpoffsets"])
                     if "dense_kpoffsets" in ev_pred else None)
        dense_in_query_frame = True
    elif dense_pred is not None:                          # separate model
        dense_cn = _np(dense_pred["coord_norm"])
        dense_sc = _np(dense_pred["kpscores_pred"])
        dense_off = (_np(dense_pred["kpoffsets_pred"])
                     if "kpoffsets_pred" in dense_pred else None)

    # Dedup duplicate queries (mask-IoU NMS — the SAME inference.dedup_queries
    # the particle masks use) so co-extensive duplicate queries don't emit
    # duplicate keypoint FPs. Needs the spacepoint mask logits.
    eff = None
    sp_mask = ev_pred.get("mask_logits", {}).get("spacepoint")
    if sp_mask is not None and args.dedup_iou_threshold > 0:
        # NOTE: pass torch tensors (not _np) — dedup_query_effective_argmax
        # operates on torch (softmax/clone) before returning a numpy argmax.
        eff = dedup_query_effective_argmax(
            ev_pred["class_logits"], sp_mask,
            no_object_class_id=n_classes - 1,
            class_prob_floor=args.kp_class_prob_floor,
            iou_threshold=args.dedup_iou_threshold)

    final, nu = decode_event_keypoints(
        class_logits=_np(ev_pred["class_logits"]),
        origin_norm=_np(ev_pred["origin"]),
        coord_norm=coord_norm, coord_cm=coord_cm,
        effective_argmax=eff,
        end_norm=(_np(ev_pred["kp_end"]) if "kp_end" in ev_pred else None),
        end_logit=(_np(ev_pred["kp_end_logit"]) if "kp_end_logit" in ev_pred
                   else None),
        dense_coord_norm=dense_cn, dense_scores=dense_sc, dense_offsets=dense_off,
        dense_in_query_frame=dense_in_query_frame,
        n_classes=n_classes,
        coord_center=args.coord_center, coord_scale=args.coord_scale,
        kp_class_prob_floor=args.kp_class_prob_floor,
        kp_end_prob_floor=args.kp_end_prob_floor,
        dense_score_thresh=args.kp_score_thresh,
        dense_cluster_radius_cm=args.kp_cluster_radius_cm)
    return keypoint_arrays_for_h5(final, nu)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--config", required=True,
                    help="CascadedParticleSegmenter config (particle_segmenter "
                         "must match the keypoint-trained particle weights).")
    ap.add_argument("--weights", default=None,
                    help="Optional full cascade ckpt (slicer + particle), "
                         "strict=False. If omitted, the cascade self-loads its "
                         "slicer/backbone from the config's weight knobs at "
                         "build time; pair with --particle-keypoint-weights to "
                         "supply the trained particle + keypoint heads.")
    ap.add_argument("--particle-keypoint-weights", default=None,
                    help="Optional: overlay a keypoint-trained particle "
                         "segmenter into the particle_segmenter slot.")
    ap.add_argument("--no-integrated-dense", action="store_true",
                    help="Do NOT use the particle segmenter's own integrated "
                         "dense head for the nu vertex (use for checkpoints "
                         "without one; pair with --keypoint-dense-* instead).")
    ap.add_argument("--keypoint-dense-config", default=None,
                    help="Optional SEPARATE LArFormerKeypoint config (Phase-1 "
                         "dense head) for the nu vertex — only used when the "
                         "integrated head is absent/disabled.")
    ap.add_argument("--keypoint-dense-weights", default=None)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--input-list", default=None)
    ap.add_argument("--split", default="val", choices=("train", "val", "test"))
    ap.add_argument("--max-events", type=int, default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available()
                    else "cpu")
    ap.add_argument("--class-prob-threshold", type=float, default=0.0)
    ap.add_argument("--dedup-iou-threshold", type=float, default=0.6)
    ap.add_argument("--deghost-threshold-val", type=float, default=None)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--deterministic", action="store_true")
    # Keypoint decode knobs.
    ap.add_argument("--coord-scale", type=float, default=179.55)
    ap.add_argument("--coord-center", type=float, nargs=3,
                    default=[125.0, 0.0, 518.0])
    ap.add_argument("--kp-class-prob-floor", type=float, default=0.3)
    ap.add_argument("--kp-end-prob-floor", type=float, default=0.5)
    ap.add_argument("--kp-score-thresh", type=float, default=0.3)
    ap.add_argument("--kp-cluster-radius-cm", type=float, default=3.0)
    args = ap.parse_args()
    args.coord_center = np.asarray(args.coord_center, dtype=np.float32)

    if args.deterministic:
        set_deterministic()

    from pointcept.utils.config import Config
    from pointcept.datasets import build_dataset, larformer_collate
    from pointcept.models.builder import build_model
    import pointcept.models    # noqa: F401
    import pointcept.datasets   # noqa: F401

    cfg = Config.fromfile(args.config)
    if cfg.model.get("type") != "CascadedParticleSegmenter":
        raise ValueError(
            f"--config must be a CascadedParticleSegmenter; got "
            f"{cfg.model.get('type')!r}")

    # ---- inject the keypoint head(s) into the particle segmenter ----
    _ps = cfg.model["particle_segmenter"]
    _ps["enable_keypoint_head"] = True
    # Integrated dense head (nu vertex) — built so the keypoint-trained
    # particle weights' dense head loads + runs. Disable for checkpoints that
    # predate the dense head (else its weights are random → bogus vertex), and
    # use --keypoint-dense-* for a separately-trained dense model instead.
    # NOTE: the injected dense cfg must match the trained checkpoint's
    # (matches larformer-keypoint-query-v1.py defaults).
    if not args.no_integrated_dense:
        _ps["enable_keypoint_dense_head"] = True
        _ps.setdefault("keypoint_dense_cfg", dict(
            n_keypoint_types=6, head_hidden_dim=256, enable_offset_head=True,
            pos_weight=50.0, weight_score=1.0, weight_offset=1.0,
            coord_scale=args.coord_scale))

    ds_cfg = dict(cfg.data[args.split])
    if args.input_list is not None:
        ds_cfg["data_list_file"] = os.path.abspath(args.input_list)
    ds_cfg["max_spacepoints"] = None
    # Inference-only: keypoints are decoded from PREDICTIONS, no GT needed.
    # gt_source="deghost" yields empty gt_instances so NO stage runs its
    # eval-with-GT Hungarian matcher — critical because the slicer is a
    # 3-class model and particle-level GT (class ids 0..7) would overflow its
    # CE target and CUDA-assert (the same reason CascadedParticleSegmenter
    # strips GT before its inner slicer).
    ds_cfg["gt_source"] = "deghost"
    dataset = build_dataset(ds_cfg)
    n_events = len(dataset)
    if args.max_events is not None:
        n_events = min(n_events, args.max_events)
    print(f"[fc-kp] dataset: {len(dataset)} events; processing {n_events}")

    print(f"[fc-kp] building cascade from {args.config}")
    model = build_model(cfg.model).to(args.device).eval()
    if args.weights:
        print(f"[fc-kp] loading weights {args.weights}")
        _load_weights_into(model, args.weights, allow_partial=True)
    else:
        print("[fc-kp] no --weights; cascade self-loaded slicer/backbone from "
              "config knobs at build time")
    if args.particle_keypoint_weights:
        print(f"[fc-kp] overlaying particle keypoint weights "
              f"{args.particle_keypoint_weights}")
        _load_weights_into(_resolve_particle_segmenter(model),
                           args.particle_keypoint_weights, allow_partial=True)

    _ps_inner = _resolve_particle_segmenter(model)
    _has_integrated_dense = bool(getattr(_ps_inner, "enable_keypoint_dense_head",
                                         False))
    if _has_integrated_dense:
        print("[fc-kp] nu vertex from the particle segmenter's INTEGRATED "
              "dense head")

    # ---- optional SEPARATE dense (vertex) model (fallback) ----
    dense_model = None
    if args.keypoint_dense_config and args.keypoint_dense_weights:
        dcfg = Config.fromfile(args.keypoint_dense_config)
        dmodel_cfg = copy.deepcopy(dcfg.model)
        dmodel_cfg.pop("backbone_weight", None)
        dense_model = build_model(dmodel_cfg).to(args.device).eval()
        _load_weights_into(dense_model, args.keypoint_dense_weights,
                           allow_partial=True)
        print(f"[fc-kp] separate dense vertex model loaded "
              f"({args.keypoint_dense_config}) — used only if no integrated head")
    elif not _has_integrated_dense:
        print("[fc-kp] no dense head — nu vertex will be absent "
              "(query start/end only)")

    # Disable train-time order shuffle for per-event reproducibility.
    for m in model.modules():
        if getattr(m, "shuffle_orders", False):
            m.shuffle_orders = False

    cascaded_slicer = _resolve_cascaded_slicer(model)
    ps_inner = _resolve_particle_segmenter(model)
    inner = getattr(model, "module", model)
    n_classes = int(ps_inner.num_classes)
    slicer_no_object = int(getattr(cascaded_slicer, "slicer",
                                   cascaded_slicer).loss_fn.no_object_class_id)
    ps_no_object = int(ps_inner.loss_fn.no_object_class_id)
    nu_class_id = int(inner.nu_class_id)
    mask_prob_threshold = float(inner.mask_prob_threshold)
    spacepoint_level = str(inner.spacepoint_level)
    recenter = bool(inner.recenter_to_slice_centroid)
    if args.deghost_threshold_val is not None:
        cascaded_slicer.deghost_threshold_val = float(args.deghost_threshold_val)

    os.makedirs(args.output_dir, exist_ok=True)
    n_done = n_drop = 0
    for ev_idx in range(n_events):
        sample = dataset[ev_idx]
        in_name = sample.get("name", f"event{ev_idx:06d}.h5")
        stem = os.path.splitext(in_name)[0]
        out_path = os.path.join(args.output_dir, f"keypointpred_{stem}.h5")
        if os.path.exists(out_path) and not args.overwrite:
            print(f"[{ev_idx+1}/{n_events}] skip-existing {stem}")
            continue
        if args.device == "cuda":
            torch.cuda.empty_cache()

        if args.deterministic:
            reseed_per_event()   # per-event order invariance (see helper)
        batched = _move_batch(larformer_collate([sample]), args.device)

        # ---- Stage 1+2 ----
        with torch.no_grad():
            slicer_out = cascaded_slicer(batched)
        slicer_event = slicer_predict_event_from_out(
            slicer_out, sample, slicer_no_object)
        slicer_preds = slicer_out.get("predictions", [])
        filtered_batch = slicer_out.get("filtered_batch", None)
        if (slicer_event.get("event_dropped") or filtered_batch is None
                or slicer_predictions_empty(slicer_preds)):
            n_drop += 1
            slicer_event["stage3_meta/event_dropped"] = int(True)
            write_event_h5(out_path, slicer_event)
            continue

        # ---- Stage 2 → 3 boundary ----
        keep = build_nu_keep_mask(
            slicer_predictions=slicer_preds,
            n_sp_per_event=filtered_batch["n_spacepoints"],
            nu_class_id=nu_class_id, mask_prob_threshold=mask_prob_threshold,
            spacepoint_level=spacepoint_level, device=args.device)
        if int(keep.sum().item()) == 0:
            n_drop += 1
            slicer_event["stage3_meta/event_dropped"] = int(True)
            write_event_h5(out_path, slicer_event)
            continue

        ps_batch = filter_batch_for_particle_segmenter(
            filtered_batch=filtered_batch, keep_mask=keep,
            recenter=recenter, drop_empty_instances=True)
        if int((ps_batch["n_spacepoints"] == 0).any().item()):
            ps_batch = drop_empty_events(ps_batch)
        if (ps_batch["n_spacepoints"].numel() == 0
                or int(ps_batch["n_spacepoints"].sum().item()) == 0):
            n_drop += 1
            slicer_event["stage3_meta/event_dropped"] = int(True)
            write_event_h5(out_path, slicer_event)
            continue

        # ---- Stage 3 (particle + keypoint heads) ----
        # Pure inference: strip GT so the particle segmenter does NOT run its
        # eval-with-GT Hungarian matcher (we only need predictions —
        # class/mask/origin/keypoints — for decoding; matching is for training
        # diagnostics only and is fragile on raw-cascade GT).
        ps_batch_infer = {k: v for k, v in ps_batch.items()
                          if k not in ("gt_instances_per_event",
                                       "n_gt_instances")}
        with torch.no_grad():
            ps_out = ps_inner(ps_batch_infer)
        ev_pred = ps_out["predictions"][0]

        ps_sample = _unpack_ps_sample(ps_batch)
        for kk in ("run", "subrun", "event", "name"):
            ps_sample[kk] = sample.get(kk, -1 if kk != "name" else "")
        stage3_event = stage3_predict_event_from_out(
            ps_out, ps_sample, ps_no_object,
            class_prob_threshold=args.class_prob_threshold,
            coord_scale=args.coord_scale,
            dedup_iou_threshold=args.dedup_iou_threshold)

        # ---- Keypoints ----
        # Prefer the integrated dense head (already in ev_pred — no extra
        # forward). Only run the SEPARATE dense model when the integrated head
        # didn't fire.
        dense_pred = None
        if dense_model is not None and "dense_kpscores" not in ev_pred:
            # Dense head trained on NON-recentered coords → run on a
            # non-recentered copy of the same nu-slice.
            ps_batch_dense = filter_batch_for_particle_segmenter(
                filtered_batch=filtered_batch, keep_mask=keep,
                recenter=False, drop_empty_instances=True)
            if int((ps_batch_dense["n_spacepoints"] == 0).any().item()):
                ps_batch_dense = drop_empty_events(ps_batch_dense)
            with torch.no_grad():
                dense_out = dense_model(ps_batch_dense)
            dpreds = dense_out.get("predictions", [])
            if dpreds:
                dense_pred = dpreds[0]
        kp_data = decode_keypoints_for_event(
            ev_pred, ps_batch, dense_pred, n_classes, args)

        merged = dict(slicer_event)
        merged.update(stage3_event)
        merged.update(kp_data)
        write_event_h5(out_path, merged)
        n_done += 1
        n_kp = kp_data.get("keypoints/n", 0)
        has_v = "keypoints/nu_vertex_cm" in kp_data
        print(f"[{ev_idx+1}/{n_events}] {stem:48s}  keypoints={n_kp}  "
              f"nu_vertex={'yes' if has_v else 'no'}")

    print(f"\n[fc-kp] done: wrote {n_done}, dropped {n_drop} → {args.output_dir}")


if __name__ == "__main__":
    main()
