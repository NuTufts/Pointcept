"""Tier-A reproducibility harness: capture the cascade's CONTINUOUS decision
tensors (and the discrete decisions they drive) per event, so the same event
captured on two different GPUs can be diffed stage-by-stage.

NO model code is modified. This script re-runs the exact full-cascade forward
(reusing run_larformer_stage3_inference's helpers and the cascade modules) and,
instead of writing a stage3pred H5, serializes the decision tensors to one .npz
per event.

What is captured per event (see the decision chain in docs/LArFormer_Reproducibility.md):

  Stage 1 (deghoster) — input is IDENTICAL across GPUs (deterministic loader),
    so this is the CLEAN, un-confounded measurement of pure GPU divergence:
      s1_pos    (N_in, 3)  pre-deghost coords (cm)   [identical across GPUs]
      s1_p_real (N_in,)    per-SP P(real)            [diverges across GPUs]
      s1_keep   (N_in,)    bool, = p_real > tau      [first hard cut]
      tau                  deghost threshold (scalar)
  Stage 2 (slicer)  — runs on post-deghost points (set differs across GPUs):
      s2_pos        (M, 3) post-deghost coords
      s2_pred_query (M,)   per-SP predicted slice (panoptic argmax)
      s2_pred_class (M,)   per-SP predicted slicer class
  Stage 3 (segmenter) — runs on the nu-candidate slice:
      s3_pos      (P, 3)   coords
      s3_class    (P,)     per-SP particle class (argmax + no_object floor)
      s3_maskprob (P,)     per-SP mask prob (for margin analysis)
  plus identity (run/subrun/event/name) and a `dropped` flag.

The Stage-1 block aligns by index (same input order both GPUs); Stages 2/3 align
by coordinate (cross_gpu_diff.py). Run twice on two GPU SKUs, then diff.

  python capture_cascade_tensors.py --config <fullcascade.py> --weights <ckpt> \
      --input-list <list.txt> --capture-dir <out> [--max-events N] [--no-deterministic]
"""
import argparse
import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", "..", "..", ".."))  # pointcept repo root
sys.path.insert(0, _REPO)                          # for pointcept.*
sys.path.insert(0, os.path.join(_REPO, "tools"))   # for run_larformer_stage3_inference

# Reuse the EXACT pipeline pieces (no duplication of model logic).
from run_larformer_stage3_inference import (   # noqa: E402
    set_deterministic, _move_batch, _resolve_cascaded_slicer,
    _resolve_particle_segmenter, _unpack_ps_sample)
from pointcept.models.LArFormer.inference import (   # noqa: E402
    slicer_predict_event_from_out, stage3_predict_event_from_out)
from pointcept.models.LArFormer.cascade_particle_filter import (   # noqa: E402
    build_nu_keep_mask, filter_batch_for_particle_segmenter,
    slicer_predictions_empty)
from pointcept.models.LArFormer.cascade_filter import drop_empty_events  # noqa: E402


def _np(x):
    return x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--input-list", required=True)
    ap.add_argument("--capture-dir", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--max-events", type=int, default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--class-prob-threshold", type=float, default=0.3)
    ap.add_argument("--deghost-threshold-val", type=float, default=None,
                    help="override Stage-1 deghoster keep threshold τ")
    ap.add_argument("--deghost-fp64", action="store_true",
                    help="run the deghoster forward in float64 (reproducibility)")
    ap.add_argument("--deghost-no-shuffle", action="store_true",
                    help="disable deghoster serialization order-shuffle (the "
                         "membership/list-dependence root cause)")
    ap.add_argument("--no-deterministic", action="store_true",
                    help="capture in DEFAULT (non-deterministic) mode — use to "
                         "measure same-GPU run-to-run instead of cross-GPU.")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    if not args.no_deterministic:
        set_deterministic()

    from pointcept.utils.config import Config
    from pointcept.datasets import build_dataset, larformer_collate
    from pointcept.models.builder import build_model
    import pointcept.models    # noqa: F401
    import pointcept.datasets   # noqa: F401

    cfg = Config.fromfile(args.config)
    ds_cfg = dict(cfg.data[args.split])
    ds_cfg["data_list_file"] = os.path.abspath(args.input_list)
    ds_cfg["max_spacepoints"] = None             # same as production full-cascade path
    dataset = build_dataset(ds_cfg)
    n_events = len(dataset) if args.max_events is None else min(args.max_events, len(dataset))

    sd = torch.load(args.weights, map_location="cpu")
    sd = sd.get("state_dict", sd)
    model = build_model(cfg.model)
    model.load_state_dict(sd, strict=False)
    model = model.to(args.device).eval()

    cascaded_slicer = _resolve_cascaded_slicer(model)
    ps_inner = _resolve_particle_segmenter(model)
    inner = getattr(model, "module", model)
    if args.deghost_threshold_val is not None:
        cascaded_slicer.deghost_threshold_val = float(args.deghost_threshold_val)
        print(f"[capture] deghost_threshold_val -> {cascaded_slicer.deghost_threshold_val}")
    if args.deghost_no_shuffle:
        cascaded_slicer.disable_deghost_order_shuffle()
    if args.deghost_fp64:
        cascaded_slicer.enable_fp64_deghoster()
    slicer_inner = getattr(cascaded_slicer, "slicer", cascaded_slicer)
    slicer_no_object = int(slicer_inner.loss_fn.no_object_class_id)
    ps_no_object = int(ps_inner.loss_fn.no_object_class_id)
    coord_scale = float(ds_cfg.get("coord_scale", 179.55))
    nu_class_id = int(inner.nu_class_id)
    mask_prob_threshold = float(inner.mask_prob_threshold)
    spacepoint_level = str(inner.spacepoint_level)
    recenter = bool(inner.recenter_to_slice_centroid)

    os.makedirs(args.capture_dir, exist_ok=True)
    print(f"[capture] {n_events} events -> {args.capture_dir}  "
          f"(deterministic={not args.no_deterministic}, "
          f"GPU={torch.cuda.get_device_name(0) if args.device=='cuda' else 'cpu'})")

    for ev_idx in range(n_events):
        sample = dataset[ev_idx]
        stem = os.path.splitext(sample.get("name", f"event{ev_idx:06d}.h5"))[0]
        out_path = os.path.join(args.capture_dir, f"capture_{stem}.npz")
        if os.path.exists(out_path) and not args.overwrite:
            continue

        cap = dict(run=int(sample.get("run", -1)),
                   subrun=int(sample.get("subrun", -1)),
                   event=int(sample.get("event", -1)),
                   name=str(sample.get("name", "")),
                   ev_idx=int(ev_idx),
                   s1_pos=_np(sample["coord"]).astype(np.float32))  # pre-deghost coords

        if args.device == "cuda":
            torch.cuda.empty_cache()
        batched = _move_batch(larformer_collate([sample]), args.device)

        try:
            with torch.no_grad():
                slicer_out = cascaded_slicer(batched)
        except RuntimeError as ex:
            if "out of memory" not in str(ex).lower():
                raise
            torch.cuda.empty_cache()
            print(f"[{ev_idx+1}/{n_events}] OOM (stage1/2) — skipped")
            continue

        # ---- Stage 1: per-SP P(real) (aligned to s1_pos by index) ----
        p_real = slicer_out.get("deghost_p_real", None)
        tau = float(slicer_out.get("deghost_tau", float("nan")))
        cap["tau"] = np.float32(tau)
        if p_real is not None:
            pr = _np(p_real).astype(np.float32)
            cap["s1_p_real"] = pr
            cap["s1_keep"] = (pr > tau)
        # sanity: p_real length should match the captured input coords
        if p_real is not None and len(cap["s1_p_real"]) != len(cap["s1_pos"]):
            cap["s1_align_warn"] = np.int64(len(cap["s1_pos"]) - len(cap["s1_p_real"]))

        # ---- Stage 2: slicer per-SP slice assignment ----
        slicer_event = slicer_predict_event_from_out(
            slicer_out, sample, slicer_no_object, slicer_dedup_iou=0.0)
        dropped = bool(slicer_event.get("event_dropped"))
        if "post/coord" in slicer_event:
            cap["s2_pos"] = _np(slicer_event["post/coord"]).astype(np.float32)
            cap["s2_pred_query"] = _np(slicer_event["post/pred_query"]).astype(np.int64)
            cap["s2_pred_class"] = _np(slicer_event["post/pred_class"]).astype(np.int64)

        slicer_predictions = slicer_out.get("predictions", [])
        filtered_batch = slicer_out.get("filtered_batch", None)
        if dropped or filtered_batch is None or slicer_predictions_empty(slicer_predictions):
            cap["dropped"] = np.int64(1)
            np.savez_compressed(out_path, **cap)
            print(f"[{ev_idx+1}/{n_events}] {stem}  dropped@slicer  N_in={len(cap['s1_pos'])}")
            continue

        # ---- Stage 2->3 boundary: nu keep mask ----
        keep = build_nu_keep_mask(
            slicer_predictions=slicer_predictions,
            n_sp_per_event=filtered_batch["n_spacepoints"],
            nu_class_id=nu_class_id, mask_prob_threshold=mask_prob_threshold,
            spacepoint_level=spacepoint_level, device=args.device)
        cap["s2_nu_keep"] = _np(keep).astype(bool)
        if int(keep.sum().item()) == 0:
            cap["dropped"] = np.int64(1)
            np.savez_compressed(out_path, **cap)
            print(f"[{ev_idx+1}/{n_events}] {stem}  dropped@nukeep")
            continue

        ps_batch = filter_batch_for_particle_segmenter(
            filtered_batch=filtered_batch, keep_mask=keep,
            recenter=recenter, drop_empty_instances=True)
        if int((ps_batch["n_spacepoints"] == 0).any().item()):
            ps_batch = drop_empty_events(ps_batch)
        if (ps_batch["n_spacepoints"].numel() == 0
                or int(ps_batch["n_spacepoints"].sum().item()) == 0):
            cap["dropped"] = np.int64(1)
            np.savez_compressed(out_path, **cap)
            continue

        # ---- Stage 3 forward ----
        try:
            with torch.no_grad():
                ps_out = ps_inner(ps_batch)
        except RuntimeError as ex:
            if "out of memory" not in str(ex).lower():
                raise
            torch.cuda.empty_cache()
            print(f"[{ev_idx+1}/{n_events}] OOM (stage3) — skipped")
            continue
        ps_sample = _unpack_ps_sample(ps_batch)
        stage3_event = stage3_predict_event_from_out(
            ps_out, ps_sample, ps_no_object,
            class_prob_threshold=args.class_prob_threshold,
            coord_scale=coord_scale, dedup_iou_threshold=0.6)
        if "stage3/coord" in stage3_event:
            cap["s3_pos"] = _np(stage3_event["stage3/coord"]).astype(np.float32)
            cap["s3_class"] = _np(stage3_event["stage3/pred_class"]).astype(np.int64)
            if "stage3/pred_mask_prob" in stage3_event:
                cap["s3_maskprob"] = _np(stage3_event["stage3/pred_mask_prob"]).astype(np.float32)
        cap["dropped"] = np.int64(0)
        np.savez_compressed(out_path, **cap)
        print(f"[{ev_idx+1}/{n_events}] {stem}  N_in={len(cap['s1_pos'])} "
              f"N_post={cap.get('s2_pos', np.zeros((0,3))).shape[0]} "
              f"N_s3={cap.get('s3_pos', np.zeros((0,3))).shape[0]}")

    print(f"[capture] done -> {args.capture_dir}")


if __name__ == "__main__":
    main()
