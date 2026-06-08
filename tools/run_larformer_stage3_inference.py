"""Inference + per-event diagnostics dump for the LArFormer Stage-3
particle segmenter.

Thin CLI around the inference helpers in
`pointcept/models/LArFormer/inference.py`. Two input modes:

  --input-mode cached  (default — the immediate use case):

    Reads Stage-1+2 cache events (the per-event H5 files produced by
    `tools/build_stage12_cache_shard.py` and augmented with
    `entry_0/particle_class_id` by
    `tools/augment_stage12_cache_particle_class_id.py`). The config
    must be one whose `model = dict(type="LArFormer", ...)` describes
    the particle segmenter (e.g.
    `configs/lartpc/larformer-particle-v1-cached-ptv3crosslevel.py`).
    For each event, runs the particle segmenter forward, writes a
    `stage3pred_<basename>.h5` with `stage3*/...` keys.

  --input-mode full-cascade:

    Reads raw merged_h5 events via a standard `LArFormerDataset`. The
    config must declare a `CascadedParticleSegmenter` model (e.g.
    `configs/lartpc/larformer-particle-v1.py`) so we can find both the
    Stage-1+2 (`cascaded_slicer`) and Stage-3 (`particle_segmenter`)
    sub-modules. For each event, runs the slicer with GT to produce a
    full slicerpred-shaped output at the top of the file, then applies
    the Stage-2 → Stage-3 boundary (build_nu_keep_mask +
    filter_batch_for_particle_segmenter) and runs the particle
    segmenter, appending `stage3*/...` keys.

The output file format is identical in both modes — the cached mode
just omits the slicer keys. So the existing slicer visualizer
(`tools/visualize_larformer_gt.py`) can already render the Stage-2
half of a full-cascade stage3pred file using its `--slicerpred-dir`
flag, and the Stage-3 visualizer (`visualize_stage3_larformer_from_cached.py`)
can render the Stage-3 half using its `--stage3pred-dir` flag.

Output file naming: `stage3pred_<input_basename_without_ext>.h5`.

Usage (cached mode):
    ./run_in_container.sh python tools/run_larformer_stage3_inference.py \\
        --config configs/lartpc/larformer-particle-v1-cached-ptv3crosslevel.py \\
        --weights exp/.../model/model_last.pth \\
        --cache-dir exp/cache_stage12_ptv3crosslevelslicer_iter_75750/val \\
        --output-dir exp/.../inference \\
        --class-prob-threshold 0.3

Usage (full-cascade mode):
    ./run_in_container.sh python tools/run_larformer_stage3_inference.py \\
        --input-mode full-cascade \\
        --config configs/lartpc/larformer-particle-v1.py \\
        --weights exp/.../model/model_last.pth \\
        --input-list inputlists/run3b_ncpi0.txt \\
        --output-dir exp/.../inference \\
        --class-prob-threshold 0.3
"""
import argparse
import copy
import os
import sys

import numpy as np
import torch


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

from pointcept.models.LArFormer.inference import (   # noqa: E402
    slicer_predict_event_from_out,
    stage3_predict_event,
    stage3_predict_event_from_out,
    write_event_h5,
)
from pointcept.models.LArFormer.cascade_particle_filter import (  # noqa: E402
    build_nu_keep_mask,
    filter_batch_for_particle_segmenter,
    slicer_predictions_empty,
)
from pointcept.models.LArFormer.cascade_filter import drop_empty_events  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_weights_into(model: torch.nn.Module, weights_path: str,
                       allow_partial: bool = True) -> None:
    """Load a checkpoint into a model with the usual prefix-stripping
    + (optional) strict=False tolerance."""
    ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
    sd = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    sd = {(k[7:] if k.startswith("module.") else k): v
          for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=not allow_partial)
    print(f"  loaded {weights_path}")
    print(f"    missing={len(missing)}  unexpected={len(unexpected)}")
    if missing[:5]:
        print(f"    first missing: {missing[:5]}")
    if unexpected[:5]:
        print(f"    first unexpected: {unexpected[:5]}")


def _move_batch(batch: dict, device: torch.device) -> dict:
    """Move per-SP tensors + nested lists of dicts/tensors onto device."""
    out = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device, non_blocking=True)
        elif isinstance(v, list) and v and isinstance(v[0], list):
            new_b = []
            for per_event in v:
                new_e = []
                for item in per_event:
                    if isinstance(item, dict):
                        new_e.append({kk: (vv.to(device, non_blocking=True)
                                           if isinstance(vv, torch.Tensor)
                                           else vv)
                                      for kk, vv in item.items()})
                    elif isinstance(item, torch.Tensor):
                        new_e.append(item.to(device, non_blocking=True))
                    else:
                        new_e.append(item)
                new_b.append(new_e)
            out[k] = new_b
        elif isinstance(v, list) and v and isinstance(v[0], torch.Tensor):
            out[k] = [vv.to(device, non_blocking=True) for vv in v]
        else:
            out[k] = v
    return out


def _resolve_particle_segmenter(top_model):
    """Given a built model, return the inner LArFormer that does the
    particle segmentation.

      - plain `LArFormer` config → top_model itself.
      - `CascadedParticleSegmenter` wrapper → top_model.particle_segmenter.
      - DDP-wrapped → reach through `.module`.
    """
    inner = getattr(top_model, "module", top_model)
    if hasattr(inner, "particle_segmenter"):
        return inner.particle_segmenter
    return inner


def _resolve_cascaded_slicer(top_model):
    """Given a built CascadedParticleSegmenter, return the inner
    CascadedSlicer."""
    inner = getattr(top_model, "module", top_model)
    if not hasattr(inner, "cascaded_slicer"):
        return None
    return inner.cascaded_slicer


def _unpack_ps_sample(ps_batch: dict) -> dict:
    """Repackage a 1-event slice of `filter_batch_for_particle_segmenter`'s
    output so it looks like a `LArFormerStage12CacheDataset` __getitem__
    return — gives `stage3_predict_event_from_out` the keys it expects."""
    n_sp = int(ps_batch["n_spacepoints"][0].item()) \
        if ps_batch["n_spacepoints"].numel() > 0 else 0

    def _take(key, default):
        v = ps_batch.get(key)
        if v is None:
            return default
        return v[:n_sp].detach().cpu().numpy()

    sample = {
        "coord":      _take("coord",      np.zeros((0, 3), dtype=np.float32)),
        "coord_norm": _take("coord_norm", np.zeros((0, 3), dtype=np.float32)),
        "feat":       _take("feat",       np.zeros((0, 6), dtype=np.float32)),
        "lm_score":   _take("lm_score",   np.zeros(0, dtype=np.float32)),
        "wire":       _take("wire",       np.zeros((0, 3), dtype=np.float32)),
        "trackid":    _take("trackid",    np.zeros(0, dtype=np.int64)),
        "pid":        _take("pid",        np.zeros(0, dtype=np.int64)),
        "origin_label": _take("origin_label", np.zeros(0, dtype=np.int64)),
        "hasmatch":   _take("hasmatch",   np.zeros(0, dtype=np.int64)),
        "ssnet_label": _take("ssnet_label", np.zeros(0, dtype=np.int64)),
        "slice_id":   _take("slice_id",   np.zeros(0, dtype=np.int64)),
        "n_spacepoints": n_sp,
    }
    # gt_instances for the post-Stage-2 batch
    gt_ev = ps_batch.get("gt_instances_per_event")
    if gt_ev:
        gts = gt_ev[0]
        sample["gt_instances"] = [
            {kk: (vv.detach().cpu().numpy()
                  if isinstance(vv, torch.Tensor) else vv)
             for kk, vv in g.items()}
            for g in gts
        ]
    else:
        sample["gt_instances"] = []
    # particle_class_id is the augmented field — in full-cascade mode
    # it isn't carried by the upstream slicer batch, so default to all
    # -1 (= ignore for the cls aux supervision; eval-time diagnostics
    # only).
    if "particle_class_id" in ps_batch:
        sample["particle_class_id"] = ps_batch["particle_class_id"][:n_sp].detach().cpu().numpy()
    else:
        sample["particle_class_id"] = -np.ones(n_sp, dtype=np.int64)
    return sample


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------

def run_cached_mode(args):
    """Iterate Stage-1+2 cache events, run particle segmenter, write
    stage3pred files."""
    from pointcept.utils.config import Config
    from pointcept.datasets import build_dataset, larformer_collate
    from pointcept.models.builder import build_model
    import pointcept.models    # noqa: F401
    import pointcept.datasets  # noqa: F401

    cfg = Config.fromfile(args.config)
    ds_cfg = dict(cfg.data[args.split])
    if args.cache_dir is not None:
        ds_cfg["data_root"] = os.path.abspath(args.cache_dir)
        ds_cfg.pop("data_list_file", None)
    dataset = build_dataset(ds_cfg)
    n_events = len(dataset)
    if args.max_events is not None:
        n_events = min(n_events, args.max_events)
    print(f"[infer] cache dataset: {len(dataset)} events; will process {n_events}.")

    print(f"[infer] Building model from {args.config}")
    if cfg.model.get("type") == "CascadedParticleSegmenter":
        ps_cfg = copy.deepcopy(cfg.model["particle_segmenter"])
        # Don't try to load a checkpoint for the wrapper's slicer when
        # we only need the particle_segmenter — strip its weight knobs.
        ps_cfg.pop("backbone_weight", None)
        model_cfg = ps_cfg
        print(f"  --- detected CascadedParticleSegmenter wrapper; "
              f"using its inner particle_segmenter")
    else:
        model_cfg = cfg.model
    model = build_model(model_cfg).to(args.device).eval()

    print(f"[infer] Loading weights from {args.weights}")
    _load_weights_into(model, args.weights, allow_partial=True)

    ps_inner = _resolve_particle_segmenter(model)
    no_object_class_id = int(ps_inner.loss_fn.no_object_class_id)
    coord_scale = float(ds_cfg.get("coord_scale", 179.55))
    print(f"[infer] no_object_class_id={no_object_class_id}  "
          f"class_prob_threshold={args.class_prob_threshold}  "
          f"coord_scale={coord_scale}")

    os.makedirs(args.output_dir, exist_ok=True)
    n_dropped = 0
    n_skipped = 0
    for ev_idx in range(n_events):
        try:
            sample = dataset[ev_idx]
        except ValueError as e:
            # Cache reader raises when source_set_filter zeroes the event.
            print(f"[{ev_idx+1:4d}/{n_events}] SKIP (empty after "
                  f"source_set_filter): {e}")
            n_skipped += 1
            continue

        # Derive a unique-per-event stem from the cache filename — the
        # cache builder's `__event<idx>` suffix is what guarantees
        # uniqueness when multiple events share the same source merged_h5.
        cache_path = dataset.data_list[ev_idx % len(dataset.data_list)]
        stem = os.path.splitext(os.path.basename(cache_path))[0]
        out_path = os.path.join(args.output_dir, f"stage3pred_{stem}.h5")
        if os.path.exists(out_path) and not args.overwrite:
            print(f"[{ev_idx+1:4d}/{n_events}] skip-existing: "
                  f"{os.path.basename(out_path)}")
            continue

        batched = larformer_collate([sample])
        batched = _move_batch(batched, args.device)

        event_data = stage3_predict_event(
            model, sample, batched, no_object_class_id,
            class_prob_threshold=args.class_prob_threshold,
            coord_scale=coord_scale,
        )
        if event_data.get("stage3_meta/event_dropped"):
            n_dropped += 1
        write_event_h5(out_path, event_data)

        n_gt = event_data.get("stage3_meta/n_gt_instances", 0)
        n_matched = event_data.get("stage3_meta/n_matched", 0)
        n_sp = event_data.get("stage3_meta/n_stage3_sp", 0)
        pair_iou = event_data.get("stage3_gt/pair_iou", np.zeros(0))
        valid_iou = pair_iou[pair_iou >= 0] if pair_iou.size else pair_iou
        mean_iou = float(np.mean(valid_iou)) if valid_iou.size > 0 else float("nan")
        l2_arr = event_data.get("stage3_gt/pair_origin_l2_cm", np.zeros(0))
        valid_l2 = l2_arr[l2_arr >= 0] if l2_arr.size else l2_arr
        mean_l2 = float(np.mean(valid_l2)) if valid_l2.size > 0 else float("nan")
        print(
            f"[{ev_idx+1:4d}/{n_events}] {stem:50s}  "
            f"n_sp={n_sp}  matched={n_matched}/{n_gt}  "
            f"mean_pair_IoU={mean_iou:.3f}  mean_origin_L2={mean_l2:.1f}cm"
        )

    print(f"\n[infer] Done. Wrote stage3pred files to {args.output_dir}  "
          f"(processed {n_events - n_skipped}, "
          f"dropped {n_dropped}, skipped {n_skipped} empty events)")


def run_full_cascade_mode(args):
    """Iterate raw merged_h5 events via LArFormerDataset, run the full
    cascade (slicer + particle segmenter), write stage3pred files
    containing both the slicer's slicerpred-shaped output AND the
    Stage-3 keys."""
    from pointcept.utils.config import Config
    from pointcept.datasets import build_dataset, larformer_collate
    from pointcept.models.builder import build_model
    import pointcept.models    # noqa: F401
    import pointcept.datasets  # noqa: F401

    cfg = Config.fromfile(args.config)
    if cfg.model.get("type") != "CascadedParticleSegmenter":
        raise ValueError(
            f"--input-mode full-cascade requires a "
            f"CascadedParticleSegmenter config; got "
            f"cfg.model.type={cfg.model.get('type')!r}. Try "
            f"`configs/lartpc/larformer-particle-v1.py`."
        )

    # Force gt_source="particle" — the full-cascade inference is
    # diagnostic, so we want particle-level GT to land in eval_loss
    # when the dataset emits it.
    ds_cfg = dict(cfg.data[args.split])
    if args.input_list is not None:
        ds_cfg["data_list_file"] = os.path.abspath(args.input_list)
    ds_cfg["max_spacepoints"] = None
    if not args.no_gt:
        ds_cfg["gt_source"] = "particle"
    dataset = build_dataset(ds_cfg)
    n_events = len(dataset)
    if args.max_events is not None:
        n_events = min(n_events, args.max_events)
    print(f"[infer] cascade dataset: {len(dataset)} events; will process {n_events}.")

    print(f"[infer] Building model from {args.config}")
    # Build the WHOLE CascadedParticleSegmenter — needed so we can run
    # the slicer with GT for slicerpred-shaped output, then continue to
    # particle segmenter via the cascade boundary helpers.
    model = build_model(cfg.model).to(args.device).eval()
    print(f"[infer] Loading weights from {args.weights}")
    _load_weights_into(model, args.weights, allow_partial=True)

    cascaded_slicer = _resolve_cascaded_slicer(model)
    ps_inner = _resolve_particle_segmenter(model)
    inner = getattr(model, "module", model)
    if cascaded_slicer is None:
        raise RuntimeError("could not find cascaded_slicer on built model")
    slicer_inner = getattr(cascaded_slicer, "slicer", cascaded_slicer)

    slicer_no_object = int(slicer_inner.loss_fn.no_object_class_id)
    ps_no_object = int(ps_inner.loss_fn.no_object_class_id)
    coord_scale = float(ds_cfg.get("coord_scale", 179.55))
    nu_class_id = int(inner.nu_class_id)
    mask_prob_threshold = float(inner.mask_prob_threshold)
    spacepoint_level = str(inner.spacepoint_level)
    recenter = bool(inner.recenter_to_slice_centroid)
    print(f"[infer] slicer no_object={slicer_no_object}  "
          f"ps no_object={ps_no_object}  "
          f"nu_class_id={nu_class_id}  "
          f"τ_loose={mask_prob_threshold}  recenter={recenter}")

    os.makedirs(args.output_dir, exist_ok=True)
    n_dropped_slicer = 0
    n_dropped_stage3 = 0
    for ev_idx in range(n_events):
        sample = dataset[ev_idx]
        in_name = sample.get("name", f"event{ev_idx:06d}.h5")
        stem = os.path.splitext(in_name)[0]
        out_path = os.path.join(args.output_dir, f"stage3pred_{stem}.h5")
        if os.path.exists(out_path) and not args.overwrite:
            print(f"[{ev_idx+1:4d}/{n_events}] skip-existing: "
                  f"{os.path.basename(out_path)}")
            continue

        batched = larformer_collate([sample])
        batched = _move_batch(batched, args.device)

        # ---- Stage 1+2 forward (WITH gt for matching diagnostics) ----
        with torch.no_grad():
            slicer_out = cascaded_slicer(batched)
        slicer_event_data = slicer_predict_event_from_out(
            slicer_out, sample, slicer_no_object,
        )
        if slicer_event_data.get("event_dropped"):
            n_dropped_slicer += 1
            # Still write what we have (slicer half + an empty stage3
            # half marked as dropped) so downstream tools see a
            # consistent file shape.
            slicer_event_data.update({
                "stage3_meta/event_dropped": int(True),
                "stage3_meta/n_stage3_sp": 0,
                "stage3_meta/n_matched": 0,
                "stage3_meta/n_gt_instances": 0,
                "stage3_meta/no_object_class_id": int(ps_no_object),
                "stage3_meta/class_prob_threshold":
                    float(args.class_prob_threshold),
                "stage3_meta/coord_scale": float(coord_scale),
                "stage3_meta/has_gt": int(False),
            })
            write_event_h5(out_path, slicer_event_data)
            print(f"[{ev_idx+1:4d}/{n_events}] {in_name:50s}  "
                  f"slicer DROPPED — wrote slicer half only")
            continue

        # ---- Stage 2 → Stage 3 boundary ----
        slicer_predictions = slicer_out.get("predictions", [])
        filtered_batch = slicer_out.get("filtered_batch", None)
        if filtered_batch is None or slicer_predictions_empty(slicer_predictions):
            n_dropped_stage3 += 1
            slicer_event_data["stage3_meta/event_dropped"] = int(True)
            slicer_event_data["stage3_meta/n_stage3_sp"] = 0
            slicer_event_data["stage3_meta/has_gt"] = int(False)
            slicer_event_data["stage3_meta/no_object_class_id"] = int(ps_no_object)
            write_event_h5(out_path, slicer_event_data)
            continue

        keep = build_nu_keep_mask(
            slicer_predictions=slicer_predictions,
            n_sp_per_event=filtered_batch["n_spacepoints"],
            nu_class_id=nu_class_id,
            mask_prob_threshold=mask_prob_threshold,
            spacepoint_level=spacepoint_level,
            device=args.device,
        )
        if int(keep.sum().item()) == 0:
            n_dropped_stage3 += 1
            slicer_event_data["stage3_meta/event_dropped"] = int(True)
            slicer_event_data["stage3_meta/n_stage3_sp"] = 0
            slicer_event_data["stage3_meta/no_object_class_id"] = int(ps_no_object)
            write_event_h5(out_path, slicer_event_data)
            continue

        ps_batch = filter_batch_for_particle_segmenter(
            filtered_batch=filtered_batch,
            keep_mask=keep,
            recenter=recenter,
            drop_empty_instances=True,
        )
        if int((ps_batch["n_spacepoints"] == 0).any().item()):
            ps_batch = drop_empty_events(ps_batch)
        if (ps_batch["n_spacepoints"].numel() == 0
                or int(ps_batch["n_spacepoints"].sum().item()) == 0):
            n_dropped_stage3 += 1
            slicer_event_data["stage3_meta/event_dropped"] = int(True)
            slicer_event_data["stage3_meta/n_stage3_sp"] = 0
            slicer_event_data["stage3_meta/no_object_class_id"] = int(ps_no_object)
            write_event_h5(out_path, slicer_event_data)
            continue

        # ---- Stage 3 forward ----
        with torch.no_grad():
            ps_out = ps_inner(ps_batch)
        ps_sample = _unpack_ps_sample(ps_batch)
        # Preserve identity info so the per-event H5 records the input
        # event's run/subrun/event/name in BOTH halves.
        ps_sample["run"]    = int(sample.get("run", -1))
        ps_sample["subrun"] = int(sample.get("subrun", -1))
        ps_sample["event"]  = int(sample.get("event", -1))
        ps_sample["name"]   = str(sample.get("name", ""))
        stage3_event_data = stage3_predict_event_from_out(
            ps_out, ps_sample, ps_no_object,
            class_prob_threshold=args.class_prob_threshold,
            coord_scale=coord_scale,
        )

        # ---- Merge slicer + stage3 halves and write ----
        merged = dict(slicer_event_data)
        merged.update(stage3_event_data)
        write_event_h5(out_path, merged)

        n_gt = merged.get("stage3_meta/n_gt_instances", 0)
        n_matched = merged.get("stage3_meta/n_matched", 0)
        n_sp = merged.get("stage3_meta/n_stage3_sp", 0)
        n_post = merged.get("meta/n_sp_post", 0)
        n_pre = merged.get("meta/n_sp_pre", 0)
        pair_iou = merged.get("stage3_gt/pair_iou", np.zeros(0))
        valid_iou = pair_iou[pair_iou >= 0] if pair_iou.size else pair_iou
        mean_iou = float(np.mean(valid_iou)) if valid_iou.size > 0 else float("nan")
        print(
            f"[{ev_idx+1:4d}/{n_events}] {in_name:50s}  "
            f"raw {n_pre} → dh {n_post} → s3 {n_sp}  "
            f"matched={n_matched}/{n_gt}  "
            f"s3_pair_IoU={mean_iou:.3f}"
        )

    print(f"\n[infer] Done. Wrote stage3pred files to {args.output_dir}  "
          f"(slicer-dropped {n_dropped_slicer}, "
          f"stage3-dropped {n_dropped_stage3} of {n_events})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawTextHelpFormatter,
    )
    ap.add_argument("--config", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--input-mode", default="cached",
                    choices=("cached", "full-cascade"),
                    help="cached: read Stage-1+2 cache events via "
                         "LArFormerStage12CacheDataset (config = "
                         "plain LArFormer or CascadedParticleSegmenter "
                         "wrapper). full-cascade: read raw merged_h5 "
                         "via LArFormerDataset, run deghoster + slicer "
                         "+ particle segmenter end-to-end (config = "
                         "CascadedParticleSegmenter).")
    ap.add_argument("--cache-dir", default=None,
                    help="(cached mode) Root of cache events. Overrides "
                         "cfg.data.<split>.data_root.")
    ap.add_argument("--input-list", default=None,
                    help="(full-cascade mode) Text file with one "
                         "merged_h5 path per line.")
    ap.add_argument("--split", default="val",
                    choices=("train", "val", "test"))
    ap.add_argument("--max-events", type=int, default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available()
                    else "cpu")
    ap.add_argument("--class-prob-threshold", type=float, default=0.0,
                    help="Confidence floor on the cls head's max softmax "
                         "probability per query. Queries below the floor "
                         "are demoted to no_object for the per-SP "
                         "panoptic assignment. 0 = strict argmax.")
    ap.add_argument("--overwrite", action="store_true",
                    help="Recompute and re-write events whose output "
                         "file already exists.")
    ap.add_argument("--no-gt", action="store_true",
                    help="(full-cascade mode) Don't request "
                         "gt_source='particle' — for real data where "
                         "no GT is available. The slicer eval-with-GT "
                         "and Stage-3 matching are skipped; only "
                         "predictions are written.")
    args = ap.parse_args()

    if args.input_mode == "cached":
        run_cached_mode(args)
    elif args.input_mode == "full-cascade":
        run_full_cascade_mode(args)
    else:  # pragma: no cover
        raise RuntimeError(args.input_mode)


if __name__ == "__main__":
    main()
