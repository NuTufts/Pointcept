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
    `configs/lartpc/larformer/stage3_particle/larformer-particle-v1-cached-ptv3crosslevel.py`).
    For each event, runs the particle segmenter forward, writes a
    `stage3pred_<basename>.h5` with `stage3*/...` keys.

  --input-mode full-cascade:

    Reads raw merged_h5 events via a standard `LArFormerDataset`. The
    config must declare a `CascadedParticleSegmenter` model (e.g.
    `configs/lartpc/larformer/stage3_particle/larformer-particle-v1.py`) so we can find both the
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
        --config configs/lartpc/larformer/stage3_particle/larformer-particle-v1-cached-ptv3crosslevel.py \\
        --weights exp/.../model/model_last.pth \\
        --cache-dir exp/cache_stage12_ptv3crosslevelslicer_iter_75750/val \\
        --output-dir exp/.../inference \\
        --class-prob-threshold 0.3

Usage (full-cascade mode):
    ./run_in_container.sh python tools/run_larformer_stage3_inference.py \\
        --input-mode full-cascade \\
        --config configs/lartpc/larformer/stage3_particle/larformer-particle-v1.py \\
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
    _prob_to_logit,
)
from pointcept.models.LArFormer.cascade_filter import drop_empty_events  # noqa: E402


# Flash-recovery imports (optional path; only used when --flash-recover-k > 0).
_TPC_BOUNDS = ((0.0, 256.0), (-116.5, 116.5), (0.0, 1036.0))


def _oob_frac_np(pts):
    if len(pts) == 0:
        return 1.0
    b = _TPC_BOUNDS
    inb = ((pts[:, 0] >= b[0][0]) & (pts[:, 0] <= b[0][1]) &
           (pts[:, 1] >= b[1][0]) & (pts[:, 1] <= b[1][1]) &
           (pts[:, 2] >= b[2][0]) & (pts[:, 2] <= b[2][1]))
    return float(1.0 - inb.mean())


def flash_recovery_keep(ev_pred, filtered_batch, input_h5_path, no_object_class_id,
                        mask_prob_threshold, gamma_beam, K, chi2_max, oob_max):
    """Expanded keep-mask term: union of the top-K slice queries whose PhotonLib-
    predicted PMT pattern best matches the in-time beam flash (Neyman chi2), cut by
    chi2 <= chi2_max and OOB <= oob_max. Returns an (n_sp,) bool tensor (or None if
    no in-time beam flash). The observed flash + raw charge come from input_h5_path."""
    import h5py
    from lartpc.larformer_analysis.slicer_eval.lib.flash_predict import (
        predict_slice_pe, select_charge_y_with_uv_fallback_np)
    from lartpc.larformer_analysis.slicer_eval.lib.flash_chi2 import neyman_chi2

    sp_mask = ev_pred["mask_logits"]["spacepoint"]          # (Q, n_sp) logits
    Q, n_sp = sp_mask.shape
    if n_sp == 0 or Q == 0:
        return None
    post_cm = filtered_batch["coord"]
    post_cm = (post_cm.detach().cpu().numpy() if torch.is_tensor(post_cm)
               else np.asarray(post_cm)).astype(np.float32)
    if post_cm.shape[0] != n_sp:        # single-event batch expected
        post_cm = post_cm[:n_sp]

    with h5py.File(input_h5_path, "r") as f:
        e = f["entry_0"]
        ipos = e["triplet_data"]["pos"][:].astype(np.float32)
        icharge = select_charge_y_with_uv_fallback_np(e["triplet_data"]["pixval"][:])
        fl = e["flashes"]
        f_pe = fl["pe"][:]; f_pid = fl["producer_id"][:]
        f_tpe = fl["total_pe"][:]; f_t = fl["time_us"][:]
    beam = np.where(f_pid == 0)[0]
    if len(beam) == 0:
        return None
    bi = beam[int(np.argmax(f_tpe[beam]))]
    pe_obs = f_pe[bi].astype(np.float64); t0 = float(f_t[bi])

    VOX = 1.0
    vc = {}
    for k, q in zip(map(tuple, np.floor(ipos / VOX).astype(np.int64)), icharge):
        vc.setdefault(k, []).append(q)
    vc = {k: float(np.mean(v)) for k, v in vc.items()}
    post_charge = np.array([vc.get(k, 0.0) for k in
                            map(tuple, np.floor(post_cm / VOX).astype(np.int64))],
                           dtype=np.float32)

    cls_argmax = ev_pred["class_logits"].argmax(dim=-1).cpu().numpy()
    tau_logit = _prob_to_logit(float(mask_prob_threshold))
    binmask = (sp_mask > tau_logit).cpu().numpy()           # (Q, n_sp)
    cand = []
    for q in range(Q):
        if int(cls_argmax[q]) == int(no_object_class_id):
            continue
        m = binmask[q]
        if m.sum() < 10:
            continue
        pts = post_cm[m]
        if _oob_frac_np(pts) > oob_max:
            continue
        pe_pred = predict_slice_pe(pts, post_charge[m], t0, producer_id=0,
                                   gamma_by_producer=(gamma_beam, gamma_beam))
        c2 = neyman_chi2(pe_pred, pe_obs)
        if np.isfinite(c2) and c2 <= chi2_max:
            cand.append((c2, q))
    cand.sort()
    flash_keep = np.zeros(n_sp, dtype=bool)
    for _, q in cand[:K]:
        flash_keep |= binmask[q]
    return torch.from_numpy(flash_keep).to(sp_mask.device)


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
    if args.cache_file_list is not None:
        # Explicit list of cache files (one per line). Overrides the
        # dataset's auto-discovery so the SLURM driver can hand each
        # task its own subset of val events.
        ds_cfg["data_list_file"] = os.path.abspath(args.cache_file_list)
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
          f"dedup_iou_threshold={args.dedup_iou_threshold}  "
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

        if args.deterministic:
            reseed_per_event()   # per-event order invariance (see helper)
        batched = larformer_collate([sample])
        batched = _move_batch(batched, args.device)

        event_data = stage3_predict_event(
            model, sample, batched, no_object_class_id,
            class_prob_threshold=args.class_prob_threshold,
            coord_scale=coord_scale,
            dedup_iou_threshold=args.dedup_iou_threshold,
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
            f"`configs/lartpc/larformer/stage3_particle/larformer-particle-v1.py`."
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

    # Defensive per-event-reproducibility fix: serialization order-shuffle
    # (shuffle_orders) is a TRAIN-TIME augmentation (random order via
    # torch.randperm, not gated by eval). Left on at inference it makes a given
    # event's output depend on RNG history -> on which OTHER events preceded it
    # in the run (membership/list/batch dependence). Force it off model-wide.
    _n_shuf = 0
    for _m in model.modules():
        if getattr(_m, "shuffle_orders", False):
            _m.shuffle_orders = False
            _n_shuf += 1
    if _n_shuf:
        print(f"[infer] disabled serialization order-shuffle on {_n_shuf} modules "
              f"(per-event-reproducible inference)")

    cascaded_slicer = _resolve_cascaded_slicer(model)
    ps_inner = _resolve_particle_segmenter(model)
    inner = getattr(model, "module", model)
    if cascaded_slicer is None:
        raise RuntimeError("could not find cascaded_slicer on built model")
    # Optional Stage-1 deghoster keep-threshold override. The deghoster keeps
    # SPs with P(real) > deghost_threshold_val; the forward reads this attr
    # dynamically in eval, so setting it here changes how aggressively Stage 1
    # prunes WITHOUT retraining. Lower it (toward the low end of the training
    # range) to retain scattered low-confidence shower SPs the segmenter needs.
    if args.deghost_threshold_val is not None:
        old = float(getattr(cascaded_slicer, "deghost_threshold_val", float("nan")))
        cascaded_slicer.deghost_threshold_val = float(args.deghost_threshold_val)
        print(f"[infer] OVERRIDE deghost_threshold_val {old} -> "
              f"{cascaded_slicer.deghost_threshold_val} "
              f"(train range U({getattr(cascaded_slicer,'deghost_threshold_min','?')}, "
              f"{getattr(cascaded_slicer,'deghost_threshold_max','?')}))")
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
          f"τ_loose={mask_prob_threshold}  recenter={recenter}  "
          f"dedup_iou_threshold={args.dedup_iou_threshold}")

    os.makedirs(args.output_dir, exist_ok=True)
    n_dropped_slicer = 0
    n_dropped_stage3 = 0
    n_dropped_oom = 0
    for ev_idx in range(n_events):
        sample = dataset[ev_idx]
        in_name = sample.get("name", f"event{ev_idx:06d}.h5")
        stem = os.path.splitext(in_name)[0]
        out_path = os.path.join(args.output_dir, f"stage3pred_{stem}.h5")
        if os.path.exists(out_path) and not args.overwrite:
            print(f"[{ev_idx+1:4d}/{n_events}] skip-existing: "
                  f"{os.path.basename(out_path)}")
            continue

        # Proactively release cached blocks between events to limit fragmentation
        # (the equivalent of Pointcept's Tester empty_cache=True, which this custom
        # loop does not go through). Cheap insurance against OOM on small GPUs.
        # LARFORMER_SKIP_EMPTY_CACHE=1 disables it for the allocator-stability
        # study (per-event empty_cache + expandable_segments make tensor layout
        # history-dependent — the leading hypothesis for the list-order dependence).
        if args.device == "cuda" and not os.environ.get("LARFORMER_SKIP_EMPTY_CACHE"):
            torch.cuda.empty_cache()

        if args.deterministic:
            reseed_per_event()   # per-event order invariance (see helper)
        batched = larformer_collate([sample])
        batched = _move_batch(batched, args.device)

        # ---- Stage 1+2 forward (WITH gt for matching diagnostics) ----
        # Per-event OOM guard: a single large/busy event must not crash the whole
        # run (esp. on 16 GB P100s). Skip the event, free cache, continue.
        try:
            with torch.no_grad():
                slicer_out = cascaded_slicer(batched)
        except RuntimeError as ex:
            if "out of memory" not in str(ex).lower():
                raise
            del batched
            torch.cuda.empty_cache()
            n_dropped_oom += 1
            print(f"[{ev_idx+1:4d}/{n_events}] {in_name:50s}  OOM (stage1/2) — skipped")
            continue
        slicer_event_data = slicer_predict_event_from_out(
            slicer_out, sample, slicer_no_object,
            slicer_dedup_iou=args.slicer_dedup_iou,
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
        # Expanded segmenter flow: also segment the top-K flash-matched slices
        # (union into the keep mask). By default ONLY rescues would-be-dropped
        # events (empty nu-keep) so it can't contaminate already-segmented ones;
        # --flash-recover-augment-all applies it to every event.
        _do_flash = (args.flash_recover_k > 0 and slicer_predictions
                     and "class_logits" in slicer_predictions[0]
                     and (args.flash_recover_augment_all or int(keep.sum().item()) == 0))
        if _do_flash:
            in_path = dataset.data_list[ev_idx % len(dataset.data_list)]
            try:
                fkeep = flash_recovery_keep(
                    slicer_predictions[0], filtered_batch, in_path,
                    no_object_class_id=slicer_no_object,
                    mask_prob_threshold=mask_prob_threshold,
                    gamma_beam=args.flash_recover_gamma,
                    K=args.flash_recover_k,
                    chi2_max=args.flash_recover_chi2_max,
                    oob_max=args.flash_recover_oob_max,
                )
                if fkeep is not None and fkeep.shape == keep.shape:
                    keep = keep | fkeep
            except Exception as _ex:
                print(f"   [flash-recover] skipped ({type(_ex).__name__}: {_ex})")
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
        try:
            with torch.no_grad():
                ps_out = ps_inner(ps_batch)
        except RuntimeError as ex:
            if "out of memory" not in str(ex).lower():
                raise
            del batched, ps_batch
            torch.cuda.empty_cache()
            n_dropped_oom += 1
            print(f"[{ev_idx+1:4d}/{n_events}] {in_name:50s}  OOM (stage3) — skipped")
            continue
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
            dedup_iou_threshold=args.dedup_iou_threshold,
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
          f"stage3-dropped {n_dropped_stage3}, "
          f"OOM-skipped {n_dropped_oom} of {n_events})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def set_deterministic(seed: int = 0):
    """Make cascade inference repeatable (same input -> same output on a fixed
    GPU/driver/lib stack). Addresses the run-to-run argmax jitter diagnosed for
    the single-photon study:

      - TF32 matmul (Ampere default) truncates to 10-bit mantissa and lets
        cuBLAS vary its split-K reduction order  -> the dominant logit jitter.
      - SerializedPooling does `torch.sort(cluster)` (non-stable for ties) then
        `segment_csr(reduce="mean")` (non-associative)  -> pooled-feature jitter
        that propagates through depth. Deterministic sort fixes the input order
        so the segmented reduction becomes reproducible.
      - cuBLAS needs CUBLAS_WORKSPACE_CONFIG set (at process start) to be
        deterministic; cuDNN benchmark autotuner must be off.

    warn_only=True: surfaces (rather than hard-failing on) any remaining op
    without a deterministic CUDA impl (e.g. some torch_scatter kernels), so a
    single un-pinned op doesn't abort a production run -- it prints a warning
    naming the op instead.
    """
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        torch.set_float32_matmul_precision("highest")
    except Exception:
        pass
    torch.use_deterministic_algorithms(True, warn_only=True)
    print("[deterministic] seeds fixed, TF32 off, cuDNN deterministic, "
          "torch deterministic algorithms on (warn_only). "
          "CUBLAS_WORKSPACE_CONFIG=:4096:8")


def reseed_per_event(seed: int = 0):
    """Reset RNG before each event so a per-event output is independent of the
    event's POSITION in the batch sequence (event-order invariance).

    `set_deterministic()` seeds only ONCE at startup, which gives same-order
    run-to-run repeatability but NOT order invariance: the forward consumes
    per-event RNG, so the Nth event sees RNG state advanced by the N-1 events
    before it. Re-seeding here makes each event's result depend only on the
    event, so re-running with the input list in a different order (or a subset)
    gives the same per-event output. Call at the top of each event iteration in
    --deterministic mode. (The dataset itself is RNG-free in the test split —
    fixed lm-score threshold, max_spacepoints=None — so this targets the
    forward-side RNG.)"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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
    ap.add_argument("--cache-file-list", default=None,
                    help="(cached mode) Text file with one cache event "
                         "path per line. Overrides the dataset's "
                         "auto-discovery — used by the SLURM driver to "
                         "hand each task its own subset of cache events. "
                         "Paths in the file are absolute, or relative to "
                         "--cache-dir if set.")
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
    ap.add_argument("--flash-recover-k", type=int, default=0,
                    help="Expanded segmenter flow: also segment the top-K slices by "
                         "in-time-beam-flash Neyman chi2 (PhotonLib). 0 = off (default). "
                         "Adds those slices' SPs to the particle-segmenter keep mask.")
    ap.add_argument("--flash-recover-chi2-max", type=float, default=500.0)
    ap.add_argument("--flash-recover-oob-max", type=float, default=0.05)
    ap.add_argument("--flash-recover-gamma", type=float, default=5.25)
    ap.add_argument("--flash-recover-augment-all", action="store_true",
                    help="Apply flash recovery to EVERY event (default: only rescue "
                         "events whose nu-keep is empty, to avoid contaminating "
                         "already-segmented events).")
    ap.add_argument("--slicer-dedup-iou", type=float, default=0.0,
                    help="Enable slicer-side query dedup (mask-IoU NMS, same as the "
                         "particle dedup) at this threshold; merges co-extensive "
                         "duplicate SLICE queries so a split photon is consolidated. "
                         "0.0 = off (default, unchanged behavior). Try 0.6.")
    ap.add_argument("--dedup-iou-threshold", type=float, default=0.6,
                    help="Query dedup (R8): merge co-extensive duplicate "
                         "queries whose binarized spacepoint-mask IoU is "
                         ">= this, before the per-SP panoptic assignment "
                         "(greedy NMS, higher panoptic score wins; the "
                         "absorbed query's class hypothesis is recorded "
                         "under stage3_queries/dedup_*). Targets the "
                         "mu/pi duplicate-pair failure mode. 0 disables.")
    ap.add_argument("--overwrite", action="store_true",
                    help="Recompute and re-write events whose output "
                         "file already exists.")
    ap.add_argument("--no-gt", action="store_true",
                    help="(full-cascade mode) Don't request "
                         "gt_source='particle' — for real data where "
                         "no GT is available. The slicer eval-with-GT "
                         "and Stage-3 matching are skipped; only "
                         "predictions are written.")
    ap.add_argument("--deghost-threshold-val", type=float, default=None,
                    help="(full-cascade) Override the Stage-1 deghoster keep "
                         "threshold τ (keep SPs with P(real)>τ). Config default "
                         "is 0.5; training sampled τ~U(0.4,0.6). Lower it (e.g. "
                         "0.4 or 0.3) to retain scattered low-confidence shower "
                         "SPs — helps the segmenter pick photon over no_object. "
                         "No retraining needed (read at eval time).")
    ap.add_argument("--deterministic", action="store_true",
                    help="Repeatable inference for physics production: fixes "
                         "seeds, disables TF32, forces deterministic cuBLAS/cuDNN "
                         "+ torch algorithms so the same input yields the same "
                         "output on a given GPU/driver/lib stack. Costs ~1.3-2x "
                         "latency on Ampere (TF32 off). Without it the per-SP "
                         "argmax jitters run-to-run (~9% event-level label flips).")
    args = ap.parse_args()

    if args.deterministic:
        set_deterministic()

    if args.input_mode == "cached":
        run_cached_mode(args)
    elif args.input_mode == "full-cascade":
        run_full_cascade_mode(args)
    else:  # pragma: no cover
        raise RuntimeError(args.input_mode)


if __name__ == "__main__":
    main()
