"""S3.2 — Stage-3 cascade benchmark.

Loads a Stage-3 config (e.g. `configs/lartpc/larformer-particle-v1.py`)
together with the trained Stage 1/2/3-backbone checkpoints specified in
the config, then measures forward / backward time and peak GPU memory
on a representative batch.

The benchmark runs TWO modes by default so the user can decide whether
the model-side cascade (current setup, S3.1) needs Stage-1+2 caching
(S3.2 of the plan):

    Mode "full":    run CascadedParticleSegmenter end-to-end every
                    iter — this is what training actually pays.
    Mode "cached":  run Stage 1+2 ONCE per sample to produce the
                    filtered_batch fed to Stage 3, then time only
                    `particle_segmenter(filtered_batch)`. This is the
                    lower bound that pre-computed Stage 1+2 outputs
                    would achieve.

Output: per-iter wall-clock (forward, backward, total), peak
`torch.cuda.max_memory_allocated`, and the Mode-A vs Mode-B speedup —
the deciding metric for whether to push S3.2's caching scheme.

Usage:

    # End-to-end with the production config + the real checkpoints,
    # on the local 10-event dev sample.
    python tools/benchmark_larformer_s3_cascade.py \\
        --config configs/lartpc/larformer-particle-v1.py \\
        --inputlist devdata_mergedh5_pi0filter_10files.txt \\
        --batch-size 2 --n-warmup 2 --n-iters 5

    # Inference-only (no backward) — useful when comparing to the
    # cached-Stage-3 cost.
    python tools/benchmark_larformer_s3_cascade.py \\
        --config configs/lartpc/larformer-particle-v1.py \\
        --inputlist devdata_mergedh5_pi0filter_10files.txt \\
        --no-backward

    # Don't load real checkpoints (use random init — useful for shape
    # sanity-checking before the real checkpoints are available).
    python tools/benchmark_larformer_s3_cascade.py \\
        --config configs/lartpc/larformer-particle-v1.py \\
        --inputlist devdata_mergedh5_pi0filter_10files.txt \\
        --skip-checkpoints
"""
from __future__ import annotations

import argparse
import copy
import os
import statistics
import time
from contextlib import nullcontext
from typing import Optional

import torch

# Side-effect: register all dataset / model types referenced in configs.
import pointcept.datasets  # noqa: F401
import pointcept.models    # noqa: F401
from pointcept.datasets.builder import build_dataset
from pointcept.datasets.larformer import larformer_collate
from pointcept.models.builder import build_model
from pointcept.models.LArFormer import (
    build_nu_keep_mask,
    filter_batch_for_particle_segmenter,
)
from pointcept.models.LArFormer.cascade_filter import drop_empty_events
from pointcept.models.LArFormer.cascade_particle_filter import (
    slicer_predictions_empty,
)
from pointcept.utils.config import Config


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _humanize_bytes(n: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(n) < 1024.0:
            return f"{n:6.2f} {unit}"
        n /= 1024.0
    return f"{n:6.2f} PiB"


def _move_batch(batch: dict, device: torch.device) -> dict:
    out = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device, non_blocking=True)
        elif isinstance(v, list) and v and isinstance(v[0], list):
            # list[B] of list[K] of dict-of-tensors (gt_instances_per_event,
            # fragment_indices_per_event)
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


def _cuda_sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _sample_indices(n_total: int, n_needed: int) -> list[int]:
    if n_total >= n_needed:
        return list(range(n_needed))
    # Repeat with wrap-around — sufficient for benchmarking.
    return [i % n_total for i in range(n_needed)]


# ---------------------------------------------------------------------------
# Stage-1+2 caching helper
# ---------------------------------------------------------------------------

def _precompute_stage3_inputs(model, batch: dict, device: torch.device,
                              nu_class_id: int, mask_prob_threshold: float,
                              spacepoint_level: str,
                              recenter: bool) -> Optional[dict]:
    """Run Stage 1+2 once and return the Stage-3 input batch (`ps_batch`)
    that the cascade would pass to its particle_segmenter.

    Returns None if the batch would result in an empty Stage-3 forward
    (slicer drops everything or no SP passes τ_loose).
    """
    model.cascaded_slicer.eval()
    with torch.no_grad():
        slicer_out = model.cascaded_slicer(batch)

    slicer_predictions = slicer_out.get("predictions", [])
    filtered_batch = slicer_out.get("filtered_batch", None)
    if filtered_batch is None or slicer_predictions_empty(slicer_predictions):
        return None

    keep = build_nu_keep_mask(
        slicer_predictions=slicer_predictions,
        n_sp_per_event=filtered_batch["n_spacepoints"],
        nu_class_id=nu_class_id,
        mask_prob_threshold=mask_prob_threshold,
        spacepoint_level=spacepoint_level,
        device=device,
    )
    if int(keep.sum().item()) == 0:
        return None

    ps_batch = filter_batch_for_particle_segmenter(
        filtered_batch=filtered_batch,
        keep_mask=keep,
        recenter=recenter,
        drop_empty_instances=True,
    )
    if int((ps_batch["n_spacepoints"] == 0).any().item()):
        ps_batch = drop_empty_events(ps_batch)
    if ps_batch["n_spacepoints"].numel() == 0 \
            or int(ps_batch["n_spacepoints"].sum().item()) == 0:
        return None
    return ps_batch


# ---------------------------------------------------------------------------
# Timing core
# ---------------------------------------------------------------------------

def _time_forward_backward(model, batch, device, do_backward: bool):
    """Returns (forward_s, backward_s) — both 0.0 if skipped."""
    _cuda_sync(device)
    t0 = time.perf_counter()
    out = model(batch)
    _cuda_sync(device)
    t1 = time.perf_counter()

    bwd_s = 0.0
    if do_backward and isinstance(out, dict) and "loss" in out \
            and out["loss"].requires_grad:
        loss = out["loss"]
        _cuda_sync(device)
        t2 = time.perf_counter()
        loss.backward()
        _cuda_sync(device)
        bwd_s = time.perf_counter() - t2
    elif do_backward:
        # Cascade returned an empty/zero-loss output (no trainable graph
        # connection for this batch). Skip backward but don't error.
        bwd_s = 0.0

    return (t1 - t0), bwd_s


def _zero_grads(model) -> None:
    for p in model.parameters():
        if p.grad is not None:
            p.grad = None


def _run_benchmark(model, batches, device, *, do_backward: bool,
                   mode_label: str) -> dict:
    """mode_label is a string for logging only."""
    fwd_times, bwd_times = [], []
    for i, batch in enumerate(batches):
        _zero_grads(model)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        fwd_s, bwd_s = _time_forward_backward(
            model, batch, device, do_backward,
        )
        fwd_times.append(fwd_s)
        bwd_times.append(bwd_s)
        peak_mb = (torch.cuda.max_memory_allocated(device) / 1024**2
                   if device.type == "cuda" else 0.0)
        print(f"  [{mode_label}] iter {i+1:2d}/{len(batches)}  "
              f"fwd={fwd_s*1000:7.1f}ms  bwd={bwd_s*1000:7.1f}ms  "
              f"peak={peak_mb:7.1f}MiB")
    return {
        "fwd_times": fwd_times,
        "bwd_times": bwd_times,
    }


def _stats(label: str, fwd_times, bwd_times) -> None:
    if not fwd_times:
        print(f"{label}: no samples")
        return
    fwd_med = statistics.median(fwd_times) * 1000
    fwd_mean = statistics.mean(fwd_times) * 1000
    fwd_std = (statistics.stdev(fwd_times) * 1000
               if len(fwd_times) > 1 else 0.0)
    bwd_med = statistics.median(bwd_times) * 1000
    bwd_mean = statistics.mean(bwd_times) * 1000
    bwd_std = (statistics.stdev(bwd_times) * 1000
               if len(bwd_times) > 1 else 0.0)
    tot = [f + b for f, b in zip(fwd_times, bwd_times)]
    tot_med = statistics.median(tot) * 1000
    tot_mean = statistics.mean(tot) * 1000
    print(f"\n  {label}:")
    print(f"    forward    median={fwd_med:7.1f}ms  "
          f"mean={fwd_mean:7.1f}ms  std={fwd_std:6.1f}ms")
    print(f"    backward   median={bwd_med:7.1f}ms  "
          f"mean={bwd_mean:7.1f}ms  std={bwd_std:6.1f}ms")
    print(f"    total      median={tot_med:7.1f}ms  mean={tot_mean:7.1f}ms")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True,
                   help="Path to a Stage-3 config (e.g. "
                        "configs/lartpc/larformer-particle-v1.py).")
    p.add_argument("--inputlist", default=None,
                   help="Override the config's val data_list_file. "
                        "Recommended for local runs where the cluster "
                        "h5lists aren't present.")
    p.add_argument("--data-root", default=None,
                   help="Override the config's data_root.")
    p.add_argument("--batch-size", type=int, default=2,
                   help="Batch size for the benchmark loop.")
    p.add_argument("--n-warmup", type=int, default=2,
                   help="Warmup iterations (timing discarded).")
    p.add_argument("--n-iters", type=int, default=5,
                   help="Measured iterations.")
    p.add_argument("--max-spacepoints", type=int, default=None,
                   help="Cap SP count per event (default: use config's "
                        "val.max_spacepoints).")
    p.add_argument("--device", default="cuda",
                   choices=("cuda", "cpu"),
                   help="Device (cuda required for spconv implicit_gemm).")
    p.add_argument("--mode", default="both",
                   choices=("full", "cached", "both"),
                   help="full = end-to-end cascade; cached = Stage-3 "
                        "only (Stage 1+2 outputs pre-computed once "
                        "per sample); both = run both for comparison.")
    p.add_argument("--no-backward", action="store_true",
                   help="Skip the backward pass (inference-only).")
    p.add_argument("--skip-checkpoints", action="store_true",
                   help="Don't load checkpoints — use the model's random "
                        "init. Useful for shape sanity checks.")
    p.add_argument("--gt-source", default=None,
                   choices=(None, "slice", "particle"),
                   help="Override dataset's gt_source. Default: use config.")
    p.add_argument("--enable-amp", action="store_true",
                   help="Wrap forward+backward in autocast(bfloat16).")
    p.add_argument("--seed", type=int, default=0,
                   help="Sampler/data seed.")
    return p.parse_args()


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)

    # ---- Apply CLI overrides to the config -----------------------------
    val_cfg = copy.deepcopy(cfg.data.val)
    if args.inputlist is not None:
        # Absolutize: data_root is often "/" so a relative path would
        # get joined into a bogus absolute path.
        val_cfg["data_list_file"] = os.path.abspath(args.inputlist)
    if args.data_root is not None:
        val_cfg["data_root"] = args.data_root
    if args.max_spacepoints is not None:
        val_cfg["max_spacepoints"] = args.max_spacepoints
    if args.gt_source is not None:
        val_cfg["gt_source"] = args.gt_source

    # ---- Strip checkpoint paths if requested ---------------------------
    model_cfg = copy.deepcopy(cfg.model)
    if args.skip_checkpoints:
        model_cfg.pop("cascaded_slicer_weight", None)
        model_cfg.pop("particle_segmenter_backbone_weight", None)
        inner = model_cfg.get("cascaded_slicer", {})
        inner.pop("deghoster_weight", None)
        inner.pop("slicer_backbone_weight", None)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA requested but unavailable; falling back to CPU.")
        device = torch.device("cpu")

    # ---- Build model + dataset ----------------------------------------
    print("=" * 70)
    print(f"Config:         {args.config}")
    print(f"Data list:      {val_cfg['data_list_file']}")
    print(f"Device:         {device}")
    print(f"Batch size:     {args.batch_size}")
    print(f"Warmup iters:   {args.n_warmup}")
    print(f"Measured iters: {args.n_iters}")
    print(f"Mode:           {args.mode}")
    print(f"Backward:       {'OFF' if args.no_backward else 'ON'}")
    print(f"AMP:            {'bfloat16' if args.enable_amp else 'OFF'}")
    print(f"Skip ckpts:     {args.skip_checkpoints}")
    print("=" * 70)

    print("\n[1/4] Building model ...")
    t_build = time.perf_counter()
    model = build_model(model_cfg).to(device)
    print(f"  build + checkpoint load took {time.perf_counter()-t_build:.1f}s")
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  params: total={n_params/1e6:.1f}M  "
          f"trainable={n_trainable/1e6:.1f}M  "
          f"frozen={(n_params-n_trainable)/1e6:.1f}M")

    print("\n[2/4] Building dataset ...")
    dataset = build_dataset(val_cfg)
    n_total = len(dataset)
    n_needed = args.n_warmup + args.n_iters
    print(f"  dataset has {n_total} samples; collecting "
          f"{n_needed} batches of size {args.batch_size}")

    # ---- Collect raw batches (CPU side) -------------------------------
    indices = _sample_indices(n_total, n_needed * args.batch_size)
    batches_cpu = []
    sp_counts = []
    for b_idx in range(n_needed):
        samples = [dataset[indices[b_idx * args.batch_size + i]]
                   for i in range(args.batch_size)]
        batch = larformer_collate(samples)
        batches_cpu.append(batch)
        sp_counts.append(int(batch["n_spacepoints"].sum().item()))

    median_sp = statistics.median(sp_counts)
    print(f"  per-batch SP count: median={median_sp}  "
          f"min={min(sp_counts)}  max={max(sp_counts)}")

    # ---- Move to device once (mimics training: 1 worker preloads) -----
    print("\n[3/4] Moving batches to device ...")
    batches_dev = [_move_batch(b, device) for b in batches_cpu]

    # ---- Benchmark ----------------------------------------------------
    print("\n[4/4] Benchmark loop ...")
    do_backward = (not args.no_backward)
    autocast_ctx = (
        torch.autocast(device_type=device.type, dtype=torch.bfloat16)
        if args.enable_amp else nullcontext()
    )

    results = {}

    if args.mode in ("full", "both"):
        print(f"\n--- Mode A: full cascade (CascadedParticleSegmenter) ---")
        model.train()
        with autocast_ctx:
            # Warmup
            for i in range(args.n_warmup):
                _zero_grads(model)
                _ = _time_forward_backward(
                    model, batches_dev[i], device, do_backward,
                )
                print(f"  [warmup] iter {i+1}/{args.n_warmup} done")
            # Measured
            r = _run_benchmark(
                model, batches_dev[args.n_warmup:], device,
                do_backward=do_backward, mode_label="full",
            )
        results["full"] = r

    if args.mode in ("cached", "both"):
        print(f"\n--- Mode B: cached Stage 1+2 + Stage-3 only "
              f"(particle_segmenter forward only) ---")
        # Precompute Stage 1+2 inputs once.
        ps_batches = []
        print("  Precomputing Stage 1+2 outputs ...")
        for i, b in enumerate(batches_dev):
            ps_b = _precompute_stage3_inputs(
                model, b, device,
                nu_class_id=model.nu_class_id,
                mask_prob_threshold=model.mask_prob_threshold,
                spacepoint_level=model.spacepoint_level,
                recenter=model.recenter_to_slice_centroid,
            )
            if ps_b is None:
                print(f"    sample {i}: cascade returns empty → skipping "
                      f"for cached mode")
                continue
            ps_batches.append(ps_b)
            n_kept = int(ps_b["n_spacepoints"].sum().item())
            print(f"    sample {i}: ps_batch has {n_kept} SPs across "
                  f"{ps_b['n_spacepoints'].numel()} events")

        if not ps_batches:
            print("  ALL samples produced empty cascade outputs — skipping "
                  "cached-mode benchmark.")
        else:
            seg = model.particle_segmenter
            seg.train()
            with autocast_ctx:
                # Warmup
                for i in range(min(args.n_warmup, len(ps_batches))):
                    _zero_grads(seg)
                    _ = _time_forward_backward(
                        seg, ps_batches[i], device, do_backward,
                    )
                    print(f"  [warmup] iter {i+1} done")
                # Measured
                measured = ps_batches[min(args.n_warmup, len(ps_batches)):]
                if not measured:
                    print("  Not enough cached samples to measure — "
                          "reusing samples.")
                    measured = ps_batches
                r = _run_benchmark(
                    seg, measured, device,
                    do_backward=do_backward, mode_label="cached",
                )
            results["cached"] = r

    # ---- Summary ------------------------------------------------------
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    if "full" in results:
        _stats("Mode A — full cascade",
               results["full"]["fwd_times"],
               results["full"]["bwd_times"])
    if "cached" in results:
        _stats("Mode B — cached Stage 1+2 + Stage-3 only",
               results["cached"]["fwd_times"],
               results["cached"]["bwd_times"])
    if "full" in results and "cached" in results:
        full_tot = (statistics.median(results["full"]["fwd_times"])
                    + statistics.median(results["full"]["bwd_times"]))
        cached_tot = (statistics.median(results["cached"]["fwd_times"])
                      + statistics.median(results["cached"]["bwd_times"]))
        if cached_tot > 0:
            speedup = full_tot / cached_tot
            print(f"\n  Caching speedup (A/B): {speedup:.2f}×")
            stage12_ms = (full_tot - cached_tot) * 1000
            print(f"  Implied Stage-1+2 per-iter cost: {stage12_ms:.1f}ms "
                  f"({100.0*(1-cached_tot/full_tot):.1f}% of full cost)")

    if device.type == "cuda":
        print(f"\n  CUDA peak alloc (last iter): "
              f"{_humanize_bytes(torch.cuda.max_memory_allocated(device))}")
        print(f"  CUDA reserved (now): "
              f"{_humanize_bytes(torch.cuda.memory_reserved(device))}")
    print()


if __name__ == "__main__":
    main()
