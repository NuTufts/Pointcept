"""SLURM-array driver for the Stage-3 predicted-mask augment (→ stage-1+2+3
cache). The shard analogue of tools/larformer/augment_stage12_cache_pred_masks.py, the
same way build_stage12_cache_shard.py is to build_stage12_cache_event.py.

Each task processes one shard of the cache .h5 files under --cache-root:
indices [shard_id, shard_id + N, shard_id + 2N, ...] of the SORTED file list
(`LArFormerStage12CacheDataset.get_data_list()` returns `sorted(...)`, so every
task sees the same ordering → stride sharding is disjoint and complete). It
builds the frozen particle segmenter ONCE, then for each owned file reproduces
the trainer's per-event input, runs the segmenter, dedups + IoU-matches, and
writes `entry_0/pred_instances` (see augment_stage12_cache_pred_masks.py for the
field format). Idempotent: files already carrying `entry_0/pred_instances` are
skipped unless --force.

Usage (single node, 1 shard = whole dir — equivalent to the serial tool):

    python tools/larformer/augment_stage12_cache_pred_masks_shard.py \\
        --keypoint-config configs/lartpc/larformer/stage4_keypoint/larformer-keypoint2-particle-v1.py \\
        --cache-root <CACHE_ROOT>/train \\
        --particle-config configs/lartpc/larformer/stage3_particle/larformer-particle-fullcascade-ptv3crosslevel.py \\
        --particle-weights exp/.../model_iter_98652.pth \\
        --shard-id 0 --n-shards 1

SLURM (one shard per array task, per split):

    #SBATCH --array=0-63
    python tools/larformer/augment_stage12_cache_pred_masks_shard.py \\
        --keypoint-config configs/lartpc/larformer/stage4_keypoint/larformer-keypoint2-particle-v1.py \\
        --cache-root /path/to/stage123_cache/train \\
        --particle-config configs/lartpc/larformer/stage3_particle/larformer-particle-fullcascade-ptv3crosslevel.py \\
        --particle-weights /path/to/model_iter_98652.pth \\
        --shard-id $SLURM_ARRAY_TASK_ID --n-shards 64
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import traceback

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import h5py  # noqa: E402

# Reuse the serial tool's core (mirrors build_shard importing from build_event).
from augment_stage12_cache_pred_masks import (  # noqa: E402
    PRED_GROUP,
    build_segmenter,
    build_loader,
    keep_indices_for,
    predict_event,
    write_pred_instances,
    mirror_path,
)


def _shard_indices(n_total: int, shard_id: int, n_shards: int,
                   max_per_shard: int | None = None) -> list[int]:
    """Indices this shard owns: stride layout {k, k+N, k+2N, ...} —
    load-balances even when the file list is sorted by size."""
    out = list(range(shard_id, n_total, n_shards))
    if max_per_shard is not None:
        out = out[:max_per_shard]
    return out


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--keypoint-config", required=True,
                   help="Keypoint training config — its data.<split> dataset "
                        "cfg (filter/recenter/coord_*) is reused so the "
                        "segmenter input == the trainer input.")
    p.add_argument("--cache-root", required=True,
                   help="Dir of cache .h5 to augment (e.g. .../stage123/train).")
    p.add_argument("--split", default="train", choices=("train", "val", "test"))
    p.add_argument("--particle-config", required=True,
                   help="Config defining `particle_segmenter_cfg` matching the "
                        "weights (larformer-particle-fullcascade-*).")
    p.add_argument("--particle-weights", required=True)
    p.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    # shard control
    p.add_argument("--shard-id", type=int, required=True,
                   help="This task's shard index in [0, n_shards). "
                        "Typically $SLURM_ARRAY_TASK_ID.")
    p.add_argument("--n-shards", type=int, required=True)
    p.add_argument("--max-per-shard", type=int, default=None,
                   help="Cap on files processed per shard (for testing).")
    p.add_argument("--progress-every", type=int, default=10)
    # dedup / matching — MUST mirror the model's particle_keypoint_cfg.pred_*.
    p.add_argument("--class-prob-floor", type=float, default=0.3)
    p.add_argument("--dedup-iou", type=float, default=0.6)
    p.add_argument("--mask-thresh", type=float, default=0.0)
    p.add_argument("--min-points", type=int, default=3)
    p.add_argument("--iou-match-thresh", type=float, default=0.3)
    p.add_argument("--output-dir", default=None,
                   help="Copy each owned file to a mirrored path here and "
                        "augment the copy (original untouched). Default: "
                        "in-place (r+).")
    p.add_argument("--force", action="store_true",
                   help="Recompute even if pred_instances already present.")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA unavailable; falling back to CPU.")
        device = torch.device("cpu")

    print(f"=== Stage-3 pred-mask augment shard {args.shard_id}/{args.n_shards} ===")
    print(f"  Cache root: {args.cache_root}")
    print(f"  Split:      {args.split}")
    print(f"  Device:     {device}")

    seg, no_obj = build_segmenter(args.particle_config, args.particle_weights,
                                  device)
    ds = build_loader(args.keypoint_config, args.split, args.cache_root)
    files = list(ds.get_data_list())          # already sorted → stable sharding
    n_total = len(files)
    indices = _shard_indices(n_total, args.shard_id, args.n_shards,
                             max_per_shard=args.max_per_shard)
    print(f"  {n_total} cache files; shard owns {len(indices)}")

    n_done = n_skip = n_none = n_inst = n_err = 0
    t0 = time.perf_counter()
    for i, idx in enumerate(indices):
        path = files[idx]
        tgt = (mirror_path(path, args.cache_root, args.output_dir)
               if args.output_dir else path)
        try:
            if not args.force:
                with h5py.File(tgt, "r") as f:
                    if PRED_GROUP in f:
                        n_skip += 1
                        continue
            sample = ds._load_cache(path)
            if sample is None:
                n_none += 1
                continue
            keep_idx = keep_indices_for(ds, path)
            insts = predict_event(seg, no_obj, sample, args, device)
            with h5py.File(tgt, "r+") as f:
                k = write_pred_instances(f, insts, keep_idx, args,
                                         args.particle_weights)
                f[PRED_GROUP].attrs["shard_id"] = int(args.shard_id)
                f[PRED_GROUP].attrs["n_shards"] = int(args.n_shards)
            n_done += 1
            n_inst += k
        except Exception as e:
            n_err += 1
            print(f"  [{i+1}/{len(indices)}] ERROR idx={idx} "
                  f"{os.path.basename(path)}: {e}")
            traceback.print_exc()
            continue

        if (i + 1) % args.progress_every == 0 or (i + 1) == len(indices):
            el = time.perf_counter() - t0
            rate = (i + 1) / max(1e-9, el)
            eta = (len(indices) - (i + 1)) / max(1e-9, rate)
            print(f"  [{i+1:5d}/{len(indices)}] done={n_done} skip={n_skip} "
                  f"empty={n_none} err={n_err} ({rate:.2f}/s eta {eta/60:.1f}min)"
                  f"  {os.path.basename(path)}")

    el = time.perf_counter() - t0
    print(f"=== Shard {args.shard_id} done in {el/60:.2f} min: "
          f"augmented={n_done} skipped={n_skip} empty_filter={n_none} "
          f"errors={n_err} total_pred_instances={n_inst} ===")
    if n_err > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
