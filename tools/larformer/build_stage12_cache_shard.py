"""SLURM-array driver for the Stage-1+2 cache.

Each task processes one shard of the input list: indices
[shard_id, shard_id + N_shards, shard_id + 2*N_shards, ...]. The output
layout mirrors the production pipeline's 3-level hashed directories so a
single directory never exceeds ~100 files:

    ${CACHE_ROOT}/<split>/<idx//1000>/<idx//100>/<basename>__event<idx>.h5

Idempotent: events with either a `.h5` cache or a `.skipped` marker are
skipped on re-run. Delete the markers to force a re-cache.

Usage (single-node test on the 10-event dev list):

    python tools/larformer/build_stage12_cache_shard.py \\
        --config configs/lartpc/larformer/stage3_particle/larformer-particle-v1.py \\
        --inputlist /abs/devdata_mergedh5_pi0filter_10files.txt \\
        --cache-root /tmp/stage12_cache_v2 \\
        --split val --shard-id 0 --n-shards 1 \\
        --max-spacepoints 80000

SLURM submission (one shard per array task):

    #SBATCH --array=0-127       # 128 shards
    python tools/larformer/build_stage12_cache_shard.py \\
        --config configs/lartpc/larformer/stage3_particle/larformer-particle-v1.py \\
        --inputlist /path/to/h5list_train.txt \\
        --cache-root /path/to/stage12_cache_v2 \\
        --split train \\
        --shard-id $SLURM_ARRAY_TASK_ID --n-shards 128
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import traceback

import torch

# Ensure tools/ is on sys.path so we can import build_stage12_cache_event.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_stage12_cache_event import (  # noqa: E402
    build_cache_event,
    build_dataset_for_caching,
    build_model_from_stage3_config,
)


# ---------------------------------------------------------------------------
# Output path layout
# ---------------------------------------------------------------------------

def _hashed_outdir(cache_root: str, split: str, idx: int) -> str:
    """Return the leaf directory for sample `idx` in the 3-level hashed
    layout. Mirrors the production pipeline's
    `<lineno/1000>/<lineno/100>/` scheme.
    """
    lvl1 = f"{idx // 1000:03d}"
    lvl2 = f"{idx // 100:04d}"
    return os.path.join(cache_root, split, lvl1, lvl2)


def _sample_basename(sample_name: str, idx: int) -> str:
    """`<merged-h5-basename-without-ext>__event<idx>`. Anchored to the
    sample index so re-shuffling the input list doesn't collide."""
    base = os.path.basename(sample_name)
    if base.endswith(".h5"):
        base = base[:-3]
    return f"{base}__event{idx:08d}"


def _output_paths(cache_root: str, split: str, idx: int,
                  sample_name: str) -> tuple[str, str]:
    outdir = _hashed_outdir(cache_root, split, idx)
    base = _sample_basename(sample_name, idx)
    return (os.path.join(outdir, base + ".h5"),
            os.path.join(outdir, base + ".skipped"))


# ---------------------------------------------------------------------------
# Shard iteration
# ---------------------------------------------------------------------------

def _shard_indices(n_total: int, shard_id: int, n_shards: int,
                   max_per_shard: int | None = None) -> list[int]:
    """Return the list of dataset indices this shard owns.

    Stride layout: shard k gets {k, k+N, k+2N, ...}. This load-balances
    across shards even when the input list is sorted by file size.
    """
    out = list(range(shard_id, n_total, n_shards))
    if max_per_shard is not None:
        out = out[:max_per_shard]
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True,
                   help="Stage-3 config used to build the cascade model "
                        "(checkpoints loaded per the config).")
    p.add_argument("--inputlist", required=True,
                   help="Absolute path to the data list for the split "
                        "being cached.")
    p.add_argument("--data-root", default=None,
                   help="Override cfg.data.<split>.data_root.")
    p.add_argument("--cache-root", required=True,
                   help="Root directory for the cache output. Per-event "
                        "files will be placed at "
                        "<cache-root>/<split>/<idx//1000>/<idx//100>/...")
    p.add_argument("--split", default="train",
                   choices=("train", "val", "test"),
                   help="Which cfg.data.<split> to use AND which "
                        "subdirectory under cache-root to write into.")
    p.add_argument("--shard-id", type=int, required=True,
                   help="This task's shard index in [0, n_shards). "
                        "Typically $SLURM_ARRAY_TASK_ID.")
    p.add_argument("--n-shards", type=int, required=True,
                   help="Total number of shards.")
    p.add_argument("--max-per-shard", type=int, default=None,
                   help="Cap on events processed per shard (for testing).")
    p.add_argument("--max-spacepoints", type=int, default=None,
                   help="Cap on per-event SPs (forwarded to the dataset).")
    p.add_argument("--device", default="cuda",
                   choices=("cuda", "cpu"))
    p.add_argument("--tau-loose-floor", type=float, default=0.2)
    p.add_argument("--tau-loose-nominal", type=float, default=0.5)
    p.add_argument("--tau-loose-delta", type=float, default=0.2)
    p.add_argument("--class-prob-floor", type=float, default=0.1)
    p.add_argument("--no-gzip", action="store_true",
                   help="Disable gzip compression on per-query mask_prob.")
    p.add_argument("--no-keypoints", action="store_true",
                   help="Do not copy the source mckeypoints group into cache "
                        "files (keypoint module won't train on this cache).")
    p.add_argument("--force-recache", action="store_true",
                   help="Re-cache events that already have an .h5 OR "
                        ".skipped marker. Default: skip idempotently.")
    p.add_argument("--skip-checkpoints", action="store_true",
                   help="(For testing only) random init the model.")
    p.add_argument("--progress-every", type=int, default=10,
                   help="Print a per-N-event progress line.")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA unavailable; falling back to CPU.")
        device = torch.device("cpu")

    print(f"=== Stage-1+2 cache shard {args.shard_id}/{args.n_shards} ===")
    print(f"  Config:      {args.config}")
    print(f"  Inputlist:   {args.inputlist}")
    print(f"  Cache root:  {args.cache_root}")
    print(f"  Split:       {args.split}")
    print(f"  Device:      {device}")
    print(f"  τ_loose:     floor={args.tau_loose_floor} "
          f"nominal={args.tau_loose_nominal} delta={args.tau_loose_delta}")

    print(f"\n[1/3] Building cascade model from {args.config} ...")
    t0 = time.perf_counter()
    cfg, model = build_model_from_stage3_config(
        args.config, device, skip_checkpoints=args.skip_checkpoints,
    )
    print(f"  built + ckpts in {time.perf_counter()-t0:.1f}s")

    print(f"\n[2/3] Building dataset (gt_source=particle) ...")
    dataset = build_dataset_for_caching(
        cfg, split=args.split,
        inputlist_override=args.inputlist,
        data_root_override=args.data_root,
        max_spacepoints_override=args.max_spacepoints,
    )
    n_total = len(dataset)
    print(f"  dataset has {n_total} samples")

    indices = _shard_indices(n_total, args.shard_id, args.n_shards,
                             max_per_shard=args.max_per_shard)
    print(f"  shard {args.shard_id}/{args.n_shards} owns {len(indices)} "
          f"sample(s)")

    print(f"\n[3/3] Caching ...")
    n_ok, n_skip_existing, n_skip_empty, n_error = 0, 0, 0, 0
    t_start = time.perf_counter()
    for i, idx in enumerate(indices):
        try:
            sample_name = dataset.data_list[idx % len(dataset.data_list)]
        except Exception:
            sample_name = f"sample_{idx}"
        out_h5, out_skip = _output_paths(
            args.cache_root, args.split, idx, sample_name,
        )

        if not args.force_recache:
            if os.path.exists(out_h5):
                n_skip_existing += 1
                if (i + 1) % args.progress_every == 0:
                    elapsed = time.perf_counter() - t_start
                    rate = (i + 1) / max(1e-9, elapsed)
                    print(f"  [{i+1:5d}/{len(indices)}] skip-existing  "
                          f"({rate:.2f}/s)  {os.path.basename(out_h5)}")
                continue
            if os.path.exists(out_skip):
                n_skip_empty += 1
                if (i + 1) % args.progress_every == 0:
                    elapsed = time.perf_counter() - t_start
                    rate = (i + 1) / max(1e-9, elapsed)
                    print(f"  [{i+1:5d}/{len(indices)}] skip-marker    "
                          f"({rate:.2f}/s)  {os.path.basename(out_skip)}")
                continue

        try:
            info = build_cache_event(
                model, dataset, idx, out_h5,
                device=device,
                tau_loose_floor=args.tau_loose_floor,
                tau_loose_nominal=args.tau_loose_nominal,
                tau_loose_delta=args.tau_loose_delta,
                class_prob_floor_for_query_storage=args.class_prob_floor,
                gzip_per_query_data=(not args.no_gzip),
                overwrite=True,
                extra_attrs={
                    "shard_id": int(args.shard_id),
                    "n_shards": int(args.n_shards),
                    "source_config": os.path.abspath(args.config),
                },
                skipped_marker_path=out_skip,
                copy_keypoints=(not args.no_keypoints),
            )
            if info.get("cache_empty"):
                n_skip_empty += 1
            else:
                n_ok += 1
        except Exception as e:
            n_error += 1
            print(f"  [{i+1:5d}/{len(indices)}] ERROR  idx={idx}  {e}")
            traceback.print_exc()
            continue

        if (i + 1) % args.progress_every == 0 or (i + 1) == len(indices):
            elapsed = time.perf_counter() - t_start
            rate = (i + 1) / max(1e-9, elapsed)
            eta = (len(indices) - (i + 1)) / max(1e-9, rate)
            base = os.path.basename(out_h5)
            print(f"  [{i+1:5d}/{len(indices)}] ok={n_ok} "
                  f"skip-existing={n_skip_existing} "
                  f"skip-empty={n_skip_empty} err={n_error}  "
                  f"({rate:.2f}/s eta {eta/60:.1f}min)  {base}")

    elapsed = time.perf_counter() - t_start
    print()
    print(f"=== Shard {args.shard_id} done in {elapsed/60:.2f} min ===")
    print(f"  ok={n_ok}  skip-existing={n_skip_existing}  "
          f"skip-empty={n_skip_empty}  errors={n_error}")
    if n_error > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
