"""Augment a Stage-1+2 cache with Stage-3 PREDICTED particle masks → a
"stage-1+2+3" cache for fast keypoint training.

WHY: the per-particle keypoint decoder can train its MAIN path on the frozen
particle segmenter's PREDICTED masks (so it sees the SAME imperfect masks as at
inference — see lartpc_data_prep/larformer_keypoint_v2/README.md). Running that
segmenter LIVE each step (`main_source="predicted"`) ~doubles batch time. This
tool precomputes the deduped predicted masks ONCE and stores them in the cache;
training then reads them (`main_source="predicted_cached"` +
`load_pred_instances=True`) with NO per-step segmenter cost.

FIDELITY: for each event we reproduce EXACTLY the keypoint trainer's per-event
input by loading it through `LArFormerStage12CacheDataset` (same
`source_set_filter` + `recenter_to_centroid`), run the frozen segmenter, and
dedup via the SAME `predicted_masks_to_instances` as CascadedKeypoint inference.
Because the no-cap cache slice + filter + recenter are deterministic and the
segmenter runs in eval (no jitter/DN), the cached masks are bit-identical to the
live path.

WRITES per event:
    entry_0/pred_instances/instance_<k>/
        truth_indices    (M,) int64  — ORIGINAL cache-SP indices (remapped at
                                        load exactly like GT truth_indices)
        @pred_class      int   — segmenter argmax class for the mask (0..6)
        @primary_trackid int   — IoU-matched GT trackid (-1 if unmatched)
        @match_iou       float — IoU to the matched GT mask (0 if unmatched)
    entry_0/pred_instances  (group attrs: provenance — segmenter ckpt, filter,
                             dedup params, n_instances)

GPU. In-place (r+) by default; --output-dir to copy+augment. Idempotent: files
that already have entry_0/pred_instances are skipped unless --force.

Usage (the no-cap dev cache, both splits):

    for SPLIT in train val; do
      python tools/augment_stage12_cache_pred_masks.py \\
        --keypoint-config configs/lartpc/larformer/stage4_keypoint/larformer-keypoint2-particle-v1.py \\
        --cache-root /mnt/ddrive/data/ub_on_tufts/devdata_cache_stage12_nocap/$SPLIT \\
        --particle-config configs/lartpc/larformer/stage3_particle/larformer-particle-fullcascade-ptv3crosslevel.py \\
        --particle-weights exp/larformer_particle_v1_cached_ptv3crosslevel_smallbatch_lr1e4_bugfixed/model_iter_98652.pth
    done
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import h5py  # noqa: E402

from pointcept.utils.config import Config  # noqa: E402
from pointcept.models import build_model  # noqa: E402
from pointcept.datasets import build_dataset, larformer_collate  # noqa: E402
from pointcept.models.LArFormer.keypoint2_cascade import (  # noqa: E402
    predicted_masks_to_instances,
)

PRED_GROUP = "entry_0/pred_instances"


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--keypoint-config", required=True,
                   help="Keypoint training config — its data.<split> dataset "
                        "cfg (source_set_filter, recenter, coord_*) is reused "
                        "VERBATIM so the segmenter input == the trainer input.")
    p.add_argument("--cache-root", required=True,
                   help="Dir of cache .h5 to augment (e.g. .../nocap/train).")
    p.add_argument("--split", default="train", choices=("train", "val", "test"),
                   help="Which data.<split> dataset cfg to clone (filter/"
                        "recenter knobs). Default train.")
    p.add_argument("--particle-config", required=True,
                   help="Config defining `particle_segmenter_cfg` matching the "
                        "weights (use larformer-particle-fullcascade-*).")
    p.add_argument("--particle-weights", required=True)
    p.add_argument("--device", default="cuda")
    # dedup / matching — MUST mirror the model's particle_keypoint_cfg.pred_*.
    p.add_argument("--class-prob-floor", type=float, default=0.3)
    p.add_argument("--dedup-iou", type=float, default=0.6)
    p.add_argument("--mask-thresh", type=float, default=0.0)
    p.add_argument("--min-points", type=int, default=3)
    p.add_argument("--iou-match-thresh", type=float, default=0.3)
    p.add_argument("--output-dir", default=None,
                   help="Copy each file to a mirrored path here and augment the "
                        "copy (original untouched). Default: in-place (r+).")
    p.add_argument("--force", action="store_true",
                   help="Recompute even if pred_instances already present.")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def build_segmenter(particle_config, weights, device):
    cfg = Config.fromfile(particle_config)
    seg = build_model(dict(cfg["particle_segmenter_cfg"]))
    sd = torch.load(weights, map_location="cpu")
    sd = sd.get("state_dict", sd.get("model", sd))
    sd = {k[len("module."):] if k.startswith("module.") else k: v
          for k, v in sd.items()}
    missing, unexpected = seg.load_state_dict(sd, strict=False)
    print(f"[segmenter] loaded {weights}: missing={len(missing)} "
          f"unexpected={len(unexpected)}")
    no_obj = int(cfg["particle_segmenter_cfg"].get("num_classes", 8)) - 1
    return seg.to(device).eval(), no_obj


def build_loader(keypoint_config, split, cache_root):
    cfg = Config.fromfile(keypoint_config)
    ds_cfg = dict(getattr(cfg.data, split))
    ds_cfg["data_root"] = cache_root
    ds_cfg.pop("data_list_file", None)
    ds_cfg["load_pred_instances"] = False        # we're CREATING them
    ds_cfg["loop"] = 1
    return build_dataset(ds_cfg)


@torch.no_grad()
def predict_event(seg, no_obj, sample, args, device):
    """Run the frozen segmenter on ONE loaded cache event → list of dicts:
    {orig_truth_indices (np int64), pred_class, primary_trackid, match_iou}."""
    batch = larformer_collate([sample])
    batch = {k: (v.to(device) if torch.is_tensor(v) else v)
             for k, v in batch.items()}
    seg_in = {k: v for k, v in batch.items()
              if k not in ("gt_instances_per_event", "n_gt_instances")}
    out = seg(seg_in)
    preds = out.get("predictions", []) if isinstance(out, dict) else []
    if not preds:
        return []
    ev = preds[0]
    cl = ev.get("class_logits")
    sm = ev.get("mask_logits", {}).get("spacepoint")
    if cl is None or sm is None:
        return []
    insts = predicted_masks_to_instances(
        cl, sm, no_obj,
        class_prob_floor=args.class_prob_floor,
        dedup_iou_threshold=args.dedup_iou,
        mask_thresh=args.mask_thresh,
        min_points=args.min_points,
        gt_instances=sample["gt_instances"],
        iou_match_thresh=args.iou_match_thresh)
    return insts


def keep_indices_for(ds, path):
    """ORIGINAL-cache indices of the kept (filtered) SPs, in load order —
    matches LArFormerStage12CacheDataset's `coord[keep]` exactly."""
    with h5py.File(path, "r") as f:
        e0 = f["entry_0"]
        source_mask = e0["source_mask"][:].astype(np.uint8)
        stage2_prob = e0["stage2_nu_mask_prob"][:].astype(np.float32)
    keep = ds._select_keep_mask(source_mask, stage2_prob)
    return np.where(keep)[0]


def write_pred_instances(f, insts, keep_idx, args, weights):
    if PRED_GROUP in f:
        del f[PRED_GROUP]
    grp = f.create_group(PRED_GROUP)
    grp.attrs["segmenter_weight"] = os.path.abspath(weights)
    grp.attrs["class_prob_floor"] = float(args.class_prob_floor)
    grp.attrs["dedup_iou"] = float(args.dedup_iou)
    grp.attrs["iou_match_thresh"] = float(args.iou_match_thresh)
    grp.attrs["mask_thresh"] = float(args.mask_thresh)
    grp.attrs["min_points"] = int(args.min_points)
    n = 0
    for inst in insts:
        fti = np.asarray(inst["truth_indices"].detach().cpu().numpy()
                         if torch.is_tensor(inst["truth_indices"])
                         else inst["truth_indices"], dtype=np.int64)
        if fti.size == 0:
            continue
        orig = keep_idx[fti].astype(np.int64)          # filtered → original
        g = grp.create_group(f"instance_{n}")
        g.create_dataset("truth_indices", data=orig, compression="gzip")
        g.attrs["pred_class"] = int(inst.get("pred_class", -1))
        g.attrs["primary_trackid"] = int(inst.get("primary_trackid", -1))
        g.attrs["match_iou"] = float(inst.get("match_iou", 0.0))
        n += 1
    grp.attrs["n_instances"] = n
    return n


def mirror_path(src_path, src_root, dst_root):
    rel = os.path.relpath(src_path, src_root)
    dst = os.path.join(dst_root, rel)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if not os.path.exists(dst):
        shutil.copy2(src_path, dst)
    return dst


def main():
    args = parse_args()
    device = torch.device(args.device)
    seg, no_obj = build_segmenter(args.particle_config, args.particle_weights,
                                  device)
    ds = build_loader(args.keypoint_config, args.split, args.cache_root)
    files = list(ds.get_data_list())
    if args.limit is not None:
        files = files[:args.limit]
    print(f">>> {len(files)} cache files under {args.cache_root}")

    n_done = n_skip = n_none = n_inst = 0
    for i, path in enumerate(files):
        tgt = (mirror_path(path, args.cache_root, args.output_dir)
               if args.output_dir else path)
        if not args.force:
            with h5py.File(tgt, "r") as f:
                if PRED_GROUP in f:
                    n_skip += 1
                    if args.verbose:
                        print(f"  [{i}] skip (has pred_instances): "
                              f"{os.path.basename(path)}")
                    continue
        sample = ds._load_cache(path)
        if sample is None:
            n_none += 1
            if args.verbose:
                print(f"  [{i}] skip (filter left <min_spacepoints): "
                      f"{os.path.basename(path)}")
            continue
        keep_idx = keep_indices_for(ds, path)
        insts = predict_event(seg, no_obj, sample, args, device)
        with h5py.File(tgt, "r+") as f:
            k = write_pred_instances(f, insts, keep_idx, args,
                                     args.particle_weights)
        n_done += 1
        n_inst += k
        if args.verbose or (i % 50 == 0):
            print(f"  [{i}] {os.path.basename(path)}: {k} pred instances "
                  f"(kept SPs={keep_idx.size})")

    print(f">>> DONE: augmented={n_done} skipped={n_skip} "
          f"empty_filter={n_none} total_pred_instances={n_inst}")


if __name__ == "__main__":
    main()
