"""
Inference script for ShowerClusteringMask2Former.

Loads a trained checkpoint, runs the model on a test dataset, writes one
HDF5 file per event under --output-dir. The per-event H5 schema (see
write_event_h5 below) is the contract consumed by
tools/visualize_shower_clustering_inference.py.

Per-spacepoint and per-pair quantities the visualizer needs are
pre-computed here so the viewer is just dataframe-style reads, not model
re-runs.

Usage
=====

    python tools/run_shower_clustering_inference.py \\
        -c configs/lartpc/shower-cluster-sonata-v1.py \\
        --checkpoint exp/shower_clustering/run1/model/model_epoch4.pth \\
        --data-list lartpc_data_prep/lantern_scripts/h5lists/h5list_bnbnu_pi0filter_validated_test.txt \\
        --output-dir exp/shower_clustering/run1/inference_epoch4/ \\
        --max-events 10 --device cuda

Default sub-sample is "first N events"; pass --shuffle to draw a random
subset (uses --seed for reproducibility).

Notes
-----
- `--cap-spacepoints` overrides cfg.data.test.max_spacepoints. Default 0
  means "no cap" (full events). Useful if you want to inference at full
  resolution regardless of how training was capped.
- The matcher is run inline using model.loss_fn.matcher with the same GT
  buildup the training loss uses, so the saved `matched/*` arrays are
  directly comparable to the loss-time matching during training.
- An aggregator that scans these per-event H5s and produces summary
  metrics across the test set is a follow-up; the per-event format is
  designed for trivial concatenation.
"""

import argparse
import os
import sys
from typing import Tuple

import h5py
import numpy as np
import torch

# Project root on path so we can import pointcept.*
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from pointcept.utils.config import Config
from pointcept.datasets.builder import DATASETS
from pointcept.datasets.shower_clustering import shower_clustering_collate
from pointcept.models.builder import build_model

# Side-effect import: registers ShowerClusteringTrainer (not strictly needed
# here since we don't call TRAINERS.build, but keeps the engines/train load
# chain identical to tools/train.py — defensive).
import pointcept.models.shower_clustering.trainer  # noqa: F401


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("-c", "--config", required=True, help="Pointcept config")
    p.add_argument("--checkpoint", required=True, help="Trained model .pth")
    p.add_argument("--data-list", default=None,
                   help="Override cfg.data.test.data_list_file")
    p.add_argument("--output-dir", required=True,
                   help="Directory for per-event H5 outputs (created if missing)")
    p.add_argument("--max-events", type=int, default=None,
                   help="Cap on total events processed. None = all.")
    p.add_argument("--shuffle", action="store_true",
                   help="Sample a random subset of `--max-events` instead "
                        "of taking the first N.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--mask-logit-thresh", type=float, default=0.0,
                   help="Spacepoint is assigned to query iff its mask logit "
                        "from that query exceeds this (default 0 ⇒ "
                        "sigmoid > 0.5)")
    p.add_argument("--active-prob-thresh", type=float, default=0.5,
                   help="Query is 'active' iff its argmax-class softmax "
                        "probability > this AND the argmax class is not "
                        "no_object (default 0.5)")
    p.add_argument("--cap-spacepoints", type=int, default=0,
                   help="Override cfg.data.test.max_spacepoints. "
                        "0 = full events (None); >0 = use that cap")
    p.add_argument("--lm-score-threshold", type=float, default=None,
                   help="Override the larmatch-score floor for the test "
                        "dataset (cfg.data.test.lm_score_val_threshold). "
                        "Default 0.15 lets noisy points through. Try 0.30 "
                        "or 0.50 for stricter ghost rejection at inference.")
    return p.parse_args()


# --------------------------------------------------------------------------
# Model + dataset setup
# --------------------------------------------------------------------------

def load_model_with_checkpoint(cfg, ckpt_path: str, device: str):
    print(f"[infer] building model")
    model = build_model(cfg.model)
    print(f"[infer] loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "state_dict" in ckpt:
        sd = ckpt["state_dict"]
    elif "model" in ckpt:
        sd = ckpt["model"]
    else:
        sd = ckpt
    sd = {k.removeprefix("module."): v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    n_loaded = sum(p.numel() for n, p in model.named_parameters()
                   if n not in missing)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"[infer] params loaded: {n_loaded:,} / {n_total:,} "
          f"({100.0 * n_loaded / max(1, n_total):.1f}%); "
          f"missing={len(missing)} unexpected={len(unexpected)}")
    if len(unexpected) > 0:
        print(f"[infer]   first unexpected: {unexpected[:3]}")
    if len(missing) > 0:
        print(f"[infer]   first missing:    {missing[:3]}")
    model = model.to(device).eval()
    return model


def build_test_dataset(cfg, args):
    dataset_cfg = cfg.data.test.copy()
    if args.data_list:
        dataset_cfg["data_list_file"] = os.path.abspath(args.data_list)
    dataset_cfg["transform"] = None
    if args.cap_spacepoints == 0:
        dataset_cfg["max_spacepoints"] = None
    elif args.cap_spacepoints > 0:
        dataset_cfg["max_spacepoints"] = args.cap_spacepoints
    if args.lm_score_threshold is not None:
        dataset_cfg["lm_score_val_threshold"] = float(args.lm_score_threshold)
        # Also bump aug_low so that if someone runs this on the train split
        # by mistake, the floor is respected.
        if dataset_cfg.get("lm_score_aug_low", 0.15) < args.lm_score_threshold:
            dataset_cfg["lm_score_aug_low"] = float(args.lm_score_threshold)
        print(f"[infer] overriding lm_score threshold to "
              f"{args.lm_score_threshold:.3f}")
    return DATASETS.build(dataset_cfg)


# --------------------------------------------------------------------------
# Move-to-device helpers (custom collate yields nested lists)
# --------------------------------------------------------------------------

def move_batch_to_device(batch: dict, device: str) -> dict:
    for k, v in list(batch.items()):
        if isinstance(v, torch.Tensor):
            batch[k] = v.to(device, non_blocking=True)
    for k in ("fragment_indices_per_event",
              "fragment_trackid_per_event",
              "fragment_pid_per_event",
              "fragment_type_per_event"):
        batch[k] = [
            ([t.to(device, non_blocking=True) for t in v]
             if isinstance(v, list)
             else v.to(device, non_blocking=True))
            for v in batch[k]
        ]
    new_gt = []
    for ev_gts in batch["gt_instances_per_event"]:
        ev_out = []
        for g in ev_gts:
            g2 = dict(g)
            ti = g2.get("truth_indices")
            if isinstance(ti, np.ndarray):
                g2["truth_indices"] = torch.as_tensor(
                    ti, dtype=torch.long, device=device,
                )
            elif isinstance(ti, torch.Tensor):
                g2["truth_indices"] = ti.to(device)
            ev_out.append(g2)
        new_gt.append(ev_out)
    batch["gt_instances_per_event"] = new_gt
    return batch


# --------------------------------------------------------------------------
# Per-event prediction extraction
# --------------------------------------------------------------------------

def compute_per_event_outputs(
    model,
    decoder_out: dict,
    gt_instances: list,
    voxel_id_local: torch.Tensor,
    n_voxels: int,
    n_sp: int,
    fragment_trackid: torch.Tensor,
    coord_center: np.ndarray,
    coord_scale: float,
    mask_logit_thresh: float,
    active_prob_thresh: float,
) -> dict:
    """Compute everything needed by the visualizer for one event."""
    device = decoder_out["final"]["class_logits"].device
    final = decoder_out["final"]
    cls_logits = final["class_logits"]                   # (Q, C)
    origin_pred_norm = final["origin"]                    # (Q, 3)
    sp_mask_logits = final["mask_logits"]["spacepoint"]   # (Q, N)
    v_mask_logits = final["mask_logits"]["voxel"]         # (Q, V)
    f_mask_logits = final["mask_logits"]["fragment"]      # (Q, F)

    Q, C = cls_logits.shape
    no_object_id = int(model.loss_fn.no_object_class_id)

    # Class predictions
    cls_probs = torch.softmax(cls_logits, dim=-1)
    pred_class = cls_logits.argmax(dim=-1)                                # (Q,)
    pred_class_prob = cls_probs.gather(
        1, pred_class.unsqueeze(-1)).squeeze(-1)                          # (Q,)
    is_active = ((pred_class != no_object_id)
                 & (pred_class_prob > active_prob_thresh))                 # (Q,)

    # Per-spacepoint assignment via argmax over active queries
    sp_query, sp_score = _scale_argmax_assignment(
        sp_mask_logits, is_active, mask_logit_thresh,
    )
    voxel_query, _ = _scale_argmax_assignment(
        v_mask_logits, is_active, mask_logit_thresh,
    )
    fragment_query, _ = _scale_argmax_assignment(
        f_mask_logits, is_active, mask_logit_thresh,
    )

    # Denormalize origins
    cc_t = torch.as_tensor(coord_center, dtype=torch.float32, device=device)
    pred_origin_cm = origin_pred_norm * coord_scale + cc_t

    # Run matcher to get matched pairs + GT tensors (consistent with training).
    loss_dict = model.loss_fn(
        decoder_output=decoder_out,
        gt_instances=gt_instances,
        voxel_id=voxel_id_local,
        n_voxels=n_voxels,
        n_spacepoints=n_sp,
        fragment_trackid=fragment_trackid,
        return_matching=True,
    )
    q_idx = loss_dict["q_idx"]
    k_idx = loss_dict["k_idx"]
    gt_classes_t = loss_dict["gt_classes"]
    gt_origin_norm_t = loss_dict["gt_origin"]
    gt_truth_indices = loss_dict["gt_truth_indices"]

    # Per-pair full-spacepoint IoU + cls match + origin error in cm
    P = len(q_idx)
    pred_bool = sp_mask_logits > mask_logit_thresh                        # (Q, N)
    iou = np.zeros(P, dtype=np.float32)
    cls_match = np.zeros(P, dtype=np.int8)
    origin_err_cm = np.zeros(P, dtype=np.float32)
    for p in range(P):
        q = int(q_idx[p])
        k = int(k_idx[p])
        gt_idx = gt_truth_indices[k]
        gt_mask = torch.zeros(n_sp, dtype=torch.bool, device=device)
        gt_mask[gt_idx] = True
        pm = pred_bool[q]
        inter = (pm & gt_mask).sum().item()
        union = (pm | gt_mask).sum().item()
        iou[p] = inter / union if union > 0 else 0.0
        cls_match[p] = int(bool((pred_class[q] == gt_classes_t[k]).item()))
        d_norm = origin_pred_norm[q] - gt_origin_norm_t[k]
        origin_err_cm[p] = float((d_norm * coord_scale).norm().item())

    # Per-spacepoint instance id (-1 if not in any GT instance)
    sp_instance_id = np.full(n_sp, -1, dtype=np.int64)
    for k, idx_t in enumerate(gt_truth_indices):
        sp_instance_id[idx_t.cpu().numpy()] = k

    # Pack GT instances into flat-and-counts arrays
    K = len(gt_instances)
    gt_trunk = np.array([int(g["trunk_trackid"]) for g in gt_instances], dtype=np.int64)
    gt_pid_arr = np.array([int(g["pid"]) for g in gt_instances], dtype=np.int32)
    gt_otype = np.array([int(g["origin_type"]) for g in gt_instances], dtype=np.int32)
    gt_npoints = np.array(
        [int(g["n_truth_points"]) for g in gt_instances], dtype=np.int32,
    )
    if K > 0:
        gt_origin_cm = np.stack([
            np.asarray(g["origin_coord_norm"], dtype=np.float32) * coord_scale
            + coord_center.astype(np.float32)
            for g in gt_instances
        ], axis=0)
    else:
        gt_origin_cm = np.zeros((0, 3), dtype=np.float32)

    return {
        # predictions/
        "cls_logits": cls_logits.detach().cpu().numpy().astype(np.float32),
        "pred_class": pred_class.cpu().numpy().astype(np.int32),
        "pred_class_prob": pred_class_prob.cpu().numpy().astype(np.float32),
        "is_active": is_active.cpu().numpy().astype(np.int8),
        "pred_origin_cm": pred_origin_cm.detach().cpu().numpy().astype(np.float32),
        "spacepoint_query": sp_query.cpu().numpy().astype(np.int32),
        "spacepoint_score": sp_score.cpu().numpy().astype(np.float32),
        "voxel_query": voxel_query.cpu().numpy().astype(np.int32),
        "fragment_query": fragment_query.cpu().numpy().astype(np.int32),
        # matched/
        "q_idx": q_idx.astype(np.int32),
        "k_idx": k_idx.astype(np.int32),
        "iou": iou,
        "cls_match": cls_match,
        "origin_err_cm": origin_err_cm,
        # gt/
        "gt_trunk_trackid": gt_trunk,
        "gt_pid": gt_pid_arr,
        "gt_origin_type": gt_otype,
        "gt_origin_cm": gt_origin_cm,
        "gt_n_truth_points": gt_npoints,
        "gt_instance_id_per_sp": sp_instance_id,
        # meta
        "no_object_class_id": no_object_id,
        "num_queries": Q,
        "num_classes": C,
    }


def _scale_argmax_assignment(
    mask_logits: torch.Tensor,         # (Q, T) where T = N or V or F
    is_active: torch.Tensor,            # (Q,) bool
    logit_thresh: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Assign each token (spacepoint / voxel / fragment) to the active
    query whose mask logit at that token is largest, IF that logit
    exceeds the threshold; otherwise -1. Returns (assignment, score)
    both of length T. Score is sigmoid of the winning logit (0 if none)."""
    Q, T = mask_logits.shape
    device = mask_logits.device
    if T == 0 or not is_active.any():
        return (torch.full((T,), -1, dtype=torch.long, device=device),
                torch.zeros(T, device=device))
    active_q_idx = is_active.nonzero(as_tuple=True)[0]            # (n_active,)
    active_logits = mask_logits[active_q_idx]                      # (n_active, T)
    best_logit, best_active_idx = active_logits.max(dim=0)         # (T,)
    best_q = active_q_idx[best_active_idx]                         # (T,)
    assignment = torch.where(
        best_logit > logit_thresh, best_q, torch.full_like(best_q, -1),
    )
    score = torch.sigmoid(best_logit).where(
        best_logit > logit_thresh, torch.zeros_like(best_logit),
    )
    return assignment, score


# --------------------------------------------------------------------------
# Per-event H5 writer
# --------------------------------------------------------------------------

def write_event_h5(
    out_path: str,
    sample: dict,
    outputs: dict,
    coord_center: np.ndarray,
    coord_scale: float,
    voxel_size_cm: float,
):
    with h5py.File(out_path, "w") as f:
        e = f.create_group("entry_0")
        e.attrs["run"] = int(sample["run"])
        e.attrs["subrun"] = int(sample["subrun"])
        e.attrs["event"] = int(sample["event"])
        e.attrs["name"] = str(sample["name"])
        e.attrs["lm_score_threshold"] = float(sample["lm_score_threshold"])
        e.attrs["coord_center"] = coord_center.astype(np.float32)
        e.attrs["coord_scale"] = float(coord_scale)
        e.attrs["voxel_size_cm"] = float(voxel_size_cm)
        e.attrs["n_spacepoints"] = int(sample["n_spacepoints"])
        e.attrs["n_voxels"] = int(sample["n_voxels"])
        e.attrs["n_fragments"] = int(sample["n_fragments"])

        # ---- per-spacepoint ----
        td = e.create_group("triplet_data")
        td.create_dataset("coord", data=sample["coord"].astype(np.float32))
        td.create_dataset("trackid", data=sample["trackid"].astype(np.int64))
        td.create_dataset("pid", data=sample["pid"].astype(np.int32))
        td.create_dataset("origin", data=sample["origin_label"].astype(np.int32))
        td.create_dataset("ssnet_label",
                          data=sample["ssnet_label"].astype(np.int32))
        td.create_dataset("hasmatch", data=sample["hasmatch"].astype(np.int8))
        td.create_dataset("lm_score", data=sample["lm_score"].astype(np.float32))
        td.create_dataset("voxel_id", data=sample["voxel_id"].astype(np.int64))
        # voxel_keys: integer grid coords (V, 3); the visualizer reconstructs
        # voxel centers in cm via (voxel_keys + 0.5) * voxel_size_cm + center.
        td.create_dataset("voxel_keys",
                          data=sample["voxel_keys"].astype(np.int64))

        # ---- shower fragments (copied from input — useful for fragment-level
        #      coloring later) ----
        sf = e.create_group("shower_fragments")
        sf.attrs["num_fragments"] = int(sample["n_fragments"])
        if sample["n_fragments"] > 0:
            flat = np.concatenate([np.asarray(idx, dtype=np.int64)
                                   for idx in sample["fragment_indices"]])
            counts = np.array([len(idx) for idx in sample["fragment_indices"]],
                              dtype=np.int64)
        else:
            flat = np.zeros(0, dtype=np.int64)
            counts = np.zeros(0, dtype=np.int64)
        sf.create_dataset("pointindices_flat", data=flat)
        sf.create_dataset("pointindices_counts", data=counts)
        sf.create_dataset("trackid",
                          data=sample["fragment_trackid"].astype(np.int64))
        sf.create_dataset("pid",
                          data=sample["fragment_pid"].astype(np.int32))
        sf.create_dataset("type",
                          data=sample["fragment_type"].astype(np.int32))

        # ---- gt instances ----
        gi = e.create_group("gt_instances")
        gi.attrs["num_instances"] = int(sample["n_gt_instances"])
        gi.create_dataset("trunk_trackid", data=outputs["gt_trunk_trackid"])
        gi.create_dataset("pid", data=outputs["gt_pid"])
        gi.create_dataset("origin_type", data=outputs["gt_origin_type"])
        gi.create_dataset("origin_cm", data=outputs["gt_origin_cm"])
        gi.create_dataset("n_truth_points", data=outputs["gt_n_truth_points"])
        td.create_dataset("gt_instance_id",
                          data=outputs["gt_instance_id_per_sp"].astype(np.int32))

        # ---- predictions ----
        pr = e.create_group("predictions")
        pr.attrs["num_queries"] = int(outputs["num_queries"])
        pr.attrs["num_classes"] = int(outputs["num_classes"])
        pr.attrs["no_object_class_id"] = int(outputs["no_object_class_id"])
        pr.create_dataset("cls_logits", data=outputs["cls_logits"])
        pr.create_dataset("pred_class", data=outputs["pred_class"])
        pr.create_dataset("pred_class_prob", data=outputs["pred_class_prob"])
        pr.create_dataset("is_active", data=outputs["is_active"])
        pr.create_dataset("pred_origin_cm", data=outputs["pred_origin_cm"])
        pr.create_dataset("spacepoint_query", data=outputs["spacepoint_query"])
        pr.create_dataset("spacepoint_score", data=outputs["spacepoint_score"])
        pr.create_dataset("voxel_query", data=outputs["voxel_query"])
        pr.create_dataset("fragment_query", data=outputs["fragment_query"])

        # ---- matched ----
        m = pr.create_group("matched")
        m.attrs["num_matched"] = int(len(outputs["q_idx"]))
        m.create_dataset("q_idx", data=outputs["q_idx"])
        m.create_dataset("k_idx", data=outputs["k_idx"])
        m.create_dataset("iou", data=outputs["iou"])
        m.create_dataset("cls_match", data=outputs["cls_match"])
        m.create_dataset("origin_err_cm", data=outputs["origin_err_cm"])


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    cfg = Config.fromfile(args.config)
    coord_center = np.asarray(cfg.coord_center, dtype=np.float32)
    coord_scale = float(cfg.coord_scale)
    voxel_size_cm = float(getattr(cfg, "voxel_size_cm", 5.0))

    dataset = build_test_dataset(cfg, args)
    n_total = len(dataset)
    if n_total == 0:
        print(f"[infer] ERROR: dataset is empty (data_list empty?)",
              file=sys.stderr)
        sys.exit(2)

    # Pick which events to process
    if args.max_events is not None and args.max_events < n_total:
        if args.shuffle:
            order = np.random.permutation(n_total)[:args.max_events]
        else:
            order = np.arange(args.max_events)
    else:
        if args.shuffle:
            order = np.random.permutation(n_total)
        else:
            order = np.arange(n_total)
    print(f"[infer] dataset has {n_total} events; processing {len(order)}")

    # Build model + load checkpoint
    model = load_model_with_checkpoint(cfg, args.checkpoint, args.device)

    os.makedirs(args.output_dir, exist_ok=True)

    for i, ev_idx in enumerate(order):
        sample = dataset[int(ev_idx)]
        batch = shower_clustering_collate([sample])
        batch = move_batch_to_device(batch, args.device)

        with torch.no_grad():
            # Forward in eval mode → predictions list (single event)
            output = model(batch)
            decoder_outs = output["predictions"]
            assert len(decoder_outs) == 1, "single-event batch expected"
            decoder_out = decoder_outs[0]

            # Per-event slice info from the model's bookkeeping
            events = model._per_event_slices(batch)
            e = events[0]
            voxel_id_local = (
                batch["voxel_id"][e["sp_slice"]] - e["voxel_id_offset"]
            )
            fragment_trackid = e["fragment_trackid"]
            if not isinstance(fragment_trackid, torch.Tensor):
                fragment_trackid = torch.as_tensor(
                    fragment_trackid, dtype=torch.long, device=args.device,
                )
            else:
                fragment_trackid = fragment_trackid.to(args.device)
            gt_instances = e["gt_instances"]

            outputs = compute_per_event_outputs(
                model=model,
                decoder_out=decoder_out,
                gt_instances=gt_instances,
                voxel_id_local=voxel_id_local,
                n_voxels=e["n_vox"],
                n_sp=e["n_sp"],
                fragment_trackid=fragment_trackid,
                coord_center=coord_center,
                coord_scale=coord_scale,
                mask_logit_thresh=args.mask_logit_thresh,
                active_prob_thresh=args.active_prob_thresh,
            )

        # Output filename: showerinference_<basename>.h5 (basename without .h5)
        in_name = sample["name"]
        if in_name.endswith(".h5"):
            in_name = in_name[:-3]
        out_path = os.path.join(args.output_dir, f"showerinference_{in_name}.h5")
        write_event_h5(out_path, sample, outputs,
                       coord_center, coord_scale, voxel_size_cm)

        n_active = int(outputs["is_active"].sum())
        K = int(sample["n_gt_instances"])
        P = int(len(outputs["q_idx"]))
        iou_mean = float(outputs["iou"].mean()) if P > 0 else 0.0
        print(f"[infer] {i+1}/{len(order)}  ev_idx={int(ev_idx)}  "
              f"n_sp={sample['n_spacepoints']:,}  "
              f"GT={K} active_q={n_active} matched={P}  "
              f"iou_mean={iou_mean:.3f}  ->  {os.path.basename(out_path)}")

    print(f"\n[infer] done. {len(order)} event H5 files in {args.output_dir}")


if __name__ == "__main__":
    main()
