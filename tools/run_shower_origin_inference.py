"""
Batch shower origin inference script.

Runs inference on all shower fragments across an entire dataset and saves
per-fragment results to an HDF5 file for downstream analysis with pandas.

Supports both the original 3-class model (INSIDE, OUTSIDE, ON_TRACK) and
the reco 5-class model (INSIDE, OUTSIDE, COSMIC_INSIDE, GHOST, TRUE_TRACK).
For reco classes, ghost (type=3) and true_track (type=4) fragments have no
meaningful ground truth origin, so GT-based distance metrics are set to NaN.

Usage:
    python tools/run_shower_origin_inference.py \
        -c configs/lartpc/shower_origin/archive/shower-origin-sonata-v1m1-v3-reco-fragments-p1cmp075.py \
        --checkpoint path/to/model.pth \
        --data-list filelist.txt \
        --output results.h5 \
        --split val \
        --device cuda

Load results:
    import h5py, pandas as pd
    with h5py.File("results.h5", "r") as f:
        df = pd.DataFrame({k: f[k][:] for k in f.keys()})
"""

import os
import sys
import argparse
import warnings

import numpy as np
import h5py
import torch
from tqdm import tqdm

# Ensure project root is on path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from pointcept.utils.config import Config
from pointcept.datasets.builder import DATASETS
from pointcept.datasets.transform import Compose, TRANSFORMS
from pointcept.datasets.utils import point_collate_fn
from pointcept.models import build_model


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Batch shower origin model inference"
    )
    parser.add_argument(
        "-c", "--config", required=True, type=str, help="Config file path"
    )
    parser.add_argument(
        "--checkpoint", required=True, type=str, help="Model checkpoint path"
    )
    parser.add_argument(
        "--data-list", type=str, default=None,
        help="Override data list file path"
    )
    parser.add_argument(
        "--output", type=str, default="results.h5",
        help="Output HDF5 file path (default: results.h5)"
    )
    parser.add_argument(
        "--split", type=str, default="val", choices=["train", "val"],
        help="Dataset split (default: val)"
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device for inference (default: cuda if available, else cpu)"
    )
    return parser.parse_args()


# ── Reused from visualize_shower_origin_inference.py ─────────────────────────

def build_dataset(cfg, split, data_list_override=None):
    """Build dataset with no transforms (applied manually per fragment)."""
    dataset_cfg = getattr(cfg.data, split).copy()
    if data_list_override is not None:
        dataset_cfg["data_list_file"] = os.path.abspath(data_list_override)
    dataset_cfg["transform"] = []
    dataset = DATASETS.build(dataset_cfg)
    return dataset


def build_inference_transforms(cfg, split):
    """Build the full val transform pipeline from config."""
    dataset_cfg = getattr(cfg.data, split)
    transform_list = dataset_cfg.get("transform", [])
    if not transform_list:
        return None
    comp = Compose(cfg=None)
    comp.transforms = [TRANSFORMS.build(t) for t in transform_list]
    return comp


def load_model(cfg, checkpoint_path, device):
    """Load model from checkpoint with DDP prefix stripping."""
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint

    state_dict = {k.removeprefix("module."): v for k, v in state_dict.items()}

    model = build_model(cfg.model)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"Missing keys ({len(missing)}): {missing[:5]}...")
    if unexpected:
        print(f"Unexpected keys ({len(unexpected)}): {unexpected[:5]}...")
    print("Model loaded successfully")

    model.to(device).eval()
    return model


def run_event_inference(dataset, event_idx, transform, model, device):
    """
    Run inference on all fragments in an event.

    Returns list of dicts per fragment with GT metadata + model predictions.
    Includes trackid and pid for merge keys.
    """
    raw_fragments = dataset.get_all_fragments(event_idx)
    if not raw_fragments:
        return []

    # Save GT metadata and apply transforms
    gt_metadata = []
    transformed = []
    for frag in raw_fragments:
        gt_metadata.append({
            "coord": frag["coord"].copy(),
            "shower_mask": frag["shower_mask"].copy(),
            "origin_coord": frag["origin_coord"].copy(),
            "origin_type": int(frag["origin_type"]),
            "start_coord": frag["start_coord"].copy(),
            "trackid": frag["trackid"],
            "pid": frag["pid"],
            "name": frag["name"],
            "n_points_raw": frag["coord"].shape[0],
        })
        if transform is not None:
            t_frag = transform(frag)
        else:
            t_frag = frag
        transformed.append(t_frag)

    # Batch all fragments
    batched = point_collate_fn(transformed)

    # Move tensors to device
    for k, v in batched.items():
        if isinstance(v, torch.Tensor):
            batched[k] = v.to(device)

    # Forward pass
    with torch.no_grad():
        output = model(batched)

    # Unpack results per fragment using offset
    offset = batched["offset"].cpu()
    n_frags = len(offset)

    # Optional: per-fragment 3D regressed origin (V3 + enable_origin_regression).
    pred_origin_raw = output.get("origin_pred_coord", None)
    all_pred_origins = None
    if pred_origin_raw is not None:
        po = pred_origin_raw.cpu()
        if po.dim() == 1:
            all_pred_origins = [po]
        else:
            all_pred_origins = [po[i] for i in range(po.shape[0])]

    if n_frags == 1:
        all_scores = [output["origin_scores"].cpu()]
        all_coords = [output["all_coords"].cpu()]
        all_classes = [output["origin_class"].cpu().item()]
        all_logits = [output["origin_class_logits"].cpu()]
    else:
        if isinstance(output["origin_scores"], list):
            all_scores = [s.cpu() for s in output["origin_scores"]]
            all_coords = [c.cpu() for c in output["all_coords"]]
        else:
            scores_cat = output["origin_scores"].cpu()
            coords_cat = output["all_coords"].cpu()
            all_scores = []
            all_coords = []
            start = 0
            for i in range(n_frags):
                end = int(offset[i])
                all_scores.append(scores_cat[start:end])
                all_coords.append(coords_cat[start:end])
                start = end

        cls_tensor = output["origin_class"].cpu()
        logits_tensor = output["origin_class_logits"].cpu()
        all_classes = [cls_tensor[i].item() for i in range(n_frags)]
        all_logits = [logits_tensor[i] for i in range(n_frags)]

    # Combine GT + predictions
    results = []
    for i in range(n_frags):
        gt = gt_metadata[i]
        scores_np = all_scores[i].float().numpy()
        coords_np = all_coords[i].float().numpy()
        logits_np = all_logits[i].float().numpy()

        result = {
            **gt,
            "pred_scores": scores_np,
            "pred_coords": coords_np,
            "pred_class": all_classes[i],
            "pred_class_logits": logits_np,
            "n_points_model": coords_np.shape[0],
        }
        if all_pred_origins is not None and i < len(all_pred_origins):
            result["pred_origin_coord"] = all_pred_origins[i].float().numpy()
        results.append(result)

    return results


# ── Per-fragment scalar extraction ────────────────────────────────────────────

def extract_row(result, event_idx, event_file, frag_idx, coord_center, coord_scale):
    """Extract a flat dict of scalars from one fragment result.

    For ghost (gt_origin_type=3) and true_track (gt_origin_type=4) fragments,
    the ground truth origin point is a placeholder (zeros) and has no physical
    meaning. GT-based distance metrics are set to NaN for these fragments.
    """
    origin = np.array(result["origin_coord"], dtype=np.float32)
    start = np.array(result["start_coord"], dtype=np.float32)
    pred_coords = result["pred_coords"]
    pred_scores = result["pred_scores"]
    logits = result["pred_class_logits"]
    gt_origin_type = int(result["origin_type"])

    # Ghost (3) and true_track (4) fragments have no meaningful GT origin.
    # Also treat type=-1 (unknown/reco-only data) the same way.
    has_gt_origin = gt_origin_type not in (-1, 3, 4)

    # Softmax for class probabilities
    probs = np.exp(logits - logits.max())
    probs = probs / probs.sum()

    # Predicted peak
    peak_idx = int(np.argmax(pred_scores))
    peak_coord = pred_coords[peak_idx]
    peak_score = float(pred_scores[peak_idx])

    if has_gt_origin:
        # Normalize GT origin to model coordinate frame for distance computation
        if coord_center is not None and coord_scale is not None:
            center = np.array(coord_center, dtype=np.float32)
            gt_norm = (origin - center) / coord_scale
        else:
            gt_norm = origin

        # GT nearest score: find spacepoint closest to GT origin in normalized coords
        dists = np.linalg.norm(pred_coords - gt_norm[np.newaxis, :], axis=1)
        nearest_idx = int(np.argmin(dists))
        gt_nearest_score = float(pred_scores[nearest_idx])
        gt_nearest_dist = float(dists[nearest_idx])

        # Peak to GT distance (in normalized coords)
        peak_gt_dist = float(np.linalg.norm(peak_coord - gt_norm))
    else:
        # No meaningful GT origin — distance metrics are undefined
        gt_nearest_score = float("nan")
        gt_nearest_dist = float("nan")
        peak_gt_dist = float("nan")

    # Shower point count (raw, before transforms)
    shower_mask = result["shower_mask"]
    n_shower_points = int(np.asarray(shower_mask).astype(bool).sum())

    row = {
        # Index / merge keys
        "event_idx": event_idx,
        "event_file": event_file,
        "fragment_idx": frag_idx,
        "trackid": int(result["trackid"]),
        "pid": int(result["pid"]),
        # Ground truth
        "gt_origin_x": float(origin[0]),
        "gt_origin_y": float(origin[1]),
        "gt_origin_z": float(origin[2]),
        "gt_start_x": float(start[0]),
        "gt_start_y": float(start[1]),
        "gt_start_z": float(start[2]),
        "gt_origin_type": gt_origin_type,
        "n_shower_points": n_shower_points,
        # Model predictions
        "pred_peak_x": float(peak_coord[0]),
        "pred_peak_y": float(peak_coord[1]),
        "pred_peak_z": float(peak_coord[2]),
        "pred_peak_score": peak_score,
        "gt_nearest_score": gt_nearest_score,
        "gt_nearest_dist": gt_nearest_dist,
        "pred_class": int(result["pred_class"]),
        "n_points_model": int(result["n_points_model"]),
        "peak_gt_dist": peak_gt_dist,
    }

    # Class probabilities (variable number of classes)
    for ci in range(len(probs)):
        row[f"class_score_{ci}"] = float(probs[ci])

    return row


# ── HDF5 output ──────────────────────────────────────────────────────────────

def save_results_h5(rows, output_path):
    """Save list of row dicts to HDF5, one dataset per column."""
    if not rows:
        print("No results to save.")
        return

    keys = list(rows[0].keys())
    n = len(rows)

    with h5py.File(output_path, "w") as f:
        for key in keys:
            values = [r[key] for r in rows]
            if isinstance(values[0], str):
                # Variable-length UTF-8 strings
                dt = h5py.string_dtype(encoding="utf-8")
                f.create_dataset(key, data=values, dtype=dt)
            elif isinstance(values[0], int):
                f.create_dataset(key, data=np.array(values, dtype=np.int64))
            elif isinstance(values[0], float):
                f.create_dataset(key, data=np.array(values, dtype=np.float32))
            else:
                f.create_dataset(key, data=np.array(values))

    print(f"Saved {n} rows x {len(keys)} columns to {output_path}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)

    # Detect reco classes from model config
    use_reco_classes = cfg.model.get("use_reco_classes", False)
    if use_reco_classes:
        CLASS_NAMES = {0: "INSIDE", 1: "OUTSIDE", 2: "COSMIC_INSIDE",
                       3: "GHOST", 4: "TRUE_TRACK"}
        # Types with no meaningful GT origin point
        NO_GT_ORIGIN_TYPES = {-1, 3, 4}
    else:
        CLASS_NAMES = {0: "INSIDE", 1: "OUTSIDE", 2: "ON_TRACK"}
        NO_GT_ORIGIN_TYPES = {-1}
    num_classes = len(CLASS_NAMES)
    print(f"Classification mode: {'reco 5-class' if use_reco_classes else 'standard 3-class'}")
    print(f"  Classes: {CLASS_NAMES}")

    # Device
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    # Build dataset (no transforms)
    dataset = build_dataset(cfg, args.split, data_list_override=args.data_list)
    n_entries = len(dataset)
    print(f"Dataset: {n_entries} entries (split={args.split})")
    print(f"  data_list: {len(dataset.data_list)} files")
    print(f"  event_index: {len(dataset.event_index)} events with shower_fragments")

    if n_entries == 0:
        print("ERROR: No events with shower_fragments found!")
        sys.exit(1)

    # Build inference transforms
    transform = build_inference_transforms(cfg, args.split)
    print(f"Inference transforms: {transform}")

    # Load model
    model = load_model(cfg, args.checkpoint, device)

    # Extract coordinate normalization params from transforms
    coord_center = None
    coord_scale = None
    val_transforms = getattr(cfg.data, args.split).get("transform", [])
    for t in val_transforms:
        if t.get("type") == "NormalizeShowerCoords":
            coord_center = list(t.get("center", [125.0, 0.0, 518.0]))
            coord_scale = float(t.get("scale", 179.55))
            break

    print(f"coord_center={coord_center}, coord_scale={coord_scale}")

    # Processing loop
    rows = []
    n_errors = 0
    n_fragments_total = 0

    for event_idx in tqdm(range(n_entries), desc="Events"):
        # Get event file basename
        file_idx = dataset.event_index[event_idx % len(dataset.event_index)]
        event_file = os.path.basename(dataset.data_list[file_idx])

        try:
            results = run_event_inference(
                dataset, event_idx, transform, model, device
            )
        except Exception as e:
            warnings.warn(f"Event {event_idx} ({event_file}) failed: {e}")
            n_errors += 1
            continue

        for frag_idx, r in enumerate(results):
            row = extract_row(
                r, event_idx, event_file, frag_idx,
                coord_center, coord_scale,
            )
            rows.append(row)
            n_fragments_total += 1

    print(f"\nProcessed {n_entries} events: "
          f"{n_fragments_total} fragments, {n_errors} errors")

    # Print summary breakdown by GT class and prediction accuracy
    if rows:
        from collections import Counter
        gt_counts = Counter(r["gt_origin_type"] for r in rows)
        pred_counts = Counter(r["pred_class"] for r in rows)

        print("\n--- GT origin type distribution ---")
        for t in sorted(gt_counts.keys()):
            name = CLASS_NAMES.get(t, f"UNKNOWN({t})")
            print(f"  {name} (type={t}): {gt_counts[t]}")

        print("\n--- Predicted class distribution ---")
        for t in sorted(pred_counts.keys()):
            name = CLASS_NAMES.get(t, f"UNKNOWN({t})")
            print(f"  {name} (class={t}): {pred_counts[t]}")

        # Classification accuracy per GT type (skip types with no GT)
        print("\n--- Classification accuracy by GT type ---")
        for t in sorted(gt_counts.keys()):
            if t in NO_GT_ORIGIN_TYPES:
                name = CLASS_NAMES.get(t, f"UNKNOWN({t})")
                print(f"  {name} (type={t}): N/A (no GT origin)")
                continue
            matching = [r for r in rows if r["gt_origin_type"] == t]
            correct = sum(1 for r in matching if r["pred_class"] == t)
            name = CLASS_NAMES.get(t, f"UNKNOWN({t})")
            acc = correct / len(matching) * 100 if matching else 0
            print(f"  {name} (type={t}): {correct}/{len(matching)} = {acc:.1f}%")

        # Origin score accuracy (only for types with GT origin)
        has_gt = [r for r in rows if r["gt_origin_type"] not in NO_GT_ORIGIN_TYPES]
        if has_gt:
            dists = [r["peak_gt_dist"] for r in has_gt if not np.isnan(r["peak_gt_dist"])]
            if dists:
                print(f"\n--- Origin prediction (types with GT, n={len(dists)}) ---")
                print(f"  Mean peak-to-GT dist: {np.mean(dists):.4f} (normalized)")
                print(f"  Median peak-to-GT dist: {np.median(dists):.4f}")
                within_5cm = sum(1 for d in dists if d < 5.0 / 179.55)
                print(f"  Within 5 cm: {within_5cm}/{len(dists)} = {within_5cm/len(dists)*100:.1f}%")

    # Save
    save_results_h5(rows, args.output)


if __name__ == "__main__":
    main()
