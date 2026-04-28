"""
Steps 5-7 of the shower origin reconstruction pipeline.

For each input merged H5 file (produced by Steps 1-4):
  Step 5: Run shower origin model inference on all fragments.
  Step 6: Merge fragments into reconstructed particle showers.
  Step 7: Write results to a ROOT TTree via uproot.

Reuses inference functions from run_shower_origin_inference.py.

Usage:
    python tools/run_shower_origin_pipeline_step567.py \
        -c configs/lartpc/shower-origin-sonata-v1m1-v3-reco-fragments-p1cmp075.py \
        --checkpoint path/to/model.pth \
        --input-h5 merged_showerorigin_entry0042.h5 \
        --output-dir /tmp/showerreco_output \
        --device cuda

    # Or process multiple files:
    python tools/run_shower_origin_pipeline_step567.py \
        -c configs/lartpc/... \
        --checkpoint path/to/model.pth \
        --input-list filelist.txt \
        --output-dir /tmp/showerreco_output
"""

import os
import sys
import argparse
import warnings
from collections import defaultdict

import numpy as np
import h5py

# Ensure project root is on path
_project_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, _project_root)

import torch

from pointcept.utils.config import Config

# Reuse inference functions from the existing script
from tools.run_shower_origin_inference import (
    build_dataset,
    build_inference_transforms,
    load_model,
    run_event_inference,
)

from lartpc_data_prep.shower_fragment_merger import (
    FragmentInfo,
    MergerConfig,
    merge_fragments,
)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Steps 5-7: inference + fragment merging + ROOT output"
    )
    p.add_argument("-c", "--config", required=True, help="Model config file")
    p.add_argument("--checkpoint", required=True, help="Model checkpoint")
    p.add_argument("--input-h5", default=None,
                   help="Single merged H5 file to process")
    p.add_argument("--input-list", default=None,
                   help="Text file with one H5 path per line")
    p.add_argument("--output-dir", required=True, help="ROOT output directory")
    p.add_argument("--split", default="val", choices=["train", "val"],
                   help="Dataset split (default: val)")
    p.add_argument("--device", default=None, help="cuda / cpu")

    # Merger parameters
    p.add_argument("--origin-proximity-radius", type=float, default=5.0,
                   help="cm (default: 5.0)")
    p.add_argument("--cone-half-angle", type=float, default=20.0,
                   help="degrees (default: 20.0)")
    p.add_argument("--cone-max-length", type=float, default=300.0,
                   help="cm (default: 300.0)")
    p.add_argument("--min-points-fraction-in-cone", type=float, default=0.5,
                   help="0-1 (default: 0.5)")
    p.add_argument("--combined-output", default=None,
                   help="If set, write all events into a single ROOT file at "
                        "this path instead of one ROOT file per input H5.")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Coordinate helpers
# ---------------------------------------------------------------------------

def denormalize_coords(coords_norm, center, scale):
    """Convert normalized model coords back to detector cm."""
    return coords_norm * scale + np.array(center, dtype=np.float32)


def get_coord_normalization(cfg, split):
    """Extract NormalizeShowerCoords params from config."""
    center = [125.0, 0.0, 518.0]
    scale = 179.55
    val_transforms = getattr(cfg.data, split).get("transform", [])
    for t in val_transforms:
        if t.get("type") == "NormalizeShowerCoords":
            center = list(t.get("center", center))
            scale = float(t.get("scale", scale))
            break
    return np.array(center, dtype=np.float32), float(scale)


# ---------------------------------------------------------------------------
# H5 metadata reader
# ---------------------------------------------------------------------------

def read_h5_fragment_metadata(h5_path, min_fragment_points=20,
                              include_cosmic_showers=True):
    """Read per-fragment metadata from H5 that get_all_fragments() doesn't
    return: istrunk, pret0shiftedoriginpt, and (run, subrun, event).

    Returns a dict with:
      - 'fragment_meta': list of dicts (one per valid fragment, same order as
        get_all_fragments()) with keys: h5_frag_idx, istrunk,
        pret0shiftedoriginpt, trackid, n_points_h5
      - 'all_truth_fragments': list of dicts for ALL truth fragments (for
        computing missed showers)
      - 'run', 'subrun', 'event': int or -1 if not stored
    """
    fragment_meta = []
    all_truth_fragments = []

    with h5py.File(h5_path, "r") as f:
        entry = f["entry_0"]

        # Read RSE if available
        run = int(entry.attrs.get("run", -1))
        subrun = int(entry.attrs.get("subrun", -1))
        event = int(entry.attrs.get("event", -1))

        if "shower_fragments" not in entry:
            return {
                "fragment_meta": [],
                "all_truth_fragments": [],
                "run": run, "subrun": subrun, "event": event,
            }

        sf = entry["shower_fragments"]
        num_frags = int(sf.attrs.get("num_fragments", 0))
        if num_frags == 0:
            return {
                "fragment_meta": [],
                "all_truth_fragments": [],
                "run": run, "subrun": subrun, "event": event,
            }

        trackids = sf["trackid"][:]
        pids = sf["pid"][:]
        istrunk_arr = sf["istrunk"][:]
        frag_types = sf["type"][:]
        index_counts = sf["pointindices_counts"][:]

        has_pret0 = "pret0shiftedoriginpt" in sf
        if has_pret0:
            pret0 = sf["pret0shiftedoriginpt"][:]
        else:
            pret0 = np.zeros((num_frags, 4), dtype=np.float32)

        # Ghost filtering — need to know effective point counts after
        # ghost removal, matching the dataset's logic
        triplet = entry["triplet_data"]
        n_points_orig = triplet["pos"].shape[0]
        ghost_mask = None
        if "hasmatch" in triplet:
            hasmatch = triplet["hasmatch"][:].astype(np.int64)
            ghost_mask = (hasmatch == 1)
            index_remap = np.full(n_points_orig, -1, dtype=np.int64)
            n_points = int(ghost_mask.sum())
            index_remap[ghost_mask] = np.arange(n_points)
        else:
            n_points = n_points_orig
            index_remap = None

        flat_indices = sf["pointindices_flat"][:]

        offset = 0
        for i in range(num_frags):
            count = int(index_counts[i])
            indices = flat_indices[offset:offset + count]
            offset += count

            # Compute effective point count after ghost filtering
            if index_remap is not None:
                valid_idx = indices[indices < n_points_orig]
                remapped = index_remap[valid_idx]
                eff_count = int((remapped >= 0).sum())
            else:
                valid_idx = indices[indices < n_points]
                eff_count = len(valid_idx)

            frag_type = int(frag_types[i])

            # Store all truth fragment info (for missed shower computation)
            all_truth_fragments.append({
                "h5_frag_idx": i,
                "trackid": int(trackids[i]),
                "pid": int(pids[i]),
                "istrunk": int(istrunk_arr[i]),
                "type": frag_type,
                "n_points": eff_count,
                "pret0shiftedoriginpt": pret0[i].copy(),
            })

            # Match the filtering logic in get_all_fragments()
            if eff_count < min_fragment_points:
                continue
            if not include_cosmic_showers and frag_type == 2:
                continue

            fragment_meta.append({
                "h5_frag_idx": i,
                "istrunk": int(istrunk_arr[i]),
                "pret0shiftedoriginpt": pret0[i].copy(),
                "trackid": int(trackids[i]),
                "n_points_h5": eff_count,
            })

    return {
        "fragment_meta": fragment_meta,
        "all_truth_fragments": all_truth_fragments,
        "run": run, "subrun": subrun, "event": event,
    }


# ---------------------------------------------------------------------------
# Per-event processing
# ---------------------------------------------------------------------------

def process_event(
    dataset,
    event_idx: int,
    transform,
    model,
    device,
    merger_config: MergerConfig,
    coord_center: np.ndarray,
    coord_scale: float,
) -> dict | None:
    """Process one event through steps 5-7.

    Returns a dict with all data needed for ROOT output, or None on error.
    """
    # Get H5 path for this event
    file_idx = dataset.event_index[event_idx % len(dataset.event_index)]
    h5_path = dataset.data_list[file_idx]

    # Step 5: Run inference
    try:
        results = run_event_inference(
            dataset, event_idx, transform, model, device
        )
    except Exception as e:
        warnings.warn(f"Event {event_idx} inference failed: {e}")
        return None

    if not results:
        return None

    # Read H5 metadata (istrunk, pret0shiftedoriginpt, RSE)
    h5_meta = read_h5_fragment_metadata(h5_path)
    frag_meta_list = h5_meta["fragment_meta"]

    # Sanity check: inference results and H5 metadata should have same count
    if len(results) != len(frag_meta_list):
        warnings.warn(
            f"Event {event_idx}: inference returned {len(results)} fragments "
            f"but H5 has {len(frag_meta_list)} valid fragments. Skipping."
        )
        return None

    # Build FragmentInfo list
    fragments = []
    for i, (result, meta) in enumerate(zip(results, frag_meta_list)):
        # Denormalize model output coordinates
        pred_coords_cm = denormalize_coords(
            result["pred_coords"], coord_center, coord_scale
        )
        pred_scores = result["pred_scores"]

        # Predicted origin = peak score location in cm
        peak_idx = int(np.argmax(pred_scores))
        pred_origin_cm = pred_coords_cm[peak_idx]

        # Raw spacepoints for this fragment (already in cm via coord_scale=1)
        shower_mask = result["shower_mask"].astype(bool)
        raw_coords_cm = result["coord"][shower_mask]

        frag = FragmentInfo(
            fragment_idx=i,
            pred_origin_cm=pred_origin_cm,
            pred_class=int(result["pred_class"]),
            pred_scores=pred_scores,
            pred_coords_cm=pred_coords_cm,
            raw_coords_cm=raw_coords_cm,
            trackid=int(result["trackid"]),
            pid=int(result["pid"]),
            gt_origin_cm=result["origin_coord"].copy(),
            gt_origin_type=int(result["origin_type"]),
            gt_pret0_origin=meta["pret0shiftedoriginpt"],
            istrunk=meta["istrunk"],
            n_points_raw=int(shower_mask.sum()),
        )
        fragments.append(frag)

    # Step 6: Merge fragments
    merged_showers, unmerged_indices = merge_fragments(fragments, merger_config)

    # Build fragment → shower_id mapping
    frag_to_shower = {}
    for shower in merged_showers:
        for fi in shower.fragment_indices:
            frag_to_shower[fi] = shower.shower_id

    # Compute missed true showers
    reco_trackids = {f.trackid for f in fragments}
    missed_true = []
    for tfrag in h5_meta["all_truth_fragments"]:
        if tfrag["trackid"] not in reco_trackids and tfrag["n_points"] > 0:
            missed_true.append(tfrag)

    # Assemble output
    return {
        "h5_path": h5_path,
        "run": h5_meta["run"],
        "subrun": h5_meta["subrun"],
        "event": h5_meta["event"],
        "fragments": fragments,
        "merged_showers": merged_showers,
        "unmerged_indices": unmerged_indices,
        "frag_to_shower": frag_to_shower,
        "missed_true": missed_true,
    }


# ---------------------------------------------------------------------------
# ROOT TTree output via uproot
# ---------------------------------------------------------------------------

def write_root_output(event_results: list[dict], output_path: str):
    """Write all event results to a ROOT file with a single TTree.

    Uses uproot + awkward for variable-length arrays.
    """
    import uproot
    import awkward as ak

    if not event_results:
        print("No results to write.")
        return

    # Collect per-event data
    data = defaultdict(list)

    for ev in event_results:
        data["run"].append(np.int32(ev["run"]))
        data["subrun"].append(np.int32(ev["subrun"]))
        data["event"].append(np.int32(ev["event"]))

        frags = ev["fragments"]
        showers = ev["merged_showers"]
        missed = ev["missed_true"]
        frag_to_shower = ev["frag_to_shower"]

        data["n_reco_fragments"].append(np.int32(len(frags)))
        data["n_merged_showers"].append(np.int32(len(showers)))
        data["n_missed_true"].append(np.int32(len(missed)))

        # Per reco fragment arrays
        data["frag_cluster_id"].append(
            np.array([f.fragment_idx for f in frags], dtype=np.int32))
        data["frag_n_points"].append(
            np.array([f.n_points_raw for f in frags], dtype=np.int32))
        data["frag_pred_origin_x"].append(
            np.array([f.pred_origin_cm[0] for f in frags], dtype=np.float32))
        data["frag_pred_origin_y"].append(
            np.array([f.pred_origin_cm[1] for f in frags], dtype=np.float32))
        data["frag_pred_origin_z"].append(
            np.array([f.pred_origin_cm[2] for f in frags], dtype=np.float32))
        data["frag_gt_origin_x"].append(
            np.array([f.gt_origin_cm[0] for f in frags], dtype=np.float32))
        data["frag_gt_origin_y"].append(
            np.array([f.gt_origin_cm[1] for f in frags], dtype=np.float32))
        data["frag_gt_origin_z"].append(
            np.array([f.gt_origin_cm[2] for f in frags], dtype=np.float32))
        data["frag_gt_origin_t"].append(
            np.array([f.gt_pret0_origin[3] for f in frags], dtype=np.float32))
        data["frag_gt_class"].append(
            np.array([f.gt_origin_type for f in frags], dtype=np.int32))
        data["frag_pred_class"].append(
            np.array([f.pred_class for f in frags], dtype=np.int32))
        data["frag_geant_trackid"].append(
            np.array([f.trackid for f in frags], dtype=np.int32))
        data["frag_is_trunk"].append(
            np.array([f.istrunk for f in frags], dtype=np.int32))
        data["frag_merged_shower_id"].append(
            np.array([frag_to_shower.get(f.fragment_idx, -1)
                       for f in frags], dtype=np.int32))

        # Per merged shower arrays
        data["shower_id"].append(
            np.array([s.shower_id for s in showers], dtype=np.int32))
        data["shower_n_fragments"].append(
            np.array([len(s.fragment_indices) for s in showers],
                      dtype=np.int32))
        data["shower_n_total_points"].append(
            np.array([sum(frags[fi].n_points_raw for fi in s.fragment_indices)
                       for s in showers], dtype=np.int32))
        data["shower_pred_origin_x"].append(
            np.array([s.predicted_origin_cm[0] for s in showers],
                      dtype=np.float32))
        data["shower_pred_origin_y"].append(
            np.array([s.predicted_origin_cm[1] for s in showers],
                      dtype=np.float32))
        data["shower_pred_origin_z"].append(
            np.array([s.predicted_origin_cm[2] for s in showers],
                      dtype=np.float32))
        data["shower_start_x"].append(
            np.array([s.start_point_cm[0] for s in showers],
                      dtype=np.float32))
        data["shower_start_y"].append(
            np.array([s.start_point_cm[1] for s in showers],
                      dtype=np.float32))
        data["shower_start_z"].append(
            np.array([s.start_point_cm[2] for s in showers],
                      dtype=np.float32))
        data["shower_axis_x"].append(
            np.array([s.shower_axis[0] for s in showers], dtype=np.float32))
        data["shower_axis_y"].append(
            np.array([s.shower_axis[1] for s in showers], dtype=np.float32))
        data["shower_axis_z"].append(
            np.array([s.shower_axis[2] for s in showers], dtype=np.float32))
        data["shower_pred_class"].append(
            np.array([s.pred_class for s in showers], dtype=np.int32))

        # Per missed true shower arrays
        data["missed_trackid"].append(
            np.array([m["trackid"] for m in missed], dtype=np.int32))
        data["missed_n_points"].append(
            np.array([m["n_points"] for m in missed], dtype=np.int32))
        data["missed_gt_origin_x"].append(
            np.array([m["pret0shiftedoriginpt"][0] for m in missed],
                      dtype=np.float32))
        data["missed_gt_origin_y"].append(
            np.array([m["pret0shiftedoriginpt"][1] for m in missed],
                      dtype=np.float32))
        data["missed_gt_origin_z"].append(
            np.array([m["pret0shiftedoriginpt"][2] for m in missed],
                      dtype=np.float32))
        data["missed_gt_origin_t"].append(
            np.array([m["pret0shiftedoriginpt"][3] for m in missed],
                      dtype=np.float32))
        data["missed_gt_class"].append(
            np.array([m["type"] for m in missed], dtype=np.int32))
        data["missed_is_trunk"].append(
            np.array([m["istrunk"] for m in missed], dtype=np.int32))

    # Convert variable-length lists to awkward arrays for uproot
    tree_data = {}
    for key, values in data.items():
        if isinstance(values[0], (np.int32, np.int64, int)):
            tree_data[key] = np.array(values)
        elif isinstance(values[0], np.ndarray):
            tree_data[key] = ak.Array(values)
        else:
            tree_data[key] = np.array(values)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    with uproot.recreate(output_path) as f:
        # Build type-only spec so mktree creates an empty TTree,
        # then extend fills it once.  Direct assignment defaults to
        # RNTuple in uproot 5.x which chokes on some awkward layouts.
        type_spec = {}
        for key, arr in tree_data.items():
            if isinstance(arr, ak.Array):
                type_spec[key] = ak.type(arr)
            else:
                type_spec[key] = arr.dtype
        f.mktree("shower_reco", type_spec)
        f["shower_reco"].extend(tree_data)

    print(f"Wrote {len(event_results)} events to {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # Resolve input files
    if args.input_h5:
        input_files = [os.path.abspath(args.input_h5)]
    elif args.input_list:
        with open(args.input_list) as fh:
            input_files = [
                line.strip() for line in fh if line.strip()
            ]
    else:
        print("ERROR: Provide --input-h5 or --input-list")
        sys.exit(1)

    print(f"Processing {len(input_files)} input H5 file(s)")

    # Load config and model
    cfg = Config.fromfile(args.config)
    coord_center, coord_scale = get_coord_normalization(cfg, args.split)
    print(f"Coord normalization: center={coord_center}, scale={coord_scale}")

    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    model = load_model(cfg, args.checkpoint, device)
    transform = build_inference_transforms(cfg, args.split)

    merger_config = MergerConfig(
        origin_proximity_radius=args.origin_proximity_radius,
        cone_half_angle=args.cone_half_angle,
        cone_max_length=args.cone_max_length,
        min_points_fraction_in_cone=args.min_points_fraction_in_cone,
    )
    print(f"Merger config: {merger_config}")

    os.makedirs(args.output_dir, exist_ok=True)

    # When --combined-output is set, accumulate all results and write once
    all_results = [] if args.combined_output else None

    # Process each input file
    for h5_path in input_files:
        h5_path = os.path.abspath(h5_path)
        if not os.path.exists(h5_path):
            warnings.warn(f"File not found: {h5_path}")
            continue

        print(f"\n--- Processing: {os.path.basename(h5_path)} ---")

        # Build a one-file dataset
        dataset = build_dataset(cfg, args.split, data_list_override=None)
        # Override the data list to just this one file
        dataset.data_list = [h5_path]
        # Rebuild event_index for this single file
        dataset.event_index = []
        try:
            with h5py.File(h5_path, "r") as f:
                if "entry_0" in f and "shower_fragments" in f["entry_0"]:
                    sf = f["entry_0"]["shower_fragments"]
                    num_frags = int(sf.attrs.get("num_fragments", 0))
                    if num_frags > 0:
                        dataset.event_index = [0]
        except Exception as e:
            warnings.warn(f"Cannot read {h5_path}: {e}")
            continue

        if not dataset.event_index:
            print(f"  No shower fragments found, skipping.")
            continue

        # Process the single event in this file
        result = process_event(
            dataset=dataset,
            event_idx=0,
            transform=transform,
            model=model,
            device=device,
            merger_config=merger_config,
            coord_center=coord_center,
            coord_scale=coord_scale,
        )

        if result is None:
            print(f"  Processing failed, skipping.")
            continue

        # Summary
        n_frags = len(result["fragments"])
        n_showers = len(result["merged_showers"])
        n_unmerged = len(result["unmerged_indices"])
        n_missed = len(result["missed_true"])
        print(f"  Fragments: {n_frags}, Merged showers: {n_showers}, "
              f"Unmerged: {n_unmerged}, Missed true: {n_missed}")

        if all_results is not None:
            all_results.append(result)
        else:
            # Write ROOT output — one file per input H5
            basename = os.path.splitext(os.path.basename(h5_path))[0]
            root_path = os.path.join(args.output_dir, f"showerreco_{basename}.root")
            write_root_output([result], root_path)

    # Write combined ROOT file with all events
    if all_results is not None:
        if all_results:
            combined_path = args.combined_output
            os.makedirs(os.path.dirname(combined_path) or ".", exist_ok=True)
            write_root_output(all_results, combined_path)
        else:
            print("No events produced results; combined ROOT file not written.")

    print("\nDone.")


if __name__ == "__main__":
    main()
