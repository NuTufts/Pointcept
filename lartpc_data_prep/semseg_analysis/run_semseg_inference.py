"""Run the SonataLoRASegmentor (semseg, 8-class) on a filelist and accumulate
per-shard confusion matrices + softmax score histograms.

Per merged H5 file in the assigned slice:

  1. Read pos, hasmatch, origin, ssnet_label, pixval, wire feat directly from
     the H5 (mirrors what LArTPCDataset.get_data does, minus the dataset's
     training-time filtering).
  2. Drop ghost points (hasmatch==0) — matches the training config's
     ``true_points_only=True``.
  3. Map ssnet_label -> 8-class truth via the dataset's SSNETLABEL_TO_CLASS
     map; ghost-mapping (class 8) and "other" (class 9) become -1 (ignored).
  4. Optional --match-val-pipeline: drop cosmics with prob + BiasedSphereCrop
     around the nu_vertex (reproduces the val pipeline used during training).
  5. Run the val-style transform chain. GridSample mode='test' for full
     coverage (default) or mode='train' (one point per voxel, val-mimic).
  6. For each fragment: forward through the model -> softmax over the 8 logits.
     Scatter back to the input layout (mode='test') or keep the deduped layout
     (mode='train').
  7. Accumulate confusion + score_hist into the shard.

The output is one .npz shard with the schema in histograms.py.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import h5py  # noqa: F401  — imported before ROOT-ish things for HDF5 ABI
import numpy as np
import torch


# ---------------------------------------------------------------------------
# Repo / pkg path bootstrap
# ---------------------------------------------------------------------------

_HERE = os.path.dirname(os.path.abspath(__file__))
POINTCEPT_ROOT = os.path.dirname(os.path.dirname(_HERE))
if POINTCEPT_ROOT not in sys.path:
    sys.path.insert(0, POINTCEPT_ROOT)

sys.path.insert(0, _HERE)
from histograms import SemSegAccumulator, N_CLASSES  # noqa: E402

from pointcept.utils.config import Config  # noqa: E402
from pointcept.models import build_model  # noqa: E402
from pointcept.datasets.transform import Compose, TRANSFORMS  # noqa: E402
import pointcept.datasets  # noqa: F401,E402  registers LArTPCDataset etc.
import pointcept.models  # noqa: F401,E402   registers SonataLoRASegmentor etc.


# Mirrors LArTPCDataset.SSNETLABEL_TO_CLASS (pointcept/datasets/lartpc.py).
SSNETLABEL_TO_CLASS = {
    0: 8,   # bg -> ghost
    1: 0,   # electron
    2: 4,   # photon -> gamma
    3: 1,   # muon
    4: 3,   # proton
    5: 2,   # pion
    6: 5,   # michel
    7: 6,   # delta
    8: 7,   # LED
    9: 9,   # OTHER
}


def _map_ssnet_to_class(ssnet_label: np.ndarray) -> np.ndarray:
    """Apply SSNETLABEL_TO_CLASS, then drop class 8 (ghost) and class 9 (other)
    by remapping them to -1 — matches the training config's
    include_ghosts=False, exclude_other=True semantics.
    """
    segment = np.full(len(ssnet_label), -1, dtype=np.int32)
    for code, idx in SSNETLABEL_TO_CLASS.items():
        segment[ssnet_label == code] = idx
    segment[segment == 8] = -1  # drop ghost class
    segment[segment == 9] = -1  # drop "other" class
    return segment


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _filter_per_point(data: dict, keep_mask: np.ndarray) -> dict:
    if keep_mask.dtype != bool:
        keep_mask = keep_mask.astype(bool)
    out = dict(data)
    for k in list(out.keys()):
        v = out[k]
        if isinstance(v, np.ndarray) and v.shape and v.shape[0] == keep_mask.shape[0]:
            out[k] = v[keep_mask]
    return out


def read_event(h5_path: str, *, skip_ghosts: bool = True) -> dict:
    """Read per-spacepoint truth + model inputs from one merged H5 entry.

    Returns a numpy data_dict in the format LArTPCDataset.get_data would
    produce (minus dataset-side filtering). When ``skip_ghosts`` is True
    (default; matches training's ``true_points_only=True``), points with
    ``hasmatch==0`` are dropped before any transforms see them.
    """
    with h5py.File(h5_path, "r") as f:
        td = f["entry_0/triplet_data"]
        coord = td["pos"][:].astype(np.float32)
        hasmatch = td["hasmatch"][:].astype(np.int64)
        origin = td["origin"][:].astype(np.int32)
        ssnet_label = td["ssnet_label"][:].astype(np.int32)

        pixval = td["pixval"][:].astype(np.float32)  # (N, 3)
        strength = (np.clip(pixval, a_min=0, a_max=1000.0) / 1.0).astype(np.float32)
        strength += 0.001  # add_min_pixval to avoid log(0)

        wire_feat = np.stack([
            td["uwire"][:], td["vwire"][:], td["ywire"][:],
        ], axis=1).astype(np.float32)

        if "mckeypoints" in f["entry_0"]:
            kp_pos = f["entry_0/mckeypoints/pos"][:].astype(np.float32)
            kp_type = f["entry_0/mckeypoints/kptype"][:].astype(np.int64)
            nu_vertices = kp_pos[kp_type == 0]
        else:
            nu_vertices = np.zeros((0, 3), dtype=np.float32)

    truth_class = _map_ssnet_to_class(ssnet_label)

    data = dict(
        coord=coord,
        strength=strength,
        color=wire_feat,
        # ``segment`` is the model's per-point label slot. We feed the
        # 8-class truth so the model's eval-mode loss computation works
        # (the value is otherwise unused for inference).
        segment=truth_class.astype(np.int32),
        nu_vertices=nu_vertices,
        # Truth bookkeeping. Registered in index_valid_keys so any transform
        # that filters per-point arrays (BiasedSphereCrop, GridSample) keeps
        # them aligned with coord.
        _truth_class=truth_class.astype(np.int32),
        _truth_origin=origin,
        _truth_hasmatch=hasmatch.astype(np.int8),
        index_valid_keys=[
            "coord", "strength", "color", "segment",
            "_truth_class", "_truth_origin", "_truth_hasmatch",
        ],
    )

    if skip_ghosts:
        keep = hasmatch == 1
        data = _filter_per_point(data, keep)

    return data


# ---------------------------------------------------------------------------
# Val-mimic preprocessing
# ---------------------------------------------------------------------------

def val_mimic_preprocess(
    data: dict,
    *,
    drop_cosmics_prob: float,
    sphere_radius: float,
    sphere_max_points: int,
    sphere_min_points: int,
    sphere_prob_random: float,
    rng: np.random.Generator,
) -> dict:
    """Reproduce the val-side data filtering before model inference:
       (1) drop_cosmics with the given probability,
       (2) BiasedSphereCrop around the nu_vertex.
    Truth labels are already in the per-point arrays and survive both steps
    via index_valid_keys.
    """
    seed = int(rng.integers(0, 2**31 - 1))
    np.random.seed(seed)

    if drop_cosmics_prob > 0 and np.random.random() < drop_cosmics_prob:
        nu_mask = data["_truth_origin"] == 1
        if int(nu_mask.sum()) > sphere_min_points:
            data = _filter_per_point(data, nu_mask)

    if data["coord"].shape[0] > sphere_max_points:
        crop = TRANSFORMS.build(dict(
            type="BiasedSphereCrop",
            anchor_points_key="nu_vertices",
            anchor_pdf_key=None,
            radius=sphere_radius,
            point_max=sphere_max_points,
            point_min=sphere_min_points,
            prob_random=sphere_prob_random,
            max_retries=100,
            fallback_to_random=True,
        ))
        data = crop(data)

    return data


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

def build_eval_transform(grid_size: float, coord_scale: float,
                         gridsample_mode: str = "test"):
    """Val-style transform chain WITHOUT BiasedSphereCrop. GridSample mode is
    "test" by default (full coverage, returns list of fragments). For the
    val-mimic mode pass "train" — one randomly-chosen point per voxel.
    """
    return Compose([
        dict(
            type="GridSample",
            grid_size=grid_size,
            hash_type="fnv",
            mode=gridsample_mode,
            return_grid_coord=True,
        ),
    ]), Compose([
        dict(type="NormalizeCoord",
             center=[125.0, 0.0, 518.0],
             scale=coord_scale),
        dict(type="LogTransform",
             min_val=0.01, max_val=1000.0, log=True,
             keys=("strength",)),
        dict(type="ToTensor"),
        dict(type="Collect",
             keys=("coord", "grid_coord", "segment"),
             feat_keys=("coord", "strength")),
    ])


# ---------------------------------------------------------------------------
# Model load
# ---------------------------------------------------------------------------

def build_segmentor(cfg_path: str, weights_path: str, device: torch.device):
    cfg = Config.fromfile(cfg_path)
    model = build_model(cfg.model)
    ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
    state = ckpt.get("state_dict", ckpt)
    clean = {}
    for k, v in state.items():
        clean[k[7:] if k.startswith("module.") else k] = v
    missing, unexpected = model.load_state_dict(clean, strict=False)
    if missing:
        print(f"[warn] missing keys in checkpoint: {len(missing)} "
              f"(first 3: {missing[:3]})")
    if unexpected:
        print(f"[warn] unexpected keys in checkpoint: {len(unexpected)} "
              f"(first 3: {unexpected[:3]})")
    model.eval().to(device)
    return model, cfg


# ---------------------------------------------------------------------------
# Forward
# ---------------------------------------------------------------------------

@torch.no_grad()
def _model_softmax(model, frag_post: dict, device: torch.device) -> np.ndarray:
    coord = frag_post["coord"].to(device)
    grid_coord = frag_post["grid_coord"].to(device)
    feat = frag_post["feat"].to(device)
    n = int(coord.shape[0])
    offset = torch.tensor([n], dtype=torch.int32, device=device)
    seg = frag_post["segment"].to(device) if "segment" in frag_post \
        else torch.zeros(n, dtype=torch.long, device=device)
    inputs = dict(coord=coord, grid_coord=grid_coord, feat=feat,
                  offset=offset, segment=seg)
    result = model(inputs)
    logits = result["seg_logits"]                  # (n, C)
    soft = torch.softmax(logits, dim=-1)
    return soft.float().cpu().numpy()


def forward_one_event(
    model, data_np: dict,
    pre_voxelize: Compose, post_transform: Compose,
    device: torch.device,
) -> dict:
    """Run the model on data_np and return aligned per-point arrays.

    GridSample mode='test' -> scatter back to original layout (one row per
    input spacepoint). GridSample mode='train' -> one row per deduped voxel;
    the truth arrays in the data_dict are deduped by the same indexing thanks
    to ``index_valid_keys``.
    """
    pipe_dict = dict(data_np)
    voxelized = pre_voxelize(pipe_dict)

    if isinstance(voxelized, list):
        # mode='test' — list of fragments, scatter back to the original layout.
        N = int(data_np["coord"].shape[0])
        soft_full = np.full((N, N_CLASSES), np.nan, dtype=np.float32)
        for frag in voxelized:
            idx = frag["index"]
            frag_post = post_transform(frag)
            soft_full[idx] = _model_softmax(model, frag_post, device)
        covered = np.isfinite(soft_full[:, 0])
        soft_full = np.nan_to_num(soft_full, nan=0.0).astype(np.float32)
        # Renormalize any rows that came in non-normalized (safety; should be
        # already-normalized softmax). Any uncovered point stays all-zeros.
        return dict(
            softmax=soft_full,
            truth_class=data_np["_truth_class"].astype(np.int32),
            origin=data_np["_truth_origin"].astype(np.int32),
            hasmatch=data_np["_truth_hasmatch"].astype(np.int8),
            n_covered=int(covered.sum()),
        )

    # mode='train' — single deduped data_dict; truth arrays already filtered.
    frag_post = post_transform(voxelized)
    soft = _model_softmax(model, frag_post, device).astype(np.float32)
    return dict(
        softmax=soft,
        truth_class=voxelized["_truth_class"].astype(np.int32),
        origin=voxelized["_truth_origin"].astype(np.int32),
        hasmatch=voxelized["_truth_hasmatch"].astype(np.int8),
        n_covered=int(soft.shape[0]),
    )


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-config", required=True,
                   help="Pointcept config .py used to build the model")
    p.add_argument("--weights", required=True,
                   help="Path to model_last.pth")
    p.add_argument("--filelist", required=True,
                   help="Text file with one merged H5 path per line")
    p.add_argument("--output-shard", required=True,
                   help="Output .npz with the accumulated histograms")
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--nfiles", type=int, default=-1)
    p.add_argument("--device", default=None,
                   help="torch device (default: cuda if available else cpu)")
    p.add_argument("--score-bins", type=int, default=1000)
    p.add_argument("--grid-size", type=float, default=None,
                   help="Override grid_size (default: cfg.grid_size)")
    p.add_argument("--coord-scale", type=float, default=None,
                   help="Override coord_scale (default: cfg.coord_scale)")
    p.add_argument("--keep-ghosts", action="store_true",
                   help="Pass ghost points (hasmatch==0) through the model "
                        "(default: skip them, matching training).")
    p.add_argument("--save-scores", type=int, default=0,
                   help="For the first N events processed, save per-spacepoint "
                        "(coord, truth_class, pred_class, softmax, origin) into "
                        "a sidecar .npz next to the shard for visualization.")
    # ---- val-mimic preprocessing ---------------------------------------------
    p.add_argument("--match-val-pipeline", action="store_true",
                   help="Reproduce the val-time data filtering: drop_cosmics "
                        "with prob + BiasedSphereCrop around the nu_vertex + "
                        "GridSample mode='train' (one point per voxel).")
    p.add_argument("--biased-sphere-radius", type=float, default=20.0)
    p.add_argument("--biased-sphere-max-points", type=int, default=10240)
    p.add_argument("--biased-sphere-min-points", type=int, default=2048)
    p.add_argument("--biased-sphere-prob-random", type=float, default=0.25)
    p.add_argument("--drop-cosmics-prob", type=float, default=0.9,
                   help="Probability of dropping cosmic-origin points in "
                        "val-mimic mode. 0 disables. (training val: 0.9)")
    p.add_argument("--val-mimic-seed", type=int, default=12345)
    args = p.parse_args()

    device = torch.device(args.device or
                          ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"[main] device: {device}")

    with open(args.filelist) as f:
        all_files = [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]
    end = (len(all_files) if args.nfiles < 0
           else min(args.start + args.nfiles, len(all_files)))
    assigned = all_files[args.start:end]
    if not assigned:
        print(f"[main] no files in range [{args.start}, {end}); exiting")
        return
    print(f"[main] processing {len(assigned)} files "
          f"(global indices {args.start}..{end - 1})")

    model, cfg = build_segmentor(args.model_config, args.weights, device)
    num_classes_cfg = int(cfg.model.get("num_classes", N_CLASSES))
    if num_classes_cfg != N_CLASSES:
        raise RuntimeError(
            f"Config num_classes={num_classes_cfg} but the accumulator schema "
            f"expects N_CLASSES={N_CLASSES}. Edit histograms.py if you want "
            f"to evaluate a model with a different class count."
        )

    grid_size = (args.grid_size if args.grid_size is not None
                 else float(cfg.get("grid_size", 0.25)))
    coord_scale = (args.coord_scale if args.coord_scale is not None
                   else float(cfg.get("coord_scale", 179.55)))
    print(f"[main] grid_size={grid_size}  coord_scale={coord_scale}")

    pre_voxelize, post_transform = build_eval_transform(
        grid_size=grid_size, coord_scale=coord_scale,
        gridsample_mode=("train" if args.match_val_pipeline else "test"),
    )
    print(f"[main] mode: "
          f"{'val-mimic (GridSample=train + BiasedSphereCrop + drop_cosmics)' if args.match_val_pipeline else 'full-event (GridSample=test, no crop, no drop_cosmics)'}")
    print(f"[main] skip_ghosts: {not args.keep_ghosts}")

    acc = SemSegAccumulator(n_bins=args.score_bins)
    save_scores_remaining = int(args.save_scores)
    saved_scores = []

    def _empty_add(file_idx, status):
        acc.add_event(
            softmax=np.zeros((0, N_CLASSES), dtype=np.float32),
            truth_class=np.zeros(0, dtype=np.int32),
            origin=np.zeros(0, dtype=np.int32),
            file_idx=file_idx, status=status,
        )

    t_start = time.time()
    for local_i, h5_path in enumerate(assigned):
        global_i = args.start + local_i
        if not os.path.exists(h5_path):
            print(f"[{global_i}] MISSING {h5_path}")
            _empty_add(global_i, status=1)
            continue
        try:
            data_np = read_event(h5_path, skip_ghosts=not args.keep_ghosts)
        except Exception as exc:
            print(f"[{global_i}] read failure on {os.path.basename(h5_path)}: {exc}")
            _empty_add(global_i, status=2)
            continue

        if args.match_val_pipeline:
            rng = np.random.default_rng(args.val_mimic_seed + global_i)
            data_np = val_mimic_preprocess(
                data_np,
                drop_cosmics_prob=args.drop_cosmics_prob,
                sphere_radius=args.biased_sphere_radius,
                sphere_max_points=args.biased_sphere_max_points,
                sphere_min_points=args.biased_sphere_min_points,
                sphere_prob_random=args.biased_sphere_prob_random,
                rng=rng,
            )

        N = int(data_np["coord"].shape[0])
        if N == 0:
            print(f"[{global_i}] N=0 after preprocessing; skipping")
            _empty_add(global_i, status=1)
            continue

        try:
            forward_out = forward_one_event(
                model, data_np, pre_voxelize, post_transform, device,
            )
        except torch.cuda.OutOfMemoryError:
            print(f"[{global_i}] OOM on {os.path.basename(h5_path)} (N={N}).")
            torch.cuda.empty_cache()
            _empty_add(global_i, status=2)
            continue
        except Exception as exc:
            print(f"[{global_i}] forward failure: {exc}")
            _empty_add(global_i, status=2)
            continue

        acc.add_event(
            softmax=forward_out["softmax"],
            truth_class=forward_out["truth_class"],
            origin=forward_out["origin"],
            file_idx=global_i,
            status=0,
        )

        if save_scores_remaining > 0:
            pred_class = forward_out["softmax"].argmax(axis=1).astype(np.int8)
            # mode='test': pos has shape (N_input,); mode='train': has shape
            # (N_voxels,) and we recover coord from the post-voxelize data.
            if forward_out["softmax"].shape[0] == data_np["coord"].shape[0]:
                pos = data_np["coord"]
            else:
                pos = None  # val-mimic: original coord not aligned to dedup
            saved_scores.append(dict(
                file_idx=np.int32(global_i),
                file_path=os.path.basename(h5_path),
                pos=pos if pos is not None else np.zeros((0, 3), dtype=np.float32),
                truth_class=forward_out["truth_class"].astype(np.int8),
                pred_class=pred_class,
                softmax=forward_out["softmax"].astype(np.float32),
                origin=forward_out["origin"].astype(np.int8),
                hasmatch=forward_out["hasmatch"].astype(np.int8),
            ))
            save_scores_remaining -= 1

        if (local_i + 1) % 10 == 0 or local_i + 1 == len(assigned):
            elapsed = time.time() - t_start
            print(f"[{global_i}] {local_i + 1}/{len(assigned)}  "
                  f"N={N:>6d}  covered={forward_out['n_covered']:>6d}  "
                  f"elapsed={elapsed:.1f}s "
                  f"({elapsed/(local_i+1):.2f}s/event)")

    os.makedirs(os.path.dirname(os.path.abspath(args.output_shard)) or ".",
                exist_ok=True)
    acc.save(
        args.output_shard, file_list=assigned,
        extra=dict(
            model_config=np.array([os.path.abspath(args.model_config)]),
            weights=np.array([os.path.abspath(args.weights)]),
            grid_size=np.float32(grid_size),
            coord_scale=np.float32(coord_scale),
            mode=np.array(["val_mimic" if args.match_val_pipeline else "full_event"]),
            skip_ghosts=np.int8(0 if args.keep_ghosts else 1),
        ),
    )
    print(f"[main] wrote shard {args.output_shard}")

    if saved_scores:
        side = args.output_shard.replace(".npz", "_scores.npz")
        np.savez_compressed(side, **{
            f"event{i}_{k}": v
            for i, ev in enumerate(saved_scores)
            for k, v in ev.items()
        })
        print(f"[main] wrote per-event sidecar {side} "
              f"({len(saved_scores)} events)")


if __name__ == "__main__":
    main()
