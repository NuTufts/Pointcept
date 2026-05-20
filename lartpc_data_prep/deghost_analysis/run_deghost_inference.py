"""Run the LoRA deghoster on a filelist and accumulate per-shard histograms.

What this does (per merged H5 file in the assigned slice):

  1. Read pos, hasmatch, origin, ssnet_label, lm_score, pixval directly
     from the H5. (Mirrors what LArTPCDataset.get_data does, but avoids
     the dataset's training-only filters.)
  2. Run a val-style transform chain on a copy of the data, swapping
     GridSample(mode='train') for GridSample(mode='test') so every
     original spacepoint lands in exactly one fragment. Skip the
     BiasedSphereCrop entirely so we cover the full event.
  3. For each test-mode fragment: forward through the model, softmax along
     the logits axis, take the real-class column → score in [0, 1].
  4. Scatter scores back to the original (N,) layout via fragment[index].
  5. Accumulate the deghost histograms into the shard (LANTERN + LoRA both).

The output is one .npz shard with the schema in histograms.py.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

# Import h5py before ROOT-ish things to dodge the HDF5 ABI clash we saw earlier.
import h5py  # noqa: F401
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
from histograms import DeghostAccumulator  # noqa: E402

# Pointcept-side imports (need POINTCEPT_ROOT on sys.path)
from pointcept.utils.config import Config  # noqa: E402
from pointcept.models import build_model  # noqa: E402
from pointcept.datasets.transform import Compose, TRANSFORMS  # noqa: E402
import pointcept.datasets  # noqa: F401,E402  registers LArTPCDataset etc.
import pointcept.models  # noqa: F401,E402   registers SonataLoRADeghostSegmentor etc.


# ---------------------------------------------------------------------------
# Data loading + transforms
# ---------------------------------------------------------------------------

def read_event(h5_path: str) -> dict:
    """Read the per-spacepoint truth + LANTERN score + model inputs from one
    merged H5 entry. Returns a numpy data_dict in the format LArTPCDataset.get_data
    would produce (minus the dataset-side filtering)."""
    with h5py.File(h5_path, "r") as f:
        td = f["entry_0/triplet_data"]
        coord = td["pos"][:].astype(np.float32)
        N = coord.shape[0]

        hasmatch = td["hasmatch"][:].astype(np.int64)
        origin = td["origin"][:].astype(np.int32)
        ssnet_label = td["ssnet_label"][:].astype(np.int32)
        # LANTERN score (already on [0,1], higher = more real)
        lm_score = td["lm_score"][:].astype(np.float32) if "lm_score" in td else None

        # Model inputs
        pixval = td["pixval"][:].astype(np.float32)              # (N, 3)
        # 'strength' as in the dataset: clip + scale; later LogTransform makes
        # it log-space. The val pipeline uses use_edep_as_strength=True.
        #strength = (np.clip(pixval, a_min=0, a_max=1000.0) / 500.0).astype(np.float32)
        strength = (np.clip(pixval, a_min=0, a_max=1000.0)).astype(np.float32)
        strength += 0.001  # add_min_pixval avoids log(0)

        wire_feat = np.stack([
            td["uwire"][:], td["vwire"][:], td["ywire"][:],
        ], axis=1).astype(np.float32)

        # Neutrino vertex for BiasedSphereCrop in val-mimic mode.
        if "mckeypoints" in f["entry_0"]:
            kp_pos = f["entry_0/mckeypoints/pos"][:].astype(np.float32)
            kp_type = f["entry_0/mckeypoints/kptype"][:].astype(np.int64)
            nu_vertices = kp_pos[kp_type == 0]
        else:
            nu_vertices = np.zeros((0, 3), dtype=np.float32)

    data = dict(
        coord=coord,                # (N, 3) cm
        strength=strength,          # (N, 3)
        color=wire_feat,            # (N, 3) - unused by the deghoster but present
        # Carry the deghoster target (1=ghost in the model's 2-class head; 0=real).
        # The val pipeline relies on RemapGhostLabel(ghost_source_index=8) on the
        # ssnet-class segment, but here we just feed hasmatch directly:
        # ghost when hasmatch==0 → 1; real when hasmatch==1 → 0.
        segment=(1 - hasmatch).astype(np.int32),
        segment_counts=np.array([[int((hasmatch == 1).sum()),
                                   int((hasmatch == 0).sum())]], dtype=np.int64),
        nu_vertices=nu_vertices,
        # Truth bookkeeping. We register them in index_valid_keys so any
        # transform that filters per-point arrays (BiasedSphereCrop,
        # GridSample) keeps them aligned with coord.
        _truth_hasmatch=hasmatch.astype(np.int8),
        _truth_origin=origin,
        _truth_ssnet=ssnet_label,
        _lantern_lm_score=lm_score if lm_score is not None
            else np.full(N, np.nan, dtype=np.float32),
        # Ignore mask is tracked as a per-point bool array so transforms
        # filter it alongside coord. Starts all-False; val_mimic_preprocess
        # sets it for ssnet==9.
        _ignore_mask=np.zeros(N, dtype=bool),
        index_valid_keys=[
            "coord", "strength", "color", "segment",
            "_truth_hasmatch", "_truth_origin", "_truth_ssnet",
            "_lantern_lm_score", "_ignore_mask",
        ],
    )
    return data


def _filter_per_point(data: dict, keep_mask: np.ndarray) -> dict:
    """In-place index every per-point array of ``data`` by ``keep_mask``."""
    if keep_mask.dtype != bool:
        keep_mask = keep_mask.astype(bool)
    out = dict(data)
    for k in list(out.keys()):
        v = out[k]
        if isinstance(v, np.ndarray) and v.shape and v.shape[0] == keep_mask.shape[0]:
            out[k] = v[keep_mask]
    # segment_counts is global — recompute if hasmatch survived
    if "_truth_hasmatch" in out:
        hm = out["_truth_hasmatch"]
        out["segment_counts"] = np.array(
            [[int((hm == 1).sum()), int((hm == 0).sum())]], dtype=np.int64)
    return out


def val_mimic_preprocess(
    data: dict,
    *,
    drop_cosmics_prob: float,
    sphere_radius: float,
    sphere_max_points: int,
    sphere_min_points: int,
    sphere_prob_random: float,
    rng: np.random.Generator,
) -> tuple[dict, np.ndarray]:
    """Reproduce the val-side data filtering before model inference:
       (1) drop_cosmics with the given probability (mask cosmic-origin points
           when the event has enough nu points after the mask),
       (2) BiasedSphereCrop around the nu_vertex (radius, point_max, etc.).
    Also switches the truth label source to ssnet-based (real=ssnet∈0..7,
    ghost=ssnet==8, ignore=ssnet==9). Returns (filtered_data, ignore_mask)
    where ignore_mask is True for points that should not contribute to
    metrics (ssnet==9 in val-mimic mode).
    """
    # (1) drop_cosmics — set numpy's global RNG so the BiasedSphereCrop
    # below (which uses np.random.*) also runs deterministically.
    seed = int(rng.integers(0, 2**31 - 1))
    np.random.seed(seed)
    if drop_cosmics_prob > 0 and np.random.random() < drop_cosmics_prob:
        nu_mask = data["_truth_origin"] == 1
        # Mimic LArTPCDataset's "only if enough nu points" guard (min_points_required
        # in the dataset is small; we use the same sphere_min_points threshold).
        if int(nu_mask.sum()) > sphere_min_points:
            data = _filter_per_point(data, nu_mask)

    # (2) BiasedSphereCrop via TRANSFORMS.build for fidelity.
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

    # (3) Build ignore mask: ssnet==9 ("other") is ignored in the val
    # RemapGhostLabel pipeline. Also switch the model's "segment" input to
    # the ssnet-based label (0=real for ssnet 0..7, 1=ghost for ssnet==8,
    # arbitrary for ssnet==9 since it's ignored).
    ssnet = data["_truth_ssnet"]
    ignore_mask = (ssnet == 9)
    seg_val = np.where(ssnet == 8, 1, 0).astype(np.int32)
    seg_val[ignore_mask] = -1
    data["segment"] = seg_val
    # Update the in-data ignore_mask so later filtering keeps it aligned.
    data["_ignore_mask"] = ignore_mask
    return data


def build_eval_transform(grid_size: float, coord_scale: float,
                          gridsample_mode: str = "test"):
    """Val-style transform chain WITHOUT BiasedSphereCrop. GridSample mode is
    "test" by default (full coverage, returns list of fragments). For the
    val-mimic mode pass "train" — one randomly-chosen point per voxel, which
    matches what the training/val loop actually scored."""
    return Compose([
        dict(
            type="GridSample",
            grid_size=grid_size,
            hash_type="fnv",
            mode=gridsample_mode,
            return_grid_coord=True,
        ),
    ]), Compose([
        # Post-test (per-fragment) — mirrors the val pipeline minus the truth
        # remap (we already wrote segment in real=0/ghost=1 layout above).
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

def build_lora_model(cfg_path: str, weights_path: str, device: torch.device):
    """Instantiate SonataLoRADeghostSegmentor and load .pth weights.
    Strips a 'module.' prefix (DDP) if present, allows partial keys."""
    cfg = Config.fromfile(cfg_path)
    model = build_model(cfg.model)
    ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
    state = ckpt.get("state_dict", ckpt)
    # Strip optional DDP prefix
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
# Forward one event
# ---------------------------------------------------------------------------

@torch.no_grad()
def _model_real_prob(model, frag_post: dict, device: torch.device) -> np.ndarray:
    """Run model on one post-Collect fragment dict; return real-class softmax
    probability as numpy (n,) float32."""
    coord = frag_post["coord"].to(device)
    grid_coord = frag_post["grid_coord"].to(device)
    feat = frag_post["feat"].to(device)
    n = int(coord.shape[0])
    offset = torch.tensor([n], dtype=torch.int32, device=device)
    seg = torch.zeros(n, dtype=torch.long, device=device)
    inputs = dict(coord=coord, grid_coord=grid_coord, feat=feat,
                  offset=offset, segment=seg)
    result = model(inputs)
    logits = result["seg_logits"]                    # (n, 2)
    real_prob = torch.softmax(logits, dim=-1)[:, 0]   # col 0 = "real"
    return real_prob.float().cpu().numpy()


def forward_one_event(
    model, data_np: dict,
    pre_voxelize: Compose, post_transform: Compose,
    device: torch.device,
) -> dict:
    """Run the model on data_np and return aligned per-point arrays for the
    set of points that were scored.

    GridSample mode='test' → scattering back gives one row per original point
    (full event coverage). GridSample mode='train' → one row per deduped
    voxel; the truth and LANTERN arrays in the data_dict are deduped by the
    same indexing thanks to ``index_valid_keys``.

    Returns:
        scores       (M, 2) float32  cols = (lantern_score, lora_score)
        hasmatch     (M,)   int8
        origin       (M,)   int32
        ssnet        (M,)   int32
        ignore_mask  (M,)   bool
    where M = N (full event) or = N_dedup (val-mimic).
    """
    pipe_dict = dict(data_np)
    voxelized = pre_voxelize(pipe_dict)

    if isinstance(voxelized, list):
        # mode='test' — list of fragments, scatter back to original layout
        N = int(data_np["coord"].shape[0])
        lora = np.full(N, np.nan, dtype=np.float32)
        for frag in voxelized:
            idx = frag["index"]
            frag_post = post_transform(frag)
            lora[idx] = _model_real_prob(model, frag_post, device)
        scores = np.stack([
            np.nan_to_num(data_np["_lantern_lm_score"], nan=0.0,
                          posinf=1.0, neginf=0.0),
            np.nan_to_num(lora, nan=0.0, posinf=1.0, neginf=0.0),
        ], axis=1).astype(np.float32)
        return dict(
            scores=scores,
            hasmatch=data_np["_truth_hasmatch"].astype(np.int8),
            origin=data_np["_truth_origin"].astype(np.int32),
            ssnet=data_np["_truth_ssnet"].astype(np.int32),
            ignore_mask=data_np["_ignore_mask"].astype(bool),
            n_lora_covered=int(np.isfinite(lora).sum()),
        )

    # mode='train' — single deduped data_dict; truth arrays are already
    # filtered by index_valid_keys to match the deduped points.
    frag_post = post_transform(voxelized)
    lora = _model_real_prob(model, frag_post, device)
    lantern = voxelized["_lantern_lm_score"]
    scores = np.stack([
        np.nan_to_num(lantern, nan=0.0, posinf=1.0, neginf=0.0),
        np.nan_to_num(lora, nan=0.0, posinf=1.0, neginf=0.0),
    ], axis=1).astype(np.float32)
    return dict(
        scores=scores,
        hasmatch=voxelized["_truth_hasmatch"].astype(np.int8),
        origin=voxelized["_truth_origin"].astype(np.int32),
        ssnet=voxelized["_truth_ssnet"].astype(np.int32),
        ignore_mask=voxelized["_ignore_mask"].astype(bool),
        n_lora_covered=int(np.isfinite(lora).sum()),
    )


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-config", required=True,
                   help="Pointcept config .py used to build the model")
    p.add_argument("--weights", required=True,
                   help="Path to model_last.pth (full LoRA-fine-tuned weights)")
    p.add_argument("--filelist", required=True,
                   help="Text file with one merged H5 path per line")
    p.add_argument("--output-shard", required=True,
                   help="Output .npz with the accumulated histograms")
    p.add_argument("--start", type=int, default=0,
                   help="First line in the filelist to process (0-indexed)")
    p.add_argument("--nfiles", type=int, default=-1,
                   help="Number of files to process (-1 = to end)")
    p.add_argument("--device", default=None,
                   help="torch device (default: cuda if available else cpu)")
    p.add_argument("--score-bins", type=int, default=1000)
    p.add_argument("--grid-size", type=float, default=None,
                   help="Override grid_size for the eval voxelization "
                        "(default: cfg.grid_size from the config file)")
    p.add_argument("--coord-scale", type=float, default=None,
                   help="Override coord_scale used by NormalizeCoord "
                        "(default: cfg.coord_scale from the config file)")
    p.add_argument("--save-scores", type=int, default=0,
                   help="For the first N events processed, save per-spacepoint "
                        "(hasmatch, origin, ssnet, lantern_score, lora_score) "
                        "into a sidecar .npz next to the shard for visualization.")
    p.add_argument("--max-points-per-chunk", type=int, default=0,
                   help="If > 0 and a fragment exceeds this many points, "
                        "split it spatially along z and forward in chunks "
                        "(no halo — small mismatches near chunk boundaries "
                        "are tolerated for this safety hatch). 0 = no chunking.")
    # ---- val-mimic preprocessing (verifies inference reproduces wandb val numbers) ----
    p.add_argument("--match-val-pipeline", action="store_true",
                   help="Reproduce the val-time data filtering: "
                        "BiasedSphereCrop around nu_vertex, drop_cosmics with prob, "
                        "GridSample mode='train' (one point per voxel), and ssnet-"
                        "based truth labels (ssnet==8=ghost, ssnet in 0..7=real, "
                        "ssnet==9 ignored). Use this to confirm the model + "
                        "inference setup reproduce the training-time val mIoU.")
    p.add_argument("--biased-sphere-radius", type=float, default=20.0)
    p.add_argument("--biased-sphere-max-points", type=int, default=10240)
    p.add_argument("--biased-sphere-min-points", type=int, default=2048)
    p.add_argument("--biased-sphere-prob-random", type=float, default=0.25)
    p.add_argument("--drop-cosmics-prob", type=float, default=0.9,
                   help="Probability of dropping cosmic-origin points in val-mimic "
                        "mode. 0 disables. (val config: 0.9)")
    p.add_argument("--val-mimic-seed", type=int, default=12345,
                   help="Seed offset for reproducible val-mimic stochastic "
                        "drop_cosmics + BiasedSphereCrop choices.")
    args = p.parse_args()

    device = torch.device(args.device or
                          ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"device: {device}")

    with open(args.filelist) as f:
        all_files = [ln.strip() for ln in f if ln.strip()]
    end = (len(all_files) if args.nfiles < 0
           else min(args.start + args.nfiles, len(all_files)))
    assigned = all_files[args.start:end]
    if not assigned:
        print(f"[main] no files in range [{args.start}, {end}); exiting")
        return
    print(f"[main] processing {len(assigned)} files "
          f"(global indices {args.start}..{end - 1})")

    model, cfg = build_lora_model(args.model_config, args.weights, device)
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
          f"{'val-mimic' if args.match_val_pipeline else 'full-event'}")

    acc = DeghostAccumulator(n_bins=args.score_bins)
    save_scores_remaining = int(args.save_scores)
    saved_scores = []  # list of dicts

    def _empty_add(file_idx, status):
        z0 = np.zeros(0, dtype=np.int32)
        acc.add_event(
            scores=np.zeros((0, 2), dtype=np.float32),
            truth_real=np.zeros(0, np.int8),
            origin=z0, ssnet_label=z0, hasmatch=z0,
            file_idx=file_idx, n_lora_covered=0, status=status,
        )

    t_start = time.time()
    for local_i, h5_path in enumerate(assigned):
        global_i = args.start + local_i
        if not os.path.exists(h5_path):
            print(f"[{global_i}] MISSING {h5_path}")
            _empty_add(global_i, status=1)
            continue
        try:
            data_np = read_event(h5_path)
        except Exception as exc:
            print(f"[{global_i}] read failure on {os.path.basename(h5_path)}: {exc}")
            _empty_add(global_i, status=2)
            continue

        # Optional val-mimic preprocessing (BiasedSphereCrop + drop_cosmics +
        # switch to ssnet-based truth + ssnet==9 ignore).
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
            print(f"[{global_i}] OOM on {os.path.basename(h5_path)} "
                  f"(N={N}). Skipping; consider re-running with "
                  f"--max-points-per-chunk.")
            torch.cuda.empty_cache()
            _empty_add(global_i, status=2)
            continue
        except Exception as exc:
            print(f"[{global_i}] forward failure: {exc}")
            _empty_add(global_i, status=2)
            continue

        # Decide truth-label source: val-mimic uses ssnet==8 as ghost; else
        # use raw hasmatch. We pass the chosen real/ghost flag plus the raw
        # hasmatch + ssnet so the accumulator can record the cross-tab.
        hm = forward_out["hasmatch"]                   # raw hasmatch (filtered)
        ss = forward_out["ssnet"]
        if args.match_val_pipeline:
            truth_real = (ss != 8).astype(np.int8)     # 0..7 real, 8 ghost
        else:
            truth_real = (hm == 1).astype(np.int8)
        acc.add_event(
            scores=forward_out["scores"],
            truth_real=truth_real,
            origin=forward_out["origin"],
            ssnet_label=ss,
            hasmatch=hm,
            file_idx=global_i,
            n_lora_covered=forward_out["n_lora_covered"],
            ignore_mask=forward_out["ignore_mask"],
            status=0,
        )

        if save_scores_remaining > 0:
            saved_scores.append(dict(
                file_idx=np.int32(global_i),
                file_path=os.path.basename(h5_path),
                pos=data_np["coord"],
                hasmatch=hm,
                origin=forward_out["origin"],
                ssnet_label=ss,
                lantern_score=forward_out["scores"][:, 0].astype(np.float32),
                lora_score=forward_out["scores"][:, 1].astype(np.float32),
                ignore_mask=forward_out["ignore_mask"],
            ))
            save_scores_remaining -= 1

        if (local_i + 1) % 10 == 0 or local_i + 1 == len(assigned):
            elapsed = time.time() - t_start
            print(f"[{global_i}] {local_i + 1}/{len(assigned)}  "
                  f"N={N:>6d}  elapsed={elapsed:.1f}s "
                  f"({elapsed/(local_i+1):.2f}s/event)")

    # ---- Save shard ----
    os.makedirs(os.path.dirname(os.path.abspath(args.output_shard)) or ".",
                exist_ok=True)
    acc.save(args.output_shard, file_list=assigned,
             extra=dict(model_config=np.array([os.path.abspath(args.model_config)]),
                        weights=np.array([os.path.abspath(args.weights)]),
                        grid_size=np.float32(grid_size),
                        coord_scale=np.float32(coord_scale)))
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
