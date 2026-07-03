"""Build a single-event Stage-1+2 cache for Stage-3 particle-segmenter
training.

Reads a Stage-3 config (e.g. `configs/lartpc/larformer/stage3_particle/larformer-particle-v1.py`),
brings up the trained Stage-1+2 cascade, picks ONE sample from the
dataset (`gt_source="particle"`), runs Stages 1 and 2 in eval mode,
and writes a self-contained HDF5 cache file whose schema mirrors the
production merged_h5 layout so `LArFormerStage12CacheDataset` (a small
loader added in the same change) reads it with the same conventions as
LArFormerDataset.

The cache's SP set is the UNION of:
    (a) deghoster-kept SPs whose `stage2_nu_mask_prob > τ_loose_floor`
        (default 0.2 — gives the trainer τ_loose ∈ [0.2, 1.0] flexibility)
    (b) deghoster-kept SPs that belong to a GT nu-origin particle
        (truth-anchor signal — used by mask denoising and the
        "GT-noise-masked input" training path)

GT-nu SPs that the deghoster cut are NOT included — Stage 1 is a hard
filter at inference time and we don't want a train/test mismatch.

`source_mask` per SP (uint8 bitmask):
    bit 0 (=1): passes stage2 nu-mask at the cache's NOMINAL τ_loose
                (default 0.5). Inference-realistic predicted-pass set.
    bit 1 (=2): belongs to a GT nu-origin particle.
    bit 2 (=4): passes at τ_loose - δ (default δ = 0.2 → τ = 0.3).
                Curriculum / ablation headroom.

The per-query slicer outputs (per-query nu-class probability + per-SP
nu mask probability for "interesting" queries) are also cached so the
trainer can vary the class-confidence cutoff as an augmentation axis
without re-running Stage 1+2.

The unit operation is `build_cache_event(model, dataset, sample_idx,
output_path, ...)`. The SLURM-array driver should import this and
loop over `sample_idx` for its assigned shard.

CLI:

    python tools/larformer/build_stage12_cache_event.py \\
        --config configs/lartpc/larformer/stage3_particle/larformer-particle-v1.py \\
        --inputlist /abs/path/devdata_mergedh5_pi0filter_10files.txt \\
        --sample-idx 0 --output /tmp/cache0.h5
"""
from __future__ import annotations

import argparse
import copy
import os
import time
from typing import Optional

import h5py
import numpy as np
import torch

# Side-effect: register dataset / model types referenced in configs.
import pointcept.datasets  # noqa: F401
import pointcept.models    # noqa: F401
from lartpc.data_prep.labels.keypoint_labels import copy_mckeypoints_group
from pointcept.datasets.builder import build_dataset
from pointcept.datasets.larformer import larformer_collate
from pointcept.models.builder import build_model
from pointcept.models.LArFormer.cascade_particle_filter import (
    slicer_predictions_empty,
)
from pointcept.utils.config import Config


CACHE_FORMAT_VERSION = 2  # bump from v1 (the early prototype this replaces)


# ---------------------------------------------------------------------------
# H5 helpers
# ---------------------------------------------------------------------------

_PER_SP_KEYS = ("coord", "coord_norm", "feat", "lm_score", "wire",
                "trackid", "pid", "origin_label", "hasmatch",
                "ssnet_label", "slice_id")


def _to_cpu_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _move_batch(batch: dict, device: torch.device) -> dict:
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


def _write_particle_instance(parent: h5py.Group, idx: int, inst: dict) -> None:
    g = parent.create_group(f"instance_{idx}")
    for k, v in inst.items():
        if isinstance(v, torch.Tensor):
            v = v.detach().cpu().numpy()
        if isinstance(v, np.ndarray):
            g.create_dataset(k, data=v)
        elif isinstance(v, (int, np.integer, bool, np.bool_)):
            g.attrs[k] = int(v) if not isinstance(v, (bool, np.bool_)) else bool(v)
        elif isinstance(v, (float, np.floating)):
            g.attrs[k] = float(v)
        elif isinstance(v, str):
            g.attrs[k] = v
        else:
            try:
                g.create_dataset(k, data=np.asarray(v))
            except Exception:
                g.attrs[k] = repr(v)


# ---------------------------------------------------------------------------
# Core: build cache for one event
# ---------------------------------------------------------------------------

def build_cache_event(
    model,
    dataset,
    sample_idx: int,
    output_path: str,
    *,
    device: torch.device,
    tau_loose_floor: float = 0.2,
    tau_loose_nominal: float = 0.5,
    tau_loose_delta: float = 0.2,
    class_prob_floor_for_query_storage: float = 0.1,
    overwrite: bool = True,
    gzip_per_query_data: bool = True,
    extra_attrs: Optional[dict] = None,
    skipped_marker_path: Optional[str] = None,
    copy_keypoints: bool = True,
) -> dict:
    """Build a Stage-1+2 cache for `dataset[sample_idx]` and write to `output_path`.

    Returns a summary dict (or writes a `.skipped` marker if the event has
    no usable Stage-3 input).
    """
    if os.path.exists(output_path) and not overwrite:
        raise FileExistsError(output_path)

    # ---- 1. Get the sample (gt_source="particle") --------------------
    sample = dataset[sample_idx]
    name = str(sample.get("name", f"sample_{sample_idx}"))
    run = int(sample.get("run", -1))
    subrun = int(sample.get("subrun", -1))
    event = int(sample.get("event", -1))
    pre_gt = sample.get("gt_instances", [])
    n_sp_dataset = int(sample["n_spacepoints"])

    # ---- 2. Run Stage 1+2 --------------------------------------------
    batch = larformer_collate([sample])
    batch_dev = _move_batch(batch, device)
    # Strip particle-level GT before the slicer call — its eval-with-GT
    # path would try to compute a 3-class CE against a 7-class target
    # and CUDA-assert. We rebuild post-cascade GT below from the raw
    # particle GT via raw_to_cache remap.
    slicer_input = {k: v for k, v in batch_dev.items()
                    if k not in ("gt_instances_per_event",
                                 "n_gt_instances")}
    model.cascaded_slicer.eval()
    with torch.no_grad():
        slicer_out = model.cascaded_slicer(slicer_input)

    slicer_predictions = slicer_out.get("predictions", [])
    filtered_batch = slicer_out.get("filtered_batch", None)
    deghost_tau = float(slicer_out.get("deghost_tau", float("nan")))
    deghost_keep_frac = float(slicer_out.get("deghost_keep_frac",
                                             float("nan")))

    deghost_p_real_t = slicer_out.get("deghost_p_real", None)
    if deghost_p_real_t is not None:
        deghost_p_real = _to_cpu_numpy(deghost_p_real_t).astype(np.float32)
        deghost_keep = (deghost_p_real > deghost_tau)
    else:
        deghost_p_real = np.zeros(n_sp_dataset, dtype=np.float32)
        deghost_keep = np.zeros(n_sp_dataset, dtype=bool)
    n_postdh = int(deghost_keep.sum())

    cascade_skipped = (filtered_batch is None
                       or slicer_predictions_empty(slicer_predictions))

    # ---- 3. Per-SP slicer evidence (dataset-index space) -------------
    nu_class_id = int(model.nu_class_id)
    spacepoint_level = str(model.spacepoint_level)
    nu_mask_prob_max_raw = np.zeros(n_sp_dataset, dtype=np.float32)
    nu_mask_prob_sum_raw = np.zeros(n_sp_dataset, dtype=np.float32)

    if not cascade_skipped:
        pred = slicer_predictions[0]
        class_logits = pred["class_logits"]                  # (Q, C)
        class_probs = class_logits.softmax(dim=-1)
        n_queries = class_logits.shape[0]
        mask_logits_sp = pred["mask_logits"][spacepoint_level]  # (Q, N_postdh)
        mask_probs_sp = mask_logits_sp.sigmoid()

        query_class_argmax = _to_cpu_numpy(
            class_probs.argmax(dim=-1)).astype(np.int64)
        query_class_max_prob = _to_cpu_numpy(
            class_probs.amax(dim=-1)).astype(np.float32)
        query_nu_prob = _to_cpu_numpy(
            class_probs[:, nu_class_id]).astype(np.float32)
        query_is_nu = (query_class_argmax == nu_class_id)

        # Per-postdh nu mask aggregates
        if bool(query_is_nu.any()):
            nu_q_idx = np.where(query_is_nu)[0]
            nu_probs_postdh = mask_probs_sp[
                torch.as_tensor(nu_q_idx, device=mask_probs_sp.device)
            ]  # (Q_nu, N_postdh)
            nu_max_postdh = _to_cpu_numpy(nu_probs_postdh.amax(dim=0)).astype(np.float32)
            nu_sum_postdh = _to_cpu_numpy(nu_probs_postdh.sum(dim=0)).astype(np.float32)
        else:
            nu_max_postdh = np.zeros(n_postdh, dtype=np.float32)
            nu_sum_postdh = np.zeros(n_postdh, dtype=np.float32)

        raw_idx_postdh = np.where(deghost_keep)[0]   # (N_postdh,) → dataset idx
        nu_mask_prob_max_raw[raw_idx_postdh] = nu_max_postdh
        nu_mask_prob_sum_raw[raw_idx_postdh] = nu_sum_postdh
    else:
        n_queries = 0
        query_class_argmax = np.empty(0, dtype=np.int64)
        query_class_max_prob = np.empty(0, dtype=np.float32)
        query_nu_prob = np.empty(0, dtype=np.float32)
        query_is_nu = np.empty(0, dtype=bool)
        nu_max_postdh = np.empty(0, dtype=np.float32)
        nu_sum_postdh = np.empty(0, dtype=np.float32)
        mask_probs_sp = None
        raw_idx_postdh = np.empty(0, dtype=np.int64)

    # ---- 4. Build GT-nu mask in dataset-index space ------------------
    gt_nu_mask_raw = np.zeros(n_sp_dataset, dtype=bool)
    for inst in pre_gt:
        ti = np.asarray(inst["truth_indices"], dtype=np.int64)
        if ti.size > 0:
            gt_nu_mask_raw[ti] = True

    # ---- 5. Determine cache inclusion --------------------------------
    # deghost_keep AND (stage2_nu_pass at τ_loose_floor OR is_GT_nu)
    passes_floor_raw = nu_mask_prob_max_raw > tau_loose_floor
    include_mask_raw = deghost_keep & (passes_floor_raw | gt_nu_mask_raw)
    n_cache = int(include_mask_raw.sum())

    if n_cache == 0:
        # Nothing usable for Stage 3 — write a .skipped marker so the
        # driver doesn't retry.
        if skipped_marker_path is not None:
            os.makedirs(os.path.dirname(os.path.abspath(skipped_marker_path)),
                        exist_ok=True)
            with open(skipped_marker_path, "w") as f:
                f.write(f"sample_idx={sample_idx}\n")
                f.write(f"name={name}\n")
                f.write(f"reason=no_cache_sp\n")
                f.write(f"cascade_skipped={cascade_skipped}\n")
                f.write(f"n_sp_dataset={n_sp_dataset}\n")
                f.write(f"n_postdh={n_postdh}\n")
                f.write(f"n_pre_gt={len(pre_gt)}\n")
        return {
            "output_path": None, "skipped_marker_path": skipped_marker_path,
            "name": name, "sample_idx": int(sample_idx),
            "run": run, "subrun": subrun, "event": event,
            "n_sp_dataset": n_sp_dataset, "n_postdh": n_postdh,
            "n_cache": 0, "n_particle_instances": 0,
            "cache_empty": True, "cascade_skipped": cascade_skipped,
            "deghost_tau": deghost_tau,
            "deghost_keep_frac": deghost_keep_frac,
        }

    cache_indices_raw = np.where(include_mask_raw)[0]
    raw_to_cache = -np.ones(n_sp_dataset, dtype=np.int64)
    raw_to_cache[cache_indices_raw] = np.arange(n_cache)

    # ---- 6. source_mask + per-SP cache arrays ------------------------
    passes_nominal = (nu_mask_prob_max_raw[cache_indices_raw]
                      > tau_loose_nominal)
    passes_delta_thresh = max(0.0, tau_loose_nominal - tau_loose_delta)
    passes_delta = (nu_mask_prob_max_raw[cache_indices_raw]
                    > passes_delta_thresh)
    is_gt_nu_cached = gt_nu_mask_raw[cache_indices_raw]

    source_mask = np.zeros(n_cache, dtype=np.uint8)
    source_mask[passes_nominal] |= 1
    source_mask[is_gt_nu_cached] |= 2
    source_mask[passes_delta] |= 4

    per_sp = {}
    for k in _PER_SP_KEYS:
        if k in sample:
            per_sp[k] = np.asarray(sample[k])[cache_indices_raw]
    per_sp["deghost_p_real"] = deghost_p_real[cache_indices_raw]
    per_sp["stage2_nu_mask_prob"] = nu_mask_prob_max_raw[cache_indices_raw]
    per_sp["stage2_nu_mask_prob_sum"] = nu_mask_prob_sum_raw[cache_indices_raw]
    per_sp["source_mask"] = source_mask

    # ---- 7. Particle instances with remapped truth_indices -----------
    particle_instances: list[dict] = []
    n_truth_orig = []
    n_truth_in_cache = []
    for inst in pre_gt:
        ti_raw = np.asarray(inst["truth_indices"], dtype=np.int64)
        if ti_raw.size == 0:
            continue
        ti_cache = raw_to_cache[ti_raw]
        ti_cache = ti_cache[ti_cache >= 0]
        if ti_cache.size == 0:
            continue
        new_inst = dict(inst)
        new_inst["truth_indices"] = ti_cache.astype(np.int64)
        new_inst["n_truth_points_orig"] = int(ti_raw.size)
        new_inst["n_truth_points_in_cache"] = int(ti_cache.size)
        particle_instances.append(new_inst)
        n_truth_orig.append(int(ti_raw.size))
        n_truth_in_cache.append(int(ti_cache.size))

    # ---- 8. Per-query slicer data (for class-threshold augmentation) -
    # Keep mask_prob for queries whose nu-class probability is above the
    # floor (default 0.1). Cuts most cold queries while preserving any
    # query the trainer might want to recompute against under a different
    # class threshold.
    if not cascade_skipped:
        keep_q_mask = query_nu_prob > class_prob_floor_for_query_storage
        keep_q_idx = np.where(keep_q_mask)[0].astype(np.int64)
        if keep_q_idx.size > 0:
            # Build the (Q_keep, N_cache) per-query mask_prob.
            # First find which postdh SPs are in cache + their cache idx.
            postdh_cache_idx = raw_to_cache[raw_idx_postdh]    # (N_postdh,) → cache or -1
            in_cache_postdh = (postdh_cache_idx >= 0)
            # Every cached SP is deghost-kept by construction → bijection.
            assert int(in_cache_postdh.sum()) == n_cache, (
                f"cache/postdh mismatch: in_cache_postdh.sum={int(in_cache_postdh.sum())} "
                f"vs n_cache={n_cache}"
            )
            # For each cache idx i, the postdh idx that maps to it:
            postdh_idx_per_cache = np.empty(n_cache, dtype=np.int64)
            in_postdh_indices = np.where(in_cache_postdh)[0]
            cache_idx_seen = postdh_cache_idx[in_cache_postdh]
            postdh_idx_per_cache[cache_idx_seen] = in_postdh_indices

            kept_mask_probs_postdh = mask_probs_sp[
                torch.as_tensor(keep_q_idx, device=mask_probs_sp.device)
            ]   # (Q_keep, N_postdh)
            per_query_mask_prob = _to_cpu_numpy(
                kept_mask_probs_postdh
            )[:, postdh_idx_per_cache].astype(np.float32)
        else:
            per_query_mask_prob = np.empty((0, n_cache), dtype=np.float32)
    else:
        keep_q_idx = np.empty(0, dtype=np.int64)
        per_query_mask_prob = np.empty((0, n_cache), dtype=np.float32)

    # ---- 9. Write H5 -------------------------------------------------
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    coord_center = (getattr(dataset, "coord_center", np.asarray((125.0, 0.0, 518.0)))
                    if not hasattr(dataset, "coord_center")
                    else np.asarray(dataset.coord_center, dtype=np.float32))
    coord_scale = float(getattr(dataset, "coord_scale", 179.55))

    compression = "gzip" if gzip_per_query_data else None
    compression_opts = 4 if gzip_per_query_data else None

    with h5py.File(output_path, "w") as f:
        # ---- top-level attrs ------------------------------------------
        f.attrs["format_version"] = CACHE_FORMAT_VERSION
        f.attrs["cache_kind"] = "stage12_for_stage3"
        f.attrs["source_h5"] = name
        f.attrs["run"] = run
        f.attrs["subrun"] = subrun
        f.attrs["event"] = event
        f.attrs["sample_idx"] = int(sample_idx)
        f.attrs["nu_class_id"] = nu_class_id
        f.attrs["spacepoint_level"] = spacepoint_level
        f.attrs["coord_center"] = np.asarray(coord_center, dtype=np.float32)
        f.attrs["coord_scale"] = coord_scale
        f.attrs["deghost_tau"] = deghost_tau
        f.attrs["deghost_keep_frac"] = deghost_keep_frac
        f.attrs["tau_loose_floor"] = float(tau_loose_floor)
        f.attrs["tau_loose_nominal"] = float(tau_loose_nominal)
        f.attrs["tau_loose_delta"] = float(tau_loose_delta)
        f.attrs["class_prob_floor_for_query_storage"] = float(
            class_prob_floor_for_query_storage)
        f.attrs["n_raw_spacepoints"] = int(sample.get("n_spacepoints_unfiltered",
                                                       n_sp_dataset))
        f.attrs["n_after_dataset_filter"] = n_sp_dataset
        f.attrs["n_after_deghost"] = n_postdh
        f.attrs["n_in_cache"] = n_cache
        f.attrs["n_passes_tau_loose"] = int(passes_nominal.sum())
        f.attrs["n_passes_tau_loose_minus_delta"] = int(passes_delta.sum())
        f.attrs["n_gt_nu_in_cache"] = int(is_gt_nu_cached.sum())
        f.attrs["n_particle_instances"] = len(particle_instances)
        f.attrs["n_queries"] = int(n_queries)
        f.attrs["n_queries_with_per_sp_data"] = int(keep_q_idx.size)
        f.attrs["cascade_skipped"] = bool(cascade_skipped)
        if extra_attrs:
            for k, v in extra_attrs.items():
                f.attrs[k] = v

        # ---- entry_0 (mirrors merged_h5 conventions) ------------------
        e0 = f.create_group("entry_0")
        e0.attrs["run"] = run
        e0.attrs["subrun"] = subrun
        e0.attrs["event"] = event

        # Per-SP arrays
        for k, v in per_sp.items():
            e0.create_dataset(k, data=v)

        # Particle GT
        e0.attrs["n_particle_instances"] = len(particle_instances)
        if particle_instances:
            pi_grp = e0.create_group("particle_instances")
            for j, inst in enumerate(particle_instances):
                _write_particle_instance(pi_grp, j, inst)

        # Slicer per-query data
        sl_grp = e0.create_group("slicer")
        sl_grp.attrs["nu_class_id"] = nu_class_id
        sl_grp.create_dataset("query_class_argmax", data=query_class_argmax)
        sl_grp.create_dataset("query_class_max_prob", data=query_class_max_prob)
        sl_grp.create_dataset("query_nu_prob", data=query_nu_prob)
        sl_grp.create_dataset("query_is_nu",
                              data=query_is_nu.astype(bool))
        sl_grp.create_dataset("query_ids_with_per_sp_data",
                              data=keep_q_idx)
        if per_query_mask_prob.size > 0:
            sl_grp.create_dataset(
                "per_query_mask_prob",
                data=per_query_mask_prob,
                compression=compression,
                compression_opts=compression_opts,
                shuffle=True if compression else False,
            )
        else:
            sl_grp.create_dataset(
                "per_query_mask_prob",
                data=per_query_mask_prob,
            )

        # ---- keypoint GT: copy mckeypoints from the source merged_h5 --
        # Makes the cache self-describing for the LArFormer keypoint module
        # (the cache reader derives kpscores + endpoints + nu-vertex from
        # this group). Resolved via the dataset's source path for this
        # sample. Non-fatal if absent.
        n_mckp = -1
        if copy_keypoints:
            data_list = getattr(dataset, "data_list", None)
            src_path = (data_list[sample_idx % len(data_list)]
                        if data_list else None)
            if src_path is not None and os.path.exists(src_path):
                try:
                    n_mckp = copy_mckeypoints_group(src_path, f, overwrite=True)
                except Exception as ex:  # noqa: BLE001 — non-fatal
                    print(f"[WARN] mckeypoints copy failed for {src_path}: {ex}")
                    n_mckp = -1
        f.attrs["has_mckeypoints"] = bool(n_mckp >= 0)

    info = {
        "output_path": output_path,
        "skipped_marker_path": None,
        "name": name, "sample_idx": int(sample_idx),
        "run": run, "subrun": subrun, "event": event,
        "n_sp_dataset": n_sp_dataset,
        "n_postdh": n_postdh,
        "n_cache": n_cache,
        "n_passes_tau_loose": int(passes_nominal.sum()),
        "n_gt_nu_in_cache": int(is_gt_nu_cached.sum()),
        "n_particle_instances": len(particle_instances),
        "cache_empty": False, "cascade_skipped": cascade_skipped,
        "deghost_tau": deghost_tau,
        "deghost_keep_frac": deghost_keep_frac,
        "n_queries_with_per_sp_data": int(keep_q_idx.size),
        "particle_summary": [
            {
                "pid": int(inst["pid"]),
                "pid_raw": int(inst.get("pid_raw", inst["pid"])),
                "class_id": int(inst.get("class_id", -1)),
                "ke_mev": float(inst.get("ke_mev", 0.0)),
                "n_truth_orig": int(inst["n_truth_points_orig"]),
                "n_truth_in_cache": int(inst["n_truth_points_in_cache"]),
            } for inst in particle_instances
        ],
    }
    return info


# ---------------------------------------------------------------------------
# Build helpers — reused by the shard driver
# ---------------------------------------------------------------------------

def build_model_from_stage3_config(cfg_path: str,
                                   device: torch.device,
                                   skip_checkpoints: bool = False):
    cfg = Config.fromfile(cfg_path)
    model_cfg = copy.deepcopy(cfg.model)
    if skip_checkpoints:
        model_cfg.pop("cascaded_slicer_weight", None)
        model_cfg.pop("particle_segmenter_backbone_weight", None)
        inner = model_cfg.get("cascaded_slicer", {})
        inner.pop("deghoster_weight", None)
        inner.pop("slicer_backbone_weight", None)
    model = build_model(model_cfg).to(device).eval()
    return cfg, model


def build_dataset_for_caching(cfg, *,
                              split: str = "val",
                              inputlist_override: Optional[str] = None,
                              data_root_override: Optional[str] = None,
                              max_spacepoints_override: Optional[int] = None,
                              gt_source_override: Optional[str] = "particle"):
    """Pick `cfg.data[split]` and force gt_source="particle" by default."""
    cfg_split = copy.deepcopy(getattr(cfg.data, split))
    if inputlist_override is not None:
        cfg_split["data_list_file"] = os.path.abspath(inputlist_override)
    if data_root_override is not None:
        cfg_split["data_root"] = data_root_override
    if max_spacepoints_override is not None:
        cfg_split["max_spacepoints"] = max_spacepoints_override
    if gt_source_override is not None:
        cfg_split["gt_source"] = gt_source_override
    return build_dataset(cfg_split)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True,
                   help="Stage-3 config (e.g. "
                        "configs/lartpc/larformer/stage3_particle/larformer-particle-v1.py).")
    p.add_argument("--inputlist", default=None,
                   help="Absolute path to a data list. Overrides cfg.")
    p.add_argument("--data-root", default=None,
                   help="Override cfg.data.<split>.data_root.")
    p.add_argument("--split", default="val",
                   choices=("train", "val", "test"))
    p.add_argument("--max-spacepoints", type=int, default=None,
                   help="Cap SPs per event for fast iteration.")
    p.add_argument("--sample-idx", type=int, required=True)
    p.add_argument("--output", required=True,
                   help="Cache .h5 output path.")
    p.add_argument("--device", default="cuda",
                   choices=("cuda", "cpu"))
    p.add_argument("--skip-checkpoints", action="store_true",
                   help="Don't load real checkpoints — random init only.")
    p.add_argument("--tau-loose-floor", type=float, default=0.2)
    p.add_argument("--tau-loose-nominal", type=float, default=0.5)
    p.add_argument("--tau-loose-delta", type=float, default=0.2)
    p.add_argument("--class-prob-floor", type=float, default=0.1,
                   help="Floor on per-query nu-class probability for "
                        "storing per-query mask_prob (default 0.1).")
    p.add_argument("--no-gzip", action="store_true",
                   help="Disable gzip compression on per-query mask_prob.")
    p.add_argument("--no-keypoints", action="store_true",
                   help="Do not copy the source mckeypoints group into the "
                        "cache (keypoint module won't train on this cache).")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--skipped-marker", default=None,
                   help="If the event has no cache SPs, write a "
                        ".skipped marker here. Default: <output>.skipped.")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA unavailable; falling back to CPU.")
        device = torch.device("cpu")

    print(f"Loading config + building model from {args.config} ...")
    t0 = time.perf_counter()
    cfg, model = build_model_from_stage3_config(
        args.config, device, skip_checkpoints=args.skip_checkpoints,
    )
    print(f"  build + checkpoint load took {time.perf_counter()-t0:.1f}s")

    print(f"Building dataset (split={args.split}, gt_source=particle) ...")
    dataset = build_dataset_for_caching(
        cfg, split=args.split,
        inputlist_override=args.inputlist,
        data_root_override=args.data_root,
        max_spacepoints_override=args.max_spacepoints,
    )
    print(f"  dataset has {len(dataset)} samples")
    if args.sample_idx < 0 or args.sample_idx >= len(dataset):
        raise IndexError(
            f"--sample-idx {args.sample_idx} out of range "
            f"[0, {len(dataset)})"
        )

    skipped_marker = (args.skipped_marker
                      if args.skipped_marker is not None
                      else args.output + ".skipped")
    print(f"Building cache for sample {args.sample_idx} → {args.output} ...")
    t0 = time.perf_counter()
    info = build_cache_event(
        model, dataset, args.sample_idx, args.output,
        device=device,
        tau_loose_floor=args.tau_loose_floor,
        tau_loose_nominal=args.tau_loose_nominal,
        tau_loose_delta=args.tau_loose_delta,
        class_prob_floor_for_query_storage=args.class_prob_floor,
        gzip_per_query_data=(not args.no_gzip),
        overwrite=args.overwrite,
        extra_attrs={"source_config": os.path.abspath(args.config)},
        skipped_marker_path=skipped_marker,
        copy_keypoints=(not args.no_keypoints),
    )
    print(f"  cache built in {time.perf_counter()-t0:.2f}s")
    print()
    print("Summary:")
    for k, v in info.items():
        if k == "particle_summary":
            print(f"  particle_summary:")
            for s in v:
                pid = s["pid"]; pid_raw = s["pid_raw"]; cls = s["class_id"]
                ke = s["ke_mev"]; n_o = s["n_truth_orig"]
                n_c = s["n_truth_in_cache"]
                print(f"    pid={pid:6d}  pid_raw={pid_raw:6d}  class={cls}  "
                      f"KE={ke:7.1f}MeV  n_orig={n_o:5d}  n_in_cache={n_c:5d}")
        else:
            print(f"  {k:30s} {v}")


if __name__ == "__main__":
    main()
