"""Stage-1+2 cache reader for Stage-3 LArFormer training.

Reads per-event HDF5 cache files produced by
`tools/build_stage12_cache_event.py` and emits dicts in the SAME shape
as `LArFormerDataset`, so the existing `larformer_collate` and the
particle segmenter's forward path can consume them unchanged.

The per-file schema (see the cache builder docstring) gives this reader
flexibility: a single cache file contains the UNION of
(Stage-2 nu-mask pass at the cache's `τ_loose_floor`) ∪ (GT-nu SPs).
The reader picks a SUBSET at __getitem__ time via `source_set_filter`,
so the trainer can run different selections per epoch / per iter
without re-generating the cache.

`source_set_filter` modes:
    "stage2_pass"             — `source_mask & 1` only (inference-realistic).
                                Default.
    "gt_nu"                   — `source_mask & 2` only (truth anchor).
    "stage2_delta"            — `source_mask & 4` (predicted-medium-confidence
                                + nominal, since 1 ⇒ 4).
    "union"                   — any nonzero source_mask (excludes the
                                floor-only noise bucket).
    "all"                     — every cached SP (incl. floor-only).
    "stage2_random_tau"       — sample τ_loose ~ U(`tau_loose_range`),
                                keep SPs with `stage2_nu_mask_prob > τ`.
                                Optionally OR with `gt_nu` when
                                `random_tau_include_gt=True`.
    "stage2_plus_gt_dropout"  — `stage2_pass` ∪ (random `gt_keep_prob`
                                fraction of GT-nu SPs not already in
                                stage2_pass). Used for curriculum /
                                mask-denoising training.

`recenter_to_centroid` (bool, default False): if True, subtract the
mean of `coord_norm` over the post-filter SP set from `coord_norm` and
each particle instance's `origin_coord_norm`. Match-time recentering
makes the model invariant to where the slice lives in the detector.

Compatibility note: the cache stores particle-level GT (`gt_source`=
"particle" semantics — origin_type=0 = nu, `class_id` in the Stage-3
7-class taxonomy, `pid_raw` for analysis, `origin_cm` for the origin
head).
"""
from __future__ import annotations

import os
from typing import Optional

import h5py
import numpy as np

from .builder import DATASETS
from .defaults import DefaultDataset


DEFAULT_BACKBONE_GRID_SIZE_CM = 0.25
DEFAULT_COORD_CENTER = (125.0, 0.0, 518.0)
DEFAULT_COORD_SCALE = 179.55

VALID_FILTERS = (
    "stage2_pass", "gt_nu", "stage2_delta",
    "union", "all",
    "stage2_random_tau", "stage2_plus_gt_dropout",
)


def _read_attr(v):
    if hasattr(v, "item") and getattr(v, "size", 1) == 1:
        return v.item()
    return v


def _read_particle_instances(grp: h5py.Group) -> list[dict]:
    out: list[dict] = []
    keys = sorted(grp.keys(),
                  key=lambda s: int(s.split("_")[-1])
                  if s.startswith("instance_") else -1)
    for k in keys:
        if not k.startswith("instance_"):
            continue
        g = grp[k]
        d: dict = {}
        for ak, av in g.attrs.items():
            d[ak] = _read_attr(av)
        for dk in g.keys():
            d[dk] = g[dk][...]
        out.append(d)
    return out


@DATASETS.register_module()
class LArFormerStage12CacheDataset(DefaultDataset):
    """Dataset over Stage-1+2 cache .h5 files for Stage-3 training.

    Args:
        data_root:               root dir holding the cache. May contain
                                  nested subdirs (the 3-level hash used by
                                  the shard driver) — files are discovered
                                  recursively.
        data_list_file:          (alternative to data_root crawl) text
                                  file with one cache .h5 path per line.
        source_set_filter:       see module docstring.
        tau_loose_range:         (low, high) for `stage2_random_tau` mode.
                                  Inclusive of low, exclusive of high.
        random_tau_include_gt:   for `stage2_random_tau`, also include
                                  `gt_nu` SPs even if their score is
                                  below the sampled τ. Lets the mask
                                  denoising path see truth anchors.
        gt_keep_prob:            for `stage2_plus_gt_dropout`, fraction of
                                  the GT-nu SPs not in stage2_pass to
                                  also include (random per __getitem__).
        recenter_to_centroid:    subtract per-event SP-centroid from
                                  coord_norm (post-filter).
        min_spacepoints:         minimum SPs that must survive
                                  `source_set_filter` for the event to be
                                  usable. Events below this are SKIPPED
                                  (the loader resamples another index)
                                  rather than raising. Default 1 (only
                                  filter-wiped, zero-SP events are skipped).
        max_resample_attempts:   how many alternate indices to try before
                                  giving up and raising. Default 20.
        coord_center, coord_scale:
                                  detector-frame normalization (must
                                  match the cache's). The cache writes
                                  these as attrs; we cross-check.
        backbone_grid_size_cm:   for recomputing `grid_coord` (the cache
                                  doesn't store it).
        loop, transform, split, test_mode, test_cfg, ignore_index, cache:
                                  standard DefaultDataset knobs.
    """

    def __init__(
        self,
        split="train",
        data_root: Optional[str] = None,
        data_list_file: Optional[str] = None,
        source_set_filter: str = "stage2_pass",
        tau_loose_range=(0.3, 0.7),
        random_tau_include_gt: bool = True,
        gt_keep_prob: float = 0.5,
        recenter_to_centroid: bool = False,
        min_spacepoints: int = 1,
        max_resample_attempts: int = 20,
        coord_center=DEFAULT_COORD_CENTER,
        coord_scale=DEFAULT_COORD_SCALE,
        backbone_grid_size_cm: float = DEFAULT_BACKBONE_GRID_SIZE_CM,
        ignore_unmatched_cache_attrs: bool = True,
        transform=None,
        loop: int = 1,
        test_mode: bool = False,
        test_cfg=None,
        cache: bool = False,
        ignore_index: int = -1,
    ):
        if source_set_filter not in VALID_FILTERS:
            raise ValueError(
                f"source_set_filter must be one of {VALID_FILTERS}; "
                f"got {source_set_filter!r}"
            )
        self.source_set_filter = source_set_filter
        self.tau_loose_range = tuple(float(x) for x in tau_loose_range)
        self.random_tau_include_gt = bool(random_tau_include_gt)
        self.gt_keep_prob = float(gt_keep_prob)
        self.recenter_to_centroid = bool(recenter_to_centroid)
        self.min_spacepoints = int(min_spacepoints)
        self.max_resample_attempts = int(max_resample_attempts)
        self.coord_center = np.asarray(coord_center, dtype=np.float32)
        self.coord_scale = float(coord_scale)
        self.backbone_grid_size_cm = float(backbone_grid_size_cm)
        self.ignore_unmatched_cache_attrs = bool(ignore_unmatched_cache_attrs)
        self.data_list_file = data_list_file
        super().__init__(
            split=split, data_root=data_root or "/",
            transform=transform, test_mode=test_mode,
            test_cfg=test_cfg, cache=cache,
            ignore_index=ignore_index, loop=loop,
        )

    # ------------------------------------------------------------------
    # DefaultDataset interface
    # ------------------------------------------------------------------

    def get_data_list(self) -> list[str]:
        data_list: list[str] = []
        if self.data_list_file is not None:
            list_file = (self.data_list_file
                         if os.path.isabs(self.data_list_file)
                         else os.path.join(self.data_root, self.data_list_file))
            if os.path.exists(list_file):
                with open(list_file, "r") as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith("#"):
                            p = (line if os.path.isabs(line)
                                 else os.path.join(self.data_root, line))
                            data_list.append(p)
        else:
            # Recursive crawl of data_root for .h5 cache files.
            for root, _dirs, files in os.walk(self.data_root):
                for fn in files:
                    if fn.endswith(".h5"):
                        data_list.append(os.path.join(root, fn))
        return sorted(data_list)

    def __len__(self) -> int:
        return len(self.data_list) * self.loop

    def get_data(self, idx: int) -> Optional[dict]:
        """Load one cache event. Returns None if fewer than
        `min_spacepoints` survive the source_set filter (caller resamples)."""
        path = self.data_list[idx % len(self.data_list)]
        return self._load_cache(path)

    def __getitem__(self, idx: int) -> dict:
        # Resample on under-populated events (filter wiped the slice below
        # min_spacepoints). Deterministic forward scan so val/test stay
        # reproducible and DDP ranks stay consistent; train shuffling is
        # handled by the sampler, not here.
        n = len(self.data_list)
        for attempt in range(self.max_resample_attempts):
            data = self.get_data(idx + attempt)
            if data is not None:
                if self.transform is not None:
                    data = self.transform(data)
                return data
        raise ValueError(
            f"{type(self).__name__}: no event with >= {self.min_spacepoints} "
            f"SPs under source_set_filter={self.source_set_filter!r} within "
            f"{self.max_resample_attempts} consecutive indices starting at "
            f"idx={idx % n}."
        )

    # ------------------------------------------------------------------
    # Cache reading
    # ------------------------------------------------------------------

    def _validate_cache_attrs(self, top_attrs: dict, path: str) -> None:
        """Cross-check the cache's coord_center / coord_scale match this
        dataset's expected values. If `ignore_unmatched_cache_attrs` is
        False, mismatch is a hard error."""
        cc = np.asarray(top_attrs.get("coord_center",
                                      self.coord_center.tolist()),
                        dtype=np.float32)
        cs = float(top_attrs.get("coord_scale", self.coord_scale))
        if not (np.allclose(cc, self.coord_center)
                and abs(cs - self.coord_scale) < 1e-3):
            msg = (
                f"Cache coord normalization mismatch for {path}: "
                f"cache center={cc.tolist()} scale={cs} vs dataset "
                f"center={self.coord_center.tolist()} scale={self.coord_scale}"
            )
            if not self.ignore_unmatched_cache_attrs:
                raise ValueError(msg)
            # else: silently accept — caller's responsibility.

    def _select_keep_mask(self, source_mask: np.ndarray,
                          stage2_prob: np.ndarray) -> np.ndarray:
        """Return a boolean keep mask over the cached SPs per the
        configured source_set_filter."""
        sm = source_mask
        if self.source_set_filter == "stage2_pass":
            return (sm & 1).astype(bool)
        if self.source_set_filter == "gt_nu":
            return (sm & 2).astype(bool)
        if self.source_set_filter == "stage2_delta":
            return (sm & 4).astype(bool)
        if self.source_set_filter == "union":
            return sm != 0
        if self.source_set_filter == "all":
            return np.ones_like(sm, dtype=bool)
        if self.source_set_filter == "stage2_random_tau":
            lo, hi = self.tau_loose_range
            tau = float(np.random.uniform(lo, hi))
            keep = stage2_prob > tau
            if self.random_tau_include_gt:
                keep = keep | (sm & 2).astype(bool)
            return keep
        if self.source_set_filter == "stage2_plus_gt_dropout":
            stage2 = (sm & 1).astype(bool)
            gt_only_extra = ((sm & 2).astype(bool)) & (~stage2)
            keep = stage2.copy()
            if gt_only_extra.any():
                rnd = np.random.rand(int(gt_only_extra.sum()))
                idx = np.where(gt_only_extra)[0]
                add = idx[rnd < self.gt_keep_prob]
                keep[add] = True
            return keep
        raise RuntimeError(self.source_set_filter)

    def _load_cache(self, path: str) -> Optional[dict]:
        with h5py.File(path, "r") as f:
            top_attrs = {k: _read_attr(v) for k, v in f.attrs.items()}
            self._validate_cache_attrs(top_attrs, path)
            e0 = f["entry_0"]
            run = int(e0.attrs.get("run",
                                    top_attrs.get("run", -1)))
            subrun = int(e0.attrs.get("subrun",
                                       top_attrs.get("subrun", -1)))
            event = int(e0.attrs.get("event",
                                      top_attrs.get("event", -1)))
            name = str(top_attrs.get("source_h5", os.path.basename(path)))

            coord = e0["coord"][:].astype(np.float32)
            coord_norm = e0["coord_norm"][:].astype(np.float32)
            feat = e0["feat"][:].astype(np.float32)
            lm_score = e0["lm_score"][:].astype(np.float32)
            wire = e0["wire"][:].astype(np.float32)
            trackid = e0["trackid"][:].astype(np.int64)
            pid = e0["pid"][:].astype(np.int64)
            origin_label = e0["origin_label"][:].astype(np.int64)
            hasmatch = e0["hasmatch"][:].astype(np.int64)
            ssnet_label = e0["ssnet_label"][:].astype(np.int64)
            slice_id = e0["slice_id"][:].astype(np.int64)
            source_mask = e0["source_mask"][:].astype(np.uint8)
            stage2_prob = e0["stage2_nu_mask_prob"][:].astype(np.float32)

            particles = (_read_particle_instances(e0["particle_instances"])
                         if "particle_instances" in e0 else [])

        # ---- 1. Apply the source_set filter ---------------------------
        keep = self._select_keep_mask(source_mask, stage2_prob)
        n_kept = int(keep.sum())
        if n_kept < self.min_spacepoints:
            # The source_set filter left too few SPs for this event to be
            # trainable (commonly 0 — the slice was wiped). Signal "skip"
            # to __getitem__, which resamples another index, instead of
            # crashing the whole DataLoader worker.
            return None

        coord = coord[keep]
        coord_norm = coord_norm[keep]
        feat = feat[keep]
        lm_score = lm_score[keep]
        wire = wire[keep]
        trackid = trackid[keep]
        pid = pid[keep]
        origin_label = origin_label[keep]
        hasmatch = hasmatch[keep]
        ssnet_label = ssnet_label[keep]
        slice_id = slice_id[keep]
        stage2_prob_kept = stage2_prob[keep]

        # ---- 2. Remap truth_indices for each particle -----------------
        remap = -np.ones(keep.shape[0], dtype=np.int64)
        remap[keep] = np.arange(n_kept)

        gt_instances: list[dict] = []
        for inst in particles:
            ti = np.asarray(inst.get("truth_indices",
                                      np.empty(0, dtype=np.int64)),
                            dtype=np.int64)
            if ti.size == 0:
                continue
            ti_kept = remap[ti]
            ti_kept = ti_kept[ti_kept >= 0]
            if ti_kept.size == 0:
                continue
            new_inst = dict(inst)
            new_inst["truth_indices"] = ti_kept.astype(np.int64)
            new_inst["n_truth_points"] = int(ti_kept.size)
            gt_instances.append(new_inst)

        # ---- 3. Optional centroid recentering -------------------------
        if self.recenter_to_centroid and n_kept > 0:
            centroid_norm = coord_norm.mean(axis=0)
            coord_norm = coord_norm - centroid_norm
            # `feat[:, :3]` is the coord_norm slot per the v6 convention
            # (built in LArFormerDataset as concat([coord_norm, pixval])).
            feat = feat.copy()
            feat[:, :3] = coord_norm
            for inst in gt_instances:
                if "origin_coord_norm" in inst:
                    inst["origin_coord_norm"] = (
                        np.asarray(inst["origin_coord_norm"],
                                   dtype=np.float32) - centroid_norm
                    ).astype(np.float32)

        # ---- 4. Backbone grid_coord -----------------------------------
        grid_coord = np.floor(
            coord / self.backbone_grid_size_cm
        ).astype(np.int64)
        if n_kept > 0:
            grid_coord -= grid_coord.min(axis=0)

        return {
            "coord": coord,
            "coord_norm": coord_norm.astype(np.float32),
            "grid_coord": grid_coord,
            "feat": feat.astype(np.float32),
            "lm_score": lm_score,
            "wire": wire,
            "trackid": trackid,
            "pid": pid,
            "origin_label": origin_label,
            "hasmatch": hasmatch,
            "ssnet_label": ssnet_label,
            "slice_id": slice_id,
            # Stage-2 telemetry — useful for ablation / per-SP weighting.
            "stage2_nu_mask_prob": stage2_prob_kept,
            "source_mask_kept": source_mask[keep],
            "gt_instances": gt_instances,
            "n_gt_instances": len(gt_instances),
            "n_spacepoints": int(n_kept),
            "lm_score_threshold": 0.0,
            "run": run, "subrun": subrun, "event": event,
            "name": name,
        }
