"""
LArFormerDataset — input pipeline for LArFormer (see
`Pointcept/docs/LArFormer.md` §7).

Reads the same merged_h5 files that `ShowerClusteringDataset` does, but:

  - voxelization is NOT done here (the model voxelizes on-demand at its
    declared voxel levels — config-only changes to grid sizes work without
    regenerating data),
  - fragment fields are OPTIONAL (only built when `emit_fragments=True`),
  - the GT-instance source is PLUGGABLE via `gt_source`:

      "shower_trunk"   — one instance per unique shower-fragment trunk
                          trackid; descendants merged. Matches the
                          ShowerClusteringDataset semantics (used to feed
                          LArFormer the same supervision as the legacy
                          shower-clustering Mask2Former).
      "slice"          — one instance per event slice. Cosmic primaries each
                          get a slice; all origin==1 (nu) primaries are
                          merged per nu_vertex into one slice. Backed by
                          `lartpc_data_prep/slice_labels.py`.
      "deghost"        — empty `gt_instances` list. Stage-1 deghosting
                          doesn't need queries / set prediction; only the
                          per-level cls head (`label_src="hasmatch"` or
                          similar) is trained.
      "particle"       — NOT IMPLEMENTED in v1; raises NotImplementedError.

Per-event dict shape (always present):

    coord            (N, 3)    detector cm
    coord_norm       (N, 3)    backbone-normalized coords
    grid_coord       (N, 3)    backbone PT-v3 dedup grid coords
    feat             (N, 6)    coord_norm + log(pixval)
    lm_score         (N,)
    wire             (N, 3)
    trackid, pid, origin_label, hasmatch, ssnet_label, slice_id  per-SP
    gt_instances     list[K] of dicts (schema depends on gt_source)
    n_spacepoints    int
    n_gt_instances   int
    lm_score_threshold float
    run, subrun, event int
    name             str

When `emit_fragments=True`, additionally:
    fragment_indices  list[F] of (M_f,) int64
    fragment_trackid  (F,) int64
    fragment_pid      (F,) int64
    fragment_type     (F,) int64
    n_fragments       int

`gt_instances[k]` dict schema by source:

  shower_trunk:
    trunk_trackid, pid, origin_type ∈ 0..4, origin_coord_norm (3,),
    truth_indices (M,), n_truth_points

  slice:
    primary_trackid, pid, primary_origin ∈ {1, 2},
    origin_type ∈ user's class map (default {1: 0, 2: 1}),
    origin_coord_norm (3,)  = normalized primary_start_pos,
    truth_indices (M,), n_truth_points

  deghost:
    no instances (`gt_instances == []`)
"""

import os
from collections import defaultdict

import h5py
import numpy as np

from .builder import DATASETS
from .defaults import DefaultDataset


SHOWER_PIDS = (11, -11, 22)
DEFAULT_COORD_CENTER = (125.0, 0.0, 518.0)
DEFAULT_COORD_SCALE = 179.55
DEFAULT_BACKBONE_GRID_SIZE_CM = 0.25
DEFAULT_SLICE_CLASS_MAP = {1: 0, 2: 1}  # primary_origin → origin_type

# gt_source="particle" default taxonomy. abs(PDG) → class_id in the 7-class
# Stage-3 taxonomy {e: 0, γ: 1, μ: 2, π: 3, p: 4, other_track: 5}; the matcher's
# no_object slot is index 7 (set by `LArFormer.num_classes - 1`).
DEFAULT_PARTICLE_CLASS_MAP = {
    11:   0,   # e±
    22:   1,   # γ
    13:   2,   # μ±
    211:  3,   # π±
    2212: 4,   # p   (compute_particle_labels relabels n → p by default,
               #      so 2112 lands here too)
}


@DATASETS.register_module()
class LArFormerDataset(DefaultDataset):
    """Per-event dataset for LArFormer.

    Args:
        gt_source:               "shower_trunk", "slice", "deghost", "particle"
        emit_fragments:          if True, also emit fragment fields. Required
                                  for any LArFormer config that lists a
                                  FragmentBuilder level.
        slice_class_map:         dict mapping primary_origin ∈ {1, 2} to the
                                  integer class slot used in gt_instance[k]
                                  ["origin_type"]. Defaults to {1: 0, 2: 1}
                                  (nu→0, cosmic→1). Set to None for the raw
                                  primary_origin values.
        merge_nu_slices:         pass-through to compute_slice_labels.
                                  Default True (collapse all origin==1
                                  primaries per nu_vertex into one slice).
        shower_trunk_label_source: "truth" / "fragment" / "union" — matches
                                    the same kwarg in ShowerClusteringDataset.
        Other kwargs              same semantics as ShowerClusteringDataset
                                  (lm_score augmentation, max_spacepoints
                                  cap, backbone grid size, etc.)
    """

    def __init__(
        self,
        split="train",
        data_root="data/lartpc",
        data_list_file=None,
        coord_center=DEFAULT_COORD_CENTER,
        coord_scale=DEFAULT_COORD_SCALE,
        backbone_grid_size_cm=DEFAULT_BACKBONE_GRID_SIZE_CM,
        max_spacepoints=None,
        lm_score_aug_low=0.15,
        lm_score_aug_high=0.40,
        lm_score_val_threshold=0.15,
        min_fragment_points_post_filter=20,
        strength_clip_max=1000.0,
        strength_log_min_val=0.01,
        strength_log_max_val=1000.0,
        wire_scale=1.0 / 3456.0,
        gt_source="slice",
        emit_fragments=False,
        slice_class_map=None,
        slice_origin_kind="primary_start_pos",
        merge_nu_slices=True,
        shower_trunk_label_source="truth",
        # ---- gt_source="particle" knobs --------------------------------
        particle_ke_thresholds=None,
        particle_other_ke_threshold=None,
        particle_never_visible_pdgs=None,
        particle_pdg_relabel=None,
        particle_nu_origin=1,
        particle_class_map=None,
        particle_other_class_id=5,
        min_particle_points_post_filter=0,
        transform=None,
        loop=1,
        test_mode=False,
        test_cfg=None,
        cache=False,
        ignore_index=-1,
        **deprecated_kwargs,
    ):
        """
        ...
        Strength preprocessing (standardized — matches the Sonata-v1m1 v6
        pretrain and the LoRA-deghoster fine-tuning pipelines exactly):

            strength = LogTransform_v6(clip(pixval, 0, strength_clip_max))

        which is, per pixval x ≥ 0:

            strength = 2 · (log10(x + min_val) − log10(min_val))
                         / (log10(max_val + min_val) − log10(min_val)) − 1

        producing values normalized into [−1, +1]. This is the canonical
        per-SP strength input that all v6-compatible backbones (Sonata
        pretrain, LoRA-fine-tuned deghoster, …) were trained on.

        Knobs (all default to the v6 settings):
            strength_clip_max:   upper clamp on pixval (default 1000).
                                  None disables the clamp.
            strength_log_min_val: LogTransform min_val (default 0.01).
            strength_log_max_val: LogTransform max_val (default 1000).
        """
        # Old log1p mode + the `log_transform_strength` boolean toggle have
        # been removed. If a config still passes `log_transform_strength`,
        # raise loudly so the user knows nothing silently changed semantics.
        for legacy in ("log_transform_strength", "strength_transform"):
            if legacy in deprecated_kwargs:
                raise TypeError(
                    f"LArFormerDataset no longer accepts `{legacy}`. The "
                    f"strength transform is now fixed at the v6 "
                    f"LogTransform (clip→log10→normalize to [-1,1]) to "
                    f"match the Sonata-v1m1 pretrained backbone. Remove "
                    f"the `{legacy}` kwarg from your config; tune via "
                    f"`strength_clip_max`, `strength_log_{{min,max}}_val`."
                )
        if deprecated_kwargs:
            raise TypeError(
                f"LArFormerDataset got unexpected kwargs: "
                f"{list(deprecated_kwargs.keys())}"
            )
        if gt_source not in ("shower_trunk", "slice", "deghost", "particle"):
            raise ValueError(
                f"gt_source must be one of "
                f"('shower_trunk', 'slice', 'deghost', 'particle'); "
                f"got {gt_source!r}"
            )
        # S3.0: gt_source="particle" wires lartpc_data_prep.slice_labels.
        # compute_particle_labels into the per-event GT pipeline. See
        # `_gt_from_particles` and `docs/LArFormer_particlesegment_stage.md`.
        if shower_trunk_label_source not in ("truth", "fragment", "union"):
            raise ValueError(
                f"shower_trunk_label_source must be 'truth' / 'fragment' "
                f"/ 'union'; got {shower_trunk_label_source!r}"
            )
        self.coord_center = np.asarray(coord_center, dtype=np.float32)
        self.coord_scale = float(coord_scale)
        self.backbone_grid_size_cm = float(backbone_grid_size_cm)
        self.max_spacepoints = (int(max_spacepoints)
                                if max_spacepoints is not None else None)
        self.lm_score_aug_low = float(lm_score_aug_low)
        self.lm_score_aug_high = float(lm_score_aug_high)
        self.lm_score_val_threshold = float(lm_score_val_threshold)
        self.min_fragment_points_post_filter = int(min_fragment_points_post_filter)
        self.strength_clip_max = (float(strength_clip_max)
                                  if strength_clip_max is not None else None)
        self.strength_log_min_val = float(strength_log_min_val)
        self.strength_log_max_val = float(strength_log_max_val)
        self.wire_scale = float(wire_scale)
        self.gt_source = gt_source
        self.emit_fragments = bool(emit_fragments)
        self.slice_class_map = (dict(slice_class_map)
                                if slice_class_map is not None
                                else dict(DEFAULT_SLICE_CLASS_MAP))
        if slice_origin_kind not in ("primary_start_pos", "centroid"):
            raise ValueError(
                f"slice_origin_kind must be 'primary_start_pos' or "
                f"'centroid'; got {slice_origin_kind!r}"
            )
        self.slice_origin_kind = slice_origin_kind
        self.merge_nu_slices = bool(merge_nu_slices)
        self.shower_trunk_label_source = shower_trunk_label_source
        # gt_source="particle" knobs — None falls through to
        # compute_particle_labels' own defaults; the class-id map defaults
        # to the 7-class taxonomy used by the Stage-3 LArFormer config.
        self.particle_ke_thresholds = (
            dict(particle_ke_thresholds)
            if particle_ke_thresholds is not None else None)
        self.particle_other_ke_threshold = (
            float(particle_other_ke_threshold)
            if particle_other_ke_threshold is not None else None)
        self.particle_never_visible_pdgs = (
            frozenset(int(p) for p in particle_never_visible_pdgs)
            if particle_never_visible_pdgs is not None else None)
        self.particle_pdg_relabel = (
            dict(particle_pdg_relabel)
            if particle_pdg_relabel is not None else None)
        self.particle_nu_origin = int(particle_nu_origin)
        # abs(PDG) → class id. Default: {e:0, γ:1, μ:2, π:3, p:4}; everything
        # else falls through to `particle_other_class_id` (default 5 =
        # "other_track" in the Stage-3 taxonomy).
        self.particle_class_map = (
            {int(k): int(v) for k, v in particle_class_map.items()}
            if particle_class_map is not None
            else dict(DEFAULT_PARTICLE_CLASS_MAP))
        self.particle_other_class_id = int(particle_other_class_id)
        self.min_particle_points_post_filter = int(
            min_particle_points_post_filter)
        self.data_list_file = data_list_file
        super().__init__(
            split=split,
            data_root=data_root,
            transform=transform,
            test_mode=test_mode,
            test_cfg=test_cfg,
            cache=cache,
            ignore_index=ignore_index,
            loop=loop,
        )

    # ---- DefaultDataset interface ------------------------------------

    def get_data_list(self):
        data_list = []
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
        if not data_list:
            split_file = os.path.join(self.data_root, f"{self.split}.txt")
            if os.path.exists(split_file):
                with open(split_file, "r") as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith("#"):
                            p = (line if os.path.isabs(line)
                                 else os.path.join(self.data_root, line))
                            data_list.append(p)
        return sorted(data_list)

    def __len__(self):
        return len(self.data_list) * self.loop

    def get_data(self, idx):
        path = self.data_list[idx % len(self.data_list)]
        with h5py.File(path, "r") as f:
            entry = f["entry_0"]
            return self._load_event(entry, path)

    def __getitem__(self, idx):
        data = self.get_data(idx)
        if self.transform is not None:
            data = self.transform(data)
        return data

    # ---- Helpers -----------------------------------------------------

    def _sample_threshold(self):
        if self.split == "train":
            return float(np.random.uniform(
                self.lm_score_aug_low, self.lm_score_aug_high))
        return self.lm_score_val_threshold

    @staticmethod
    def _build_children_map(mpt_group):
        tids = mpt_group["trackid"][:].astype(np.int64)
        parents = mpt_group["parent_trackid"][:].astype(np.int64)
        pids = mpt_group["pid"][:].astype(np.int64)
        children = defaultdict(list)
        tid_to_pid = {}
        for i, t in enumerate(tids):
            children[int(parents[i])].append(int(t))
            tid_to_pid[int(t)] = int(pids[i])
        return children, tid_to_pid

    @staticmethod
    def _descendants(root_tid, children_map):
        out = set()
        stack = [int(root_tid)]
        while stack:
            cur = stack.pop()
            if cur in out:
                continue
            out.add(cur)
            stack.extend(children_map.get(cur, []))
        return out

    # ---- Main loader -------------------------------------------------

    def _load_event(self, entry, path):
        td = entry["triplet_data"]
        sf = entry["shower_fragments"]
        mpt = entry["mc_particle_tree"] if "mc_particle_tree" in entry else None

        pos = td["pos"][:].astype(np.float32)
        n_sp = pos.shape[0]
        if n_sp == 0:
            raise ValueError(f"empty event in {path}")

        lm_score = td["lm_score"][:].astype(np.float32)
        pixval = td["pixval"][:].astype(np.float32)
        # v6 strength preprocessing (standard, non-optional — must match the
        # Sonata-v1m1 v6 pretrain / LoRA-deghoster pipeline).
        #
        # See pointcept.datasets.transform.LogTransform — this code
        # reproduces it inline so the dataset's output `feat` column matches
        # what the trained backbone weights were trained on, without
        # requiring the caller to also assemble a `transform=[LogTransform]`
        # pipeline.
        pixval = np.clip(pixval, 0.0, self.strength_clip_max).astype(np.float32)
        pixval = pixval + self.strength_log_min_val
        log_min = np.log10(self.strength_log_min_val)
        log_max = np.log10(self.strength_log_max_val + self.strength_log_min_val)
        pixval = 2.0 * (np.log10(pixval) - log_min) / (log_max - log_min) - 1.0
        if all(k in td for k in ("uwire", "vwire", "ywire")):
            wire = np.stack([
                td["uwire"][:].astype(np.float32) * self.wire_scale,
                td["vwire"][:].astype(np.float32) * self.wire_scale,
                td["ywire"][:].astype(np.float32) * self.wire_scale,
            ], axis=-1)
        else:
            wire = np.zeros((n_sp, 3), dtype=np.float32)

        sp_trackid = td["trackid"][:].astype(np.int64)
        sp_pid = td["pid"][:].astype(np.int64)
        sp_origin = (td["origin"][:].astype(np.int64)
                     if "origin" in td else np.full(n_sp, -1, dtype=np.int64))
        sp_hasmatch = (td["hasmatch"][:].astype(np.int64)
                       if "hasmatch" in td else np.ones(n_sp, dtype=np.int64))
        sp_ssnet = (td["ssnet_label"][:].astype(np.int64)
                    if "ssnet_label" in td else np.full(n_sp, -1, dtype=np.int64))

        # ---- 1. lm_score aug filter --------------------------------
        tau = self._sample_threshold()
        keep = lm_score >= tau
        n_keep = int(keep.sum())
        if n_keep == 0:
            raise ValueError(
                f"lm_score threshold {tau:.3f} dropped all spacepoints in {path}"
            )
        remap = np.full(n_sp, -1, dtype=np.int64)
        remap[keep] = np.arange(n_keep)

        pos_k = pos[keep]; lm_score_k = lm_score[keep]; pixval_k = pixval[keep]
        wire_k = wire[keep]; sp_trackid_k = sp_trackid[keep]
        sp_pid_k = sp_pid[keep]; sp_origin_k = sp_origin[keep]
        sp_hasmatch_k = sp_hasmatch[keep]; sp_ssnet_k = sp_ssnet[keep]

        # ---- 2. Backbone grid_coord + dedup ------------------------
        grid_coord_full = np.floor(
            pos_k / self.backbone_grid_size_cm
        ).astype(np.int64)
        grid_coord_full -= grid_coord_full.min(axis=0)
        _, first_idx = np.unique(grid_coord_full, axis=0, return_index=True)
        keep_dedup = np.sort(first_idx)

        pos_k = pos_k[keep_dedup]; lm_score_k = lm_score_k[keep_dedup]
        pixval_k = pixval_k[keep_dedup]; wire_k = wire_k[keep_dedup]
        sp_trackid_k = sp_trackid_k[keep_dedup]; sp_pid_k = sp_pid_k[keep_dedup]
        sp_origin_k = sp_origin_k[keep_dedup]; sp_hasmatch_k = sp_hasmatch_k[keep_dedup]
        sp_ssnet_k = sp_ssnet_k[keep_dedup]; grid_coord = grid_coord_full[keep_dedup]

        post_lm_to_dedup = np.full(n_keep, -1, dtype=np.int64)
        post_lm_to_dedup[keep_dedup] = np.arange(len(keep_dedup))
        valid_mask = remap >= 0
        final_remap = np.full(n_sp, -1, dtype=np.int64)
        final_remap[valid_mask] = post_lm_to_dedup[remap[valid_mask]]
        remap = final_remap
        n_keep = len(keep_dedup)

        # ---- 2c. max_spacepoints cap -------------------------------
        if (self.max_spacepoints is not None
                and n_keep > self.max_spacepoints):
            cap_perm = np.random.permutation(n_keep)[:self.max_spacepoints]
            cap_perm.sort()
            pos_k = pos_k[cap_perm]; lm_score_k = lm_score_k[cap_perm]
            pixval_k = pixval_k[cap_perm]; wire_k = wire_k[cap_perm]
            sp_trackid_k = sp_trackid_k[cap_perm]; sp_pid_k = sp_pid_k[cap_perm]
            sp_origin_k = sp_origin_k[cap_perm]; sp_hasmatch_k = sp_hasmatch_k[cap_perm]
            sp_ssnet_k = sp_ssnet_k[cap_perm]; grid_coord = grid_coord[cap_perm]

            dedup_to_cap = np.full(n_keep, -1, dtype=np.int64)
            dedup_to_cap[cap_perm] = np.arange(self.max_spacepoints)
            valid_mask = remap >= 0
            new_remap = np.full(n_sp, -1, dtype=np.int64)
            new_remap[valid_mask] = dedup_to_cap[remap[valid_mask]]
            remap = new_remap
            n_keep = self.max_spacepoints

        # ---- 3. Normalize coords + backbone feat -------------------
        coord_norm = (pos_k - self.coord_center) / self.coord_scale
        feat = np.concatenate([coord_norm, pixval_k], axis=-1).astype(np.float32)

        # ---- 4. (Optional) per-SP slice_id -------------------------
        if mpt is not None:
            from lartpc_data_prep.slice_labels import compute_slice_labels
            slice_info = compute_slice_labels(
                mpt, sp_trackid_k, sp_hasmatch=sp_hasmatch_k,
                merge_nu_slices=self.merge_nu_slices,
            )
            slice_id_k = slice_info["slice_id"]
        else:
            slice_info = None
            slice_id_k = np.full(n_keep, -1, dtype=np.int64)

        # ---- 5. Fragments (always compute; conditional emit) -------
        # Compute is cheap; emit only if requested OR if a FragmentBuilder
        # level needs them (caller controls this via emit_fragments).
        fragment_indices, fragment_trackid, fragment_pid, fragment_type = \
            self._build_fragments(sf, remap, n_sp)

        # ---- 6. Build gt_instances per gt_source -------------------
        if self.gt_source == "shower_trunk":
            gt_instances = self._gt_from_shower_trunks(
                mpt=mpt, sf=sf, remap=remap, sp_trackid=sp_trackid,
                n_sp=n_sp,
                fragment_indices=fragment_indices,
                fragment_trackid=fragment_trackid,
            )
        elif self.gt_source == "slice":
            gt_instances = self._gt_from_slices(slice_info, n_keep, coord_norm)
        elif self.gt_source == "particle":
            gt_instances = self._gt_from_particles(
                mpt, sp_trackid_k, sp_hasmatch_k, n_keep,
            )
        elif self.gt_source == "deghost":
            gt_instances = []
        else:  # pragma: no cover — already validated in __init__
            raise RuntimeError(self.gt_source)

        # ---- 7. Identity -------------------------------------------
        run = int(entry.attrs.get("run", -1))
        subrun = int(entry.attrs.get("subrun", -1))
        event = int(entry.attrs.get("event", -1))

        # Per-SP particle class id. Real when GT instances carry a class_id
        # (gt_source="particle"); otherwise -1. Emitted so a particle
        # segmenter whose voxel level declares cls supervision on
        # `particle_class_id` (e.g. the ptv3crosslevel config used for
        # full-cascade inference) can build its per-level target. A -1 stub
        # produces an all-ignore target — the trained cls head still runs
        # (it feeds mixed_query_selection); only the unused loss target is
        # masked out. This mirrors the cache's augmented `particle_class_id`
        # field for the raw-merged_h5 inference path.
        particle_class_id_k = np.full(n_keep, -1, dtype=np.int64)
        for g in gt_instances:
            if "class_id" in g and "truth_indices" in g:
                ti = np.asarray(g["truth_indices"], dtype=np.int64)
                if ti.size:
                    particle_class_id_k[ti] = int(g["class_id"])

        out = {
            "coord": pos_k,
            "coord_norm": coord_norm.astype(np.float32),
            "grid_coord": grid_coord,
            "feat": feat,
            "lm_score": lm_score_k,
            "wire": wire_k,
            "trackid": sp_trackid_k,
            "pid": sp_pid_k,
            "origin_label": sp_origin_k,
            "hasmatch": sp_hasmatch_k,
            "ssnet_label": sp_ssnet_k,
            "slice_id": slice_id_k,
            "particle_class_id": particle_class_id_k,
            "gt_instances": gt_instances,
            "n_gt_instances": len(gt_instances),
            "n_spacepoints": int(n_keep),
            "n_spacepoints_unfiltered": int(n_sp),
            "lm_score_threshold": float(tau),
            "run": run, "subrun": subrun, "event": event,
            "name": os.path.basename(path),
        }
        if self.emit_fragments:
            out["fragment_indices"] = fragment_indices
            out["fragment_trackid"] = np.asarray(fragment_trackid, dtype=np.int64)
            out["fragment_pid"] = np.asarray(fragment_pid, dtype=np.int64)
            out["fragment_type"] = np.asarray(fragment_type, dtype=np.int64)
            out["n_fragments"] = len(fragment_indices)
        return out

    # ---- Fragment & GT builders --------------------------------------

    def _build_fragments(self, sf, remap, n_sp):
        num_frags_orig = int(sf.attrs.get("num_fragments", 0))
        if num_frags_orig == 0:
            return [], [], [], []
        flat = sf["pointindices_flat"][:].astype(np.int64)
        counts = sf["pointindices_counts"][:].astype(np.int64)
        frag_trackid_orig = (sf["trackid"][:].astype(np.int64)
                             if "trackid" in sf
                             else np.full(num_frags_orig, -1, dtype=np.int64))
        frag_pid_orig = (sf["pid"][:].astype(np.int64)
                         if "pid" in sf
                         else np.full(num_frags_orig, -1, dtype=np.int64))
        frag_type_orig = (sf["type"][:].astype(np.int64)
                          if "type" in sf
                          else np.full(num_frags_orig, -1, dtype=np.int64))
        fragment_indices, fragment_trackid, fragment_pid, fragment_type = \
            [], [], [], []
        offset = 0
        for fi in range(num_frags_orig):
            n = int(counts[fi])
            orig_idx = flat[offset:offset + n]
            offset += n
            orig_idx = orig_idx[(orig_idx >= 0) & (orig_idx < n_sp)]
            new_idx = remap[orig_idx]
            new_idx = new_idx[new_idx >= 0]
            if len(new_idx) < self.min_fragment_points_post_filter:
                continue
            fragment_indices.append(new_idx.astype(np.int64))
            fragment_trackid.append(int(frag_trackid_orig[fi]))
            fragment_pid.append(int(frag_pid_orig[fi]))
            fragment_type.append(int(frag_type_orig[fi]))
        return fragment_indices, fragment_trackid, fragment_pid, fragment_type

    def _gt_from_shower_trunks(self, mpt, sf, remap, sp_trackid, n_sp,
                               fragment_indices, fragment_trackid):
        """Reproduce ShowerClusteringDataset's trunk-walk gt_instances."""
        if mpt is None or not fragment_trackid:
            return []
        num_frags_orig = int(sf.attrs.get("num_fragments", 0))
        children_map, tid_to_pid = self._build_children_map(mpt)
        orig_originpt = (sf["originpt"][:].astype(np.float32)
                         if "originpt" in sf
                         else np.zeros((num_frags_orig, 3), dtype=np.float32))
        # Map trackid → (originpt, origin_type) from the first matching
        # *original* fragment in the H5 (before our min-fragment filter).
        if "trackid" in sf:
            sf_trackid = sf["trackid"][:].astype(np.int64)
        else:
            sf_trackid = np.full(num_frags_orig, -1, dtype=np.int64)
        if "type" in sf:
            sf_type = sf["type"][:].astype(np.int64)
        else:
            sf_type = np.full(num_frags_orig, -1, dtype=np.int64)
        tid_to_origin = {}
        tid_to_type = {}
        for fi in range(num_frags_orig):
            tid = int(sf_trackid[fi])
            if tid <= 0:
                continue
            tid_to_origin.setdefault(tid, orig_originpt[fi].copy())
            tid_to_type.setdefault(tid, int(sf_type[fi]))

        gt_instances = []
        seen_trunks = set()
        for tid in fragment_trackid:
            if tid <= 0 or tid in seen_trunks:
                continue
            seen_trunks.add(tid)
            desc = self._descendants(tid, children_map)
            if not desc:
                continue
            desc_arr = np.fromiter(desc, dtype=np.int64, count=len(desc))
            truth_idx_orig = np.where(np.isin(sp_trackid, desc_arr))[0]
            truth_part = remap[truth_idx_orig]
            truth_part = truth_part[truth_part >= 0].astype(np.int64)
            desc_set = set(int(x) for x in desc_arr.tolist())
            frag_chunks = [
                idx for idx, ft in zip(fragment_indices, fragment_trackid)
                if int(ft) in desc_set
            ]
            if frag_chunks:
                fragment_part = np.unique(
                    np.concatenate(frag_chunks)
                ).astype(np.int64)
            else:
                fragment_part = np.zeros(0, dtype=np.int64)
            if self.shower_trunk_label_source == "truth":
                truth_idx_new = truth_part
            elif self.shower_trunk_label_source == "fragment":
                truth_idx_new = fragment_part
            else:
                if truth_part.size and fragment_part.size:
                    truth_idx_new = np.unique(
                        np.concatenate([truth_part, fragment_part])
                    ).astype(np.int64)
                elif truth_part.size:
                    truth_idx_new = truth_part
                else:
                    truth_idx_new = fragment_part
            if len(truth_idx_new) == 0:
                continue
            origin_cm = tid_to_origin.get(tid, np.zeros(3, dtype=np.float32))
            origin_norm = (origin_cm - self.coord_center) / self.coord_scale
            gt_instances.append({
                "trunk_trackid": int(tid),
                "pid": int(tid_to_pid.get(int(tid), -1)),
                "origin_type": int(tid_to_type.get(tid, -1)),
                "origin_coord_norm": origin_norm.astype(np.float32),
                "truth_indices": truth_idx_new.astype(np.int64),
                "n_truth_points": int(len(truth_idx_new)),
            })
        return gt_instances

    def _gt_from_slices(self, slice_info, n_keep, coord_norm):
        """One instance per non-ghost slice.

        `origin_coord_norm` is either:
          - the primary particle start position (default), or
          - the centroid of the slice's spacepoints in coord_norm space,
        depending on `self.slice_origin_kind`.
        """
        if slice_info is None:
            return []
        slice_id = slice_info["slice_id"]                # (N,)
        primary_trackid = slice_info["primary_trackid"]  # (S,)
        primary_pid = slice_info["primary_pid"]
        primary_origin = slice_info["primary_origin"]
        primary_start_pos = slice_info["primary_start_pos"]
        primary_n = slice_info["primary_n_spacepoints"]
        gt_instances = []
        for k, key in enumerate(primary_trackid):
            key = int(key)
            truth_idx = np.where(slice_id == key)[0].astype(np.int64)
            if truth_idx.size == 0:
                continue
            origin = int(primary_origin[k])
            origin_type = self.slice_class_map.get(origin, origin)
            if self.slice_origin_kind == "centroid":
                origin_norm = coord_norm[truth_idx].mean(axis=0)
            else:
                origin_norm = (
                    primary_start_pos[k] - self.coord_center
                ) / self.coord_scale
            gt_instances.append({
                "primary_trackid": key,
                "primary_origin": origin,
                "pid": int(primary_pid[k]),
                "origin_type": int(origin_type),
                "origin_coord_norm": origin_norm.astype(np.float32),
                "truth_indices": truth_idx,
                "n_truth_points": int(primary_n[k]),
            })
        return gt_instances

    def _gt_from_particles(self, mpt, sp_trackid_k, sp_hasmatch_k, n_keep):
        """Per-particle GT instances for Stage-3 particle clustering.

        Each instance corresponds to a single visible nu-origin particle
        (post-KE-threshold + post-blacklist + parent-walk merging into
        the nearest visible ancestor). Truth indices index into the
        post-filter SP array (length `n_keep`). PDGs are POST-relabel
        (e.g. n → p) per `compute_particle_labels`.
        """
        if mpt is None:
            return []
        from lartpc_data_prep.slice_labels import compute_particle_labels
        kwargs = dict(nu_origin=self.particle_nu_origin)
        if self.particle_ke_thresholds is not None:
            kwargs["ke_thresholds"] = self.particle_ke_thresholds
        if self.particle_other_ke_threshold is not None:
            kwargs["other_ke_threshold"] = self.particle_other_ke_threshold
        if self.particle_never_visible_pdgs is not None:
            kwargs["never_visible_pdgs"] = self.particle_never_visible_pdgs
        if self.particle_pdg_relabel is not None:
            kwargs["pdg_relabel"] = self.particle_pdg_relabel
        pinfo = compute_particle_labels(
            mpt, sp_trackid_k, sp_hasmatch=sp_hasmatch_k, **kwargs,
        )
        slice_id = pinfo["slice_id"]                    # (N_keep,)
        primary_trackid = pinfo["primary_trackid"]      # (S,)
        primary_pid = pinfo["primary_pid"]              # post-relabel
        primary_pid_raw = pinfo["primary_pid_raw"]
        primary_ke_MeV = pinfo["primary_ke_MeV"]
        primary_start_pos = pinfo["primary_start_pos"]        # raw truth (cm)
        # SCE-applied start position — aligns with reco SP frame. Falls back
        # to raw start_pos inside compute_particle_labels if start_pos_sce
        # isn't in the H5, so this field is always present.
        primary_start_pos_sce = pinfo["primary_start_pos_sce"]
        primary_n = pinfo["primary_n_spacepoints"]
        gt_instances = []
        for k, tid in enumerate(primary_trackid):
            tid = int(tid)
            truth_idx = np.where(slice_id == tid)[0].astype(np.int64)
            if truth_idx.size < self.min_particle_points_post_filter:
                continue
            pdg = int(primary_pid[k])
            class_id = int(self.particle_class_map.get(
                abs(pdg), self.particle_other_class_id))
            origin_cm = primary_start_pos_sce[k].astype(np.float32)
            origin_cm_truth = primary_start_pos[k].astype(np.float32)
            origin_norm = (
                (origin_cm - self.coord_center) / self.coord_scale
            ).astype(np.float32)
            gt_instances.append({
                "primary_trackid": tid,
                "pid": pdg,
                "pid_raw": int(primary_pid_raw[k]),
                # LArFormerLoss reads `origin_type` as the per-instance
                # class slot for the matcher (legacy name from the slicer
                # GT schema, where origin_type ∈ {0=nu, 1=cosmic}). For
                # the particle taxonomy we map it to class_id; the
                # original "always nu-origin" flag is implicit since
                # compute_particle_labels only walks nu-origin tracks.
                "origin_type": class_id,
                "class_id": class_id,
                # origin_cm / origin_coord_norm are the SCE-applied (reco-
                # frame) birth point — this is what the origin head should
                # predict and what the visualizer plots. The raw-truth
                # birth point is kept separately for analysis only.
                "origin_coord_norm": origin_norm,
                "origin_cm": origin_cm,
                "origin_cm_truth": origin_cm_truth,
                "ke_mev": float(primary_ke_MeV[k]),
                "truth_indices": truth_idx,
                "n_truth_points": int(truth_idx.size),
            })
        return gt_instances


# ---------------------------------------------------------------------------
# Collate
# ---------------------------------------------------------------------------

def larformer_collate(batch):
    """Custom collate for LArFormerDataset.

    Flat-batches per-spacepoint tensors with `offset` (cumulative count) for
    per-event slicing. Per-event lists (fragments, gt_instances) are kept
    nested. Voxel fields are NOT emitted (LArFormer voxelizes model-side).

    Output dict keys (always):
        coord, coord_norm, grid_coord, feat, lm_score, wire,
        trackid, pid, origin_label, hasmatch, ssnet_label, slice_id
            — flat tensors with per-event concatenation
        offset                       (B,) cumulative spacepoint count
        gt_instances_per_event        list[B] of list[K_b] of dict
        n_spacepoints, n_gt_instances, lm_score_threshold,
        names, runs, subruns, events  per-event scalars

    When the dataset emits fragments (emit_fragments=True), additionally:
        fragment_indices_per_event   list[B] of list[F_b] of (M,) tensors
        fragment_trackid_per_event   list[B] of (F_b,) tensors
        fragment_pid_per_event       list[B] of (F_b,) tensors
        fragment_type_per_event      list[B] of (F_b,) tensors
        n_fragments                  (B,) tensor
    """
    import torch

    keys_flat = ("coord", "coord_norm", "grid_coord", "feat", "lm_score",
                 "wire", "trackid", "pid", "origin_label", "hasmatch",
                 "ssnet_label", "slice_id")
    # Optional per-SP fields — concatenated only when every sample in the
    # batch has them. Lets `LArFormerStage12CacheDataset` emit extra fields
    # (e.g. `particle_class_id`) without breaking legacy datasets.
    optional_flat_keys = ("particle_class_id",)
    out = {}
    for k in keys_flat:
        arrs = [s[k] for s in batch]
        out[k] = torch.from_numpy(np.concatenate(arrs, axis=0))
    for k in optional_flat_keys:
        if all(k in s for s in batch):
            arrs = [s[k] for s in batch]
            out[k] = torch.from_numpy(np.concatenate(arrs, axis=0))

    n_per_event = np.array([s["n_spacepoints"] for s in batch], dtype=np.int64)
    out["offset"] = torch.from_numpy(np.cumsum(n_per_event))

    out["gt_instances_per_event"] = [s["gt_instances"] for s in batch]
    out["n_spacepoints"] = torch.from_numpy(n_per_event)
    out["n_gt_instances"] = torch.tensor(
        [s["n_gt_instances"] for s in batch], dtype=torch.int64)
    out["lm_score_threshold"] = torch.tensor(
        [s["lm_score_threshold"] for s in batch], dtype=torch.float32)
    out["names"] = [s["name"] for s in batch]
    out["runs"] = [s["run"] for s in batch]
    out["subruns"] = [s["subrun"] for s in batch]
    out["events"] = [s["event"] for s in batch]

    if "fragment_indices" in batch[0]:
        out["fragment_indices_per_event"] = [
            [torch.from_numpy(idx) for idx in s["fragment_indices"]]
            for s in batch
        ]
        out["fragment_trackid_per_event"] = [
            torch.from_numpy(s["fragment_trackid"]) for s in batch
        ]
        out["fragment_pid_per_event"] = [
            torch.from_numpy(s["fragment_pid"]) for s in batch
        ]
        out["fragment_type_per_event"] = [
            torch.from_numpy(s["fragment_type"]) for s in batch
        ]
        out["n_fragments"] = torch.tensor(
            [s["n_fragments"] for s in batch], dtype=torch.int64)
    return out
