"""
ShowerClusteringDataset — input pipeline for the Mask2Former-style shower
clustering model. See `pointcept/docs/shower_clustering_design.md` (Phase 2).

Each sample is one event from a merged H5 (post the new merge step that
preserves `mc_particle_tree`). Per `__getitem__` the dataset returns a dict:

    coord            (N, 3)    detector cm
    coord_norm       (N, 3)    backbone-normalized coords
    feat             (N, 6)    coord_norm + log(pixval)  (backbone input)
    lm_score         (N,)      per-spacepoint larmatch score
    trackid          (N,)      per-spacepoint truth Geant4 trackid
    pid              (N,)      per-spacepoint truth PDG
    origin_label     (N,)      per-spacepoint truth origin (0=cosmic, 1=neutrino)
    hasmatch         (N,)      0 = ghost spacepoint, 1 = real
    ssnet_label      (N,)      per-spacepoint SSNet label
    wire             (N, 3)    per-spacepoint normalized wire coords
    voxel_id         (N,)      voxel index per spacepoint at fixed 5 cm grid
    voxel_keys       (V, 3)    integer voxel grid coordinates
    n_voxels         int       V
    fragment_indices list[ndarray]   per-fragment surviving spacepoint indices
    fragment_trackid (F,)      per-fragment plurality trackid
    fragment_pid     (F,)      per-fragment plurality PDG
    fragment_type    (F,)      per-fragment origin type (0..4)
    n_fragments      int       F
    gt_instances     list[dict] per-instance dict (see below)
    n_gt_instances   int
    n_spacepoints    int       N (after lm_score filter)
    lm_score_threshold float   τ used this sample
    run, subrun, event int     identity
    name             str       basename

Each `gt_instances[i]` dict has:
    trunk_trackid    int
    pid              int       (dominant truth PDG of this trunk's particle)
    truth_indices    (M,)      remapped surviving spacepoint indices
    n_truth_points   int

GT instances are built by walking `mc_particle_tree.parent_trackid` from each
unique non-(-1) `shower_fragments/trackid` (the trunk set), collecting all
descendants, then collecting per-spacepoint truth indices. See §4d of the
design doc for context.

Augmentation: per-event lm_score threshold τ. Sampled from
Uniform(`lm_score_aug_low`, `lm_score_aug_high`) on training; fixed at
`lm_score_val_threshold` on val/test. See §4e of the design doc.
"""

import os
from collections import defaultdict

import h5py
import numpy as np
import torch

from .builder import DATASETS
from .defaults import DefaultDataset


SHOWER_PIDS = (11, -11, 22)
DEFAULT_COORD_CENTER = (125.0, 0.0, 518.0)
DEFAULT_COORD_SCALE = 179.55
DEFAULT_VOXEL_SIZE_CM = 5.0


@DATASETS.register_module()
class ShowerClusteringDataset(DefaultDataset):
    """Per-event dataset for the shower-clustering Mask2Former model."""

    def __init__(
        self,
        split="train",
        data_root="data/lartpc",
        data_list_file=None,
        coord_center=DEFAULT_COORD_CENTER,
        coord_scale=DEFAULT_COORD_SCALE,
        voxel_size_cm=DEFAULT_VOXEL_SIZE_CM,
        backbone_grid_size_cm=0.25,
        max_spacepoints=None,
        lm_score_aug_low=0.15,
        lm_score_aug_high=0.40,
        lm_score_val_threshold=0.15,
        min_fragment_points_post_filter=20,
        log_transform_strength=True,
        wire_scale=1.0 / 3456.0,
        transform=None,
        loop=1,
        test_mode=False,
        test_cfg=None,
        cache=False,
        ignore_index=-1,
    ):
        self.coord_center = np.asarray(coord_center, dtype=np.float32)
        self.coord_scale = float(coord_scale)
        self.voxel_size_norm = float(voxel_size_cm) / self.coord_scale
        self.backbone_grid_size_cm = float(backbone_grid_size_cm)
        self.max_spacepoints = (int(max_spacepoints)
                                if max_spacepoints is not None else None)
        self.lm_score_aug_low = float(lm_score_aug_low)
        self.lm_score_aug_high = float(lm_score_aug_high)
        self.lm_score_val_threshold = float(lm_score_val_threshold)
        self.min_fragment_points_post_filter = int(min_fragment_points_post_filter)
        self.log_transform_strength = bool(log_transform_strength)
        self.wire_scale = float(wire_scale)
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

    # ---- DefaultDataset interface --------------------------------------------

    def get_data_list(self):
        data_list = []
        if self.data_list_file is not None:
            list_file = (
                self.data_list_file
                if os.path.isabs(self.data_list_file)
                else os.path.join(self.data_root, self.data_list_file)
            )
            if os.path.exists(list_file):
                with open(list_file, "r") as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith("#"):
                            p = line if os.path.isabs(line) else os.path.join(
                                self.data_root, line)
                            data_list.append(p)
        if not data_list:
            split_file = os.path.join(self.data_root, f"{self.split}.txt")
            if os.path.exists(split_file):
                with open(split_file, "r") as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith("#"):
                            p = line if os.path.isabs(line) else os.path.join(
                                self.data_root, line)
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

    # ---- Helpers --------------------------------------------------------------

    def _sample_threshold(self):
        if self.split == "train":
            return float(np.random.uniform(
                self.lm_score_aug_low, self.lm_score_aug_high))
        return self.lm_score_val_threshold

    def _build_voxel_ids(self, coord_norm):
        """5 cm voxel grid in normalized coords. Returns voxel_id (N,) and
        voxel_keys (V, 3) integer grid coords."""
        v_int = np.floor(coord_norm / self.voxel_size_norm).astype(np.int64)
        v_keys, voxel_id = np.unique(v_int, axis=0, return_inverse=True)
        return voxel_id.astype(np.int64), v_keys.astype(np.int64)

    @staticmethod
    def _build_children_map(mpt_group):
        """Build {parent_trackid: [child_trackid, ...]} from mc_particle_tree."""
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
        """BFS over children_map starting at root_tid (inclusive)."""
        out = set()
        stack = [int(root_tid)]
        while stack:
            cur = stack.pop()
            if cur in out:
                continue
            out.add(cur)
            stack.extend(children_map.get(cur, []))
        return out

    def _load_event(self, entry, path):
        td = entry["triplet_data"]
        sf = entry["shower_fragments"]
        mpt = entry["mc_particle_tree"] if "mc_particle_tree" in entry else None

        pos = td["pos"][:].astype(np.float32)
        n_sp = pos.shape[0]
        if n_sp == 0:
            raise ValueError(f"empty event in {path}")

        lm_score = td["lm_score"][:].astype(np.float32)

        # Strength: log(pixval) per plane
        pixval = td["pixval"][:].astype(np.float32)
        if self.log_transform_strength:
            pixval = np.log1p(np.clip(pixval, 0.0, None))

        # Wire features (normalized)
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

        # ---- 1. lm_score augmentation filter ----
        tau = self._sample_threshold()
        keep = lm_score >= tau
        n_keep = int(keep.sum())
        if n_keep == 0:
            raise ValueError(
                f"lm_score threshold {tau:.3f} dropped all spacepoints in {path}"
            )

        # remap orig_idx -> new_idx (or -1 if dropped)
        remap = np.full(n_sp, -1, dtype=np.int64)
        remap[keep] = np.arange(n_keep)

        # Apply filter to per-spacepoint arrays
        pos_k = pos[keep]
        lm_score_k = lm_score[keep]
        pixval_k = pixval[keep]
        wire_k = wire[keep]
        sp_trackid_k = sp_trackid[keep]
        sp_pid_k = sp_pid[keep]
        sp_origin_k = sp_origin[keep]
        sp_hasmatch_k = sp_hasmatch[keep]
        sp_ssnet_k = sp_ssnet[keep]

        # ---- 2. Backbone grid_coord + dedup at 0.25 cm ---------------------
        # PT-v3's serialized attention assumes one entry per grid cell (V3's
        # pipeline runs GridSample before feeding the backbone). Without
        # dedup, duplicate grid_coord values cause out-of-bounds indexing in
        # the encoder's stride pooling. We replicate V3's behavior here.
        grid_coord_full = np.floor(
            pos_k / self.backbone_grid_size_cm
        ).astype(np.int64)
        grid_coord_full -= grid_coord_full.min(axis=0)
        # First-occurrence dedup, preserving original order.
        _, first_idx = np.unique(grid_coord_full, axis=0, return_index=True)
        keep_dedup = np.sort(first_idx)

        # Apply dedup to the post-lm-filter arrays
        pos_k = pos_k[keep_dedup]
        lm_score_k = lm_score_k[keep_dedup]
        pixval_k = pixval_k[keep_dedup]
        wire_k = wire_k[keep_dedup]
        sp_trackid_k = sp_trackid_k[keep_dedup]
        sp_pid_k = sp_pid_k[keep_dedup]
        sp_origin_k = sp_origin_k[keep_dedup]
        sp_hasmatch_k = sp_hasmatch_k[keep_dedup]
        sp_ssnet_k = sp_ssnet_k[keep_dedup]
        grid_coord = grid_coord_full[keep_dedup]

        # Compose the lm-filter and dedup-filter into a single
        # original_index -> final_index remap. `remap` was previously the
        # lm-filter remap (orig -> post-lm); we extend it through dedup so
        # that downstream fragment / GT bookkeeping uses final-array indices.
        n_post_lm = n_keep
        n_dedup = len(keep_dedup)
        post_lm_to_dedup = np.full(n_post_lm, -1, dtype=np.int64)
        post_lm_to_dedup[keep_dedup] = np.arange(n_dedup)
        valid_mask = remap >= 0
        final_remap = np.full(n_sp, -1, dtype=np.int64)
        final_remap[valid_mask] = post_lm_to_dedup[remap[valid_mask]]
        remap = final_remap
        n_keep = n_dedup

        # ---- 2c. Optional spacepoint cap (memory bound) --------------------
        # When `max_spacepoints` is set and the post-dedup count exceeds it,
        # randomly subsample down. Mainly used to bound per-event VRAM during
        # training; if you want the cap on training only, override the
        # train-split kwargs in the config.
        # We compose this filter into `remap` so all downstream
        # fragment / gt-instance bookkeeping uses the final indexing.
        if (self.max_spacepoints is not None
                and n_keep > self.max_spacepoints):
            cap_perm = np.random.permutation(n_keep)[:self.max_spacepoints]
            cap_perm.sort()  # preserve original ordering
            pos_k = pos_k[cap_perm]
            lm_score_k = lm_score_k[cap_perm]
            pixval_k = pixval_k[cap_perm]
            wire_k = wire_k[cap_perm]
            sp_trackid_k = sp_trackid_k[cap_perm]
            sp_pid_k = sp_pid_k[cap_perm]
            sp_origin_k = sp_origin_k[cap_perm]
            sp_hasmatch_k = sp_hasmatch_k[cap_perm]
            sp_ssnet_k = sp_ssnet_k[cap_perm]
            grid_coord = grid_coord[cap_perm]

            n_post_dedup = n_keep
            n_after_cap = self.max_spacepoints
            dedup_to_cap = np.full(n_post_dedup, -1, dtype=np.int64)
            dedup_to_cap[cap_perm] = np.arange(n_after_cap)
            valid_mask = remap >= 0
            new_remap = np.full(n_sp, -1, dtype=np.int64)
            new_remap[valid_mask] = dedup_to_cap[remap[valid_mask]]
            remap = new_remap
            n_keep = n_after_cap

        # ---- 3. Normalize coords + build backbone input feat ---------------
        coord_norm = (pos_k - self.coord_center) / self.coord_scale
        feat = np.concatenate([coord_norm, pixval_k], axis=-1).astype(np.float32)

        # ---- 4. Voxel ids on surviving spacepoints (5 cm) ------------------
        voxel_id, voxel_keys = self._build_voxel_ids(coord_norm)

        # ---- 5. Fragment membership (filter & remap) ----
        num_frags_orig = int(sf.attrs.get("num_fragments", 0))
        if num_frags_orig > 0:
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
        else:
            flat = np.zeros(0, dtype=np.int64)
            counts = np.zeros(0, dtype=np.int64)
            frag_trackid_orig = np.zeros(0, dtype=np.int64)
            frag_pid_orig = np.zeros(0, dtype=np.int64)
            frag_type_orig = np.zeros(0, dtype=np.int64)

        fragment_indices = []
        fragment_trackid = []
        fragment_pid = []
        fragment_type = []
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

        # ---- 6. GT instances from mc_particle_tree descendants ----
        # Per GT instance we need: descendant truth-spacepoint indices (for
        # mask losses), the origin class (for the CE loss), and the
        # normalized origin coordinate (for the auxiliary L1 loss).
        # Origin type and origin point come from the first surviving fragment
        # with this trunk trackid — multiple fragments with the same trackid
        # share the same shower so should agree.
        gt_instances = []
        if mpt is not None and len(fragment_trackid) > 0:
            children_map, tid_to_pid = self._build_children_map(mpt)
            # Map from trackid -> originpt and origin_type taken from the
            # first matching fragment in the original (pre-filter) shower
            # fragments table. originpt is in detector cm, normalize before
            # storing.
            orig_originpt = sf["originpt"][:].astype(np.float32) \
                if "originpt" in sf else np.zeros((num_frags_orig, 3),
                                                  dtype=np.float32)
            tid_to_origin: dict = {}
            tid_to_type: dict = {}
            for fi in range(num_frags_orig):
                tid = int(frag_trackid_orig[fi])
                if tid <= 0:
                    continue
                tid_to_origin.setdefault(tid, orig_originpt[fi].copy())
                tid_to_type.setdefault(tid, int(frag_type_orig[fi]))

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
                truth_idx_new = remap[truth_idx_orig]
                truth_idx_new = truth_idx_new[truth_idx_new >= 0]
                if len(truth_idx_new) == 0:
                    continue
                origin_cm = tid_to_origin.get(
                    tid, np.zeros(3, dtype=np.float32))
                origin_norm = (origin_cm - self.coord_center) / self.coord_scale
                gt_instances.append({
                    "trunk_trackid": int(tid),
                    "pid": int(tid_to_pid.get(int(tid), -1)),
                    "origin_type": int(tid_to_type.get(tid, -1)),
                    "origin_coord_norm": origin_norm.astype(np.float32),
                    "truth_indices": truth_idx_new.astype(np.int64),
                    "n_truth_points": int(len(truth_idx_new)),
                })

        # ---- 7. Identity ----
        run = int(entry.attrs.get("run", -1))
        subrun = int(entry.attrs.get("subrun", -1))
        event = int(entry.attrs.get("event", -1))

        return {
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
            "voxel_id": voxel_id,
            "voxel_keys": voxel_keys,
            "n_voxels": int(voxel_keys.shape[0]),
            "fragment_indices": fragment_indices,
            "fragment_trackid": np.asarray(fragment_trackid, dtype=np.int64),
            "fragment_pid": np.asarray(fragment_pid, dtype=np.int64),
            "fragment_type": np.asarray(fragment_type, dtype=np.int64),
            "n_fragments": len(fragment_indices),
            "gt_instances": gt_instances,
            "n_gt_instances": len(gt_instances),
            "n_spacepoints": int(n_keep),
            "n_spacepoints_unfiltered": int(n_sp),
            "lm_score_threshold": float(tau),
            "run": run,
            "subrun": subrun,
            "event": event,
            "name": os.path.basename(path),
        }


def shower_clustering_collate(batch):
    """
    Custom collate for ShowerClusteringDataset.

    The dataset returns variable-size per-event tensors plus a list of
    fragments and a list of GT-instance dicts. We don't try to flatten
    fragments/instances across events; instead we pack per-spacepoint and
    per-voxel arrays with batch offsets, and keep the lists nested.

    Output dict keys:
        coord, coord_norm, feat, lm_score, wire, trackid, pid, origin_label,
        hasmatch, ssnet_label, voxel_id, voxel_keys
            — flat tensors with per-event concatenation
        offset (B,)        cumulative spacepoint count per event
        voxel_offset (B,)  cumulative voxel count per event
        fragment_indices_per_event   list[B] of list[F_b] of (M,) tensors
        fragment_trackid_per_event   list[B] of (F_b,) tensors
        fragment_pid_per_event       list[B] of (F_b,) tensors
        fragment_type_per_event      list[B] of (F_b,) tensors
        gt_instances_per_event       list[B] of list[K_b] of dict
        names, runs, subruns, events                — per-event identity
        n_spacepoints, n_voxels, n_fragments,
        n_gt_instances, lm_score_threshold          — per-event scalars
    """
    keys_flat = ("coord", "coord_norm", "grid_coord", "feat", "lm_score",
                 "wire", "trackid", "pid", "origin_label", "hasmatch",
                 "ssnet_label", "voxel_id")
    out = {}
    # Per-spacepoint flat concatenation
    for k in keys_flat:
        arrs = [s[k] for s in batch]
        out[k] = torch.from_numpy(np.concatenate(arrs, axis=0))

    # voxel_id needs to be globally unique across batch — offset per-event
    voxel_offsets_local = [0]
    for s in batch[:-1]:
        voxel_offsets_local.append(voxel_offsets_local[-1] + s["n_voxels"])
    # rewrite voxel_id with offset
    rewritten = []
    sp_so_far = 0
    for i, s in enumerate(batch):
        n = s["n_spacepoints"]
        rewritten.append(out["voxel_id"][sp_so_far:sp_so_far + n] +
                         voxel_offsets_local[i])
        sp_so_far += n
    out["voxel_id"] = torch.cat(rewritten, dim=0)

    # voxel_keys concatenated (B-event keys laid end to end)
    vk = [torch.from_numpy(s["voxel_keys"]) for s in batch]
    out["voxel_keys"] = torch.cat(vk, dim=0)

    # offsets (cumulative counts) — convention compatible with point_collate_fn
    n_per_event = np.array([s["n_spacepoints"] for s in batch], dtype=np.int64)
    out["offset"] = torch.from_numpy(np.cumsum(n_per_event))
    nvox_per_event = np.array([s["n_voxels"] for s in batch], dtype=np.int64)
    out["voxel_offset"] = torch.from_numpy(np.cumsum(nvox_per_event))

    # Per-event lists (kept as Python lists; model handles per-event)
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
    out["gt_instances_per_event"] = [s["gt_instances"] for s in batch]

    # Per-event scalars
    out["n_spacepoints"] = torch.from_numpy(n_per_event)
    out["n_voxels"] = torch.from_numpy(nvox_per_event)
    out["n_fragments"] = torch.tensor(
        [s["n_fragments"] for s in batch], dtype=torch.int64)
    out["n_gt_instances"] = torch.tensor(
        [s["n_gt_instances"] for s in batch], dtype=torch.int64)
    out["lm_score_threshold"] = torch.tensor(
        [s["lm_score_threshold"] for s in batch], dtype=torch.float32)

    # Identity
    out["names"] = [s["name"] for s in batch]
    out["runs"] = [s["run"] for s in batch]
    out["subruns"] = [s["subrun"] for s in batch]
    out["events"] = [s["event"] for s in batch]

    return out
