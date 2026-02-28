"""
Shower Origin Dataset

Dataset class for shower origin prediction task. Extends LArTPCDataset
to include shower fragment masks and origin point information.

Supports both single-shower (V2 compatibility) and multi-shower modes.
Handles photon showers (pid=22) and electron/positron showers (pid=11/-11).

Supports two HDF5 formats:
1. New flat format (from C++ ShowerFragmentOriginMaker):
   - shower_fragments/@num_fragments: int attribute
   - shower_fragments/trackid: (F,) int
   - shower_fragments/pid: (F,) int
   - shower_fragments/istrunk: (F,) int
   - shower_fragments/type: (F,) int — 0=nu-inside, 1=outside, 2=cosmic-inside
   - shower_fragments/startpt: (F, 3) float
   - shower_fragments/originpt: (F, 3) float — prediction target (moved to startpt for outside showers)
   - shower_fragments/pret0shiftedoriginpt: (F, 4) float — true MC origin (x,y,z,t), no t0 shift
   - shower_fragments/pointindices_flat: (T,) long
   - shower_fragments/pointindices_counts: (F,) int

   Origin type is determined in C++ using pret0shiftedoriginpt to check
   if the true origin is inside/outside the TPC.

2. Legacy per-fragment format (from Python labeler):
   - shower_fragments/fragment_0/, fragment_1/, etc.

Author: Claude Code Assistant
"""

import os
import h5py
import numpy as np
from collections.abc import Sequence

import torch

from .builder import DATASETS
from .defaults import DefaultDataset


# Sentinel value for padded (invalid) shower entries
SHOWER_PAD_SENTINEL = -999.0


@DATASETS.register_module()
class ShowerOriginDataset(DefaultDataset):
    """
    Dataset for shower origin prediction task with multi-shower support.

    Each sample contains:
    - Full event point cloud
    - All valid shower fragments for the event (up to max_showers_per_event)
    - Ground truth origin location and type per shower
    - Padded arrays for batching compatibility

    Indexing is per-EVENT (not per-fragment). Each event can contribute
    multiple showers, all returned in a single sample.

    Args:
        split: Dataset split (train/val/test)
        data_root: Root directory for data
        data_list_file: Path to file listing HDF5 files
        transform: List of transform configs
        coord_scale: Scale factor for coordinates (default: 1.0)
        use_reco_coords: Whether to use reconstructed coordinates
        log_transform_edep: Whether to apply log transform to energy deposits
        gaussian_sigma: Sigma for Gaussian distance labels (in coord units)
        max_showers_per_event: Maximum showers to include per event (default: 4)
        min_shower_points: Minimum points for a valid shower fragment (legacy format)
        min_fragment_points: Minimum points for a DBSCAN fragment to be included (default: 20)
        include_cosmic_showers: Whether to include cosmic-origin showers
        balance_origins: Whether to balance neutrino/cosmic samples
        **kwargs: Additional arguments passed to DefaultDataset
    """

    VALID_ASSETS = [
        "coord",
        "strength",
        "color",
        "segment",
        "shower_mask",
        "shower_masks",
        "origin_coord",
        "origin_coords",
        "origin_types",
        "origin_distance",
        "num_showers",
        "valid_showers_mask",
        "particle_pids",
        "start_coords",
        "fragment_trackids",
    ]

    ORIGIN_TYPES = ["inside", "outside", "on_track"]

    def __init__(
        self,
        split="train",
        data_root="data/lartpc",
        data_list_file=None,
        transform=None,
        coord_scale=1.0,
        use_reco_coords=True,
        log_transform_edep=True,
        gaussian_sigma=4.0,
        max_showers_per_event=4,
        min_shower_points=10,
        min_fragment_points=20,
        include_cosmic_showers=True,
        balance_origins=False,
        wire_scale=1.0/3456.0,
        **kwargs,
    ):
        self.coord_scale = coord_scale
        self.use_reco_coords = use_reco_coords
        self.log_transform_edep = log_transform_edep
        self.gaussian_sigma = gaussian_sigma
        self.max_showers_per_event = max_showers_per_event
        self.min_shower_points = min_shower_points
        self.min_fragment_points = min_fragment_points
        self.include_cosmic_showers = include_cosmic_showers
        self.balance_origins = balance_origins
        self.wire_scale = wire_scale
        self.data_list_file = data_list_file

        super().__init__(
            split=split,
            data_root=data_root,
            transform=transform,
            **kwargs,
        )

        # Build event-level index: sample_idx -> file_idx
        self._build_fragment_index()

    def get_data_list(self):
        """Get list of HDF5 file paths."""
        data_list = []

        # Option 1: Explicit file list
        if self.data_list_file is not None:
            if os.path.isabs(self.data_list_file):
                list_file = self.data_list_file
            else:
                list_file = os.path.join(self.data_root, self.data_list_file)

            if os.path.exists(list_file):
                with open(list_file, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#'):
                            if os.path.isabs(line):
                                data_list.append(line)
                            else:
                                data_list.append(os.path.join(self.data_root, line))

        # Option 2: Look for split.txt file
        if not data_list:
            split_file = os.path.join(self.data_root, f"{self.split}.txt")
            if os.path.exists(split_file):
                with open(split_file, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#'):
                            if os.path.isabs(line):
                                data_list.append(line)
                            else:
                                data_list.append(os.path.join(self.data_root, line))

        # Option 3: Look for split directory
        if not data_list:
            split_dir = os.path.join(self.data_root, self.split)
            if os.path.isdir(split_dir):
                for f in sorted(os.listdir(split_dir)):
                    if f.endswith('.h5') or f.endswith('.hdf5'):
                        data_list.append(os.path.join(split_dir, f))

        return sorted(data_list)

    def _build_fragment_index(self):
        """
        Build event-level index mapping sample_idx -> file_idx.

        Each event (file) is one sample, regardless of how many shower
        fragments it contains. The per-fragment filtering (min points,
        origin type) is applied at load time in get_data().

        Supports both the new flat format (pointindices_flat) and
        the legacy per-fragment group format.
        """
        self.event_index = []

        for file_idx, filepath in enumerate(self.data_list):
            try:
                with h5py.File(filepath, 'r') as f:
                    # Check if shower_fragments group exists
                    if 'entry_0/shower_fragments' not in f:
                        continue

                    sf = f['entry_0/shower_fragments']

                    # New flat format: check num_fragments attribute
                    if 'pointindices_flat' in sf:
                        num_frags = int(sf.attrs.get('num_fragments', 0))
                        if num_frags > 0:
                            self.event_index.append(file_idx)
                    else:
                        # Legacy per-fragment group format
                        fragment_names = [k for k in sf.keys()
                                         if k.startswith('fragment_')]
                        n_valid = 0
                        for frag_name in fragment_names:
                            frag = sf[frag_name]
                            if 'has_valid_tracking' in frag:
                                if not bool(frag['has_valid_tracking'][()]):
                                    continue
                            if 'mask' in frag:
                                mask = frag['mask'][:]
                                if mask.sum() < self.min_shower_points:
                                    continue
                            if not self.include_cosmic_showers and 'origin_type' in frag:
                                origin_type = frag['origin_type'][()]
                                if origin_type == 1:
                                    continue
                            n_valid += 1
                        if n_valid > 0:
                            self.event_index.append(file_idx)

            except Exception as e:
                print(f"Warning: Could not process {filepath}: {e}")
                continue

        # Legacy compatibility: also keep fragment_index for V2 fallback
        self.fragment_index = self.event_index

    def __len__(self):
        return len(self.event_index) * self.loop

    def get_data(self, idx):
        """
        Load all valid shower fragments for a single event.

        Returns:
            data_dict: Dict containing:
                - coord: (N, 3) point coordinates
                - strength: (N, C) energy/charge features
                - color: (N, 3) wire features
                - shower_mask: (N,) union of all shower masks
                - shower_masks: (max_showers, N) padded individual shower masks
                - origin_coords: (max_showers, 3) padded origin coordinates
                - origin_types: (max_showers,) padded origin types
                - num_showers: int, actual number of valid showers
                - valid_showers_mask: (max_showers,) bool, which slots are valid
                - origin_is_vertex: (max_showers,) bool, whether origin is a vertex
                - origin_coord: (3,) first shower origin (V2 compatibility)
                - origin_distance: (N,) distance to first origin (V2 compatibility)
                - name: sample identifier
        """
        idx = idx % len(self.event_index)
        file_idx = self.event_index[idx]
        filepath = self.data_list[file_idx]

        with h5py.File(filepath, 'r') as f:
            entry = f['entry_0']
            triplet = entry['triplet_data']

            # Load point cloud data
            if self.use_reco_coords and 'pos' in triplet:
                coord = triplet['pos'][:].astype(np.float32) * self.coord_scale
            else:
                coord = triplet['pos'][:].astype(np.float32) * self.coord_scale

            n_points = coord.shape[0]

            # Load energy features
            if 'pixval' in triplet:
                pixval = triplet['pixval'][:].astype(np.float32)
                if self.log_transform_edep:
                    pixval = np.log1p(np.clip(pixval, 0, None))
                strength = pixval
            else:
                strength = np.ones((n_points, 1), dtype=np.float32)

            # Load wire coordinates as color/additional features
            if all(k in triplet for k in ['uwire', 'vwire', 'ywire']):
                uwire = triplet['uwire'][:].astype(np.float32) * self.wire_scale
                vwire = triplet['vwire'][:].astype(np.float32) * self.wire_scale
                ywire = triplet['ywire'][:].astype(np.float32) * self.wire_scale
                color = np.stack([uwire, vwire, ywire], axis=-1)
            else:
                color = np.zeros((n_points, 3), dtype=np.float32)

            # Collect all valid shower fragments for this event
            shower_masks_list = []
            origin_coords_list = []
            start_coords_list = []
            origin_types_list = []
            is_vertex_list = []
            particle_pids_list = []
            fragment_trackids_list = []

            if 'shower_fragments' in entry:
                sf = entry['shower_fragments']

                # Detect format: new flat format vs legacy per-fragment groups
                if 'pointindices_flat' in sf:
                    # === New flat format from C++ ShowerFragmentOriginMaker ===
                    num_frags = int(sf.attrs.get('num_fragments', 0))
                    if num_frags > 0:
                        trackids = sf['trackid'][:]
                        pids = sf['pid'][:]
                        istrunk = sf['istrunk'][:]
                        frag_types = sf['type'][:]
                        startpts = sf['startpt'][:]
                        originpts = sf['originpt'][:]
                        flat_indices = sf['pointindices_flat'][:]
                        index_counts = sf['pointindices_counts'][:]

                        offset = 0
                        for i in range(num_frags):
                            if len(shower_masks_list) >= self.max_showers_per_event:
                                break

                            count = int(index_counts[i])
                            indices = flat_indices[offset:offset+count]
                            offset += count

                            # Build boolean mask
                            fmask = np.zeros(n_points, dtype=bool)
                            valid_idx = indices[indices < n_points]
                            fmask[valid_idx] = True

                            if fmask.sum() < self.min_fragment_points:
                                continue

                            # Origin type determined by C++ using pret0shiftedoriginpt:
                            # 0=nu-inside, 1=outside, 2=cosmic-inside
                            frag_origin_type = int(frag_types[i])

                            if not self.include_cosmic_showers and frag_origin_type == 2:
                                continue

                            frag_origin = originpts[i].astype(np.float32) * self.coord_scale
                            frag_start = startpts[i].astype(np.float32) * self.coord_scale
                            frag_is_vertex = (frag_origin_type == 0)

                            shower_masks_list.append(fmask)
                            origin_coords_list.append(frag_origin)
                            start_coords_list.append(frag_start)
                            origin_types_list.append(frag_origin_type)
                            is_vertex_list.append(frag_is_vertex)
                            particle_pids_list.append(int(pids[i]))
                            fragment_trackids_list.append(int(trackids[i]))
                else:
                    # === Legacy per-fragment group format ===
                    fragment_names = sorted(
                        [k for k in sf.keys() if k.startswith('fragment_')]
                    )
                    for frag_name in fragment_names:
                        if len(shower_masks_list) >= self.max_showers_per_event:
                            break
                        frag = sf[frag_name]
                        if 'has_valid_tracking' in frag:
                            if not bool(frag['has_valid_tracking'][()]):
                                continue
                        if 'mask' in frag:
                            fmask = frag['mask'][:].astype(bool)
                        elif 'indices' in frag:
                            fmask = np.zeros(n_points, dtype=bool)
                            indices = frag['indices'][:]
                            fmask[indices] = True
                        else:
                            continue
                        if fmask.sum() < self.min_shower_points:
                            continue
                        if 'origin_type' in frag:
                            frag_origin_type = int(frag['origin_type'][()])
                        else:
                            frag_origin_type = 0
                        if not self.include_cosmic_showers and frag_origin_type == 1:
                            continue
                        if 'origin_coord' in frag:
                            frag_origin = frag['origin_coord'][:].astype(np.float32) * self.coord_scale
                        else:
                            frag_origin = coord[fmask].mean(axis=0)
                        if 'is_vertex' in frag:
                            frag_is_vertex = bool(frag['is_vertex'][()])
                        else:
                            frag_is_vertex = (frag_origin_type == 0)
                        if 'particle_pid' in frag:
                            frag_pid = int(frag['particle_pid'][()])
                        else:
                            frag_pid = 22
                        shower_masks_list.append(fmask)
                        origin_coords_list.append(frag_origin)
                        start_coords_list.append(frag_origin)  # legacy: no startpt, use origin
                        origin_types_list.append(frag_origin_type)
                        is_vertex_list.append(frag_is_vertex)
                        particle_pids_list.append(frag_pid)
                        fragment_trackids_list.append(-1)  # legacy: no trackid

        num_showers = len(shower_masks_list)
        max_s = self.max_showers_per_event

        # Build union mask
        if num_showers > 0:
            union_mask = np.zeros(n_points, dtype=bool)
            for m in shower_masks_list:
                union_mask = union_mask | m
        else:
            union_mask = np.zeros(n_points, dtype=bool)

        # Pad shower_masks to (max_showers_per_event, N)
        padded_shower_masks = np.zeros((max_s, n_points), dtype=np.float32)
        for i in range(num_showers):
            padded_shower_masks[i, :] = shower_masks_list[i].astype(np.float32)

        # Pad origin_coords to (max_showers_per_event, 3)
        padded_origin_coords = np.full((max_s, 3), SHOWER_PAD_SENTINEL, dtype=np.float32)
        for i in range(num_showers):
            padded_origin_coords[i, :] = origin_coords_list[i]

        # Pad origin_types to (max_showers_per_event,)
        padded_origin_types = np.full((max_s,), -1, dtype=np.int64)
        for i in range(num_showers):
            padded_origin_types[i] = origin_types_list[i]

        # Valid showers mask
        valid_showers_mask = np.zeros(max_s, dtype=bool)
        valid_showers_mask[:num_showers] = True

        # Pad is_vertex
        padded_is_vertex = np.zeros(max_s, dtype=bool)
        for i in range(num_showers):
            padded_is_vertex[i] = is_vertex_list[i]

        # Pad particle_pids
        padded_particle_pids = np.full((max_s,), 0, dtype=np.int64)
        for i in range(num_showers):
            padded_particle_pids[i] = particle_pids_list[i]

        # Pad start_coords (auxiliary supervision target)
        padded_start_coords = np.full((max_s, 3), SHOWER_PAD_SENTINEL, dtype=np.float32)
        for i in range(num_showers):
            padded_start_coords[i, :] = start_coords_list[i]

        # Pad fragment_trackids
        padded_fragment_trackids = np.full((max_s,), -1, dtype=np.int64)
        for i in range(num_showers):
            padded_fragment_trackids[i] = fragment_trackids_list[i]

        # V2 compatibility: single-shower fallback
        if num_showers > 0:
            first_origin = origin_coords_list[0]
        else:
            first_origin = coord.mean(axis=0)

        origin_distance = np.linalg.norm(
            coord - first_origin.reshape(1, 3), axis=1
        ).astype(np.float32)

        # Build data dict
        data_dict = {
            "coord": coord,
            "strength": strength,
            "color": color,
            # Union shower mask (backward compatible)
            "shower_mask": union_mask.astype(np.float32),
            # Multi-shower data (padded for batching)
            "shower_masks": padded_shower_masks,
            "origin_coords": padded_origin_coords,
            "origin_types": padded_origin_types,
            "particle_pids": padded_particle_pids,
            "start_coords": padded_start_coords,
            "fragment_trackids": padded_fragment_trackids,
            "num_showers": np.int64(num_showers),
            "valid_showers_mask": valid_showers_mask,
            "origin_is_vertex": padded_is_vertex,
            # V2 single-shower compatibility
            "origin_coord": first_origin,
            "origin_distance": origin_distance,
            "name": os.path.basename(filepath),
        }

        return data_dict


@DATASETS.register_module()
class ShowerOriginDatasetSimple(DefaultDataset):
    """
    Simplified shower origin dataset that generates synthetic shower masks.

    This version works with standard LArTPC HDF5 files and generates
    shower masks from semantic segmentation labels (gamma class).

    Useful for initial development and testing before full shower
    fragment labeling is available.

    Args:
        gamma_class_id: Class ID for gamma/EM shower in segmentation (default: 4)
        min_shower_points: Minimum points to form a valid shower
        **kwargs: Arguments passed to parent class
    """

    VALID_ASSETS = [
        "coord",
        "strength",
        "color",
        "segment",
        "shower_mask",
        "origin_coord",
        "origin_distance",
    ]

    def __init__(
        self,
        split="train",
        data_root="data/lartpc",
        data_list_file=None,
        transform=None,
        coord_scale=1.0,
        use_reco_coords=True,
        log_transform_edep=True,
        wire_scale=1.0/3456.0,
        gamma_class_id=4,
        min_shower_points=10,
        **kwargs,
    ):
        self.coord_scale = coord_scale
        self.use_reco_coords = use_reco_coords
        self.log_transform_edep = log_transform_edep
        self.wire_scale = wire_scale
        self.gamma_class_id = gamma_class_id
        self.min_shower_points = min_shower_points
        self.data_list_file = data_list_file

        super().__init__(
            split=split,
            data_root=data_root,
            transform=transform,
            **kwargs,
        )

    def get_data_list(self):
        """Get list of HDF5 file paths."""
        data_list = []

        # Option 1: Explicit file list
        if self.data_list_file is not None:
            if os.path.isabs(self.data_list_file):
                list_file = self.data_list_file
            else:
                list_file = os.path.join(self.data_root, self.data_list_file)

            if os.path.exists(list_file):
                with open(list_file, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#'):
                            if os.path.isabs(line):
                                data_list.append(line)
                            else:
                                data_list.append(os.path.join(self.data_root, line))

        # Option 2: Look for split.txt file
        if not data_list:
            split_file = os.path.join(self.data_root, f"{self.split}.txt")
            if os.path.exists(split_file):
                with open(split_file, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#'):
                            if os.path.isabs(line):
                                data_list.append(line)
                            else:
                                data_list.append(os.path.join(self.data_root, line))

        # Option 3: Look for split directory
        if not data_list:
            split_dir = os.path.join(self.data_root, self.split)
            if os.path.isdir(split_dir):
                for f in sorted(os.listdir(split_dir)):
                    if f.endswith('.h5') or f.endswith('.hdf5'):
                        data_list.append(os.path.join(split_dir, f))

        return sorted(data_list)

    def get_data(self, idx):
        """
        Load sample and generate shower mask from gamma class.

        For this simplified version, we use all gamma-class points as
        a single "shower" and use the mean position as a placeholder origin.
        """
        idx = idx % len(self.data_list)
        filepath = self.data_list[idx]

        with h5py.File(filepath, 'r') as f:
            entry = f['entry_0']
            triplet = entry['triplet_data']

            # Load coordinates
            coord = triplet['pos'][:].astype(np.float32) * self.coord_scale

            # Load energy features
            if 'pixval' in triplet:
                pixval = triplet['pixval'][:].astype(np.float32)
                if self.log_transform_edep:
                    pixval = np.log1p(np.clip(pixval, 0, None))
                strength = pixval
            else:
                strength = np.ones((coord.shape[0], 1), dtype=np.float32)

            # Load wire features
            if all(k in triplet for k in ['uwire', 'vwire', 'ywire']):
                uwire = triplet['uwire'][:].astype(np.float32) * self.wire_scale
                vwire = triplet['vwire'][:].astype(np.float32) * self.wire_scale
                ywire = triplet['ywire'][:].astype(np.float32) * self.wire_scale
                color = np.stack([uwire, vwire, ywire], axis=-1)
            else:
                color = np.zeros((coord.shape[0], 3), dtype=np.float32)

            # Load segmentation labels
            if 'pid' in triplet:
                pid = triplet['pid'][:].astype(np.int64)
                # Map PDG to class (simplified)
                segment = np.zeros_like(pid)
                segment[pid == 22] = self.gamma_class_id  # gamma
            else:
                segment = np.zeros(coord.shape[0], dtype=np.int64)

            # Generate shower mask from gamma class
            shower_mask = (segment == self.gamma_class_id)

            # Use neutrino vertex if available, else shower centroid
            if 'nu_vtx' in entry:
                origin_coord = entry['nu_vtx'][:].astype(np.float32) * self.coord_scale
            elif shower_mask.sum() > 0:
                origin_coord = coord[shower_mask].mean(axis=0)
            else:
                origin_coord = coord.mean(axis=0)

            # Compute distances
            origin_distance = np.linalg.norm(
                coord - origin_coord.reshape(1, 3), axis=1
            ).astype(np.float32)

        data_dict = {
            "coord": coord,
            "strength": strength,
            "color": color,
            "segment": segment,
            "shower_mask": shower_mask.astype(np.float32),
            "origin_coord": origin_coord,
            "origin_distance": origin_distance,
            "name": os.path.basename(filepath),
        }

        return data_dict
