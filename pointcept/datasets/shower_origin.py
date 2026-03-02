"""
Shower Origin Dataset

Dataset class for shower origin prediction task.
Supports multi-shower mode with per-fragment masks and origin labels.
Handles photon showers (pid=22) and electron/positron showers (pid=11/-11).

HDF5 format (from C++ ShowerFragmentOriginMaker):
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
   - shower_fragments/nu_vertex_is_visible: int (scalar) — 1 if a neutrino
     vertex keypoint (kNuVertex) exists, 0 otherwise.

   For type-0 (nu-inside) fragments, the origin is overridden to the trunk
   startpt when BOTH: (a) nu_vertex_is_visible==0, and (b) there is only
   one visible neutrino-origin shower. In this case the vertex cannot be
   inferred from the available information.
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
        min_fragment_points: Minimum points for a DBSCAN fragment to be included (default: 20)
        include_cosmic_showers: Whether to include cosmic-origin showers
        include_ghosts: Whether to include ghost points (default: False)
        balance_origins: Whether to balance neutrino/cosmic samples
        **kwargs: Additional arguments passed to DefaultDataset
    """

    VALID_ASSETS = [
        "coord",
        "strength",
        "color",
        "segment",
        "shower_mask",
        "fragment_centroid",
        "origin_coord",
        "origin_type",
        "origin_distance",
        "start_coord",
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
        include_ghosts=False,
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
        self.include_ghosts = include_ghosts
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
        """
        self.event_index = []

        for file_idx, filepath in enumerate(self.data_list):
            try:
                with h5py.File(filepath, 'r') as f:
                    if 'entry_0/shower_fragments' not in f:
                        continue
                    sf = f['entry_0/shower_fragments']
                    if 'pointindices_flat' not in sf:
                        continue
                    num_frags = int(sf.attrs.get('num_fragments', 0))
                    if num_frags > 0:
                        self.event_index.append(file_idx)
            except Exception as e:
                print(f"Warning: Could not process {filepath}: {e}")
                continue

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

            n_points_orig = coord.shape[0]

            # Ghost point filtering using hasmatch (0=ghost, 1=true)
            ghost_mask = None
            if not self.include_ghosts and 'hasmatch' in triplet:
                hasmatch = triplet['hasmatch'][:].astype(np.int64)
                ghost_mask = (hasmatch == 1)
                coord = coord[ghost_mask]

            n_points = coord.shape[0]

            # Load energy features
            if 'pixval' in triplet:
                pixval = triplet['pixval'][:].astype(np.float32)
                if ghost_mask is not None:
                    pixval = pixval[ghost_mask]
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
                if ghost_mask is not None:
                    uwire = uwire[ghost_mask]
                    vwire = vwire[ghost_mask]
                    ywire = ywire[ghost_mask]
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

            n_inside = 0

            if 'shower_fragments' in entry:
                sf = entry['shower_fragments']

                # Build index remap for ghost filtering: original index -> filtered index
                # Fragment point indices reference the original (unfiltered) array,
                # so we need to map them to the filtered (ghost-removed) array.
                if ghost_mask is not None:
                    index_remap = np.full(n_points_orig, -1, dtype=np.int64)
                    index_remap[ghost_mask] = np.arange(n_points)
                else:
                    index_remap = None

                num_frags = int(sf.attrs.get('num_fragments', 0))
                #print("num fragments: ",num_frags)
                if num_frags > 0:
                    trackids = sf['trackid'][:]
                    pids = sf['pid'][:]
                    istrunk = sf['istrunk'][:]
                    frag_types = sf['type'][:]
                    startpts = sf['startpt'][:]
                    originpts = sf['originpt'][:]
                    flat_indices = sf['pointindices_flat'][:]
                    index_counts = sf['pointindices_counts'][:]

                    # Load nu_vertex_is_visible flag (single scalar).
                    # 1 = a neutrino vertex keypoint (kNuVertex) exists,
                    # meaning the vertex is visible from other particles.
                    if 'nu_vertex_is_visible' in sf:
                        nu_vtx_visible = int(sf['nu_vertex_is_visible'][()])
                    else:
                        nu_vtx_visible = 1  # default: assume visible for old data

                    # First pass: count unique visible nu-origin (type-0) trackids
                    visible_nu_trackids = set()
                    offset = 0
                    for i in range(num_frags):
                        count = int(index_counts[i])
                        if count >= self.min_fragment_points and int(frag_types[i]) == 0:
                            visible_nu_trackids.add(int(trackids[i]))
                        offset += count
                    n_visible_nu_showers = len(visible_nu_trackids)

                    # Determine if type-0 origins should be moved to startpt:
                    # When the nu vertex is NOT visible AND there is only
                    # one visible nu-origin shower, the vertex cannot be
                    # inferred — override origin to trunk startpt.
                    override_nu_origin = (nu_vtx_visible == 0
                                          and n_visible_nu_showers <= 1)

                    # Find trunk startpt per trackid for origin override
                    trunk_startpt = {}
                    if override_nu_origin:
                        for i in range(num_frags):
                            tid = int(trackids[i])
                            if tid in visible_nu_trackids and int(istrunk[i]) == 1:
                                trunk_startpt[tid] = startpts[i].astype(np.float32)

                    offset = 0
                    for i in range(num_frags):
                        #if len(shower_masks_list) >= self.max_showers_per_event:
                        #    break

                        count = int(index_counts[i])
                        indices = flat_indices[offset:offset+count]
                        offset += count

                        # Build boolean mask, remapping indices if ghosts were removed
                        fmask = np.zeros(n_points, dtype=bool)
                        if index_remap is not None:
                            valid_idx = indices[indices < n_points_orig]
                            remapped = index_remap[valid_idx]
                            remapped = remapped[remapped >= 0]  # drop ghost points
                            fmask[remapped] = True
                        else:
                            valid_idx = indices[indices < n_points]
                            fmask[valid_idx] = True

                        if fmask.sum() < self.min_fragment_points:
                            continue

                        # Origin type determined by C++ using pret0shiftedoriginpt:
                        # 0=nu-inside, 1=outside, 2=cosmic-inside
                        frag_origin_type = int(frag_types[i])

                        if not self.include_cosmic_showers and frag_origin_type == 2:
                            continue

                        if frag_origin_type==0:
                            n_inside += 1

                        frag_origin = originpts[i].astype(np.float32) * self.coord_scale
                        frag_start = startpts[i].astype(np.float32) * self.coord_scale
                        frag_tid = int(trackids[i])

                        # Override origin for type-0 fragments when the nu
                        # vertex is not visible and there's only one visible
                        # nu-origin shower (vertex cannot be inferred).
                        if override_nu_origin and frag_origin_type == 0:
                            if frag_tid in trunk_startpt:
                                frag_origin = trunk_startpt[frag_tid] * self.coord_scale
                            else:
                                frag_origin = frag_start

                        frag_is_vertex = (frag_origin_type == 0
                                          and not override_nu_origin)

                        shower_masks_list.append(fmask)
                        origin_coords_list.append(frag_origin)
                        start_coords_list.append(frag_start)
                        origin_types_list.append(frag_origin_type)
                        is_vertex_list.append(frag_is_vertex)
                        particle_pids_list.append(int(pids[i]))
                        fragment_trackids_list.append(frag_tid)

        num_showers = len(shower_masks_list)
        #print("num inside fragments:  ",n_inside)

        # Randomly select 1 fragment for this training example
        if num_showers > 0:
            idx = np.random.randint(num_showers)
            #print("selected fragment idx=",idx," of ",num_showers)
            selected_mask = shower_masks_list[idx]
            selected_origin = origin_coords_list[idx]
            selected_type = origin_types_list[idx]
            selected_start = start_coords_list[idx]
        else:
            # No valid fragments: use a dummy zero mask and centroid fallback
            selected_mask = np.zeros(n_points, dtype=bool)
            selected_origin = coord.mean(axis=0)
            selected_type = -1
            selected_start = coord.mean(axis=0)

        # Compute fragment centroid for BiasedSphereCrop anchor
        if selected_mask.any():
            fragment_centroid = coord[selected_mask].mean(axis=0).reshape(1, 3)
        else:
            fragment_centroid = coord.mean(axis=0).reshape(1, 3)

        # Distance from every point to the selected origin
        origin_distance = np.linalg.norm(
            coord - selected_origin.reshape(1, 3), axis=1
        ).astype(np.float32)

        # Validate data for NaN/Inf before returning
        _arrays_to_check = {
            "coord": coord, "strength": strength,
            "selected_origin": selected_origin,
            "selected_start": selected_start,
            "origin_distance": origin_distance,
            "fragment_centroid": fragment_centroid,
        }
        for _name, _arr in _arrays_to_check.items():
            if not np.all(np.isfinite(_arr)):
                import logging
                logger = logging.getLogger(__name__)
                logger.error(
                    f"NaN/Inf in '{_name}' for file={os.path.basename(filepath)}, "
                    f"fragment_idx={idx if num_showers > 0 else 'N/A'}/{num_showers}, "
                    f"origin_type={selected_type}, "
                    f"n_points={len(coord)}, "
                    f"nan_count={np.isnan(_arr).sum()}, inf_count={np.isinf(_arr).sum()}"
                )

        sample_id = (
            f"{os.path.basename(filepath)}:frag{idx if num_showers > 0 else 'none'}"
            f"/{num_showers}:type{selected_type}"
        )

        data_dict = {
            "coord": coord,
            "strength": strength,
            "color": color,
            "shower_mask": selected_mask.astype(np.float32),
            "fragment_centroid": fragment_centroid.astype(np.float32),
            "origin_coord": selected_origin.astype(np.float32),
            "origin_type": np.int64(selected_type),
            "start_coord": selected_start.astype(np.float32),
            "origin_distance": origin_distance,
            "name": sample_id,
            # Register per-point keys for index_operator (used by
            # BiasedSphereCrop, GridSample, SphereCrop, etc.)
            "index_valid_keys": [
                "coord", "color", "normal", "superpoint", "strength",
                "segment", "instance",
                # shower-specific per-point keys
                "shower_mask", "origin_distance",
            ],
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
