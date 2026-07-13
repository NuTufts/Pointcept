"""
LArTPC (Liquid Argon Time Projection Chamber) Dataset

Dataset for 3D point cloud semantic segmentation of particle physics data.
Reads HDF5 files produced by SimChTripletLabelMaker from the ubdl/larflow package.

Author: Generated for LArTPC physics analysis
"""

import os
import glob
import h5py
import numpy as np

from .builder import DATASETS
from .defaults import DefaultDataset


@DATASETS.register_module()
class LArTPCDataset(DefaultDataset):
    """
    Dataset for Liquid Argon TPC point cloud data from HDF5 files.

    Reads triplet data exported by SimChTripletLabelMaker and formats it
    for Pointcept's training pipeline.

    Args:
        split (str): Dataset split - 'train', 'val', or 'test'
        data_root (str): Root directory containing split subdirectories (used as base
            for relative paths in file lists)
        data_list_file (str, optional): Explicit path to a text file containing paths
            to HDF5 files (one per line). If provided, this takes priority over
            directory scanning. Paths in the file can be absolute or relative to
            data_root. Supports comments (lines starting with #) and blank lines.
        transform (list): List of transform configs
        use_reco_coords (bool): If True, use reconstructed coordinates (pos_*_reco),
            otherwise use true coordinates (pos_*)
        use_edep_as_strength (bool): If True, use energy deposition as strength feature
        label_mode (str): How to generate semantic labels:
            - 'pid': Use particle ID (PDG code) mapped to classes
            - 'origin': Use origin type (neutrino vs cosmic)
        coord_scale (float): Scale factor for coordinates (e.g., 0.01 to convert cm to m)
        log_transform_edep (bool): Apply log(1+x) transform to energy deposition
        drop_cosmics (bool): default False. If True, drop cosmic and ghost spacepoints with probability, drop_cosmics_prob.
        drop_cosmics_prob (float): default 0.5. If drop_cosmics is true, drop cosmics with this probability.
        filter_larmatch (bool): default False. If True and 'larmatch_score' exists in the
            HDF5 file, filter points by a randomly drawn larmatch score threshold.
        larmatch_threshold_range (tuple): default (0.15, 0.75). The (low, high) range from
            which the larmatch score threshold is drawn uniformly each time a sample is loaded.
            Points with larmatch_score > threshold are kept.
        **kwargs: Additional arguments passed to DefaultDataset

    Data Loading Priority:
        1. If `data_list_file` is provided and exists, use it
        2. Else if `data_root/{split}.txt` exists, use it
        3. Else scan `data_root/{split}/` directory for .h5 files

    Example config with explicit file lists:
        data = dict(
            train=dict(
                type="LArTPCDataset",
                split="train",
                data_root="data/lartpc",
                data_list_file="/path/to/my_train_files.txt",
                ...
            ),
            val=dict(
                type="LArTPCDataset",
                split="val",
                data_root="data/lartpc",
                data_list_file="/path/to/my_val_files.txt",
                ...
            ),
        )
    """

    # Assets that can be loaded (for compatibility with DefaultDataset)
    VALID_ASSETS = ["coord", "strength", "segment", "instance", "color"]

    # Default particle class names
    CLASS_NAMES_PID = [
        "electron",   # 0: e+/e-
        "muon",       # 1: mu+/mu-
        "pion",       # 2: pi+/pi-
        "proton",     # 3: proton
        "gamma",      # 4: photon
        "michel",     # 5: michel
        "delta",      # 6: delta
        "led"         # 7: low energy deposit
        "ghost",      # 8: ghost
        "other",      # 9: everything else
    ]

    # Default origin class names
    CLASS_NAMES_ORIGIN = [
        "unknown",    # 0: unknown origin
        "neutrino",   # 1: neutrino interaction
        "cosmic",     # 2: cosmic ray
    ]

    # PDG code to class index mapping
    # See: https://pdg.lbl.gov/2023/mcdata/mc_particle_id_contents.html
    PID_TO_CLASS = {
        # Electrons
        11: 0,      # electron
        -11: 0,     # positron
        # Muons
        13: 1,      # muon-
        -13: 1,     # muon+
        # Pions
        211: 2,     # pi+
        -211: 2,    # pi-
        111: 4,     # pi0 -> decays to gammas, treat as gamma
        # Proton
        2212: 3,    # proton
        # Photon
        22: 4,      # gamma
        # Neutron (often invisible, but include)
        2112: 3,    # neutron -> proton
        # Kaons
        321: 6,     # K+
        -321: 6,    # K-
        # Other common particles go to "other"
        # ghost points
        0:8
    }

    # SSNET LABEL FROM larflow::prep::SimChTripletLabelMaker
    # 0:'bg',
    # 1:'electron',
    # 2:'photon',
    # 3:'muon',
    # 4:'proton',
    # 5:'pion',
    # 6:'michel',
    # 7:'delta',
    # 8:'led',
    # 9:'other'
    SSNETLABEL_TO_CLASS = {
        0:8, # ghost
        1:0, # electron
        2:4, # photon
        3:1, # muon
        4:3, # proton
        5:2, # pion
        6:5, # michel
        7:6, # delta
        8:7, # LED
        9:9, # OTHER
    }

    def __init__(
        self,
        split="train",
        data_root="data/lartpc",
        data_list_file=None,
        transform=None,
        use_reco_coords=True,
        use_edep_as_strength=True,
        label_mode="pid",
        coord_scale=1.0,
        wire_scale=1.0/3456.0,
        adc_scale=1.0,
        test_mode=False,
        test_cfg=None,
        cache=False,
        ignore_index=-1,
        loop=1,
        include_ghosts=False,
        exclude_other=True,
        true_points_only=False,
        ghost_keep_frac=None,
        drop_cosmics=False,
        drop_cosmics_prob=0.5,
        add_min_pixval=1.0e-2,
        data_only=False,
        filter_larmatch=False,
        larmatch_threshold_range=(0.15, 0.75),
        **kwargs
    ):
        self.use_reco_coords = use_reco_coords
        self.use_edep_as_strength = use_edep_as_strength
        self.label_mode = label_mode
        self.coord_scale = coord_scale
        self.wire_scale = wire_scale
        self.adc_scale = adc_scale
        self.add_min_pixval = add_min_pixval
        self.include_ghosts = include_ghosts
        self.exclude_other = exclude_other
        self.data_list_file = data_list_file
        self.true_points_only = true_points_only
        # ghost_keep_frac: None disables; float in [0,1] keeps all real points
        # (hasmatch==1) and each ghost point with this probability. Controlled
        # contamination for the ghost dose-response studies (P0.4 / P2A).
        # ghost_keep_frac=0.0 is equivalent to true_points_only=True;
        # ghost_keep_frac=1.0 keeps everything.
        if ghost_keep_frac is not None:
            assert 0.0 <= ghost_keep_frac <= 1.0, \
                f"ghost_keep_frac must be in [0,1], got {ghost_keep_frac}"
            assert not true_points_only, \
                "ghost_keep_frac and true_points_only are mutually exclusive"
        self.ghost_keep_frac = ghost_keep_frac
        self.drop_cosmics = drop_cosmics
        self.drop_cosmics_prob = drop_cosmics_prob
        self.data_only = data_only
        self.filter_larmatch = filter_larmatch
        self.larmatch_threshold_range = larmatch_threshold_range
        self.min_points_required = 100

        # Call parent init (this will call get_data_list)
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

    def get_data_list(self):
        """
        Find all HDF5 files for this dataset split.

        Loading priority:
            1. If `data_list_file` is provided and exists, use it
            2. Else if `data_root/{split}.txt` exists, use it
            3. Else scan `data_root/{split}/` directory for .h5 files

        File list format:
            - One file path per line
            - Paths can be absolute or relative to data_root
            - Lines starting with # are treated as comments
            - Blank lines are ignored
        """
        data_list = []

        # Priority 1: Explicit data_list_file parameter
        if self.data_list_file is not None:
            if os.path.isfile(self.data_list_file):
                data_list = self._load_file_list(self.data_list_file)
                if data_list:
                    return sorted(data_list)
            else:
                raise FileNotFoundError(
                    f"data_list_file not found: {self.data_list_file}"
                )

        # Handle multiple splits (for combined datasets)
        if isinstance(self.split, str):
            split_list = [self.split]
        else:
            split_list = list(self.split)

        for split in split_list:
            # Priority 2: Check for split file (train.txt, val.txt, etc.) in data_root
            split_file = os.path.join(self.data_root, f"{split}.txt")
            if os.path.isfile(split_file):
                data_list.extend(self._load_file_list(split_file))
            else:
                # Priority 3: Look for HDF5 files in split directory (including subdirectories)
                split_dir = os.path.join(self.data_root, split)
                if os.path.isdir(split_dir):
                    # Support multiple extensions with recursive glob (**)
                    for ext in ['*.h5', '*.hdf5', '*.hdf']:
                        # Search in top-level directory
                        data_list.extend(glob.glob(os.path.join(split_dir, ext)))
                        # Search recursively in subdirectories
                        data_list.extend(glob.glob(os.path.join(split_dir, '**', ext), recursive=True))

        return sorted(data_list)

    def _load_file_list(self, file_path):
        """
        Load a list of file paths from a text file.

        Args:
            file_path: Path to the text file containing file paths

        Returns:
            List of resolved file paths
        """
        data_list = []
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                # Skip empty lines and comments
                if not line or line.startswith('#'):
                    continue
                # Handle both absolute and relative paths
                if os.path.isabs(line):
                    data_list.append(line)
                else:
                    data_list.append(os.path.join(self.data_root, line))
        return data_list

    def get_data_name(self, idx):
        """Return the sample name (filename without extension)."""
        path = self.data_list[idx % len(self.data_list)]
        return os.path.splitext(os.path.basename(path))[0]

    def get_split_name(self, idx):
        """Return the split name for this sample."""
        path = self.data_list[idx % len(self.data_list)]
        # Try to infer from parent directory
        parent = os.path.basename(os.path.dirname(path))
        if parent in ['train', 'val', 'test']:
            return parent
        # Fall back to configured split
        if isinstance(self.split, str):
            return self.split
        return self.split[0]

    def _has_truth(self, f):
        """Check whether the HDF5 file contains truth-level fields.

        Returns:
            bool: True if truth fields should be loaded.
        """
        if self.data_only is True:
            return False
        if self.data_only is False:
            return True
        # data_only="auto": probe the file
        return 'origin' in f['/entry_0/triplet_data']

    def get_data(self, idx):
        """
        Load data from HDF5 file and format for Pointcept.

        Returns:
            dict with keys: coord, segment, strength, color, instance, name, split
        """
        data_path = self.data_list[idx % len(self.data_list)]
        name = self.get_data_name(idx)
        split = self.get_split_name(idx)

        with h5py.File(data_path, 'r') as f:
            # Load coordinates
            coord = np.array(f['/entry_0/triplet_data/pos'], dtype=np.float32)

            # Apply coordinate scaling
            if self.coord_scale != 1.0:
                coord = coord * self.coord_scale

            N = coord.shape[0]
            has_truth = self._has_truth(f)

            if has_truth:
                # Load origin tag
                # origin flags: 0=no ground truth, 1=neutrino, 2=cosmic
                origin = f['/entry_0/triplet_data/origin'][:]

                # Load and map semantic labels
                if self.label_mode == 'pid':
                    pid = f['/entry_0/triplet_data/pid'][:]
                    segment = self._map_pid_to_class(pid)
                elif self.label_mode == 'ssnet':
                    ssnetlabels = f['/entry_0/triplet_data/ssnet_label'][:]
                    segment = self._map_ssnetlabel_to_class(ssnetlabels)
                elif self.label_mode == 'origin':
                    segment = self._map_origin_to_class(origin)
                else:
                    raise ValueError(f"Unknown label_mode: {self.label_mode}")

                # Count points per class for the active classes only
                if self.label_mode == 'ssnet':
                    active_classes = list(range(8))
                    ghost_class = 8
                    other_class = 9
                else:
                    active_classes = list(range(5))
                    ghost_class = 5
                    other_class = 6
                if self.include_ghosts:
                    active_classes.append(ghost_class)
                if not self.exclude_other:
                    active_classes.append(other_class)

                nclasses = len(active_classes)
                classcounts = np.zeros((1, nclasses), dtype=np.int64)
                for i, class_idx in enumerate(active_classes):
                    classcounts[0, i] = (segment == class_idx).sum()

                # Load instance labels (track IDs)
                trackid = f['/entry_0/triplet_data/trackid'][:]
                instance = trackid.astype(np.int32)

                # Load true-ghost labels. 0=ghost, 1=true
                hasmatch = np.array(f['/entry_0/triplet_data/hasmatch'], dtype=np.int64)

                # Get keypoints and pass neutrino keypoint for biased sampling
                keypoint_pos  = np.array(f['/entry_0/mckeypoints/pos'], dtype=np.float32)
                keypoint_type = np.array(f['/entry_0/mckeypoints/kptype'], dtype=np.int64)
                nu_keypoint_mask = keypoint_type == 0
                nu_vertices = keypoint_pos[nu_keypoint_mask[:], :]
            else:
                # Data-only path: no truth fields available
                origin = np.zeros(N, dtype=np.int32)
                segment = np.full(N, fill_value=-1, dtype=np.int32)
                instance = np.full(N, fill_value=-1, dtype=np.int32)
                hasmatch = np.ones(N, dtype=np.int64)
                nu_vertices = np.zeros((0, 3), dtype=np.float32)
                classcounts = np.zeros((1, 0), dtype=np.int64)

            # Load energy deposition as strength
            if self.use_edep_as_strength:
                edep = np.array(f['/entry_0/triplet_data/pixval'], dtype=np.float32)
                # Scale pixval (N, 3) - pixel intensities from 3 wire planes
                # Each column is the signal from u, v, y wire plane images
                strength = (np.clip(edep, a_min=0, a_max=1000.0) / self.adc_scale).astype(np.float32)
            else:
                strength = np.ones((N, 1), dtype=np.float32)
            # Add a small value, so that LogTransform (transform.py) does not break
            strength += self.add_min_pixval

            # Load wire coordinates as "color" feature (3 channels like RGB)
            wire_feat = np.stack([
                f['/entry_0/triplet_data/uwire'][:],
                f['/entry_0/triplet_data/vwire'][:],
                f['/entry_0/triplet_data/ywire'][:]
            ], axis=1).astype(np.float32)
            wire_feat *= self.wire_scale

            # Load larmatch score if available and filtering is enabled
            larmatch_score = None
            if self.filter_larmatch and 'larmatch_score' in f['/entry_0/triplet_data']:
                larmatch_score = np.array(
                    f['/entry_0/triplet_data/larmatch_score'], dtype=np.float32
                )

        data_dict = {
            "coord": coord,
            "strength": strength,
            "color": wire_feat,
            "segment": segment,
            "instance": instance,
            "name": name,
            "split": split,
            "segment_counts": classcounts,
            "nu_vertices": nu_vertices,
            "origin": origin,
            # ``hasmatch`` is the producer's per-spacepoint deghost truth
            # (1 = real with simch edep, 0 = ghost). The HasmatchAsGhost
            # transform reads this field to write ``segment``. In data-only
            # mode it's filled with ones above so the transform behaves like
            # "everything is real".
            "hasmatch": hasmatch,
            # Register the LArTPC-specific per-point fields so transforms
            # that filter (BiasedSphereCrop, SphereCrop, GridSample) keep
            # them aligned with coord. The default index_valid_keys in
            # transform.py doesn't know about ``hasmatch`` or ``origin``.
            "index_valid_keys": [
                "coord", "color", "normal", "superpoint",
                "strength", "segment", "instance",
                "hasmatch", "origin",
            ],
        }

        filtered = False
        # Masks applied so far, in order, each in the coordinates of the array
        # it was applied to — used to bring the larmatch mask (defined on the
        # original points) into the current filtered frame.
        applied_masks = []
        if has_truth and self.true_points_only:
            # mask out ghost points
            ghost_mask = hasmatch == 1
            for k in data_dict:
                if k in ['coord', 'strength', 'color', 'segment', 'instance', 'origin', 'hasmatch']:
                    data_dict[k] = data_dict[k][ghost_mask[:]]
            applied_masks.append(ghost_mask)
            filtered = True
        elif has_truth and self.ghost_keep_frac is not None:
            # keep all real points and a random ghost_keep_frac of the ghosts
            keep_mask = (hasmatch == 1) | (
                np.random.random(hasmatch.shape[0]) < self.ghost_keep_frac
            )
            for k in data_dict:
                if k in ['coord', 'strength', 'color', 'segment', 'instance', 'origin', 'hasmatch']:
                    data_dict[k] = data_dict[k][keep_mask[:]]
            applied_masks.append(keep_mask)
            filtered = True

        if has_truth and self.drop_cosmics and np.random.random() < self.drop_cosmics_prob:
            nu_mask = data_dict['origin'] == 1
            if nu_mask.sum() > self.min_points_required:
                for k in data_dict:
                    if k in ['coord', 'strength', 'color', 'segment', 'instance', 'origin', 'hasmatch']:
                        data_dict[k] = data_dict[k][nu_mask[:]]
                applied_masks.append(nu_mask)
                filtered = True

        if larmatch_score is not None:
            # Draw a random threshold uniformly from the configured range
            lo, hi = self.larmatch_threshold_range
            threshold = np.random.uniform(lo, hi)
            lm_mask = larmatch_score > threshold
            # Bring the mask (defined on the original points) into the current
            # filtered frame by replaying the masks that were actually applied.
            if filtered and len(lm_mask) != data_dict['coord'].shape[0]:
                for prior_mask in applied_masks:
                    lm_mask = lm_mask[prior_mask[:]]
            if lm_mask.sum() >= self.min_points_required:
                for k in data_dict:
                    if k in ['coord', 'strength', 'color', 'segment', 'instance', 'origin', 'hasmatch']:
                        data_dict[k] = data_dict[k][lm_mask[:]]
                filtered = True

        if filtered:
            # Check if we have enough points after filtering
            if data_dict['coord'].shape[0] < self.min_points_required:
                # Track retry attempts to prevent infinite recursion
                if not hasattr(self, '_retry_count'):
                    self._retry_count = 0
                self._retry_count += 1

                max_retries = 10
                if self._retry_count > max_retries:
                    self._retry_count = 0
                    raise RuntimeError(
                        f"Failed to find valid sample after {max_retries} retries. "
                        f"Too many events have < {self.min_points_required} true points. "
                        f"Consider checking your data or lowering true_points_only requirements."
                    )

                # Log warning and retry with a different sample
                print(
                    f"WARNING: Event '{name}' has only {data_dict['coord'].shape[0]} true points "
                    f"(< {self.min_points_required}). Sampling a different event (retry {self._retry_count}/{max_retries})."
                )
                # Get a different random sample
                new_idx = np.random.randint(0, len(self.data_list))
                result = self.get_data(new_idx)
                self._retry_count = 0  # Reset on success
                return result

        return data_dict

    def _map_pid_to_class(self, pid):
        """
        Map PDG particle IDs to class indices.

        Args:
            pid: Array of PDG codes

        Returns:
            Array of class indices (int32), with -1 for unknown/ignored particles
        """
        # Default to -1 (ignore_index) for unknown particles
        # PID=0 means no particle ID assigned - should be ignored during training
        segment = np.full(len(pid), fill_value=-1, dtype=np.int32)

        for pdg_code, class_idx in self.PID_TO_CLASS.items():
            segment[pid == pdg_code] = class_idx

        if not self.include_ghosts:
            segment[ segment==5 ] = -1
        if self.exclude_other:
            segment[ segment==6 ] = -1

        return segment

    def _map_ssnetlabel_to_class(self, ssnetlabel):
        """
        Map PDG particle IDs to class indices.

        Args:
            pid: Array of PDG codes

        Returns:
            Array of class indices (int32), with -1 for unknown/ignored particles
        """
        # Default to -1 (ignore_index) for unknown particles
        # PID=0 means no particle ID assigned - should be ignored during training
        segment = np.full(len(ssnetlabel), fill_value=-1, dtype=np.int32)

        for ssnet_code, class_idx in self.SSNETLABEL_TO_CLASS.items():
            segment[ssnetlabel == ssnet_code] = class_idx

        if not self.include_ghosts:
            segment[ segment==8 ] = -1
        if self.exclude_other:
            segment[ segment==9 ] = -1

        return segment

    def _map_origin_to_class(self, origin):
        """
        Map origin type to class indices.

        Origin values (from LArSoft/ubdl):
            0: unknown
            1: neutrino
            2: cosmic

        Args:
            origin: Array of origin types

        Returns:
            Array of class indices (int32)
        """
        # Origin is already suitable as class index
        return origin.astype(np.int32)

    @property
    def class_names(self):
        """Return class names based on label mode."""
        if self.label_mode == 'pid':
            return self.CLASS_NAMES_PID
        elif self.label_mode == 'origin':
            return self.CLASS_NAMES_ORIGIN
        return None

    @property
    def num_classes(self):
        """Return number of classes based on label mode."""
        if self.label_mode == 'pid':
            return len(self.CLASS_NAMES_PID)
        elif self.label_mode == 'origin':
            return len(self.CLASS_NAMES_ORIGIN)
        return None


@DATASETS.register_module()
class LArTPCInstanceDataset(LArTPCDataset):
    """
    LArTPC dataset configured for instance segmentation.

    Includes both semantic labels (segment) and instance labels (instance)
    for training instance segmentation models.
    """

    VALID_ASSETS = ["coord", "strength", "segment", "instance", "color"]

    def get_data(self, idx):
        """Load data with instance labels emphasized."""
        data_dict = super().get_data(idx)

        # Ensure instance labels are properly formatted
        # Remap instance IDs to be contiguous starting from 0
        instance = data_dict["instance"]
        unique_ids = np.unique(instance)
        id_map = {old_id: new_id for new_id, old_id in enumerate(unique_ids)}
        data_dict["instance"] = np.array(
            [id_map[i] for i in instance], dtype=np.int32
        )

        return data_dict
