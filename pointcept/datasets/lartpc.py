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
        data_root (str): Root directory containing split subdirectories
        transform (list): List of transform configs
        use_reco_coords (bool): If True, use reconstructed coordinates (pos_*_reco),
            otherwise use true coordinates (pos_*)
        use_edep_as_strength (bool): If True, use energy deposition as strength feature
        label_mode (str): How to generate semantic labels:
            - 'pid': Use particle ID (PDG code) mapped to classes
            - 'origin': Use origin type (neutrino vs cosmic)
        coord_scale (float): Scale factor for coordinates (e.g., 0.01 to convert cm to m)
        log_transform_edep (bool): Apply log(1+x) transform to energy deposition
        **kwargs: Additional arguments passed to DefaultDataset
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
        "other",      # 5: everything else
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
        2112: 5,    # neutron -> other
        # Kaons
        321: 5,     # K+
        -321: 5,    # K-
        # Other common particles go to "other"
    }

    def __init__(
        self,
        split="train",
        data_root="data/lartpc",
        transform=None,
        use_reco_coords=True,
        use_edep_as_strength=True,
        label_mode="pid",
        coord_scale=1.0,
        log_transform_edep=True,
        test_mode=False,
        test_cfg=None,
        cache=False,
        ignore_index=-1,
        loop=1,
        **kwargs
    ):
        self.use_reco_coords = use_reco_coords
        self.use_edep_as_strength = use_edep_as_strength
        self.label_mode = label_mode
        self.coord_scale = coord_scale
        self.log_transform_edep = log_transform_edep

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
        Find all HDF5 files in the data directory.

        Supports both directory-based organization:
            data_root/train/*.h5
            data_root/val/*.h5

        And file-list based organization:
            data_root/train.txt (containing paths to .h5 files)
        """
        if isinstance(self.split, str):
            split_list = [self.split]
        else:
            split_list = list(self.split)

        data_list = []
        for split in split_list:
            # Check for split file (train.txt, val.txt, etc.)
            split_file = os.path.join(self.data_root, f"{split}.txt")
            if os.path.isfile(split_file):
                with open(split_file, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            # Handle both absolute and relative paths
                            if os.path.isabs(line):
                                data_list.append(line)
                            else:
                                data_list.append(os.path.join(self.data_root, line))
            else:
                # Look for HDF5 files in split directory
                split_dir = os.path.join(self.data_root, split)
                if os.path.isdir(split_dir):
                    # Support multiple extensions
                    for ext in ['*.h5', '*.hdf5', '*.hdf']:
                        data_list.extend(glob.glob(os.path.join(split_dir, ext)))

        return sorted(data_list)

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
            #entrydata = f['/entry_0']
            #print(entrydata.keys())
            #tripletdata = entrydata['triplet_data']
            #print(tripletdata.keys())
            
            # Load coordinates
            coord = np.array(f['/entry_0/triplet_data/pos'], dtype=np.float32)

            # Apply coordinate scaling
            if self.coord_scale != 1.0:
                coord = coord * self.coord_scale

            # Load and map semantic labels
            if self.label_mode == 'pid':
                pid = f['/entry_0/triplet_data/pid'][:]
                segment = self._map_pid_to_class(pid)
            elif self.label_mode == 'origin':
                origin = f['/entry_0/triplet_data/origin'][:]
                segment = self._map_origin_to_class(origin)
            else:
                raise ValueError(f"Unknown label_mode: {self.label_mode}")

            # Load energy deposition as strength
            if self.use_edep_as_strength:
                edep = np.array(f['/entry_0/triplet_data/pixval'], dtype=np.float32)
                if self.log_transform_edep:
                    # Log transform for better scaling (edep can vary by orders of magnitude)
                    #strength = np.log1p(edep).reshape(-1, 1).astype(np.float32)
                    strength = edep/500.0
                else:
                    strength = edep.reshape(-1, 1).astype(np.float32)
            else:
                strength = np.ones((coord.shape[0], 1), dtype=np.float32)

            # Load wire coordinates as "color" feature (3 channels like RGB)
            wire_feat = np.stack([
                f['/entry_0/triplet_data/uwire'][:],
                f['/entry_0/triplet_data/vwire'][:],
                f['/entry_0/triplet_data/ywire'][:]
            ], axis=1).astype(np.float32)
            wire_feat /= 1000.0

            # Load instance labels (track IDs)
            trackid = f['/entry_0/triplet_data/trackid'][:]
            instance = trackid.astype(np.int32)

        data_dict = {
            "coord": coord,
            "strength": strength,
            "color": wire_feat,
            "segment": segment,
            "instance": instance,
            "name": name,
            "split": split,
        }

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
