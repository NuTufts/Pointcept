# Creating a Custom Pointcept Dataset for LArTPC Data

This guide documents how to integrate Liquid Argon TPC (LArTPC) point cloud data with the Pointcept repository for training PointTransformerV3 and other models.

## Overview

LArTPC detectors produce 3D point cloud data from particle interactions. This guide shows how to create a custom dataset that reads HDF5 files produced by `SimChTripletLabelMaker` and formats them for Pointcept's training pipeline.

## Source Data Format (HDF5)

The `SimChTripletLabelMaker` class (from `ubdl/larflow/larflow/PrepFlowMatchData/`) exports triplet data to HDF5 with the following structure:

```
/triplet_data/
├── pos_x          (N,) float - True X position
├── pos_y          (N,) float - True Y position
├── pos_z          (N,) float - True Z position
├── pos_x_reco     (N,) float - Reconstructed X position
├── pos_y_reco     (N,) float - Reconstructed Y position
├── pos_z_reco     (N,) float - Reconstructed Z position
├── edep           (N,) float - Energy deposition
├── trackid        (N,) long  - Track ID
├── pid            (N,) int   - Particle ID (PDG code)
├── aid            (N,) int   - Ancestor ID
├── origin         (N,) int   - Origin type (neutrino, cosmic, etc.)
├── uwire          (N,) int   - U-plane wire coordinate
├── vwire          (N,) int   - V-plane wire coordinate
├── ywire          (N,) int   - Y-plane wire coordinate
├── tick           (N,) int   - Time tick
└── row            (N,) int   - Image row
```

## Pointcept Expected Format

Pointcept's `DefaultDataset` expects a dictionary with these keys:

| Key | Required | Shape | Type | Description |
|-----|----------|-------|------|-------------|
| `coord` | Yes | (N, 3) | float32 | Point coordinates |
| `segment` | Yes | (N,) | int32 | Semantic labels (class IDs), -1 for ignore |
| `color` | Optional | (N, 3) | float32 | RGB colors or features |
| `normal` | Optional | (N, 3) | float32 | Surface normals |
| `strength` | Optional | (N, 1) | float32 | Point intensity |
| `instance` | Optional | (N,) | int32 | Instance IDs |
| `name` | Yes | str | - | Sample identifier |
| `split` | Yes | str | - | Dataset split (train/val/test) |

## Implementation: HDF5-Native Dataset

### Dataset Class

The `LArTPCDataset` class directly reads HDF5 files without requiring format conversion:

```python
# Located at: pointcept/datasets/lartpc.py

@DATASETS.register_module()
class LArTPCDataset(DefaultDataset):
    """Dataset for Liquid Argon TPC point cloud data from HDF5 files."""

    def get_data(self, idx):
        # Reads HDF5 directly and returns Pointcept-compatible dict
        ...
```

### Key Design Decisions

1. **Coordinates**: Use `pos_*_reco` (reconstructed) by default, with option to use true positions
2. **Semantic Labels**: Map PDG particle IDs to class indices
3. **Features**:
   - `strength`: Energy deposition (log-transformed)
   - `color`: Wire coordinates (u, v, y) as 3-channel feature
4. **Instance Labels**: Track IDs for instance segmentation

### Particle Class Mapping

Default mapping from PDG codes to class indices:

| Class | PDG Codes | Particle |
|-------|-----------|----------|
| 0 | 11, -11 | electron/positron |
| 1 | 13, -13 | muon |
| 2 | 211, -211 | charged pion |
| 3 | 2212 | proton |
| 4 | 22 | gamma |
| 5 | other | other particles |

You can customize this mapping in the dataset class.

## Data Directory Structure

Organize your HDF5 files as follows:

```
data/lartpc/
├── train/
│   ├── event_0001.h5
│   ├── event_0002.h5
│   └── ...
├── val/
│   ├── event_1001.h5
│   └── ...
└── test/
    ├── event_2001.h5
    └── ...
```

## Configuration

### Example Config File

See `configs/lartpc/semseg-pt-v3-lartpc.py` for a complete example.

Key configuration options:

```python
data = dict(
    num_classes=6,
    ignore_index=-1,
    train=dict(
        type="LArTPCDataset",
        split="train",
        data_root="data/lartpc",
        use_reco_coords=True,      # Use reconstructed coordinates
        use_edep_as_strength=True, # Use energy deposition as feature
        transform=[...],
    ),
)

model = dict(
    backbone=dict(
        type="PT-v3",
        in_channels=7,  # coord(3) + strength(1) + wire_coords(3)
        ...
    ),
)
```

### Feature Channels

The `in_channels` for the model depends on which features you include:

| Features | Channels |
|----------|----------|
| coord only | 3 |
| coord + strength | 4 |
| coord + wire_coords | 6 |
| coord + strength + wire_coords | 7 |

Adjust `feat_keys` in the `Collect` transform to match.

## Transforms for LArTPC Data

### Recommended Training Transforms

```python
transform=[
    dict(type="CenterShift", apply_z=True),
    dict(type="RandomRotate", angle=[-1, 1], axis="z", p=0.5),
    dict(type="RandomRotate", angle=[-1/24, 1/24], axis="x", p=0.5),
    dict(type="RandomRotate", angle=[-1/24, 1/24], axis="y", p=0.5),
    dict(type="RandomScale", scale=[0.9, 1.1]),
    dict(type="RandomFlip", p=0.5),
    dict(type="GridSample", grid_size=0.02, mode="train"),
    dict(type="SphereCrop", sample_rate=0.8, mode="random"),
    dict(type="ToTensor"),
    dict(
        type="Collect",
        keys=("coord", "segment"),
        feat_keys=("coord", "strength", "color"),
    ),
]
```

### Grid Size Considerations

- Default: `grid_size=0.02` (2cm voxels)
- Adjust based on your detector resolution
- Smaller values = more points, higher memory usage
- Larger values = faster training, lower resolution

## LArTPC-Specific Considerations

### Coordinate System
- LArTPC typically uses centimeters
- Consider scaling to meters for consistency with other datasets
- The dataset includes a `coord_scale` option for this

### Sparse Data
- LArTPC point clouds can be very sparse
- PointTransformerV3 handles sparsity well with serialization
- Consider adjusting `SphereCrop` sample_rate if needed

### Class Imbalance
- Particle physics has severe class imbalance (many more cosmic rays than neutrino events)
- Consider:
  - Class weights in loss function
  - Focal loss (`dict(type="FocalLoss", ...)`)
  - Oversampling minority classes

### Origin-Based Labeling
- Alternative to particle-type labeling
- Use `origin` field: neutrino (1) vs cosmic (2)
- Useful for cosmic rejection tasks

## Training

```bash
# Single GPU
python tools/train.py --config-file configs/lartpc/semseg-pt-v3-lartpc.py

# Multi-GPU
python tools/train.py --config-file configs/lartpc/semseg-pt-v3-lartpc.py \
    --num-gpus 4
```

## Extending the Dataset

### Adding New Features

1. Modify `get_data()` to load additional HDF5 fields
2. Add to `VALID_ASSETS` if using standard loading
3. Update `feat_keys` in config to include new features
4. Adjust `in_channels` in model config

### Custom Label Mappings

Override `_map_pid_to_class()` or `_map_origin_to_class()` methods:

```python
class MyLArTPCDataset(LArTPCDataset):
    PID_TO_CLASS = {
        11: 0,    # custom mapping
        ...
    }
```

### Multiple Label Types

The dataset supports both:
- `segment`: Semantic labels (particle type or origin)
- `instance`: Instance labels (track IDs)

Enable instance segmentation by including instance labels in training.

## Files

- Dataset: `pointcept/datasets/lartpc.py`
- Config: `configs/lartpc/semseg-pt-v3-lartpc.py`
- This guide: `docs/LArTPC_Dataset_Guide.md`
