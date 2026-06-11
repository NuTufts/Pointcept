# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Pointcept is a codebase for point cloud perception research, supporting semantic/instance segmentation and pre-training methods. This fork (nutufts_lartpc branch) extends it for Liquid Argon Time Projection Chamber (LArTPC) particle physics data.

The Primary Models we are developing/training are:

- the Sonata model to learn reusable representations for LArTPC data represented as space points
- the LArFormer model to find cosmic and neutrino interactions. For neutrino interactions, to also parse the interaction into individual particles. See `docs/LArFormer.md`.

## Common Commands

### Training
```bash
# Using train script (recommended)
sh scripts/train.sh -g <NUM_GPU> -d <DATASET> -c <CONFIG_NAME> -n <EXP_NAME>

# Examples:
sh scripts/train.sh -g 4 -d lartpc -c semseg-pt-v3m1-0-base -n my-experiment
sh scripts/train.sh -g 4 -d scannet -c semseg-pt-v2m2-0-base -n semseg-test

# Resume training
sh scripts/train.sh -g 4 -d lartpc -c semseg-pt-v3m1-0-base -n my-experiment -r true

# Direct invocation
export PYTHONPATH=./
python tools/train.py --config-file configs/lartpc/semseg-pt-v3m1-0-base.py --num-gpus 4 --options save_path=exp/lartpc/my-experiment
```

### Testing
```bash
sh scripts/test.sh -g <NUM_GPU> -d <DATASET> -n <EXP_NAME> -w <WEIGHT_NAME>

# Example:
sh scripts/test.sh -g 4 -d lartpc -n my-experiment -w model_best
```

### Build Custom CUDA Operations
```bash
cd libs/pointops && python setup.py install && cd ../..
cd libs/pointgroup_ops && python setup.py install && cd ../..
```

## Architecture

### Directory Structure
- `pointcept/` - Main package
  - `datasets/` - Dataset classes (DefaultDataset base, LArTPCDataset for physics data)
  - `models/` - Model architectures (PTv3, SpUNet, Sonata, etc.)
  - `engines/` - Training/testing engines and hooks
  - `utils/` - Utilities (config, logging, registry)
- `configs/` - Configuration files organized by dataset (lartpc/, scannet/, s3dis/, etc.)
- `tools/` - Entry points (train.py, test.py)
- `scripts/` - Shell scripts for training/testing
- `libs/` - Custom CUDA extensions (pointops, pointgroup_ops)

### Config System
Configs use inheritance via `_base_` and Python dict format:
```python
_base_ = ["../_base_/default_runtime.py"]
model = dict(type="DefaultSegmentorV2", backbone=dict(type="PT-v3m1", ...))
data = dict(train=dict(type="LArTPCDataset", transform=[...]), ...)
```

### Registry Pattern
Models, datasets, and transforms use a registry pattern:
```python
from .builder import DATASETS
@DATASETS.register_module()
class MyDataset(DefaultDataset): ...
```

### Data Flow
1. Dataset `get_data()` returns dict with `coord`, `segment`, `strength`, `color`, etc.
2. Transforms modify data dict (GridSample, RandomRotate, ToTensor, Collect)
3. `Collect` transform selects final keys and `feat_keys` for model input
4. Model receives `coord`, `grid_coord`, `feat` (concatenated features), `segment`

## LArTPC-Specific

### Dataset
`LArTPCDataset` (pointcept/datasets/lartpc.py) reads HDF5 files from SimChTripletLabelMaker:
- `pos`: 3D coordinates
- `pid`: PDG particle codes mapped to classes (electron=0, muon=1, pion=2, proton=3, gamma=4, ghost=5)
- `pixval`: Wire plane signals (3 channels)
- `uwire/vwire/ywire`: Wire coordinates

### Configs
- `configs/lartpc/semseg-pt-v3m1-0-base.py` - PTv3 semantic segmentation
- `configs/lartpc/pretrain-sonata-v1m1-lartpc.py` - Sonata pre-training
- `configs/lartpc/linearprobe-sonata-v1m1-lartpc.py` - Linear probe evaluation

### Data Organization
```
data/lartpc/
├── train.txt  # or train/ directory with .h5 files
├── val.txt
└── test.txt
```

## Key Classes

### Models
- `DefaultSegmentorV2` - Wrapper combining backbone + head for segmentation
- `PT-v3m1` (PointTransformerV3) - Main backbone, uses serialized attention
- `SpUNet` - Sparse convolution U-Net backbone
- `Sonata` - Self-supervised pre-training framework

### Transforms
- `GridSample` - Voxelization (critical: `grid_size` must match train/val/test)
- `SphereCrop` - Limit points per sample
- `Collect` - Select output keys; `feat_keys` determines model input channels

### Training
- Experiments saved to `exp/<dataset>/<exp_name>/`
- Code backed up to `exp/.../code/`
- Uses OneCycleLR scheduler by default
- Supports wandb logging (`enable_wandb=True`)

## Important Parameters

- `in_channels` in model must match total channels from `feat_keys` in Collect
- `grid_size` must be consistent across train/val/test transforms
- `num_classes` must match dataset class count
- `coord_scale` in LArTPCDataset (0.001 converts mm to m)

## Container Environment

The code in this repository is assumed to be run within a singularity/apptainer container environment.

- When on the Tufts cluster, use `run_in_tufts_pointcept_container.sh`
- When on the local machine use `run_in_local_pointcept_container.sh`

## Data set Information

- MicroBooNE datasets on the Tufts cluster are described in `docs/MicroBooNE_Datasets_on_Tufts.md`