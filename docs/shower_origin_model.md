# Shower Origin Model Architecture (V3)

## Overview

`ShowerOriginPredictorV3` is a query-conditioned architecture for identifying the origin points of electromagnetic showers in LArTPC data. It uses a frozen Sonata encoder with Slot Attention for query formation, virtual grid points for empty-space predictions, and multi-round cross-attention for refinement.

**Source**: `pointcept/models/shower_origin/shower_origin_model.py`

**Registered as**: `ShowerOriginPredictorV3`

## Architecture Diagram

```
Input: Full event point cloud (coord, feat, shower_mask, etc.)
       │
┌──────▼──────────────────────────────────────┐
│ 1. FROZEN SONATA BACKBONE                    │
│    Input: coord, feat, offset                │
│    Output: per-point features (N, 496)       │
│            per-point coords (N, 3)           │
└──────┬───────────────────────────────────────┘
       │
       ├──── shower_mask ────┐
       │                     ▼
       │             ┌───────────────┐
       │             │ Shower points │
       │             │ (S, 496)      │
       │             └───────┬───────┘
       │                     │
       │                     ▼
       │             ┌───────────────────┐
       │             │ 2. SLOT ATTENTION  │
       │             │  K=4 slots         │
       │             │  3 iterations      │
       │             │  Input: (1,S,496)  │
       │             │  Output: (K, 496)  │
       │             └───────┬───────────┘
       │                     │
       ▼                     │
┌─────────────────────────┐  │
│ 3. VIRTUAL GRID         │  │
│  Dense: 2cm, r=30cm     │  │
│  Sparse: 10cm, full TPC │  │
│  kNN interp + pos enc   │  │
│  → (N+V, 496) combined  │  │
└──────┬──────────────────┘  │
       │                     │
       ▼                     ▼
┌──────────────────────────────────────────────┐
│ 4. CROSS-ATTENTION (×3 layers)               │
│    Queries: slot vectors (K, 496)            │
│    Keys/Values: all_features (N+V, 496)      │
│    8 heads, pre-norm, MLP + residual         │
│    → refined queries (K, 496)                │
│    → LayerNorm                               │
└──────┬───────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────┐
│ 5. PER-SLOT PREDICTIONS (slot k=0 only)      │
│                                              │
│  a) Origin Score + Distance:                 │
│     broadcast slot[0] → (N+V, 496)           │
│     concat [event_feat, slot] → (N+V, 992)   │
│     Combiner MLP → (N+V, 496)               │
│     OriginScoreHead:                         │
│       shared MLP (496→256→256)               │
│       → score_head → sigmoid → scores (N+V,) │
│       → dist_head → softplus → dists (N+V,)  │
│                                              │
│  b) Origin Classification:                   │
│     slot[0] vector (496,) →                  │
│     MLP (496→128→3) → logits (3,)            │
│     Classes: 0=inside, 1=outside, 2=on_track │
└──────────────────────────────────────────────┘
       │
       ▼
  Output (inference, virtual pts trimmed):
    - origin_scores: (N,) per-point origin probability
    - all_coords: (N, 3) real point coordinates
    - origin_class: int, argmax of class logits
    - origin_class_logits: (3,) raw logits
    - origin_distances: (N,) predicted distance to origin
```

## Components

### 1. Frozen Sonata Backbone

The Sonata self-supervised encoder extracts per-point features for the full event point cloud. The backbone is frozen during training — only the downstream heads are learned.

To save GPU memory, the student network and non-backbone teacher components are deleted at init time:

```python
if hasattr(self.backbone, 'student'):
    del self.backbone.student
if hasattr(self.backbone, 'teacher'):
    for key in list(self.backbone.teacher.keys()):
        if key != 'backbone':
            del self.backbone.teacher[key]
```

**Output dimension**: 496 (after Sonata upcast)

**Note**: Shower-specific keys (`origin_coord`, `origin_type`, `origin_distance`, `shower_mask`, `start_coord`) are temporarily removed from `input_dict` before the backbone forward pass because PT-v3m2's GridPooling would misinterpret `origin_coord` as per-point coordinates.

### 2. Slot Attention

Aggregates shower point features into K fixed-size slot vectors via iterative competitive attention.

- **Source**: `pointcept/models/shower_origin/slot_attention.py`
- **Key mechanism**: Softmax is applied over the *slot* dimension (not inputs), making slots compete for each input point
- **Parameters**: K=1 slots, 3 iterations, GRU state update + MLP refinement
- **Input**: Shower points selected by `shower_mask` from backbone output
- **Padding**: If fewer than K shower points, pads with mean feature

Runs in float32 to avoid numerical issues with bfloat16.

### 3. Virtual Grid Generator

Generates virtual grid points in empty space where no physical measurements exist, enabling origin prediction outside the detector active volume.

- **Source**: `pointcept/models/shower_origin/virtual_grid.py`
- **Two grid levels**:
  - **Dense grid**: Fine spacing (2 cm) within a 30 cm radius sphere around the shower centroid
  - **Sparse grid**: Coarse spacing (10 cm) covering the full MicroBooNE TPC volume
  - Sparse points within any dense region are removed to avoid overlap
- **Feature computation**:
  1. kNN interpolation (k=8) from real backbone features using inverse-distance weighting
  2. Learned sinusoidal positional encoding (8 frequency bands, MLP: input→128→128→64)
  3. Linear projection: concat [interpolated (496) + pos_enc (64)] → LayerNorm → GELU → (496)

### 4. Cross-Attention

Three layers of standard transformer cross-attention where slot queries attend to all point features (real + virtual).

- **Queries**: Slot vectors (K, 496)
- **Keys/Values**: All event features (N+V, 496)
- **Architecture per layer**: Pre-norm → Multi-head attention (8 heads) → Residual → Pre-norm → MLP (4× expansion) → Residual
- **Final**: LayerNorm after the cross-attention stack

Runs in float32.

### 5. Prediction Heads

Only **slot 0** is used for predictions (single-fragment-per-event training mode).

#### a) Origin Score Head (`OriginScoreHead`)

Per-point prediction of origin likelihood:

```
point_features (N+V, 496) ──┐
                             ├─ concat → (N+V, 992)
slot[0] broadcast (N+V, 496)─┘
         │
    Combiner: Linear(992→496) + LayerNorm + GELU
         │
    Shared MLP: Linear(496→256) + LN + GELU + Linear(256→256) + LN + GELU
         │
    ├─ score_head: Linear(256→1) → sigmoid → origin_scores (N+V,)
    └─ dist_head:  Linear(256→1) → softplus → origin_distances (N+V,)
```

At inference, virtual point scores are trimmed — only real point scores (N,) are returned.

#### b) Origin Classification Head (`OriginClassificationHead`)

Per-fragment classification of origin type from the slot vector directly:

```
slot[0] (496,) → Linear(496→128) → GELU → Linear(128→3) → logits
```

Classes: `0 = inside TPC`, `1 = outside TPC`, `2 = on_track (cosmic)`

## Loss Functions

All computed in float32 with `nan_to_num` safety.

| Loss | Formula | Weight |
|------|---------|--------|
| Score | BCE with logits against Gaussian soft labels: `target = exp(-0.5 * (d / σ)²)` | `score_loss_weight` (default: 1.0) |
| Classification | Cross-entropy on origin type logits | `classification_loss_weight` (default: 1.0) |
| Distance | Smooth L1 between predicted and actual distance to origin | `distance_loss_weight` (default: 0.1) |

`gaussian_sigma` controls the width of the Gaussian target around the ground truth origin point (in coordinate units).

## Model Versions

| Version | Class | Key Features |
|---------|-------|-------------|
| V1 | `ShowerOriginPredictor` | Basic: backbone → slot attention → cross-attention → score head. Direct loss on origin_coord. |
| V2 | `ShowerOriginPredictorV2` | DefaultSegmentorV2-compatible interface. Uses `origin_distance` from data dict. |
| **V3** | `ShowerOriginPredictorV3` | Virtual grid, per-slot predictions, classification head, per-sample batch processing. **Currently used.** |

## Configuration

Example config: `configs/lartpc/shower-origin-sonata-v1m1-v3.py`

Key parameters:

```python
model = dict(
    type="ShowerOriginPredictorV3",
    backbone=dict(type="Sonata-v1m1", ...),
    backbone_out_channels=496,
    num_slots=1,
    slot_iterations=3,
    num_cross_attn_layers=3,
    num_heads=8,
    hidden_channels=256,
    predict_distance=True,
    freeze_backbone=True,
    gaussian_sigma=0.022,       # in normalized coordinate units
    num_origin_classes=3,
    virtual_grid_enabled=True,
    dense_spacing=2.0,          # cm (or normalized units)
    dense_radius=30.0,
    sparse_spacing=10.0,
)
```

## Inference

See `tools/visualize_shower_origin_inference.py` for interactive visualization and `tools/run_shower_origin_inference.py` for batch inference over a dataset.

Coordinate normalization (applied by `NormalizeShowerCoords` transform):
- Center: `[125.0, 0.0, 518.0]` (cm)
- Scale: `179.44` (cm)

The model operates in normalized coordinates. To convert predicted distances back to cm, multiply by `coord_scale`.
