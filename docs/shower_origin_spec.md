# Shower Origin Model — Consolidated Specification

Predicting the 3D origin point and inside/outside-TPC classification of electromagnetic (photon) showers in MicroBooNE LArTPC data.

---

## Table of Contents

1. [Motivation](#motivation)
2. [Ground Truth Logic](#ground-truth-logic)
3. [Architecture](#architecture)
4. [Sonata Backbone Pre-training](#sonata-backbone-pre-training)
5. [Shower Origin Loss Functions](#shower-origin-loss-functions)
6. [Data Pipeline — Training (MC Simulation)](#data-pipeline--training-mc-simulation)
7. [Data Pipeline — Reconstruction (Real/Reco Data)](#data-pipeline--reconstruction-realreco-data)
8. [HDF5 Data Schema](#hdf5-data-schema)
9. [Input Datasets](#input-datasets)
10. [Configuration](#configuration)
11. [Inference and Visualization](#inference-and-visualization)
12. [File Inventory](#file-inventory)
13. [Completion Status](#completion-status)
14. [Known Issues and Fixes](#known-issues-and-fixes)
15. [Open Questions](#open-questions)

---

## Motivation

Photon showers in LArTPC detectors originate from a point that may be displaced from the visible ionization. A photon travels invisibly from its production vertex (e.g., a pi0 decay) to its conversion point where it pair-produces and creates detectable charge. Reconstructing this origin is critical for associating showers with their parent interaction vertices.

Key challenges:

1. **Context-dependent ground truth** -- the "correct" origin depends on how many sibling showers are visible and whether the origin is inside or outside the detector.
2. **Empty-space prediction** -- the PointTransformer backbone only produces features at ionization locations, but origins may be in empty space.
3. **Multi-shower events** -- the model must predict origins for all showers simultaneously.

---

## Ground Truth Logic

Ground truth is produced by the C++ `ShowerFragmentOriginMaker`, which clusters each shower's spacepoints with DBSCAN and assigns per-fragment origin labels. For each shower fragment, the labeler determines an origin point, start point, and classification.

### Origin Type Classification

The inside/outside check uses `pret0shiftedoriginpt` (raw Geant4 truth position without t0 shift or SCE correction), NOT the apparent position which is always inside the TPC by construction.

| Type | Condition | Description |
|------|-----------|-------------|
| `0` (INSIDE) | Origin inside TPC + neutrino-origin (MCParticleGraph origin==1) | Neutrino-induced, inside detector |
| `1` (OUTSIDE) | Origin outside TPC (regardless of origin flag) | Particle created outside detector |
| `2` (ON_TRACK) | Origin inside TPC + cosmic-origin (MCParticleGraph origin==2) | Cosmic-induced, inside detector |

TPC active volume bounds: X: [0, 255.6], Y: [-116.5, 116.5], Z: [0.5, 1035.5] cm.

### Origin Point Assignment

| Condition | Origin Point | Notes |
|---|---|---|
| Inside TPC (type 0 or 2), origin within image bounds, keypoint matched, vertex inferable | True physics vertex (SCE-corrected apparent position) | e.g., pi0 decay point with both photons visible |
| Inside TPC (type 0), pi0 vertex but sibling photon not visible | Trunk fragment start point | Pi0 vertex cannot be inferred from single photon (data loader override) |
| Outside TPC (type 1) | Trunk fragment start point | True origin unreachable by model |
| Origin tick outside image bounds (tick < 2410 or > 8438) | Trunk fragment start point | Origin not visible in detector readout |
| No keypoint matched for this particle | Trunk fragment start point | Particle not tracked by MCKeypointMaker |

The raw Geant4 truth position is always preserved in `pret0shiftedoriginpt` (4D: x,y,z,t).

### Definitions

- **Fragment**: A DBSCAN cluster of spacepoints belonging to a single MC particle (electron, positron, or photon shower).
- **Trunk fragment**: The fragment whose start point is closest to the shower's first energy deposition (preferring fragments with >20 points).
- **Start point**: Most upstream point in the fragment along the shower axis direction.
- **Inside TPC**: True origin (from `pret0shiftedoriginpt`) within MicroBooNE active volume.
- **Conversion point**: Where the photon first creates detectable ionization (`first_edep_pos` from MCParticleGraph).

---

## Architecture

### Model: ShowerOriginPredictorV3

**Source**: `pointcept/models/shower_origin/shower_origin_model.py`

**Registered as**: `ShowerOriginPredictorV3`

Built on top of a frozen Sonata-pretrained PointTransformer V3 backbone.

```
Input: Full event point cloud (coord, feat, shower_mask, etc.)
       │
┌──────▼──────────────────────────────────────┐
│ 1. FROZEN SONATA BACKBONE                    │
│    Input: coord, feat, offset                │
│    Output: per-point features (N, C)         │
│            per-point coords (N, 3)           │
└──────┬───────────────────────────────────────┘
       │
       ├──── shower_mask ────┐
       │                     ▼
       │             ┌───────────────┐
       │             │ Shower points │
       │             │ (S, C)        │
       │             └───────┬───────┘
       │                     │
       │                     ▼
       │             ┌───────────────────┐
       │             │ 2. SLOT ATTENTION  │
       │             │  K slots           │
       │             │  3 iterations      │
       │             │  Input: (1,S,C)    │
       │             │  Output: (K, C)    │
       │             └───────┬───────────┘
       │                     │
       ▼                     │
┌─────────────────────────┐  │
│ 3. VIRTUAL GRID         │  │
│  Dense: 2cm, r=30cm     │  │
│  Sparse: 10cm, full TPC │  │
│  kNN interp + pos enc   │  │
│  → (N+V, C) combined    │  │
└──────┬──────────────────┘  │
       │                     │
       ▼                     ▼
┌──────────────────────────────────────────────┐
│ 4. CROSS-ATTENTION (×3 layers)               │
│    Queries: slot vectors (K, C)              │
│    Keys/Values: all_features (N+V, C)        │
│    8 heads, pre-norm, MLP + residual         │
│    → refined queries (K, C)                  │
│    → LayerNorm                               │
└──────┬───────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────┐
│ 5. PER-SLOT PREDICTIONS (slot k=0 only)      │
│                                              │
│  a) Origin Score + Distance:                 │
│     broadcast slot[0] → (N+V, C)             │
│     concat [event_feat, slot] → (N+V, 2C)    │
│     Combiner MLP → (N+V, C)                 │
│     OriginScoreHead:                         │
│       shared MLP (C→256→256)                 │
│       → score_head → sigmoid → scores (N+V,) │
│       → dist_head → softplus → dists (N+V,)  │
│                                              │
│  b) Origin Classification:                   │
│     slot[0] vector (C,) →                    │
│     MLP (C→128→3) → logits (3,)              │
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

**Current training mode (K=1):** Single-shower prediction per crop. Each training sample is a BiasedSphereCrop around one fragment centroid. The model predicts a Gaussian origin heatmap over all input points (not just shower points) and a 3-class origin type.

### Component Details

#### 1. Frozen Sonata Backbone

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

**Note**: Shower-specific keys (`origin_coord`, `origin_type`, `origin_distance`, `shower_mask`, `start_coord`) are temporarily removed from `input_dict` before the backbone forward pass because PT-v3m2's GridPooling would misinterpret `origin_coord` as per-point coordinates.

#### 2. Slot Attention

Aggregates shower point features into K fixed-size slot vectors via iterative competitive attention.

- **Source**: `pointcept/models/shower_origin/slot_attention.py`
- **Key mechanism**: Softmax is applied over the *slot* dimension (not inputs), making slots compete for each input point
- **Parameters**: K=1 slots, 3 iterations, GRU state update + MLP refinement
- **Input**: Shower points selected by `shower_mask` from backbone output
- **Padding**: If fewer than K shower points, pads with mean feature

Runs in float32 to avoid numerical issues with bfloat16.

#### 3. Virtual Grid Generator

Generates virtual grid points in empty space where no physical measurements exist, enabling origin prediction outside the detector active volume.

- **Source**: `pointcept/models/shower_origin/virtual_grid.py`
- **Two grid levels**:
  - **Dense grid**: Fine spacing (2 cm) within a 30 cm radius sphere around the shower centroid
  - **Sparse grid**: Coarse spacing (10 cm) covering the full MicroBooNE TPC volume
  - Sparse points within any dense region are removed to avoid overlap
- **Feature computation**:
  1. kNN interpolation (k=8) from real backbone features using inverse-distance weighting
  2. Learned sinusoidal positional encoding (8 frequency bands, MLP: input→128→128→64)
  3. Linear projection: concat [interpolated + pos_enc] → LayerNorm → GELU → output dim
- **Status**: Currently disabled by default (OOM on <=16GB GPUs with ~91K virtual points × 1088 channels).

#### 4. Cross-Attention

Three layers of standard transformer cross-attention where slot queries attend to all point features (real + virtual).

- **Queries**: Slot vectors (K, C)
- **Keys/Values**: All event features (N+V, C)
- **Architecture per layer**: Pre-norm → Multi-head attention (8 heads) → Residual → Pre-norm → MLP (4× expansion) → Residual
- **Final**: LayerNorm after the cross-attention stack

Runs in float32.

#### 5. Prediction Heads

Only **slot 0** is used for predictions (single-fragment-per-event training mode).

**a) Origin Score Head (`OriginScoreHead`)**

Per-point prediction of origin likelihood:

```
point_features (N+V, C) ──┐
                           ├─ concat → (N+V, 2C)
slot[0] broadcast (N+V, C)─┘
         │
    Combiner: Linear(2C→C) + LayerNorm + GELU
         │
    Shared MLP: Linear(C→256) + LN + GELU + Linear(256→256) + LN + GELU
         │
    ├─ score_head: Linear(256→1) → sigmoid → origin_scores (N+V,)
    └─ dist_head:  Linear(256→1) → softplus → origin_distances (N+V,)
```

At inference, virtual point scores are trimmed — only real point scores (N,) are returned.

**b) Origin Classification Head (`OriginClassificationHead`)**

Per-fragment classification of origin type from the slot vector directly:

```
slot[0] (C,) → Linear(C→128) → GELU → Linear(128→3) → logits
```

Classes: `0 = inside TPC`, `1 = outside TPC`, `2 = on_track (cosmic)`

#### Additional Components

- **Hungarian Matching**: Optimal bipartite assignment of K slots to J ground-truth showers during training (scipy `linear_sum_assignment`). Not used when K=1.
- **ShowerOriginEvaluator** (`engines/hooks/shower_origin_evaluator.py`): Validation hook computing origin recall, specificity, peak distance, and classification accuracy.

### Model Versions

| Version | Class | Key Features |
|---------|-------|-------------|
| V1 | `ShowerOriginPredictor` | Basic: backbone → slot attention → cross-attention → score head. Direct loss on origin_coord. |
| V2 | `ShowerOriginPredictorV2` | DefaultSegmentorV2-compatible interface. Uses `origin_distance` from data dict. |
| **V3** | `ShowerOriginPredictorV3` | Virtual grid, per-slot predictions, classification head, per-sample batch processing. **Currently used.** |

---

## Sonata Backbone Pre-training

The Sonata backbone used by the shower origin model is pre-trained using a self-supervised **teacher-student architecture** (similar to DINO/SwAV) with online clustering. The model learns representations by predicting cluster assignments, where:

- **Student network**: Trainable, receives augmented/masked views
- **Teacher network**: Exponential moving average (EMA) of student weights, provides target assignments

All three losses compute **cross-entropy** between:
- **Student**: Softmax-normalized predictions (temperature = 0.1)
- **Teacher**: Sinkhorn-Knopp normalized cluster assignments (temperature ~0.04-0.07)

### Sonata Loss Functions

#### 1. `mask_loss` (default weight: 2/8)

**Purpose**: Masked point cloud reconstruction.

The student sees a **masked** version of the global point cloud where random patches are hidden or jittered. The teacher sees the **unmasked** version. The loss encourages the student to predict the same cluster assignments as the teacher for spatially matched points.

**What it teaches**: Robustness to missing data; ability to infer masked regions from context.

#### 2. `roll_mask_loss` (default weight: 2/8)

**Purpose**: Cross-view consistency with masking.

Requires `num_global_view >= 2`. The "roll" operation swaps the two global views. The student predicts from a masked view, but the teacher target comes from a **different augmentation** of the same scene.

**What it teaches**: Features that are invariant to augmentations and robust to masking simultaneously.

#### 3. `unmask_loss` (default weight: 4/8)

**Purpose**: Local-to-global view consistency (no masking).

The student processes **local views** (smaller crops), while the teacher processes the **principal global view** (unmasked). Points are matched by spatial proximity.

**What it teaches**: Scale-invariant features; local regions should have representations consistent with the global context.

#### Combined Loss

```python
loss = mask_loss * (2/8) + roll_mask_loss * (2/8) + unmask_loss * (4/8)
```

The `unmask_loss` has the highest weight because local-to-global consistency is the primary pretext task.

### How Prototype Vectors Are Updated

Prototypes are stored as the **weights of a linear layer** with weight normalization in the `OnlineCluster` head:

```python
self.prototype = weight_norm(nn.Linear(embed_channels, num_prototypes, bias=False))
```

The weight matrix has shape `[num_prototypes, embed_channels]` = `[4096, 512]`. Each row is a prototype vector. The **magnitude is fixed to 1** and frozen — only the **direction** is learned. All prototypes live on the unit hypersphere.

The model maintains **two separate clustering heads**, each with its own set of 4096 prototypes:

1. **`mask_head`**: Used by `mask_loss` and `roll_mask_loss`
2. **`unmask_head`**: Used by `unmask_loss`

| Loss | Student Head Updated | Prototypes Affected |
|------|---------------------|---------------------|
| `mask_loss` | `student.mask_head` | `student.mask_head.prototype` |
| `roll_mask_loss` | `student.mask_head` | `student.mask_head.prototype` |
| `unmask_loss` | `student.unmask_head` | `student.unmask_head.prototype` |

The **teacher's prototypes are NOT trained by gradient descent**. They are updated via EMA:

```python
teacher = momentum * teacher + (1 - momentum) * student
```

where momentum starts at 0.996 and increases to 1.0 over training.

### Interpreting Sonata Loss Values

#### Mathematical Basis

The loss is cross-entropy over K prototypes (default K=4096):

$$L = -\sum_{i}^{N} \sum_{k}^{K} Q_{ik} \log(p_{ik})$$

where:
- $Q_{ik}$ = teacher's probability (from Sinkhorn-Knopp optimal transport) for sample $i$ to belong to prototype $k$
- $p_{ik}$ = student's probability for sample $i$ to belong to prototype $k$
- Both normalize to 1 over $i$: $\sum_{i}^{N} Q_{ik} = 1$ and $\sum_{i}^{N} p_{ik} = 1$

Notes on the Sinkhorn-Knopp target:
- $Q_{ik}$ is the optimal transport plan between uniformly distributed mass over $K$ clusters and $N$ points.
- This ensures all cluster prototypes are used, avoiding representation collapse.
- However, individual samples can still have low-entropy distributions over prototypes.
- Works best with enough diversity of patterns per batch — may need large batch sizes or accumulation.
- For LArTPC data with large class imbalances, enough "semantic diversity" must always be present.

The student probability is:

$$ p_{ik} = \frac{ e^{\frac{1}{\tau} z_i^T c_k }}{\sum_{k'}^{K}e^{\frac{1}{\tau} z_i^T c_{k'} }} $$

where $z_i$ is the feature vector and $c_k$ is the prototype vector.

#### Random Baseline

At initialization: $L_{random} = \ln(4096) \approx 8.32$

#### Loss Value Reference Table

| Loss Value | Interpretation |
|------------|----------------|
| ~8.3 | Random predictions (maximum entropy over 4096 prototypes) |
| ~2.0-2.5 | Typical mid-training values; structured representations emerging |
| ~1.5-2.0 | Well-trained model with confident cluster assignments |
| ~0 | Perfect prediction (or potential mode collapse - verify carefully) |

#### Effective Number of Clusters

A loss value L corresponds to: $N_{effective} = e^L$

- Loss = 2.0 → ~7.4 effective clusters
- Loss = 2.5 → ~12.2 effective clusters
- Loss = 1.5 → ~4.5 effective clusters

### Sonata Training Dynamics

1. **Initial drop**: Loss falls rapidly from ~8.3 as model learns basic structure
2. **Warmup bumps**: Around 5% of training, schedulers transition from "easy" to "hard" settings (`mask_size`: 0.1→0.4, `mask_ratio`: 0.3→0.7, `teacher_temp`: 0.04→0.07). Can cause temporary loss increases.
3. **Gradual descent**: Continued improvement with high batch-to-batch variance
4. **Plateau**: Eventually settles to a data-dependent floor

#### Warning Signs

| Symptom | Potential Issue |
|---------|-----------------|
| Loss drops to ~0 quickly | Mode collapse (trivial solution) |
| Loss plateaus early, never improves | Learning rate too low or architecture problem |
| Loss diverges (increases steadily) | Learning rate too high |
| All three losses become identical | Representation collapse |
| Extremely high variance that doesn't decrease | Batch size too small or learning rate too high |

### Prototypes vs. Semantic Classes

For LArTPC data with 6 semantic classes, the loss does **not** settle around ln(6) ≈ 1.79 because:

1. **Prototypes capture finer structure**: Within "muon" alone, the model might learn straight track segments, endpoints, scattering vertices, different dE/dx patterns.
2. **Sinkhorn-Knopp forces utilization**: All 4096 prototypes are used roughly equally.
3. **Sub-class structure**: The model discovers structure at semantic, sub-class, and geometric levels.

**The loss value alone does not tell you if representations are useful for your downstream task.** Evaluate via linear probing, fine-tuning, or kNN evaluation.

### Key Sonata Components

| Component | Purpose |
|-----------|---------|
| `OnlineCluster` head | MLP that projects features to prototype space |
| `sinkhorn_knopp()` | Soft cluster assignment normalization for teacher (prevents collapse) |
| `match_neighbour()` | Spatial matching between views (default radius < 0.08) |
| EMA teacher update | teacher = momentum * teacher + (1-momentum) * student |

**References**: `pointcept/models/sonata/sonata_v1m1_base.py`, `configs/lartpc/pretrain-sonata-v1m1-lartpc.py`

---

## Shower Origin Loss Functions

All computed in float32 with `nan_to_num` safety.

| Loss | Formula | Weight |
|------|---------|--------|
| Score | BCE with logits against Gaussian soft labels: `target = exp(-0.5 * (d / σ)²)` | `score_loss_weight` (default: 1.0) |
| Classification | Cross-entropy on origin type logits | `classification_loss_weight` (default: 1.0) |
| Distance | Smooth L1 between predicted and actual distance to origin | `distance_loss_weight` (default: 0.1) |

`gaussian_sigma` controls the width of the Gaussian target around the ground truth origin point (in coordinate units).

---

## Data Pipeline — Training (MC Simulation)

### Step 1: ROOT to HDF5 Conversion

Convert `dlmerged` ROOT files to per-event HDF5 using `SimChTripletLabelMaker`:

```bash
cd lartpc_data_prep
python process_dlmerged_to_hdf5_event_files.py \
    --input /path/to/dlmerged_file.root \
    --output-dir ./output/
```

The C++ `SimChTripletLabelMaker` (modified in `ubdl/larflow`) exports an `mc_particle_tree` group alongside the standard `triplet_data` and `mckeypoints`.

**Note:** Shower fragment labels (`shower_fragments/` group) are now produced directly by the C++ `ShowerFragmentOriginMaker` during Step 1 — no separate post-processing step is needed.

### Step 2: Training

```bash
# Single GPU
python tools/train.py \
    --config-file configs/lartpc/shower-origin-sonata-v1m1-v3.py \
    --num-gpus 1 \
    --options save_path=exp/shower_origin/v3

# Multi-GPU via SLURM
sbatch scripts/slurm/train_shower_origin.sh
```

### MC Visualization

Inspect labeled events interactively with the Dash/Plotly 3D viewer:

```bash
cd lartpc_data_prep

# Show only shower fragments + origin markers
python vis_shower_fragments.py -i event_file.h5 -e 0

# With non-shower context points
python vis_shower_fragments.py -i event_file.h5 -e 0 --show-non-shower

# Color by origin type (green=INSIDE, red=OUTSIDE)
python vis_shower_fragments.py -i event_file.h5 -e 0 -c origin --show-non-shower

# With MC keypoints overlaid
python vis_shower_fragments.py -i event_file.h5 -e 0 --show-keypoints

# Include ghost points
python vis_shower_fragments.py -i event_file.h5 -e 0 --show-non-shower --include-ghosts
```

The viewer displays:
- Shower spacepoints colored by fragment index, origin type, or trackid
- Origin markers: green spheres (INSIDE), red spheres (OUTSIDE), labeled O0, O1, ...
- Conversion point markers: cyan diamonds (when different from origin), labeled C0, C1, ...
- Dashed lines from origin to conversion point (photon flight path)
- Summary table with per-fragment metadata

### Dataset Output Keys (ShowerOriginDataset)

| Key | Shape | Description |
|---|---|---|
| `coord` | (N, 3) | Point coordinates |
| `feat` | (N, C) | Point features (from Collect transform) |
| `shower_masks` | (max_showers, N) | Per-shower boolean masks, padded |
| `origin_coords` | (max_showers, 3) | Ground truth origin coordinates |
| `origin_types` | (max_showers,) | 0=INSIDE, 1=OUTSIDE, 2=ON_TRACK |
| `valid_showers_mask` | (max_showers,) | Which shower slots are real vs padding |
| `num_showers` | int | Actual number of showers in event |

---

## Data Pipeline — Reconstruction (Real/Reco Data)

This section describes how to apply the shower origin model within the MicroBooNE reconstruction pipeline for real or simulated data (without MC truth).

### Overview of Integration Steps

1. Make spacepoints from wire-plane image data, remove ghost points, and apply SSNet labels. (ROOT→ROOT, completed.)
2. Convert the ROOT data file into HDF5 compatible with `ShowerOriginDataset`. (Per-event HDF5 files.)
3. Apply the shower origin model to each event file. (Inference produces a single result HDF5.)
4. Integrate the output into the LANTERN reco chain as an optional module.

### Step 1: Making Spacepoints with SSNet Labels

The input is processed through `ubdl/lantern_scripts.sh` which runs `ubdl/larflow/larmatchnet/larmatch/deploy_larmatchme.py`:

```bash
python3 deploy_larmatchme.py \
    --config-file config_larmatchme_deploycpu.yaml \
    --supera merged_dlana_d5cd7f5c-67e6-4bee-8c3a-dcefb42a63c0.root \
    --weights /cluster/home/ubdl/larflow/larmatchnet/larmatch/larmatch_ckpt78k.pt \
    --output output_test.root \
    --min-score 0.5 --adc-name wire --chstatus-name wire --device-name cpu --use-skip-limit -tb
```

Note: This runs in the production MicroBooNE "lantern" container (requires MicroBooNE CVMFS access).

The script produces two ROOT files: `_larcv.root` and `_larlite.root`.

The key outputs are in the ROOT TTree `larflow3dhit_larmatch_tree`. Each hit is a `larflow3dhit` (inherits from `std::vector<float>`):

```
[0-2]:   x, y, z
[3-9]:   flow direction scores (deprecated; [9] is larmatch score for triplet)
[10-16]: 7 ssnet scores (bg, track, shower) from larmatch
[17-22]: 6 keypoint label scores [nu, track-start, track-end, nu-shower, delta, michel]
[23-25]: reserved for plane charge
[26-28]: 3D flow direction
```

Additional `larflow3dhit` member variables:
- `hit.tick` (int): Image row (tick) — used to sample pixel values from wire-plane images.
- `hit.targetwire` (std::vector\<int\>): Wire indices per plane. `[0]`=U, `[1]`=V, `[2]`=Y.
- `hit.renormed_shower_score` (float): Combined SSNet shower probability (sum of renormalized shower + delta + michel scores).

To read data from the larlite ROOT file:

```python
import ROOT as rt
from larlite import larlite

io = larlite.storage_manager(larlite.storage_manager.kREAD)
io.add_in_filename("output_larlite.root")
io.open()

for ientry in range(io.get_entries()):
    event_hits = io.get_data(larlite.data.kLArFlow3DHit, "larmatch")
    for ihit in range(event_hits.size()):
        hit = event_hits.at(ihit)
        # ...
```

### Step 2: Convert Larlite ROOT to ShowerOriginDataset HDF5

**Script**: `lartpc_data_prep/convert_larlite_to_showerorigin_h5.py`

Since this is reco data (no MC truth), shower fragments are created by:
1. Selecting shower-like hits using `hit.renormed_shower_score >= threshold`
2. Clustering with DBSCAN (eps=3.0 cm, min_samples=4, matching the C++ `ShowerFragmentOriginMaker`)
3. Computing a PCA-based start point per fragment (most upstream point along principal axis)

Truth-only fields (`originpt`, `type`, `pret0shiftedoriginpt`) are filled with placeholders.

**Usage:**

```bash
python lartpc_data_prep/convert_larlite_to_showerorigin_h5.py \
    --input-larlite output_larlite.root \
    --output-dir ./showerorigin_h5/ \
    --input-larcv output_larcv.root \
    --shower-threshold 0.5
```

**Command line arguments:**

| Argument | Default | Description |
|---|---|---|
| `--input-larlite` | (required) | Larlite ROOT file from `deploy_larmatchme.py` |
| `--input-larcv` | None | Larcv ROOT file for wire-plane pixel values. If not provided, `pixval` is filled with ones. |
| `--output-dir` | (required) | Output directory for per-event HDF5 files |
| `--min-score` | None | Optional stricter larmatch score filter |
| `--shower-threshold` | 0.5 | Threshold on `renormed_shower_score` for DBSCAN input |
| `--dbscan-eps` | 3.0 | DBSCAN neighborhood radius (cm) |
| `--dbscan-min-samples` | 4 | DBSCAN minimum cluster size |
| `--min-fragment-points` | 20 | Minimum points per fragment to keep |
| `--hit-producer` | "larmatch" | larlite producer name for larflow3dhit |
| `-n` / `--nentries` | -1 | Max entries to process (-1 = all) |
| `--start-entry` | 0 | First entry to process |

**Key implementation details:**

- `extract_hits()`: Reads `larflow3dhit` objects via `larlite.storage_manager`, extracting `pos`, `tick`, `targetwire`, `renormed_shower_score`, and larmatch score.
- `LArCVPixelReader`: Manages a `larcv.IOManager` for sampling wire-plane pixel values. Uses `hit.tick` → `meta.row(tick)` and `hit.targetwire[plane]` → `meta.col(wire)`.
- `cluster_shower_fragments()`: Selects shower hits by threshold, runs `sklearn.cluster.DBSCAN`, filters clusters below `min_fragment_points`.
- `compute_start_point()`: PCA via SVD on cluster points; picks the point with smallest projection along the principal axis.

**Output file naming:** `showerorigin_<input_basename>_entry<NNNNNN>.h5`

### Reco Visualization: `tools/visualize_shower_origin_reco.py`

Interactive Dash/Plotly 3D viewer for inspecting the reco HDF5 output.

```bash
# Single file
python tools/visualize_shower_origin_reco.py --input /path/to/event.h5

# Directory of files
python tools/visualize_shower_origin_reco.py --input-dir ./showerorigin_h5/

# File list
python tools/visualize_shower_origin_reco.py --data-list /path/to/filelist.txt
```

**Panels:**
- **Row 1, Left — Selected Fragment**: Currently selected fragment (red) with PCA start point (cyan cross). Non-fragment points in gray. MicroBooNE detector outline shown.
- **Row 1, Right — Shower Score**: All points colored by `renormed_shower_score` (0–1 colormap).
- **Row 2, Full Width — All Fragments**: Each DBSCAN cluster in a distinct color. All start points labeled S0, S1, etc.

### Analyzing Output Before Full Integration

Before performing Step 4, data up to Step 3 can be used to create a tree merged with LANTERN analysis. This enables event selection by asking whether an "inside" shower event exists and whether the predicted origin satisfies selection criteria.

---

## HDF5 Data Schema

### Training Data (MC Simulation)

#### mc_particle_tree (from SimChTripletLabelMaker)

```
entry_0/mc_particle_tree/
    trackid:              (M,) int     -- unique MC track IDs
    pid:                  (M,) int     -- PDG particle code
    parent_trackid:       (M,) int     -- parent's trackid (-1 for primaries)
    origin:               (M,) int     -- 1=neutrino, 2=cosmic
    start_pos:            (M, 3) float -- true start position [cm]
    start_pos_sce:        (M, 3) float -- SCE-corrected start position [cm]
    energy_mev:           (M,) float   -- kinetic energy [MeV]
    process_code:         (M,) int     -- creation process (0=primary, 1=Decay, 2=compt, 3=conv, ...)
    num_daughters:        (M,) int     -- number of daughters per particle
    daughter_start_indices: (M,) int   -- index into flattened daughter list
    daughter_trackids:    (D,) int     -- flattened daughter trackid list
    nu_vertices:          (V, 3) float -- neutrino interaction vertices [cm]
```

#### shower_fragments (from C++ ShowerFragmentOriginMaker)

```
entry_0/shower_fragments/
    @num_fragments: int              -- attribute: total number of DBSCAN fragments
    trackid:              (F,) int   -- Geant4 track ID per fragment
    pid:                  (F,) int   -- PDG code (22, 11, -11)
    istrunk:              (F,) int   -- 1=trunk, 2=secondary fragment
    type:                 (F,) int   -- 0=nu-inside, 1=outside, 2=cosmic-inside
    startpt:              (F, 3) float -- most upstream point per fragment [cm]
    originpt:             (F, 3) float -- prediction target origin [cm] (apparent; for outside/unreachable, equals trunk startpt)
    pret0shiftedoriginpt: (F, 4) float -- raw Geant4 truth origin (x,y,z,t) [cm, ns]
    pointindices_flat:    (T,) long  -- concatenated point indices into triplet_data
    pointindices_counts:  (F,) int   -- number of points per fragment
```

Where F = number of fragments, T = total points across all fragments.

### Reco Data (from `convert_larlite_to_showerorigin_h5.py`)

```
entry_0/
  triplet_data/
    pos:          (N, 3) float32  — x, y, z coordinates (cm)
    pixval:       (N, 3) float32  — wire-plane ADC values [U, V, Y]
    uwire:        (N,)   float32  — U wire index
    vwire:        (N,)   float32  — V wire index
    ywire:        (N,)   float32  — Y wire index
    hasmatch:     (N,)   int64    — all ones (ghosts already removed)
    shower_score: (N,)   float32  — renormed_shower_score per hit
    lm_score:     (N,)   float32  — larmatch score per hit
  shower_fragments/
    @num_fragments: int
    pointindices_flat:   (T,) int64   — concatenated point indices
    pointindices_counts: (F,) int64   — points per fragment
    startpt:             (F, 3) float32 — PCA-based start point
    trackid:             (F,) int64   — sequential dummy IDs (0, 1, 2, ...)
    pid:                 (F,) int64   — all 22 (photon, generic shower)
    istrunk:             (F,) int64   — all 1 (each fragment independent)
    type:                (F,) int64   — all -1 (unknown, model predicts this)
    originpt:            (F, 3) float32 — placeholder zeros (model predicts)
    pret0shiftedoriginpt:(F, 4) float32 — placeholder zeros (no MC truth)
    nu_vertex_is_visible: int64        — 0 (unknown for reco data)
```

---

## Input Datasets

List files are in `/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/pointcept/lartpc_data_prep/inputlists/`.

| Sample Name | File List | Num Files | Description |
|---|---|---|---|
| bnb_nu_corsika | bnb_nu_corsika_prod2.txt | 5000 | BNB neutrino flux (all flavors). Nu interactions in LAr volume. Includes cosmics. |
| bnb_nu_pi0_corsika | bnb_nu_pi0_corsika_prod2.txt | 4995 | Same as bnb_nu_corsika (intended pi0 filter did not work). Sample name kept. |
| bnb_nu_pi0filter_corsika | bnb_nu_pi0filter_corsika.txt | 8313 | BNB flux, all flavors. Only nu interactions with at least 1 pi0. Includes cosmics. |
| bnb_nue_corsika | bnb_nue_corsika_prod2.txt | 4999 | BNB flux, nue flavor only. Nu interactions inside TPC. Includes cosmics. |
| bnb_nu_chargedpiplus_corsika | bnb_nu_chargedpiplus_corsika_prod2.txt | 8050 | BNB flux, all flavors. Only nu interactions with at least 1 charged pion. Includes cosmics. |

### Production Scripts

Located at `/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/pointcept/lartpc_data_prep/`.

Each data set has a SLURM submission script calling a bash script that runs `process_dlmerged_to_hdf5_event_files.py` on N files (the "stride").

Output folder: `/cluster/tufts/wongjiradlab/larbys/data/ub_on_tufts/hdf5/v3_showerfragments/[sample name]`

| Sample Name | Jobs Required | Stride | SLURM Script | Bash Script |
|---|---|---|---|---|
| bnb_nu_corsika | 100 | 50 | submit_bnbnu_corsika.sh | run_corsika_bnb_nu.sh |
| bnb_nu_pi0_corsika | 100 | 50 | submit_bnbnu_pi0_corsika.sh | run_corsika_bnb_nu_pi0.sh |
| bnb_nu_pi0filter_corsika | 83 | 100 | submit_bnbnu_pi0filter_corsika.sh | run_corsika_bnb_nu_pi0filter.sh |
| bnb_nue_corsika | 100 | 50 | submit_bnbnue_corsika.sh | run_corsika_bnb_nue.sh |
| bnb_nu_chargedpiplus_corsika | 81 | 100 | submit_bnbnu_chargedpiplus_corsika.sh | run_corsika_bnb_nu_chargedpiplus.sh |

### File Size Impact

Including shower fragments and mc particle tree in HDF5 produces a tolerable size increase (~1-2%). When done, v2 files can be removed.

---

## Configuration

Example config: `configs/lartpc/shower-origin-sonata-v1m1-v3.py`

Key parameters:

```python
model = dict(
    type="ShowerOriginPredictorV3",
    backbone=dict(type="Sonata-v1m1", ...),
    backbone_out_channels=496,       # or 1088 for v6 config
    num_slots=1,
    slot_iterations=3,
    num_cross_attn_layers=3,
    num_heads=8,
    hidden_channels=256,
    predict_distance=True,
    freeze_backbone=True,
    gaussian_sigma=0.022,            # in normalized coordinate units
    num_origin_classes=3,
    virtual_grid_enabled=True,       # disabled by default (OOM risk)
    dense_spacing=2.0,               # cm (or normalized units)
    dense_radius=30.0,
    sparse_spacing=10.0,
)
```

Coordinate normalization (applied by `NormalizeShowerCoords` transform):
- Center: `[125.0, 0.0, 518.0]` (cm)
- Scale: `179.44` (cm)

The model operates in normalized coordinates. To convert predicted distances back to cm, multiply by `coord_scale`.

---

## Inference and Visualization

- **Batch inference**: `tools/run_shower_origin_inference.py`
- **Interactive visualization (MC data)**: `tools/visualize_shower_origin_inference.py`
- **Data loader visualization**: `tools/visualize_shower_origin.py` — Dash 3-panel viewer: shower mask, origin_distance, Gaussian target
- **Reco data visualization**: `tools/visualize_shower_origin_reco.py`
- **Raw HDF5 viewer**: `lartpc_data_prep/vis_shower_fragments.py`

---

## File Inventory

### Workstream 1: C++ Ground Truth (ubdl/larflow)

| File | Description |
|---|---|
| `ubdl/larflow/.../ShowerFragmentOriginMaker.h` | Header: fragment data struct forward decl, method declarations |
| `ubdl/larflow/.../ShowerFragmentOriginMaker.cxx` | DBSCAN clustering, trunk/type/origin logic, HDF5 export |
| `ubdl/larflow/.../ShowerFragmentOrigin.h` | Data struct for fragment arrays |
| `ubdl/larflow/.../SimChTripletLabelMaker.h` | Added `save_mc_particle_tree()` declaration |
| `ubdl/larflow/.../SimChTripletLabelMaker.cxx` | Calls `_shower_fragment_maker.save_entry_to_hdf()` + `save_mc_particle_tree()` |

### Workstream 2: V3 Model Architecture (Pointcept)

| File | Description |
|---|---|
| `pointcept/models/shower_origin/shower_origin_model.py` | V1, V2, and V3 model definitions |
| `pointcept/models/shower_origin/virtual_grid.py` | `VirtualGridGenerator`, `PositionalEncoding3D` |
| `pointcept/models/shower_origin/slot_attention.py` | `SlotAttention` module |
| `pointcept/models/shower_origin/__init__.py` | Exports V3, classification head, virtual grid |
| `configs/lartpc/shower-origin-sonata-v1m1-v3.py` | V3 training config |

### Workstream 3: Data Pipeline (Pointcept)

| File | Description |
|---|---|
| `pointcept/datasets/shower_origin.py` | `ShowerOriginDataset` with multi-shower support, dual format loading |
| `lartpc_data_prep/convert_larlite_to_showerorigin_h5.py` | Larlite ROOT → ShowerOriginDataset HDF5 conversion |
| `scripts/slurm/prep_shower_origin_data.sh` | SLURM array job for data prep |
| `scripts/slurm/train_shower_origin.sh` | SLURM training job |

### Workstream 4: Training Infrastructure (Pointcept)

| File | Description |
|---|---|
| `pointcept/engines/hooks/shower_origin_evaluator.py` | `ShowerOriginEvaluator` hook — validation metrics |
| `pointcept/datasets/transform.py` | `RandomScale` and `RandomFlipAxis` extended with `extra_coord_keys` |

### Workstream 5: Visualization & Documentation (Pointcept)

| File | Description |
|---|---|
| `tools/visualize_shower_origin.py` | Dash 3-panel viewer for data loader output |
| `tools/visualize_shower_origin_inference.py` | Interactive inference visualization |
| `tools/visualize_shower_origin_reco.py` | Reco HDF5 3D viewer |
| `lartpc_data_prep/vis_shower_fragments.py` | Raw HDF5 3D viewer |
| `docs/LArTPC_HDF5_Data_Format.md` | HDF5 schema documentation |
| `docs/shower_fragment_origin_spec.md` | Shower fragment origin implementation specification |

---

## Completion Status

### Done

- [x] **Phase 1a**: All three workstreams implemented in parallel
  - [x] C++ `save_mc_particle_tree()` added to `SimChTripletLabelMaker` and compiled
  - [x] C++ `ShowerFragmentOriginMaker` — DBSCAN clustering, per-fragment origin/start/type, HDF5 export
  - [x] `pret0shiftedoriginpt` field added — true Geant4 origin preserved for diagnostics
  - [x] Origin reassignment for outside-TPC and unreachable origins (moved to trunk startpt)
  - [x] `ShowerOriginPredictorV3` with virtual grid, Hungarian matching, classification head
  - [x] `ShowerOriginDataset` rewritten for multi-shower with padded batching (supports new flat format + legacy)
  - [x] `vis_shower_fragments.py` interactive 3D viewer with dual format support, pret0 hover text
  - [x] SLURM scripts for data prep and training
  - [x] HDF5 data format documentation (`LArTPC_HDF5_Data_Format.md`)
  - [x] Shower fragment origin specification (`shower_fragment_origin_spec.md`)
- [x] **Phase 1a-data**: ROOT files reprocessed with C++ ShowerFragmentOriginMaker
  - [x] 10 events visually inspected and validated
- [x] **Phase 1b**: Validate training data labels
  - [x] Verified origin labels for type-0 showers where pi0 vertex is unreachable
  - [x] Added Python data loader override: origin → trunk startpt when sibling photon has no visible fragments
  - [x] Validation on 8 events (17 files total): 572 fragments (60 type-0, 81 type-1, 431 type-2), 6 pi0 pairs (4 both-visible, 2 single-visible), 2 single-photon cases correctly overridden
  - [x] Validation script: `lartpc_data_prep/validate_shower_origins.py`
- [x] **Phase 2**: Integration testing
  - [x] End-to-end test: dataset → model forward pass (batch support added, tested with batch_size=16)
  - [x] ShowerOriginEvaluator hook with validation metrics
  - [x] Data augmentation fix: `RandomScale` and `RandomFlipAxis` now transform `origin_coord`/`start_coord` consistently
  - [x] BiasedSphereCrop `prob_random` set to 0.0 to prevent crops that miss the target fragment
  - [x] Overfitting test on single event (43 fragments, 10 epochs) — model learns successfully
- [x] Reco pipeline: conversion script and visualization tool implemented

#### Single-Event Overfit Results (43 fragments, 10 epochs, batch_size=16)

| Metric | Value |
|---|---|
| Train loss (final epoch avg) | 0.0495 |
| Val loss | 0.0486 |
| Origin Recall | 0.8925 |
| Non-origin Specificity | 0.9672 |
| Mean Peak Distance (normalized) | 0.0146 (~2.6 cm) |
| Median Peak Distance (normalized) | 0.0097 (~1.7 cm) |
| Classification Accuracy | 1.0000 |

Peak distance in cm: value × coord_scale (179.55).

### In Progress

- [ ] **Phase 3**: Data production
  - [x] Submit test job and look at file size increase. Not very large.
  - [x] Submit for bnb_nu_pi0filter_corsika sample
  - [ ] Submit for bnb_nu_corsika sample
  - [ ] Submit for bnb_nu_pi0_corsika sample
  - [ ] Submit for bnb_nue_corsika sample
  - [ ] Submit for bnb_nu_chargedpiplus_corsika

### Remaining

- [ ] **Phase 3 (cont.)**: Production
  - [ ] Full-scale data prep via SLURM (all available ROOT files → HDF5 → shower_fragments)
  - [ ] Multi-GPU training run on full dataset
  - [ ] Consider validation check for new shower fragment data (past version used `lartpc_data_prep/validate_hdf5_files.py`)
- [ ] **Phase 4**: Evaluation
  - [ ] Analyze prediction quality on held-out data (origin localization error, inside/outside accuracy)
  - [ ] Ablation: with/without virtual grid
  - [ ] Hyperparameter tuning (grid spacing, min_visible_points threshold, num_slots for multi-shower)
- [ ] Reco pipeline: Test that `ShowerOriginDataset` can load reco HDF5 files end-to-end (the `type=-1` placeholder passes through without filtering issues)

---

## Known Issues and Fixes

### Cosmic Photon Labeling (Fixed)

The initial Python labeler only found photons via `mckeypoints` with `kptype==3` (shower start). Since `MCKeypointMaker` only creates keypoints for neutrino-origin particles, all cosmic photons were missed. Now handled by the C++ `ShowerFragmentOriginMaker` which scans `MCParticleGraph` directly.

### Pi0 Single-Photon Origin (Fixed in Python Data Loader)

When a pi0 decays inside the TPC into two photons but only one produces visible fragments, the C++ code assigns the pi0 decay vertex as the origin. However, with only one shower direction, the model cannot geometrically infer the pi0 vertex. The Python data loader (`shower_origin.py`) now overrides the origin to the trunk startpt for these cases. Validated on 8 events: 2 single-photon pi0 cases, both correctly overridden.

### Data Augmentation Inconsistency (Fixed)

`RandomScale` and `RandomFlipAxis` transforms only modified `coord`, leaving `origin_coord` and `start_coord` in the original frame. After augmentation, the Gaussian loss target was garbage — the model could not learn. Fixed by adding `extra_coord_keys=("origin_coord", "start_coord")` to both transforms.

### BiasedSphereCrop Missing Fragment (Fixed)

With `prob_random=0.25`, 25% of crops used a random center, often far from the target fragment. Fixed by setting `prob_random=0.0`.

### Fragment Sampling Bias (Fixed)

A cap on max fragments per event caused the same 4 fragments to be selected repeatedly. Fixed to sample from all available fragments.

---

## Open Questions

1. **No-object loss weight**: DETR-style penalty for unmatched slots. Current weight may need adjustment to balance false positive suppression vs recall.
2. **Minimum fragment points**: Currently 20 in data loader. May need tuning.
