# LArTPC HDF5 Data Format Documentation

> **Status: REFERENCE** — HDF5 schema for spacepoint training data.

This document describes the structure of HDF5 files produced by `SimChTripletLabelMaker` from the `ubdl/larflow` package. These files contain 3D space points (triplets) and 2D wire plane images for training point cloud neural networks on Liquid Argon Time Projection Chamber (LArTPC) data.

## File Structure Overview

Each HDF5 file contains one or more entries, organized as:

```
file.h5
└── entry_0/
    ├── triplet_data/      # 3D space points with labels
    ├── image_data/        # 2D wire plane images (sparse)
    ├── mckeypoints/       # Monte Carlo truth keypoints
    ├── triplet_truth/     # Truth-matched triplet subset
    ├── mc_particle_tree/  # MC particle hierarchy from Geant4
    └── shower_fragments/  # DBSCAN-clustered EM shower fragments
```

For files with multiple entries, additional groups `entry_1/`, `entry_2/`, etc. are present.

---

## Triplet Data (`/entry_N/triplet_data/`)

The triplet data contains 3D reconstructed space points formed by matching charge deposits across the three wire planes. Each point has associated features and labels.

| Dataset | Shape | Dtype | Description |
|---------|-------|-------|-------------|
| `pos` | (N, 3) | float32 | 3D position in detector coordinates (x, y, z) in cm |
| `pixval` | (N, 3) | float32 | Pixel/ADC values from each wire plane (U, V, Y) |
| `pid` | (N,) | int32 | PDG particle ID code (see [PDG Codes](#pdg-particle-id-codes)) |
| `ssnet_label` | (N,) | int32 | Semantic segmentation label (see [SSNET Labels](#ssnet-class-labels)) |
| `origin` | (N,) | int32 | Origin type: 0=unknown, 1=neutrino, 2=cosmic |
| `hasmatch` | (N,) | int32 | Truth match flag: 1=true point, 0=ghost point |
| `trackid` | (N,) | int32 | Geant4 track ID for truth-matched points |
| `uwire` | (N,) | int32 | U plane wire number |
| `vwire` | (N,) | int32 | V plane wire number |
| `ywire` | (N,) | int32 | Y (collection) plane wire number |
| `tick` | (N,) | int32 | Time tick value |
| `edep` | (N, 3) | float32 | Energy deposition per plane (MeV) |
| `aid` | (N,) | int32 | Ancestor track ID |
| `kpscores` | (N, 6) | float32 | Keypoint proximity scores |
| `ssnet_boundary` | (N,) | int32 | Boundary/endpoint flag |

### Coordinate Systems

- **3D Position (`pos`)**: Detector coordinates in centimeters
  - X: Drift direction (0 to ~256 cm for MicroBooNE)
  - Y: Vertical (-117 to 117 cm)
  - Z: Beam direction (0 to 1036 cm)

- **Wire Coordinates**: Integer wire numbers for each plane
  - U plane: 0-2399 (induction)
  - V plane: 0-2399 (induction)
  - Y plane: 0-3455 (collection)

- **Time Tick**: Raw TPC readout time tick (typically 0-9600)

### Converting Tick to Image Row

To convert from time tick to the row coordinate used in wire plane images:

```python
row = (tick - 2400) / 6.0
```

This accounts for the trigger offset and downsampling factor.

---

## Wire Plane Images (`/entry_N/image_data/`)

Wire plane images are stored in sparse format for efficiency. Each plane has its own subgroup.

### Structure

```
image_data/
├── plane0/          # U plane
│   ├── coord        # Pixel coordinates
│   ├── feat         # Pixel values (ADC)
│   ├── dims         # Image dimensions
│   ├── origin       # Image origin offset
│   └── pixsize      # Pixel size
├── plane1/          # V plane
│   └── ...
├── plane2/          # Y plane
│   └── ...
└── triplet_imgpix_index  # Mapping from triplets to image pixels
```

### Plane Data Fields

| Dataset | Shape | Dtype | Description |
|---------|-------|-------|-------------|
| `coord` | (M, 2) | int32 | Sparse pixel coordinates [wire, row] |
| `feat` | (M,) | float32 | ADC/pixel values |
| `dims` | (2,) | int32 | Image dimensions [n_wires, n_rows] |
| `origin` | (2,) | float32 | Image origin offset |
| `pixsize` | (2,) | float32 | Pixel size in each dimension |

### Typical Dimensions

For MicroBooNE data:
- **dims**: [3456, 1008] (wires x rows)
- Wire range: 0-3455
- Row range: 0-1007

### Triplet to Image Mapping

The `triplet_imgpix_index` dataset maps each 3D triplet to its corresponding 2D pixel locations:

| Dataset | Shape | Dtype | Description |
|---------|-------|-------|-------------|
| `triplet_imgpix_index` | (N, 4) | int32 | [triplet_idx, u_pix, v_pix, y_pix] |

---

## MC Keypoints (`/entry_N/mckeypoints/`)

Monte Carlo truth keypoints mark important physics locations in the event.

| Dataset | Shape | Dtype | Description |
|---------|-------|-------|-------------|
| `pos` | (K, 3) | float32 | 3D position (x, y, z) in cm |
| `imgcoord` | (K, 4) | float32 | Image coordinates |
| `kptype` | (K,) | int32 | Keypoint type (see below) |
| `pid` | (K,) | int32 | PDG particle ID |
| `trackid` | (K,) | int32 | Geant4 track ID |
| `startpos` | (K,3) | float32 | 3D creation position (x, y, z) in cm. Only different than pos when particle is photon (pid=22) |

### Keypoint Types

| Value | Name | Description |
|-------|------|-------------|
| 0 | Nu Vertex | Neutrino interaction vertex |
| 1 | Track Start | Beginning of a track |
| 2 | Track End | End of a track |
| 3 | Shower | Shower start point |
| 4 | Michel | Michel electron candidate |
| 5 | Delta | Delta ray origin |

---

## Triplet Truth (`/entry_N/triplet_truth/`)

A subset of triplets with guaranteed truth matching, useful for training on clean samples.

| Dataset | Shape | Dtype | Description |
|---------|-------|-------|-------------|
| `pos` | (T, 3) | float32 | True 3D position |
| `pos_reco` | (T, 3) | float32 | Reconstructed 3D position |
| `pid` | (T,) | int32 | PDG particle ID |
| `origin` | (T,) | int32 | Origin type |
| `trackid` | (T,) | int32 | Geant4 track ID |
| `aid` | (T,) | int32 | Ancestor track ID |
| `edep` | (T, 3) | float32 | Energy deposition per plane |
| `uwire` | (T,) | int32 | U plane wire |
| `vwire` | (T,) | int32 | V plane wire |
| `ywire` | (T,) | int32 | Y plane wire |
| `tick` | (T,) | int32 | Time tick |
| `row` | (T,) | int32 | Image row |

---

## MC Particle Tree (`/entry_N/mc_particle_tree/`)

The Geant4 Monte Carlo particle hierarchy, stored from the `MCParticleGraph`. Contains per-particle truth information and parent-daughter relationships.

### Per-Particle Datasets

| Dataset | Shape | Dtype | Description |
|---------|-------|-------|-------------|
| `trackid` | (P,) | int32 | Geant4 track ID |
| `pid` | (P,) | int32 | PDG particle ID code |
| `parent_trackid` | (P,) | int32 | Parent particle's track ID (-1 if none) |
| `origin` | (P,) | int32 | Origin: 0=unknown, 1=neutrino, 2=cosmic |
| `energy_mev` | (P,) | float32 | Particle kinetic energy (MeV) |
| `process_code` | (P,) | int32 | Geant4 creation process (see below) |
| `start_pos` | (P, 3) | float32 | True start position (x, y, z) in cm |
| `start_pos_sce` | (P, 3) | float32 | SCE-corrected start position (x, y, z) in cm |

Where P = number of particles in the event.

### Daughter Relationship Datasets

| Dataset | Shape | Dtype | Description |
|---------|-------|-------|-------------|
| `num_daughters` | (P,) | int32 | Number of daughter particles per node |
| `daughter_start_indices` | (P,) | int32 | Start index into `daughter_trackids` for each node |
| `daughter_trackids` | (D,) | int32 | Flattened array of daughter track IDs |

Where D = total daughter entries across all particles. To get the daughters of particle `i`:

```python
start = daughter_start_indices[i]
count = num_daughters[i]
daughters = daughter_trackids[start:start+count]
```

### Neutrino Vertex Dataset

| Dataset | Shape | Dtype | Description |
|---------|-------|-------|-------------|
| `nu_vertices` | (V, 3) | float32 | Neutrino interaction vertex positions (x, y, z) in cm |

Where V = number of neutrino vertices in the event (typically 0 or 1).

### Process Codes

| Code | Geant4 Process | Description |
|------|----------------|-------------|
| 0 | primary | Primary particle from event generator |
| 1 | Decay | Particle decay |
| 2 | compt | Compton scattering |
| 3 | conv | Pair production (photon conversion) |
| 4 | phot | Photoelectric effect |
| 5 | eBrem | Electron bremsstrahlung |
| 6 | eIoni | Electron ionization |
| 7 | muIoni | Muon ionization |
| 8 | muBrems | Muon bremsstrahlung |
| 9 | muPairProd | Muon pair production |
| 10 | hIoni | Hadron ionization |
| 11 | hadElastic | Hadronic elastic scattering |
| 12 | neutronInelastic | Neutron inelastic scattering |
| 13 | protonInelastic | Proton inelastic scattering |
| 14 | pi+Inelastic | Pi+ inelastic scattering |
| 15 | pi-Inelastic | Pi- inelastic scattering |
| 16 | muMinusCaptureAtRest | Muon capture at rest |
| 17 | nCapture | Neutron capture |
| 18 | annihil | Positron annihilation |
| 19 | CoulombScat | Coulomb scattering |
| 20 | photonNuclear | Photonuclear interaction |
| -1 | null | No process information |
| 99 | other | Other/unknown process |

---

## Shower Fragments (`/entry_N/shower_fragments/`)

DBSCAN-clustered shower fragments produced by `ShowerFragmentOriginMaker`. Each EM shower (photon or electron/positron) is split into multiple fragments based on spatial clustering. Each fragment has its own origin point (where the particle was created) and start point (most upstream point in the cluster along the shower axis).

### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `num_fragments` | int | Total number of fragment clusters (F) |

### Datasets

| Dataset | Shape | Dtype | Description |
|---------|-------|-------|-------------|
| `trackid` | (F,) | int32 | Geant4 track ID per fragment |
| `pid` | (F,) | int32 | PDG code per fragment (22=photon, 11=electron, -11=positron) |
| `istrunk` | (F,) | int32 | 1=trunk (closest to shower start), 2=secondary fragment |
| `type` | (F,) | int32 | Origin type (see below) |
| `startpt` | (F, 3) | float32 | Most upstream point per fragment (SCE-corrected, cm) |
| `originpt` | (F, 3) | float32 | Particle creation point in apparent coordinates (cm). For outside-TPC showers, this is set equal to startpt |
| `pret0shiftedoriginpt` | (F, 4) | float32 | True MC origin (x, y, z, t) without t0 shift. Used to determine inside/outside TPC classification |
| `pointindices_flat` | (T,) | int64 | Concatenated indices into triplet_data arrays for all fragments |
| `pointindices_counts` | (F,) | int32 | Number of points per fragment (used to split pointindices_flat) |

Where F = number of fragments, T = total points across all fragments.

### Origin Type (`type`)

| Value | Name | Description |
|-------|------|-------------|
| 0 | nu-inside | Neutrino-origin particle, origin inside TPC |
| 1 | outside | Particle origin outside TPC active volume |
| 2 | cosmic-inside | Cosmic-origin particle, origin inside TPC |

The inside/outside determination uses the `pret0shiftedoriginpt` position (true MC coordinates without t0 shift) against the TPC active volume bounds:
- X: [0.0, 255.6] cm
- Y: [-116.5, 116.5] cm
- Z: [0.5, 1035.5] cm

### Reconstructing Per-Fragment Masks

The `pointindices_flat` and `pointindices_counts` arrays store variable-length index lists in a flat format:

```python
import numpy as np

sf = f['entry_0/shower_fragments']
num_frags = int(sf.attrs['num_fragments'])
flat_indices = sf['pointindices_flat'][:]
index_counts = sf['pointindices_counts'][:]
n_points = f['entry_0/triplet_data/pos'].shape[0]

offset = 0
for i in range(num_frags):
    count = int(index_counts[i])
    indices = flat_indices[offset:offset+count]
    offset += count

    mask = np.zeros(n_points, dtype=bool)
    mask[indices[indices < n_points]] = True
    # mask now selects triplet_data points belonging to fragment i
```

### Notes

- Multiple fragments can share the same `trackid` — they are DBSCAN clusters from the same shower particle.
- The trunk fragment (`istrunk=1`) is the cluster whose start point is closest to the original shower start.
- `originpt` is in SCE-corrected "apparent" coordinates, suitable as a prediction target for models.
- For showers originating outside the TPC, `originpt` is set equal to `startpt` since the true origin is not a physically meaningful prediction target in detector coordinates.
- `pret0shiftedoriginpt` is the raw Geant4 truth position without any time-zero shift applied; it is used only for classification, not as a prediction target.

---

## Label Definitions

### PDG Particle ID Codes

Standard PDG Monte Carlo particle numbering:

| PDG Code | Particle |
|----------|----------|
| 11 | electron (e-) |
| -11 | positron (e+) |
| 13 | muon (mu-) |
| -13 | antimuon (mu+) |
| 22 | photon (gamma) |
| 111 | neutral pion (pi0) |
| 211 | positive pion (pi+) |
| -211 | negative pion (pi-) |
| 2212 | proton |
| 2112 | neutron |
| 321 | positive kaon (K+) |
| -321 | negative kaon (K-) |
| 0 | ghost/unmatched |

### SSNET Class Labels

Semantic segmentation network labels from `SimChTripletLabelMaker`:

| Value | Class | Description |
|-------|-------|-------------|
| 0 | background | Ghost/unmatched points |
| 1 | electron | Electron showers |
| 2 | photon | Photon showers |
| 3 | muon | Muon tracks |
| 4 | proton | Proton tracks |
| 5 | pion | Charged pion tracks |
| 6 | michel | Michel electrons |
| 7 | delta | Delta rays |
| 8 | led | Low energy deposits |
| 9 | other | Other particles |

### Origin Labels

| Value | Origin | Description |
|-------|--------|-------------|
| 0 | unknown | No truth information |
| 1 | neutrino | From neutrino interaction |
| 2 | cosmic | Cosmic ray origin |

---

## Python Usage Examples

### Reading Basic Data

```python
import h5py
import numpy as np

with h5py.File('data.h5', 'r') as f:
    # Get 3D positions and labels
    pos = f['/entry_0/triplet_data/pos'][:]
    ssnet_labels = f['/entry_0/triplet_data/ssnet_label'][:]
    origin = f['/entry_0/triplet_data/origin'][:]

    # Filter to true (non-ghost) points
    hasmatch = f['/entry_0/triplet_data/hasmatch'][:]
    true_mask = hasmatch == 1
    true_pos = pos[true_mask]
```

### Reading Wire Plane Images

```python
with h5py.File('data.h5', 'r') as f:
    # Read Y plane (collection) sparse image
    plane = f['/entry_0/image_data/plane2']
    coords = plane['coord'][:]  # (M, 2) - [wire, row]
    feats = plane['feat'][:]    # (M,) - ADC values
    dims = plane['dims'][:]     # [n_wires, n_rows]

    # Convert to dense image if needed
    dense_img = np.zeros((dims[1], dims[0]), dtype=np.float32)
    dense_img[coords[:, 1], coords[:, 0]] = feats
```

### Getting 2D Pixel for a 3D Point

```python
with h5py.File('data.h5', 'r') as f:
    # Get wire coordinates for point i
    uwire = f['/entry_0/triplet_data/uwire'][i]
    vwire = f['/entry_0/triplet_data/vwire'][i]
    ywire = f['/entry_0/triplet_data/ywire'][i]
    tick = f['/entry_0/triplet_data/tick'][i]

    # Convert tick to row
    row = (tick - 2400) / 6.0

    # Now (uwire, row), (vwire, row), (ywire, row) are the
    # 2D coordinates in planes 0, 1, 2 respectively
```

### Filtering by Particle Type

```python
with h5py.File('data.h5', 'r') as f:
    pos = f['/entry_0/triplet_data/pos'][:]
    ssnet = f['/entry_0/triplet_data/ssnet_label'][:]

    # Get only muon points
    muon_mask = ssnet == 3
    muon_pos = pos[muon_mask]

    # Get only neutrino-origin points
    origin = f['/entry_0/triplet_data/origin'][:]
    nu_mask = origin == 1
    nu_pos = pos[nu_mask]
```

---

## Visualization Tool

A visualization tool is available at `tools/viz/visualize_lartpc_h5data.py`:

```bash
python tools/viz/visualize_lartpc_h5data.py /path/to/file.h5 --port 8050
```

Features:
- 3D point cloud visualization with multiple coloring options
- 2D wire plane image display
- Click-to-zoom: click a 3D point to see 50x50 heatmap around corresponding 2D pixels
- Toggle ghost point visibility
- Keypoint display

---

## Data Production

These HDF5 files are produced by `SimChTripletLabelMaker` from the `ubdl/larflow` package:
- Source: `ubdl/larflow/larflow/PrepFlowMatchData/SimChTripletLabelMaker.cxx`
- The `save_entry_sparseimg()` method writes the sparse image data
- The `save_entry()` method writes the triplet, truth, and shower fragment data
- Shower fragments are produced by `ShowerFragmentOriginMaker` (`ShowerFragmentOriginMaker.cxx`) using DBSCAN clustering and exported via its `save_entry_to_hdf()` method

---

## Related Documentation

- `docs/reference/LArTPC_Dataset_Guide.md` - Guide to using LArTPCDataset in Pointcept
- `pointcept/datasets/lartpc.py` - Dataset class implementation
