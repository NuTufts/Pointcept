# MinkowskiEngine Migration Guide

> **Status: REFERENCE** — Sparse-convolution backend migration notes.

This document describes what would be needed to replace spconv with MinkowskiEngine for CPU inference support.

## Background

The spconv library (used for sparse convolutions in PT-v3 and other models) is compiled as GPU-only. Its C++ code contains hard assertions that reject CPU tensors:

```
!features.is_cpu() assert failed. bias and act don't support cpu.
```

MinkowskiEngine is an alternative sparse convolution library that supports CPU-only builds.

## Does MinkowskiEngine Support CPU?

**Yes**. From the [MinkowskiEngine documentation](https://github.com/NVIDIA/MinkowskiEngine):

> "The Minkowski Engine supports CPU only build on other platforms that do not have NVidia GPUs."

This is a key advantage over spconv for CPU inference use cases.

## Current spconv Usage in Pointcept PT-v3

The model uses spconv in these locations:

| File | Line | Usage |
|------|------|-------|
| `pointcept/models/utils/structure.py` | 139 | `spconv.SparseConvTensor` - creates sparse tensor from points |
| `pointcept/models/point_transformer_v3/point_transformer_v3m2_sonata.py` | 356 | `spconv.SubMConv3d` - CPE (Conditional Position Encoding) layer |
| `pointcept/models/modules.py` | 84, 106 | `spconv.modules.is_spconv_module()` and `spconv.SparseConvTensor` type checks |

## Layer Equivalence Table

| spconv | MinkowskiEngine | Notes |
|--------|-----------------|-------|
| `spconv.SparseConvTensor` | `ME.SparseTensor` | Different constructor arguments |
| `spconv.SubMConv3d` | `ME.MinkowskiConvolution(stride=1)` | stride=1 means submanifold sparse conv |
| `spconv.SparseConv3d` | `ME.MinkowskiConvolution(stride>1)` | Regular sparse conv with downsampling |
| `spconv.SparseInverseConv3d` | `ME.MinkowskiConvolutionTranspose` | Upsampling/transpose conv |
| `spconv.SparseMaxPool3d` | `ME.MinkowskiMaxPooling` | Sparse max pooling |

## Required Code Changes

### 1. Sparse Tensor Creation (`structure.py`)

**Current spconv code:**
```python
sparse_conv_feat = spconv.SparseConvTensor(
    features=self.feat,
    indices=torch.cat(
        [self.batch.unsqueeze(-1).int(), self.grid_coord.int()], dim=1
    ).contiguous(),
    spatial_shape=sparse_shape,
    batch_size=self.batch[-1].tolist() + 1,
)
```

**MinkowskiEngine equivalent:**
```python
import MinkowskiEngine as ME

sparse_conv_feat = ME.SparseTensor(
    features=self.feat,
    coordinates=torch.cat(
        [self.batch.unsqueeze(-1).int(), self.grid_coord.int()], dim=1
    ).contiguous(),
    device=self.feat.device,
)
```

Key differences:
- `indices` -> `coordinates`
- No need to specify `spatial_shape` or `batch_size`
- Need to specify `device`

### 2. SubMConv3d Replacement (`point_transformer_v3m2_sonata.py`)

**Current spconv code:**
```python
spconv.SubMConv3d(
    channels,
    channels,
    kernel_size=3,
    bias=True,
    indice_key=cpe_indice_key,
)
```

**MinkowskiEngine equivalent:**
```python
ME.MinkowskiConvolution(
    in_channels=channels,
    out_channels=channels,
    kernel_size=3,
    stride=1,  # stride=1 means submanifold sparse conv
    bias=True,
    dimension=3,
)
```

Key differences:
- `SubMConv3d` -> `MinkowskiConvolution` with `stride=1`
- No `indice_key` parameter (MinkowskiEngine handles caching differently)
- Must specify `dimension=3` for 3D convolutions

### 3. Module Type Checking (`modules.py`)

**Current spconv code:**
```python
elif spconv.modules.is_spconv_module(module):
    if isinstance(input, Point):
        input.sparse_conv_feat = module(input.sparse_conv_feat)
        input.feat = input.sparse_conv_feat.features
    else:
        input = module(input)
# ...
elif isinstance(input, spconv.SparseConvTensor):
    if input.indices.shape[0] != 0:
        input = input.replace_feature(module(input.features))
```

**MinkowskiEngine equivalent:**
```python
elif isinstance(module, ME.MinkowskiModuleBase):
    if isinstance(input, Point):
        input.sparse_conv_feat = module(input.sparse_conv_feat)
        input.feat = input.sparse_conv_feat.F  # .F instead of .features
    else:
        input = module(input)
# ...
elif isinstance(input, ME.SparseTensor):
    if input.C.shape[0] != 0:  # .C instead of .indices
        # MinkowskiEngine applies module directly, no replace_feature needed
        input = ME.SparseTensor(
            features=module(input.F),
            coordinate_map_key=input.coordinate_map_key,
            coordinate_manager=input.coordinate_manager,
        )
```

### 4. Feature/Coordinate Access

| Operation | spconv | MinkowskiEngine |
|-----------|--------|-----------------|
| Get features | `.features` | `.F` or `.features` |
| Get coordinates | `.indices` | `.C` or `.coordinates` |
| Replace features | `.replace_feature(new_feat)` | Create new SparseTensor |

## Recommended Implementation Approach

### Option A: Backend Abstraction Layer

Create a `sparse_backend.py` that provides a unified interface:

```python
# pointcept/models/utils/sparse_backend.py

import torch

# Try to import both backends
try:
    import spconv.pytorch as spconv
    SPCONV_AVAILABLE = True
except ImportError:
    SPCONV_AVAILABLE = False

try:
    import MinkowskiEngine as ME
    MINKOWSKI_AVAILABLE = True
except ImportError:
    MINKOWSKI_AVAILABLE = False


def get_backend(prefer_cpu=False):
    """Get the appropriate sparse convolution backend."""
    if prefer_cpu and MINKOWSKI_AVAILABLE:
        return "minkowski"
    elif SPCONV_AVAILABLE:
        return "spconv"
    elif MINKOWSKI_AVAILABLE:
        return "minkowski"
    else:
        raise ImportError("No sparse convolution backend available")


def create_sparse_tensor(features, coordinates, spatial_shape=None, batch_size=None,
                         backend="spconv", device=None):
    """Create a sparse tensor using the specified backend."""
    if backend == "spconv":
        return spconv.SparseConvTensor(
            features=features,
            indices=coordinates,
            spatial_shape=spatial_shape,
            batch_size=batch_size,
        )
    elif backend == "minkowski":
        return ME.SparseTensor(
            features=features,
            coordinates=coordinates,
            device=device or features.device,
        )


def create_subm_conv3d(in_channels, out_channels, kernel_size, bias=True,
                       indice_key=None, backend="spconv"):
    """Create a submanifold sparse 3D convolution."""
    if backend == "spconv":
        return spconv.SubMConv3d(
            in_channels, out_channels, kernel_size,
            bias=bias, indice_key=indice_key,
        )
    elif backend == "minkowski":
        return ME.MinkowskiConvolution(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1,
            bias=bias,
            dimension=3,
        )
```

### Option B: Separate Model Variants

Create MinkowskiEngine-specific versions of the model files for CPU inference:
- `point_transformer_v3m2_sonata_minkowski.py`

This is more code duplication but cleaner separation.

## Weight Conversion

Pretrained spconv weights would need conversion for MinkowskiEngine:

1. **Weight layout**: spconv 2.x uses KRSC layout, MinkowskiEngine may differ
2. **State dict keys**: Layer names will differ if using different module classes

A conversion script would be needed:

```python
def convert_spconv_to_minkowski(spconv_state_dict):
    """Convert spconv checkpoint to MinkowskiEngine format."""
    new_state_dict = {}
    for key, value in spconv_state_dict.items():
        # Map layer names
        new_key = key.replace('.cpe.0.', '.cpe.0.')  # Example mapping

        # Transpose weights if needed (check layouts)
        if 'weight' in key and len(value.shape) == 5:
            # spconv: (K, R, S, C_in, C_out) or similar
            # ME: may need transposition
            pass

        new_state_dict[new_key] = value
    return new_state_dict
```

## Caveats and Considerations

### 1. No `indice_key` in MinkowskiEngine
spconv uses `indice_key` to cache and reuse sparse indices between layers for efficiency. MinkowskiEngine handles this differently through its coordinate manager. This may affect performance.

### 2. Weight Format Differences
Pretrained spconv weights may need conversion. Test thoroughly after conversion.

### 3. GPU Performance
On GPU, spconv is generally faster than MinkowskiEngine, especially with Tensor Core optimizations. Consider using spconv for GPU and MinkowskiEngine only for CPU.

### 4. Coordinate Manager
MinkowskiEngine uses a `CoordinateManager` to track sparse coordinates across the network. This is different from spconv's approach and may require additional handling.

## Estimated Effort

| Component | Lines to Change | Complexity |
|-----------|-----------------|------------|
| `structure.py` | ~15 lines | Low |
| `modules.py` | ~10 lines | Medium |
| `point_transformer_v3m2_sonata.py` | ~5 lines | Low |
| Backend abstraction (optional) | ~100 lines | Medium |
| Weight conversion script | ~50 lines | Medium |
| Testing and validation | - | High |

## References

- [MinkowskiEngine GitHub](https://github.com/NVIDIA/MinkowskiEngine)
- [MinkowskiEngine Convolution Documentation](https://nvidia.github.io/MinkowskiEngine/convolution.html)
- [MinkowskiEngine SparseTensor Documentation](https://nvidia.github.io/MinkowskiEngine/sparse_tensor.html)
- [spconv GitHub](https://github.com/traveller59/spconv)
- [spconv Usage Guide](https://github.com/traveller59/spconv/blob/master/docs/USAGE.md)

---

## Example Prompt for Implementation

Use the following prompt to start the MinkowskiEngine migration project:

```
I want to implement CPU inference support for the PT-v3 model by replacing spconv
with MinkowskiEngine. Please read docs/reference/MinkowskiEngine_Migration_Guide.md for the
research and planning that was done previously.

The goal is to create a backend abstraction layer that allows switching between
spconv (for GPU) and MinkowskiEngine (for CPU) at runtime.

Please implement:
1. A sparse_backend.py module in pointcept/models/utils/ that provides unified
   interfaces for both backends
2. Modifications to structure.py to use the backend abstraction for sparse tensor
   creation
3. Modifications to modules.py to handle both spconv and MinkowskiEngine module types
4. Modifications to point_transformer_v3m2_sonata.py to use the backend abstraction
   for the CPE sparse convolutions
5. A weight conversion utility if needed for loading spconv-trained weights into
   MinkowskiEngine layers

The backend selection should be automatic based on the device (CPU -> MinkowskiEngine,
GPU -> spconv) but also allow manual override.

Start by reading the migration guide, then propose a detailed implementation plan
before writing code.
```
