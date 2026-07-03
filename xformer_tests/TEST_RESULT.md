# Example Comparison between default and xformer backend

These are example results comparing the outputs of a pre-trained encoder using both the `default` and `xformer` flash attention backends.

The model with the `default` backend ran on a RTX 3080 (Ampere), while the model using the `xformer` backend ran on a P100 (Pascal).

In both setups, we use the script, `extract_encoder_vectors.py`, to output a file containing the embedding vectors output by the same encoder on the same event. We then used `compare_encoder_vectors.py` to produce the output below.

One issue is that the GridSample stage of the input processing creates slightly different input coordinates probably due to
randomness how the representative point within a grid voxel is chosen. Here is what Claude suggested when asked to investigate possible sources
of randomness in the inputs:

```
  Root Cause

  The differences are due to GridSample's random point selection within voxels. When multiple raw points fall into the same voxel grid cell, GridSample randomly picks one. Even with np.random.seed(42):

  - Different numpy versions may have different RNG implementations
  - Different platforms may have different floating-point hash collision behaviors
  - The order of operations in np.unique() can vary

  The Good News

  Your xformers backend is working correctly! Despite selecting slightly different points:
  - Cosine similarity: 0.9999 (nearly identical)
  - Per-point cosine min: 0.986 (still very good)
  - The encoder is producing consistent features regardless of backend

  For Truly Deterministic Comparison

  If you want exact coordinate matching for rigorous testing, you could modify the extraction script to use deterministic voxel selection (always pick the first point, or centroid):

  # In GridSample, change from random selection:
  idx_select = np.cumsum(np.insert(count, 0, 0)[0:-1]) + 0  # Always pick first point
```

Here is the result of the test:

```
python3 xformer_tests/compare_encoder_vectors.py --file1 encoder_out_flashattn_rtx3090_fixedseed.pt --file2 encoder_out_p100_fixseed.pt 
Loading file 1: encoder_out_flashattn_rtx3090_fixedseed.pt
Loading file 2: encoder_out_p100_fixseed.pt
WARNING: Coordinates differ (max diff: 1.500244e-01)
The encoder outputs may not be directly comparable if coordinates differ.

COORDINATE DIFFERENCE ANALYSIS:
----------------------------------------
Total points: 4761

Per-axis statistics (absolute difference):
  X-axis: mean=0.000000e+00, max=0.000000e+00, std=0.000000e+00
  Y-axis: mean=1.381829e-03, max=1.154709e-01, std=5.565131e-03
  Z-axis: mean=1.688986e-03, max=1.500244e-01, std=6.981442e-03

L2 distance statistics:
  Mean: 2.355318e-03
  Max:  1.732256e-01
  Std:  8.884011e-03

Points with L2 diff > 0.001:
  Count: 898 (18.86%)

Top 10 largest coordinate differences:
     Idx |         Coord1 (x,y,z)         |         Coord2 (x,y,z)         |    L2 Dist
  -------+--------------------------------+--------------------------------+-----------
    3234 | (   96.075,   -19.382,   788.050) | (   96.075,   -19.468,   788.200) |   0.173226
    4209 | (  175.790,    26.518,    47.950) | (  175.790,    26.604,    47.800) |   0.173206
    4716 | (  279.880,    60.379,   743.800) | (  279.880,    60.264,   743.800) |   0.115471
    4111 | (  159.869,    86.667,    31.750) | (  159.869,    86.723,    31.666) |   0.101230
    2634 | (   69.723,   104.316,   647.175) | (   69.723,   104.258,   647.250) |   0.094608
    4249 | (  187.264,    36.279,    35.781) | (  187.264,    36.188,    35.800) |   0.092142
    4152 | (  160.663,    87.005,    31.493) | (  160.663,    87.047,    31.414) |   0.089577
    2736 | (   72.325,    46.056,   743.806) | (   72.325,    46.013,   743.883) |   0.087798
    3078 | (   91.793,     0.580,   764.275) | (   91.793,     0.624,   764.200) |   0.086666
    2435 | (   61.323,   108.400,   740.425) | (   61.323,   108.357,   740.500) |   0.086615

Pattern analysis:
  X-axis contributes 0.0% of differences
  Y-axis contributes 45.0% of differences
  Z-axis contributes 55.0% of differences
  Differing points span: X=328.82, Y=232.25, Z=1034.12


Computing comparison metrics...

======================================================================
ENCODER VECTORS COMPARISON REPORT
======================================================================

FILE INFORMATION:
----------------------------------------
File 1:
  Path: pointceptdata_dlmerged_coriska_bnb_nue_fileno000228_entry000003.h5
  Backend: flash_attn
  Config: configs/lartpc/sonata_pretrain/archive/pretrain-sonata-v1m1-lartpc.py
  Checkpoint: epoch_80.pth

File 2:
  Path: /cluster/tufts/wongjiradlab//larbys/data/ub_on_tufts/hdf5/bnb_nue_corsika/000/002/pointceptdata_dlmerged_coriska_bnb_nue_fileno000228_entry000003.h5
  Backend: xformers
  Config: configs/lartpc/sonata_pretrain/archive/pretrain-sonata-v1m1-lartpc.py
  Checkpoint: epoch_80.pth

TENSOR INFORMATION:
----------------------------------------
Features 1 shape: (4761, 256)
Features 2 shape: (4761, 256)
Features 1 dtype: torch.float32
Features 2 dtype: torch.float32

COMPARISON METRICS:
----------------------------------------

Absolute Error Metrics:
  Mean Absolute Error (MAE):     2.082142e-03
  Root Mean Squared Error:       5.847588e-03
  Max Absolute Difference:       5.274086e-01

Relative Error Metrics:
  Mean Relative Error:           5.8015%
  Max Relative Error:            159415.6984%

Similarity Metrics:
  Cosine Similarity (overall):   0.99991339
  Cosine Similarity (per-point mean): 0.99990638
  Cosine Similarity (per-point min):  0.98635290
  Cosine Similarity (per-point std):  0.00057216
  Pearson Correlation:           0.99990631

Tolerance Check:
  Values within tolerance:       61.47%

Per-Channel MAE:
  Mean:                          2.082142e-03
  Max:                           5.119864e-03
  Min:                           5.288051e-04

Difference Distribution (absolute):
  50th percentile (median):      6.682370e-04
  90th percentile:               4.843757e-03
  99th percentile:               2.187230e-02

INTERPRETATION:
----------------------------------------
  [EXCELLENT] Outputs are nearly identical (cosine sim > 0.9999)
  Relative error is significant (5.8015%)

======================================================================
```


# Test that we have left the default `flash_attn` backend unmodified

Here we run the encoder with the default `flash_attn` backend twice: once with the dev branch with the xformer option, and once with unmodified code in the `lartpc` branch. The test is to ensure that the modifications to provide the `xformer` flash-attention backend has not altered running with the default backend. Both runs were on the same RTX 3080 card.


The tests seem to indicate that the default backend is unchanged:

```
python3 compare_encoder_vectors.py --file1 encoder_out_flashattn_rtx3090_fixedseed.pt --file2 encoder_out_rtx3080_lartpcbranch.pt 
Loading file 1: encoder_out_flashattn_rtx3090_fixedseed.pt
Loading file 2: encoder_out_rtx3080_lartpcbranch.pt
OK: Coordinates match

Computing comparison metrics...

======================================================================
ENCODER VECTORS COMPARISON REPORT
======================================================================

FILE INFORMATION:
----------------------------------------
File 1:
  Path: pointceptdata_dlmerged_coriska_bnb_nue_fileno000228_entry000003.h5
  Backend: flash_attn
  Config: configs/lartpc/sonata_pretrain/archive/pretrain-sonata-v1m1-lartpc.py
  Checkpoint: epoch_80.pth

File 2:
  Path: pointceptdata_dlmerged_coriska_bnb_nue_fileno000228_entry000003.h5
  Backend: flash_attn
  Config: configs/lartpc/sonata_pretrain/archive/pretrain-sonata-v1m1-lartpc.py
  Checkpoint: epoch_80.pth

TENSOR INFORMATION:
----------------------------------------
Features 1 shape: (4761, 256)
Features 2 shape: (4761, 256)
Features 1 dtype: torch.float32
Features 2 dtype: torch.float32

COMPARISON METRICS:
----------------------------------------

Absolute Error Metrics:
  Mean Absolute Error (MAE):     1.086295e-05
  Root Mean Squared Error:       2.269823e-05
  Max Absolute Difference:       1.442432e-03

Relative Error Metrics:
  Mean Relative Error:           0.0498%
  Max Relative Error:            4204.9106%

Similarity Metrics:
  Cosine Similarity (overall):   1.00000000
  Cosine Similarity (per-point mean): 1.00000000
  Cosine Similarity (per-point min):  0.99999977
  Cosine Similarity (per-point std):  0.00000001
  Pearson Correlation:           1.00000000

Tolerance Check:
  Values within tolerance:       100.00%

Per-Channel MAE:
  Mean:                          1.086295e-05
  Max:                           1.632520e-05
  Min:                           4.831166e-06

Difference Distribution (absolute):
  50th percentile (median):      4.440546e-06
  90th percentile:               2.720952e-05
  99th percentile:               9.000301e-05

INTERPRETATION:
----------------------------------------
  [EXCELLENT] Outputs are nearly identical (cosine sim > 0.9999)
  Relative error is negligible (0.0498%)

======================================================================
```