```
python3 xformer_tests/compare_encoder_vectors.py --file1 encoder_out_flashattn_rtx3090_fixedseed.pt --file2 encoder_out_p100_fixseed.pt 
Loading file 1: encoder_out_flashattn_rtx3090_fixedseed.pt
/home/twongjirad/working/larbys/gen2/container_u22/dev_xformers/pointcept_xformers/xformer_tests/compare_encoder_vectors.py:375: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
  data1 = torch.load(args.file1, map_location="cpu")
Loading file 2: encoder_out_p100_fixseed.pt
/home/twongjirad/working/larbys/gen2/container_u22/dev_xformers/pointcept_xformers/xformer_tests/compare_encoder_vectors.py:378: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
  data2 = torch.load(args.file2, map_location="cpu")
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
  Config: configs/lartpc/pretrain-sonata-v1m1-lartpc.py
  Checkpoint: epoch_80.pth

File 2:
  Path: /cluster/tufts/wongjiradlab//larbys/data/ub_on_tufts/hdf5/bnb_nue_corsika/000/002/pointceptdata_dlmerged_coriska_bnb_nue_fileno000228_entry000003.h5
  Backend: xformers
  Config: configs/lartpc/pretrain-sonata-v1m1-lartpc.py
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