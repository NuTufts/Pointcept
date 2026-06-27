# `reco/` — score-field keypoint reconstruction (v1)

PyTorch port of larflow's `larflow::reco::KeypointReco` greedy peak-finder,
operating on the attempt-2 LArFormer keypoint cascade's **dense score maps**.
Design + rationale: [`../keypoint_reco_spec.md`](../keypoint_reco_spec.md).

**v1 scope:** score-field only (no vote/offset head yet — see spec §4.1 Method 3
and the open items). Produces keypoint *positions + scores*; the `object` head is
type-less so its keypoints are generic (typing is future work).

## What it does

For each dense head (nu-vertex @ full spacepoint resolution; object @ voxelized
`ptv3_dec2`), greedily: find the max-score point → isolate neighbors within
`radius_cm` (10) → fit a fixed-σ (3 cm) 3D Gaussian mean → subtract that Gaussian
from the residual field → repeat until the running max drops below
`score_thresh` (0.67). The subtraction is the non-maximum suppression.

## Files

| File | Role |
|---|---|
| `gaussian_fit.py` | Fitters: `fit_gaussian_nls` (Method 1, default), `fit_gaussian_loglinear` (Method 2), `fit_gaussian_centroid` (fallback), `caruana_full` (parity). |
| `keypoint_reco.py` | `KeypointRecoTorch` + `KeypointRecoParams` + `Keypoint`; the greedy peel loop, fit dispatch w/ fallback, off-support guards. |
| `io.py` | `load_score_maps` (read `--save-score-maps` H5) + `write_reco_h5`. |
| `run_keypoint_reco.py` | CLI over a dir of score-map H5s; optional `--eval` vs GT. |
| `test_keypoint_reco.py` | Synthetic tests incl. truncated / one-sided / off-support peaks. |

## Usage (inside the pointcept container)

```bash
# 1. produce score maps:
python tools/run_larformer_keypoint2_cascade_inference.py \
    --config configs/lartpc/larformer-keypoint2-fullcascade.py \
    --input-list <list> --output-dir kp2_out --save-score-maps

# 2. reconstruct keypoints from them:
python -m lartpc_data_prep.larformer_keypoint_v2.reco.run_keypoint_reco \
    kp2_out --output-dir kp_reco_out --eval

# tests:
python -m lartpc_data_prep.larformer_keypoint_v2.reco.test_keypoint_reco
```

Dependencies: `numpy`, `torch`, `h5py`, and `scipy` (KDTree radius queries; a
`torch.cdist` fallback runs without it). No larflow / ROOT dependency.
