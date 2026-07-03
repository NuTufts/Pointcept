# LArFormer Analysis

Performance analysis of the LArFormer cascade stages and downstream physics.

## Active

- `slicer_eval/` — Stage-2 event-slicer val/test evaluation: 5-stage pipeline
  (manifests → flashinfo → inference+analysis → aggregation → plots) with flash-match
  χ² ranking and γ tuning. The flash-matching library in `slicer_eval/lib/`
  (`flash_predict.py`, `flash_chi2.py`) is slated to become the shared
  `lartpc/flashmatch/` package.
- `particle_eval/` — Stage-3 particle-segmenter val/test evaluation (mask IoU,
  class accuracy, origin error; per-pair parquet records).
- `physics/` — physics-level studies:
  - `single_photon/` — single-photon (1γ0X) selection studies on official uboone sim;
    forerunner of the unified reco + flash-matching inference (next work stage —
    see `docs/Reorganization_Plan.md` §1b).
  - `repeatability_tests/` — cross-GPU / determinism checks of cascade inference.

## archive/ (completed one-off studies)

| Directory | What it answered |
|---|---|
| `semseg_analysis/` | LoRA semantic-segmentation classifier evaluation; includes the resolved mIoU-0.03 debug brief (pixval preprocessing drift). |
| `deghost_analysis/` | Ghost-point-removal performance evaluation. |
| `shower_origin_reco_scripts/` | Shower-origin reconstruction validation (exploratory project). |
| `extbnb_larmatch/` | ExtBNB + LArMatch H5 conversion/validation for Sonata pre-training data. |
| `keypoint_v1/` | Keypoint attempt 1: heads on the frozen Stage-3 particle masker. Plateaued short of the 1 cm target; superseded by keypoint v2 (`lartpc_data_prep/larformer_keypoint_v2`, moving to `lartpc/larformer_reco/`). |
