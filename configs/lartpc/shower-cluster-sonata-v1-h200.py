"""
Config for the shower-clustering Mask2Former model. Phases 2–8 of the design
(see pointcept/docs/shower_clustering_design.md).

Builds:
  - ShowerClusteringDataset                   (Phase 2)
  - ShowerClusteringMask2Former               (Phase 7)
  - ShowerClusteringTrainer                   (Phase 8)
  - SonataCheckpointLoader hook for backbone  (loads cfg.weight)

Defaults are tuned for a Phase 8 one-event smoke test on a single P100:
batch_size=1, epoch=1, num_worker=0, evaluate=False. Bump these for real
training runs.

The visualizer at tools/visualize_shower_clustering.py reads this config so
the GT labels shown match what the model sees during training.
"""

_base_ = ["./shower-cluster-sonata-v1.py"]

# =============================================================================
# Use 'flash_attn' backend for backbone on H200
flash_backend = "flash_attn"
model["backbone"]["backbone"]["flash_backend"] = flash_backend