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

# =============================================================================
# Training loop knobs (one-event smoke-test defaults; bump for real runs)
# =============================================================================
save_path = "exp/shower_clustering/run2_h200"
epoch = 100
eval_epoch = 100
batch_size = 96
batch_size_val  = 48
batch_size_test = 48
num_worker = 24          # 0 to keep stack traces sane during smoke test
num_worker_val = 12
evaluate = True          # flip to True (and uncomment evaluator hook below)
                         # for real training runs with val metrics
clip_grad = 5.0          # mild gradient clipping
enable_amp = True        # AMP roughly halves memory on P100 (xformers fp16).
                         # Watch for instability — see GradScalerMonitor hook
                         # below + the comment block at the end of this file.
amp_dtype = "bfloat16"   # P100 has no bf16 hardware; bf16 falls back to fp32.
enable_wandb = True      # turn on for real runs (ShowerClusteringEvaluator
                         # will then push val/* metrics to wandb)
wandb_project = "pointcept-shower-clustering"
empty_cache = True       # 16 GB P100 — free CUDA cache between fwd / bwd