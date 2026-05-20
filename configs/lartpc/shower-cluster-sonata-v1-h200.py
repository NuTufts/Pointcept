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
# Override only the fields that differ from the base. Pointcept's Config
# does deep-merge of nested dicts (see _merge_a_into_b in
# pointcept/utils/config.py), so we redefine just the leaf path. We
# CANNOT do `model["backbone"]["backbone"]["flash_backend"] = ...` here:
# at file-import time the override module is executed standalone before
# the base is merged in, so `model` is not yet defined in this namespace
# (NameError).
flash_backend = "flash_attn"
model = dict(
    backbone=dict(
        backbone=dict(
            flash_backend=flash_backend,
        ),
    ),
)

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
enable_amp = False       # AMP halves memory; H200 has native bf16 (no
                         # GradScaler overflow loop like fp16 on P100).
amp_dtype = "bfloat16"   # H200 has native bf16 hardware.
enable_wandb = True      # turn on for real runs (ShowerClusteringEvaluator
                         # will then push val/* metrics to wandb)
wandb_project = "pointcept-shower-clustering"
empty_cache = True       # 16 GB P100 — free CUDA cache between fwd / bwd

# =============================================================================
# Optimizer / scheduler
# =============================================================================
# Flat LR with two decay triggers (see pointcept/utils/scheduler.py
# FlatWithDecayLR docstring):
#   - epoch trigger:   cut LR by `gamma` every `step_period_epochs`.
#   - plateau trigger: cut LR by `gamma` after `patience_epochs` of no
#                      val/loss improvement > `min_delta`, then suppress
#                      further plateau triggers for `cooldown_epochs`.
#   - mode="both":     either trigger fires (whichever first).
# Driven by the LREpochScheduler hook (must sit between the evaluator
# and CheckpointSaver in `hooks=` below). When resuming a run trained
# with this scheduler, set `extend_scheduler=False` on the checkpoint
# loader — the OneCycleLR-aware extend path rewrites max_lr/initial_lr
# and would corrupt FlatWithDecayLR state.
base_lr = 1.0e-4
optimizer = dict(type="AdamW", lr=base_lr, weight_decay=0.01)
scheduler = dict(
    # Pointcept's config merge deep-merges nested dicts, so without
    # _delete_=True the base config's OneCycleLR keys (max_lr, pct_start,
    # …) leak through and FlatWithDecayLR.__init__ explodes with an
    # 'unexpected keyword argument' TypeError. See
    # pointcept/utils/config.py::_merge_a_into_b for the _delete_ docs.
    _delete_=True,
    type="FlatWithDecayLR",
    mode="both",
    gamma=0.5,
    min_lr=1e-7,
    step_period_epochs=20,
    patience_epochs=10,
    min_delta=1e-4,
    cooldown_epochs=4,
    # Manual LR override on resume. When not None, FlatWithDecayLR's
    # load_state_dict overwrites every param_group's lr to this value
    # AFTER optimizer.load_state_dict() and the scheduler counter
    # restore — i.e. it is the final word for the next iteration.
    # Useful when the saved checkpoint's LR is too high after a long
    # run and you want to drop it before continuing.
    #
    # Caveat: this is applied EVERY time load_state_dict runs. After
    # your first resume produces a new checkpoint, set back to None
    # (or just delete the line) — otherwise the next resume will reset
    # the LR to this value again, undoing any decays in between.
    #
    # Set `reset_counters=True` to also wipe best_val_loss and the
    # plateau / period counters (use when reset_lr changes the loss
    # landscape enough that the old "best" is no longer comparable).
    reset_lr=1.0e-5,    
    reset_counters=True,
)

# =============================================================================
# Hooks: override the base list to insert LREpochScheduler between the
# evaluator (which writes val_loss into comm_info) and CheckpointSaver
# (which persists scheduler.state_dict()). Pointcept's config merge
# replaces lists wholesale, so we restate the full hook order here.
# =============================================================================
hooks = [
    dict(type="SonataCheckpointLoader"),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="GradScalerMonitor", log_frequency=10, warn_on_low_scale=4.0),
    dict(type="InformationWriter"),
    dict(type="ShowerClusteringEvaluator",
         eval_freq=0,
         best_metric="mask_iou_mean",
         empty_cache=True,
         log_per_event=False),
    dict(type="LREpochScheduler"),
    dict(type="CheckpointSaver", save_freq=10),
]