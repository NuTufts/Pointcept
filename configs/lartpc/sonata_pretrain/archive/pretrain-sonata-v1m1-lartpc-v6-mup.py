"""
SONATA Pretraining for LArTPC Point Clouds -- with muP (Maximal Update Parameterization)

This config extends pretrain-sonata-v1m1-lartpc-v6.py with muP scaling:
  - Attention scaling: 1/d_head instead of 1/sqrt(d_head)
  - Initialization: fan-in scaling instead of fixed std=0.02
  - Per-stage LR scaling: base_lr * layer_decay / m, where m = stage_width / base_width
  - Output layer (OnlineCluster prototype): zero-init

The benefit of muP is that hyperparameters tuned on a narrow proxy model
(e.g., 4x narrower) transfer directly to the full-width model.

See docs/reference/muP_for_Sonata.md for full explanation.
"""

_base_ = ["./pretrain-sonata-v1m1-lartpc-v6.py"]

# =============================================================================
# muP Configuration
# =============================================================================
# base_width: the width of the narrowest proxy model you would tune HPs on.
# When training the full model, m = enc_channels[s] / mup_base_width gives
# the per-stage width multiplier.
# When training the proxy model, set enc_channels to (mup_base_width, 2*mup_base_width, ...)
# so that m=1 at stage 0 and HPs transfer directly.
mup_enabled = True
mup_base_width = 48  # matches enc_channels[0] of the full model

# Inherit all model settings from v6 and add muP flags
# The backbone dict from _base_ is merged; we override with muP params
model = dict(
    backbone=dict(
        mup_enabled=mup_enabled,
        mup_base_width=mup_base_width,
    ),
    mup_enabled=mup_enabled,
    mup_base_width=mup_base_width,
)

# =============================================================================
# muP Learning Rate Scaling
# =============================================================================
# Re-read enc_channels and enc_depths from base config
# (These are defined in the _base_ config and merged into model)
enc_channels = (48, 96, 192, 384, 512)
enc_depths = (3, 3, 3, 9, 3)

# Import lr settings from base (re-specify since we need them for computation)
base_lr = 0.002
lr_decay = 0.9

# Per-stage muP LR scaling:
# For each encoder block, LR = base_lr * layer_decay^depth_index / m
# where m = enc_channels[stage] / mup_base_width
total_depth = sum(enc_depths)
param_dicts = []
for e in range(len(enc_depths)):
    m = enc_channels[e] / mup_base_width  # width multiplier for this stage
    for b in range(enc_depths[e]):
        depth_idx = total_depth - sum(enc_depths[:e]) - b - 1
        param_dicts.append(dict(
            keyword=f"enc{e}.block{b}.",
            lr=base_lr * lr_decay ** depth_idx / m,
        ))

# Head param groups with muP scaling
# The heads have hidden_channels=2048 as the widest hidden layer
head_hidden_channels = 2048
head_m = head_hidden_channels / mup_base_width
param_dicts.append(dict(keyword="mask_head.", lr=base_lr / head_m))
param_dicts.append(dict(keyword="unmask_head.", lr=base_lr / head_m))

# GridPooling layers (between encoder stages) -- also hidden-to-hidden
# These are captured by the "enc{s}.down." keyword
for s in range(1, len(enc_depths)):
    m = enc_channels[s] / mup_base_width
    param_dicts.append(dict(keyword=f"enc{s}.down.", lr=base_lr / m))

# =============================================================================
# Scheduler (must match param group count)
# =============================================================================
lr_pct_start = 0.06
lr_final_div_factor = 1000.0

optimizer = dict(type="AdamW", lr=base_lr, weight_decay=0.04)
scheduler = dict(
    type="OneCycleLR",
    max_lr=[base_lr] + [g["lr"] for g in param_dicts],
    pct_start=lr_pct_start,
    anneal_strategy="cos",
    div_factor=1000.0,
    final_div_factor=lr_final_div_factor,
)

# Update save path to distinguish muP runs
save_path = "sonata/lartpc_v6_h200_noghosts_pretrain_logspace_mup"

del enc_channels, enc_depths, total_depth, head_hidden_channels, head_m
