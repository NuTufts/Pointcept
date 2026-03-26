"""
SONATA Pretraining -- muP Proxy Model (3x narrower)

Narrower version of the full muP model for:
  - Validating that muP changes work correctly
  - Quick single-GPU training runs
  - Hyperparameter tuning (HPs transfer to full model via muP)

Channel widths: (16, 32, 64, 128, 256) vs full model (48, 96, 192, 384, 512)
head_dim = 16 throughout (same as full model: 48/3 = 16)
~16x fewer parameters, fits easily on 1 GPU.

mup_base_width = 48 (same as full model -- this is what makes HP transfer work).
"""

_base_ = ["./pretrain-sonata-v1m1-lartpc-v6.py"]

# =============================================================================
# muP Configuration (must match full model)
# =============================================================================
mup_enabled = True
mup_base_width = 48  # SAME as full model -- do not change

# =============================================================================
# Narrower proxy model
# =============================================================================
enc_channels = (16, 32, 64, 128, 256)
enc_depths = (3, 3, 3, 9, 3)  # keep depth the same
enc_num_head = (1, 2, 4, 8, 16)  # head_dim = 16 throughout

# head_in_channels = sum of channels from up_cast_level=2 onward:
# channels[2] + channels[3] + channels[4] = 64 + 128 + 256 = 448
head_in_channels = 448
head_hidden_channels = 512  # scaled down from 2048

# select backend depending on GPU
#flash_backend='flash_attn' # default backend
#amp_dtype = "bfloat16" # use this for default 'flash_attn' backend
flash_backend='xformers'   # backend needed to run on P100
amp_dtype = "float16"      # use this with xformer backend on P100

model = dict(
    backbone=dict(
        enc_channels=enc_channels,
        enc_num_head=enc_num_head,
        mup_enabled=mup_enabled,
        mup_base_width=mup_base_width,
        flash_backend=flash_backend,
    ),
    head_in_channels=head_in_channels,
    head_hidden_channels=head_hidden_channels,
    mup_enabled=mup_enabled,
    mup_base_width=mup_base_width,
)

# =============================================================================
# Training settings for validation run
# =============================================================================
epoch = 5
eval_epoch = 5
batch_size = 8  # smaller model, can use smaller batch on 1 GPU
enable_wandb = True
save_path = "sonata/lartpc_v6_mup_proxy_validation"
num_worker = 4


# =============================================================================
# muP Learning Rate Scaling (same formula as full model, different channels)
# =============================================================================
base_lr = 0.002
lr_decay = 0.9

total_depth = sum(enc_depths)
param_dicts = []
for e in range(len(enc_depths)):
    m = enc_channels[e] / mup_base_width  # m < 1 for narrow stages -- that's correct
    for b in range(enc_depths[e]):
        depth_idx = total_depth - sum(enc_depths[:e]) - b - 1
        param_dicts.append(dict(
            keyword=f"enc{e}.block{b}.",
            lr=base_lr * lr_decay ** depth_idx / m,
        ))

# Head param groups
head_m = head_hidden_channels / mup_base_width
param_dicts.append(dict(keyword="mask_head.", lr=base_lr / head_m))
param_dicts.append(dict(keyword="unmask_head.", lr=base_lr / head_m))

# GridPooling layers
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

del total_depth, head_m
