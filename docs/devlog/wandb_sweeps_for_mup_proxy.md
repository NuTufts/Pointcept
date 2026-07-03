# W&B Sweeps for muP Proxy Model Hyperparameter Optimization

> **Status: WORKLOG (2026-03)** — W&B sweep setup for the muP proxy-model tuning campaign.

## Context

With muP implemented for the Sonata model, optimal hyperparameters (LR, weight decay, temperatures) tuned on a narrow proxy model transfer to the full-width model. This document describes how to use W&B Sweeps to efficiently search the HP space on a cheap proxy, then apply those HPs to the full model.

---

## Part 1: How W&B Sweeps Work

### Architecture

There are three components:

1. **Sweep controller** -- runs on W&B's servers (cloud). It decides which HP combinations to try next based on the sweep configuration and results so far.

2. **Sweep agent** -- a lightweight process you run on your cluster node(s). It polls the controller for the next set of HPs, then executes your training script with those HPs. When the run finishes, it asks for the next set and repeats.

3. **Your training script** -- a normal Python script that reads HPs from `wandb.config` and reports metrics via `wandb.log()`.

### Workflow

```bash
# Step 1: Define sweep config (YAML or Python dict)
# Step 2: Create the sweep (returns a sweep ID)
wandb sweep sweep_config.yaml
# prints: "Created sweep with ID: abc123"

# Step 3: Launch agent(s) on your cluster
wandb agent your-entity/pointcept/abc123
```

### Key behaviors

- **Agents run trials sequentially** within a single process/job. The loop is: get HPs from controller -> run training -> report results -> repeat.
- **Parallelism**: Launch multiple agents (separate SLURM jobs or nodes) pointing at the same sweep ID. The controller coordinates so they don't duplicate work.
- **No central launcher needed**: The controller lives on W&B's cloud. You only run agents.
- **HP delivery**: The controller injects HPs into `wandb.config`. Your script reads them.

### Sweep methods

| Method | How it picks HPs | Best for |
|---|---|---|
| `grid` | All combinations | Small discrete spaces (< 20 combos) |
| `random` | Random sampling | Broad exploration, any space |
| `bayes` | Bayesian optimization (Gaussian process) | Continuous spaces, limited trial budget |

For proxy model tuning with ~3 HPs and ~20-30 trials, **`bayes`** is the best choice.

---

## Part 2: Which HPs to Sweep

### Transfers across widths (sweep these):

| HP | Why it transfers | Sweep range |
|---|---|---|
| `base_lr` | Primary muP guarantee | `[1e-4, 1e-2]` log-uniform |
| `base_wd` | Update-to-weight ratio stabilized by muP | `[0.01, 0.2]` |
| `student_temp` | Logit magnitudes are O(1) under muP, so temperature meaning is stable | `[0.05, 0.2]` |
| `teacher_temp_base` | Same reason as student_temp | `[0.04, 0.15]` |

### Does NOT transfer (keep fixed):

| HP | Why | Recommendation |
|---|---|---|
| Batch size | Gradient noise, data property | Use same as full model (or scale LR if different) |
| Max points per event | Sequence length, orthogonal to width | Keep same as full model |
| Number of prototypes | Changes the problem, not width | Keep at 4096 |
| Masking schedule | Data augmentation | Keep same |
| Number of views | Architecture choice | Keep same |
| Drop path rate | Regularization need changes with capacity | May need separate tuning at full width |
| Number of epochs | Convergence speed differs | Proxy trains faster; use enough to assess quality |

---

## Part 3: Implementation Plan

### Step 1: Create a proxy model config

Create `configs/lartpc/sonata_pretrain/archive/pretrain-sonata-v1m1-lartpc-v6-mup-proxy.py` that inherits from the muP config but with 4x narrower channels:

```python
_base_ = ["./pretrain-sonata-v1m1-lartpc-v6-mup.py"]

# 4x narrower proxy model
# head_dim = 48/3 = 16 in the full model, keep it consistent
enc_channels = (12, 24, 48, 96, 128)
enc_depths = (3, 3, 3, 9, 3)  # keep depth the same
enc_num_head = (1, 2, 3, 6, 8)  # channels / head_dim

model = dict(
    backbone=dict(
        enc_channels=enc_channels,
        enc_num_head=enc_num_head,
    ),
    # head_in_channels = sum of channels from up_cast_level onward
    # up_cast_level=2: channels[2] + channels[3] + channels[4] = 48 + 96 + 128 = 272
    head_in_channels=272,
    head_hidden_channels=512,  # scale down proportionally
)

# Shorter training for sweep trials (enough to assess HP quality)
epoch = 5
eval_epoch = 5

save_path = "sonata/lartpc_v6_mup_proxy_sweep"
```

### Step 2: Create the sweep training wrapper

Create `tools/sweep_train.py`:

```python
"""
W&B Sweep training wrapper for Sonata muP proxy model.

This script is called by wandb agent. It reads HPs from wandb.config,
patches the Pointcept config, and runs training.

Usage (standalone test):
    python tools/sweep_train.py \
        --config configs/lartpc/sonata_pretrain/archive/pretrain-sonata-v1m1-lartpc-v6-mup-proxy.py

Usage (via wandb agent -- see sweep_config.yaml):
    wandb agent <entity>/<project>/<sweep_id>
"""
import os
import sys
import copy
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import wandb
from pointcept.utils.config import Config


def build_mup_param_dicts(base_lr, lr_decay, enc_channels, enc_depths,
                          mup_base_width, head_hidden_channels):
    """Generate muP-scaled parameter groups from sweep HPs."""
    total_depth = sum(enc_depths)
    param_dicts = []

    for e in range(len(enc_depths)):
        m = enc_channels[e] / mup_base_width
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

    return param_dicts


def train(config_path):
    """Single training run with HPs from wandb.config."""
    # Initialize wandb (HPs come from sweep controller)
    run = wandb.init()

    # Load base config
    cfg = Config.fromfile(config_path)

    # Override HPs from sweep
    base_lr = wandb.config.get("base_lr", cfg.optimizer.lr)
    base_wd = wandb.config.get("base_wd", cfg.optimizer.weight_decay)
    student_temp = wandb.config.get("student_temp", cfg.model.student_temp)
    teacher_temp_base = wandb.config.get(
        "teacher_temp_base", cfg.model.teacher_temp_base
    )

    # Apply to config
    cfg.optimizer.lr = base_lr
    cfg.optimizer.weight_decay = base_wd
    cfg.model.student_temp = student_temp
    cfg.model.teacher_temp_base = teacher_temp_base

    # Rebuild param_dicts with new base_lr
    enc_channels = cfg.model.backbone.enc_channels
    enc_depths = cfg.model.backbone.enc_depths
    mup_base_width = cfg.model.get("mup_base_width", 48)
    lr_decay = wandb.config.get("lr_decay", 0.9)
    head_hidden_channels = cfg.model.get("head_hidden_channels", 2048)

    param_dicts = build_mup_param_dicts(
        base_lr, lr_decay, enc_channels, enc_depths,
        mup_base_width, head_hidden_channels,
    )
    cfg.param_dicts = param_dicts

    # Rebuild scheduler max_lr to match param groups
    cfg.scheduler.max_lr = [base_lr] + [g["lr"] for g in param_dicts]

    # Unique save path per run
    cfg.save_path = f"sonata/sweep_{run.id}"

    # Enable wandb in Pointcept
    cfg.enable_wandb = True
    cfg.wandb_project = wandb.run.project

    # Import and run trainer
    from pointcept.engines.train import DefaultTrainer
    trainer = DefaultTrainer(cfg)
    trainer.train()

    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/lartpc/sonata_pretrain/archive/pretrain-sonata-v1m1-lartpc-v6-mup-proxy.py",
    )
    args = parser.parse_args()
    train(args.config)
```

### Step 3: Create the sweep configuration

Create `configs/sweeps/mup_proxy_sweep.yaml`:

```yaml
program: tools/sweep_train.py
method: bayes
metric:
  name: loss
  goal: minimize
parameters:
  base_lr:
    min: 0.0001
    max: 0.01
    distribution: log_uniform_values
  base_wd:
    min: 0.01
    max: 0.2
    distribution: uniform
  student_temp:
    values: [0.05, 0.075, 0.1, 0.125, 0.15, 0.2]
  teacher_temp_base:
    values: [0.04, 0.05, 0.07, 0.1]
early_terminate:
  type: hyperband
  min_iter: 500
  max_iter: 5000
  s: 2
```

Notes on the config:
- `method: bayes` uses Gaussian process optimization, good for ~20-50 trials
- `log_uniform_values` for LR ensures even sampling across orders of magnitude
- `early_terminate` with Hyperband kills bad runs early, saving compute
- `min_iter` / `max_iter` refer to the number of logged steps

### Step 4: Create the SLURM submission script

Create `submit_mup_sweep.sh`:

```bash
#!/bin/bash

#SBATCH --job-name=mup_sweep
#SBATCH --output=mup_sweep.%j.%N.log
#SBATCH --mem-per-cpu=2000
#SBATCH --cpus-per-task=16
#SBATCH --time=1-00:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --error=mup_sweep.%j.%N.err

# Proxy model is ~16x fewer params, 1 GPU is sufficient
# Each trial trains for ~5 epochs

WORKDIR=/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/pointcept/
container=/cluster/tufts/wongjiradlabnu/larbys/larbys-container/pointcept_cuml.sif

module load modtree/deprecated
module load apptainer/1.2.4-suid

# SWEEP_ID should be set before submitting, e.g.:
#   export SWEEP_ID=abc123
#   sbatch submit_mup_sweep.sh
# Or create the sweep first:
#   wandb sweep configs/sweeps/mup_proxy_sweep.yaml

apptainer exec --nv --bind /cluster:/cluster $container bash -c "
    cd ${WORKDIR}
    source ../setenv_pointcept_only.sh
    wandb agent nutufts/pointcept/${SWEEP_ID} --count 10
"
```

### Step 5: Launch the sweep

```bash
# 1. Create the sweep (do this once, from login node)
cd /cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/pointcept
wandb sweep configs/sweeps/mup_proxy_sweep.yaml
# Output: "Created sweep with ID: abc123"

# 2. Launch agents (submit multiple jobs for parallelism)
export SWEEP_ID=abc123

# Option A: Single agent, runs 10 trials sequentially
sbatch submit_mup_sweep.sh

# Option B: 3 parallel agents, each runs 10 trials (30 total)
sbatch submit_mup_sweep.sh
sbatch submit_mup_sweep.sh
sbatch submit_mup_sweep.sh
```

### Step 6: Transfer HPs to full model

Once the sweep completes, find the best run on the W&B dashboard:

```bash
# Query best run via CLI
wandb api runs nutufts/pointcept --filter '{"sweep": "abc123"}' \
    --order '-summary_metrics.loss' --limit 1
```

Then create the full-width training config using those HPs:

```python
# configs/lartpc/pretrain-sonata-v1m1-lartpc-v6-mup-tuned.py
_base_ = ["./pretrain-sonata-v1m1-lartpc-v6-mup.py"]

# Best HPs from proxy sweep (example values)
base_lr = 0.0032      # from sweep
base_wd = 0.06        # from sweep
lr_final_div_factor = 1000.0

model = dict(
    student_temp=0.1,          # from sweep
    teacher_temp_base=0.07,    # from sweep
)

# ... rebuild param_dicts and scheduler with the new base_lr ...
```

---

## Part 4: Practical Considerations

### Compute budget

| Component | Proxy (4x narrow) | Full model |
|---|---|---|
| Params | ~1/16th | Full |
| GPU memory | ~1/4th | Full |
| GPUs needed | 1 | 2 |
| Time per epoch | ~4x faster | Full |
| Sweep trials | 20-30 | 1 (transfer) |
| Total sweep cost | ~5-8 GPU-hours | -- |

### Tips

1. **Sweep metric**: Use the training loss averaged over the last 20% of steps, not the final loss (which is noisy). Or use the linear probe accuracy if `PretrainEvaluator` is enabled.

2. **Early termination**: Hyperband (configured above) will kill runs that are clearly worse than the median after `min_iter` steps. This saves ~30-50% compute on a typical sweep.

3. **Trial length**: 5 epochs on the proxy is usually enough to distinguish good vs bad HPs. The learning rate is the most sensitive -- bad LRs show clearly within 1-2 epochs.

4. **Parallel agents**: Submit 3-4 agents for a 30-trial Bayesian sweep. More agents means faster wall-clock time but slightly less efficient Bayesian optimization (agents don't wait for each other's results before starting new trials).

5. **Reproducibility**: W&B logs all HPs, metrics, and system info per run. The sweep dashboard shows parallel coordinates and importance plots to understand which HPs matter most.

6. **After the sweep**: Run the coord check (`tools/mup_coord_check.py`) with the best HPs at both proxy and full width to verify the HPs actually transferred before committing to a full training run.

---

## Part 5: File Locations

| Component | Path |
|---|---|
| muP full config | `configs/lartpc/sonata_pretrain/archive/pretrain-sonata-v1m1-lartpc-v6-mup.py` |
| Proxy config (to create) | `configs/lartpc/sonata_pretrain/archive/pretrain-sonata-v1m1-lartpc-v6-mup-proxy.py` |
| Sweep wrapper (to create) | `tools/sweep_train.py` |
| Sweep YAML (to create) | `configs/sweeps/mup_proxy_sweep.yaml` |
| SLURM script (to create) | `submit_mup_sweep.sh` |
| Coord check | `tools/mup_coord_check.py` |
| muP documentation | `docs/reference/muP_for_Sonata.md` |
