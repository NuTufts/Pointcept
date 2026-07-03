# Implementing muP (Maximal Update Parameterization) for Sonata

## Context

We are pre-training a Sonata self-supervised model on LArTPC data using config `configs/lartpc/sonata_pretrain/archive/pretrain-sonata-v1m1-lartpc-v6.py`. The backbone is PT-v3m2 (PointTransformerV3 mode 2) with encoder channels `(48, 96, 192, 384, 512)` and OnlineCluster heads. Currently, all Linear layers use `trunc_normal_(std=0.02)` initialization and a single base learning rate (with layer-wise decay). This is Standard Parameterization (SP), where optimal hyperparameters (especially learning rate) change as you scale model width. muP would let you tune hyperparameters on a small/narrow model and transfer them to your full-size model, saving significant GPU time on HP searches.

---

## Part 1: What is Maximal Update Parameterization (muP)?

### The Problem muP Solves

In standard parameterization (SP -- what we have now), when you change model width (e.g., double `enc_channels`), the optimal learning rate, initialization scale, and other hyperparameters shift unpredictably. A learning rate that works for a 48-channel model won't work for a 512-channel model. This means you must re-tune hyperparameters at every scale -- expensive for large models.

### The Core Idea

muP (from Yang et al., "Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer", 2022) defines a parameterization where **the size of feature updates in each layer remains O(1) regardless of model width**. When this holds, optimal hyperparameters transfer across widths.

The key insight: in a standard transformer, as width `d` grows, different layers need different scaling of their learning rates and initialization to keep updates stable. muP prescribes exactly what those scalings should be.

### muP Rules for Transformers

Define a "base width" `d_base` (e.g., 48, the narrowest encoder stage) and the actual width `d`. The **width multiplier** is `m = d / d_base`.

| Layer Type | SP Init | muP Init | SP LR | muP LR | SP Output Scale | muP Output Scale |
|---|---|---|---|---|---|---|
| **Embedding** (input to first hidden) | `O(1)` | `O(1)` -- same | `eta` | `eta` -- same | `1` | `1` |
| **Hidden-to-Hidden** (all intermediate Linear, QKV, MLP, projections) | `O(1/sqrt(d))` | `O(1/sqrt(d))` -- same | `eta` | `eta / m` | `1` | `1` |
| **Output/Readout** (last linear before loss) | `O(1/sqrt(d))` | **`O(1/d)`** | `eta` | `eta / m` | `1` | **`1/m`** (or init to zero) |
| **Attention logits** | `1/sqrt(d_head)` | **`1/d_head`** | -- | -- | -- | -- |

In plain English:

1. **Embedding layer**: No change needed. Input-to-hidden is fine as-is.
2. **Hidden layers** (QKV projections, output projections, MLP fc1/fc2, GridPooling projections, CPE layers): **Scale the learning rate down by `1/m`** where `m = width / base_width`. Initialization stays `1/sqrt(fan_in)` (or `trunc_normal_(std=0.02)` -- but see note below).
3. **Output/readhead layer** (the OnlineCluster head's final projection to prototypes): **Initialize smaller by `1/m`** AND **scale LR down by `1/m`**. Alternatively, initialize to zero.
4. **Attention scaling**: Change from `1/sqrt(d_head)` to `1/d_head`. This is critical -- it prevents attention entropy from collapsing at large widths.

### Why This Works (Intuition)

In SP, a hidden layer with width `d` has weights `W ~ O(1/sqrt(d))` and input activations `x ~ O(1)`. The output `Wx ~ O(1)` (good). But the **update** `delta_W * x` scales as `O(eta/sqrt(d))` which shrinks with width -- deeper layers barely learn. muP fixes this by scaling `eta` with `1/m` for hidden layers so updates remain `O(1)` at every width.

For attention: `Q^T K / sqrt(d)` has logits that grow as `O(sqrt(d))` in SP at initialization, causing attention to sharpen/collapse for wide models. Using `1/d` instead of `1/sqrt(d)` keeps logits `O(1)`.

### What This Means Practically

1. Pick a narrow "proxy" model (e.g., `enc_channels=(12, 24, 48, 96, 128)` -- 4x narrower)
2. Tune learning rate, weight decay, temperature schedules, etc. on the proxy
3. Scale up to full width `(48, 96, 192, 384, 512)` -- the tuned hyperparameters transfer directly
4. Only the width multiplier `m` changes the per-layer LR scaling; the base LR stays the same

---

## Part 2: Specific Changes Needed for the Sonata Model

### 2.1 Files to Modify

| File | What Changes |
|---|---|
| `pointcept/models/point_transformer_v3/point_transformer_v3m2_sonata.py` | Attention scaling, init, optional width-aware LR tags |
| `pointcept/models/sonata/sonata_v1m1_base.py` | Head initialization, output scaling |
| `pointcept/utils/optimizer.py` | Support muP per-parameter LR multipliers |
| `configs/lartpc/sonata_pretrain/archive/pretrain-sonata-v1m1-lartpc-v6.py` (or new config) | muP-specific param_dicts, base_width setting |

### 2.2 Change 1: Attention Scaling (`1/sqrt(d_head)` -> `1/d_head`)

**File**: `pointcept/models/point_transformer_v3/point_transformer_v3m2_sonata.py`, class `SerializedAttention`

Currently:
```python
self.scale = qk_scale or (channels // num_heads) ** -0.5  # 1/sqrt(d_head)
```

Change to (when muP is enabled):
```python
head_dim = channels // num_heads
if mup_enabled:
    self.scale = 1.0 / head_dim          # 1/d_head (muP)
else:
    self.scale = qk_scale or head_dim ** -0.5  # 1/sqrt(d_head) (SP, default)
```

**Why**: This is the single most impactful muP change. It prevents attention logits from growing with width, keeping the softmax distribution stable across scales.

### 2.3 Change 2: Initialization Scaling

**File**: `point_transformer_v3m2_sonata.py`, method `_init_weights`

Currently all Linear layers use `trunc_normal_(std=0.02)` regardless of fan-in. This is non-standard even for SP (which should use `std ~ 1/sqrt(fan_in)`). For muP:

```python
@staticmethod
def _init_weights(module, mup_enabled=False, base_width=None):
    if isinstance(module, nn.Linear):
        fan_in = module.in_features
        if mup_enabled and base_width is not None:
            m = fan_in / base_width  # width multiplier (approximate)
            std = 1.0 / math.sqrt(fan_in)  # standard fan-in scaling
            # Hidden layers: same as SP. Output layers handled separately.
        else:
            std = 0.02  # legacy behavior
        trunc_normal_(module.weight, std=std)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
```

**For the output layer** (OnlineCluster prototype or final MLP layer):
```python
# Output layer init: scale down by 1/m
std = 1.0 / fan_in  # O(1/d) instead of O(1/sqrt(d))
# OR simply: nn.init.zeros_(module.weight)
```

**File**: `pointcept/models/sonata/sonata_v1m1_base.py`, class `OnlineCluster._init_weights`

The prototype layer uses weight normalization so its effective output is already normalized. The MLP layers (`Linear(1088, 2048)` and `Linear(2048, 256)`) are hidden layers and should use fan-in scaling. The prototype itself (256 -> 4096) acts as the readout -- its direction vector `v` should be initialized smaller or to zero under muP.

### 2.35 Change 2b: Sparse 3D Convolutions (CPE Layers)

Each transformer block contains a Conditional Positional Encoding (CPE) module with a `spconv.SubMConv3d(channels, channels, kernel_size=3)` followed by `nn.Linear(channels, channels)`. These are **hidden-to-hidden** layers under muP.

**Fan-in for 3D convolution**: `fan_in = k^3 * C_in` where `k=3` (kernel size) and `C_in = channels`.

| Stage | channels | fan_in = 27 * C | Current init (std=0.02) | muP init std |
|---|---|---|---|---|
| 0 | 48 | 1296 | 0.020 | 0.028 |
| 1 | 96 | 2592 | 0.020 | 0.020 |
| 2 | 192 | 5184 | 0.020 | 0.014 |
| 3 | 384 | 10368 | 0.020 | 0.010 |
| 4 | 512 | 13824 | 0.020 | 0.0085 |

Note: the current fixed `std=0.02` happens to match stage 1 (96 channels) but over-initializes the deeper/wider stages and under-initializes stage 0.

**Initialization change** in `_init_weights`:
```python
elif isinstance(module, spconv.SubMConv3d):
    if mup_enabled:
        # fan_in = kernel_volume * in_channels
        # spconv weight shape is typically (out_ch, kernel_vol, in_ch) -- verify at runtime
        fan_in = module.weight.shape[1] * module.weight.shape[2]
        std = 1.0 / math.sqrt(fan_in)
    else:
        std = 0.02  # legacy behavior
    trunc_normal_(module.weight, std=std)
    if module.bias is not None:
        nn.init.zeros_(module.bias)
```

**Learning rate**: Same `1/m` scaling as other hidden layers. The CPE lives inside the block, so it is **already captured** by the `keyword=f"enc{e}.block{b}."` param_dict matching -- no additional config change needed.

**SubMConv3d specifics**: Submanifold sparse convolutions only compute outputs at active (non-zero) input sites, and the kernel only sees neighboring active voxels. This means the *effective* fan-in per output can be less than `k^3 * C_in` due to sparsity. However, for muP we use the **architectural fan-in** (`k^3 * C_in`), not the data-dependent effective fan-in, because the parameterization must be independent of the input data.

### 2.4 Change 3: Per-Layer Learning Rate Scaling

This is the most involved change. muP requires:
- **Embedding layer** (`backbone.embedding`): LR = `base_lr` (no scaling)
- **All hidden layers** (encoder blocks, GridPooling, CPE, head MLP): LR = `base_lr / m` where `m = width / base_width`
- **Output layer** (prototype): LR = `base_lr / m`

**Problem**: The model has variable width across stages (48, 96, 192, 384, 512). In a strict muP interpretation with a single `base_width`, each stage has a different `m`. There are two approaches:

**Approach A: Single base_width, per-stage multiplier (Theoretically correct)**
```python
base_width = 48  # narrowest stage
enc_channels = (48, 96, 192, 384, 512)

# For each encoder stage, compute m = channels / base_width
# Stage 0: m=1, Stage 1: m=2, Stage 2: m=4, Stage 3: m=8, Stage 4: m=10.67
# LR for stage s = base_lr / m_s

param_dicts = []
for e in range(len(enc_depths)):
    m = enc_channels[e] / base_width
    for b in range(enc_depths[e]):
        layer_decay = lr_decay ** (total_depth - depth_index - 1)
        param_dicts.append(dict(
            keyword=f"enc{e}.block{b}.",
            lr=base_lr * layer_decay / m,  # muP: divide by width multiplier
        ))
```

**Approach B: Uniform multiplier (Simpler, "good enough" in practice)**

Treat the model as having a single effective width (e.g., the largest stage, 512) and apply a uniform `1/m` scaling to all hidden layers. Less theoretically pure but simpler and often works well.

**Config changes** (`pretrain-sonata-v1m1-lartpc-v6.py`):
```python
# muP settings
mup_enabled = True
mup_base_width = 48  # Width of the narrowest proxy model

# Model config addition
model = dict(
    ...,
    mup_enabled=mup_enabled,
    mup_base_width=mup_base_width,
)

# Param dicts with muP LR scaling
param_dicts = []
for e in range(len(enc_depths)):
    m = enc_channels[e] / mup_base_width
    for b in range(enc_depths[e]):
        depth_idx = sum(enc_depths) - sum(enc_depths[:e]) - b - 1
        param_dicts.append(dict(
            keyword=f"enc{e}.block{b}.",
            lr=base_lr * lr_decay**depth_idx / m,
        ))

# Head params (output layer scaling)
param_dicts.append(dict(
    keyword="mask_head.",
    lr=base_lr / (head_hidden_channels / mup_base_width),
))
param_dicts.append(dict(
    keyword="unmask_head.",
    lr=base_lr / (head_hidden_channels / mup_base_width),
))
```

### 2.5 Change 4: Optimizer Integration

**File**: `pointcept/utils/optimizer.py`

The existing `param_dicts` keyword matching system already supports per-parameter learning rates. No fundamental change needed -- the config-level `param_dicts` can encode muP scaling directly.

However, for cleaner integration, a utility function could be added:

```python
def build_mup_param_dicts(model, base_lr, base_width, lr_decay=1.0, enc_depths=None, enc_channels=None):
    """Generate muP-scaled parameter groups."""
    param_dicts = []
    # ... generate per-layer LR based on width multiplier
    return param_dicts
```

### 2.6 Change 5: Weight Decay

Under muP, weight decay should generally **not** be scaled with width (it's already per-parameter). The current setup uses a WD schedule from `base_wd=0.04` to `final_wd=0.2` via `WeightDecaySchedular` -- this can stay as-is. However, some practitioners scale WD by `m` for hidden layers; this is optional and can be tuned on the proxy model.

### 2.7 Change 6: Teacher Model Considerations

The teacher in Sonata is an EMA copy of the student. Since the teacher's parameters are derived from the student via `momentum * teacher + (1 - momentum) * student`, the muP scaling naturally propagates. **No special handling needed for the teacher.**

### 2.8 Summary of All Changes

| Component | Current (SP) | muP Change | Complexity |
|---|---|---|---|
| Attention scale | `1/sqrt(d_head)` | `1/d_head` | 1 line |
| Hidden layer init | `trunc_normal_(std=0.02)` | `trunc_normal_(std=1/sqrt(fan_in))` | ~5 lines in `_init_weights` |
| Output layer init | `trunc_normal_(std=0.02)` | `zeros_` or `std=1/fan_in` | ~3 lines |
| Hidden layer LR | `base_lr * layer_decay` | `base_lr * layer_decay / m` | Config change |
| Output layer LR | `base_lr` | `base_lr / m` | Config change |
| Embedding LR | `base_lr` | `base_lr` (unchanged) | None |
| Weight decay | Scheduled | No change (optional: scale by `m`) | None |
| Teacher EMA | Momentum schedule | No change | None |

---

## Part 3: Implementation Strategy

### Phase 1: Add muP flag and attention scaling (Minimal, high-impact)
1. Add `mup_enabled` and `mup_base_width` parameters to PT-v3m2 backbone
2. Change attention scaling from `1/sqrt(d_head)` to `1/d_head` when enabled
3. This alone is the single biggest muP improvement

### Phase 2: Fix initialization
1. Modify `_init_weights` in PT-v3m2 to use fan-in scaling when muP enabled
2. Modify OnlineCluster `_init_weights` for output layer zero-init
3. Add `mup_enabled` flag to Sonata model class

### Phase 3: LR scaling in config
1. Create a new config `pretrain-sonata-v1m1-lartpc-v6-mup.py`
2. Compute per-stage width multipliers and apply to `param_dicts`
3. Add head param groups with appropriate scaling
4. Adjust `scheduler.max_lr` list to match new param groups

### Phase 4: Proxy model for HP transfer (the payoff)
1. Create a narrow proxy config (e.g., 4x narrower: channels `(12, 24, 48, 96, 128)`)
2. Tune base_lr, weight_decay, temperatures on the proxy (~4x cheaper per run)
3. Transfer those HPs to full-width model -- they should work without re-tuning

---

## Part 4: Verification

### 4.1 Coordinate Check Test (Primary Verification)

The "coord check" is the standard verification method from the muP paper. The idea: **if muP is correctly implemented, average activation magnitudes at each layer should remain constant across model widths**. If they grow or shrink with width, muP is not implemented correctly.

#### Procedure

1. **Define width multipliers** to test (e.g., 1x, 2x, 4x, 8x of a base width):
   ```python
   # Base (narrowest proxy):
   width_configs = [
       (12, 24, 48, 96, 128),    # 0.25x
       (24, 48, 96, 192, 256),    # 0.5x
       (48, 96, 192, 384, 512),   # 1x (your full model)
       (96, 192, 384, 768, 1024), # 2x (if GPU memory allows)
   ]
   ```

2. **For each width**, train for 10 steps with a **fixed learning rate** (e.g., `base_lr=0.002`), recording the mean absolute activation after each layer type:
   - Post-embedding activations
   - Post-attention activations (before residual add)
   - Post-MLP activations (before residual add)
   - Post-CPE activations
   - Head output (pre-prototype logits)

3. **Plot**: For each layer type, plot `mean |activation|` (y-axis) vs `width multiplier` (x-axis) after step 10.

#### Implementation: Coord Check Script

Create `tools/mup_coord_check.py`:

```python
"""
muP Coordinate Check for Sonata/PT-v3m2.

Trains models at multiple widths for a few steps and records per-layer
activation statistics. Under correct muP, activations should be O(1)
regardless of width.

Usage:
    python tools/mup_coord_check.py \
        --config-file configs/lartpc/sonata_pretrain/archive/pretrain-sonata-v1m1-lartpc-v6-mup.py \
        --num-steps 10 \
        --widths 0.25,0.5,1.0,2.0
"""
import argparse
import copy
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from collections import defaultdict

from pointcept.utils.config import Config
from pointcept.models import build_model
from pointcept.utils.optimizer import build_optimizer


def register_activation_hooks(model):
    """Register forward hooks on key layers to record activation magnitudes."""
    stats = defaultdict(list)  # layer_name -> list of mean |activation|

    def make_hook(name):
        def hook(module, input, output):
            if isinstance(output, torch.Tensor):
                stats[name].append(output.detach().abs().mean().item())
            # Handle PointModule outputs where feat is in point.feat
            elif hasattr(output, 'feat'):
                stats[name].append(output.feat.detach().abs().mean().item())
        return hook

    hooks = []
    for name, module in model.named_modules():
        # Track key layer types
        if any(key in name for key in [
            'embedding', 'attn.proj', 'mlp.fc2', 'cpe',
            'mask_head.mlp', 'unmask_head.mlp'
        ]):
            hooks.append(module.register_forward_hook(make_hook(name)))
    return stats, hooks


def run_coord_check(config_path, num_steps=10, width_multipliers=[0.25, 0.5, 1.0, 2.0]):
    """Run coord check across multiple widths."""
    base_cfg = Config.fromfile(config_path)
    base_channels = base_cfg.model.backbone.enc_channels
    base_heads = base_cfg.model.backbone.enc_num_head

    results = {}  # width_mult -> {layer_name: mean_activation_at_step_N}

    for mult in width_multipliers:
        cfg = copy.deepcopy(base_cfg)

        # Scale width
        cfg.model.backbone.enc_channels = tuple(
            max(int(c * mult), 1) for c in base_channels
        )
        # Scale heads proportionally (keep head_dim constant)
        head_dim = base_channels[0] // base_heads[0]
        cfg.model.backbone.enc_num_head = tuple(
            max(c // head_dim, 1) for c in cfg.model.backbone.enc_channels
        )
        # Update head_in_channels for OnlineCluster
        enc_ch = cfg.model.backbone.enc_channels
        up_cast_level = cfg.model.get('up_cast_level', 2)
        cfg.model.head_in_channels = sum(enc_ch[up_cast_level:])

        # Build model
        model = build_model(cfg.model).cuda()
        model.train()

        # Register hooks
        stats, hooks = register_activation_hooks(model.student)

        # Build optimizer with muP param dicts
        # (need to generate param_dicts for this width)
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.optimizer.lr)

        # Train for num_steps
        for step in range(num_steps):
            # Generate synthetic data matching expected input format
            # (In practice, use a real dataloader with a small batch)
            loss = model.forward_train(synthetic_batch)  # adapt to actual API
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        # Record final activation stats
        results[mult] = {
            name: values[-1] if values else 0.0
            for name, values in stats.items()
        }

        # Cleanup
        for h in hooks:
            h.remove()
        del model, optimizer
        torch.cuda.empty_cache()

    return results


def plot_coord_check(results, output_path="coord_check.png"):
    """Plot activation magnitudes vs width multiplier."""
    widths = sorted(results.keys())
    layer_names = sorted(results[widths[0]].keys())

    fig, axes = plt.subplots(1, len(layer_names), figsize=(4 * len(layer_names), 4))
    if len(layer_names) == 1:
        axes = [axes]

    for ax, name in zip(axes, layer_names):
        values = [results[w].get(name, 0) for w in widths]
        ax.plot(widths, values, 'o-')
        ax.set_xlabel("Width multiplier")
        ax.set_ylabel("Mean |activation|")
        ax.set_title(name.split('.')[-2] + '.' + name.split('.')[-1])
        ax.set_xscale('log', base=2)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Coord check plot saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", required=True)
    parser.add_argument("--num-steps", type=int, default=10)
    parser.add_argument("--widths", type=str, default="0.25,0.5,1.0,2.0")
    args = parser.parse_args()

    width_mults = [float(x) for x in args.widths.split(",")]
    results = run_coord_check(args.config_file, args.num_steps, width_mults)
    plot_coord_check(results)
```

#### Expected Results

**Under correct muP**: For each layer type, `mean |activation|` should be approximately flat across all width multipliers. The plot should show horizontal lines.

**Under SP (incorrect/no muP)**: Activations will typically:
- **Grow** with width for layers where init is too large (e.g., fixed `std=0.02` on wide layers)
- **Shrink** for layers where updates are too small (hidden layers with un-scaled LR)
- **Attention logits** will grow as `O(sqrt(width))` with `1/sqrt(d_head)` scaling

#### Interpretation Guide

| Layer | Flat? | If NOT flat |
|---|---|---|
| Post-embedding | Should be flat (embedding is not width-scaled) | Check embedding init |
| Post-attention | Should be flat | Check attention scale (`1/d_head`?) and QKV init |
| Post-MLP | Should be flat | Check MLP init and LR scaling |
| Post-CPE | Should be flat | Check SubMConv3d init (fan-in scaling?) |
| Head output | Should be flat | Check output layer init (zero/small?) and LR |

#### Running the Test

```bash
# Quick check with 2 widths (fast)
python tools/mup_coord_check.py \
    --config-file configs/lartpc/sonata_pretrain/archive/pretrain-sonata-v1m1-lartpc-v6-mup.py \
    --num-steps 10 --widths 0.5,1.0

# Full check with 4 widths (thorough)
python tools/mup_coord_check.py \
    --config-file configs/lartpc/sonata_pretrain/archive/pretrain-sonata-v1m1-lartpc-v6-mup.py \
    --num-steps 10 --widths 0.25,0.5,1.0,2.0
```

### 4.2 Additional Verification Methods

1. **LR sweep coord check**: Train proxy and full-width models for ~100 steps each at several learning rates (e.g., 1e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2). Plot final loss vs LR. Under muP, the optimal LR should be approximately the same for both widths. Under SP, it will shift.
2. **Attention entropy**: Log attention entropy. Under muP with `1/d_head` scaling, entropy should be similar across widths. With `1/sqrt(d_head)`, wider models will have lower entropy (sharper attention).
3. **Training stability**: The full-width model with transferred HPs should train stably without NaN/Inf. Monitor existing `GradScalerMonitor` and `AdamStateMonitor` hooks.

---

## Part 5: Backward Compatibility

All muP changes are gated behind a `mup_enabled` flag (default: `False`). When muP is **not** enabled:

- Attention scale remains `1/sqrt(d_head)` (unchanged)
- Initialization remains `trunc_normal_(std=0.02)` (unchanged)
- Learning rate scaling remains as currently configured (unchanged)
- No new parameters or modules are added

Existing configs, checkpoints, and training runs are completely unaffected.

---

## Part 6: Risks and Considerations

1. **Variable width across stages**: The backbone has 5 different widths (48-512). Pure muP assumes a single width. Approach A (per-stage multiplier) is theoretically correct but adds complexity. Consider starting with Approach B (uniform multiplier based on largest stage).

2. **Interaction with layer-wise LR decay**: The existing `lr_decay=0.9` interacts with muP's `1/m` scaling. Both are multiplicative so they compose naturally, but the combined effect needs validation.

3. **OneCycleLR compatibility**: The scheduler needs `max_lr` as a list matching param group count. Adding head param groups changes the list length -- make sure to update it.

4. **Flash attention**: Verified that all three attention code paths pass `self.scale` as a plain float:
   - `flash_attn.flash_attn_varlen_qkvpacked_func(..., softmax_scale=self.scale)` (line 252)
   - `xops.memory_efficient_attention(..., scale=self.scale)` (line 288)
   - Manual path: `q * self.scale @ k.T` (line 238)

   All accept any float value -- changing from `1/sqrt(d_head)` to `1/d_head` requires no API changes.

5. **Existing checkpoint compatibility**: muP changes initialization and LR scaling. You cannot resume from an SP checkpoint with muP settings -- you must train from scratch. However, you can use muP-tuned HPs with SP initialization if you only want the HP transfer benefit without changing the model code.

6. **The `mup` Python package**: Microsoft provides a `mup` package that automates much of this. However, it requires wrapping layers with `mup.MuReadout`, `mup.MuLinear`, etc., which is more invasive. The manual approach described here is simpler for this codebase.

---

## Appendix: Sonata Model Architecture Reference

### Forward Pass Flow
```
Input: Point cloud features (N, in_channels=6)
  |
Embedding: Linear(6, 48) + norm + act
  |
Encoder Stage 0: 3 x Block(48, 3_heads, ...)
  |
Encoder Stage 1: GridPooling(48->96) + 3 x Block(96, 6_heads, ...)
  |
Encoder Stage 2: GridPooling(96->192) + 3 x Block(192, 12_heads, ...)
  |
Encoder Stage 3: GridPooling(192->384) + 9 x Block(384, 24_heads, ...)
  |
Encoder Stage 4: GridPooling(384->512) + 3 x Block(512, 32_heads, ...)
  |
Up-cast (up_cast_level=2):
  - Concatenate features from levels 2, 3, 4
  - Output: (N, 192+384+512=1088)
  |
Head (OnlineCluster):
  - Linear(1088, 2048) + GELU
  - Linear(2048, 256)
  - L2 normalize
  - WeightNorm Linear(256, 4096)
  |
Output: (N, 4096) logits for prototype assignment
```

### Transformer Block Structure
Each block contains:
- **CPE** (Conditional Positional Encoding): SubMConv3d + Linear
- **Attention**: LayerNorm -> SerializedAttention(QKV + output proj) -> LayerScale -> DropPath
- **MLP**: LayerNorm -> Linear(d, 4d) + GELU + Linear(4d, d) -> LayerScale -> DropPath
- Pre-norm residual connections throughout

### Current Initialization
- All Linear weights: `trunc_normal_(std=0.02)`
- All Linear biases: `zeros`
- All SubMConv3d: `trunc_normal_(std=0.02)`
- LayerScale initial: `1e-5`
- Mask token: `zeros`

### Key File Locations
| Component | File Path |
|-----------|-----------|
| Sonata Model | `pointcept/models/sonata/sonata_v1m1_base.py` |
| PT-v3m2 Backbone | `pointcept/models/point_transformer_v3/point_transformer_v3m2_sonata.py` |
| Optimizer Builder | `pointcept/utils/optimizer.py` |
| Scheduler Builder | `pointcept/utils/scheduler.py` |
| Training Engine | `pointcept/engines/train.py` |
| Config | `configs/lartpc/sonata_pretrain/archive/pretrain-sonata-v1m1-lartpc-v6.py` |
