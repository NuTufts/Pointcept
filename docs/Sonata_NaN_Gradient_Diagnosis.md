# Sonata Training NaN/Inf Gradient Diagnosis and Fixes

**Date:** 2026-03-30
**Config:** `configs/lartpc/sonata_pretrain/archive/pretrain-sonata-v6-extbnb.py`
**Jobs:** `larsonata.v6.extbnb.373354.pax008` (epochs 1-7), `larsonata.v6.extbnb.resume.epoch7.382201.pax008` (resumed epoch 8, crashed)

## Symptom

Training with bfloat16 AMP periodically produces `inf` gradients in the CPE (Conditional Position Encoding) sparse convolution layer. The GradScaler handles these initially by skipping optimizer steps and halving its scale factor. However, the overflows accelerate over time, eroding the scale factor until training eventually produces a NaN loss and crashes with a CUDA `index out of bounds` assertion.

## Root Cause Analysis

### 1. Inf Gradients in the CPE Layer

The inf gradients occur exclusively in `module.student.backbone.enc.enc0.block{0,2}.cpe.0` -- the `SubMConv3d` sparse convolution at the finest voxel resolution (grid_size=0.25cm, 48 channels).

Key observations from the logs:
- Always **inf** (not NaN), indicating overflow rather than undefined operations
- Only 1-43 out of 62,208 weight elements overflow per event
- The `finite_abs_max` of gradients is small (0.05-0.43), so the vast majority of gradients are well-behaved
- Different batches trigger it -- not caused by specific "bad" samples

**Cause:** The CPE sparse convolution operates at the finest resolution with the most active voxels. During backpropagation, gradient accumulation through the sparse kernel can overflow bfloat16's limited 8-bit mantissa (~3 decimal digits of precision). Certain voxel neighborhood patterns concentrate gradient flow through a small number of kernel elements, pushing them past representable values.

### 2. GradScaler Death Spiral

The config uses `enable_amp=True` with `amp_dtype="bfloat16"`. However, GradScaler was designed for **float16**, which has only 5 exponent bits and genuinely needs loss scaling to prevent gradient underflow. **bfloat16 has 8 exponent bits (same as float32) and does not need loss scaling.**

Using GradScaler with bfloat16 creates a positive feedback loop:

1. CPE produces occasional inf gradients (a bfloat16 precision issue)
2. GradScaler detects the overflow, skips the step, and halves the scale factor
3. Lower scale factor means smaller gradient magnitudes, increasing the chance of further precision issues
4. More overflows occur, further eroding the scale factor
5. Eventually the scale becomes so low that effective gradients underflow, causing NaN

The scale factor erosion in the crashed run (epoch 8):

| Step | Scale | Overflows | Gap (steps) |
|------|-------|-----------|-------------|
| 1644 | 2^19 -> 2^18 | 1 | -- |
| 3776 | 2^19 -> 2^18 | 2 | 2132 (recovered) |
| 3918 | 2^18 -> 2^17 | 3 | 142 |
| 4864 | 2^17 -> 2^16 | 4 | 946 |
| 7140 | 2^17 -> 2^16 | 5 | 2276 (recovered) |
| 7408 | 2^16 -> 2^15 | 6 | 268 |
| 7632 | 2^15 -> 2^14 | 7 | 224 |
| 8206 | 2^14 -> 2^13 | 8 | 574 |
| 8535 | 2^13 -> 2^12 | 9 | 329 |
| 8576 | 2^12 -> 2^11 | 10 | **41** |
| 8767 | 2^11 -> 2^10 | 11 | 191 |
| 9243 | 2^10 -> 2^9 | 12 | 476 |
| **10025** | -- | **NaN crash** | 782 |

Note the accelerating pattern: early overflows are spaced 1000-2000 steps apart, but later they cluster within 40-200 steps.

### 3. The Crash Sequence

At iteration 10025 with scale=512 (2^9):

1. Forward pass produces **NaN in `mask_loss` and `roll_mask_loss`** (but NOT `unmask_loss`)
2. Both losses use `F.log_softmax(pred / student_temp)` -- when precision degrades, softmax inputs become degenerate, causing `log(0) = -inf`, which multiplied by target probabilities yields NaN
3. The code logged the NaN but still called `backward()`
4. NaN values propagated through the backward pass, corrupting index tensors
5. Corrupted indices (NaN cast to int = garbage) were passed to `torch.scatter`/`torch.gather`
6. CUDA kernel assertion: `idx_dim >= 0 && idx_dim < index_size && "index out of bounds"`
7. Device-side assert killed the process

## Fixes Implemented

### Fix 1: Disable GradScaler for bfloat16

**File:** `pointcept/engines/train.py` (`build_scaler` method)

`build_scaler()` now returns `None` when `amp_dtype == "bfloat16"`. GradScaler is only created for float16 AMP. All scaler usage in `run_step()` is guarded with `self.scaler is not None`, and checkpoint save/load in `pointcept/engines/hooks/misc.py` handles `scaler=None` gracefully.

**Impact:** Eliminates the death spiral entirely. Without GradScaler, bfloat16 AMP operates with its native dynamic range, which matches float32 and does not need loss scaling.

### Fix 2: Run CPE Sparse Convolution in float32

**File:** `pointcept/models/point_transformer_v3/point_transformer_v3m2_sonata.py` (`Block.forward`)

The CPE block now:
1. Casts `point.feat` and `sparse_conv_feat` to float32 before the CPE
2. Wraps the CPE call in `torch.amp.autocast("cuda", enabled=False)` to prevent re-casting
3. Casts back to the original dtype after CPE completes

**Impact:** Prevents the inf gradients at their source. The SubMConv3d gradient accumulation now has full float32 precision, avoiding the overflow that triggered the entire cascade.

### Fix 3: Clamp log_softmax in Sonata Loss

**File:** `pointcept/models/sonata/sonata_v1m1_base.py` (loss computation)

Added `.clamp(min=-100)` to all three `F.log_softmax()` calls (mask_loss, roll_mask_loss, unmask_loss). This prevents `-inf` from `log(0)` from propagating as NaN through the cross-entropy computation.

**Impact:** Even if softmax produces near-zero probabilities due to precision issues, the loss remains finite rather than becoming NaN.

### Fix 4: Skip NaN Loss Batches

**File:** `pointcept/engines/train.py` (`run_step` method)

When NaN/Inf loss is detected, the batch is now **skipped entirely**: gradients are zeroed, the accumulation counter is reset, and the method returns early. Previously, the code logged the error but still called `backward()`, which caused NaN to corrupt index tensors and trigger the fatal CUDA assertion.

**Impact:** Adds resilience so that occasional NaN batches do not crash training. The training log will still report the NaN event for monitoring.

## Monitoring Recommendations

After applying these fixes, monitor the following in wandb/logs:

1. **NaN batch skip frequency** -- If NaN batches occur more than ~1 per epoch, investigate whether specific samples or data transforms are producing degenerate inputs.
2. **Inf gradient frequency** -- Fix 2 should dramatically reduce or eliminate these. If they persist, the issue may be in a different layer.
3. **Feature std** -- The `FeatureStdMonitor` tracks representation collapse. Healthy values should remain in the range 1.5-3.0 for both student and teacher.
4. **Channel std max** -- The max channel std was reaching 8-9x during the crashed run (e.g., `max=8.7302`). If this grows monotonically, it may indicate an unstable channel that could seed future overflows.
