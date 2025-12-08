# Sonata Loss Functions

This document explains the loss functions used in the Sonata self-supervised pre-training framework and how to interpret their values during training.

## Overview

Sonata uses a **teacher-student architecture** (similar to DINO/SwAV) with online clustering. The model learns representations by predicting cluster assignments, where:

- **Student network**: Trainable, receives augmented/masked views
- **Teacher network**: Exponential moving average (EMA) of student weights, provides target assignments

All three losses compute **cross-entropy** between:
- **Student**: Softmax-normalized predictions (temperature = 0.1)
- **Teacher**: Sinkhorn-Knopp normalized cluster assignments (temperature ~0.04-0.07)

## Loss Functions

### 1. `mask_loss` (default weight: 2/8)

**Purpose**: Masked point cloud reconstruction.

The student sees a **masked** version of the global point cloud where random patches are hidden or jittered (controlled by `mask_size` and `mask_ratio` parameters). The teacher sees the **unmasked** version.

The loss encourages the student to predict the same cluster assignments as the teacher for spatially matched points between the masked and unmasked views.

```
Student input: masked global view → predict cluster assignments
Teacher input: unmasked global view → ground truth cluster assignments
Loss: Cross-entropy between student predictions and teacher's Sinkhorn-Knopp assignments
```

**What it teaches**: Robustness to missing data; ability to infer masked regions from context.

### 2. `roll_mask_loss` (default weight: 2/8)

**Purpose**: Cross-view consistency with masking.

This requires `num_global_view >= 2`. The "roll" operation swaps the two global views:
- Original order: `[view1_A, view1_B, view2_A, view2_B]`
- After roll: `[view1_B, view1_A, view2_B, view2_A]`

The student predicts from a masked view, but the teacher target comes from a **different augmentation** of the same scene (the "rolled" view).

```
Student input: masked global view A → predict cluster assignments
Teacher input: unmasked global view B (rolled) → ground truth cluster assignments
Loss: Cross-entropy between student predictions and rolled teacher assignments
```

**What it teaches**: Features that are invariant to augmentations and robust to masking simultaneously.

### 3. `unmask_loss` (default weight: 4/8)

**Purpose**: Local-to-global view consistency (no masking).

The student processes **local views** (smaller crops), while the teacher processes the **principal global view** (unmasked). Points are matched by spatial proximity between local and global views.

```
Student input: local view (smaller crop) → predict cluster assignments
Teacher input: principal global view (unmasked) → ground truth cluster assignments
Loss: Cross-entropy between local student predictions and global teacher assignments
```

**What it teaches**: Scale-invariant features; local regions should have representations consistent with the global context.

### Combined Loss

```python
loss = mask_loss * (2/8) + roll_mask_loss * (2/8) + unmask_loss * (4/8)
```

The `unmask_loss` has the highest weight because local-to-global consistency is the primary pretext task, while the masking losses provide additional regularization.

## Interpreting Loss Values

### Mathematical Basis

The loss is cross-entropy over K prototypes (default K=4096):

$$L = -\sum_{k=1}^{K} q_k \log(p_k)$$

where:
- $q_k$ = teacher's soft assignment for prototype k
- $p_k$ = student's predicted probability for prototype k

Notes:
-  This $L$ term is for one "sample", i.e. for one point with some point set of N points.
-  $q_k$ comes from the Sinkhorn-Knopp calculation, which finds the optical transport plan
   between a uniformly distributed mass over $K$ clusters to a uniformly distributed mass over $N$ points.
   The optimal transport cost comes from the entropy(?) between the clusters and spacepoint features: 

     $$ C_{ik}=[e^{-\frac{1}{\tau}z_i^Tc_{k}}]_{ik} $$

-  $p_k$ is a probability calculated from
   
     $$ p_k = \frac{ e^{\frac{1}{\tau} z^T c_k }}{\sum_{k'}^{K}e^{\frac{1}{\tau} z^T c_{k'} }} $$

   where:
     - $z$ is the feature vector, $z \in R^{d}$, of a given point, and
     - $c_k$ is the vector in feature space for prototype, $k$.





### Random Baseline

At initialization, both networks produce approximately uniform distributions:

$$L_{random} = -\sum_{k=1}^{K} \frac{1}{K} \log\left(\frac{1}{K}\right) = \log(K)$$

With K = 4096:

$$L_{random} = \ln(4096) = 12 \ln(2) \approx 8.32$$

### Loss Value Reference Table

| Loss Value | Interpretation |
|------------|----------------|
| ~8.3 | Random predictions (maximum entropy over 4096 prototypes) |
| ~2.0-2.5 | Typical mid-training values; structured representations emerging |
| ~1.5-2.0 | Well-trained model with confident cluster assignments |
| ~0 | Perfect prediction (or potential mode collapse - verify carefully) |

### Effective Number of Clusters

A loss value L corresponds to an "effective" number of clusters:

$$N_{effective} = e^L$$

For example:
- Loss = 2.0 → ~7.4 effective clusters
- Loss = 2.5 → ~12.2 effective clusters
- Loss = 1.5 → ~4.5 effective clusters

This represents the entropy of the predictions, not the actual number of prototypes used.

## What Does the Loss "Settle" To?

The final loss value depends on multiple factors and does **not** directly correspond to the number of semantic classes in your data.

### Factors Affecting Final Loss

| Factor | Effect |
|--------|--------|
| Intrinsic data complexity | More distinct local patterns → lower achievable loss |
| Noise/ambiguity in data | More noise → higher loss floor |
| Temperature settings | Lower temperatures → sharper assignments → lower loss |
| Prototype count vs. true structure | Mismatch can increase loss |

### Prototypes vs. Semantic Classes

For LArTPC data with 6 semantic classes (electron, muon, pion, proton, gamma, ghost), you might expect the loss to settle around ln(6) ≈ 1.79. However, this is typically **not** what happens because:

1. **Prototypes capture finer structure**: The 4096 prototypes learn local geometric/feature patterns, not semantic categories. Within "muon" alone, the model might learn:
   - Straight track segments
   - Track endpoints
   - Scattering vertices
   - Different dE/dx patterns

2. **Sinkhorn-Knopp forces utilization**: The normalization ensures all prototypes are used roughly equally, forcing the model to find distinguishable patterns across all 4096 clusters.

3. **Sub-class structure**: The model discovers structure at multiple scales:
   - Semantic level: 6 classes
   - Sub-class level: Track vs. shower topology, Bragg peaks, delta rays
   - Geometric level: Local curvature, density, wire plane response patterns

### Evaluating Representation Quality

**The loss value alone does not tell you if learned representations are useful for your downstream task.**

The standard evaluation approaches are:

1. **Linear probing**: Freeze the backbone, train only a linear classifier on labeled data
2. **Fine-tuning**: Train the full model on the downstream task
3. **kNN evaluation**: Use k-nearest neighbors on frozen features

For LArTPC, evaluate on your semantic segmentation task at various checkpoints (e.g., 25%, 50%, 75%, 100% of training).

## Training Dynamics

### Expected Behavior

1. **Initial drop**: Loss falls rapidly from ~8.3 as model learns basic structure
2. **Warmup bumps**: Around 5% of training, Sonata's schedulers transition from "easy" to "hard" settings:
   - `mask_size`: 0.1 → 0.4
   - `mask_ratio`: 0.3 → 0.7
   - `teacher_temp`: 0.04 → 0.07

   This can cause temporary loss increases.
3. **Gradual descent**: Continued improvement with high batch-to-batch variance
4. **Plateau**: Eventually settles to a data-dependent floor

### Warning Signs

| Symptom | Potential Issue |
|---------|-----------------|
| Loss drops to ~0 quickly | Mode collapse (trivial solution) |
| Loss plateaus early, never improves | Learning rate too low or architecture problem |
| Loss diverges (increases steadily) | Learning rate too high |
| All three losses become identical | Representation collapse |
| Extremely high variance that doesn't decrease | Batch size too small or learning rate too high |

## Key Model Components

| Component | Purpose |
|-----------|---------|
| `OnlineCluster` head | MLP that projects features to prototype space |
| `sinkhorn_knopp()` | Soft cluster assignment normalization for teacher (prevents collapse) |
| `match_neighbour()` | Spatial matching between views (default radius < 0.08) |
| EMA teacher update | teacher = momentum * teacher + (1-momentum) * student |

## References

- Sonata implementation: `pointcept/models/sonata/sonata_v1m1_base.py`
- Configuration example: `configs/lartpc/pretrain-sonata-v1m1-lartpc.py`
