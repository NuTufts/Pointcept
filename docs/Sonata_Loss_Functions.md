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

## How Prototype Vectors Are Updated

The prototype vectors $c_k$ are central to the clustering mechanism. Understanding how they are learned helps interpret what the model is actually optimizing.

### Prototype Storage

Prototypes are stored as the **weights of a linear layer** with weight normalization in the `OnlineCluster` head:

```python
self.prototype = weight_norm(nn.Linear(embed_channels, num_prototypes, bias=False))
```

The weight matrix has shape `[num_prototypes, embed_channels]` = `[4096, 512]`. Each row is a prototype vector $c_k$.

### Weight Normalization

The prototypes use `weight_norm`, which decomposes weights into magnitude and direction:

$$W = g \cdot \frac{V}{\|V\|}$$

Critically, the **magnitude is fixed to 1** and frozen:

```python
self.prototype.weight_g.data.fill_(1)
self.prototype.weight_g.requires_grad = False
```

This means only the **direction** of each prototype vector is learned. All prototypes live on the unit hypersphere in the 512-dimensional embedding space.

### Two Independent Sets of Prototypes

The model maintains **two separate clustering heads**, each with its own set of 4096 prototypes:

1. **`mask_head`**: Used by `mask_loss` and `roll_mask_loss`
2. **`unmask_head`**: Used by `unmask_loss`

Both student and teacher networks have copies of these heads.

### Which Loss Updates Which Prototypes

| Loss | Student Head Updated | Prototypes Affected |
|------|---------------------|---------------------|
| `mask_loss` | `student.mask_head` | `student.mask_head.prototype` |
| `roll_mask_loss` | `student.mask_head` | `student.mask_head.prototype` |
| `unmask_loss` | `student.unmask_head` | `student.unmask_head.prototype` |

So:
- **`mask_loss` + `roll_mask_loss`** jointly train one set of prototypes (in `mask_head`)
- **`unmask_loss`** trains a separate set of prototypes (in `unmask_head`)

### Teacher Prototype Updates

The **teacher's prototypes are NOT trained by gradient descent**. Instead, they are updated via exponential moving average (EMA) after each training step:

```python
# after_step()
teacher = momentum * teacher + (1 - momentum) * student
```

where momentum starts at 0.996 and increases to 1.0 over training. This provides stable, slowly-evolving targets for the student to match.

### Summary

All three losses influence prototype learning, but through two independent sets of prototypes. The prototypes are unit-normalized vectors in the 512-dimensional embedding space, evolving to represent distinct local geometric/feature patterns in your point cloud data. The teacher's prototypes provide stable targets via EMA, while the student's prototypes are directly optimized through backpropagation.

## Interpreting Loss Values

### Mathematical Basis

The loss is cross-entropy over K prototypes (default K=4096):

$$L = -\sum_{i}^{N} \sum_{k}^{K} Q_{ik} \log(p_{ik})$$

where:
- $Q_{ik}$ = teacher's probability for sample $i$ to belong to prototype $k$
- $p_{ik}$ = student's probability for sample $i$ to belong to prototype $k$
- Both $Q_{ik}$ and $p_{ik}$ normalize to 1 over $i$, i.e. $\sum_{i}^{N} Q_{ik} = 1$ and $\sum_{i}^{N} p_{ik} = 1$

Notes:
-  This $L$ term is for one batch -- or rather training update -- i.e. it sums over $N$ samples indexed by $i$.
-  $Q_{ik}$ comes from the Sinkhorn-Knopp calculation and is the optical transport plan
   between a uniformly distributed mass over $K$ clusters to a uniformly distributed mass over $N$ points.
-  This way of calculating the target makes sure that all the cluster prototypes are used.
-  This avoids representation collapse that might come about by all the sample feature vectors mapping to one or a few prototypes.
-  However, enforcing the transport across all prototypes does not mean that the $p_{ik}$ must be smeared out.
   As long as there are other samples in the the batch that map to other prototypes, an individual sample can
   have a low entropy distribution over the prototypes, i.e. have high probability only over a few prototypes.
   Indeed, this is what we are trying to achieve by minimizing with the cross-entropy above.
   So this works best if the batch has enough diversity of patterns. So we might play around with large batch sizes or accomulate samples over multiple batches.
   Is the code setup to do this for the Sinkhorn-Knopp calculation?
-  For our data, which can have large class imbalances, it seems we need to make sure that enough "semantic diversity" is always present.
-  The optimal transport cost comes from the negative-similarity between the clusters and spacepoint features: 

  $$ C_{ik}=[e^{-\frac{1}{\tau}z_i^Tc_{k}}]_{ik} $$

-  $p_{ik}$ is a probability mass for sample $i$ with feature vector, $z_i$, calculated by
   
  $$ p_{ik} = \frac{ e^{\frac{1}{\tau} z_i^T c_k }}{\sum_{k'}^{K}e^{\frac{1}{\tau} z_i^T c_{k'} }} $$

  where:

   - $i$ is the index over $N$ samples in the batch (i.e. the (downsampled) points)
   - $z_i$ is the feature vector, $z_i \in \mathbb{R}^{d}$, of a given point, and
   - $c_k$ is the vector, $c_k \in \mathbb{R}^{d}$ in feature space for prototype $k$ out of $K$ prototypes.
- I assume something like the Sinkhorn algorithm is being applied here. It takes in a cost matrix $C_{ik}$ above and the mass distributions over $N$ samples and $K$ prototypes.
  This is probably $\frac{1}{N}$ and $\frac{1}{K}$ for the samples and prototypes, respectively. The output of the Sinkhorn algorithm is the matrix,
  $Q_{ik}$, which is the transport plan for how the mass of sample $i$ is to be distributed over each of the $K$ prototypes.





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
