# LArFormer — Design and Implementation Plan

**Status:** P1–P6 implemented (scaffold → multi-level voxel → fragment builder → `LArFormerDataset` + GT visualizer → Stage-1 deghoster → Stage-2 cascaded slicer). Currently in slicer-debug iteration on a 10-event dev sample; see §15. A **`TokenRefiner`** abstraction (§16) was added to sit between the tokenizer and the decoder so multi-scale feature refinement can be ablated independently of the rest of the pipeline. P7 (particle clusterer) not started.
**Owner:** taritree.wongjirad@tufts.edu
**Generalizes:** [`ShowerClusteringMask2Former`](../pointcept/models/shower_clustering/model.py) (kept frozen for the trained shower-origin pipeline).
**Lives at:** `pointcept/models/LArFormer/`.

This document is the living design reference. Update as decisions change or phases complete.

---

## 1. Motivation

`ShowerClusteringMask2Former` is a hierarchical Mask2Former operating on three hardcoded token streams: spacepoints, 5 cm voxels, and DBSCAN fragments. The architecture is sound, but the structure is wired such that:

- The decoder enumerates exactly three scale names (`voxel`, `fragment`, `spacepoint`) — `SCALES` constant, `mask_logits` dict keys, attn-mask dispatch.
- The tokenizer builds exactly these three streams.
- The loss has hand-coded `_build_voxel_gt_mask` / `_build_fragment_gt_mask`, and a per-scale loss weight per term.
- Voxel resolution is fixed by the dataset (5 cm).

The result is a model that's hard to repurpose for related tasks (deghosting, event slicing, particle clustering) and hard to ablate over multi-resolution voxel hierarchies.

**LArFormer** lifts the same architecture into a configurable list of *levels*. A level is any per-event token stream — spacepoints, voxels at any resolution, DBSCAN fragments, or anything else with a registered builder. The decoder's cross-attention pattern, the per-level supervision (mask + optional per-token classifier), and the loss budget are all driven by the config.

The same scaffold then carries three downstream tasks for the MicroBooNE Gen-2 pipeline:

1. **Deghosting** — per-spacepoint binary `real / ghost`.
2. **Event slicing** — per-spacepoint instance assignment into cosmic-primary slices + a merged nu-vertex slice (see [Event_Slicer_Spec.md](Event_Slicer_Spec.md)).
3. **Particle clustering of the neutrino slice** — instance segmentation of the nu slice into per-particle trajectories.

These are trained sequentially as a **cascade**: deghoster first, then slicer with frozen deghoster, then particle clusterer with frozen upstream. The cascade pattern is described in §6.

---

## 2. Core abstraction: `Level`

A **Level** is a per-event token stream defined by:

```python
{
    "name":       "voxel_10cm",          # stable key — used for mask logits / logging
    "builder":    "voxel",                # registered builder type
    "builder_cfg": {"voxel_size_cm": 10.0},
    "supervision": {                      # all sub-keys optional
        "mask": {"weight": 1.0, "mode": "aux"},
        "cls":  {"num_classes": 3,
                 "label_src": "origin_label",
                 "reduce": "plurality",   # spacepoint→level aggregation
                 "weight": 1.0,
                 "loss": "ce"},
    },
}
```

A registered **`LevelBuilder`** takes the per-spacepoint backbone features plus the per-event data dict and returns:

| field             | shape       | meaning                                                  |
|-------------------|-------------|----------------------------------------------------------|
| `tokens`          | (M, D)      | per-token feature vector                                  |
| `coords`          | (M, 3)      | per-token position in normalized coords (for the decoder's pos-emb) |
| `sp_to_level_id`  | (N,)        | per-spacepoint mapping into [0, M) (or −1 if a SP is unmapped at this level) |

The third return is the key generalization. Any per-spacepoint truth array (`origin_label`, `slice_id`, `pid`, `hasmatch`, …) can be lifted into a per-level GT mask or per-level class target by scatter-reducing through `sp_to_level_id`. This subsumes the existing hand-coded `_build_voxel_gt_mask` / `_build_fragment_gt_mask`.

### v1 builders

| Builder       | Tokens per event | `tokens` source                | `coords` source         | `sp_to_level_id`                 |
|---------------|-------------------|--------------------------------|--------------------------|----------------------------------|
| `spacepoint`  | N                | `Linear(sp_feat)`              | `coord_norm`             | `arange(N)`                      |
| `voxel`       | V (config size) | `mean_pool(sp_feat, voxel_id)` | voxel-center coords      | `floor(coord_norm/v)` + unique  |
| `fragment`    | F (DBSCAN)      | `FragmentPool(sp_feat, frags)` | per-fragment centroids   | union of `fragment_indices`, or −1 if SP not in any fragment |

Voxel builders are **model-side** (no dataset change to switch resolution — just a config edit). Multiple voxel levels with different `voxel_size_cm` can coexist; each gets its own unique `name`.

The fragment builder is **self-contained** under `pointcept/models/LArFormer/builders/fragment.py` — `FragmentPool` and `FragmentContentEnricher` are copied in rather than imported from `pointcept/models/shower_clustering/`. LArFormer must have zero import dependencies on the legacy module so the two can evolve independently. For tasks that don't use fragments, this builder is simply not listed and the dataset's `fragment_indices` field is ignored.

Adding a new builder later (e.g., flash-conditioned pseudo-clusters for the slicer) means writing one `LevelBuilder` subclass and registering it; no decoder/loss changes.

---

## 3. Configurable scale patterns

The decoder's `scale_pattern` is an ordered list of level names. Each layer's masked cross-attention attends to the listed level. The default Mask2Former-style coarse-to-fine pattern for the slicer might look like:

```python
levels = [
    Level("voxel_20cm", builder="voxel", cfg=dict(voxel_size_cm=20)),
    Level("voxel_10cm", builder="voxel", cfg=dict(voxel_size_cm=10)),
    Level("voxel_5cm",  builder="voxel", cfg=dict(voxel_size_cm=5)),
    Level("spacepoint", builder="spacepoint"),
]
scale_pattern = ["voxel_20cm", "voxel_10cm", "voxel_10cm",
                 "voxel_5cm",  "voxel_5cm",  "spacepoint"]
```

A fragment-aware shower-clustering setup (close to today's `ShowerClusteringMask2Former`) becomes:

```python
levels = [
    Level("voxel",      builder="voxel", cfg=dict(voxel_size_cm=5)),
    Level("fragment",   builder="fragment"),
    Level("spacepoint", builder="spacepoint"),
]
scale_pattern = ["voxel", "fragment", "voxel", "fragment",
                 "spacepoint", "spacepoint"]
```

Decoder layer count = `len(scale_pattern)`. All levels listed anywhere in `scale_pattern` are built exactly once per event (cached, reused across layers + the mask head).

---

## 4. Per-level supervision

Two independent supervisions per level, both optional:

### 4a. Per-level mask supervision (set loss)

Every level listed in `supervision.mask` gets a BCE mask loss against the lifted GT mask `(K, M_level)` at every decoder layer. One level is declared **primary** — that level uses the existing PointRend-style per-pair point sampling + Dice (expensive but high signal). All others use full-mask BCE (cheap, fine because token count is small).

Default selection: if `spacepoint` is in `scale_pattern`, it's primary. Otherwise the highest-resolution voxel level is primary.

Per-level mask losses respect a token-count guard: if a level's `M_level` exceeds `aux_max_tokens` (default 10 000), the aux BCE silently falls back to the same point-sampled scheme as the primary. This keeps a 20 cm voxel scale (~hundreds of tokens) and a spacepoint scale (~100k) from both running full-mask BCE.

### 4b. Per-level classification supervision (per-token)

If a level declares `supervision.cls`, a small linear head is attached to the level's tokens and trained against a per-token target lifted from a named per-spacepoint truth field. Concretely:

```python
target_per_level = scatter_reduce(
    src       = label_remap_if_set(data["spacepoint"][label_src]),  # (N,)
    index     = sp_to_level_id,                                      # (N,)
    dim_size  = M_level,
    reduce    = "amax" | "plurality",
)
```

`label_src` names a field already in the dataset (e.g., `hasmatch`, `origin_label`, `slice_id`, `pid`). `reduce="plurality"` is implemented as scatter-mode (most common label per level). `reduce="amax"` is right for binary labels (deghosting) and for **priority pooling** (see below). `mean` is not exposed — token cls is an integer label, not a regression target.

**Priority pooling via `label_remap`.** For multi-class labels where one class should dominate ("any voxel touched by a nu spacepoint should be a nu voxel, even if cosmics outnumber"), set `label_remap` so the priority class has the largest integer code, then use `reduce="amax"`. Example for slicer per-voxel cls on the dataset's `origin_label = {0=ghost, 1=nu, 2=cosmic}`:

```python
cls=dict(
    num_classes=3,
    label_src="origin_label",
    label_remap={0: 0, 1: 2, 2: 1},   # post-remap: 0=ghost, 1=cosmic, 2=nu
    reduce="amax",                     # any-nu voxel → nu; else any-cosmic → cosmic; else ghost
    weight=0.5, loss="ce", ignore_index=-1,
)
```

`label_remap` is identity for any keys not listed, and `ignore_index` entries are passed through unremapped. This keeps the priority semantics in the model config rather than baking them into the dataset, so the same `LArFormerDataset` instance can serve multiple tasks with different priority orders.

This is the single mechanism that handles:

- per-spacepoint `real / ghost` (deghoster)
- per-voxel `nu / cosmic / no-slice` (slicer aux)
- per-spacepoint `origin_type` (shower clustering, optional)
- per-spacepoint `pid` (particle clustering, optional)

The class set, target field, and loss weight are config-only. No code changes between tasks.

### 4c. Query-side supervision (set-prediction class)

The decoder's per-query class head and Hungarian matcher stay essentially as today — the *query* class is decoupled from per-level classification. Per-query class is the "what instance type is this slot": origin_type for shower clustering, `nu / cosmic` for the slicer, particle PID for the particle clusterer.

---

## 5. Loss budget defaults

Inheriting from `ShowerClusteringLoss`, generalized. Default weights below are the spec defaults; the v1 cascaded-slicer configs have tuned values noted in the rightmost column (see [`larformer-slicer-v1-cascaded-loradeghost.py`](../configs/lartpc/larformer-slicer-v1-cascaded-loradeghost.py) for the canonical set).

| Component                                          | Spec default | Slicer-v1 value | Notes                                              |
|----------------------------------------------------|--------------|-----------------|----------------------------------------------------|
| Query class CE (matched)                           | 2.0          | 2.0             | Hungarian-matched query → GT class                 |
| Query class CE (no-object)                         | × 0.1        | × 0.1           | Down-weight for unmatched queries                  |
| Primary-level mask BCE (per-pair, sampled)         | 5.0          | 5.0             | Point-sampled à la PointRend; balanced pos/neg     |
| Primary-level Dice                                 | 5.0          | 5.0             | Same sampled set                                   |
| Aux mask BCE per non-primary level                 | 1.0          | 0.3             | Full-mask BCE if `M_level ≤ aux_max_tokens` (=20 000) |
| Per-level cls CE (if declared)                     | 1.0          | 0.5             | Per-token CE on the level's tokens                 |
| Origin L1 (matched, if origin head on)             | 1.0          | 0.0 / 0.5       | 0 = head off (default); 0.5 when `ENABLE_ORIGIN_HEAD_WITH_CENTROID=True` |

**`num_sample_points`** for the primary BCE/Dice sampler is **8192** in the slicer-v1 configs (bumped from the 4096 default after dropping the `lm_score` pre-filter and so per-event SP counts ~3× larger).

**Importance sampling** (PointRend-style hard-negative mining for the per-pair mask losses) is enabled in slicer-v1: `use_importance_sampling=True`, `importance_oversample_ratio=3.0`, `importance_ratio=0.75`. Negatives are drawn preferentially from the `|sigmoid(logit) - 0.5|` halo around the predicted mask boundary. Ported from `shower_clustering` where it gave a measurable uplift on the same set-prediction loss shape.

Deep supervision: all the above are applied at the init prediction + every decoder layer, then summed. This is unchanged from `ShowerClusteringLoss`.

---

## 6. The cascade

Three independently trained `LArFormer` instances. Each is a standard model file with its own config; the chain is only assembled at inference time and (later) when a downstream stage is jointly fine-tuned with frozen upstream.

```
spacepoints (raw, with ghosts)
        │
        ▼  Stage 1: LArFormer-deghost
            per-SP cls head on the spacepoint level → real/ghost score
        │  threshold drops ghost spacepoints
        ▼
deghosted spacepoints
        │
        ▼  Stage 2: LArFormer-slicer
            query set predicts slice instances over voxel + spacepoint levels
            optional flash-match loss (see Event_Slicer_Spec.md §1)
        │  one slice selected (matched to in-time beam flash) as nu candidate
        ▼
nu-slice spacepoints
        │
        ▼  Stage 3: LArFormer-particle-cluster
            query set predicts per-particle trajectories
            per-query class = particle PID
        ▼
per-particle instance segmentation of the nu interaction
```

Training order, per user direction:

1. **Stage 1 alone.** Standard supervised training against `hasmatch`.
2. **Stage 2 with stage 1 frozen.** Slicer dataset uses spacepoints filtered by the frozen deghoster's output. The deghoster's per-SP score can also be carried in as an extra feature (cheap; the slicer doesn't have to relearn the signal).
3. **Stage 3 with stages 1+2 frozen.** Particle clusterer sees only the matched nu slice from stage 2.

Joint fine-tuning (unfreezing all three stages end-to-end) is a future option and not in scope for v1.

### Conditioning mechanisms

Open spec, expected to land per-stage as the cascade is built:

- **Stage 1 → Stage 2.** Deghoster's per-SP `real` score is added to the slicer's input `feat`. The slicer's input set is the filter `score > τ` (τ sampled at train time, fixed at eval).
- **Stage 2 → Stage 3.** Slicer's chosen nu-slice mask defines stage 3's input spacepoints. Slice metadata (flash time, vertex position) can ride along as extra per-event tokens.

**Implementation note (Stage 1 → Stage 2).** v1 ended up doing the cascade *model-side* rather than dataset-side: [`CascadedSlicer`](../pointcept/models/LArFormer/cascaded.py) wraps the deghoster + slicer as one `nn.Module`. Forward: run the deghoster, threshold its per-SP `p(real)` against a τ sampled per-batch (`U(min,max)` in train, `val` at eval), call [`filter_batch_by_keep_mask`](../pointcept/models/LArFormer/cascade_filter.py) to drop ghost SPs from every per-SP tensor + recompute `offset`, then run the slicer on the surviving SPs. GT instances are filtered to the survivors too (`filtered_gt_instances_per_event` is exposed on the eval output so the evaluator's Hungarian uses the post-filter K). This keeps the dataset task-agnostic; the model knows about the cascade.

Two cascade variants exist:

- [`larformer-slicer-v1-cascaded.py`](../configs/lartpc/larformer-slicer-v1-cascaded.py) — LArFormer-flavored Stage-1 deghoster (per-token cls on the spacepoint level, class 1 = real). The deghoster is a separately-trained `LArFormer` checkpoint produced from [`larformer-deghost-v0.py`](../configs/lartpc/larformer-deghost-v0.py).
- [`larformer-slicer-v1-cascaded-loradeghost.py`](../configs/lartpc/larformer-slicer-v1-cascaded-loradeghost.py) — uses the existing trained [`SonataLoRADeghostSegmentor`](../pointcept/models/sonata_lora_deghost.py) as Stage 1 (LoRA-finetuned Sonata-v1m1, class 0 = real via the HasmatchAsGhost convention). `CascadedSlicer._run_deghoster_p_real` accepts either output convention and picks the right softmax column based on `deghoster_class_index_real` on the cascade config. **This is the active variant for the current debug effort** because the LoRA deghoster is already trained and gives `real_recall=0.65` / `ghost_reject=0.83` at τ=0.5 on the dev sample.

---

## 7. Dataset: `LArFormerDataset`

A **new** dataset, registered alongside `ShowerClusteringDataset` (which stays untouched for the deployed shower-origin model). Both read the same merged H5 files. Differences from `ShowerClusteringDataset`:

| Aspect                | `ShowerClusteringDataset`                    | `LArFormerDataset`                                  |
|-----------------------|----------------------------------------------|------------------------------------------------------|
| Voxelization          | Hardcoded 5 cm grid; emits `voxel_id`, `voxel_keys` | Not done in dataset — moved to model-side builders |
| Fragment fields       | Required (shower DBSCAN)                     | Optional; emitted iff `emit_fragments=True`         |
| `gt_instances` source | Trunk-walk on `mc_particle_tree`             | Pluggable: `gt_source="shower_trunk" / "slice" / "particle" / "deghost"` |
| Per-SP labels         | All of trackid / pid / origin / ssnet / hasmatch | Same (deghost / slicer / cluster all need subsets)  |
| Slice GT              | Not emitted                                  | Emitted iff `gt_source="slice"`, via `slice_labels.py` |

The instance-source plug is a callable `(per_sp_truth, mc_particle_tree) → list[gt_instance_dict]`. Implementations:

- `shower_trunk`: existing logic from `ShowerClusteringDataset` (trunks → descendant sets).
- `slice`: existing [`lartpc_data_prep/slice_labels.py`](../lartpc_data_prep/slice_labels.py) — one instance per cosmic primary, one merged nu instance.
- `particle`: one instance per Geant4 primary trajectory; filtered to the input slice for stage 3.
- `deghost`: pseudo-empty (no instance segmentation needed; only the per-SP cls head is trained). A single "all-real" instance can stand in for the matcher or queries are disabled entirely.

The collate function mirrors `shower_clustering_collate` but drops the voxel-specific fields. Fragment fields are only carried when `emit_fragments=True`.

Implementation file: `pointcept/datasets/larformer.py`.

---

## 8. Backbone handling

v1: take the backbone via subconfig and build it through `MODELS.register_module` — same pattern as `ShowerClusteringMask2Former.__init__`. Supports any registered Sonata variant (frozen, LoRA-fine-tuned, or full-finetune); the LArFormer model itself only needs `freeze_backbone` (bool) to know whether to wrap the forward pass in `torch.no_grad`.

**Future extension (not v1):** allow the LArFormer config to wrap the backbone with LoRA adapters in-place, with `lora_cfg` controlling rank / target modules / scaling. This is cleaner than building a separate "deghost-tuned backbone" model whose adapters are baked into the checkpoint, and lets one base Sonata checkpoint serve all three cascade stages with stage-specific LoRA deltas. Defer until v1 is working.

---

## 9. File layout

```
pointcept/models/LArFormer/
├── __init__.py
├── model.py                # LArFormer top-level (analog of ShowerClusteringMask2Former)
├── tokenizer.py            # CompositeTokenizer: iterates builders, returns Level list
├── decoder.py              # Mask2FormerDecoder generalized to named levels
├── losses.py               # LArFormerLoss generalized to per-level supervision
├── matcher.py              # HungarianMatcher (lifted, parameterized by primary level)
├── heads.py                # PerTokenClsHead, FlashMatchHead (stub for slicer)
└── builders/
    ├── __init__.py         # BUILDERS registry + Level dataclass
    ├── base.py             # LevelBuilder ABC; Level / LevelOutput dataclasses
    ├── spacepoint.py       # SpacepointBuilder (identity)
    ├── voxel.py            # VoxelBuilder (model-side voxelization)
    └── fragment.py         # FragmentBuilder (reuses FragmentPool/Enricher)

pointcept/datasets/
└── larformer.py            # LArFormerDataset + larformer_collate

tools/
└── visualize_larformer_gt.py   # config-driven GT visualizer (see §11)
```

Existing `pointcept/models/shower_clustering/` and `pointcept/datasets/shower_clustering.py` are not touched in v1. LArFormer is self-contained — no imports from `shower_clustering/` (the fragment pool / content enricher are copied into `builders/fragment.py`).

---

## 10. Forward API

```python
out = model(data_dict)
if model.training:
    loss = out["loss"]
    # plus per-component scalars: loss_cls, loss_mask_primary, loss_dice,
    # loss_aux_mask_<level>, loss_cls_<level>, loss_origin, ...
else:
    # out["predictions"] = list[B] of:
    #   {
    #     "class_logits":   (Q, C),
    #     "origin":         (Q, 3),
    #     "mask_logits":    {<level_name>: (Q, M_level)},
    #     "per_level_cls":  {<level_name>: (M_level, C_level)},   # if cls head declared
    #     "matched":        (q_idx, k_idx)   # only if not in pure inference
    #   }
```

Per-event slicing logic is the same as today: the backbone runs flat-batched once, then everything downstream loops over events because Hungarian matching is per-event.

---

## 11. Ground-truth visualizer

A standalone tool, [`tools/visualize_larformer_gt.py`](../tools/visualize_larformer_gt.py) (TBD), for visually checking the GT labels produced at each level. The design constraint: **the visualizer must not reimplement any of the level construction or GT lifting** — it queries the same code paths the model and loss use. If the visualizer and the trainer ever disagree, the bug is in shared code, not in two parallel implementations.

### What it shows

For each event in a user-selected sample, for each level declared in the config:

- 3D scatter of the level's token coords, colored by:
  - **`instance_id`** — which GT instance each token belongs to (per Hungarian-target source: shower trunk, slice primary, particle trajectory). Tokens unassigned to any instance are gray.
  - **`cls_target`** — the per-level classification target (if the level declares `supervision.cls`).
- Per-spacepoint overlay showing `sp_to_level_id` (helpful for sanity-checking voxel binning or fragment membership).
- Side panel listing instance metadata (origin_type / pid / primary_trackid / origin_coord, depending on `gt_source`).

The 3D widget reuses the same plotly conventions as [`tools/visualize_lartpc_h5data.py`](../tools/visualize_lartpc_h5data.py) for consistency. For spacepoint-in-detector rendering patterns (TPC bounds, axis orientation, hover tooltips, instance-color cycling), refer to [`tools/visualize_shower_clustering.py`](../tools/visualize_shower_clustering.py) and [`tools/visualize_slice_flash_match.py`](../tools/visualize_slice_flash_match.py) — both are already-working examples of the same rough idiom (load H5 → render coords + per-point labels in a detector-frame 3D scatter).

### Required convenience hooks (drives a small API contract)

To make the visualizer trivially correct, the model and loss expose two pure helpers that the visualizer calls directly:

```python
# In LArFormer/model.py (no backbone forward, no decoder, no matcher):
@torch.no_grad()
def build_levels(self, data_dict) -> dict[str, LevelOutput]:
    """Run every level builder using zero-filled per-SP features (we only
    need coords + sp_to_level_id for GT visualization). Returns the same
    LevelOutput objects the forward pass uses."""

# In LArFormer/losses.py (or a sibling gt_targets.py used by both loss and viz):
def build_per_level_gt(
    data_dict, levels: dict[str, LevelOutput],
    gt_instances: list[dict],
) -> dict[str, dict]:
    """Returns:
        {<level_name>: {
            "instance_mask": (K, M_level) float in {0, 1},
            "cls_target":    (M_level,) long or None,
        }, ...}
    """
```

`build_per_level_gt` is the function the training loss already calls internally to construct `gt_masks_voxel` / `gt_masks_fragment` (today) — so making it a public, importable helper is just an extraction, not new code. Same applies to the per-level cls target reducer (§4b).

### Usage

```bash
python tools/visualize_larformer_gt.py \
    --config       configs/larformer/larformer-slicer-v1.py \
    --h5           /path/to/merged_..._entry000000.h5 \
    --level        voxel_10cm,spacepoint \
    --color-by     instance_id
```

The config is the same one used for training; only the dataset block and the model's `levels` config are consulted (the backbone weights are not loaded because the visualizer doesn't run the forward pass). This keeps the visualizer fast (no GPU required) and means a `levels` config change is reflected immediately on the next visualization run.

---

## 12. What's NOT in v1

Logged here so they're not forgotten:

- **Inter-level information flow (UNet-style pyramid).** Each level is built independently from spacepoint features via pooling. This is OK because the Sonata backbone is itself PTv3 with 2 levels of upsampling, so the per-SP features already encode some multi-scale context. Add a small encoder pyramid if level-specific aux losses plateau.
- **LoRA-wrapped backbone in-config.** §8.
- **Joint fine-tuning across cascade stages.** §6 — for now, one stage at a time, downstream training freezes upstream.
- **Flash-match loss for the slicer.** Specced in [Event_Slicer_Spec.md](Event_Slicer_Spec.md). A `FlashMatchHead` stub is in the file layout but its loss term is out of v1; the slicer trains on mask + cls + query CE only for the first pass.
- **Pre-computed multi-resolution voxel grids in the dataset.** Decided to keep voxelization model-side. Revisit if dataloader becomes the bottleneck.
- **Custom builders beyond spacepoint / voxel / fragment.** Easy to add later via the registry.
- **Shared-backbone cascade (the "deghost-decoder" path).** v0 `CascadedSlicer` runs two independent Sonata backbones (one per stage, ~400M params total, two forward passes per training step). The clean alternative is to keep the backbone vanilla + train the **deghoster as a per-SP decoder head** on top of those shared features instead of LoRA-tuning the backbone — then the slicer reads the same backbone features, one pass total. Cost: retrain the deghoster from scratch with a non-LoRA architecture (e.g., a PTv3 decoder block per `larformer-deghost-v0-ptv3decoder.py` or a deeper per-SP MLP). Benefits: 1× backbone compute, naturally extensible to a single-pass 3-stage model. Revisit when (a) the trained Stage-1 deghoster reaches good val mIoU with the LoRA approach AND (b) the 2× backbone cost becomes a practical bottleneck.

---

## 13. Resolved design questions (2026-05-19)

Logged so the reasoning isn't lost.

1. **Cascade vs. single multi-task model.** → Cascade. Three independently trained `LArFormer` instances. Downstream stages are trained with upstream frozen. Joint fine-tuning is a future extension.
2. **Dataset.** → New `LArFormerDataset` (extend, don't modify). `ShowerClusteringDataset` is preserved for the trained shower-origin pipeline.
3. **Multi-resolution voxels.** → Model-side voxelization. Cheap (`floor` + `unique` + scatter), and lets config-only changes flip resolutions without H5 regeneration.
4. **Per-level classification.** → Single per-level cls head with configurable `label_src` naming a per-SP truth field, and a `reduce` rule for spacepoint → level aggregation.
5. **Per-level mask loss budget.** → Point-sampled BCE/Dice at one declared "primary" level (default: highest-resolution level); full-mask BCE at all others; auto-fallback to sampling if a level exceeds `aux_max_tokens`.
6. **Inter-level information flow.** → Independent pooling for v1. Sonata's own upsampling provides enough abstraction. Note as a future extension.
7. **Backbone handling.** → Take via subconfig, support `freeze_backbone`. LoRA-in-config deferred.

---

## 14. Implementation phases (proposed)

Each phase ends with a runnable training config and at least one overfit / sanity test. Smoke-test scripts for the model-side phases live in [`tools/smoke_test_larformer_p{2,3,4,5,6}.py`](../tools/).

| Phase | Scope | Status | Notes |
|-------|-------|--------|-------|
| **P1 — Scaffold** | `builders/{base,spacepoint,voxel}.py`, generic decoder + loss with one-level (spacepoint-only) config | **Done** | Single-event overfit works. |
| **P2 — Multi-level voxel** | Add 2–3 voxel levels; verify per-level mask aux losses + scale-pattern dispatch | **Done** | Smoke test in `smoke_test_larformer_p2.py`. |
| **P3 — Fragment builder** | Port `FragmentPool` + content enricher into `builders/fragment.py`; reproduce `ShowerClusteringMask2Former` behavior | **Done** | Smoke test in `smoke_test_larformer_p3.py`. |
| **P4 — `LArFormerDataset`** | Pluggable `gt_source`; slice GT via `slice_labels.py`; collate handles optional fragments | **Done** | `gt_source="slice"` + `"shower_trunk"` + `"deghost"` all working. `gt_source="particle"` (P7) raises `NotImplementedError`. |
| **P4b — GT visualizer** | Extract `build_levels` + `build_per_level_gt` into pure helpers (§11); wire `tools/visualize_larformer_gt.py` | **Done** | Visualizer at [`tools/visualize_larformer_gt.py`](../tools/visualize_larformer_gt.py), extended in v1 with a prediction-panel mode (see §15). |
| **P5 — Stage 1: deghoster** | Per-level cls head on the spacepoint level; train on `hasmatch` | **Done (LArFormer flavor); LoRA variant adopted for cascade** | Both [`larformer-deghost-v0.py`](../configs/lartpc/larformer-deghost-v0.py) (LArFormer-flavored) and the LoRA-finetuned [`SonataLoRADeghostSegmentor`](../pointcept/models/sonata_lora_deghost.py) work as Stage 1. v1 cascade defaults to the LoRA variant (see §6 implementation note). |
| **P6 — Stage 2: slicer** | Slicer config, frozen Stage 1 wired in-model via `CascadedSlicer`, query-set predicts slices | **Done (architecture); in active debug** | Trains end-to-end and reaches `nu_mIoU=0.67 / cosmic_mIoU=0.47 / mIoU=0.48` on a 10-event dev sample before plateauing — see §15 for the open failure mode and the toggles being tried against it. Flash-match loss still not wired (out of v1 scope). |
| **P7 — Stage 3: particle clusterer** | Particle config, frozen stages 1+2 in the dataset wrapper | **Not started** | Blocked on P6 reaching usable val mIoU. |

Phases 1–4 are model + dataset plumbing and can be done without committing to any downstream task. Phases 5–7 are the cascade itself and depend on having sufficient training data and the upstream stages working.

---

## 15. Current state (slicer-debug, 2026-05-21)

### Setup

- **Config in active iteration:** [`configs/lartpc/larformer-slicer-v1-cascaded-loradeghost.py`](../configs/lartpc/larformer-slicer-v1-cascaded-loradeghost.py).
- **Stage-1 deghoster:** frozen `SonataLoRADeghostSegmentor` from `sonata/lora_deghost_v6_hasmatch/model/epoch_30.pth` (HasmatchAsGhost convention, class 0 = real).
- **Stage-2 slicer:** Sonata-v1m1 backbone (frozen, initialized from `sonata/lartpc_v6_h200_noghosts_pretrain_logspace_resume/model/epoch_42.pth`) + Mask2Former decoder (`token_dim=256`, `num_queries=64`, `num_heads=4`, 6-layer scale pattern: `voxel_20cm → voxel_10cm → voxel_10cm → voxel_5cm → spacepoint → spacepoint`).
- **Training data:** 10-event dev sample, single `_DEFAULT_LIST` used for train/val/test (intentional — this run is for overfit + failure-mode characterization, not generalization).
- **Optimizer / schedule:** AdamW @ `1e-4`, `FlatWithDecayLR(plateau, gamma=0.5, patience_epochs=4, cooldown_epochs=2, min_lr=1e-7)` driven by the `LREpochScheduler` hook.
- **Augmentation knob (cascade-specific):** the dataset's `lm_score_aug_low/high/val` were all dropped to `0.0` (no `lm_score` pre-filter at the dataset stage). The previous default (`0.15/0.40/0.15`) pre-filtered SPs at the LArMatch stage and then handed the deghoster a non-representative subset; that double-filtering depressed the measured `real_recall` from `~0.71` (inference-script value, 100-event eval) to `~0.13` (in-training at τ=0.5 with the pre-filter still on). With pre-filter off, in-training `real_recall=0.65` and `ghost_reject=0.83` at τ=0.5, matching the inference script.

### Tooling added during the debug iteration

- [`tools/run_slicer_inference.py`](../tools/run_slicer_inference.py) — reads an input list + a `CascadedSlicer` checkpoint and writes a per-event HDF5 with `pre/post/queries/gt/meta` groups (pre- and post-deghost SPs, query predictions, matched GT slice IDs, run/subrun/event ids). Used both for offline mIoU measurement (matches the in-training evaluator) and as the source-of-truth for the visualizer's prediction panel.
- [`tools/visualize_larformer_gt.py`](../tools/visualize_larformer_gt.py) — extended with a `--slicerpred-dir` mode that overlays the inference script's HDF5 output as a second 3D scene stacked under the GT scene. Color modes: `pred_correct` (matched / mismatched / unmatched), `pred_slice_id`, `slice_id_gt`, `pred_class`, `p_real`. Track-ID color cycling is shared between GT and pred for visual alignment.

### Current results & failure mode

Best val metrics on the 10-event dev sample, plateauing around iter 2800:

| Metric          | Value |
|-----------------|-------|
| `nu_mIoU`       | 0.67  |
| `cosmic_mIoU`   | 0.47  |
| `mIoU`          | 0.48  |

Hand-scan of two events (37 GT slices total) using the visualizer's `pred_correct` mode:

| Outcome                                              | Count |
|------------------------------------------------------|-------|
| Good slice prediction (~1-to-1 matched to a GT slice) | 13/37 |
| Mirror-symmetric / parallel over-clustering          | 13/37 |
| Slice predicted as several small disjoint pieces     |  7/37 |

The dominant new failure pattern (driving the cosmic mIoU drag) is **two content-similar tracks at very different positions getting bound to the same query** — often two roughly parallel tracks in opposite halves of the TPC, or a track plus its mirror across a TPC axis. The model has no good way to break the tie: with `enable_origin_head=False`, each query is just a learnable slot identity with no spatial anchor, and the decoder's only spatial signal is a learnable 3-layer MLP `pos_emb` that can collapse to mirror-symmetric features early in training.

### Toggles added to combat the mirror-symmetric pathology

Both configs (`larformer-slicer-v1-cascaded.py` and `…-cascaded-loradeghost.py`) carry two top-of-file flags that select between the spec's defaults and the new positional priors:

```python
USE_SINUSOIDAL_POS_EMB = False               # decoder.pos_emb_kind
ENABLE_ORIGIN_HEAD_WITH_CENTROID = False     # enable_origin_head + slice_origin_kind + weight_origin
```

1. **`USE_SINUSOIDAL_POS_EMB`** swaps the learnable MLP `pos_emb` in [`Mask2FormerDecoder`](../pointcept/models/LArFormer/decoder.py) for a fixed NeRF-style sinusoidal embedding (`SinusoidalPosEmb3D`: 3 axes × `(sin, cos)` × `num_freq` log-spaced freqs in `[1, 256]` + a single Linear projection). Every coord gets a unique structured signature out of the box.

   *Status (2026-05-21): tried; no measurable improvement, possibly slightly worse than the MLP baseline. Currently `False`.*

2. **`ENABLE_ORIGIN_HEAD_WITH_CENTROID`** re-enables the per-query origin head + sets the regression target to the **slice centroid** (`coord_norm[truth_indices].mean(0)`) rather than `primary_start_pos`. Centroid is well-bounded inside the TPC and tightly correlated with the slice's own spacepoints — for cosmics in particular, `primary_start_pos` is the cosmic-ray start above the TPC surface and can sit ~16+ norm-units away from the slice (i.e. ~2900 cm), which makes the regression target essentially noise. The centroid target also matches the spatial scale of the decoder's `pos_emb` input. When `True`, the loss adds `weight_origin=0.5`, the dataset's `slice_origin_kind="centroid"`, and the dynamic `query_pos = query_pos + pos_emb(predicted_origin)` is re-enabled per layer so the spatial anchor refines layer-to-layer.

   Implementation: `slice_origin_kind` is a new kwarg on [`LArFormerDataset`](../pointcept/datasets/larformer.py) (default `"primary_start_pos"` for backward compatibility); the `"centroid"` branch in `_gt_from_slices` computes the per-slice mean of the coord-normalized spacepoint positions for that slice. The origin head itself is a direct query → 3-vec regression (per-layer MLP `Linear(D,D) → GELU → Linear(D,3)`, see [`_PerLayerHeads`](../pointcept/models/LArFormer/decoder.py)) with an L1 loss on Hungarian-matched (query, slice) pairs — **no mask involvement**.

   *Status (2026-05-21): in active trial.*

If the origin head alone doesn't fix the mirror-symmetric merger, the next thing to try is **initializing the layer-0 origin from the layer-0 mask centroid** (currently the layer-0 origin is whatever the init MLP outputs from the raw learnable query token, with no spatial signal) or stop-grad'ing the predicted origin into `query_pos` for the first layer so a bad init can't pin down the spatial anchor for the rest of the stack.

---

## 16. The `TokenRefiner` abstraction

### Motivation

The mirror-symmetric merger analysis in §15 narrowed the failure to: `mask_embed(q)` can't separate two content-similar tracks at different positions, because the only spatial information the mask head sees is `pos_emb(coord)` added to each level's tokens. In vanilla LArFormer, the level tokens themselves are static mean-pooled backbone features — they never learn to encode "I am in *this* region of the detector with *these* neighbors." A merger-rate diagnostic on the dev sample confirmed the model's confidently-wrong cluster mass is concentrated at the *voxel scales* (7–10% confident over-cluster mass at voxel_5/10/20cm vs. only 1.7% at the spacepoint scale, where most over-cluster appearance is panoptic-argmax noise).

This is the canonical setup for Mask2Former-style architectures: between the backbone (which gives you per-token features) and the transformer decoder (which cross-attends queries to those tokens), Mask2Former inserts a **pixel decoder** — a trainable module whose job is to refine the per-token features into mask-friendly multi-scale representations. LArFormer originally skipped this stage entirely (queries cross-attend straight against pooled backbone features). The `TokenRefiner` abstraction adds it back, in a way that respects LArFormer's flexible-levels design.

### Contract

A `TokenRefiner` is a drop-in transform on the tokenizer's output:

```python
OrderedDict[level_name → LevelOutput]   →   OrderedDict[level_name → LevelOutput]
```

Same keys in, same keys out; per-level `(coords, sp_to_level_id, name)` are preserved; only `tokens` may change. Token count `M_level` MUST stay the same (the decoder's mask logits and the loss's per-level GT masks are indexed by token position).

The refiner runs once per event, between [`CompositeTokenizer`](../pointcept/models/LArFormer/tokenizer.py) and [`Mask2FormerDecoder`](../pointcept/models/LArFormer/decoder.py). The integration point in [`LArFormer.forward`](../pointcept/models/LArFormer/model.py) is one line:

```python
levels = self.tokenizer(sp_feat, coord_norm, event_dict)
levels = self.token_refiner(levels)            # ← the new step
decoder_out = self.decoder(levels)
```

The decoder, loss, evaluator, and visualizer are unchanged. The refiner is opt-in via config; the default `IdentityRefiner` pass-through reproduces pre-refiner behavior bit-exactly.

### File layout

```
pointcept/models/LArFormer/refiners/
├── __init__.py            # REFINERS registry + build_token_refiner()
├── base.py                # TokenRefiner ABC + IdentityRefiner
├── pos_emb.py             # Shared MLPPosEmb / SinusoidalPosEmb3D / build_pos_emb
├── per_level_sa.py        # PerLevelSelfAttn   (Option 1)
└── cross_level.py         # CrossLevelAttn     (Option 2)
```

### Available refiners

| Refiner | Levels touched | What it does | Params (D=256, default cfg) |
|---|---|---|---|
| `IdentityRefiner` | none | Pass-through. Baseline. | 0 |
| `PerLevelSelfAttn` | each voxel level independently | N transformer-style SA + FFN blocks per voxel level, with the level's own `pos_emb` on Q/K. No cross-level interaction. | ~4.94M (2 layers × 3 voxel levels × heads=4 × mlp_ratio=4) |
| `CrossLevelAttn` | voxel levels (Q); all levels incl. SP as K/V | For each target voxel level, cross-attention against the concatenated token pool of all source levels. Shared `pos_emb` bridges scales via coords. Voxel tokens can READ from per-SP context. Updated tokens feed back into the K/V pool for the next layer (scales co-evolve). | ~4.78M |
| (deferred) `CrossLevelAttn(attn_kind="deformable")` | as above | kNN-based deformable sampling; defer until the full-attn variant is shown to be the right kind of mechanism. | ~5–6M projected |

All three live behind a single config knob — `slicer_cfg.token_refiner` — and the LArFormer model auto-injects `dim=token_dim` and `levels_cfg` so the per-level submodules build EAGERLY at `__init__` (required for DDP and post-construction `.to(device)`).

### Why the spacepoint level is excluded by default

`PerLevelSelfAttn` skips spacepoint and `CrossLevelAttn` excludes it from *targets* (it stays available as a *source* — voxel queries can attend to per-SP keys). Two reasons:

- **Cost.** Full self-attention on ~50K post-deghost SPs is infeasible: ~10⁹ attention ops per layer per head, and the attention matrix alone is ~10GB. No shape-friendly version of this exists without windowing.
- **PTv3's own windowed attention** already mixes SP context inside the (frozen) backbone via serialization-based patches. A non-windowed SA layer on top would be duplicating that work at huge cost.

Voxel levels at our token counts (~500–2500 per level) are cheap (full-attention attention matrix is ~MB, microseconds) AND are exactly where the merger-rate diagnostic localized the confident-FP problem.

### Hybrid: PTv3 native decoder as a "refiner"

PT-v3m2 has a built-in learned decoder (`self.dec`, gated by `enc_mode=False`) that mirrors the encoder's pyramid. Each `dec{s}` stage is a `GridUnpooling` + `dec_depths[s]` transformer Blocks operating at the encoder's native stride. Turning this on gives a learned multi-scale refinement that's *structurally inside the backbone* rather than bolted on between backbone and decoder.

To use it, set `USE_PTV3_DECODER_LEVELS = True` in the cascaded-loradeghost config:

- `enc_mode=False` on the inner PT-v3m2 → builds `self.dec`
- `up_cast_level=0` on the Sonata-v1m1 wrapper → bypasses the (now-unneeded) post-encoder concat-scatter upcast
- `backbone_out_channels = dec_channels[0] = 64` → SP-level features are now the final dec0 output, not the encoder's concatenated pyramid
- `levels = [ptv3_dec3, ptv3_dec2, ptv3_dec1, spacepoint]` with [`PTv3DecoderStageLevel`](../pointcept/models/LArFormer/builders/ptv3_decoder_stage.py) builders that consume each captured decoder stage
- `capture_decoder_stages=True` on the LArFormer model → registers forward hooks on `teacher.backbone.dec.dec{s}` so each stage's output Point is captured into `self._dec_stage_capture`, then sliced per-event in `_build_decoder_stages_per_event()`
- `unfreeze_decoder=True` → frees parameters whose name contains `.dec.` so the decoder is trainable from random init (the encoder stays frozen on the pretrain weights). `_encode` skips its `no_grad` wrapper when the decoder is trainable so the gradient graph spans the decoder.
- `token_refiner=None` (Identity) — the PTv3 decoder IS the refiner in this setup. A non-Identity refiner could also stack on top as a hybrid.

A standalone config preset lives at [`configs/lartpc/larformer-slicer-v1-cascaded-ptv3decoder.py`](../configs/lartpc/larformer-slicer-v1-cascaded-ptv3decoder.py).

**Caveat — PTv3 pyramid is finer than user-defined voxels.** PT-v3m2's strides `(2,2,2,2)` apply to the input grid_coord, which the dataset emits at `backbone_grid_size_cm=0.25`. So dec1/dec2/dec3 have effective grids of 0.5/1/2 cm respectively. Compared with `voxel_5/10/20cm`, the PTv3 pyramid pools *much less aggressively* on LArTPC-spaced data (real spacepoints sit ~1–3 cm apart, finer than dec3's 2 cm grid). To get the PTv3 pyramid to pool as coarsely as `voxel_20cm` would require bumping `backbone_grid_size_cm` substantially (changing the encoder's input grid, a non-trivial decision).

### How to switch (cascaded-loradeghost config)

```python
TOKEN_REFINER_KIND = "identity"     # "identity" | "per_level" | "cross_level"
TOKEN_REFINER_LAYERS = 2
USE_PTV3_DECODER_LEVELS = False     # PTv3 native decoder; pair with TOKEN_REFINER_KIND="identity"
```

The three `TOKEN_REFINER_KIND` choices and the orthogonal `USE_PTV3_DECODER_LEVELS` flag give a 4-by configuration matrix:

| `USE_PTV3_DECODER_LEVELS` | `TOKEN_REFINER_KIND` | Levels | Refiner / multi-scale learning lives in… |
|---|---|---|---|
| `False` | `"identity"` | voxel_20/10/5cm + SP | nowhere (the baseline; what existed before the abstraction) |
| `False` | `"per_level"` | voxel_20/10/5cm + SP | `PerLevelSelfAttn` per voxel level (no cross-scale flow) |
| `False` | `"cross_level"` | voxel_20/10/5cm + SP | `CrossLevelAttn` across all levels |
| `True`  | `"identity"` | ptv3_dec3/2/1 + SP | PT-v3m2's `self.dec` (learned native pyramid) |
| `True`  | `"per_level"` or `"cross_level"` | ptv3_dec* + SP | PTv3 decoder *and* refiner on top (hybrid) |

### Ablation plan

Recommended order, paired with the merger-rate diagnostic ([`tools/measure_merger_rates.py`](../tools/measure_merger_rates.py)) at both `--min-mask-prob 0` (panoptic) and `--min-mask-prob 0.5` (confident-only) for headline comparisons:

| # | Setup | Hypothesis tested | Status (2026-05-21) |
|---|---|---|---|
| 1 | baseline = `identity` + voxels | reference for everything else | done |
| 2 | `per_level` + voxels | per-level intra-token mixing helps the voxel-scale masks separate content-similar distant tracks | in progress |
| 3 | `cross_level` + voxels | cross-level information flow (voxel tokens read per-SP context, coarse↔fine flows both ways) beats per-level | queued after #2 |
| 4 | `identity` + PTv3 decoder | the encoder-native pyramid + learned decoder is the "right" multi-scale machinery for LArTPC | queued after #3; requires deciding the right `backbone_grid_size_cm` so the pyramid pools at physically meaningful scales |
| 5 | `cross_level` + PTv3 decoder (hybrid) | combine PTv3-native pyramid with bridge-style cross-level mixing; closest analog to Mask2Former + MSDeformAttn | only if #4 looks promising |

For each setup, the per-level merger rates (and the gap between `min_mask_prob=0` and `min_mask_prob=0.5` numbers) are the most informative metric, alongside `nu_mIoU` and `cosmic_mIoU`. Hand-scan a small sample with the visualizer's new `sp_by_level_inst` color mode at coarser levels — that lets you literally see whether two physically distinct objects pool into the same cluster at a given level (which the loss can't unmerge there).

### Design points worth flagging

- **`pos_emb` is per-refiner, not shared with the decoder.** Both `PerLevelSelfAttn` and `CrossLevelAttn` build their own pos_emb (`build_pos_emb` from [`refiners/pos_emb.py`](../pointcept/models/LArFormer/refiners/pos_emb.py)) so ablations can vary the refiner's `pos_emb_kind` independently of the decoder's. `CrossLevelAttn` further uses ONE shared pos_emb across all levels (positions are in the same `coord_norm` frame, so no per-level duplication).
- **DDP / `.to(device)` safety.** Both Option 1 and Option 2 build their per-level submodules eagerly in `__init__` (via the `levels_cfg` kwarg the LArFormer auto-injects). The lazy-build fallback remains for standalone smoke tests but is documented as DDP-unsafe.
- **The SP source asymmetry in `CrossLevelAttn`.** Voxel target tokens can read SP-level keys (huge K), but SP tokens are never *updated*. This keeps the cost asymmetric: full Q×K with Q in the thousands and K in the tens of thousands is bounded and (with FlashAttention) fast. A `max_source_tokens_per_level` knob (default `8192` in the active config) randomly subsamples big sources per forward to keep K bounded.
- **PTv3 decoder is trainable from scratch.** The Sonata pretrain ran with `enc_mode=True`, so decoder weights are NOT in the pretrain checkpoint. They initialize randomly. The encoder stays frozen on the pretrain. Expect higher trainable-param counts (~22M including the new decoder, vs ~7M for refiner-only setups) and somewhat slower training; memory needs the encoder's activations retained for backward through the decoder.

---

## 17. Mixed Query Selection (Phase A)

### Motivation

After §16 added a learned `TokenRefiner` between the backbone and the decoder, two related pathologies remained:

1. **Matching instability.** Hungarian assignments shuffle across iterations because the K decoder queries are initialized from a single learnable embedding bank — at init they're nearly indistinguishable, so which query "owns" which GT slice flips depending on small perturbations in the per-token features. The decoder spends many epochs untangling query identities before mask losses can specialize.

2. **Over-claim.** Instrumented via the `pair_iou` vs `argmax_iou` gap from `tools/run_slicer_inference.py` (v2 fields) and aggregated by [`tools/measure_overclaim.py`](../tools/measure_overclaim.py). Multiple queries compete to claim the same easy SPs of a confident track. With identical init, queries cluster on whichever GT slice has the cleanest signal; spatial diversity has to emerge from training alone.

DINO and Mask-DINO solve both with **mixed query selection**: replace the learnable query bank with the top-K most "object-like" tokens picked from the encoder/refiner output, and seed each query's positional prior with the picked token's coordinates. Each query now starts at a concrete spatial anchor with a meaningful content vector — matching is more stable (queries are pre-specialized to a region) and over-claim drops (anchors are spread, not piled on top of each other).

### Adaptation for LArTPC

LArTPC slices vary enormously in size. One long cosmic produces hundreds of high-score `voxel_8cm` tokens; smaller tracks produce a handful. Pure DINO-style top-K would happily put dozens of queries on the longest cosmic and miss several smaller tracks entirely. The selection pipeline therefore has a diversity step:

1. **Score** every token in the chosen source level (`voxel_8cm` by default) by `1 - p(no_object)` from that level's already-trained per-token cls head. No new supervision needed — the cls head is supervised by [`larformer-slicer-v1-cascaded-ptv3hybrid_perlevel.py`](../configs/lartpc/larformer-slicer-v1-cascaded-ptv3hybrid_perlevel.py) anyway.
2. **Top-M filter:** keep the M = K × `score_filter_multiplier` (default 4) highest-scoring tokens.
3. **FPS** (farthest-point sampling) from the M survivors picks K spatially-diverse anchors. FPS is seeded with the highest-score token (index 0 of the topk result), so the most confident candidate is always selected.

The smoke test [`tools/smoke_test_larformer_p7_mixed_query.py`](../tools/smoke_test_larformer_p7_mixed_query.py) verifies that on a synthetic event with one dominant high-score cluster and four smaller clusters, FPS gives ~1.6× the mean-nearest-neighbor distance of pure top-K.

### Decoder wiring

[`Mask2FormerDecoder.forward`](../pointcept/models/LArFormer/decoder.py) accepts two new optional kwargs:

```
forward(levels,
        init_query_content: (Q, D),    # token features at selected anchors
        init_anchor_coords: (Q, 3))     # anchor coords (coord_norm frame)
```

When provided:

- `queries = init_query_content + self.query_content` (additive delta on top of anchors).
- `anchor_pe = self.pos_emb(init_anchor_coords)` is computed once and added into every layer's `query_pos_dyn`.

The pre-existing `self.query_content` and `self.query_pos` parameters become **zero-initialized learnable deltas** when the [`MixedQuerySelector`](../pointcept/models/LArFormer/query_selection.py) is active. That zero-init is the standard DETR/Mask-DINO trick — at init the decoder sees pure anchor features + pure anchor PE; the deltas only become non-trivial as training progresses.

### Config knob

The `LArFormer` ctor gains one new kwarg:

```python
model = dict(
    type="LArFormer",
    ...,
    mixed_query_selection=dict(
        source_level="voxel_8cm",              # must declare supervision.cls
        score_source="cls_head",               # only mode in v1
        selection_mode="top_m_then_fps",       # or "top_k"
        score_filter_multiplier=4,             # M = K × this
    ),
)
```

When `mixed_query_selection` is omitted the model behaves exactly as before (learnable query bank, no anchors). Enabling it requires the configured source level to have a `supervision.cls` block; the selector raises a clear `ValueError` at construction time otherwise.

### What's deliberately NOT in Phase A

- **No dedicated proposer head.** Score source is fixed at `cls_head`. Adding a separate proposer + aux BCE loss (DINO's "dedicated" mode) is deferred to Phase B/C, gated on Phase A failing to lift the matching-stability / over-claim metrics.
- **No denoising queries.** Mask DINO's mask-denoising path is the planned next architectural addition, but kept separate from Phase A so each piece can be evaluated in isolation.

---

## 18. Mask Denoising (Phase B)

### Motivation

Even with Phase A (mixed query selection) seeding the K queries from `1 - p(no_object)`-scored anchor tokens, two failure modes remained visible in early training:

- **Slow mask specialization.** Hungarian matching is unstable until the per-query mask logits are sharp enough to be discriminative; in the first few thousand iters, most queries' masks are diffuse and the matcher cost is dominated by random class-logit noise. The result is high-variance assignments — a query "owns" one GT slice this iteration and a different one the next.
- **Decoder needs an easy task to bootstrap.** Set-prediction with Hungarian is a hard end-to-end task: the gradients only kick in *after* matching, which itself depends on the predictions. A strong auxiliary signal that gives the decoder a "find a mask given approximately where it is and what it is" task — solvable independently of matching — accelerates the regular path indirectly through shared decoder weights.

Mask DINO introduces **mask denoising** for exactly this: take each GT, perturb its position + identity, feed those perturbed copies in as extra "denoising queries," and supervise each one to **directly** reconstruct its original GT (no Hungarian — a fixed index assignment). The denoising queries are kept isolated from the regular ones via attention masking, so they don't pollute the regular query specialization, but the decoder weights they update are shared. The reported gain in image-domain Mask DINO is +1.5–3 mAP on top of mixed query selection.

### Adaptation for LArTPC (Phase B.1, the minimum viable version)

The simplest formulation that captures most of the benefit, scoped to land before any contrastive / mask-domain refinements:

- **Per-GT replication.** For each event with K GT instances, emit `dn_groups × K` DN queries (default `dn_groups = 3`). Each DN query is supervised to its corresponding GT.
- **Per-event cap.** `max_dn_per_event = 96` upper-bounds the DN query count. Above the cap, a random subset of (group, gt) pairs is kept. With ~20 GT slices per typical event and dn_groups=3 we get ~60 DN queries; the cap kicks in only on cosmic-heavy outliers (e.g. 40+ GT slices).
- **Anchor noise.** GT `origin_coord_norm` + Gaussian jitter (σ = 0.05 in coord_norm space ≈ 9 cm — one `voxel_8cm` cell). Same jitter sampled independently per query so different DN groups for the same GT see different anchors and aren't trivial copies.
- **Content init.** A learnable per-class embedding lookup (`nn.Embedding(num_classes, D)`). **Deliberately does NOT depend on which SPs are in the GT slice** — keeps the init from baking GT-mask structure into the content vector, which would make the denoising task suspiciously easy.
- **Attention isolation.** A `Q_total × Q_total` boolean mask passed to every decoder layer's self-attention: regular ↔ regular allowed, regular ↔ DN blocked both ways, DN cross-group blocked, DN within-group allowed. Cross-attention is untouched — DN queries attend to keys normally.

### Decoder wiring

[`Mask2FormerDecoder.forward`](../pointcept/models/LArFormer/decoder.py) accepts a new optional kwarg `dn_self_attn_mask: (Q_total, Q_total) bool` and tolerates `init_query_content` / `init_anchor_coords` larger than `self.query_content.shape[0]`:

- `queries[:K_reg] = init_query_content[:K_reg] + self.query_content[:K_reg]` (Phase-A delta applies to regular slots).
- `queries[K_reg:] = init_query_content[K_reg:]` (DN slots pass through the class-embedding init unchanged).
- `query_pos_dyn[:K_reg] = self.query_pos[:K_reg] + anchor_pe[:K_reg]`; `query_pos_dyn[K_reg:] = 0 + anchor_pe[K_reg:]` (the learnable `query_pos` is zero-padded for DN; DN's positional prior comes from its anchor + optional origin-head refinement).

`_MaskedDecoderLayer.forward` gains a `self_attn_mask` parameter, threaded into `self.self_attn(..., attn_mask=self_attn_mask, ...)`. Cross-attn mask stays untouched.

### Loss wiring

[`LArFormerLoss`](../pointcept/models/LArFormer/losses.py) gains a `compute_dn_loss(decoder_output_dn, per_level_gt_mask, gt_classes, gt_origin, gt_target_idx)` method that:

1. Builds the **direct** assignment `q_idx = arange(Q_dn), k_idx = gt_target_idx` — no Hungarian.
2. Reuses `_compute_layer_loss` (the existing per-pair sampled BCE + Dice + class CE + origin L1).
3. Aggregates using the same per-component weights as the regular path (`weight_class`, `weight_mask_primary`, …), then scales the final sum by a single knob `weight_dn_loss` (default `1.0`).
4. Returns a scalar dict that the model prefixes with `dn_` and merges into the regular loss dict.

In training mode, [`LArFormer.forward`](../pointcept/models/LArFormer/model.py) computes the loss with `return_matching=True` (so `compute_dn_loss` can reuse the already-built `per_level_gt_mask` / `gt_classes` / `gt_origin` rather than recomputing them), strips the non-scalar matching keys before appending to the per-event loss list, and runs `compute_dn_loss` on the DN slice.

### Config knob

`LArFormer.__init__` gains a `mask_denoising: Optional[dict]` kwarg:

```python
model = dict(
    type="LArFormer",
    ...,
    mixed_query_selection=dict(source_level="voxel_8cm", ...),  # required
    mask_denoising=dict(
        dn_groups=3,
        max_dn_per_event=96,
        anchor_jitter_std=0.05,
    ),
    # loss_kwargs may also set weight_dn_loss (default 1.0)
)
```

The ctor enforces that `mixed_query_selection` is also enabled — otherwise `self.decoder.query_content` / `query_pos` aren't zero-init'd and they'd bleed into the DN slots (the decoder zero-pads them past K_reg, but the *learnable* values for the first K_reg slots could still steer the trailing DN content through the residual additions in cross-attn → mask_embed). Cleaner to make Phase A a hard prerequisite.

### Train / eval semantics

- **Train only.** DN queries are built per-event during `model.forward` when `self.training is True`. Eval and inference skip the path entirely — both the decoder forward and the loss avoid the extra Q.
- **Per-iter cost.** Worst case (96 DN queries on top of the configured K_reg = 128 / 64) ~doubles the decoder's Q. Cross-attn cost stays dominated by keys (~5K–20K SPs per event); self-attn becomes O(Q²) instead of O(K²) but stays small in absolute terms.
- **`origin_head` interaction.** When `enable_origin_head = False` (the live config setting), DN queries simply don't have an origin loss term (`weight_origin = 0` already disables it for regular queries too). Their anchor still feeds the per-layer `query_pos_dyn` via `anchor_pe`.
- **Backward fan-in.** DN losses flow into: (a) the decoder's per-layer heads (cls/origin/mask_embed), (b) the cross-attn / self-attn / FFN weights shared with the regular path, (c) the `MaskDenoiser.class_embedding` table, (d) Phase A's `MixedQuerySelector` is NOT in the DN gradient path — its outputs only feed the regular slice. So Phase A and Phase B optimize independent slices of the input space; they share only the decoder body.

### Smoke test

[`tools/smoke_test_larformer_p8_mask_denoising.py`](../tools/smoke_test_larformer_p8_mask_denoising.py) covers, in isolation from the heavy backbone:

1. `MaskDenoiser` direct construction — Q_dn shape, `gt_target_idx` / `group_id` structure, jitter within 4σ, cap enforcement, empty / no_object filtering.
2. `build_self_attn_mask` structure — regular ↔ regular allowed, regular ↔ DN blocked, DN cross-group blocked.
3. End-to-end decoder forward + backward with combined `[regular | DN]` queries; verifies the self-attn mask actually changes layer output (re-runs with the mask vs `None` after un-zeroing `self_attn.out_proj` so the change is observable through PyTorch's MHA).
4. `LArFormerLoss.compute_dn_loss` on synthetic DN decoder output; checks scalar finiteness, backward into `class_logits` / `mask_logits`, and the `Q_dn = 0` empty path.

### What's deliberately NOT in Phase B.1

- **Mask-domain noise (drop + add SPs).** The query content is class-embedding only, not mask-pooled. Mask-domain perturbation would only matter if content depended on which SPs survived noising.
- **Contrastive denoising (CDN positive/negative pairs).** Mask DINO's CDN doubles DN query count to teach "negative" queries to predict no_object. Deferred — adds a knob that would confuse the diagnosis if B.1 underperforms.
- **Per-group noise scheduling.** All groups use the same `anchor_jitter_std`. Annealing (e.g., σ → 0 over training) is a Phase B.3 follow-up.
- **DN aux-mask losses on non-primary levels.** Reusing `_compute_layer_loss` means aux-mask BCE *is* computed at every supervised level — but DN queries are most informative at the primary level (where the actual mask supervision lives). The `weight_aux_mask` override per level still applies, so the user can dial DN aux contribution down without touching the regular path.

---

## 19. References

- Existing model being generalized:
  [`pointcept/models/shower_clustering/`](../pointcept/models/shower_clustering/) — model / tokenizer / decoder / losses / matcher.
- Existing dataset being extended:
  [`pointcept/datasets/shower_clustering.py`](../pointcept/datasets/shower_clustering.py).
- Shower-clustering design notes (the architecture LArFormer generalizes):
  [`docs/shower_clustering_design.md`](shower_clustering_design.md).
- Event slicer requirements (the primary stage-2 use case):
  [`docs/Event_Slicer_Spec.md`](Event_Slicer_Spec.md).
- Slice ground-truth builder used by `gt_source="slice"`:
  [`lartpc_data_prep/slice_labels.py`](../lartpc_data_prep/slice_labels.py).
- LArTPC H5 schema (per-spacepoint truth fields used as `label_src` values):
  [`docs/LArTPC_HDF5_Data_Format.md`](LArTPC_HDF5_Data_Format.md).
- Visualizer patterns to reuse for the per-level GT viewer (§11):
  [`tools/visualize_shower_clustering.py`](../tools/visualize_shower_clustering.py),
  [`tools/visualize_slice_flash_match.py`](../tools/visualize_slice_flash_match.py),
  [`tools/visualize_lartpc_h5data.py`](../tools/visualize_lartpc_h5data.py).
