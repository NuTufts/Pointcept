# LArFormer — Design and Implementation Plan

**Status:** Draft / pre-implementation. Not yet started in code.
**Owner:** taritree.wongjirad@tufts.edu
**Generalizes:** [`ShowerClusteringMask2Former`](../pointcept/models/shower_clustering/model.py) (kept frozen for the trained shower-origin pipeline).
**Lives at:** `pointcept/models/LArFormer/` (new).

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
    src       = data["spacepoint"][label_src],     # (N,)
    index     = sp_to_level_id,                    # (N,)
    dim_size  = M_level,
    reduce    = "amax" | "plurality" | "mean",
)
```

`label_src` names a field already in the dataset (e.g., `hasmatch`, `origin_label`, `slice_id`, `pid`). `reduce="plurality"` is implemented as scatter-mode (most common label per level). For binary labels (deghosting), `amax` works.

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

Inheriting from `ShowerClusteringLoss`, generalized:

| Component                               | Default weight | Notes                                              |
|-----------------------------------------|----------------|----------------------------------------------------|
| Query class CE (matched)                | 2.0            | Hungarian-matched query → GT class                 |
| Query class CE (no-object)              | × 0.1          | Down-weight for unmatched queries                  |
| Primary-level mask BCE (per-pair, sampled, S=4096) | 5.0  | Point-sampled à la PointRend; balanced pos/neg     |
| Primary-level Dice                      | 5.0            | Same sampled set                                   |
| Aux mask BCE per non-primary level       | 1.0            | Full-mask BCE if `M_level ≤ aux_max_tokens`        |
| Per-level cls CE (if declared)          | 1.0            | Per-token CE on the level's tokens                  |
| Origin L1 (matched, if origin head on)  | 1.0            | As today                                            |

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

- **Stage 1 → Stage 2.** Deghoster's per-SP `real` score is added to the slicer's input `feat`. The slicer's input set is the filter `score > τ` (τ matches deghoster training).
- **Stage 2 → Stage 3.** Slicer's chosen nu-slice mask defines stage 3's input spacepoints. Slice metadata (flash time, vertex position) can ride along as extra per-event tokens.

These hookpoints live in dataset-side wrappers (`LArFormerDeghostedDataset`, `LArFormerSliceCarvedDataset`) that take the prior stage's frozen checkpoint as a kwarg and run it once at `__getitem__` time. The downstream model itself doesn't know about the cascade.

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

Each phase ends with a runnable training config and at least one overfit / sanity test.

| Phase | Scope | Done when |
|-------|-------|-----------|
| **P1 — Scaffold** | `builders/{base,spacepoint,voxel}.py`, generic decoder + loss with one-level (spacepoint-only) config | Overfits a single event, no fragment dependency |
| **P2 — Multi-level voxel** | Add 2–3 voxel levels in config; verify per-level mask aux losses work; verify scale pattern dispatch | A pure-voxel + spacepoint slicer-style config trains for 1 epoch on a small sample |
| **P3 — Fragment builder** | Port `FragmentPool` + content enricher into `builders/fragment.py`; reproduce `ShowerClusteringMask2Former` behavior to within numerical tolerance | Equivalent config gives mask_iou parity vs the existing model on a small eval set |
| **P4 — `LArFormerDataset`** | New dataset with pluggable `gt_source`; pull slice GT via `slice_labels.py`; collate handles optional fragments | Loads the canonical example for `gt_source="slice"` and `gt_source="shower_trunk"` |
| **P4b — GT visualizer** | Extract `build_levels` + `build_per_level_gt` into pure helpers (§11); wire `tools/visualize_larformer_gt.py` against them | Per-level coords + instance / cls coloring renders for the canonical example under both `gt_source` settings |
| **P5 — Stage 1: deghoster** | Per-level cls head on the spacepoint level; minimal/no queries; train on `hasmatch` | Beats or matches the existing LoRA deghoster on val mIoU |
| **P6 — Stage 2: slicer** | Slicer config, frozen Stage 1 in the dataset wrapper, query-set predicts slices | Pure-mask + cls training converges on a small sample; flash loss not yet wired |
| **P7 — Stage 3: particle clusterer** | Particle config, frozen stages 1+2 in the dataset wrapper | Overfit one nu slice end-to-end |

Phases 1–4 are model + dataset plumbing and can be done without committing to any downstream task. Phases 5–7 are the cascade itself and depend on having sufficient training data and the upstream stages working.

---

## 15. References

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
