# LArFormer cascade — data processing pipeline

Driver scripts that take MicroBooNE `merged_dlreco.root` files (custom
C++/ROOT objects) and run them through the trained **LArFormer cascade**
(deghost → event-slice → particle-segment), producing per-event HDF5.

Everything runs inside the **pointcept** Apptainer/Singularity container.
There is **no LArMatch and no SSNet, and no lantern container** — the LoRA
deghoster (cascade Stage 1) does the deghosting, so the 3D spacepoints come
straight from the `SimChTripletLabelMaker` triplet proposals.

See the model design in [`../../docs/LArFormer.md`](../../docs/LArFormer.md)
(§0 is the project hub) and
[`../../docs/LArFormer_particlesegment_stage.md`](../../docs/LArFormer_particlesegment_stage.md).
For split-level validation metrics on the resulting `stage3pred_*.h5`
files, see
[`../larformer_particle_analysis/README.md`](../larformer_particle_analysis/README.md).

---

## The two stages

| Stage | Container | Entrypoint | Output |
|-------|-----------|------------|--------|
| A | pointcept | [`convert_dlmerged_to_larformer_h5.py`](convert_dlmerged_to_larformer_h5.py) | `merged_<TAG>_fileno<NNNNN>_entry<N>.h5` |
| B | pointcept (GPU) | [`run_larformer_stage3_inference.py --input-mode full-cascade`](../../tools/run_larformer_stage3_inference.py) | `stage3pred_<basename>.h5` |

### Stage A — convert

`merged_dlreco.root` → per-event H5 via `SimChTripletLabelMaker`, the same
truth maker used to build the LArFormer *training* data. Each event H5 has
the exact on-disk schema `pointcept.datasets.larformer.LArFormerDataset`
reads:

- `entry_0/triplet_data/` — `pos`, `pixval`, `uwire/vwire/ywire`, `tick`,
  `trackid`, `pid`, `origin`, `hasmatch` (0/1 = real/ghost, the deghoster's
  target), `ssnet_label`, and a **dummy `lm_score`=1.0** (LArFormerDataset
  reads `lm_score` unconditionally; the cascade disables the lm_score
  pre-filter, so the value is irrelevant).
- `entry_0/mc_particle_tree/` — MC truth (simulation only).
- `entry_0/shower_fragments/` — empty placeholder (LArFormer voxelizes
  model-side; fragments are not used, but the group is opened
  unconditionally).
- **Folded-in flash** (no separate flashinfo file): `entry_0/flashes/`
  (`simpleFlashBeam` + `simpleFlashCosmic`: `pe`, `total_pe`, `time_us`,
  `tpc_tick`, `producer_id`, `y_center`, `z_center`) and
  `entry_0/pmt_positions` (32×3). Detector-level only — the truth-level
  slice↔flash matching is a downstream analysis concern.
- `entry_0` attrs `run / subrun / event`.

The full ghost-included triplet set is kept (`hasmatch` 0/1) — that is what
the deghoster was trained to classify. No pre-filtering.

### Stage B — full cascade

Runs the trained `CascadedParticleSegmenter` (LoRA deghoster → ptv3crosslevel
slicer → ptv3crosslevel particle segmenter) in a single forward via
[`run_larformer_stage3_inference.py --input-mode full-cascade`]. Output
`stage3pred_*.h5` carries both the slicer half (`pre/ post/ queries/ gt/
meta/ levels/`) and the particle half (`stage3/ stage3_queries/ stage3_gt/
stage3_levels/ stage3_meta/`). Visualize with
[`tools/visualize_stage3_larformer_from_cached.py --stage3pred-dir`] and
[`tools/visualize_larformer_gt.py --slicerpred-dir`].

Config: [`../../configs/lartpc/larformer/stage3_particle/larformer-particle-fullcascade-ptv3crosslevel.py`](../../configs/lartpc/larformer/stage3_particle/larformer-particle-fullcascade-ptv3crosslevel.py)
(derived from the trained cached config `larformer-particle-v1-cached-ptv3crosslevel.py`).

**Query dedup (2026-06-11).** `run_larformer_stage3_inference.py` now
applies a mask-IoU NMS over the Stage-3 queries by default
(`--dedup-iou-threshold 0.6`; `0` disables): co-extensive duplicate
queries — characteristically a μ + π pair hedging one ambiguous track —
are merged before the per-SP panoptic assignment, with the absorbed
query's class hypothesis recorded under `stage3_queries/dedup_*` and the
pre-dedup assignment preserved in `stage3/pred_query_nodedup`. Design +
schema: [`../../docs/LArFormer_Stage3_TrainingStability.md`](../../docs/LArFormer_Stage3_TrainingStability.md) §7.

### Visualizing the output

[`tools/visualize_full_cascade.py`](../../tools/visualize_full_cascade.py) —
two camera-synced 3D rows: PREDICTION (predicted slices colored by query, with
the nu-candidate slice rendered as the Stage-3 particle segmentation) and
GROUND TRUTH (truth slices + nu-slice truth particles). Because Stage B runs
GT-less, the truth panel is recomputed from the source `merged_h5` via
`LArFormerDataset` (reusing the exact label code); pass `--merged-dir` to
enable it. Real-data events (no `mc_particle_tree`) show an empty GT panel.

```bash
python tools/visualize_full_cascade.py \
    --stage3pred-dir OUTPUT_DIR/000/000 \
    --merged-dir     OUTPUT_DIR/000/000        # optional, for the GT row
# open http://<host>:8051
```

Runs in the pointcept container, CPU-only (no GPU needed). Built on the same
`pointcept/models/LArFormer/viz_inference.py` color/figure helpers as
`tools/visualize_stage3_larformer_from_cached.py` and the slicer overlay in
`tools/visualize_larformer_gt.py`.

**Pure-inference, GT-less.** The whole cascade runs with `gt_source="deghost"`
+ `--no-gt`. The frozen 3-class slicer cannot consume 7-class particle GT in
its eval matcher (CUDA-asserts — see `docs/LArFormer_particlesegment_stage.md`
§13.2). GT-matched evaluation metrics are a separate concern: for those, use
the **cached** inference path (`tools/build_stage12_cache_shard.py` →
`run_larformer_stage3_inference.py --input-mode cached`), which has the
`particle_class_id` augmentation and slice-level GT.

**Checkpoint re-prefix.** The trained Stage-3 weights were saved from a
*standalone* `LArFormer` (un-prefixed keys). [`run_stepB_cascade_wconfig.sh`]
re-prefixes them to `particle_segmenter.*` via
[`prefix_particle_ckpt.py`](prefix_particle_ckpt.py) before passing them as
`--weights`; the deghoster + slicer weights load via the cascade config's
`cascaded_slicer_weight`/`deghoster_weight` in `__init__`.

---

## Config-driven workflow

Everything is driven by a sourced bash config in
[`larformer_configs/`](larformer_configs/). Three are provided:

| Config | Sample | Flags |
|--------|--------|-------|
| `bnb_nu_pi0filter_corsika.conf` | newer sim (training sample) | `--adc wiremc`, tick-forward, MC truth |
| `mcc9_v29e_nue_overlay.conf` | older official sim | `--adc wire -tb --mcc9` |
| `bnb5e19.conf` | real detector data | `--adc wire -tb --is-data` |

The only per-dataset differences are these flags + paths/tag. Real data:
`SimChTripletLabelMaker` runs in `--is-data` mode (writes a bogus
`mc_particle_tree` and `trackid/pid` = -1); the cascade runs GT-less so that
is harmless.

### Run end-to-end

```bash
cd Pointcept/lartpc_data_prep/larformer_scripts
# (bare node / inside or outside the container — the bootstrap re-execs into
#  the pointcept container automatically)
source run_larformer_wconfig.sh larformer_configs/bnb_nu_pi0filter_corsika.conf
```

`SLURM_ARRAY_TASK_ID` (default 0) selects the stride block:
`lineno = OFFSET + stride*SLURM_ARRAY_TASK_ID + i`, `i=1..stride`. Outputs go
to `OUTPUT_DIR/<lineno/1000>/<lineno/100>/` (3-level hash).

### Run a single stage standalone

```bash
source run_stepA_convert_wconfig.sh larformer_configs/bnb5e19.conf [lineno]
source run_stepB_cascade_wconfig.sh larformer_configs/bnb5e19.conf [lineno]
```

Each subscript re-execs into the pointcept container if not already inside it.

### Key config knobs

| Knob | Meaning |
|------|---------|
| `INPUTLIST` / `TAG` / `OUTPUT_DIR` | dataset list, tag, output tree |
| `ADCNAME` / `TBFLAG` / `MCC9FLAG` / `ISDATAFLAG` | per-dataset conversion flags |
| `RUN_CASCADE` | 0 = Stage A only; 1 = A+B |
| `CLASS_PROB_THRESHOLD` | Stage-3 panoptic confidence floor (default 0.3) |
| `WORKDIR_BASE` | parent of per-file workdirs (`/tmp` ephemeral; persistent path to keep intermediates) |
| `KEEP_INTERMEDIATES` | 1 = don't wipe workdir, skip stages whose outputs exist |
| `MAX_EVENTS` | `-n N` to the converter (-1 = all) |
| `BIND_FOLDERS` | container binds (laptop: `/mnt/ddrive:/mnt/ddrive,/home:/home`; cluster: `/cluster:/cluster`) |
| `LARFORMER_*_CKPT` | optional cascade checkpoint overrides (else config defaults) |

---

## Open item — slicer checkpoint provenance

The Stage-3 weights (`model_iter_98652.pth`) were trained on a Stage-1+2 cache
built with a ptv3crosslevel slicer at **iter 75750**
(`exp/cache_stage12_ptv3crosslevelslicer_iter_75750/`). That exact slicer
checkpoint was not present in this tree; the cascade config defaults
`LARFORMER_SLICER_CKPT` to the best available ptv3crosslevel slicer
(`..._iter_37625.pth`). For the most faithful cascade, set
`LARFORMER_SLICER_CKPT` to the iter-75750 checkpoint if it can be recovered.

---

## What changed in the model repo

One small additive change to support full-cascade inference on raw
merged_h5: `pointcept/datasets/larformer.py` now emits a per-SP
`particle_class_id` field (real when particle GT instances are present, else
a `-1` stub). The model (`_per_sp_labels_for_event`) and `larformer_collate`
already supported the key; the dataset just wasn't emitting it. The `-1`
stub yields an all-ignore per-level cls *target* — the trained cls *head*
still runs (it feeds `mixed_query_selection`); only the unused loss target is
masked. No effect on training (Stage-3 trains from the cache dataset) or on
predictions.
