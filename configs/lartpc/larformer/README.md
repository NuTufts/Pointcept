# LArFormer Training Configs

Staged training of the LArFormer cascade. Each stage folder holds the production
config(s) at its top level and past experiments in `archive/`. See
`docs/LArFormer.md` for the architecture and `../README.md` for the full manifest.

```
Stage 1  deghost      per-spacepoint P(real), LoRA on ghost-aware Sonata backbone
   ↓ frozen
Stage 2  slicer       Mask2Former queries → interaction slices
   ↓ frozen (or cached)
Stage 3  particle     per-slice particle instance segmentation (7 classes)
   ↓ frozen (or cached)
Stage 4  keypoint     per-particle start/end/vertex + dense nu-vertex scores
```

| Stage | Production config |
|---|---|
| 1 | `stage1_deghost/lorafinetune-sonata-v1m1-lartpc-v6-deghost-extbnb-larmatch.py` |
| 2 | `stage2_slicer/larformer-slicer-v1-cascaded-ptv3hybrid-crosslevel.py` |
| 3 | `stage3_particle/larformer-particle-v1-cached-ptv3crosslevel-decaylrsched.py` (warm-started from the flat-LR `larformer-particle-v1-cached-ptv3crosslevel.py`) |
| 4 | `stage4_keypoint/larformer-keypoint2-particle-predmask-cached-v1.py` |
| full-cascade inference | `stage4_keypoint/larformer-keypoint2-fullcascade.py` (keypoint2); `stage3_particle/larformer-particle-fullcascade-ptv3crosslevel.py` (official-data stepB) |

Training generally uses the **cached** configs (Stage-1+2 outputs precomputed with
`tools/build_stage12_cache_*.py`); the full-cascade configs are for inference on raw
merged H5.

## Per-stage notes

- **stage1_deghost/**: the production config is a duplicate of the one in
  `../lora_finetune/` (that project owns it; the copy here completes the stage chain).
  `archive/` holds the v0 standalone deghost trainings that predate the LoRA approach.
- **stage2_slicer/**: `archive/` holds the v0 standalone slicer and the v1-cascaded
  refiner ablations (identity / per-level / cross-level / ptv3decoder /
  ptv3hybrid_perlevel / loradeghost).
- **stage3_particle/**: `larformer-particle-v1.py` stays at top level — it is the
  config `tools/build_stage12_cache_{event,shard}.py` load to build the training cache.
  `archive/` holds the pre-ptv3crosslevel cached baseline and the v1.1 tweak.
- **stage4_keypoint/**: the production cached config inherits
  `./larformer-keypoint2-particle-v1.py`, which inherits
  `./larformer-keypoint2-slice-v1.py` — those two are live parents, do **not**
  archive them. `archive/` holds keypoint phases 1–2 (dense heatmap and query-decoder
  attempts on the frozen Stage-3 model; superseded by the keypoint-v2 design).
  Note: `archive/larformer-keypoint-query-v1.py` no longer loads
  (`RESUME_STEP` undefined — pre-existing breakage, kept as-is for provenance).
