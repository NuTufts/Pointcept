# LArTPC Configs

Configs are organized by **project**, with per-stage subfolders for LArFormer.
Within each folder, the production/current configs sit at the top level and past
experiments live in `archive/` — each folder's README documents what the current
and archived configs were for. Reorg background: `docs/Reorganization_Plan.md` §2.

```
configs/lartpc/
├── sonata_pretrain/     Sonata self-supervised backbone pre-training (+ probes/, archive/)
├── lora_finetune/       LoRA finetune performance project (deghost, semseg)
├── larformer/           staged LArFormer training
│   ├── stage1_deghost/  per-spacepoint ghost removal
│   ├── stage2_slicer/   interaction slicing
│   ├── stage3_particle/ particle instance segmentation
│   └── stage4_keypoint/ per-particle keypoints + nu-vertex
├── semseg/              early supervised semantic segmentation (historical)
└── shower_origin/       shower origin / clustering studies (exploratory, archived)
```

## Production configuration chain

| Stage | Config | Notes |
|---|---|---|
| Sonata pre-training (sim-only backbone) | `sonata_pretrain/pretrain-sonata-v1m1-lartpc-v6-logspace-resume.py` | Fixes a data-presentation mistake in the earlier v6 config (now archived). Feeds the slicer / particle / keypoint stages. |
| Sonata pre-training (ghost-aware backbone) | `sonata_pretrain/pretrain-sonata-v7-extbnb-larmatch.py` | Trained on data including ghost points and real (extbnb) data; backbone for deghosting. |
| **Stage 1 — deghosting** | `larformer/stage1_deghost/lorafinetune-sonata-v1m1-lartpc-v6-deghost-extbnb-larmatch.py` | LoRA finetune on the v7 backbone. Owned by `lora_finetune/` (duplicate kept in the stage chain). |
| **Stage 2 — event slicing** | `larformer/stage2_slicer/larformer-slicer-v1-cascaded-ptv3hybrid-crosslevel.py` | PTv3-hybrid tokenization + cross-level attention refiner. |
| **Stage 3 — particle segmentation** | `larformer/stage3_particle/larformer-particle-v1-cached-ptv3crosslevel-decaylrsched.py` | Final decay-LR run, warm-started from a flat-LR run of `larformer-particle-v1-cached-ptv3crosslevel.py`. |
| **Stage 4 — keypoints** | `larformer/stage4_keypoint/larformer-keypoint2-particle-predmask-cached-v1.py` | Trained on the Stage-1+2+3 cache with predicted masks. Inherits the (live, non-archived) `larformer-keypoint2-particle-v1.py` → `larformer-keypoint2-slice-v1.py` chain. |

## Production inference / cache-building configs

| Purpose | Config | Used by |
|---|---|---|
| Full-cascade keypoint2 inference | `larformer/stage4_keypoint/larformer-keypoint2-fullcascade.py` | `tools/run_larformer_keypoint2_cascade_inference.py`, keypoint_v2 submit scripts |
| Full-cascade Stage-3 inference (official data pipeline) | `larformer/stage3_particle/larformer-particle-fullcascade-ptv3crosslevel.py` | Default in `lartpc_data_prep/larformer_scripts/run_stepB_cascade_wconfig.sh` |
| Stage-1+2 cache building | `larformer/stage3_particle/larformer-particle-v1.py` | `tools/build_stage12_cache_{event,shard}.py`, `tools/benchmark_larformer_s3_cascade.py` |

## Conventions

- `scripts/train.sh` takes subfolder configs as
  `-c larformer/stage2_slicer/larformer-slicer-v1-cascaded-ptv3hybrid-crosslevel`.
- `_base_` paths are relative to each config file; configs one folder deep use
  `../../_base_/`, stage folders `../../../_base_/`, stage archives `../../../../_base_/`.
- Never delete a config from `archive/` without checking nothing inherits from it
  (`grep -rn "_base_" configs/lartpc/ | grep <name>`).
