# LArTPC Configs — Production Manifest

This directory holds all configs for the LArTPC projects (Sonata pre-training,
LoRA finetuning, LArFormer staged training, and earlier semseg / shower-origin work).
A reorganization into per-project folders is planned — see
[`docs/Reorganization_Plan.md`](../../docs/Reorganization_Plan.md) §2. Until then,
this manifest records **which configs produced the current production checkpoints**.

## Production configuration chain

| Stage | Config | Notes |
|---|---|---|
| Sonata pre-training (sim-only backbone) | `pretrain-sonata-v1m1-lartpc-v6-logspace-resume.py` | Fixes a data-presentation mistake in the earlier `pretrain-sonata-v1m1-lartpc-v6.py`. This backbone feeds the slicer / particle / keypoint stages. |
| Sonata pre-training (ghost-aware backbone) | `pretrain-sonata-v7-extbnb-larmatch.py` | Trained on a dataset that includes ghost points and real data; used as the backbone for the deghosting stage. |
| **Stage 1 — deghosting** | `lorafinetune-sonata-v1m1-lartpc-v6-deghost-extbnb-larmatch.py` | LoRA finetune on the v7 extbnb-larmatch backbone. |
| **Stage 2 — event slicing** | `larformer-slicer-v1-cascaded-ptv3hybrid-crosslevel.py` | PTv3-hybrid tokenization + cross-level attention refiner. |
| **Stage 3 — particle segmentation** | `larformer-particle-v1-cached-ptv3crosslevel-decaylrsched.py` | Final decay-LR run. Warm-started from a flat-LR run of `larformer-particle-v1-cached-ptv3crosslevel.py`. |
| **Stage 4 — keypoints** | `larformer-keypoint2-particle-predmask-cached-v1.py` | Trained on the Stage-1+2+3 cache with predicted masks. |

## Production inference / cache-building configs

| Purpose | Config | Used by |
|---|---|---|
| Full-cascade keypoint2 inference | `larformer-keypoint2-fullcascade.py` | `tools/run_larformer_keypoint2_cascade_inference.py`, keypoint_v2 submit scripts |
| Full-cascade Stage-3 inference (official data pipeline) | `larformer-particle-fullcascade-ptv3crosslevel.py` | Default in `lartpc_data_prep/larformer_scripts/run_stepB_cascade_wconfig.sh` |
| Stage-1+2 cache building | `larformer-particle-v1.py` | `tools/build_stage12_cache_{event,shard}.py`, `tools/benchmark_larformer_s3_cascade.py` |

## Everything else

All other configs here are earlier generations or ablation experiments
(v0 deghost/slicer, refiner variants, keypoint phases 1–2, semseg finetunes,
shower-origin/clustering studies, μP studies). They will move into per-project
`archive/` folders with per-folder READMEs during the config reorg.
