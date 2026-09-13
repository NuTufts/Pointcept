# Running keypoint2 cascade inference on another cluster (Polaris)

What a fresh clone of this repo is missing, and how to point it at the assets.

## Not in git (transfer these)

Staged by `lartpc/data_prep/squashfs/stage_inference_assets.sh <out>` into one
directory (Tufts copy: `/cluster/tufts/wongjiradlab/larbys/data/larformer/polaris_assets/`,
~18 GB, `sha256sums.txt` + `MANIFEST.txt` included):

| file | size | role |
|---|---|---|
| `kpv2_assets/sonata/lora_deghost_v6noghosts_lantern/model/epoch_25.pth` | 0.40 GB | deghoster (self-contained: backbone + LoRA) |
| `kpv2_assets/exp/larformer_slicer_s1_mixenriched_v1/model/epoch_2.pth` | 1.47 GB | slicer |
| `kpv2_assets/exp/larformer_particle_s1cache_m2frecipe_rebal_v2_warm2_cew/model/epoch_6.pth` | 1.08 GB | stage-3 segmenter (cew6) |
| `kpv2_assets/exp/larformer_keypoint2_particle_cachedpredmask_v1/model/epoch_30.pth` | 0.95 GB | keypoint head |
| `kpv2_assets/lartpc/flashmatch/data/photonlib_v6_70kV.npz` | 0.06 GB | PhotonLib cache (flash prediction) |
| `oldrepo_assets/sonata/lartpc_v6_h200_noghosts_pretrain_logspace_resume/model/epoch_42.pth` | 1.57 GB | Sonata pretrain, loaded into the slicer/segmenter backbones at model build (then overwritten by the checkpoints above; still must exist) |
| `pointcept_cuml.sif` | 12.6 GB | the container (custom CUDA ops are built inside it) |

Everything else (configs, code, LLR tables, SCE maps, shower BDTs, calo
calibration, flash gamma table) is tracked in git. NOTE: `.gitignore` excludes
`*.npz` and `*.so`, so new data files of those types must be `git add -f`'ed
(the s1ep2p8 attachment LLR table and `lib_wirecell_fiducial_volume.so` were
missing on Polaris until 2026-09-13; `pol_check_env` now asserts them).

## Roots and environment

The configs used to hardcode the Tufts checkout paths; they now read two env
variables with the Tufts defaults, so a clone anywhere works without edits:

* `LARFORMER_KPV2_ROOT` -- where `sonata/`, `exp/` and `lartpc/flashmatch/data/`
  live (the clone itself if you copy `kpv2_assets/*` into it, or the
  `kpv2_assets` directory).
* `LARFORMER_OLD_REPO` -- where the Sonata pretrain lives (`oldrepo_assets`).

`<assets>/env.sh` sets both plus the per-stage checkpoint variables (all of
them must be exported: `LARFORMER_SONATA_PRETRAIN` is also read by the keypoint
model's config, whose base file hard-codes the Tufts path)
(`LARFORMER_BATTERY_SLICER_CKPT`, `LARFORMER_KP_PARTICLE_CKPT`,
`LARFORMER_KP_KEYPOINT_CKPT`, `LARFORMER_SONATA_PRETRAIN`). Edit the two roots
at its top, then `source env.sh`.

## Data

bnb5e19 merged_sp as 12 squashfs images (`lartpc/data_prep/squashfs/README.md`):
mount with `mount_args.sh`, build the list with `list_merged_sp.py` (stable
fileno/entry order = identical index linkage to the Tufts production).

## Command (per shard; Polaris is PBS, so replace the Slurm array with a PBS
array or a per-node loop over --start-event/--n-events)

    source <assets>/env.sh
    apptainer exec --nv $(bash lartpc/data_prep/squashfs/mount_args.sh <images> /data/bnb5e19/merged_sp) \
      <assets>/pointcept_cuml.sif bash -c "
      cd <clone> && export PYTHONPATH=\$PWD && \
      python3 tools/larformer/run_larformer_keypoint2_cascade_inference.py \
        --config configs/lartpc/larformer/stage4_keypoint/larformer-keypoint2-fullcascade-v6lantern-envslicer.py \
        --input-list <list> --output-dir <out>/keypoint2_streams/ \
        --start-event \$START --n-events \$N \
        --deterministic --save-score-maps --device cuda --output-tree --no-gt \
        --gamma-run-scale table --sample-kind data --flash-window 2.8,5.0 \
        --photonlib \$LARFORMER_KPV2_ROOT/lartpc/flashmatch/data/photonlib_v6_70kV.npz"

Flags match `lartpc/larformer_reco/slurm/submit_inference_shard.sh` +
`submit_extbnb_chain.sh` (gamma table cell (data, 1) = 0.5449, beam-on run-1
window 2.8-5.0 us). Determinism is per GPU type: A100 outputs differ from the
Tufts ones at the float32 level in the slicer partition (18/20 events
bit-identical in the Tufts regression; the rest differ upstream of the flash
table).

Downstream stages (nu_reco, LArPID, export) are CPU-only and can run either
side; LArPID needs its network weights (`lartpc/larformer_reco/larpid/`,
check `select_checkpoint`) and the exporter the shower BDTs (in git).
