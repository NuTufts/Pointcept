# Running the bnb5e19 inference chain on Polaris (ALCF) — handoff

This document is for a fresh Claude session (or a person) working on Polaris.
It says what the job is, where everything is, which code to read, and how to
test and launch. Read it fully before touching the cluster.

## 1. Goal

Run the LArFormer **keypoint2 cascade GPU inference** over the full MicroBooNE
**bnb5e19** sample (run-1 beam-on data, 176,302 events) with the **calibrated
flash light-yield gamma** (2026-09-12), and then either run the CPU tail
(nu_reco -> LArPID -> ntuple export -> hadd) on Polaris too, or ship the
cascade outputs back to Tufts for the tail. The product is the run-1 beam-data
ntuple of the "table-gamma" campaign (data vs simulation with all run-1
inputs: EXT half sample done at Tufts, run-1 MC overlay next, then this
bnb5e19 reprocess). The previous bnb5e19 production (cew6, at Tufts) used the
legacy gamma (0.80 -> gamma_eff 4.20) and is being replaced.

Why Polaris: A100 GPUs in quantity. At Tufts one A100-40GB shard does ~1.9 s
per event (6,533 events in 3.3 h), so the full sample is ~95 GPU-hours: on
16 Polaris nodes x 4 A100 = 64 shards of ~2,750 events, about 1.5 h.

## 2. Where things are on Polaris

| what | path |
|---|---|
| repo clone (branch `nutufts_lartpc_keypointdev_v2`) | `/eagle/neutrinoGPU/twongj01/larformer_pointcept` |
| assets (checkpoints, PhotonLib cache, container, `env.sh`, `sha256sums.txt`) | `/eagle/neutrinoGPU/twongj01/polaris_assets` |
| bnb5e19 merged_sp as 12 squashfs images + manifests | `/eagle/neutrinoGPU/twongj01/data/uboone/mcc9_v28_wctagger_bnb5e19` |
| project / allocation | `neutrinoGPU` |

Assets layout (see `docs/reference/Running_Inference_Offsite.md`):
`kpv2_assets/{sonata,exp,lartpc/flashmatch/data}/...` (deghoster, slicer,
segmenter, keypoint checkpoints, PhotonLib cache), `oldrepo_assets/sonata/...`
(Sonata pretrain, loaded into the backbones at model build), `pointcept_cuml.sif`.

## 3. What to read in the repo (in this order)

1. `docs/reference/Running_Inference_Offsite.md` — the asset list, the two
   env roots (`LARFORMER_KPV2_ROOT`, `LARFORMER_OLD_REPO`) and the per-shard
   command. `polaris_assets/env.sh` sets every env var; edit its two roots.
2. `lartpc/data_prep/squashfs/README.md` + `mount_args.sh` — how the 12
   images are bind-mounted into the container so the tree looks like
   `<mount>/NNN/NN/merged_bnb5e19_filenoNNNNN_entryNNNNNN.h5`.
3. `lartpc/polaris/make_polaris_list.py` — builds the input list in the
   Tufts PRODUCTION order from `bnb5e19_production_basenames.txt.gz`
   (176,302 of the 176,336 files; the order defines the cascade event index,
   keep it).
4. `tools/larformer/run_larformer_keypoint2_cascade_inference.py` — the entry
   point. Read its argparse block (`--gamma-run-scale`, `--sample-kind`,
   `--flash-window`, `--deterministic`, `--save-score-maps`, `--output-tree`,
   `--no-gt`, `--start-event/--n-events`, `--photonlib`) and the event loop
   near the end (it skips unreadable inputs; writes `keypoint2_event{i:05d}_0.h5`
   into a 2-level index tree).
5. `lartpc/larformer_reco/slurm/submit_inference_shard.sh` and
   `submit_extbnb_chain.sh` — how Tufts shards the list (contiguous ranges,
   `NSHARDS`), the retry loop, and the pinned "chain version" block (config +
   checkpoints + BDTs + LLR tables). Polaris is PBS, so the sbatch array
   becomes a PBS job array or a per-node loop; the python command is the same.
6. `lartpc/flashmatch/flash_calib.py` — the gamma table: cell
   `("data", 1) = 0.5449` -> gamma_eff = 5.25 x 0.5449 = 2.861; beam-on run-1
   flash window `(2.8, 5.0)` us. The cascade records `flash` attrs
   `gamma_spec/gamma_scale/gamma_eff/sample_kind/flash_window` in every file.
7. `lartpc/larformer_reco/larformer_reco_output_data_schema.md` — output format.
8. `lartpc/larformer_analysis/model_and_output_file_versions.md` — the chain
   version (v2_s1ep2p8cew6) and the calibrated-gamma policy.
9. For the CPU tail: `lartpc/larformer_reco/slurm/submit_nu_reco_shard.sh`,
   `submit_larpid_shard_cpu.sh`, `submit_export_shard.sh`,
   `submit_export_merge.sh`, `regen_kp2_list.sh`, and
   `lartpc/larformer_reco/README.md`.

## 4. Test, then launch

### 4.1 Environment sanity (debug queue, 1 node)

    module load apptainer
    cd /eagle/neutrinoGPU/twongj01/larformer_pointcept
    (cd /eagle/neutrinoGPU/twongj01/polaris_assets && sha256sum -c sha256sums.txt)   # all OK
    export LARFORMER_KPV2_ROOT=/eagle/neutrinoGPU/twongj01/polaris_assets/kpv2_assets
    export LARFORMER_OLD_REPO=/eagle/neutrinoGPU/twongj01/polaris_assets/oldrepo_assets
    source /eagle/neutrinoGPU/twongj01/polaris_assets/env.sh
    export HDF5_USE_FILE_LOCKING=FALSE        # Lustre: h5py file locking is unreliable
    IMG=/eagle/neutrinoGPU/twongj01/data/uboone/mcc9_v28_wctagger_bnb5e19
    SIF=/eagle/neutrinoGPU/twongj01/polaris_assets/pointcept_cuml.sif
    apptainer exec --nv $(bash lartpc/data_prep/squashfs/mount_args.sh $IMG /data/bnb5e19/merged_sp) $SIF \
      bash -c "nvidia-smi -L; cd $PWD && export PYTHONPATH=\$PWD && python3 -c 'import torch, pointops; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))' && ls /data/bnb5e19/merged_sp | head -3"

Expect: 12 checksums OK, 4 GPUs listed, `True NVIDIA A100...`, dirs `000 001 002`.
If `pointops` fails to import, the container was built for the Tufts CUDA
driver; check `nvidia-smi` driver vs the container's CUDA (apptainer `--nv`
usually handles it).

### 4.2 Input list (once)

    apptainer exec $(bash lartpc/data_prep/squashfs/mount_args.sh $IMG /data/bnb5e19/merged_sp) $SIF \
      bash -c "cd $PWD && python3 lartpc/polaris/make_polaris_list.py --merged-sp /data/bnb5e19/merged_sp \
               --out lartpc/larformer_reco/inputlists/merged_sp_bnb5e19_polaris.txt"

Expect `wrote 176302 paths in production order`. The paths are container
paths (`/data/bnb5e19/merged_sp/...`), so every inference process must mount
the images at the same `/data/bnb5e19/merged_sp`.

### 4.3 Two-event smoke (interactive, one GPU)

    OUT=/eagle/neutrinoGPU/twongj01/data/uboone/bnb5e19_kp2_polaris
    apptainer exec --nv $(bash lartpc/data_prep/squashfs/mount_args.sh $IMG /data/bnb5e19/merged_sp) $SIF bash -c "
      cd $PWD && export PYTHONPATH=\$PWD && export HDF5_USE_FILE_LOCKING=FALSE && \
      python3 tools/larformer/run_larformer_keypoint2_cascade_inference.py \
        --config configs/lartpc/larformer/stage4_keypoint/larformer-keypoint2-fullcascade-v6lantern-envslicer.py \
        --input-list lartpc/larformer_reco/inputlists/merged_sp_bnb5e19_polaris.txt \
        --output-dir $OUT/smoke/ --start-event 0 --n-events 2 \
        --deterministic --save-score-maps --device cuda --output-tree --no-gt \
        --gamma-run-scale table --sample-kind data --flash-window 2.8,5.0 \
        --photonlib \$LARFORMER_KPV2_ROOT/lartpc/flashmatch/data/photonlib_v6_70kV.npz"

Check the log for the chain being built (`[DeghostSegmentor] Injecting LoRA`,
slicer/particle checkpoints = the mixenriched epoch_2 / cew epoch_6 paths),
two `n_particles=` lines and `DONE`; then
`h5dump -a /flash/gamma_eff -a /flash/gamma_spec -a /flash/sample_kind -a /flash/flash_window <file>`
must show `2.86073`, `"table"`, `"data"`, `"2.8,5.0"`, and `/stream` = `"nu"`.
`--no-gt` is right for data (no truth). Do NOT pass `--no-flash`.

### 4.4 Throughput (one node, 4 GPUs, ~200 events each)

Launch 4 processes on one node, one per GPU (`CUDA_VISIBLE_DEVICES=k`), with
`--start-event k*200 --n-events 200` into a scratch `--output-dir`. Measure
seconds/event from the log timestamps; expect ~1.5-2 s/event. Memory: the
Tufts shards run with 32 GB host RAM per process; 4 processes/node fit.

### 4.5 Production launch

Shard the 176,302-event list into contiguous ranges (`NSHARDS` = nodes x 4;
`PER=ceil(176302/NSHARDS)`, shard k = `--start-event k*PER --n-events PER`),
one process per GPU, all writing into ONE `--output-dir` with `--output-tree`
(files never collide: names carry the event index). PBS sketch:

    qsub -A neutrinoGPU -q prod -l select=16:system=polaris -l walltime=03:00:00 \
         -l filesystems=home:eagle -l place=scatter launch_bnb5e19.pbs

where `launch_bnb5e19.pbs` runs, on each node (`mpiexec -n <nodes> --ppn 1`
or a `$PBS_NODEFILE` loop), a script that starts 4 background python
processes (one per `CUDA_VISIBLE_DEVICES`) for that node's 4 shard indices
and waits. Give every shard a log `logs/inference/polaris_<shard>.log`; the
Tufts shard script's retry loop (3 attempts) is worth copying. Keep per-job
walltime generous (a shard of 2,750 events is ~1.5 h; ask for 3 h).

### 4.6 Verify

* `find $OUT -name 'keypoint2_event*_0.h5' | wc -l` ~ 176k (events with no nu
  slice have no file; Tufts bnb5e19 had one file per event within ~0.1%).
* Every shard log ends with `DONE`; no `Traceback` in the logs; count
  `SKIP unreadable` lines (Tufts bnb5e19 had no corrupt inputs).
* Spot-check attrs on a few files (4.3), and compare `n_particles` per event
  for ~20 events against Tufts (`/cluster/tufts/wongjiradlab/larbys/data/ub_on_tufts/larformer_bnb5e19_s1ep2p8cew6/keypoint2_streams`):
  partitions should match; `pred_pe`/`chi2` differ by the gamma (2.861 vs 4.20)
  and at the float32 level (different GPU type).

### 4.7 After the GPU pass

Either (a) run the CPU tail on Polaris: `regen_kp2_list.sh` (nu/fm lists) ->
`run_nu_reco.py` shards -> LArPID (`lartpc/larformer_reco/larpid/apply_larpid.py`;
needs the external `prongCNN` repo + its `checkpoints/LArPID_default_network_weights.pt`,
NOT in this repo: `lartpc/larformer_reco/larpid/model.py` `PRONGCNN_DIR`) ->
`export_gen2ntuple.py` (data mode, both cew6 shower BDTs in git) -> hadd; or
(b) pack `$OUT/keypoint2_streams` with
`lartpc/data_prep/squashfs/pack_merged_sp_squashfs.sh` (it packs any 2-level
tree; ~11 GB per 100k events, so ~20 GB total) and Globus it back to Tufts,
where `submit_extbnb_chain.sh RESUME_AFTER_INF=...` style resumption of the
tail is available. (b) is simpler; (a) needs prongCNN shipped over.

## 5. Gotchas

* Set `HDF5_USE_FILE_LOCKING=FALSE` in every process (Lustre).
* The squashfs mounts are read-only and must be mounted at the same
  container path the list was built with.
* The configs resolve checkpoint paths from `LARFORMER_KPV2_ROOT` /
  `LARFORMER_OLD_REPO`; if you copy assets INTO the clone instead, the
  defaults in `env.sh` can point at the clone.
* Never launch the same shard twice into the same output tree concurrently
  (two writers on one file name); the event index in the name is the only key.
* Determinism (`--deterministic`) is per GPU type; A100 vs Tufts L40S/V100
  differ at float32 level upstream of the flash table — expected.
* The Tufts production flags for this sample (for a like-for-like ntuple
  later): `--output-tree --gamma-run-scale table --sample-kind data
  --flash-window 2.8,5.0 --deterministic --save-score-maps`, default LArPID
  weights (run-1 data), export in data mode.

## 6. Report back

When done, record in `lartpc/larformer_analysis/flashmodel_calib/CALIBRATION_LOG.md`
(or a new dated note in `docs/devlog/`): number of files, any skipped events,
GPU-hours used, where the outputs live on Eagle, and the spot-check result.
