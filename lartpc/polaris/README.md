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
| bnb5e19 merged_sp as 12 squashfs images + manifests | `/eagle/neutrinoGPU/twongj01/data/uboone/mcc9_v28_wctagger_bnb5e19/squashfs/` |
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
   `<mount>/NNN/NN/merged_bnb5e19_filenoNNNNN_entryNNNNNN.h5`. The mount root
   must exist in the container: `polaris_env.sh` uses a host directory
   (`$POL_DATADIR/merged_sp_mnt`, 12 empty `NNN/` mount points) so that only
   bind mounts are needed; a root like `/data/...` that is absent from the
   image needs `--writable-tmpfs` or `--fakeroot` (mkdir of the mount point
   fails otherwise: seen at Tufts with apptainer 1.5).
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

## 4. Scripts (`lartpc/polaris/`) and the procedure

Everything below is scripted; the hand-typed commands of the first version of
this document are superseded. Site-specific values live in ONE file.

| file | role |
|---|---|
| `polaris_env.sh` | all Eagle paths, accounts, chain-version pins, `pol_exec` (container + squashfs mounts), `pol_check_env` |
| `make_worklist.py` | shard ranges -> text worklist (cascade / range / nu_reco / larpid / export) |
| `node_launcher.sh` | per-node runner: takes its share of a worklist, runs `ppn` tasks concurrently (`SLOT` -> GPU) |
| `run_cascade_shard.sh` | one GPU shard: retry x3, timestamped log, `.done` / `.running` markers, idempotent |
| `cascade.pbs`, `qsub_cascade.sh` | the GPU pass (16 nodes x 4 GPUs) and its qsub builder with guards |
| `test_cascade.pbs` | first job: checksums, imports, list, 2-event smoke, 4-GPU throughput, Tufts comparison |
| `check_cascade.py` | per-shard verification + `.resume` worklist + s/event (stdlib, host python) |
| `check_kp2_attrs.py` | asserts gamma_eff / gamma_spec / sample_kind / flash_window / stream (container) |
| `make_tufts_reference.py`, `tufts_ref_bnb5e19_idx0-499.json`, `compare_to_tufts.py` | event-by-event comparison with the Tufts cew6 production, keyed on `src_file` |
| `run_tail_task.sh`, `tail_driver.sh`, `tail.pbs`, `qsub_tail.sh`, `hadd_check.py` | CPU tail on Polaris: regen -> nu_reco -> LArPID -> export -> hadd (+ entry-count assert) |
| `compare_platforms.py` | cross-platform measurement: kp2 tree vs tree, ntuple vs ntuple (flip rates + errors) |
| `make_polaris_list.py` | input list in production order (unchanged) |

Layout of `$POL_DATADIR` (default `/eagle/neutrinoGPU/twongj01/data/uboone/bnb5e19_kp2_polaris`):
`lists/` (merged_sp list, worklists), `logs/<tag>/`, `merged_sp_mnt/` (12 empty
mount points; the images appear there INSIDE the container only),
`tests/{smoke,throughput}/`, `keypoint2_streams/` (production cascade tree),
`nu_reco_streams_{nu,fm}/`, `nu_reco_larpid_{nu,fm}/`, the final ntuple.

Polaris facts the scripts assume (checked 2026-09-12, `docs/reference/Polaris_Site_Info_2026-09-12.md`):
apptainer 1.4.1 via `module use /soft/modulefiles; module load spack-pe-base apptainer`;
queues `debug` 1-2 nodes / 1 h / 1 running job per user, `prod` -> `small`
10-24 nodes / 3 h, `preemptable` 1-10 nodes / 72 h; `-A <project>::<suballocation>`
is mandatory (submanagement): `neutrinoGPU::DetsimGPU` (100 node-hours, production)
and `neutrinoGPU::debug` (tests) -- confirm with
`sbank-list-allocations -r polaris -p neutrinoGPU -f "+subname users_list"` that
your user is on both. PBS charges elapsed wall x nodes only, so walltimes are
generous. Budget for this campaign: ~25-30 node-hours GPU pass, ~10 tail, <1 tests.

### 4.1 Once: pull, configure, check (login node, no job)

    cd /eagle/neutrinoGPU/twongj01/larformer_pointcept && git pull
    $EDITOR lartpc/polaris/polaris_env.sh      # the POL_* defaults at the top; edit only if a path differs
    source lartpc/polaris/polaris_env.sh && pol_check_env   # every asset, 12 images, apptainer -> PASS

### 4.2 Test job (debug queue, 1 node, <= 45 min, `-A neutrinoGPU::debug`)

    bash lartpc/polaris/qsub_cascade.sh --test         # DRYRUN=1 to only print the qsub line

Runs `test_cascade.pbs`: (0) `sha256sum -c`, `pol_check_env`, 4 GPUs, in-container
`import torch, pointops` + squashfs listing; (1) builds
`$POL_DATADIR/lists/merged_sp_bnb5e19_polaris.txt` (176,302 paths, production
order); (2) 2-event smoke into `tests/smoke/` and the attr assertions
(gamma_eff 2.86073, table, data, 2.8,5.0, stream nu); (3) 4 GPUs x 50 events
(indices 0-199) into `tests/throughput/` with s/event per GPU; (4)
`compare_to_tufts.py` against indices 0-199 of the Tufts cew6 production. Read
`$POL_DATADIR/logs/test/kp2test.o<jobid>`: every step prints PASS/FAIL; all PASS
writes `tests/throughput/.PASS`, which the production qsub requires.
Expect ~1.5-2.5 s/event. If it is much slower with idle GPUs the single-stripe
images are the bottleneck (`lfs migrate -c 8 <image>`, optional).

Then the tail on the same 200 events (debug queue, 1 node):

    TAG=throughput MODE=debug MAX_EVENTS=200 bash lartpc/polaris/qsub_tail.sh

Expect in `logs/throughput/tail/`: nu_reco/larpid/export tasks DONE and the hadd
line `merged EventTree entries: 200 (== shard sum), potTree: 0, showerCosmicScore branch: True`;
ntuple at `tests/throughput/tail/dlgen2_larformer_ntuple_bnb5e19_throughput.root`.

### 4.3 Production GPU pass (`prod` queue, `-A neutrinoGPU::DetsimGPU`)

    source lartpc/polaris/polaris_env.sh
    python3 lartpc/polaris/make_worklist.py --mode cascade \
        --list $POL_DATADIR/lists/merged_sp_bnb5e19_polaris.txt --nshards 64 --out $POL_DATADIR/lists/cascade_prod.wl
    MODE=prod NODES=16 WALLTIME=03:00:00 TAG=prod WORKLIST=$POL_DATADIR/lists/cascade_prod.wl \
        bash lartpc/polaris/qsub_cascade.sh

64 shards of 2,755 events, 4 per node, one wave on 16 nodes (~1.5 h at 2 s/event;
the `small` cap is 3 h). If the test showed > 2.3 s/event use `--nshards 96` and
`NODES=24` (1,837 events per shard). The job ends with `check_cascade.py`; if
anything is unfinished it writes `cascade_prod.wl.resume` (each line resumes a
shard from its last logged event; `--deterministic` makes the resumed range
bit-identical) and prints the relaunch line, e.g.

    MODE=debug NODES=1 TAG=prod OUTDIR=$POL_DATADIR/keypoint2_streams WORKLIST=$POL_DATADIR/lists/cascade_prod.wl.resume bash lartpc/polaris/qsub_cascade.sh

Never launch a relaunch while the first job still runs (`qstat -u $USER`): the
qsub wrapper refuses if `keypoint2_streams/.shards/*.running` exist (FORCE=1 after
you checked). Re-check any time with

    python3 lartpc/polaris/check_cascade.py --worklist $POL_DATADIR/lists/cascade_prod.wl \
        --outdir $POL_DATADIR/keypoint2_streams --logdir $POL_LOGDIR/prod --tag prod --throughput

### 4.4 Production tail (`prod` queue, 10 nodes = the queue minimum, ~1 h)

    TAG=prod MODE=prod NODES=10 WALLTIME=03:00:00 bash lartpc/polaris/qsub_tail.sh

`tail_driver.sh` runs regen (find-based nu/fm lists; refuses to regenerate once
nu_reco outputs exist -- gidx = line number must stay fixed), nu_reco (120 shards
per stream, 24 per node), LArPID (one task per nu_reco shard, 32 per node, CPU,
default run-1 weights via `PRONGCNN_DIR`), export (96 shards, data mode:
`--truth-dir` absent, `--weights-pkl none`, cew6 shower BDTs) and hadd (ROOT
from `/opt/root` inside the container). Every task has a `.done` marker under
`$POL_DATADIR/.done/`, so resubmitting the same command resumes;
`STAGES=export,hadd` runs a subset. Result:
`$POL_DATADIR/dlgen2_larformer_ntuple_bnb5e19_prod.root` (export shards are kept
next to it).

### 4.5 Platform conformance (do this BEFORE production; result 2026-09-13)

The first Polaris test job reproduced only 99/200 Tufts partitions bit for bit
(155/200 same particle count, no bias: +22/-23 events), while a Tufts A100 on
the same code reproduces 200/200. Per `docs/reference/LArFormer_Reproducibility.md`
§4 determinism follows the driver + library stack, so Polaris (different host
driver, A100-SXM4) is a separate conformance family. Data reconstructed on
Polaris against MC/EXT reconstructed at Tufts would mix families; the size of
that systematic is measured at the analysis-variable level on 2,000 events:

    # Polaris: 2,000-event cascade (4 GPUs x 500, ~15 min) + tail, debug queue
    cd /eagle/neutrinoGPU/twongj01/larformer_pointcept && git pull
    NTHR=500 bash lartpc/polaris/qsub_cascade.sh --test          # step 4 now WARNS on partition agreement
    TAG=throughput MODE=debug MAX_EVENTS=2000 bash lartpc/polaris/qsub_tail.sh
    # ship the cascade tree (~220 MB) + ntuple to Tufts
    source lartpc/polaris/polaris_env.sh; cd $POL_DATADIR/tests
    tar czf polaris_throughput_2k.tgz throughput/0* throughput/tail/dlgen2_larformer_ntuple_bnb5e19_throughput.root
    scp polaris_throughput_2k.tgz <user>@login.pax.tufts.edu:/cluster/tufts/wongjiradlab/larbys/data/larformer/polaris_it_tufts/from_polaris/

    # Tufts: the same 2,000 events (indices 0-1999) through the same scripts
    # (polaris_it_tufts/cas2k.sbatch + tail), then
    pol_exec "python3 lartpc/polaris/compare_platforms.py kp2 --a <tufts tree> --b <polaris tree>"
    pol_exec "python3 lartpc/polaris/compare_platforms.py ntuple --a <tufts.root> --b <polaris.root> --csv diff.csv"

`compare_platforms.py` prints event-level flip rates with binomial errors
(foundVertex, prong counts, PID multisets, shower energy, cosmic BDT score,
flash chi2, vertex distance). Decide with those numbers whether to (a) quote
the shift as a systematic, (b) move MC/EXT to Polaris too (one family), or
(c) keep bnb5e19 at Tufts. The Hopper family in the reproducibility study
(1.9% event drop-flips) was ruled non-conforming.

### 4.6 Verify

* `check_cascade.py` exits 0: 64 markers, every log ends `DONE`, 0 Tracebacks,
  SKIP count reported (Tufts bnb5e19 had none); nu-file count ~176k.
* `pol_exec "python3 lartpc/polaris/check_kp2_attrs.py --tree $POL_DATADIR/keypoint2_streams --sample 50"`.
* `pol_exec "python3 lartpc/polaris/compare_to_tufts.py --ref lartpc/polaris/tufts_ref_bnb5e19_idx0-499.json --tree $POL_DATADIR/keypoint2_streams"`:
  `src_file` identical for all 500 (index linkage), identical partitions for the
  large majority (A100 vs Tufts float32 differences change a minority upstream
  of the flash table); `pred_pe`/`chi2` differ by the gamma (2.861 vs 4.20).
* hadd line: EventTree entries == shard sum, potTree 0 (data), `showerCosmicScore` present.

## 5. Gotchas

* Set `HDF5_USE_FILE_LOCKING=FALSE` in every process (Lustre).
* `LARFORMER_SONATA_PRETRAIN` (from the assets `env.sh`) must be exported: the
  keypoint model's config inherits a hard-coded Tufts path for the Sonata
  pretrain and only the production config honours the env var (fixed
  2026-09-12 after the Tufts smoke of these scripts loaded it from the old
  repo path). `pol_check_env` asserts the file; a missing file kills the cascade
  at model build.
* The squashfs mounts are read-only and must be mounted at the same
  container path the list was built with (`POL_MSP_MNT`; the list stores it).
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
