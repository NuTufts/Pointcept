# Split processing workflow: Tufts (step A) -> Polaris GPUs (step B) -> Tufts (step C)

**Use this file when** a MicroBooNE sample is to be converted at Tufts (CPU),
run through the LArFormer keypoint2 cascade on Polaris A100s, and finished
(nu_reco / LArPID / ntuple) back at Tufts. It is the operator runbook; the
single-site procedure is `Sample_Processing_Procedure.md` and the original
all-on-Polaris notes are `lartpc/polaris/README.md` + `Running_Inference_Offsite.md`.
First sample: **run4ovl_4a** (mcc9.10 Run4a BNB nu overlay, 119 files,
103,384 events), run 2026-10-15 to demonstrate and measure N-node scaling for
an ALCF allocation proposal.

Claude sessions run only at Tufts. Everything on Polaris is `git pull` + a
handful of commands typed by the user (section 3); results come back as files.

```
TUFTS    A1 spec + 12-event smoke (geometry check decides TICK)
         A2 convert (sbatch sequencer, all filenos concurrent) -> merged_sp/ truth_sidecar/ markers/
         A3 finisher: retry pass -> pack merged_sp into fileno-group squashfs images, wave 1
         A4 globus_transfer.sh to-polaris (images + manifests), verify_images.sh on Eagle
            delete local images, pack wave 2, transfer, verify, delete
POLARIS  B0 git pull; pol_check_env (sample env)
         B1 qsub_stageB.sh --test   (debug, 1 node: list, 2-event smoke via the pack path, PPG/MPS matrix)
         B2 qsub_stageB.sh NODES=16 PPG=.. MPS=..  (prod, 1 wave; per-shard images on Eagle, GPU monitor)
         B3 check_cascade.py --images ; report_<tag>.md (utilization, events per node-hour)
         B4 globus_transfer.sh to-tufts (kp2_images/ + logs/)
TUFTS    C1 launch_stageC_from_polaris.sh: verify + unpack images -> keypoint2_streams/, RESUME_AFTER_INF chain
         C2 verify (STEP 5 of the procedure) + register
```

Paths: `K=/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/kpv2_pointcept`,
sample dir `D=/cluster/tufts/wongjiradlabnu/nutufts/data/larformer/mcc9.10_run4_bnbnu_overlay_4a`
(wongjiradlabnu: wongjiradlab had 432 GB free), Eagle `E=/eagle/neutrinoGPU/twongj01`.

---

## 1. Step A at Tufts (CPU)

Decisions for run4ovl_4a (survey `logs/data_prep/run4ovl4a_survey.5263874.log`,
smoke `logs/data_prep/run4ovl4a_smoke.5264533.log`): MODE=mc, period 4,
**TICK=forward** (geometry check: median centroid-minus-true x = -3.1 cm),
4.6 MB/event -> ~475 GB merged_sp, sidecar totPOT 1.11e18 per file (the
survey's POT sum is garbage for this production; `totGoodPOT` in the sidecar
is too -- use `totPOT`).

1. Spec: `lartpc/data_prep/uboone_official/tranche_run4ovl_4a.spec` (one batch, filenos 1-119).
2. Sequencer as a slurm job (sources are tier1 -> no login node needed):
   ```bash
   sbatch logs/data_prep/run4ovl4a_seq.sbatch      # THROTTLE=%119: all files at once
   ```
   It runs `run_tier2_tranche.sh` then the finisher `logs/data_prep/run4ovl4a_finish.run.sh`:
   retry pass (`retry_incomplete_tranche.sh --max-rounds 2`, now login-free for
   tier1 lists) and the **wave-1 pack** (groups 0-5). Progress:
   `ls $D/markers/*.ok | wc -l`, logs `logs/data_prep/run4ovl4a_seq.*.log`,
   per-file `$D/markers/logs/fileno*.log`. ~24 s/event -> the largest file
   (1,463 events) takes ~10 h; the whole batch finishes in about that time.
3. Packing (`lartpc/data_prep/squashfs/pack_merged_sp_squashfs.sh`, new
   `GROUP_FILENOS` mode): image k holds filenos [10k, 10k+9] as
   `run4ovl_4a_merged_sp_gKK.sqfs` (image root = the `NNN/` dirs; staged with
   hard links, so no extra disk during the pack), plus `.manifest`
   (counts, bytes, sha256) and `.ok`. lz4, ~40-50 GB each, 12 images.
   Two waves because merged_sp + images exceed the free space:
   ```bash
   # wave 1 is submitted by the finisher (array 0-5); wave 2 by hand once wave 1 is on Eagle and deleted:
   SRC=$D/merged_sp OUT=$D/squashfs SAMPLE=run4ovl_4a GROUP_FILENOS=10 sbatch --array=6-11 \
       lartpc/data_prep/squashfs/pack_merged_sp_squashfs.sh
   bash lartpc/data_prep/squashfs/verify_images.sh $D/squashfs --quick
   ```
4. Transfer (Globus CLI; once: `pip install --user globus-cli; globus login`;
   copy `lartpc/data_prep/squashfs/globus_endpoints.env.example` to
   `globus_endpoints.env` with the Tufts and `alcf#dtn_eagle` UUIDs from
   `globus endpoint search ...`):
   ```bash
   bash lartpc/data_prep/squashfs/globus_transfer.sh to-polaris $D/squashfs $E/data/uboone/run4ovl_4a/squashfs --wait
   # on Polaris: bash lartpc/data_prep/squashfs/verify_images.sh /eagle/neutrinoGPU/twongj01/data/uboone/run4ovl_4a/squashfs
   rm $D/squashfs/run4ovl_4a_merged_sp_g0[0-5].sqfs      # only after the far-side verify; manifests/.ok stay
   ```
   Set wide striping on the Eagle destination before the first transfer
   (`lfs setstripe -c 8 <dir>`, squashfs README). The transfer is
   `--sync-level checksum --verify-checksum`, so re-running it resumes.
   Keep the dlmerged copies at Tufts until this workflow has been exercised
   end to end (user decision 2026-10-15); long term they live on Polaris.

## 2. What is on Polaris (sample-specific values live in ONE file)

`lartpc/polaris/samples/run4ovl_4a.env` (sourced by `polaris_env.sh` when
`POL_SAMPLE_ENV` names it): images `$E/data/uboone/run4ovl_4a/squashfs`
(12), outputs `$E/data/uboone/run4ovl_4a_stageB/` (`lists/`, `logs/<tag>/`,
`tests/`, `kp2_images/`), list name, expected event count, and the GPU-pass
flags: `--gamma-run-scale 0.8325` ((mc,3) proxy, no (mc,4) cell; approved
2026-10-15) `--sample-kind mc --flash-window 3.6,5.2`, GT emitted (no
`--no-gt`, like the Tufts MC chain), deterministic, score maps, output tree.
Expected kp2 attrs: gamma_eff 4.370625, spec 0.8325, kind mc, window 3.6,5.2.
No CPU-tail assets are checked (`POL_NEED_TAIL=0`).

Stage-B mechanics (`lartpc/polaris/`):

| file | role |
|---|---|
| `qsub_stageB.sh` | qsub builder: `--test` (debug, 1 node) or prod (`NODES`, `PPG` processes per GPU, `MPS` 0/1); builds a worklist of NODES x 4 x PPG equal shards (one uniform wave) |
| `test_stageB.pbs` | assets + image sha256, list build, 2-event smoke through the pack path + attr assertions, then the **PPG/MPS matrix** (`MATRIX="1:0 2:1 3:1"`, `NTHR` events per process) -> one line per config; writes `tests/stageB/.PASS` |
| `stageB.pbs` | the production job: per node MPS daemon (optional) + `gpu_monitor.sh`; `node_launcher.sh` runs 4 x PPG shards per node |
| `run_cascade_shard.sh` | one shard: GPU = SLOT % 4; writes its tree to `/local/scratch/$USER/kp2_<tag>_<job>/shardNNN`, packs it into one file per shard, `kp2_images/kp2_<tag>_shardNNN.{tar,sqfs}` + `.manifest` (format below), copies to Eagle, verifies sha256, then `.done` |
| `gpu_monitor.sh` | `nvidia-smi` sampler per node -> `logs/<tag>/gpumon_<host>.csv` (readable live from a login node: `tail -f`) |
| `summarize_gpu_util.py` | report: per-node/GPU mean utilization in the active window, per-shard s/event, aggregate events/s, events per GPU-hour and **per node-hour**, node-hours charged; 1-min utilization time series CSV |
| `check_cascade.py --images` | shard done = marker + DONE + image + manifest; unfinished shards are resumed whole (their per-event files were node-local) |
| `mps_enable.sh` / `mps_disable.sh` | ALCF recipe (`nvidia-cuda-mps-control -d`, pipe dir `/tmp/nvidia-mps`, `start_server -uid`) |
| `build_squashfs_tools.sh` | builds `mksquashfs`/`unsquashfs` 4.6.1 *inside the container* into `$POL_ASSETS/bin` (gzip, xz); no container rebuild, no site package needed |

Shard image format (`POL_PACK_FMT`, passed through `qsub_stageB.sh`):
`tar` (default) is a plain uncompressed tar -- Polaris has no site
`mksquashfs` (checked 2026-10-15), and for shipping back to Tufts a tar is as
good as a squashfs image (one file per shard, sha256 in the manifest,
`unpack_kp2_images.sh` extracts both). `sqfs` is worth it only if the images
are to be *mounted* later (e.g. a future stage C on Krux/Polaris reading
`kp2_images` through `mount_args.sh` instead of unpacking): then run
`bash lartpc/polaris/build_squashfs_tools.sh` once on a login node (mksquashfs
is pure user space -- no kernel module, no privilege -- so the container-built
binary works on login, GPU and Krux nodes) and submit with
`POL_PACK_FMT=sqfs`; the shard script runs it through `pol_exec`
(`POL_SQFS_VIA=container`), gzip level 1 unless the build found lz4 headers.
Both routes passed the Tufts integration test (jobs 5267836 tar, 5267947 sqfs).

Why node-local + per-shard images: Lustre is slow at per-file metadata
(bnb5e19 export went from 48 to 0.8 s/event after cutting file opens), so no
per-event file is ever created on Eagle; a shard is ~800 events ~ 160 MB,
one sequential write. The images are what Globus ships back (128 files
instead of 135k). A walltime kill loses at most the running shards.

Why MPS: one cascade process leaves the GPU idle during its CPU work
(collate, flash matching, h5 writes). Several processes per GPU fill the
gaps; without MPS their kernels time-slice, with MPS they overlap. The test
job measures PPG=1 (no MPS) vs 2 and 3 (MPS) and prints events/s and GPU
utilization for each; pick the best for production. Memory: ~10 GB GPU and
~20 GB host per process; PPG=3 = 12 processes per node, within 40 GB/512 GB.

## 3. Polaris commands (user, login node)

```bash
cd /eagle/neutrinoGPU/twongj01/larformer_pointcept && git pull
export POL_SAMPLE_ENV=lartpc/polaris/samples/run4ovl_4a.env
source lartpc/polaris/polaris_env.sh && pol_check_env          # 12 images, assets, apptainer -> PASS
sbank-list-allocations -r polaris -p neutrinoGPU -f "+subname users_list"   # DetsimGPU / debug still valid?
lfs setstripe -c 8 $POL_DATADIR/kp2_images 2>/dev/null; mkdir -p $POL_DATADIR/kp2_images

# B1 test (debug queue, 1 node, <= 1 h; log $POL_DATADIR/logs/testB/<jobid>...OU)
bash lartpc/polaris/qsub_stageB.sh --test                        # NTHR=40 MATRIX="1:0 2:1 3:1" by default
#   read the MATRIX SUMMARY lines (tag=testB_p<PPG>m<MPS> ... agg=<ev/s> ... gpu_util_mean=..); .PASS written

# B2 production (prod -> small queue, 16 nodes, 3 h cap; charges DetsimGPU)
MODE=prod NODES=16 PPG=2 MPS=1 TAG=prod bash lartpc/polaris/qsub_stageB.sh   # PPG/MPS = best matrix row
qstat -u $USER;  tail -f $POL_DATADIR/logs/prod/gpumon_*.csv    # live utilization while it runs

# B3 verify + report (the job does both at its end; re-run any time)
python3 lartpc/polaris/check_cascade.py --worklist $POL_DATADIR/lists/stageB_prod_n16_p2.wl \
    --outdir $POL_DATADIR/kp2_images --images $POL_DATADIR/kp2_images --logdir $POL_LOGDIR/prod --tag prod --throughput
python3 lartpc/polaris/summarize_gpu_util.py --logdir $POL_LOGDIR/prod --tag prod --nodes 16 --ppg 2 --mps 1 \
    --elapsed <launcher elapsed s from the job log> --out $POL_LOGDIR/prod/report_prod.md --csv $POL_LOGDIR/prod/util_prod.csv
#   unfinished shards: check_cascade prints the qsub_stageB.sh relaunch line (MODE=debug, WORKLIST=<wl>.resume)
bash lartpc/data_prep/squashfs/verify_images.sh $POL_DATADIR/kp2_images --quick

# B4 ship back (from Tufts, or from Polaris if the CLI is installed there)
bash lartpc/data_prep/squashfs/globus_transfer.sh to-tufts $E/data/uboone/run4ovl_4a_stageB/kp2_images $D/kp2_images --wait
bash lartpc/data_prep/squashfs/globus_transfer.sh to-tufts $E/data/uboone/run4ovl_4a_stageB/logs      $D/polaris_logs --wait
```

Budget: 103,384 events x ~2 s / 4 GPUs = ~14 node-hours of compute; one
uniform wave on 16 nodes is ~35-40 min wall (808 events per shard at PPG=1,
plus ~5 min model build and the pack/copy) = ~10-12 node-hours charged,
plus ~1 for the test job and a reserve for a resume job: ~15, <= 25.

## 4. Step C at Tufts (CPU tail)

```bash
bash lartpc/data_prep/squashfs/verify_images.sh $D/kp2_images            # sha256 vs manifests
SPEC=lartpc/data_prep/uboone_official/tranche_run4ovl_4a.spec TAG=run4ovl_4a IMGDIR=$D/kp2_images \
  GAMMA_SPEC=0.8325 FLASH_WINDOW=3.6,5.2 WEIGHTS_PKL=none \
  sbatch --export=ALL lartpc/larformer_reco/slurm/launch_stageC_from_polaris.sh
```
The launcher asserts the kp2 attrs against GAMMA_SPEC/FLASH_WINDOW/kind,
builds the merged_sp list (`list_merged_sp.py`; the tail resolves kp2
`src_file` basenames against it, so the Polaris path prefix never matters),
submits `unpack_kp2_images.sh` (verify -> `tar -xf` / `unsquashfs -f` every image into
`$D/keypoint2_streams`) and calls `launch_tranche_chain.sh` with
`RESUME_AFTER_INF=<unpack job>`: `submit_extbnb_chain.sh` skips prep +
inference and hangs regen on the unpack. `WEIGHTS_PKL=none` because no run-4
`bnb_nu` xsec pickle exists (xsecWeight = -1); LArPID default weights (not
run 3). Then STEP 5 / STEP 6 of `Sample_Processing_Procedure.md`
(verify reco-vs-truth, register in the versions doc with the proxy gamma).

Conformance note: Polaris A100-SXM4 + driver 580 is a separate determinism
family from Tufts (`Polaris_Conformance_2026-09-16.md`: ~4% event-level
flips vs Tufts, unbiased). A sample processed this way must be compared only
with samples whose GPU pass also ran on Polaris, or the difference quoted.

## 5. Tufts-side integration test of the Polaris scripts

`lartpc/polaris/tufts_it_stageB.sbatch` (GPU job at Tufts) packs the smoke
merged_sp into a g00 image with the new packer, mounts it with the new
`mount_args.sh`, builds the list, runs 2 shards x 2 events through
`node_launcher.sh` (2 processes on one GPU, GPU monitor, MPS if the node has
it) with node-local output + per-shard packing, unpacks the images with
`unpack_kp2_images.sh` logic, asserts the kp2 attrs and prints the
utilization report. Run it after any change to `lartpc/polaris/`.
