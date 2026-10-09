# Processing a new MicroBooNE sample: dlmerged ROOT → LArFormer analysis ntuple

**Use this file when** you start a fresh session to process a new official
MicroBooNE sample (MC overlay, pure MC, particle gun, beam-on data, or EXT) from
`merged_dlreco*.root` files into a `dlgen2_larformer_ntuple_<TAG>.root`. It is
the operational procedure. The underlying details live in:

| topic | doc |
|---|---|
| Step A converter, merged_sp schema | `lartpc/data_prep/uboone_official/LARFORMER_DATAPREP.md` |
| reco chain stages, streams, flash products | `lartpc/larformer_reco/README.md` |
| frozen chain version, all finished ntuples + normalizations | `lartpc/larformer_analysis/model_and_output_file_versions.md` |
| POT / spill normalization per sample | `lartpc/data_prep/README.md` § Normalization |
| flash light-yield (gamma) calibration | `lartpc/larformer_analysis/flashmodel_calib/PROTOCOL.md`, `lartpc/flashmatch/flash_calib.py` |
| ntuple schema | `lartpc/larformer_reco/larformer_reco_output_data_schema.md` |

**Do not write new pipeline scripts.** Every step below already has a tool, and
most new samples need only a **spec file** plus **one launcher call**. If a
sample needs new behaviour, generalize the existing script (env knob or
auto-detection) as was done for `stage_overlay_train_batch.sh`.

Paths: `K=/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/kpv2_pointcept`
(run everything from here), `L=/cluster/tufts/wongjiradlab/larbys/data/larformer`
(output root; one directory per sample), container
`/cluster/tufts/wongjiradlabnu/larbys/containers/pointcept_cuml.sif`, ubdl env
`/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/ubdl/setenv_pointcept_container.sh`.

---

## Pipeline at a glance

```
inputlists/<sample>.txt  (one dlmerged path per line; line number = fileno)
  │
  ├─ STEP 1  survey_dlmerged_sample.py         -> decisions (period, kind, window, disk)
  ├─ STEP 2  smoke: 5 events convert + sidecar  -> MB/event, s/event, truth sanity
  ├─ STEP 3  tranche_<x>.spec
  │          tmux_tier2_tranche.sh  (login node)
  │            └ run_tier2_tranche.sh -> stage_overlay_train_batch.sh (stage/symlink)
  │                 └ sbatch array: submit_overlay_train_convert.sh, per fileno:
  │                     convert_dlmerged_to_larformer_h5.py -> merged_sp/<fileno/100>/*.h5
  │                     [MC] extract_truth_sidecar.py       -> truth_sidecar/truth_fileno*.h5
  │                     [MC] complete_labels.py (label completion)
  │                     marker markers/fileno*.ok  (only if the entry loop finished)
  │          finisher (tmux after-command): retry_incomplete_tranche.sh -> sbatch launcher
  ├─ STEP 4  launch_tranche_chain.sh (guards) -> submit_extbnb_chain.sh:
  │            prep -> inference (GPU) -> regen -> nu_reco nu/fm -> larpid nu/fm
  │            -> export (60 shards) -> hadd -> $L/<sample>/dlgen2_larformer_ntuple_<TAG>.root
  ├─ STEP 5  verify
  └─ STEP 6  register (versions doc, data_prep README normalization, sample table below)
```

Wall time: conversion ~24 s/event over 60 concurrent tasks (about 4-12 h for
50-200k events); the chain takes about a day (GPU waves of ~3.5 h, then CPU stages).

---

## STEP 1: survey the sample and make the per-sample decisions

```bash
sbatch --partition=batch,preempt --mem=4G --time=1:00:00 --output=logs/data_prep/<x>_survey.%j.log \
  --wrap="module load apptainer; apptainer exec --bind /cluster:/cluster <container> bash -c \
  'cd <ubdl> && source setenv_pointcept_container.sh >/dev/null 2>&1 && cd $K && \
   python3 lartpc/data_prep/uboone_official/survey_dlmerged_sample.py <inputlist>'"
```

It prints the file count and size, whether sources are on tier2, event counts,
run range → run period, MC vs data, pure-MC (CORSIKA) vs overlay, POT, generator
PDG/time/energy, and the in-time flash-time histogram. Use it to fill this table
(write the answers into the spec header and the launcher call):

| decision | rule |
|---|---|
| **MODE** (`mc`/`data`) | `mc` if `mctruth_generator_tree` exists (overlay, pure MC, gun). `data` for beam-on and EXT. Sets `--mcc9` vs `--is-data`, truth sidecars, label completion. |
| **TAG** | short, no dots or spaces (it goes into every file name and job name), e.g. `isogamma_run45`, `nue_run1_half`. Used as the converter `SAMPLE` *and* the chain `TAG`. |
| **DATADIR** | `$L/<sample name>/`, a fresh directory. Never reuse another production's DATADIR: chain prep `rm -rf`s its downstream dirs. |
| **Staging** | tier2 sources (`/cluster/tier2/...`) are login-only and get **copied** to `.../mcc9_scratch/tier2_staging/<TAG>` batch by batch. Other paths (e.g. tier1 `.../larbys/data/globus/`) get **symlinked** automatically. |
| **GAMMA_SPEC** | the flash light-yield scale, baked into the GPU pass. `table` if `flash_calib.GAMMA_SCALE_TABLE` has a `(kind, period)` cell: currently (data,1) (data,3) (mc,1) (mc,3). For any other cell, `table` raises a KeyError. Then either calibrate the cell first (PROTOCOL.md: needs in-time through-going muons, so overlays/data only) or pass an explicit float and **record the proxy choice** in the launcher and the versions doc. MC cells are nearly period-independent ((mc,1)/(mc,3) = 1.03), so the nearest MC cell is the standard proxy for MC. Never use a data cell for MC or vice versa. |
| **FLASH_WINDOW** | read it off the survey's flash-time histogram and compare with `flash_calib.FLASH_WINDOW_US`: MC 3.6,5.2; run-1 beam-on 2.8,5.0; EXT 3.2,5.4 (EXT **must** pass it explicitly). The launcher requires it. |
| **TICK** (spec column 10) | larcv image tick order. `backward` (default, converter `-tb`) for the mcc9 v28/v29e productions. **`forward` for mcc9.10** (verified on isogamma_run45). A wrong choice mirrors the drift coordinate (`x_reco ~ -x_true - 80 cm`), and the whole chain still runs "successfully" on garbage. Never assume: STEP 2's geometry check decides. |
| **SAMPLE_KIND** | derived from MODE by the launcher (`mc`/`data`). |
| **WEIGHTS_PKL** | GENIE CV xsec-weight pickle in `pointcept_env/gen2ntuple/event_weighting/` matching the sample (run period and nu flavour), e.g. `weights_forCV_v48_Sep24_{bnb_nu,intrinsic_nue}_run{1,3}.pkl`. Use `none` for data and for non-GENIE samples (particle guns); every event then gets `xsecWeight=-1`. |
| **LArPID weights** | `larpid/model.py:select_checkpoint`: the substring `run3` in `LARPID_SAMPLE_TAG` → alternate weights, else default. The run-period rule: only run-3 samples get `run3` in the tag. The launcher defaults `LARPID_SAMPLE_TAG=$TAG`, so do **not** put `run3` in TAG unless it is a run-3 sample. |
| **NINF** | GPU inference shards: ~1 per 5k events (12 for 55k, 16 for 80k, 32 for 180k). |
| **Disk** | merged_sp is ~5-6 MB/event (measure in the smoke). Downstream adds <10% (keypoint2 ~0.2 MB/event, nu_reco + larpid ~1 GB per 80k events, ntuple ~2 kB/event). `df -h /cluster/tufts/wongjiradlab`: the stager refuses to start a batch below 500 GB free, and the volume runs at 97-98% full. If it does not fit, stop and ask the user (options: squashfs-pack old merged_sp trees, or put DATADIR on `wongjiradlabnu`). |
| **Subsets** | an analysis half or a fraction needs a subset list (see `training_data_ledger/` for md5-parity TRAINPOOL/RESERVED splits). Never mix training-pool files into an analysis sample. |

## STEP 2: smoke test (5 events)

Before launching hundreds of tasks, convert 5 events of file 1, extract the
sidecar and run label completion in one short sbatch job. Write into a
`smoke/` dir **outside** `merged_sp/` and delete it afterwards. Commands are the
same as in `submit_overlay_train_convert.sh`:

```bash
python3 lartpc/data_prep/uboone_official/convert_dlmerged_to_larformer_h5.py -i <file1> -o <smoke>/merged_sp \
   --tag <TAG> --fileno-tag fileno00001 --adc wire -tb <--mcc9|--is-data> -v 1 -n 5
python3 lartpc/larformer_reco/export/extract_truth_sidecar.py --input-dlmerged <file1> --out <smoke>/truth_fileno00001.h5 --max-events 5   # MC
PYTHONPATH=$K python3 lartpc/data_prep/uboone_official/complete_labels.py --h5 <smoke>/merged_sp/*.h5                                     # MC
python3 lartpc/data_prep/uboone_official/check_merged_sp_geometry.py --truth <smoke>/truth_fileno00001.h5 <smoke>/merged_sp/*.h5      # MC: MUST print ok
```
(Drop `-tb` for TICK=forward. Use `-n 12` or more so that several events have truth-matched points.)

Check: **the geometry check prints `ok`** (median centroid-minus-true x of a few cm; a ~100 cm or larger offset or negative centroids = wrong TICK). For data (no truth), the all-points x range must sit roughly within [-40, 290] cm. Then: `Done. N entries processed`; MB/event (for the disk budget); s/event
(the largest file must finish inside the 24 h task limit); the truth sidecar
attrs (`trueNuPDG`, `trueVtx*`, `truePrimPart`) make sense for the generator.
Non-neutrino generators go through the same code path: a particle gun is
wrapped as an MCNeutrino whose `Nu()` is the gun particle, so `trueNuPDG` = gun
PDG, `trueNuE` = gun energy and `trueVtx*` = gun start.

## STEP 3: conversion (tranche spec + tmux sequencer + finisher)

1. **Spec** `lartpc/data_prep/uboone_official/tranche_<x>.spec`: one line per
   batch, `SAMPLE LIST OUT_ROOT MODE TRUTH_DIR STAGE MARK FIRST LAST`
   (`TRUTH_DIR=-` for data). FIRST/LAST are 1-based list line numbers = filenos.
   One batch is fine up to ~2000 files on tier1. For tier2, use 250-file batches
   (they are run serially to bound the scratch copy). Put the sample facts in the
   `#` header: events, period, MB/event. Copy `tranche_isogamma_run45.spec` (tier1, one
   batch) or `tranche_ovl_run1_half2.spec` (tier2, 20 batches).
2. **Finisher** `logs/data_prep/<x>_finish.run.sh`: waits for the conversion
   arrays to drain, runs `retry_incomplete_tranche.sh <spec> --max-rounds 2`,
   then sbatches the launcher with the STEP-1 decisions. Copy
   `logs/data_prep/isog_finish.run.sh`.
3. **Start on a login node** (tier2 is visible only there; login cleanup kills
   nohup/setsid processes, but tmux survives):
   ```bash
   bash lartpc/data_prep/uboone_official/tmux_tier2_tranche.sh <spec> <name> "bash logs/data_prep/<x>_finish.run.sh"
   ```
   Log: `logs/data_prep/<name>.log`. Progress: `ls $DATADIR/markers/*.ok | wc -l`.
   Per-file converter logs: `$DATADIR/markers/logs/fileno*.log`. The sequencer is
   idempotent: re-running it resumes from markers.

Expected failures (handled, do not chase them): ~1% of files segfault or hang in
the converter (rc=129, or a stuck ROOT teardown: `scancel` that task). Those leave
**partial** h5s with no marker. The retry tool deletes the partial outputs and
retries them. Deterministic crashers are dropped whole into `<spec>.dropped`.
Never let a partial file into a sample: its sidecar carries the full-file POT.
Conversion tasks run on `batch,preempt,wongjiradlab` and are **not requeued**,
so preempted tasks show up as `missing` filenos (21 of 248 for isogamma_run45);
the retry pass recovers them. The retry relaxes the stager's free-disk guard to
`RETRY_MIN_FREE_GB` (default 100 GB; full batches need 500 GB, `MIN_FREE_GB`).
If a retry batch cannot be submitted at all, the tool **stops with exit 2 and
drops nothing**, and the finisher does not launch the chain. (Before 2026-10-09 it
silently dropped such filenos as if they were crashers, and the launcher accepted
the loss.) Only filenos that still fail after real retries go to `<spec>.dropped`.
To re-attempt a dropped fileno, remove its line from `.dropped` and re-run the tool.
Files with no MC truth trees (reco-only upstream files) fail every event and
cannot be recovered (the nue run-3 sample lost 33% this way).

## STEP 4: reco chain (generic launcher)

`lartpc/larformer_reco/slurm/launch_tranche_chain.sh` is spec-driven. It reads
DATADIR, MODE and TRUTH_DIR from the spec, and refuses (exit 2) if markers <
files − (dropped + NDUD), sidecars lag markers, conversion jobs are still queued,
partial filenos exist, or `$DATADIR/.chain_submitted` exists. It then builds the
stable `(fileno, entry)`-sorted merged_sp list and calls `submit_extbnb_chain.sh`.
That script is the general chain despite its name, and it pins the frozen chain
version (configs, checkpoints, BDTs, LLR).

```bash
SPEC=lartpc/data_prep/uboone_official/tranche_<x>.spec TAG=<tag> \
  GAMMA_SPEC=<table|float> FLASH_WINDOW=<lo,hi> WEIGHTS_PKL=<pkl|none> NINF=<n> \
  sbatch --export=ALL lartpc/larformer_reco/slurm/launch_tranche_chain.sh
```

The finisher normally makes this call. Launcher log:
`logs/export/tranche_chain_launch.<jobid>.log` (it prints every stage's job id).
Outputs in DATADIR: `keypoint2_streams/`, `nu_reco_streams_{nu,fm}/`,
`nu_reco_larpid_{nu,fm}/`, `dlgen2_larformer_ntuple_<TAG>.root`. The lists go
to `lartpc/larformer_reco/{inputlists,outputlists}/*_<TAG>*.txt`. **Never
regenerate** the kp2 lists after the fact: nu_reco and larpid shards are keyed
by line number.

Other knobs (env, pass through): **`EXCLUDE_NODES=pax007`** (its GPUs intermittently fail at model load with "CUDA-capable device(s) is/are busy or unavailable": 3 shards in each of the two isogamma launches) and `pax051` (flaky on huge events). Use it by default,
`NNR`/`NEXP` (CPU shards, default 60), `NU_RECO_EXTRA_ARGS`, `INF_EXTRA_ARGS`.
Partition rules: GPU arrays only `gpu,preempt`, never `wongjiradlab`; CPU arrays
`batch,wongjiradlab`, ≥60 shards, ≥24 h limits. Do not hunt for limits empirically.

**Recovering failed inference shards.** A failed shard leaves the downstream
jobs in `DependencyNeverSatisfied`. Typical causes: a bad GPU node
(`CUDA-capable device(s) is/are busy or unavailable` at model load, as with
pax007 on 2026-10-07), or one huge event crashing a node (then add
`--skip-events <i>` to that shard, run the event on another node and copy its
output into the tree). Procedure: `scancel` the stalled tail jobs, rerun only the
failed array indices with the chain's inference env, then resubmit the CPU tail
in resume mode. Example (isogamma_run45, shards 9-11):
```bash
export CONFIG=configs/lartpc/larformer/stage4_keypoint/larformer-keypoint2-fullcascade-v6lantern-envslicer.py
export LARFORMER_BATTERY_SLICER_CKPT=exp/larformer_slicer_s1_mixenriched_v1/model/epoch_2.pth
export LARFORMER_KP_PARTICLE_CKPT=exp/larformer_particle_s1cache_m2frecipe_rebal_v2_warm2_cew/model/epoch_6.pth
INF=$(INPUT_LIST=lartpc/larformer_reco/inputlists/merged_sp_<TAG>.txt OUTPUT_DIR=$DATADIR/keypoint2_streams/ NSHARDS=<NINF> \
  EXTRA_INF_ARGS="--output-tree --gamma-run-scale <g> --sample-kind <kind> --flash-window <lo,hi>" \
  sbatch --parsable --exclude=<badnode> --export=ALL --array=9-11 --time=24:00:00 lartpc/larformer_reco/slurm/submit_inference_shard.sh)
TAG=<TAG> DATADIR=$DATADIR TRUTH_DIR=$DATADIR/truth_sidecar LARPID_SAMPLE_TAG=<TAG> WEIGHTS_PKL=<pkl|none> \
  GAMMA_SPEC=<g> SAMPLE_KIND=<kind> FLASH_WINDOW=<lo,hi> NINF=<NINF> EXCLUDE_NODES=<badnode> RESUME_AFTER_INF=$INF \
  bash lartpc/larformer_reco/slurm/submit_extbnb_chain.sh
```
`NSHARDS` must equal the original NINF (shards are contiguous slices of the
list). Resume mode keeps the merged_sp list and keypoint2_streams. Before any
*re-inference* of shards that already wrote output, clean the downstream dirs:
stale shards poison the glob consumers.

## STEP 5: verify (all must hold before registering)

```bash
grep -l Traceback logs/export/<TAG>_* logs/export/ntupshard.<expjob>_* 2>/dev/null   # expect none
sacct -j <inf>,<regen>,<nr_nu>,<nr_fm>,<lp_nu>,<lp_fm>,<exp>,<hadd> -X -n --format=JobID,State | sort -k2 | uniq -c -f1
wc -l lartpc/larformer_reco/outputlists/keypoint2_out_<TAG>_{nu,fm}.txt
```
and in the container (uproot is available):
```python
import uproot, numpy as np
f = uproot.open("<DATADIR>/dlgen2_larformer_ntuple_<TAG>.root")
ev = f["EventTree"]; print(ev.num_entries)              # == merged_sp list length (minus known unreadables)
print(np.unique(ev["flashGammaEff"].array()))           # == 5.25 * gamma scale actually used
print(ev["foundVertex"].array().mean())
pot = f["potTree"]; print(pot.num_entries, pot["totPOT"].array().sum())   # MC: rows == converted files
```
MC: **reco must follow truth**: `vtxDistToTrue` for foundVertex events should have a median of a few cm (nu samples: within 5 cm for ~87%), and efficiency must depend on truth (e.g. photon converts in the TPC vs not). Flat efficiency vs truth, or vertices hundreds of cm away, means the inputs are wrong (this is how the TICK error was found). Also check that truth is filled (`trueNuE > 0` for all events, or the gun's E), and
`xsecWeight` is as expected (−1 for `none`; a handful of 0/−1 is normal with a
pickle). Compare foundVertex and the stream split (nu vs fm counts) with a
sibling production in the versions doc.

## STEP 6: register

1. `lartpc/larformer_analysis/model_and_output_file_versions.md`: add a row
   (ntuple path, events, normalization, gamma_eff + window, LArPID weights,
   pickle) and a short paragraph (composition, dropped files, launcher, caveats
   such as a proxy gamma).
2. `lartpc/data_prep/README.md` § Normalization: POT (MC) or spills (EXT) of
   what was actually converted (subset rule: file fraction × full-sample spills).
3. The **sample table below**.
4. Commit the spec, `.dropped` lists, finisher and any script generalization
   (never commit data or `logs/`).

---

## Gotchas collected from past productions

- **Tick order (mcc9.10)**: the first isogamma_run45 production (2026-10-07) used the
  default `-tb`. Every job completed cleanly and the ntuple looked complete, but all
  spacepoints were mirrored in x, giving foundVertex 20% flat in photon energy and
  vertices a median 4 m from truth. It was discarded and reconverted with TICK=forward.
  A clean chain is not a correct chain: always run the STEP 2 geometry check and the
  STEP 5 reco-vs-truth check.

- **Gamma/window are baked into the GPU pass.** A wrong choice means a full
  re-inference. Legacy `GAMMA_SPEC=auto` exists only for reproducing old files.
- **merged_sp detection**: data-mode files carry an *empty* `mc_particle_tree`.
  `flash_calib.detect_kind` counts trackids, so do not test for group presence.
- **The (fileno, entry) list order is the linkage** between cascade index,
  nu_reco group and ntuple row. Always build it with `list_merged_sp.py` (the
  launcher does).
- **potTree rows come from the shard holding a fileno's first event.** POT for
  partially converted files is wrong. That is why markers and the partial guard exist.
- **Spill counts for EXT subsets** = file fraction × full spills (user decision 2026-09-12).
- **MC truth missing upstream**: `MCParticleGraph::buildgraph` throws (it was
  fixed from a segfault in ubdl/ublarcvapp). Such files produce no events.
- **LArMatch score name differs** (MC `lm_score` vs EXT `larmatch_score`). The
  cascade disables the lm pre-filter, so this only matters for training data.
- **Check the conversion count before trusting a launch**: the launcher's marker guard
  allows `dropped + NDUD` missing files. Read `<spec>.dropped` and make sure every entry
  is a real crasher (failed after retries), not a never-retried fileno.
- **Heavy work goes to batch** (sbatch). Only the tmux sequencer and the
  tier2 copies run on a login node.

## Sample table (productions made with this procedure)

`$L=/cluster/tufts/wongjiradlab/larbys/data/larformer`. Full details and
normalizations: versions doc §3c.

| sample | kind / period | spec | launcher | TAG | gamma / window | status |
|---|---|---|---|---|---|---|
| run-1 EXT half / half2 | data EXT / 1 | `tranche_ext_run1_{A,B,C}.spec` | `launch_run1ext_half{,2}_chain.sh` | `extbnb_run1_half{,2}` | table (data,1) / 3.2,5.4 | done |
| run-1 BNB nu overlay half / half2 | mc overlay / 1 | `tranche_ovl_run1_half{,2}.spec` | `launch_run1ovl_half{,2}_chain.sh` | `bnbovl_run1_half{,2}` | table (mc,1) / 3.6,5.2 | done |
| run-1 intrinsic nue half | mc overlay / 1 | `tranche_nue_run1_half.spec` | `launch_run1nue_half_chain.sh` | `nue_run1_half` | table (mc,1) / 3.6,5.2 | done |
| run-1 intrinsic nue TRAIN half | mc overlay / 1 | sidecar-only pass (`run_sidecar_only_tranche.sh`) | `launch_run1nue_trainhalf_chain.sh` | `nue_run1_trainhalf` | table (mc,1) / 3.6,5.2 | done (BDT training only) |
| bnb5e19 | data beam-on / 1 | (pre-existing merged_sp) | `launch_bnb5e19_table_chain.sh` | `bnb5e19_run1_table` | table (data,1) / 2.8,5.0 | done |
| mcc9.10 isotropic 1γ gun, runs 4/5 | pure MC (CORSIKA) / 4 | `tranche_isogamma_run45.spec` (tier1, symlink staging, **TICK=forward**) | `launch_tranche_chain.sh` via `logs/data_prep/isog_finish.run.sh` | `isogamma_run45` | **0.8325 = (mc,3) proxy** (no (mc,4) cell; no in-time muons to calibrate) / 3.6,5.2 (verified) ; WEIGHTS_PKL none ; LArPID default | DONE 2026-10-09 (247/248 files, 53,938 ev; registered in versions doc §3c) |

Per-sample notes: **isogamma_run45** has 248 files and 54,670 events (3-829
per file). The first production (247 files, fileno 79 dropped) was discarded for the tick-order
error. The reconversion (TICK=forward, truth sidecars reused) passed the geometry check and the
reco-vs-truth check (foundVertex 70%, rising with photon energy; reco vertex within 5 cm of the
true conversion point in 50% of found-vertex events). **Gun-sample caveat:** `trueVtx*` and
hence `vtxDistToTrue` refer to the photon ORIGIN; the conversion point is the photon's
`trueSimPart` EDep in the truth sidecar (`physics/single_photon/verify_isogamma_ntuple.py`).
The user approved the (mc,3) proxy gamma. All runs are 18961-19752, i.e. period 4, despite the "run45" name.
One γ per event, 0-9.8 GeV (median 0.39), gun t0 3.13-4.72 µs, isotropic, and
some start outside the TPC. Truth: `trueNuPDG=22`, `trueNuE`=Eγ, `trueVtx`=gun
start. `trueSimPart` EDep=inf when the photon never converts in the TPC.
potTree POT is meaningless for a gun. In the smoke event, `triplet_data/pid`
held only 0 and 22, so the CORSIKA cosmics do not appear to be labelled. Check
this before using the sample for training labels.
