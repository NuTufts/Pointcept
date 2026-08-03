# LArFormer Reco — nu-interaction reconstruction

Post-processes the keypoint2-cascade inference output (per-event H5) into neutrino
interaction candidates: nu-vertex candidates → track building (sliding-PCA +
MCS-RDP fit/stitch) → shower attachment → interaction tree → per-particle
4-momenta (range-based for tracks, calorimetric for showers).

Run everything from the **repo root** inside the pointcept container with
`PYTHONPATH=./` (the entry scripts also self-insert the repo root, so direct
`python3 path/to/script.py` works too).

## Pipeline

```
tools/larformer/run_larformer_keypoint2_cascade_inference.py     (GPU; slurm/submit_inference_shard.sh)
  → keypoint2_event*.h5
scripts/run_nu_reco.py                                 (CPU; slurm/submit_nu_reco_shard.sh)
  → nu_reco_shard*.h5
eval/eval_reco_performance.py                          (slurm/submit_eval_reco_{shard,merge}.sh)
  → eval_shard*.npz → merged records + plots
```

**Output file schema** (every dataset/attr, units, sentinels, class-id codes,
stream + linkage conventions):
[larformer_reco_output_data_schema.md](larformer_reco_output_data_schema.md).

## Running the full chain (merged_sp → ntuple)

The whole downstream campaign is orchestrated by one dependency-chained SLURM
submitter, **`slurm/submit_extbnb_chain.sh`** (the name is historical — it is the
*general* chain and handles both data and MC via the `TRUTH_DIR` knob). Stages:

```
merged_sp/  (built upstream by data_prep Step A — see
             ../data_prep/uboone_official/LARFORMER_DATAPREP.md)
  [MC only] truth_sidecar/   slurm/submit_truth_sidecar_shard.sh
  └─ submit_extbnb_chain.sh:
       0 prep       build merged_sp list (list_merged_sp.py) + clean downstream dirs
       1 inference  GPU array: merged_sp → keypoint2_streams/ (nu + fm kp2 h5)
       2 regen      split keypoint2_streams → per-stream nu / fm lists
       3 nu_reco    CPU arrays, per stream → nu_reco_streams_{nu,fm}/ (run_nu_reco.py)
       4 larpid     CPU arrays, per stream → nu_reco_larpid_{nu,fm}/ (apply_larpid.py)
       5 export     gen2ntuple shards (export_gen2ntuple.py)
       6 hadd       merge → dlgen2_larformer_ntuple_<TAG>.root
```

Output lands in `DATADIR/` (the same dir that holds `merged_sp/`), mirroring the
reference `output/mcc9_bnbnu_overlay_1500_full_satfix/` layout.

**Data vs MC knobs** (env vars to the chain):
- `TRUTH_DIR` — set to a real `truth_sidecar/` dir → MC mode (fills truth
  branches + `potTree` + `xsecWeight` in the ntuple). Leave unset → data mode.
- `LARPID_SAMPLE_TAG` — `apply_larpid.py`'s `select_checkpoint`: **`run3` anywhere
  in the tag → alternate (run-1 MC) weights**, else default. MC overlay samples
  (bnb_nu, intrinsic_nue) use the alternate weights → put `run3` in the tag.
- `WEIGHTS_PKL` — per-sample CV xsecWeight pickle in
  `pointcept_env/gen2ntuple/event_weighting/` (e.g.
  `weights_forCV_v48_Sep24_intrinsic_nue_run3.pkl` for the nue overlay;
  `..._bnb_nu_run3.pkl` for bnb overlay). Propagates to the export shard via
  `--export=ALL`. Wrong/missing pickle → per-event `xsecWeight=-1` (tolerated).
- `NINF` / `NNR` / `NEXP` — shard counts (inference GPU / nu_reco per stream /
  export). `EXCLUDE_NODES` — comma-sep bad nodes. `DEP` — `afterany:` gate job(s)
  (e.g. the Step-A + truth_sidecar array ids) so the chain auto-starts when the
  merged_sp are ready.

### Worked example — mcc9 v29e run3b intrinsic-nue overlay (MC)

```bash
# 1) merged_sp (Step A) — see ../data_prep/uboone_official/LARFORMER_DATAPREP.md
CONF=lartpc/data_prep/uboone_official/larformer_configs/mcc9_v29e_nue_overlay_tufts.conf
STEPA=$(CONF=$CONF STRIDE=5 sbatch --parsable --array=0-666%100 \
        lartpc/data_prep/uboone_official/submit_stepA_shard.sh)

# 2) truth_sidecar (MC only), same dlmerged list so fileno tags align with Step A
DATADIR=/cluster/tufts/wongjiradlabnu/nutufts/data/larformer/mcc9_v29e_dl_run3b_bnb_intrinsic_nue_overlay_nocrtremerge
DLLIST=lartpc/data_prep/uboone_official/inputlists/mcc9_v29e_dl_run3b_bnb_intrinsic_nue_overlay_nocrtremerge.txt
TRUTHSC=$(INPUT_LIST=$DLLIST OUTPUT_DIR=${DATADIR}/truth_sidecar NSHARDS=50 \
          sbatch --parsable --array=0-49 lartpc/larformer_reco/slurm/submit_truth_sidecar_shard.sh)

# 3) downstream chain, gated on Step A + truth_sidecar
DEP=${STEPA}:${TRUTHSC} TAG=mcc9_v29e_nue_overlay DATADIR=$DATADIR \
  TRUTH_DIR=${DATADIR}/truth_sidecar \
  LARPID_SAMPLE_TAG=mcc9_v29e_dl_run3b_bnb_intrinsic_nue_overlay \
  WEIGHTS_PKL=/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/gen2ntuple/event_weighting/weights_forCV_v48_Sep24_intrinsic_nue_run3.pkl \
  NINF=8 NNR=8 NEXP=4 \
  bash lartpc/larformer_reco/slurm/submit_extbnb_chain.sh
# final ntuple: ${DATADIR}/dlgen2_larformer_ntuple_mcc9_v29e_nue_overlay.root
```

## Streams and flash products

Cascade inference emits up to TWO labeled slice streams per event (file attr
`stream`; analyzers pick by stream):

- `stream="nu"` — the slicer's nu-union slice (the production path;
  `keypoint2_event{i}_0.h5`). Becomes `"nu,flashmatch"` when the nu slice is
  also the best flash-match.
- `stream="flashmatch"` — Stage-3+keypoints rerun on the best flash-χ² slice
  when that is NOT the nu union (`keypoint2_event{i}_fm_0.h5`, attrs
  `slice_label="cosmicQQ"`, `flash_chi2`). Runs with the loose single-object
  fallback enabled (per-particle `loose_pass` attr). Recovers events with no nu
  slice at all. NOTE the single-photon study found a blind K=1 rescue is
  net-negative for 1γ0X selection — cut on `flash_chi2`/χ²-margin in analysis.

Every keypoint2 file also carries the flash-match products (from the input
merged_sp `entry_0/flashes` + `lartpc/flashmatch` prediction):

```
flash/observed_pe (32,)      in-time beam flash (producer 0, max total PE);
      attrs time_us total_pe producer_id flash_index has_beam_flash
flash/all/{pe,producer_id,total_pe,time_us}          full flash table
slices/{label,query,n_points,pred_pe (S,32),chi2,oob_frac,chi2_rank,p_nu}
       per-slice flash-match table over the WHOLE event (nu union + every
       cosmic slice >= --slice-min-points); query=-5 is the nu union;
       chi2_rank is 1-based among slices with oob_frac <= --flash-oob-max
slices/nu_queries/{query,p_nu}                       individual nu queries
```

Knobs: `--gamma-beam` (5.25), `--flash-f-sys` (0.10), `--flash-eps` (1.0),
`--flash-oob-max` (0.05), `--photonlib`, `--no-flash`, `--no-flashmatch-stream`.

Downstream: `slurm/regen_kp2_list.sh` writes a per-stream list
(`..._flashmatch.txt`); run `scripts/run_nu_reco.py` and
`eval/eval_reco_performance.py --stream flashmatch` per list — the gidx linkage
(list line number ↔ nu_reco event group) means streams must never be mixed in
one list. nu_reco event groups carry `stream`/`slice_label`/`flash_chi2` attrs.

## Layout

- `scripts/` — `run_nu_reco.py` (main driver), `dump_schema.py`
- `trajfit/` — the reconstruction library (`nu_interaction.py` is the core;
  `cluster_fit_stitch`, `particle_momentum`, `range_momentum`, `calo`,
  `shower_{trunk,connect,truth}`, `mcs_rdp`, `run_elpigraph`).
  Modules use relative imports — run their CLIs as
  `PYTHONPATH=./ python3 -m lartpc.larformer_reco.trajfit.particle_momentum`.
  `trajfit/data/` holds the lookup tables (`range2ke_lar.npz` from
  `make_range2ke_npz.py`, `calo_calib.npz` from the calo-calibration shards) —
  committed to git; production depends on them.
- `keypoint/` — score-field nu-vertex/keypoint peak fitter (greedy Gaussian NMS;
  see `specs/keypoint_reco_spec.md`). `nu_interaction.vertex_candidates()` imports
  this; if the import fails it **silently falls back** to the dense `nu_vertex_cm`
  decode (broad `except Exception`) — reco still runs but vertices degrade.
  Tests: `PYTHONPATH=./ python3 -m lartpc.larformer_reco.keypoint.test_keypoint_reco`.
- `eval/` — `eval_reco_performance.py` (per-species efficiency: segmentation,
  attachment+kinematics, slice coverage), `eval_keypoint2_inference.py`,
  `keypoint/eval_nu_vertex_reco.py`.
- `viz/` — `visualize_cascade_output.py` interactive event display
  (temporarily imports `lartpc_data_prep.detectoroutline` until the shared viz
  library exists).
- `studies/` — one-off spikes: ElPiGraph sweeps, shower-direction study,
  shower-attach scans, single-photon selection/recovery analyses.
- `slurm/` — cluster submission scripts (inference → nu-reco → eval chain).
- `specs/` — design specs per subsystem + `DEVLOG.md` (the dated development
  journal formerly serving as this README).
- `utils.py` — shared helpers (`read_list`).
- `inputlists/`, `outputlists/` — dataset/file lists (gitignored contents).
  `output/`, `plots/`, `logs/` — run artifacts (gitignored).

## Configs & checkpoints

Training/inference configs: `configs/lartpc/larformer/stage4_keypoint/`
(see `configs/lartpc/README.md` for the production chain).
