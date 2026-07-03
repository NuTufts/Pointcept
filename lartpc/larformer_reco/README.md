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
tools/run_larformer_keypoint2_cascade_inference.py     (GPU; slurm/submit_inference_shard.sh)
  → keypoint2_event*.h5
scripts/run_nu_reco.py                                 (CPU; slurm/submit_nu_reco_shard.sh)
  → nu_reco_shard*.h5
eval/eval_reco_performance.py                          (slurm/submit_eval_reco_{shard,merge}.sh)
  → eval_shard*.npz → merged records + plots
```

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
