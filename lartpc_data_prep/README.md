# lartpc_data_prep — mostly moved to `lartpc/`

This directory has been reorganized (see `docs/Reorganization_Plan.md` §3):

- Data prep pipelines, label makers, QA → `lartpc/data_prep/`
  (`training_data/` = former `lantern_scripts/`, `uboone_official/` = former
  `larformer_scripts/`, plus `labels/`, `validation/`, `archive/`)
- Analysis directories → `lartpc/larformer_analysis/`
  (`slicer_eval/` = former `larformer_analysis/`, `particle_eval/` = former
  `larformer_particle_analysis/`, `physics/` = former `larformer_physics/`,
  completed one-offs under `archive/`)

Data processing uses code from the [larflow](https://github.com/nutufts/larflow)
repository, part of the [ubdl](https://github.com/larbys/ubdl) stack.

- Nu-interaction reconstruction post-processing (former `larformer_keypoint_v2/`)
  → `lartpc/larformer_reco/`

## Still here (pending the shared-viz-library phase)

- `detectoroutline.py` + `vis_lartpc_hdfdata.py`, `view_particle_gt.py`,
  `vis_shower_fragments.py`, `characterize_fragments.py` — visualization/diagnostic
  scripts.
- `inputlists/` — dataset list files.
