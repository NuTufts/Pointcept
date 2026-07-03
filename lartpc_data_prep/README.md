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

## Still here (pending later reorg phases)

- `larformer_keypoint_v2/` — nu-interaction reconstruction post-processing;
  moving to `lartpc/larformer_reco/` in a later phase.
- `detectoroutline.py` + `vis_lartpc_hdfdata.py`, `view_particle_gt.py`,
  `vis_shower_fragments.py`, `characterize_fragments.py` — visualization/diagnostic
  scripts; moving to the shared viz library phase.
- `inputlists/` — dataset list files.
