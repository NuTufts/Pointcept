# Generation-1 Training Data Prep (superseded)

Direct dlmerged→H5 conversion drivers for the corsika samples, without the LANTERN
step (no truepoint scores). Superseded by the generation-2 pipeline in
`../../training_data/` (2026-05). Kept for provenance of the v3_showerfragments-era
datasets.

Notes:
- `process_dlmerged_to_hdf5_event_files.py` and `merge_reco_truth_showerorigin.py`
  here are older copies; the live versions (with `--fileno-tag` support) are in
  `../../training_data/`.
- The `run_corsika_*.sh` WORKDIR paths point at the old
  `pointcept_env/pointcept` checkout.
