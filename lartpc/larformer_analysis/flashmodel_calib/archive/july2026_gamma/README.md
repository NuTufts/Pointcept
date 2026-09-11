# Archived: July-September 2026 gamma estimators

Superseded by `../../gammacal` + `../../scripts/fit_gamma.py` (see
`../../PROTOCOL.md`). Kept because they produced the numbers in
`../../CALIBRATION_LOG.md` and serve as the regression reference.

* `fit_gamma_run.py` -- "clean muon" estimator on the NU-UNION prediction stored
  in the cascade files (`--gamma-beam` must equal the production `gamma_eff`).
* `measure_bulk_gamma.py` -- nu-union bulk median; `out/bulk_*.npz`.
* `run_gamma_2x2.sh`, `run_bulk_gamma_refs.sh` -- the sbatch drivers (paths updated
  to this directory).
* `test_gamma_scale_chi2.py`, `compare_chi2_mc_data.py`, `compare_predobs_mc_data.py`,
  `flashmatch_impact.py` -- July diagnostics of the run-1 0.80 correction.
* `gamma_*.npz`, `out/`, `plots/` -- their outputs (not tracked).

Why superseded: the nu-union prediction sums charge from everything the slicer
put in the union (in EXT several out-of-time cosmics), so its obs/pred ratio is
dominated by mis-association and depends on sample composition. The new code
predicts from the calibration muon's own spacepoints.
