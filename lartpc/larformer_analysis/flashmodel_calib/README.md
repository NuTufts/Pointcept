# flashmodel_calib: flash light-yield (gamma) calibration

Calibrates the charge-to-PE scale of the LArFormer flash model per (sample
kind, run period). Read `PROTOCOL.md` for the procedure and
`CALIBRATION_LOG.md` for the history, measurements and open questions.

## Layout

| path | what |
|---|---|
| `PROTOCOL.md` | the standing calibration procedure |
| `CALIBRATION_LOG.md` | append-only log: deployed constants, measurements, decisions |
| `gammacal/` | library: sample registry, muon selection, own-points prediction at the reference gamma, live-PMT mask, estimators, provenance |
| `scripts/build_records.py` | one pass over a sample -> per-event / per-muon record npz |
| `scripts/fit_gamma.py` | apply the protocol cuts, fit `s`, scans, JSON + plots |
| `scripts/rank1_gamma_check.py` | MC: true-nu slice rank-1 rate vs gamma (ranking sensitivity) |
| `slurm/run_build_records.sh` | sharded record build (batch partitions, container) |
| `results/<chain>/` | `*.json` results (tracked), `records/` shards and `plots/` (not tracked) |
| `archive/saturation_forensics/` | July 2026 PMT-saturation diagnosis (the hole mask now in `lartpc/flashmatch/saturation.py`); see its README |
| `archive/july2026_gamma/` | superseded estimators (`fit_gamma_run.py`, `measure_bulk_gamma.py`, ...) and their outputs, kept for regression |

## Where the scale lives in production

`lartpc/flashmatch/dead_channels.py` (`GAMMA_SCALE_BY_PERIOD`, keyed on run
period only) multiplies `--gamma-beam 5.25` inside
`tools/larformer/run_larformer_keypoint2_cascade_inference.py` before the flash
chi2 ranking, so the value is baked into the GPU pass. Per-sample overrides go
through `INF_EXTRA_ARGS="--gamma-run-scale <s>"` on the reco chain. Every
cascade file records `flash` attrs `gamma_beam, gamma_scale, gamma_eff,
dead_opdets`.

## Quick start

    K=/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/kpv2_pointcept
    C=/cluster/tufts/wongjiradlabnu/larbys/containers/pointcept_cuml.sif
    apptainer exec --bind /cluster:/cluster $C bash -c "cd $K && export PYTHONPATH=$K && \
      python3 lartpc/larformer_analysis/flashmodel_calib/scripts/fit_gamma.py --sample bnb5e19_cew6 --arm muon"

Samples are registered in `gammacal/samples.py` (paths, kind, period, chain,
flash window). Add a new production there first.
