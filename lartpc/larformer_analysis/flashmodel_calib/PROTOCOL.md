# Flash light-yield scale (gamma) calibration protocol

Standing procedure for measuring the charge-to-PE scale of the flash model, one
value per (sample kind, run period), on a given reconstruction chain. The code
is `gammacal/` + `scripts/`; results and the running log are in
`results/<chain>/` and `CALIBRATION_LOG.md`.

## 0. What is being calibrated

    pred_pe[j] = gamma_eff * sum_sp q_comb(sp) * Vis(sp, j)

`q_comb` is the de-double-counted wire-plane pixel value (uncalibrated ADC-like
units), `Vis` the PhotonLib visibility, and `gamma_eff = gamma_beam * gamma_scale`
the effective scale that folds (i) charge calibration (ADC per electron; differs
by run period AND by data vs simulation), (ii) PMT PE calibration, and (iii) the
scintillation light yield (Run 3 ~40% lower than Run 1 from contaminants). Only
(iii) carries the "Run 1 brighter" expectation, so the protocol MEASURES the
direction per sample rather than assuming it, and includes a charge-side vs
light-side decomposition.

Conventions:
* `GAMMA_BEAM_REF = 5.25` is frozen. A calibrated cell is a multiplier `s` on it
  (`gamma_eff = 5.25 * s`).
* Every prediction in this folder is recomputed at `GAMMA_BEAM_REF` from the
  calibration muon's OWN spacepoints (CPU PhotonLib). The gamma the production
  ran at is recorded per event for audit only, so there is no "which gamma was
  this pred_pe stored at" arithmetic.
* Live PMTs = not dead in this run period (`dead_channels.dead_opdets_for_run`)
  and not a saturation hole in this flash (`saturation.find_saturated`).

## 1. Sample kinds and what light they measure

| kind | samples | in-time light source |
|---|---|---|
| data | beam-on (bnb5e19), beam-off EXT | real detector light |
| mc   | overlay: simulated nu on unbiased-trigger beam-off data | SIMULATED nu light (a coincident data cosmic in the window is rare) |

Consequence: an in-time muon in an overlay is the simulated CC muon, so the same
muon selection measures the MC light scale there and the data light scale in
data. Table cells are keyed `(kind, period)`; an overlay of run period P is used
for `(mc, P)`, EXT/beam data of period P for `(data, P)`.

The in-time flash window differs by sample (software trigger / beam gate):
beam data run-1 ~[2.8, 5.0] us, run-3 EXT ~[3.2, 5.4] us, run-3 MC ~[3.6, 5.2] us.
The window is part of the sample registry (`gammacal/samples.py`) and must be
checked on the flash-time histogram whenever a new sample is added. EXT emulates
the beam trigger, so a cosmic flash near a window edge can be partly truncated:
a 0.2 us edge margin is applied and the ratio is scanned vs position in window.

## 2. The calibration event

**Calibration stream (preferred, 2026-09-12).** Run the cascade with
`--flash-calib-mode` (`slurm/run_calib_inference.sh`): the deghosted cloud is
clustered into connected components, light is predicted for every cluster, the
cluster whose PMT pattern matches the in-time flash (cosine >= 0.9) is the
flash source and is written as stream='calib'. The calibration object is the
WHOLE flash-matched cluster (`fit_gamma.py --calib-object cluster
--no-require-mu-class --rms-perp-max 4 --lin-min 0.95 --n-boundary 2
--min-len 120 --iso-track-ke 1e9 --iso-shower-e 1e9`): track-like,
through-going and long, so it is complete by construction. Segmenter
instances hold only ~78% of a cosmic muon's points and bias the ratio high
(30-35% in data); short clusters are fragments and bias it high too. On
overlays add `--nu-qfrac-min 0.3` (the in-time light is the neutrino's, so the
calibration slice must hold the true-nu charge; removes the aligned-cosmic and
dim-flash mis-associations). Data cells use `--n-boundary -1` (>= 1 boundary
end); the through-going subset (`--n-boundary 2`) is the cleanest cross-check.

**Production stream (cross-check).** The selection below on the nu-stream
kp2 + nu_reco output. Kept because it needs no extra inference and because
on overlays the nu slice is the natural object (the in-time light is the
neutrino's; see the MC caveat in CALIBRATION_LOG 2026-09-12).

An isolated, one-boundary, minimum-ionising muon track that is the source of
the in-time flash. Defined identically in every sample; all cuts are
`scripts/fit_gamma.py` knobs (defaults in brackets) applied to the record file.

| step | cut | why |
|---|---|---|
| candidates | segmenter tracks with LArFormer class mu (nu_reco particles AND vertex-less kp2 instances), length > 30 cm at record time | no vertex requirement: EXT/cosmic muons have none |
| flash | brightest simpleFlashBeam flash in [2, 7] us; at fit time inside the sample window minus margin [0.2 us], the ONLY flash > 20 PE in the window | in-time, untruncated, unambiguous |
| length | > [50] cm | MIP, well reconstructed |
| topology | exactly [1] endpoint within 10 cm of a TPC face | one-boundary: entering-stopping cosmic, or exiting nu muon |
| boundary end | drift x > [125] cm (cathode side) | light from the outside-TPC segment is attenuated before the PMTs; checked by the ratio-vs-x scan |
| primary | not a secondary (kink/decay daughter) | |
| isolation | no other track with KE > [50] MeV, no shower with E > [30] MeV in the slice; muon comb charge >= [0.8] of the slice's | the flash must come from this muon |
| proton veto | no proton track with KE > [50] MeV | recombination-biased q/L |
| prediction | sum_live pred_ref >= [50] PE, sum_live obs > 0 | |
| source test | cosine similarity(obs, pred_mu) over live PMTs >= [0.9]; pred/obs light centroid within [60] cm in y, [100] cm in z | scale-free shape match: the muon IS what lit the PMTs; not circular |
| one per event | longest surviving muon | independence |

`r = sum_live obs / sum_live pred_ref` per muon.

## 3. Statistic and validity gates

Primary: `s = median(r)` with a 2000-resample bootstrap error and the 16-84%
band. Gates (a cell failing them is reported, not promoted): the trimmed core
median (iterated median of the events within +-0.15 dex of the running centre)
must hold >= 50% of the events and lie within 5% of the plain median. A
background of mis-associated flashes pulls the two apart, which is exactly what
the gate is for.
Secondary, reported alongside: the per-event Neyman-weighted closed-form
multiplier (the gamma that minimises the production chi2) and its pooled value,
and the geometric mean.

## 4. Robustness scans (all from the record file)

boundary-end x bins (attenuation residual: must be flat), muon length, observed
brightness, cosine threshold, flash position in window (edge truncation),
run-number quantile groups (stability within the period), vertexed vs
vertex-less, MC truth origin (nu vs cosmic). A stable cell has every scan within
+-0.05 of the headline.

## 5. Cross-checks

* union arm: nu-union prediction (recomputed at the reference) for the same
  events -- the legacy population; diagnostic only.
* truth_nu arm (MC): union arm restricted to events whose nu union holds
  >= 90% of the true neutrino charge; tests whether shower-rich nu light needs
  a different scale than MIP light.
* beam-on vs EXT of the same period must agree (two triggers, two populations,
  same light).
* `scripts/rank1_gamma_check.py`: ranking sensitivity to gamma on MC.

## 6. Decomposition for period-to-period differences

From the same muons: charge side = muon comb charge per cm (MIP dQ/dx) per
period and kind; light side = observed PE per unit predicted light. If `s`
moves with the charge side it is charge calibration; if with the light side it
is light yield.

## 7. Closure

Re-infer >= 5k events of the cell at `--gamma-run-scale s`; rebuild records;
require `median r = 1.00 +- 0.05`, scans flat, and (MC) the true-nu rank-1
rate not below the reference. Only then write the cell into the production
table with the result JSON as provenance.

## 8. Running it

    K=<repo>; export PYTHONPATH=$K
    # records (sbatch, sharded; --union-every for the union/truth arms)
    SAMPLE=extbnb200k_cew6 NSHARDS=16 EXTRA=--union-every \
        sbatch --array=0-15 lartpc/larformer_analysis/flashmodel_calib/slurm/run_build_records.sh
    # fit
    python3 lartpc/larformer_analysis/flashmodel_calib/scripts/fit_gamma.py \
        --sample extbnb200k_cew6 --arm muon --plots lartpc/larformer_analysis/flashmodel_calib/results/s1ep2p8cew6/plots
