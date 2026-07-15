"""Per-run flash-match config: dead PMT (opdet) list + light-yield gamma scale.

DEAD PMTs read observed PE ~= 0 every event, but the PhotonLib still predicts
light on them, so the Neyman term (0 - pred)^2 / eps ~ pred^2 spuriously
dominates the flash chi2 (diagnosed 2026-07-14: opdet 15 was ~98% of the chi2
in the high-chi2 tail of run3 MC). flash_chi2.neyman_chi2(dead_opdets=...)
excludes these channels from the sum.

  - Run 1 (bnb5e19, runs ~5121-5946): opdet 15 is LIVE  -> mask nothing.
  - Run 3 (mcc9 v29e overlay + EXT, runs ~14121-18794): opdet 15 is DEAD.

GAMMA SCALE is a per-run multiplier on gamma_beam (the q->PE light-yield scale)
that absorbs the run-to-run pred/obs offset. Measured (flashmodel_calib,
2026-07-14) from in-time MIP muons and cross-checked on pi0 shower slices:
run1 bnb5e19 OVER-predicts PE vs run3 MC by ~1.25-1.4x (obs/pred ~ 0.79 muons /
0.85 showers) -> apply ~0.80 for run1. This is OPPOSITE the scintillation
light-yield-vs-time curve (which would make run1 brighter), so it is dominated
by a run1<->run3 charge or PMT-PE calibration/electronics difference, not
scintillation; regardless, it is the correct multiplier to center the flash
match. gamma_beam=5.25 is tuned to run3 MC -> run3 scale = 1.0.

TODO: (a) replace dead list with the official MicroBooNE bad-optical-channel
list per period; (b) measure the run3-DATA gamma from the full EXT sample (the
val gave N=11); if run3 data also offsets from run3 MC, the effect is data/MC
not purely per-run and this table should be split accordingly.

Standard MicroBooNE run-period run-number boundaries (inclusive lower edge):
  run1 < 7771 <= run2 < 13697 <= run3 < 18961 <= run4 < 22270 <= run5
"""

# opdet indices dead in each run period (see module docstring).
DEAD_OPDETS_BY_PERIOD = {
    1: (),        # channel 15 live
    2: (),        # (not yet processed; assume none pending official list)
    3: (15,),     # opdet 15 dead (data-driven)
    4: (15,),     # assume 15 stays dead (needs official list)
    5: (15,),     # assume 15 stays dead (needs official list)
}

# per-run-period multiplier on gamma_beam (light-yield / pred-PE scale).
# 1.0 = run3 (gamma_beam is tuned to run3 MC). See module docstring.
GAMMA_SCALE_BY_PERIOD = {
    1: 0.80,      # bnb5e19 run1: measured (muon 0.79 / shower 0.85)
    2: 1.0,       # not yet measured -> reference
    3: 1.0,       # run3 MC reference (gamma_beam)
    4: 1.0,       # not yet measured
    5: 1.0,       # not yet measured
}

# (period, run_low_inclusive) upper-open boundaries.
_PERIOD_BOUNDS = ((1, 0), (2, 7771), (3, 13697), (4, 18961), (5, 22270))


def run_period(run):
    """MicroBooNE run period (1-5) for a run number."""
    run = int(run)
    period = 1
    for p, lo in _PERIOD_BOUNDS:
        if run >= lo:
            period = p
    return period


def dead_opdets_for_run(run):
    """Tuple of dead opdet indices for the given run number (run-period based).
    Returns () for run1/2 (channel 15 live) and (15,) for run3+."""
    return DEAD_OPDETS_BY_PERIOD.get(run_period(run), ())


def resolve_dead_opdets(spec, run):
    """Resolve a CLI --dead-opdets spec into a tuple of opdet indices.

    spec: "auto"  -> dead_opdets_for_run(run)  (run-period based)
          "none"  -> ()  (score all 32 PMTs; reproduces pre-fix behavior)
          "15" / "15,7" -> that explicit list (forces, ignores run)
    """
    if spec is None or spec == "auto":
        return dead_opdets_for_run(run)
    if spec == "none":
        return ()
    return tuple(int(x) for x in str(spec).split(",") if x.strip() != "")


def gamma_scale_for_run(run):
    """Per-run multiplier on gamma_beam (1.0 = run3 reference)."""
    return GAMMA_SCALE_BY_PERIOD.get(run_period(run), 1.0)


def resolve_gamma_scale(spec, run):
    """Resolve a CLI --gamma-run-scale spec into a float multiplier.

    spec: "auto" -> gamma_scale_for_run(run)  (run-period based)
          "1" / "0.8" -> that explicit multiplier (forces, ignores run)
    """
    if spec is None or spec == "auto":
        return gamma_scale_for_run(run)
    return float(spec)
