"""Per-run dead/disabled PMT (opdet) list for the flash-match chi2.

Dead PMTs read observed PE ~= 0 every event, but the PhotonLib still predicts
light on them, so the Neyman term (0 - pred)^2 / eps ~ pred^2 spuriously
dominates the flash chi2 (diagnosed 2026-07-14: opdet 15 was ~98% of the chi2
in the high-chi2 tail of run3 MC). flash_chi2.neyman_chi2(dead_opdets=...)
excludes these channels from the sum.

The lists here are indexed by OPDET (the 32-element PE-array order, verified
opdet-indexed by the charge<->light spatial alignment). Currently derived
data-driven: the channel(s) whose observed PE is ~0 across a whole run.

  - Run 1 (bnb5e19, runs ~5121-5946): opdet 15 is LIVE  -> mask nothing.
  - Run 3 (mcc9 v29e overlay + EXT, runs ~14121-18794): opdet 15 is DEAD.

TODO: replace with the official MicroBooNE bad-optical-channel list per run
period (resolves whether run3 has a second, intermittently-dead channel, and
fills run2/4/5 which are not yet processed here).

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
