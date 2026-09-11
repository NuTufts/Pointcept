"""Live-PMT mask: run-aware dead list + observed-flash saturation holes.

One function for every estimator so the muon arm, the legacy arms and the
closure test all sum over the same tubes.
"""
import numpy as np

from lartpc.flashmatch.dead_channels import dead_opdets_for_run
from lartpc.flashmatch.saturation import find_saturated

N_PMTS = 32


def live_mask(run, obs_pe, saturation=True, max_masked=4):
    """(live[32] bool, dead tuple, saturated tuple)."""
    dead = tuple(dead_opdets_for_run(int(run)))
    live = np.ones(N_PMTS, bool)
    if dead:
        live[list(dead)] = False
    sat = ()
    if saturation:
        try:
            sat = tuple(int(x) for x in find_saturated(
                np.clip(np.asarray(obs_pe, np.float64), 0, None),
                dead=dead, max_masked=max_masked))
        except Exception:
            sat = ()
        if sat:
            live[list(sat)] = False
    return live, dead, sat
