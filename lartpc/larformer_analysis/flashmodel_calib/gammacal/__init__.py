"""gammacal: flash light-yield (gamma) calibration library.

The flash prediction is  pred_pe[j] = gamma_eff * sum_sp q_comb(sp) * Vis(sp, j).
Everything in this package predicts at the FROZEN reference GAMMA_BEAM_REF, from
the calibration muon's own spacepoints, so a sample's calibrated scale is simply

    s = stat( sum_live obs / sum_live pred_ref )

with no dependence on the gamma the production was run at (that value is
recorded per event for audit only). See ../PROTOCOL.md.
"""
import os

GAMMA_BEAM_REF = 5.25      # frozen reference; table entries are multipliers on it
VERSION = "gammacal-1"

REPO_ROOT = os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", ".."))
CALIB_DIR = os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), ".."))
