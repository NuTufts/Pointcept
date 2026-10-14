"""
MicroBooNE flash model: today's lartpc/flashmatch behaviour expressed as a FlashModel.

In the unified formula this is the special case
    n(P)  = de-double-counted Y-plane comb charge (charge rule applied by the caller)
    v_dir = v6 70 kV photon library (trilinear), v_refl = none, ε_dir = 1
    G     = gamma_beam (5.25) x gamma scale(run period, data/MC) x readout factor(producer)
and it must reproduce flash_predict.predict_many_slices_pe / predict_slice_pe,
flash_chi2.neyman_chi2 / oob_mask, dead_channels and saturation bit-for-bit
(tests/test_uboone_regression.py).

Constants still come from the legacy modules (MicroBooNE has no detinfo file
yet); moving them to detinfo is a later step that must keep this test green.
"""
import numpy as np

from .. import base
from ... import dead_channels, flash_calib, flash_predict, saturation
from ...flash_chi2 import (DRIFT_VELOCITY_CM_PER_US, N_PMTS, TPC_X_MAX, TPC_X_MIN, TPC_Y_MAX,
                           TPC_Y_MIN, TPC_Z_MAX, TPC_Z_MIN)


class UBooNEFlashModel(base.FlashModel):
    detector = "uboone"
    n_channels = N_PMTS
    drift_dir_x = (-1.0,)                                  # anode at x = 0, drift toward -x
    active_boxes = (((TPC_X_MIN, TPC_Y_MIN, TPC_Z_MIN), (TPC_X_MAX, TPC_Y_MAX, TPC_Z_MAX)),)
    v_drift_cm_per_us = DRIFT_VELOCITY_CM_PER_US
    channel_tpc = None

    def __init__(self, photonlib_cache=None, device=None, score=None):
        super().__init__(score=score, device=device)
        self.photonlib_cache = photonlib_cache
        self._pl = None

    # the legacy process-wide singleton, so mixing old and new call sites shares one LUT
    @property
    def photonlib(self):
        if self._pl is None:
            self._pl = flash_predict.get_photonlib(cache_path=self.photonlib_cache,
                                                   device=self._device)
        return self._pl

    @property
    def device(self):
        return self.photonlib.vis_table.device

    @property
    def vis_dtype(self):
        return self.photonlib.vis_table.dtype

    def visibility_torch(self, pos):
        return self.photonlib.visibility(pos), None

    # ----------------------------------------------------------- light yield
    @staticmethod
    def readout_factors():
        """(beam, cosmic) readout factors (fast/slow scintillation in the readout window)."""
        return flash_predict.default_readout_factors()

    @staticmethod
    def gamma_scale(spec, run, kind=None):
        """Calibrated multiplier on GAMMA_BEAM_REF (flash_calib.resolve)."""
        return flash_calib.resolve(spec, run, kind)

    def light_yield(self, gamma_by_producer, producer_id=0, readout_factor_by_producer=None):
        """G exactly as the legacy code forms it: float(gamma[pid]) * float(rf[pid])."""
        rf = readout_factor_by_producer or self.readout_factors()
        pid = int(producer_id)
        return float(gamma_by_producer[pid]) * float(rf[pid])

    def predict_legacy(self, pos_cm, charge, cluster_id, n_clusters, flash_t0_us, producer_id=0,
                       gamma_by_producer=(flash_predict.DEFAULT_GAMMA, flash_predict.DEFAULT_GAMMA),
                       readout_factor_by_producer=None):
        """Drop-in for flash_predict.predict_many_slices_pe (same arguments, same output)."""
        G = self.light_yield(gamma_by_producer, producer_id, readout_factor_by_producer)
        return self.predict(pos_cm, charge, cluster_id, n_clusters, flash_t0_us, light_yield=G)

    # ----------------------------------------------------------- masks
    @staticmethod
    def dead_channels(spec="auto", run=None):
        return dead_channels.resolve_dead_opdets(spec, run)

    def channel_mask(self, pe_obs, run=None, dead_spec="auto", mask_saturated=True,
                     max_saturated=4):
        """dead tubes for the run period ∪ saturation holes in the observed flash
        (the keypoint2-cascade rule)."""
        dead = self.dead_channels(dead_spec, run)
        sat = (saturation.find_saturated(pe_obs, dead=dead, max_masked=int(max_saturated))
               if mask_saturated else ())
        return tuple(sorted(set(dead) | set(sat)))

    @staticmethod
    def charge_from_pixval(pixval):
        """Y-plane charge with U/V fallback (numpy), the legacy charge selection."""
        return flash_predict.select_charge_y_with_uv_fallback_np(pixval)
